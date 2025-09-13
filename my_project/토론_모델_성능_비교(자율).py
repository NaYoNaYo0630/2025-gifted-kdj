import os
import re
import json
import time
import random
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
import ollama
from utils import check_ollama

# ================== 상수/전역 ==================
LETTERS = ["A", "B", "C", "D"]

# ================== 공용 유틸 ==================
def clean_surrogates(text: str) -> str:
    return re.sub(r'[\ud800-\udfff]', '', text or "")

def safe_json_loads(payload: str) -> Optional[dict]:
    """
    모델이 코드펜스나 앞뒤 잡음을 덧붙였을 때도 JSON만 추출해서 파싱.
    """
    if not payload:
        return None
    m = re.search(r"\{.*\}|\[.*\]", payload, re.S)  # 객체/배열 모두 허용
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ---- 발언 추출 보강 ----
def _extract_talk(value) -> str:
    """모델 출력에서 발언 텍스트를 최대한 안전하게 뽑아낸다."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for k in ("talk", "speech", "content", "text", "message"):
            v = value.get(k)
            if isinstance(v, str):
                return v.strip()
        parts = [str(v).strip() for v in value.values() if isinstance(v, (str, int, float))]
        if parts:
            return " ".join(parts)
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if isinstance(x, (str, int, float))]
        if parts:
            return " ".join(parts)
    return ""

def _normalize_debate_json(data: dict) -> Dict[str, str]:
    """
    허용 형태:
      1) {"A":{"talk":"..."}, "B":{"talk":"..."}, ...}
      2) {"A":"...", "B":"...", ...}
      3) 키 대소문자 섞임
    반환: {"A": str, "B": str, "C": str, "D": str}
    """
    if not isinstance(data, dict):
        return {k: "" for k in LETTERS}
    upper = {str(k).upper(): v for k, v in data.items()}
    out = {}
    for L in LETTERS:
        v = upper.get(L)
        out[L] = _extract_talk(v) if v is not None else ""
    return out

@st.cache_data(show_spinner=False)
def load_questions():
    """
    도메인(분야) -> {qid: item} 형태/기존 평탄 형태 모두 지원.
    반환: (flat_questions, domain_index, qid2domain)
    """
    candidates = [
        "pages/mmlu_debate_questions.json",
        "mmlu_debate_questions.json",
        r"C:\Users\USER\ollama\pages\mmlu_debate_questions.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)

            def looks_like_question(d: dict) -> bool:
                return isinstance(d, dict) and {"question", "choices", "answer"}.issubset(d.keys())

            is_nested = all(isinstance(v, dict) for v in raw.values()) and not any(
                looks_like_question(v) for v in raw.values()
            )

            flat, domain_index, qid2domain = {}, {}, {}
            if is_nested:
                for domain, items in raw.items():
                    if not isinstance(items, dict):
                        continue
                    domain_index.setdefault(domain, [])
                    for qid, item in items.items():
                        if not looks_like_question(item):
                            continue
                        flat[qid] = item
                        flat[qid]["topic"] = item.get("topic") or domain
                        domain_index[domain].append(qid)
                        qid2domain[qid] = domain
            else:
                for qid, item in raw.items():
                    if not looks_like_question(item):
                        continue
                    flat[qid] = item
                    dom = item.get("topic", "Misc")
                    domain_index.setdefault(dom, []).append(qid)
                    qid2domain[qid] = dom
                    flat[qid]["topic"] = flat[qid].get("topic", dom)
            return flat, domain_index, qid2domain

    st.error("mmlu_debate_questions.json 파일을 찾지 못했습니다. 위치를 확인하세요.")
    st.stop()

def extract_choice_strict(text: str) -> str:
    """
    (구) Judge 전용: 태그 한 줄 or '정답/answer/final: X' 한 줄만 인정.
    """
    t = (text or "").strip()
    m = re.search(r"<answer>\s*([ABCD])\s*</answer>", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.fullmatch(r"(?:정답|answer|final)\s*(?:is|:|=)?\s*([ABCD])\s*", t, re.I)
    if m:
        return m.group(1).upper()
    return ""

def extract_choice(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"<answer>\s*([ABCD])\s*</answer>", t, re.I)
    if m: return m.group(1).upper()
    m = re.search(r"(정답|answer|final)\s*(is|:|=)?\s*([ABCD])\b", t, re.I)
    if m: return m.group(3).upper()
    for line in t.splitlines():
        m = re.match(r"^\s*[\(\[\{<]?\s*([ABCD])\s*[\)\]\}>]?\s*$", line.strip(), re.I)
        if m: return m.group(1).upper()
    m = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", t.upper())
    return m[0] if m else ""

def extract_leading_choice_from_line(line: str) -> str:
    """
    자유 선택 모드의 각 발언은 '[X : 주장]' 형식.
    여기서 X(A/B/C/D)를 파싱한다.
    """
    if not line:
        return ""
    s = line.strip()

    # 1) 권장 형식: [X : ...] 또는 [X: ...]
    m = re.match(r"^\[\s*([ABCD])\s*[:\|\-]\s*", s, re.I)
    if m:
        return m.group(1).upper()

    # 2) 대체 허용: X : ... (대괄호 없음)
    m = re.match(r"^\s*([ABCD])\s*[:\|\-]\s*", s, re.I)
    if m:
        return m.group(1).upper()

    # 3) 대괄호-마감 실수: [X] ...:
    m = re.match(r"^\[\s*([ABCD])\s*\]\s*[:\|\-]\s*", s, re.I)
    if m:
        return m.group(1).upper()

    return ""

def normalize_item_in_place(q: dict):
    # 키 이름 보정
    if "question" not in q and "prompt" in q:
        q["question"] = q["prompt"]
    if "choices" not in q and "answers" in q:
        q["choices"] = q["answers"]
    if "answer" not in q and "gold" in q:
        q["answer"] = q["gold"]

    # 선택지 형태 보정 → ABCD dict
    if isinstance(q.get("choices"), list):
        q["choices"] = {LETTERS[i]: q["choices"][i] for i in range(min(4, len(q["choices"])))}
    elif isinstance(q.get("choices"), dict):
        ks = list(q["choices"].keys())
        if any(k not in LETTERS for k in ks):
            vals = list(q["choices"].values())
            q["choices"] = {LETTERS[i]: vals[i] for i in range(min(4, len(vals)))}

    # topic 기본
    if not q.get("topic"):
        q["topic"] = "(미분류)"

def validate_item_or_stop(qid: str, q: dict):
    need = ["question", "choices", "answer"]
    miss = [k for k in need if k not in q]
    if miss:
        st.error(f"[{qid}] 문항 키 누락: {miss}")
        st.json(q); st.stop()
    if not isinstance(q["choices"], dict):
        st.error(f"[{qid}] choices가 dict가 아닙니다.")
        st.json(q); st.stop()
    # 선택지는 A/B/C/D만 유지
    for k in list(q["choices"].keys()):
        if k not in LETTERS:
            del q["choices"][k]
    if len(q["choices"]) < 2:
        st.error(f"[{qid}] 선택지가 부족합니다.")
        st.json(q); st.stop()
    # 4지선다 가정
    if any(k not in q["choices"] for k in LETTERS):
        st.error(f"[{qid}] 토론 모드용 4지선다(ABCD) 형식이 아닙니다.")
        st.json(q); st.stop()

# ================== Ollama 래퍼 ==================
def chat_once(model: str, messages: list, temperature: float, top_p: float,
              keep_alive: str = "5m", **options) -> str:
    """
    한 번 호출. keep_alive로 모델을 메모리에 유지해 반복 호출 비용을 줄임.
    추가 옵션: num_ctx, seed, top_k, repeat_penalty 등 전달 가능.
    """
    opts = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "keep_alive": keep_alive,
    }
    opts.update(options or {})
    res = ollama.chat(model=model, messages=messages, stream=False, options=opts)
    return clean_surrogates(res.get("message", {}).get("content", ""))

# ================== (1) 자유 선택 토론 프롬프트 ==================
def make_free_choice_system(ai_role: str, max_sents: int) -> str:
    return (
        f"역할: {ai_role}\n"
        "목표: 문제의 4개 선택지 중 너가 옳다고 믿는 하나(X∈A,B,C,D)를 스스로 고르고, 그 선택지를 강력히 옹호하라.\n"
        "형식: 출력은 단 한 줄, 정확히 다음 형식을 지켜라 — `[X : 주장]` (대괄호와 콜론 포함, X는 A/B/C/D)\n"
        "규칙:\n"
        f"- 한국어로만, {max_sents}문장 이내로 간결하게.\n"
        "- 자신의 선택(X)을 첫 글자에 대괄호로 명시: 예) [C : ...]\n"
        "- 최소 2개의 다른 선택지(예: A와 D)의 약점을 구체적으로 2가지 이상 지적.\n"
        "- '정답은 X' 같은 메타 발언 금지, 논거와 반박 위주.\n"
        "- 오직 한 줄만 출력하고, 코드펜스/설명 금지.\n"
    )

def make_free_choice_user(q: dict) -> str:
    choices_str = "\n".join([f"{k}: {v}" for k, v in q["choices"].items()])
    return (
        "문제:\n"
        f"{q['question']}\n\n"
        "선택지:\n" + choices_str + "\n\n"
        "지시: 너가 옳다고 믿는 하나의 선택지를 골라 위 형식으로 한 줄만 출력."
    )

# ================== (3) 타당성 채점 프롬프트(품질 점수) ==================
def make_quality_judge_system() -> str:
    return (
        "역할: 토론 채점관\n"
        "목표: 각 발언의 '타당성/논리 일관성/근거의 구체성/반론 처리'를 0~100으로 평가한다.\n"
        "입력은 여러 줄이며 각 줄은 정확히 `[X : 주장]` 형식으로 시작한다(X는 A/B/C/D).\n"
        "출력은 오직 JSON 배열 한 줄. 각 원소는 다음 키를 포함:\n"
        '[{"line":1,"X":"B","quality":78}, ...]\n'
        "설명/코드펜스/여분 텍스트 금지."
    )

def make_quality_judge_user(q: dict, debate_lines: List[str]) -> str:
    choices_str = "\n".join([f"{k}: {v}" for k, v in q["choices"].items()])
    lines_join = "\n".join(debate_lines)
    return (
        "문제와 선택지:\n"
        f"{q['question']}\n" + choices_str + "\n\n"
        "다음은 토론자들의 발언이다(각 줄은 [X : 주장] 형식):\n" +
        lines_join + "\n\n"
        "각 줄의 X와 주장 내용을 고려해 품질 점수(0~100)를 부여하고 JSON 배열만 출력."
    )

def parse_quality_array(payload: str, n_expected: int) -> List[Dict]:
    data = safe_json_loads(payload)
    out = []
    if isinstance(data, list):
        for it in data:
            if not isinstance(it, dict): continue
            line = it.get("line")
            X = it.get("X")
            quality = it.get("quality", it.get("score", it.get("q", None)))
            try:
                line = int(line)
            except Exception:
                continue
            if isinstance(X, str) and X.upper() in LETTERS:
                try:
                    qv = float(quality)
                except Exception:
                    qv = 0.0
                out.append({"line": line, "X": X.upper(), "quality": max(0.0, min(100.0, qv))})
    # 보정: 길이 맞추기
    if len(out) != n_expected:
        # 라인 번호 채우기
        seen = {d["line"] for d in out}
        for i in range(1, n_expected + 1):
            if i not in seen:
                out.append({"line": i, "X": "", "quality": 0.0})
        out.sort(key=lambda x: x["line"])
    return out

# ================== 멀티보이스 생성기(기존 번들: 유지) ==================
def make_bundle_system(max_sents: int) -> str:
    return (
        "너는 네 명의 토론자(A,B,C,D)를 동시에 연기한다.\n"
        "각 토론자는 자신의 선택지만 옹호하고, 다른 선택지의 약점을 최소 2가지 지적한다.\n"
        f"각 발언은 한국어로 {max_sents}문장 이내.\n"
        "출력은 **오직 하나의 JSON 객체**로 하고, 코드펜스/설명 금지.\n"
        '권장 형식: {"A":{"talk":"..."}, "B":{"talk":"..."}, "C":{"talk":"..."}, "D":{"talk":"..."}}\n'
        '허용 형식: {"A":"...", "B":"...", "C":"...", "D":"..."}\n'
    )

def make_bundle_user(q: dict) -> str:
    choices_str = "\n".join([f"{k}: {v}" for k, v in q["choices"].items()])
    return (
        "문제:\n"
        f"{q['question']}\n\n"
        "선택지:\n" + choices_str + "\n\n"
        "위 JSON 형식으로 각 토론자의 발언을 생성하라."
    )

def generate_debate_bundle_single(model: str, q: dict, max_sents: int,
                                  temperature: float, top_p: float,
                                  num_ctx: int = 8192, seed: Optional[int] = None,
                                  top_k: Optional[int] = None, repeat_penalty: Optional[float] = None
                                  ) -> Tuple[List[dict], List[str]]:
    """
    (백업용) A/B/C/D 강제 배정 번들 생성.
    """
    system_prompt = make_bundle_system(max_sents)
    user_msg = make_bundle_user(q)
    extra = {}
    if num_ctx: extra["num_ctx"] = int(num_ctx)
    if seed is not None: extra["seed"] = int(seed)
    if top_k is not None: extra["top_k"] = int(top_k)
    if repeat_penalty is not None: extra["repeat_penalty"] = float(repeat_penalty)

    raw = chat_once(
        model,
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_msg}],
        temperature=temperature, top_p=top_p, **extra
    )
    data = safe_json_loads(raw)
    talks = _normalize_debate_json(data)
    debaters, debate_blocks = [], []
    for i, letter in enumerate(LETTERS):
        ai_role = f"AI{i+1}"
        content = talks.get(letter, "")
        debaters.append({"role": ai_role, "letter": letter, "content": content})
        debate_blocks.append(f"{ai_role} [{letter}]: {content}")
    return debaters, debate_blocks

# ================== (1) 자유 선택 토론 생성기 ==================
def generate_debate_free_choice(q: dict,
                                models_for_roles: List[str],
                                max_sents: int, temperature: float, top_p: float,
                                num_ctx: int = 8192, seed_base: int = 42,
                                top_k: int = 40, repeat_penalty: float = 1.1
                                ) -> Tuple[List[dict], List[str], List[str]]:
    """
    네 명의 AI가 각자 스스로 X∈A/B/C/D를 선택하고 한 줄로 '[X : 주장]'을 생성.
    반환:
      - debaters: [{"role":"AI1","picked":"B","content":"[B : ...]"}, ...]
      - debate_lines: ["[B : ...]","[C : ...]", ...]  # Judge 입력용
      - picked_letters: ["B","C","B","A"]
    """
    debaters, lines, picks = [], [], []
    for i in range(4):
        role = f"AI{i+1}"
        model_name = models_for_roles[i % max(1, len(models_for_roles))]
        sys = make_free_choice_system(role, max_sents)
        usr = make_free_choice_user(q)
        try:
            out = chat_once(
                model_name,
                [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                temperature=temperature, top_p=top_p,
                num_ctx=num_ctx, seed=seed_base + i,
                top_k=top_k, repeat_penalty=repeat_penalty
            ).strip()
        except Exception as e:
            out = f"[A : (오류) 모델 응답 실패: {e}]"

        # 한 줄만 보장: 여러 줄이면 첫 줄만 사용
        first_line = out.splitlines()[0].strip()
        # 대괄호 누락 시 보정 (가능한 경우에만)
        if not first_line.startswith("["):
            # 앞단에 형식 보정 시도: 첫 글자에 선택지가 있으면 감싸기
            guess = extract_leading_choice_from_line(first_line)
            if guess:
                # ❗ f-string 내부에서 re.sub 호출하지 말고, 먼저 계산해서 넣기
                cleaned = re.sub(r'^\s*[ABCD]\s*[:\|\-]\s*', '', first_line, flags=re.I)
                first_line = f"[{guess} : {cleaned}]"
            else:
                # 최후 보정: A로 라벨링
                first_line = f"[A : {first_line}]"

        X = extract_leading_choice_from_line(first_line) or "A"
        debaters.append({"role": role, "picked": X, "content": first_line})
        lines.append(first_line)
        picks.append(X)
    return debaters, lines, picks

# ================== (3) Judge: 품질 점수화 & 가중 카운트 집계 ==================
def judge_quality_and_aggregate(judge_model: str, q: dict, debate_lines: List[str],
                                num_ctx: int = 8192, seed: int = 777) -> Tuple[Dict[str, float], Dict[str, int], str, List[Dict]]:
    """
    - debate_lines: ['[B : ...]', '[C : ...]', ...] (길이 4)
    - Judge가 각 줄의 품질(0~100)을 평가 → 품질 가중 카운트 계산
    반환:
      weighted_counts: {'A': wA, 'B': wB, ...}  # sum(quality/100)
      raw_counts: {'A': nA, 'B': nB, ...}       # 단순 빈도
      raw_judge_payload: Judge 원문
      per_line: [{'line':1,'X':'B','quality':78}, ...]
    """
    # 원시 카운트(형식 파서 기반)
    raw_counts = {L: 0 for L in LETTERS}
    for ln in debate_lines:
        x = extract_leading_choice_from_line(ln)
        if x in raw_counts:
            raw_counts[x] += 1

    instr = make_quality_judge_system()
    usr = make_quality_judge_user(q, debate_lines)
    raw = chat_once(
        judge_model,
        [{"role": "system", "content": instr}, {"role": "user", "content": usr}],
        temperature=0.0, top_p=1.0, num_ctx=num_ctx, seed=seed
    )
    parsed = parse_quality_array(raw, n_expected=len(debate_lines))

    # 가중 카운트 = quality/100을 가산
    weighted_counts = {L: 0.0 for L in LETTERS}
    for d in parsed:
        X = d.get("X", "")
        score = float(d.get("quality", 0.0))
        if X in weighted_counts:
            weighted_counts[X] += max(0.0, min(1.0, score / 100.0))

    return weighted_counts, raw_counts, raw, parsed

# ================== (4) (옵션) 기존 저지(단표/번들) 백업 구현 ==================
def make_judge_payload(q: dict, debate_blocks: List[str], perm: List[str]) -> Tuple[str, str]:
    choices_lines = [f"{k}: {q['choices'][k]}" for k in perm]
    judge_instruction = (
        "너는 채점관이다.\n"
        "출력 형식은 반드시 `<answer>X</answer>` 한 줄만 출력하라. (X는 A/B/C/D 중 하나)\n"
        "설명/번역/추가 텍스트/개행/마크다운 금지. 오직 태그 1줄만."
    )
    # 자유 선택 모드에서도 호환되도록 debate_blocks 그대로 사용
    judge_user = (
        "문제와 선택지(무작위 순서):\n"
        f"Q: {q['question']}\n" + "\n".join(choices_lines) +
        "\n\n다음은 토론자들의 발언 일부이다. 가장 설득력 있는 선택지를 하나만 고르고 형식 지시를 지켜라.\n\n" +
        "\n\n".join(debate_blocks)
    )
    return judge_instruction, judge_user

def _parse_score_json(payload: str) -> Dict[str, float]:
    d = safe_json_loads(payload) or {}
    out = {}
    for k in LETTERS:
        v = d.get(k)
        try:
            out[k] = float(v)
        except Exception:
            out[k] = float("-inf")
    return out

def run_one_judge_vote(judge_model_name: str, q: dict, debate_blocks: List[str],
                       num_ctx: int, seed: int) -> Tuple[str, str]:
    """단표 모드(백업)."""
    perm = LETTERS[:]
    random.shuffle(perm)
    instr, user = make_judge_payload(q, debate_blocks, perm)
    raw = chat_once(
        judge_model_name,
        [{"role": "system", "content": instr}, {"role": "user", "content": user}],
        temperature=0.0, top_p=1.0, num_ctx=num_ctx, seed=seed, stop=["\n"],
    )
    pick = extract_choice_strict(raw) or extract_choice(raw)
    return (pick if pick in LETTERS else ""), raw

# ================== 평가(저지) ==================
def evaluate_one(qid: str, q: dict, debater_models: List[str], judge_model: str,
                 temperature: float, top_p: float, max_sents: int,
                 num_debaters: int, retry_judge: bool = True, domain: Optional[str] = None,
                 multi_voice: bool = True, single_debater_model: Optional[str] = None,
                 # === 정확도 향상 인자들 ===
                 n_debate: int = 1, num_ctx: int = 8192, seed_base: int = 42,
                 top_k: int = 40, repeat_penalty: float = 1.1,
                 # === NEW 자유 선택 모드 ===
                 free_choice_mode: bool = True):
    """
    - free_choice_mode=True: 네 명이 스스로 X를 선택(중복 허용) → Judge가 각 줄 품질을 0~100으로 채점 →
      가중 카운트(quality/100)로 최종 승자 결정.
    - free_choice_mode=False: (백업) 기존 강제 배정 로직 사용 후 단표 저지 1회.
    """
    normalize_item_in_place(q)
    validate_item_or_stop(qid, q)

    last_debaters = []
    last_lines = []
    last_raws = []
    weighted_counts_acc = {L: 0.0 for L in LETTERS}
    raw_counts_acc = {L: 0 for L in LETTERS}
    per_line_all: List[Dict] = []

    for d_idx in range(n_debate):
        # --- 토론 생성 ---
        if free_choice_mode:
            if multi_voice and single_debater_model:
                models_for_roles = [single_debater_model] * 4
            else:
                # 여러 모델이면 순서대로 할당 (AI1→m0, AI2→m1, ...)
                models_for_roles = [debater_models[i % max(1, len(debater_models))] for i in range(4)]

            debaters, debate_lines, picked = generate_debate_free_choice(
                q=q, models_for_roles=models_for_roles,
                max_sents=max_sents, temperature=temperature, top_p=top_p,
                num_ctx=num_ctx, seed_base=seed_base + d_idx * 1000,
                top_k=top_k, repeat_penalty=repeat_penalty
            )
            last_debaters = debaters
            last_lines = debate_lines

            # --- Judge 품질 채점 + 집계 ---
            weighted_counts, raw_counts, raw, per_line = judge_quality_and_aggregate(
                judge_model=judge_model, q=q, debate_lines=debate_lines,
                num_ctx=num_ctx, seed=seed_base + d_idx * 1000 + 123
            )
            last_raws.append(raw)
            for L in LETTERS:
                weighted_counts_acc[L] += weighted_counts[L]
                raw_counts_acc[L] += raw_counts[L]
            # 라인 기록(최근 샘플만 별도 저장)
            per_line_all = per_line

        else:
            # 백업: 기존 강제 배정 번들 + 단표 저지
            if multi_voice and single_debater_model:
                debaters, debate_blocks = generate_debate_bundle_single(
                    single_debater_model, q, max_sents, temperature, top_p,
                    num_ctx=num_ctx, seed=seed_base + d_idx,
                    top_k=top_k, repeat_penalty=repeat_penalty
                )
            else:
                debaters, debate_blocks = [], []
                for i, letter in enumerate(LETTERS):
                    ai_role = f"AI{i+1}"
                    claim = q["choices"][letter]
                    system_prompt = (
                        f"역할: {ai_role}\n"
                        f"담당 선택지: {letter}\n"
                        f"목표: 오직 선택지 {letter}({claim})가 옳다고 강력히 옹호하라.\n\n"
                        "규칙:\n"
                        f"- '{letter}' 외 다른 선택지의 우위 인정 금지.\n"
                        "- 반대 선택지의 약점을 최소 2가지 지적.\n"
                        f"- 한국어로만, {max_sents}문장 이내."
                    )
                    user_msg = (
                        f"{q['question']}\n\n선택지:\n" +
                        "\n".join([f"{k}: {v}" for k, v in q["choices"].items()]) +
                        f"\n\n지시: 선택지 {letter}가 옳은 이유 3가지를 제시하고, 다른 선택지의 약점 2가지를 짚어라."
                    )
                    model_for_role = debater_models[i % max(1, num_debaters)]
                    try:
                        content = chat_once(
                            model_for_role,
                            [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_msg}],
                            temperature=temperature, top_p=top_p,
                            num_ctx=num_ctx, seed=seed_base + d_idx,
                            top_k=top_k, repeat_penalty=repeat_penalty
                        )
                    except Exception as e:
                        content = f"[오류] 모델 응답 실패: {e}"
                    debaters.append({"role": ai_role, "letter": letter, "content": content})
                    debate_blocks.append(f"{ai_role} [{letter}]: {content}")

            last_debaters = debaters
            last_lines = [d.get("content","") for d in debaters]
            # 단표 저지
            pick, raw = run_one_judge_vote(judge_model, q, last_lines, num_ctx=num_ctx, seed=seed_base + d_idx * 10000)
            last_raws.append(raw)
            # 단표 결과를 가중 카운트에 1.0로 반영 (백업 동작)
            if pick in LETTERS:
                weighted_counts_acc[pick] += 1.0
                raw_counts_acc[pick] += 1

    # 최종 결정: 가중 카운트가 최대인 선택지
    final_choice = max(LETTERS, key=lambda L: (weighted_counts_acc[L], raw_counts_acc[L], L))
    correct = (final_choice == q["answer"])

    # 반환용 부가 정보
    # 자유선택 모드에서는 A/B/C/D별 텍스트 고정이 없으므로 공란 유지
    a_text = b_text = c_text = d_text = ""

    judge_log = (
        "=== 품질 기반 가중 집계 ===\n"
        f"weighted_counts={weighted_counts_acc}\nraw_counts={raw_counts_acc}\n"
        f"last_judge_raw={last_raws[-1] if last_raws else ''}"
    )

    return {
        "qid": qid,
        "domain": domain or "",
        "topic": q.get("topic", ""),
        "question": q["question"],
        "gold": q["answer"],
        "judge": final_choice or "",
        "correct": bool(correct),
        "judge_raw": judge_log,
        "A_text": a_text, "B_text": b_text, "C_text": c_text, "D_text": d_text,
        "weighted_counts": weighted_counts_acc,
        "raw_counts": raw_counts_acc,
        "last_lines": last_lines,       # ['[B : ...]', ...]
        "per_line": per_line_all,       # [{'line':1,'X':'B','quality':..}, ...]
    }

# ================== 토픽(문제 유형) 필터 ==================
def topic_of(q: dict) -> str:
    t = (q.get("topic") or "").strip()
    return t if t else "(미분류)"

def list_topics(mmlu_data: dict) -> List[str]:
    return sorted({topic_of(q) for q in mmlu_data.values()})

def filter_qids_by_topic(mmlu_data: dict, qids: List[str], picked_topics: List[str]) -> List[str]:
    picked = set(picked_topics)
    return [qid for qid in qids if topic_of(mmlu_data[qid]) in picked]

# ================== 앱 ==================
st.set_page_config(page_title="MMLU Debate Evaluation (Free-Choice)", layout="wide")
st.sidebar.title("🧠 MMLU 토론 평가 (자유 선택 모드)")

# 데이터 로드
mmlu_data, domain_index, qid2domain = load_questions()
all_qids = list(mmlu_data.keys())
if not all_qids:
    st.error("문항이 없습니다. JSON 파일을 확인하세요."); st.stop()

# 토픽(문제 유형) 필터
st.sidebar.markdown("### 🗂 문제 유형(토픽) 필터")
all_topics = list_topics(mmlu_data)
picked_topics = st.sidebar.multiselect("유형 선택(복수 가능)", all_topics, default=all_topics)

# Ollama 준비 & 모델 목록
check_ollama()
try:
    model_list = [m["model"] for m in ollama.list()["models"]]
except Exception as e:
    st.error(f"Ollama 모델 목록 오류: {e}"); st.stop()
if not model_list:
    st.error("설치된 모델이 없습니다. `ollama pull mistral` 등으로 설치하세요."); st.stop()

# 공통 하이퍼파라미터
st.sidebar.markdown("### ⚙️ 공통 설정")
temperature = st.sidebar.slider("temperature", 0.0, 1.5, 0.2, 0.1)
top_p = st.sidebar.slider("top_p", 0.1, 1.0, 0.9, 0.05)
max_sents = st.sidebar.slider("발언 문장 수(권장 최대)", 3, 8, 6, 1)

# 1인다역 여부
st.sidebar.markdown("---")
multi_voice = st.sidebar.checkbox("⚡ 1인 다역(한 모델로 4명 모두 생성)", value=True)

# 자유 선택 모드(중복 허용) — 기본 ON
st.sidebar.markdown("---")
free_choice_mode = st.sidebar.checkbox(
    "🧠 자유 선택 토론(중복 허용)", value=True,
    help="각 AI가 스스로 A/B/C/D 중 하나를 선택해 '[X : 주장]'을 생성. 같은 선택지를 여러 명이 골라도 됩니다."
)

# 모델 선택
st.sidebar.markdown("### 🤖 토론자 & 저지 모델")
if multi_voice:
    single_debater_model = st.sidebar.selectbox("토론자(1인 다역) 모델", model_list, key="single_deb_model")
    debater_models = [single_debater_model]
    num_debaters = 1
else:
    num_debaters = st.sidebar.slider("토론자 수", 1, 4, 4)
    debater_models = [st.sidebar.selectbox(f"토론자 모델 {i+1}", model_list, key=f"deb_model_{i+1}") for i in range(num_debaters)]
    single_debater_model = None

judge_default = model_list.index("mistral") if "mistral" in model_list else 0
judge_model = st.sidebar.selectbox("Judge(품질 채점) 모델", model_list, index=judge_default)

# 정확도 향상 옵션
st.sidebar.markdown("---")
n_debate = st.sidebar.slider("샘플 반복 수(자기일관성)", 1, 5, 1)
num_ctx = st.sidebar.number_input("num_ctx(컨텍스트)", 2048, 32768, 8192, step=1024)
top_k = st.sidebar.number_input("top_k", 16, 200, 40, step=8)
repeat_penalty = st.sidebar.number_input("repeat_penalty", 1.0, 2.0, 1.1, step=0.05)
seed_base = st.sidebar.number_input("seed(재현성)", 0, 10_000_000, 42, step=1)

# 실행 모드
mode = st.radio("실행 모드 선택", ["단일 문제", "전체 평가"], horizontal=True)

# ---------------- 단일 문제 ----------------
if mode == "단일 문제":
    filtered_qids = filter_qids_by_topic(mmlu_data, all_qids, picked_topics)
    if not filtered_qids:
        st.warning("선택된 토픽에 해당하는 문항이 없습니다. 사이드바에서 토픽을 조정하세요."); st.stop()

    topics_for_pick = sorted({topic_of(mmlu_data[qid]) for qid in filtered_qids})
    chosen_topic = st.selectbox("문제 유형(토픽)", topics_for_pick)
    topic_qids = [qid for qid in filtered_qids if topic_of(mmlu_data[qid]) == chosen_topic]

    qid = st.selectbox("문제 선택", topic_qids, key="single_qid")
    q = mmlu_data[qid]

    normalize_item_in_place(q)
    validate_item_or_stop(qid, q)

    st.caption(f"유형: {topic_of(q)}")
    st.markdown(f"### ❓ {q['question']}")
    st.markdown("**선택지**")
    for k in LETTERS:
        st.markdown(f"- **{k}**: {q['choices'][k]}")

    if st.button("🚀 토론 시작 + 품질 기반 Judge"):
        with st.spinner("토론 생성 중..."):
            if free_choice_mode:
                # 모델 리스트 구성
                if multi_voice and single_debater_model:
                    models_for_roles = [single_debater_model] * 4
                else:
                    models_for_roles = [debater_models[i % max(1, len(debater_models))] for i in range(4)]
                debaters, debate_lines, picks = generate_debate_free_choice(
                    q=q, models_for_roles=models_for_roles,
                    max_sents=max_sents, temperature=temperature, top_p=top_p,
                    num_ctx=num_ctx, seed_base=seed_base, top_k=top_k, repeat_penalty=repeat_penalty
                )
            else:
                # 백업: 번들 강제 모드
                debaters, debate_blocks = generate_debate_bundle_single(
                    (single_debater_model or debater_models[0]),
                    q, max_sents, temperature, top_p,
                    num_ctx=num_ctx, seed=seed_base, top_k=top_k, repeat_penalty=repeat_penalty
                )
                debate_lines = [d["content"] for d in debaters]

        st.subheader("💬 AI 발언 (자유 선택)")
        for d in debaters:
            st.markdown(f"**{d['role']}**  선택: **{d['picked']}**")
            st.info(d["content"])

        # Judge 품질 채점 + 가중 집계
        st.subheader("⚖️ Judge 품질 채점 & 가중 집계")
        weighted_counts, raw_counts, raw_judge, per_line = judge_quality_and_aggregate(
            judge_model=judge_model, q=q, debate_lines=debate_lines,
            num_ctx=num_ctx, seed=seed_base + 9999
        )

        # 결과 표
        wrow = {L: f"{weighted_counts[L]:.2f}" for L in LETTERS}
        rrow = {L: raw_counts[L] for L in LETTERS}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**가중 카운트(quality/100 합)**")
            st.table(pd.DataFrame([wrow], index=["weighted"]))
        with c2:
            st.markdown("**단순 빈도(선택 수)**")
            st.table(pd.DataFrame([rrow], index=["raw"]))

        # 최종 판단
        final_choice = max(LETTERS, key=lambda L: (weighted_counts[L], raw_counts[L], L))
        st.markdown(f"**최종 선택(가중 기준): {final_choice}**")
        st.markdown(f"**정답:** {q['answer']}")
        st.success("정답과 일치합니다! ✅") if final_choice == q["answer"] else st.error("정답과 불일치! ❌")

        with st.expander("Judge 원문(JSON) 보기"):
            st.code(raw_judge)
        with st.expander("라인별 품질 점수"):
            st.json(per_line)

# ---------------- 전체 평가 ----------------
else:
    st.markdown("### 📚 전체 평가 설정")

    filtered_qids = filter_qids_by_topic(mmlu_data, all_qids, picked_topics)
    if not filtered_qids:
        st.warning("선택된 토픽에 해당하는 문항이 없습니다. 사이드바에서 토픽을 조정하세요."); st.stop()

    random_order = st.checkbox("문제 순서 섞기", value=False)
    seed = st.number_input("랜덤 시드", value=42, step=1)
    max_count = st.slider("평가할 문제 개수", 1, len(filtered_qids), len(filtered_qids))
    show_logs = st.checkbox("문항별 상세 로그 표시", value=False)

    selected_qids = filtered_qids.copy()
    if random_order:
        random.Random(seed).shuffle(selected_qids)
    selected_qids = selected_qids[:max_count]

    if st.button("🧪 전체 평가 시작"):
        results, start_time = [], time.time()
        progress = st.progress(0, text="시작 중...")

        for idx, qid in enumerate(selected_qids, start=1):
            q = mmlu_data[qid]
            normalize_item_in_place(q)
            validate_item_or_stop(qid, q)

            progress.progress(idx / max_count, text=f"{idx}/{max_count} 평가 중: {qid}")

            res = evaluate_one(
                qid=qid, q=q,
                debater_models=(debater_models if not (multi_voice and single_debater_model) else [single_debater_model]),
                judge_model=judge_model,
                temperature=temperature, top_p=top_p,
                max_sents=max_sents, num_debaters=(1 if (multi_voice and single_debater_model) else len(debater_models)),
                retry_judge=True, domain=qid2domain.get(qid),
                multi_voice=multi_voice, single_debater_model=single_debater_model,
                n_debate=n_debate, num_ctx=num_ctx, seed_base=seed_base + idx * 10000,
                top_k=top_k, repeat_penalty=repeat_penalty,
                free_choice_mode=free_choice_mode
            )
            results.append(res)

            if show_logs:
                with st.expander(f"[{idx}] {qid} | 분야: {qid2domain.get(qid, q.get('topic',''))} | 정답: {res['gold']} | 최종: {res['judge']} | {'✅' if res['correct'] else '❌'}"):
                    st.write(q["question"])
                    st.write({k: v for k, v in q["choices"].items()})
                    st.markdown("**AI 발언(최근 샘플)**")
                    for line in res.get("last_lines", []):
                        st.info(line)
                    st.markdown("**가중/빈도 카운트**")
                    st.json({"weighted_counts": res.get("weighted_counts"), "raw_counts": res.get("raw_counts")})
                    st.markdown("**Judge 원문/JSON(일부)**"); st.code(res.get("judge_raw",""))
                    st.markdown("**라인별 점수(최근)**"); st.json(res.get("per_line", []))

        total_time = time.time() - start_time
        progress.progress(1.0, text="완료")

        # 요약
        df = pd.DataFrame(results)
        st.markdown("## 📈 결과 요약")
        if df.empty:
            st.warning("결과가 비어 있습니다."); st.stop()
        acc = df["correct"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("총 문항", len(df))
        c2.metric("정확도(%)", f"{acc*100:.1f}")
        c3.metric("총 소요시간(초)", f"{total_time:.1f}")

        # 분야별/토픽별 정확도
        st.markdown("### 🧭 분야별 정확도 (domain)")
        if "domain" in df.columns:
            dom_acc = df.groupby("domain")["correct"].mean().sort_values(ascending=False)
            st.dataframe(dom_acc.to_frame("accuracy").assign(accuracy=lambda x: (x["accuracy"]*100).round(1)))

        st.markdown("### 🧭 주제별 정확도 (topic)")
        if "topic" in df.columns:
            topic_acc = df.groupby("topic")["correct"].mean().sort_values(ascending=False)
            st.dataframe(topic_acc.to_frame("accuracy").assign(accuracy=lambda x: (x["accuracy"]*100).round(1)))

        # 혼동 행렬 (가중 최종판단 기준)
        st.markdown("### 🔁 혼동 행렬 (최종 판단 vs 정답)")
        cm = pd.crosstab(df["gold"], df["judge"], dropna=False).reindex(index=LETTERS, columns=LETTERS, fill_value=0)
        st.dataframe(cm)

        # CSV 다운로드
        st.markdown("### ⬇️ 결과 저장")
        # 불러오기 쉬운 필드만 저장
        save_cols = ["qid","domain","topic","question","gold","judge","correct","weighted_counts","raw_counts"]
        for c in save_cols:
            if c not in df.columns:
                df[c] = None
        csv = df[save_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("결과 CSV 다운로드", data=csv, file_name="mmlu_freechoice_results.csv", mime="text/csv")

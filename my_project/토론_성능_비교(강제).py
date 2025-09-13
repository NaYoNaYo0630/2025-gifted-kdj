

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
    m = re.search(r"\{.*\}", payload, re.S)
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
    Judge 전용: 태그 한 줄 or '정답/answer/final: X' 한 줄만 인정.
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

# ================== 토론 프롬프트 ==================
def make_debater_system(ai_role: str, letter: str, claim: str, max_sents: int) -> str:
    return (
        f"역할: {ai_role}\n"
        f"담당 선택지: {letter}\n"
        f"목표: 오직 선택지 {letter}({claim})가 옳다고 강력히 옹호하라.\n\n"
        "규칙:\n"
        f"- '{letter}' 외 다른 선택지가 더 낫다고 말하지 마라.\n"
        "- '정답은 C' 같은 표현 금지.\n"
        "- 반대 선택지의 약점을 최소 2가지 지적하라.\n"
        f"- 한국어로만, {max_sents}문장 이내로 간결하게.\n"
    )

def make_debater_user(q: dict, letter: str) -> str:
    choices_str = "\n".join([f"{k}: {v}" for k, v in q["choices"].items()])
    return (
        f"{q['question']}\n\n"
        "선택지:\n" + choices_str + "\n\n"
        f"지시: 선택지 {letter}가 옳은 이유 3가지를 제시하고, 다른 선택지의 약점 2가지를 짚어라."
    )

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

# ================== 멀티보이스 생성기 ==================
def generate_debate_bundle_single(model: str, q: dict, max_sents: int,
                                  temperature: float, top_p: float,
                                  num_ctx: int = 8192, seed: Optional[int] = None,
                                  top_k: Optional[int] = None, repeat_penalty: Optional[float] = None
                                  ) -> Tuple[List[dict], List[str]]:
    """
    한 번의 호출로 A/B/C/D 발언을 JSON으로 받아 debaters 리스트와 문자열 블록을 반환.
    중첩·평평 JSON 모두 허용.
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

# ================== 저지 프롬프트(태그 모드) ==================
def make_judge_payload(q: dict, debate_blocks: List[str], perm: List[str]) -> Tuple[str, str]:
    # 선택지/발언을 같은 무작위 순서로 재배열
    choices_lines = [f"{k}: {q['choices'][k]}" for k in perm]
    letter2text = {}
    for line in debate_blocks:
        m = re.match(r"^(.+?)\s+\[([ABCD])\]:\s*(.*)$", line, re.S)
        if m:
            letter2text[m.group(2)] = (m.group(1), m.group(3))
    debate_lines = []
    for L in perm:
        role, txt = letter2text.get(L, (f"AI?{L}", ""))  # 안전
        debate_lines.append(f"{role} [{L}]: {txt}")

    judge_instruction = (
        "너는 채점관이다.\n"
        "출력 형식은 반드시 `<answer>X</answer>` 한 줄만 출력하라. (X는 A/B/C/D 중 하나)\n"
        "설명/번역/추가 텍스트/개행/마크다운 금지. 오직 태그 1줄만."
    )
    judge_user = (
        "문제와 선택지(무작위 순서):\n"
        f"Q: {q['question']}\n" +
        "\n".join(choices_lines) +
        "\n\n다음은 각 AI의 주장(무작위 순서)이다. 논리성/근거의 구체성/반론 처리 관점에서 가장 설득력 있는 하나를 고르고, "
        "형식 지시를 지켜서 답하라.\n\n" +
        "\n\n".join(debate_lines)
    )
    return judge_instruction, judge_user

# ================== (선택) 점수 JSON 저지 모드 ==================
def make_judge_scores_payload(q: dict, debate_blocks: List[str], perm: List[str]) -> Tuple[str, str]:
    choices_lines = [f"{k}: {q['choices'][k]}" for k in perm]
    letter2text = {}
    for line in debate_blocks:
        m = re.match(r"^(.+?)\s+\[([ABCD])\]:\s*(.*)$", line, re.S)
        if m:
            letter2text[m.group(2)] = (m.group(1), m.group(3))
    debate_lines = []
    for L in perm:
        role, txt = letter2text.get(L, (f"AI?{L}", ""))  # 안전
        debate_lines.append(f"{role} [{L}]: {txt}")

    judge_instruction = (
        "너는 채점관이다.\n"
        "각 선택지 A/B/C/D에 대해 0~100 점수를 매겨라(높을수록 설득력). "
        "출력은 오직 아래 JSON 한 줄만 허용(코드펜스/주석/문장 금지).\n"
        '{"A": 0, "B": 0, "C": 0, "D": 0}'
    )
    judge_user = (
        "문제와 선택지(무작위 순서):\n"
        f"Q: {q['question']}\n" + "\n".join(choices_lines) +
        "\n\n다음은 각 AI의 주장(무작위 순서)이다. 점수만 JSON으로 내라.\n\n" +
        "\n\n".join(debate_lines)
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
                       mode_scores: bool, num_ctx: int, seed: int) -> Tuple[str, str]:
    """단표 모드. mode_scores=True면 JSON 점수, False면 태그 1줄."""
    perm = LETTERS[:]
    random.shuffle(perm)

    if mode_scores:
        instr, user = make_judge_scores_payload(q, debate_blocks, perm)
        raw = chat_once(
            judge_model_name,
            [{"role": "system", "content": instr}, {"role": "user", "content": user}],
            temperature=0.0, top_p=1.0, num_ctx=num_ctx, seed=seed,
        )
        scores = _parse_score_json(raw)
        best = max(LETTERS, key=lambda L: (scores.get(L, float("-inf")), L))
        return best, raw
    else:
        instr, user = make_judge_payload(q, debate_blocks, perm)
        raw = chat_once(
            judge_model_name,
            [{"role": "system", "content": instr}, {"role": "user", "content": user}],
            temperature=0.0, top_p=1.0, num_ctx=num_ctx, seed=seed, stop=["\n"],
        )
        pick = extract_choice_strict(raw) or extract_choice(raw)
        if pick not in LETTERS:
            raw2 = chat_once(
                judge_model_name,
                [{"role": "system", "content": instr + "\n반드시 예: <answer>C</answer> 형식."},
                 {"role": "user", "content": user}],
                temperature=0.0, top_p=1.0, num_ctx=num_ctx, seed=seed + 999, stop=["\n"],
            )
            pick = extract_choice_strict(raw2) or extract_choice(raw2)
            raw = raw2 if pick in LETTERS else raw
        return (pick if pick in LETTERS else ""), raw

# ================== NEW: 1인N역 번들 저지 ==================
def make_judge_bundle_payload(q: dict, debate_blocks: List[str], perm: List[str], size: int = 5) -> Tuple[str, str]:
    """J1..J{size} 심사위원이 동시에 한 표씩(A/B/C/D) 뽑아 JSON으로만 출력."""
    choices_lines = [f"{k}: {q['choices'][k]}" for k in perm]

    # 토론문 재배열
    letter2text = {}
    for line in debate_blocks:
        m = re.match(r"^(.+?)\s+\[([ABCD])\]:\s*(.*)$", line, re.S)
        if m:
            letter2text[m.group(2)] = (m.group(1), m.group(3))
    debate_lines = []
    for L in perm:
        role, txt = letter2text.get(L, (f"AI?{L}", ""))
        debate_lines.append(f"{role} [{L}]: {txt}")

    judge_instruction = (
        f"너는 서로 독립적인 {size}명의 심사위원(J1..J{size})을 동시에 연기한다.\n"
        "각 심사위원은 가장 설득력 있는 선택지 한 개(A/B/C/D)만 고른다.\n"
        "출력은 **오직 하나의 JSON 객체**로, 예시는 다음과 같다(설명/코드펜스/빈줄 금지):\n"
        '{"J1":"A","J2":"C","J3":"B","J4":"C","J5":"D"}'
    )
    judge_user = (
        "문제와 선택지(무작위 순서):\n"
        f"Q: {q['question']}\n" + "\n".join(choices_lines) +
        "\n\n다음은 각 AI의 주장(무작위 순서)이다. 각 심사위원의 선택만 JSON으로 출력하라.\n\n" +
        "\n\n".join(debate_lines)
    )
    return judge_instruction, judge_user

def _extract_pick(v) -> str:
    """값에서 A/B/C/D 1글자만 뽑기."""
    if isinstance(v, str):
        m = re.search(r"\b([ABCD])\b", v.upper())
        return m.group(1) if m else ""
    if isinstance(v, dict):
        for kk in ("pick", "answer", "choice", "final"):
            s = v.get(kk)
            if isinstance(s, str) and s.upper() in LETTERS:
                return s.upper()
    return ""

def _parse_bundle_vote_json(payload: str, size: int = 5) -> List[str]:
    """{"J1":"A",...} 같이 온 JSON을 [p1..pN] 리스트로 변환. 키는 대소문자/공백 허용."""
    d = safe_json_loads(payload) or {}
    picks = [""] * size
    if not isinstance(d, dict):
        return picks

    for k, v in d.items():
        kstr = str(k)
        m = re.match(r"^\s*(?:J|Judge)?\s*(\d+)\s*$", kstr, re.I)
        if not m:
            continue
        idx = int(m.group(1))
        if 1 <= idx <= size:
            picks[idx - 1] = _extract_pick(v)
    return picks

def run_bundle_judges(judge_model_name: str, q: dict, debate_blocks: List[str],
                      size: int, num_ctx: int, seed: int) -> Tuple[List[str], str]:
    """저지 1인N역: 한 번의 호출로 N표 반환."""
    perm = LETTERS[:]
    random.shuffle(perm)
    instr, user = make_judge_bundle_payload(q, debate_blocks, perm, size=size)
    raw = chat_once(
        judge_model_name,
        [{"role": "system", "content": instr}, {"role": "user", "content": user}],
        temperature=0.0, top_p=1.0, num_ctx=num_ctx, seed=seed
    )
    picks = _parse_bundle_vote_json(raw, size=size)
    return picks, raw

# ================== 평가(저지) ==================
def evaluate_one(qid: str, q: dict, debater_models: List[str], judge_model: str,
                 temperature: float, top_p: float, max_sents: int,
                 num_debaters: int, retry_judge: bool = True, domain: Optional[str] = None,
                 multi_voice: bool = True, single_debater_model: Optional[str] = None,
                 # === 정확도 향상 인자들 ===
                 n_debate: int = 1, n_judge: int = 1, judge_models_multi: Optional[List[str]] = None,
                 use_score_mode: bool = False, num_ctx: int = 8192, seed_base: int = 42,
                 top_k: int = 40, repeat_penalty: float = 1.1,
                 # === NEW 번들 저지 ===
                 use_bundle_judge: bool = True, judge_bundle_size: int = 5):
    """다중 토론·다중 저지 앙상블 (번들 저지 지원)."""
    judge_models_multi = judge_models_multi or [judge_model]

    normalize_item_in_place(q)
    validate_item_or_stop(qid, q)

    tally = {L: 0 for L in LETTERS}
    last_debaters, last_blocks = [], []
    last_raws = []

    for d_idx in range(n_debate):
        # --- 토론 샘플 생성 ---
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
                system_prompt = make_debater_system(ai_role, letter, claim, max_sents)
                user_msg = make_debater_user(q, letter)
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

        # --- 저지 ---
        if use_bundle_judge:
            jm = judge_models_multi[0]  # 번들 모드에서는 대표 1개 모델만 사용(필요시 라운드 로빈 가능)
            picks, raw = run_bundle_judges(
                jm, q, debate_blocks, size=judge_bundle_size,
                num_ctx=num_ctx, seed=seed_base + 10_000 * d_idx
            )
            last_raws.append(f"[bundle {jm} x{judge_bundle_size}] {raw}")
            for p in picks:
                if p in LETTERS:
                    tally[p] += 1
        else:
            for v in range(n_judge):
                jm = judge_models_multi[v % len(judge_models_multi)]
                pick, raw = run_one_judge_vote(
                    jm, q, debate_blocks,
                    mode_scores=use_score_mode,
                    num_ctx=num_ctx,
                    seed=seed_base + 10_000 * d_idx + v
                )
                last_raws.append(f"[{jm}] {raw}")
                if pick in LETTERS:
                    tally[pick] += 1

        last_debaters, last_blocks = debaters, debate_blocks

    final_choice = max(LETTERS, key=lambda L: (tally[L], L))
    correct = (final_choice == q["answer"])

    a_text = next((d["content"] for d in last_debaters if d["letter"] == "A"), "")
    b_text = next((d["content"] for d in last_debaters if d["letter"] == "B"), "")
    c_text = next((d["content"] for d in last_debaters if d["letter"] == "C"), "")
    d_text = next((d["content"] for d in last_debaters if d["letter"] == "D"), "")

    return {
        "qid": qid,
        "domain": domain or "",
        "topic": q.get("topic", ""),
        "question": q["question"],
        "gold": q["answer"],
        "judge": final_choice or "",
        "correct": bool(correct),
        "judge_raw": f"tally={tally}\n" + "\n".join(last_raws[-min(len(last_raws), 5):]),
        "A_text": a_text, "B_text": b_text, "C_text": c_text, "D_text": d_text,
        "perm": "",
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
st.set_page_config(page_title="MMLU Debate Evaluation", layout="wide")
st.sidebar.title("🧠 MMLU 토론 평가")

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

# === (옵션) 사이드바: 번호 형식 의견 생성기 ===
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🧪 번호 형식 의견 생성")
    N_num = st.number_input("줄 수(N)", min_value=2, max_value=6, value=2, step=1, key="sb_lines")
    topic_num = st.text_input("주제(예: 무슨 옷을 입을까?)", key="sb_topic_numbered")
    default_name = "gemma3:latest"
    default_idx = model_list.index(default_name) if default_name in model_list else 0
    gen_model = st.selectbox("실행 모델", model_list, index=default_idx, key="sb_model_numbered")
    sb_temp = st.slider("temperature(opinion)", 0.0, 1.5, 0.6, 0.1, key="sb_temp_numbered")
    sb_topp = st.slider("top_p(opinion)", 0.1, 1.0, 0.95, 0.05, key="sb_topp_numbered")
    if st.button("▶ 번호 형식 생성", key="sb_make_numbered"):
        if not (topic_num or "").strip():
            st.warning("주제를 입력하세요.")
        else:
            sys = (
                "너는 사용자 주제에 대해 서로 대비되는 여러 입장을 만든다.\n"
                f"출력은 **오직 {N_num}줄**, 각 줄은 숫자와 점으로 시작해야 한다. 다른 말/코드펜스/빈 줄 금지.\n"
                f"형식 예시: 1. …\\n2. …\\n...\\n{N_num}. …\n"
                "한국어만 사용."
            )
            usr = f"주제: {topic_num}"
            raw = chat_once(
                gen_model,
                [{"role": "system", "content": sys},
                 {"role": "user", "content": usr}],
                temperature=sb_temp, top_p=sb_topp
            )
            text = (raw or "").strip()
            pairs = re.findall(r"(?m)^\s*(\d+)\.\s*(.+?)\s*$", text)
            by_num = {}
            for num_str, content in pairs:
                try:
                    k = int(num_str)
                except ValueError:
                    continue
                if 1 <= k <= N_num and k not in by_num:
                    by_num[k] = content.strip()
            if len(by_num) < N_num:
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                for l in lines:
                    if len(by_num) >= N_num:
                        break
                    if not re.match(r"^\d+\.\s*", l):
                        by_num[len(by_num) + 1] = l
            contents = [by_num.get(i, "") for i in range(1, int(N_num) + 1)]
            final = topic_num.strip() + "\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(contents, 1))
            st.markdown("**결과**")
            st.code(final)

# 공통 하이퍼파라미터
st.sidebar.markdown("### ⚙️ 공통 설정")
temperature = st.sidebar.slider("temperature", 0.0, 1.5, 0.2, 0.1)
top_p = st.sidebar.slider("top_p", 0.1, 1.0, 0.9, 0.05)
max_sents = st.sidebar.slider("발언 문장 수(권장 최대)", 3, 8, 6, 1)

# 모드 전환: 1인 다역(멀티보이스)
st.sidebar.markdown("---")
multi_voice = st.sidebar.checkbox("⚡ 1인 다역(한 모델로 A/B/C/D 생성, 1콜)", value=True)

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
judge_model = st.sidebar.selectbox("Judge 기본 모델", model_list, index=judge_default)

# === 정확도 향상 옵션 ===
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 정확도 향상 옵션")
n_debate = st.sidebar.slider("Debate 샘플 수(자기일관성)", 1, 5, 3)
# 번들 저지: 1인5역 고정
use_bundle_judge = st.sidebar.checkbox("1인 N역 번들 저지 사용", value=True)
judge_bundle_size = st.sidebar.number_input("번들 저지 인원(N)", min_value=5, max_value=5, value=5, step=0, help="요청에 따라 5로 고정")
# (비번들용) 저지 투표수/모드
n_judge = st.sidebar.slider("저지 투표 수(비번들 모드용)", 1, 9, 5, step=2)
scoring_mode = st.sidebar.selectbox("비번들 저지 방식", ["태그 1줄(`<answer>X</answer>`)","JSON 점수({\"A\":0-100,...})"], index=0)

judge_models_multi = st.sidebar.multiselect(
    "저지 모델(복수 선택 가능, 번들은 첫 모델만 사용)",
    model_list,
    default=[judge_model] if judge_model in model_list else []
)
if not judge_models_multi:
    judge_models_multi = [judge_model]

# 모델 옵션 보강
num_ctx = st.sidebar.number_input("num_ctx(컨텍스트)", 2048, 32768, 8192, step=1024)
top_k = st.sidebar.number_input("top_k", 16, 200, 40, step=8)
repeat_penalty = st.sidebar.number_input("repeat_penalty", 1.0, 2.0, 1.1, step=0.05)
seed_base = st.sidebar.number_input("seed(재현성)", 0, 10_000_000, 42, step=1)
st.sidebar.caption("번들 저지: 한 번 호출로 5표 생성 → 속도/안정성 향상")

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

    if st.button("🚀 토론 시작 + 저지 채점"):
        with st.spinner("토론 생성 중..."):
            if multi_voice and single_debater_model:
                debaters, debate_blocks = generate_debate_bundle_single(
                    single_debater_model, q, max_sents, temperature, top_p,
                    num_ctx=num_ctx, seed=seed_base, top_k=top_k, repeat_penalty=repeat_penalty
                )
            else:
                debaters, debate_blocks = [], []
                for i, letter in enumerate(LETTERS):
                    ai_role = f"AI{i+1}"
                    claim = q["choices"][letter]
                    system_prompt = make_debater_system(ai_role, letter, claim, max_sents)
                    user_msg = make_debater_user(q, letter)
                    model_for_role = debater_models[i % max(1, num_debaters)]
                    try:
                        content = chat_once(
                            model_for_role,
                            [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_msg}],
                            temperature=temperature, top_p=top_p,
                            num_ctx=num_ctx, seed=seed_base, top_k=top_k, repeat_penalty=repeat_penalty
                        )
                    except Exception as e:
                        content = f"[오류] 모델 응답 실패: {e}"
                    debaters.append({"role": ai_role, "letter": letter, "content": content})
                    debate_blocks.append(f"{ai_role} [{letter}]: {content}")

        st.subheader("💬 AI 발언")
        for d in debaters:
            st.markdown(f"**{d['role']} ({d['letter']} 주장)**")
            st.info(d["content"])

        # 저지 (앙상블)
        st.subheader("⚖️ Judge 모델 판단")
        res = evaluate_one(
            qid=qid, q=q,
            debater_models=debater_models,
            judge_model=judge_model,
            temperature=temperature, top_p=top_p,
            max_sents=max_sents, num_debaters=(num_debaters if not multi_voice else 1),
            retry_judge=True, domain=qid2domain.get(qid),
            multi_voice=multi_voice, single_debater_model=single_debater_model,
            # 정확도 향상 인자
            n_debate=n_debate, n_judge=n_judge, judge_models_multi=judge_models_multi,
            use_score_mode=(scoring_mode.startswith("JSON")),
            num_ctx=num_ctx, seed_base=seed_base, top_k=top_k, repeat_penalty=repeat_penalty,
            # 번들 저지(1인5역)
            use_bundle_judge=use_bundle_judge, judge_bundle_size=judge_bundle_size
        )

        final_choice = res["judge"]
        judge_raw = res["judge_raw"]

        if not final_choice:
            st.error("Judge 출력에서 A/B/C/D를 파싱하지 못했습니다.")
        else:
            st.markdown(f"**Judge 선택:** {final_choice}")
            st.markdown(f"**정답:** {q['answer']}")
            st.success("정답과 일치합니다! ✅") if final_choice == q["answer"] else st.error("정답과 불일치! ❌")
            with st.expander("저지 로그 보기"):
                st.code(judge_raw)

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
                debater_models=debater_models,
                judge_model=judge_model,
                temperature=temperature, top_p=top_p,
                max_sents=max_sents, num_debaters=(num_debaters if not multi_voice else 1),
                retry_judge=True, domain=qid2domain.get(qid),
                multi_voice=multi_voice, single_debater_model=single_debater_model,
                # 정확도 향상 인자
                n_debate=n_debate, n_judge=n_judge, judge_models_multi=judge_models_multi,
                use_score_mode=(scoring_mode.startswith("JSON")),
                num_ctx=num_ctx, seed_base=seed_base, top_k=top_k, repeat_penalty=repeat_penalty,
                # 번들 저지(1인5역)
                use_bundle_judge=use_bundle_judge, judge_bundle_size=judge_bundle_size
            )
            results.append(res)

            if show_logs:
                with st.expander(f"[{idx}] {qid} | 분야: {qid2domain.get(qid, q.get('topic',''))} | 정답: {res['gold']} | Judge: {res['judge']} | {'✅' if res['correct'] else '❌'}"):
                    st.write(q["question"])
                    st.write({k: v for k, v in q["choices"].items()})
                    st.markdown("**Judge 원문/로그(일부)**"); st.code(res["judge_raw"])
                    st.markdown("**AI1 (A) 발언**"); st.info(res["A_text"])
                    st.markdown("**AI2 (B) 발언**"); st.info(res["B_text"])
                    st.markdown("**AI3 (C) 발언**"); st.info(res["C_text"])
                    st.markdown("**AI4 (D) 발언**"); st.info(res["D_text"])

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

        # 혼동 행렬
        st.markdown("### 🔁 혼동 행렬 (Judge vs 정답)")
        cm = pd.crosstab(df["gold"], df["judge"], dropna=False).reindex(index=LETTERS, columns=LETTERS, fill_value=0)
        st.dataframe(cm)

        # CSV 다운로드
        st.markdown("### ⬇️ 결과 저장")
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("결과 CSV 다운로드", data=csv, file_name="mmlu_batch_results.csv", mime="text/csv")

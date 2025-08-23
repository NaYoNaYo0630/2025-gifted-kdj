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

# ================== 상수 ==================
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
    # 가장 바깥 { ... } 블록만 추출
    m = re.search(r"\{.*\}", payload, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

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
def chat_once(model: str, messages: list, temperature: float, top_p: float, keep_alive: str = "5m", **options) -> str:
    """
    한 번 호출. keep_alive로 모델을 메모리에 유지해 반복 호출 비용을 줄임.
    """
    opts = {"temperature": float(temperature), "top_p": float(top_p), "keep_alive": keep_alive}
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
        "출력은 **오직 아래 JSON 한 덩어리**로만 하라. 코드펜스/주석/설명 금지.\n"
        '{\n'
        '  "A": {"talk": "..."},\n'
        '  "B": {"talk": "..."},\n'
        '  "C": {"talk": "..."},\n'
        '  "D": {"talk": "..."}}\n'
    )

def make_bundle_user(q: dict) -> str:
    choices_str = "\n".join([f"{k}: {v}" for k, v in q["choices"].items()])
    return (
        "문제:\n"
        f"{q['question']}\n\n"
        "선택지:\n" + choices_str + "\n\n"
        "각 토론자의 발언을 위 JSON 형식으로 생성하라."
    )

# ================== 멀티보이스 생성기 ==================
def generate_debate_bundle_single(model: str, q: dict, max_sents: int, temperature: float, top_p: float) -> Tuple[List[dict], List[str]]:
    """
    한 번의 호출로 A/B/C/D 발언을 JSON으로 받아 debaters 리스트와 문자열 블록을 반환.
    """
    system_prompt = make_bundle_system(max_sents)
    user_msg = make_bundle_user(q)
    raw = chat_once(model, [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg}],
                    temperature=temperature, top_p=top_p)
    data = safe_json_loads(raw)
    debaters, debate_blocks = [], []
    if not isinstance(data, dict):
        # 실패 시 안전하게 빈 텍스트로 채움
        data = {k: {"talk": ""} for k in LETTERS}

    for i, letter in enumerate(LETTERS):
        content = str((data.get(letter) or {}).get("talk") or "")
        ai_role = f"AI{i+1}"
        debaters.append({"role": ai_role, "letter": letter, "content": content})
        debate_blocks.append(f"{ai_role} [{letter}]: {content}")
    return debaters, debate_blocks

# ================== 평가(저지) ==================
def make_judge_payload(q: dict, debate_blocks: List[str], perm: List[str]) -> Tuple[str, str]:
    # 선택지/발언을 같은 무작위 순서로 재배열
    choices_lines = [f"{k}: {q['choices'][k]}" for k in perm]
    # debate_blocks는 "AIi [L]: text" 형식이라 그대로 섞을 수 있도록 매핑 생성
    # 여기서는 perm 순서대로 필터링
    # 먼저 letter별 텍스트 추출
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

def evaluate_one(qid: str, q: dict, debater_models: List[str], judge_model: str,
                 temperature: float, top_p: float, max_sents: int,
                 num_debaters: int, retry_judge: bool = True, domain: Optional[str] = None,
                 multi_voice: bool = True, single_debater_model: Optional[str] = None):
    """단일 문항 평가 -> dict 결과"""
    # 안전 보정/검증
    normalize_item_in_place(q)
    validate_item_or_stop(qid, q)

    # 토론 생성
    debaters, debate_blocks = [], []
    if multi_voice and single_debater_model:
        debaters, debate_blocks = generate_debate_bundle_single(
            single_debater_model, q, max_sents, temperature, top_p
        )
    else:
        # 기존 방식: 각 토론자 개별 호출
        for i, letter in enumerate(LETTERS):
            ai_role = f"AI{i+1}"
            claim = q["choices"][letter]
            system_prompt = make_debater_system(ai_role, letter, claim, max_sents)
            user_msg = make_debater_user(q, letter)
            model_for_role = debater_models[i % num_debaters]
            try:
                content = chat_once(
                    model_for_role,
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_msg}],
                    temperature=temperature, top_p=top_p,
                )
            except Exception as e:
                content = f"[오류] 모델 응답 실패: {e}"
            debaters.append({"role": ai_role, "letter": letter, "content": content})
            debate_blocks.append(f"{ai_role} [{letter}]: {content}")

    # 저지
    perm = LETTERS[:]
    random.shuffle(perm)
    judge_instruction, judge_user = make_judge_payload(q, debate_blocks, perm)

    try:
        judge_raw = chat_once(
            judge_model,
            [{"role": "system", "content": judge_instruction},
             {"role": "user", "content": judge_user}],
            temperature=0.0, top_p=1.0,
            stop=["\n"],  # 한 줄 강제
        )
        judge_choice = extract_choice(judge_raw)
        if not judge_choice and retry_judge:
            judge_raw2 = chat_once(
                judge_model,
                [{"role": "system", "content": judge_instruction + "\n반드시 예: <answer>C</answer> 형식."},
                 {"role": "user", "content": judge_user}],
                temperature=0.0, top_p=1.0, stop=["\n"],
            )
            jc2 = extract_choice(judge_raw2)
            if jc2:
                judge_choice, judge_raw = jc2, judge_raw2
    except Exception as e:
        judge_choice, judge_raw = "", f"[오류] Judge 모델 실패: {e}"

    correct = (judge_choice == q["answer"])
    # debaters를 펼쳐서 텍스트 저장
    a_text = next((d["content"] for d in debaters if d["letter"] == "A"), "")
    b_text = next((d["content"] for d in debaters if d["letter"] == "B"), "")
    c_text = next((d["content"] for d in debaters if d["letter"] == "C"), "")
    d_text = next((d["content"] for d in debaters if d["letter"] == "D"), "")

    return {
        "qid": qid,
        "domain": domain or "",
        "topic": q.get("topic", ""),
        "question": q["question"],
        "gold": q["answer"],
        "judge": judge_choice or "",
        "correct": bool(correct),
        "judge_raw": judge_raw,
        "A_text": a_text,
        "B_text": b_text,
        "C_text": c_text,
        "D_text": d_text,
        "perm": "".join(perm),  # 무작위 제시 순서(로그용)
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

# Ollama 준비
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

# 모드 전환: 1인 다역(멀티보이스)
st.sidebar.markdown("---")
multi_voice = st.sidebar.checkbox("⚡ 1인 다역(한 모델로 A/B/C/D 생성, 1콜)", value=True)

# 모델 선택
st.sidebar.markdown("### 🤖 토론자 & 저지 모델")
if multi_voice:
    single_debater_model = st.sidebar.selectbox("토론자(1인 다역) 모델", model_list, key="single_deb_model")
    debater_models = [single_debater_model]  # placeholder
    num_debaters = 1
else:
    num_debaters = st.sidebar.slider("토론자 수", 1, 4, 4)
    debater_models = [st.sidebar.selectbox(f"토론자 모델 {i+1}", model_list, key=f"deb_model_{i+1}") for i in range(num_debaters)]
    single_debater_model = None

judge_default = model_list.index("mistral") if "mistral" in model_list else 0
judge_model = st.sidebar.selectbox("Judge 모델 선택", model_list, index=judge_default)
st.sidebar.caption("1인 다역: 토론자 생성 1콜 + 저지 1콜. 기존 방식: 토론자 최대 4콜 + 저지 1콜.")

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
                    single_debater_model, q, max_sents, temperature, top_p
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
                        )
                    except Exception as e:
                        content = f"[오류] 모델 응답 실패: {e}"
                    debaters.append({"role": ai_role, "letter": letter, "content": content})
                    debate_blocks.append(f"{ai_role} [{letter}]: {content}")

        st.subheader("💬 AI 발언")
        for d in debaters:
            st.markdown(f"**{d['role']} ({d['letter']} 주장)**")
            st.info(d["content"])

        # 저지
        st.subheader("⚖️ Judge 모델 판단")
        perm = LETTERS[:]
        random.shuffle(perm)
        judge_instruction, judge_user = make_judge_payload(q, debate_blocks, perm)

        with st.spinner("Judge 평가 중..."):
            final_choice, judge_raw = "", ""
            for attempt in range(3):
                try:
                    judge_raw = chat_once(
                        judge_model,
                        [
                            {"role": "system", "content": judge_instruction},
                            {"role": "user", "content": judge_user if attempt == 0
                             else judge_user + "\n\n오직 한 줄로 `<answer>X</answer>`만 출력."},
                        ],
                        temperature=0.0, top_p=1.0, stop=["\n"],
                    )
                except Exception as e:
                    judge_raw = f"[오류] Judge 모델 실패: {e}"
                    break
                final_choice = extract_choice_strict(judge_raw)
                if final_choice:
                    break

        if not final_choice:
            st.error("Judge 출력에서 A/B/C/D를 파싱하지 못했습니다.")
        else:
            st.markdown(f"**Judge 선택:** {final_choice}")
            st.markdown(f"**정답:** {q['answer']}")
            st.success("정답과 일치합니다! ✅") if final_choice == q["answer"] else st.error("정답과 불일치! ❌")

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
                multi_voice=multi_voice, single_debater_model=single_debater_model
            )
            results.append(res)

            if show_logs:
                with st.expander(f"[{idx}] {qid} | 분야: {qid2domain.get(qid, q.get('topic',''))} | 정답: {res['gold']} | Judge: {res['judge']} | {'✅' if res['correct'] else '❌'}"):
                    st.write(q["question"])
                    st.write({k: v for k, v in q["choices"].items()})
                    st.markdown("**Judge 원문 출력**"); st.code(res["judge_raw"])
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

import os
import re
import json
import time
import random
import pandas as pd
import streamlit as st
import ollama
from utils import check_ollama

# ================== 상수 ==================
LETTERS = ["A", "B", "C", "D"]

# ================== 공용 유틸 ==================
def clean_surrogates(text: str) -> str:
    return re.sub(r'[\ud800-\udfff]', '', text or "")

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
    # A~D 중 빠진 항목이 있으면 에러 (토론 모드는 4지선다 가정)
    if any(k not in q["choices"] for k in LETTERS):
        st.error(f"[{qid}] 토론 모드용 4지선다(ABCD) 형식이 아닙니다.")
        st.json(q); st.stop()

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

# ================== 토론 프롬프트 ==================
letters = ["A", "B", "C", "D"]

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

def chat_once(model: str, messages: list, **options) -> str:
    res = ollama.chat(
        model=model,
        messages=messages,
        stream=False,
        options=options or {},
    )
    return clean_surrogates(res.get("message", {}).get("content", ""))

# ================== 토론 실행 ==================
def evaluate_one(qid: str, q: dict, debater_models: list, judge_model: str,
                 temperature: float, top_p: float, max_sents: int,
                 num_debaters: int, retry_judge: bool = True, domain: str | None = None):
    """단일 문항 평가 (4명 토론 + Judge) -> dict 결과"""
    # 안전 보정/검증
    normalize_item_in_place(q)
    validate_item_or_stop(qid, q)

    debaters, debate_blocks = [], []
    for i, letter in enumerate(letters):
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
                temperature=float(temperature),
                top_p=float(top_p),
            )
        except Exception as e:
            content = f"[오류] 모델 응답 실패: {e}"
        debaters.append({"role": ai_role, "letter": letter, "content": content})
        debate_blocks.append(f"{ai_role} [{letter}]: {content}")

    judge_instruction = (
        "너는 채점관이다.\n"
        "출력 형식은 반드시 `<answer>X</answer>` 한 줄만 출력하라. (X는 A/B/C/D 중 하나)\n"
        "설명/번역/추가 텍스트/개행/마크다운 금지. 오직 태그 1줄만."
    )
    judge_user = (
        "문제와 선택지:\n"
        f"Q: {q['question']}\n" +
        "\n".join([f"{k}: {v}" for k, v in q["choices"].items()]) +
        "\n\n다음은 각 AI의 주장이다. 논리성/근거의 구체성/반론 처리 관점에서 가장 설득력 있는 선택지 하나를 고르고, "
        "형식 지시를 지켜서 답하라.\n\n" +
        "\n\n".join(debate_blocks)
    )
    try:
        judge_raw = chat_once(
            judge_model,
            [{"role": "system", "content": judge_instruction},
             {"role": "user", "content": judge_user}],
            temperature=0.0, top_p=1.0,
        )
        judge_choice = extract_choice(judge_raw)
        if not judge_choice and retry_judge:
            judge_raw2 = chat_once(
                judge_model,
                [{"role": "system", "content": judge_instruction + "\n반드시 예: <answer>C</answer> 형식."},
                 {"role": "user", "content": judge_user}],
                temperature=0.0, top_p=1.0,
            )
            jc2 = extract_choice(judge_raw2)
            if jc2:
                judge_choice, judge_raw = jc2, judge_raw2
    except Exception as e:
        judge_choice, judge_raw = "", f"[오류] Judge 모델 실패: {e}"

    correct = (judge_choice == q["answer"])
    return {
        "qid": qid,
        "domain": domain or "",
        "topic": q.get("topic", ""),
        "question": q["question"],
        "gold": q["answer"],
        "judge": judge_choice or "",
        "correct": bool(correct),
        "judge_raw": judge_raw,
        "A_text": debaters[0]["content"],
        "B_text": debaters[1]["content"],
        "C_text": debaters[2]["content"],
        "D_text": debaters[3]["content"],
    }

# ================== 토픽(문제 유형) 필터 ==================
def topic_of(q: dict) -> str:
    t = (q.get("topic") or "").strip()
    return t if t else "(미분류)"

def list_topics(mmlu_data: dict) -> list[str]:
    return sorted({topic_of(q) for q in mmlu_data.values()})

def filter_qids_by_topic(mmlu_data: dict, qids: list[str], picked_topics: list[str]) -> list[str]:
    picked = set(picked_topics)
    # BUGFIX: 선택된 토픽에 속하는 QID만 통과
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

# 모델 선택 (가변 개수)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 토론자 & 저지 모델")
num_debaters = st.sidebar.slider("토론자 수", 1, 4, 4)
debater_models = [st.sidebar.selectbox(f"토론자 모델 {i+1}", model_list, key=f"deb_model_{i+1}") for i in range(num_debaters)]
judge_default = model_list.index("mistral") if "mistral" in model_list else 0
judge_model = st.sidebar.selectbox("Judge 모델 선택", model_list, index=judge_default)
st.sidebar.caption("선택한 토론자 모델은 A/B/C/D 역할에 순환 배치됩니다.")

# 모드 선택
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

    # 안전 보정/검증
    normalize_item_in_place(q)
    validate_item_or_stop(qid, q)

    st.caption(f"유형: {topic_of(q)}")
    st.markdown(f"### ❓ {q['question']}")
    st.markdown("**선택지**")
    for k in LETTERS:
        st.markdown(f"- **{k}**: {q['choices'][k]}")

    if st.button("🚀 토론 시작 및 채점"):
        st.subheader("💬 AI 발언")
        debate_blocks = []
        for i, letter in enumerate(letters):
            ai_role = f"AI{i+1}"
            claim = q["choices"][letter]
            system_prompt = make_debater_system(ai_role, letter, claim, max_sents)
            user_msg = make_debater_user(q, letter)
            model_for_role = debater_models[i % num_debaters]
            with st.spinner(f"{ai_role} ({letter}) 생성 중..."):
                try:
                    content = chat_once(
                        model_for_role,
                        [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_msg}],
                        temperature=float(temperature), top_p=float(top_p),
                    )
                except Exception as e:
                    content = f"[오류] 모델 응답 실패: {e}"
            st.markdown(f"**{ai_role} ({letter} 주장)**"); st.info(content)
            debate_blocks.append(f"{ai_role} [{letter}]: {content}")

        st.subheader("⚖️ Judge 모델 판단")
        judge_instruction = (
            "너는 채점관이다.\n"
            "출력 형식은 반드시 `<answer>X</answer>` 한 줄만 출력하라. (X는 A/B/C/D 중 하나)\n"
            "설명/번역/추가 텍스트/개행/마크다운 금지. 오직 태그 1줄만."
        )
        judge_user = (
            "문제와 선택지:\n"
            f"Q: {q['question']}\n" +
            "\n".join([f"{k}: {v}" for k, v in q["choices"].items()]) +
            "\n\n다음은 각 AI의 주장이다. 논리성/근거의 구체성/반론 처리 관점에서 가장 설득력 있는 선택지 하나를 고르고, "
            "형식 지시를 지켜서 답하라.\n\n" +
            "\n\n".join(debate_blocks)
        )
        with st.spinner("Judge 평가 중..."):
            try:
                judge_raw = chat_once(
                    judge_model,
                    [{"role": "system", "content": judge_instruction},
                     {"role": "user", "content": judge_user}],
                    temperature=0.0, top_p=1.0,
                )
                final_choice = extract_choice(judge_raw)
                if not final_choice:
                    judge_raw_retry = chat_once(
                        judge_model,
                        [{"role": "system", "content": judge_instruction + "\n반드시 예: <answer>C</answer> 형식."},
                         {"role": "user", "content": judge_user}],
                        temperature=0.0, top_p=1.0,
                    )
                    final_choice = extract_choice(judge_raw_retry) or final_choice
            except Exception as e:
                final_choice = ""
                judge_raw = f"[오류] Judge 모델 실패: {e}"

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
                max_sents=max_sents, num_debaters=num_debaters,
                retry_judge=True, domain=qid2domain.get(qid)
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
        cm = pd.crosstab(df["gold"], df["judge"], dropna=False).reindex(index=letters, columns=letters, fill_value=0)
        st.dataframe(cm)

        # CSV 다운로드
        st.markdown("### ⬇️ 결과 저장")
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("결과 CSV 다운로드", data=csv, file_name="mmlu_batch_results.csv", mime="text/csv")


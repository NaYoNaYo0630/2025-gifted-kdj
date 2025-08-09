import os
import re
import json
import time
import random
import pandas as pd
import streamlit as st
import ollama
from utils import check_ollama

# -------------------- 유틸 --------------------
def clean_surrogates(text: str) -> str:
    return re.sub(r'[\ud800-\udfff]', '', text)

@st.cache_data(show_spinner=False)
def load_questions() -> dict:
    for p in ["pages/mmlu_debate_questions.json", "mmlu_debate_questions.json"]:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    st.error("mmlu_debate_questions.json 파일을 찾지 못했습니다. 위치를 확인하세요.")
    st.stop()

def extract_choice(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, re.I)
    if m: return m.group(1).upper()
    m = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", text.upper())
    return m[0] if m else ""

LETTERS = ["A", "B", "C", "D"]

def build_single_system_prompt() -> str:
    # 출력 강제: 태그 한 줄만
    return (
        "너는 객관식 채점 대상 문제의 정답을 고르는 모델이다.\n"
        "출력 형식은 반드시 `<answer>X</answer>` 한 줄만 출력하라. (X는 A/B/C/D 중 하나)\n"
        "설명/번역/추가 텍스트/개행/마크다운 금지. 오직 태그 1줄만."
    )

def build_single_user_prompt(q: dict) -> str:
    choices_str = "\n".join([f"{k}: {v}" for k, v in q["choices"].items()])
    return (
        f"문제:\n{q['question']}\n\n선택지:\n{choices_str}\n\n"
        "지시: 가장 옳은 한 가지 선택지를 고르고 지정된 출력 형식만 따르라."
    )

def chat_once(model: str, messages: list, **options) -> str:
    res = ollama.chat(
        model=model,
        messages=messages,
        stream=False,
        options=options or {},
    )
    return clean_surrogates(res["message"]["content"])

def predict_one(qid: str, q: dict, model: str,
                temperature: float, top_p: float, retry: bool = True):
    """단일 모델이 직접 A/B/C/D를 고름 -> 결과 dict"""
    sys = build_single_system_prompt()
    usr = build_single_user_prompt(q)
    try:
        raw = chat_once(
            model,
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            temperature=float(temperature),
            top_p=float(top_p),
        )
        pred = extract_choice(raw)
        # 파싱 실패 시 재시도(지시 강화)
        if not pred and retry:
            raw2 = chat_once(
                model,
                [{"role": "system", "content": sys + "\n예: <answer>C</answer> 형식만 허용."},
                 {"role": "user", "content": usr}],
                temperature=0.0, top_p=1.0,
            )
            p2 = extract_choice(raw2)
            if p2:
                raw, pred = raw2, p2
    except Exception as e:
        raw, pred = f"[오류] 모델 실패: {e}", ""

    correct = (pred == q["answer"])
    return {
        "qid": qid,
        "topic": q.get("topic", ""),
        "question": q["question"],
        "gold": q["answer"],
        "pred": pred or "",
        "correct": bool(correct),
        "raw": raw,
    }

# -------------------- 앱 --------------------
st.set_page_config(page_title="MMLU Single-Model Evaluation")
st.sidebar.title("🧠 MMLU 단일 모델 평가")

mmlu_data = load_questions()
all_qids = list(mmlu_data.keys())

# Ollama 준비
check_ollama()
try:
    model_list = [m["model"] for m in ollama.list()["models"]]
except Exception as e:
    st.error(f"Ollama 모델 목록 오류: {e}")
    st.stop()
if not model_list:
    st.error("설치된 모델이 없습니다. `ollama pull mistral` 등으로 설치하세요.")
    st.stop()

# 하이퍼파라미터 & 모델
st.sidebar.markdown("### ⚙️ 설정")
temperature = st.sidebar.slider("temperature", 0.0, 1.5, 0.2, 0.1)
top_p = st.sidebar.slider("top_p", 0.1, 1.0, 0.9, 0.05)
single_model = st.sidebar.selectbox("사용 모델(1개)", model_list, index=0)

mode = st.radio("실행 모드 선택", ["단일 문제", "전체 평가"], horizontal=True)

# -------- 단일 문제 --------
if mode == "단일 문제":
    qid = st.selectbox("문제 선택", all_qids, key="single_qid")
    q = mmlu_data[qid]

    st.markdown(f"### ❓ {q['question']}")
    for k, v in q["choices"].items():
        st.markdown(f"- **{k}**: {v}")

    if st.button("🚀 모델 예측"):
        with st.spinner("모델 예측 중..."):
            res = predict_one(
                qid=qid, q=q, model=single_model,
                temperature=temperature, top_p=top_p, retry=True
            )

        st.markdown(f"**모델 선택:** {res['pred'] or '(파싱 실패)'}")
        st.markdown(f"**정답:** {res['gold']}")
        st.markdown("**원문 출력(raw):**")
        st.code(res["raw"])
        if res["pred"]:
            if res["correct"]:
                st.success("정답과 일치합니다! ✅")
            else:
                st.error("정답과 불일치! ❌")
        else:
            st.warning("모델 출력에서 A/B/C/D 파싱 실패.")

# -------- 전체 평가 --------
else:
    st.markdown("### 📚 전체 평가 설정")
    random_order = st.checkbox("문제 순서 섞기", value=False)
    seed = st.number_input("랜덤 시드", value=42, step=1)
    max_count = st.slider("평가할 문제 개수", 1, len(all_qids), len(all_qids))
    show_logs = st.checkbox("문항별 상세 로그 표시", value=False)

    selected_qids = all_qids.copy()
    if random_order:
        random.Random(seed).shuffle(selected_qids)
    selected_qids = selected_qids[:max_count]

    if st.button("🧪 전체 평가 시작"):
        results = []
        progress = st.progress(0, text="시작 중...")
        start_time = time.time()

        for idx, qid in enumerate(selected_qids, start=1):
            q = mmlu_data[qid]
            progress.progress(idx / max_count, text=f"{idx}/{max_count} 평가 중: {qid}")

            res = predict_one(
                qid=qid, q=q, model=single_model,
                temperature=temperature, top_p=top_p, retry=True
            )
            results.append(res)

            if show_logs:
                with st.expander(f"[{idx}] {qid} | 정답: {res['gold']} | 예측: {res['pred']} | {'✅' if res['correct'] else '❌'}"):
                    st.write(q["question"])
                    st.write({k: v for k, v in q["choices"].items()})
                    st.markdown("**모델 원문 출력(raw)**")
                    st.code(res["raw"])

        total_time = time.time() - start_time
        progress.progress(1.0, text="완료")

        # 요약
        df = pd.DataFrame(results)
        acc = df["correct"].mean() if not df.empty else 0.0
        st.markdown("## 📈 결과 요약")
        col1, col2, col3 = st.columns(3)
        col1.metric("총 문항", len(df))
        col2.metric("정확도(%)", f"{acc*100:.1f}")
        col3.metric("총 소요시간(초)", f"{total_time:.1f}")

        # 주제별 정확도
        st.markdown("### 🧭 주제별 정확도")
        if "topic" in df.columns:
            topic_acc = df.groupby("topic")["correct"].mean().sort_values(ascending=False)
            st.dataframe(topic_acc.to_frame("accuracy").assign(accuracy=lambda x: (x["accuracy"]*100).round(1)))

        # 혼동 행렬
        st.markdown("### 🔁 혼동 행렬 (예측 vs 정답)")
        if not df.empty:
            cm = pd.crosstab(df["gold"], df["pred"], dropna=False)\
                   .reindex(index=LETTERS, columns=LETTERS, fill_value=0)
            st.dataframe(cm)

        # CSV 다운로드
        st.markdown("### ⬇️ 결과 저장")
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("결과 CSV 다운로드", data=csv, file_name="mmlu_single_model_results.csv", mime="text/csv")

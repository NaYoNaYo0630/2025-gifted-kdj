import os
import re
import json
import time
import random
import pandas as pd
import streamlit as st
import ollama
from utils import check_ollama

# ============================== 상수 ==============================
LETTERS = ["A", "B", "C", "D"]

# ============================== 유틸 ==============================
def clean_surrogates(text: str) -> str:
    return re.sub(r'[\ud800-\udfff]', '', text or "")

@st.cache_data(show_spinner=False)
def load_questions() -> dict:
    """
    지원 형태
      - 평탄: {qid: {question, choices, answer, ...}}
      - 중첩: {domain: {qid: {question, choices, answer, ...}}}
    반환: 평탄 dict (qid -> item)
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

            if not is_nested:
                # 평탄 구조 그대로 반환
                return raw

            # 중첩 → 평탄화
            flat = {}
            for domain, items in raw.items():
                if not isinstance(items, dict):
                    continue
                for qid, item in items.items():
                    if not looks_like_question(item):
                        continue
                    if not item.get("topic"):
                        item["topic"] = domain
                    flat[qid] = item
            return flat

    st.error("mmlu_debate_questions.json 파일을 찾지 못했습니다. 위치를 확인하세요.")
    st.stop()

def normalize_item_in_place(q: dict):
    """키/타입 보정: prompt→question, answers→choices, gold→answer, choices 재매핑, topic 채우기"""
    if "question" not in q and "prompt" in q:
        q["question"] = q["prompt"]
    if "choices" not in q and "answers" in q:
        q["choices"] = q["answers"]
    if "answer" not in q and "gold" in q:
        q["answer"] = q["gold"]

    # choices 보정
    if isinstance(q.get("choices"), list):
        q["choices"] = {LETTERS[i]: q["choices"][i] for i in range(min(4, len(q["choices"])))}
    elif isinstance(q.get("choices"), dict):
        ks = list(q["choices"].keys())
        if any(k not in LETTERS for k in ks):
            vals = list(q["choices"].values())
            q["choices"] = {LETTERS[i]: vals[i] for i in range(min(4, len(vals)))}

    if not q.get("topic"):
        q["topic"] = "(미분류)"

def normalize_all_in_place(mmlu_data: dict):
    for _qid, _q in list(mmlu_data.items()):
        if isinstance(_q, dict):
            normalize_item_in_place(_q)

def validate_item_or_stop(qid: str, q: dict):
    need = ["question", "choices", "answer"]
    miss = [k for k in need if k not in q]
    if miss:
        st.error(f"[{qid}] 문항 키 누락: {miss}")
        st.json(q)
        st.stop()
    if not isinstance(q["choices"], dict):
        st.error(f"[{qid}] choices가 dict가 아닙니다.")
        st.json(q)
        st.stop()
    # 선택지는 A/B/C/D만 유지
    for k in list(q["choices"].keys()):
        if k not in LETTERS:
            del q["choices"][k]
    if len(q["choices"]) < 2:
        st.error(f"[{qid}] 선택지가 부족합니다.")
        st.json(q)
        st.stop()

def extract_choice(text: str) -> str:
    """모델 출력에서 A/B/C/D 1글자 추출(강화 버전)"""
    t = (text or "").strip()
    # 1) 태그 우선
    m = re.search(r"<answer>\s*([ABCD])\s*</answer>", t, re.I)
    if m:
        return m.group(1).upper()
    # 2) '정답/answer/final is: X'
    m = re.search(r"(정답|answer|final)\s*(is|:|=)?\s*([ABCD])\b", t, re.I)
    if m:
        return m.group(3).upper()
    # 3) 줄 전체가 한 글자
    for line in t.splitlines():
        m = re.match(r"^\s*[\(\[\{<]?\s*([ABCD])\s*[\)\]\}>]?\s*$", line.strip(), re.I)
        if m:
            return m.group(1).upper()
    # 4) 마지막 fallback
    m = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", t.upper())
    return m[0] if m else ""

def build_single_system_prompt() -> str:
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

def ollama_options(temperature: float, top_p: float, seed: int | None):
    opt = {
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if seed is not None:
        opt["seed"] = int(seed)
    return opt

def chat_once(model: str, messages: list, **options) -> str:
    res = ollama.chat(
        model=model,
        messages=messages,
        stream=False,
        options=options or {},
    )
    return clean_surrogates(res.get("message", {}).get("content", ""))

def predict_one(qid: str, q: dict, model: str,
                temperature: float, top_p: float,
                seed: int | None = None, retry: bool = True):
    """단일 모델이 직접 A/B/C/D를 고름 -> 결과 dict"""
    normalize_item_in_place(q)
    validate_item_or_stop(qid, q)

    sys = build_single_system_prompt()
    usr = build_single_user_prompt(q)
    try:
        raw = chat_once(
            model,
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            **ollama_options(temperature, top_p, seed),
        )
        pred = extract_choice(raw)
        # 파싱 실패 시 재시도(지시 강화 + 보수 세팅)
        if not pred and retry:
            raw2 = chat_once(
                model,
                [{"role": "system", "content": sys + "\n예: <answer>C</answer> 형식만 허용."},
                 {"role": "user", "content": usr}],
                **ollama_options(temperature=0.0, top_p=1.0, seed=seed),
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
        "correct": bool(correct) if pred else False,
        "raw": raw,
    }

# ---------- 토픽(문제 유형) 필터 ----------
def topic_of(q: dict) -> str:
    t = (q.get("topic") or "").strip()
    return t if t else "(미분류)"

def list_topics(mmlu_data: dict) -> list[str]:
    return sorted({topic_of(q) for q in mmlu_data.values()})

def filter_qids_by_topic(mmlu_data: dict, qids: list[str], picked_topics: list[str]) -> list[str]:
    picked = set(picked_topics)
    return [qid for qid in qids if topic_of(mmlu_data[qid]) in picked]

# ============================== 앱 ==============================
st.set_page_config(page_title="MMLU Single-Model Evaluation", layout="wide")
st.sidebar.title("🧠 MMLU 단일 모델 평가")

# 데이터 로드 → 정규화
mmlu_data = load_questions()
normalize_all_in_place(mmlu_data)
all_qids = list(mmlu_data.keys())

if not all_qids:
    st.error("문항이 없습니다. JSON 파일을 확인하세요.")
    st.stop()

# 토픽(문제 유형) 필터
st.sidebar.markdown("### 🗂 문제 유형(토픽) 필터")
all_topics = list_topics(mmlu_data)
picked_topics = st.sidebar.multiselect("유형 선택(복수 가능)", all_topics, default=all_topics)

# Ollama 준비
check_ollama()
try:
    model_list = [m["model"] for m in ollama.list().get("models", [])]
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
use_seed = st.sidebar.checkbox("seed 고정(재현성)", value=True)
seed = st.sidebar.number_input("seed", value=42, step=1) if use_seed else None
single_model = st.sidebar.selectbox("사용 모델(1개)", model_list, index=0)

mode = st.radio("실행 모드 선택", ["단일 문제", "전체 평가"], horizontal=True)

# ------------------------ 단일 문제 ------------------------
if mode == "단일 문제":
    filtered_qids = filter_qids_by_topic(mmlu_data, all_qids, picked_topics)
    if not filtered_qids:
        st.warning("선택된 토픽에 해당하는 문항이 없습니다. 사이드바에서 토픽을 조정하세요.")
        st.stop()

    # 토픽 먼저 선택 → 해당 토픽에서 문제 선택
    topics_for_pick = sorted({topic_of(mmlu_data[qid]) for qid in filtered_qids})
    chosen_topic = st.selectbox("문제 유형(토픽)", topics_for_pick)
    topic_qids = [qid for qid in filtered_qids if topic_of(mmlu_data[qid]) == chosen_topic]

    qid = st.selectbox("문제 선택", topic_qids, key="single_qid")
    q = mmlu_data[qid]

    # 표시 전 보정/검증
    normalize_item_in_place(q)
    validate_item_or_stop(qid, q)

    st.caption(f"유형: {topic_of(q)}")
    st.markdown(f"### ❓ {q['question']}")
    for k in LETTERS:
        if k in q["choices"]:
            st.markdown(f"- **{k}**: {q['choices'][k]}")

    if st.button("🚀 모델 예측"):
        with st.spinner("모델 예측 중..."):
            res = predict_one(
                qid=qid, q=q, model=single_model,
                temperature=temperature, top_p=top_p,
                seed=seed, retry=True
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

# ------------------------ 전체 평가 ------------------------
else:
    st.markdown("### 📚 전체 평가 설정")

    filtered_qids = filter_qids_by_topic(mmlu_data, all_qids, picked_topics)
    if not filtered_qids:
        st.warning("선택된 토픽에 해당하는 문항이 없습니다. 사이드바에서 토픽을 조정하세요.")
        st.stop()

    random_order = st.checkbox("문제 순서 섞기", value=False)
    seed_shuffle = st.number_input("랜덤 시드(문항 순서)", value=42, step=1)
    max_count = st.slider("평가할 문제 개수", 1, len(filtered_qids), len(filtered_qids))
    show_logs = st.checkbox("문항별 상세 로그 표시", value=False)

    selected_qids = filtered_qids.copy()
    if random_order:
        random.Random(seed_shuffle).shuffle(selected_qids)
    selected_qids = selected_qids[:max_count]

    if st.button("🧪 전체 평가 시작"):
        results = []
        progress = st.progress(0, text="시작 중...")
        start_time = time.time()

        for idx, qid in enumerate(selected_qids, start=1):
            q = mmlu_data[qid]
            normalize_item_in_place(q)
            validate_item_or_stop(qid, q)

            progress.progress(idx / max_count, text=f"{idx}/{max_count} 평가 중: {qid}")
            res = predict_one(
                qid=qid, q=q, model=single_model,
                temperature=temperature, top_p=top_p,
                seed=seed, retry=True
            )
            results.append(res)

            if show_logs:
                with st.expander(f"[{idx}] {qid} | 유형: {topic_of(q)} | 정답: {res['gold']} | 예측: {res['pred']} | {'✅' if res['correct'] else '❌'}"):
                    st.write(q["question"])
                    st.write({k: v for k, v in q["choices"].items()})
                    st.markdown("**모델 원문 출력(raw)**")
                    st.code(res["raw"])

        total_time = time.time() - start_time
        progress.progress(1.0, text="완료")

        # 요약
        df = pd.DataFrame(results)
        st.markdown("## 📈 결과 요약")
        if df.empty:
            st.warning("결과가 비어 있습니다.")
            st.stop()

        acc = df["correct"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("총 문항", len(df))
        c2.metric("정확도(%)", f"{acc*100:.1f}")
        c3.metric("총 소요시간(초)", f"{total_time:.1f}")

        # 토픽별 정확도 + 문제 수
        st.markdown("### 🧭 문제 유형별 정확도")
        df["topic_label"] = df["topic"].apply(lambda t: t.strip() if isinstance(t, str) and t.strip() else "(미분류)")
        topic_group = df.groupby("topic_label").agg(
            n=("correct", "size"),
            accuracy=("correct", "mean")
        ).reset_index()
        topic_group["accuracy(%)"] = (topic_group["accuracy"] * 100).round(1)
        st.dataframe(topic_group.sort_values(["accuracy(%)", "n"], ascending=[False, False]), use_container_width=True)

        # 혼동 행렬
        st.markdown("### 🔁 혼동 행렬 (예측 vs 정답)")
        cm = pd.crosstab(df["gold"], df["pred"], dropna=False).reindex(index=LETTERS, columns=LETTERS, fill_value=0)
        st.dataframe(cm, use_container_width=True)

        # CSV 다운로드
        st.markdown("### ⬇️ 결과 저장")
        file_name = f"mmlu_single_model_results_{single_model.replace(':','_')}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("결과 CSV 다운로드", data=csv, file_name=file_name, mime="text/csv")

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

            # 도메인 중첩 여부 판별
            def looks_like_question(d: dict) -> bool:
                return isinstance(d, dict) and {"question", "choices", "answer"}.issubset(d.keys())

            is_nested = all(isinstance(v, dict) for v in raw.values()) and not any(
                looks_like_question(v) for v in raw.values()
            )

            flat = {}
            domain_index = {}   # domain -> [qids]
            qid2domain = {}     # qid -> domain

            if is_nested:
                for domain, items in raw.items():
                    if not isinstance(items, dict):
                        continue
                    domain_index.setdefault(domain, [])
                    for qid, item in items.items():
                        if not looks_like_question(item):
                            # 방어적으로 스킵
                            continue
                        flat[qid] = item
                        # topic 없으면 domain으로 채움
                        flat[qid]["topic"] = item.get("topic") or domain
                        domain_index[domain].append(qid)
                        qid2domain[qid] = domain
            else:
                # 이미 평탄. topic을 도메인으로 사용
                for qid, item in raw.items():
                    if not looks_like_question(item):
                        continue
                    flat[qid] = item
                    dom = item.get("topic", "Misc")
                    domain_index.setdefault(dom, []).append(qid)
                    qid2domain[qid] = dom

            return flat, domain_index, qid2domain

    st.error("mmlu_debate_questions.json 파일을 찾지 못했습니다. 위치를 확인하세요.")
    st.stop()

def extract_choice(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, re.I)
    if m: return m.group(1).upper()
    m = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", text.upper())
    return m[0] if m else ""

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
    return clean_surrogates(res["message"]["content"])

def evaluate_one(qid: str, q: dict, debater_models: list, judge_model: str,
                 temperature: float, top_p: float, max_sents: int,
                 num_debaters: int, retry_judge: bool = True, domain: str | None = None):
    """단일 문항 평가 (4명 토론 + Judge) -> dict 결과"""
    debaters = []
    debate_blocks = []
    # 4명 발언 (모델은 순환 할당)
    for i, letter in enumerate(letters):
        ai_role = f"AI{i+1}"
        claim = q["choices"][letter]
        system_prompt = make_debater_system(ai_role, letter, claim, max_sents)
        user_msg = make_debater_user(q, letter)

        model_for_role = debater_models[i % num_debaters]

        try:
            content = chat_once(
                model_for_role,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=float(temperature),
                top_p=float(top_p),
            )
        except Exception as e:
            content = f"[오류] 모델 응답 실패: {e}"
        debaters.append({"role": ai_role, "letter": letter, "content": content})
        debate_blocks.append(f"{ai_role} [{letter}]: {content}")

    # Judge
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
            [
                {"role": "system", "content": judge_instruction},
                {"role": "user", "content": judge_user},
            ],
            temperature=0.0,
            top_p=1.0,
        )
        judge_choice = extract_choice(judge_raw)
        if not judge_choice and retry_judge:
            judge_raw2 = chat_once(
                judge_model,
                [
                    {"role": "system", "content": judge_instruction + "\n반드시 예: <answer>C</answer> 형식."},
                    {"role": "user", "content": judge_user},
                ],
                temperature=0.0,
                top_p=1.0,
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

# -------------------- 앱 --------------------
st.set_page_config(page_title="MMLU Debate Evaluation")
st.sidebar.title("🧠 MMLU 토론 평가")

# 데이터 로드 (평탄화 + 인덱스)
mmlu_data, domain_index, qid2domain = load_questions()

# 분야(도메인) 필터
all_domains = ["(전체)"] + sorted(domain_index.keys())
sel_domain = st.sidebar.selectbox("분야 필터", all_domains, index=0)

def current_qids():
    if sel_domain == "(전체)":
        return list(mmlu_data.keys())
    return domain_index.get(sel_domain, [])

all_qids = current_qids()

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

# 공통 하이퍼파라미터
st.sidebar.markdown("### ⚙️ 공통 설정")
temperature = st.sidebar.slider("temperature", 0.0, 1.5, 0.2, 0.1)
top_p = st.sidebar.slider("top_p", 0.1, 1.0, 0.9, 0.05)
max_sents = st.sidebar.slider("발언 문장 수(권장 최대)", 3, 8, 6, 1)

# 모델 선택 (가변 개수)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 토론자 & 저지 모델")
num_debaters = st.sidebar.slider("토론자 수", 1, 4, 4)
debater_models = []
for i in range(num_debaters):
    debater_models.append(
        st.sidebar.selectbox(f"토론자 모델 {i+1}", model_list, key=f"deb_model_{i+1}")
    )
judge_default = model_list.index("mistral") if "mistral" in model_list else 0
judge_model = st.sidebar.selectbox("Judge 모델 선택", model_list, index=judge_default)
st.sidebar.caption("선택한 토론자 모델은 A/B/C/D 역할에 순환 배치됩니다.")

# 모드 선택
mode = st.radio("실행 모드 선택", ["단일 문제", "전체 평가"], horizontal=True)

if mode == "단일 문제":
    if not all_qids:
        st.warning("선택한 분야에 문제가 없습니다.")
        st.stop()

    qid = st.selectbox("문제 선택", all_qids, key="single_qid")
    q = mmlu_data[qid]

    st.caption(f"분야: {qid2domain.get(qid, q.get('topic',''))}")
    st.markdown(f"### ❓ {q['question']}")
    for key, val in q["choices"].items():
        st.markdown(f"- **{key}**: {val}")

    if st.button("🚀 토론 시작 및 채점"):
        st.subheader("💬 AI 발언")
        # 각 AI
        debate_blocks = []
        for i, letter in enumerate(letters):
            ai_role = f"AI{i+1}"
            claim = q["choices"][letter]
            system_prompt = make_debater_system(ai_role, letter, claim, max_sents)
            user_msg = make_debater_user(q, letter)

            # 순환 할당
            model_for_role = debater_models[i % num_debaters]

            with st.spinner(f"{ai_role} ({letter}) 생성 중..."):
                try:
                    content = chat_once(
                        model_for_role,
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=float(temperature),
                        top_p=float(top_p),
                    )
                except Exception as e:
                    content = f"[오류] 모델 응답 실패: {e}"
            st.markdown(f"**{ai_role} ({letter} 주장)**")
            st.info(content)
            debate_blocks.append(f"{ai_role} [{letter}]: {content}")

        # Judge
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
                    [
                        {"role": "system", "content": judge_instruction},
                        {"role": "user", "content": judge_user},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                )
                final_choice = extract_choice(judge_raw)
                if not final_choice:
                    judge_raw_retry = chat_once(
                        judge_model,
                        [
                            {"role": "system", "content": judge_instruction + "\n반드시 예: <answer>C</answer> 형식."},
                            {"role": "user", "content": judge_user},
                        ],
                        temperature=0.0,
                        top_p=1.0,
                    )
                    final_choice = extract_choice(judge_raw_retry) or final_choice
            except Exception as e:
                final_choice = ""
                judge_raw = f"[오류] Judge 모델 실패: {e}"

        if not final_choice:
            st.error("Judge 출력에서 A/B/C/D를 파싱하지 못했습니다.")
        else:
            st.markdown(f"**Judge 선택: {final_choice}**")
            st.markdown(f"**정답: {q['answer']}**")
            if final_choice == q["answer"]:
                st.success("정답과 일치합니다! 모델의 판단이 정확했습니다.")
            else:
                st.error("정답과 불일치! 모델 판단이 틀렸습니다.")

else:
    # -------- 전체 평가 모드 --------
    st.markdown("### 📚 전체 평가 설정")
    if not all_qids:
        st.warning("선택한 분야에 문제가 없습니다.")
        st.stop()

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

            res = evaluate_one(
                qid=qid,
                q=q,
                debater_models=debater_models,
                judge_model=judge_model,
                temperature=temperature,
                top_p=top_p,
                max_sents=max_sents,
                num_debaters=num_debaters,
                retry_judge=True,
                domain=qid2domain.get(qid)
            )
            results.append(res)

            if show_logs:
                with st.expander(f"[{idx}] {qid} | 분야: {qid2domain.get(qid, q.get('topic',''))} | 정답: {res['gold']} | Judge: {res['judge']} | {'✅' if res['correct'] else '❌'}"):
                    st.write(q["question"])
                    st.write({k: v for k, v in q["choices"].items()})
                    st.markdown("**Judge 원문 출력**")
                    st.code(res["judge_raw"])
                    st.markdown("**AI1 (A) 발언**")
                    st.info(res["A_text"])
                    st.markdown("**AI2 (B) 발언**")
                    st.info(res["B_text"])
                    st.markdown("**AI3 (C) 발언**")
                    st.info(res["C_text"])
                    st.markdown("**AI4 (D) 발언**")
                    st.info(res["D_text"])

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

        # 분야별 정확도
        st.markdown("### 🧭 분야별 정확도 (domain)")
        if "domain" in df.columns:
            dom_acc = df.groupby("domain")["correct"].mean().sort_values(ascending=False)
            st.dataframe(dom_acc.to_frame("accuracy").assign(accuracy=lambda x: (x["accuracy"]*100).round(1)))

        # (선택) topic별 정확도도 같이 보고 싶으면 유지
        st.markdown("### 🧭 주제별 정확도 (topic)")
        if "topic" in df.columns:
            topic_acc = df.groupby("topic")["correct"].mean().sort_values(ascending=False)
            st.dataframe(topic_acc.to_frame("accuracy").assign(accuracy=lambda x: (x["accuracy"]*100).round(1)))

        # 혼동 행렬
        st.markdown("### 🔁 혼동 행렬 (Judge vs 정답)")
        if not df.empty:
            cm = pd.crosstab(df["gold"], df["judge"], dropna=False).reindex(index=letters, columns=letters, fill_value=0)
            st.dataframe(cm)

        # CSV 다운로드
        st.markdown("### ⬇️ 결과 저장")
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("결과 CSV 다운로드", data=csv, file_name="mmlu_batch_results.csv", mime="text/csv")

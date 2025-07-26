

import uuid
import json
import streamlit as st
import ollama
from pages.utils import check_ollama
import re

def clean_surrogates(text):
    return re.sub(r'[\ud800-\udfff]', '', text)

st.set_page_config(page_title="MMLU Debate Evaluation")
st.sidebar.title("🧠 MMLU 토론 평가")

# Load MMLU 문제
with open("mmlu_debate_questions.json", "r", encoding="utf-8") as f:
    mmlu_data = json.load(f)

# 문제 선택
qid = st.sidebar.selectbox("문제 선택", list(mmlu_data.keys()))
q = mmlu_data[qid]

# 문제 및 선택지 출력
st.markdown(f"### ❓ {q['question']}")
for key, val in q['choices'].items():
    st.markdown(f"- **{key}**: {val}")

# 모델 선택
check_ollama()
model_list = [m["model"] for m in ollama.list()["models"]]
models = []
for i in range(4):
    models.append(st.sidebar.selectbox(f"모델 {i+1} 선택", model_list, key=f"model{i+1}"))

# 토론 실행
if st.button("🚀 토론 시작 및 채점"):
    st.subheader("💬 AI 발언")
    debate_messages = []

    for i in range(4):
        ai_role = f"AI{i+1}"
        system_prompt = f"You are {ai_role}. You are debating to defend the answer choice {chr(65+i)}: {q['choices'][chr(65+i)]}."
        user_msg = q['question'] + "\nChoices:\n" + "\n".join([f"{k}: {v}" for k,v in q['choices'].items()]) + f"\nExplain why {chr(65+i)} is the correct answer."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]

        res = ollama.chat(model=models[i], messages=messages, stream=False)
        content = res['message']['content']
        st.markdown(f"**{ai_role} ({chr(65+i)} 주장)**")
        st.info(content)

        debate_messages.append({"role": ai_role, "content": content})

    # Judge 모델 사용
    st.subheader("🧑‍⚖️ Judge 모델 판단")
    judge_instruction = (
        "위 토론 내용을 참고하여 가장 논리적으로 옳은 선택지를 하나만 골라라. "
        "답변은 오직 'A', 'B', 'C', 'D' 중 하나만 단답으로 출력하라. 다른 설명은 하지 마라."
    )

    judge_prompt = [
        {"role": "system", "content": judge_instruction},
        *debate_messages
    ]

    judge_res = ollama.chat(model="mistral", messages=judge_prompt, stream=False)
    final_choice = judge_res["message"]["content"].strip().upper()

    st.markdown(f"**🎯 Judge 선택: {final_choice}**")
    st.markdown(f"**✅ 정답: {q['answer']}**")

    if final_choice == q['answer']:
        st.success("정답과 일치합니다! 모델의 판단이 정확했습니다.")
    else:
        st.error("정답과 불일치! 모델 판단이 틀렸습니다.")

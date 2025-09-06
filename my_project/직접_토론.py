import streamlit as st
import re
import json
import uuid
import ollama
from utils import check_ollama

# ============== 유틸 ==============
def clean_surrogates(text: str) -> str:
    return re.sub(r'[\ud800-\udfff]', '', text or "")

def chat_once(model: str, messages: list, temperature: float, top_p: float, keep_alive: str = "5m"):
    res = ollama.chat(
        model=model,
        messages=messages,
        stream=False,
        options={
            "temperature": float(temperature),
            "top_p": float(top_p),
            "keep_alive": keep_alive
        }
    )
    return clean_surrogates(res.get("message", {}).get("content", ""))

# ============== 앱 ==============
st.set_page_config(page_title="User vs AI Debate", page_icon="🤖")
st.title("👤 사용자 vs 🤖 AI Debate")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "judge_result" not in st.session_state:
    st.session_state.judge_result = ""

if "model" not in st.session_state:
    check_ollama()
    try:
        models = [m["model"] for m in ollama.list()["models"]]
    except Exception:
        models = []
    st.session_state.model = models[0] if models else ""

# ── 사이드바 설정 ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Debate Settings")

    # 토론 주제 입력
    topic = st.text_input("토론 주제 입력", key="debate_topic")

    # 주제 추천 (AI 생성)
    prefer = st.text_input("추천 받을 주제 입력", key="prefer")
    if st.button("🎲 주제 추천 받기"):
        sys = "다양한 토론 주제를 5개 제시해라. 간결하고 한국어로."
        usr = f"{prefer}과 관련된 흥미로운 토론 주제를 추천해줘."
        raw = chat_once("mistral", [{"role":"system","content":sys},{"role":"user","content":usr}], 0.7, 0.9)
        st.session_state.recommended_topics = raw.strip().split("\n")

    if "recommended_topics" in st.session_state:
        st.markdown("#### 추천 주제")
        for t in st.session_state.recommended_topics:
            st.markdown(t)

    # AI 모델 선택
    model = st.selectbox("AI 모델 선택", ollama.list()["models"], format_func=lambda m: m["model"], key="model_select")

    # 난이도 슬라이더 (1=쉬움, 3=어려움)
    difficulty_level = st.slider("난이도", 1, 3, 2)
    difficulty_map = {
        1: "AI는 약간 허점이 있고, 논리를 완벽히 전개하지 않는다.",
        2: "AI는 논리적으로 균형 잡힌 반박을 한다.",
        3: "AI는 매우 날카롭고 공격적으로 반박하며, 상대방의 약점을 집요하게 파고든다."
    }
    difficulty_prompt = difficulty_map[difficulty_level]

    # AI 주장(세팅)
    debate_role = st.text_area("AI 주장 세팅", value="예) 차갑고 무뚝뚝하다.")

    # 파라미터
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)

# ── 메인 채팅 영역 ─────────────────────────────────────────
# 대화 기록 출력
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 사용자 입력
user_input = st.chat_input("당신의 주장이나 반박을 입력하세요")
if user_input:
    # 사용자 메시지 저장
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # AI 응답 준비
    sys_prompt = (
        f"당신은 토론 참여자입니다. 주제는 '{topic}' 입니다.\n"
        "절대 중립적이지 말고, 자신의 입장을 강하게 옹호하세요. "
        "상대방의 약점을 반드시 지적하고 반박해야 합니다.\n"
        f"{difficulty_prompt}\n\n"
        f"아래는 당신의 고정된 주장 세팅입니다:\n{debate_role}"
    )
    messages = [{"role": "system", "content": sys_prompt}] + st.session_state.messages

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("AI가 반박을 준비 중..."):
            try:
                ai_reply = chat_once(st.session_state.model, messages, temperature, top_p)
            except Exception as e:
                ai_reply = f"(에러 발생: {e})"
            st.markdown(ai_reply)

    # AI 메시지 저장
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})

# ── Judge 모델 ─────────────────────────────────────────
if st.button("🧑‍⚖️ Judge 평가하기"):
    judge_instruction = (
        f"당신은 공정한 토론 심판자입니다. 지금까지의 대화를 읽고 User와 AI 중 누가 더 논리적이고 설득력 있었는지 판정하세요.\n"
        f"승자를 정하고, 각자의 승률을 백분율로 나누어 주세요.\n"
        f"출력 형식:\n\n"
        f"[승자] : User 또는 AI\n[승률] : User xx% - AI xx%\n[이유] : ..."
    )
    judge_prompt = [{"role": "system", "content": judge_instruction}] + st.session_state.messages
    try:
        judge_result = chat_once("mistral", judge_prompt, temperature=0.0, top_p=1.0)
    except Exception:
        judge_result = "판정 실패"
    st.session_state.judge_result = judge_result

if st.session_state.judge_result:
    st.markdown("### 🧑‍⚖️ Judge 결과")
    st.info(st.session_state.judge_result)

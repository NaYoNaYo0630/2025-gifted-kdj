import uuid
import streamlit as st
import ollama
from utils import check_ollama

st.set_page_config(page_title="AI Debate Room")
st.sidebar.title("Settings")

# 모델 설정
if "model" not in st.session_state:
    check_ollama()
    st.session_state.model = ""
models = [m["model"] for m in ollama.list()["models"]]
st.session_state.model = st.sidebar.selectbox("Choose model", models)

system_prompt = st.sidebar.text_area("System Prompt")

# 세션 초기화
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# 새 채팅 만들기
if st.sidebar.button("\u2795 New Chat"):
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat_id = chat_id
    st.session_state.chats[chat_id] = {
        "name": "새로운 채팅",
        "messages": [{"role": "system", "content": system_prompt}]
    }

AI1_setting = st.sidebar.text_area("AI1 설정", help="AI1의 주장 경향 및 스타일")
AI2_setting = st.sidebar.text_area("AI2 설정", help="AI2의 주장 경향 및 스타일")

# 채팅 선택
for cid, chat_info in st.session_state.chats.items():
    if st.sidebar.button(chat_info["name"], key=cid):
        st.session_state.current_chat_id = cid

# 채팅 진행
chat_id = st.session_state.current_chat_id
if chat_id:
    chat = st.session_state.chats[chat_id]
    st.title(chat["name"])

    # 이전 메시지 출력
    for msg in chat["messages"]:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Enter debate topic or reply")

    # 오직 사용자가 입력했을 때만 토론 시작
    if user_input:
        prompt = f"{user_input}\n이 주제에 대해 딱 하나의 주장만 말해주세요. 인사말이나 서론 없이 본론만."
        chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 제목 요약
        if len(chat["messages"]) == 2 and chat["name"] == "새로운 채팅" and not chat.get("summarized", False):
            summary_prompt = [
                {"role": "system", "content": "You are a helpful assistant that creates concise chat titles."},
                {"role": "user", "content": f"Generate a short (max 5 words) title summarizing the conversation:\nUser: {chat['messages'][1]['content']}"}
            ]
            try:
                new_title = ""
                stream = ollama.chat(model=st.session_state.model, messages=summary_prompt, stream=True)
                for chunk in stream:
                    new_title += chunk.message.content
                chat["name"] = new_title.strip()
            except:
                pass
            st.rerun()

        # AI 토론 4턴 (AI1 → AI2 → AI1 → AI2)
        responses = []

        max_turn = 10
        for i in range(max_turn):
            ai_role = "AI1" if i % 2 == 0 else "AI2"
            setting = AI1_setting if ai_role == "AI1" else AI2_setting
            response = ""

            with st.chat_message(ai_role):
                st.markdown(f"__{ai_role}__")
                placeholder = st.empty()

                # 메시지 구성
                messages = [{"role": "system", "content": setting}] + chat["messages"]

                stream = ollama.chat(model=st.session_state.model, messages=messages, stream=True)
                for chunk in stream:
                    response += chunk.message.content
                    placeholder.markdown(response)

            chat["messages"].append({"role": "assistant", "content": response})
            responses.append(response)

            # 다음 턴용 사용자 지시 삽입
            rebuttal = f"The counterpart just said: {response}\nPlease rebut and make your own argument."
            chat["messages"].append({"role": "user", "content": rebuttal})
            judge_prompt = [
            {"role": "system", "content": "You are a debate observer. Decide if the debate has reached a meaningful conclusion. Reply only with 'stop' or 'continue'."},
            *chat["messages"]
            ]

            judge_response = ""
            stream = ollama.chat(model=st.session_state.model, messages=judge_prompt, stream=True)
            for chunk in stream:
                judge_response += chunk.message.content.lower().strip()

                if "stop" in judge_response:
                    st.success("🧑‍⚖️ Judge: This debate has reached a conclusion.")
                    break

else:
    st.info("왼쪽에서 채팅을 선택하거나 새 채팅을 만들어주세요.")

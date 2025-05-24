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

# 세션 상태 초기화
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
AI1_setting = st.sidebar.text_area(
        "AI1 설정",
        help="이 설정은 AI1의 주장 경향, 가치관을 조절합니다."
    )
AI2_setting= st.sidebar.text_area(
        "A12 설정",
        value="이 설정은 AI2의 주장 경향, 가치관을 조절합니다."
    )


# 채팅 선택
for cid, chat_info in st.session_state.chats.items():
    if st.sidebar.button(chat_info["name"], key=cid):
        st.session_state.current_chat_id = cid

# 채팅 활성화
chat_id = st.session_state.current_chat_id
if chat_id:
    chat = st.session_state.chats[chat_id]
    st.title(chat["name"])

    for msg in chat["messages"]:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Enter debate topic or reply")
    if user_input:
        prompt = f"{user_input}. 이거에 대해서 딱 오직 하나의 의견만 내줘"
        chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 🔍 제목 생성: system + user 메시지 2개일 때만 실행
        if len(chat["messages"]) == 2 and chat["name"] == "새로운 채팅" and not chat.get("summarized", False):
            summary_prompt = [
                {"role": "system", "content": "You are a helpful assistant that creates concise chat titles."},
                {"role": "user", "content": f"Generate a short (max 5 words) title summarizing the conversation:\nUser: {chat['messages'][1]['content']}"}
            ]
            try:
                new_title = ""
                stream = ollama.chat(
                    model=st.session_state.model,
                    messages=summary_prompt,
                    stream=True,
                )
                for chunk in stream:
                    new_title += chunk.message.content
                chat["name"] = new_title.strip()
            except:
                pass
            st.rerun()

    responses = []  # 각 AI의 응답을 모아둠

    for i in range(4):
        if i % 2 == 0:
            response = ""
            ai_role = "AI1"
            with st.chat_message(ai_role):
                st.markdown(f"__{ai_role}__")
                placeholder = st.empty()
                
                if i == 0:
                    messages = chat["messages"]
                else:
                    messages = (
                            [{"role": "system", "content": AI1_setting}] +
                            chat["messages"] +
                            [{"role": "assistant", "content": r} for r in responses]
                    )

                stream = ollama.chat(
                    model=st.session_state.model,
                    messages=messages,
                    stream=True
                )
                for chunk in stream:
                    response += chunk.message.content
                    placeholder.markdown(response)
        chat["messages"].append({'role': 'user', 'content': str([response + " Conterpart argue like this. Please rebut and refute this and make your own argue about subject"])})
        if i % 2 == 1:
            ai_role = "AI2"
            response = ""
            with st.chat_message(ai_role):
                st.markdown(f"__{ai_role}__")
                placeholder = st.empty()
                
                if i == 0:
                    messages = chat["messages"]
                else:
                    messages = (
                                [{"role": "system", "content":AI2_setting}] +
                                chat["messages"] +
                                [{"role": "assistant", "content": r} for r in responses]
                            )


                stream = ollama.chat(
                    model=st.session_state.model,
                    messages=messages,
                    stream=True
                )
                for chunk in stream:
                    response += chunk.message.content
                    placeholder.markdown(response)
        chat["messages"].append({'role': 'user', 'content': str([response + " Conterpart argue like this, Please rebut and refute this and make your own argue about subject"])})
        responses.append(response)
        chat["messages"].append({"role": ai_role, "content": response})    
    else:
        st.info("왼쪽에서 채팅을 선택하거나 새 채팅을 만들어주세요.")

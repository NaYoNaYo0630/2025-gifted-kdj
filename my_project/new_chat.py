import uuid
import streamlit as st
import ollama
import requests
from utils import check_ollama

# LibreTranslate API 함수 (API 키 불필요)
def translate_libre(text, source="auto", target="en"):
    try:
        url = "https://libretranslate.com/translate"
        payload = {
            "q": text,
            "source": source,
            "target": target,
            "format": "text"
        }
        response = requests.post(url, data=payload)
        return response.json().get("translatedText", "[번역 실패]")
    except Exception as e:
        return f"[번역 오류: {str(e)}]"

st.set_page_config(page_title="AI Debate Room")
st.sidebar.title("Settings")

# 언어 설정
if "languages" not in st.session_state:
    st.session_state.languages = ""
languages = ['Korean', 'English']
st.session_state.languages = st.sidebar.selectbox("Choose languages", languages)

# 모델 설정
if "model" not in st.session_state:
    check_ollama()
    st.session_state.model = ""
models = [m["model"] for m in ollama.list()["models"]]
st.session_state.model = st.sidebar.selectbox("Choose model", models)
system_prompt = st.sidebar.text_area("System Prompt")

# 채팅 상태 초기화
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# 새 채팅 생성
if st.sidebar.button("\u2795 New Chat"):
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat_id = chat_id
    st.session_state.chats[chat_id] = {
        "name": "새로운 채팅",
        "messages": [{"role": "system", "content": system_prompt}]
    }

AI1_setting = st.sidebar.text_area("AI1 설정", help="AI1의 주장 경향 및 스타일")
AI2_setting = st.sidebar.text_area("AI2 설정", help="AI2의 주장 경향 및 스타일")

for cid, chat_info in st.session_state.chats.items():
    if st.sidebar.button(chat_info["name"], key=cid):
        st.session_state.current_chat_id = cid

# 채팅 시작
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
        display_input = user_input
        if st.session_state.languages == 'Korean':
            system_input = f"{user_input}. 이 주제에 대해서 이 중에서 하나의 주장만 확고하게 펼치시오."
        else:
            system_input = f"{user_input}. About this topic, please make a firm statement about just one of these."

        chat["messages"].append({"role": "user", "content": system_input})
        with st.chat_message("user"):
            st.markdown(display_input)

        if len(chat["messages"]) == 4 and chat["name"] == "새로운 채팅":
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

        max_turns = 2
        responses = []

        for turn in range(max_turns):
            for ai_role, setting in [("AI1", AI1_setting), ("AI2", AI2_setting)]:
                messages = [{"role": "system", "content": setting}] + chat["messages"]
                response = ollama.chat(model=st.session_state.model, messages=messages, stream=False)["message"]["content"]

                if st.session_state.languages == "Korean":
                    response_display = response
                else:
                    response_display = translate_libre(response, source="en", target="ko")

                with st.chat_message(ai_role):
                    st.markdown(response_display)

                chat["messages"].append({"role": "assistant", "content": response})
                chat["messages"].append({
                    "role": "user",
                    "content": f"The counterpart just said: {response}\nPlease rebut and make your own argument."
                })

            judge_instruction = {
                "Korean": "당신은 토론 감시자입니다. 이 토론이 충분한 결론에 도달했는지 판단하십시오. 오직 'stop' 또는 'continue'만 응답하세요.",
                "English": "You are a debate observer. Decide if the debate has reached a meaningful conclusion. Reply only with 'stop' or 'continue'."
            }
            judge_prompt = [
                {"role": "system", "content": judge_instruction[st.session_state.languages]},
                *chat["messages"][-6:]
            ]

            judge_model = "mistral"
            judge_result = ollama.chat(model=judge_model, messages=judge_prompt, stream=False)["message"]["content"]
            if judge_result.strip().lower().startswith("stop"):
                st.success("\U0001f9d1‍⚖️ Judge: This debate has reached a conclusion.")
                break
else:
    st.info("왼쪽에서 채팅을 선택하거나 새 채팅을 만들어주세요.")

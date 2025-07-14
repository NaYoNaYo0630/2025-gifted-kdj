import uuid
import streamlit as st
import ollama
from utils import check_ollama

st.set_page_config(page_title="AI Debate Room")
st.sidebar.title("Settings")

# 한어/영어 설정
if "languages" not in st.session_state:
    st.session_state.languages = ""
languages = ['Korean', 'English']
st.session_state.languages = st.sidebar.selectbox("Choose languages", languages)

# AI 개수 선택 & 설정 입력
NumberOfAi = [2, 3, 4, 5]
if "NumberOfAi" not in st.session_state:
    st.session_state.NumberOfAi = 2

selected_num = st.sidebar.selectbox(
    "Choose the number of AI",
    NumberOfAi,
    index=NumberOfAi.index(st.session_state["NumberOfAi"])
)
if selected_num != st.session_state["NumberOfAi"]:
    st.session_state["NumberOfAi"] = selected_num

for i in range(max(NumberOfAi)):
    key = f"AI{i+1}_setting"
    if i < st.session_state["NumberOfAi"]:
        st.session_state[key] = st.sidebar.text_area(
            f"AI{i+1} 설정",
            value=st.session_state.get(key, ""),
            help=f"AI{i+1}의 주장 가능성 메시지 설정"
        )
    else:
        st.sidebar.markdown(f"AI{i+1} 설정: _(숨김)_")

# 모델 선택
if "model" not in st.session_state:
    check_ollama()
    st.session_state.model = ""
models = [m["model"] for m in ollama.list()["models"]]
st.session_state.model = st.sidebar.selectbox("Choose model", models)
system_prompt = st.sidebar.text_area("System Prompt")

# 채팅 초기화
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

for cid, chat_info in st.session_state.chats.items():
    if st.sidebar.button(chat_info["name"], key=cid):
        st.session_state.current_chat_id = cid

# 아이콘 배정 (하나만)
if "avatar_map" not in st.session_state:
    emoji_numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    st.session_state.avatar_map = {"user": "👤"}
    for i in range(st.session_state.NumberOfAi):
        st.session_state.avatar_map[f"AI{i+1}"] = emoji_numbers[i]

avatar_map = st.session_state.avatar_map
num_ai = st.session_state.NumberOfAi

# 버튼 상태 관리
st.session_state.setdefault("show_user_judge", False)
st.session_state.setdefault("show_model_judge", False)

# 채팅 시작
chat_id = st.session_state.current_chat_id
if chat_id:
    chat = st.session_state.chats[chat_id]
    st.title(chat["name"])

    for msg in chat["messages"]:
        if msg["role"] != "system":
            avatar = avatar_map.get(msg["role"], "💬")
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    user_input = st.chat_input("Enter debate topic or reply")

    if user_input:
        display_input = user_input
        if st.session_state.languages == 'Korean':
            system_input = f"{user_input}. 이 주제에 대해서 이 중에서 하나의 주장만 확고하게 평치하십시오."
        else:
            system_input = f"{user_input}. About this topic, please make a firm statement about just one of these."

        chat["messages"].append({"role": "user", "content": system_input})
        with st.chat_message("user", avatar=avatar_map["user"]):
            st.markdown(display_input)

        if len(chat["messages"]) == 3 and chat["name"] == "새로운 채팅":
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
                st.rerun()
            except:
                pass

        max_turns = 5
        for turn in range(max_turns):
            for i in range(num_ai):
                ai_role = f"AI{i+1}"
                setting_key = f"AI{i+1}_setting"
                setting = st.session_state.get(setting_key, "")
                opponents = [f"AI{j+1}" for j in range(num_ai) if j != i]
                opponent_str = ", ".join(opponents)

                system_intro = f"You are {ai_role}. You are debating against {opponent_str}. " + setting
                messages = [{"role": "system", "content": system_intro}] + chat["messages"]

                response = ollama.chat(model=st.session_state.model, messages=messages, stream=False)["message"]["content"]

                with st.chat_message(ai_role, avatar=avatar_map[ai_role]):
                    st.markdown(response)

                chat["messages"].append({"role": "assistant", "content": response})
                chat["messages"].append({
                    "role": "user",
                    "content": f"{opponent_str} just said: {response}\nPlease rebut and make your own argument. No more than 3 sentences."
                })

        st.divider()
        st.subheader("\ud83d\udce3 \ud1a0\ub860이 \uc885료되어요. \ud310단을 \b0b4려보세요.")

        if st.button("👤 사용자 승자 선택"):
            st.session_state.show_user_judge = True
        if st.session_state.show_user_judge:
            choice = st.selectbox("승자를 선택하세요", [f"AI{i+1}" for i in range(num_ai)], key="user_choice")
            st.success(f"👤 사용자 판단: {choice} 승리!")

        if st.button("🧐 JudgeModel 조언"):
            st.session_state.show_model_judge = True
        if st.session_state.show_model_judge:
            judge_instruction = {
                "Korean": "당신은 토론 감시자입니다. 이 토론의 승자와 이유를 제시해 주십시오.",
                "English": "You are a debate observer. Please indicate the winner of this debate and the reasons why."
            }
            judge_prompt = [
                {"role": "system", "content": judge_instruction[st.session_state.languages]},
                *chat["messages"]
            ]
            judge_model = "mistral"
            judge_result = ollama.chat(model=judge_model, messages=judge_prompt, stream=False)["message"]["content"]
            st.markdown(judge_result)
else:
    st.info("\uc624\ub298\uc758 \ud1a0\ub860 \uc8fc\uc81c\ub294 \ubb34\uc5c7\uc778\uac00\uc694?\n\uc67c\ucabd\uc5d0\uc11c \ucc44\ud305\uc744 \uc120\ud0dd\ud558\uac70\ub098 \uc0c8 \ucc44\ud305\uc744 \ub9cc\ub4e4\uc5b4\uc8fc\uc138\uc694.")

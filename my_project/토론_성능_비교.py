"""
timHan/llama3.2korean3B4QKM:latest
openchat:latest
mistral:latest
exaone3.5:latest
gemma3:latest
llama3.2:latest
    """

import uuid
import streamlit as st
import ollama
from utils import check_ollama
import re

def clean_surrogates(text):
    return re.sub(r'[\ud800-\udfff]', '', text)

st.set_page_config(page_title="AI Debate Room")
st.sidebar.title("Settings")

if "languages" not in st.session_state:
    st.session_state.languages = ""
languages = ['Korean', 'English']
st.session_state.languages = st.sidebar.selectbox("Choose languages", languages)

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
            f"AI{i+1} 주장 경향성 설정",
            value=st.session_state.get(key, ""),
            help=f"AI{i+1}의 주장 가능성 메시지 설정"
        )

if "model" not in st.session_state:
    check_ollama()
    model_list = [m["model"] for m in ollama.list()["models"]]
    st.session_state.model = model_list[0] if model_list else ""

models = [m["model"] for m in ollama.list()["models"]]
for i in range(st.session_state["NumberOfAi"]):
    key = f"AI{i+1}_model"
    default_index = 0
    if key in st.session_state and st.session_state[key] in models:
        default_index = models.index(st.session_state[key])

    st.sidebar.selectbox(
        f"AI{i+1} 모델 설정",
        models,
        index=default_index,
        key=key
    )

# 공통 하이퍼파라미터
st.sidebar.markdown("### ⚙️ 공통 설정")
temperature = st.sidebar.slider("temperature", 0.0, 1.5, 0.2, 0.1)
top_p = st.sidebar.slider("top_p", 0.1, 1.0, 0.9, 0.05)
max_sents = st.sidebar.slider("발언 문장 수(권장 최대)", 3, 8, 6, 1)

system_prompt = st.sidebar.text_area("System Prompt")

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

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

emoji_numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
st.session_state.avatar_map = {f"AI{i+1}": emoji_numbers[i] for i in range(st.session_state["NumberOfAi"])}
st.session_state.avatar_map["user"] = "👤"

avatar_map = st.session_state.avatar_map
num_ai = st.session_state.NumberOfAi

st.session_state.setdefault("show_user_judge", False)
st.session_state.setdefault("show_model_judge", False)
st.session_state.setdefault("user_judge_choice", "")
st.session_state.setdefault("judge_result", "")

chat_id = st.session_state.current_chat_id
if chat_id:
    chat = st.session_state.chats[chat_id]
    st.title(chat["name"])

    for msg in chat["messages"]:
        if msg["role"] == "system":
            continue
        if msg["role"] == "user" and (
            "just said:" in msg["content"] 
            or "Please rebut" in msg["content"]
            or "⚠️ 당신의 발언이 너무 중립적입니다. 자신의 입장을 강하게 다시 주장하세요. 중립적 비교는 허용되지 않습니다." in msg["content"]
            or "⚠️ Your response was too neutral. Reassert your stance strongly. Neutral comparisons are not allowed." in msg["content"]
        ):
            continue
        if msg["role"].endswith("_instruction"):
            continue

        avatar = avatar_map.get(msg["role"], "💬")
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    user_input = st.chat_input("Enter debate topic or reply")

    if user_input:
        chat["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar=avatar_map["user"]):
            st.markdown(user_input)

        if len(chat["messages"]) == 3 and chat["name"] == "새로운 채팅":
            summary_prompt = [
                {"role": "system", "content": "You are a helpful assistant that creates concise chat titles."},
                {"role": "user", "content": f"Generate a short (max 5 words) title summarizing the conversation:\nUser: {chat['messages'][1]['content']}"}
            ]
            try:
                stream = ollama.chat(model=st.session_state.model, messages=summary_prompt, stream=True)
                new_title = ""
                for chunk in stream:
                    new_title += chunk.message.content
                chat["name"] = new_title.strip()
                st.rerun()
            except:
                pass


        max_turns = 3
        for turn in range(max_turns):
            st.divider()
            for i in range(num_ai):
                ai_role = f"AI{i+1}"
                setting_key = f"AI{i+1}_setting"
                setting = st.session_state.get(setting_key, "")
                opponents = [f"AI{j+1}" for j in range(num_ai) if j != i]
                opponent_str = ", ".join(opponents)
                response = ""
                previous_response = response
                system_intro = (
                    f"You are {ai_role} and are debating with {opponent_str}. "
                    f"Your fixed position is {setting} and you should never deviate from this argument. "
                    f"Rebut your opponent's argument, {previous_response}. Write in 3 sentences at most."
                ) if st.session_state.languages == "English" else (
                    f"당신은 {ai_role}이며, {opponent_str}와 토론 중입니다. "
                    f"당신의 고정 입장은 {setting}이며 이 주장에서 절대로 벗어나지 마세요. "
                    f"상대의 주장인 {previous_response}를 반박하세요. 최대 3문장으로 작성하세요."
                )

                messages = [{"role": "system", "content": system_intro}] + chat["messages"]
                response = ollama.chat(
                    model=st.session_state.model,
                    messages=messages,
                    stream=False,
                    options={
                        "top_p": 1.4,
                        "temperature": 1.0
                    }
                )["message"]["content"]
                
                with st.chat_message(ai_role, avatar=avatar_map[ai_role]):
                    st.markdown(response)

                chat["messages"].append({"role": ai_role, "content": response})
                neutral_keywords = [
                    "both", "depends", "personal preference", "중립", "선호", "취향", "균형", "장단점", "equally valid"
                ]
                if any(keyword.lower() in response.lower() for keyword in neutral_keywords):
                    feedback = (
                        "⚠️ Your response was too neutral. Reassert your stance strongly. Neutral comparisons are not allowed."
                        if st.session_state.languages == "English"
                        else "⚠️ 당신의 발언이 너무 중립적입니다. 자신의 입장을 강하게 다시 주장하세요. 중립적 비교는 허용되지 않습니다."
                    )
                    chat["messages"].append({
                        "role": "user",
                        "content": feedback
                    })
                chat["messages"].append({
                    "role": "user",
                    "content": f"{opponent_str} just said: {response}\nPlease rebut and make your own argument. No more than 3 sentences."
                })

            # 1. 사용자 판단 버튼
    if st.button("👤 사용자 승자 선택", key="user_judge_btn"):
        st.session_state.show_user_judge = True
        st.rerun()

    # 2. 판단 선택 창
    if st.session_state.get("show_user_judge", False):
        choice = st.selectbox(
            "승자를 선택하세요",
            [f"AI{i+1}" for i in range(num_ai)],
            key="user_choice_select"
        )
        st.session_state.user_judge_choice = choice
        st.success(f"👤 사용자 판단: {choice} 승리!")

            # 3. 선택된 AI 이어가기
        if st.button("선택된 AI가 주장 이어가기", key="continue_arg"):
            ai_idx = int(choice.replace("AI", "")) - 1
            ai_role = f"AI{ai_idx+1}"
            setting = st.session_state.get(f"AI{ai_idx+1}_setting", "")
            opponent_str = ", ".join([f"AI{j+1}" for j in range(num_ai) if j != ai_idx])
            system_intro = (
                f"You are {ai_role}. Continue your argument logically against {opponent_str}. " +
                setting + " " +
                ("Do not mention the prompt. Expand your idea concisely." if st.session_state.languages == "English"
                else "프롬프트에 대해 말하지 말고 자신의 주장을 간결하고 논리적으로 이어가세요.")
            )
            messages = [{"role": "system", "content": system_intro}] + chat["messages"]
            response = ollama.chat(model=st.session_state.model, messages=messages, stream=False)["message"]["content"]
            with st.chat_message(ai_role, avatar=avatar_map[ai_role]):
                st.markdown(response)
                chat["messages"].append({"role": ai_role, "content": response})
                st.session_state.show_user_judge = False

        if st.button("🧐 JudgeModel 조언"):
            st.session_state.show_model_judge = True
        if st.session_state.show_model_judge:
            judge_instruction = {
                "Korean": (
                    "많은 참가자 중에 제일 논리적이고 설득력 있는 사람을 딱 1명만 뽑아줘. 그 사람의 이름만 말해. 예를 들어 'AI1' 이렇게."
                ),
                "English": (
                        "You are a fair debate judge. Read the conversation and decide which AI presented the more logical and persuasive argument.\n"
                        "Output in the following format:\n\n"
                        "[Winner] : AI1 or AI2\n[Reason] : Detailed explanation"
                )
            }
            judge_prompt = [
                {"role": "system", "content": judge_instruction[st.session_state.languages]},
                *chat["messages"]
            ]
            judge_model = "mistral"
            judge_result = ollama.chat(model=judge_model, messages=judge_prompt, stream=False)["message"]["content"]
            st.session_state.judge_result = judge_result

        if st.session_state.judge_result:
            st.markdown(f"### 🧑‍⚖️ JudgeModel 판단 결과\n{st.session_state.judge_result}")

else:
    st.info("오늘의 토론 주제는 무엇인가요?\n왼쪽에서 채팅을 선택하거나 새 채팅을 만들어주세요.")

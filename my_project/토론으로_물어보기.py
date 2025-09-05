"""
timHan/llama3.2korean3B4QKM:latest
openchat:latest
mistral:latest
exaone3.5:latest
gemma3:latest
llama3.2:latest
"""

import uuid
import json
import re
import streamlit as st
import ollama
from utils import check_ollama

# ============== 유틸 ==============
def clean_surrogates(text: str) -> str:
    return re.sub(r'[\ud800-\udfff]', '', text or "")

def safe_json_loads(payload: str):
    if not payload:
        return None
    m = re.search(r"\{.*\}", payload, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

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

def make_bundle_system(roles, lang: str, max_sents: int):
    lang_line = "한국어만 사용" if lang == "Korean" else "Use English only"
    keys_line = ", ".join(roles)
    # JSON 키를 정확히 강제
    json_schema_lines = ",\n".join([f'  "{r}": ""' for r in roles])
    return (
        f"너는 다음 여러 참가자를 동시에 연기한다: {keys_line}.\n"
        f"{lang_line}. 각 참가자는 자신의 고정 입장(setting)을 강하게 옹호하고, 중립적 표현을 피하며, "
        f"다른 참가자의 주장 약점을 최소 1회 지적한다. 각 발언은 최대 {max_sents}문장.\n\n"
        "출력 형식은 **오직 하나의 JSON 객체**로 하고, 다른 설명/코드펜스/주석 금지. "
        "키는 아래와 정확히 일치해야 한다.\n"
        "{\n" + json_schema_lines + "\n}"
    )

def make_bundle_user(all_messages_for_context: list, settings_map: dict, lang: str):
    # 간단한 컨텍스트 + 역할별 setting만 전달 (대화 기록은 messages로 이미 제공)
    if lang == "Korean":
        header = "아래는 역할별 고정 입장(setting)이다. 각 역할은 자신의 설정을 바탕으로 발언을 생성하라."
        settings_txt = "\n".join([f"- {k}: {v}" for k, v in settings_map.items()])
        return header + "\n" + settings_txt
    else:
        header = "Below are fixed positions (settings) per role. Generate each role's speech based on its setting."
        settings_txt = "\n".join([f"- {k}: {v}" for k, v in settings_map.items()])
        return header + "\n" + settings_txt

# ============== 앱 ==============
st.set_page_config(page_title="AI Debate Room")
st.sidebar.title("Settings")

if "languages" not in st.session_state:
    st.session_state.languages = ""
languages = ['Korean', 'English']
st.session_state.languages = st.sidebar.selectbox("Choose languages", languages)

NumberOfAi = [2, 3, 4, 5]
if "NumberOfAi" not in st.session_state:
    st.session_state.NumberOfAi = 2

# === 🧪 주장 자동 생성(사이드바 · 1,2,3 형식) ===
if "models" not in st.session_state:
    check_ollama()
    def _load_models():
        try:
            return [m["model"] for m in ollama.list()["models"]]
        except Exception:
            return []
    st.session_state.models = _load_models()

models = st.session_state.get("models", [])
if not models:
    st.sidebar.error("설치된 Ollama 모델이 없습니다. 예: `ollama pull mistral`")
    st.stop()

with st.sidebar:
    st.markdown("### 🧪 번호 형식 의견 생성")

    # N은 사이드바에서 고른 AI 수와 동일하게 사용 (1~N 의견)
    N = int(st.session_state.get("NumberOfAi", 2))

    topic = st.text_input("주제(예: 무슨 옷을 입을까?)", key="sb_topic_numbered")

    default_name = "gemma3:latest"
    default_idx = models.index(default_name) if default_name in models else 0
    gen_model = st.selectbox("실행 모델", models, index=default_idx, key="sb_model_numbered")

    sb_temp = st.slider("temperature", 0.0, 1.5, 0.6, 0.1, key="sb_temp_numbered")
    sb_topp = st.slider("top_p", 0.1, 1.0, 0.95, 0.05, key="sb_topp_numbered")

    if st.button("▶ 번호 형식 생성", key="sb_make_numbered"):
        if not (topic or "").strip():
            st.warning("주제를 입력하세요.")
        else:
            # 프롬프트: 정확히 N줄, 1. ~ N. 형식만 허용
            sys = (
                "너는 사용자 주제에 대해 서로 대비되는 여러 입장을 만든다.\n"
                f"출력은 **오직 {N}줄**, 각 줄은 숫자와 점으로 시작해야 한다. 다른 말/코드펜스/빈 줄 금지.\n"
                f"형식 예시: 1. …\\n2. …\\n...\\n{N}. …\n"
                "각 줄은 한 문장으로 간결하게, 한국어만 사용."
            )
            usr = f"주제: {topic}"

            raw = chat_once(
                gen_model,
                [{"role": "system", "content": sys},
                 {"role": "user", "content": usr}],
                temperature=sb_temp, top_p=sb_topp
            )

            text = (raw or "").strip()

            # --- 번호 라인 파싱: "1. ..." ~ "N. ..." 만 추출 ---
            pairs = re.findall(r"(?m)^\s*(\d+)\.\s*(.+?)\s*$", text)
            by_num = {}
            for num_str, content in pairs:
                try:
                    k = int(num_str)
                except ValueError:
                    continue
                if 1 <= k <= N and k not in by_num:
                    by_num[k] = content.strip()

            # 부족하면 일반 줄에서 보충
            if len(by_num) < N:
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                # 번호 없이 온 문장들을 채워넣기
                for l in lines:
                    if len(by_num) >= N:
                        break
                    if not re.match(r"^\d+\.\s*", l):
                        by_num[len(by_num) + 1] = l

            # 여전히 부족하면 빈칸으로 패딩
            contents = [by_num.get(i, "") for i in range(1, N + 1)]

            # 최종 출력
            final = topic.strip() + "\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(contents, 1))
            st.markdown("**결과**")
            st.code(final)

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

# 모델 목록
if "model" not in st.session_state:
    check_ollama()
    model_list = [m["model"] for m in ollama.list()["models"]]
    st.session_state.model = model_list[0] if model_list else ""

models = [m["model"] for m in ollama.list()["models"]]

# 각 AI별 모델 선택 (이 값으로 그룹핑하여 모델별 일인이역 실행)
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
max_turns = st.sidebar.slider("토론 최대 턴수", 3,8,6,1)

system_prompt = st.sidebar.text_area("System Prompt")

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# ── 다이얼로그 정의 ─────────────────────────────────────────
@st.dialog("새 채팅 만들기")
def new_chat_dialog():
    chatings_name = st.text_input("채팅 이름을 입력하세요", key="dlg_new_chat_name")
    if st.button("확인", key="dlg_new_chat_ok"):
        name = (chatings_name or "").strip() or "Untitled Chat"
        chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = chat_id
        st.session_state.chats[chat_id] = {
            "name": name,
            "messages": []
        }
        st.rerun()

# ── 새 채팅 버튼(사이드바) ──────────────────────────────────
if st.sidebar.button("➕ New Chat", key="sidebar_new_chat"):
    new_chat_dialog()

# ── 기존 잘못된 name 값 정리(예방) ──────────────────────────
for _cid, _info in st.session_state.chats.items():
    nm = _info.get("name")
    if not isinstance(nm, str) or not nm.strip():
        _info["name"] = "Untitled Chat"

# ── 채팅 목록 1회만 렌더 + key 네임스페이스 부여 ────────────
for cid, chat_info in list(st.session_state.chats.items()):
    label = (chat_info.get("name") or "").strip() or "Untitled Chat"
    btn_key = f"chat_btn_{cid}"          # ← 고유 key로 충돌 방지
    if st.sidebar.button(label, key=btn_key):
        st.session_state.current_chat_id = cid
        st.rerun()


emoji_numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
st.session_state.avatar_map = {f"AI{i+1}": emoji_numbers[i] for i in range(st.session_state["NumberOfAi"]) }
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

    # 기록 렌더
    for msg in chat["messages"]:
        if msg["role"] == "system":
            continue
        if msg["role"] == "user" and (
            "just said:" in msg["content"]
            or "Please rebut" in msg["content"]
            or "⚠️ 당신의 발언이 너무 중립적입니다." in msg["content"]
            or "⚠️ Your response was too neutral." in msg["content"]
        ):
            continue
        if msg["role"].endswith("_instruction"):
            continue
        avatar = avatar_map.get(msg["role"], "💬")
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # 사용자 입력
    user_input = st.chat_input("Enter debate topic or reply")
    if user_input:
        chat["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar=avatar_map["user"]):
            st.markdown(user_input)

        # ========== 일인이역(모델별 멀티보이스) 토론 루프 ==========
        max_turns = 3
        for turn in range(max_turns):
            st.divider()

            # 1) 모델별로 AI 역할 묶기
            model_groups = {}  # model_name -> [ai_index...]
            for i in range(num_ai):
                mname = st.session_state.get(f"AI{i+1}_model", st.session_state.model or (models[0] if models else ""))
                model_groups.setdefault(mname, []).append(i)

            # 2) 각 모델 그룹에 대해 1콜씩 JSON 번들 생성
            for mname, idxs in model_groups.items():
                roles = [f"AI{j+1}" for j in idxs]
                settings_map = {f"AI{j+1}": st.session_state.get(f"AI{j+1}_setting", "") for j in idxs}

                bundle_system = make_bundle_system(roles, st.session_state.languages, max_sents)
                bundle_user = make_bundle_user(chat["messages"], settings_map, st.session_state.languages)

                # system + 기존 대화 기록 + user(설정 요약)
                messages = [{"role": "system", "content": bundle_system}] + chat["messages"] + [{"role": "user", "content": bundle_user}]

                try:
                    raw = chat_once(mname, messages, temperature=temperature, top_p=top_p)
                except Exception as e:
                    raw = ""

                data = safe_json_loads(raw) or {}

                # 3) 번들에서 각 역할의 발언 추출 후 기록/렌더
                for j in idxs:
                    ai_role = f"AI{j+1}"
                    text = data.get(ai_role, "")
                    text = clean_surrogates(str(text))
                    if not text:
                        # 비상시 짧은 기본 응답
                        text = ("주장을 간결히 정리합니다." if st.session_state.languages == "Korean"
                                else "I will summarize my stance concisely.")
                    with st.chat_message(ai_role, avatar=avatar_map[ai_role]):
                        st.markdown(text)
                    chat["messages"].append({"role": ai_role, "content": text})

                    # 중립 경고 및 반박 유도 프롬프트
                    neutral_keywords = [
                        "both", "depends", "personal preference",
                        "중립", "선호", "취향", "균형", "장단점", "equally valid"
                    ]
                    if any(k.lower() in text.lower() for k in neutral_keywords):
                        feedback = (
                            "⚠️ Your response was too neutral. Reassert your stance strongly. Neutral comparisons are not allowed."
                            if st.session_state.languages == "English"
                            else "⚠️ 당신의 발언이 너무 중립적입니다. 자신의 입장을 강하게 다시 주장하세요. 중립적 비교는 허용되지 않습니다."
                        )
                        chat["messages"].append({"role": "user", "content": feedback})

                # 4) 다음 턴을 위한 일반 반박 유도(간단 지시)
                opponent_str = ", ".join([f"AI{k+1}" for k in range(num_ai)])
                chat["messages"].append({
                    "role": "user",
                    "content": (f"{opponent_str} just said their points. Please rebut and make your own argument. No more than {max_sents} sentences.")
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
                f"You are {ai_role}. Continue your argument logically against {opponent_str}. {setting} "
                + ("Do not mention the prompt. Expand your idea concisely. Don't rebut." if st.session_state.languages == "English"
                   else "프롬프트를 언급하지 말고 자신의 주장을 간결하고 논리적으로 이어가세요. 반박은 하지마시오.")
            )
            messages = [{"role": "system", "content": system_intro}] + chat["messages"]
            try:
                response = chat_once(
                    st.session_state.get(f"AI{ai_idx+1}_model", st.session_state.model),
                    messages, temperature=temperature, top_p=top_p
                )
            except Exception:
                response = "계속 주장을 전개합니다."
            with st.chat_message(ai_role, avatar=avatar_map[ai_role]):
                st.markdown(response)
            chat["messages"].append({"role": ai_role, "content": response})
            st.session_state.show_user_judge = False

        # JudgeModel 조언
        if st.button("🧐 JudgeModel 조언"):
            st.session_state.show_model_judge = True
        if st.session_state.show_model_judge:
            judge_instruction = (
                    "당신은 공정한 토론 심판자입니다. 대화를 읽고 어떤 AI가 더 논리적이고 설득력 있는 주장을 펼쳤는지 판단해 보세요.\n"
                    "다음 형식으로 출력합니다.\n\n"
                    "[우승] : AI1 또는 AI2\n[이유] : 자세한 설명"
                    )  if st.session_state.languages == "Korean" else (
                    "You are a fair debate judge. Read the conversation and decide which AI presented the more logical and persuasive argument.\n"
                    "Output in the following format:\n\n"
                    "[Winner] : AI1 or AI2\n[Reason] : Detailed explanation"
                    )
            judge_prompt = [
                {"role": "system", "content": judge_instruction},
                *chat["messages"]
            ]
            judge_model = "mistral"
            try:
                judge_result = chat_once(judge_model, judge_prompt, temperature=0.0, top_p=1.0)
            except Exception:
                judge_result = "판단 실패"
            st.session_state.judge_result = judge_result

        if st.session_state.judge_result:
            st.markdown(f"### 🧑‍⚖️ JudgeModel 판단 결과\n{st.session_state.judge_result}")

else:
    st.info("오늘의 토론 주제는 무엇인가요?\n왼쪽에서 채팅을 선택하거나 새 채팅을 만들어주세요.")

import uuid
import streamlit as st




st.set_page_config(
    page_title="AI Debate Room",
    page_icon="🤖",
    layout="wide"
)

# 상단 타이틀
st.title("🤖 AI Debate Room")
st.markdown("AI 모델들을 불러와서 토론을 진행하고 성능을 비교하는 앱입니다.")

# 1) Hero 섹션 (중앙 강조)
st.markdown("""
<div style="padding:30px; border-radius:15px; background:linear-gradient(90deg, #00C9FF, #92FE9D); color:white; text-align:center; font-size:22px;">
    🚀 원하는 모델을 선택하고, 토론을 시작해보세요!
</div>
""", unsafe_allow_html=True)

st.write("")

# 2) 기능 카드 (columns + container)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 💬 토론하기")
    st.info("여러 AI를 동시에 불러서 토론을 진행할 수 있습니다.")
    st.page_link("pages/토론으로_물어보기.py", label="👉 이동")

with col2:
    st.markdown("### 📊 성능 비교")
    st.info("모델별 성능을 단일 또는 다중 기준으로 비교합니다.")
    st.page_link("pages/성능_비교.py", label="👉 이동")

with col3:
    st.markdown("### 🧑 직접 토론")
    st.info("내가 직접 AI와 토론으로 맞붙을수가?")
    st.page_link("pages/직접_토론.py", label="👉 이동")

st.write("")

# 3) 하단 탭 (소개/가이드)
tab1, tab2, tab3 = st.tabs(["ℹ️ 토론으로 물어보기", "ℹ️ 토론 모델 성능 비교", "ℹ️ 직접 토론하기"])

with tab1:
    st.markdown("""
    - 여러 AI 모델을 동시에 실행해 토론을 시뮬레이션합니다.  
    - 각 AI는 고정된 주장 세팅을 가지고 자신의 입장을 강하게 옹호합니다.  
    - 사용자는 토론 도중 개입하거나 승자를 직접 선택할 수도 있습니다.  

    1. 왼쪽 사이드바에서 모델과 토론 참가자 수를 선택합니다.  
    2. 주제를 입력하고 토론을 시작하세요.  
    3. 토론이 끝나면 **Judge 모델** 또는 **사용자**가 승자를 판정할 수 있습니다.  
    """)

with tab2:
    st.markdown("""
    - 과연 토론 모델이 단일 모델 보다 강할지 시뮬레이션합니다.
    - 어떤 분야에서 토론 모델이 더 강한지, 더 약한지를 각 분야별로 채점합니다.
    - 총문제 150개 이상의 많은 데이터 양

    1. 문제 유형을 골라서 채점할 분야를 선택하세요.
    2. 토론자와 채점관의 모델을 선택하세요.
    3. 다중 채점 복잡도와 횟수를 설정하게요.
    """)

with tab3:
    st.markdown("""
    - 내가 원하는 토론 모델과 직접 맞붙을 수 있습니다.
    - 난이도 설정 (1~3)
    - **Judge Model**의 승자와 이유 제시

    1.
    
    )

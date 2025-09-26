import uuid
import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

# ===== CSS =====
st.markdown("""
<style>
.card {
  border-radius: 14px;
  padding: 28px;
  text-align: center;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(0,0,0,0.1);
  transition: all .2s ease;
  cursor: pointer;
}
.card:hover {
  background: rgba(14,165,233,0.1);
  border-color: rgba(14,165,233,0.4);
  transform: translateY(-3px);
}
.card h2 {
  margin: 0;
  font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏠 Home")
st.markdown("원하는 비교 모드를 선택하세요!")

# ===== 2열 카드 =====
col1, col2 = st.columns(2)

with col1:
    if st.button("다중 토론 모델 성능", use_container_width=True):
        st.switch_page("pages/토론_모델_성능_비교.py")

with col2:
    if st.button("단일 모델 성능", use_container_width=True):
        st.switch_page("pages/단일_모델_성능_비교.py")

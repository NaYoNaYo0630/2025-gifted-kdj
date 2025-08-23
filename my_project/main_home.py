import streamlit as st

st.set_page_config(page_title="Home")

st.title("AI 토론 시스템")

if st.button("🗣️ 토론하러 가기"):
    st.switch_page("pages/discuss_battle.py")
elif st.button("📊 모델 성능 비교하러 가기"):
    st.switch_page("pages/compare_model.py")

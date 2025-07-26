import uuid
import streamlit as st


st.set_page_config(page_title="Home")

if st.button("🗣️ 토론하러 가기"):
    st.switch_page("pages/1_discuss_battle.py")
elif st.button("모델 성능 비교하러 가기"):
    st.switch_page("pages/2_compare_model.py")

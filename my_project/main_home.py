import uuid
import streamlit as st


st.set_page_config(page_title="Home")

if st.button("🗣️ 토론하러 가기"):
    st.switch_page("pages/토론으로_물어보기.py")
elif st.button("모델 성능 비교하러 가기"):
    st.switch_page("pages/성능_비교.py")

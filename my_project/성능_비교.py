import uuid
import streamlit as st


st.set_page_config(page_title="Home")

if st.button("다중 토론 모델 성능"):
    st.switch_page("pages/토론_모델_성능_비교.py")
elif st.button("단일 모델 성능"):
    st.switch_page("pages/단일_모델_성능_비교.py")

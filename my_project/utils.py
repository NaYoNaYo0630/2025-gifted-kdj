import streamlit as st
import ollama

def check_ollama():
    try:
        _ = ollama.list()
    except Exception as e:
        st.error("❌ Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요.")
        raise e
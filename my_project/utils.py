import streamlit as st
import subprocess

def check_ollama():
    """Ollama 서버가 실행 중인지 확인"""
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True)
    except Exception as e:
        st.error("⚠️ Ollama가 실행 중이 아닙니다. 터미널에서 `ollama serve` 를 먼저 실행하세요.")
        st.stop()

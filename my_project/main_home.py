# -*- coding: utf-8 -*-
import streamlit as st
import uuid

st.set_page_config(
    page_title="AI Debate Room",
    page_icon="🤖",
    layout="wide"
)

# =========================
# Global CSS (Glass / Hover)
# =========================
st.markdown("""
<style>
:root {
  --bg-grad: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
  --glass-bg: rgba(255,255,255,0.08);
  --glass-bd: rgba(255,255,255,0.18);
  --text-hero: #ffffff;
  --card-title: #0ea5e9;
  --shadow: 0 10px 30px rgba(0,0,0,.15);
}

/* Container max width */
.main .block-container{max-width:1200px}

/* Hero section */
.hero {
  margin-top: .75rem;
  padding: 42px 28px;
  border-radius: 20px;
  background: var(--bg-grad);
  color: var(--text-hero);
  text-align: center;
  box-shadow: var(--shadow);
}
.hero h1 {
  margin: 0 0 8px 0;
  font-size: 44px;
  line-height: 1.1;
  letter-spacing: -0.5px;
}
.hero p {
  margin: 0;
  font-size: 18px;
  opacity: .95;
}

/* Section titles */
.section-title {
  margin: 18px 0 8px 0;
  font-size: 22px;
  font-weight: 700;
}

/* Glass Card */
.card {
  position: relative;
  border-radius: 16px;
  padding: 18px 18px 16px 18px;
  background: var(--glass-bg);
  border: 1px solid var(--glass-bd);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
  box-shadow: 0 2px 18px rgba(0,0,0,.06);
  height: 100%;
}
.card:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 30px rgba(0,0,0,.18);
  border-color: rgba(255,255,255,.28);
}
.card h3 {
  margin: 0 0 8px 0;
  font-size: 20px;
  color: var(--card-title);
  display: flex; 
  align-items: center;
  gap: 8px;
}
.card p {
  margin: 0 0 12px 0;
  color: rgba(255,255,255,.88);
}

/* Buttons as links */
.card .btn-row{
  display: flex; 
  gap: 8px;
  flex-wrap: wrap;
}
a.btn {
  text-decoration: none !important;
  padding: 8px 12px;
  border-radius: 10px;
  background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.24);
  color: #fff !important;
  transition: background .15s ease, transform .1s ease, border-color .15s ease;
  font-size: 14px;
}
a.btn:hover{ 
  background: rgba(255,255,255,.18);
  border-color: rgba(255,255,255,.36);
  transform: translateY(-1px);
}

/* Footer */
.footer {
  margin-top: 28px;
  padding: 12px 14px;
  border-radius: 12px;
  border: 1px dashed rgba(255,255,255,.2);
  text-align: center;
  color: rgba(255,255,255,.75);
  font-size: 13px;
}

/* Light mode tune */
@media (prefers-color-scheme: light) {
  :root{
    --glass-bg: rgba(255,255,255,0.6);
    --glass-bd: rgba(0,0,0,0.06);
    --shadow: 0 10px 30px rgba(14,165,233,.12);
  }
  .card p { color: rgba(0,0,0,.7) }
  a.btn { color: #111 !important; background: rgba(0,0,0,.04); border-color: rgba(0,0,0,.12) }
  a.btn:hover { background: rgba(0,0,0,.07) }
  .footer { color: rgba(0,0,0,.55); border-color: rgba(0,0,0,.15) }
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
col_logo, col_title = st.columns([1,9])
with col_logo:
  st.markdown("### 🤖")
with col_title:
  st.markdown("## AI Debate Room")
  st.caption("AI 모델 토론 · 성능 비교 · 멀티 저지 앙상블")

# =========================
# Hero Section
# =========================
st.markdown("""
<div class="hero">
  <h1>모델을 고르고, 토론으로 검증하세요</h1>
  <p>여러 AI를 동시에 토론시키고, 저지(Judge) 앙상블로 더 공정하게 점수를 매깁니다.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# Feature Cards
# =========================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
      <h3>💬 토론으로 물어보기</h3>
      <p>여러 AI를 동시에 불러 토론을 시뮬레이션합니다.  
      각 AI는 고정 세팅으로 강하게 주장하고, 저지 피드백을 반영합니다.</p>
      <div class="btn-row">
        <a class="btn" href="/pages/토론으로_물어보기" target="_self">👉 이동</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
      <h3>📊 성능 비교</h3>
      <p>토론 모델 vs 단일 모델. 여러 분야·문항에서 성능을 점수화하고  
      평균/표준편차로 안정성을 확인합니다.</p>
      <div class="btn-row">
        <a class="btn" href="/pages/성능_비교" target="_self">👉 이동</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
      <h3>🧑‍⚖️ 저지 앙상블</h3>
      <p>여러 저지 모델로 표를 모아 평균/분산을 산출.  
      판정의 일관성과 신뢰도를 높입니다.</p>
      <div class="btn-row">
        <a class="btn" href="#guide" class="btn">가이드 보기</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================
# Tabs (Guide)
# =========================
tab1, tab2, tab3 = st.tabs(["ℹ️ 토론으로 물어보기", "ℹ️ 토론 모델 성능 비교", "ℹ️ 직접 토론하기"])

with tab1:
    st.markdown("""
**어떻게 동작하나요?**
- 여러 AI 모델을 동시에 실행해 **토론을 구성**합니다.  
- 각 AI는 고정된 **주장 세팅**(톤/스타일/입장)을 따릅니다.  
- 턴 종료마다 **저지 JSON 피드백**을 반영해 다음 턴 품질을 개선합니다.  

**빠른 시작**
1) 사이드바에서 모델·참가자 수를 선택  
2) 주제를 입력하고 토론 시작  
3) 마지막에 저지/사용자 판정으로 승자 결정
    """)

with tab2:
    st.markdown("""
**무엇을 비교하나요?**
- 토론 기반 모델이 **단일 모델**보다 강한지 시뮬레이션  
- **분야별/문제별** 점수화 + **표준편차**로 안정성 평가  

**사용법**
1) 문제 유형/분야 선택  
2) 토론자와 채점관 모델 선택  
3) 복수 저지 표 수와 난이도 설정
    """)

with tab3:
    st.markdown("""
**직접 개입하여 토론하기**
- 진행 중 원하는 순간 **사용자 승자 판정**  
- 선택한 AI가 **주장 이어가기**로 계속 전개  
- 저지 로그/점수 패널에서 근거 확인
    """)


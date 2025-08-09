# --- 사이드바나 페이지 하단 아무 곳에 붙여 넣으세요 ---
import os, json, re, random, io
import pandas as pd
import streamlit as st

OUT_PATH = r"C:\Users\USER\ollama\pages\mmlu_debate_questions.json"
SEED = 42
N_PER_DOMAIN = 5
SPLITS = ["test", "dev", "train"]  # 우선순위

def _clean(s: str) -> str:
    return re.sub(r'[\ud800-\udfff]', '', str(s or "")).strip()

def _idx_to_letter(ans):
    try:
        a = int(ans)
    except:
        return None
    if a in (1,2,3,4): return "ABCD"[a-1]
    if a in (0,1,2,3): return "ABCD"[a]
    return None

def _row_to_item(row, topic_fallback):
    choices = {k: _clean(row.get(k, "")) for k in ["A","B","C","D"]}
    if any(v == "" for v in choices.values()):
        return None
    letter = _idx_to_letter(row.get("answer"))
    if letter not in "ABCD":
        return None
    q_text = _clean(row.get("question", ""))
    if not q_text:
        return None
    topic = _clean(row.get("Category", topic_fallback))
    return q_text, choices, letter, topic

def _pick_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    # 재현 가능하게 앞에서 k개 (원하면 sample로 바꾸세요)
    return df.head(k)

def build_kmmlu_subset_json(n_per_domain=N_PER_DOMAIN, seed=SEED):
    try:
        from datasets import load_dataset, get_dataset_config_names
    except Exception as e:
        st.error("🤖 `datasets` 패키지가 필요합니다. 터미널에서 `pip install datasets` 실행 후 다시 눌러주세요.")
        raise

    random.seed(seed)
    configs = get_dataset_config_names("HAERAE-HUB/KMMLU")  # 모든 분야 이름
    out_nested = {}  # {domain: {qid: item}}
    skipped = []

    prog = st.progress(0.0, text="KMMLU 분야 목록 로드 완료...")

    for idx, domain in enumerate(configs, start=1):
        ds = None
        last_err = None
        # split 우선 시도
        for sp in SPLITS:
            try:
                ds = load_dataset("HAERAE-HUB/KMMLU", name=domain, split=sp)
                break
            except Exception as e:
                last_err = e
                continue
        if ds is None or len(ds) == 0:
            skipped.append((domain, f"no split ({last_err})"))
            prog.progress(idx/len(configs), text=f"{domain}: 건너뜀")
            continue

        # HF Dataset -> pandas
        df = pd.DataFrame(ds)
        # 필요한 컬럼만 남기고 누락 제거
        needed = ["question","answer","A","B","C","D","Category"]
        for col in needed:
            if col not in df.columns:
                df[col] = None
        # 샘플 k개
        df_k = _pick_k(df, n_per_domain)
        bucket = {}
        count_ok = 0
        for i, row in df_k.iterrows():
            item = _row_to_item(row, topic_fallback=domain)
            if not item:
                continue
            q_text, choices, letter, topic = item
            qid = f"{domain}_q{count_ok+1}"
            bucket[qid] = {
                "question": q_text,
                "choices": choices,
                "answer": letter,
                "topic": topic
            }
            count_ok += 1

        if count_ok > 0:
            out_nested[domain] = bucket
            prog.progress(idx/len(configs), text=f"{domain}: {count_ok}개 수집")
        else:
            skipped.append((domain, "no valid rows"))
            prog.progress(idx/len(configs), text=f"{domain}: 유효 항목 없음")

    return out_nested, skipped

st.markdown("### 📦 KMMLU 5문항/분야 JSON 생성")
if st.button("🚀 생성 시작"):
    with st.spinner("KMMLU에서 각 분야 5문항씩 수집 중..."):
        data, skipped = build_kmmlu_subset_json()

    # 파일로 저장
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    st.success(f"완료! → {OUT_PATH}")

    # 즉시 다운로드 버튼도 제공
    buf = io.BytesIO(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8-sig"))
    st.download_button("⬇️ mmlu_debate_questions.json 다운로드",
                       data=buf, file_name="mmlu_debate_questions.json",
                       mime="application/json")

    if skipped:
        with st.expander("건너뛴 분야(참고)"):
            st.write(skipped)

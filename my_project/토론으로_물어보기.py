# -*- coding: utf-8 -*-
# 토론으로 물어보기 (성능비교 메커니즘 + 정확도 향상 옵션 + 우세 표시 + max_turns 루프)
#
# 유지/개선 요약:
#  - 모델별 "역할 번들(JSON)" 1콜 생성(성능비교식 안전 파싱)
#  - 매 턴 종료 시: 저지(JSON) → 각 역할 조언 숨김주입 → 다음 턴에 반영
#  - 매 턴 종료 시: 저지 앙상블 점수(0~100) 평균/표준편차 표시, 우세 표시
#  - 마지막에: 최종 저지 요약(내용 요약, 각 선수 장/단점, 최종 우위/우승) 표시
#  - 사용자 승자 선택 & 이어가기 버튼 제공
#  - 견고 JSON 파서(safe_json_loads v2), 저지 원문 디버그(expander)
#  - max_turns 만큼 자동 진행

import re
import json
import uuid
from typing import Dict, List, Optional

import streamlit as st
import ollama
from utils import check_ollama

# =============== 공통 상수 ===============
MAX_AI = 5

# =============== 유틸 ===============
def clean_surrogates(text: str) -> str:
    return re.sub(r'[\ud800-\udfff]', '', text or "")

def safe_json_loads(payload: str) -> Optional[dict]:
    """
    모델이 코드펜스/설명/홑따옴표/꼬리콤마를 섞어도 최대한 복구해서 dict로 반환.
    실패 시 None.
    """
    if not payload:
        return None

    text = str(payload)

    # 0) 코드펜스/마크다운 제거
    text = re.sub(r"^```[\w-]*\s*|\s*```$", "", text.strip(), flags=re.S)

    # 1) 가장 바깥 {} 블록만 정확히 추출(스택 방식)
    def extract_outer_json_blob(s: str) -> Optional[str]:
        start = None
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(s):
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return s[start:i+1]
        return None

    blob = extract_outer_json_blob(text)
    if not blob:
        m = re.search(r"\{.*\}", text, re.S)
        blob = m.group(0) if m else None
    if not blob:
        return None

    # 2) 1차 시도
    try:
        return json.loads(blob)
    except Exception:
        pass

    fixed = blob

    # 3) 홑따옴표 → 쌍따옴표(키/문자열 값만)
    def smart_quotes(s: str) -> str:
        s = re.sub(r"(?P<prefix>[\{\s,])'(?P<key>[^'\n\r\"]+?)'\s*:", r'\g<prefix>"\g<key>":', s)
        s = re.sub(r":\s*'(?P<val>[^'\n\r\"]+?)'(?P<tail>\s*[,}\]])", r': "\g<val>"\g<tail>', s)
        return s

    fixed = smart_quotes(fixed)

    # 4) 트레일링 콤마 제거
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    # 5) 비표준 치환
    fixed = fixed.replace("NaN", "null").replace("Infinity", "1e9999").replace("-Infinity", "-1e9999")
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)

    try:
        return json.loads(fixed)
    except Exception:
        # 7) 최후 fallback: 점수만 회수
        scores = dict(re.findall(r'"(AI\d+)"\s*:\s*([+-]?\d+(?:\.\d+)?)', fixed))
        if scores:
            return {"scores": {k: float(v) for k, v in scores.items()}}
        return None


def chat_once(model: str, messages: list, temperature: float, top_p: float,
              keep_alive: str = "5m", **options) -> str:
    opts = {"temperature": float(temperature), "top_p": float(top_p), "keep_alive": keep_alive}
    opts.update(options or {})
    res = ollama.chat(model=model, messages=messages, stream=False, options=opts)
    return clean_surrogates(res.get("message", {}).get("content", ""))

# =============== 번들 토론 생성(성능비교식) ===============
def _extract_talk(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for k in ("talk", "speech", "content", "text", "message"):
            v = value.get(k)
            if isinstance(v, str):
                return v.strip()
        parts = [str(v).strip() for v in value.values() if isinstance(v, (str, int, float))]
        if parts:
            return " ".join(parts)
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if isinstance(x, (str, int, float))]
        if parts:
            return " ".join(parts)
    return ""

def _normalize_debate_json(data: dict, roles: List[str]) -> Dict[str, str]:
    out = {r: "" for r in roles}
    if not isinstance(data, dict):
        return out
    normalized = {}
    for k, v in data.items():
        kstr = str(k).strip()
        m = re.match(r"^\s*AI\s*([1-9]\d*)\s*$", kstr, re.I)
        if m:
            normalized[f"AI{int(m.group(1))}"] = v
        else:
            normalized[kstr] = v
    for r in roles:
        v = normalized.get(r) or normalized.get(r.upper()) or normalized.get(r.lower())
        if v is not None:
            out[r] = _extract_talk(v)
    return out

def make_bundle_system(roles: List[str], lang: str, max_sents: int) -> str:
    lang_line = "한국어만 사용" if lang == "Korean" else "Use English only"
    keys_line = ", ".join(roles)
    json_schema_lines = ",\n".join([f'  "{r}": ""' for r in roles])
    return (
        f"너는 다음 참가자들을 동시에 연기한다: {keys_line}.\n"
        f"{lang_line}. 각 참가자는 자신의 고정 입장(setting)을 강하게 옹호하고, 중립 표현을 피하며, "
        f"다른 참가자의 주장 약점을 최소 1회 지적한다. 각 발언은 최대 {max_sents}문장.\n\n"
        "출력은 **오직 하나의 JSON 객체**로 하고, 다른 설명/코드펜스/주석 금지. "
        "키는 아래와 정확히 일치해야 한다.\n"
        "{\n" + json_schema_lines + "\n}"
    )

def make_bundle_user(settings_map: Dict[str, str], topic_or_user_turn: str, lang: str) -> str:
    if lang == "Korean":
        header = "아래는 역할별 고정 입장(setting)과 현재 토론 맥락이다. 각 역할은 자신의 설정을 바탕으로 발언을 생성하라."
    else:
        header = "Below are role settings and the current debate context. Generate each role's speech accordingly."
    settings_txt = "\n".join([f"- {k}: {v}" for k, v in settings_map.items()])
    return f"{header}\n\n[SETTINGS]\n{settings_txt}\n\n[CONTEXT]\n{topic_or_user_turn}"

def generate_bundle_for_group(
    model_name: str, roles: List[str], settings_map: Dict[str, str],
    lang: str, topic_or_user_turn: str,
    max_sents: int, temperature: float, top_p: float,
    num_ctx: int, seed: int, top_k: int, repeat_penalty: float
) -> Dict[str, str]:
    sys = make_bundle_system(roles, lang, max_sents)
    usr = make_bundle_user(settings_map, topic_or_user_turn, lang)
    raw = chat_once(
        model_name,
        [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        temperature=temperature, top_p=top_p,
        num_ctx=int(num_ctx), seed=int(seed), top_k=int(top_k), repeat_penalty=float(repeat_penalty)
    )
    data = safe_json_loads(raw) or {}
    talks = _normalize_debate_json(data, roles)
    for r in roles:
        if not talks.get(r, "").strip():
            talks[r] = "내 입장을 간결하게 재강조하고, 상대 논지의 약점을 한 가지 이상 지적한다."
    return talks

# =============== Judge(점수/조언/최종요약) ===============
JUDGE_MODEL = "mistral"  # 기본 저지 모델

def build_judge_prompt(lang: str):
    if lang == "Korean":
        sys = (
            "당신은 공정하고 엄격한 토론 심판자입니다. "
            "대화 기록 전체를 읽고, 각 AI의 최신 주장 품질을 평가하고 다음 라운드를 위한 구체적 개선 지침을 제공합니다.\n"
            "오직 하나의 JSON으로만 출력하세요. 다른 말/코드펜스/주석 금지. "
            "markdown 금지, ``` 금지, 반드시 스키마를 채워 출력.\n"
            "스키마 예시:\n"
            "{\n"
            '  "winner": "AI1",\n'
            '  "scores": {"AI1": 0~10, "AI2": 0~10, ...},\n'
            '  "per_ai_advice": {\n'
            '     "AI1": {\n'
            '        "summary": "1~2문장 핵심 요약",\n'
            '        "rebut_targets": ["AI2", "AI3"],\n'
            '        "fixes": ["논리전개 개선점 A", "증거/예시 B"],\n'
            '        "evidence_requests": ["통계/사례/정의 등 요청 포인트"]\n'
            '     }\n'
            '  }\n'
            "}\n"
            "평가 기준: 논리적 일관성, 반박의 정확도, 근거의 구체성, 주장 강도(비중립성)."
        )
    else:
        sys = (
            "You are a rigorous debate judge. Read the entire conversation and output JSON with "
            "winner, per-AI numeric scores (0-10), and next-turn, actionable guidance for each AI. "
            "JSON only, no markdown, no codefence, fill the schema.\n"
            '{ "winner":"AI1", "scores":{"AI1":0-10,...}, "per_ai_advice":{"AI1":{"summary":"..","rebut_targets":[],"fixes":[],"evidence_requests":[]}} }'
        )
    return [{"role": "system", "content": sys}]

def get_judge_feedback(chat_messages: list, lang: str):
    prompt = build_judge_prompt(lang) + chat_messages
    raw = ""
    try:
        raw = chat_once(JUDGE_MODEL, prompt, temperature=0.0, top_p=1.0)
    except Exception:
        return None, "저지 호출 실패"
    data = safe_json_loads(raw)
    if not data:
        st.session_state["_last_judge_raw"] = raw
        return None, "JSON 파싱 실패"
    scores = {k: float(v) for k, v in (data.get("scores") or {}).items()
              if str(v).replace('.','',1).isdigit()}
    data["scores"] = scores
    return data, None

def make_judge_scores_payload(chat_messages: List[dict], roles: List[str], lang: str) -> Dict[str, str]:
    if lang == "Korean":
        instr = (
            "너는 공정한 토론 심판자다.\n"
            "현재 턴의 각 참가자 발언을 논리성/근거의 구체성/반론 처리 기준으로 0~100 점수화하라.\n"
            "출력은 **오직 하나의 JSON** 한 줄만. 예: {\"AI1\":73,\"AI2\":64}\n"
        )
    else:
        instr = (
            "You are a fair debate judge.\n"
            "Score each participant's current-turn argument (0~100) for logic/evidence/rebuttals.\n"
            "Output ONLY one JSON line, e.g., {\"AI1\":73,\"AI2\":64}."
        )
    last_turn = [m for m in chat_messages if str(m.get("role","")).startswith("AI")]
    K = len(roles)
    last_turn = last_turn[-K:] if last_turn else []
    lines = [f"{m['role']}: {m['content']}" for m in last_turn]
    user = "\n".join(lines) if lines else "현재 턴 발언이 없습니다."
    return {"system": instr, "user": user}

def parse_role_scores(payload: str, roles: List[str]) -> Dict[str, float]:
    d = safe_json_loads(payload) or {}
    out = {}
    for r in roles:
        try:
            out[r] = float(d.get(r, 0))
        except Exception:
            out[r] = 0.0
    return out

def make_judge_advice_payload(chat_messages: List[dict], roles: List[str], lang: str) -> Dict[str, str]:
    if lang == "Korean":
        instr = (
            "너는 토론 심판/코치다.\n"
            "각 참가자(AI1..N)에 대해 다음 JSON만 출력하라(코드펜스/설명 금지):\n"
            '{"AI1":{"tip":"다음 턴에서 개선할 1~2문장 팁","rebut":"반박해야 할 상대 핵심 1~2개"}, ...}\n'
        )
    else:
        instr = (
            "You are a debate judge/coach.\n"
            "Return ONLY one JSON: "
            '{"AI1":{"tip":"1-2 sentence actionable tip","rebut":"1-2 key opponent points to rebut"}, ...}'
        )
    last_turn = [m for m in chat_messages if str(m.get("role","")).startswith("AI")]
    K = len(roles)
    last_turn = last_turn[-K:] if last_turn else []
    lines = [f"{m['role']}: {m['content']}" for m in last_turn]
    user = "\n".join(lines) if lines else "현재 턴 발언이 없습니다."
    return {"system": instr, "user": user}

def parse_advice(payload: str, roles: List[str]) -> Dict[str, Dict[str, str]]:
    d = safe_json_loads(payload) or {}
    out = {r: {"tip": "", "rebut": ""} for r in roles}
    if not isinstance(d, dict):
        return out
    for r in roles:
        v = d.get(r) or d.get(r.upper()) or d.get(r.lower())
        if isinstance(v, dict):
            tip = v.get("tip") or v.get("advice") or v.get("guide") or ""
            reb = v.get("rebut") or v.get("target") or v.get("opp") or ""
            out[r]["tip"] = str(tip).strip()
            out[r]["rebut"] = str(reb).strip()
        elif isinstance(v, str):
            parts = re.split(r"\s*;\s*|\n+", v.strip(), maxsplit=1)
            out[r]["tip"] = parts[0]
            out[r]["rebut"] = parts[1] if len(parts) > 1 else ""
    return out

def _strict_advice_call(model: str, payload: Dict[str, str], num_ctx: int, seed: int) -> str:
    return chat_once(
        model,
        [{"role": "system", "content": payload["system"]},
         {"role": "user", "content": payload["user"]}],
        temperature=0.0, top_p=1.0, num_ctx=int(num_ctx), seed=int(seed)
    )

def get_advice_with_retries(judge_models_multi: List[str], messages: List[dict], roles: List[str],
                            lang: str, num_ctx: int, seed: int) -> Dict[str, Dict[str, str]]:
    """
    - 1차: 멀티저지 첫 모델로 엄격 프롬프트 호출
    - 2차: 실패/빈칸 있으면 다른 저지 모델로 재시도
    - 그래도 빈칸 있으면 휴리스틱으로 채움
    """
    payload = make_judge_advice_payload(messages, roles, lang)

    tried = []
    # 1차 시도
    m1 = judge_models_multi[0]
    tried.append(m1)
    raw1 = _strict_advice_call(m1, payload, num_ctx=num_ctx, seed=seed)
    adv = parse_advice(raw1, roles)

    def _has_empty(d):
        for r in roles:
            if not d.get(r, {}).get("tip") or not d.get(r, {}).get("rebut"):
                return True
        return False

    if not _has_empty(adv):
        return adv  # 성공

    # 2차 시도(다른 모델로)
    if len(judge_models_multi) > 1:
        m2 = judge_models_multi[1]
    else:
        m2 = judge_models_multi[0]
    if m2 not in tried:
        raw2 = _strict_advice_call(m2, payload, num_ctx=num_ctx, seed=seed+777)
        adv2 = parse_advice(raw2, roles)
        # adv에 빈칸인 항목만 보완
        for r in roles:
            if not adv.get(r, {}).get("tip") and adv2.get(r, {}).get("tip"):
                adv[r]["tip"] = adv2[r]["tip"]
            if not adv.get(r, {}).get("rebut") and adv2.get(r, {}).get("rebut"):
                adv[r]["rebut"] = adv2[r]["rebut"]

    # 3) 휴리스틱 보정
    return fill_missing_advice(adv, roles, messages)

def fill_missing_advice(adv: Dict[str, Dict[str, str]], roles: List[str], messages: List[dict]) -> Dict[str, Dict[str, str]]:
    """
    - 최근 저지 피드백(per_ai_advice)에서 summary/fixes/targets를 가져와 채움
    - 없으면 최근 턴의 상대 발언을 기준으로 기본 문장 생성
    """
    last_judge = st.session_state.get("last_judge", {}) or {}
    per_ai = (last_judge.get("per_ai_advice") or {}) if isinstance(last_judge, dict) else {}

    # 최근 턴의 AI 발언 맵
    last_ai_msgs = [m for m in messages if str(m.get("role","")).startswith("AI")]
    role_to_text = {}
    if last_ai_msgs:
        # 마지막 라운드만 추정
        k = len(roles)
        for m in last_ai_msgs[-k:]:
            role_to_text[m["role"]] = m.get("content","")

    all_roles = set(roles)
    for r in roles:
        tip = adv.get(r, {}).get("tip", "").strip()
        rebut = adv.get(r, {}).get("rebut", "").strip()

        # 저지 원자료에서 끌어오기
        src = per_ai.get(r, {}) if isinstance(per_ai, dict) else {}
        if not tip:
            fixes = src.get("fixes") or src.get("tip")
            summary = src.get("summary", "")
            if isinstance(fixes, list) and fixes:
                tip = " / ".join([str(x) for x in fixes[:2]])
            elif isinstance(fixes, str) and fixes.strip():
                tip = fixes.strip()
            elif summary:
                tip = summary.strip()

        if not rebut:
            # 저지가 지정한 반박 대상
            targets = src.get("rebut_targets") or src.get("rebut") or []
            if isinstance(targets, str):
                targets = [targets]
            targets = [t for t in targets if str(t).upper() in all_roles]
            if targets:
                rebut = f"{', '.join(targets)}의 핵심 논지를 구체적 근거로 반박하라."
            else:
                # 상대 중 하나 선택(간단 휴리스틱)
                others = [x for x in roles if x != r]
                # 상대 최신 발언 중 길이가 긴 쪽을 우선 타깃
                if others:
                    best = max(others, key=lambda o: len(role_to_text.get(o,"")))
                    rebut = f"{best}가 제시한 최근 근거의 전제/수치의 타당성을 검증해 반박하라."

        # 최종 비어있음 방지: 기본 문장
        if not tip:
            tip = "핵심 주장 한 문단을 더 선명하게 재구성하고, 구체적 수치·사례 1개를 추가하라."
        if not rebut:
            # 남은 상대 중 하나
            others = [x for x in roles if x != r]
            target = others[0] if others else "상대"
            rebut = f"{target}의 가장 강한 주장 하나를 골라 논리적 전제와 근거의 신뢰도를 짚어 반박하라."

        adv.setdefault(r, {})
        adv[r]["tip"] = tip
        adv[r]["rebut"] = rebut

    return adv


def ensemble_judge_scores(
    judge_models_multi: List[str], n_judge: int,
    messages: List[dict], roles: List[str],
    num_ctx: int, seed_base: int
) -> Dict[str, float]:
    totals = {r: 0.0 for r in roles}
    squares = {r: 0.0 for r in roles}
    logs = []
    for v in range(n_judge):
        jm = judge_models_multi[v % len(judge_models_multi)]
        payload = make_judge_scores_payload(messages, roles, st.session_state.languages)
        raw = chat_once(
            jm,
            [{"role": "system", "content": payload["system"]},
             {"role": "user", "content": payload["user"]}],
            temperature=0.0, top_p=1.0, num_ctx=int(num_ctx), seed=int(seed_base + v)
        )
        scores = parse_role_scores(raw, roles)
        logs.append(f"[{jm}] {raw}")
        for r in roles:
            s = float(scores.get(r, 0.0))
            totals[r] += s
            squares[r] += s * s
    means = {r: totals[r] / max(1, n_judge) for r in roles}
    if n_judge > 1:
        stds = {r: (squares[r] / n_judge - means[r] ** 2) ** 0.5 for r in roles}
    else:
        stds = {r: 0.0 for r in roles}
    st.session_state["_judge_logs"] = logs[-min(5, len(logs)):]
    st.session_state["_judge_stds"] = stds
    return means

# 최종 요약 저지
def build_final_summary_prompt(lang: str):
    if lang == "Korean":
        sys = (
            "너는 토론 최종 심판자다. 전체 대화를 읽고 다음 JSON만 출력하라(코드펜스/설명 금지):\n"
            "{\n"
            '  "summary": "전체 토론 핵심을 3~5문장으로 요약",\n'
            '  "per_ai": {\n'
            '     "AI1": {"strengths": ["강점1","강점2"], "weaknesses": ["약점1","약점2"]},\n'
            '     "AI2": {"strengths": [...], "weaknesses": [...]}\n'
            '  },\n'
            '  "final_winner": "AI1",\n'
            '  "reason": "최종 우위/판단 근거를 2~4문장"\n'
            "}\n"
            "JSON만 출력. markdown/텍스트 금지."
        )
    else:
        sys = (
            "You are the final debate judge. Read the whole conversation and output ONLY JSON:\n"
            '{ "summary":"3-5 sentence recap", "per_ai":{"AI1":{"strengths":[],"weaknesses":[]},...}, "final_winner":"AI1", "reason":"2-4 sentences" }'
        )
    return [{"role": "system", "content": sys}]

def get_final_summary(chat_messages: list, roles: List[str], lang: str):
    prompt = build_final_summary_prompt(lang) + chat_messages
    raw = chat_once(JUDGE_MODEL, prompt, temperature=0.0, top_p=1.0)
    data = safe_json_loads(raw)
    if not data:
        st.session_state["_last_final_judge_raw"] = raw
        return None, "최종 요약 JSON 파싱 실패"
    # per_ai 키 보정
    per_ai = {}
    got = data.get("per_ai") or {}
    for r in roles:
        v = got.get(r) or got.get(r.upper()) or got.get(r.lower()) or {}
        per_ai[r] = {
            "strengths": list(v.get("strengths") or []),
            "weaknesses": list(v.get("weaknesses") or [])
        }
    data["per_ai"] = per_ai
    return data, None

def _parse_last_judge_scores(messages: List[dict], roles: List[str]) -> Optional[Dict[str, float]]:
    """
    chat_messages에 우리가 저장해둔 system 메시지: {"role":"system","content":"[JUDGE_SCORES]{...}"}
    를 뒤에서부터 찾아 평균점수 dict를 복구.
    """
    for m in reversed(messages):
        if m.get("role") == "system":
            c = str(m.get("content", ""))
            if c.startswith("[JUDGE_SCORES]"):
                blob = c[len("[JUDGE_SCORES]"):]
                try:
                    d = safe_json_loads(blob) or json.loads(blob)
                    # roles만 필터
                    return {r: float(d.get(r, 0.0)) for r in roles}
                except Exception:
                    return None
    return None

def get_final_summary_robust(chat_messages: list, roles: List[str], lang: str,
                             judge_models_multi: List[str]):

    # 1) 더 엄격한 프롬프트(빈 값 금지, 미기재 시 규칙)
    if lang == "Korean":
        sys = (
            "너는 토론 최종 심판자다. 반드시 다음 **완전한 JSON**만 출력하라.\n"
            "{\n"
            '  "summary": "빈 문자열 금지. 3~5문장.",\n'
            '  "per_ai": {\n'
            '     "AI1": {"strengths": ["최소1개"], "weaknesses": ["최소1개"]},\n'
            '     "AI2": {"strengths": ["최소1개"], "weaknesses": ["최소1개"]}\n'
            '  },\n'
            '  "final_winner": "AI1 또는 AI2 등 정확한 키",\n'
            '  "reason": "빈 문자열 금지. 2~4문장."\n'
            "}\n"
            "키 누락/빈 문자열/빈 배열 금지. 어떤 경우에도 위 스키마를 충족시켜 출력할 것. "
            "애매하면 가장 일관된 논리를 보인 참가자를 우승자로 선택하라."
        )
    else:
        sys = (
            "You are the final debate judge. Output ONLY a **complete JSON**:\n"
            '{ "summary":"3-5 sentences, not empty",'
            '  "per_ai":{"AI1":{"strengths":[">=1"],"weaknesses":[">=1"]},"AI2":{"strengths":[">=1"],"weaknesses":[">=1"]}},'
            '  "final_winner":"AI1|AI2|...", "reason":"2-4 sentences, not empty" }\n'
            "No missing keys, no empty strings/arrays. If uncertain, pick the participant with the most consistent logic."
        )

    prompt = [{"role": "system", "content": sys}] + chat_messages

    # 2) 1차 시도: 기본 JUDGE_MODEL
    raw = ""
    try:
        raw = chat_once(JUDGE_MODEL, prompt, temperature=0.0, top_p=1.0)
        data = safe_json_loads(raw)
    except Exception:
        data = None

    # 3) 2차 시도: 다른 judge 모델로 재시도
    if not data and judge_models_multi:
        alt = judge_models_multi[0]
        try:
            raw = chat_once(alt, prompt, temperature=0.0, top_p=1.0)
            data = safe_json_loads(raw)
        except Exception:
            data = None

    # 4) 보정: 최소 필드 강제
    result = {"summary": "", "per_ai": {}, "final_winner": "", "reason": ""}
    if isinstance(data, dict):
        result["summary"] = str(data.get("summary", "") or "").strip()
        result["final_winner"] = str(data.get("final_winner", "") or "").strip()
        result["reason"] = str(data.get("reason", "") or "").strip()
        got = data.get("per_ai") or {}
        for r in roles:
            v = got.get(r) or got.get(r.upper()) or got.get(r.lower()) or {}
            result["per_ai"][r] = {
                "strengths": list(v.get("strengths") or []),
                "weaknesses": list(v.get("weaknesses") or []),
            }

    # 5) 빈 값 보정 로직
    # (a) 우승자 비었으면: 마지막 Judge 앙상블 점수 또는 직전 judge_json의 winner로 보정
    if not result["final_winner"]:
        by_scores = _parse_last_judge_scores(chat_messages, roles) or {}
        if by_scores:
            result["final_winner"] = max(by_scores, key=by_scores.get)
        else:
            last_j = st.session_state.get("last_judge", {}) or {}
            w = str((last_j.get("winner") or "")).strip()
            if w in roles:
                result["final_winner"] = w

    # (b) summary/ reason 비었으면 간이 요약 생성
    if not result["summary"]:
        turns = sum(1 for m in chat_messages if str(m.get("role","")).startswith("AI"))
        result["summary"] = f"참가자들은 총 {max(1, turns//len(roles))} 라운드 동안 핵심 논점을 주고받았다. 각자는 자신의 입장을 강화하고 상호 반박을 제시했다."
    if not result["reason"]:
        winner = result["final_winner"] or roles[0]
        result["reason"] = f"{winner}가 논리적 일관성과 구체적 근거 제시에서 상대를 앞선 것으로 판단했다."

    # (c) per_ai의 strengths/weaknesses 채우기
    per_ai_judge = (st.session_state.get("last_judge", {}) or {}).get("per_ai_advice", {}) or {}
    for r in roles:
        sw = result["per_ai"].setdefault(r, {"strengths": [], "weaknesses": []})
        if not sw["strengths"]:
            # 힌트: fixes/summary에서 강점 유추
            src = per_ai_judge.get(r, {})
            summary = (src.get("summary") or "").strip()
            if summary:
                sw["strengths"].append(summary)
            sw["strengths"] = sw["strengths"] or ["핵심 주장을 일관되게 강조함"]
        if not sw["weaknesses"]:
            src = per_ai_judge.get(r, {})
            reqs = src.get("evidence_requests") or []
            if isinstance(reqs, list) and reqs:
                sw["weaknesses"].append(f"근거 보강 필요: {reqs[0]}")
            sw["weaknesses"] = sw["weaknesses"] or ["정량적 근거 또는 반례 제시가 부족함"]

    return result, None


def _continue_ai_callback(chat_id: str, ai_choice: str):
    if chat_id not in st.session_state.chats:
        st.warning("채팅을 찾을 수 없습니다.")
        return
    chat = st.session_state.chats[chat_id]
    try:
        ai_idx = int(ai_choice.replace("AI", "")) - 1
    except Exception:
        st.warning("올바른 AI를 선택하세요.")
        return
    ai_role = f"AI{ai_idx+1}"

    setting = st.session_state.get(f"AI{ai_idx+1}_setting", "") or ""
    opponents = ", ".join([r for r in ai_roles if r != ai_role])

    # 마지막 사용자 입력 또는 토론 주제 추출
    last_user_msg = ""
    for m in reversed(chat["messages"]):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break

    if st.session_state.languages == "Korean":
        sys_prompt = (
            f"당신은 {ai_role}이다. {opponents}의 논점을 염두에 두되, 자신의 논지를 더 깊고 구체적으로 전개하라. {setting} (프롬프트 언급 금지.)"
        )
        usr_prompt = (
                f"이전 대화 주제 [{last_user_msg}]와 지금까지의 토론 맥락을 기반으로 "
                f"{ai_role}의 주장을 이어가라. "
                "새로운 근거 1개 이상 포함하고, 저지의 조언을 반영하며, 지정된 반박 대상을 우선 반박하라."
            )
    else:
        sys_prompt = (
            f"You are {ai_role}. Consider {opponents}' points but extend your case with depth and specifics. {setting} (Do not mention the prompt.)"
        )
        usr_prompt = "Continue your key argument in ≤4 sentences. Include at least one new piece of evidence."

    last_judge = st.session_state.get("last_judge", {}) or {}
    per_ai = last_judge.get("per_ai_advice", {}) or {}
    adv = per_ai.get(ai_role, {}) or {}

    bullets = []
    rts = adv.get("rebut_targets") or adv.get("rebut") or []
    if isinstance(rts, str) and rts:
        rts = [rts]
    if rts:
        bullets.append("우선 반박 대상: " + ", ".join(map(str, rts)))
    fixes = adv.get("fixes") or adv.get("tip") or []
    if isinstance(fixes, str) and fixes:
        fixes = [fixes]
    if fixes:
        bullets.append("개선 지시: " + "; ".join(map(str, fixes)))
    reqs = adv.get("evidence_requests") or []
    if reqs:
        bullets.append("근거 보강: " + "; ".join(map(str, reqs)))
    if bullets:
        judge_hint = "[저지 조언]\n- " + "\n- ".join(bullets)
        sys_prompt = judge_hint + "\n" + sys_prompt

    try:
        response = chat_once(
            st.session_state.get(f"AI{ai_idx+1}_model", models[0]),
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
            temperature=temperature, top_p=top_p, num_ctx=int(num_ctx), seed=int(seed_base + 30_000)
        ) or "계속 주장을 전개합니다."
    except Exception as e:
        response = f"계속 주장을 전개합니다. (생성 실패: {e})"

    chat.setdefault("messages", [])
    chat["messages"].append({"role": ai_role, "content": response})

    st.session_state.show_user_judge = False
    st.session_state.user_judge_choice = ""
    st.session_state.last_manual_continue = {"ai": ai_role, "text": response}


def _manual_judge_advice_callback(chat_id: str):
    if chat_id not in st.session_state.chats:
        st.warning("채팅을 찾을 수 없습니다.")
        return
    chat_msgs = st.session_state.chats[chat_id].get("messages", [])
    payload = make_judge_advice_payload(chat_msgs, ai_roles, st.session_state.languages)
    try:
        raw_adv = chat_once(
            judge_models_multi[0],
            [{"role": "system", "content": payload["system"]}, {"role": "user", "content": payload["user"]}],
            temperature=0.0, top_p=1.0, num_ctx=int(num_ctx), seed=int(seed_base + 40_000)
        )
        adv = parse_advice(raw_adv, ai_roles)
    except Exception as e:
        adv = {r: {"tip": f"저지 호출 실패: {e}", "rebut": ""} for r in ai_roles}
        raw_adv = "(error)"

    st.session_state.judge_result = json.dumps(adv, ensure_ascii=False, indent=2)
    st.session_state.last_manual_judge_raw = raw_adv
    st.session_state.last_judge = {"winner": None, "scores": {}, "per_ai_advice": adv}


# =============== Streamlit 앱 ===============
st.set_page_config(page_title="AI Debate Room (개선+정확도옵션+max_turns)", layout="wide")
st.sidebar.title("Settings")

# 언어
if "languages" not in st.session_state:
    st.session_state.languages = "Korean"
st.session_state.languages = st.sidebar.selectbox("Choose languages", ["Korean", "English"], index=0)

# AI 수
if "NumberOfAi" not in st.session_state:
    st.session_state.NumberOfAi = 2
num_ai = st.sidebar.slider("AI 인원", 2, MAX_AI, st.session_state.NumberOfAi, 1)
st.session_state.NumberOfAi = num_ai
ai_roles = [f"AI{i+1}" for i in range(num_ai)]

# 모델 목록
check_ollama()
try:
    models = [m["model"] for m in ollama.list()["models"]]
except Exception:
    models = []
if not models:
    st.sidebar.error("설치된 Ollama 모델이 없습니다. 예: `ollama pull mistral`")
    st.stop()

# 각 AI별 모델 선택(모델별 그룹 → 그룹 1콜 번들 생성)
for i in range(num_ai):
    key = f"AI{i+1}_model"
    default_idx = 1 if st.session_state.get(key) not in models else models.index(st.session_state[key])
    st.sidebar.selectbox(f"AI{i+1} 모델", models, index=default_idx, key=key)

# 각 AI 주장 성향
for i in range(num_ai):
    key = f"AI{i+1}_setting"
    st.session_state[key] = st.sidebar.text_area(
        f"AI{i+1} 주장 경향성",
        value=st.session_state.get(key, ""),
        help="예) '사과를 더 선호해', '어린이 말투로'... 등등"
    )

# 번호 형식 자동생성(원기능 유지)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 번호 형식 의견 생성")
st.session_state.setdefault("numbered_topic", "")
st.session_state.setdefault("numbered_contents", None)

topic = st.sidebar.text_input("주제(예: 무슨 옷을 입을까?)", key="sb_topic_numbered")
default_name = "gemma3:latest"
default_idx = models.index(default_name) if default_name in models else 1
gen_model = st.sidebar.selectbox("실행 모델", models, index=default_idx, key="sb_model_numbered")
sb_temp = st.sidebar.slider("temperature(opinion)", 0.0, 1.5, 0.6, 0.1, key="sb_temp_numbered")
sb_topp = st.sidebar.slider("top_p(opinion)", 0.1, 1.0, 0.95, 0.05, key="sb_topp_numbered")

def _make_numbered(gen_model: str, topic: str, N: int) -> List[str]:
    sys = (
        "너는 사용자 주제에 대해 서로 대비되는 여러 입장을 만든다.\n"
        f"출력은 **오직 {N}줄**, 각 줄은 숫자와 점으로 시작. 코드펜스/빈줄 금지.\n"
        f"예: 1. …\\n2. …\\n...\\n{N}. …\n"
        "각 줄은 한 문장으로, 한국어만."
    )
    usr = f"주제: {topic}"
    raw = chat_once(gen_model, [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                    temperature=sb_temp, top_p=sb_topp)
    text = (raw or "").strip()
    pairs = re.findall(r"(?m)^\s*(\d+)\.\s*(.+?)\s*$", text)
    by_num = {}
    for num_str, content in pairs:
        try:
            k = int(num_str)
        except ValueError:
            continue
        if 1 <= k <= N and k not in by_num:
            by_num[k] = content.strip()
    if len(by_num) < N:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for l in lines:
            if len(by_num) >= N:
                break
            if not re.match(r"^\d+\.\s*", l):
                by_num[len(by_num) + 1] = l
    return [by_num.get(i, "") for i in range(1, N + 1)]

if st.sidebar.button("▶ 번호 형식 생성", key="sb_make_numbered"):
    if not (topic or "").strip():
        st.sidebar.warning("주제를 입력하세요.")
    else:
        st.session_state["numbered_topic"] = (topic or "").strip()
        st.session_state["numbered_contents"] = _make_numbered(gen_model, topic, num_ai)
        st.sidebar.success(f"의견 {num_ai}개 저장 완료")

if st.session_state.get("numbered_contents"):
    st.sidebar.markdown("**결과**")
    if st.session_state.get("numbered_topic"):
        st.sidebar.markdown(f"- 주제: {st.session_state['numbered_topic']}")
    for i, c in enumerate(st.session_state["numbered_contents"], 1):
        st.sidebar.markdown(f"**{i}.**")
        st.sidebar.code(c or "", language="text")


# 공통 생성 하이퍼파라미터
st.sidebar.markdown("### ⚙️ 생성 파라미터 (조정해도 유의미한 변화 없음)")
temperature = st.sidebar.slider("temperature", 0.0, 1.5, 0.4, 0.1)
top_p = st.sidebar.slider("top_p", 0.1, 1.0, 0.9, 0.05)
max_sents = st.sidebar.slider("발언 문장 수(권장 최대)", 3, 8, 6, 1)
max_turns = st.sidebar.slider("토론 최대 턴수", 1, 8, 3, 1)

# 정확도 향상 옵션(저지 앙상블)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 정확도 향상 옵션 (Judge 앙상블)")
num_ctx = st.sidebar.number_input("num_ctx", 2048, 32768, 8192, step=1024)
top_k = st.sidebar.number_input("top_k", 8, 200, 40, step=8)
repeat_penalty = st.sidebar.number_input("repeat_penalty", 1.0, 2.0, 1.1, step=0.05)
seed_base = st.sidebar.number_input("seed", 0, 10_000_000, 42, step=1)

n_judge = st.sidebar.slider("저지 표 수(n_judge)", 1, 9, 5, step=2, help='이 값은 토론 시간에 큰 영향을 미칩니다. 1~3 추천')
judge_models_multi = st.sidebar.multiselect(
    "저지 모델(복수 선택 가능)",
    models,
    default=[m for m in ["mistral", "gemma3:latest"] if m in models] or [models[1]]
)
if not judge_models_multi:
    judge_models_multi = [models[0]]
st.sidebar.caption("여러 저지 모델 × 여러 표 → 평균/표준편차로 안정화")

st.sidebar.markdown("### 🧑‍⚖️ 저지 설정", help= '두 개 다 체크해 놓으시는 것을 권장합니다.')
use_judge_guidance = st.sidebar.checkbox("턴 종료마다 저지 피드백 반영", value=True)
show_judge_panel = st.sidebar.checkbox("저지 결과 패널 보이기", value=True)


# =============== 채팅 세션 ===============
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

@st.dialog("새 채팅 만들기")
def new_chat_dialog():
    chatings_name = st.text_input("채팅 이름을 입력하세요", key="dlg_new_chat_name")
    if st.button("확인", key="dlg_new_chat_ok"):
        name = (chatings_name or "").strip() or "Untitled Chat"
        chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = chat_id
        st.session_state.chats[chat_id] = {"name": name, "messages": []}
        st.rerun()

if st.sidebar.button("➕ New Chat", key="sidebar_new_chat"):
    new_chat_dialog()

# 채팅 목록
for cid, chat_info in list(st.session_state.chats.items()):
    label = (chat_info.get("name") or "").strip() or "Untitled Chat"
    if st.sidebar.button(label, key=f"chat_btn_{cid}"):
        st.session_state.current_chat_id = cid
        st.rerun()

# 아바타
emoji_numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
avatar_map = {f"AI{i+1}": emoji_numbers[i] for i in range(num_ai)}
avatar_map["user"] = "👤"
st.session_state.avatar_map = avatar_map

# 판단/조언 상태
st.session_state.setdefault("show_user_judge", False)
st.session_state.setdefault("show_model_judge", False)
st.session_state.setdefault("user_judge_choice", "")
st.session_state.setdefault("judge_result", "")
st.session_state.setdefault("last_role_scores", {})
st.session_state.setdefault("_judge_logs", [])
st.session_state.setdefault("_judge_stds", {})

# =============== 본문 ===============
chat_id = st.session_state.current_chat_id
if chat_id:
    chat = st.session_state.chats[chat_id]
    st.title(chat["name"])

    # 기록 렌더(숨김/내부 지시 제거)
    for msg in chat["messages"]:
        if msg["role"] == "system" or msg.get("_hidden"):
            continue
        if str(msg["role"]).endswith("_instruction"):
            continue
        with st.chat_message(msg["role"], avatar=avatar_map.get(msg["role"], "💬")):
            st.markdown(msg["content"])

    # 사용자 입력
    user_input = st.chat_input(topic)
    if user_input:
        # 사용자 메시지 저장/표시
        chat["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar=avatar_map["user"]):
            st.markdown(user_input)

        # 🔁 max_turns 루프 시작
        for turn in range(1, max_turns + 1):
            st.markdown(f"### 🔄 Turn {turn}/{max_turns}")
            # ===== 1) 모델별 그룹 → 번들 1콜 생성 =====
            model_groups: Dict[str, List[int]] = {}
            for i in range(num_ai):
                mname = st.session_state.get(f"AI{i+1}_model", models[0])
                model_groups.setdefault(mname, []).append(i)

            for mname, idxs in model_groups.items():
                roles = [f"AI{j+1}" for j in idxs]
                settings_map = {f"AI{j+1}": st.session_state.get(f"AI{j+1}_setting", "") for j in idxs}
                talks = generate_bundle_for_group(
                    model_name=mname, roles=roles, settings_map=settings_map,
                    lang=st.session_state.languages, topic_or_user_turn=user_input if turn == 1 else "이전 턴 발언과 저지 조언을 반영해 계속 전개",
                    max_sents=max_sents, temperature=temperature, top_p=top_p,
                    num_ctx=num_ctx, seed=seed_base + 1000*turn, top_k=top_k, repeat_penalty=repeat_penalty
                )
                for r in roles:
                    text = talks.get(r, "")
                    # 중립 경고(숨김 피드백)
                    if any(k in text.lower() for k in ["both", "depends", "personal preference", "중립", "균형", "equally valid"]):
                        warn = "⚠️ 당신의 발언이 너무 중립적입니다. 자신의 입장을 강하게 다시 주장하세요."
                        chat["messages"].append({"role": "user", "content": warn, "_hidden": True})
                    with st.chat_message(r, avatar=avatar_map[r]):
                        st.markdown(text)
                    chat["messages"].append({"role": r, "content": text})

            # ========== 2) 턴 종료: 저지 평가 & 다음 턴 개선 지시 주입 ==========
            if use_judge_guidance:
                judge_json, jerr = get_judge_feedback(chat["messages"], st.session_state.languages)
                st.session_state["last_judge"] = judge_json or {}
                st.session_state["last_judge_err"] = jerr

                if judge_json:
                    if show_judge_panel:
                        with st.expander(f"🧑‍⚖️ 저지 피드백 (Turn {turn})", expanded=True):
                            winner = judge_json.get("winner", "N/A")
                            scores = judge_json.get("scores", {})
                            st.markdown(f"**우승(임시 판단)**: {winner}")
                            if scores:
                                cols = st.columns(max(2, len(scores)))
                                for i, (k, v) in enumerate(sorted(scores.items())):
                                    with cols[i % len(cols)]:
                                        st.metric(k, f"{v:.1f}/10")
                            st.code(json.dumps(judge_json, ensure_ascii=False, indent=2), language="json")

                    # 숨김 지시문 주입(다음 턴 반영)
                    per_ai = judge_json.get("per_ai_advice", {}) or {}
                    for i in range(num_ai):
                        ai_role = f"AI{i+1}"
                        adv = per_ai.get(ai_role, {})
                        if not adv:
                            continue
                        guide_lines = []
                        if adv.get("summary"):
                            guide_lines.append(f"[저지 요약] {adv['summary']}")
                        rts = adv.get("rebut_targets") or []
                        if rts:
                            guide_lines.append(f"[반박 대상] {', '.join(rts)}")
                        fixes = adv.get("fixes") or []
                        if fixes:
                            guide_lines.append("[개선 지시]\n- " + "\n- ".join(map(str, fixes)))
                        reqs = adv.get("evidence_requests") or []
                        if reqs:
                            guide_lines.append("[근거 보강]\n- " + "\n- ".join(map(str, reqs)))
                        setting = st.session_state.get(f"{ai_role}_setting", "")
                        setting_line = f"[고정 세팅 재확인] {setting}" if setting else ""
                        instruction_text = "\n".join([setting_line] + guide_lines).strip()
                        if instruction_text:
                            chat["messages"].append({"role": f"{ai_role}_instruction", "content": instruction_text})

                    chat["messages"].append({
                        "role": "user",
                        "content": ("저지의 조언을 반영하여 다음 턴에서 주장을 더 강하게 전개하고, 지정된 반박 대상을 우선 공략하세요. 중립적 결론 금지.")
                    })
                else:
                    pass
                    #if show_judge_panel and jerr:
                     #   with st.expander("저지 원문(파싱 실패 디버그)", expanded=False):
                      #      st.code(st.session_state.get("_last_judge_raw", "(원문 없음)"))
                       # st.warning(f"저지 피드백 오류: {jerr}")

            # ========== 3) 턴 종료: 다중 저지 앙상블 스코어 ==========
            means = ensemble_judge_scores(
                judge_models_multi=judge_models_multi,
                n_judge=int(n_judge),
                messages=chat["messages"],
                roles=ai_roles,
                num_ctx=int(num_ctx),
                seed_base=int(seed_base + 10_000 + 1000*turn)
            )
            stds = st.session_state["_judge_stds"]
            leader = max(ai_roles, key=lambda r: means.get(r, 0.0))
            with st.expander(f"⚖️ Judge 앙상블 점수 (Turn {turn})"):
                st.write({r: round(means.get(r, 0.0), 1) for r in ai_roles})
                if stds: 
                    st.caption("표준편차(낮을수록 합의 높음): " + ", ".join([f"{r}:{stds[r]:.1f}" for r in ai_roles]))
                st.info(f"이번 턴 우세: **{leader}** (평균 {means.get(leader, 0):.1f})")
                with st.expander("저지 로그(일부)"):
                    for line in st.session_state["_judge_logs"]:
                        st.code(line)

            chat["messages"].append({"role": "system", "content": f"[JUDGE_SCORES]{json.dumps(means, ensure_ascii=False)}", "_hidden": True})

            # ========== 4) Judge 모델 '다음 턴 가이드' 생성 & 숨김 주입 ==========
            adv = get_advice_with_retries(
                judge_models_multi=judge_models_multi,
                messages=chat["messages"],
                roles=ai_roles,
                lang=st.session_state.languages,
                num_ctx=int(num_ctx),
                seed=int(seed_base + 20_000 + 1000*turn)
            )


            with st.expander(f"🧭 Judge 모델 조언(다음 턴 가이드 · Turn {turn})"):
                for r in ai_roles:
                    tip = adv[r]["tip"]; reb = adv[r]["rebut"]
                    st.markdown(f"- **{r}** → TIP: {tip or '(비어있음)'} / REBUT: {reb or '(비어있음)'}")

            chat["messages"].append({
                "role": "user",
                "content": f"[JUDGE_FEEDBACK]{json.dumps({'leader':leader,'advice':adv}, ensure_ascii=False)}",
                "_hidden": True
            })

        # 🔚 모든 턴 종료 후: 최종 요약/우승
        st.divider()
        st.markdown("## 🏁 최종 결과")
        final_json, ferr = get_final_summary_robust(
            chat["messages"], ai_roles, st.session_state.languages, judge_models_multi
        )
        if final_json:
            st.markdown("### 📜 내용 요약")
            st.write(final_json.get("summary", ""))
            st.markdown("### 🧩 각 AI 장·단점")
            for r in ai_roles:
                item = final_json["per_ai"].get(r, {"strengths": [], "weaknesses": []})
                st.markdown(f"**{r}**")
                st.write("- 강점: " + (", ".join(item.get("strengths") or []) or "없음"))
                st.write("- 약점: " + (", ".join(item.get("weaknesses") or []) or "없음"))
            st.markdown("### 🏆 최종 우승자")
            st.success(f'우승: **{final_json.get("final_winner","N/A")}**')
            st.caption(final_json.get("reason", ""))
        else:
            with st.expander("최종 저지 원문(파싱 실패 디버그)", expanded=False):
                st.code(st.session_state.get("_last_final_judge_raw", "(원문 없음)"))
            st.warning(f"최종 요약 오류: {ferr or '알 수 없는 오류'}")

    # ---- 👤 사용자 승자 선택 / 이어가기 ----
    if st.button("👤 사용자 승자 선택"):
        st.session_state.show_user_judge = True

    if st.session_state.get("show_user_judge", False):
        choice = st.selectbox("승자를 선택하세요", ai_roles, key="user_choice_select")
        st.session_state.user_judge_choice = choice
        st.success(f"👤 사용자 판단: {choice} 승리!")

        st.button(
            "선택된 AI가 주장 이어가기",
            on_click=_continue_ai_callback,
            args=(st.session_state.current_chat_id, choice)
        )

        st.button(
            "🧐 JudgeModel 조언(수동)",
            on_click=_manual_judge_advice_callback,
            args=(st.session_state.current_chat_id,)
        )

        if st.session_state.get("judge_result"):
            st.markdown("### 🧭 JudgeModel 판단/조언 (수동 저장)")
            st.code(st.session_state.judge_result)
            if st.session_state.get("last_manual_judge_raw"):
                with st.expander("JudgeModel 원문(파싱 실패 디버그)", expanded=False):
                    st.code(st.session_state.last_manual_judge_raw)


else:
    st.info("오늘의 토론 주제는 무엇인가요?\n왼쪽에서 채팅을 선택하거나 새 채팅을 만들어주세요.")

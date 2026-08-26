from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from full_feature_extractor import extract_103_features
from predictor import predict_research_score

st.set_page_config(page_title="Cognitive Aim", page_icon="🎯", layout="wide")
ROOT = Path(__file__).parent
FEATURE_COLUMNS = json.loads((ROOT / "model_feature_columns.json").read_text(encoding="utf-8"))
MODEL_META = json.loads((ROOT / "deployment_model_metadata.json").read_text(encoding="utf-8"))
REFERENCE = json.loads((ROOT / "research_reference_stats.json").read_text(encoding="utf-8"))


def load_static_component(component_name, folder_name):
    parts = sorted((ROOT / folder_name).glob("part*.txt"))
    if not parts:
        raise FileNotFoundError(f"{folder_name} component parts were not found.")
    html = "".join(p.read_text(encoding="utf-8") for p in parts)
    runtime_dir = Path(tempfile.gettempdir()) / f"{component_name}_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "index.html").write_text(html, encoding="utf-8")
    return components.declare_component(component_name, path=str(runtime_dir))


aim_component = load_static_component("cognitive_aim_component_v1", "aim_component_parts")
ctmt_component = load_static_component("ctmt_mouse_component_v2", "ctmt_component_parts")

st.markdown("""
<style>
:root{--ink:#07111f;--cyan:#1fd7c5;--mint:#73ffd2;--line:#d7e4ea}
.block-container{max-width:1180px;padding-top:1.45rem;padding-bottom:3rem}
.hero{position:relative;overflow:hidden;padding:30px 32px;border:1px solid #18364a;border-radius:22px;background:linear-gradient(135deg,#07111f,#0d2334 58%,#12354a);margin:6px 0 20px;color:white}
.hero:after{content:"";position:absolute;inset:0;background:repeating-linear-gradient(180deg,transparent 0 5px,rgba(255,255,255,.018) 6px);pointer-events:none}
.kicker{font-size:12px;font-weight:900;letter-spacing:.14em;color:#73ffd2;text-transform:uppercase}
.hero h1{font-size:40px;letter-spacing:-.045em;margin:7px 0 8px;color:#f7fcff}
.hero p{font-size:15px;line-height:1.7;color:#b9cbd6;max-width:900px;margin:0}
.mission-strip{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 0}.mission-chip{border:1px solid #2b5368;background:rgba(10,31,47,.72);border-radius:999px;padding:7px 11px;font-size:11px;font-weight:850;letter-spacing:.05em;color:#d8f6ef}
.operator{display:flex;gap:14px;align-items:center;border:1px solid #cde6e2;border-radius:17px;padding:15px 17px;background:#f4fbfa;margin:10px 0 18px}.operator-avatar{width:48px;height:48px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#0d2334;font-size:26px;flex:0 0 48px}.operator .name{font-size:11px;font-weight:900;letter-spacing:.08em;color:#0b756c;text-transform:uppercase}.operator .msg{font-size:14px;line-height:1.55;color:#344054;margin-top:3px}
.card{border:1px solid #dbe6eb;border-radius:17px;padding:17px 18px;background:white;height:100%}.card .small{font-size:11px;font-weight:900;color:#0b756c;letter-spacing:.08em;text-transform:uppercase}.card .big{font-size:27px;font-weight:950;color:#172033;margin:4px 0}.card .desc{font-size:13px;color:#667085;line-height:1.5}
.score{border:1px solid #9adbd2;background:linear-gradient(145deg,#effbf8,#f8fcff);border-radius:22px;padding:24px}.score .label{font-size:12px;font-weight:900;letter-spacing:.06em;color:#0b756c;text-transform:uppercase}.score .number{font-size:52px;font-weight:950;color:#087f72;letter-spacing:-.05em;line-height:1.05;margin:6px 0}.score .desc{font-size:12px;color:#667085;line-height:1.5}
.notice{border-left:4px solid #1ba899;background:#f4fbfa;border-radius:9px;padding:13px 15px;color:#475467;font-size:13px;line-height:1.55;margin-top:16px}.summary-card{border:1px solid #b9e1dc;background:#f4fbfa;border-radius:16px;padding:17px 19px;margin:14px 0 18px}.summary-card .eyebrow{font-size:11px;font-weight:900;letter-spacing:.08em;color:#0b756c;margin-bottom:6px;text-transform:uppercase}.summary-card .headline{font-size:19px;font-weight:850;color:#172033;line-height:1.45}.summary-card .sub{font-size:12px;color:#667085;line-height:1.5;margin-top:7px}.section-tag{font-size:12px;font-weight:900;letter-spacing:.1em;color:#0b756c;text-transform:uppercase;margin-bottom:2px}
div[data-testid="stMetric"]{border:1px solid #dbe6eb;border-radius:15px;padding:14px;background:#fff}
</style>
""", unsafe_allow_html=True)


def percentile(name, value):
    r = REFERENCE[name]
    xs = np.array([r["min"], r["q1"], r["median"], r["q3"], r["max"]], dtype=float)
    ps = np.array([0, 25, 50, 75, 100], dtype=float)
    ux, up = [], []
    for x, p in zip(xs, ps):
        if ux and x <= ux[-1]:
            up[-1] = max(up[-1], p)
        else:
            ux.append(x)
            up.append(p)
    if len(ux) == 1:
        return 50.0
    return float(np.interp(value, np.array(ux), np.array(up), left=0, right=100))


def fmt(name, value):
    if name == "non_cut_correct_targets_touches_PART_B":
        return f"{value:.2f} / 15"
    if name == "is_valid_sum_B":
        return f"{value:.0f}%"
    if name == "non_cut_rt_PART_B":
        return f"{value/1000:.2f} s"
    if name == "max_duration_PART_B":
        return f"{value:.3f} s"
    return f"{value:.3f}"


def build_one_line_summary(features):
    p_correct = percentile("non_cut_correct_targets_touches_PART_B", features["non_cut_correct_targets_touches_PART_B"])
    p_stable = percentile("is_valid_sum_B", features["is_valid_sum_B"])
    p_switch_acc = percentile("non_cut_correct_targets_touches_B_A_ratio", features["non_cut_correct_targets_touches_B_A_ratio"])
    p_rt = percentile("non_cut_rt_PART_B", features["non_cut_rt_PART_B"])
    p_hes = percentile("max_duration_PART_B", features["max_duration_PART_B"])
    p_trans = percentile("state_transitions_B_A_ratio", features["state_transitions_B_A_ratio"])
    strengths, burdens = [], []
    if p_correct >= 75: strengths.append("정확성")
    if p_stable >= 75: strengths.append("수행 안정성")
    if p_switch_acc >= 75: strengths.append("전환 정확성")
    if p_rt >= 75: burdens.append("반응 지연")
    if p_hes >= 75: burdens.append("망설임")
    if p_trans >= 75: burdens.append("행동 전환 증가")
    if len(strengths) >= 2 and not burdens:
        return "정확성과 수행 안정성이 높게 나타났고, 전환 과제에서도 비교적 안정적인 수행 패턴이 관찰되었습니다."
    if len(burdens) >= 2 and not strengths:
        return f"전환 과제에서 {', '.join(burdens[:2])} 특성이 상대적으로 두드러진 행동 패턴이 관찰되었습니다."
    if strengths and burdens:
        return f"{', '.join(strengths[:2])}은 비교적 높게 나타났지만, {', '.join(burdens[:2])} 특성도 함께 관찰되었습니다."
    if strengths:
        return f"{', '.join(strengths[:2])} 지표가 연구표본에서 상대적으로 높은 위치에 나타났습니다."
    if burdens:
        return f"{', '.join(burdens[:2])} 특성이 연구표본에서 상대적으로 높은 위치에 나타났습니다."
    return "핵심 인지행동 지표가 연구표본의 중앙 구간과 대체로 유사한 패턴으로 나타났습니다."


def aim_round_stats(r):
    clicks = r.get("clicks", [])
    correct = [c for c in clicks if c.get("correct")]
    mean_rt = float(np.mean([c.get("reaction_ms", np.nan) for c in correct])) if correct else np.nan
    pts = r.get("pointer_events", [])
    distance = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        distance += float(np.hypot(b["x_norm"] - a["x_norm"], b["y_norm"] - a["y_norm"]))
    return {
        "part": r.get("part"),
        "correct": r.get("correct_hits", 0),
        "wrong": r.get("wrong_target_clicks", 0),
        "miss": r.get("miss_clicks", 0),
        "duration_s": r.get("duration_ms", 0) / 1000,
        "mean_rt_ms": mean_rt,
        "trajectory_distance_norm": distance,
    }


st.markdown("""
<div class="hero">
  <div class="kicker">Cognitive Aim // Game-based Mouse Biomarker</div>
  <h1>FPS형 인지행동 Aim 미션</h1>
  <p>두 개의 훈련 타깃 중 올바른 숫자·문자를 선택해 조준하고 클릭합니다. 정답 타깃이 제거되면 다음 타깃이 새 위치에 등장하며, 전 과정의 마우스 궤적·반응시간·오선택을 기록합니다.</p>
  <div class="mission-strip"><span class="mission-chip">2 TARGET ROLLING</span><span class="mission-chip">AIM + CLICK</span><span class="mission-chip">SEQUENCE / SWITCHING</span><span class="mission-chip">MOUSE TRAJECTORY</span></div>
</div>
""", unsafe_allow_html=True)

tab_intro, tab_aim, tab_research, tab_result = st.tabs(["01 BRIEFING", "02 COGNITIVE AIM", "03 RESEARCH cTMT", "04 RESEARCH REPORT"])

with tab_intro:
    st.markdown("""
    <div class="operator"><div class="operator-avatar">🎯</div><div><div class="name">Mission Concept</div><div class="msg">서비스 모드에서는 두 타깃을 동시에 제시하고 Aim + Click 행동을 측정합니다. 기존 논문 기반 cTMT는 별도의 Research Mode로 보존해 기술 검증 근거와 서비스 확장을 분리합니다.</div></div></div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        (c1, "Round A", "Sequential Aim", "1 → 2 → 3 순서로 두 타깃 중 올바른 훈련 타깃을 선택합니다."),
        (c2, "Round B", "Switching Aim", "1 → A → 2 → B처럼 숫자와 문자를 번갈아 선택합니다."),
        (c3, "Behavior", "Aim + Click", "마우스 궤적, 반응시간, 오선택, miss, 이동거리를 기록합니다."),
    ]
    for col, small, big, desc in cards:
        with col:
            st.markdown(f'<div class="card"><div class="small">{small}</div><div class="big">{big}</div><div class="desc">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="notice"><b>연구 범위</b><br>Cognitive Aim은 기존 cTMT를 그대로 재현한 검사가 아니라, cTMT의 순차 처리와 전환 규칙을 FPS형 상호작용으로 확장한 서비스 프로토타입입니다. 게임 모드 결과에는 기존 cTMT SVM을 적용하지 않습니다.</div>', unsafe_allow_html=True)

with tab_aim:
    st.markdown('<div class="section-tag">SERVICE MODE // TWO-TARGET COGNITIVE AIM</div>', unsafe_allow_html=True)
    st.subheader("Cognitive Aim Mission")
    st.caption("항상 2개의 훈련용 타깃이 보입니다. 올바른 코드의 머리를 조준하고 클릭하면 해당 타깃이 사라지고 다음 코드가 새 위치에 등장합니다.")
    aim_data = aim_component(key="cognitive_aim_v1", default=None)
    if aim_data:
        st.session_state["aim_session"] = aim_data
    aim_pack = st.session_state.get("aim_session")
    if aim_pack and aim_pack.get("completed"):
        st.markdown("### AFTER ACTION REPORT // GAME MODE")
        summary = aim_pack.get("summary", {})
        rounds = [aim_round_stats(r) for r in aim_pack.get("rounds", [])]
        rmap = {r["part"]: r for r in rounds}
        a, b = rmap.get("A", {}), rmap.get("B", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target Accuracy", f"{summary.get('accuracy', 0)*100:.1f}%")
        c2.metric("Correct Hits", f"{summary.get('correct_hits', 0)} / 30")
        c3.metric("Wrong Target", summary.get("wrong_target_clicks", 0))
        c4.metric("Miss Click", summary.get("miss_clicks", 0))
        d1, d2, d3 = st.columns(3)
        rt_a = a.get("mean_rt_ms", np.nan)
        rt_b = b.get("mean_rt_ms", np.nan)
        d1.metric("Round A Mean RT", "-" if np.isnan(rt_a) else f"{rt_a/1000:.2f} s")
        d2.metric("Round B Mean RT", "-" if np.isnan(rt_b) else f"{rt_b/1000:.2f} s")
        switch_cost = rt_b - rt_a if not np.isnan(rt_a) and not np.isnan(rt_b) else np.nan
        d3.metric("Switching Cost", "-" if np.isnan(switch_cost) else f"{switch_cost/1000:+.2f} s")
        if rounds:
            st.dataframe(pd.DataFrame(rounds).rename(columns={"part":"Round","correct":"정답","wrong":"오선택","miss":"Miss","duration_s":"수행시간(s)","mean_rt_ms":"평균 반응시간(ms)","trajectory_distance_norm":"정규화 이동거리"}), use_container_width=True, hide_index=True)
        st.markdown('<div class="notice"><b>게임형 행동 리포트</b><br>이 결과는 새로운 Cognitive Aim 과제의 수행 특성을 요약한 값입니다. 현재 단계에서는 MCI 또는 인지장애를 판정하지 않으며 기존 cTMT SVM 점수와 동일하게 해석하지 않습니다.</div>', unsafe_allow_html=True)
        st.download_button("COGNITIVE AIM JSON 다운로드", data=json.dumps(aim_pack, ensure_ascii=False, indent=2).encode("utf-8"), file_name=f"{aim_pack.get('session_id','cognitive_aim')}.json", mime="application/json")

with tab_research:
    st.markdown('<div class="section-tag">VALIDATION MODE // ORIGINAL cTMT PIPELINE</div>', unsafe_allow_html=True)
    st.subheader("Research cTMT")
    st.caption("선행연구 기반 103-feature extraction과 SVM 검증을 위한 연구 모드입니다. 서비스형 Cognitive Aim과 데이터 파이프라인을 분리합니다.")
    with st.expander("RESEARCH MODE RULES", expanded=False):
        st.markdown("- Part A: `1 → 2 → 3 → ...`\n- Part B: `1 → A → 2 → B → ...`\n- Training A/B + 분석 20 trials\n- 각 trial 최대 25초\n- 목표는 클릭하지 않고 포인터로 통과")
    session_data = ctmt_component(key="ctmt_step12", default=None)
    if session_data:
        st.session_state["ctmt_session"] = session_data
        try:
            extracted = extract_103_features(session_data, FEATURE_COLUMNS)
            if extracted["model_ready"]:
                prediction = predict_research_score(extracted["features"])
                st.session_state["ctmt_result"] = {"extracted": extracted, "prediction": prediction}
                st.success("Research cTMT 분석 완료. **04 RESEARCH REPORT**에서 확인하세요.")
            else:
                st.warning("103개 Feature를 모두 생성하지 못했습니다. 유효 trial을 확인해주세요.")
        except Exception as exc:
            st.error(f"분석 처리 중 오류: {exc}")

with tab_result:
    pack = st.session_state.get("ctmt_result")
    raw = st.session_state.get("ctmt_session")
    if not pack:
        st.info("먼저 **03 RESEARCH cTMT**를 완료하고 분석 데이터를 전송해주세요.")
    else:
        extracted = pack["extracted"]
        pred = pack["prediction"]
        f = extracted["features"]
        score = pred["research_probability_mci_pattern"] * 100
        one_line_summary = build_one_line_summary(f)
        st.markdown('<div class="section-tag">RESEARCH REPORT // DIGITAL MOUSE BIOMARKER</div>', unsafe_allow_html=True)
        st.subheader("Research cTMT Analysis")
        left, right = st.columns([0.42, 0.58], gap="large")
        with left:
            st.markdown(f'<div class="score"><div class="label">인지행동 패턴 지수</div><div class="number">{score:.1f}</div><div class="desc">SVM MCI-class 출력값을 0–100 스케일로 표시한 연구용 지수이며 질병 발생확률이나 진단값이 아닙니다.</div></div>', unsafe_allow_html=True)
            st.progress(min(max(score / 100, 0.0), 1.0))
            st.caption(f"Digital-only SVM Nested CV 재현 ROC-AUC {MODEL_META['official_reproduction_auc_nested_cv']:.3f}")
        with right:
            st.markdown("#### RESEARCH QUALITY")
            q1, q2, q3 = st.columns(3)
            q1.metric("Valid Part A", f"{f['is_valid_sum_A']:.0f}%")
            q2.metric("Valid Part B", f"{f['is_valid_sum_B']:.0f}%")
            q3.metric("Features", "103")
        st.markdown(f'<div class="summary-card"><div class="eyebrow">Research Summary</div><div class="headline">{one_line_summary}</div><div class="sub">74명 연구표본 내 상대 위치 기반 연구용 행동 요약입니다.</div></div>', unsafe_allow_html=True)
        names = ["non_cut_correct_targets_touches_PART_B","is_valid_sum_B","non_cut_correct_targets_touches_B_A_ratio","non_cut_rt_PART_B","max_duration_PART_B","state_transitions_B_A_ratio"]
        st.markdown("#### 핵심 행동지표 6개")
        for group in (names[:3], names[3:]):
            cols = st.columns(3)
            for i, name in enumerate(group):
                with cols[i]:
                    st.metric(REFERENCE[name]["label"], fmt(name, f[name]))
                    st.caption(REFERENCE[name]["desc"])
        st.markdown("#### 연구표본 내 상대 위치")
        rows = [{"지표":REFERENCE[n]["label"],"현재값":fmt(n,f[n]),"연구표본 중앙값":fmt(n,REFERENCE[n]["median"]),"상대 위치(약)":round(percentile(n,f[n]),1)} for n in names]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown('<div class="notice"><b>해석 주의</b><br>본 결과는 연구용 디지털 행동 분석 결과입니다. MCI 또는 치매를 진단하지 않으며 의료진의 평가를 대체하지 않습니다.</div>', unsafe_allow_html=True)
        payload = {"session_id":raw.get("session_id") if raw else None,"one_line_summary":one_line_summary,"research_model_output":pred,"core_features":{n:f[n] for n in names},"all_103_features":f,"model_metadata":{"training_subjects":MODEL_META["training_subjects"],"nested_cv_reproduction_auc":MODEL_META["official_reproduction_auc_nested_cv"],"input_features":MODEL_META["input_features"],"selected_features":MODEL_META["selected_features"]}}
        st.download_button("RESEARCH REPORT JSON 다운로드", data=json.dumps(payload,ensure_ascii=False,indent=2).encode("utf-8"), file_name=f"{payload['session_id'] or 'ctmt'}_result.json", mime="application/json")
        with st.expander("연구·기술 정보"):
            st.json({"training_subjects":MODEL_META["training_subjects"],"input_features":MODEL_META["input_features"],"selected_features":MODEL_META["selected_features"],"deployment_params":MODEL_META["best_params_full_data_inner_cv"],"nested_cv_reproduction_auc":MODEL_META["official_reproduction_auc_nested_cv"],"model_class_at_default_threshold":pred["model_class_at_default_threshold"]})

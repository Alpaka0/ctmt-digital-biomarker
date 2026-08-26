from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from full_feature_extractor import extract_103_features
from predictor import predict_research_score

st.set_page_config(page_title="cTMT Cognitive Recon", page_icon="🛰️", layout="wide")
ROOT = Path(__file__).parent
FEATURE_COLUMNS = json.loads((ROOT / "model_feature_columns.json").read_text(encoding="utf-8"))
MODEL_META = json.loads((ROOT / "deployment_model_metadata.json").read_text(encoding="utf-8"))
REFERENCE = json.loads((ROOT / "research_reference_stats.json").read_text(encoding="utf-8"))

_component_parts = sorted((ROOT / "ctmt_component_parts").glob("part*.txt"))
if not _component_parts:
    raise FileNotFoundError("cTMT component parts were not found.")
_component_html = "".join(p.read_text(encoding="utf-8") for p in _component_parts)
_component_dir = Path(tempfile.gettempdir()) / "ctmt_component_runtime"
_component_dir.mkdir(parents=True, exist_ok=True)
(_component_dir / "index.html").write_text(_component_html, encoding="utf-8")
ctmt_component = components.declare_component("ctmt_mouse_component_v2", path=str(_component_dir))

st.markdown("""
<style>
:root{--ink:#07111f;--ink2:#0d1b2a;--cyan:#1fd7c5;--mint:#73ffd2;--line:#d7e4ea}
.block-container{max-width:1180px;padding-top:1.5rem;padding-bottom:3rem}
.hero{position:relative;overflow:hidden;padding:30px 32px;border:1px solid #18364a;border-radius:22px;background:linear-gradient(135deg,#07111f,#0d2334 58%,#12354a);margin:6px 0 20px;color:white}
.hero:after{content:"";position:absolute;inset:0;background:repeating-linear-gradient(180deg,transparent 0 5px,rgba(255,255,255,.018) 6px);pointer-events:none}
.kicker{font-size:12px;font-weight:900;letter-spacing:.14em;color:#73ffd2;text-transform:uppercase}
.hero h1{font-size:40px;letter-spacing:-.045em;margin:7px 0 8px;color:#f7fcff}
.hero p{font-size:15px;line-height:1.7;color:#b9cbd6;max-width:850px;margin:0}
.mission-strip{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0 0}
.mission-chip{border:1px solid #2b5368;background:rgba(10,31,47,.72);border-radius:999px;padding:7px 11px;font-size:11px;font-weight:850;letter-spacing:.05em;color:#d8f6ef}
.operator{display:flex;gap:14px;align-items:center;border:1px solid #cde6e2;border-radius:17px;padding:15px 17px;background:#f4fbfa;margin:10px 0 18px}
.operator-avatar{width:48px;height:48px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#0d2334;font-size:26px;flex:0 0 48px}
.operator .name{font-size:11px;font-weight:900;letter-spacing:.08em;color:#0b756c;text-transform:uppercase}
.operator .msg{font-size:14px;line-height:1.55;color:#344054;margin-top:3px}
.card{border:1px solid #dbe6eb;border-radius:17px;padding:17px 18px;background:white;height:100%}
.card .small{font-size:11px;font-weight:900;color:#0b756c;letter-spacing:.08em;text-transform:uppercase}
.card .big{font-size:27px;font-weight:950;color:#172033;margin:4px 0}
.card .desc{font-size:13px;color:#667085;line-height:1.5}
.score{border:1px solid #9adbd2;background:linear-gradient(145deg,#effbf8,#f8fcff);border-radius:22px;padding:24px}
.score .label{font-size:12px;font-weight:900;letter-spacing:.06em;color:#0b756c;text-transform:uppercase}
.score .number{font-size:52px;font-weight:950;color:#087f72;letter-spacing:-.05em;line-height:1.05;margin:6px 0}
.score .desc{font-size:12px;color:#667085;line-height:1.5}
.notice{border-left:4px solid #1ba899;background:#f4fbfa;border-radius:9px;padding:13px 15px;color:#475467;font-size:13px;line-height:1.55;margin-top:16px}
.summary-card{border:1px solid #b9e1dc;background:#f4fbfa;border-radius:16px;padding:17px 19px;margin:14px 0 18px}
.summary-card .eyebrow{font-size:11px;font-weight:900;letter-spacing:.08em;color:#0b756c;margin-bottom:6px;text-transform:uppercase}
.summary-card .headline{font-size:19px;font-weight:850;color:#172033;line-height:1.45}
.summary-card .sub{font-size:12px;color:#667085;line-height:1.5;margin-top:7px}
.section-tag{font-size:12px;font-weight:900;letter-spacing:.1em;color:#0b756c;text-transform:uppercase;margin-bottom:2px}
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

    strengths = []
    if p_correct >= 75:
        strengths.append("정확성")
    if p_stable >= 75:
        strengths.append("수행 안정성")
    if p_switch_acc >= 75:
        strengths.append("전환 정확성")

    burdens = []
    if p_rt >= 75:
        burdens.append("반응 지연")
    if p_hes >= 75:
        burdens.append("망설임")
    if p_trans >= 75:
        burdens.append("행동 전환 증가")

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


st.markdown("""
<div class="hero">
  <div class="kicker">Cognitive Recon // Digital Mouse Biomarker</div>
  <h1>cTMT 인지 정찰 미션</h1>
  <p>검사 구조는 그대로 유지한 채 FPS의 Mission HUD 경험을 적용했습니다. 마우스 궤적에서 103개 행동 특성을 생성하고 공개 연구데이터 기반 SVM으로 인지행동 패턴을 분석합니다.</p>
  <div class="mission-strip">
    <span class="mission-chip">MISSION 20 TRIALS</span>
    <span class="mission-chip">TRACKING 103 FEATURES</span>
    <span class="mission-chip">MODEL SVM</span>
    <span class="mission-chip">RESEARCH PROTOTYPE</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_intro, tab_test, tab_result = st.tabs(["01 BRIEFING", "02 MISSION", "03 AFTER ACTION REPORT"])

with tab_intro:
    st.markdown("""
    <div class="operator">
      <div class="operator-avatar">🧑‍🚀</div>
      <div>
        <div class="name">Mission Operator // Echo</div>
        <div class="msg">이번 미션의 목표는 숫자와 문자를 순서대로 추적하는 것입니다. 사격이나 게임 조작은 없습니다. 평소처럼 마우스를 움직여 경로만 자연스럽게 따라가 주세요.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    cards = [
        (c1, "Mission", "20 Trials", "Part A/B를 교대로 수행하며 X/Y 좌표와 timestamp를 연속 기록합니다."),
        (c2, "Tracking", "103 Features", "속도·가속도·경로·Search/Travel/Hesitation을 동일 feature pipeline으로 계산합니다."),
        (c3, "Evidence", f"AUC {MODEL_META['official_reproduction_auc_nested_cv']:.3f}", "74명 공개데이터에서 재현한 digital-only SVM nested-CV 성능입니다."),
    ]
    for col, small, big, desc in cards:
        with col:
            st.markdown(f'<div class="card"><div class="small">{small}</div><div class="big">{big}</div><div class="desc">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("### MISSION FLOW")
    st.write("Training A 1회 → Training B 1회 → Mission A/B 20회. 각 trial은 중앙 `⊕`를 클릭해 tracking을 초기화한 뒤 25초 이내에 목표를 순서대로 포인터로 통과합니다.")
    st.markdown('<div class="notice"><b>Research safety notice</b><br>본 서비스는 연구·교육용 프로토타입입니다. 의료기기나 진단도구가 아니며 결과를 실제 MCI 발생확률 또는 의료적 진단으로 해석해서는 안 됩니다.</div>', unsafe_allow_html=True)

with tab_test:
    st.markdown('<div class="section-tag">LIVE MISSION // TRACKING MODE</div>', unsafe_allow_html=True)
    st.subheader("cTMT Mission Console")
    st.caption("검사 도중 창 크기를 바꾸지 말고 평소 사용하는 마우스 또는 터치패드를 사용하세요. FPS 스타일은 HUD에만 적용되며 검사 규칙은 기존 cTMT와 동일합니다.")
    st.markdown("""
    <div class="operator">
      <div class="operator-avatar">🧑‍🚀</div>
      <div>
        <div class="name">Operator Echo // Mission Brief</div>
        <div class="msg"><b>Part A</b>는 숫자를 순서대로, <b>Part B</b>는 숫자와 문자를 번갈아 추적합니다. 중앙 ⊕를 클릭한 뒤 목표를 클릭하지 말고 포인터로 통과하세요.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("MISSION RULES", expanded=False):
        st.markdown("- Part A: `1 → 2 → 3 → ...`\n- Part B: `1 → A → 2 → B → ...`\n- 각 trial 최대 25초\n- 목표는 클릭하지 않고 포인터로 통과\n- 완료 후 **ANALYZE / Python 전송**")

    session_data = ctmt_component(key="ctmt_step12", default=None)
    if session_data:
        st.session_state["ctmt_session"] = session_data
        try:
            extracted = extract_103_features(session_data, FEATURE_COLUMNS)
            if extracted["model_ready"]:
                prediction = predict_research_score(extracted["features"])
                st.session_state["ctmt_result"] = {"extracted": extracted, "prediction": prediction}
                st.success("MISSION DATA 분석 완료. **03 AFTER ACTION REPORT** 탭에서 확인할 수 있습니다.")
            else:
                st.warning("103개 Feature를 모두 생성하지 못했습니다. 유효 trial을 확인해주세요.")
                with st.expander("미생성 Feature"):
                    st.write(extracted["missing_features"])
        except Exception as exc:
            st.error(f"분석 처리 중 오류: {exc}")

with tab_result:
    pack = st.session_state.get("ctmt_result")
    raw = st.session_state.get("ctmt_session")
    if not pack:
        st.info("먼저 **02 MISSION** 탭에서 cTMT를 완료하고 분석 데이터를 전송해주세요.")
    else:
        extracted = pack["extracted"]
        pred = pack["prediction"]
        f = extracted["features"]
        score = pred["research_probability_mci_pattern"] * 100
        one_line_summary = build_one_line_summary(f)

        st.markdown('<div class="section-tag">AFTER ACTION REPORT // COGNITIVE BEHAVIOR PROFILE</div>', unsafe_allow_html=True)
        st.subheader("Mission Analysis Complete")
        left, right = st.columns([0.42, 0.58], gap="large")
        with left:
            st.markdown(
                f'<div class="score"><div class="label">인지행동 패턴 지수</div>'
                f'<div class="number">{score:.1f}</div>'
                f'<div class="desc">SVM의 MCI-class 출력값을 0–100 스케일로 표시한 연구용 지수입니다. 질병 발생확률이나 진단값이 아닙니다.</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(score / 100, 0.0), 1.0))
            st.caption(f"Digital-only SVM Nested CV 재현 ROC-AUC {MODEL_META['official_reproduction_auc_nested_cv']:.3f}")
        with right:
            st.markdown("#### MISSION QUALITY")
            q1, q2, q3 = st.columns(3)
            q1.metric("Valid Part A", f"{f['is_valid_sum_A']:.0f}%")
            q2.metric("Valid Part B", f"{f['is_valid_sum_B']:.0f}%")
            q3.metric("Tracked Features", "103")
            if f["is_valid_sum_A"] < 80 or f["is_valid_sum_B"] < 80:
                st.warning("유효 trial 비율이 낮아 결과 해석에 주의가 필요합니다.")

        st.markdown(
            f'<div class="summary-card"><div class="eyebrow">Operator Summary</div>'
            f'<div class="headline">{one_line_summary}</div>'
            f'<div class="sub">74명 연구표본 내 상대 위치를 바탕으로 한 연구용 행동 요약이며 의료적 진단을 의미하지 않습니다.</div></div>',
            unsafe_allow_html=True,
        )

        names = [
            "non_cut_correct_targets_touches_PART_B",
            "is_valid_sum_B",
            "non_cut_correct_targets_touches_B_A_ratio",
            "non_cut_rt_PART_B",
            "max_duration_PART_B",
            "state_transitions_B_A_ratio",
        ]
        st.markdown("#### BEHAVIOR LOADOUT // 핵심 행동지표 6개")
        for group in (names[:3], names[3:]):
            cols = st.columns(3)
            for i, name in enumerate(group):
                with cols[i]:
                    st.metric(REFERENCE[name]["label"], fmt(name, f[name]))
                    st.caption(REFERENCE[name]["desc"])

        st.markdown("#### RESEARCH REFERENCE // 연구표본 내 상대 위치")
        st.caption("74명 연구표본 요약분포 기준입니다. 높은 값이 반드시 좋거나 나쁜 뜻은 아닙니다.")
        rows = [
            {
                "지표": REFERENCE[name]["label"],
                "현재값": fmt(name, f[name]),
                "연구표본 중앙값": fmt(name, REFERENCE[name]["median"]),
                "상대 위치(약)": round(percentile(name, f[name]), 1),
            }
            for name in names
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        chart = pd.DataFrame({
            "지표": [REFERENCE[name]["label"] for name in names],
            "연구표본 내 상대 위치(%)": [percentile(name, f[name]) for name in names],
        }).set_index("지표")
        st.bar_chart(chart)

        notable = [REFERENCE[name]["label"] for name in names if percentile(name, f[name]) <= 25 or percentile(name, f[name]) >= 75]
        st.markdown("#### OPERATOR NOTE // 분석 요약")
        st.markdown(f"**요약:** {one_line_summary}")
        if notable:
            st.write(f"이번 검사에서는 **{', '.join(notable[:3])}** 지표가 연구표본의 중앙 구간에서 비교적 벗어난 위치에 있었습니다. 장치·마우스 사용 습관·피로도·검사 환경 등의 영향이 있으므로 단독으로 해석하지 않습니다.")
        else:
            st.write("이번 검사에서 핵심 행동지표들은 연구표본의 중앙 구간에 주로 위치했습니다. 단일 검사 결과만으로 인지상태를 판단할 수는 없습니다.")

        st.markdown('<div class="notice"><b>해석 주의</b><br>본 결과는 연구용 디지털 행동 분석 결과입니다. MCI 또는 치매를 진단하지 않으며 의료진의 평가를 대체하지 않습니다. 반복 측정과 외부검증이 추가로 필요합니다.</div>', unsafe_allow_html=True)

        payload = {
            "session_id": raw.get("session_id") if raw else None,
            "one_line_summary": one_line_summary,
            "research_model_output": pred,
            "core_features": {name: f[name] for name in names},
            "all_103_features": f,
            "model_metadata": {
                "training_subjects": MODEL_META["training_subjects"],
                "nested_cv_reproduction_auc": MODEL_META["official_reproduction_auc_nested_cv"],
                "input_features": MODEL_META["input_features"],
                "selected_features": MODEL_META["selected_features"],
            },
        }
        st.download_button(
            "AFTER ACTION REPORT JSON 다운로드",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{payload['session_id'] or 'ctmt'}_result.json",
            mime="application/json",
        )

        with st.expander("연구·기술 정보"):
            st.json({
                "training_subjects": MODEL_META["training_subjects"],
                "input_features": MODEL_META["input_features"],
                "selected_features": MODEL_META["selected_features"],
                "deployment_params": MODEL_META["best_params_full_data_inner_cv"],
                "nested_cv_reproduction_auc": MODEL_META["official_reproduction_auc_nested_cv"],
                "model_class_at_default_threshold": pred["model_class_at_default_threshold"],
            })

        with st.expander("103개 전체 Feature"):
            st.dataframe(pd.DataFrame({"feature": FEATURE_COLUMNS, "value": [f[c] for c in FEATURE_COLUMNS]}), use_container_width=True, hide_index=True)

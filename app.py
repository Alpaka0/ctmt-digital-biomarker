from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from full_feature_extractor import extract_103_features
from predictor import predict_research_score

st.set_page_config(page_title="cTMT Digital Biomarker", page_icon="🖱️", layout="wide")
ROOT = Path(__file__).parent
FEATURE_COLUMNS = json.loads((ROOT / "model_feature_columns.json").read_text(encoding="utf-8"))
MODEL_META = json.loads((ROOT / "deployment_model_metadata.json").read_text(encoding="utf-8"))
REFERENCE = json.loads((ROOT / "research_reference_stats.json").read_text(encoding="utf-8"))
ctmt_component = components.declare_component("ctmt_mouse_component_v2", path=str(ROOT / "ctmt_component"))

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:2rem;padding-bottom:3rem}
.hero{padding:28px 30px;border:1px solid #e4e7ec;border-radius:22px;background:linear-gradient(135deg,#f8fbff,#ffffff 58%,#f5f7ff);margin:6px 0 22px}
.kicker{font-size:12px;font-weight:800;letter-spacing:.08em;color:#3154b8;text-transform:uppercase}
.hero h1{font-size:38px;letter-spacing:-.04em;margin:7px 0 8px;color:#172033}
.hero p{font-size:16px;line-height:1.65;color:#667085;max-width:820px;margin:0}
.card{border:1px solid #e4e7ec;border-radius:17px;padding:17px 18px;background:white;height:100%}
.card .small{font-size:12px;font-weight:800;color:#667085;letter-spacing:.04em;text-transform:uppercase}
.card .big{font-size:27px;font-weight:900;color:#172033;margin:4px 0}
.card .desc{font-size:13px;color:#667085;line-height:1.45}
.score{border:1px solid #cfe0ff;background:#f8fbff;border-radius:22px;padding:24px}
.score .label{font-size:14px;font-weight:800;color:#475467}
.score .number{font-size:52px;font-weight:950;color:#1d4ed8;letter-spacing:-.05em;line-height:1.05;margin:6px 0}
.notice{border-left:4px solid #98a2b3;background:#f9fafb;border-radius:9px;padding:13px 15px;color:#475467;font-size:13px;line-height:1.55;margin-top:16px}
div[data-testid="stMetric"]{border:1px solid #e4e7ec;border-radius:15px;padding:14px;background:#fff}
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
            ux.append(x); up.append(p)
    if len(ux) == 1:
        return 50.0
    return float(np.interp(value, np.array(ux), np.array(up), left=0, right=100))


def fmt(name, value):
    if name == "non_cut_correct_targets_touches_PART_B": return f"{value:.2f} / 15"
    if name == "is_valid_sum_B": return f"{value:.0f}%"
    if name == "non_cut_rt_PART_B": return f"{value/1000:.2f} s"
    if name == "max_duration_PART_B": return f"{value:.3f} s"
    return f"{value:.3f}"


st.markdown("""
<div class="hero">
  <div class="kicker">Digital Mouse Biomarker Research Prototype</div>
  <h1>마우스 움직임으로 보는 인지행동 패턴</h1>
  <p>cTMT 수행 중 마우스 궤적을 수집해 103개 디지털 행동 특성을 생성하고, 공개 연구데이터로 학습된 SVM과 연결합니다.</p>
</div>
""", unsafe_allow_html=True)

tab_intro, tab_test, tab_result = st.tabs(["01 소개", "02 검사", "03 결과"])

with tab_intro:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><div class="small">Capture</div><div class="big">20 Trials</div><div class="desc">Part A/B를 교대로 수행하며 X/Y 좌표와 timestamp를 연속 기록합니다.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="small">Analyze</div><div class="big">103 Features</div><div class="desc">속도·가속도·경로·Search/Travel/Hesitation을 공개 코드 정의에 맞춰 계산합니다.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><div class="small">Model</div><div class="big">AUC {MODEL_META["official_reproduction_auc_nested_cv"]:.3f}</div><div class="desc">74명 공개데이터에서 재현한 nested-CV 성능입니다.</div></div>', unsafe_allow_html=True)
    st.subheader("검사 흐름")
    st.write("연습 A 1회 → 연습 B 1회 → 분석 A/B 20회. 각 trial은 중앙 `+` 클릭 후 시작하고 25초 이내에 원을 순서대로 포인터로 통과합니다.")
    st.markdown('<div class="notice">본 서비스는 연구·교육용 프로토타입입니다. 의료기기나 진단도구가 아니며 결과를 실제 MCI 발생확률 또는 의료적 진단으로 해석해서는 안 됩니다.</div>', unsafe_allow_html=True)

with tab_test:
    st.subheader("cTMT 검사")
    st.caption("검사 도중 창 크기를 바꾸지 말고 평소 사용하는 마우스 또는 터치패드를 사용하세요.")
    with st.expander("검사 전 확인", expanded=False):
        st.markdown("- Part A: `1 → 2 → 3 → ...`\n- Part B: `1 → A → 2 → B → ...`\n- 각 trial 최대 25초\n- 목표는 클릭하지 않고 포인터로 통과\n- 완료 후 **Python으로 전송**")

    session_data = ctmt_component(key="ctmt_step11", default=None)
    if session_data:
        st.session_state["ctmt_step11_session"] = session_data
        try:
            extracted = extract_103_features(session_data, FEATURE_COLUMNS)
            if extracted["model_ready"]:
                prediction = predict_research_score(extracted["features"])
                st.session_state["ctmt_step11_result"] = {"extracted": extracted, "prediction": prediction}
                st.success("분석 완료. **03 결과** 탭에서 확인할 수 있습니다.")
            else:
                st.warning("103개 Feature를 모두 생성하지 못했습니다. 유효 trial을 확인해주세요.")
                with st.expander("미생성 Feature"):
                    st.write(extracted["missing_features"])
        except Exception as exc:
            st.error(f"분석 처리 중 오류: {exc}")

with tab_result:
    pack = st.session_state.get("ctmt_step11_result")
    raw = st.session_state.get("ctmt_step11_session")
    if not pack:
        st.info("먼저 **02 검사** 탭에서 cTMT를 완료하고 Python으로 전송해주세요.")
    else:
        extracted = pack["extracted"]
        pred = pack["prediction"]
        f = extracted["features"]
        score = pred["research_probability_mci_pattern"] * 100

        st.subheader("인지행동 분석 결과")
        left, right = st.columns([0.42, 0.58], gap="large")
        with left:
            st.markdown(f'<div class="score"><div class="label">MCI 관련 행동패턴 점수</div><div class="number">{score:.1f}%</div><div class="desc">SVM의 MCI-class 출력값이며 실제 질병 발생확률은 아닙니다.</div></div>', unsafe_allow_html=True)
            st.progress(min(max(score / 100, 0.0), 1.0))
            st.caption(f"Nested CV 재현 ROC-AUC {MODEL_META['official_reproduction_auc_nested_cv']:.3f}")
        with right:
            st.markdown("#### 검사 품질")
            q1, q2, q3 = st.columns(3)
            q1.metric("Valid Part A", f"{f['is_valid_sum_A']:.0f}%")
            q2.metric("Valid Part B", f"{f['is_valid_sum_B']:.0f}%")
            q3.metric("분석 Feature", "103개")
            if f["is_valid_sum_A"] < 80 or f["is_valid_sum_B"] < 80:
                st.warning("유효 trial 비율이 낮아 결과 해석에 주의가 필요합니다.")

        names = [
            "non_cut_correct_targets_touches_PART_B",
            "is_valid_sum_B",
            "non_cut_correct_targets_touches_B_A_ratio",
            "non_cut_rt_PART_B",
            "max_duration_PART_B",
            "state_transitions_B_A_ratio",
        ]
        st.markdown("#### 핵심 행동지표 6개")
        cols = st.columns(3)
        for i, n in enumerate(names[:3]):
            with cols[i]:
                st.metric(REFERENCE[n]["label"], fmt(n, f[n])); st.caption(REFERENCE[n]["desc"])
        cols = st.columns(3)
        for i, n in enumerate(names[3:]):
            with cols[i]:
                st.metric(REFERENCE[n]["label"], fmt(n, f[n])); st.caption(REFERENCE[n]["desc"])

        st.markdown("#### 연구표본 내 상대 위치")
        st.caption("74명 연구표본 요약분포 기준입니다. 높은 값이 반드시 좋거나 나쁜 뜻은 아닙니다.")
        rows = [{"지표": REFERENCE[n]["label"], "현재값": fmt(n, f[n]), "연구표본 중앙값": fmt(n, REFERENCE[n]["median"]), "상대 위치(약)": round(percentile(n, f[n]), 1)} for n in names]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        chart = pd.DataFrame({"지표": [REFERENCE[n]["label"] for n in names], "연구표본 내 상대 위치(%)": [percentile(n, f[n]) for n in names]}).set_index("지표")
        st.bar_chart(chart)

        notable = [REFERENCE[n]["label"] for n in names if percentile(n, f[n]) <= 25 or percentile(n, f[n]) >= 75]
        st.markdown("#### 분석 요약")
        if notable:
            st.write(f"이번 검사에서는 **{', '.join(notable[:3])}** 지표가 연구표본의 중앙 구간에서 비교적 벗어난 위치에 있었습니다. 장치·마우스 사용 습관·피로도·검사 환경 등의 영향이 있으므로 단독으로 해석하지 않습니다.")
        else:
            st.write("이번 검사에서 핵심 행동지표들은 연구표본의 중앙 구간에 주로 위치했습니다. 단일 검사 결과만으로 인지상태를 판단할 수는 없습니다.")

        st.markdown('<div class="notice"><b>해석 주의</b><br>본 결과는 연구용 디지털 행동 분석 결과입니다. MCI 또는 치매를 진단하지 않으며 의료진의 평가를 대체하지 않습니다. 반복 측정과 외부검증이 추가로 필요합니다.</div>', unsafe_allow_html=True)

        payload = {
            "session_id": raw.get("session_id") if raw else None,
            "research_model_output": pred,
            "core_features": {n: f[n] for n in names},
            "all_103_features": f,
            "model_metadata": {
                "training_subjects": MODEL_META["training_subjects"],
                "nested_cv_reproduction_auc": MODEL_META["official_reproduction_auc_nested_cv"],
                "input_features": MODEL_META["input_features"],
                "selected_features": MODEL_META["selected_features"],
            },
        }
        st.download_button("분석 결과 JSON 다운로드", data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), file_name=f"{payload['session_id'] or 'ctmt'}_result.json", mime="application/json")

        with st.expander("연구·기술 정보"):
            st.json({"training_subjects": MODEL_META["training_subjects"], "input_features": MODEL_META["input_features"], "selected_features": MODEL_META["selected_features"], "deployment_params": MODEL_META["best_params_full_data_inner_cv"], "nested_cv_reproduction_auc": MODEL_META["official_reproduction_auc_nested_cv"], "model_class_at_default_threshold": pred["model_class_at_default_threshold"]})

        with st.expander("103개 전체 Feature"):
            st.dataframe(pd.DataFrame({"feature": FEATURE_COLUMNS, "value": [f[c] for c in FEATURE_COLUMNS]}), use_container_width=True, hide_index=True)

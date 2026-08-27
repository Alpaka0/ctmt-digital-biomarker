from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from full_feature_extractor import extract_103_features
from predictor import predict_research_score

st.set_page_config(page_title="Cognitive Mole", page_icon="🧠", layout="wide")
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


mole_component = load_static_component("cognitive_mole_component_v2", "aim_component_parts")
ctmt_component = load_static_component("ctmt_mouse_component_v2", "ctmt_component_parts")

st.markdown("""
<style>
:root{--green:#4fae58;--green-dark:#245b2b;--green-soft:#f2faee;--yellow:#ffd85e;--line:#d8e8d3;--text:#243127;--muted:#6d786f}
.block-container{max-width:1180px;padding-top:1.45rem;padding-bottom:3rem}
.hero{position:relative;overflow:hidden;padding:30px 32px;border:1px solid #cfe5c9;border-radius:22px;background:linear-gradient(135deg,#eff9e9,#fafff7 58%,#f4fbe9);margin:6px 0 20px;color:var(--text);box-shadow:0 10px 28px rgba(62,108,55,.08)}
.hero:after{content:"";position:absolute;right:-30px;bottom:-65px;width:220px;height:220px;border-radius:50%;background:radial-gradient(circle,#dff1c8 0,#ecf7df 45%,transparent 68%);pointer-events:none}
.kicker{font-size:12px;font-weight:900;letter-spacing:.14em;color:#3d8d47;text-transform:uppercase}
.hero h1{font-size:40px;letter-spacing:-.045em;margin:7px 0 8px;color:#1f4726}
.hero p{font-size:15px;line-height:1.7;color:#5d6c60;max-width:920px;margin:0}
.mission-strip{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 0}.mission-chip{border:1px solid #aad7a8;background:rgba(255,255,255,.78);border-radius:999px;padding:7px 11px;font-size:11px;font-weight:850;letter-spacing:.05em;color:#367b3e}
.operator{display:flex;gap:14px;align-items:center;border:1px solid #d4e8d0;border-radius:17px;padding:15px 17px;background:#f6fbf3;margin:10px 0 18px}.operator-avatar{width:48px;height:48px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#e4f3db;font-size:25px;flex:0 0 48px}.operator .name{font-size:11px;font-weight:900;letter-spacing:.08em;color:#3b8a43;text-transform:uppercase}.operator .msg{font-size:14px;line-height:1.55;color:#435046;margin-top:3px}
.card{border:1px solid #dce9d8;border-radius:17px;padding:17px 18px;background:white;height:100%}.card .small{font-size:11px;font-weight:900;color:#3d8d47;letter-spacing:.08em;text-transform:uppercase}.card .big{font-size:25px;font-weight:950;color:#26352a;margin:4px 0}.card .desc{font-size:13px;color:#68736a;line-height:1.5}
.score{border:1px solid #9adbd2;background:linear-gradient(145deg,#effbf8,#f8fcff);border-radius:22px;padding:24px}.score .label{font-size:12px;font-weight:900;letter-spacing:.06em;color:#0b756c;text-transform:uppercase}.score .number{font-size:52px;font-weight:950;color:#087f72;letter-spacing:-.05em;line-height:1.05;margin:6px 0}.score .desc{font-size:12px;color:#667085;line-height:1.5}
.notice{border-left:4px solid #59ad5f;background:#f5fbf2;border-radius:9px;padding:13px 15px;color:#526057;font-size:13px;line-height:1.55;margin-top:16px}.summary-card{border:1px solid #b9e1dc;background:#f4fbfa;border-radius:16px;padding:17px 19px;margin:14px 0 18px}.summary-card .eyebrow{font-size:11px;font-weight:900;letter-spacing:.08em;color:#0b756c;margin-bottom:6px;text-transform:uppercase}.summary-card .headline{font-size:19px;font-weight:850;color:#172033;line-height:1.45}.summary-card .sub{font-size:12px;color:#667085;line-height:1.5;margin-top:7px}.section-tag{font-size:12px;font-weight:900;letter-spacing:.1em;color:#3d8d47;text-transform:uppercase;margin-bottom:2px}
div[data-testid="stMetric"]{border:1px solid #dbe6eb;border-radius:15px;padding:14px;background:#fff}
.traj-title{font-size:12px;font-weight:900;color:#385b3d;margin:4px 0 6px}
.research-shell{margin-top:28px}
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
    if len(strengths) >= 2 and not burdens: return "정확성과 수행 안정성이 높게 나타났고, 전환 과제에서도 비교적 안정적인 수행 패턴이 관찰되었습니다."
    if len(burdens) >= 2 and not strengths: return f"전환 과제에서 {', '.join(burdens[:2])} 특성이 상대적으로 두드러진 행동 패턴이 관찰되었습니다."
    if strengths and burdens: return f"{', '.join(strengths[:2])}은 비교적 높게 나타났지만, {', '.join(burdens[:2])} 특성도 함께 관찰되었습니다."
    if strengths: return f"{', '.join(strengths[:2])} 지표가 연구표본에서 상대적으로 높은 위치에 나타났습니다."
    if burdens: return f"{', '.join(burdens[:2])} 특성이 연구표본에서 상대적으로 높은 위치에 나타났습니다."
    return "핵심 인지행동 지표가 연구표본의 중앙 구간과 대체로 유사한 패턴으로 나타났습니다."


def _safe_float(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _path_efficiency(r):
    events = r.get("pointer_events", [])
    hits = [c for c in r.get("clicks", []) if c.get("correct")]
    if not events or not hits:
        return np.nan
    prev_t = 0.0
    prev_point = None
    vals = []
    for hit in hits:
        hit_t = _safe_float(hit.get("t_ms"), np.nan)
        if np.isnan(hit_t):
            continue
        pts = [
            (_safe_float(p.get("x_norm")), _safe_float(p.get("y_norm")))
            for p in events
            if prev_t <= _safe_float(p.get("t_ms"), -1) <= hit_t
        ]
        pts = [(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y)]
        if prev_point is not None:
            pts.insert(0, prev_point)
        hit_point = (_safe_float(hit.get("x_norm")), _safe_float(hit.get("y_norm")))
        if not all(np.isfinite(hit_point)):
            continue
        pts.append(hit_point)
        if len(pts) >= 2:
            actual = sum(float(np.hypot(b[0]-a[0], b[1]-a[1])) for a, b in zip(pts[:-1], pts[1:]))
            direct = float(np.hypot(pts[-1][0]-pts[0][0], pts[-1][1]-pts[0][1]))
            if actual > 0:
                vals.append(max(0.0, min(1.0, direct / actual)))
        prev_t = hit_t
        prev_point = hit_point
    return float(np.mean(vals)) if vals else np.nan


def mole_round_stats(r):
    clicks = r.get("clicks", [])
    correct = [c for c in clicks if c.get("correct")]
    rts = [_safe_float(c.get("reaction_ms")) for c in correct]
    rts = [x for x in rts if np.isfinite(x)]
    pts = r.get("pointer_events", [])
    distance = sum(
        float(np.hypot(_safe_float(b.get("x_norm")) - _safe_float(a.get("x_norm")),
                       _safe_float(b.get("y_norm")) - _safe_float(a.get("y_norm"))))
        for a, b in zip(pts[:-1], pts[1:])
        if all(np.isfinite([
            _safe_float(a.get("x_norm")), _safe_float(a.get("y_norm")),
            _safe_float(b.get("x_norm")), _safe_float(b.get("y_norm"))
        ]))
    )
    correct_hits = int(r.get("correct_hits", 0))
    wrong = int(r.get("wrong_target_clicks", 0))
    miss = int(r.get("miss_clicks", 0))
    total = max(1, correct_hits + wrong + miss)
    return {
        "part": r.get("part"),
        "correct_hits": correct_hits,
        "wrong_target_clicks": wrong,
        "miss_clicks": miss,
        "error_count": wrong + miss,
        "accuracy": correct_hits / total,
        "duration_ms": _safe_float(r.get("duration_ms"), 0.0),
        "mean_rt_ms": float(np.mean(rts)) if rts else np.nan,
        "median_rt_ms": float(np.median(rts)) if rts else np.nan,
        "trajectory_distance_norm": distance,
        "path_efficiency": _path_efficiency(r),
        "pointer_events": len(pts),
    }


def mole_trajectory_svg(r):
    if not r:
        return '<div style="height:280px;border:1px solid #dce9d8;border-radius:12px;background:#f7fbf4;display:flex;align-items:center;justify-content:center;color:#7a877d;font-size:12px">trajectory data 없음</div>'
    events = r.get("pointer_events", [])
    hits = [c for c in r.get("clicks", []) if c.get("correct")]
    max_pts = 420
    step = max(1, int(np.ceil(len(events) / max_pts))) if events else 1
    sample = events[::step]
    coords = []
    for p in sample:
        x, y = _safe_float(p.get("x_norm")), _safe_float(p.get("y_norm"))
        if np.isfinite(x) and np.isfinite(y):
            coords.append(f"{x*960:.1f},{y*540:.1f}")
    polyline = " ".join(coords)
    marks = []
    for i, c in enumerate(hits, start=1):
        x, y = _safe_float(c.get("x_norm")), _safe_float(c.get("y_norm"))
        if np.isfinite(x) and np.isfinite(y):
            marks.append(
                f'<circle cx="{x*960:.1f}" cy="{y*540:.1f}" r="8" fill="#fff" stroke="#3f9f4c" stroke-width="4"/>'
                f'<text x="{x*960:.1f}" y="{y*540+4:.1f}" text-anchor="middle" font-size="10" font-weight="900" fill="#245b2b">{i}</text>'
            )
    return (
        '<svg viewBox="0 0 960 540" width="100%" role="img" '
        f'aria-label="Round {r.get("part","")} mouse trajectory" '
        'style="display:block;background:#f7fbf4;border:1px solid #dce9d8;border-radius:12px">'
        f'<polyline points="{polyline}" fill="none" stroke="#79a985" stroke-width="3" '
        'stroke-linecap="round" stroke-linejoin="round" opacity=".82"/>'
        + "".join(marks) + '</svg>'
    )


def _signed(v, digits=0, suffix=""):
    if not np.isfinite(v):
        return "-"
    return f"{v:+.{digits}f}{suffix}"


st.markdown("""
<div class="hero">
  <div class="kicker">COGNITIVE MOLE // PAPER-INSPIRED MOUSE TASK</div>
  <h1>순서 기억 두더지 게임</h1>
  <p>순차 처리와 과제 전환 규칙을 바탕으로 구성한 게임형 인지행동 과제입니다. 올바른 순서의 두더지를 선택하는 과정에서 마우스 궤적, 반응시간, 정확성, 오선택 패턴을 기록합니다.</p>
  <div class="mission-strip"><span class="mission-chip">RANDOM HOLES</span><span class="mission-chip">CLICK TASK</span><span class="mission-chip">SEQUENCE / SWITCHING</span><span class="mission-chip">MOUSE TRAJECTORY</span></div>
</div>
""", unsafe_allow_html=True)

tab_intro, tab_mole = st.tabs(["01 BRIEFING", "02 COGNITIVE MOLE"])

with tab_intro:
    st.markdown("""
    <div class="operator"><div class="operator-avatar">🧠</div><div><div class="name">Service Concept</div><div class="msg">선행연구의 cTMT에서 사용된 순차 처리, 과제 전환, 마우스 궤적 측정 개념을 두더지 게임형 상호작용으로 확장했습니다. 기존 cTMT 검증 모드는 별도로 유지합니다.</div></div></div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        (c1, "Round A", "Sequential Processing", "1 → 2 → 3 순서에 따라 올바른 두더지를 찾아 클릭합니다."),
        (c2, "Round B", "Task Switching", "1 → A → 2 → B처럼 숫자와 문자를 번갈아 선택합니다."),
        (c3, "Behavior", "Mouse Trajectory", "이동 궤적, 반응시간, 오선택, MISS, 이동거리를 기록합니다."),
    ]
    for col, small, big, desc in cards:
        with col:
            st.markdown(f'<div class="card"><div class="small">{small}</div><div class="big">{big}</div><div class="desc">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="notice"><b>연구 범위</b><br>Cognitive Mole은 기존 cTMT를 그대로 재현한 검사가 아니라, 선행연구의 순차 처리·과제 전환·mouse trajectory 개념을 게임형 상호작용으로 확장한 서비스 프로토타입입니다. 게임 결과에는 기존 cTMT SVM을 적용하지 않습니다.</div>', unsafe_allow_html=True)

with tab_mole:
    st.markdown('<div class="section-tag">SERVICE MODE // COGNITIVE MOLE</div>', unsafe_allow_html=True)
    st.subheader("순서 기억 두더지 게임")
    st.caption("매 순간 현재 정답, 다음 후보, 방해 두더지가 서로 다른 구멍에 등장합니다. 규칙에 맞는 두더지를 찾아 망치로 클릭하세요.")
    mole_data = mole_component(key="cognitive_mole_v2", default=None)
    if mole_data:
        st.session_state["mole_session"] = mole_data
    mole_pack = st.session_state.get("mole_session")

    if mole_pack and mole_pack.get("completed"):
        summary = mole_pack.get("summary", {})
        component_report = mole_pack.get("report", {})
        rounds = component_report.get("rounds") or [mole_round_stats(r) for r in mole_pack.get("rounds", [])]
        rmap = {r.get("part"): r for r in rounds}
        a, b = rmap.get("A", {}), rmap.get("B", {})
        raw_rmap = {r.get("part"): r for r in mole_pack.get("rounds", [])}

        correct_hits = int(summary.get("correct_hits", 0))
        wrong = int(summary.get("wrong_target_clicks", 0))
        miss = int(summary.get("miss_clicks", 0))
        completion_rate = correct_hits / 30 * 100
        selection_accuracy = float(summary.get("accuracy", 0)) * 100

        rt_a = _safe_float(a.get("median_rt_ms"))
        rt_b = _safe_float(b.get("median_rt_ms"))
        rt_delta = rt_b - rt_a if np.isfinite(rt_a) and np.isfinite(rt_b) else np.nan
        duration_a = _safe_float(a.get("duration_ms"), 0.0)
        duration_b = _safe_float(b.get("duration_ms"), 0.0)
        duration_delta = duration_b - duration_a
        err_a = int(a.get("error_count", int(a.get("wrong_target_clicks", 0)) + int(a.get("miss_clicks", 0))))
        err_b = int(b.get("error_count", int(b.get("wrong_target_clicks", 0)) + int(b.get("miss_clicks", 0))))
        err_delta = err_b - err_a
        eff_a = _safe_float(a.get("path_efficiency"))
        eff_b = _safe_float(b.get("path_efficiency"))
        eff_delta = eff_b - eff_a if np.isfinite(eff_a) and np.isfinite(eff_b) else np.nan

        st.markdown("### 게임형 인지행동 리포트")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("과제 완료율", f"{completion_rate:.1f}%")
        c2.metric("선택 정확도", f"{selection_accuracy:.1f}%")
        c3.metric("정답 목표", f"{correct_hits} / 30")
        c4.metric("오선택", wrong)
        c5.metric("MISS", miss)

        d1, d2, d3 = st.columns(3)
        d1.metric("Round A 중앙 반응시간", "-" if not np.isfinite(rt_a) else f"{rt_a:.0f} ms")
        d2.metric("Round B 중앙 반응시간", "-" if not np.isfinite(rt_b) else f"{rt_b:.0f} ms")
        d3.metric("교대 규칙 차이 (B−A)", "-" if not np.isfinite(rt_delta) else f"{rt_delta:+.0f} ms")

        summary_parts = []
        if np.isfinite(rt_delta):
            rt_pct = rt_delta / rt_a * 100 if rt_a > 0 else np.nan
            text = f"Round B 중앙 반응시간은 A보다 {_signed(rt_delta, 0, ' ms')}"
            if np.isfinite(rt_pct):
                text += f" ({_signed(rt_pct, 1, '%')})"
            summary_parts.append(text)
        summary_parts.append(f"오선택·MISS는 {err_delta:+d}회 변화")
        if np.isfinite(eff_delta):
            summary_parts.append(f"경로 효율은 {_signed(eff_delta*100, 1, '%p')} 변화")
        one_line = " · ".join(summary_parts) + "."
        st.markdown(
            f'<div class="summary-card"><div class="eyebrow">Performance Summary</div>'
            f'<div class="headline">{one_line}</div>'
            '<div class="sub">Round B−A는 이번 게임 내 교대 규칙 조건에서 관찰된 수행 차이이며 임상적 판정값이 아닙니다.</div></div>',
            unsafe_allow_html=True,
        )

        compare_rows = [
            {
                "지표": "수행시간",
                "Round A": f"{duration_a/1000:.2f} s",
                "Round B": f"{duration_b/1000:.2f} s",
                "B−A": f"{duration_delta/1000:+.2f} s",
            },
            {
                "지표": "중앙 반응시간",
                "Round A": "-" if not np.isfinite(rt_a) else f"{rt_a:.0f} ms",
                "Round B": "-" if not np.isfinite(rt_b) else f"{rt_b:.0f} ms",
                "B−A": "-" if not np.isfinite(rt_delta) else f"{rt_delta:+.0f} ms",
            },
            {
                "지표": "오선택 + MISS",
                "Round A": f"{err_a}회",
                "Round B": f"{err_b}회",
                "B−A": f"{err_delta:+d}회",
            },
            {
                "지표": "경로 효율",
                "Round A": "-" if not np.isfinite(eff_a) else f"{eff_a*100:.1f}%",
                "Round B": "-" if not np.isfinite(eff_b) else f"{eff_b*100:.1f}%",
                "B−A": "-" if not np.isfinite(eff_delta) else f"{eff_delta*100:+.1f}%p",
            },
        ]
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Mouse trajectory")
        t1, t2 = st.columns(2, gap="medium")
        with t1:
            st.markdown('<div class="traj-title">ROUND A · Sequential processing</div>', unsafe_allow_html=True)
            st.markdown(mole_trajectory_svg(raw_rmap.get("A")), unsafe_allow_html=True)
        with t2:
            st.markdown('<div class="traj-title">ROUND B · Alternating rule</div>', unsafe_allow_html=True)
            st.markdown(mole_trajectory_svg(raw_rmap.get("B")), unsafe_allow_html=True)

        st.markdown('<div class="notice"><b>해석 범위</b><br>본 리포트는 Cognitive Mole 게임 수행 중 나타난 행동 특성을 요약합니다. 선택 정확도는 정답 클릭을 전체 클릭(정답+오선택+MISS)으로 나눈 값이며, 과제 완료율과 구분됩니다. Round B−A 차이는 게임 내 조건 차이일 뿐 임상적으로 검증된 switching-cost cut-off가 아닙니다. MCI 또는 인지장애를 판정하지 않으며 기존 cTMT SVM 결과와 동일하게 해석하지 않습니다.</div>', unsafe_allow_html=True)
        st.download_button(
            "COGNITIVE MOLE JSON 다운로드",
            data=json.dumps(mole_pack, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{mole_pack.get('session_id','cognitive_mole')}.json",
            mime="application/json",
        )

st.markdown('<div class="research-shell"></div>', unsafe_allow_html=True)
with st.expander("RESEARCH VALIDATION // ORIGINAL cTMT · 103-FEATURE · SVM", expanded=False):
    st.caption("논문 재현 및 모델 검증을 위한 연구 기능입니다. 서비스용 Cognitive Mole과 데이터·해석 파이프라인을 분리해 유지합니다.")
    tab_research, tab_result = st.tabs(["RESEARCH cTMT", "RESEARCH REPORT"])

    with tab_research:
        st.markdown('<div class="section-tag">VALIDATION MODE // ORIGINAL cTMT PIPELINE</div>', unsafe_allow_html=True)
        st.subheader("Research cTMT")
        st.caption("선행연구 기반 103-feature extraction과 SVM 검증을 위한 연구 모드입니다. 서비스형 Cognitive Mole과 데이터 파이프라인을 분리합니다.")
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
                    st.success("Research cTMT 분석 완료. 같은 연구 검증 영역의 **RESEARCH REPORT**에서 확인하세요.")
                else:
                    st.warning("103개 Feature를 모두 생성하지 못했습니다. 유효 trial을 확인해주세요.")
            except Exception as exc:
                st.error(f"분석 처리 중 오류: {exc}")

    with tab_result:
        pack = st.session_state.get("ctmt_result")
        raw = st.session_state.get("ctmt_session")
        if not pack:
            st.info("먼저 **RESEARCH cTMT**를 완료하고 분석 데이터를 전송해주세요.")
        else:
            extracted, pred = pack["extracted"], pack["prediction"]
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

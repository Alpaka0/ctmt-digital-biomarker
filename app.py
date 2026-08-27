from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Research reproduction modules remain preserved in the repository.
from full_feature_extractor import extract_103_features  # noqa: F401
from predictor import predict_research_score  # noqa: F401

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
# Keep the original cTMT component loadable without exposing it as a service tab.
ctmt_component = load_static_component("ctmt_mouse_component_v2", "ctmt_component_parts")

st.markdown("""
<style>
:root{--green:#4fae58;--green-dark:#245b2b;--green-soft:#f2faee;--line:#d8e8d3;--text:#243127;--muted:#6d786f}
.block-container{max-width:1180px;padding-top:1.45rem;padding-bottom:3rem}
.hero{position:relative;overflow:hidden;padding:30px 32px;border:1px solid #cfe5c9;border-radius:22px;background:linear-gradient(135deg,#eff9e9,#fafff7 58%,#f4fbe9);margin:6px 0 20px;color:var(--text);box-shadow:0 10px 28px rgba(62,108,55,.08)}
.hero:after{content:"";position:absolute;right:-30px;bottom:-65px;width:220px;height:220px;border-radius:50%;background:radial-gradient(circle,#dff1c8 0,#ecf7df 45%,transparent 68%);pointer-events:none}
.kicker{font-size:12px;font-weight:900;letter-spacing:.14em;color:#3d8d47;text-transform:uppercase}.hero h1{font-size:40px;letter-spacing:-.045em;margin:7px 0 8px;color:#1f4726}.hero p{font-size:15px;line-height:1.7;color:#5d6c60;max-width:920px;margin:0}
.mission-strip{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 0}.mission-chip{border:1px solid #aad7a8;background:rgba(255,255,255,.78);border-radius:999px;padding:7px 11px;font-size:11px;font-weight:850;letter-spacing:.05em;color:#367b3e}
.operator{display:flex;gap:14px;align-items:center;border:1px solid #d4e8d0;border-radius:17px;padding:15px 17px;background:#f6fbf3;margin:10px 0 18px}.operator-avatar{width:48px;height:48px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#e4f3db;font-size:25px;flex:0 0 48px}.operator .name{font-size:11px;font-weight:900;letter-spacing:.08em;color:#3b8a43;text-transform:uppercase}.operator .msg{font-size:14px;line-height:1.55;color:#435046;margin-top:3px}
.card{border:1px solid #dce9d8;border-radius:17px;padding:17px 18px;background:white;height:100%}.card .small{font-size:11px;font-weight:900;color:#3d8d47;letter-spacing:.08em;text-transform:uppercase}.card .big{font-size:25px;font-weight:950;color:#26352a;margin:4px 0}.card .desc{font-size:13px;color:#68736a;line-height:1.5}
.notice{border-left:4px solid #59ad5f;background:#f5fbf2;border-radius:9px;padding:13px 15px;color:#526057;font-size:13px;line-height:1.55;margin-top:16px}.summary-card{border:1px solid #b9e1dc;background:#f4fbfa;border-radius:16px;padding:17px 19px;margin:14px 0 18px}.summary-card .eyebrow{font-size:11px;font-weight:900;letter-spacing:.08em;color:#0b756c;margin-bottom:6px;text-transform:uppercase}.summary-card .headline{font-size:19px;font-weight:850;color:#172033;line-height:1.45}.summary-card .sub{font-size:12px;color:#667085;line-height:1.5;margin-top:7px}.section-tag{font-size:12px;font-weight:900;letter-spacing:.1em;color:#3d8d47;text-transform:uppercase;margin-bottom:2px}
div[data-testid="stMetric"]{border:1px solid #dbe6eb;border-radius:15px;padding:14px;background:#fff}.traj-title{font-size:12px;font-weight:900;color:#385b3d;margin:4px 0 6px}.benchmark{border:1px solid #cfe4ca;background:#fbfefa;border-radius:17px;padding:16px 18px;margin:18px 0}.benchmark-title{font-size:12px;font-weight:950;letter-spacing:.09em;color:#327b3c;margin-bottom:10px}.benchmark-note{font-size:11px;line-height:1.55;color:#667268;margin-top:10px}
</style>
""", unsafe_allow_html=True)


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
        if not np.isfinite(hit_t):
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


def _motion_features(events):
    rows = []
    for p in events:
        t = _safe_float(p.get("t_ms")) / 1000.0
        x, y = _safe_float(p.get("x_norm")), _safe_float(p.get("y_norm"))
        if all(np.isfinite([t, x, y])):
            rows.append((t, x, y))
    if len(rows) < 2:
        return {
            "mean_speed": np.nan, "std_speed": np.nan, "peak_speed": np.nan,
            "mean_acceleration": np.nan, "std_acceleration": np.nan, "peak_abs_acceleration": np.nan,
            "mean_move_interval_ms": np.nan, "max_move_interval_ms": np.nan,
            "x_span": np.nan, "y_span": np.nan,
        }
    speeds, dts = [], []
    for a, b in zip(rows[:-1], rows[1:]):
        dt = b[0] - a[0]
        if dt <= 0:
            continue
        dts.append(dt)
        speeds.append(float(np.hypot(b[1]-a[1], b[2]-a[2]) / dt))
    acc = []
    for i in range(1, len(speeds)):
        dt = dts[i] if i < len(dts) else np.nan
        if np.isfinite(dt) and dt > 0:
            acc.append((speeds[i] - speeds[i-1]) / dt)
    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    return {
        "mean_speed": float(np.mean(speeds)) if speeds else np.nan,
        "std_speed": float(np.std(speeds)) if speeds else np.nan,
        "peak_speed": float(np.max(speeds)) if speeds else np.nan,
        "mean_acceleration": float(np.mean(acc)) if acc else np.nan,
        "std_acceleration": float(np.std(acc)) if acc else np.nan,
        "peak_abs_acceleration": float(np.max(np.abs(acc))) if acc else np.nan,
        "mean_move_interval_ms": float(np.mean(dts)*1000) if dts else np.nan,
        "max_move_interval_ms": float(np.max(dts)*1000) if dts else np.nan,
        "x_span": float(max(xs)-min(xs)),
        "y_span": float(max(ys)-min(ys)),
    }


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
    motion = _motion_features(pts)
    correct_hits = int(r.get("correct_hits", 0))
    wrong = int(r.get("wrong_target_clicks", 0))
    miss = int(r.get("miss_clicks", 0))
    total = max(1, correct_hits + wrong + miss)
    target_pts = [
        (_safe_float(c.get("target_x_norm")), _safe_float(c.get("target_y_norm")))
        for c in correct if np.isfinite(_safe_float(c.get("target_x_norm"))) and np.isfinite(_safe_float(c.get("target_y_norm")))
    ]
    target_jumps = [float(np.hypot(b[0]-a[0], b[1]-a[1])) for a, b in zip(target_pts[:-1], target_pts[1:])]
    click_times = [_safe_float(c.get("t_ms")) for c in correct if np.isfinite(_safe_float(c.get("t_ms")))]
    click_intervals = [b-a for a, b in zip(click_times[:-1], click_times[1:]) if b > a]
    result = {
        "part": r.get("part"),
        "correct_hits": correct_hits,
        "wrong_target_clicks": wrong,
        "miss_clicks": miss,
        "error_count": wrong + miss,
        "accuracy": correct_hits / total,
        "duration_ms": _safe_float(r.get("duration_ms"), 0.0),
        "mean_rt_ms": float(np.mean(rts)) if rts else np.nan,
        "median_rt_ms": float(np.median(rts)) if rts else np.nan,
        "std_rt_ms": float(np.std(rts)) if rts else np.nan,
        "min_rt_ms": float(np.min(rts)) if rts else np.nan,
        "max_rt_ms": float(np.max(rts)) if rts else np.nan,
        "q25_rt_ms": float(np.percentile(rts, 25)) if rts else np.nan,
        "q75_rt_ms": float(np.percentile(rts, 75)) if rts else np.nan,
        "trajectory_distance_norm": distance,
        "path_efficiency": _path_efficiency(r),
        "pointer_events": len(pts),
        "mean_correct_interval_ms": float(np.mean(click_intervals)) if click_intervals else np.nan,
        "median_correct_interval_ms": float(np.median(click_intervals)) if click_intervals else np.nan,
        "target_jump_mean_norm": float(np.mean(target_jumps)) if target_jumps else np.nan,
        "target_jump_total_norm": float(np.sum(target_jumps)) if target_jumps else np.nan,
        "unique_holes": len({c.get("hole_id") for c in correct if c.get("hole_id")}),
    }
    result.update(motion)
    return result


def mole_trajectory_svg(r):
    if not r:
        return '<div style="height:280px;border:1px solid #dce9d8;border-radius:12px;background:#f7fbf4;display:flex;align-items:center;justify-content:center;color:#7a877d;font-size:12px">trajectory data 없음</div>'
    events = r.get("pointer_events", [])
    hits = [c for c in r.get("clicks", []) if c.get("correct")]
    step = max(1, int(np.ceil(len(events) / 420))) if events else 1
    sample = events[::step]
    coords = []
    for p in sample:
        x, y = _safe_float(p.get("x_norm")), _safe_float(p.get("y_norm"))
        if np.isfinite(x) and np.isfinite(y):
            coords.append(f"{x*960:.1f},{y*540:.1f}")
    marks = []
    for i, c in enumerate(hits, start=1):
        x, y = _safe_float(c.get("x_norm")), _safe_float(c.get("y_norm"))
        if np.isfinite(x) and np.isfinite(y):
            marks.append(
                f'<circle cx="{x*960:.1f}" cy="{y*540:.1f}" r="8" fill="#fff" stroke="#3f9f4c" stroke-width="4"/>'
                f'<text x="{x*960:.1f}" y="{y*540+4:.1f}" text-anchor="middle" font-size="10" font-weight="900" fill="#245b2b">{i}</text>'
            )
    return (
        '<svg viewBox="0 0 960 540" width="100%" role="img" style="display:block;background:#f7fbf4;border:1px solid #dce9d8;border-radius:12px">'
        f'<polyline points="{" ".join(coords)}" fill="none" stroke="#79a985" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" opacity=".82"/>'
        + "".join(marks) + '</svg>'
    )


def _signed(v, digits=0, suffix=""):
    if not np.isfinite(v):
        return "-"
    return f"{v:+.{digits}f}{suffix}"


def _feature_value(v, percent=False, digits=3):
    if not np.isfinite(_safe_float(v)):
        return "-"
    value = float(v)
    if percent:
        return f"{value*100:.1f}%"
    return f"{value:.{digits}f}"


def detailed_feature_table(a, b):
    specs = [
        ("correct_hits", "정답 수", "count"),
        ("wrong_target_clicks", "오선택", "count"),
        ("miss_clicks", "MISS", "count"),
        ("accuracy", "선택 정확도", "pct"),
        ("duration_ms", "수행시간", "ms"),
        ("mean_rt_ms", "평균 반응시간", "ms"),
        ("median_rt_ms", "중앙 반응시간", "ms"),
        ("std_rt_ms", "반응시간 표준편차", "ms"),
        ("min_rt_ms", "최소 반응시간", "ms"),
        ("max_rt_ms", "최대 반응시간", "ms"),
        ("q25_rt_ms", "반응시간 Q25", "ms"),
        ("q75_rt_ms", "반응시간 Q75", "ms"),
        ("trajectory_distance_norm", "정규화 이동거리", "num"),
        ("path_efficiency", "경로 효율", "pct"),
        ("pointer_events", "Pointer events", "count"),
        ("mean_speed", "평균 이동속도", "num"),
        ("std_speed", "이동속도 표준편차", "num"),
        ("peak_speed", "최대 이동속도", "num"),
        ("mean_acceleration", "평균 가속도", "num"),
        ("std_acceleration", "가속도 표준편차", "num"),
        ("peak_abs_acceleration", "최대 절대가속도", "num"),
        ("mean_move_interval_ms", "평균 포인터 샘플 간격", "ms"),
        ("max_move_interval_ms", "최대 포인터 샘플 간격", "ms"),
        ("mean_correct_interval_ms", "평균 정답 클릭 간격", "ms"),
        ("median_correct_interval_ms", "중앙 정답 클릭 간격", "ms"),
        ("target_jump_mean_norm", "평균 목표 간 거리", "num"),
        ("target_jump_total_norm", "전체 목표 간 거리", "num"),
        ("x_span", "X축 탐색 범위", "num"),
        ("y_span", "Y축 탐색 범위", "num"),
        ("unique_holes", "사용된 구멍 수", "count"),
    ]
    rows = []
    for key, label, kind in specs:
        av, bv = _safe_float(a.get(key)), _safe_float(b.get(key))
        delta = bv-av if np.isfinite(av) and np.isfinite(bv) else np.nan
        def fmt(v):
            if not np.isfinite(v): return "-"
            if kind == "count": return f"{int(round(v))}"
            if kind == "pct": return f"{v*100:.1f}%"
            if kind == "ms": return f"{v:.1f} ms"
            return f"{v:.3f}"
        if kind == "pct": delta_txt = "-" if not np.isfinite(delta) else f"{delta*100:+.1f}%p"
        elif kind == "count": delta_txt = "-" if not np.isfinite(delta) else f"{int(round(delta)):+d}"
        elif kind == "ms": delta_txt = "-" if not np.isfinite(delta) else f"{delta:+.1f} ms"
        else: delta_txt = "-" if not np.isfinite(delta) else f"{delta:+.3f}"
        rows.append({"행동 Feature": label, "Round A": fmt(av), "Round B": fmt(bv), "B−A": delta_txt})
    return pd.DataFrame(rows)


def click_event_table(mole_pack):
    rows = []
    for r in mole_pack.get("rounds", []):
        for i, c in enumerate(r.get("clicks", []), start=1):
            rows.append({
                "Round": r.get("part"),
                "event": i,
                "type": c.get("type"),
                "expected": c.get("expected"),
                "clicked": c.get("clicked", "-"),
                "correct": bool(c.get("correct", False)),
                "reaction_ms": c.get("reaction_ms", np.nan),
                "t_ms": c.get("t_ms", np.nan),
                "hole_id": c.get("hole_id", "-"),
                "x_norm": c.get("x_norm", np.nan),
                "y_norm": c.get("y_norm", np.nan),
            })
    return pd.DataFrame(rows)


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
    <div class="operator"><div class="operator-avatar">🧠</div><div><div class="name">Service Concept</div><div class="msg">선행연구 cTMT의 순차 처리, 과제 전환, 마우스 궤적 측정 개념을 게임형 상호작용으로 확장했습니다. 원 cTMT의 103-feature/SVM 재현 코드는 저장소에 그대로 보존하며, 서비스 화면에서는 Cognitive Mole 수행 데이터와 연구 근거를 하나의 리포트에서 제시합니다.</div></div></div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        (c1, "Round A", "Sequential Processing", "1 → 2 → 3 순서에 따라 올바른 두더지를 찾아 클릭합니다."),
        (c2, "Round B", "Task Switching", "1 → A → 2 → B처럼 숫자와 문자를 번갈아 선택합니다."),
        (c3, "Behavior", "Mouse Trajectory", "이동 궤적, 반응시간, 오선택, MISS, 이동 특성을 기록합니다."),
    ]
    for col, small, big, desc in cards:
        with col:
            st.markdown(f'<div class="card"><div class="small">{small}</div><div class="big">{big}</div><div class="desc">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="notice"><b>연구 범위</b><br>Cognitive Mole은 원 cTMT를 그대로 재현한 검사가 아니라 paper-inspired 서비스 프로토타입입니다. 따라서 원 cTMT의 103-feature SVM 점수를 Cognitive Mole 세션에 직접 적용하지 않습니다. 게임 리포트에는 게임에서 실제로 계산 가능한 행동 Feature와 원 연구 모델의 검증 정보를 함께 표시합니다.</div>', unsafe_allow_html=True)

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
        raw_rmap = {r.get("part"): r for r in mole_pack.get("rounds", [])}
        a = mole_round_stats(raw_rmap.get("A", {}))
        b = mole_round_stats(raw_rmap.get("B", {}))

        correct_hits = int(summary.get("correct_hits", 0))
        wrong = int(summary.get("wrong_target_clicks", 0))
        miss = int(summary.get("miss_clicks", 0))
        completion_rate = correct_hits / 30 * 100
        selection_accuracy = float(summary.get("accuracy", 0)) * 100
        rt_a, rt_b = _safe_float(a.get("median_rt_ms")), _safe_float(b.get("median_rt_ms"))
        rt_delta = rt_b - rt_a if np.isfinite(rt_a) and np.isfinite(rt_b) else np.nan
        err_delta = int(b.get("error_count", 0)) - int(a.get("error_count", 0))
        eff_a, eff_b = _safe_float(a.get("path_efficiency")), _safe_float(b.get("path_efficiency"))
        eff_delta = eff_b-eff_a if np.isfinite(eff_a) and np.isfinite(eff_b) else np.nan

        st.markdown("### Cognitive Mole 통합 행동 리포트")
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
            pct = rt_delta/rt_a*100 if rt_a > 0 else np.nan
            text = f"Round B 중앙 반응시간은 A보다 {_signed(rt_delta,0,' ms')}"
            if np.isfinite(pct): text += f" ({_signed(pct,1,'%')})"
            summary_parts.append(text)
        summary_parts.append(f"오선택·MISS는 {err_delta:+d}회 변화")
        if np.isfinite(eff_delta): summary_parts.append(f"경로 효율은 {_signed(eff_delta*100,1,'%p')} 변화")
        st.markdown(
            f'<div class="summary-card"><div class="eyebrow">Performance Summary</div><div class="headline">{" · ".join(summary_parts)}.</div>'
            '<div class="sub">Round B−A는 이번 게임 내 조건 차이이며 임상적 판정값이 아닙니다.</div></div>', unsafe_allow_html=True)

        st.markdown("#### Round A / B 상세 행동 Feature")
        feature_df = detailed_feature_table(a, b)
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

        st.markdown("#### Mouse trajectory")
        t1, t2 = st.columns(2, gap="medium")
        with t1:
            st.markdown('<div class="traj-title">ROUND A · Sequential processing</div>', unsafe_allow_html=True)
            st.markdown(mole_trajectory_svg(raw_rmap.get("A")), unsafe_allow_html=True)
        with t2:
            st.markdown('<div class="traj-title">ROUND B · Alternating rule</div>', unsafe_allow_html=True)
            st.markdown(mole_trajectory_svg(raw_rmap.get("B")), unsafe_allow_html=True)

        st.markdown('<div class="benchmark"><div class="benchmark-title">PAPER REPRODUCTION BENCHMARK // 103-FEATURE · SVM</div>', unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("원 연구 입력 Feature", f"{MODEL_META['input_features']}")
        b2.metric("SVM 선택 Feature", f"{MODEL_META['selected_features']}")
        b3.metric("재현 Nested-CV AUC", f"{MODEL_META['official_reproduction_auc_nested_cv']:.3f}")
        b4.metric("Mole SVM 출력", "미적용")
        st.markdown('<div class="benchmark-note">원 cTMT 재현 파이프라인은 20개 분석 trial과 pointer-through-target 프로토콜을 전제로 103개 Feature를 생성합니다. Cognitive Mole은 2개 라운드의 click-based 동적 타깃 게임이므로 같은 SVM 점수를 직접 계산하면 연구적으로 동일한 입력이 아닙니다. 따라서 원 모델은 검증 근거로 보존하고, 현재 서비스 리포트는 게임에서 실제 측정된 행동 Feature를 제시합니다.</div></div>', unsafe_allow_html=True)

        with st.expander("클릭 이벤트 상세 로그", expanded=False):
            click_df = click_event_table(mole_pack)
            if len(click_df):
                st.dataframe(click_df, use_container_width=True, hide_index=True)
            else:
                st.info("클릭 이벤트가 없습니다.")

        st.markdown('<div class="notice"><b>해석 범위</b><br>본 리포트는 Cognitive Mole 게임 수행 중 나타난 행동 특성을 요약합니다. 의료 진단이나 MCI/치매 판정을 제공하지 않으며, 원 cTMT의 103-feature SVM 출력값과 동일하게 해석하지 않습니다. Cognitive Mole용 SVM을 사용하려면 별도의 게임 수행 데이터와 정답 라벨을 수집해 재학습·검증해야 합니다.</div>', unsafe_allow_html=True)

        payload = {
            "session_id": mole_pack.get("session_id"),
            "cognitive_mole_summary": summary,
            "round_A_features": a,
            "round_B_features": b,
            "research_benchmark": {
                "original_input_features": MODEL_META["input_features"],
                "original_selected_features": MODEL_META["selected_features"],
                "original_nested_cv_auc": MODEL_META["official_reproduction_auc_nested_cv"],
                "svm_applied_to_cognitive_mole": False,
            },
            "raw_session": mole_pack,
        }
        j1, j2 = st.columns(2)
        with j1:
            st.download_button("통합 리포트 JSON 다운로드", data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), file_name=f"{mole_pack.get('session_id','cognitive_mole')}_report.json", mime="application/json", use_container_width=True)
        with j2:
            st.download_button("행동 Feature CSV 다운로드", data=feature_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{mole_pack.get('session_id','cognitive_mole')}_features.csv", mime="text/csv", use_container_width=True)

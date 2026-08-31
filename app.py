from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Original cTMT reproduction code is preserved in the repository.
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


mole_component = load_static_component("cognitive_mole_component_v3", "aim_component_parts")
# Keep the original cTMT component loadable for reproducibility without exposing it in service UX.
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
        hit_t = _safe_float(hit.get("t_ms"))
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
        for c in correct
        if np.isfinite(_safe_float(c.get("target_x_norm"))) and np.isfinite(_safe_float(c.get("target_y_norm")))
    ]
    target_jumps = [float(np.hypot(b[0]-a[0], b[1]-a[1])) for a, b in zip(target_pts[:-1], target_pts[1:])]
    click_times = [_safe_float(c.get("t_ms")) for c in correct if np.isfinite(_safe_float(c.get("t_ms")))]
    click_intervals = [b-a for a, b in zip(click_times[:-1], click_times[1:]) if b > a]
    result = {
        "part": r.get("part"),
        "stage_index": int(r.get("stage_index", 0) or 0),
        "trial_index": int(r.get("trial_index", 0) or 0),
        "end_reason": r.get("end_reason"),
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


def aggregate_mole_trials(raw_trials, part):
    stats = [mole_round_stats(r) for r in raw_trials if r.get("part") == part]
    if not stats:
        return {"part": part, "trials": 0}
    numeric_keys = [
        "correct_hits", "wrong_target_clicks", "miss_clicks", "error_count", "accuracy",
        "duration_ms", "mean_rt_ms", "median_rt_ms", "std_rt_ms", "min_rt_ms", "max_rt_ms",
        "q25_rt_ms", "q75_rt_ms", "trajectory_distance_norm", "path_efficiency", "pointer_events",
        "mean_speed", "std_speed", "peak_speed", "mean_acceleration", "std_acceleration",
        "peak_abs_acceleration", "mean_move_interval_ms", "max_move_interval_ms",
        "mean_correct_interval_ms", "median_correct_interval_ms", "target_jump_mean_norm",
        "target_jump_total_norm", "x_span", "y_span", "unique_holes",
    ]
    out = {"part": part, "trials": len(stats)}
    for key in numeric_keys:
        vals = [_safe_float(s.get(key)) for s in stats]
        vals = [v for v in vals if np.isfinite(v)]
        out[key] = float(np.mean(vals)) if vals else np.nan
    out["correct_hits_total"] = int(sum(s["correct_hits"] for s in stats))
    out["wrong_target_clicks_total"] = int(sum(s["wrong_target_clicks"] for s in stats))
    out["miss_clicks_total"] = int(sum(s["miss_clicks"] for s in stats))
    out["analysis_ready_trials"] = int(sum(s["correct_hits"] >= 8 for s in stats))
    return out


def mole_trajectory_svg(r):
    if not r:
        return '<div style="height:280px;border:1px solid #dce9d8;border-radius:12px;background:#f7fbf4;display:flex;align-items:center;justify-content:center;color:#7a877d;font-size:12px">trajectory data 없음</div>'
    events = r.get("pointer_events", [])
    hits = [c for c in r.get("clicks", []) if c.get("correct")]
    step = max(1, int(np.ceil(len(events) / 420))) if events else 1
    coords = []
    for p in events[::step]:
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


def detailed_feature_table(a, b):
    specs = [
        ("correct_hits", "평균 정답 수 / trial", "count"),
        ("wrong_target_clicks", "평균 오선택 / trial", "count"),
        ("miss_clicks", "평균 MISS / trial", "count"),
        ("accuracy", "평균 선택 정확도", "pct"),
        ("duration_ms", "평균 수행시간", "ms"),
        ("mean_rt_ms", "평균 반응시간", "ms"),
        ("median_rt_ms", "중앙 반응시간의 trial 평균", "ms"),
        ("std_rt_ms", "반응시간 표준편차", "ms"),
        ("min_rt_ms", "최소 반응시간", "ms"),
        ("max_rt_ms", "최대 반응시간", "ms"),
        ("q25_rt_ms", "반응시간 Q25", "ms"),
        ("q75_rt_ms", "반응시간 Q75", "ms"),
        ("trajectory_distance_norm", "정규화 이동거리", "num"),
        ("path_efficiency", "경로 효율", "pct"),
        ("pointer_events", "Pointer events / trial", "count"),
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
            if not np.isfinite(v):
                return "-"
            if kind == "count":
                return f"{v:.1f}"
            if kind == "pct":
                return f"{v*100:.1f}%"
            if kind == "ms":
                return f"{v:.1f} ms"
            return f"{v:.3f}"

        if kind == "pct":
            delta_txt = "-" if not np.isfinite(delta) else f"{delta*100:+.1f}%p"
        elif kind == "count":
            delta_txt = "-" if not np.isfinite(delta) else f"{delta:+.1f}"
        elif kind == "ms":
            delta_txt = "-" if not np.isfinite(delta) else f"{delta:+.1f} ms"
        else:
            delta_txt = "-" if not np.isfinite(delta) else f"{delta:+.3f}"
        rows.append({
            "행동 Feature": label,
            f"Round A 평균 (n={a.get('trials', 0)})": fmt(av),
            f"Round B 평균 (n={b.get('trials', 0)})": fmt(bv),
            "B−A": delta_txt,
        })
    return pd.DataFrame(rows)


def trial_summary_table(mole_pack):
    rows = []
    for r in mole_pack.get("rounds", []):
        s = mole_round_stats(r)
        rows.append({
            "Trial": s["trial_index"],
            "Stage": s["stage_index"],
            "Round": s["part"],
            "종료": s["end_reason"],
            "정답": f'{s["correct_hits"]}/15',
            "오선택": s["wrong_target_clicks"],
            "MISS": s["miss_clicks"],
            "수행시간(s)": round(s["duration_ms"]/1000, 2),
            "중앙 RT(ms)": None if not np.isfinite(s["median_rt_ms"]) else round(s["median_rt_ms"], 1),
            "경로효율(%)": None if not np.isfinite(s["path_efficiency"]) else round(s["path_efficiency"]*100, 1),
        })
    return pd.DataFrame(rows)


def click_event_table(mole_pack):
    rows = []
    for r in mole_pack.get("rounds", []):
        for i, c in enumerate(r.get("clicks", []), start=1):
            rows.append({
                "Trial": r.get("trial_index"),
                "Stage": r.get("stage_index"),
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
  <p>원 cTMT의 반복 측정 구조에 맞춰 Round A 10회와 Round B 10회, 총 20 trials를 수행합니다. 각 trial에서 마우스 궤적, 반응시간, 정확성, 오선택 패턴을 기록하고 A/B 반복 수행의 평균 차이를 분석합니다.</p>
  <div class="mission-strip"><span class="mission-chip">20 TRIALS</span><span class="mission-chip">A × 10 / B × 10</span><span class="mission-chip">25 SEC / TRIAL</span><span class="mission-chip">MOUSE TRAJECTORY</span></div>
</div>
""", unsafe_allow_html=True)

tab_intro, tab_mole = st.tabs(["01 BRIEFING", "02 COGNITIVE MOLE"])

with tab_intro:
    st.markdown("""
    <div class="operator"><div class="operator-avatar">🧠</div><div><div class="name">Service Concept</div><div class="msg">선행연구 cTMT의 순차 처리·과제 전환·반복 trial·mouse trajectory 분석 구조를 게임형 상호작용으로 확장했습니다. A/B를 각각 10회 반복하여 단일 2-round 결과보다 안정적인 개인 행동 패턴을 집계합니다.</div></div></div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        (c1, "Round A × 10", "Sequential Processing", "각 trial에서 1 → 2 → 3 → … → 15 순서로 두더지를 선택합니다."),
        (c2, "Round B × 10", "Alternating Rule", "각 trial에서 1 → A → 2 → B → … → 8 순서로 선택합니다."),
        (c3, "20 Trials", "Repeated Behavior", "trial별 궤적·RT·오류·속도 등을 계산하고 A/B 평균과 차이를 산출합니다."),
    ]
    for col, small, big, desc in cards:
        with col:
            st.markdown(f'<div class="card"><div class="small">{small}</div><div class="big">{big}</div><div class="desc">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="notice"><b>연구 범위</b><br>반복 횟수는 원 cTMT 분석 구조와 맞춰 A 10 trials + B 10 trials로 구성했습니다. 다만 Cognitive Mole은 동적 두더지를 클릭하는 서비스형 과제이므로 원 연구의 pointer-through-target 데이터와 생성 방식이 동일하지 않습니다. 따라서 기존 103-feature SVM은 게임 데이터에 직접 적용하지 않고, 20-trial 반복 행동 Feature를 별도로 리포트합니다.</div>', unsafe_allow_html=True)

with tab_mole:
    st.markdown('<div class="section-tag">SERVICE MODE // 20-TRIAL COGNITIVE MOLE</div>', unsafe_allow_html=True)
    st.subheader("순서 기억 두더지 게임")
    st.caption("A→B를 한 세트로 총 10세트(20 trials) 수행합니다. 각 trial은 최대 25초이며, 완료 후 화면 안내에 따라 다음 trial로 이동합니다.")
    mole_data = mole_component(key="cognitive_mole_v3", default=None)
    if mole_data:
        st.session_state["mole_session"] = mole_data
    mole_pack = st.session_state.get("mole_session")

    if mole_pack and mole_pack.get("completed"):
        summary = mole_pack.get("summary", {})
        raw_trials = mole_pack.get("rounds", [])
        a_trials = [r for r in raw_trials if r.get("part") == "A"]
        b_trials = [r for r in raw_trials if r.get("part") == "B"]
        a = aggregate_mole_trials(raw_trials, "A")
        b = aggregate_mole_trials(raw_trials, "B")

        correct_hits = int(summary.get("correct_hits", 0))
        wrong = int(summary.get("wrong_target_clicks", 0))
        miss = int(summary.get("miss_clicks", 0))
        trials_done = len(raw_trials)
        max_targets = 15 * max(1, trials_done)
        completion_rate = correct_hits / max_targets * 100
        selection_accuracy = float(summary.get("accuracy", 0)) * 100
        rt_a, rt_b = _safe_float(a.get("median_rt_ms")), _safe_float(b.get("median_rt_ms"))
        rt_delta = rt_b - rt_a if np.isfinite(rt_a) and np.isfinite(rt_b) else np.nan
        err_a, err_b = _safe_float(a.get("error_count")), _safe_float(b.get("error_count"))
        err_delta = err_b - err_a if np.isfinite(err_a) and np.isfinite(err_b) else np.nan
        eff_a, eff_b = _safe_float(a.get("path_efficiency")), _safe_float(b.get("path_efficiency"))
        eff_delta = eff_b-eff_a if np.isfinite(eff_a) and np.isfinite(eff_b) else np.nan

        st.markdown("### Cognitive Mole 20-Trial 통합 행동 리포트")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("완료 Trials", f"{trials_done} / 20")
        c2.metric("과제 완료율", f"{completion_rate:.1f}%")
        c3.metric("선택 정확도", f"{selection_accuracy:.1f}%")
        c4.metric("오선택", wrong)
        c5.metric("MISS", miss)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("A 중앙 RT · 10-trial 평균", "-" if not np.isfinite(rt_a) else f"{rt_a:.0f} ms")
        d2.metric("B 중앙 RT · 10-trial 평균", "-" if not np.isfinite(rt_b) else f"{rt_b:.0f} ms")
        d3.metric("교대 규칙 차이 (B−A)", "-" if not np.isfinite(rt_delta) else f"{rt_delta:+.0f} ms")
        d4.metric("정답 목표", f"{correct_hits} / {15*20}")

        summary_parts = []
        if np.isfinite(rt_delta):
            pct = rt_delta/rt_a*100 if rt_a > 0 else np.nan
            text = f"B의 trial별 중앙 RT 평균은 A보다 {_signed(rt_delta,0,' ms')}"
            if np.isfinite(pct):
                text += f" ({_signed(pct,1,'%')})"
            summary_parts.append(text)
        if np.isfinite(err_delta):
            summary_parts.append(f"trial당 오선택·MISS는 {_signed(err_delta,1,'회')} 변화")
        if np.isfinite(eff_delta):
            summary_parts.append(f"경로 효율은 {_signed(eff_delta*100,1,'%p')} 변화")
        st.markdown(
            f'<div class="summary-card"><div class="eyebrow">Repeated-Trial Summary</div><div class="headline">{" · ".join(summary_parts)}.</div>'
            '<div class="sub">A와 B 각각 10개 trial의 행동 Feature를 trial 단위로 계산한 뒤 평균하여 비교했습니다. B−A는 게임 내 조건 차이이며 임상적 판정값이 아닙니다.</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### 20 Trials 수행 결과")
        trial_df = trial_summary_table(mole_pack)
        st.dataframe(trial_df, use_container_width=True, hide_index=True)

        st.markdown("#### Round A / B 반복 측정 Feature")
        feature_df = detailed_feature_table(a, b)
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

        st.markdown("#### 대표 Mouse trajectory · 마지막 A/B trial")
        last_a = a_trials[-1] if a_trials else None
        last_b = b_trials[-1] if b_trials else None
        t1, t2 = st.columns(2, gap="medium")
        with t1:
            st.markdown('<div class="traj-title">ROUND A · Trial 19 / Stage 10</div>', unsafe_allow_html=True)
            st.markdown(mole_trajectory_svg(last_a), unsafe_allow_html=True)
        with t2:
            st.markdown('<div class="traj-title">ROUND B · Trial 20 / Stage 10</div>', unsafe_allow_html=True)
            st.markdown(mole_trajectory_svg(last_b), unsafe_allow_html=True)

        st.markdown('<div class="benchmark"><div class="benchmark-title">PAPER REPRODUCTION BENCHMARK // 103-FEATURE · SVM</div>', unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("원 연구 입력 Feature", f"{MODEL_META['input_features']}")
        b2.metric("SVM 선택 Feature", f"{MODEL_META['selected_features']}")
        b3.metric("재현 Nested-CV AUC", f"{MODEL_META['official_reproduction_auc_nested_cv']:.3f}")
        b4.metric("Mole SVM 출력", "미적용")
        st.markdown('<div class="benchmark-note">Cognitive Mole도 이제 A 10 + B 10의 총 20-trial 반복 구조로 데이터를 수집하며, trial 단위 Feature를 계산한 뒤 A/B별로 집계합니다. 이는 원 cTMT의 반복 측정 분석 구조에 더 가깝게 맞춘 것입니다. 다만 원 연구는 고정 target을 포인터가 통과하는 프로토콜이고 Cognitive Mole은 랜덤 동적 target을 클릭하는 프로토콜이므로 동일한 103-feature 입력으로 간주할 수 없습니다. 기존 SVM은 연구 재현 근거로만 유지하며 게임 전용 모델은 별도 데이터 수집·학습·검증이 필요합니다.</div></div>', unsafe_allow_html=True)

        with st.expander("클릭 이벤트 상세 로그", expanded=False):
            click_df = click_event_table(mole_pack)
            if len(click_df):
                st.dataframe(click_df, use_container_width=True, hide_index=True)
            else:
                st.info("클릭 이벤트가 없습니다.")

        st.markdown('<div class="notice"><b>해석 범위</b><br>본 리포트는 20-trial Cognitive Mole 수행 중 나타난 반복 행동 특성을 요약합니다. 의료 진단이나 MCI/치매 판정을 제공하지 않으며 원 cTMT의 103-feature SVM 출력값과 동일하게 해석하지 않습니다.</div>', unsafe_allow_html=True)

        payload = {
            "session_id": mole_pack.get("session_id"),
            "protocol": mole_pack.get("protocol"),
            "cognitive_mole_summary": summary,
            "round_A_10trial_mean_features": a,
            "round_B_10trial_mean_features": b,
            "trial_features": [mole_round_stats(r) for r in raw_trials],
            "research_benchmark": {
                "original_input_features": MODEL_META["input_features"],
                "original_selected_features": MODEL_META["selected_features"],
                "original_nested_cv_auc": MODEL_META["official_reproduction_auc_nested_cv"],
                "svm_applied_to_cognitive_mole": False,
                "reason": "Cognitive Mole uses dynamic click-based targets rather than the original pointer-through-target cTMT protocol.",
            },
            "raw_session": mole_pack,
        }
        j1, j2 = st.columns(2)
        with j1:
            st.download_button(
                "통합 리포트 JSON 다운로드",
                data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{mole_pack.get('session_id','cognitive_mole')}_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with j2:
            st.download_button(
                "A/B 평균 Feature CSV 다운로드",
                data=feature_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{mole_pack.get('session_id','cognitive_mole')}_features.csv",
                mime="text/csv",
                use_container_width=True,
            )

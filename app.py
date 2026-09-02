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

st.set_page_config(page_title="뇌굴뇌굴", page_icon="🧠", layout="wide")
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
.radar-card{border:1px solid #dde2ef;border-radius:18px;background:#fff;padding:8px 10px 4px;margin:8px 0 8px;box-shadow:0 5px 18px rgba(62,72,108,.05)}
.radar-legend{display:flex;gap:18px;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:#55606b;margin:3px 0 0}.radar-dot{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}.radar-a{background:#6f8fae}.radar-b{background:#8358d8}
.mci-result{border-radius:20px;padding:22px 24px;margin:10px 0 12px;border:1px solid #dce3e8;box-shadow:0 7px 22px rgba(38,48,60,.06)}
.mci-result.stable{background:#f3fbf4;border-color:#b9dfbf}.mci-result.watch{background:#fff9e9;border-color:#ead38c}.mci-result.check{background:#fff2ef;border-color:#efb5aa}
.mci-result .mci-top{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}.mci-result .mci-kicker{font-size:11px;font-weight:950;letter-spacing:.1em;color:#667085}.mci-result .mci-badge{display:inline-block;border-radius:999px;padding:6px 11px;font-size:11px;font-weight:950;background:#fff;border:1px solid rgba(70,80,90,.18)}
.mci-result h3{font-size:26px;line-height:1.35;margin:9px 0 7px;color:#263238}.mci-result p{font-size:13px;line-height:1.65;color:#59636c;margin:0}.mci-action{margin-top:13px;padding-top:12px;border-top:1px solid rgba(70,80,90,.12);font-size:13px;font-weight:850;color:#344054}.mci-evidence{margin-top:9px;font-size:11px;color:#667085;line-height:1.55}.mci-prototype{font-size:11px;color:#7a8490;margin:8px 0 16px}
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
            "mean_acceleration": np.nan, "std_acceleration": np.nan, "peak_acceleration": np.nan,
            "mean_abs_acceleration": np.nan, "std_abs_acceleration": np.nan, "peak_abs_acceleration": np.nan,
            "mean_negative_acceleration": np.nan, "std_negative_acceleration": np.nan, "peak_negative_acceleration": np.nan,
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
    abs_acc = np.abs(acc) if acc else []
    neg_acc = [v for v in acc if v < 0]
    return {
        "mean_speed": float(np.mean(speeds)) if speeds else np.nan,
        "std_speed": float(np.std(speeds)) if speeds else np.nan,
        "peak_speed": float(np.max(speeds)) if speeds else np.nan,
        "mean_acceleration": float(np.mean(acc)) if acc else np.nan,
        "std_acceleration": float(np.std(acc)) if acc else np.nan,
        "peak_acceleration": float(np.max(acc)) if acc else np.nan,
        "mean_abs_acceleration": float(np.mean(abs_acc)) if len(abs_acc) else np.nan,
        "std_abs_acceleration": float(np.std(abs_acc)) if len(abs_acc) else np.nan,
        "peak_abs_acceleration": float(np.max(abs_acc)) if len(abs_acc) else np.nan,
        "mean_negative_acceleration": float(np.mean(neg_acc)) if neg_acc else np.nan,
        "std_negative_acceleration": float(np.std(neg_acc)) if neg_acc else np.nan,
        "peak_negative_acceleration": float(np.min(neg_acc)) if neg_acc else np.nan,
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
    rt_to_8_s = click_times[7] / 1000.0 if len(click_times) >= 8 else np.nan
    path_efficiency = _path_efficiency(r)
    distance_difference_proxy = (
        distance * (1.0 - path_efficiency)
        if np.isfinite(path_efficiency) else np.nan
    )
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
        "path_efficiency": path_efficiency,
        "rt_to_8_s": rt_to_8_s,
        "distance_difference_proxy": distance_difference_proxy,
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
        "peak_acceleration", "mean_abs_acceleration", "std_abs_acceleration", "peak_abs_acceleration",
        "mean_negative_acceleration", "std_negative_acceleration", "peak_negative_acceleration",
        "rt_to_8_s", "distance_difference_proxy", "mean_move_interval_ms", "max_move_interval_ms",
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


RADAR_DIMENSIONS = [
    ("completion", "완료도"),
    ("accuracy", "선택 정확도"),
    ("rt_stability", "반응 안정성"),
    ("time_efficiency", "시간 효율"),
    ("path_efficiency", "경로 효율"),
    ("wrong_control", "오선택 억제"),
    ("miss_control", "MISS 억제"),
]

CHANGE_RADAR_DIMENSIONS = [
    ("rt_increase", "반응시간 증가"),
    ("accuracy_drop", "정확도 감소"),
    ("path_drop", "경로효율 감소"),
    ("wrong_increase", "오선택 증가"),
    ("miss_increase", "MISS 증가"),
]


def _score_pct(value):
    if not np.isfinite(_safe_float(value)):
        return np.nan
    return float(np.clip(value, 0.0, 100.0))


def mole_performance_profile(stats):
    correct = _safe_float(stats.get("correct_hits"))
    accuracy = _safe_float(stats.get("accuracy"))
    mean_rt = _safe_float(stats.get("mean_rt_ms"))
    std_rt = _safe_float(stats.get("std_rt_ms"))
    duration = _safe_float(stats.get("duration_ms"))
    path_eff = _safe_float(stats.get("path_efficiency"))
    wrong = _safe_float(stats.get("wrong_target_clicks"))
    miss = _safe_float(stats.get("miss_clicks"))

    rt_stability = np.nan
    if np.isfinite(mean_rt) and mean_rt > 0 and np.isfinite(std_rt):
        rt_stability = 100.0 * (1.0 - min(std_rt / mean_rt, 1.0))

    return {
        "completion": _score_pct(correct / 15.0 * 100.0) if np.isfinite(correct) else np.nan,
        "accuracy": _score_pct(accuracy * 100.0) if np.isfinite(accuracy) else np.nan,
        "rt_stability": _score_pct(rt_stability),
        "time_efficiency": _score_pct((1.0 - min(duration / 25000.0, 1.0)) * 100.0) if np.isfinite(duration) else np.nan,
        "path_efficiency": _score_pct(path_eff * 100.0) if np.isfinite(path_eff) else np.nan,
        "wrong_control": _score_pct((1.0 - min(wrong / 15.0, 1.0)) * 100.0) if np.isfinite(wrong) else np.nan,
        "miss_control": _score_pct((1.0 - min(miss / 15.0, 1.0)) * 100.0) if np.isfinite(miss) else np.nan,
    }


def mole_change_profile(a, b):
    """Within-session A→B change profile used only for visualization.

    The radar shows change in the burden-increase direction. It is not a normal-control
    comparison and does not use a diagnostic threshold. Each dimension is normalized to
    a visualization ceiling chosen only to make the within-session pattern readable.
    These ceilings are not normative or clinical cut-offs.
    """
    rt_a = _safe_float(a.get("median_rt_ms"))
    rt_b = _safe_float(b.get("median_rt_ms"))
    rt_change_pct = (
        (rt_b - rt_a) / rt_a * 100.0
        if np.isfinite(rt_a) and rt_a > 0 and np.isfinite(rt_b) else np.nan
    )

    acc_a = _safe_float(a.get("accuracy"))
    acc_b = _safe_float(b.get("accuracy"))
    accuracy_change_pp = (
        (acc_b - acc_a) * 100.0
        if np.isfinite(acc_a) and np.isfinite(acc_b) else np.nan
    )

    path_a = _safe_float(a.get("path_efficiency"))
    path_b = _safe_float(b.get("path_efficiency"))
    path_change_pp = (
        (path_b - path_a) * 100.0
        if np.isfinite(path_a) and np.isfinite(path_b) else np.nan
    )

    wrong_a = _safe_float(a.get("wrong_target_clicks"))
    wrong_b = _safe_float(b.get("wrong_target_clicks"))
    wrong_change = wrong_b - wrong_a if np.isfinite(wrong_a) and np.isfinite(wrong_b) else np.nan

    miss_a = _safe_float(a.get("miss_clicks"))
    miss_b = _safe_float(b.get("miss_clicks"))
    miss_change = miss_b - miss_a if np.isfinite(miss_a) and np.isfinite(miss_b) else np.nan

    raw = {
        "rt_change_pct": rt_change_pct,
        "accuracy_change_pp": accuracy_change_pp,
        "path_change_pp": path_change_pp,
        "wrong_change_per_trial": wrong_change,
        "miss_change_per_trial": miss_change,
    }

    def positive(v):
        return max(0.0, v) if np.isfinite(v) else np.nan

    scores = {
        "rt_increase": _score_pct(positive(rt_change_pct) / 30.0 * 100.0),
        "accuracy_drop": _score_pct(positive(-accuracy_change_pp) / 10.0 * 100.0) if np.isfinite(accuracy_change_pp) else np.nan,
        "path_drop": _score_pct(positive(-path_change_pp) / 15.0 * 100.0) if np.isfinite(path_change_pp) else np.nan,
        "wrong_increase": _score_pct(positive(wrong_change) / 2.0 * 100.0) if np.isfinite(wrong_change) else np.nan,
        "miss_increase": _score_pct(positive(miss_change) / 1.0 * 100.0) if np.isfinite(miss_change) else np.nan,
    }
    return {"raw": raw, "scores": scores}


def mole_radar_svg(a_stats, b_stats):
    change = mole_change_profile(a_stats, b_stats)
    profile = change["scores"]
    n = len(CHANGE_RADAR_DIMENSIONS)
    cx, cy, radius = 420.0, 245.0, 172.0
    angles = [-np.pi / 2 + (2 * np.pi * i / n) for i in range(n)]

    def point(angle, value):
        scale = (value / 100.0) if np.isfinite(value) else 0.0
        r = radius * scale
        return cx + r * np.cos(angle), cy + r * np.sin(angle)

    def polygon(level=None):
        pts = []
        for (key, _), angle in zip(CHANGE_RADAR_DIMENSIONS, angles):
            value = level if level is not None else _safe_float(profile.get(key), 0.0)
            x, y = point(angle, value)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    parts = [
        '<div class="radar-card">',
        '<div class="radar-legend"><span><i class="radar-dot radar-b"></i>A → B 행동 변화량</span></div>',
        '<svg viewBox="0 0 840 500" width="100%" role="img" aria-label="Round A 순차 수행 대비 Round B 교대 수행의 행동 변화 레이더 차트" style="display:block;max-height:520px">',
    ]

    for level in (20, 40, 60, 80, 100):
        parts.append(
            f'<polygon points="{polygon(level=level)}" fill="none" stroke="#d9dee7" stroke-width="1"/>'
        )

    for angle in angles:
        x, y = point(angle, 100)
        parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#d7dce5" stroke-width="1"/>')

    parts.append(f'<polygon points="{polygon()}" fill="#8358d8" fill-opacity="0.24" stroke="#7043c7" stroke-width="3" stroke-linejoin="round"/>')

    for (key, _), angle in zip(CHANGE_RADAR_DIMENSIONS, angles):
        value = _safe_float(profile.get(key))
        if not np.isfinite(value):
            continue
        x, y = point(angle, value)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#fff" stroke="#7043c7" stroke-width="3"/>')

    for (key, label), angle in zip(CHANGE_RADAR_DIMENSIONS, angles):
        lx = cx + (radius + 52) * np.cos(angle)
        ly = cy + (radius + 52) * np.sin(angle)
        anchor = "middle"
        if lx < cx - 25:
            anchor = "end"
        elif lx > cx + 25:
            anchor = "start"
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" dominant-baseline="middle" '
            f'font-size="14" font-weight="800" fill="#44505c">{label}</text>'
        )

    parts.append('<text x="420" y="472" text-anchor="middle" font-size="11" fill="#7b8490">지표별 시각화 범위로 정규화 · 바깥쪽일수록 A 대비 B 변화가 큼 · 정상/위험 기준 아님</text>')
    parts.append('</svg></div>')
    return "".join(parts)


def radar_raw_table(a, b):
    raw = mole_change_profile(a, b)["raw"]

    def fmt(v, kind):
        if not np.isfinite(_safe_float(v)):
            return "-"
        if kind == "pct":
            return f"{v:+.1f}%"
        if kind == "pp":
            return f"{v:+.1f}%p"
        if kind == "count":
            return f"{v:+.2f}회/trial"
        return f"{v:+.2f}"

    return pd.DataFrame([
        {"지표": "반응시간", "B−A 실제 변화": fmt(raw["rt_change_pct"], "pct"), "레이더 방향": "증가 시 바깥쪽"},
        {"지표": "정확도", "B−A 실제 변화": fmt(raw["accuracy_change_pp"], "pp"), "레이더 방향": "감소 시 바깥쪽"},
        {"지표": "경로 효율", "B−A 실제 변화": fmt(raw["path_change_pp"], "pp"), "레이더 방향": "감소 시 바깥쪽"},
        {"지표": "오선택", "B−A 실제 변화": fmt(raw["wrong_change_per_trial"], "count"), "레이더 방향": "증가 시 바깥쪽"},
        {"지표": "MISS", "B−A 실제 변화": fmt(raw["miss_change_per_trial"], "count"), "레이더 방향": "증가 시 바깥쪽"},
    ])


def build_mole_research_model_proxy(a, b):
    """Project 뇌굴뇌굴 features into the reproduced cTMT SVM as an exploratory proxy.

    Only features with a reasonably direct behavioral analogue are populated from Mole.
    Original cTMT-only features remain NaN and are handled by the deployment pipeline's
    training-mean SimpleImputer. The resulting score is not a validated clinical probability.
    """
    features = {c: np.nan for c in FEATURE_COLUMNS}
    mapping = {
        "distance_difference_from_ideal": "distance_difference_proxy",
        "mean_abs_acceleration": "mean_abs_acceleration",
        "mean_acceleration": "mean_acceleration",
        "mean_negative_acceleration": "mean_negative_acceleration",
        "mean_speed": "mean_speed",
        "non_cut_correct_targets_touches": "correct_hits",
        "non_cut_rt": "duration_ms",
        "peak_abs_acceleration": "peak_abs_acceleration",
        "peak_acceleration": "peak_acceleration",
        "peak_negative_acceleration": "peak_negative_acceleration",
        "peak_speed": "peak_speed",
        "rt": "rt_to_8_s",
        "std_abs_acceleration": "std_abs_acceleration",
        "std_acceleration": "std_acceleration",
        "std_negative_acceleration": "std_negative_acceleration",
        "std_speed": "std_speed",
        "total_distance": "trajectory_distance_norm",
    }

    for part, stats in (("A", a), ("B", b)):
        for research_name, mole_name in mapping.items():
            col = f"{research_name}_PART_{part}"
            if col not in features:
                continue
            value = _safe_float(stats.get(mole_name))
            if np.isfinite(value):
                features[col] = float(value)

    features["is_valid_sum_A"] = 100.0 * _safe_float(a.get("analysis_ready_trials"), 0.0) / 10.0
    features["is_valid_sum_B"] = 100.0 * _safe_float(b.get("analysis_ready_trials"), 0.0) / 10.0

    suffix = "_B_A_ratio"
    for col in FEATURE_COLUMNS:
        if not col.endswith(suffix):
            continue
        base = col[:-len(suffix)]
        av = _safe_float(features.get(f"{base}_PART_A"))
        bv = _safe_float(features.get(f"{base}_PART_B"))
        if np.isfinite(av) and np.isfinite(bv) and av != 0:
            features[col] = float(bv / av)

    mapped = int(sum(np.isfinite(_safe_float(features.get(c))) for c in FEATURE_COLUMNS))
    try:
        prediction = predict_research_score(features)
        return {
            "available": True,
            "score": float(prediction["research_probability_mci_pattern"]),
            "model_class_at_default_threshold": int(prediction["model_class_at_default_threshold"]),
            "mapped_features": mapped,
            "total_features": len(FEATURE_COLUMNS),
            "imputed_features": len(FEATURE_COLUMNS) - mapped,
            "linkage_mode": "exploratory_research_model_proxy",
            "clinical_probability": False,
            "interpretation": "원 cTMT 재현 SVM에 뇌굴뇌굴 대응 Feature를 투영한 탐색적 참고점수",
        }
    except Exception as exc:
        return {
            "available": False,
            "score": np.nan,
            "mapped_features": mapped,
            "total_features": len(FEATURE_COLUMNS),
            "imputed_features": len(FEATURE_COLUMNS) - mapped,
            "linkage_mode": "exploratory_research_model_proxy",
            "clinical_probability": False,
            "error": str(exc),
        }


def mci_signal_screening(a, b):
    a_profile = mole_performance_profile(a)
    b_profile = mole_performance_profile(b)
    gaps = []
    for key, _ in RADAR_DIMENSIONS:
        av = _safe_float(a_profile.get(key))
        bv = _safe_float(b_profile.get(key))
        if np.isfinite(av) and np.isfinite(bv):
            gaps.append(bv - av)

    mean_gap = float(np.mean(gaps)) if gaps else np.nan
    declined_dimensions = int(sum(g <= -10.0 for g in gaps))

    rt_a = _safe_float(a.get("median_rt_ms"))
    rt_b = _safe_float(b.get("median_rt_ms"))
    rt_pct = ((rt_b - rt_a) / rt_a * 100.0) if np.isfinite(rt_a) and rt_a > 0 and np.isfinite(rt_b) else np.nan

    acc_a = _safe_float(a.get("accuracy"))
    acc_b = _safe_float(b.get("accuracy"))
    accuracy_delta_pp = (acc_b - acc_a) * 100.0 if np.isfinite(acc_a) and np.isfinite(acc_b) else np.nan

    err_a = _safe_float(a.get("error_count"))
    err_b = _safe_float(b.get("error_count"))
    error_delta = err_b - err_a if np.isfinite(err_a) and np.isfinite(err_b) else np.nan

    path_a = _safe_float(a.get("path_efficiency"))
    path_b = _safe_float(b.get("path_efficiency"))
    path_delta_pp = (path_b - path_a) * 100.0 if np.isfinite(path_a) and np.isfinite(path_b) else np.nan

    points = 0
    if np.isfinite(rt_pct):
        points += 2 if rt_pct >= 25 else (1 if rt_pct >= 10 else 0)
    if np.isfinite(error_delta):
        points += 2 if error_delta >= 1.5 else (1 if error_delta >= 0.5 else 0)
    if np.isfinite(accuracy_delta_pp):
        points += 2 if accuracy_delta_pp <= -8 else (1 if accuracy_delta_pp <= -3 else 0)
    if np.isfinite(path_delta_pp):
        points += 2 if path_delta_pp <= -10 else (1 if path_delta_pp <= -5 else 0)
    points += 2 if declined_dimensions >= 4 else (1 if declined_dimensions >= 2 else 0)

    if points >= 6:
        level = "확인 필요"
        css_class = "check"
        headline = "MCI와 관련될 수 있는 행동 변화 신호가 비교적 크게 관찰되었습니다."
        action = "신경과·기억장애 클리닉 등 전문 의료기관에서 표준 인지검사와 전문 평가 상담을 권장합니다."
    elif points >= 3:
        level = "관찰"
        css_class = "watch"
        headline = "MCI와 관련될 수 있는 일부 행동 변화 신호가 관찰되었습니다."
        action = "동일한 조건으로 추적 재검을 권장하며, 변화가 반복되거나 일상 기능 저하가 느껴지면 전문 인지평가를 고려하세요."
    else:
        level = "안정"
        css_class = "stable"
        headline = "현재 수행에서 뚜렷한 MCI 관련 행동 위험 신호는 낮게 관찰되었습니다."
        action = "현재 결과를 기준선으로 보관하고 정기적으로 추적해 변화 여부를 확인하는 것을 권장합니다."

    return {
        "level": level,
        "css_class": css_class,
        "headline": headline,
        "action": action,
        "prototype_points": int(points),
        "declined_dimensions": declined_dimensions,
        "mean_profile_gap": mean_gap,
        "rt_change_pct": rt_pct,
        "accuracy_change_pp": accuracy_delta_pp,
        "error_change_per_trial": error_delta,
        "path_efficiency_change_pp": path_delta_pp,
        "basis": "within-session Round A vs Round B rule-based prototype",
        "clinical_probability": False,
    }


def mci_result_html(result):
    evidence = []
    if np.isfinite(_safe_float(result.get("rt_change_pct"))):
        evidence.append(f"반응시간 {result['rt_change_pct']:+.1f}%")
    if np.isfinite(_safe_float(result.get("accuracy_change_pp"))):
        evidence.append(f"정확도 {result['accuracy_change_pp']:+.1f}%p")
    if np.isfinite(_safe_float(result.get("error_change_per_trial"))):
        evidence.append(f"오류 {result['error_change_per_trial']:+.1f}회/trial")
    if np.isfinite(_safe_float(result.get("path_efficiency_change_pp"))):
        evidence.append(f"경로효율 {result['path_efficiency_change_pp']:+.1f}%p")
    evidence.append(f"저하 지표 {result.get('declined_dimensions', 0)}개")
    evidence_text = " · ".join(evidence)

    return (
        f'<div class="mci-result {result["css_class"]}">'
        '<div class="mci-top"><div class="mci-kicker">MCI RISK-SIGNAL SCREENING // PROTOTYPE</div>'
        f'<span class="mci-badge">{result["level"]}</span></div>'
        f'<h3>{result["headline"]}</h3>'
        '<p>순차 수행 대비 교대 수행에서 나타난 반응시간, 정확도, 오류, 경로 효율과 다차원 행동 변화를 종합한 서비스용 선별 결과입니다.</p>'
        f'<div class="mci-action">다음 단계 · {result["action"]}</div>'
        f'<div class="mci-evidence">관찰 근거 · {evidence_text}</div>'
        '</div>'
    )


def mci_level_table():
    return pd.DataFrame([
        {
            "선별 결과": "안정",
            "한 줄 결과": "MCI 관련 행동 위험 신호가 낮게 관찰되었습니다.",
            "권장 행동": "현재 결과를 기준선으로 저장하고 정기 추적",
        },
        {
            "선별 결과": "관찰",
            "한 줄 결과": "MCI와 관련될 수 있는 일부 행동 변화 신호가 관찰되었습니다.",
            "권장 행동": "추적 재검 후 변화가 반복되면 전문 인지평가 고려",
        },
        {
            "선별 결과": "확인 필요",
            "한 줄 결과": "MCI와 관련될 수 있는 행동 변화 신호가 비교적 크게 관찰되었습니다.",
            "권장 행동": "신경과·기억장애 클리닉 등 전문 의료기관 인지평가 권장",
        },
    ])


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
  <div class="kicker">뇌굴뇌굴 // PAPER-INSPIRED MOUSE TASK</div>
  <h1>뇌굴뇌굴</h1>
  <p>원 cTMT의 반복 측정 구조에 맞춰 Round A 10회와 Round B 10회, 총 20 trials를 수행합니다. 각 trial에서 마우스 궤적, 반응시간, 정확성, 오선택 패턴을 기록하고 A/B 반복 수행의 평균 차이를 분석합니다.</p>
  <div class="mission-strip"><span class="mission-chip">20 TRIALS</span><span class="mission-chip">A × 10 / B × 10</span><span class="mission-chip">25 SEC / TRIAL</span><span class="mission-chip">MOUSE TRAJECTORY</span></div>
</div>
""", unsafe_allow_html=True)

tab_intro, tab_mole = st.tabs(["01 BRIEFING", "02 뇌굴뇌굴"])

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
    st.markdown('<div class="notice"><b>연구 범위</b><br>반복 횟수는 원 cTMT 분석 구조와 맞춰 A 10 trials + B 10 trials로 구성했습니다. 뇌굴뇌굴은 동적 두더지를 클릭하는 서비스형 과제라 원 연구와 상호작용 방식은 다르지만, 대응 가능한 mouse-behavior Feature를 원 103-feature 구조에 연결해 재현 SVM의 탐색적 참고점수까지 함께 제시합니다.</div>', unsafe_allow_html=True)

with tab_mole:
    st.markdown('<div class="section-tag">SERVICE MODE // 20-TRIAL 뇌굴뇌굴</div>', unsafe_allow_html=True)
    st.subheader("뇌굴뇌굴")
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
        research_model_proxy = build_mole_research_model_proxy(a, b)

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

        st.markdown("### 뇌굴뇌굴 20-Trial 통합 행동 리포트")
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

        st.markdown("#### 인지 전환 행동 변화 프로필 · B−A")
        st.caption("Round A(순차 수행) 대비 Round B(교대 수행)에서 행동이 얼마나 변했는지를 한 개의 변화 프로필로 표시합니다.")
        st.markdown(mole_radar_svg(a, b), unsafe_allow_html=True)

        change_profile = mole_change_profile(a, b)
        change_raw = change_profile["raw"]
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("반응시간 변화", "-" if not np.isfinite(change_raw["rt_change_pct"]) else f'{change_raw["rt_change_pct"]:+.1f}%')
        r2.metric("정확도 변화", "-" if not np.isfinite(change_raw["accuracy_change_pp"]) else f'{change_raw["accuracy_change_pp"]:+.1f}%p')
        r3.metric("경로효율 변화", "-" if not np.isfinite(change_raw["path_change_pp"]) else f'{change_raw["path_change_pp"]:+.1f}%p')
        r4.metric("오선택 변화", "-" if not np.isfinite(change_raw["wrong_change_per_trial"]) else f'{change_raw["wrong_change_per_trial"]:+.2f}회')
        r5.metric("MISS 변화", "-" if not np.isfinite(change_raw["miss_change_per_trial"]) else f'{change_raw["miss_change_per_trial"]:+.2f}회')
        st.caption("※ 오선택·MISS 변화량은 trial당 평균 발생 횟수 차이입니다.")

        st.caption(
            "두 조건 모두 현재 사용자의 실제 수행입니다. 레이더는 지표별 시각화 상한(반응시간 30%, 정확도 10%p, 경로효율 15%p, 오선택 2회/trial, MISS 1회/trial)에 맞춰 상대 크기만 정규화합니다. "
            "이 상한은 그래프 가독성을 위한 시각화 스케일이며 정상군 평균·정상/위험 판정·임상적 기준값이 아닙니다."
        )
        st.dataframe(radar_raw_table(a, b), use_container_width=True, hide_index=True)

        screening_result = mci_signal_screening(a, b)
        st.markdown("### MCI 관련 행동 신호 참고 결과 · 프로토타입")
        st.markdown(mci_result_html(screening_result), unsafe_allow_html=True)
        if research_model_proxy.get("available"):
            proxy_pct = research_model_proxy["score"] * 100.0
            st.markdown(
                f'<div class="mci-prototype"><b>논문 재현 SVM 기반 MCI 패턴 참고점수 · {proxy_pct:.1f}%</b><br>'
                f'뇌굴뇌굴에서 대응 가능한 {research_model_proxy["mapped_features"]}/{research_model_proxy["total_features"]}개 Feature를 연결하고, '
                '대응되지 않는 원 cTMT 전용 Feature는 재현 모델의 학습 평균으로 보완한 탐색적 proxy입니다. 임상적 MCI 확률이 아닙니다.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="mci-prototype">※ 논문 재현 SVM 탐색적 연계 점수는 이번 세션에서 산출되지 않았습니다.</div>', unsafe_allow_html=True)
        st.markdown('<div class="mci-prototype">※ 서비스 선별 결과는 뇌굴뇌굴 내부 A/B 수행 차이에 기반한 프로토타입 규칙이며, 정상군 비교·MCI 진단 또는 의료적 확진 결과가 아닙니다.</div>', unsafe_allow_html=True)
        st.markdown("#### 결과 단계 및 후속 안내")
        st.dataframe(mci_level_table(), use_container_width=True, hide_index=True)

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
        proxy_value = "산출 불가"
        if research_model_proxy.get("available"):
            proxy_value = f"{research_model_proxy['score']*100:.1f}%"
        b4.metric("뇌굴뇌굴 SVM 참고점수", proxy_value)
        st.markdown(
            f'<div class="benchmark-note">원 연구의 20-trial·103-feature·SVM 파이프라인을 재현한 뒤, 뇌굴뇌굴에서도 A 10 + B 10 구조와 mouse trajectory를 유지했습니다. '
            f'게임에서 원 연구와 직접 대응 가능한 {research_model_proxy.get("mapped_features", 0)}/{research_model_proxy.get("total_features", len(FEATURE_COLUMNS))}개 Feature를 103-feature 공간에 연결하고, '
            '나머지 원 cTMT 전용 Feature는 재현 SVM pipeline의 학습 평균 imputation으로 보완해 탐색적 Research-model proxy를 산출합니다. '
            '따라서 표시된 점수는 서비스 설계를 위한 연구모델 참고값이며, 원 논문의 AUC 0.670을 뇌굴뇌굴의 검증 성능이나 임상적 MCI 확률로 해석하지 않습니다.</div></div>',
            unsafe_allow_html=True,
        )

        with st.expander("클릭 이벤트 상세 로그", expanded=False):
            click_df = click_event_table(mole_pack)
            if len(click_df):
                st.dataframe(click_df, use_container_width=True, hide_index=True)
            else:
                st.info("클릭 이벤트가 없습니다.")

        st.markdown('<div class="notice"><b>해석 범위</b><br>본 리포트의 레이더는 동일 사용자의 Round A→B 행동 변화만 시각화하며 정상군 규준 비교가 아닙니다. 원 논문 재현 SVM의 탐색적 proxy는 별도로 제시합니다. 서비스 선별 결과와 SVM 참고점수 모두 의료 진단이나 MCI/치매 확진·임상 확률이 아니며, 뇌굴뇌굴 자체 임상 검증을 위해서는 동일 게임 프로토콜로 정상군·임상군 데이터를 별도 수집해 재학습·검증해야 합니다.</div>', unsafe_allow_html=True)

        payload = {
            "session_id": mole_pack.get("session_id"),
            "protocol": mole_pack.get("protocol"),
            "cognitive_mole_summary": summary,
            "round_A_10trial_mean_features": a,
            "round_B_10trial_mean_features": b,
            "radar_profile": {
                "type": "within-session B-A behavioral change profile",
                "comparison": "Round A sequential vs Round B alternating",
                "normal_reference": False,
                "change_profile": change_profile,
            },
            "mci_risk_signal_screening": screening_result,
            "mole_research_model_proxy": research_model_proxy,
            "trial_features": [mole_round_stats(r) for r in raw_trials],
            "research_benchmark": {
                "original_input_features": MODEL_META["input_features"],
                "original_selected_features": MODEL_META["selected_features"],
                "original_nested_cv_auc": MODEL_META["official_reproduction_auc_nested_cv"],
                "svm_linkage_mode": "exploratory_research_model_proxy",
                "clinical_probability": False,
                "reason": "뇌굴뇌굴 maps compatible mouse-behavior features into the reproduced cTMT SVM; cTMT-only features are mean-imputed, so the output is exploratory rather than a validated Mole prediction.",
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
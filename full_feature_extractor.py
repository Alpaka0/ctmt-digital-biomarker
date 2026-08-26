from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import json
import numpy as np

TARGET_RADIUS = 0.0275
CORRECT_THRESHOLD = 8
CONSECUTIVE_POINTS = 5

BASE_VARS = [
    "rt", "total_distance",
    "non_cut_correct_targets_touches", "non_cut_zigzag_amplitude", "non_cut_rt",
    "mean_speed", "std_speed", "peak_speed",
    "mean_acceleration", "std_acceleration", "peak_acceleration",
    "mean_abs_acceleration", "std_abs_acceleration", "peak_abs_acceleration",
    "mean_negative_acceleration", "std_negative_acceleration", "peak_negative_acceleration",
    "hesitation_time", "travel_time", "search_time",
    "hesitation_distance", "travel_distance", "search_distance",
    "hesitation_avg_speed", "travel_avg_speed", "search_avg_speed",
    "state_transitions", "total_hesitations", "average_duration", "max_duration",
    "zigzag_amplitude", "distance_difference_from_ideal",
    "area_difference_from_ideal", "intra_target_time", "inter_target_time",
]

@dataclass(frozen=True)
class Point:
    x: float
    y: float
    t: float

@dataclass
class Trial:
    part: str
    points: List[Point]
    targets: List[Tuple[str, float, float]]
    rt_ms: float
    trial_index: int = 0
    is_practice: bool = False
    start_idx: int | None = None


def _distance_xy(ax, ay, bx, by):
    return hypot(ax - bx, ay - by)


def _distance(a: Point, b: Point):
    return hypot(a.x - b.x, a.y - b.y)


def _from_start(trial: Trial) -> List[Point]:
    if trial.start_idx is None:
        return []
    return trial.points[trial.start_idx:]


def _find_first_target_hit(trial: Trial, radius: float = TARGET_RADIUS):
    if not trial.targets:
        return None
    _, tx, ty = trial.targets[0]
    for i, p in enumerate(trial.points):
        if _distance_xy(p.x, p.y, tx, ty) <= radius:
            return i
    return None


def trial_from_streamlit(raw: Dict[str, Any]) -> Trial:
    events = [e for e in raw.get("events", []) if e.get("event_type") == "pointermove"]
    events = sorted(events, key=lambda e: float(e.get("t_ms", 0)))
    points = []
    last_t = -1.0
    for e in events:
        t = float(e["t_ms"]) / 1000.0
        if t <= last_t:
            continue
        points.append(Point(float(e["x_norm"]), float(e["y_norm"]), t))
        last_t = t

    targets_raw = sorted(raw.get("targets", []), key=lambda x: int(x["target_order"]))
    targets = [(str(t["target_label"]), float(t["target_x_norm"]), float(t["target_y_norm"])) for t in targets_raw]

    trial = Trial(
        part=str(raw["part"]),
        points=points,
        targets=targets,
        rt_ms=float(int(float(raw.get("trial_end_ms") or 0.0))),
        trial_index=int(raw.get("trial_index") or 0),
        is_practice=bool(raw.get("is_practice", False)),
    )
    trial.start_idx = _find_first_target_hit(trial)
    return trial


def _touched_indices(point: Point, trial: Trial, radius: float = TARGET_RADIUS):
    result = []
    for i, (_, x, y) in enumerate(trial.targets):
        if _distance_xy(point.x, point.y, x, y) < radius:
            result.append(i)
    return result


def _segments_to_expected_targets(trial: Trial):
    pts = _from_start(trial)
    segments = []
    cursor_i = 0
    for expected_i in range(1, len(trial.targets)):
        current = []
        found = False
        while cursor_i < len(pts):
            p = pts[cursor_i]
            cursor_i += 1
            current.append(p)
            if expected_i in _touched_indices(p, trial):
                segments.append((expected_i, current.copy()))
                found = True
                break
        if not found:
            break
    return segments


def _correct_count(trial: Trial):
    if trial.start_idx is None:
        return 0
    return len(_segments_to_expected_targets(trial)) + 1


def _intervals(trial: Trial):
    return [(idx, seg[0], seg[-1]) for idx, seg in _segments_to_expected_targets(trial) if seg]


def _cut_trial_repository_compatible(trial: Trial, minimum: int = CORRECT_THRESHOLD):
    if _correct_count(trial) < minimum:
        return None
    intervals = _intervals(trial)
    if len(intervals) < minimum:
        return None
    target_to_find = minimum - 1
    cutoff = None
    for idx, _, end in intervals:
        if idx == target_to_find:
            cutoff = end.t
            break
    if cutoff is None:
        return None
    pts = [p for p in trial.points if p.t <= cutoff]
    cut = Trial(
        part=trial.part,
        points=pts,
        targets=trial.targets[:minimum],
        rt_ms=float(cutoff),
        trial_index=trial.trial_index,
        is_practice=trial.is_practice,
    )
    cut.start_idx = _find_first_target_hit(cut)
    return cut


def _speeds(points: Sequence[Point]):
    if len(points) < 2:
        raise ValueError("At least two cursor points are required.")
    out = []
    for prev, cur in zip(points[:-1], points[1:]):
        dt = cur.t - prev.t
        if dt <= 0:
            raise ValueError("Cursor timestamps must strictly increase.")
        speed = _distance(prev, cur) / dt
        if speed > 50:
            raise ValueError("Repository-compatible speed guard exceeded (>50).")
        out.append(speed)
    return out


def _safe_stats(values, peak_min=False):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    peak = np.min(arr) if peak_min else np.max(arr)
    return float(np.mean(arr)), float(np.std(arr)), float(peak)


def _speed_acc_metrics(trial: Trial):
    pts = _from_start(trial)
    speeds = _speeds(pts)
    accelerations = []
    for i in range(1, len(speeds)):
        dt = pts[i + 1].t - pts[i].t
        if dt <= 0:
            raise ValueError("Invalid acceleration interval.")
        accelerations.append((speeds[i] - speeds[i - 1]) / dt)
    abs_acc = np.abs(accelerations)
    neg_acc = [a for a in accelerations if a < 0]
    ms, ss, ps = _safe_stats(speeds)
    ma, sa, pa = _safe_stats(accelerations)
    maa, saa, paa = _safe_stats(abs_acc)
    mna, sna, pna = _safe_stats(neg_acc, peak_min=True)
    return {
        "mean_speed": ms, "std_speed": ss, "peak_speed": ps,
        "mean_acceleration": ma, "std_acceleration": sa, "peak_acceleration": pa,
        "mean_abs_acceleration": maa, "std_abs_acceleration": saa, "peak_abs_acceleration": paa,
        "mean_negative_acceleration": mna, "std_negative_acceleration": sna, "peak_negative_acceleration": pna,
    }


def _total_distance(trial: Trial):
    pts = _from_start(trial)
    return float(sum(_distance(a, b) for a, b in zip(pts[:-1], pts[1:])))


def _speed_threshold(trials: Sequence[Trial]):
    values = []
    for trial in trials:
        if trial.part != "A" or trial.start_idx is None:
            continue
        pts = _from_start(trial)
        stimuli = trial.targets[:2]
        segments = []
        segment_start = 0
        cursor_i = 0
        for _, tx, ty in stimuli:
            found = False
            while cursor_i < len(pts):
                if _distance_xy(pts[cursor_i].x, pts[cursor_i].y, tx, ty) <= TARGET_RADIUS:
                    found = True
                    break
                cursor_i += 1
            if found:
                segments.append(pts[segment_start:cursor_i + 1])
                segment_start = cursor_i
                cursor_i += 1
            else:
                segments.append(pts[segment_start:])
                break
        if len(segments) < 2 or len(segments[1]) < 2:
            continue
        try:
            values.append(float(np.mean(_speeds(segments[1]))))
        except ValueError:
            continue
    if not values:
        raise ValueError("No valid Part-A trial available for speed threshold.")
    return float(np.percentile(values, 50))


def _over_target_flags(trial: Trial):
    pts = _from_start(trial)
    flags = []
    current_target_i = 0
    on_current = False
    for p in pts:
        if current_target_i == len(trial.targets):
            _, x, y = trial.targets[current_target_i - 1]
            flags.append((True, (x, y)))
            continue
        _, tx, ty = trial.targets[current_target_i]
        d = _distance_xy(p.x, p.y, tx, ty)
        if on_current:
            if d < TARGET_RADIUS:
                over = True
            else:
                over = False
                on_current = False
                current_target_i += 1
        else:
            if d < TARGET_RADIUS:
                over = True
                on_current = True
            else:
                over = False
        flags.append((over, (tx, ty)))
    return flags


def _above_threshold_n(speeds, cursor_i, threshold):
    speed_i = cursor_i - 1
    if speed_i < CONSECUTIVE_POINTS:
        return False
    for i in range(speed_i - CONSECUTIVE_POINTS + 1, speed_i + 1):
        if i <= 0 or speeds[i] <= threshold:
            return False
    return True


def _below_threshold_n(speeds, cursor_i, threshold):
    speed_i = cursor_i - 1
    if speed_i < CONSECUTIVE_POINTS:
        return False
    for i in range(speed_i - CONSECUTIVE_POINTS + 1, speed_i + 1):
        if i <= 0 or speeds[i] > threshold:
            return False
    return True


def _classify_states(trial: Trial, threshold: float):
    pts = _from_start(trial)
    speeds = _speeds(pts)
    flags = _over_target_flags(trial)
    state = "Search"
    last_target = flags[0][1]
    classified = []
    for i, p in enumerate(pts):
        over = flags[i][0]
        if over:
            last_target = flags[i][1]
            state = "Search"
        elif _above_threshold_n(speeds, i, threshold):
            state = "Travel"
        elif state == "Travel" and _below_threshold_n(speeds, i, threshold):
            state = "Hesitation"
        elif state == "Search" and not over:
            if _distance_xy(p.x, p.y, last_target[0], last_target[1]) > 2 * TARGET_RADIUS:
                state = "Travel"
        classified.append((state, p))
    return classified


def _segmentation_metrics(trial: Trial, threshold: float):
    classified = _classify_states(trial, threshold)
    if not classified:
        raise ValueError("No classified cursor positions.")
    state_times = {"Search": 0.0, "Travel": 0.0, "Hesitation": 0.0}
    state_dist = {"Search": 0.0, "Travel": 0.0, "Hesitation": 0.0}
    state_speeds = {"Search": [], "Travel": [], "Hesitation": []}
    prev_state, prev_point = classified[0]
    prev_time = prev_point.t
    for state, point in classified[1:]:
        dt = point.t - prev_time
        state_times[prev_state] += dt
        dist = _distance(prev_point, point)
        state_dist[prev_state] += dist
        if dt > 0:
            state_speeds[prev_state].append(dist / dt)
        prev_state = state
        prev_point = point
        prev_time = point.t
    avg_speed = {k: (sum(v) / len(v) if v else 0.0) for k, v in state_speeds.items()}
    transitions = 0
    previous_state = classified[0][0]
    for state, _ in classified[1:]:
        if state != previous_state:
            transitions += 1
        previous_state = state
    periods = []
    in_hesitation = False
    start_time = 0.0
    for state, point in classified:
        if state == "Hesitation":
            if not in_hesitation:
                in_hesitation = True
                start_time = point.t
        elif in_hesitation:
            in_hesitation = False
            periods.append(point.t - start_time)
    if in_hesitation:
        periods.append(classified[-1][1].t - start_time)
    return {
        "hesitation_time": state_times["Hesitation"],
        "travel_time": state_times["Travel"],
        "search_time": state_times["Search"],
        "hesitation_distance": state_dist["Hesitation"],
        "travel_distance": state_dist["Travel"],
        "search_distance": state_dist["Search"],
        "hesitation_avg_speed": avg_speed["Hesitation"],
        "travel_avg_speed": avg_speed["Travel"],
        "search_avg_speed": avg_speed["Search"],
        "state_transitions": transitions,
        "total_hesitations": len(periods),
        "average_duration": sum(periods) / len(periods) if periods else 0.0,
        "max_duration": max(periods) if periods else 0.0,
    }


def _distance_difference_from_ideal(segment):
    actual = sum(_distance(a, b) for a, b in zip(segment[:-1], segment[1:]))
    ideal = _distance(segment[0], segment[-1])
    return abs(actual - ideal)


def _area_difference(segment):
    coords = np.array([[p.x, p.y] for p in segment], dtype=float)
    if len(coords) < 2:
        return 0.0
    start = coords[0]
    end = coords[-1]
    vector = end - start
    length_sq = float(np.dot(vector, vector))
    if length_sq == 0:
        return 0.0
    length = float(np.sqrt(length_sq))
    offsets = coords - start
    projection_factors = (offsets @ vector) / length_sq
    projection_points = start + projection_factors[:, None] * vector
    perpendicular = np.linalg.norm(coords - projection_points, axis=1)
    line_positions = projection_factors * length
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(perpendicular, line_positions))
    return float(np.trapz(perpendicular, line_positions))


def _ideal_metrics(trial: Trial):
    segments = _segments_to_expected_targets(trial)
    differences = []
    areas = []
    for _, segment in segments:
        if not segment:
            continue
        differences.append(_distance_difference_from_ideal(segment))
        areas.append(_area_difference(segment))
    return {
        "distance_difference_from_ideal": float(np.mean(differences)),
        "area_difference_from_ideal": float(np.mean(areas)),
    }


def _target_times(trial: Trial):
    intervals = _intervals(trial)
    if not intervals:
        raise ValueError("No correct target intervals.")
    interval_times = [end.t - start.t for _, start, end in intervals]
    intra = float(np.mean(interval_times))
    total_dwell = float(np.sum(interval_times))
    pts = _from_start(trial)
    finish_time = pts[-1].t
    start_time = pts[0].t
    inter = float((finish_time - start_time) - total_dwell)
    return {"intra_target_time": intra, "inter_target_time": inter}


def _zigzag_amplitude(trial: Trial):
    if trial.part != "B":
        return np.nan
    intervals = _intervals(trial)
    differences = []
    for i in range(0, len(intervals) - 1, 2):
        first_idx, first_start, _ = intervals[i]
        second_idx, second_start, _ = intervals[i + 1]
        first_label = trial.targets[first_idx][0]
        second_label = trial.targets[second_idx][0]
        if not first_label.isalpha() or not second_label.isdigit():
            raise ValueError("Unexpected Part-B interval order.")
        differences.append(first_start.t - second_start.t)
    return float(np.mean(differences)) if differences else np.nan


def _metrics_for_trial(trial: Trial, threshold: float):
    result = {
        "zigzag_amplitude": _zigzag_amplitude(trial),
        "total_distance": _total_distance(trial),
        "rt": trial.rt_ms,
        "correct_targets_touches": _correct_count(trial),
    }
    result.update(_speed_acc_metrics(trial))
    result.update(_segmentation_metrics(trial, threshold))
    result.update(_ideal_metrics(trial))
    result.update(_target_times(trial))
    return result


def extract_103_features(session: Dict[str, Any], feature_columns: Sequence[str] | None = None):
    trials = [trial_from_streamlit(t) for t in session.get("trials", []) if not bool(t.get("is_practice", False))]
    if len(trials) != 20:
        raise ValueError(f"Research-mode input requires 20 analysis trials; received {len(trials)}.")
    threshold = _speed_threshold(trials)
    rows = []
    for trial in trials:
        valid_from_mapper = trial.start_idx is not None and len(_from_start(trial)) > 2
        if not valid_from_mapper:
            rows.append({"part": trial.part, "valid": False})
            continue
        cut = _cut_trial_repository_compatible(trial)
        if cut is None:
            rows.append({"part": trial.part, "valid": False})
            continue
        try:
            cut_metrics = _metrics_for_trial(cut, threshold)
            non_cut_metrics = _metrics_for_trial(trial, threshold)
        except Exception:
            rows.append({"part": trial.part, "valid": False})
            continue
        row = {"part": trial.part, "valid": True}
        for var in BASE_VARS:
            if var.startswith("non_cut_"):
                row[var] = non_cut_metrics[var[len("non_cut_"):]]
            else:
                row[var] = cut_metrics[var]
        rows.append(row)
    features = {}
    for part in ("A", "B"):
        valid_rows = [r for r in rows if r["part"] == part and r["valid"]]
        for var in BASE_VARS:
            values = [r.get(var) for r in valid_rows if r.get(var) is not None and not np.isnan(r.get(var))]
            if values:
                features[f"{var}_PART_{part}"] = float(np.mean(values))
        features[f"is_valid_sum_{part}"] = 100.0 * len(valid_rows) / 10.0
    for var in BASE_VARS:
        a = features.get(f"{var}_PART_A")
        b = features.get(f"{var}_PART_B")
        if a is not None and b is not None and not np.isnan(a) and not np.isnan(b):
            features[f"{var}_B_A_ratio"] = float(b / a) if a != 0 else np.nan
    if feature_columns is None:
        feature_columns = json.loads((Path(__file__).parent / "model_feature_columns.json").read_text(encoding="utf-8"))
    missing = [c for c in feature_columns if c not in features or not np.isfinite(features[c])]
    if missing:
        return {"model_ready": False, "missing_features": missing, "speed_threshold": threshold, "features": features, "trial_rows": rows}
    ordered = {c: float(features[c]) for c in feature_columns}
    return {"model_ready": True, "missing_features": [], "speed_threshold": threshold, "features": ordered, "trial_rows": rows}

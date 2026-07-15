"""Closed-loop progress correction and D0.1 trajectory classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from d0 import CLASSIFIER_VERSION


RAW_OUTCOME_MAP = {
    "following": "follow",
    "overtaking": "overtake",
    "collision": "collision",
}

EQUALITY_FIELDS = (
    "archived_outcome_raw",
    "archived_outcome3",
    "ego_collision",
    "opp_collision",
    "collision_any",
    "alignment_status",
    "alignment_k",
    "rel_start_hex",
    "rel_terminal_hex",
    "ego_wrap_count",
    "opp_wrap_count",
    "physics_status",
    "terminal_gap_hex",
    "frame_spacing_status",
    "censored",
    "interaction_attempt",
    "confirmed_safe_pass",
    "attempted_follow_no_collision",
    "corrected_outcome3",
    "four_state",
    "collision_involvement",
    "collision_cause",
    "collision_phase",
    "collision_final_dist_hex",
)


@dataclass(frozen=True)
class UnwrapResult:
    values: np.ndarray
    wrap_count: int


@dataclass(frozen=True)
class RelSeries:
    values: np.ndarray
    raw_values: np.ndarray
    k: int
    length: float
    ego_wrap_count: int
    opp_wrap_count: int
    status: str


@dataclass(frozen=True)
class CollisionEvent:
    ego_collision: bool
    opp_collision: bool
    involvement: str
    cause: str
    phase: str
    final_dist_hex: str
    terminal_rel_hex: str
    classifier_version: str = CLASSIFIER_VERSION


@dataclass(frozen=True)
class OutcomeRecord:
    archived_outcome_raw: str
    archived_outcome3: str
    ego_collision: bool
    opp_collision: bool
    collision_any: bool
    alignment_status: str
    alignment_k: int | str
    rel_start_hex: str
    rel_terminal_hex: str
    ego_wrap_count: int
    opp_wrap_count: int
    physics_status: str
    terminal_gap_hex: str
    frame_spacing_status: str
    censored: bool
    interaction_attempt: bool | str
    confirmed_safe_pass: bool | str
    attempted_follow_no_collision: bool | str
    corrected_outcome3: str
    four_state: str
    collision_involvement: str
    collision_cause: str
    collision_phase: str
    collision_final_dist_hex: str
    rel_series: RelSeries


def _array(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _scalar(value, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    result = float(array.reshape(()))
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bool_scalar(value, name: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    return bool(array.reshape(()))


def centerline_length(asset_root: str | Path, map_name: str) -> float:
    path = Path(asset_root) / map_name / "raceline1.csv"
    rows = np.loadtxt(path, delimiter=";", skiprows=1, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] < 3:
        raise ValueError(f"invalid centerline asset: {path}")
    xy = rows[:, [1, 2]]
    if not np.all(np.isfinite(xy)):
        raise ValueError(f"nonfinite centerline asset: {path}")
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def unwrap_progress(raw: Sequence[float], length: float) -> UnwrapResult:
    values = _array(raw, "progress")
    length = float(length)
    if not math.isfinite(length) or length <= 0:
        raise ValueError("centerline length must be finite and positive")
    output = np.empty_like(values, dtype=np.float64)
    output[0] = values[0]
    wraps = 0
    half = length / 2.0
    for index in range(1, len(values)):
        delta = values[index] - values[index - 1]
        if delta > half:
            delta -= length
            wraps += 1
        elif delta < -half:
            delta += length
            wraps += 1
        output[index] = output[index - 1] + delta
    return UnwrapResult(values=output, wrap_count=wraps)


def align_rel(
    ego_recorded: Sequence[float],
    opp_recorded: Sequence[float],
    ego_terminal: float,
    opp_terminal: float,
    length: float,
) -> RelSeries:
    ego = _array(ego_recorded, "ego_progress")
    opp = _array(opp_recorded, "opp_progress")
    if len(ego) != len(opp):
        raise ValueError("ego/opponent progress lengths differ")
    ego_all = np.concatenate([ego, [_scalar(ego_terminal, "final_ego_progress")]])
    opp_all = np.concatenate([opp, [_scalar(opp_terminal, "final_opp_progress")]])
    ego_unwrapped = unwrap_progress(ego_all, length)
    opp_unwrapped = unwrap_progress(opp_all, length)
    raw_rel = ego_unwrapped.values - opp_unwrapped.values
    length = float(length)
    k = int(math.floor((length / 2.0 - float(raw_rel[0])) / length))
    aligned = raw_rel + k * length
    in_principal = -length / 2.0 < float(aligned[0]) <= length / 2.0
    status = "ok" if in_principal and float(aligned[0]) < 0.0 else "alignment_failure"
    return RelSeries(
        values=aligned,
        raw_values=raw_rel,
        k=k,
        length=length,
        ego_wrap_count=ego_unwrapped.wrap_count,
        opp_wrap_count=opp_unwrapped.wrap_count,
        status=status,
    )


def normalize_archived_outcome(raw: str) -> str:
    if not isinstance(raw, str) or raw not in RAW_OUTCOME_MAP:
        raise ValueError(
            f"archived outcome must be one of {sorted(RAW_OUTCOME_MAP)}, got {raw!r}"
        )
    return RAW_OUTCOME_MAP[raw]


def _time_status(npz: Mapping[str, Any]) -> tuple[str, str, float]:
    try:
        time = _array(npz["time"], "time")
        final_time = _scalar(npz["final_time"], "final_time")
    except (KeyError, ValueError):
        return "invalid", "invalid", float("nan")
    gap = final_time - float(time[-1])
    tolerance = 1e-12
    gap_ok = gap > 0.0 and abs(gap - 0.01) <= 0.005 + tolerance
    if len(time) <= 1:
        spacing_ok = True
    else:
        deltas = np.diff(time)
        spacing_ok = bool(
            np.all(np.isfinite(deltas))
            and np.all(deltas > 0.0)
            and np.all(deltas >= 0.005 - tolerance)
            and np.all(deltas <= 0.015 + tolerance)
        )
    spacing = "ok" if spacing_ok else "invalid"
    physics = "ok" if gap_ok and spacing_ok else "invalid"
    return physics, spacing, gap


def classify_collision(npz: Mapping[str, Any], rel_series: RelSeries) -> CollisionEvent:
    ego_collision = _bool_scalar(npz["ego_collision"], "ego_collision")
    opp_collision = _bool_scalar(npz["opp_collision"], "opp_collision")
    collision_any = ego_collision or opp_collision
    terminal_rel = float(rel_series.values[-1])
    if not collision_any:
        return CollisionEvent(
            ego_collision=False,
            opp_collision=False,
            involvement="not_applicable",
            cause="not_applicable",
            phase="not_applicable",
            final_dist_hex="not_applicable",
            terminal_rel_hex=terminal_rel.hex(),
        )
    involvement = "both" if ego_collision and opp_collision else (
        "ego_only" if ego_collision else "opp_only"
    )
    ego_pose = np.asarray(npz["final_ego_pose"], dtype=np.float64)
    opp_pose = np.asarray(npz["final_opp_pose"], dtype=np.float64)
    if ego_pose.shape != (3,) or opp_pose.shape != (3,) or not np.all(np.isfinite(ego_pose)) or not np.all(np.isfinite(opp_pose)):
        raise ValueError("terminal collision poses must be finite 3-vectors")
    distance = float(np.linalg.norm(ego_pose[:2] - opp_pose[:2]))
    cause = "car" if distance <= 1.0 else "wall"
    if rel_series.status != "ok":
        phase = "unknown"
    elif abs(terminal_rel) < 0.6 and not math.isclose(
        abs(terminal_rel), 0.6, rel_tol=0.0, abs_tol=1e-12
    ):
        phase = "alongside"
    elif terminal_rel < 0.0:
        phase = "pre"
    else:
        phase = "post"
    return CollisionEvent(
        ego_collision=ego_collision,
        opp_collision=opp_collision,
        involvement=involvement,
        cause=cause,
        phase=phase,
        final_dist_hex=distance.hex(),
        terminal_rel_hex=terminal_rel.hex(),
    )


def classify_outcome(
    npz: Mapping[str, Any],
    json_episode: Mapping[str, Any],
    length: float,
    *,
    attempt_threshold: float = 0.6,
    lead_threshold: float = 2.0,
    hold_seconds: float = 0.7,
) -> OutcomeRecord:
    raw = json_episode.get("outcome")
    state_label = json_episode.get("state_label")
    if raw != state_label:
        raise ValueError("results.json outcome/state_label disagreement")
    archived3 = normalize_archived_outcome(raw)
    if "state_label" in npz:
        npz_label = str(np.asarray(npz["state_label"]).reshape(()))
        if npz_label != raw:
            raise ValueError("JSON/NPZ raw label disagreement")

    ego_collision = _bool_scalar(npz["ego_collision"], "ego_collision")
    opp_collision = _bool_scalar(npz["opp_collision"], "opp_collision")
    collision_any = ego_collision or opp_collision
    aggregate_collision = _bool_scalar(npz["collision"], "collision")
    if aggregate_collision != collision_any:
        raise ValueError("collision != ego_collision or opp_collision")
    if bool(json_episode.get("ego_collision")) != ego_collision or bool(json_episode.get("opp_collision")) != opp_collision:
        raise ValueError("JSON/NPZ collision flag disagreement")

    ego_progress = _array(npz["ego_progress"], "ego_progress")
    opp_progress = _array(npz["opp_progress"], "opp_progress")
    rel_series = align_rel(
        ego_progress,
        opp_progress,
        npz["final_ego_progress"],
        npz["final_opp_progress"],
        length,
    )
    physics_status, spacing_status, gap = _time_status(npz)
    terminal_gap_hex = gap.hex() if math.isfinite(gap) else "unknown"
    event = classify_collision(npz, rel_series)

    time = _array(npz["time"], "time")
    final_time = _scalar(npz["final_time"], "final_time")
    trace_span = final_time - float(time[0])
    censored = bool(not collision_any and trace_span + 1e-12 < hold_seconds)
    valid_mechanism = physics_status == "ok" and rel_series.status == "ok"

    if collision_any:
        corrected3: str = "collision"
    elif rel_series.status == "ok":
        corrected3 = "overtake" if float(rel_series.values[-1]) > 0.0 else "follow"
    else:
        corrected3 = "unknown"

    if not valid_mechanism:
        interaction_attempt: bool | str = "unknown"
        confirmed_safe_pass: bool | str = False if collision_any else "unknown"
        attempted_follow: bool | str = "unknown"
        four_state = "collision" if collision_any else "unknown"
        if not collision_any:
            censored = True
    else:
        interaction_attempt = bool(np.any(np.abs(rel_series.values) <= attempt_threshold))
        if collision_any:
            confirmed_safe_pass = False
        elif censored:
            confirmed_safe_pass = False
        else:
            all_times = np.concatenate([time, [final_time]])
            cutoff = final_time - hold_seconds
            mask = (all_times >= cutoff - 1e-12) & (all_times <= final_time + 1e-12)
            confirmed_safe_pass = bool(np.any(mask) and np.all(rel_series.values[mask] >= lead_threshold))
        if confirmed_safe_pass and corrected3 != "overtake":
            raise AssertionError("confirmed_safe_pass must imply corrected overtake")
        if collision_any:
            four_state = "collision"
        elif corrected3 == "overtake" and confirmed_safe_pass:
            four_state = "confirmed_pass"
        elif corrected3 == "overtake":
            four_state = "terminal_overtake_only"
        elif corrected3 == "follow":
            four_state = "safe_follow"
        else:
            four_state = "unknown"
        attempted_follow = bool(
            interaction_attempt and four_state == "safe_follow" and not collision_any
        )

    return OutcomeRecord(
        archived_outcome_raw=raw,
        archived_outcome3=archived3,
        ego_collision=ego_collision,
        opp_collision=opp_collision,
        collision_any=collision_any,
        alignment_status=rel_series.status,
        alignment_k=rel_series.k,
        rel_start_hex=float(rel_series.values[0]).hex(),
        rel_terminal_hex=float(rel_series.values[-1]).hex(),
        ego_wrap_count=rel_series.ego_wrap_count,
        opp_wrap_count=rel_series.opp_wrap_count,
        physics_status=physics_status,
        terminal_gap_hex=terminal_gap_hex,
        frame_spacing_status=spacing_status,
        censored=censored,
        interaction_attempt=interaction_attempt,
        confirmed_safe_pass=confirmed_safe_pass,
        attempted_follow_no_collision=attempted_follow,
        corrected_outcome3=corrected3,
        four_state=four_state,
        collision_involvement=event.involvement,
        collision_cause=event.cause,
        collision_phase=event.phase,
        collision_final_dist_hex=event.final_dist_hex,
        rel_series=rel_series,
    )


def equality_vector(record: OutcomeRecord) -> tuple:
    return tuple(getattr(record, field) for field in EQUALITY_FIELDS)

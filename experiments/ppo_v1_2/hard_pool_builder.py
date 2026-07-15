"""Deterministic expanded-scenario and fixed-seed hard-pool construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from rl.ppo_scenarios import (
    EGO_RACELINE,
    EVALUATION_STARTPOINTS,
    EXPANDED_INTERVAL_IDXS,
    EXPANDED_SPEED_SCALES,
    MAP_NAME,
    OPPONENT_RACELINES,
    ScenarioSpec,
)
from utils import find_corresponding_waypoint, load_raceline_waypoints

from .experiment_spec import PROJECT_ROOT, canonical_hash


EXPANDED_PANEL_ID = "train_austin_expanded_v1_2"
PASS1_SEEDS = (20260715, 20260716, 20260717, 20260718)
PASS2_SEEDS = (20260719, 20260720, 20260721, 20260722)


def deterministic_expanded_startpoints() -> tuple[int, ...]:
    """Select 100 progress-spaced indices satisfying the evaluation-distance gate."""

    data = np.loadtxt(
        PROJECT_ROOT / "f1tenth_racetracks" / "Austin" / "raceline1.csv",
        delimiter=";",
        comments="#",
        dtype=np.float64,
    )
    if np.linalg.norm(data[-1, 1:3] - data[0, 1:3]) > 1e-9:
        raise ValueError("Austin raceline1 must contain the duplicated closing endpoint")
    unique = data[:-1]
    if len(unique) != 2096:
        raise ValueError(f"Expected 2096 unique Austin raceline1 waypoints, got {len(unique)}")
    evaluation_xy = data[np.asarray(EVALUATION_STARTPOINTS, dtype=np.int64) % len(unique), 1:3]
    distances = np.linalg.norm(unique[:, None, 1:3] - evaluation_xy[None, :, :], axis=2)
    allowed = np.flatnonzero(np.min(distances, axis=1) >= 1.0 - 1e-12)
    if len(allowed) < 100:
        raise RuntimeError("Fewer than 100 Austin waypoints satisfy the expanded-pool separation gate")
    track_length = float(data[-1, 0])
    selected: list[int] = []
    for target in (np.arange(100, dtype=np.float64) + 0.5) * track_length / 100.0:
        progress_delta = np.abs(unique[allowed, 0] - target)
        progress_delta = np.minimum(progress_delta, track_length - progress_delta)
        order = np.lexsort((allowed, progress_delta))
        index = next(int(allowed[position]) for position in order if int(allowed[position]) not in selected)
        selected.append(index)
    ordered = tuple(sorted(selected, key=lambda index: float(unique[index, 0])))
    if len(ordered) != 100 or len(set(ordered)) != 100:
        raise AssertionError("Expanded startpoint selection is not 100 unique indices")
    if any(np.min(np.linalg.norm(unique[index, 1:3] - evaluation_xy, axis=1)) < 1.0 - 1e-12 for index in ordered):
        raise AssertionError("Expanded startpoint violates evaluation XY separation")
    return ordered


def _expanded_id(ordinal: int, ego_idx: int, raceline: str, interval: int, speed: float) -> str:
    return f"v12-sp{ordinal:03d}-ego{ego_idx:04d}-{raceline}-i{interval:02d}-v{int(round(speed*100)):03d}"


def expanded_scenarios(startpoints: Sequence[int] | None = None) -> tuple[ScenarioSpec, ...]:
    startpoints = tuple(deterministic_expanded_startpoints() if startpoints is None else startpoints)
    if len(startpoints) != 100 or len(set(startpoints)) != 100:
        raise ValueError("Expanded pool requires exactly 100 unique startpoints")
    ego_waypoints = load_raceline_waypoints(MAP_NAME, f"{EGO_RACELINE}.csv")
    opponent_waypoints = {
        raceline: load_raceline_waypoints(MAP_NAME, f"{raceline}.csv")
        for raceline in OPPONENT_RACELINES
    }
    rows: list[ScenarioSpec] = []
    for ordinal, ego_idx in enumerate(startpoints):
        ego_waypoint = ego_waypoints[ego_idx]
        for raceline in OPPONENT_RACELINES:
            mapped = ego_idx if raceline == EGO_RACELINE else int(find_corresponding_waypoint(ego_waypoint, opponent_waypoints[raceline]))
            for interval in EXPANDED_INTERVAL_IDXS:
                opponent_index = (mapped + interval) % (len(opponent_waypoints[raceline]) - 1)
                for speed in EXPANDED_SPEED_SCALES:
                    rows.append(
                        ScenarioSpec(
                            scenario_id=_expanded_id(ordinal, ego_idx, raceline, interval, speed),
                            pool=EXPANDED_PANEL_ID,
                            startpoint_ordinal=ordinal,
                            ego_idx=int(ego_idx),
                            opp_idx=int(opponent_index),
                            opp_raceline=raceline,
                            opp_speedscale=float(speed),
                            interval_idx=int(interval),
                        )
                    )
    ids = [row.scenario_id for row in rows]
    if len(rows) != 10_800 or len(ids) != len(set(ids)):
        raise AssertionError("Expanded scenario Cartesian product must contain 10,800 unique IDs")
    return tuple(rows)


def validate_candidates(
    candidates: Sequence[ScenarioSpec],
    preflight: Callable[[ScenarioSpec], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run every preflight independently; errors are explicit invalid rows."""

    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for scenario in candidates:
        try:
            checks = dict(preflight(scenario))
            required = ("reset", "poses_finite", "observation_finite", "initial_collision_free", "rectangles_disjoint", "planner_constructed")
            passed = all(bool(checks.get(name, False)) for name in required)
            row = {**asdict(scenario), "preflight": checks, "valid": passed}
        except Exception as error:
            row = {
                **asdict(scenario),
                "preflight": {},
                "valid": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        (valid if row["valid"] else invalid).append(row)
    summary = {
        "candidate_count": len(candidates),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "complete": len(valid) + len(invalid) == len(candidates),
    }
    return valid, invalid, summary


def classify_stochastic_rows(rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Derive the exact H2 sets from persisted eight-seed outcomes."""

    core: list[str] = []
    boundary: list[str] = []
    all_ids: list[str] = []
    for row in rows:
        outcomes = row.get("seed_outcomes", {})
        if sorted(map(int, outcomes)) != sorted((*PASS1_SEEDS, *PASS2_SEEDS)):
            raise ValueError(f"Stochastic row lacks the fixed eight seeds: {row.get('scenario_id')}")
        collision_count = sum(value.get("outcome") == "ego_collision" for value in outcomes.values())
        if int(row.get("collision_count", collision_count)) != collision_count:
            raise ValueError(f"collision_count mismatch: {row.get('scenario_id')}")
        scenario_id = str(row["scenario_id"])
        if collision_count >= 1:
            all_ids.append(scenario_id)
        if collision_count == 1:
            boundary.append(scenario_id)
        if collision_count >= 2:
            core.append(scenario_id)
    return {
        "H2_STOCH_CORE": sorted(core),
        "H2_STOCH_BOUNDARY": sorted(boundary),
        "H2_STOCH_ALL": sorted(all_ids),
    }


def union_pool(*groups: Iterable[str]) -> list[str]:
    return sorted(set().union(*(set(group) for group in groups)))


def pool_manifest(pool_id: str, scenario_ids: Sequence[str], by_id: dict[str, ScenarioSpec]) -> dict[str, Any]:
    ids = sorted(map(str, scenario_ids))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Pool {pool_id} contains duplicate IDs")
    missing = sorted(set(ids) - set(by_id))
    if missing:
        raise ValueError(f"Pool {pool_id} references unknown IDs: {missing[:3]}")
    scenarios = [by_id[scenario_id] for scenario_id in ids]
    distributions = {
        "interval_idx": dict(sorted(Counter(str(row.interval_idx) for row in scenarios).items())),
        "opp_raceline": dict(sorted(Counter(row.opp_raceline for row in scenarios).items())),
        "opp_speedscale": dict(sorted(Counter(f"{row.opp_speedscale:.2f}" for row in scenarios).items())),
        "startpoint_ordinal": dict(sorted(Counter(str(row.startpoint_ordinal) for row in scenarios).items(), key=lambda pair: int(pair[0]))),
    }
    content = {
        "pool_id": pool_id,
        "scenario_ids": ids,
        "scenarios": [asdict(row) for row in scenarios],
        "count": len(ids),
        "distributions": distributions,
    }
    content["manifest_hash"] = canonical_hash(content)
    return content


def collision_step_summary(seed_outcomes: dict[str, dict[str, Any]]) -> dict[str, float | int | None]:
    steps = [int(row["collision_step"]) for row in seed_outcomes.values() if row.get("outcome") == "ego_collision" and row.get("collision_step") is not None]
    return {
        "collision_count": len(steps),
        "collision_rate": len(steps) / 8.0,
        "mean_collision_step": float(np.mean(steps)) if steps else None,
        "min_collision_step": min(steps, default=None),
        "max_collision_step": max(steps, default=None),
    }

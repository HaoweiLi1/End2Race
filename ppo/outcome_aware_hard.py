"""Outcome-aware hard-neighbor collision pool (isolated, review-before-merge).

This module is *additive*: it imports the existing boundary-discovery machinery
from :mod:`ppo.hard_neighbors` read-only and never mutates the baseline or
``boundary-aware-v1`` caches. It builds an independent cache whose final
collision pool is a *pure function* of

    (richer BC replay labels) x (an explicit, recorded filter specification)

so the pool is reproducible and self-verifying on load.

Why this exists
---------------
The shipped hard-neighbor cache defines a boundary by the binary
``ego_collision`` vs ``other`` label. But ``other`` mixes genuine safe
overtakes, follows (opponent too fast to pass), and near-misses. A boundary
collision produced next to a *follow* endpoint is a different failure than one
produced next to a *safe overtake*. This module replays both sides of every
boundary pair with the frozen BC actor, records the real terminal outcome
(overtake / follow) and clearance for the ``other`` endpoint, and records the
collision-moment geometry (post-overtake rear contact vs rear-end vs wall, plus
a fishtail severity flag) for the collision endpoint. A configurable filter then
selects which boundary collisions enter training.

Contract preservation
----------------------
The replay worker is byte-for-byte the frozen-BC rollout used by the shipped
classifier (same ``make_environment(0, map_name)`` env: postpass OFF, privileged
OFF; same action clip; same reset-validity check). Therefore the ``ego_collision``
labels this module produces for the boundary candidates are identical to
``boundary-aware-v1``'s, and ``--filter_mode all`` reconstructs that cache's
final pool exactly. The cleanup-era regression contract for that reconstruction
is preserved in ``.agents/EXPERIMENTS.md``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np
import torch

from model import End2Race
from ppo.collision_classification import (
    collision_classification_config,
    load_collision_cache_artifacts,
    resolve_collision_scenarios,
)
from ppo.environment import EXTERNAL_RESET_OPTION, make_environment
from ppo.hard_neighbors import (
    BASE_CACHE_FILES,
    BOUNDARY_GENERATOR_SCHEMA,
    CLASSIFIER_CONTRACT,
    SPEED_FIXED_POINT_SCALE,
    _asset_hashes,
    _sha256_file,
    _speed_milli,
    discover_boundary_candidates,
    materialize_boundary_candidates,
)
from ppo.policy import END2RACE_LIDAR_SIZE, STEERING_BOUND
from ppo.scenarios import (
    COLLISION_INTERVAL_INDICES,
    COLLISION_SPEED_SCALES,
    EGO_RACELINE,
    OPPONENT_RACELINES,
    ScenarioSpec,
)
from ppo.vec_env import limit_worker_threads


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Independent schema so this cache can never be confused with the shipped hard
# cache. Bump on any change to the label record or filter semantics.
OUTCOME_AWARE_CACHE_SCHEMA = 1
LABEL_RECORD_SCHEMA = 1

# Geometry thresholds. These reproduce the offline collision-mode taxonomy that
# was reviewed and accepted for the 0722 analysis; keep them in one place so the
# audit and the training filter agree by construction.
WALL_CENTER_DISTANCE_M = 0.75          # car-car center distance above this => wall hit
REAR_CONTACT_BEARING_DEG = 80.0        # opponent behind ego beam half-angle
REAR_END_BEARING_DEG = 60.0            # opponent ahead of ego beam half-angle
REAR_END_REL_TRACK_M = -0.10           # ego trails opponent along the track
FISHTAIL_SLIP_DEG = 8.0                # kinematic slip severity gate
FISHTAIL_HEADING_DEG = 20.0            # inter-car heading divergence gate
SLIP_WINDOW_STEPS = 60                 # pre-impact window for max |slip|
SLIP_STRIDE_STEPS = 5                  # finite-difference stride for velocity heading
SLIP_MIN_SPEED_MPS = 1.0               # ignore slip when nearly stopped

VALID_FILTER_MODES = ("all", "safe_overtake", "fishtail", "fishtail_rearend")
COLLISION_MODES = ("post_overtake_rear", "rear_end_opp", "wall", "side_other")


# --------------------------------------------------------------------------- #
# Richer per-candidate label record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CandidateLabel:
    """One frozen-BC replay outcome enriched beyond the binary collision flag."""

    candidate_index: int
    scenario_id: str
    outcome: str                       # ego_collision | overtake | follow | invalid
    terminal_relative_position_m: float
    min_obb_clearance_m: float
    steps: int
    elapsed_time_s: float
    # collision-only geometry; None-valued when outcome != "ego_collision"
    collision_time_s: float | None = None
    collision_center_distance_m: float | None = None
    collision_bearing_deg: float | None = None
    collision_rel_track_m: float | None = None
    collision_delta_heading_deg: float | None = None
    collision_ego_slip_max_deg: float | None = None
    collision_true_ego_slip_deg: float | None = None
    collision_partner_is_opponent: bool | None = None
    collision_mode: str | None = None
    collision_is_fishtail: bool | None = None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["label_schema"] = LABEL_RECORD_SCHEMA
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "CandidateLabel":
        payload = dict(record)
        schema = payload.pop("label_schema", None)
        if schema != LABEL_RECORD_SCHEMA:
            raise RuntimeError(
                f"Label record schema {schema!r} does not match {LABEL_RECORD_SCHEMA}"
            )
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        if set(payload) != field_names:
            raise RuntimeError(
                f"Label record fields do not match CandidateLabel for {payload.get('scenario_id')!r}"
            )
        return cls(**payload)


# --------------------------------------------------------------------------- #
# Frozen-BC replay worker (contract-identical to the shipped classifier)
# --------------------------------------------------------------------------- #
_LABEL_ENV = None
_LABEL_ACTOR = None


def _label_worker_init(pretrained_model_path: str, hidden_scale: int, map_name: str) -> None:
    global _LABEL_ENV, _LABEL_ACTOR
    limit_worker_threads()
    _LABEL_ENV = make_environment(0, map_name)()
    _LABEL_ACTOR = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)
    _LABEL_ACTOR.load_state_dict(
        torch.load(pretrained_model_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    _LABEL_ACTOR.eval()


def _wrap_angle_rad(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _true_ego_slip_deg(env: Any) -> float | None:
    """Best-effort read of the simulator's true tire slip (state index 6).

    Read-only; does not affect dynamics, so it never changes the label outcome.
    Returns None if the private attribute path is unavailable.
    """
    try:
        state = env.f110_env.sim.agents[0].state
        return float(np.degrees(float(state[6])))
    except Exception:
        return None


def _classify_collision_mode(
    partner_is_opponent: bool,
    passed_before: bool,
    rel_track_m: float,
    bearing_deg: float,
) -> str:
    if not partner_is_opponent:
        return "wall"
    if passed_before and rel_track_m > 0.0 and abs(bearing_deg) > REAR_CONTACT_BEARING_DEG:
        return "post_overtake_rear"
    if rel_track_m < REAR_END_REL_TRACK_M and abs(bearing_deg) < REAR_END_BEARING_DEG:
        return "rear_end_opp"
    return "side_other"


def _replay_label(index: int, scenario: ScenarioSpec) -> CandidateLabel:
    """Roll out the frozen BC actor once and produce a CandidateLabel."""
    env, actor = _LABEL_ENV, _LABEL_ACTOR
    if env is None or actor is None:
        raise RuntimeError("Outcome-aware label worker is not initialized")
    observation, _info = env.reset(
        options={EXTERNAL_RESET_OPTION: scenario.to_reset_spec("collision")}
    )
    raw = env._raw_observation
    finite = np.isfinite(observation).all() and all(
        np.isfinite(np.asarray(value)).all()
        for value in raw.values()
        if isinstance(value, (list, tuple, np.ndarray))
    )
    if not finite or np.asarray(raw["collisions"], dtype=bool).any():
        return CandidateLabel(
            candidate_index=index,
            scenario_id=scenario.scenario_id,
            outcome="invalid",
            terminal_relative_position_m=0.0,
            min_obb_clearance_m=0.0,
            steps=0,
            elapsed_time_s=0.0,
        )

    # Rolling pre-impact ego-pose window for kinematic slip.
    ego_x_hist: list[float] = []
    ego_y_hist: list[float] = []
    ego_theta_hist: list[float] = []
    time_hist: list[float] = []
    passed_before = False
    hidden = None
    while True:
        actor_observation = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            actions, hidden = actor(
                actor_observation[:END2RACE_LIDAR_SIZE].reshape(1, 1, -1),
                actor_observation[END2RACE_LIDAR_SIZE:].reshape(1, 1, 1),
                hidden,
            )
        action = actions[0, -1].numpy().copy()
        action[0] = np.clip(action[0], -STEERING_BOUND, STEERING_BOUND)
        if not np.isfinite(action).all():
            raise RuntimeError(f"Non-finite BC action for {scenario.scenario_id}")

        raw_before = env._raw_observation
        ego_x_hist.append(float(np.asarray(raw_before["poses_x"])[0]))
        ego_y_hist.append(float(np.asarray(raw_before["poses_y"])[0]))
        ego_theta_hist.append(float(np.asarray(raw_before["poses_theta"])[0]))
        time_hist.append(float(env._elapsed_time))

        observation, _reward, terminated, truncated, info = env.step(action)
        if float(info["relative_position_m"]) > 0.0:
            passed_before = True
        if terminated or truncated:
            outcome = str(info["episode_outcome"])
            base = dict(
                candidate_index=index,
                scenario_id=scenario.scenario_id,
                outcome=outcome,
                terminal_relative_position_m=float(info["relative_position_m"]),
                min_obb_clearance_m=float(info["episode_min_obb_clearance_m"]),
                steps=int(info["episode_steps"]),
                elapsed_time_s=float(info["elapsed_time"]),
            )
            if outcome != "ego_collision":
                return CandidateLabel(**base)
            return _finalize_collision_label(env, info, base, ego_x_hist, ego_y_hist, ego_theta_hist, time_hist, passed_before)


def _finalize_collision_label(
    env: Any,
    info: dict[str, Any],
    base: dict[str, Any],
    ego_x_hist: list[float],
    ego_y_hist: list[float],
    ego_theta_hist: list[float],
    time_hist: list[float],
    passed_before: bool,
) -> CandidateLabel:
    raw = env._raw_observation
    ex = float(np.asarray(raw["poses_x"])[0])
    ey = float(np.asarray(raw["poses_y"])[0])
    eth = float(np.asarray(raw["poses_theta"])[0])
    ox = float(np.asarray(raw["poses_x"])[1])
    oy = float(np.asarray(raw["poses_y"])[1])
    oth = float(np.asarray(raw["poses_theta"])[1])
    collisions = np.asarray(raw["collisions"], dtype=bool).reshape(-1)

    dxw, dyw = ox - ex, oy - ey
    center_distance = float(np.hypot(dxw, dyw))
    ca, sa = np.cos(eth), np.sin(eth)
    dx = ca * dxw + sa * dyw            # +x forward in ego body frame
    dy = -sa * dxw + ca * dyw           # +y left in ego body frame
    bearing_deg = float(np.degrees(np.arctan2(dy, dx)))
    delta_heading_deg = float(np.degrees(_wrap_angle_rad(oth - eth)))
    rel_track_m = float(info["relative_position_m"])
    partner_is_opponent = bool(collisions[1]) or center_distance <= WALL_CENTER_DISTANCE_M

    slip_max_deg = _kinematic_slip_max_deg(ego_x_hist, ego_y_hist, ego_theta_hist, time_hist)
    mode = _classify_collision_mode(partner_is_opponent, passed_before, rel_track_m, bearing_deg)
    is_fishtail = bool(slip_max_deg >= FISHTAIL_SLIP_DEG or abs(delta_heading_deg) >= FISHTAIL_HEADING_DEG)

    return CandidateLabel(
        **base,
        collision_time_s=float(info["elapsed_time"]),
        collision_center_distance_m=round(center_distance, 6),
        collision_bearing_deg=round(bearing_deg, 3),
        collision_rel_track_m=round(rel_track_m, 6),
        collision_delta_heading_deg=round(delta_heading_deg, 3),
        collision_ego_slip_max_deg=round(slip_max_deg, 3),
        collision_true_ego_slip_deg=_maybe_round(_true_ego_slip_deg(env), 3),
        collision_partner_is_opponent=partner_is_opponent,
        collision_mode=mode,
        collision_is_fishtail=is_fishtail,
    )


def _maybe_round(value: float | None, ndigits: int) -> float | None:
    return None if value is None else round(float(value), ndigits)


def _kinematic_slip_max_deg(
    xs: list[float],
    ys: list[float],
    thetas: list[float],
    times: list[float],
) -> float:
    n = len(xs)
    if n <= SLIP_STRIDE_STEPS:
        return 0.0
    lo = max(0, n - SLIP_WINDOW_STEPS)
    x = np.asarray(xs[lo:], dtype=np.float64)
    y = np.asarray(ys[lo:], dtype=np.float64)
    th = np.asarray(thetas[lo:], dtype=np.float64)
    t = np.asarray(times[lo:], dtype=np.float64)
    k = SLIP_STRIDE_STEPS
    if len(x) <= k:
        return 0.0
    dt = t[k:] - t[:-k]
    dt = np.where(np.abs(dt) < 1e-9, np.nan, dt)
    vx = (x[k:] - x[:-k]) / dt
    vy = (y[k:] - y[:-k]) / dt
    speed = np.hypot(vx, vy)
    slip = np.degrees(np.array([_wrap_angle_rad(a) for a in (np.arctan2(vy, vx) - th[k:])]))
    slip = np.where(speed > SLIP_MIN_SPEED_MPS, slip, 0.0)
    slip = np.where(np.isfinite(slip), slip, 0.0)
    return float(np.max(np.abs(slip))) if slip.size else 0.0


def classify_labeled_scenarios(
    pretrained_model_path: str | Path,
    hidden_scale: int,
    map_name: str,
    env_workers: int,
    candidates: Sequence[ScenarioSpec],
    start_method: str,
) -> tuple[list[CandidateLabel], dict[str, Any]]:
    """Frozen-BC richer replay over ``candidates`` in candidate order."""
    candidates = tuple(candidates)
    candidate_count = len(candidates)
    if candidate_count == 0:
        raise ValueError("classify_labeled_scenarios requires a non-empty candidate set")
    context = mp.get_context(start_method)
    labels: list[CandidateLabel | None] = [None] * candidate_count
    with ProcessPoolExecutor(
        max_workers=env_workers,
        mp_context=context,
        initializer=_label_worker_init,
        initargs=(str(Path(pretrained_model_path).expanduser().resolve()), hidden_scale, map_name),
    ) as executor:
        for completed, label in enumerate(
            executor.map(_replay_label_task, enumerate(candidates), chunksize=4), start=1
        ):
            if label.candidate_index != completed - 1 or label.scenario_id != candidates[label.candidate_index].scenario_id:
                raise RuntimeError(f"Outcome-aware replay returned out-of-order result at {completed - 1}")
            labels[label.candidate_index] = label
            if completed % 100 == 0 or completed == candidate_count:
                print(
                    f"Outcome-aware replay: {completed}/{candidate_count}",
                    flush=True,
                )
    if any(label is None for label in labels):
        raise RuntimeError("Outcome-aware replay did not label every candidate")
    metadata = {
        "candidate_count": candidate_count,
        "env_workers": int(env_workers),
    }
    return [label for label in labels if label is not None], metadata


def _replay_label_task(task: tuple[int, ScenarioSpec]) -> CandidateLabel:
    index, scenario = task
    return _replay_label(index, scenario)


# --------------------------------------------------------------------------- #
# Filter specification and pure selection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FilterSpec:
    mode: str = "all"
    safe_clearance_m: float = 0.10
    require_all_source_pairs_safe: bool = True
    keep_rear_end_quota: bool = True
    drop_unsafe_base: bool = False

    def validate(self) -> "FilterSpec":
        if self.mode not in VALID_FILTER_MODES:
            raise ValueError(f"filter_mode must be one of {VALID_FILTER_MODES}; got {self.mode!r}")
        if not np.isfinite(self.safe_clearance_m) or self.safe_clearance_m < 0.0:
            raise ValueError("safe_clearance_m must be finite and non-negative")
        return self

    def to_config(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "safe_clearance_m": float(self.safe_clearance_m),
            "require_all_source_pairs_safe": bool(self.require_all_source_pairs_safe),
            "keep_rear_end_quota": bool(self.keep_rear_end_quota),
            "drop_unsafe_base": bool(self.drop_unsafe_base),
        }


def _other_endpoint_id(pair: dict[str, Any]) -> str:
    if pair["low_outcome"] == "other":
        return str(pair["low_scenario_id"])
    return str(pair["high_scenario_id"])


def _pair_is_safe_overtake(
    pair: dict[str, Any],
    other_labels_by_id: dict[str, CandidateLabel],
    spec: FilterSpec,
) -> bool:
    label = other_labels_by_id.get(_other_endpoint_id(pair))
    if label is None:
        raise RuntimeError(f"Missing other-endpoint label for pair {pair['pair_id']}")
    return bool(label.outcome == "overtake" and label.min_obb_clearance_m >= spec.safe_clearance_m)


def apply_outcome_aware_filter(
    spec: FilterSpec,
    base_collisions: Sequence[ScenarioSpec],
    boundary_candidates: Sequence[ScenarioSpec],
    boundary_labels: Sequence[CandidateLabel],
    pair_records: Sequence[dict[str, Any]],
    pair_other_labels: Sequence[CandidateLabel],
    base_collision_labels: Sequence[CandidateLabel] | None = None,
) -> tuple[tuple[ScenarioSpec, ...], dict[str, Any]]:
    """Pure, deterministic selection of the final collision pool + audit."""
    spec.validate()
    if len(boundary_candidates) != len(boundary_labels):
        raise ValueError("boundary candidates and labels must be aligned")

    label_by_id = {label.scenario_id: label for label in boundary_labels}
    other_labels_by_id = {label.scenario_id: label for label in pair_other_labels}

    # "all" is defined purely by the collision labels, so it never needs the
    # (expensive) other-endpoint replay. Every other mode does.
    needs_safety = spec.mode != "all"
    pair_safe: dict[str, bool] = {}
    if needs_safety:
        for pair in pair_records:
            pair_safe[str(pair["pair_id"])] = _pair_is_safe_overtake(pair, other_labels_by_id, spec)

    # Map every generated boundary candidate to the pairs that produced it.
    source_pairs_by_candidate: dict[str, list[str]] = defaultdict(list)
    for pair in pair_records:
        for scenario_id in pair.get("selected_scenario_ids", []):
            source_pairs_by_candidate[str(scenario_id)].append(str(pair["pair_id"]))

    kept: list[ScenarioSpec] = []
    per_pair_decisions: list[dict[str, Any]] = []
    mode_counts: Counter = Counter()
    kept_mode_counts: Counter = Counter()
    reason_counts: Counter = Counter()

    for scenario in boundary_candidates:
        label = label_by_id[scenario.scenario_id]
        if label.outcome != "ego_collision":
            continue
        mode_counts[label.collision_mode] += 1
        source_pairs = source_pairs_by_candidate.get(scenario.scenario_id, [])
        if needs_safety:
            safe_flags = [pair_safe[p] for p in source_pairs]
            if spec.require_all_source_pairs_safe:
                other_safe = bool(safe_flags) and all(safe_flags)
            else:
                other_safe = any(safe_flags)
            other_safe_record: bool | None = other_safe
        else:
            other_safe = False
            other_safe_record = None

        keep, reason = _keep_boundary_collision(spec, label, other_safe)
        reason_counts[reason] += 1
        per_pair_decisions.append(
            {
                "scenario_id": scenario.scenario_id,
                "collision_mode": label.collision_mode,
                "is_fishtail": label.collision_is_fishtail,
                "other_side_safe_overtake": other_safe_record,
                "source_pair_ids": source_pairs,
                "kept": keep,
                "reason": reason,
            }
        )
        if keep:
            kept.append(scenario)
            kept_mode_counts[label.collision_mode] += 1

    # Base collisions: kept intact by default (they are the original BC failures).
    base_kept = list(base_collisions)
    base_dropped = 0
    if spec.drop_unsafe_base:
        if base_collision_labels is None:
            raise ValueError("drop_unsafe_base requires base_collision_labels")
        base_label_by_id = {label.scenario_id: label for label in base_collision_labels}
        filtered = []
        for scenario in base_collisions:
            label = base_label_by_id.get(scenario.scenario_id)
            if label is not None and label.collision_mode == "wall":
                base_dropped += 1
                continue
            filtered.append(scenario)
        base_kept = filtered

    final = tuple(base_kept) + tuple(kept)
    audit = {
        "filter_spec": spec.to_config(),
        "base_collision_count": len(base_collisions),
        "base_collision_kept": len(base_kept),
        "base_collision_dropped": base_dropped,
        "boundary_collision_total": int(sum(mode_counts.values())),
        "boundary_collision_kept": len(kept),
        "boundary_mode_counts": dict(sorted(mode_counts.items())),
        "boundary_kept_mode_counts": dict(sorted(kept_mode_counts.items())),
        "keep_reason_counts": dict(sorted(reason_counts.items())),
        "final_collision_count": len(final),
        "per_candidate_decisions": per_pair_decisions,
    }
    return final, audit


def _keep_boundary_collision(
    spec: FilterSpec,
    label: CandidateLabel,
    other_safe: bool,
) -> tuple[bool, str]:
    mode = label.collision_mode
    if spec.mode == "all":
        return True, "all"
    if spec.mode == "safe_overtake":
        return (other_safe, "safe_overtake" if other_safe else "unsafe_other_side")
    if spec.mode == "fishtail":
        if other_safe and mode == "post_overtake_rear" and bool(label.collision_is_fishtail):
            return True, "safe_fishtail"
        return False, "not_safe_fishtail"
    if spec.mode == "fishtail_rearend":
        if other_safe and mode == "post_overtake_rear" and bool(label.collision_is_fishtail):
            return True, "safe_fishtail"
        if spec.keep_rear_end_quota and mode == "rear_end_opp":
            return True, "rear_end_quota"
        return False, "not_kept"
    raise ValueError(f"Unhandled filter mode: {spec.mode}")


# --------------------------------------------------------------------------- #
# Cache identity, build, and self-verifying load
# --------------------------------------------------------------------------- #
SEMANTIC_CACHE_FILES = (
    "classification_config.json",
    "base_candidate_outcomes.jsonl",
    "boundary_pairs.jsonl",
    "boundary_candidate_labels.jsonl",
    "pair_other_labels.jsonl",
    "base_collision_labels.jsonl",
    "collision_scenarios.json",
    "filter_audit.json",
    "classification_summary.json",
)
CACHE_FILES = SEMANTIC_CACHE_FILES + ("build_metadata.json", "manifest.sha256")


def outcome_aware_cache_config(
    args: Any,
    base_cache_dir: Path,
    base_candidate_count: int,
    boundary_pair_count: int,
    boundary_candidate_count: int,
    spec: FilterSpec,
) -> dict[str, Any]:
    actor_path = Path(args.pretrained_model_path).expanduser().resolve()
    map_hashes, raceline_hashes = _asset_hashes(str(args.map_name))
    lattice_config = PROJECT_ROOT / "latticeplanner" / "lattice_config.yaml"
    if not lattice_config.is_file():
        raise FileNotFoundError(f"Opponent planner config is missing: {lattice_config}")
    base_hashes = {name: _sha256_file(base_cache_dir / name) for name in BASE_CACHE_FILES}
    return {
        "outcome_aware_schema": OUTCOME_AWARE_CACHE_SCHEMA,
        "label_record_schema": LABEL_RECORD_SCHEMA,
        "classifier_contract": CLASSIFIER_CONTRACT,
        "base_actor_path": str(actor_path),
        "base_actor_sha256": _sha256_file(actor_path),
        "hidden_scale": int(args.hidden_scale),
        "map_name": str(args.map_name),
        "map_asset_sha256": map_hashes,
        "raceline_asset_sha256": raceline_hashes,
        "opponent_planner_config_sha256": _sha256_file(lattice_config),
        "base_cache_dir": str(base_cache_dir),
        "base_cache_sha256": base_hashes,
        "base_candidate_count": int(base_candidate_count),
        "base_candidate_generator": {
            "ego_raceline": EGO_RACELINE,
            "opponent_racelines": list(OPPONENT_RACELINES),
            "interval_indices": list(COLLISION_INTERVAL_INDICES),
            "speed_scales": list(COLLISION_SPEED_SCALES),
        },
        "boundary_generator": {
            "schema": BOUNDARY_GENERATOR_SCHEMA,
            "speed_fixed_point_scale": SPEED_FIXED_POINT_SCALE,
        },
        "boundary_pair_count": int(boundary_pair_count),
        "boundary_candidate_count": int(boundary_candidate_count),
        "geometry_thresholds": {
            "wall_center_distance_m": WALL_CENTER_DISTANCE_M,
            "rear_contact_bearing_deg": REAR_CONTACT_BEARING_DEG,
            "rear_end_bearing_deg": REAR_END_BEARING_DEG,
            "rear_end_rel_track_m": REAR_END_REL_TRACK_M,
            "fishtail_slip_deg": FISHTAIL_SLIP_DEG,
            "fishtail_heading_deg": FISHTAIL_HEADING_DEG,
            "slip_window_steps": SLIP_WINDOW_STEPS,
            "slip_stride_steps": SLIP_STRIDE_STEPS,
        },
        "filter_spec": spec.to_config(),
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def _summary(
    base_outcomes: Sequence[dict[str, Any]],
    pair_records: Sequence[dict[str, Any]],
    boundary_labels: Sequence[CandidateLabel],
    pair_other_labels: Sequence[CandidateLabel],
    final_scenarios: Sequence[ScenarioSpec],
    audit: dict[str, Any],
) -> dict[str, Any]:
    base_counts = Counter(record["outcome"] for record in base_outcomes)
    boundary_outcome_counts = Counter(label.outcome for label in boundary_labels)
    other_outcome_counts = Counter(label.outcome for label in pair_other_labels)
    per_raceline = Counter(scenario.opp_raceline for scenario in final_scenarios)
    per_speed = Counter(
        f"{_speed_milli(scenario.opp_speedscale) / SPEED_FIXED_POINT_SCALE:.3f}"
        for scenario in final_scenarios
    )
    return {
        "base_candidate_count": len(base_outcomes),
        "base_collision_count": base_counts["ego_collision"],
        "base_other_count": base_counts["other"],
        "base_invalid_count": base_counts["invalid"],
        "boundary_pair_count": len(pair_records),
        "boundary_candidate_count": len(boundary_labels),
        "boundary_outcome_counts": dict(sorted(boundary_outcome_counts.items())),
        "pair_other_endpoint_count": len(pair_other_labels),
        "pair_other_outcome_counts": dict(sorted(other_outcome_counts.items())),
        "final_collision_count": len(final_scenarios),
        "final_per_raceline": dict(sorted(per_raceline.items())),
        "final_per_speed": dict(sorted(per_speed.items(), key=lambda item: float(item[0]))),
        "boundary_kept_mode_counts": audit["boundary_kept_mode_counts"],
        "boundary_mode_counts": audit["boundary_mode_counts"],
    }


def _labels_to_records(labels: Sequence[CandidateLabel]) -> list[dict[str, Any]]:
    return [label.to_record() for label in labels]


def _records_to_labels(records: Sequence[dict[str, Any]]) -> list[CandidateLabel]:
    return [CandidateLabel.from_record(record) for record in records]


def build_outcome_aware_pool(
    *,
    base_collisions: Sequence[ScenarioSpec],
    base_outcomes: Sequence[dict[str, Any]],
    pair_records: Sequence[dict[str, Any]],
    boundary_candidates: Sequence[ScenarioSpec],
    boundary_labels: Sequence[CandidateLabel],
    pair_other_labels: Sequence[CandidateLabel],
    base_collision_labels: Sequence[CandidateLabel],
    spec: FilterSpec,
) -> tuple[tuple[ScenarioSpec, ...], dict[str, Any]]:
    final, audit = apply_outcome_aware_filter(
        spec,
        base_collisions,
        boundary_candidates,
        boundary_labels,
        pair_records,
        pair_other_labels,
        base_collision_labels,
    )
    if not final:
        raise RuntimeError("Outcome-aware filter produced an empty collision pool")
    if len({scenario.scenario_id for scenario in final}) != len(final):
        raise RuntimeError("Outcome-aware final pool has duplicate scenario IDs")
    return final, audit


def publish_outcome_aware_cache(
    cache_dir: Path,
    config: dict[str, Any],
    base_outcomes: Sequence[dict[str, Any]],
    pair_records: Sequence[dict[str, Any]],
    boundary_labels: Sequence[CandidateLabel],
    pair_other_labels: Sequence[CandidateLabel],
    base_collision_labels: Sequence[CandidateLabel],
    final_scenarios: Sequence[ScenarioSpec],
    audit: dict[str, Any],
    summary: dict[str, Any],
    build_metadata: dict[str, Any],
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite outcome-aware cache; choose a new directory: {cache_dir}"
        )
    temporary = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}.building-", dir=cache_dir.parent))
    try:
        _write_json(temporary / "classification_config.json", config)
        _write_jsonl(temporary / "base_candidate_outcomes.jsonl", list(base_outcomes))
        _write_jsonl(temporary / "boundary_pairs.jsonl", list(pair_records))
        _write_jsonl(temporary / "boundary_candidate_labels.jsonl", _labels_to_records(boundary_labels))
        _write_jsonl(temporary / "pair_other_labels.jsonl", _labels_to_records(pair_other_labels))
        _write_jsonl(temporary / "base_collision_labels.jsonl", _labels_to_records(base_collision_labels))
        _write_json(
            temporary / "collision_scenarios.json",
            [asdict(scenario) for scenario in final_scenarios],
        )
        _write_json(temporary / "filter_audit.json", audit)
        _write_json(temporary / "classification_summary.json", summary)
        _write_json(temporary / "build_metadata.json", build_metadata)
        manifest = "".join(
            f"{_sha256_file(temporary / name)}  {name}\n" for name in SEMANTIC_CACHE_FILES
        )
        (temporary / "manifest.sha256").write_text(manifest, encoding="utf-8")
        os.rename(temporary, cache_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _verify_manifest(cache_dir: Path) -> None:
    lines = (cache_dir / "manifest.sha256").read_text(encoding="utf-8").splitlines()
    entries: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or Path(parts[1]).name != parts[1]:
            raise RuntimeError("Outcome-aware manifest has an invalid entry")
        digest, name = parts
        if name in entries or len(digest) != 64:
            raise RuntimeError("Outcome-aware manifest has duplicate or invalid hashes")
        entries[name] = digest
    if set(entries) != set(SEMANTIC_CACHE_FILES):
        raise RuntimeError("Outcome-aware manifest does not cover the semantic cache files")
    for name, expected in entries.items():
        if _sha256_file(cache_dir / name) != expected:
            raise RuntimeError(f"Outcome-aware cache hash mismatch: {name}")


def cache_exists(cache_dir: Path) -> bool:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return False
    if not cache_dir.is_dir():
        raise RuntimeError(f"Outcome-aware cache path is not a directory: {cache_dir}")
    names = {path.name for path in cache_dir.iterdir()}
    if names != set(CACHE_FILES):
        raise RuntimeError(
            "Outcome-aware cache is incomplete or has unexpected files; use a new empty directory"
        )
    return True


def load_outcome_aware_cache(
    cache_dir: Path,
    current_config: dict[str, Any],
    base_collisions: Sequence[ScenarioSpec],
    base_outcomes: Sequence[dict[str, Any]],
    pair_records: Sequence[dict[str, Any]],
    boundary_candidates: Sequence[ScenarioSpec],
    spec: FilterSpec,
) -> tuple[tuple[ScenarioSpec, ...], dict[str, Any]]:
    """Load and re-derive the pool from stored labels; the pool is a pure function."""
    cache_dir = Path(cache_dir)
    _verify_manifest(cache_dir)
    if _read_json(cache_dir / "classification_config.json") != current_config:
        raise RuntimeError(
            "Outcome-aware cache identity does not match the current actor, assets, base cache, "
            "generator, geometry thresholds, or filter spec; use a new cache directory"
        )
    if _read_jsonl(cache_dir / "base_candidate_outcomes.jsonl") != list(base_outcomes):
        raise RuntimeError("Outcome-aware cache base outcomes do not match the validated base cache")
    if _read_jsonl(cache_dir / "boundary_pairs.jsonl") != list(pair_records):
        raise RuntimeError("Outcome-aware boundary pairs are not deterministically reconstructable")

    boundary_labels = _records_to_labels(_read_jsonl(cache_dir / "boundary_candidate_labels.jsonl"))
    pair_other_labels = _records_to_labels(_read_jsonl(cache_dir / "pair_other_labels.jsonl"))
    base_collision_labels = _records_to_labels(_read_jsonl(cache_dir / "base_collision_labels.jsonl"))
    _check_labels_align(boundary_labels, boundary_candidates, "boundary")

    final, audit = build_outcome_aware_pool(
        base_collisions=base_collisions,
        base_outcomes=base_outcomes,
        pair_records=pair_records,
        boundary_candidates=boundary_candidates,
        boundary_labels=boundary_labels,
        pair_other_labels=pair_other_labels,
        base_collision_labels=base_collision_labels,
        spec=spec,
    )
    records = _read_json(cache_dir / "collision_scenarios.json")
    if records != [asdict(scenario) for scenario in final]:
        raise RuntimeError("Outcome-aware final pool does not match the re-derived filter output")
    if _read_json(cache_dir / "filter_audit.json") != audit:
        raise RuntimeError("Outcome-aware filter audit does not match the re-derived decisions")
    summary = _summary(base_outcomes, pair_records, boundary_labels, pair_other_labels, final, audit)
    if _read_json(cache_dir / "classification_summary.json") != summary:
        raise RuntimeError("Outcome-aware summary does not match its evidence")
    return final, {**summary, "filter_audit": audit}


def _check_labels_align(
    labels: Sequence[CandidateLabel],
    candidates: Sequence[ScenarioSpec],
    label: str,
) -> None:
    if len(labels) != len(candidates):
        raise RuntimeError(f"{label} labels ({len(labels)}) do not cover candidates ({len(candidates)})")
    for index, (record, candidate) in enumerate(zip(labels, candidates)):
        if record.candidate_index != index or record.scenario_id != candidate.scenario_id:
            raise RuntimeError(f"{label} label misaligned at candidate {index}")


# --------------------------------------------------------------------------- #
# Drop-in resolver (mirrors ppo.hard_neighbors.resolve_training_collision_scenarios)
# --------------------------------------------------------------------------- #
def _unique_other_endpoint_scenarios(
    pair_records: Sequence[dict[str, Any]],
    base_candidates: Sequence[ScenarioSpec],
) -> tuple[ScenarioSpec, ...]:
    by_id = {scenario.scenario_id: scenario for scenario in base_candidates}
    seen: dict[str, ScenarioSpec] = {}
    for pair in pair_records:
        other_id = _other_endpoint_id(pair)
        if other_id not in by_id:
            raise RuntimeError(f"Boundary pair references unknown base scenario {other_id}")
        seen.setdefault(other_id, by_id[other_id])
    # deterministic order by scenario_id
    return tuple(seen[key] for key in sorted(seen))


def resolve_outcome_aware_collision_scenarios(
    args: Any,
    base_candidates: tuple[ScenarioSpec, ...],
    start_method: str,
) -> tuple[tuple[ScenarioSpec, ...], dict[str, Any]]:
    """Resolve the outcome-aware filtered collision pool, building the cache if absent.

    Wiring note (apply after review): in ``train_ppo.py`` dispatch to this when
    ``args.outcome_aware_hard`` is set, instead of
    ``resolve_training_collision_scenarios``. It returns the same
    ``(scenarios, info)`` contract, so ``vec_env`` / scheduler / reward are
    untouched. ``scenario.pool`` tags are preserved ("collision" for base,
    "hard_neighbor" for boundary), so ``--hard_neighbor_fraction`` still works.
    """
    spec = FilterSpec(
        mode=str(getattr(args, "outcome_aware_filter_mode", "all")),
        safe_clearance_m=float(getattr(args, "outcome_aware_safe_clearance_m", 0.10)),
        require_all_source_pairs_safe=bool(getattr(args, "outcome_aware_require_all_pairs_safe", True)),
        keep_rear_end_quota=bool(getattr(args, "outcome_aware_keep_rear_end", True)),
        drop_unsafe_base=bool(getattr(args, "outcome_aware_drop_unsafe_base", False)),
    ).validate()

    base_collisions, base_cache_hit, base_reclassified = resolve_collision_scenarios(
        args, base_candidates, start_method
    )
    base_cache_dir = Path(args.collision_cache_dir).expanduser().resolve()
    base_config = collision_classification_config(args, len(base_candidates))
    loaded_base_collisions, base_outcomes, _summary_unused = load_collision_cache_artifacts(
        base_cache_dir, base_config, base_candidates
    )
    if loaded_base_collisions != base_collisions:
        raise RuntimeError("Resolved base collision pool changed while loading its evidence")

    discovery = discover_boundary_candidates(base_candidates, base_outcomes)
    boundary_candidates = materialize_boundary_candidates(discovery.candidates, base_candidates)
    if not boundary_candidates:
        raise RuntimeError("The base cache contains no refinable collision/other boundaries")
    other_endpoints = _unique_other_endpoint_scenarios(discovery.pair_records, base_candidates)

    cache_dir = Path(args.outcome_aware_cache_dir).expanduser().resolve()
    current_config = outcome_aware_cache_config(
        args,
        base_cache_dir,
        len(base_candidates),
        len(discovery.pair_records),
        len(boundary_candidates),
        spec,
    )

    if cache_exists(cache_dir):
        final, summary = load_outcome_aware_cache(
            cache_dir, current_config, base_collisions, base_outcomes,
            discovery.pair_records, boundary_candidates, spec,
        )
        cache_hit = True
        print(
            f"Outcome-aware cache hit [{spec.mode}]: {len(final)} collision scenarios",
            flush=True,
        )
    else:
        print(
            f"Outcome-aware cache miss [{spec.mode}]: replaying "
            f"{len(boundary_candidates)} boundary + {len(other_endpoints)} other-endpoint "
            f"+ {len(base_collisions)} base-collision scenarios",
            flush=True,
        )
        boundary_labels, boundary_meta = classify_labeled_scenarios(
            args.pretrained_model_path, args.hidden_scale, args.map_name,
            args.env_workers, boundary_candidates, start_method,
        )
        other_labels, _ = classify_labeled_scenarios(
            args.pretrained_model_path, args.hidden_scale, args.map_name,
            args.env_workers, other_endpoints, start_method,
        )
        base_collision_labels, _ = classify_labeled_scenarios(
            args.pretrained_model_path, args.hidden_scale, args.map_name,
            args.env_workers, base_collisions, start_method,
        )
        _check_labels_align(boundary_labels, boundary_candidates, "boundary")
        final, audit = build_outcome_aware_pool(
            base_collisions=base_collisions,
            base_outcomes=base_outcomes,
            pair_records=discovery.pair_records,
            boundary_candidates=boundary_candidates,
            boundary_labels=boundary_labels,
            pair_other_labels=other_labels,
            base_collision_labels=base_collision_labels,
            spec=spec,
        )
        summary = _summary(
            base_outcomes, discovery.pair_records, boundary_labels,
            other_labels, final, audit,
        )
        publish_outcome_aware_cache(
            cache_dir, current_config, base_outcomes, discovery.pair_records,
            boundary_labels, other_labels, base_collision_labels, final,
            audit, summary, boundary_meta,
        )
        final, summary = load_outcome_aware_cache(
            cache_dir, current_config, base_collisions, base_outcomes,
            discovery.pair_records, boundary_candidates, spec,
        )
        cache_hit = False
        print(f"Outcome-aware cache built [{spec.mode}]: {len(final)} collision scenarios", flush=True)

    info = {
        "mode": "outcome_aware",
        "hard_neighbors": True,
        "outcome_aware": True,
        "cache_dir": str(cache_dir),
        "base_cache_dir": str(base_cache_dir),
        "base_cache_hit": base_cache_hit,
        "base_reclassified": base_reclassified,
        "base_candidate_count": len(base_candidates),
        "base_collision_count": len(base_collisions),
        "outcome_aware_cache_hit": cache_hit,
        "filter_spec": spec.to_config(),
        "boundary_pair_count": summary["boundary_pair_count"],
        "boundary_kept_mode_counts": summary["boundary_kept_mode_counts"],
        "collision_count": len(final),
    }
    return final, info

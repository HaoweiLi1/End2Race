#!/usr/bin/env python3
"""Replay saved eval episodes to validate post-pass rear-clearance sparsity.

This script does not modify training rewards. It audits pure geometry over
collision, overtake, and follow episodes, sweeps a small fixed set of phase
gates/deadbands, and writes reproducible JSON/CSV evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.postpass_reward_calculation import (
    bounded_postpass_reward,
    ego_induced_rear_closing,
    postpass_penalty_basis,
    signed_rear_longitudinal_gap,
)


VEHICLE_LENGTH_M = 0.58
VEHICLE_WIDTH_M = 0.31
PASS_MARGINS_M = (0.0, 0.05, 0.10, 0.30)
SAFE_REAR_GAPS_M = (0.30, 0.60)
REAR_CLOSING_DEADBANDS_MPS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
CLEAR_MODES = ("latched", "reactive")
PROPOSED_REWARD_WEIGHT_PER_M = 0.25
PROPOSED_MAXIMUM_STEP_PENALTY = 0.005
PROPOSED_MAXIMUM_EPISODE_PENALTY = 0.05
DEFAULT_PANELS = (
    "end2race_Austin",
    "ppo_privilege_gru_0722_long_clip020_u0030_Austin",
    "ppo_privilege_gru_0722_long45_clip020_u0045_Austin",
    "ppo_privilege_gru_0722_long45_clip020_hard_u0045_Austin",
)


class TrackProjector:
    """Vectorized nearby-segment projection onto the dense Austin raceline."""

    def __init__(self, path: Path) -> None:
        reference = np.loadtxt(path, delimiter=";", comments="#", dtype=np.float64)
        self.track_length = float(reference[-1, 0])
        if np.linalg.norm(reference[-1, 1:3] - reference[0, 1:3]) <= 1e-9:
            reference = reference[:-1]
        self.progress = reference[:, 0]
        self.points = reference[:, 1:3]
        self.tree = cKDTree(self.points)
        self.segment_vectors = np.roll(self.points, -1, axis=0) - self.points
        self.segment_norm_sq = np.einsum(
            "ij,ij->i",
            self.segment_vectors,
            self.segment_vectors,
        )
        self.segment_progress = np.concatenate(
            (
                np.diff(self.progress),
                np.asarray([self.track_length - self.progress[-1]]),
            )
        )

    def project_progress(self, query: np.ndarray) -> np.ndarray:
        points = np.asarray(query, dtype=np.float64)
        _distance, nearest = self.tree.query(points)
        candidates = np.column_stack((nearest, (nearest - 1) % len(self.points)))
        starts = self.points[candidates]
        vectors = self.segment_vectors[candidates]
        offset = points[:, None, :] - starts
        fraction = np.clip(
            np.einsum("nci,nci->nc", offset, vectors)
            / self.segment_norm_sq[candidates],
            0.0,
            1.0,
        )
        closest = starts + fraction[..., None] * vectors
        distance_sq = np.einsum(
            "nci,nci->nc",
            points[:, None, :] - closest,
            points[:, None, :] - closest,
        )
        choice = np.argmin(distance_sq, axis=1)
        row = np.arange(len(points))
        segments = candidates[row, choice]
        return (
            self.progress[segments]
            + fraction[row, choice] * self.segment_progress[segments]
        ) % self.track_length

    def relative_progress(
        self,
        ego_xy: np.ndarray,
        opponent_xy: np.ndarray,
    ) -> np.ndarray:
        ego_progress = self.project_progress(ego_xy)
        opponent_progress = self.project_progress(opponent_xy)
        raw = (
            ego_progress
            - opponent_progress
            + 0.5 * self.track_length
        ) % self.track_length - 0.5 * self.track_length
        unwrapped = np.empty_like(raw)
        unwrapped[0] = raw[0]
        deltas = (
            np.diff(raw) + 0.5 * self.track_length
        ) % self.track_length - 0.5 * self.track_length
        unwrapped[1:] = raw[0] + np.cumsum(deltas)
        return unwrapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        default=list(DEFAULT_PANELS),
        help="Evaluation directory names under eval_results",
    )
    parser.add_argument(
        "--tail-labels",
        type=Path,
        default=Path(
            "analysis_results/ppo_all_experiments_20260723/"
            "collision_episode_kinematics.csv"
        ),
        help="Existing primary tail-classifier output used only as a reference label",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results/postpass_reward_validation"),
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else root / expanded


def panel_label(directory_name: str) -> str:
    if directory_name == "end2race_Austin":
        return "BC"
    suffix = "_Austin"
    if not directory_name.endswith(suffix):
        raise ValueError(f"Panel directory must end in {suffix!r}: {directory_name}")
    return directory_name[: -len(suffix)]


def load_reference_tail_labels(path: Path) -> dict[tuple[str, str], bool]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    labels: dict[tuple[str, str], bool] = {}
    for row in rows:
        key = (row["panel"], row["scenario_id"])
        if key in labels:
            raise ValueError(f"Duplicate reference tail label: {key}")
        labels[key] = row["merge_tail_primary"] == "True"
    return labels


def validate_trace(
    path: Path,
    *,
    expected_collision: bool,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        required = {
            "time_s",
            "ego_pose",
            "opp_pose",
            "collisions",
            "action_applied",
            "terminal_post_step",
        }
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        result = {name: np.asarray(arrays[name]) for name in required}

    lengths = {len(value) for value in result.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError(f"{path}: inconsistent or too-short trace arrays")
    length = next(iter(lengths))
    if result["ego_pose"].shape != (length, 3):
        raise ValueError(f"{path}: invalid ego_pose shape")
    if result["opp_pose"].shape != (length, 3):
        raise ValueError(f"{path}: invalid opp_pose shape")
    if result["collisions"].shape != (length, 2):
        raise ValueError(f"{path}: invalid collisions shape")
    if result["action_applied"].shape != (length,):
        raise ValueError(f"{path}: invalid action_applied shape")
    if result["terminal_post_step"].shape != (length,):
        raise ValueError(f"{path}: invalid terminal_post_step shape")
    numeric = (result["time_s"], result["ego_pose"], result["opp_pose"])
    if not all(np.isfinite(value).all() for value in numeric):
        raise ValueError(f"{path}: non-finite trace geometry")
    time_s = np.asarray(result["time_s"], dtype=np.float64)
    if time_s.shape != (length,) or np.any(np.diff(time_s) <= 0.0):
        raise ValueError(f"{path}: time_s must be finite and strictly increasing")
    terminal = np.asarray(result["terminal_post_step"], dtype=bool)
    actions = np.asarray(result["action_applied"], dtype=bool)
    if (
        np.count_nonzero(terminal) != 1
        or not terminal[-1]
        or np.count_nonzero(actions) != length - 1
        or actions[-1]
    ):
        raise ValueError(f"{path}: invalid terminal post-step contract")
    if bool(np.asarray(result["collisions"], dtype=bool)[-1, 0]) != expected_collision:
        raise ValueError(f"{path}: collision marker disagrees with results_multi.json")
    return result


def vectorized_signed_rear_gap(
    ego_pose: np.ndarray,
    opponent_pose: np.ndarray,
) -> np.ndarray:
    """Closed-form equivalent of signed_rear_longitudinal_gap for equal OBBs."""

    relative = opponent_pose[:, :2] - ego_pose[:, :2]
    cosine = np.cos(ego_pose[:, 2])
    sine = np.sin(ego_pose[:, 2])
    opponent_body_x = relative[:, 0] * cosine + relative[:, 1] * sine
    heading_difference = opponent_pose[:, 2] - ego_pose[:, 2]
    opponent_half_extent = (
        0.5 * VEHICLE_LENGTH_M * np.abs(np.cos(heading_difference))
        + 0.5 * VEHICLE_WIDTH_M * np.abs(np.sin(heading_difference))
    )
    return -0.5 * VEHICLE_LENGTH_M - (
        opponent_body_x + opponent_half_extent
    )


def vectorized_signed_rear_gap_from_vertices(
    ego_pose: np.ndarray,
    opponent_pose: np.ndarray,
) -> np.ndarray:
    """Independent all-row OBB-vertex implementation for audit cross-checking."""

    local_vertices = np.asarray(
        (
            (-0.5 * VEHICLE_LENGTH_M, 0.5 * VEHICLE_WIDTH_M),
            (-0.5 * VEHICLE_LENGTH_M, -0.5 * VEHICLE_WIDTH_M),
            (0.5 * VEHICLE_LENGTH_M, -0.5 * VEHICLE_WIDTH_M),
            (0.5 * VEHICLE_LENGTH_M, 0.5 * VEHICLE_WIDTH_M),
        ),
        dtype=np.float64,
    )

    def world_vertices(poses: np.ndarray) -> np.ndarray:
        cosine = np.cos(poses[:, 2])
        sine = np.sin(poses[:, 2])
        rotation = np.stack(
            (
                np.stack((cosine, -sine), axis=1),
                np.stack((sine, cosine), axis=1),
            ),
            axis=1,
        )
        return (
            np.einsum("nij,vj->nvi", rotation, local_vertices)
            + poses[:, None, :2]
        )

    ego_vertices = world_vertices(ego_pose)
    opponent_vertices = world_vertices(opponent_pose)
    ego_axis = np.stack(
        (np.cos(ego_pose[:, 2]), np.sin(ego_pose[:, 2])),
        axis=1,
    )
    ego_projection = np.einsum("nvi,ni->nv", ego_vertices, ego_axis)
    opponent_projection = np.einsum(
        "nvi,ni->nv",
        opponent_vertices,
        ego_axis,
    )
    return np.min(ego_projection, axis=1) - np.max(
        opponent_projection,
        axis=1,
    )


def first_crossing(relative: np.ndarray, margin_m: float) -> int | None:
    crossings = np.flatnonzero(
        (relative[:-1] <= margin_m) & (relative[1:] > margin_m)
    )
    return int(crossings[0] + 1) if crossings.size else None


def replay_episode_geometry(
    trace: dict[str, np.ndarray],
    projector: TrackProjector,
) -> dict[str, Any]:
    ego_pose = np.asarray(trace["ego_pose"], dtype=np.float64)
    opponent_pose = np.asarray(trace["opp_pose"], dtype=np.float64)
    relative = projector.relative_progress(ego_pose[:, :2], opponent_pose[:, :2])
    signed_gap = vectorized_signed_rear_gap(ego_pose, opponent_pose)
    vertex_signed_gap = vectorized_signed_rear_gap_from_vertices(
        ego_pose,
        opponent_pose,
    )
    if not np.allclose(
        signed_gap,
        vertex_signed_gap,
        rtol=0.0,
        atol=1e-12,
    ):
        maximum_error = float(np.max(np.abs(signed_gap - vertex_signed_gap)))
        raise RuntimeError(
            "Closed-form signed rear gap disagrees with all-row OBB vertices: "
            f"max error {maximum_error}"
        )
    opponent_collision_latched = np.maximum.accumulate(
        np.asarray(trace["collisions"], dtype=bool)[:, 1]
    )

    # Cross-check the fast formula against the production-pure function at
    # beginning, midpoint, and terminal rows of every episode.
    for index in sorted({0, len(ego_pose) // 2, len(ego_pose) - 1}):
        exact = signed_rear_longitudinal_gap(
            ego_pose[index],
            opponent_pose[index],
            VEHICLE_LENGTH_M,
            VEHICLE_WIDTH_M,
        )
        if abs(exact - float(signed_gap[index])) > 1e-10:
            raise RuntimeError("Vectorized signed rear gap disagrees with pure geometry")

    earliest_entry = first_crossing(relative, min(PASS_MARGINS_M))
    closing = np.zeros(len(ego_pose), dtype=np.float64)
    current_clearance = np.full(len(ego_pose), np.nan, dtype=np.float64)
    if earliest_entry is not None:
        candidates = np.flatnonzero(
            (np.arange(len(ego_pose)) >= earliest_entry)
            & (signed_gap < max(SAFE_REAR_GAPS_M))
        )
        for index in candidates:
            if index == 0:
                continue
            result = ego_induced_rear_closing(
                ego_pose[index - 1],
                ego_pose[index],
                opponent_pose[index],
                VEHICLE_LENGTH_M,
                VEHICLE_WIDTH_M,
            )
            closing[index] = result.closing_m
            current_clearance[index] = result.current_clearance_m
    return {
        "relative_progress_m": relative,
        "signed_rear_gap_m": signed_gap,
        "ego_induced_rear_closing_m": closing,
        "rear_half_clearance_m": current_clearance,
        "opponent_collision_latched": opponent_collision_latched,
    }


def setting_episode_result(
    geometry: dict[str, Any],
    time_s: np.ndarray,
    *,
    pass_margin_m: float,
    safe_rear_gap_m: float,
    closing_deadband_mps: float,
    clear_mode: str,
) -> dict[str, Any]:
    relative = geometry["relative_progress_m"]
    signed_gap = geometry["signed_rear_gap_m"]
    closing = geometry["ego_induced_rear_closing_m"]
    opponent_collision_latched = geometry["opponent_collision_latched"]
    entry = first_crossing(relative, pass_margin_m)
    active = np.zeros(len(relative), dtype=bool)
    if entry is not None:
        entered = np.arange(len(relative)) >= entry
        unsafe = signed_gap < safe_rear_gap_m
        active = entered & unsafe
        if clear_mode == "latched":
            clear_indices = np.flatnonzero(entered & ~unsafe)
            if clear_indices.size:
                active[int(clear_indices[0]) :] = False
        elif clear_mode != "reactive":
            raise ValueError(f"Unknown clear mode: {clear_mode}")
        active &= ~opponent_collision_latched

    transition_dt_s = np.zeros(len(relative), dtype=np.float64)
    transition_dt_s[1:] = np.diff(np.asarray(time_s, dtype=np.float64))
    trigger = active & (closing > closing_deadband_mps * transition_dt_s)
    preterminal_trigger = trigger.copy()
    preterminal_trigger[-1] = False
    basis = np.zeros(len(relative), dtype=np.float64)
    proposed_reward = np.zeros(len(relative), dtype=np.float64)
    episode_penalty_used = 0.0
    for index in np.flatnonzero(trigger):
        basis[index] = postpass_penalty_basis(
            float(closing[index]),
            float(signed_gap[index]),
            safe_rear_gap_m,
            active=True,
        )
        proposed_reward[index] = bounded_postpass_reward(
            float(basis[index]),
            PROPOSED_REWARD_WEIGHT_PER_M,
            PROPOSED_MAXIMUM_STEP_PENALTY,
            episode_penalty_used,
            PROPOSED_MAXIMUM_EPISODE_PENALTY,
        )
        episode_penalty_used = min(
            PROPOSED_MAXIMUM_EPISODE_PENALTY,
            episode_penalty_used - float(proposed_reward[index]),
        )
    return {
        "pass_detected": entry is not None,
        "pass_index": entry,
        "active_steps": int(np.count_nonzero(active)),
        "trigger_steps": int(np.count_nonzero(trigger)),
        "preterminal_trigger_steps": int(np.count_nonzero(preterminal_trigger)),
        "triggered": bool(np.any(trigger)),
        "preterminal_triggered": bool(np.any(preterminal_trigger)),
        "basis_sum_m": float(np.sum(basis)),
        "basis_max_step_m": float(np.max(basis, initial=0.0)),
        "proposed_reward_sum": float(np.sum(proposed_reward)),
        "proposed_penalty_sum": float(-np.sum(proposed_reward)),
        "proposed_penalty_nonzero_steps": int(
            np.count_nonzero(proposed_reward)
        ),
        "proposed_step_cap_hit": bool(
            np.any(
                basis * PROPOSED_REWARD_WEIGHT_PER_M
                > PROPOSED_MAXIMUM_STEP_PENALTY
            )
        ),
        "proposed_episode_cap_hit": bool(
            episode_penalty_used
            >= PROPOSED_MAXIMUM_EPISODE_PENALTY - 1e-12
        ),
        "minimum_signed_rear_gap_m": (
            float(np.min(signed_gap[entry:])) if entry is not None else None
        ),
        "first_trigger_index": (
            int(np.flatnonzero(trigger)[0]) if np.any(trigger) else None
        ),
        "last_trigger_index": (
            int(np.flatnonzero(trigger)[-1]) if np.any(trigger) else None
        ),
        "first_trigger_time_s": (
            float(time_s[np.flatnonzero(trigger)[0]]) if np.any(trigger) else None
        ),
        "first_trigger_lead_to_terminal_s": (
            float(time_s[-1] - time_s[np.flatnonzero(trigger)[0]])
            if np.any(trigger)
            else None
        ),
    }


def load_panel(
    root: Path,
    directory_name: str,
    projector: TrackProjector,
    tail_labels: dict[tuple[str, str], bool],
) -> list[dict[str, Any]]:
    label = panel_label(directory_name)
    multiagents = root / "eval_results" / directory_name / "multiagents"
    result_path = multiagents / "results_multi.json"
    document = json.loads(result_path.read_text(encoding="utf-8"))
    episodes = document.get("episodes")
    final = document.get("final")
    if not isinstance(episodes, dict) or not isinstance(final, dict):
        raise ValueError(f"{result_path}: missing final/episodes objects")
    if len(episodes) != 600 or int(final.get("total_episodes", -1)) != 600:
        raise ValueError(f"{result_path}: expected a complete 600-episode panel")
    if int(final.get("error_count", -1)) != 0:
        raise ValueError(f"{result_path}: evaluation contains errors")
    trace_paths = sorted((multiagents / "traces").glob("*.npz"))
    expected_trace_stems = set(episodes)
    observed_trace_stems = {path.stem for path in trace_paths}
    if observed_trace_stems != expected_trace_stems:
        raise ValueError(
            f"{result_path}: trace/result episode-key sets do not match"
        )
    scenario_ids = [str(episode["scenario_id"]) for episode in episodes.values()]
    if len(set(scenario_ids)) != 600:
        raise ValueError(f"{result_path}: scenario IDs must be unique within panel")

    output = []
    for episode_key, episode in sorted(episodes.items()):
        raw_outcome = str(episode["outcome"])
        outcome = "collision" if raw_outcome == "ego_collision" else raw_outcome
        if outcome not in {"collision", "overtake", "follow"}:
            raise ValueError(f"{result_path}: unexpected outcome {outcome!r}")
        collision = bool(episode["ego_collision_occurred"])
        if collision != (outcome == "collision"):
            raise ValueError(f"{result_path}: collision/outcome mismatch for {episode_key}")
        trace_path = multiagents / "traces" / f"{episode_key}.npz"
        trace = validate_trace(trace_path, expected_collision=collision)
        transition_count = len(trace["time_s"]) - 1
        if int(episode["steps"]) != transition_count:
            raise ValueError(f"{trace_path}: trace length disagrees with steps")
        if not math.isclose(
            float(episode["simulation_time_s"]),
            float(trace["time_s"][-1]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"{trace_path}: terminal time disagrees with simulation_time_s"
            )
        ego_collision_indices = np.flatnonzero(
            np.asarray(trace["collisions"], dtype=bool)[:, 0]
        )
        reported_collision_time = episode.get("ego_collision_time_s")
        if collision:
            if reported_collision_time is None or ego_collision_indices.size == 0:
                raise ValueError(f"{trace_path}: missing ego collision time/row")
            observed_collision_time = float(
                trace["time_s"][int(ego_collision_indices[0])]
            )
            if not math.isclose(
                float(reported_collision_time),
                observed_collision_time,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{trace_path}: collision time disagrees with trace"
                )
        elif reported_collision_time is not None or ego_collision_indices.size:
            raise ValueError(f"{trace_path}: non-collision has collision time/row")
        observed_opponent_collision = bool(
            np.any(np.asarray(trace["collisions"], dtype=bool)[:, 1])
        )
        if observed_opponent_collision != bool(
            episode["opp_collision_occurred"]
        ):
            raise ValueError(
                f"{trace_path}: opponent collision marker disagrees with "
                "results_multi.json"
            )
        geometry = replay_episode_geometry(trace, projector)
        if not math.isclose(
            float(episode["final_relative_position_m"]),
            float(geometry["relative_progress_m"][-1]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"{trace_path}: replayed final relative position disagrees "
                "with results_multi.json"
            )
        scenario_id = str(episode["scenario_id"])
        reference_primary_tail = tail_labels.get((label, scenario_id), False)
        if reference_primary_tail and not collision:
            raise ValueError(
                f"{trace_path}: primary-tail reference labels a non-collision"
            )
        output.append(
            {
                "panel": label,
                "panel_directory": directory_name,
                "scenario_id": scenario_id,
                "episode_key": episode_key,
                "outcome": outcome,
                "ego_collision": collision,
                "reference_primary_tail": reference_primary_tail,
                "trace_steps": transition_count,
                "collision_time_s": episode.get("ego_collision_time_s"),
                "time_s": np.asarray(trace["time_s"], dtype=np.float64),
                "geometry": geometry,
            }
        )
    observed = Counter(row["outcome"] for row in output)
    expected = {
        "collision": int(final["collision_count"]),
        "overtake": int(final["overtaking_count"]),
        "follow": int(final["following_count"]),
    }
    if observed != Counter(expected):
        raise ValueError(
            f"{result_path}: replay outcomes {dict(observed)} != final {expected}"
        )
    return output


def quantile(values: list[float], probability: float) -> float | None:
    return (
        float(np.quantile(np.asarray(values, dtype=np.float64), probability))
        if values
        else None
    )


def setting_key(
    pass_margin_m: float,
    safe_rear_gap_m: float,
    closing_deadband_mps: float,
    clear_mode: str,
) -> str:
    return (
        f"pass{pass_margin_m:.2f}_safe{safe_rear_gap_m:.2f}_"
        f"dead{closing_deadband_mps:.2f}mps_{clear_mode}"
    )


def evaluate_settings(
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    setting_rows = []
    for pass_margin in PASS_MARGINS_M:
        for safe_gap in SAFE_REAR_GAPS_M:
            for deadband_mps in REAR_CLOSING_DEADBANDS_MPS:
                for clear_mode in CLEAR_MODES:
                    key = setting_key(
                        pass_margin,
                        safe_gap,
                        deadband_mps,
                        clear_mode,
                    )
                    evaluated = []
                    for episode in episodes:
                        result = setting_episode_result(
                            episode["geometry"],
                            episode["time_s"],
                            pass_margin_m=pass_margin,
                            safe_rear_gap_m=safe_gap,
                            closing_deadband_mps=deadband_mps,
                            clear_mode=clear_mode,
                        )
                        row = {
                            "setting": key,
                            "pass_margin_m": pass_margin,
                            "safe_rear_gap_m": safe_gap,
                            "closing_deadband_mps": deadband_mps,
                            "clear_mode": clear_mode,
                            "panel": episode["panel"],
                            "scenario_id": episode["scenario_id"],
                            "episode_key": episode["episode_key"],
                            "outcome": episode["outcome"],
                            "ego_collision": episode["ego_collision"],
                            "reference_primary_tail": episode[
                                "reference_primary_tail"
                            ],
                            "trace_steps": episode["trace_steps"],
                            **result,
                        }
                        evaluated.append(row)

                    total_steps = sum(row["trace_steps"] for row in evaluated)
                    active_steps = sum(row["active_steps"] for row in evaluated)
                    trigger_steps = sum(row["trigger_steps"] for row in evaluated)
                    tails = [
                        row for row in evaluated if row["reference_primary_tail"]
                    ]
                    by_outcome = {
                        outcome: [
                            row for row in evaluated if row["outcome"] == outcome
                        ]
                        for outcome in ("collision", "overtake", "follow")
                    }
                    basis = [
                        row["basis_sum_m"]
                        for row in evaluated
                        if row["triggered"]
                    ]
                    proposed_penalties = [
                        row["proposed_penalty_sum"]
                        for row in evaluated
                        if row["triggered"]
                    ]
                    setting_rows.append(
                        {
                            "setting": key,
                            "pass_margin_m": pass_margin,
                            "safe_rear_gap_m": safe_gap,
                            "closing_deadband_mps": deadband_mps,
                            "clear_mode": clear_mode,
                            "episode_count": len(evaluated),
                            "panel_count": len(
                                {row["panel"] for row in evaluated}
                            ),
                            "total_transition_steps": total_steps,
                            "active_steps": active_steps,
                            "trigger_steps": trigger_steps,
                            "active_step_fraction": active_steps / total_steps,
                            "trigger_step_fraction": trigger_steps / total_steps,
                            "trigger_episode_count": sum(
                                row["triggered"] for row in evaluated
                            ),
                            "trigger_episode_fraction": sum(
                                row["triggered"] for row in evaluated
                            )
                            / len(evaluated),
                            "reference_primary_tail_count": len(tails),
                            "reference_primary_tail_captured": sum(
                                row["preterminal_triggered"] for row in tails
                            ),
                            "reference_primary_tail_capture_rate": (
                                sum(row["preterminal_triggered"] for row in tails)
                                / len(tails)
                                if tails
                                else None
                            ),
                            **{
                                f"{outcome}_episode_count": len(rows)
                                for outcome, rows in by_outcome.items()
                            },
                            **{
                                f"{outcome}_trigger_episode_count": sum(
                                    row["triggered"] for row in rows
                                )
                                for outcome, rows in by_outcome.items()
                            },
                            **{
                                f"{outcome}_trigger_episode_rate": (
                                    sum(row["triggered"] for row in rows) / len(rows)
                                    if rows
                                    else None
                                )
                                for outcome, rows in by_outcome.items()
                            },
                            "triggered_basis_sum_m": float(sum(basis)),
                            "triggered_basis_episode_median_m": quantile(
                                basis,
                                0.50,
                            ),
                            "triggered_basis_episode_q90_m": quantile(
                                basis,
                                0.90,
                            ),
                            "proposed_penalty_episode_median": quantile(
                                proposed_penalties,
                                0.50,
                            ),
                            "proposed_penalty_episode_q90": quantile(
                                proposed_penalties,
                                0.90,
                            ),
                            "proposed_penalty_sum": float(
                                sum(proposed_penalties)
                            ),
                            "proposed_step_cap_episode_count": sum(
                                row["proposed_step_cap_hit"]
                                for row in evaluated
                            ),
                            "proposed_episode_cap_episode_count": sum(
                                row["proposed_episode_cap_hit"]
                                for row in evaluated
                            ),
                            **{
                                f"{outcome}_proposed_penalty_median": quantile(
                                    [
                                        row["proposed_penalty_sum"]
                                        for row in rows
                                        if row["triggered"]
                                    ],
                                    0.50,
                                )
                                for outcome, rows in by_outcome.items()
                            },
                            **{
                                f"{outcome}_proposed_penalty_q90": quantile(
                                    [
                                        row["proposed_penalty_sum"]
                                        for row in rows
                                        if row["triggered"]
                                    ],
                                    0.90,
                                )
                                for outcome, rows in by_outcome.items()
                            },
                        }
                    )
    return setting_rows


def evaluate_candidate_episodes(
    episodes: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        result = setting_episode_result(
            episode["geometry"],
            episode["time_s"],
            pass_margin_m=float(candidate["pass_margin_m"]),
            safe_rear_gap_m=float(candidate["safe_rear_gap_m"]),
            closing_deadband_mps=float(candidate["closing_deadband_mps"]),
            clear_mode=str(candidate["clear_mode"]),
        )
        rows.append(
            {
                "setting": candidate["setting"],
                "pass_margin_m": candidate["pass_margin_m"],
                "safe_rear_gap_m": candidate["safe_rear_gap_m"],
                "closing_deadband_mps": candidate[
                    "closing_deadband_mps"
                ],
                "clear_mode": candidate["clear_mode"],
                "panel": episode["panel"],
                "scenario_id": episode["scenario_id"],
                "episode_key": episode["episode_key"],
                "outcome": episode["outcome"],
                "ego_collision": episode["ego_collision"],
                "reference_primary_tail": episode[
                    "reference_primary_tail"
                ],
                "trace_steps": episode["trace_steps"],
                **result,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def choose_candidate(settings: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer complete reference-tail capture, then the sparsest trigger."""

    best_capture = max(
        float(row["reference_primary_tail_capture_rate"] or 0.0)
        for row in settings
    )
    candidates = [
        row
        for row in settings
        if math.isclose(
            float(row["reference_primary_tail_capture_rate"] or 0.0),
            best_capture,
        )
    ]
    return min(
        candidates,
        key=lambda row: (
            row["follow_trigger_episode_count"],
            row["proposed_episode_cap_episode_count"],
            row["proposed_penalty_sum"],
            row["trigger_step_fraction"],
            row["overtake_trigger_episode_rate"],
            row["pass_margin_m"],
            -row["safe_rear_gap_m"],
            -row["closing_deadband_mps"],
            row["clear_mode"],
        ),
    )


def candidate_breakdown(
    candidate: dict[str, Any],
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        row for row in episode_rows if row["setting"] == candidate["setting"]
    ]
    groups = {
        "reference_primary_tail": [
            row for row in selected if row["reference_primary_tail"]
        ],
        "non_reference_collision": [
            row
            for row in selected
            if row["outcome"] == "collision"
            and not row["reference_primary_tail"]
        ],
        "successful_overtake": [
            row for row in selected if row["outcome"] == "overtake"
        ],
        "final_follow": [
            row for row in selected if row["outcome"] == "follow"
        ],
    }
    breakdown = {}
    for name, rows in groups.items():
        triggered = [row for row in rows if row["triggered"]]
        penalties = [row["proposed_penalty_sum"] for row in triggered]
        breakdown[name] = {
            "episode_count": len(rows),
            "trigger_episode_count": len(triggered),
            "trigger_episode_rate": len(triggered) / len(rows) if rows else None,
            "preterminal_trigger_episode_count": sum(
                row["preterminal_triggered"] for row in rows
            ),
            "step_cap_episode_count": sum(
                row["proposed_step_cap_hit"] for row in rows
            ),
            "episode_cap_episode_count": sum(
                row["proposed_episode_cap_hit"] for row in rows
            ),
            "penalty_q10": quantile(penalties, 0.10),
            "penalty_median": quantile(penalties, 0.50),
            "penalty_q90": quantile(penalties, 0.90),
            "penalty_sum": float(sum(penalties)),
        }
    tail_leads = [
        row["first_trigger_lead_to_terminal_s"]
        for row in groups["reference_primary_tail"]
        if row["first_trigger_lead_to_terminal_s"] is not None
    ]
    breakdown["reference_primary_tail"]["first_signal_lead_s_q10"] = (
        quantile(tail_leads, 0.10)
    )
    breakdown["reference_primary_tail"]["first_signal_lead_s_median"] = (
        quantile(tail_leads, 0.50)
    )
    breakdown["reference_primary_tail"]["first_signal_lead_s_q90"] = (
        quantile(tail_leads, 0.90)
    )
    breakdown["triggered_final_follow_episodes"] = [
        {
            "panel": row["panel"],
            "scenario_id": row["scenario_id"],
            "penalty": row["proposed_penalty_sum"],
        }
        for row in groups["final_follow"]
        if row["triggered"]
    ]
    return breakdown


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    tail_path = resolve(root, args.tail_labels)
    output_dir = resolve(root, args.output_dir)
    tail_labels = load_reference_tail_labels(tail_path)
    projector = TrackProjector(
        root / "f1tenth_racetracks" / "Austin" / "raceline1.csv"
    )

    episodes: list[dict[str, Any]] = []
    for panel in args.panels:
        print(f"Replaying {panel}", flush=True)
        episodes.extend(load_panel(root, panel, projector, tail_labels))

    settings = evaluate_settings(episodes)
    candidate = choose_candidate(settings)
    episode_rows = evaluate_candidate_episodes(episodes, candidate)
    breakdown = candidate_breakdown(candidate, episode_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "setting_sweep.csv", settings)
    write_csv(output_dir / "episode_setting_results.csv", episode_rows)
    panel_counts = {
        panel: dict(Counter(row["outcome"] for row in episodes if row["panel"] == panel))
        for panel in sorted({row["panel"] for row in episodes})
    }
    summary = {
        "contract": {
            "vehicle_length_m": VEHICLE_LENGTH_M,
            "vehicle_width_m": VEHICLE_WIDTH_M,
            "pass_margins_m": list(PASS_MARGINS_M),
            "safe_rear_gaps_m": list(SAFE_REAR_GAPS_M),
            "rear_closing_deadbands_mps": list(
                REAR_CLOSING_DEADBANDS_MPS
            ),
            "clear_modes": list(CLEAR_MODES),
            "proposed_reward_weight_per_m": PROPOSED_REWARD_WEIGHT_PER_M,
            "proposed_maximum_step_penalty": (
                PROPOSED_MAXIMUM_STEP_PENALTY
            ),
            "proposed_maximum_episode_penalty": (
                PROPOSED_MAXIMUM_EPISODE_PENALTY
            ),
            "tail_reference": str(tail_path.relative_to(root)),
            "tail_reference_note": (
                "Reference labels are the existing primary heuristic, not "
                "ground-truth contact-point annotations."
            ),
            "episode_setting_results_scope": (
                "Only the selected candidate's 2,400 episode rows; all "
                "setting-level aggregates are in setting_sweep.csv."
            ),
        },
        "data_quality": {
            "panels": list(args.panels),
            "episode_count": len(episodes),
            "evaluated_setting_count": len(settings),
            "unique_panel_episode_count": len(
                {(row["panel"], row["episode_key"]) for row in episodes}
            ),
            "unique_panel_scenario_count": len(
                {(row["panel"], row["scenario_id"]) for row in episodes}
            ),
            "panel_outcome_counts": panel_counts,
            "all_traces_numeric_aligned_terminal_v2": True,
            "all_trace_collision_markers_match_results": True,
            "all_trace_opponent_collision_markers_match_results": True,
            "all_trace_file_sets_match_result_keys": True,
            "all_panel_scenario_ids_unique": True,
            "all_trace_steps_and_times_match_results": True,
            "all_collision_times_match_trace": True,
            "all_replayed_final_relative_positions_match_results": True,
            "all_row_signed_gap_formula_matches_obb_vertices": True,
            "transition_dt_s_min": min(
                float(np.min(np.diff(row["time_s"]))) for row in episodes
            ),
            "transition_dt_s_max": max(
                float(np.max(np.diff(row["time_s"]))) for row in episodes
            ),
        },
        "selection_rule": (
            "maximum existing-primary-tail episode capture; then zero/minimum "
            "follow triggers; then minimum capped-episode count; then minimum "
            "total bounded penalty; then minimum trigger-step fraction; then "
            "minimum successful-overtake trigger rate"
        ),
        "candidate_sparse_setting": candidate,
        "candidate_episode_breakdown": breakdown,
    }
    temporary = output_dir / ".summary.json.tmp"
    temporary.write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_dir / "summary.json")
    print(
        "Candidate sparse setting: "
        f"{candidate['setting']}, tail capture "
        f"{candidate['reference_primary_tail_captured']}/"
        f"{candidate['reference_primary_tail_count']}, trigger steps "
        f"{candidate['trigger_step_fraction']:.6%}, follow triggers "
        f"{candidate['follow_trigger_episode_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate the isolated post-pass reward and its PPO-loss consequences.

The default ``unit`` mode is intentionally tiny. ``saved-episodes`` reads
existing evaluation JSON/NPZ files only and refuses to start when a competing
training, evaluation, simulator, or post-pass process is visible.
"""

from __future__ import annotations

import os

# Bound numerical libraries before importing NumPy/SciPy.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import sys
from typing import Any, Iterator
import zipfile

import numpy as np

from shadow_contract import (
    PostpassState,
    RewardConfig,
    bounded_negative_reward,
    clipped_ppo_policy_loss,
    ego_induced_rear_closing,
    fixed_prediction_value_loss_delta,
    gae_delta_from_reward_delta,
    masked_follow_teacher_huber_loss,
    mean_squared_value_loss,
    normalized_advantages,
    oriented_rectangle_vertices,
    postpass_reward_step,
    rear_half_clearance,
    rectangle_clearance,
    signed_rear_longitudinal_gap,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "outputs"
DEFAULT_PANELS = (
    "end2race_Austin",
    "ppo_privilege_gru_0722_long_clip020_u0030_Austin",
    "ppo_privilege_gru_0722_long45_clip020_u0045_Austin",
    "ppo_privilege_gru_0722_long45_clip020_hard_u0045_Austin",
)
BLOCKING_COMMAND_FRAGMENTS = (
    "run.sh",
    "train_ppo.py",
    "eval_multiagent.py",
    "sumo ",
    "/sumo",
    "validate_postpass_reward.py",
    "outcome_aware_hard",
    "select_critic_lr_candidate.py",
)
GAMMA = 0.999
GAE_LAMBDA = 0.995
VALUE_LOSS_COEFFICIENT = 0.5
ACCEPT_MIN_TAIL_CAPTURE = 0.90
ACCEPT_MAX_OVERTAKE_TRIGGER = 0.20
ACCEPT_MAX_FOLLOW_TRIGGER = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("unit", "saved-episodes", "all"),
        default="unit",
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panels", nargs="+", default=list(DEFAULT_PANELS))
    parser.add_argument(
        "--tail-labels",
        type=Path,
        default=Path(
            "analysis_results/ppo_all_experiments_20260723/"
            "collision_episode_kinematics.csv"
        ),
    )
    parser.add_argument(
        "--candidate-module",
        type=Path,
        default=None,
        help="Optional pure candidate module for independent geometry cross-check",
    )
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            _json_ready(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def process_visibility_receipt() -> dict[str, Any]:
    """Read /proc and report whether the host process view is trustworthy."""

    blockers = []
    own_pid = os.getpid()
    process_entries = [
        entry
        for entry in Path("/proc").iterdir()
        if entry.name.isdigit()
    ]
    readable_commands = 0
    for entry in process_entries:
        if int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8",
                errors="replace",
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not command:
            continue
        readable_commands += 1
        own_script_paths = (
            str(Path(__file__).resolve()),
            str(HERE / "live_episode_probe.py"),
        )
        # Parent shells launched by this validator contain its absolute path.
        if any(path in command for path in own_script_paths):
            continue
        named_blocker = next(
            (
                fragment
                for fragment in BLOCKING_COMMAND_FRAGMENTS
                if fragment in command
            ),
            None,
        )
        workspace_python = False
        try:
            process_cwd = (entry / "cwd").resolve(strict=True)
            workspace_python = (
                (
                    process_cwd == PROJECT_ROOT
                    or PROJECT_ROOT in process_cwd.parents
                )
                and "python" in command.lower()
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
        if named_blocker is not None or workspace_python:
            blockers.append(
                {
                    "pid": int(entry.name),
                    "reason": (
                        f"named_fragment:{named_blocker}"
                        if named_blocker is not None
                        else "workspace_python_process"
                    ),
                    "command": command.strip(),
                }
            )
    return {
        "numeric_proc_entry_count": len(process_entries),
        "readable_nonempty_cmdline_count": readable_commands,
        "minimum_required_proc_entries": 16,
        "minimum_required_readable_cmdlines": 8,
        "host_visibility_sufficient": (
            len(process_entries) >= 16 and readable_commands >= 8
        ),
        "blockers": sorted(blockers, key=lambda row: row["pid"]),
    }


def visible_blockers() -> list[dict[str, Any]]:
    return process_visibility_receipt()["blockers"]


def require_host_process_idle() -> dict[str, Any]:
    receipt = process_visibility_receipt()
    if not receipt["host_visibility_sufficient"]:
        raise RuntimeError(
            "Refusing heavy validation because /proc does not expose the host "
            f"process table: {json.dumps(receipt, ensure_ascii=False)}"
        )
    if receipt["blockers"]:
        raise RuntimeError(
            "Refusing heavy validation while competing processes are visible: "
            + json.dumps(receipt["blockers"], ensure_ascii=False)
        )
    return receipt


@contextmanager
def exclusive_output_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".shadow_validation.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError(f"Another shadow validation owns {lock_path}") from error
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _assert_close(actual: Any, expected: Any, *, tolerance: float = 1e-10) -> None:
    if not np.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        raise AssertionError(f"{actual!r} != {expected!r}")


def _rotated_pose(pose: np.ndarray, angle: float, translation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
        dtype=np.float64,
    )
    output = np.asarray(pose, dtype=np.float64).copy()
    output[:2] = rotation @ output[:2] + translation
    output[2] += angle
    return output


def run_unit_tests(candidate_module: Path | None) -> dict[str, Any]:
    passed: list[str] = []

    ego = np.asarray((0.0, 0.0, 0.0))
    opponent_behind = np.asarray((-2.0, 0.0, 0.0))
    _assert_close(
        signed_rear_longitudinal_gap(ego, opponent_behind, 0.58, 0.31),
        1.42,
    )
    _assert_close(rear_half_clearance(ego, opponent_behind, 0.58, 0.31), 1.42)
    passed.append("known_axis_aligned_gap")

    overlapping = oriented_rectangle_vertices(ego, 0.58, 0.31)
    _assert_close(rectangle_clearance(overlapping, overlapping), 0.0)
    passed.append("overlap_clearance_zero")

    angle = 0.73
    translation = np.asarray((4.2, -1.7))
    rotated_ego = _rotated_pose(ego, angle, translation)
    rotated_opponent = _rotated_pose(opponent_behind, angle, translation)
    _assert_close(
        signed_rear_longitudinal_gap(
            rotated_ego,
            rotated_opponent,
            0.58,
            0.31,
        ),
        1.42,
    )
    _assert_close(
        rear_half_clearance(rotated_ego, rotated_opponent, 0.58, 0.31),
        1.42,
    )
    passed.append("rigid_transform_invariance")

    closing = ego_induced_rear_closing(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((-0.10, 0.0, 0.0)),
        np.asarray((-1.0, 0.0, 0.0)),
        0.58,
        0.31,
    )
    _assert_close(closing.closing_m, 0.10)
    opening = ego_induced_rear_closing(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.10, 0.0, 0.0)),
        np.asarray((-1.0, 0.0, 0.0)),
        0.58,
        0.31,
    )
    _assert_close(opening.closing_m, 0.0)
    passed.append("ego_induced_closing_direction")

    state = PostpassState()
    config = RewardConfig(
        activation_clearance_m=0.20,
        maximum_ttc_s=0.75,
    )
    triggered = postpass_reward_step(
        previous_relative_progress_m=0.0,
        current_relative_progress_m=0.10,
        previous_ego_pose=np.asarray((0.0, 0.60, 0.0)),
        current_ego_pose=np.asarray((0.0, 0.40, 0.0)),
        current_opponent_pose=np.asarray((0.0, 0.0, 0.0)),
        opponent_collision_latched=False,
        transition_dt_s=0.10,
        config=config,
        state=state,
    )
    if not triggered.entered or not triggered.phase_active or not triggered.triggered:
        raise AssertionError("Expected a close post-pass transition to trigger")
    if not -config.maximum_step_penalty <= triggered.reward < 0.0:
        raise AssertionError("Triggered reward has the wrong sign or cap")
    passed.append("postpass_trigger_and_sign")

    inactive = postpass_reward_step(
        previous_relative_progress_m=-0.20,
        current_relative_progress_m=-0.10,
        previous_ego_pose=np.asarray((0.0, 0.60, 0.0)),
        current_ego_pose=np.asarray((0.0, 0.40, 0.0)),
        current_opponent_pose=np.asarray((0.0, 0.0, 0.0)),
        opponent_collision_latched=False,
        transition_dt_s=0.10,
        config=config,
        state=PostpassState(),
    )
    if inactive.phase_active or inactive.triggered or inactive.reward != 0.0:
        raise AssertionError("Pre-pass transition must remain inactive")
    passed.append("prepass_zero")

    collision_suppressed = postpass_reward_step(
        previous_relative_progress_m=0.0,
        current_relative_progress_m=0.10,
        previous_ego_pose=np.asarray((0.0, 0.60, 0.0)),
        current_ego_pose=np.asarray((0.0, 0.40, 0.0)),
        current_opponent_pose=np.asarray((0.0, 0.0, 0.0)),
        opponent_collision_latched=True,
        transition_dt_s=0.10,
        config=config,
        state=PostpassState(),
    )
    if collision_suppressed.phase_active or collision_suppressed.reward != 0.0:
        raise AssertionError("Opponent-collision latch must suppress the reward")
    passed.append("opponent_collision_suppression")

    clear_state = PostpassState(entered=True)
    cleared = postpass_reward_step(
        previous_relative_progress_m=0.10,
        current_relative_progress_m=0.20,
        previous_ego_pose=np.asarray((0.0, 0.0, 0.0)),
        current_ego_pose=np.asarray((0.0, 0.0, 0.0)),
        current_opponent_pose=np.asarray((-2.0, 0.0, 0.0)),
        opponent_collision_latched=False,
        transition_dt_s=0.10,
        config=config,
        state=clear_state,
    )
    if not cleared.cleared or cleared.phase_active or cleared.reward != 0.0:
        raise AssertionError("Safe rear gap must latch the phase clear")
    returned = postpass_reward_step(
        previous_relative_progress_m=0.20,
        current_relative_progress_m=0.30,
        previous_ego_pose=np.asarray((0.0, 0.60, 0.0)),
        current_ego_pose=np.asarray((0.0, 0.40, 0.0)),
        current_opponent_pose=np.asarray((0.0, 0.0, 0.0)),
        opponent_collision_latched=False,
        transition_dt_s=0.10,
        config=config,
        state=clear_state,
    )
    if returned.phase_active or returned.reward != 0.0:
        raise AssertionError("Latched-clear phase must not reactivate")
    passed.append("safe_gap_latched_clear")

    cap_config = RewardConfig(
        reward_weight_per_m=1.0,
        maximum_step_penalty=0.02,
        maximum_episode_penalty=0.03,
        activation_clearance_m=None,
        maximum_ttc_s=None,
    )
    cap_state = PostpassState()
    rewards = [
        bounded_negative_reward(1.0, cap_config, cap_state),
        bounded_negative_reward(1.0, cap_config, cap_state),
        bounded_negative_reward(1.0, cap_config, cap_state),
    ]
    _assert_close(rewards, (-0.02, -0.01, 0.0))
    _assert_close(cap_state.penalty_used, 0.03)
    passed.append("step_and_episode_caps")

    relative = np.asarray((-0.10, 0.00, 0.06, 0.10, 0.15))
    time_s = np.arange(len(relative), dtype=np.float64) * 0.10
    ego_poses = np.asarray(
        (
            (0.0, 0.70, 0.0),
            (0.0, 0.55, 0.0),
            (0.0, 0.35, 0.0),
            (0.0, 0.25, 0.0),
            (0.0, 0.20, 0.0),
        ),
        dtype=np.float64,
    )
    opponent_poses = np.zeros_like(ego_poses)
    signed_gap = np.asarray(
        [
            signed_rear_longitudinal_gap(ego_pose, opponent_pose, 0.58, 0.31)
            for ego_pose, opponent_pose in zip(ego_poses, opponent_poses)
        ]
    )
    active = np.asarray((False, False, True, True, True))
    clearance = np.full(len(relative), np.inf)
    closing_distance = np.zeros(len(relative))
    for index in np.flatnonzero(active):
        closing_geometry = ego_induced_rear_closing(
            ego_poses[index - 1],
            ego_poses[index],
            opponent_poses[index],
            0.58,
            0.31,
        )
        clearance[index] = closing_geometry.current_clearance_m
        closing_distance[index] = closing_geometry.closing_m
    dt = np.asarray((0.0, 0.10, 0.10, 0.10, 0.10))
    closing_speed = np.divide(
        closing_distance,
        dt,
        out=np.zeros_like(closing_distance),
        where=dt > 0.0,
    )
    ttc = np.divide(
        clearance,
        closing_speed,
        out=np.full_like(clearance, np.inf),
        where=closing_speed > 0.0,
    )
    vectorized = evaluate_setting(
        {
            "time_s": time_s,
            "active": active,
            "signed_gap": signed_gap,
            "clearance": clearance,
            "closing": closing_distance,
            "closing_speed": closing_speed,
            "ttc": ttc,
        },
        config,
        retain_arrays=True,
    )
    sequential_state = PostpassState()
    sequential_reward = np.zeros(len(relative))
    sequential_trigger = np.zeros(len(relative), dtype=bool)
    for index in range(1, len(relative)):
        step = postpass_reward_step(
            previous_relative_progress_m=float(relative[index - 1]),
            current_relative_progress_m=float(relative[index]),
            previous_ego_pose=ego_poses[index - 1],
            current_ego_pose=ego_poses[index],
            current_opponent_pose=opponent_poses[index],
            opponent_collision_latched=False,
            transition_dt_s=float(dt[index]),
            config=config,
            state=sequential_state,
        )
        sequential_reward[index] = step.reward
        sequential_trigger[index] = step.triggered
    _assert_close(vectorized["reward"], sequential_reward)
    _assert_close(vectorized["trigger"], sequential_trigger)
    _assert_close(
        vectorized["episode_penalty_used"],
        sequential_state.penalty_used,
    )
    passed.append("sequential_vectorized_episode_equivalence")

    factor = GAMMA * GAE_LAMBDA
    reward_delta = np.asarray((0.0, 0.0, -1.0, 0.0))
    episode_end = np.asarray((False, False, True, True))
    delta = gae_delta_from_reward_delta(
        reward_delta,
        episode_end,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
    )
    _assert_close(delta, (-factor * factor, -factor, -1.0, 0.0))
    passed.append("gae_impulse_closed_form")

    reward_delta = np.asarray((-1.0, -2.0))
    episode_end = np.asarray((True, True))
    delta = gae_delta_from_reward_delta(
        reward_delta,
        episode_end,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
    )
    _assert_close(delta, reward_delta)
    passed.append("gae_no_cross_episode_leak")

    rng = np.random.default_rng(31071995)
    reward_delta = -rng.uniform(0.0, 0.01, size=37)
    episode_end = np.zeros(37, dtype=bool)
    episode_end[[6, 18, 36]] = True
    recursive = gae_delta_from_reward_delta(
        reward_delta,
        episode_end,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
    )
    explicit = np.zeros_like(recursive)
    for start, stop in ((0, 7), (7, 19), (19, 37)):
        for index in range(start, stop):
            powers = factor ** np.arange(stop - index)
            explicit[index] = float(
                np.sum(powers * reward_delta[index:stop])
            )
    _assert_close(recursive, explicit)
    if np.any(recursive > 0.0):
        raise AssertionError("Non-positive rewards created a positive GAE delta")
    passed.append("gae_multi_episode_explicit_convolution")

    if wilson_interval(0, 0) != (None, None):
        raise AssertionError("Empty Wilson interval must be undefined")
    zero_interval = wilson_interval(0, 10)
    full_interval = wilson_interval(10, 10)
    _assert_close(zero_interval, (0.0, 0.2775327998628892))
    _assert_close(full_interval, (0.7224672001371107, 1.0))
    passed.append("wilson_interval_boundary_cases")

    selection_common = {
        "follow_trigger_rate": 0.0,
        "trigger_step_fraction": 0.01,
        "tail_first_signal_lead_s_median": 0.2,
    }
    selected = select_setting(
        [
            {
                **selection_common,
                "setting": "broad",
                "acceptance_pass": True,
                "tail_preterminal_capture_rate": 1.0,
                "overtake_trigger_rate": 0.19,
            },
            {
                **selection_common,
                "setting": "selective",
                "acceptance_pass": True,
                "tail_preterminal_capture_rate": 0.92,
                "overtake_trigger_rate": 0.03,
            },
        ]
    )
    if selected["setting"] != "selective":
        raise AssertionError("Accepted settings must prioritize selectivity")
    fallback = select_setting(
        [
            {
                **selection_common,
                "setting": "high_capture",
                "acceptance_pass": False,
                "tail_preterminal_capture_rate": 0.80,
                "overtake_trigger_rate": 0.50,
            },
            {
                **selection_common,
                "setting": "low_capture",
                "acceptance_pass": False,
                "tail_preterminal_capture_rate": 0.70,
                "overtake_trigger_rate": 0.01,
            },
        ]
    )
    if (
        fallback["setting"] != "high_capture"
        or fallback["selected_from_accepted_set"]
    ):
        raise AssertionError("Rejected fallback selection is inconsistent")
    passed.append("accepted_selectivity_and_rejected_fallback_order")

    with tempfile.TemporaryDirectory(prefix="postpass_trace_schema_") as temporary:
        temporary_root = Path(temporary)
        count = 3
        valid_arrays = {
            "time_s": np.arange(count, dtype=np.float64) * 0.01,
            "ego_lidar_360": np.ones((count, 360), dtype=np.float32),
            "opp_lidar_360": np.ones((count, 360), dtype=np.float32),
            "ego_raw_action": np.zeros((count, 2), dtype=np.float32),
            "ego_executed_action": np.zeros((count, 2), dtype=np.float32),
            "opp_executed_action": np.zeros((count, 2), dtype=np.float32),
            "ego_measured_speed_mps": np.ones(count, dtype=np.float32),
            "opp_measured_speed_mps": np.ones(count, dtype=np.float32),
            "ego_pose": np.zeros((count, 3), dtype=np.float64),
            "opp_pose": np.zeros((count, 3), dtype=np.float64),
            "collisions": np.zeros((count, 2), dtype=np.bool_),
            "action_applied": np.asarray((True, True, False)),
            "terminal_post_step": np.asarray((False, False, True)),
        }
        valid_path = temporary_root / "valid.npz"
        np.savez_compressed(valid_path, **valid_arrays)
        schema = trace_archive_schema(valid_path)
        if (
            schema["trace_length"] != count
            or schema["member_count"] != len(TRACE_ARCHIVE_FIELDS)
            or not schema["all_member_crcs_verified"]
        ):
            raise AssertionError("Valid NPZ schema receipt is incorrect")
        invalid_arrays = dict(valid_arrays)
        invalid_arrays["ego_pose"] = invalid_arrays["ego_pose"].astype(
            np.float32
        )
        invalid_path = temporary_root / "invalid_dtype.npz"
        np.savez_compressed(invalid_path, **invalid_arrays)
        try:
            trace_archive_schema(invalid_path)
        except ValueError:
            pass
        else:
            raise AssertionError("Trace schema accepted a wrong pose dtype")
    passed.append("npz_crc_shape_dtype_schema")

    policy_loss, samples = clipped_ppo_policy_loss(
        np.asarray((1.0, -1.0)),
        np.asarray((1.3, 0.7)),
        clip_range=0.20,
        normalize_advantage=False,
    )
    _assert_close(samples, (-1.2, 0.8))
    _assert_close(policy_loss, -0.2)
    passed.append("clipped_policy_loss_manual_case")

    normalized = normalized_advantages(np.asarray((1.0, 2.0, 3.0)))
    _assert_close(normalized, (-0.99999999, 0.0, 0.99999999))
    normalized_policy_loss, normalized_samples = clipped_ppo_policy_loss(
        np.asarray((1.0, 2.0, 3.0)),
        np.asarray((1.0, 1.3, 0.7)),
        clip_range=0.20,
        normalize_advantage=True,
    )
    _assert_close(
        normalized_samples,
        (0.99999999, 0.0, -0.699999993),
    )
    _assert_close(normalized_policy_loss, 0.099999999)
    try:
        normalized_advantages(np.asarray((1.0,)))
    except ValueError:
        pass
    else:
        raise AssertionError("One-sample normalization must fail closed")
    passed.append("torch_sample_std_advantage_normalization")

    predictions = np.asarray((0.0, 1.0))
    baseline_returns = np.asarray((1.0, 1.0))
    return_delta = np.asarray((-0.5, -0.25))
    baseline_value_loss, baseline_samples = mean_squared_value_loss(
        predictions,
        baseline_returns,
    )
    treatment_value_loss, treatment_samples = mean_squared_value_loss(
        predictions,
        baseline_returns + return_delta,
    )
    value_loss_delta, value_loss_delta_samples = (
        fixed_prediction_value_loss_delta(
            predictions,
            baseline_returns,
            return_delta,
        )
    )
    _assert_close(baseline_value_loss, 0.5)
    _assert_close(baseline_samples, (1.0, 0.0))
    _assert_close(treatment_value_loss, 0.15625)
    _assert_close(treatment_samples, (0.25, 0.0625))
    _assert_close(value_loss_delta, treatment_value_loss - baseline_value_loss)
    _assert_close(
        value_loss_delta_samples,
        treatment_samples - baseline_samples,
    )
    passed.append("fixed_prediction_value_loss_delta")

    actions = np.asarray(((0.1, 4.0), (-0.2, 5.0), (0.3, 6.0)))
    zero_loss, zero_gradient = masked_follow_teacher_huber_loss(
        actions,
        actions.copy(),
        np.asarray((True, True, False)),
    )
    _assert_close(zero_loss, 0.0)
    _assert_close(zero_gradient, np.zeros_like(actions))
    empty_loss, empty_gradient = masked_follow_teacher_huber_loss(
        actions,
        np.zeros_like(actions),
        np.zeros(3, dtype=bool),
    )
    _assert_close(empty_loss, 0.0)
    _assert_close(empty_gradient, np.zeros_like(actions))
    passed.append("follow_teacher_zero_and_empty_mask")

    teacher = np.zeros_like(actions)
    active = np.asarray((True, False, False))
    loss, gradient = masked_follow_teacher_huber_loss(actions, teacher, active)
    epsilon = 1e-6
    perturbed = actions.copy()
    perturbed[0, 0] += epsilon
    next_loss, _ = masked_follow_teacher_huber_loss(perturbed, teacher, active)
    finite_difference = (next_loss - loss) / epsilon
    _assert_close(finite_difference, gradient[0, 0], tolerance=2e-6)
    if np.any(gradient[~active] != 0.0):
        raise AssertionError("Masked teacher-loss rows received a gradient")
    passed.append("follow_teacher_gradient_finite_difference")

    candidate_crosscheck = None
    if candidate_module is not None:
        candidate_crosscheck = crosscheck_candidate_module(candidate_module)
        passed.append("candidate_module_crosscheck")

    return {
        "status": "passed",
        "test_count": len(passed),
        "tests": passed,
        "candidate_crosscheck": candidate_crosscheck,
    }


def crosscheck_candidate_module(path: Path) -> dict[str, Any]:
    """Compare common pure geometry/cap behavior against a fixed file hash."""

    module_path = path.expanduser().resolve()
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    before = _sha256(module_path)
    spec = importlib.util.spec_from_file_location(
        f"candidate_postpass_{before[:12]}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    project_root_text = str(PROJECT_ROOT)
    added_project_root = project_root_text not in sys.path
    if added_project_root:
        sys.path.insert(0, project_root_text)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        if added_project_root:
            sys.path.remove(project_root_text)
    rng = np.random.default_rng(20260723)
    maximum_error = {
        "signed_gap": 0.0,
        "counterfactual_clearance": 0.0,
        "current_clearance": 0.0,
        "closing": 0.0,
        "bounded_reward": 0.0,
    }
    case_count = 256
    for _ in range(case_count):
        previous_ego = rng.normal(size=3)
        current_ego = previous_ego + rng.normal(scale=(0.1, 0.1, 0.03), size=3)
        opponent = rng.normal(size=3)
        ours_gap = signed_rear_longitudinal_gap(
            current_ego,
            opponent,
            0.58,
            0.31,
        )
        theirs_gap = module.signed_rear_longitudinal_gap(
            current_ego,
            opponent,
            0.58,
            0.31,
        )
        maximum_error["signed_gap"] = max(
            maximum_error["signed_gap"],
            abs(ours_gap - theirs_gap),
        )
        ours = ego_induced_rear_closing(
            previous_ego,
            current_ego,
            opponent,
            0.58,
            0.31,
        )
        theirs = module.ego_induced_rear_closing(
            previous_ego,
            current_ego,
            opponent,
            0.58,
            0.31,
        )
        for field, theirs_name in (
            ("counterfactual_clearance", "counterfactual_previous_clearance_m"),
            ("current_clearance", "current_clearance_m"),
            ("closing", "closing_m"),
        ):
            ours_name = (
                "counterfactual_previous_clearance_m"
                if field == "counterfactual_clearance"
                else f"{field}_m"
            )
            maximum_error[field] = max(
                maximum_error[field],
                abs(getattr(ours, ours_name) - getattr(theirs, theirs_name)),
            )
        basis = float(rng.uniform(0.0, 0.1))
        used = float(rng.uniform(0.0, 0.04))
        cap_config = RewardConfig(
            reward_weight_per_m=0.25,
            maximum_step_penalty=0.005,
            maximum_episode_penalty=0.05,
        )
        ours_state = PostpassState(penalty_used=used)
        ours_reward = bounded_negative_reward(basis, cap_config, ours_state)
        theirs_reward = module.bounded_postpass_reward(
            basis,
            0.25,
            0.005,
            used,
            0.05,
        )
        maximum_error["bounded_reward"] = max(
            maximum_error["bounded_reward"],
            abs(ours_reward - theirs_reward),
        )
    after = _sha256(module_path)
    if before != after:
        raise RuntimeError(f"Candidate module changed during cross-check: {module_path}")
    if max(maximum_error.values()) > 1e-10:
        raise AssertionError(f"Candidate geometry mismatch: {maximum_error}")
    return {
        "path": str(module_path),
        "sha256": before,
        "case_count": case_count,
        "maximum_absolute_error": maximum_error,
    }


class TrackProjector:
    def __init__(self, path: Path) -> None:
        from scipy.spatial import cKDTree

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
        row = np.arange(len(points))
        choice = np.argmin(distance_sq, axis=1)
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
        ego = self.project_progress(ego_xy)
        opponent = self.project_progress(opponent_xy)
        wrapped = (
            ego - opponent + 0.5 * self.track_length
        ) % self.track_length - 0.5 * self.track_length
        output = np.empty_like(wrapped)
        output[0] = wrapped[0]
        increments = (
            np.diff(wrapped) + 0.5 * self.track_length
        ) % self.track_length - 0.5 * self.track_length
        output[1:] = wrapped[0] + np.cumsum(increments)
        return output


def panel_label(directory: str) -> str:
    if directory == "end2race_Austin":
        return "BC"
    suffix = "_Austin"
    if not directory.endswith(suffix):
        raise ValueError(f"Unexpected panel directory: {directory}")
    return directory[: -len(suffix)]


def resolve(root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _strict_csv_bool(value: str, *, field: str, key: tuple[str, str]) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"{key}: {field} is not a strict boolean: {value!r}")
    return value == "True"


def load_tail_labels(
    path: Path,
    allowed_panels: set[str] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {
            "panel",
            "scenario_id",
            "episode_key",
            "panel_valid",
            "trace_format",
            "trace_length",
            "terminal_marker_valid",
            "trace_collision_marker_matches_json",
            "pass_detected",
            "merge_tail_primary",
            "trace_path",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path}: missing label columns "
                f"{sorted(required - set(reader.fieldnames or ())) }"
            )
        for row in reader:
            if (
                allowed_panels is not None
                and row["panel"] not in allowed_panels
            ):
                continue
            key = (row["panel"], row["scenario_id"])
            if key in labels:
                raise ValueError(f"Duplicate tail label: {key}")
            try:
                trace_length = int(row["trace_length"])
            except ValueError as error:
                raise ValueError(f"{key}: invalid trace_length") from error
            if trace_length < 2:
                raise ValueError(f"{key}: trace_length must be at least two")
            labels[key] = {
                "episode_key": row["episode_key"],
                "panel_valid": _strict_csv_bool(
                    row["panel_valid"],
                    field="panel_valid",
                    key=key,
                ),
                "trace_format": row["trace_format"],
                "trace_length": trace_length,
                "terminal_marker_valid": _strict_csv_bool(
                    row["terminal_marker_valid"],
                    field="terminal_marker_valid",
                    key=key,
                ),
                "trace_collision_marker_matches_json": _strict_csv_bool(
                    row["trace_collision_marker_matches_json"],
                    field="trace_collision_marker_matches_json",
                    key=key,
                ),
                "pass_detected": _strict_csv_bool(
                    row["pass_detected"],
                    field="pass_detected",
                    key=key,
                ),
                "merge_tail_primary": _strict_csv_bool(
                    row["merge_tail_primary"],
                    field="merge_tail_primary",
                    key=key,
                ),
                "trace_path": row["trace_path"],
            }
    if allowed_panels is not None:
        missing_panels = allowed_panels - {
            panel for panel, _scenario_id in labels
        }
        if missing_panels:
            raise ValueError(
                f"{path}: no collision labels for panels {sorted(missing_panels)}"
            )
    return labels


TRACE_ARCHIVE_FIELDS = (
    "time_s",
    "ego_lidar_360",
    "opp_lidar_360",
    "ego_raw_action",
    "ego_executed_action",
    "opp_executed_action",
    "ego_measured_speed_mps",
    "opp_measured_speed_mps",
    "ego_pose",
    "opp_pose",
    "collisions",
    "action_applied",
    "terminal_post_step",
)

TRACE_ANALYSIS_FIELDS = (
    "time_s",
    "ego_raw_action",
    "ego_pose",
    "opp_pose",
    "collisions",
    "action_applied",
    "terminal_post_step",
)

TRACE_DTYPES = {
    "time_s": np.dtype(np.float64),
    "ego_lidar_360": np.dtype(np.float32),
    "opp_lidar_360": np.dtype(np.float32),
    "ego_raw_action": np.dtype(np.float32),
    "ego_executed_action": np.dtype(np.float32),
    "opp_executed_action": np.dtype(np.float32),
    "ego_measured_speed_mps": np.dtype(np.float32),
    "opp_measured_speed_mps": np.dtype(np.float32),
    "ego_pose": np.dtype(np.float64),
    "opp_pose": np.dtype(np.float64),
    "collisions": np.dtype(np.bool_),
    "action_applied": np.dtype(np.bool_),
    "terminal_post_step": np.dtype(np.bool_),
}


def trace_archive_schema(path: Path) -> dict[str, Any]:
    """Validate every NPZ member's CRC, shape, and dtype without retaining it."""

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        member_names = [info.filename for info in infos]
        if len(member_names) != len(set(member_names)):
            raise ValueError(f"{path}: duplicate ZIP member")
        expected_members = {f"{name}.npy" for name in TRACE_ARCHIVE_FIELDS}
        if set(member_names) != expected_members:
            missing = sorted(expected_members - set(member_names))
            unexpected = sorted(set(member_names) - expected_members)
            raise ValueError(
                f"{path}: archive members differ; missing={missing}, "
                f"unexpected={unexpected}"
            )
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"{path}: CRC failure in {bad_member}")
        schemas: dict[str, dict[str, Any]] = {}
        fingerprint_rows = []
        for info in sorted(infos, key=lambda value: value.filename):
            field = info.filename[: -len(".npy")]
            with archive.open(info, "r") as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran_order, dtype = np.lib.format._read_array_header(
                    stream,
                    version,
                )
            dtype = np.dtype(dtype)
            schemas[field] = {
                "shape": tuple(int(value) for value in shape),
                "fortran_order": bool(fortran_order),
                "dtype": str(dtype),
            }
            fingerprint_rows.append(
                (
                    info.filename,
                    int(info.CRC),
                    int(info.file_size),
                    int(info.compress_size),
                    tuple(int(value) for value in shape),
                    str(dtype),
                    bool(fortran_order),
                )
            )
    count_shape = schemas["time_s"]["shape"]
    if len(count_shape) != 1 or count_shape[0] < 2:
        raise ValueError(f"{path}: invalid time_s shape {count_shape}")
    count = int(count_shape[0])
    expected_shapes = {
        "time_s": (count,),
        "ego_lidar_360": (count, 360),
        "opp_lidar_360": (count, 360),
        "ego_raw_action": (count, 2),
        "ego_executed_action": (count, 2),
        "opp_executed_action": (count, 2),
        "ego_measured_speed_mps": (count,),
        "opp_measured_speed_mps": (count,),
        "ego_pose": (count, 3),
        "opp_pose": (count, 3),
        "collisions": (count, 2),
        "action_applied": (count,),
        "terminal_post_step": (count,),
    }
    for name in TRACE_ARCHIVE_FIELDS:
        if schemas[name]["shape"] != expected_shapes[name]:
            raise ValueError(
                f"{path}: {name} shape {schemas[name]['shape']} "
                f"!= {expected_shapes[name]}"
            )
        if np.dtype(schemas[name]["dtype"]) != TRACE_DTYPES[name]:
            raise ValueError(
                f"{path}: {name} dtype {schemas[name]['dtype']} "
                f"!= {TRACE_DTYPES[name]}"
            )
        if schemas[name]["fortran_order"]:
            raise ValueError(f"{path}: unexpected Fortran-order array {name}")
    stat = path.stat()
    member_digest = hashlib.sha256(
        json.dumps(fingerprint_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "trace_path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "trace_length": count,
        "member_count": len(fingerprint_rows),
        "uncompressed_bytes": sum(row[2] for row in fingerprint_rows),
        "compressed_member_bytes": sum(row[3] for row in fingerprint_rows),
        "member_crc_schema_sha256": member_digest,
        "all_member_crcs_verified": True,
    }


def load_trace(
    path: Path,
    expected_collision: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    archive_schema = trace_archive_schema(path)
    with np.load(path, allow_pickle=False) as archive:
        trace = {
            name: np.asarray(archive[name])
            for name in TRACE_ANALYSIS_FIELDS
        }
    lengths = {len(value) for value in trace.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError(f"{path}: inconsistent trace lengths")
    count = next(iter(lengths))
    numeric = (
        trace["time_s"],
        trace["ego_raw_action"],
        trace["ego_pose"],
        trace["opp_pose"],
    )
    if not all(np.isfinite(value).all() for value in numeric):
        raise ValueError(f"{path}: non-finite numeric array")
    time_s = np.asarray(trace["time_s"], dtype=np.float64)
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError(f"{path}: time is not strictly increasing")
    terminal = np.asarray(trace["terminal_post_step"], dtype=bool)
    actions = np.asarray(trace["action_applied"], dtype=bool)
    if (
        np.count_nonzero(terminal) != 1
        or not terminal[-1]
        or np.count_nonzero(actions) != count - 1
        or actions[-1]
    ):
        raise ValueError(f"{path}: invalid terminal/action contract")
    if bool(np.asarray(trace["collisions"], dtype=bool)[-1, 0]) != expected_collision:
        raise ValueError(f"{path}: collision marker mismatch")
    return trace, archive_schema


def setting_name(config: RewardConfig) -> str:
    clearance = (
        "none"
        if config.activation_clearance_m is None
        else f"{config.activation_clearance_m:.2f}"
    )
    ttc = "none" if config.maximum_ttc_s is None else f"{config.maximum_ttc_s:.2f}"
    return (
        f"clear{clearance}_ttc{ttc}_"
        f"dead{config.closing_deadband_mps:.2f}"
    )


def sweep_configs() -> list[RewardConfig]:
    configs = [
        RewardConfig(
            activation_clearance_m=None,
            maximum_ttc_s=None,
            closing_deadband_mps=0.10,
        )
    ]
    for clearance in (0.05, 0.10, 0.15, 0.20, 0.30):
        for ttc in (0.25, 0.50, 0.75, 1.00):
            for deadband in (0.05, 0.10):
                configs.append(
                    RewardConfig(
                        activation_clearance_m=clearance,
                        maximum_ttc_s=ttc,
                        closing_deadband_mps=deadband,
                    )
                )
    names = [setting_name(config) for config in configs]
    if len(set(names)) != len(names):
        raise RuntimeError("Reward sweep settings are not unique")
    return configs


def _first_crossing(relative: np.ndarray, margin: float) -> int | None:
    indices = np.flatnonzero(
        (relative[:-1] <= margin) & (relative[1:] > margin)
    )
    return int(indices[0] + 1) if indices.size else None


def prepare_episode_geometry(
    trace: dict[str, np.ndarray],
    projector: TrackProjector,
    base_config: RewardConfig,
) -> dict[str, np.ndarray | int | None]:
    time_s = np.asarray(trace["time_s"], dtype=np.float64)
    ego_pose = np.asarray(trace["ego_pose"], dtype=np.float64)
    opponent_pose = np.asarray(trace["opp_pose"], dtype=np.float64)
    relative = projector.relative_progress(ego_pose[:, :2], opponent_pose[:, :2])
    entry = _first_crossing(relative, base_config.pass_margin_m)
    signed_gap = np.asarray(
        [
            signed_rear_longitudinal_gap(
                ego,
                opponent,
                base_config.vehicle_length_m,
                base_config.vehicle_width_m,
            )
            for ego, opponent in zip(ego_pose, opponent_pose)
        ],
        dtype=np.float64,
    )
    opponent_collision = np.maximum.accumulate(
        np.asarray(trace["collisions"], dtype=bool)[:, 1]
    )
    active = np.zeros(len(time_s), dtype=bool)
    if entry is not None:
        active[entry:] = True
        clear = np.flatnonzero(
            (np.arange(len(time_s)) >= entry)
            & (signed_gap >= base_config.safe_rear_gap_m)
        )
        if clear.size:
            active[int(clear[0]) :] = False
    active &= ~opponent_collision
    clearance = np.full(len(time_s), np.inf, dtype=np.float64)
    closing = np.zeros(len(time_s), dtype=np.float64)
    for index in np.flatnonzero(active):
        if index == 0:
            continue
        result = ego_induced_rear_closing(
            ego_pose[index - 1],
            ego_pose[index],
            opponent_pose[index],
            base_config.vehicle_length_m,
            base_config.vehicle_width_m,
        )
        clearance[index] = result.current_clearance_m
        closing[index] = result.closing_m
    dt = np.zeros(len(time_s), dtype=np.float64)
    dt[1:] = np.diff(time_s)
    closing_speed = np.divide(
        closing,
        dt,
        out=np.zeros_like(closing),
        where=dt > 0.0,
    )
    ttc = np.divide(
        clearance,
        closing_speed,
        out=np.full_like(clearance, np.inf),
        where=closing_speed > 0.0,
    )
    return {
        "time_s": time_s,
        "relative": relative,
        "entry": entry,
        "signed_gap": signed_gap,
        "opponent_collision_latched": opponent_collision,
        "active": active,
        "clearance": clearance,
        "closing": closing,
        "dt": dt,
        "closing_speed": closing_speed,
        "ttc": ttc,
    }


def evaluate_setting(
    geometry: dict[str, Any],
    config: RewardConfig,
    *,
    retain_arrays: bool = False,
) -> dict[str, Any]:
    active = np.asarray(geometry["active"], dtype=bool)
    gap = np.asarray(geometry["signed_gap"], dtype=np.float64)
    clearance = np.asarray(geometry["clearance"], dtype=np.float64)
    closing = np.asarray(geometry["closing"], dtype=np.float64)
    closing_speed = np.asarray(geometry["closing_speed"], dtype=np.float64)
    ttc = np.asarray(geometry["ttc"], dtype=np.float64)
    unsafe = np.clip(
        (config.safe_rear_gap_m - gap) / config.safe_rear_gap_m,
        0.0,
        1.0,
    )
    if config.activation_clearance_m is None:
        proximity = np.ones_like(clearance)
        clearance_gate = np.ones_like(active)
    else:
        proximity = np.clip(
            (config.activation_clearance_m - clearance)
            / config.activation_clearance_m,
            0.0,
            1.0,
        )
        clearance_gate = clearance < config.activation_clearance_m
    ttc_gate = (
        np.ones_like(active)
        if config.maximum_ttc_s is None
        else ttc <= config.maximum_ttc_s
    )
    trigger = (
        active
        & clearance_gate
        & ttc_gate
        & (closing_speed > config.closing_deadband_mps)
    )
    basis = np.where(
        trigger,
        closing * unsafe * unsafe * proximity * proximity,
        0.0,
    )
    reward = np.zeros_like(basis)
    used = 0.0
    for index in np.flatnonzero(trigger):
        state = PostpassState(penalty_used=used)
        reward[index] = bounded_negative_reward(float(basis[index]), config, state)
        used = state.penalty_used
    transition_reward = reward[1:]
    endings = np.zeros(len(transition_reward), dtype=bool)
    endings[-1] = True
    gae_delta = gae_delta_from_reward_delta(
        transition_reward,
        endings,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
    )
    preterminal = trigger.copy()
    preterminal[-1] = False
    first = int(np.flatnonzero(trigger)[0]) if np.any(trigger) else None
    result = {
        "triggered": bool(np.any(trigger)),
        "preterminal_triggered": bool(np.any(preterminal)),
        "trigger_steps": int(np.count_nonzero(trigger)),
        "preterminal_trigger_steps": int(np.count_nonzero(preterminal)),
        "active_steps": int(np.count_nonzero(active)),
        "penalty_sum": float(-np.sum(reward)),
        "reward_delta_min": float(np.min(reward, initial=0.0)),
        "reward_delta_max": float(np.max(reward, initial=0.0)),
        "maximum_step_penalty_used": float(np.max(-reward, initial=0.0)),
        "episode_penalty_used": float(used),
        "step_cap_hit": bool(
            np.any(
                config.reward_weight_per_m * basis
                >= config.maximum_step_penalty
            )
        ),
        "episode_cap_hit": bool(
            used >= config.maximum_episode_penalty - 1e-12
        ),
        "first_trigger_index": first,
        "first_trigger_lead_s": (
            float(geometry["time_s"][-1] - geometry["time_s"][first])
            if first is not None
            else None
        ),
        "gae_delta_sum": float(np.sum(gae_delta)),
        "gae_delta_min": float(np.min(gae_delta, initial=0.0)),
        "gae_delta_max": float(np.max(gae_delta, initial=0.0)),
        "gae_affected_steps": int(np.count_nonzero(gae_delta)),
    }
    if retain_arrays:
        result.update(
            {
                "trigger": trigger,
                "reward": reward,
                "gae_delta": gae_delta,
            }
        )
    return result


def quantile(values: list[float], probability: float) -> float | None:
    return (
        float(np.quantile(np.asarray(values, dtype=np.float64), probability))
        if values
        else None
    )


def wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    probability = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def load_panel_episodes(
    root: Path,
    directory: str,
    labels: dict[tuple[str, str], dict[str, Any]],
    projector: TrackProjector,
    configs: list[RewardConfig],
    process_guard_checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    label = panel_label(directory)
    panel_root = root / "eval_results" / directory / "multiagents"
    result_path = panel_root / "results_multi.json"
    document = json.loads(result_path.read_text(encoding="utf-8"))
    episodes = document.get("episodes")
    final = document.get("final")
    if not isinstance(episodes, dict) or not isinstance(final, dict):
        raise ValueError(f"{result_path}: missing episodes/final")
    if (
        len(episodes) != 600
        or int(final.get("total_episodes", -1)) != 600
        or int(final.get("error_count", -1)) != 0
    ):
        raise ValueError(f"{result_path}: panel is incomplete or has errors")
    output = []
    observed = Counter()
    observed_scenarios: set[str] = set()
    observed_collision_keys: set[tuple[str, str]] = set()
    for episode_ordinal, (episode_key, episode) in enumerate(
        sorted(episodes.items())
    ):
        if episode_ordinal % 50 == 0:
            process_guard_checks.append(require_host_process_idle())
        if str(episode.get("episode_key")) != episode_key:
            raise ValueError(f"{result_path}: episode-key payload mismatch")
        scenario_id = str(episode["scenario_id"])
        if scenario_id in observed_scenarios:
            raise ValueError(f"{result_path}: duplicate scenario_id {scenario_id}")
        observed_scenarios.add(scenario_id)
        raw_outcome = str(episode["outcome"])
        outcome = "collision" if raw_outcome == "ego_collision" else raw_outcome
        if outcome not in {"collision", "overtake", "follow"}:
            raise ValueError(f"{result_path}: unexpected outcome {outcome}")
        if str(episode.get("collision_scope")) != "ego":
            raise ValueError(f"{result_path}: episode lacks ego collision scope")
        if (
            not bool(episode.get("observation_finite"))
            or not bool(episode.get("action_finite"))
            or bool(episode.get("initial_ego_collision"))
            or bool(episode.get("initial_opponent_collision"))
        ):
            raise ValueError(f"{result_path}: invalid finite/initial state")
        expected_collision = bool(episode["ego_collision_occurred"])
        if expected_collision != (outcome == "collision"):
            raise ValueError(f"{result_path}: collision/outcome mismatch")
        trace_path = panel_root / "traces" / f"{episode_key}.npz"
        trace, archive_schema = load_trace(trace_path, expected_collision)
        trace_count = len(trace["time_s"])
        if int(episode["steps"]) != trace_count - 1:
            raise ValueError(f"{trace_path}: JSON/trace step-count mismatch")
        _assert_close(
            float(episode["simulation_time_s"]),
            float(trace["time_s"][-1]),
            tolerance=1e-8,
        )
        observed_ego_collision = bool(
            np.any(np.asarray(trace["collisions"], dtype=bool)[:, 0])
        )
        if observed_ego_collision != expected_collision:
            raise ValueError(f"{trace_path}: ego collision history mismatch")
        observed_opponent_collision = bool(
            np.any(np.asarray(trace["collisions"], dtype=bool)[:, 1])
        )
        if observed_opponent_collision != bool(
            episode["opp_collision_occurred"]
        ):
            raise ValueError(f"{trace_path}: opponent collision mismatch")
        geometry = prepare_episode_geometry(trace, projector, configs[0])
        terminal_relative_error = abs(
            float(np.asarray(geometry["relative"])[-1])
            - float(episode["final_relative_position_m"])
        )
        if terminal_relative_error > 1e-6:
            raise ValueError(
                f"{trace_path}: terminal relative-progress error "
                f"{terminal_relative_error}"
            )
        label_key = (label, scenario_id)
        label_record = labels.get(label_key)
        if outcome == "collision":
            observed_collision_keys.add(label_key)
            if label_record is None:
                raise ValueError(f"{trace_path}: collision has no kinematic label")
            if (
                not label_record["panel_valid"]
                or label_record["trace_format"] != "post_step_v2"
                or not label_record["terminal_marker_valid"]
                or not label_record["trace_collision_marker_matches_json"]
                or label_record["episode_key"] != episode_key
                or label_record["trace_length"] != trace_count
            ):
                raise ValueError(f"{trace_path}: invalid/stale kinematic label")
            labelled_trace_path = resolve(root, Path(label_record["trace_path"]))
            if labelled_trace_path != trace_path.resolve():
                raise ValueError(f"{trace_path}: kinematic trace path mismatch")
            if bool(geometry["entry"] is not None) != label_record["pass_detected"]:
                raise ValueError(f"{trace_path}: pass-detection mismatch")
        elif label_record is not None:
            raise ValueError(f"{trace_path}: non-collision has a collision label")
        reference_primary_tail = bool(
            label_record is not None and label_record["merge_tail_primary"]
        )
        retain_arrays = label == "BC" and reference_primary_tail
        results = {
            setting_name(config): evaluate_setting(
                geometry,
                config,
                retain_arrays=retain_arrays,
            )
            for config in configs
        }
        output.append(
            {
                "panel": label,
                "directory": directory,
                "scenario_id": scenario_id,
                "episode_key": episode_key,
                "outcome": outcome,
                "reference_primary_tail": reference_primary_tail,
                "transition_count": len(trace["time_s"]) - 1,
                "pass_detected": geometry["entry"] is not None,
                "terminal_relative_progress_error_m": terminal_relative_error,
                "teacher_actions": (
                    np.asarray(trace["ego_raw_action"], dtype=np.float64)[:-1].copy()
                    if retain_arrays
                    else None
                ),
                "settings": results,
                "trace_path": trace_path,
                "trace_archive": {
                    **archive_schema,
                    "panel": label,
                    "scenario_id": scenario_id,
                    "episode_key": episode_key,
                },
            }
        )
        observed[outcome] += 1
    expected = Counter(
        {
            "collision": int(final["collision_count"]),
            "overtake": int(final["overtaking_count"]),
            "follow": int(final["following_count"]),
        }
    )
    if observed != expected:
        raise ValueError(f"{result_path}: {observed} != {expected}")
    labelled_collision_keys = {
        key for key in labels if key[0] == label
    }
    if observed_collision_keys != labelled_collision_keys:
        raise ValueError(
            f"{result_path}: collision-label coverage mismatch; "
            f"missing={sorted(observed_collision_keys - labelled_collision_keys)}, "
            f"extra={sorted(labelled_collision_keys - observed_collision_keys)}"
        )
    return output


def aggregate_settings(
    episodes: list[dict[str, Any]],
    configs: list[RewardConfig],
    *,
    scope: str = "all_panels",
) -> list[dict[str, Any]]:
    rows = []
    for config in configs:
        name = setting_name(config)
        evaluated = [
            {
                **episode,
                "result": episode["settings"][name],
            }
            for episode in episodes
        ]
        groups = {
            "tail": [
                row for row in evaluated if row["reference_primary_tail"]
            ],
            "other_collision": [
                row
                for row in evaluated
                if row["outcome"] == "collision"
                and not row["reference_primary_tail"]
            ],
            "overtake": [
                row for row in evaluated if row["outcome"] == "overtake"
            ],
            "follow": [
                row for row in evaluated if row["outcome"] == "follow"
            ],
        }

        def rate(group: str, field: str) -> float:
            values = groups[group]
            return (
                sum(bool(row["result"][field]) for row in values) / len(values)
                if values
                else 0.0
            )

        total_steps = sum(row["transition_count"] for row in evaluated)
        tail_leads = [
            row["result"]["first_trigger_lead_s"]
            for row in groups["tail"]
            if row["result"]["first_trigger_lead_s"] is not None
        ]
        penalties = [
            row["result"]["penalty_sum"]
            for row in evaluated
            if row["result"]["triggered"]
        ]
        tail_capture_count = sum(
            row["result"]["preterminal_triggered"]
            for row in groups["tail"]
        )
        overtake_trigger_count = sum(
            row["result"]["triggered"] for row in groups["overtake"]
        )
        follow_trigger_count = sum(
            row["result"]["triggered"] for row in groups["follow"]
        )
        other_collision_trigger_count = sum(
            row["result"]["triggered"]
            for row in groups["other_collision"]
        )
        tail_ci = wilson_interval(tail_capture_count, len(groups["tail"]))
        overtake_ci = wilson_interval(
            overtake_trigger_count,
            len(groups["overtake"]),
        )
        follow_ci = wilson_interval(
            follow_trigger_count,
            len(groups["follow"]),
        )
        row = {
            "scope": scope,
            "setting": name,
            "activation_clearance_m": config.activation_clearance_m,
            "maximum_ttc_s": config.maximum_ttc_s,
            "closing_deadband_mps": config.closing_deadband_mps,
            "episode_count": len(evaluated),
            "transition_count": total_steps,
            "tail_count": len(groups["tail"]),
            "tail_preterminal_capture_count": tail_capture_count,
            "tail_preterminal_capture_rate": rate(
                "tail",
                "preterminal_triggered",
            ),
            "tail_preterminal_capture_wilson95_low": tail_ci[0],
            "tail_preterminal_capture_wilson95_high": tail_ci[1],
            "other_collision_count": len(groups["other_collision"]),
            "other_collision_trigger_count": other_collision_trigger_count,
            "other_collision_trigger_rate": rate(
                "other_collision",
                "triggered",
            ),
            "overtake_count": len(groups["overtake"]),
            "overtake_trigger_count": overtake_trigger_count,
            "overtake_trigger_rate": rate("overtake", "triggered"),
            "overtake_trigger_wilson95_low": overtake_ci[0],
            "overtake_trigger_wilson95_high": overtake_ci[1],
            "follow_count": len(groups["follow"]),
            "follow_trigger_count": follow_trigger_count,
            "follow_trigger_rate": rate("follow", "triggered"),
            "follow_trigger_wilson95_low": follow_ci[0],
            "follow_trigger_wilson95_high": follow_ci[1],
            "trigger_step_count": sum(
                row["result"]["trigger_steps"] for row in evaluated
            ),
            "trigger_step_fraction": (
                sum(row["result"]["trigger_steps"] for row in evaluated)
                / total_steps
            ),
            "episode_cap_count": sum(
                row["result"]["episode_cap_hit"] for row in evaluated
            ),
            "step_cap_count": sum(
                row["result"]["step_cap_hit"] for row in evaluated
            ),
            "triggered_penalty_median": quantile(penalties, 0.50),
            "triggered_penalty_q90": quantile(penalties, 0.90),
            "tail_first_signal_lead_s_q10": quantile(tail_leads, 0.10),
            "tail_first_signal_lead_s_median": quantile(tail_leads, 0.50),
            "gae_delta_sum": float(
                sum(row["result"]["gae_delta_sum"] for row in evaluated)
            ),
            "gae_affected_step_count": sum(
                row["result"]["gae_affected_steps"] for row in evaluated
            ),
        }
        row["acceptance_pass"] = bool(
            row["tail_preterminal_capture_rate"] >= ACCEPT_MIN_TAIL_CAPTURE
            and row["overtake_trigger_rate"] <= ACCEPT_MAX_OVERTAKE_TRIGGER
            and row["follow_trigger_rate"] <= ACCEPT_MAX_FOLLOW_TRIGGER
        )
        rows.append(row)
    return rows


def select_setting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["acceptance_pass"]]
    if accepted:
        selected = min(
            accepted,
            key=lambda row: (
                float(row["overtake_trigger_rate"]),
                float(row["follow_trigger_rate"]),
                -float(row["tail_preterminal_capture_rate"]),
                float(row["trigger_step_fraction"]),
                -(float(row["tail_first_signal_lead_s_median"] or 0.0)),
                row["setting"],
            ),
        )
    else:
        selected = min(
            rows,
            key=lambda row: (
                -float(row["tail_preterminal_capture_rate"]),
                float(row["overtake_trigger_rate"]),
                float(row["follow_trigger_rate"]),
                float(row["trigger_step_fraction"]),
                -(float(row["tail_first_signal_lead_s_median"] or 0.0)),
                row["setting"],
            ),
        )
    return {
        **selected,
        "selected_from_accepted_set": bool(accepted),
        "selection_interpretation": (
            "offline_acceptance_candidate"
            if accepted
            else "diagnostic_fallback_only_no_setting_met_acceptance"
        ),
    }


def attach_panel_selectivity_guards(
    aggregate_rows: list[dict[str, Any]],
    panel_rows: list[dict[str, Any]],
) -> None:
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in panel_rows:
        by_setting.setdefault(str(row["setting"]), []).append(row)
    for row in aggregate_rows:
        panels = by_setting.get(str(row["setting"]), [])
        if not panels:
            raise ValueError(f"Missing per-panel rows for {row['setting']}")
        maximum_overtake = max(
            float(panel["overtake_trigger_rate"])
            for panel in panels
        )
        maximum_follow = max(
            float(panel["follow_trigger_rate"])
            for panel in panels
        )
        panel_guard = bool(
            maximum_overtake <= ACCEPT_MAX_OVERTAKE_TRIGGER
            and maximum_follow <= ACCEPT_MAX_FOLLOW_TRIGGER
        )
        row["maximum_panel_overtake_trigger_rate"] = maximum_overtake
        row["maximum_panel_follow_trigger_rate"] = maximum_follow
        row["panel_selectivity_guard_pass"] = panel_guard
        row["acceptance_pass"] = bool(row["acceptance_pass"] and panel_guard)


def selected_episode_rows(
    episodes: list[dict[str, Any]],
    selected: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    name = str(selected["setting"])
    for episode in episodes:
        result = episode["settings"][name]
        output.append(
            {
                "panel": episode["panel"],
                "scenario_id": episode["scenario_id"],
                "episode_key": episode["episode_key"],
                "outcome": episode["outcome"],
                "reference_primary_tail": episode["reference_primary_tail"],
                "transition_count": episode["transition_count"],
                "pass_detected": episode["pass_detected"],
                "terminal_relative_progress_error_m": (
                    episode["terminal_relative_progress_error_m"]
                ),
                "triggered": result["triggered"],
                "preterminal_triggered": result["preterminal_triggered"],
                "trigger_steps": result["trigger_steps"],
                "penalty_sum": result["penalty_sum"],
                "episode_cap_hit": result["episode_cap_hit"],
                "step_cap_hit": result["step_cap_hit"],
                "first_trigger_lead_s": result["first_trigger_lead_s"],
                "gae_delta_sum": result["gae_delta_sum"],
                "gae_delta_min": result["gae_delta_min"],
                "gae_affected_steps": result["gae_affected_steps"],
            }
        )
    return output


def audit_selected_invariants(
    episodes: list[dict[str, Any]],
    selected: dict[str, Any],
) -> dict[str, Any]:
    name = str(selected["setting"])
    nonpositive_reward = True
    nonpositive_gae = True
    caps_valid = True
    bc_tail_teacher_self_loss = []
    for episode in episodes:
        result = episode["settings"][name]
        nonpositive_reward &= bool(result["reward_delta_max"] <= 1e-15)
        nonpositive_gae &= bool(result["gae_delta_max"] <= 1e-15)
        caps_valid &= bool(
            result["episode_penalty_used"] <= 0.05 + 1e-12
            and result["maximum_step_penalty_used"] <= 0.005 + 1e-12
        )
        if episode["panel"] == "BC" and episode["reference_primary_tail"]:
            actions = np.asarray(episode["teacher_actions"], dtype=np.float64)
            trigger_state = np.asarray(result["trigger"], dtype=bool)
            # Reward at state i belongs to action/transition i-1.
            mask = trigger_state[1:]
            loss, gradient = masked_follow_teacher_huber_loss(
                actions,
                actions.copy(),
                mask,
            )
            bc_tail_teacher_self_loss.append(
                {
                    "scenario_id": episode["scenario_id"],
                    "masked_step_count": int(np.count_nonzero(mask)),
                    "loss": loss,
                    "gradient_max_abs": float(
                        np.max(np.abs(gradient), initial=0.0)
                    ),
                }
            )
    if not nonpositive_reward or not nonpositive_gae or not caps_valid:
        raise AssertionError("Selected reward/loss invariants failed")
    if any(
        row["loss"] != 0.0 or row["gradient_max_abs"] != 0.0
        for row in bc_tail_teacher_self_loss
    ):
        raise AssertionError("BC teacher self-loss must be exactly zero")
    return {
        "all_reward_deltas_nonpositive": nonpositive_reward,
        "all_gae_deltas_nonpositive": nonpositive_gae,
        "all_step_and_episode_caps_valid": caps_valid,
        "maximum_terminal_relative_progress_error_m": max(
            episode["terminal_relative_progress_error_m"]
            for episode in episodes
        ),
        "all_trace_member_crcs_verified": all(
            episode["trace_archive"]["all_member_crcs_verified"]
            for episode in episodes
        ),
        "bc_tail_follow_teacher_self_loss": bc_tail_teacher_self_loss,
        "follow_teacher_interpretation": (
            "The BC teacher produces exactly zero imitation loss on its own "
            "unsafe tail episodes, so this loss supplies no corrective signal "
            "at the inherited failure and cannot be treated as a safety loss."
        ),
        "exact_real_training_policy_loss_available": False,
        "missing_training_loss_inputs": [
            "rollout baseline advantages",
            "old log probabilities",
            "new log probabilities",
            "critic value predictions",
        ],
        "loss_boundary": (
            "The reward-to-GAE/return delta is exact with values held fixed. "
            "Exact realized clipped policy/value loss deltas require a training "
            "rollout buffer and are not reconstructed from deterministic eval "
            "traces. The reference value loss matches the unscaled logged MSE; "
            "the production critic gradient multiplies it by 0.5."
        ),
    }


def run_saved_episode_audit(
    root: Path,
    panels: list[str],
    tail_labels_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    process_guard = require_host_process_idle()
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    labels = load_tail_labels(
        tail_labels_path,
        {panel_label(panel) for panel in panels},
    )
    projector_path = (
        root / "f1tenth_racetracks" / "Austin" / "raceline1.csv"
    )
    result_paths = [
        root / "eval_results" / panel / "multiagents" / "results_multi.json"
        for panel in panels
    ]
    protected_inputs = [
        HERE / "shadow_contract.py",
        Path(__file__).resolve(),
        tail_labels_path,
        projector_path,
        root / "ppo" / "algorithm.py",
        *result_paths,
    ]
    input_hashes_before = {
        str(path): _sha256(path)
        for path in protected_inputs
    }
    projector = TrackProjector(projector_path)
    configs = sweep_configs()
    episodes: list[dict[str, Any]] = []
    process_guard_checks = [process_guard]
    for panel, result_path in zip(panels, result_paths):
        episodes.extend(
            load_panel_episodes(
                root,
                panel,
                labels,
                projector,
                configs,
                process_guard_checks,
            )
        )
    process_guard_checks.append(require_host_process_idle())
    if len(episodes) != 600 * len(panels):
        raise AssertionError("Unexpected saved-episode total")
    setting_rows = aggregate_settings(episodes, configs)
    panel_setting_rows = []
    for panel in panels:
        panel_setting_rows.extend(
            aggregate_settings(
                [
                    episode
                    for episode in episodes
                    if episode["directory"] == panel
                ],
                configs,
                scope=panel,
            )
        )
    attach_panel_selectivity_guards(setting_rows, panel_setting_rows)
    selected = select_setting(setting_rows)
    selected_by_panel = [
        row
        for row in panel_setting_rows
        if row["setting"] == selected["setting"]
    ]
    episode_rows = selected_episode_rows(episodes, selected)
    invariants = audit_selected_invariants(episodes, selected)
    trace_manifest_rows = [
        episode["trace_archive"]
        for episode in episodes
    ]
    input_hashes_after = {
        str(path): _sha256(path)
        for path in protected_inputs
    }
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("A protected source changed during saved-episode replay")
    trace_metadata_unchanged = all(
        episode["trace_path"].stat().st_size
        == episode["trace_archive"]["size_bytes"]
        and episode["trace_path"].stat().st_mtime_ns
        == episode["trace_archive"]["mtime_ns"]
        for episode in episodes
    )
    if not trace_metadata_unchanged:
        raise RuntimeError("A protected NPZ trace changed during replay")
    _write_csv(output_dir / "setting_sweep.csv", setting_rows)
    _write_csv(output_dir / "panel_setting_sweep.csv", panel_setting_rows)
    _write_csv(output_dir / "selected_episode_results.csv", episode_rows)
    _write_csv(output_dir / "input_trace_manifest.csv", trace_manifest_rows)
    summary = {
        "status": "passed",
        "contract": {
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "production_value_loss_gradient_coefficient": (
                VALUE_LOSS_COEFFICIENT
            ),
            "acceptance": {
                "minimum_tail_preterminal_capture_rate": ACCEPT_MIN_TAIL_CAPTURE,
                "maximum_overtake_trigger_rate": ACCEPT_MAX_OVERTAKE_TRIGGER,
                "maximum_follow_trigger_rate": ACCEPT_MAX_FOLLOW_TRIGGER,
                "same_overtake_and_follow_limits_required_in_every_panel": True,
            },
            "panel_count": len(panels),
            "episode_count": len(episodes),
            "setting_count": len(configs),
            "output_isolation_root": str(HERE),
            "host_process_guard": process_guard,
            "host_process_guard_check_count": len(process_guard_checks),
            "host_process_guard_minimum_visible_processes": min(
                receipt["numeric_proc_entry_count"]
                for receipt in process_guard_checks
            ),
        },
        "input_fingerprints": {
            "protected_inputs_before": input_hashes_before,
            "protected_inputs_after": input_hashes_after,
            "trace_manifest": _sha256(output_dir / "input_trace_manifest.csv"),
        },
        "input_quality": {
            "all_npz_member_crcs_verified": True,
            "exact_npz_member_schema_verified": True,
            "trace_json_step_time_collision_reconciled": True,
            "terminal_relative_progress_reconciled": True,
            "collision_label_coverage_reconciled": True,
            "observation_and_action_finite_flags_required": True,
            "protected_sources_unchanged_during_replay": True,
            "trace_size_and_mtime_unchanged_during_replay": (
                trace_metadata_unchanged
            ),
            "trace_count": len(trace_manifest_rows),
            "trace_archive_size_bytes": sum(
                row["size_bytes"] for row in trace_manifest_rows
            ),
            "trace_uncompressed_member_bytes": sum(
                row["uncompressed_bytes"] for row in trace_manifest_rows
            ),
        },
        "panel_outcomes": {
            panel: dict(
                Counter(
                    episode["outcome"]
                    for episode in episodes
                    if episode["directory"] == panel
                )
            )
            for panel in panels
        },
        "reference_primary_tail_count": sum(
            episode["reference_primary_tail"] for episode in episodes
        ),
        "candidate_decision": (
            "eligible_for_bounded_live_consistency_probe_only"
            if selected["selected_from_accepted_set"]
            else "rejected_by_offline_selectivity_gate"
        ),
        "production_integration_authorized": False,
        "selected_setting": selected,
        "selected_setting_by_panel": selected_by_panel,
        "selected_invariants": invariants,
        "setting_sweep_path": str(output_dir / "setting_sweep.csv"),
        "panel_setting_sweep_path": str(
            output_dir / "panel_setting_sweep.csv"
        ),
        "selected_episode_results_path": str(
            output_dir / "selected_episode_results.csv"
        ),
        "input_trace_manifest_path": str(
            output_dir / "input_trace_manifest.csv"
        ),
    }
    _atomic_json(output_dir / "saved_episode_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if root != PROJECT_ROOT:
        required = (root / "eval_results", root / "f1tenth_racetracks")
        if not all(path.exists() for path in required):
            raise ValueError(f"Invalid End2Race root: {root}")
    if HERE not in output_dir.parents and output_dir != HERE / "outputs":
        raise ValueError(
            f"Output must remain beneath the isolated directory: {HERE}"
        )
    candidate = (
        None
        if args.candidate_module is None
        else resolve(root, args.candidate_module)
    )
    with exclusive_output_lock(output_dir):
        receipt: dict[str, Any] = {
            "mode": args.mode,
            "unit": None,
            "saved_episodes": None,
        }
        if args.mode in {"unit", "all"}:
            receipt["unit"] = run_unit_tests(candidate)
            _atomic_json(output_dir / "unit_receipt.json", receipt["unit"])
        if args.mode in {"saved-episodes", "all"}:
            tail_labels = resolve(root, args.tail_labels)
            receipt["saved_episodes"] = run_saved_episode_audit(
                root,
                list(args.panels),
                tail_labels,
                output_dir,
            )
        _atomic_json(output_dir / "validation_receipt.json", receipt)
    print(json.dumps(_json_ready(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

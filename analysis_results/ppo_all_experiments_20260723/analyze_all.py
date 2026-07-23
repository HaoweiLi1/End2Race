#!/usr/bin/env python3
"""Audit every current PPO run/eval and diagnose post-overtake rear-sweep collisions.

The script treats ``results_multi.json`` as terminal outcome truth and uses NPZ
poses/actions for kinematic mechanism analysis.  Legacy 0721 traces do not have
the terminal post-step row, so their collision bit is never used as the label.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable
import zipfile

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binomtest, pearsonr, spearmanr
import torch


ROOT = Path(__file__).resolve().parents[2]
POST_TRAINED = ROOT / "post-trained"
EVAL_RESULTS = ROOT / "eval_results"
OUT = ROOT / "analysis_results" / "ppo_all_experiments_20260723"
BC_MODEL = ROOT / "pretrained" / "end2race.pth"
RACELINE = ROOT / "f1tenth_racetracks" / "Austin" / "raceline1.csv"

PANEL_PATTERN = re.compile(r"(?P<run>ppo_.+)_u(?P<update>\d{4})_Austin$")
SCENARIO_PATTERN = re.compile(
    r"evaluation-sp(?P<startpoint>\d+)-ego(?P<ego>\d+)-"
    r"raceline(?P<raceline>\d+)-v(?P<speed>[0-9.]+)$"
)

TRAINABLE_PREFIXES = ("gru.", "output_layer.")
CHECKPOINT_UPDATES = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.number)):
        return bool(np.isfinite(value))
    if isinstance(value, dict):
        return all(finite_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(child) for child in value)
    return True


def mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(clean)) if clean else None


def median(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(statistics.median(clean)) if clean else None


def quantile(values: Iterable[float | int | None], q: float) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.quantile(clean, q)) if clean else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(events: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = events / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def exact_paired_p(resolved: int, created: int) -> float:
    discordant = resolved + created
    if discordant == 0:
        return 1.0
    return float(binomtest(min(resolved, created), discordant, 0.5, alternative="two-sided").pvalue)


def holm_adjust(rows: list[dict[str, Any]], p_field: str, output_field: str) -> None:
    indexed = [(index, float(row[p_field])) for index, row in enumerate(rows) if row.get("valid")]
    indexed.sort(key=lambda item: item[1])
    adjusted = [math.nan] * len(rows)
    running = 0.0
    total = len(indexed)
    for rank, (index, p_value) in enumerate(indexed):
        running = max(running, (total - rank) * p_value)
        adjusted[index] = min(1.0, running)
    for index, row in enumerate(rows):
        row[output_field] = adjusted[index]


def state_delta_stats(
    state: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    keys: Iterable[str],
) -> dict[str, float]:
    delta_sq = 0.0
    baseline_sq = 0.0
    max_abs = 0.0
    count = 0
    for key in keys:
        delta = (state[key] - baseline[key]).double()
        reference = baseline[key].double()
        delta_sq += float(torch.sum(delta * delta))
        baseline_sq += float(torch.sum(reference * reference))
        max_abs = max(max_abs, float(delta.abs().max()))
        count += delta.numel()
    return {
        "delta_l2": math.sqrt(delta_sq),
        "relative_l2": math.sqrt(delta_sq / baseline_sq) if baseline_sq else math.nan,
        "delta_rms": math.sqrt(delta_sq / count) if count else math.nan,
        "max_abs_delta": max_abs,
        "parameter_count": count,
    }


def wrap_angle(values: np.ndarray | float) -> np.ndarray | float:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


class TrackProjector:
    """Fast exact-nearby segment projection onto the dense Austin raceline."""

    def __init__(self, path: Path) -> None:
        reference = np.loadtxt(path, delimiter=";", comments="#", dtype=np.float64)
        self.track_length = float(reference[-1, 0])
        if np.linalg.norm(reference[-1, 1:3] - reference[0, 1:3]) <= 1e-9:
            reference = reference[:-1]
        self.progress = reference[:, 0]
        self.points = reference[:, 1:3]
        self.headings = reference[:, 3]
        self.tree = cKDTree(self.points)
        self.segment_vectors = np.roll(self.points, -1, axis=0) - self.points
        self.segment_norm_sq = np.einsum("ij,ij->i", self.segment_vectors, self.segment_vectors)
        self.segment_progress = np.concatenate(
            (np.diff(self.progress), np.asarray([self.track_length - self.progress[-1]]))
        )

    def project(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(query, dtype=np.float64)
        _distance, nearest = self.tree.query(points)
        candidates = np.column_stack((nearest, (nearest - 1) % len(self.points)))
        starts = self.points[candidates]
        vectors = self.segment_vectors[candidates]
        offset = points[:, None, :] - starts
        fraction = np.clip(
            np.einsum("nci,nci->nc", offset, vectors) / self.segment_norm_sq[candidates],
            0.0,
            1.0,
        )
        closest = starts + fraction[..., None] * vectors
        distance_sq = np.einsum("nci,nci->nc", points[:, None, :] - closest, points[:, None, :] - closest)
        choice = np.argmin(distance_sq, axis=1)
        row = np.arange(len(points))
        segment = candidates[row, choice]
        selected_fraction = fraction[row, choice]
        projected = closest[row, choice]
        progress = (
            self.progress[segment] + selected_fraction * self.segment_progress[segment]
        ) % self.track_length
        direction = self.segment_vectors[segment]
        heading = np.arctan2(direction[:, 1], direction[:, 0])
        normal = np.column_stack((-np.sin(heading), np.cos(heading)))
        lateral = np.einsum("ni,ni->n", points - projected, normal)
        return progress, lateral, heading

    def relative_progress(self, ego_xy: np.ndarray, opponent_xy: np.ndarray) -> np.ndarray:
        ego_progress, _ego_lateral, _ego_heading = self.project(ego_xy)
        opponent_progress, _opponent_lateral, _opponent_heading = self.project(opponent_xy)
        raw = (
            ego_progress - opponent_progress + 0.5 * self.track_length
        ) % self.track_length - 0.5 * self.track_length
        unwrapped = np.empty_like(raw)
        unwrapped[0] = raw[0]
        for index in range(1, len(raw)):
            delta = (
                raw[index] - raw[index - 1] + 0.5 * self.track_length
            ) % self.track_length - 0.5 * self.track_length
            unwrapped[index] = unwrapped[index - 1] + delta
        return unwrapped


def parse_scenario(scenario_id: str) -> dict[str, Any]:
    match = SCENARIO_PATTERN.fullmatch(scenario_id)
    if match is None:
        raise ValueError(f"Unexpected scenario id: {scenario_id}")
    return {
        "startpoint": int(match.group("startpoint")),
        "ego_idx": int(match.group("ego")),
        "opponent_raceline": f"raceline{match.group('raceline')}",
        "opponent_speed_scale": float(match.group("speed")),
    }


def panel_identity(directory: Path) -> tuple[str, int]:
    if directory.name == "end2race_Austin":
        return "BC", 0
    match = PANEL_PATTERN.fullmatch(directory.name)
    if match is None:
        raise ValueError(f"Unexpected evaluation directory: {directory}")
    return match.group("run"), int(match.group("update"))


def actor_checkpoint(run: str, update: int) -> Path:
    if run == "BC":
        return BC_MODEL
    return POST_TRAINED / run / "checkpoints" / f"actor_u{update:04d}.pth"


def read_runs() -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    run_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    training_by_run: dict[str, list[dict[str, Any]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for config_path in sorted(POST_TRAINED.glob("ppo_*/run_config.json")):
        run = config_path.parent.name
        config = load_json(config_path)
        configs[run] = config
        args = config["args"]
        metrics = read_jsonl(config_path.parent / "metrics.jsonl")
        warmup = [row for row in metrics if row.get("phase") == "warmup"]
        formal = [row for row in metrics if row.get("phase") == "formal"]
        training_by_run[run] = formal
        expected_updates = int(args["num_updates"])
        actor_files = sorted((config_path.parent / "checkpoints").glob("actor_u*.pth"))
        critic_files = sorted((config_path.parent / "checkpoints").glob("critic_u*.pt"))
        collision_info = load_json(config_path.parent / "collision_cache_info.json")
        run_row = {
            "run": run,
            "started_at": config.get("started_at"),
            "critic": args["critic"],
            "hidden_scale": args["hidden_scale"],
            "n_envs": args["n_envs"],
            "env_workers": args["env_workers"],
            "n_steps": args["n_steps"],
            "batch_size": args["batch_size"],
            "num_updates": expected_updates,
            "actor_epochs": args["actor_epochs"],
            "critic_epochs": args["critic_epochs"],
            "gru_learning_rate": args["gru_learning_rate"],
            "head_learning_rate": args["head_learning_rate"],
            "critic_learning_rate": args["critic_learning_rate"],
            "steering_latent_std": args["steering_latent_std"],
            "speed_physical_std": args["speed_physical_std"],
            "gamma": args["gamma"],
            "gae_lambda": args["gae_lambda"],
            "clip_range": args["clip_range"],
            "target_kl": args.get("target_kl"),
            "hard_neighbors": bool(args.get("hard_neighbors", False)),
            "collision_pool_count": collision_info.get("collision_count"),
            "base_collision_pool_count": collision_info.get("base_collision_count"),
            "boundary_collision_count": collision_info.get("boundary_collision_count", 0),
            "seed_recorded": "seed" in args,
            "source_commit_recorded": any(key in config for key in ("git_commit", "source_commit", "commit")),
            "warmup_rows": len(warmup),
            "formal_rows": len(formal),
            "formal_updates_sequential": [int(row["update"]) for row in formal] == list(range(1, expected_updates + 1)),
            "metrics_finite": finite_tree(metrics),
            "actor_checkpoint_count": len(actor_files),
            "critic_checkpoint_count": len(critic_files),
            "checkpoint_complete": len(actor_files) == expected_updates and len(critic_files) == expected_updates,
            "warmup_best_validation_loss": warmup[0].get("best_validation_loss") if len(warmup) == 1 else None,
            "early_stop_updates": sum(bool(row.get("actor_early_stop_triggered")) for row in formal),
            "actor_steps_completed": sum(int(row.get("actor_optimizer_steps_completed", 0)) for row in formal),
            "actor_steps_planned": sum(int(row.get("actor_optimizer_steps_planned", 0)) for row in formal),
            "mean_approx_kl": mean(row.get("approx_kl_mean") for row in formal),
            "max_approx_kl": max(float(row["approx_kl_max"]) for row in formal),
            "mean_clip_fraction": mean(row.get("clip_fraction_mean") for row in formal),
            "final_explained_variance_post": formal[-1].get("explained_variance_post_update"),
            "final_value_loss_post": formal[-1].get("value_loss_post_update"),
            "mean_training_collision_rate": mean(
                float(row["ego_collision_count"]) / float(row["episode_count"])
                for row in formal
                if row.get("episode_count")
            ),
            "rollout_wall_seconds_total": sum(float(row.get("rollout_wall_seconds", 0.0)) for row in formal),
            "actor_wall_seconds_total": sum(float(row.get("actor_train_wall_seconds", 0.0)) for row in formal),
            "critic_wall_seconds_total": sum(float(row.get("critic_train_wall_seconds", 0.0)) for row in formal),
            "collision_scenarios_sha256": sha256(config_path.parent / "collision_scenarios.json"),
            "ordinary_scenarios_sha256": sha256(config_path.parent / "ordinary_scenarios.json"),
        }
        run_rows.append(run_row)
        for row in formal:
            training_rows.append(
                {
                    "run": run,
                    "update": int(row["update"]),
                    "formal_training_timesteps": row.get("formal_training_timesteps"),
                    "policy_gradient_loss": row.get("policy_gradient_loss"),
                    "value_loss_pre": row.get("value_loss_pre_update"),
                    "value_loss_post": row.get("value_loss_post_update"),
                    "explained_variance_pre": row.get("explained_variance_pre_update"),
                    "explained_variance_post": row.get("explained_variance_post_update"),
                    "collision_value_loss_pre": row.get("collision_value_loss_pre"),
                    "collision_value_loss_post": row.get("collision_value_loss_post"),
                    "ordinary_value_loss_pre": row.get("ordinary_value_loss_pre"),
                    "ordinary_value_loss_post": row.get("ordinary_value_loss_post"),
                    "collision_explained_variance_post": row.get("collision_explained_variance_post"),
                    "ordinary_explained_variance_post": row.get("ordinary_explained_variance_post"),
                    "approx_kl_mean": row.get("approx_kl_mean"),
                    "approx_kl_max": row.get("approx_kl_max"),
                    "clip_fraction_mean": row.get("clip_fraction_mean"),
                    "clip_fraction_max": row.get("clip_fraction_max"),
                    "actor_grad_norm_mean_preclip": row.get("actor_grad_norm_mean"),
                    "actor_grad_norm_max_preclip": row.get("actor_grad_norm_max"),
                    "critic_grad_norm_mean_preclip": row.get("critic_grad_norm_mean"),
                    "actor_optimizer_steps_completed": row.get("actor_optimizer_steps_completed"),
                    "actor_optimizer_steps_planned": row.get("actor_optimizer_steps_planned"),
                    "actor_early_stop_triggered": row.get("actor_early_stop_triggered"),
                    "training_ego_collision_count": row.get("ego_collision_count"),
                    "training_episode_count": row.get("episode_count"),
                    "training_collision_rate": (
                        float(row["ego_collision_count"]) / float(row["episode_count"])
                        if row.get("episode_count") else None
                    ),
                    "mean_episode_return": row.get("mean_episode_return"),
                    "mean_relative_position_m": row.get("mean_relative_position_m"),
                    "mean_episode_min_obb_clearance_m": row.get("mean_episode_min_obb_clearance_m"),
                    "mean_episode_min_wall_clearance_m": row.get("mean_episode_min_wall_clearance_m"),
                    "mean_episode_risk_active_fraction": row.get("mean_episode_risk_active_fraction"),
                    "rollout_wall_seconds": row.get("rollout_wall_seconds"),
                    "actor_train_wall_seconds": row.get("actor_train_wall_seconds"),
                    "critic_train_wall_seconds": row.get("critic_train_wall_seconds"),
                }
            )
    return run_rows, training_by_run, configs, training_rows


def read_panels(configs: dict[str, dict[str, Any]]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, int], dict[str, dict[str, Any]]],
]:
    panel_directories = sorted(
        path for path in EVAL_RESULTS.iterdir()
        if path.is_dir() and (path / "multiagents" / "results_multi.json").is_file()
    )
    panel_directories.sort(key=lambda path: (path.name != "end2race_Austin", path.name))
    panel_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    episodes_by_panel: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    bc_scenarios: set[str] | None = None
    for directory in panel_directories:
        run, update = panel_identity(directory)
        if run != "BC" and run not in configs:
            raise RuntimeError(f"Evaluation has no training config: {directory.name}")
        multiagents = directory / "multiagents"
        document = load_json(multiagents / "results_multi.json")
        episodes = document.get("episodes", {})
        final = document.get("final", {})
        rows = list(episodes.values())
        scenario_ids = [str(row.get("scenario_id")) for row in rows]
        unique_scenarios = set(scenario_ids)
        if run == "BC":
            bc_scenarios = unique_scenarios
        assert bc_scenarios is not None
        trace_stems = {path.stem for path in (multiagents / "traces").glob("*.npz")}
        episode_keys = {str(row.get("episode_key")) for row in rows}
        collision_rows = [row for row in rows if row.get("ego_collision_occurred") is True]
        collision_count = len(collision_rows)
        overtake_count = sum(row.get("outcome") == "overtake" for row in rows)
        follow_count = sum(row.get("outcome") == "follow" for row in rows)
        aggregate_consistent = (
            final.get("collision_count") == collision_count
            and final.get("overtaking_count") == overtake_count
            and final.get("following_count") == follow_count
            and final.get("success_count") == overtake_count + follow_count
            and final.get("total_episodes") == collision_count + overtake_count + follow_count + int(final.get("error_count", 0))
        )
        valid = (
            len(rows) == 600
            and len(unique_scenarios) == 600
            and final.get("total_episodes") == 600
            and final.get("error_count") == 0
            and aggregate_consistent
            and all(row.get("action_finite") is True and row.get("observation_finite") is True for row in rows)
            and trace_stems == episode_keys
            and (run == "BC" or unique_scenarios == bc_scenarios)
        )
        checkpoint = actor_checkpoint(run, update)
        policy_hash = sha256(checkpoint)
        result_hash = sha256(multiagents / "results_multi.json")
        low, high = wilson_interval(collision_count, 600)
        panel_name = "BC" if run == "BC" else f"{run}_u{update:04d}"
        panel_rows.append(
            {
                "panel": panel_name,
                "eval_directory": directory.name,
                "run": run,
                "update": update,
                "policy_sha256": policy_hash,
                "results_sha256": result_hash,
                "episode_rows": len(rows),
                "unique_scenarios": len(unique_scenarios),
                "trace_files": len(trace_stems),
                "missing_scenarios_vs_bc": ";".join(sorted(bc_scenarios - unique_scenarios)),
                "extra_scenarios_vs_bc": ";".join(sorted(unique_scenarios - bc_scenarios)),
                "collision_count": collision_count,
                "collision_rate": collision_count / 600.0,
                "collision_rate_ci_low": low,
                "collision_rate_ci_high": high,
                "overtake_count": overtake_count,
                "follow_count": follow_count,
                "success_count": overtake_count + follow_count,
                "success_rate": (overtake_count + follow_count) / 600.0,
                "error_count": final.get("error_count"),
                "collision_with_opponent_count": sum(bool(row.get("opp_collision_occurred")) for row in collision_rows),
                "wall_like_collision_count": sum(not bool(row.get("opp_collision_occurred")) for row in collision_rows),
                "mean_collision_time_s": mean(row.get("ego_collision_time_s") for row in collision_rows),
                "median_collision_time_s": median(row.get("ego_collision_time_s") for row in collision_rows),
                "avg_speed_mean": final.get("avg_speed_mean"),
                "speed_variance_mean": final.get("speed_variance_mean"),
                "total_distance_mean": final.get("total_distance_mean"),
                "near_proximity_episode_count": sum(bool(row.get("proximity_below_threshold_timesteps")) for row in rows),
                "steering_anomaly_episode_count": sum(bool(row.get("steering_anomaly_timesteps")) for row in rows),
                "all_action_observation_finite": all(
                    row.get("action_finite") is True and row.get("observation_finite") is True for row in rows
                ),
                "trace_keys_match_episode_keys": trace_stems == episode_keys,
                "matches_bc_scenario_set": run == "BC" or unique_scenarios == bc_scenarios,
                "aggregate_consistent": aggregate_consistent,
                "valid": valid,
            }
        )
        simplified: dict[str, dict[str, Any]] = {}
        for row in rows:
            scenario_id = str(row["scenario_id"])
            parsed = parse_scenario(scenario_id)
            output = {
                "panel": panel_name,
                "eval_directory": directory.name,
                "run": run,
                "update": update,
                "panel_valid": valid,
                "policy_sha256": policy_hash,
                "episode_key": str(row["episode_key"]),
                "scenario_id": scenario_id,
                **parsed,
                "outcome": row.get("outcome"),
                "ego_collision": bool(row.get("ego_collision_occurred")),
                "opponent_collision": bool(row.get("opp_collision_occurred")),
                "opponent_only_collision": bool(row.get("opponent_only_collision")),
                "initial_ego_collision": bool(row.get("initial_ego_collision")),
                "ego_collision_time_s": row.get("ego_collision_time_s"),
                "simulation_time_s": row.get("simulation_time_s"),
                "steps": row.get("steps"),
                "final_relative_position_m": row.get("final_relative_position_m"),
                "avg_speed": row.get("avg_speed"),
                "speed_variance": row.get("speed_variance"),
                "total_distance": row.get("total_distance"),
                "global_min_surface_dist": row.get("global_min_surface_dist"),
                "ego_min_lidar": row.get("ego_min_lidar"),
                "ego_max_abs_steer": row.get("ego_max_abs_steer"),
                "max_steer_delta": row.get("max_steer_delta"),
                "max_steer_reversals": row.get("max_steer_reversals"),
                "steer_autocorr_lag1": row.get("steer_autocorr_lag1"),
                "near_proximity": bool(row.get("proximity_below_threshold_timesteps")),
                "steering_anomaly": bool(row.get("steering_anomaly_timesteps")),
                "trace_path": str((multiagents / "traces" / f"{row['episode_key']}.npz").relative_to(ROOT)),
            }
            episode_rows.append(output)
            simplified[scenario_id] = output
        episodes_by_panel[(run, update)] = simplified
    return panel_rows, episode_rows, episodes_by_panel


def analyze_collision_trace(episode: dict[str, Any], projector: TrackProjector) -> dict[str, Any]:
    trace_path = ROOT / episode["trace_path"]
    with np.load(trace_path, allow_pickle=False) as arrays:
        time_s = np.asarray(arrays["time_s"], dtype=np.float64)
        ego_pose = np.asarray(arrays["ego_pose"], dtype=np.float64)
        opponent_pose = np.asarray(arrays["opp_pose"], dtype=np.float64)
        ego_action = np.asarray(arrays["ego_executed_action"], dtype=np.float64)
        collisions = np.asarray(arrays["collisions"], dtype=bool)
        is_post_step_v2 = "terminal_post_step" in arrays.files and "action_applied" in arrays.files
        if is_post_step_v2:
            terminal_marker = np.asarray(arrays["terminal_post_step"], dtype=bool)
            action_applied = np.asarray(arrays["action_applied"], dtype=bool)
        else:
            terminal_marker = np.zeros(len(time_s), dtype=bool)
            action_applied = np.ones(len(time_s), dtype=bool)

    ego_progress, ego_lateral, ego_track_heading = projector.project(ego_pose[:, :2])
    opponent_progress, opponent_lateral, opponent_track_heading = projector.project(opponent_pose[:, :2])
    raw_relative = (
        ego_progress - opponent_progress + 0.5 * projector.track_length
    ) % projector.track_length - 0.5 * projector.track_length
    relative_progress = np.empty_like(raw_relative)
    relative_progress[0] = raw_relative[0]
    for index in range(1, len(raw_relative)):
        relative_progress[index] = relative_progress[index - 1] + (
            raw_relative[index] - raw_relative[index - 1] + 0.5 * projector.track_length
        ) % projector.track_length - 0.5 * projector.track_length

    relative_world = opponent_pose[:, :2] - ego_pose[:, :2]
    cosine = np.cos(ego_pose[:, 2])
    sine = np.sin(ego_pose[:, 2])
    opponent_body_x = relative_world[:, 0] * cosine + relative_world[:, 1] * sine
    opponent_body_y = -relative_world[:, 0] * sine + relative_world[:, 1] * cosine
    center_distance = np.linalg.norm(relative_world, axis=1)
    heading_difference = wrap_angle(opponent_pose[:, 2] - ego_pose[:, 2])
    ego_heading_error = wrap_angle(ego_pose[:, 2] - ego_track_heading)

    crossing = np.flatnonzero((relative_progress[:-1] <= 0.0) & (relative_progress[1:] > 0.0))
    pass_index = int(crossing[0] + 1) if crossing.size else None
    collision_time = float(episode["ego_collision_time_s"])
    pass_time = float(time_s[pass_index]) if pass_index is not None else None
    pass_to_collision_s = collision_time - pass_time if pass_time is not None else None
    lateral_gap = np.abs(ego_lateral - opponent_lateral)
    lateral_convergence = (
        float(lateral_gap[pass_index] - lateral_gap[-1]) if pass_index is not None else None
    )

    strict_candidates = np.flatnonzero(relative_progress >= 0.1)
    strict_index = None
    if strict_candidates.size:
        candidate = int(strict_candidates[0])
        if float(np.min(relative_progress[: candidate + 1])) <= -0.1:
            strict_index = candidate
    strict_lead_s = collision_time - float(time_s[strict_index]) if strict_index is not None else None
    strict_lateral_convergence = (
        float(lateral_gap[strict_index] - lateral_gap[-1]) if strict_index is not None else None
    )

    opponent_contact = bool(episode["opponent_collision"])
    opponent_behind = bool(opponent_body_x[-1] < 0.0)
    post_pass_rear_contact = bool(
        pass_index is not None
        and pass_to_collision_s is not None
        and pass_to_collision_s >= 0.10
        and opponent_contact
        and opponent_behind
    )
    merge_tail_relaxed = bool(
        pass_index is not None
        and pass_to_collision_s is not None
        and pass_to_collision_s >= 0.05
        and opponent_contact
        and opponent_behind
        and lateral_convergence is not None
        and lateral_convergence >= 0.05
    )
    merge_tail_primary = bool(
        post_pass_rear_contact
        and lateral_convergence is not None
        and lateral_convergence >= 0.10
    )
    merge_tail_strict = bool(
        strict_index is not None
        and strict_lead_s is not None
        and strict_lead_s >= 0.15
        and opponent_contact
        and opponent_behind
        and strict_lateral_convergence is not None
        and strict_lateral_convergence >= 0.10
    )

    if merge_tail_primary:
        collision_class = "post_overtake_merge_rear_sweep"
    elif post_pass_rear_contact:
        collision_class = "post_overtake_rear_contact_without_merge_threshold"
    elif opponent_contact:
        collision_class = "prepass_or_front_side_opponent_contact"
    else:
        collision_class = "ego_only_wall_like"

    # Pose-derived 0.10 s kinematics.  This is a slip proxy, not the simulator's
    # internal tire slip-angle state (which the evaluator does not save).
    target_time = float(time_s[-1] - 0.10)
    window_start = int(np.searchsorted(time_s, target_time, side="left"))
    window_start = min(window_start, len(time_s) - 2)
    delta_t = max(float(time_s[-1] - time_s[window_start]), 1e-9)
    displacement = ego_pose[-1, :2] - ego_pose[window_start, :2]
    motion_heading = math.atan2(float(displacement[1]), float(displacement[0]))
    kinematic_slip_proxy = float(wrap_angle(motion_heading - ego_pose[-1, 2]))
    yaw_rate = float(wrap_angle(ego_pose[-1, 2] - ego_pose[window_start, 2]) / delta_t)
    tail_xy = ego_pose[:, :2] - 0.29 * np.column_stack((cosine, sine))
    tail_displacement = tail_xy[-1] - tail_xy[window_start]
    terminal_lateral_axis = np.asarray((-math.sin(ego_pose[-1, 2]), math.cos(ego_pose[-1, 2])))
    tail_lateral_velocity = float(np.dot(tail_displacement, terminal_lateral_axis) / delta_t)

    applied_actions = ego_action[action_applied]
    terminal_collision_bit = bool(collisions[-1, 0]) if is_post_step_v2 else bool(np.any(collisions[:, 0]))
    terminal_valid = None
    if is_post_step_v2:
        terminal_valid = bool(
            np.count_nonzero(terminal_marker) == 1
            and terminal_marker[-1]
            and np.count_nonzero(action_applied) == len(action_applied) - 1
            and not action_applied[-1]
        )
    return {
        "panel": episode["panel"],
        "run": episode["run"],
        "update": episode["update"],
        "panel_valid": episode["panel_valid"],
        "policy_sha256": episode["policy_sha256"],
        "scenario_id": episode["scenario_id"],
        "episode_key": episode["episode_key"],
        "trace_format": "post_step_v2" if is_post_step_v2 else "legacy_pre_post_step",
        "trace_length": len(time_s),
        "terminal_marker_valid": terminal_valid,
        "trace_collision_marker_matches_json": terminal_collision_bit == bool(episode["ego_collision"]),
        "collision_time_s": collision_time,
        "trace_end_time_s": float(time_s[-1]),
        "trace_end_lag_to_collision_s": collision_time - float(time_s[-1]),
        "opponent_collision": opponent_contact,
        "pass_detected": pass_index is not None,
        "pass_time_s": pass_time,
        "pass_to_collision_s": pass_to_collision_s,
        "max_relative_progress_m": float(np.max(relative_progress)),
        "terminal_relative_progress_m": float(relative_progress[-1]),
        "json_terminal_relative_progress_m": episode["final_relative_position_m"],
        "relative_progress_terminal_abs_error_m": abs(
            float(relative_progress[-1]) - float(episode["final_relative_position_m"])
        ),
        "lateral_gap_at_pass_m": float(lateral_gap[pass_index]) if pass_index is not None else None,
        "terminal_lateral_gap_m": float(lateral_gap[-1]),
        "post_pass_lateral_convergence_m": lateral_convergence,
        "terminal_opponent_body_x_m": float(opponent_body_x[-1]),
        "terminal_opponent_body_y_m": float(opponent_body_y[-1]),
        "terminal_center_distance_m": float(center_distance[-1]),
        "terminal_heading_difference_rad": float(heading_difference[-1]),
        "terminal_ego_track_heading_error_rad": float(ego_heading_error[-1]),
        "terminal_abs_kinematic_slip_proxy_rad": abs(kinematic_slip_proxy),
        "terminal_yaw_rate_proxy_radps": yaw_rate,
        "terminal_tail_lateral_velocity_proxy_mps": tail_lateral_velocity,
        "terminal_abs_steer_command_rad": float(abs(applied_actions[-1, 0])) if len(applied_actions) else None,
        "post_pass_rear_contact": post_pass_rear_contact,
        "merge_tail_relaxed": merge_tail_relaxed,
        "merge_tail_primary": merge_tail_primary,
        "merge_tail_strict": merge_tail_strict,
        "collision_class": collision_class,
        "trace_path": episode["trace_path"],
    }


def analyze_collision_traces(
    panel_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], dict[str, Any]]]:
    projector = TrackProjector(RACELINE)
    collision_rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for episode in episode_rows:
        if not episode["ego_collision"]:
            continue
        trace = analyze_collision_trace(episode, projector)
        collision_rows.append(trace)
        lookup[(episode["run"], int(episode["update"]), episode["scenario_id"])] = trace
        episode.update(
            {
                "collision_class": trace["collision_class"],
                "post_pass_rear_contact": trace["post_pass_rear_contact"],
                "merge_tail_relaxed": trace["merge_tail_relaxed"],
                "merge_tail_primary": trace["merge_tail_primary"],
                "merge_tail_strict": trace["merge_tail_strict"],
            }
        )
    for episode in episode_rows:
        if episode["ego_collision"]:
            continue
        episode.update(
            {
                "collision_class": "none",
                "post_pass_rear_contact": False,
                "merge_tail_relaxed": False,
                "merge_tail_primary": False,
                "merge_tail_strict": False,
            }
        )

    collisions_by_panel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in collision_rows:
        collisions_by_panel[row["panel"]].append(row)
    for panel in panel_rows:
        rows = collisions_by_panel[panel["panel"]]
        panel["post_pass_rear_contact_count"] = sum(bool(row["post_pass_rear_contact"]) for row in rows)
        panel["merge_tail_relaxed_count"] = sum(bool(row["merge_tail_relaxed"]) for row in rows)
        panel["merge_tail_primary_count"] = sum(bool(row["merge_tail_primary"]) for row in rows)
        panel["merge_tail_strict_count"] = sum(bool(row["merge_tail_strict"]) for row in rows)
        panel["merge_tail_primary_share_of_collisions"] = (
            panel["merge_tail_primary_count"] / panel["collision_count"]
            if panel["collision_count"] else 0.0
        )

    bc = next(row for row in panel_rows if row["run"] == "BC")
    expected = {
        "merge_tail_relaxed_count": 11,
        "merge_tail_primary_count": 11,
        "merge_tail_strict_count": 8,
    }
    actual = {key: bc[key] for key in expected}
    if actual != expected:
        raise RuntimeError(f"BC tail-signature calibration changed: expected {expected}, got {actual}")
    return collision_rows, lookup


def numpy_header(stream) -> tuple[tuple[int, ...], np.dtype]:
    major, minor = np.lib.format.read_magic(stream)
    if (major, minor) == (1, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif (major, minor) == (2, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        shape, _fortran, dtype = np.lib.format._read_array_header(stream, (major, minor))
    return tuple(shape), np.dtype(dtype)


def audit_trace_file(path: Path, expected_collision: bool) -> dict[str, Any]:
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, np.dtype] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as stream:
                shape, dtype = numpy_header(stream)
            shapes[member[:-4]] = shape
            dtypes[member[:-4]] = dtype
    leading = {shape[0] for shape in shapes.values() if shape}
    numeric = all(
        np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)
        for dtype in dtypes.values()
    )
    is_new = "terminal_post_step" in shapes and "action_applied" in shapes
    terminal_valid = None
    with np.load(path, allow_pickle=False) as arrays:
        collisions = np.asarray(arrays["collisions"], dtype=bool)
        if is_new:
            terminal = np.asarray(arrays["terminal_post_step"], dtype=bool)
            applied = np.asarray(arrays["action_applied"], dtype=bool)
            terminal_valid = bool(
                np.count_nonzero(terminal) == 1
                and terminal[-1]
                and np.count_nonzero(applied) == len(applied) - 1
                and not applied[-1]
            )
            collision_matches = bool(collisions[-1, 0]) == expected_collision
        else:
            collision_matches = bool(np.any(collisions[:, 0])) == expected_collision
    return {
        "format": "post_step_v2" if is_new else "legacy_pre_post_step",
        "array_count": len(shapes),
        "aligned": len(leading) == 1,
        "numeric": numeric,
        "terminal_valid": terminal_valid,
        "collision_marker_matches": collision_matches,
    }


def scan_npz(episode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    by_format: dict[str, Counter[str]] = defaultdict(Counter)
    panel_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for episode in episode_rows:
        result = audit_trace_file(ROOT / episode["trace_path"], bool(episode["ego_collision"]))
        fmt = result["format"]
        totals["files"] += 1
        by_format[fmt]["files"] += 1
        panel_counts[episode["panel"]]["files"] += 1
        for field in ("aligned", "numeric", "collision_marker_matches"):
            key = f"{field}_{result[field]}"
            totals[key] += 1
            by_format[fmt][key] += 1
            panel_counts[episode["panel"]][key] += 1
        if result["terminal_valid"] is not None:
            key = f"terminal_valid_{result['terminal_valid']}"
            totals[key] += 1
            by_format[fmt][key] += 1
            panel_counts[episode["panel"]][key] += 1
    issues = [
        {"panel": panel, **dict(counts)}
        for panel, counts in sorted(panel_counts.items())
        if counts.get("aligned_False") or counts.get("numeric_False") or counts.get("terminal_valid_False")
    ]
    return {
        "totals": dict(totals),
        "by_format": {name: dict(counts) for name, counts in sorted(by_format.items())},
        "structural_panel_issues": issues,
        "interpretation": (
            "Legacy 0721 traces omit terminal_post_step/action_applied and usually omit the terminal "
            "collision state; results_multi.json controls collision truth for every format."
        ),
    }


def audit_model_parameters(
    panel_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline = torch.load(BC_MODEL, map_location="cpu", weights_only=True)
    all_keys = list(baseline)
    trainable_keys = [key for key in all_keys if key.startswith(TRAINABLE_PREFIXES)]
    gru_keys = [key for key in trainable_keys if key.startswith("gru.")]
    head_keys = [key for key in trainable_keys if key.startswith("output_layer.")]
    fixed_keys = [key for key in all_keys if key not in trainable_keys]
    model_rows: list[dict[str, Any]] = []
    for panel in panel_rows:
        path = actor_checkpoint(panel["run"], int(panel["update"]))
        state = torch.load(path, map_location="cpu", weights_only=True)
        if list(state) != all_keys:
            raise RuntimeError(f"Actor key mismatch: {path}")
        total = state_delta_stats(state, baseline, trainable_keys)
        gru = state_delta_stats(state, baseline, gru_keys)
        head = state_delta_stats(state, baseline, head_keys)
        fixed = state_delta_stats(state, baseline, fixed_keys)
        model_rows.append(
            {
                "panel": panel["panel"],
                "run": panel["run"],
                "update": panel["update"],
                "policy_sha256": panel["policy_sha256"],
                "trainable_actor_parameter_count": total["parameter_count"],
                "fixed_actor_parameter_count": fixed["parameter_count"],
                "actor_relative_l2_from_bc": total["relative_l2"],
                "actor_delta_rms_from_bc": total["delta_rms"],
                "actor_max_abs_delta_from_bc": total["max_abs_delta"],
                "gru_relative_l2_from_bc": gru["relative_l2"],
                "head_relative_l2_from_bc": head["relative_l2"],
                "fixed_frontend_delta_l2": fixed["delta_l2"],
                "checkpoint_path": str(path.relative_to(ROOT)),
            }
        )
        del state

    critic_rows: list[dict[str, Any]] = []
    run_by_name = {row["run"]: row for row in run_rows}
    for run, run_row in sorted(run_by_name.items()):
        run_dir = POST_TRAINED / run
        final_update = int(run_row["num_updates"])
        warm_path = run_dir / "checkpoints" / "critic_warmup.pt"
        final_path = run_dir / "checkpoints" / f"critic_u{final_update:04d}.pt"
        warm = torch.load(warm_path, map_location="cpu", weights_only=True)
        final = torch.load(final_path, map_location="cpu", weights_only=True)
        keys = list(warm)
        delta = state_delta_stats(final, warm, keys)
        projection = final.get("privileged_projection.weight")
        actor_final = torch.load(run_dir / "actor_final.pth", map_location="cpu", weights_only=True)
        actor_last = torch.load(
            run_dir / "checkpoints" / f"actor_u{final_update:04d}.pth",
            map_location="cpu",
            weights_only=True,
        )
        actor_final_equal = all(torch.equal(actor_final[key], actor_last[key]) for key in actor_final)
        run_row["actor_final_equals_last_checkpoint"] = actor_final_equal
        run_row["actor_final_sha256"] = sha256(run_dir / "actor_final.pth")
        run_row["last_actor_checkpoint_sha256"] = sha256(
            run_dir / "checkpoints" / f"actor_u{final_update:04d}.pth"
        )
        critic_rows.append(
            {
                "run": run,
                "critic": run_row["critic"],
                "final_update": final_update,
                "critic_parameter_count": sum(value.numel() for value in final.values()),
                "critic_relative_l2_from_warmup": delta["relative_l2"],
                "critic_delta_rms_from_warmup": delta["delta_rms"],
                "privileged_projection_l2": (
                    float(torch.linalg.vector_norm(projection.double())) if projection is not None else None
                ),
                "warmup_checkpoint_path": str(warm_path.relative_to(ROOT)),
                "final_checkpoint_path": str(final_path.relative_to(ROOT)),
            }
        )

    architecture = {
        "pretrained_sha256": sha256(BC_MODEL),
        "actor_observation_dimensions": 361,
        "privileged_critic_observation_dimensions": 381,
        "action_dimensions": 2,
        "actor_total_parameter_count": sum(value.numel() for value in baseline.values()),
        "actor_trainable_parameter_count": sum(baseline[key].numel() for key in trainable_keys),
        "actor_fixed_parameter_count": sum(baseline[key].numel() for key in fixed_keys),
        "actor_gru_parameter_count": sum(baseline[key].numel() for key in gru_keys),
        "actor_head_parameter_count": sum(baseline[key].numel() for key in head_keys),
        "actor_state_dict_keys": len(baseline),
        "gru_input_size": 420,
        "gru_hidden_size": 1680,
        "privileged_feature_count": 20,
        "critic_parameter_counts_by_observed_variant": {
            row["critic"]: row["critic_parameter_count"] for row in critic_rows
        },
        "frozen_actor_components": ["k", "speed_mlp", "dummy_embedding"],
        "trainable_actor_components": ["gru", "output_layer"],
    }
    return model_rows, critic_rows, architecture


def trajectory_equivalence(training_by_run: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    pairs = (
        (
            "base_vs_long_clip015_u1_u20",
            "ppo_privilege_gru_0721_base",
            "ppo_privilege_gru_0722_long_clip015",
            range(1, 21),
        ),
        (
            "long_clip020_vs_long45_clip020_u1_u30",
            "ppo_privilege_gru_0722_long_clip020",
            "ppo_privilege_gru_0722_long45_clip020",
            range(1, 31),
        ),
    )
    for name, left_run, right_run, updates in pairs:
        checkpoint_rows = []
        for update in updates:
            left_path = POST_TRAINED / left_run / "checkpoints" / f"actor_u{update:04d}.pth"
            right_path = POST_TRAINED / right_run / "checkpoints" / f"actor_u{update:04d}.pth"
            left_hash = sha256(left_path)
            right_hash = sha256(right_path)
            checkpoint_rows.append(
                {
                    "update": update,
                    "left_sha256": left_hash,
                    "right_sha256": right_hash,
                    "byte_equal": left_hash == right_hash,
                }
            )
        left_metrics = {int(row["update"]): row for row in training_by_run[left_run]}
        right_metrics = {int(row["update"]): row for row in training_by_run[right_run]}
        numeric_equal = []
        for update in updates:
            left = {key: value for key, value in left_metrics[update].items() if key not in ("actor_checkpoint", "critic_checkpoint")}
            right = {key: value for key, value in right_metrics[update].items() if key not in ("actor_checkpoint", "critic_checkpoint")}
            numeric_equal.append(left == right)
        comparisons[name] = {
            "left_run": left_run,
            "right_run": right_run,
            "updates": checkpoint_rows,
            "all_actor_checkpoints_byte_equal": all(row["byte_equal"] for row in checkpoint_rows),
            "all_nonpath_training_metrics_equal": all(numeric_equal),
        }
    return comparisons


def collision_set(panel: dict[str, dict[str, Any]]) -> set[str]:
    return {scenario for scenario, row in panel.items() if row["ego_collision"]}


def tail_set(panel: dict[str, dict[str, Any]]) -> set[str]:
    return {scenario for scenario, row in panel.items() if row["merge_tail_primary"]}


def cluster_bootstrap_difference(
    baseline: dict[str, dict[str, Any]],
    comparison: dict[str, dict[str, Any]],
    field: str,
    seed: int,
    draws: int = 5000,
) -> tuple[float, float]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for scenario in sorted(baseline):
        cluster = scenario.rsplit("-v", 1)[0]
        clusters[cluster].append(int(bool(baseline[scenario][field])) - int(bool(comparison[scenario][field])))
    cluster_values = np.asarray([sum(values) for values in clusters.values()], dtype=np.int16)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.int16)
    for start in range(0, draws, 1000):
        end = min(draws, start + 1000)
        indices = rng.integers(0, len(cluster_values), size=(end - start, len(cluster_values)))
        samples[start:end] = cluster_values[indices].sum(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def paired_vs_bc(
    panel_rows: list[dict[str, Any]],
    episodes_by_panel: dict[tuple[str, int], dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = episodes_by_panel[("BC", 0)]
    bc_collisions = collision_set(baseline)
    bc_tail = tail_set(baseline)
    rows: list[dict[str, Any]] = []
    tail_scenario_rows: list[dict[str, Any]] = []
    panel_by_key = {(row["run"], int(row["update"])): row for row in panel_rows}
    for index, ((run, update), panel) in enumerate(sorted(episodes_by_panel.items())):
        if run == "BC":
            continue
        collisions = collision_set(panel)
        tails = tail_set(panel)
        collision_resolved = len(bc_collisions - collisions)
        collision_created = len(collisions - bc_collisions)
        tail_resolved = len(bc_tail - tails)
        tail_created = len(tails - bc_tail)
        valid = bool(panel_by_key[(run, update)]["valid"])
        collision_ci = (math.nan, math.nan)
        tail_ci = (math.nan, math.nan)
        if valid:
            collision_ci = cluster_bootstrap_difference(baseline, panel, "ego_collision", 1000 + index)
            tail_ci = cluster_bootstrap_difference(baseline, panel, "merge_tail_primary", 2000 + index)
        row = {
            "panel": panel_by_key[(run, update)]["panel"],
            "run": run,
            "update": update,
            "valid": valid,
            "policy_sha256": panel_by_key[(run, update)]["policy_sha256"],
            "bc_collision_count": len(bc_collisions),
            "ppo_collision_count": len(collisions),
            "collision_shared": len(bc_collisions & collisions),
            "collision_resolved": collision_resolved,
            "collision_created": collision_created,
            "collision_net_reduction": len(bc_collisions) - len(collisions),
            "collision_jaccard": len(bc_collisions & collisions) / len(bc_collisions | collisions),
            "collision_exact_p": exact_paired_p(collision_resolved, collision_created),
            "collision_cluster_bootstrap_diff_ci_low": collision_ci[0],
            "collision_cluster_bootstrap_diff_ci_high": collision_ci[1],
            "bc_tail_count": len(bc_tail),
            "ppo_tail_count": len(tails),
            "tail_shared": len(bc_tail & tails),
            "tail_resolved": tail_resolved,
            "tail_created": tail_created,
            "tail_net_reduction": len(bc_tail) - len(tails),
            "tail_jaccard": len(bc_tail & tails) / len(bc_tail | tails) if (bc_tail | tails) else 1.0,
            "tail_exact_p": exact_paired_p(tail_resolved, tail_created),
            "tail_cluster_bootstrap_diff_ci_low": tail_ci[0],
            "tail_cluster_bootstrap_diff_ci_high": tail_ci[1],
            "bc_tail_scenarios_still_any_collision": sum(panel[scenario]["ego_collision"] for scenario in bc_tail),
            "bc_tail_scenarios_now_overtake": sum(panel[scenario]["outcome"] == "overtake" for scenario in bc_tail),
            "bc_tail_scenarios_now_follow": sum(panel[scenario]["outcome"] == "follow" for scenario in bc_tail),
        }
        rows.append(row)
        for scenario in sorted(bc_tail):
            episode = panel[scenario]
            tail_scenario_rows.append(
                {
                    "panel": row["panel"],
                    "run": run,
                    "update": update,
                    "valid": valid,
                    "scenario_id": scenario,
                    "outcome": episode["outcome"],
                    "ego_collision": episode["ego_collision"],
                    "merge_tail_primary": episode["merge_tail_primary"],
                    "collision_class": episode["collision_class"],
                    "final_relative_position_m": episode["final_relative_position_m"],
                }
            )
    holm_adjust(rows, "collision_exact_p", "collision_exact_p_holm_all_valid_panels")
    holm_adjust(rows, "tail_exact_p", "tail_exact_p_holm_all_valid_panels")
    return rows, tail_scenario_rows


def build_group_rows(
    panel_rows: list[dict[str, Any]],
    episodes_by_panel: dict[tuple[str, int], dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    panel_by_key = {(row["run"], int(row["update"])): row for row in panel_rows}
    regular_20 = (1, 5, 10, 15, 20)
    regular_30 = (1, 5, 10, 15, 20, 25, 30)
    regular_45 = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45)

    def one_run(run: str, updates: Iterable[int]) -> list[tuple[str, int]]:
        return [(run, update) for update in updates]

    combined_clip020 = [
        ("ppo_privilege_gru_0722_long_clip020", update) for update in regular_30
    ] + [
        ("ppo_privilege_gru_0722_long45_clip020", update) for update in (35, 40, 45)
    ]
    specifications = [
        ("G1 critic", "independent_gru", one_run("ppo_independent_gru_0721_base", regular_20)),
        ("G1 critic", "privilege_mlp", one_run("ppo_privilege_mlp_0721_base", regular_20)),
        ("G1 critic", "privilege_gru", one_run("ppo_privilege_gru_0721_base", regular_20)),
        ("G2 batch", "batch 12,800", one_run("ppo_privilege_gru_0721_base", regular_20)),
        ("G2 batch", "batch 25,600", one_run("ppo_privilege_gru_0721_bs25600", regular_20)),
        ("G2 batch", "batch 51,200", one_run("ppo_privilege_gru_0721_bs51200", regular_20)),
        ("G3 legacy clip workers=8", "clip 0.10", one_run("ppo_privilege_gru_0721_clip010", regular_20)),
        ("G3 legacy clip workers=8", "clip 0.20", one_run("ppo_privilege_gru_0721_clip020", regular_20)),
        ("G4 actor LR", "LR 1x", one_run("ppo_privilege_gru_0722_lr1_tkloff", regular_20)),
        ("G4 actor LR", "LR 3x", one_run("ppo_privilege_gru_0721_base", regular_20)),
        ("G4 actor LR", "LR 5x", one_run("ppo_privilege_gru_0722_lr5_tkloff", regular_20)),
        ("G5 30U clip", "clip 0.15", one_run("ppo_privilege_gru_0722_long_clip015", regular_30)),
        ("G5 30U clip", "clip 0.20", one_run("ppo_privilege_gru_0722_long_clip020", regular_30)),
        ("G6 target-KL", "off", one_run("ppo_privilege_gru_0721_base", regular_20)),
        ("G6 target-KL", "0.02 workers=8", one_run("ppo_privilege_gru_0722_clip015_tkl002", regular_20)),
        ("G6 target-KL", "0.04", one_run("ppo_privilege_gru_0722_clip015_tkl004", regular_20)),
        ("G7 45U extension", "clip 0.20", combined_clip020),
        ("G8 45U clip", "clip 0.20", combined_clip020),
        ("G8 45U clip", "clip 0.25", one_run("ppo_privilege_gru_0722_long45_clip025", regular_45)),
        ("G9 hard-neighbor", "baseline pool 479", combined_clip020),
        ("G9 hard-neighbor", "boundary-aware pool 805", one_run("ppo_privilege_gru_0722_long45_clip020_hard", regular_45)),
    ]
    output: list[dict[str, Any]] = []
    for group, arm, keys in specifications:
        panels = [panel_by_key[key] for key in keys]
        valid_panels = [panel for panel in panels if panel["valid"]]
        collision_jaccards = []
        tail_jaccards = []
        collision_flips = []
        tail_flips = []
        for left_key, right_key in zip(keys[:-1], keys[1:]):
            left_collision = collision_set(episodes_by_panel[left_key])
            right_collision = collision_set(episodes_by_panel[right_key])
            left_tail = tail_set(episodes_by_panel[left_key])
            right_tail = tail_set(episodes_by_panel[right_key])
            collision_union = left_collision | right_collision
            tail_union = left_tail | right_tail
            collision_jaccards.append(
                len(left_collision & right_collision) / len(collision_union) if collision_union else 1.0
            )
            tail_jaccards.append(len(left_tail & right_tail) / len(tail_union) if tail_union else 1.0)
            collision_flips.append(len(left_collision ^ right_collision))
            tail_flips.append(len(left_tail ^ right_tail))
        late = [panel for panel in valid_panels if int(panel["update"]) >= 25]
        best = min(valid_panels, key=lambda row: (row["collision_count"], row["merge_tail_primary_count"], -row["overtake_count"]))
        final = valid_panels[-1]
        output.append(
            {
                "group": group,
                "arm": arm,
                "source_runs": ";".join(dict.fromkeys(key[0] for key in keys)),
                "updates": "/".join(str(key[1]) for key in keys),
                "valid_path": "/".join("1" if panel["valid"] else "0" for panel in panels),
                "collision_path": " / ".join(str(panel["collision_count"]) for panel in panels),
                "merge_tail_path": " / ".join(str(panel["merge_tail_primary_count"]) for panel in panels),
                "overtake_path": " / ".join(str(panel["overtake_count"]) for panel in panels),
                "mean_collisions_all_valid": mean(panel["collision_count"] for panel in valid_panels),
                "mean_tail_all_valid": mean(panel["merge_tail_primary_count"] for panel in valid_panels),
                "mean_collisions_u25_plus": mean(panel["collision_count"] for panel in late),
                "mean_tail_u25_plus": mean(panel["merge_tail_primary_count"] for panel in late),
                "best_update": best["update"],
                "best_collision_count": best["collision_count"],
                "best_tail_count": best["merge_tail_primary_count"],
                "best_overtake_count": best["overtake_count"],
                "final_update": final["update"],
                "final_collision_count": final["collision_count"],
                "final_tail_count": final["merge_tail_primary_count"],
                "final_overtake_count": final["overtake_count"],
                "mean_adjacent_collision_jaccard": mean(collision_jaccards),
                "mean_adjacent_tail_jaccard": mean(tail_jaccards),
                "mean_adjacent_collision_flips": mean(collision_flips),
                "mean_adjacent_tail_flips": mean(tail_flips),
                "unique_policy_count": len({panel["policy_sha256"] for panel in panels}),
            }
        )
    return output


def group_control_audit(configs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ignored = {"output_dir", "num_updates"}

    def differences(left: str, right: str) -> list[str]:
        first = configs[left]["args"]
        second = configs[right]["args"]
        return sorted(
            key for key in set(first) | set(second)
            if key not in ignored and first.get(key) != second.get(key)
        )

    comparisons = [
        ("G1", "ppo_privilege_gru_0721_base", "ppo_independent_gru_0721_base", {"critic"}),
        ("G1", "ppo_privilege_gru_0721_base", "ppo_privilege_mlp_0721_base", {"critic"}),
        ("G2", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0721_bs25600", {"batch_size"}),
        ("G2", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0721_bs51200", {"batch_size"}),
        ("G3 context", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0721_clip010", {"clip_range"}),
        ("G3 context", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0721_clip020", {"clip_range"}),
        ("G3 workers=8", "ppo_privilege_gru_0721_clip010", "ppo_privilege_gru_0721_clip020", {"clip_range"}),
        ("G4", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0722_lr1_tkloff", {"gru_learning_rate", "head_learning_rate"}),
        ("G4", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0722_lr5_tkloff", {"gru_learning_rate", "head_learning_rate"}),
        ("G5", "ppo_privilege_gru_0722_long_clip015", "ppo_privilege_gru_0722_long_clip020", {"clip_range"}),
        ("G6", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0722_clip015_tkl002", {"target_kl"}),
        ("G6", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0722_clip015_tkl004", {"target_kl"}),
        ("G8", "ppo_privilege_gru_0722_long45_clip020", "ppo_privilege_gru_0722_long45_clip025", {"clip_range"}),
        ("G9", "ppo_privilege_gru_0722_long45_clip020", "ppo_privilege_gru_0722_long45_clip020_hard", {"hard_neighbors"}),
    ]
    rows = []
    for group, baseline, arm, intended in comparisons:
        actual = differences(baseline, arm)
        expected_extra = set()
        if group == "G9":
            # The explicit switch changes the resolved collision pool by design;
            # hard_neighbor_cache_dir itself is equal in both configs.
            expected_extra = set()
        confounds = sorted(set(actual) - intended - expected_extra)
        rows.append(
            {
                "group": group,
                "baseline_run": baseline,
                "arm_run": arm,
                "intended_differences": ", ".join(sorted(intended)),
                "recorded_differences": ", ".join(actual),
                "confounds": ", ".join(confounds),
                "strict_single_axis": not confounds,
            }
        )
    return rows


def duplicate_policy_audit(panel_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for panel in panel_rows:
        if panel["run"] != "BC" and panel["valid"]:
            grouped[panel["policy_sha256"]].append(panel)
    rows = []
    selected: set[tuple[str, int]] = set()
    for policy_hash, panels in sorted(grouped.items()):
        panels.sort(key=lambda row: (row["run"], int(row["update"])))
        selected.add((panels[0]["run"], int(panels[0]["update"])))
        if len(panels) > 1:
            rows.append(
                {
                    "policy_sha256": policy_hash,
                    "panel_count": len(panels),
                    "panels": ";".join(panel["panel"] for panel in panels),
                    "result_hash_count": len({panel["results_sha256"] for panel in panels}),
                    "results_identical": len({panel["results_sha256"] for panel in panels}) == 1,
                }
            )
    return rows, selected


def scenario_frequency(
    selected_policy_keys: set[tuple[str, int]],
    episodes_by_panel: dict[tuple[str, int], dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key in selected_policy_keys:
        for scenario, row in episodes_by_panel[key].items():
            counts[scenario]["collision"] += int(row["ego_collision"])
            counts[scenario]["opponent_collision"] += int(row["ego_collision"] and row["opponent_collision"])
            counts[scenario]["tail"] += int(row["merge_tail_primary"])
            counts[scenario]["overtake"] += int(row["outcome"] == "overtake")
            counts[scenario]["follow"] += int(row["outcome"] == "follow")
    output = []
    denominator = len(selected_policy_keys)
    for scenario, counter in counts.items():
        output.append(
            {
                "scenario_id": scenario,
                **parse_scenario(scenario),
                "unique_policy_panels": denominator,
                "collision_panels": counter["collision"],
                "collision_rate": counter["collision"] / denominator,
                "opponent_collision_panels": counter["opponent_collision"],
                "merge_tail_panels": counter["tail"],
                "merge_tail_rate": counter["tail"] / denominator,
                "overtake_panels": counter["overtake"],
                "follow_panels": counter["follow"],
            }
        )
    output.sort(key=lambda row: (-row["collision_panels"], -row["merge_tail_panels"], row["scenario_id"]))
    return output


def collision_commonality(
    collision_rows: list[dict[str, Any]],
    selected_policy_keys: set[tuple[str, int]],
    scenario_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    unique_policy_collisions = [
        row for row in collision_rows
        if row["panel_valid"] and (row["run"], int(row["update"])) in selected_policy_keys
    ]
    bc_collisions = [row for row in collision_rows if row["run"] == "BC"]
    bc_tail = [row for row in bc_collisions if row["merge_tail_primary"]]
    bc_other = [row for row in bc_collisions if not row["merge_tail_primary"]]

    def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
        values = Counter(str(parse_scenario(row["scenario_id"])[field]) for row in rows)
        return dict(sorted(values.items()))

    return {
        "unique_valid_ppo_policy_panels": len(selected_policy_keys),
        "unique_policy_collision_events": len(unique_policy_collisions),
        "unique_policy_collision_class_counts": dict(Counter(row["collision_class"] for row in unique_policy_collisions)),
        "unique_policy_opponent_raceline_counts": distribution(unique_policy_collisions, "opponent_raceline"),
        "unique_policy_speed_scale_counts": distribution(unique_policy_collisions, "opponent_speed_scale"),
        "unique_policy_collision_time_quantiles_s": {
            "q10": quantile((row["collision_time_s"] for row in unique_policy_collisions), 0.10),
            "median": median(row["collision_time_s"] for row in unique_policy_collisions),
            "q90": quantile((row["collision_time_s"] for row in unique_policy_collisions), 0.90),
        },
        "bc_collision_count": len(bc_collisions),
        "bc_tail_count": len(bc_tail),
        "bc_tail_raceline_counts": distribution(bc_tail, "opponent_raceline"),
        "bc_tail_speed_scale_counts": distribution(bc_tail, "opponent_speed_scale"),
        "bc_tail_pass_to_collision_s": {
            "min": min(row["pass_to_collision_s"] for row in bc_tail),
            "median": median(row["pass_to_collision_s"] for row in bc_tail),
            "max": max(row["pass_to_collision_s"] for row in bc_tail),
        },
        "bc_tail_lateral_convergence_m": {
            "min": min(row["post_pass_lateral_convergence_m"] for row in bc_tail),
            "median": median(row["post_pass_lateral_convergence_m"] for row in bc_tail),
            "max": max(row["post_pass_lateral_convergence_m"] for row in bc_tail),
        },
        "bc_tail_abs_slip_proxy_median_rad": median(row["terminal_abs_kinematic_slip_proxy_rad"] for row in bc_tail),
        "bc_other_collision_abs_slip_proxy_median_rad": median(
            row["terminal_abs_kinematic_slip_proxy_rad"] for row in bc_other
        ),
        "top_collision_scenarios": scenario_rows[:20],
    }


def training_eval_correlations(
    panel_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    selected_policy_keys: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    training = {(row["run"], int(row["update"])): row for row in training_rows}
    selected = [
        row for row in panel_rows
        if row["run"] != "BC" and row["valid"] and (row["run"], int(row["update"])) in selected_policy_keys
    ]
    metrics = (
        "training_collision_rate",
        "mean_episode_return",
        "value_loss_post",
        "explained_variance_post",
        "approx_kl_mean",
        "clip_fraction_mean",
    )
    output = []
    target = np.asarray([float(row["collision_count"]) for row in selected])
    for field in metrics:
        values = np.asarray([float(training[(row["run"], int(row["update"]))][field]) for row in selected])
        output.append(
            {
                "metric": field,
                "unique_policy_panels": len(values),
                "pearson_with_eval_collision_count": float(pearsonr(values, target).statistic),
                "spearman_with_eval_collision_count": float(spearmanr(values, target).statistic),
                "scope": "pooled descriptive correlation across heterogeneous single-seed runs; not causal",
            }
        )
    return output


def run_eval_summary(panel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in panel_rows:
        if row["run"] != "BC":
            grouped[row["run"]].append(row)
    output = []
    for run, panels in sorted(grouped.items()):
        panels.sort(key=lambda row: int(row["update"]))
        valid = [row for row in panels if row["valid"]]
        best = min(valid, key=lambda row: (row["collision_count"], row["merge_tail_primary_count"], -row["overtake_count"]))
        final = valid[-1]
        output.append(
            {
                "run": run,
                "eval_panel_count": len(panels),
                "valid_eval_panel_count": len(valid),
                "invalid_updates": ";".join(str(row["update"]) for row in panels if not row["valid"]),
                "updates": "/".join(str(row["update"]) for row in panels),
                "collision_path": " / ".join(str(row["collision_count"]) for row in panels),
                "merge_tail_path": " / ".join(str(row["merge_tail_primary_count"]) for row in panels),
                "overtake_path": " / ".join(str(row["overtake_count"]) for row in panels),
                "best_update": best["update"],
                "best_collision_count": best["collision_count"],
                "best_tail_count": best["merge_tail_primary_count"],
                "best_overtake_count": best["overtake_count"],
                "latest_valid_update": final["update"],
                "latest_valid_collision_count": final["collision_count"],
                "latest_valid_tail_count": final["merge_tail_primary_count"],
                "latest_valid_overtake_count": final["overtake_count"],
            }
        )
    return output


def selected_panel_record(
    panel: dict[str, Any],
    paired_by_panel: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    paired = paired_by_panel.get(panel["panel"], {})
    return {
        "panel": panel["panel"],
        "run": panel["run"],
        "update": panel["update"],
        "valid": panel["valid"],
        "collision_count": panel["collision_count"],
        "collision_rate": panel["collision_rate"],
        "overtake_count": panel["overtake_count"],
        "follow_count": panel["follow_count"],
        "merge_tail_relaxed_count": panel["merge_tail_relaxed_count"],
        "merge_tail_primary_count": panel["merge_tail_primary_count"],
        "merge_tail_strict_count": panel["merge_tail_strict_count"],
        "tail_shared_with_bc": paired.get("tail_shared"),
        "bc_tail_resolved": paired.get("tail_resolved"),
        "new_tail_events": paired.get("tail_created"),
        "bc_tail_scenarios_still_any_collision": paired.get("bc_tail_scenarios_still_any_collision"),
        "bc_tail_scenarios_now_overtake": paired.get("bc_tail_scenarios_now_overtake"),
        "collision_net_reduction_vs_bc": paired.get("collision_net_reduction"),
        "tail_net_reduction_vs_bc": paired.get("tail_net_reduction"),
        "collision_exact_p": paired.get("collision_exact_p"),
        "tail_exact_p": paired.get("tail_exact_p"),
        "collision_cluster_bootstrap_diff_ci": [
            paired.get("collision_cluster_bootstrap_diff_ci_low"),
            paired.get("collision_cluster_bootstrap_diff_ci_high"),
        ] if paired else None,
        "tail_cluster_bootstrap_diff_ci": [
            paired.get("tail_cluster_bootstrap_diff_ci_low"),
            paired.get("tail_cluster_bootstrap_diff_ci_high"),
        ] if paired else None,
        "policy_sha256": panel["policy_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-npz-audit", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/8] Reading all training runs and metrics", flush=True)
    run_rows, training_by_run, configs, training_rows = read_runs()
    print("[2/8] Reading all evaluation panels and episode records", flush=True)
    panel_rows, episode_rows, episodes_by_panel = read_panels(configs)
    print("[3/8] Reconstructing every collision trace", flush=True)
    collision_rows, _collision_lookup = analyze_collision_traces(panel_rows, episode_rows)
    print("[4/8] Auditing NPZ structure and terminal semantics", flush=True)
    npz_path = OUT / "npz_audit.json"
    if args.reuse_npz_audit and npz_path.is_file():
        npz_audit = load_json(npz_path)
    else:
        npz_audit = scan_npz(episode_rows)
    print("[5/8] Auditing actor/critic parameters and continuation equivalence", flush=True)
    model_rows, critic_rows, architecture = audit_model_parameters(panel_rows, run_rows)
    equivalence = trajectory_equivalence(training_by_run)
    print("[6/8] Building paired BC comparisons and experiment groups", flush=True)
    paired_rows, bc_tail_scenario_rows = paired_vs_bc(panel_rows, episodes_by_panel)
    group_rows = build_group_rows(panel_rows, episodes_by_panel)
    control_rows = group_control_audit(configs)
    duplicate_rows, selected_policy_keys = duplicate_policy_audit(panel_rows)
    scenario_rows = scenario_frequency(selected_policy_keys, episodes_by_panel)
    commonality = collision_commonality(collision_rows, selected_policy_keys, scenario_rows)
    correlation_rows = training_eval_correlations(panel_rows, training_rows, selected_policy_keys)
    run_eval_rows = run_eval_summary(panel_rows)

    print("[7/8] Writing bounded audit tables", flush=True)
    write_csv(OUT / "run_inventory.csv", run_rows)
    write_csv(OUT / "training_metrics.csv", training_rows)
    write_csv(OUT / "eval_panels.csv", panel_rows)
    write_csv(OUT / "eval_episode_outcomes.csv", episode_rows)
    write_csv(OUT / "collision_episode_kinematics.csv", collision_rows)
    write_csv(OUT / "paired_vs_bc.csv", paired_rows)
    write_csv(OUT / "bc_tail_scenario_outcomes.csv", bc_tail_scenario_rows)
    write_csv(OUT / "group_summary.csv", group_rows)
    write_csv(OUT / "group_control_audit.csv", control_rows)
    write_csv(OUT / "run_eval_summary.csv", run_eval_rows)
    write_csv(OUT / "actor_parameter_deltas.csv", model_rows)
    write_csv(OUT / "critic_parameter_summary.csv", critic_rows)
    write_csv(OUT / "duplicate_policy_audit.csv", duplicate_rows)
    write_csv(OUT / "scenario_frequency_unique_policies.csv", scenario_rows)
    write_csv(OUT / "training_eval_correlations.csv", correlation_rows)
    npz_path.write_text(json.dumps(npz_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "model_architecture.json").write_text(
        json.dumps(architecture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT / "trajectory_equivalence.json").write_text(
        json.dumps(equivalence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT / "collision_commonality.json").write_text(
        json.dumps(commonality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    panel_by_key = {(row["run"], int(row["update"])): row for row in panel_rows}
    paired_by_panel = {row["panel"]: row for row in paired_rows}
    bc_panel = panel_by_key[("BC", 0)]
    valid_ppo_panels = [row for row in panel_rows if row["run"] != "BC" and row["valid"]]
    best_collision = min(
        valid_ppo_panels,
        key=lambda row: (row["collision_count"], row["merge_tail_primary_count"], -row["overtake_count"]),
    )
    best_tail = min(
        valid_ppo_panels,
        key=lambda row: (row["merge_tail_primary_count"], row["collision_count"], -row["overtake_count"]),
    )
    selected_keys = [
        ("ppo_privilege_gru_0721_base", 20),
        ("ppo_privilege_gru_0722_long_clip020", 30),
        ("ppo_privilege_gru_0722_long45_clip020", 40),
        ("ppo_privilege_gru_0722_long45_clip020", 45),
        ("ppo_privilege_gru_0722_long45_clip025", 45),
        ("ppo_privilege_gru_0722_long45_clip020_hard", 35),
        ("ppo_privilege_gru_0722_long45_clip020_hard", 45),
    ]
    selected_records = [selected_panel_record(panel_by_key[key], paired_by_panel) for key in selected_keys]
    bc_tail_ids = sorted(tail_set(episodes_by_panel[("BC", 0)]))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_head": "4e0043ed26e57950546c685d7bddba83e756c1a5",
        "scope": {
            "training_runs": len(run_rows),
            "ppo_eval_panels": sum(row["run"] != "BC" for row in panel_rows),
            "bc_eval_panels": 1,
            "valid_ppo_eval_panels": sum(row["run"] != "BC" and row["valid"] for row in panel_rows),
            "invalid_ppo_eval_panels": [row["panel"] for row in panel_rows if row["run"] != "BC" and not row["valid"]],
            "episode_rows": len(episode_rows),
            "collision_episode_traces_analyzed": len(collision_rows),
            "npz_files_audited": npz_audit["totals"].get("files"),
            "unique_valid_ppo_policies": len(selected_policy_keys),
            "duplicate_policy_groups": len(duplicate_rows),
        },
        "data_quality": {
            "valid_panels": sum(bool(row["valid"]) for row in panel_rows),
            "total_panels": len(panel_rows),
            "invalid_panels": [row for row in panel_rows if not row["valid"]],
            "npz": npz_audit,
            "duplicate_policies": duplicate_rows,
            "all_run_metrics_finite": all(row["metrics_finite"] for row in run_rows),
            "all_run_checkpoints_complete": all(row["checkpoint_complete"] for row in run_rows),
            "all_actor_final_equals_last": all(row["actor_final_equals_last_checkpoint"] for row in run_rows),
            "seed_recorded_run_count": sum(row["seed_recorded"] for row in run_rows),
            "source_commit_recorded_run_count": sum(row["source_commit_recorded"] for row in run_rows),
        },
        "model_architecture": architecture,
        "bc": selected_panel_record(bc_panel, paired_by_panel),
        "best_collision_panel": selected_panel_record(best_collision, paired_by_panel),
        "best_tail_panel": selected_panel_record(best_tail, paired_by_panel),
        "selected_panels": selected_records,
        "tail_failure_mode": {
            "operational_definition": (
                "ego crosses from nonpositive to positive relative progress at least 0.10 s before collision; "
                "opponent also collides and is behind ego at terminal/pre-terminal pose; absolute track-lateral "
                "gap closes by at least 0.10 m after the pass"
            ),
            "bc_collision_count": bc_panel["collision_count"],
            "bc_merge_tail_primary_count": bc_panel["merge_tail_primary_count"],
            "bc_merge_tail_relaxed_count": bc_panel["merge_tail_relaxed_count"],
            "bc_merge_tail_strict_count": bc_panel["merge_tail_strict_count"],
            "bc_tail_scenario_ids": bc_tail_ids,
            "verdict": "mitigated_but_not_eliminated",
            "verdict_basis": (
                "Several PPO checkpoints resolve most or all original BC tail scenarios, but every selected "
                "candidate still has newly created or persistent merge-tail events; the mechanism moves across scenarios."
            ),
            "literal_tire_slip_limit": (
                "Eval NPZ does not save simulator slip-angle state. Pose-derived slip proxy is reported, so the "
                "evidence establishes a post-pass merge/rear-sweep contact signature, not a causal tire-dynamics label."
            ),
        },
        "collision_commonality": commonality,
        "trajectory_equivalence": equivalence,
        "provenance_limits": [
            "run_config.json intentionally omits seed; run.sh/current CLI imply default seed 42 but the run artifact does not persist it",
            "run_config.json does not persist a Git/source commit",
            "one LR5 U20 eval has 599 episode rows and one error and is excluded from rankings/inference",
            "0721 legacy NPZ traces omit the post-step terminal row; results_multi.json is terminal collision truth",
            "all training arms are single-run and all evals use the same fixed Austin600 panel",
        ],
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[8/8] Analysis complete", flush=True)
    print(json.dumps({
        "output": str(OUT),
        "valid_panels": summary["data_quality"]["valid_panels"],
        "episode_rows": len(episode_rows),
        "bc_collisions": bc_panel["collision_count"],
        "bc_tail": bc_panel["merge_tail_primary_count"],
        "best_panel": best_collision["panel"],
        "best_collisions": best_collision["collision_count"],
        "best_panel_tail": best_collision["merge_tail_primary_count"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()




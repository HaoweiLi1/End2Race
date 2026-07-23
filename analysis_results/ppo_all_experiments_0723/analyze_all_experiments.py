#!/usr/bin/env python3
"""Audit every PPO training/evaluation artifact and diagnose tail-swing collisions.

The script is read-only with respect to training and evaluation artifacts.  It
writes reproducible, bounded tables under ``analysis_results/ppo_all_experiments_0723``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_results" / "ppo_all_experiments_0723"
TRAINED = ROOT / "post-trained"
EVAL = ROOT / "eval_results"
MAP_DIR = ROOT / "f1tenth_racetracks" / "Austin"
OUT.mkdir(parents=True, exist_ok=True)

TRACK_LENGTH_M = 419.12425
VEHICLE_LENGTH_M = 0.58
VEHICLE_WIDTH_M = 0.31
EXPECTED_PANEL_EPISODES = 600


@dataclass(frozen=True)
class RunSpec:
    run: str
    group: str
    arm: str


RUN_SPECS = (
    RunSpec("ppo_independent_gru_0721_base", "G1", "independent_gru"),
    RunSpec("ppo_privilege_mlp_0721_base", "G1", "privilege_mlp"),
    RunSpec("ppo_privilege_gru_0721_base", "G1", "privilege_gru base"),
    RunSpec("ppo_privilege_gru_0721_bs25600", "G2", "batch 25600"),
    RunSpec("ppo_privilege_gru_0721_bs51200", "G2", "batch 51200"),
    RunSpec("ppo_privilege_gru_0721_clip010", "G3", "clip 0.10"),
    RunSpec("ppo_privilege_gru_0721_clip020", "G3", "clip 0.20 legacy 8-worker"),
    RunSpec("ppo_privilege_gru_0722_lr1_tkloff", "G4", "actor LR low"),
    RunSpec("ppo_privilege_gru_0722_lr5_tkloff", "G4", "actor LR high"),
    RunSpec("ppo_privilege_gru_0722_long_clip015", "G5", "clip 0.15 x 30U"),
    RunSpec("ppo_privilege_gru_0722_long_clip020", "G5", "clip 0.20 x 30U"),
    RunSpec("ppo_privilege_gru_0722_clip015_tkl002", "G6", "target-KL 0.02"),
    RunSpec("ppo_privilege_gru_0722_clip015_tkl004", "G6", "target-KL 0.04"),
    RunSpec("ppo_privilege_gru_0722_long45_clip020", "G7", "clip 0.20 x 45U"),
    RunSpec("ppo_privilege_gru_0722_long45_clip025", "G8", "clip 0.25 x 45U"),
    RunSpec("ppo_privilege_gru_0722_long45_clip020_hard", "G9", "hard-neighbor x 45U"),
)
SPEC_BY_RUN = {spec.run: spec for spec in RUN_SPECS}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(numeric) if numeric else None


def finite_median(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(numeric) if numeric else None


def safe_min(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(numeric) if numeric else None


def safe_max(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(numeric) if numeric else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def exact_paired_p(resolved: int, created: int) -> float:
    discordant = resolved + created
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(resolved, created) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 1.0
    count = len(p_values)
    for rank_from_end, index in enumerate(reversed(order), start=1):
        rank = count - rank_from_end + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def wrap_track_delta(value: np.ndarray | float) -> np.ndarray | float:
    return (value + TRACK_LENGTH_M / 2.0) % TRACK_LENGTH_M - TRACK_LENGTH_M / 2.0


def wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


class FastProgressProjector:
    """Vectorized nearest-segment projection onto raceline1."""

    def __init__(self) -> None:
        data = np.loadtxt(MAP_DIR / "raceline1.csv", delimiter=";", comments="#")
        if np.linalg.norm(data[-1, 1:3] - data[0, 1:3]) <= 1e-9:
            data = data[:-1]
        self.progress = data[:, 0]
        self.xy = data[:, 1:3]
        self.tree = cKDTree(self.xy)
        self.segment_vector = np.roll(self.xy, -1, axis=0) - self.xy
        self.segment_norm_sq = np.einsum("ij,ij->i", self.segment_vector, self.segment_vector)
        self.segment_progress = np.concatenate(
            (np.diff(self.progress), np.asarray([TRACK_LENGTH_M - self.progress[-1]]))
        )

    def project(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        _distance, nearest = self.tree.query(points, k=1)
        candidates = np.stack(((nearest - 1) % len(self.xy), nearest), axis=1)
        starts = self.xy[candidates]
        vectors = self.segment_vector[candidates]
        fractions = np.sum((points[:, None, :] - starts) * vectors, axis=2) / np.sum(vectors * vectors, axis=2)
        fractions = np.clip(fractions, 0.0, 1.0)
        closest = starts + fractions[:, :, None] * vectors
        distances_sq = np.sum((points[:, None, :] - closest) ** 2, axis=2)
        choice = np.argmin(distances_sq, axis=1)
        indices = candidates[np.arange(len(points)), choice]
        chosen_fraction = fractions[np.arange(len(points)), choice]
        return (self.progress[indices] + chosen_fraction * self.segment_progress[indices]) % TRACK_LENGTH_M


PROJECTOR = FastProgressProjector()


def parse_episode_key(key: str) -> tuple[int, int, int, float]:
    match = re.fullmatch(r"ol(\d+)_e(\d+)_o(\d+)_s([0-9.]+)", key)
    if not match:
        raise ValueError(f"Unexpected episode key: {key}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), float(match.group(4))


def raceline_data(index: int) -> np.ndarray:
    path = MAP_DIR / f"raceline{index}.csv"
    return np.loadtxt(path, delimiter=";", comments="#")


RACELINES = {index: raceline_data(index) for index in range(3)}


def physical_key(episode_key: str) -> str:
    line, ego_index, opponent_index, speed_scale = parse_episode_key(episode_key)
    ego = RACELINES[1][ego_index % len(RACELINES[1]), [1, 2, 3, 5]]
    opponent = RACELINES[line][opponent_index % len(RACELINES[line]), [1, 2, 3, 5]]
    values = [*ego.tolist(), *opponent.tolist(), speed_scale]
    return "|".join(f"{value:.9f}" for value in values)


def eval_directory(run: str, update: int) -> Path:
    if run == "BC":
        return EVAL / "end2race_Austin" / "multiagents"
    return EVAL / f"{run}_u{update:04d}_Austin" / "multiagents"


def discover_panels() -> list[tuple[str, int, Path]]:
    panels: list[tuple[str, int, Path]] = [("BC", 0, eval_directory("BC", 0))]
    pattern = re.compile(r"^(ppo_.+)_u(\d{4})_Austin$")
    for directory in sorted(EVAL.iterdir()):
        if not directory.is_dir():
            continue
        match = pattern.match(directory.name)
        if match and match.group(1) in SPEC_BY_RUN:
            panels.append((match.group(1), int(match.group(2)), directory / "multiagents"))
    panels.sort(key=lambda item: ("" if item[0] == "BC" else item[0], item[1]))
    return panels


def state_dict_stats(path: Path) -> tuple[int, int, str]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        state = state.state_dict()
    tensor_count = len(state)
    parameter_count = sum(int(value.numel()) for value in state.values() if torch.is_tensor(value))
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode())
        if torch.is_tensor(value):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return tensor_count, parameter_count, digest.hexdigest()


def state_dict_equal(left: Path, right: Path) -> tuple[bool, float]:
    lhs = torch.load(left, map_location="cpu", weights_only=True)
    rhs = torch.load(right, map_location="cpu", weights_only=True)
    if not isinstance(lhs, dict):
        lhs = lhs.state_dict()
    if not isinstance(rhs, dict):
        rhs = rhs.state_dict()
    if set(lhs) != set(rhs):
        return False, math.inf
    maximum = max(float((lhs[key] - rhs[key]).abs().max()) for key in lhs)
    return maximum == 0.0, maximum


def read_run_configs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    configs: dict[str, dict[str, Any]] = {}
    for spec in RUN_SPECS:
        run_dir = TRAINED / spec.run
        config = load_json(run_dir / "run_config.json")
        args = config["args"]
        configs[spec.run] = config
        actor_tensors, actor_parameters, actor_sha = state_dict_stats(run_dir / "actor_final.pth")
        critic_path = run_dir / "checkpoints" / f"critic_u{int(args['num_updates']):04d}.pt"
        critic_tensors, critic_parameters, critic_sha = state_dict_stats(critic_path)
        rows.append(
            {
                "group": spec.group,
                "arm": spec.arm,
                "run": spec.run,
                "critic": args["critic"],
                "env_workers": args["env_workers"],
                "n_envs": args["n_envs"],
                "n_steps": args["n_steps"],
                "batch_size": args["batch_size"],
                "num_updates": args["num_updates"],
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
                "collision_pool_count": len(load_json(run_dir / "collision_scenarios.json")),
                "ordinary_pool_count": len(load_json(run_dir / "ordinary_scenarios.json")),
                "actor_tensor_count": actor_tensors,
                "actor_parameter_count": actor_parameters,
                "actor_sha256": actor_sha,
                "critic_tensor_count": critic_tensors,
                "critic_parameter_count": critic_parameters,
                "critic_sha256": critic_sha,
                "checkpoint_actor_count": len(list((run_dir / "checkpoints").glob("actor_u*.pth"))),
                "checkpoint_critic_count": len(list((run_dir / "checkpoints").glob("critic_u*.pt"))),
            }
        )
    return rows, configs


TRAINING_FIELDS = (
    "policy_gradient_loss",
    "value_loss_post_update",
    "explained_variance_post_update",
    "collision_explained_variance_post",
    "ordinary_explained_variance_post",
    "approx_kl_mean",
    "approx_kl_max",
    "clip_fraction_mean",
    "clip_fraction_max",
    "actor_optimizer_steps_planned",
    "actor_optimizer_steps_completed",
    "actor_early_stop_triggered",
    "actor_grad_norm_mean",
    "actor_grad_norm_max",
    "critic_grad_norm_mean",
    "critic_grad_norm_max",
    "ego_collision_count",
    "episode_count",
    "overtake_count",
    "follow_count",
    "mean_episode_return",
    "mean_collision_episode_return",
    "mean_ordinary_episode_return",
    "mean_relative_position_m",
    "mean_collision_relative_position_m",
    "mean_ordinary_relative_position_m",
    "mean_ego_collision_time",
    "mean_episode_min_obb_clearance_m",
    "mean_episode_min_wall_clearance_m",
    "mean_episode_risk_active_fraction",
    "mean_episode_reward_progress",
    "mean_episode_reward_relative",
    "mean_episode_reward_collision",
    "mean_episode_reward_risk",
    "rollout_wall_seconds",
    "actor_train_wall_seconds",
    "critic_train_wall_seconds",
)


def read_training(configs: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    update_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    base_ids = {row["scenario_id"] for row in load_json(TRAINED / "collision-cache" / "default" / "collision_scenarios.json")}
    boundary_ids = {
        row["scenario_id"]
        for row in load_json(TRAINED / "collision-cache" / "boundary-aware-v1" / "collision_scenarios.json")
    } - base_ids

    for spec in RUN_SPECS:
        metrics = load_jsonl(TRAINED / spec.run / "metrics.jsonl")
        formal = [row for row in metrics if row.get("phase") == "formal"]
        expected = int(configs[spec.run]["args"]["num_updates"])
        for row in formal:
            output = {
                "group": spec.group,
                "arm": spec.arm,
                "run": spec.run,
                "update": int(row["update"]),
                "rollout_policy_update": int(row["rollout_policy_update"]),
                "checkpoint_update": int(row["checkpoint_update"]),
                "rollout_collision_rate": float(row["ego_collision_count"]) / float(row["episode_count"]),
                "rollout_overtake_rate": float(row["overtake_count"]) / float(row["episode_count"]),
            }
            output.update({field: row.get(field) for field in TRAINING_FIELDS})
            update_rows.append(output)

        episodes = load_jsonl(TRAINED / spec.run / "episodes.jsonl")
        for update in range(1, expected + 1):
            rows = [row for row in episodes if row.get("phase") == "formal" and int(row.get("formal_update", -1)) == update]
            collision_role = [row for row in rows if row.get("env_role") == "collision"]
            ordinary_role = [row for row in rows if row.get("env_role") == "ordinary"]
            base_role = [row for row in collision_role if row.get("scenario_id") in base_ids]
            boundary_role = [row for row in collision_role if row.get("scenario_id") in boundary_ids]
            source_rows.append(
                {
                    "group": spec.group,
                    "run": spec.run,
                    "update": update,
                    "episode_rows": len(rows),
                    "collision_role_episodes": len(collision_role),
                    "ordinary_role_episodes": len(ordinary_role),
                    "base_collision_source_episodes": len(base_role),
                    "boundary_collision_source_episodes": len(boundary_role),
                    "unknown_collision_source_episodes": len(collision_role) - len(base_role) - len(boundary_role),
                    "base_source_realized_collision_rate": (
                        sum(bool(row.get("ego_collision")) for row in base_role) / len(base_role) if base_role else None
                    ),
                    "boundary_source_realized_collision_rate": (
                        sum(bool(row.get("ego_collision")) for row in boundary_role) / len(boundary_role) if boundary_role else None
                    ),
                    "ordinary_role_realized_collision_rate": (
                        sum(bool(row.get("ego_collision")) for row in ordinary_role) / len(ordinary_role) if ordinary_role else None
                    ),
                }
            )

        final = formal[-1]
        summary_rows.append(
            {
                "group": spec.group,
                "arm": spec.arm,
                "run": spec.run,
                "formal_metric_rows": len(formal),
                "expected_updates": expected,
                "updates_complete": len(formal) == expected and [int(row["update"]) for row in formal] == list(range(1, expected + 1)),
                "mean_rollout_collision_rate": finite_mean(
                    float(row["ego_collision_count"]) / float(row["episode_count"]) for row in formal
                ),
                "final_rollout_collision_rate": float(final["ego_collision_count"]) / float(final["episode_count"]),
                "mean_episode_return": finite_mean(row.get("mean_episode_return") for row in formal),
                "final_episode_return": final.get("mean_episode_return"),
                "mean_relative_position_m": finite_mean(row.get("mean_relative_position_m") for row in formal),
                "final_relative_position_m": final.get("mean_relative_position_m"),
                "median_approx_kl_mean": finite_median(row.get("approx_kl_mean") for row in formal),
                "max_approx_kl_max": safe_max(row.get("approx_kl_max") for row in formal),
                "updates_approx_kl_mean_gt_0p05": sum(float(row.get("approx_kl_mean", 0.0)) > 0.05 for row in formal),
                "updates_approx_kl_max_gt_0p5": sum(float(row.get("approx_kl_max", 0.0)) > 0.5 for row in formal),
                "mean_clip_fraction": finite_mean(row.get("clip_fraction_mean") for row in formal),
                "median_actor_grad_norm": finite_median(row.get("actor_grad_norm_mean") for row in formal),
                "max_actor_grad_norm": safe_max(row.get("actor_grad_norm_max") for row in formal),
                "final_explained_variance": final.get("explained_variance_post_update"),
                "final_collision_explained_variance": final.get("collision_explained_variance_post"),
                "final_ordinary_explained_variance": final.get("ordinary_explained_variance_post"),
                "early_stop_updates": sum(bool(row.get("actor_early_stop_triggered")) for row in formal),
                "actor_steps_completed": sum(int(row.get("actor_optimizer_steps_completed", 0)) for row in formal),
                "actor_steps_planned": sum(int(row.get("actor_optimizer_steps_planned", 0)) for row in formal),
                "rollout_wall_seconds": sum(float(row.get("rollout_wall_seconds", 0.0)) for row in formal),
                "actor_wall_seconds": sum(float(row.get("actor_train_wall_seconds", 0.0)) for row in formal),
                "critic_wall_seconds": sum(float(row.get("critic_train_wall_seconds", 0.0)) for row in formal),
            }
        )
    return update_rows, summary_rows, source_rows


def panel_payload(directory: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = load_json(directory / "results_multi.json")
    by_scenario = {str(row["scenario_id"]): row for row in document["episodes"].values()}
    return document, by_scenario


def collision_set(rows: dict[str, dict[str, Any]]) -> set[str]:
    return {scenario for scenario, row in rows.items() if bool(row.get("ego_collision_occurred"))}


def classify_transition(bc_collision: bool, model_collision: bool) -> str:
    if bc_collision and not model_collision:
        return "resolved"
    if bc_collision and model_collision:
        return "persistent"
    if not bc_collision and model_collision:
        return "created"
    return "both_success"


def read_eval_panels(
    panels: list[tuple[str, int, Path]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, int], dict[str, dict[str, Any]]],
    dict[tuple[str, int], Path],
    dict[str, str],
]:
    bc_document, bc_rows = panel_payload(eval_directory("BC", 0))
    del bc_document
    bc_scenarios = set(bc_rows)
    bc_collisions = collision_set(bc_rows)
    scenario_physical = {scenario: physical_key(row["episode_key"]) for scenario, row in bc_rows.items()}
    panel_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    all_episodes: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    panel_dirs: dict[tuple[str, int], Path] = {}

    for run, update, directory in panels:
        document, rows = panel_payload(directory)
        all_episodes[(run, update)] = rows
        panel_dirs[(run, update)] = directory
        final = document["final"]
        scenarios = set(rows)
        collisions = collision_set(rows)
        overtake = sum(row.get("outcome") == "overtake" for row in rows.values())
        follow = sum(row.get("outcome") == "follow" for row in rows.values())
        traces = {path.stem for path in (directory / "traces").glob("*.npz")}
        expected_trace_keys = {str(row["episode_key"]) for row in rows.values()}
        aggregate_consistent = (
            final.get("collision_count") == len(collisions)
            and final.get("overtaking_count") == overtake
            and final.get("following_count") == follow
            and final.get("success_count") == overtake + follow
            and len(collisions) + overtake + follow == len(rows)
            and int(final.get("error_count", 0)) == EXPECTED_PANEL_EPISODES - len(rows)
        )
        paired_scenarios = bc_scenarios & scenarios
        paired_bc_collisions = bc_collisions & paired_scenarios
        paired_collisions = collisions & paired_scenarios
        resolved = len(paired_bc_collisions - paired_collisions)
        created = len(paired_collisions - paired_bc_collisions)
        persistent = len(paired_bc_collisions & paired_collisions)
        p_value = exact_paired_p(resolved, created) if run != "BC" else 1.0
        ci_low, ci_high = wilson_interval(len(collisions), EXPECTED_PANEL_EPISODES)
        collision_rows = [row for row in rows.values() if row.get("ego_collision_occurred")]
        spec = SPEC_BY_RUN.get(run)
        panel_rows.append(
            {
                "group": "BC" if run == "BC" else spec.group,
                "arm": "BC" if run == "BC" else spec.arm,
                "run": run,
                "update": update,
                "panel": "BC" if run == "BC" else f"{run}_u{update:04d}",
                "episode_rows": len(document["episodes"]),
                "unique_scenario_ids": len(scenarios),
                "unique_physical_initial_conditions": len(set(scenario_physical[scenario] for scenario in scenarios)),
                "missing_scenario_ids": len(bc_scenarios - scenarios),
                "extra_scenario_ids": len(scenarios - bc_scenarios),
                "trace_files": len(traces),
                "missing_trace_files": len(expected_trace_keys - traces),
                "unexpected_trace_files": len(traces - expected_trace_keys),
                "collision_count": len(collisions),
                "collision_rate": len(collisions) / EXPECTED_PANEL_EPISODES,
                "collision_rate_among_recorded_rows": len(collisions) / len(rows),
                "collision_rate_wilson_low": ci_low,
                "collision_rate_wilson_high": ci_high,
                "success_count": overtake + follow,
                "success_rate": (overtake + follow) / EXPECTED_PANEL_EPISODES,
                "overtake_count": overtake,
                "overtake_rate": overtake / EXPECTED_PANEL_EPISODES,
                "follow_count": follow,
                "follow_rate": follow / EXPECTED_PANEL_EPISODES,
                "vehicle_collision_count": sum(bool(row.get("opp_collision_occurred")) for row in collision_rows),
                "wall_only_collision_count": sum(not bool(row.get("opp_collision_occurred")) for row in collision_rows),
                "initial_ego_collision_count": sum(bool(row.get("initial_ego_collision")) for row in collision_rows),
                "mean_avg_speed_mps": finite_mean(row.get("avg_speed") for row in rows.values()),
                "mean_speed_variance": finite_mean(row.get("speed_variance") for row in rows.values()),
                "mean_total_distance_m": finite_mean(row.get("total_distance") for row in rows.values()),
                "mean_final_relative_position_m": finite_mean(row.get("final_relative_position_m") for row in rows.values()),
                "mean_global_min_surface_dist_m": finite_mean(row.get("global_min_surface_dist") for row in rows.values()),
                "median_collision_time_s": finite_median(row.get("ego_collision_time_s") for row in collision_rows),
                "mean_collision_time_s": finite_mean(row.get("ego_collision_time_s") for row in collision_rows),
                "steering_anomaly_episode_count": sum(bool(row.get("steering_anomaly_timesteps")) for row in rows.values()),
                "near_proximity_episode_count": sum(bool(row.get("proximity_below_threshold_timesteps")) for row in rows.values()),
                "bc_collisions_resolved": resolved,
                "bc_collisions_persistent": persistent,
                "new_collisions_created": created,
                "paired_scenario_count": len(paired_scenarios),
                "net_collision_change_vs_bc": len(collisions) - len(bc_collisions),
                "paired_exact_p_unadjusted": p_value,
                "aggregate_consistent": aggregate_consistent,
                "scenario_set_matches_bc": scenarios == bc_scenarios,
                "all_action_observation_finite": all(
                    row.get("action_finite") is True and row.get("observation_finite") is True for row in rows.values()
                ),
                "summary_valid": (
                    len(document["episodes"]) == EXPECTED_PANEL_EPISODES
                    and len(scenarios) == EXPECTED_PANEL_EPISODES
                    and final.get("total_episodes") == EXPECTED_PANEL_EPISODES
                    and final.get("error_count") == 0
                    and aggregate_consistent
                    and scenarios == bc_scenarios
                ),
                "trace_complete": traces == expected_trace_keys,
            }
        )

        for scenario in sorted(bc_scenarios):
            bc = bc_rows[scenario]
            current = rows.get(scenario)
            bc_collision = bool(bc.get("ego_collision_occurred"))
            model_collision = bool(current.get("ego_collision_occurred")) if current is not None else None
            episode_rows.append(
                {
                    "group": "BC" if run == "BC" else spec.group,
                    "run": run,
                    "update": update,
                    "scenario_id": scenario,
                    "episode_key": current["episode_key"] if current is not None else None,
                    "physical_initial_condition_key": scenario_physical[scenario],
                    "bc_outcome": bc.get("outcome"),
                    "model_outcome": current.get("outcome") if current is not None else "missing/error",
                    "bc_collision": bc_collision,
                    "model_collision": model_collision,
                    "collision_transition": (
                        classify_transition(bc_collision, model_collision) if model_collision is not None else "missing/error"
                    ),
                    "bc_ego_collision_time_s": bc.get("ego_collision_time_s"),
                    "model_ego_collision_time_s": current.get("ego_collision_time_s") if current is not None else None,
                    "collision_time_delta_s": (
                        float(current["ego_collision_time_s"]) - float(bc["ego_collision_time_s"])
                        if current is not None
                        and current.get("ego_collision_time_s") is not None
                        and bc.get("ego_collision_time_s") is not None
                        else None
                    ),
                    "bc_avg_speed_mps": bc.get("avg_speed"),
                    "model_avg_speed_mps": current.get("avg_speed") if current is not None else None,
                    "avg_speed_delta_mps": (
                        float(current["avg_speed"]) - float(bc["avg_speed"]) if current is not None else None
                    ),
                    "bc_final_relative_position_m": bc.get("final_relative_position_m"),
                    "model_final_relative_position_m": (
                        current.get("final_relative_position_m") if current is not None else None
                    ),
                    "final_relative_position_delta_m": (
                        float(current["final_relative_position_m"]) - float(bc["final_relative_position_m"])
                        if current is not None
                        else None
                    ),
                    "bc_global_min_surface_dist_m": bc.get("global_min_surface_dist"),
                    "model_global_min_surface_dist_m": (
                        current.get("global_min_surface_dist") if current is not None else None
                    ),
                    "global_min_surface_dist_delta_m": (
                        float(current["global_min_surface_dist"]) - float(bc["global_min_surface_dist"])
                        if current is not None
                        else None
                    ),
                }
            )

    p_values = [float(row["paired_exact_p_unadjusted"]) for row in panel_rows if row["run"] != "BC"]
    adjusted = benjamini_hochberg(p_values)
    index = 0
    for row in panel_rows:
        if row["run"] == "BC":
            row["paired_exact_q_bh_all_panels"] = 1.0
        else:
            row["paired_exact_q_bh_all_panels"] = adjusted[index]
            index += 1
    return panel_rows, episode_rows, all_episodes, panel_dirs, scenario_physical


def backward_kinematic_series(pose: np.ndarray, time_s: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    sideslip = np.full(len(time_s), np.nan, dtype=np.float64)
    yaw_rate = np.full(len(time_s), np.nan, dtype=np.float64)
    if len(time_s) <= steps:
        return sideslip, yaw_rate
    delta_xy = pose[steps:, :2] - pose[:-steps, :2]
    course = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])
    sideslip[steps:] = wrap_angle(course - pose[steps:, 2])
    yaw = np.unwrap(pose[:, 2])
    delta_t = time_s[steps:] - time_s[:-steps]
    yaw_rate[steps:] = np.divide(yaw[steps:] - yaw[:-steps], delta_t, out=np.full_like(delta_t, np.nan), where=delta_t > 0)
    return sideslip, yaw_rate


def trace_reference_index(
    row: dict[str, Any], time_s: np.ndarray, collisions: np.ndarray | None, center_distance: np.ndarray
) -> tuple[int, str]:
    if row.get("ego_collision_occurred"):
        if collisions is not None and collisions.ndim == 2 and collisions.shape[1] >= 1 and np.any(collisions[:, 0]):
            return int(np.flatnonzero(collisions[:, 0])[0]), "terminal_collision_flag"
        collision_time = row.get("ego_collision_time_s")
        if collision_time is not None:
            return int(np.argmin(np.abs(time_s - float(collision_time)))), "nearest_preterminal_frame"
        return len(time_s) - 1, "last_frame_collision_fallback"
    return int(np.argmin(center_distance)), "closest_approach"


def extract_trace_features(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as trace:
        files = set(trace.files)
        time_s = np.asarray(trace["time_s"], dtype=np.float64)
        ego_pose = np.asarray(trace["ego_pose"], dtype=np.float64)
        opp_pose = np.asarray(trace["opp_pose"], dtype=np.float64)
        collisions = np.asarray(trace["collisions"], dtype=bool) if "collisions" in files else None
        terminal = np.asarray(trace["terminal_post_step"], dtype=bool) if "terminal_post_step" in files else None

    relative_world = opp_pose[:, :2] - ego_pose[:, :2]
    cos_heading = np.cos(ego_pose[:, 2])
    sin_heading = np.sin(ego_pose[:, 2])
    opponent_longitudinal = cos_heading * relative_world[:, 0] + sin_heading * relative_world[:, 1]
    opponent_lateral = -sin_heading * relative_world[:, 0] + cos_heading * relative_world[:, 1]
    center_distance = np.linalg.norm(relative_world, axis=1)
    reference_index, reference_kind = trace_reference_index(row, time_s, collisions, center_distance)
    reference_index = min(max(reference_index, 0), len(time_s) - 1)

    ego_progress = PROJECTOR.project(ego_pose[: reference_index + 1, :2])
    opp_progress = PROJECTOR.project(opp_pose[: reference_index + 1, :2])
    raw_relative = np.asarray(wrap_track_delta(ego_progress - opp_progress), dtype=np.float64)
    unwrapped_relative = np.empty_like(raw_relative)
    unwrapped_relative[0] = raw_relative[0]
    if len(raw_relative) > 1:
        unwrapped_relative[1:] = raw_relative[0] + np.cumsum(wrap_track_delta(np.diff(raw_relative)))
    overtake_indices = np.flatnonzero(unwrapped_relative > 0.0)
    first_overtake_time = float(time_s[overtake_indices[0]]) if len(overtake_indices) else None

    sideslip_5, yaw_rate_5 = backward_kinematic_series(ego_pose, time_s, steps=5)
    sideslip_10, _yaw_rate_10 = backward_kinematic_series(ego_pose, time_s, steps=10)
    half_second_index = max(0, reference_index - 50)
    one_second_index = max(0, reference_index - 100)
    half_window = slice(half_second_index, reference_index + 1)
    one_window = slice(one_second_index, reference_index + 1)

    x_at_reference = float(opponent_longitudinal[reference_index])
    y_at_reference = float(opponent_lateral[reference_index])
    rear_distance = math.hypot(x_at_reference + VEHICLE_LENGTH_M / 2.0, y_at_reference)
    front_distance = math.hypot(x_at_reference - VEHICLE_LENGTH_M / 2.0, y_at_reference)
    max_sideslip_5_half = math.degrees(float(np.nanmax(np.abs(sideslip_5[half_window]))))
    max_sideslip_10_half = math.degrees(float(np.nanmax(np.abs(sideslip_10[half_window]))))
    max_yaw_rate_half = float(np.nanmax(np.abs(yaw_rate_5[half_window])))
    lateral_closure_half = float(abs(opponent_lateral[half_second_index]) - abs(y_at_reference))
    lateral_closure_one = float(abs(opponent_lateral[one_second_index]) - abs(y_at_reference))
    vehicle_collision = bool(row.get("ego_collision_occurred") and row.get("opp_collision_occurred"))
    post_overtake = bool(len(overtake_indices))
    rear_contact = bool(x_at_reference < 0.0 and rear_distance <= front_distance)
    merge_closing = bool(lateral_closure_half >= 0.10)
    structural = bool(vehicle_collision and post_overtake and rear_contact)
    return {
        "trace_frames": len(time_s),
        "trace_duration_s": float(time_s[-1]),
        "terminal_post_step_present": bool(terminal is not None),
        "terminal_post_step_true_count": int(np.count_nonzero(terminal)) if terminal is not None else 0,
        "collision_flag_present_in_trace": bool(collisions is not None and np.any(collisions[:, 0])),
        "reference_index": reference_index,
        "reference_kind": reference_kind,
        "reference_time_s": float(time_s[reference_index]),
        "json_collision_time_s": row.get("ego_collision_time_s"),
        "trace_reference_lag_to_collision_s": (
            float(row["ego_collision_time_s"]) - float(time_s[reference_index])
            if row.get("ego_collision_time_s") is not None
            else None
        ),
        "initial_relative_progress_m": float(unwrapped_relative[0]),
        "max_relative_progress_before_reference_m": float(np.max(unwrapped_relative)),
        "relative_progress_at_reference_m": float(unwrapped_relative[-1]),
        "first_overtake_time_s": first_overtake_time,
        "time_from_overtake_to_reference_s": (
            float(time_s[reference_index]) - first_overtake_time if first_overtake_time is not None else None
        ),
        "opponent_longitudinal_in_ego_frame_m": x_at_reference,
        "opponent_lateral_in_ego_frame_m": y_at_reference,
        "center_distance_at_reference_m": float(center_distance[reference_index]),
        "rear_center_distance_at_reference_m": rear_distance,
        "front_center_distance_at_reference_m": front_distance,
        "max_abs_estimated_sideslip_deg_0p5s_5step": max_sideslip_5_half,
        "max_abs_estimated_sideslip_deg_0p5s_10step": max_sideslip_10_half,
        "max_abs_estimated_yaw_rate_radps_0p5s": max_yaw_rate_half,
        "lateral_closure_m_0p5s": lateral_closure_half,
        "lateral_closure_m_1p0s": lateral_closure_one,
        "min_center_distance_m_1p0s": float(np.min(center_distance[one_window])),
        "vehicle_collision": vehicle_collision,
        "wall_only_collision": bool(row.get("ego_collision_occurred") and not row.get("opp_collision_occurred")),
        "overtake_before_reference": post_overtake,
        "opponent_behind_at_reference": rear_contact,
        "lateral_merge_closing_0p5s": merge_closing,
        "post_overtake_rear_contact": structural,
        "post_overtake_rear_merge_contact": bool(structural and merge_closing),
        "high_sideslip_tail_3deg": bool(structural and merge_closing and max_sideslip_5_half >= 3.0),
        "high_sideslip_tail_5deg": bool(structural and merge_closing and max_sideslip_5_half >= 5.0),
        "high_sideslip_tail_8deg": bool(structural and merge_closing and max_sideslip_5_half >= 8.0),
    }


def trace_analysis(
    panels: list[tuple[str, int, Path]],
    all_episodes: dict[tuple[str, int], dict[str, dict[str, Any]]],
    panel_dirs: dict[tuple[str, int], Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[str], set[str]]:
    bc_rows = all_episodes[("BC", 0)]
    bc_collision_features: dict[str, dict[str, Any]] = {}
    for scenario, row in bc_rows.items():
        if not row.get("ego_collision_occurred"):
            continue
        path = panel_dirs[("BC", 0)] / "traces" / f"{row['episode_key']}.npz"
        bc_collision_features[scenario] = extract_trace_features(path, row)
    bc_structural = {scenario for scenario, values in bc_collision_features.items() if values["post_overtake_rear_contact"]}
    bc_strict_5deg = {scenario for scenario, values in bc_collision_features.items() if values["high_sideslip_tail_5deg"]}

    collision_features: list[dict[str, Any]] = []
    inherited_rows: list[dict[str, Any]] = []
    panel_tail_rows: list[dict[str, Any]] = []
    for run, update, _directory in panels:
        rows = all_episodes[(run, update)]
        spec = SPEC_BY_RUN.get(run)
        collision_scenarios = {scenario for scenario, row in rows.items() if row.get("ego_collision_occurred")}
        selected = collision_scenarios | bc_structural
        feature_cache: dict[str, dict[str, Any]] = {}
        for scenario in sorted(selected):
            row = rows.get(scenario)
            if row is None:
                feature_cache[scenario] = {"trace_missing": True, "episode_missing": True}
                continue
            path = panel_dirs[(run, update)] / "traces" / f"{row['episode_key']}.npz"
            if not path.exists():
                feature_cache[scenario] = {"trace_missing": True}
                continue
            values = extract_trace_features(path, row)
            values["trace_missing"] = False
            feature_cache[scenario] = values
            if scenario in collision_scenarios:
                collision_features.append(
                    {
                        "group": "BC" if run == "BC" else spec.group,
                        "run": run,
                        "update": update,
                        "scenario_id": scenario,
                        "episode_key": row["episode_key"],
                        "outcome": row.get("outcome"),
                        "opp_raceline": parse_episode_key(row["episode_key"])[0],
                        "opponent_speed_scale": parse_episode_key(row["episode_key"])[3],
                        "avg_speed_mps": row.get("avg_speed"),
                        "global_min_surface_dist_m": row.get("global_min_surface_dist"),
                        "final_relative_position_m_json": row.get("final_relative_position_m"),
                        **values,
                    }
                )

        for scenario in sorted(bc_structural):
            row = rows.get(scenario)
            values = feature_cache[scenario]
            inherited_rows.append(
                {
                    "group": "BC" if run == "BC" else spec.group,
                    "run": run,
                    "update": update,
                    "scenario_id": scenario,
                    "episode_key": row["episode_key"] if row is not None else None,
                    "bc_strict_5deg_tail": scenario in bc_strict_5deg,
                    "outcome": row.get("outcome") if row is not None else "missing/error",
                    "ego_collision": bool(row.get("ego_collision_occurred")) if row is not None else None,
                    "opp_collision": bool(row.get("opp_collision_occurred")) if row is not None else None,
                    "collision_time_s": row.get("ego_collision_time_s") if row is not None else None,
                    "avg_speed_mps": row.get("avg_speed") if row is not None else None,
                    "final_relative_position_m": row.get("final_relative_position_m") if row is not None else None,
                    **values,
                }
            )

        panel_collision_features = [
            row for row in collision_features if row["run"] == run and int(row["update"]) == update
        ]
        structural_scenarios = {
            row["scenario_id"] for row in panel_collision_features if row.get("post_overtake_rear_contact")
        }
        strict3 = {row["scenario_id"] for row in panel_collision_features if row.get("high_sideslip_tail_3deg")}
        strict5 = {row["scenario_id"] for row in panel_collision_features if row.get("high_sideslip_tail_5deg")}
        strict8 = {row["scenario_id"] for row in panel_collision_features if row.get("high_sideslip_tail_8deg")}
        panel_tail_rows.append(
            {
                "group": "BC" if run == "BC" else spec.group,
                "arm": "BC" if run == "BC" else spec.arm,
                "run": run,
                "update": update,
                "collision_trace_rows": len(panel_collision_features),
                "missing_collision_traces": sum(bool(row.get("trace_missing")) for row in panel_collision_features),
                "post_overtake_rear_contact_count": len(structural_scenarios),
                "post_overtake_rear_merge_contact_count": sum(
                    bool(row.get("post_overtake_rear_merge_contact")) for row in panel_collision_features
                ),
                "high_sideslip_tail_3deg_count": len(strict3),
                "high_sideslip_tail_5deg_count": len(strict5),
                "high_sideslip_tail_8deg_count": len(strict8),
                "bc_structural_tail_scenarios_total": len(bc_structural),
                "bc_structural_tail_scenarios_still_collision": len(bc_structural & collision_scenarios),
                "bc_structural_tail_scenarios_resolved": len(bc_structural - collision_scenarios),
                "bc_strict_5deg_tail_scenarios_total": len(bc_strict_5deg),
                "bc_strict_5deg_tail_scenarios_still_collision": len(bc_strict_5deg & collision_scenarios),
                "bc_strict_5deg_tail_scenarios_resolved": len(bc_strict_5deg - collision_scenarios),
                "new_structural_tail_scenarios_vs_bc": len(structural_scenarios - bc_structural),
                "new_strict_5deg_tail_scenarios_vs_bc": len(strict5 - bc_strict_5deg),
            }
        )
    return collision_features, inherited_rows, panel_tail_rows, bc_structural, bc_strict_5deg


def checkpoint_reproducibility() -> list[dict[str, Any]]:
    comparisons = (
        ("G5 clip0.20 -> G7 extension", "ppo_privilege_gru_0722_long_clip020", "ppo_privilege_gru_0722_long45_clip020", range(1, 31)),
        ("G1 base -> G5 clip0.15 extension", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0722_long_clip015", range(1, 21)),
    )
    rows: list[dict[str, Any]] = []
    for label, left_run, right_run, updates in comparisons:
        left_metrics = {int(row["update"]): row for row in load_jsonl(TRAINED / left_run / "metrics.jsonl") if row.get("phase") == "formal"}
        right_metrics = {int(row["update"]): row for row in load_jsonl(TRAINED / right_run / "metrics.jsonl") if row.get("phase") == "formal"}
        volatile = {
            "actor_checkpoint",
            "critic_checkpoint",
            "rollout_wall_seconds",
            "actor_train_wall_seconds",
            "critic_train_wall_seconds",
        }
        for update in updates:
            left_path = TRAINED / left_run / "checkpoints" / f"actor_u{update:04d}.pth"
            right_path = TRAINED / right_run / "checkpoints" / f"actor_u{update:04d}.pth"
            tensor_equal, maximum = state_dict_equal(left_path, right_path)
            left_core = {key: value for key, value in left_metrics[update].items() if key not in volatile}
            right_core = {key: value for key, value in right_metrics[update].items() if key not in volatile}
            rows.append(
                {
                    "comparison": label,
                    "left_run": left_run,
                    "right_run": right_run,
                    "update": update,
                    "actor_tensors_exact": tensor_equal,
                    "max_abs_actor_parameter_difference": maximum,
                    "non_walltime_metrics_exact": left_core == right_core,
                }
            )
    return rows


def scenario_frequency(
    panels: list[tuple[str, int, Path]],
    all_episodes: dict[tuple[str, int], dict[str, dict[str, Any]]],
    collision_features: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest_by_run: dict[str, int] = {}
    for run, update, _directory in panels:
        if run != "BC":
            latest_by_run[run] = max(update, latest_by_run.get(run, 0))
    bc_rows = all_episodes[("BC", 0)]
    ppo_panels = [(run, update) for run, update, _directory in panels if run != "BC"]
    tail_lookup = {
        (row["run"], int(row["update"]), row["scenario_id"]): row for row in collision_features
    }
    rows: list[dict[str, Any]] = []
    for scenario in sorted(bc_rows):
        observed_panels = [
            (run, update) for run, update in ppo_panels if scenario in all_episodes[(run, update)]
        ]
        observed_finals = [
            (run, update) for run, update in latest_by_run.items() if scenario in all_episodes[(run, update)]
        ]
        all_count = sum(
            bool(all_episodes[(run, update)][scenario].get("ego_collision_occurred"))
            for run, update in observed_panels
        )
        final_count = sum(
            bool(all_episodes[(run, update)][scenario].get("ego_collision_occurred"))
            for run, update in observed_finals
        )
        structural_count = sum(
            bool(tail_lookup.get((run, update, scenario), {}).get("post_overtake_rear_contact"))
            for run, update in ppo_panels
        )
        strict_count = sum(
            bool(tail_lookup.get((run, update, scenario), {}).get("high_sideslip_tail_5deg"))
            for run, update in ppo_panels
        )
        rows.append(
            {
                "scenario_id": scenario,
                "episode_key": bc_rows[scenario]["episode_key"],
                "bc_collision": bool(bc_rows[scenario].get("ego_collision_occurred")),
                "ppo_eval_panel_count": len(ppo_panels),
                "ppo_observed_panel_count": len(observed_panels),
                "ppo_collision_panel_count": all_count,
                "ppo_collision_panel_rate": all_count / len(observed_panels),
                "ppo_final_run_count": len(latest_by_run),
                "ppo_observed_final_run_count": len(observed_finals),
                "ppo_final_collision_count": final_count,
                "ppo_final_collision_rate": final_count / len(observed_finals),
                "ppo_structural_tail_panel_count": structural_count,
                "ppo_strict_5deg_tail_panel_count": strict_count,
            }
        )
    rows.sort(key=lambda row: (-row["ppo_collision_panel_count"], -int(row["bc_collision"]), row["scenario_id"]))
    return rows


def selected_pairwise(
    all_episodes: dict[tuple[str, int], dict[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    pairs = (
        ("BC -> G1 base U20", ("BC", 0), ("ppo_privilege_gru_0721_base", 20)),
        ("BC -> G5 clip0.20 U30", ("BC", 0), ("ppo_privilege_gru_0722_long_clip020", 30)),
        ("BC -> G7 clip0.20 U40", ("BC", 0), ("ppo_privilege_gru_0722_long45_clip020", 40)),
        ("BC -> G7 clip0.20 U45", ("BC", 0), ("ppo_privilege_gru_0722_long45_clip020", 45)),
        ("BC -> G8 clip0.25 U45", ("BC", 0), ("ppo_privilege_gru_0722_long45_clip025", 45)),
        ("BC -> G9 hard U20", ("BC", 0), ("ppo_privilege_gru_0722_long45_clip020_hard", 20)),
        ("BC -> G9 hard U45", ("BC", 0), ("ppo_privilege_gru_0722_long45_clip020_hard", 45)),
        ("G7 vs G9 U30", ("ppo_privilege_gru_0722_long_clip020", 30), ("ppo_privilege_gru_0722_long45_clip020_hard", 30)),
        ("G7 vs G9 U45", ("ppo_privilege_gru_0722_long45_clip020", 45), ("ppo_privilege_gru_0722_long45_clip020_hard", 45)),
        ("G7 vs G8 U45", ("ppo_privilege_gru_0722_long45_clip020", 45), ("ppo_privilege_gru_0722_long45_clip025", 45)),
    )
    rows: list[dict[str, Any]] = []
    for label, left_key, right_key in pairs:
        left = collision_set(all_episodes[left_key])
        right = collision_set(all_episodes[right_key])
        resolved = len(left - right)
        created = len(right - left)
        rows.append(
            {
                "comparison": label,
                "left_run": left_key[0],
                "left_update": left_key[1],
                "right_run": right_key[0],
                "right_update": right_key[1],
                "left_collisions": len(left),
                "right_collisions": len(right),
                "shared_collisions": len(left & right),
                "resolved_left_collisions": resolved,
                "created_right_collisions": created,
                "net_collision_change": len(right) - len(left),
                "exact_paired_p_unadjusted": exact_paired_p(resolved, created),
            }
        )
    return rows


def logical_clip020_path(panel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["run"], int(row["update"])): row for row in panel_rows}
    rows: list[dict[str, Any]] = []
    for update in (1, 5, 10, 15, 20, 25, 30):
        source = by_key[("ppo_privilege_gru_0722_long_clip020", update)]
        rows.append({**source, "logical_run": "clip0.20_45U", "source_run": source["run"]})
    for update in (35, 40, 45):
        source = by_key[("ppo_privilege_gru_0722_long45_clip020", update)]
        rows.append({**source, "logical_run": "clip0.20_45U", "source_run": source["run"]})
    return rows


def main() -> None:
    config_rows, configs = read_run_configs()
    training_updates, training_summary, training_sources = read_training(configs)
    panels = discover_panels()
    panel_rows, episode_rows, all_episodes, panel_dirs, scenario_physical = read_eval_panels(panels)
    collision_features, inherited_rows, tail_rows, bc_structural, bc_strict_5deg = trace_analysis(
        panels, all_episodes, panel_dirs
    )
    reproducibility = checkpoint_reproducibility()
    frequencies = scenario_frequency(panels, all_episodes, collision_features)
    pairwise = selected_pairwise(all_episodes)
    logical_path = logical_clip020_path(panel_rows)

    tail_by_panel = {(row["run"], int(row["update"])): row for row in tail_rows}
    for row in panel_rows:
        tail = tail_by_panel[(row["run"], int(row["update"]))]
        row.update({key: value for key, value in tail.items() if key not in {"group", "arm", "run", "update"}})

    physical_groups: dict[str, list[str]] = {}
    for scenario, key in scenario_physical.items():
        physical_groups.setdefault(key, []).append(scenario)
    duplicate_rows = [
        {
            "physical_initial_condition_key": key,
            "scenario_count": len(scenarios),
            "scenario_ids": ";".join(sorted(scenarios)),
        }
        for key, scenarios in physical_groups.items()
        if len(scenarios) > 1
    ]

    panel_by_key = {(row["run"], int(row["update"])): row for row in panel_rows}
    current_keys = {
        "BC": ("BC", 0),
        "G5_U30": ("ppo_privilege_gru_0722_long_clip020", 30),
        "G7_U40": ("ppo_privilege_gru_0722_long45_clip020", 40),
        "G7_U45": ("ppo_privilege_gru_0722_long45_clip020", 45),
        "G8_U45": ("ppo_privilege_gru_0722_long45_clip025", 45),
        "G9_U20": ("ppo_privilege_gru_0722_long45_clip020_hard", 20),
        "G9_U45": ("ppo_privilege_gru_0722_long45_clip020_hard", 45),
    }
    current_snapshot = {label: panel_by_key[key] for label, key in current_keys.items()}

    best_collision_count = min(int(row["collision_count"]) for row in panel_rows if row["run"] != "BC")
    best_panels = [
        {key: row[key] for key in ("group", "arm", "run", "update", "collision_count", "overtake_count", "follow_count")}
        for row in panel_rows
        if row["run"] != "BC" and int(row["collision_count"]) == best_collision_count
    ]

    summary = {
        "as_of": "2026-07-23",
        "git_commit": "4e0043ed26e57950546c685d7bddba83e756c1a5",
        "run_count": len(RUN_SPECS),
        "training_formal_update_rows": len(training_updates),
        "eval_panel_count_including_bc": len(panel_rows),
        "ppo_eval_panel_count": sum(row["run"] != "BC" for row in panel_rows),
        "valid_ppo_eval_panel_count": sum(
            row["run"] != "BC" and bool(row["summary_valid"]) for row in panel_rows
        ),
        "eval_episode_comparison_rows": len(episode_rows),
        "ppo_episode_comparison_rows": sum(row["run"] != "BC" for row in episode_rows),
        "bc_self_reference_rows": sum(row["run"] == "BC" for row in episode_rows),
        "expected_nominal_eval_episodes": len(panel_rows) * EXPECTED_PANEL_EPISODES,
        "summary_valid_panel_count": sum(bool(row["summary_valid"]) for row in panel_rows),
        "trace_complete_panel_count": sum(bool(row["trace_complete"]) for row in panel_rows),
        "panel_unique_scenario_ids": len(all_episodes[("BC", 0)]),
        "panel_unique_physical_initial_conditions": len(set(scenario_physical.values())),
        "duplicate_physical_initial_condition_groups": len(duplicate_rows),
        "duplicate_nominal_episode_rows": sum(int(row["scenario_count"]) - 1 for row in duplicate_rows),
        "bc_collision_count": len(collision_set(all_episodes[("BC", 0)])),
        "bc_structural_post_overtake_rear_contact_count": len(bc_structural),
        "bc_strict_high_sideslip_tail_5deg_count": len(bc_strict_5deg),
        "best_ppo_collision_count": best_collision_count,
        "best_ppo_panels": best_panels,
        "current_snapshot": current_snapshot,
        "long_extension_actor_exact_u1_u30": all(
            bool(row["actor_tensors_exact"])
            for row in reproducibility
            if row["comparison"] == "G5 clip0.20 -> G7 extension"
        ),
        "data_quality_issues": [
            "The nominal 600-scenario panel contains 592 unique physical initial conditions because the closed-raceline endpoint duplicates the start for 8 cross-raceline combinations.",
            "ppo_privilege_gru_0722_lr5_tkloff_u0020 has one failed/missing episode (599 JSON rows and 599 NPZ traces; error_count=1), so it is excluded from claims requiring a complete panel.",
            "0721 traces predate terminal-frame recording; collision labels/times therefore come from results_multi.json and their last trace frame is 0.01 s pre-impact.",
        ],
        "tail_classifier_definition": {
            "structural": "ego-opponent collision after relative progress crossed zero, with opponent center behind ego at impact/pre-impact",
            "merge": "absolute ego-frame lateral separation closed by at least 0.10 m during the preceding 0.5 s",
            "strict_5deg": "structural + merge + maximum pose-derived sideslip at least 5 degrees in the preceding 0.5 s",
            "sensitivity_thresholds_deg": [3, 5, 8],
        },
    }

    write_csv(OUT / "run_config_matrix.csv", config_rows)
    write_csv(OUT / "training_updates.csv", training_updates)
    write_csv(OUT / "training_run_summary.csv", training_summary)
    write_csv(OUT / "training_collision_source_by_update.csv", training_sources)
    write_csv(OUT / "eval_panels.csv", panel_rows)
    write_csv(OUT / "eval_episode_comparison.csv", episode_rows)
    write_csv(OUT / "collision_episode_features.csv", collision_features)
    write_csv(OUT / "bc_tail_scenario_outcomes.csv", inherited_rows)
    write_csv(OUT / "tail_issue_by_panel.csv", tail_rows)
    write_csv(OUT / "collision_scenario_frequency.csv", frequencies)
    write_csv(OUT / "checkpoint_reproducibility.csv", reproducibility)
    write_csv(OUT / "selected_paired_comparisons.csv", pairwise)
    write_csv(OUT / "logical_clip020_45u_path.csv", logical_path)
    write_csv(OUT / "physical_duplicate_groups.csv", duplicate_rows)
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

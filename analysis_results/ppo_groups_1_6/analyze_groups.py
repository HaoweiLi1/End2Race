#!/usr/bin/env python3
"""Reproducible audit of the End2Race PPO baseline and Groups 1-6.

The script is read-only with respect to training/evaluation artifacts. It writes
bounded derived CSV/JSON files into its own analysis output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent
POST_TRAINED = REPO / "post-trained"
EVAL_RESULTS = REPO / "eval_results"
CHECKPOINT_UPDATES_20 = (1, 5, 10, 15, 20)
CHECKPOINT_UPDATES_30 = (1, 5, 10, 15, 20, 25, 30)


@dataclass(frozen=True)
class RunSpec:
    run: str
    label: str
    group_memberships: tuple[str, ...]
    axis_value: str


RUN_SPECS = (
    RunSpec("ppo_independent_gru_0721_base", "independent_gru", ("Group 1",), "independent_gru"),
    RunSpec("ppo_privilege_mlp_0721_base", "privilege_mlp", ("Group 1",), "privilege_mlp"),
    RunSpec("ppo_privilege_gru_0721_base", "privilege_gru / shared baseline", ("Group 1", "Group 2", "Group 3", "Group 4", "Group 6"), "baseline"),
    RunSpec("ppo_privilege_gru_0721_bs25600", "batch 25600", ("Group 2",), "25600"),
    RunSpec("ppo_privilege_gru_0721_bs51200", "batch 51200", ("Group 2",), "51200"),
    RunSpec("ppo_privilege_gru_0721_clip010", "clip 0.10 (legacy workers=8)", ("Group 3",), "0.10"),
    RunSpec("ppo_privilege_gru_0721_clip020", "clip 0.20 (legacy workers=8)", ("Group 3",), "0.20"),
    RunSpec("ppo_privilege_gru_0722_lr1_tkloff", "actor LR 1x", ("Group 4",), "1e-6 / 1e-5"),
    RunSpec("ppo_privilege_gru_0722_lr5_tkloff", "actor LR 5x", ("Group 4",), "5e-6 / 5e-5"),
    RunSpec("ppo_privilege_gru_0722_long_clip015", "30 updates, clip 0.15", ("Group 5",), "0.15"),
    RunSpec("ppo_privilege_gru_0722_long_clip020", "30 updates, clip 0.20", ("Group 5",), "0.20"),
    RunSpec("ppo_privilege_gru_0722_clip015_tkl002", "target-KL 0.02 (workers=8)", ("Group 6",), "0.02"),
    RunSpec("ppo_privilege_gru_0722_clip015_tkl004", "target-KL 0.04", ("Group 6",), "0.04"),
)
RUN_SPEC_BY_NAME = {spec.run: spec for spec in RUN_SPECS}


GROUP_ARMS = {
    "Group 1": [
        "ppo_independent_gru_0721_base",
        "ppo_privilege_mlp_0721_base",
        "ppo_privilege_gru_0721_base",
    ],
    "Group 2": [
        "ppo_privilege_gru_0721_base",
        "ppo_privilege_gru_0721_bs25600",
        "ppo_privilege_gru_0721_bs51200",
    ],
    "Group 3": [
        "ppo_privilege_gru_0721_clip010",
        "ppo_privilege_gru_0721_base",
        "ppo_privilege_gru_0721_clip020",
    ],
    "Group 4": [
        "ppo_privilege_gru_0722_lr1_tkloff",
        "ppo_privilege_gru_0721_base",
        "ppo_privilege_gru_0722_lr5_tkloff",
    ],
    "Group 5": [
        "ppo_privilege_gru_0722_long_clip015",
        "ppo_privilege_gru_0722_long_clip020",
    ],
    "Group 6": [
        "ppo_privilege_gru_0721_base",
        "ppo_privilege_gru_0722_clip015_tkl002",
        "ppo_privilege_gru_0722_clip015_tkl004",
    ],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def mean_or_none(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(statistics.mean(clean)) if clean else None


def median_or_none(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(statistics.median(clean)) if clean else None


def rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def correlation(x: list[float], y: list[float], *, ranked: bool = False) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    xa = rankdata(x) if ranked else np.asarray(x, dtype=np.float64)
    ya = rankdata(y) if ranked else np.asarray(y, dtype=np.float64)
    if np.std(xa) == 0 or np.std(ya) == 0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_run_data() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    training_by_run: dict[str, list[dict[str, Any]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for spec in RUN_SPECS:
        directory = POST_TRAINED / spec.run
        config_doc = load_json(directory / "run_config.json")
        args = config_doc["args"]
        configs[spec.run] = config_doc
        metric_rows = [load_json_line for load_json_line in (
            json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
        )]
        warmup = [row for row in metric_rows if row.get("phase") == "warmup"]
        formal = [row for row in metric_rows if row.get("phase") == "formal"]
        training_by_run[spec.run] = formal
        expected_updates = list(range(1, int(args["num_updates"]) + 1))
        update_sequence_valid = [int(row["update"]) for row in formal] == expected_updates
        checkpoint_complete = all(
            (directory / "checkpoints" / f"actor_u{update:04d}.pth").is_file()
            and (directory / "checkpoints" / f"critic_u{update:04d}.pt").is_file()
            for update in expected_updates
        )
        planned_steps_per_update = int(args["actor_epochs"] * args["n_envs"] * args["n_steps"] / args["batch_size"])
        actual_steps = sum(int(row.get("actor_optimizer_steps_completed", planned_steps_per_update)) for row in formal)
        planned_steps = sum(int(row.get("actor_optimizer_steps_planned", planned_steps_per_update)) for row in formal)
        clear_high = [
            float(value)
            for row in formal
            for value in (row.get("privileged_feature_saturation_high") or [])[9:12]
        ]
        run_rows.append({
            "run": spec.run,
            "label": spec.label,
            "groups": ", ".join(spec.group_memberships),
            "axis_value": spec.axis_value,
            "started_at": config_doc["started_at"],
            "critic": args["critic"],
            "n_envs": args["n_envs"],
            "env_workers": args["env_workers"],
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
            "warmup_epochs": warmup[0]["epochs"],
            "warmup_best_epoch": warmup[0]["best_epoch"],
            "warmup_best_validation_loss": warmup[0]["best_validation_loss"],
            "formal_rows": len(formal),
            "formal_updates_sequential": update_sequence_valid,
            "metrics_finite": all(finite_tree(row) for row in metric_rows),
            "checkpoint_complete": checkpoint_complete,
            "planned_actor_steps_per_update": planned_steps_per_update,
            "actor_steps_completed": actual_steps,
            "actor_steps_planned": planned_steps,
            "early_stop_updates": sum(bool(row.get("actor_early_stop_triggered")) for row in formal),
            "mean_approx_kl": mean_or_none(row["approx_kl_mean"] for row in formal),
            "max_approx_kl": max(float(row["approx_kl_max"]) for row in formal),
            "mean_clip_fraction": mean_or_none(row["clip_fraction_mean"] for row in formal),
            "median_actor_grad_norm_preclip": median_or_none(row["actor_grad_norm_mean"] for row in formal),
            "max_actor_grad_norm_preclip": max(float(row["actor_grad_norm_max"]) for row in formal),
            "mean_value_loss_reduction": mean_or_none(
                (float(row["value_loss_pre_update"]) - float(row["value_loss_post_update"]))
                / float(row["value_loss_pre_update"])
                for row in formal if float(row["value_loss_pre_update"]) != 0
            ),
            "final_value_loss_post": formal[-1]["value_loss_post_update"],
            "final_explained_variance_post": formal[-1]["explained_variance_post_update"],
            "final_collision_explained_variance_post": formal[-1]["collision_explained_variance_post"],
            "final_ordinary_explained_variance_post": formal[-1]["ordinary_explained_variance_post"],
            "mean_training_collision_rate": mean_or_none(
                float(row["ego_collision_count"]) / float(row["episode_count"])
                for row in formal if row.get("episode_count")
            ),
            "max_clearance_exact_high_saturation": max(clear_high) if clear_high else None,
            "rollout_wall_seconds_total": sum(float(row["rollout_wall_seconds"]) for row in formal),
            "actor_wall_seconds_total": sum(float(row["actor_train_wall_seconds"]) for row in formal),
            "critic_wall_seconds_total": sum(float(row["critic_train_wall_seconds"]) for row in formal),
        })
    return run_rows, training_by_run, configs


def eval_directory(run: str, update: int | None) -> Path:
    if run == "BC":
        return EVAL_RESULTS / "end2race_Austin" / "multiagents"
    assert update is not None
    return EVAL_RESULTS / f"{run}_u{update:04d}_Austin" / "multiagents"


def eval_updates(run: str, configs: dict[str, dict[str, Any]]) -> tuple[int, ...]:
    if run == "BC":
        return (0,)
    num_updates = int(configs[run]["args"]["num_updates"])
    return CHECKPOINT_UPDATES_30 if num_updates == 30 else CHECKPOINT_UPDATES_20


def read_eval_data(configs: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, dict[str, Any]]], set[str]]:
    panel_rows: list[dict[str, Any]] = []
    episodes_by_panel: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    bc_scenarios: set[str] = set()
    for run in ("BC", *(spec.run for spec in RUN_SPECS)):
        for update in eval_updates(run, configs):
            directory = eval_directory(run, update)
            document = load_json(directory / "results_multi.json")
            episodes = document["episodes"]
            final = document["final"]
            scenario_rows = {str(row["scenario_id"]): row for row in episodes.values()}
            episodes_by_panel[(run, update)] = scenario_rows
            scenario_ids = set(scenario_rows)
            if run == "BC":
                bc_scenarios = scenario_ids
            collisions = {sid for sid, row in scenario_rows.items() if row.get("ego_collision_occurred") is True}
            overtake = sum(row.get("outcome") == "overtake" for row in scenario_rows.values())
            follow = sum(row.get("outcome") == "follow" for row in scenario_rows.values())
            trace_files = {path.stem for path in (directory / "traces").glob("*.npz")}
            aggregate_consistent = (
                final.get("collision_count") == len(collisions)
                and final.get("overtaking_count") == overtake
                and final.get("following_count") == follow
                and final.get("success_count") == overtake + follow
            )
            valid = (
                len(episodes) == 600
                and len(scenario_ids) == 600
                and final.get("total_episodes") == 600
                and final.get("error_count") == 0
                and aggregate_consistent
                and len(collisions) + overtake + follow == 600
                and all(row.get("action_finite") is True and row.get("observation_finite") is True for row in scenario_rows.values())
                and trace_files == set(episodes)
            )
            collision_rows = [row for row in scenario_rows.values() if row.get("ego_collision_occurred")]
            panel_rows.append({
                "run": run,
                "update": update,
                "panel": "BC" if run == "BC" else f"{run}_u{update:04d}",
                "episode_rows": len(episodes),
                "unique_scenarios": len(scenario_ids),
                "trace_files": len(trace_files),
                "collision_count": len(collisions),
                "collision_rate": len(collisions) / 600.0,
                "overtake_count": overtake,
                "follow_count": follow,
                "success_count": overtake + follow,
                "success_rate": (overtake + follow) / 600.0,
                "error_count": final.get("error_count"),
                "avg_speed_mean": final.get("avg_speed_mean"),
                "speed_variance_mean": final.get("speed_variance_mean"),
                "total_distance_mean": final.get("total_distance_mean"),
                "median_collision_time_s": median_or_none(row.get("ego_collision_time_s") for row in collision_rows),
                "mean_collision_time_s": mean_or_none(row.get("ego_collision_time_s") for row in collision_rows),
                "initial_ego_collision_count": sum(bool(row.get("initial_ego_collision")) for row in collision_rows),
                "collision_with_opponent_count": sum(bool(row.get("opp_collision_occurred")) for row in collision_rows),
                "steering_anomaly_episode_count": sum(bool(row.get("steering_anomaly_timesteps")) for row in scenario_rows.values()),
                "near_proximity_episode_count": sum(bool(row.get("proximity_below_threshold_timesteps")) for row in scenario_rows.values()),
                "aggregate_consistent": aggregate_consistent,
                "all_action_observation_finite": all(row.get("action_finite") is True and row.get("observation_finite") is True for row in scenario_rows.values()),
                "trace_keys_match_episode_keys": trace_files == set(episodes),
                "matches_bc_scenario_set": None if run == "BC" else scenario_ids == bc_scenarios,
                "valid": valid,
            })
    return panel_rows, episodes_by_panel, bc_scenarios


def join_training_eval(panel_rows: list[dict[str, Any]], training_by_run: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for panel in panel_rows:
        if panel["run"] == "BC":
            continue
        training = {int(row["update"]): row for row in training_by_run[panel["run"]]}[int(panel["update"])]
        output.append({
            **panel,
            "training_ego_collision_count": training["ego_collision_count"],
            "training_episode_count": training["episode_count"],
            "training_collision_rate": float(training["ego_collision_count"]) / float(training["episode_count"]),
            "mean_episode_return": training["mean_episode_return"],
            "value_loss_post": training["value_loss_post_update"],
            "explained_variance_post": training["explained_variance_post_update"],
            "collision_explained_variance_post": training["collision_explained_variance_post"],
            "ordinary_explained_variance_post": training["ordinary_explained_variance_post"],
            "approx_kl_mean": training["approx_kl_mean"],
            "approx_kl_max": training["approx_kl_max"],
            "clip_fraction_mean": training["clip_fraction_mean"],
            "actor_grad_norm_mean_preclip": training["actor_grad_norm_mean"],
            "critic_grad_norm_mean_preclip": training["critic_grad_norm_mean"],
            "actor_optimizer_steps_completed": training.get("actor_optimizer_steps_completed"),
            "actor_optimizer_steps_planned": training.get("actor_optimizer_steps_planned"),
            "actor_early_stop_triggered": training.get("actor_early_stop_triggered"),
        })
    return output


def collision_set(panel: dict[str, dict[str, Any]]) -> set[str]:
    return {sid for sid, row in panel.items() if row.get("ego_collision_occurred") is True}


def pairwise_row(label: str, a_key: tuple[str, int], b_key: tuple[str, int], episodes_by_panel: dict[tuple[str, int], dict[str, dict[str, Any]]]) -> dict[str, Any]:
    a = collision_set(episodes_by_panel[a_key])
    b = collision_set(episodes_by_panel[b_key])
    union = a | b
    return {
        "comparison": label,
        "a_run": a_key[0],
        "a_update": a_key[1],
        "b_run": b_key[0],
        "b_update": b_key[1],
        "a_collisions": len(a),
        "b_collisions": len(b),
        "shared": len(a & b),
        "resolved": len(a - b),
        "created": len(b - a),
        "net_change": len(b) - len(a),
        "jaccard": len(a & b) / len(union) if union else 1.0,
    }


def build_pairwise(episodes_by_panel: dict[tuple[str, int], dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    baseline = ("ppo_privilege_gru_0721_base", 20)
    pairs = [
        ("BC -> shared PPO baseline U20", ("BC", 0), baseline),
        ("G1 independent -> privilege_gru U20", ("ppo_independent_gru_0721_base", 20), baseline),
        ("G1 privilege_mlp -> privilege_gru U20", ("ppo_privilege_mlp_0721_base", 20), baseline),
        ("G2 batch 12800 -> 25600 U20", baseline, ("ppo_privilege_gru_0721_bs25600", 20)),
        ("G2 batch 12800 -> 51200 U20", baseline, ("ppo_privilege_gru_0721_bs51200", 20)),
        ("G3 clip 0.15 -> 0.10 U20", baseline, ("ppo_privilege_gru_0721_clip010", 20)),
        ("G3 clip 0.15 -> legacy 0.20 U20", baseline, ("ppo_privilege_gru_0721_clip020", 20)),
        ("G4 LR3 -> LR1 U20", baseline, ("ppo_privilege_gru_0722_lr1_tkloff", 20)),
        ("G4 LR3 -> LR5 U15", baseline, ("ppo_privilege_gru_0722_lr5_tkloff", 15)),
        ("G5 clip 0.15 U20 -> U30", ("ppo_privilege_gru_0722_long_clip015", 20), ("ppo_privilege_gru_0722_long_clip015", 30)),
        ("G5 clip 0.20 U20 -> U30", ("ppo_privilege_gru_0722_long_clip020", 20), ("ppo_privilege_gru_0722_long_clip020", 30)),
        ("G5 clip 0.15 U30 -> 0.20 U30", ("ppo_privilege_gru_0722_long_clip015", 30), ("ppo_privilege_gru_0722_long_clip020", 30)),
        ("G6 KL off -> 0.02 U20", baseline, ("ppo_privilege_gru_0722_clip015_tkl002", 20)),
        ("G6 KL off -> 0.04 U20", baseline, ("ppo_privilege_gru_0722_clip015_tkl004", 20)),
        ("BC -> selected clip 0.20 U30", ("BC", 0), ("ppo_privilege_gru_0722_long_clip020", 30)),
    ]
    return [pairwise_row(label, a, b, episodes_by_panel) for label, a, b in pairs]


def group_summary(panel_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    panels = {(row["run"], int(row["update"])): row for row in panel_rows}
    runs = {row["run"]: row for row in run_rows}
    output = []
    for group, arms in GROUP_ARMS.items():
        for run in arms:
            updates = CHECKPOINT_UPDATES_30 if runs[run]["num_updates"] == 30 else CHECKPOINT_UPDATES_20
            eval_rows = [panels[(run, update)] for update in updates]
            stable = [row for row in eval_rows if row["valid"] and int(row["update"]) >= 5]
            valid_rows = [row for row in eval_rows if row["valid"]]
            best = min(valid_rows, key=lambda row: (row["collision_count"], -row["overtake_count"], row["update"]))
            final = eval_rows[-1]
            output.append({
                "group": group,
                "run": run,
                "label": RUN_SPEC_BY_NAME[run].label,
                "axis_value": RUN_SPEC_BY_NAME[run].axis_value,
                "env_workers": runs[run]["env_workers"],
                "planned_actor_steps_per_update": runs[run]["planned_actor_steps_per_update"],
                "collision_path": " / ".join(str(row["collision_count"]) + ("*" if not row["valid"] else "") for row in eval_rows),
                "mean_collision_u5_plus": mean_or_none(row["collision_count"] for row in stable),
                "best_update": best["update"],
                "best_collision_count": best["collision_count"],
                "best_overtake_count": best["overtake_count"],
                "final_update": final["update"],
                "final_collision_count": final["collision_count"],
                "final_overtake_count": final["overtake_count"],
                "final_follow_count": final["follow_count"],
                "final_eval_valid": final["valid"],
                "warmup_best_validation_loss": runs[run]["warmup_best_validation_loss"],
                "mean_approx_kl": runs[run]["mean_approx_kl"],
                "max_approx_kl": runs[run]["max_approx_kl"],
                "mean_clip_fraction": runs[run]["mean_clip_fraction"],
                "final_explained_variance_post": runs[run]["final_explained_variance_post"],
                "early_stop_updates": runs[run]["early_stop_updates"],
                "actor_steps_completed": runs[run]["actor_steps_completed"],
                "actor_steps_planned": runs[run]["actor_steps_planned"],
            })
    return output


def group_control_audit(configs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = [
        ("Group 1", "ppo_privilege_gru_0721_base", "ppo_independent_gru_0721_base", "critic architecture"),
        ("Group 1", "ppo_privilege_gru_0721_base", "ppo_privilege_mlp_0721_base", "critic architecture"),
        ("Group 2", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0721_bs25600", "batch size"),
        ("Group 2", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0721_bs51200", "batch size"),
        ("Group 3", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0721_clip010", "clip range"),
        ("Group 3", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0721_clip020", "clip range"),
        ("Group 3 within workers=8", "ppo_privilege_gru_0721_clip010", "ppo_privilege_gru_0721_clip020", "clip range"),
        ("Group 4", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0722_lr1_tkloff", "actor GRU/head learning rates"),
        ("Group 4", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0722_lr5_tkloff", "actor GRU/head learning rates"),
        ("Group 5", "ppo_privilege_gru_0722_long_clip015", "ppo_privilege_gru_0722_long_clip020", "clip range"),
        ("Group 6", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0722_clip015_tkl002", "target-KL"),
        ("Group 6", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0722_clip015_tkl004", "target-KL"),
    ]
    ignored = {"output_dir"}
    output = []
    for group, baseline, arm, intended_axis in comparisons:
        left = configs[baseline]["args"]
        right = configs[arm]["args"]
        keys = sorted((set(left) | set(right)) - ignored)
        differences = [key for key in keys if left.get(key) != right.get(key)]
        intended_keys = {
            "critic architecture": {"critic"},
            "batch size": {"batch_size"},
            "clip range": {"clip_range"},
            "actor GRU/head learning rates": {"gru_learning_rate", "head_learning_rate"},
            "target-KL": {"target_kl"},
        }[intended_axis]
        confounds = [key for key in differences if key not in intended_keys]
        output.append({
            "group": group,
            "baseline_run": baseline,
            "arm_run": arm,
            "intended_axis": intended_axis,
            "recorded_differences": ", ".join(differences),
            "confounds": ", ".join(confounds),
            "strict_single_axis": not confounds,
        })
    return output


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
    dtypes: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            key = member[:-4]
            with archive.open(member) as stream:
                shape, dtype = numpy_header(stream)
            shapes[key] = shape
            dtypes[key] = str(dtype)
    leading_lengths = {shape[0] for shape in shapes.values() if shape}
    numeric = all(
        np.issubdtype(np.dtype(dtype), np.number) or np.issubdtype(np.dtype(dtype), np.bool_)
        for dtype in dtypes.values()
    )
    is_new = "terminal_post_step" in shapes and "action_applied" in shapes
    terminal_valid = None
    collision_marker_matches = None
    if is_new:
        with np.load(path, allow_pickle=False) as arrays:
            terminal = arrays["terminal_post_step"]
            applied = arrays["action_applied"]
            collisions = arrays["collisions"]
            terminal_valid = (
                int(np.count_nonzero(terminal)) == 1
                and bool(terminal[-1])
                and int(np.count_nonzero(applied)) == len(applied) - 1
                and not bool(applied[-1])
            )
            collision_marker_matches = bool(collisions[-1, 0]) == expected_collision
    else:
        with np.load(path, allow_pickle=False) as arrays:
            collisions = arrays["collisions"]
            collision_marker_matches = bool(np.any(collisions[:, 0])) == expected_collision
    return {
        "format": "post_step_v2" if is_new else "legacy_pre_post_step",
        "array_count": len(shapes),
        "aligned": len(leading_lengths) == 1,
        "numeric": numeric,
        "terminal_valid": terminal_valid,
        "collision_marker_matches": collision_marker_matches,
    }


def scan_npz(episodes_by_panel: dict[tuple[str, int], dict[str, dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    by_format: dict[str, Counter] = {}
    panel_issues: list[dict[str, Any]] = []
    for (run, update), episodes in sorted(episodes_by_panel.items()):
        trace_dir = eval_directory(run, update) / "traces"
        panel_counts = Counter()
        for _scenario_id, episode in sorted(episodes.items()):
            episode_key = str(episode["episode_key"])
            trace_path = trace_dir / f"{episode_key}.npz"
            result = audit_trace_file(trace_path, bool(episode["ego_collision_occurred"]))
            fmt = result["format"]
            format_counts = by_format.setdefault(fmt, Counter())
            panel_counts["files"] += 1
            totals["files"] += 1
            for name in ("aligned", "numeric"):
                panel_counts[f"{name}_{result[name]}"] += 1
                format_counts[f"{name}_{result[name]}"] += 1
                totals[f"{name}_{result[name]}"] += 1
            if result["terminal_valid"] is not None:
                panel_counts[f"terminal_valid_{result['terminal_valid']}"] += 1
                format_counts[f"terminal_valid_{result['terminal_valid']}"] += 1
                totals[f"terminal_valid_{result['terminal_valid']}"] += 1
            panel_counts[f"collision_marker_matches_{result['collision_marker_matches']}"] += 1
            format_counts[f"collision_marker_matches_{result['collision_marker_matches']}"] += 1
            totals[f"collision_marker_matches_{result['collision_marker_matches']}"] += 1
            format_counts["files"] += 1
        if panel_counts.get("aligned_False") or panel_counts.get("numeric_False") or panel_counts.get("terminal_valid_False"):
            panel_issues.append({"run": run, "update": update, **dict(panel_counts)})
    return {
        "totals": dict(totals),
        "by_format": {name: dict(counts) for name, counts in by_format.items()},
        "structural_panel_issues": panel_issues,
        "interpretation": (
            "Legacy 0721 traces omit terminal_post_step/action_applied and therefore do not record the "
            "terminal collision state. Collision truth for those panels must come from results_multi.json."
        ),
    }


def tensor_equivalence() -> dict[str, Any]:
    import torch

    results: dict[str, Any] = {}
    for spec in RUN_SPECS:
        directory = POST_TRAINED / spec.run
        config = load_json(directory / "run_config.json")["args"]
        update = int(config["num_updates"])
        actor_final = torch.load(directory / "actor_final.pth", map_location="cpu", weights_only=True)
        checkpoint = torch.load(directory / "checkpoints" / f"actor_u{update:04d}.pth", map_location="cpu", weights_only=True)
        keys_match = set(actor_final) == set(checkpoint)
        max_abs_difference = max(
            float((actor_final[key] - checkpoint[key]).abs().max().item())
            for key in actor_final if torch.is_tensor(actor_final[key])
        )
        results[spec.run] = {
            "keys_match": keys_match,
            "tensor_equal": keys_match and max_abs_difference == 0.0,
            "max_abs_difference": max_abs_difference,
        }
    comparisons = {}
    for name, a, b in (
        ("shared_baseline_vs_long_clip015", "ppo_privilege_gru_0721_base", "ppo_privilege_gru_0722_long_clip015"),
        ("legacy_clip020_vs_long_clip020", "ppo_privilege_gru_0721_clip020", "ppo_privilege_gru_0722_long_clip020"),
    ):
        updates = {}
        for update in CHECKPOINT_UPDATES_20:
            left = torch.load(POST_TRAINED / a / "checkpoints" / f"actor_u{update:04d}.pth", map_location="cpu", weights_only=True)
            right = torch.load(POST_TRAINED / b / "checkpoints" / f"actor_u{update:04d}.pth", map_location="cpu", weights_only=True)
            max_diff = max(float((left[key] - right[key]).abs().max().item()) for key in left)
            updates[str(update)] = {"tensor_equal": max_diff == 0.0, "max_abs_difference": max_diff}
        comparisons[name] = updates
    return {"final_actor_vs_last_checkpoint": results, "cross_run": comparisons}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def training_eval_correlations(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in joined if row["valid"]]
    fields = [
        ("training_collision_rate", "Training rollout collision rate"),
        ("mean_episode_return", "Training mean episode return"),
        ("value_loss_post", "Critic post-update value loss"),
        ("explained_variance_post", "Critic post-update explained variance"),
        ("approx_kl_mean", "Actor mean approximate KL"),
        ("clip_fraction_mean", "Actor clip fraction"),
    ]
    output = []
    y = [float(row["collision_count"]) for row in valid]
    for field, label in fields:
        x = [float(row[field]) for row in valid]
        output.append({
            "metric": field,
            "label": label,
            "panels": len(x),
            "pearson_with_eval_collision_count": correlation(x, y),
            "spearman_with_eval_collision_count": correlation(x, y, ranked=True),
            "scope": "pooled descriptive correlation across heterogeneous runs; not causal",
        })
    return output


def scenario_frequency(episodes_by_panel: dict[tuple[str, int], dict[str, dict[str, Any]]], panel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_keys = {(row["run"], int(row["update"])) for row in panel_rows if row["valid"] and row["run"] != "BC"}
    frequencies = Counter()
    for key in valid_keys:
        frequencies.update(collision_set(episodes_by_panel[key]))
    return [
        {"scenario_id": scenario, "collision_panels": count, "valid_ppo_panels": len(valid_keys), "rate": count / len(valid_keys)}
        for scenario, count in frequencies.most_common()
    ]


def trace_summary_selected(episodes_by_panel: dict[tuple[str, int], dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    selected = [
        ("BC", 0, "BC"),
        ("ppo_privilege_gru_0721_base", 20, "Shared PPO baseline U20"),
        ("ppo_privilege_gru_0722_long_clip015", 30, "Group 5 clip 0.15 U30"),
        ("ppo_privilege_gru_0722_long_clip020", 30, "Group 5 clip 0.20 U30"),
        ("ppo_privilege_gru_0722_clip015_tkl002", 20, "Group 6 target-KL 0.02 U20"),
        ("ppo_privilege_gru_0722_clip015_tkl004", 20, "Group 6 target-KL 0.04 U20"),
    ]
    output = []
    for run, update, label in selected:
        collisions = [row for row in episodes_by_panel[(run, update)].values() if row["ego_collision_occurred"]]
        output.append({
            "label": label,
            "run": run,
            "update": update,
            "collision_count": len(collisions),
            "collision_with_opponent_count": sum(bool(row["opp_collision_occurred"]) for row in collisions),
            "initial_collision_count": sum(bool(row["initial_ego_collision"]) for row in collisions),
            "min_collision_time_s": min(float(row["ego_collision_time_s"]) for row in collisions),
            "median_collision_time_s": statistics.median(float(row["ego_collision_time_s"]) for row in collisions),
            "max_collision_time_s": max(float(row["ego_collision_time_s"]) for row in collisions),
            "raceline0_count": sum("-raceline0-" in row["scenario_id"] for row in collisions),
            "raceline1_count": sum("-raceline1-" in row["scenario_id"] for row in collisions),
            "raceline2_count": sum("-raceline2-" in row["scenario_id"] for row in collisions),
            "speed_0_5_count": sum(row["scenario_id"].endswith("-v0.5") for row in collisions),
            "speed_0_6_count": sum(row["scenario_id"].endswith("-v0.6") for row in collisions),
            "speed_0_7_count": sum(row["scenario_id"].endswith("-v0.7") for row in collisions),
            "speed_0_8_count": sum(row["scenario_id"].endswith("-v0.8") for row in collisions),
        })
    return output


def leakage_check(episodes_by_panel: dict[tuple[str, int], dict[str, dict[str, Any]]]) -> dict[str, Any]:
    training_run = POST_TRAINED / "ppo_privilege_gru_0721_base"
    collision_pool = load_json(training_run / "collision_scenarios.json")
    ordinary_pool = load_json(training_run / "ordinary_scenarios.json")
    training_ego = {int(row["ego_idx"]) for row in collision_pool + ordinary_pool}
    evaluation_ego = {
        int(re.search(r"-ego(\d+)-", scenario_id).group(1))
        for scenario_id in episodes_by_panel[("BC", 0)]
    }
    return {
        "collision_pool_rows": len(collision_pool),
        "ordinary_pool_rows": len(ordinary_pool),
        "training_unique_ego_indices": len(training_ego),
        "evaluation_unique_ego_indices": len(evaluation_ego),
        "ego_index_overlap_count": len(training_ego & evaluation_ego),
        "collision_pool_sha256": sha256(training_run / "collision_scenarios.json"),
        "ordinary_pool_sha256": sha256(training_run / "ordinary_scenarios.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-npz-audit", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows, training_by_run, configs = read_run_data()
    panel_rows, episodes_by_panel, bc_scenarios = read_eval_data(configs)
    joined = join_training_eval(panel_rows, training_by_run)
    group_rows = group_summary(panel_rows, run_rows)
    control_audit = group_control_audit(configs)
    pairwise = build_pairwise(episodes_by_panel)
    correlations = training_eval_correlations(joined)
    scenario_rows = scenario_frequency(episodes_by_panel, panel_rows)
    trace_selected = trace_summary_selected(episodes_by_panel)
    leakage = leakage_check(episodes_by_panel)
    tensor_checks = tensor_equivalence()

    npz_path = output_dir / "npz_audit.json"
    if args.reuse_npz_audit and npz_path.is_file():
        npz_audit = load_json(npz_path)
    else:
        npz_audit = scan_npz(episodes_by_panel)

    valid_panels = [row for row in panel_rows if row["valid"]]
    invalid_panels = [row for row in panel_rows if not row["valid"]]
    best_panel = min(valid_panels, key=lambda row: (row["collision_count"], -row["overtake_count"]))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(REPO),
        "scope": {
            "baseline": "pretrained/end2race.pth",
            "groups": list(GROUP_ARMS),
            "evaluation": "Austin, 50 ego starts x 3 opponent racelines x 4 speed scales = 600 scenarios",
            "collision_scope": "ego",
            "primary_metric": "ego collision count; follow and overtake are both success outcomes",
        },
        "quality": {
            "panels": len(panel_rows),
            "valid_panels": len(valid_panels),
            "invalid_panels": invalid_panels,
            "all_training_metrics_finite": all(row["metrics_finite"] for row in run_rows),
            "all_training_updates_complete": all(row["formal_updates_sequential"] and row["checkpoint_complete"] for row in run_rows),
            "common_scenario_set_valid_panels": all(row["matches_bc_scenario_set"] is not False for row in valid_panels),
            "npz": npz_audit,
            "tensor_checks": tensor_checks,
            "leakage": leakage,
        },
        "best_panel": best_panel,
        "baseline_panel": next(row for row in panel_rows if row["run"] == "BC"),
        "group_summary": group_rows,
        "group_control_audit": control_audit,
        "pairwise": pairwise,
        "correlations": correlations,
        "trace_summary_selected": trace_selected,
        "persistent_scenarios": scenario_rows[:30],
    }

    write_csv(output_dir / "run_summary.csv", run_rows)
    write_csv(output_dir / "panel_metrics.csv", panel_rows)
    write_csv(output_dir / "training_eval_join.csv", joined)
    write_csv(output_dir / "group_summary.csv", group_rows)
    write_csv(output_dir / "group_control_audit.csv", control_audit)
    write_csv(output_dir / "scenario_pairwise.csv", pairwise)
    write_csv(output_dir / "training_eval_correlations.csv", correlations)
    write_csv(output_dir / "scenario_frequency.csv", scenario_rows)
    write_csv(output_dir / "trace_summary_selected.csv", trace_selected)
    npz_path.write_text(json.dumps(npz_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(output_dir),
        "panels": len(panel_rows),
        "valid_panels": len(valid_panels),
        "invalid_panels": len(invalid_panels),
        "best_panel": best_panel["panel"],
        "best_collisions": best_panel["collision_count"],
        "npz_files": npz_audit["totals"].get("files"),
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build reproducible evidence for deciding whether to test clip_range > 0.20."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_results" / "clip_extension_decision"
TRAINED = ROOT / "post-trained"
EVAL = ROOT / "eval_results"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "clip 0.15": "ppo_privilege_gru_0722_long_clip015",
    "clip 0.20": "ppo_privilege_gru_0722_long_clip020",
}
UPDATES = (1, 5, 10, 15, 20, 25, 30)
SCENARIO_PATTERN = re.compile(
    r"collision-sp(?P<startpoint>\d+)-ego(?P<ego>\d+)-(?P<raceline>raceline\d+)-i(?P<interval>\d+)-v(?P<speed>\d+)"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def panel(run: str, update: int) -> dict[str, dict[str, Any]]:
    path = EVAL / f"{run}_u{update:04d}_Austin" / "multiagents" / "results_multi.json"
    payload = json.loads(path.read_text())
    rows = list(payload["episodes"].values())
    if len(rows) != 600:
        raise ValueError(f"Expected 600 Austin scenarios at {path}, got {len(rows)}")
    ids = [row["scenario_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate scenario ids at {path}")
    return {row["scenario_id"]: row for row in rows}


def collision_set(rows: dict[str, dict[str, Any]]) -> set[str]:
    return {scenario for scenario, row in rows.items() if row["ego_collision_occurred"]}


def exact_paired_p(resolved: int, created: int) -> float:
    n = resolved + created
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(resolved, created) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    path = OUT / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


eval_rows: list[dict[str, Any]] = []
panels: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
for clip_label, run in RUNS.items():
    for update in UPDATES:
        rows = panel(run, update)
        panels[(clip_label, update)] = rows
        collisions = [row for row in rows.values() if row["ego_collision_occurred"]]
        eval_rows.append(
            {
                "clip": clip_label,
                "update": update,
                "collisions": len(collisions),
                "success_rate": 1.0 - len(collisions) / 600.0,
                "wall_like_collisions": sum(not bool(row["opp_collision_occurred"]) for row in collisions),
                "overtakes": sum(row["outcome"] == "overtake" for row in rows.values()),
                "follows": sum(row["outcome"] == "follow" for row in rows.values()),
                "mean_speed_mps": sum(float(row["avg_speed"]) for row in rows.values()) / 600.0,
                "mean_min_surface_m": sum(float(row["global_min_surface_dist"]) for row in rows.values()) / 600.0,
                "scenario_count": 600,
            }
        )

paired_rows: list[dict[str, Any]] = []
for update in UPDATES:
    left = collision_set(panels[("clip 0.15", update)])
    right = collision_set(panels[("clip 0.20", update)])
    resolved = len(left - right)
    created = len(right - left)
    paired_rows.append(
        {
            "update": update,
            "clip015_collisions": len(left),
            "clip020_collisions": len(right),
            "shared": len(left & right),
            "resolved_by_clip020": resolved,
            "created_by_clip020": created,
            "net_collision_change": len(right) - len(left),
            "paired_exact_p_unadjusted": exact_paired_p(resolved, created),
        }
    )

telemetry_rows: list[dict[str, Any]] = []
for clip_label, run in RUNS.items():
    formal = [row for row in read_jsonl(TRAINED / run / "metrics.jsonl") if row.get("phase") == "formal" and int(row["update"]) >= 2]
    clip_fractions = [float(row["clip_fraction_mean"]) for row in formal]
    kl_means = [float(row["approx_kl_mean"]) for row in formal]
    kl_maxima = [float(row["approx_kl_max"]) for row in formal]
    telemetry_rows.append(
        {
            "clip": clip_label,
            "updates": len(formal),
            "mean_clip_fraction": statistics.fmean(clip_fractions),
            "median_clip_fraction": statistics.median(clip_fractions),
            "mean_approx_kl": statistics.fmean(kl_means),
            "updates_kl_max_gt_0_5": sum(value > 0.5 for value in kl_maxima),
            "updates_kl_max_gt_1": sum(value > 1.0 for value in kl_maxima),
            "largest_kl_max": max(kl_maxima),
        }
    )

# U20-U30 is the observed late-training window.  Formal row U contains update-U
# optimization telemetry but rollout outcomes from policy U-1; preserve that label.
long020_formal = {
    int(row["update"]): row
    for row in read_jsonl(TRAINED / RUNS["clip 0.20"] / "metrics.jsonl")
    if row.get("phase") == "formal"
}


def actor_state(update: int) -> dict[str, torch.Tensor]:
    path = TRAINED / RUNS["clip 0.20"] / "checkpoints" / f"actor_u{update:04d}.pth"
    return torch.load(path, map_location="cpu", weights_only=True)


late_training_rows: list[dict[str, Any]] = []
previous_actor = actor_state(19)
for update in range(20, 31):
    current_actor = actor_state(update)
    squared_delta = sum(float(((current_actor[key] - previous_actor[key]).double() ** 2).sum()) for key in current_actor)
    squared_previous = sum(float((previous_actor[key].double() ** 2).sum()) for key in previous_actor)
    row = long020_formal[update]
    late_training_rows.append(
        {
            "update": update,
            "rollout_policy_update": int(row["rollout_policy_update"]),
            "approx_kl_mean": float(row["approx_kl_mean"]),
            "approx_kl_max": float(row["approx_kl_max"]),
            "clip_fraction_mean": float(row["clip_fraction_mean"]),
            "explained_variance_post": float(row["explained_variance_post_update"]),
            "rollout_collision_count": int(row["ego_collision_count"]),
            "rollout_episode_count": int(row["episode_count"]),
            "actor_step_l2": math.sqrt(squared_delta),
            "actor_step_relative_l2": math.sqrt(squared_delta / squared_previous),
        }
    )
    previous_actor = current_actor

# Gamma is fixed at 0.999.  These rows make the temporal reach of possible
# lambda changes explicit at the simulator's 100 Hz step rate.
discount_rows: list[dict[str, Any]] = []
for gae_lambda in (0.99, 0.995, 0.9975, 1.0):
    decay = 0.999 * gae_lambda
    discount_rows.append(
        {
            "gamma": 0.999,
            "gae_lambda": gae_lambda,
            "gamma_times_lambda": decay,
            "td_error_half_life_seconds": math.log(0.5) / math.log(decay) / 100.0,
            "geometric_horizon_seconds": 1.0 / (1.0 - decay) / 100.0,
            "weight_after_1s": decay**100,
            "weight_after_2s": decay**200,
            "weight_after_4s": decay**400,
            "weight_after_8s": decay**800,
        }
    )

# Recompute the hard-neighbor evidence directly from the cache outcome lattice.
hard_probe = json.loads((ROOT / "analysis_results" / "hard_neighbor_probe" / "summary.json").read_text())
cache_summary = json.loads((TRAINED / "collision-cache" / "default" / "classification_summary.json").read_text())
outcomes = read_jsonl(TRAINED / "collision-cache" / "default" / "candidate_outcomes.jsonl")
parsed: dict[tuple[int, str, int, float], dict[str, Any]] = {}
for row in outcomes:
    match = SCENARIO_PATTERN.fullmatch(row["scenario_id"])
    if match is None:
        raise ValueError(f"Unexpected collision scenario id: {row['scenario_id']}")
    key = (
        int(match.group("ego")),
        match.group("raceline"),
        int(match.group("interval")),
        int(match.group("speed")) / 100.0,
    )
    parsed[key] = row

boundary_pairs: set[tuple[str, str]] = set()
intervals = (8, 10, 12, 15)
speeds = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
for key, row in parsed.items():
    ego, raceline, interval, speed = key
    for axis_values, value, build_neighbor in (
        (intervals, interval, lambda other: (ego, raceline, other, speed)),
        (speeds, speed, lambda other: (ego, raceline, interval, other)),
    ):
        index = axis_values.index(value)
        if index + 1 >= len(axis_values):
            continue
        other_row = parsed[build_neighbor(axis_values[index + 1])]
        if {row["outcome"], other_row["outcome"]} == {"ego_collision", "other"}:
            boundary_pairs.add(tuple(sorted((row["scenario_id"], other_row["scenario_id"]))))

boundary_scenarios = set().union(*(set(pair) for pair in boundary_pairs))
outcome_by_id = {row["scenario_id"]: row["outcome"] for row in outcomes}
collision_boundary = {scenario for scenario in boundary_scenarios if outcome_by_id[scenario] == "ego_collision"}
other_boundary = {scenario for scenario in boundary_scenarios if outcome_by_id[scenario] == "other"}
if (len(boundary_pairs), len(boundary_scenarios), len(collision_boundary), len(other_boundary)) != (1042, 1360, 446, 914):
    raise AssertionError("Boundary-lattice recomputation does not match the recorded hard-neighbor audit")

hard_neighbor_rows = [{
    "base_candidate_count": int(cache_summary["candidate_count"]),
    "base_collision_count": int(cache_summary["collision_count"]),
    "base_valid_collision_rate": float(hard_probe["global_candidate_collision_rate"]),
    "probe_neighbor_count": int(hard_probe["neighbor_count"]),
    "probe_neighbor_collisions": int(hard_probe["neighbor_collision_count"]),
    "probe_neighbor_collision_rate": float(hard_probe["neighbor_collision_rate"]),
    "probe_enrichment": float(hard_probe["collision_rate_enrichment"]),
    "outcome_flip_boundary_pairs": len(boundary_pairs),
    "boundary_unique_scenarios": len(boundary_scenarios),
    "boundary_collision_side_scenarios": len(collision_boundary),
    "boundary_other_side_scenarios": len(other_boundary),
    "pipeline_integrated": bool(hard_probe["config"]["pipeline_integration"]),
}]

# Audit the nominal Austin600 panel construction.  evaluate.sh includes both
# endpoints i=0 and i=NUM_STARTPOINTS-1; eval_multiagent then reduces indices
# modulo the raceline length, so the last endpoint aliases the first one.
raceline_path = ROOT / "f1tenth_racetracks" / "Austin" / "raceline1.csv"
max_waypoints = len(raceline_path.read_text().splitlines()[2:])
num_startpoints = 50
scenarios_per_start = 3 * 4


def effective_eval_starts(offset: int) -> list[int]:
    raw = [i * max_waypoints // (num_startpoints - 1) + offset for i in range(num_startpoints)]
    return [index % max_waypoints for index in raw]


base_start_set = set(effective_eval_starts(0))
evaluation_panel_rows: list[dict[str, Any]] = []
for offset in (0, 1):
    starts = effective_eval_starts(offset)
    unique_starts = set(starts)
    evaluation_panel_rows.append(
        {
            "ego_idx_offset": offset,
            "raceline_waypoints": max_waypoints,
            "nominal_startpoints": num_startpoints,
            "unique_effective_startpoints": len(unique_starts),
            "duplicated_startpoint_slots": num_startpoints - len(unique_starts),
            "nominal_scenarios": num_startpoints * scenarios_per_start,
            "unique_physical_scenarios": len(unique_starts) * scenarios_per_start,
            "duplicated_physical_scenario_slots": (num_startpoints - len(unique_starts)) * scenarios_per_start,
            "effective_start_overlap_with_offset0": len(unique_starts & base_start_set),
        }
    )

# Two paired examples show why a universal '+/-4.2 collisions' or '>8
# collisions' rule is invalid on a fixed, paired panel.  Significance depends
# on discordant scenario identities, not only on the net count.
bc_panel = {
    row["scenario_id"]: row
    for row in json.loads((EVAL / "end2race_Austin" / "multiagents" / "results_multi.json").read_text())["episodes"].values()
}
base_u20 = {
    row["scenario_id"]: row
    for row in json.loads((EVAL / "ppo_privilege_gru_0721_base_u0020_Austin" / "multiagents" / "results_multi.json").read_text())["episodes"].values()
}


def paired_example(label: str, left_rows: dict[str, dict[str, Any]], right_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left = collision_set(left_rows)
    right = collision_set(right_rows)
    resolved = len(left - right)
    created = len(right - left)
    return {
        "comparison": label,
        "left_collisions": len(left),
        "right_collisions": len(right),
        "net_reduction": len(left) - len(right),
        "resolved": resolved,
        "created": created,
        "discordant_pairs": resolved + created,
        "paired_exact_p_unadjusted": exact_paired_p(resolved, created),
    }


paired_stat_examples = [
    paired_example("BC -> privilege_gru base U20", bc_panel, base_u20),
    paired_example("long clip0.15 U30 -> long clip0.20 U30", panels[("clip 0.15", 30)], panels[("clip 0.20", 30)]),
]

roadmap_rows = [
    {
        "priority": 1,
        "axis": "Training horizon",
        "treatment": "clip 0.20, 45 updates",
        "comparator": "existing clip 0.20 through U30",
        "decision": "Run now as a fresh-start horizon diagnostic; assess U35/U40/U45 late-window mean and scenario churn.",
    },
    {
        "priority": 1,
        "axis": "Clip boundary",
        "treatment": "clip 0.25, 45 updates",
        "comparator": "clip 0.20, 45 updates under the same seed/source/cache",
        "decision": "Maximum clip probe; do not open 0.30/0.40 unless 0.25 wins without safety regression.",
    },
    {
        "priority": 2,
        "axis": "Collision curriculum",
        "treatment": "fixed schema-2 boundary-aware collision cache",
        "comparator": "current fixed base collision cache",
        "decision": "Run only after clip/horizon freeze; preserve 50/50 collision/ordinary transitions and fresh-start both arms.",
    },
    {
        "priority": 3,
        "axis": "GAE lambda",
        "treatment": "0.99 diagnostic",
        "comparator": "current 0.995",
        "decision": "Defer until advantage statistics are logged; this is a stability hypothesis, not a current evidence-backed improvement.",
    },
    {
        "priority": 4,
        "axis": "Gamma",
        "treatment": "keep 0.999",
        "comparator": "none",
        "decision": "Freeze: current code couples gamma to PPO discounting and potential-based risk reward.",
    },
]

collisions_by_clip = {
    label: [row["collisions"] for row in eval_rows if row["clip"] == label]
    for label in RUNS
}
u30 = next(row for row in paired_rows if row["update"] == 30)
clip020_telemetry = next(row for row in telemetry_rows if row["clip"] == "clip 0.20")
summary = {
    "decision": "Cap clip exploration at 0.25, extend the clean 0.20/0.25 comparison to 45 updates, freeze gamma, defer GAE, then test a fixed boundary-aware collision cache as a separate axis.",
    "as_of": "2026-07-22",
    "checkpoint_updates": list(UPDATES),
    "clip015_collision_path": collisions_by_clip["clip 0.15"],
    "clip020_collision_path": collisions_by_clip["clip 0.20"],
    "clip015_mean_collisions_all_checkpoints": statistics.fmean(collisions_by_clip["clip 0.15"]),
    "clip020_mean_collisions_all_checkpoints": statistics.fmean(collisions_by_clip["clip 0.20"]),
    "clip020_u30_collisions": u30["clip020_collisions"],
    "clip015_u30_collisions": u30["clip015_collisions"],
    "u30_resolved_by_clip020": u30["resolved_by_clip020"],
    "u30_created_by_clip020": u30["created_by_clip020"],
    "u30_paired_exact_p_unadjusted": u30["paired_exact_p_unadjusted"],
    "clip020_mean_clip_fraction_u2_u30": clip020_telemetry["mean_clip_fraction"],
    "clip020_updates_kl_max_gt_0_5_u2_u30": clip020_telemetry["updates_kl_max_gt_0_5"],
    "clip020_updates_kl_max_gt_1_u2_u30": clip020_telemetry["updates_kl_max_gt_1"],
    "late_actor_relative_step_min_u20_u30": min(row["actor_step_relative_l2"] for row in late_training_rows),
    "late_actor_relative_step_max_u20_u30": max(row["actor_step_relative_l2"] for row in late_training_rows),
    "late_kl_mean_min_u20_u30": min(row["approx_kl_mean"] for row in late_training_rows),
    "late_kl_mean_max_u20_u30": max(row["approx_kl_mean"] for row in late_training_rows),
    "gamma_is_reward_coupled": True,
    "current_gamma": 0.999,
    "current_gae_lambda": 0.995,
    "current_gae_td_error_half_life_seconds": next(row["td_error_half_life_seconds"] for row in discount_rows if row["gae_lambda"] == 0.995),
    "hard_neighbor_probe_collisions": hard_neighbor_rows[0]["probe_neighbor_collisions"],
    "hard_neighbor_probe_count": hard_neighbor_rows[0]["probe_neighbor_count"],
    "hard_neighbor_boundary_pairs": hard_neighbor_rows[0]["outcome_flip_boundary_pairs"],
    "hard_neighbor_pipeline_integrated": hard_neighbor_rows[0]["pipeline_integrated"],
    "austin_nominal_scenarios": evaluation_panel_rows[0]["nominal_scenarios"],
    "austin_unique_physical_scenarios": evaluation_panel_rows[0]["unique_physical_scenarios"],
    "austin_duplicated_physical_scenario_slots": evaluation_panel_rows[0]["duplicated_physical_scenario_slots"],
    "recommendation_order": [
        "Run clean 45-update clip 0.20 and 0.25 arms with the current fixed collision cache; cap clip at 0.25.",
        "Evaluate U35/U40/U45 and use the late-window mean plus scenario churn to assess convergence; do not select only the minimum collision checkpoint.",
        "Freeze gamma at 0.999 and defer GAE lambda until advantage-by-role telemetry exists.",
        "After choosing clip and horizon, implement and A/B a fixed boundary-aware hard-neighbor cache without changing reward or the 50/50 role mix.",
    ],
}

write_csv("checkpoint_eval.csv", eval_rows)
write_csv("paired_by_checkpoint.csv", paired_rows)
write_csv("training_telemetry.csv", telemetry_rows)
write_csv("late_training_u20_u30.csv", late_training_rows)
write_csv("discount_horizons.csv", discount_rows)
write_csv("hard_neighbor_evidence.csv", hard_neighbor_rows)
write_csv("evaluation_panel_audit.csv", evaluation_panel_rows)
write_csv("paired_stat_examples.csv", paired_stat_examples)
write_csv("experiment_roadmap.csv", roadmap_rows)
(OUT / "decision_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))

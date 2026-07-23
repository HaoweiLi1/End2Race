#!/usr/bin/env python3
"""Reproduce the high-impact checks in the Claude Groups 1-6 review.

This script is read-only with respect to training/evaluation artifacts.  It
writes bounded validation tables into its own analysis directory.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_results" / "ppo_groups_1_6_validation"
TRAINED = ROOT / "post-trained"
EVAL = ROOT / "eval_results"
OUT.mkdir(parents=True, exist_ok=True)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def config(run: str) -> dict[str, Any]:
    return json.loads((TRAINED / run / "run_config.json").read_text())


def metrics(run: str, phase: str) -> list[dict[str, Any]]:
    return [row for row in jsonl(TRAINED / run / "metrics.jsonl") if row.get("phase") == phase]


def episodes(run: str, phase: str, rollout_index: int) -> list[dict[str, Any]]:
    return [
        row
        for row in jsonl(TRAINED / run / "episodes.jsonl")
        if row.get("phase") == phase and int(row.get("rollout_index", -1)) == rollout_index
    ]


def eval_panel(directory: str) -> dict[str, dict[str, Any]]:
    payload = json.loads((EVAL / directory / "multiagents" / "results_multi.json").read_text())
    return {row["scenario_id"]: row for row in payload["episodes"].values()}


def run_eval(run: str, update: int) -> dict[str, dict[str, Any]]:
    return eval_panel(f"{run}_u{update:04d}_Austin")


def collision_set(panel: dict[str, dict[str, Any]]) -> set[str]:
    return {scenario for scenario, row in panel.items() if row["ego_collision_occurred"]}


def exact_paired_p(resolved: int, created: int) -> float:
    """Two-sided exact McNemar/binomial p-value on discordant scenario pairs."""
    n = resolved + created
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(resolved, created) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def pairwise(label: str, before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left = collision_set(before)
    right = collision_set(after)
    resolved = len(left - right)
    created = len(right - left)
    return {
        "comparison": label,
        "before_collisions": len(left),
        "after_collisions": len(right),
        "shared": len(left & right),
        "resolved": resolved,
        "created": created,
        "net_change": len(right) - len(left),
        "paired_exact_p_unadjusted": exact_paired_p(resolved, created),
    }


def actor_state(run: str, update: int) -> dict[str, torch.Tensor]:
    path = TRAINED / run / "checkpoints" / f"actor_u{update:04d}.pth"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload if isinstance(payload, dict) else payload.state_dict()


def actor_difference(left_run: str, right_run: str, update: int) -> dict[str, Any]:
    left = actor_state(left_run, update)
    right = actor_state(right_run, update)
    keys_match = set(left) == set(right)
    maximum = max(float((left[key] - right[key]).abs().max()) for key in left) if keys_match else None
    return {
        "update": update,
        "keys_match": keys_match,
        "tensor_equal": bool(keys_match and maximum == 0.0),
        "max_abs_parameter_difference": maximum,
    }


old_clip = "ppo_privilege_gru_0721_clip020"
long_clip = "ppo_privilege_gru_0722_long_clip020"
base = "ppo_privilege_gru_0721_base"
long015 = "ppo_privilege_gru_0722_long_clip015"
tkl002 = "ppo_privilege_gru_0722_clip015_tkl002"
tkl004 = "ppo_privilege_gru_0722_clip015_tkl004"

old_args = config(old_clip)["args"]
long_args = config(long_clip)["args"]
config_differences = {
    key: {"old_g3": old_args.get(key), "g5_long": long_args.get(key)}
    for key in sorted(set(old_args) | set(long_args))
    if old_args.get(key) != long_args.get(key)
}

old_warm = metrics(old_clip, "warmup")[0]
long_warm = metrics(long_clip, "warmup")[0]
old_eps = episodes(old_clip, "warmup", 1)
long_eps = episodes(long_clip, "warmup", 1)
first_value_difference = None
first_structural_difference = None
for index, (left, right) in enumerate(zip(old_eps, long_eps)):
    differing = [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
    if differing and first_value_difference is None:
        first_value_difference = {
            "row": index,
            "scenario_id": left["scenario_id"],
            "fields": differing,
            "old_episode_return": left["episode_return"],
            "long_episode_return": right["episode_return"],
        }
    if (
        left["scenario_id"] != right["scenario_id"]
        or left["episode_steps"] != right["episode_steps"]
        or left["episode_outcome"] != right["episode_outcome"]
    ) and first_structural_difference is None:
        first_structural_difference = {
            "row": index,
            "old_scenario_id": left["scenario_id"],
            "long_scenario_id": right["scenario_id"],
            "old_steps": left["episode_steps"],
            "long_steps": right["episode_steps"],
            "old_outcome": left["episode_outcome"],
            "long_outcome": right["episode_outcome"],
        }

checkpoint_differences = [actor_difference(old_clip, long_clip, update) for update in (1, 5, 10, 15, 20)]
baseline_extension = [actor_difference(base, long015, update) for update in (1, 5, 10, 15, 20)]

target_rows = []
for run, label in ((tkl002, "target-KL 0.02"), (tkl004, "target-KL 0.04")):
    for row in metrics(run, "formal"):
        target_rows.append(
            {
                "run": run,
                "label": label,
                "update": int(row["update"]),
                "threshold": row["actor_kl_stop_threshold"],
                "steps_completed": int(row["actor_optimizer_steps_completed"]),
                "steps_planned": int(row["actor_optimizer_steps_planned"]),
                "early_stop": bool(row["actor_early_stop_triggered"]),
                "stop_epoch": row["actor_early_stop_epoch"],
                "stop_minibatch": row["actor_early_stop_minibatch"],
                "trigger_kl": row["actor_early_stop_approx_kl"],
                "approx_kl_mean": row["approx_kl_mean"],
                "approx_kl_max": row["approx_kl_max"],
                "rollout_policy_update": int(row["rollout_policy_update"]),
                "checkpoint_update": int(row["checkpoint_update"]),
                "rollout_collision_count": int(row["ego_collision_count"]),
                "rollout_episode_count": int(row["episode_count"]),
            }
        )

tkl_eval_rows = []
for update in (1, 5, 10, 15, 20):
    panel = run_eval(tkl004, update)
    collisions = [row for row in panel.values() if row["ego_collision_occurred"]]
    tkl_eval_rows.append(
        {
            "update": update,
            "collisions": len(collisions),
            "opponent_collisions": sum(bool(row["opp_collision_occurred"]) for row in collisions),
            "ego_or_wall_collisions": sum(not bool(row["opp_collision_occurred"]) for row in collisions),
            "mean_min_surface_distance_m": sum(row["global_min_surface_dist"] for row in panel.values()) / len(panel),
            "mean_speed_mps": sum(row["avg_speed"] for row in panel.values()) / len(panel),
        }
    )

bc = eval_panel("end2race_Austin")
base_u20 = run_eval(base, 20)
old_clip_u20 = run_eval(old_clip, 20)
long_clip_u30 = run_eval(long_clip, 30)
tkl004_u20 = run_eval(tkl004, 20)
paired_rows = [
    pairwise("BC -> base U20", bc, base_u20),
    pairwise("BC -> old G3 clip 0.20 U20", bc, old_clip_u20),
    pairwise("BC -> G5 long clip 0.20 U30", bc, long_clip_u30),
    pairwise("base U20 -> G5 long clip 0.20 U30", base_u20, long_clip_u30),
    pairwise("BC -> target-KL 0.04 U20", bc, tkl004_u20),
]

final_sets = []
for run_dir in sorted(TRAINED.glob("ppo_*")):
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        continue
    run_config = json.loads(config_path.read_text())
    update = int(run_config["args"]["num_updates"])
    panel_path = EVAL / f"{run_dir.name}_u{update:04d}_Austin" / "multiagents" / "results_multi.json"
    if panel_path.exists():
        final_sets.append((run_dir.name, collision_set(run_eval(run_dir.name, update))))
final_union = set().union(*(items for _run, items in final_sets))
final_intersection = set.intersection(*(items for _run, items in final_sets))

persistent_rows = []
for scenario in sorted(final_intersection):
    bc_row = bc[scenario]
    long_row = long_clip_u30[scenario]
    persistent_rows.append(
        {
            "scenario_id": scenario,
            "ppo_final_runs": len(final_sets),
            "bc_collision_time_s": bc_row["ego_collision_time_s"],
            "long_clip_u30_collision_time_s": long_row["ego_collision_time_s"],
            "bc_initial_collision": bool(bc_row["initial_ego_collision"]),
            "long_initial_collision": bool(long_row["initial_ego_collision"]),
            "same_time_exactly": bc_row["ego_collision_time_s"] == long_row["ego_collision_time_s"],
        }
    )

claim_rows = [
    {
        "claim": "G1 privilege_gru is the strongest critic arm",
        "assessment": "supported",
        "severity": "none",
        "evidence": "U20 collisions 14 vs privilege_mlp 25 and independent_gru 34",
        "required_revision": "Remove the causal phrase that independent_gru gives harmful advantage directions; the eval path does not identify that mechanism.",
    },
    {
        "claim": "Batch 12800 is best and larger batches also reduce optimizer-step count",
        "assessment": "supported with nuance",
        "severity": "low",
        "evidence": "16/8/4 actor steps per update for batch 12800/25600/51200 under fixed epochs",
        "required_revision": "Call it the effect of changing batch under fixed epochs; it does not isolate batch aggregation from optimization budget.",
    },
    {
        "claim": "G3 old clip runs are confounded against the 12-worker baseline",
        "assessment": "supported",
        "severity": "none",
        "evidence": "Warm-up episode values and episode lengths diverge before actor updates; 8-worker arms match each other and 12-worker arms match each other.",
        "required_revision": "Keep the planner shallow-copy explanation explicitly hypothetical; root cause is not proven.",
    },
    {
        "claim": "G5 long clip 0.20 continuously improves through U30",
        "assessment": "incorrect wording",
        "severity": "medium",
        "evidence": "Collision path is 21/18/16/13/13/17/11, including a U25 regression.",
        "required_revision": "Use 'best final result with a non-monotonic path', not 'continuous improvement'.",
    },
    {
        "claim": "target-KL 0.02 early-stops 10 of 20 updates",
        "assessment": "incorrect number",
        "severity": "low",
        "evidence": "metrics.jsonl records 11 early-stop updates and 206/320 completed actor steps.",
        "required_revision": "Replace 10/20 with 11/20.",
    },
    {
        "claim": "target-KL injects random step length and is purely harmful",
        "assessment": "overstated",
        "severity": "medium",
        "evidence": "The step count is deterministic but state/minibatch-dependent; target-KL 0.04 is a clean single-axis failure for this seed/config.",
        "required_revision": "Say 'irregular adaptive optimization budget and path dependence'; restrict harm to the tested targets/configuration.",
    },
    {
        "claim": "Training rollout metrics and same-number eval describe the same policy",
        "assessment": "incorrect alignment",
        "severity": "high",
        "evidence": "Each formal row records rollout_policy_update = update - 1 and checkpoint_update = update.",
        "required_revision": "Compare rollout row U+1 with checkpoint U when available, and keep KL/gradient metrics associated with update U separate from rollout outcomes.",
    },
    {
        "claim": "A binomial ±4.2 collision noise floor means differences below 8 are not significant",
        "assessment": "methodologically incorrect",
        "severity": "high",
        "evidence": "Austin600 is a fixed paired panel; exact paired inference depends on resolved vs created scenarios, not count difference alone.",
        "required_revision": "Use paired scenario analysis/McNemar and adjust for checkpoint selection; do not use a universal eight-collision threshold.",
    },
    {
        "claim": "The two persistent scenarios are identical or near-impossible starts",
        "assessment": "unsupported inference",
        "severity": "medium",
        "evidence": "They collide in all 13 PPO final checkpoints, but initial_ego_collision is false and BC/long collision times are close, not exactly equal.",
        "required_revision": "Call them persistent hard cases pending counterfactual trajectory or reset-geometry analysis.",
    },
    {
        "claim": "LR5 trades overtaking for conservative following to maximize return",
        "assessment": "plausible but not proven",
        "severity": "medium",
        "evidence": "The U20 eval has 599 rows/one error and does show 318 overtakes, 255 follows and mean relative position 4.78.",
        "required_revision": "Label the behavior explanation as a hypothesis and keep the invalid-panel caveat attached.",
    },
]


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


write_csv("claim_review.csv", claim_rows)
write_csv("target_kl_steps.csv", target_rows)
write_csv("target_kl_eval.csv", tkl_eval_rows)
write_csv("paired_scenario_tests.csv", paired_rows)
write_csv("persistent_scenarios.csv", persistent_rows)
write_csv("old_vs_long_actor_diff.csv", checkpoint_differences)
write_csv("base_vs_long015_actor_diff.csv", baseline_extension)

summary = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "repo_head": "4e0043ed26e57950546c685d7bddba83e756c1a5",
    "overall_assessment": "needs revision before being used as a decision record; directionally strong but contains two high-severity methodology errors and several causal overclaims",
    "g3_vs_g5": {
        "config_differences": config_differences,
        "old_warmup": {
            "epochs": old_warm["epochs"],
            "best_epoch": old_warm["best_epoch"],
            "best_validation_loss": old_warm["best_validation_loss"],
            "rollout1_episode_count": len(old_eps),
        },
        "long_warmup": {
            "epochs": long_warm["epochs"],
            "best_epoch": long_warm["best_epoch"],
            "best_validation_loss": long_warm["best_validation_loss"],
            "rollout1_episode_count": len(long_eps),
        },
        "first_value_difference": first_value_difference,
        "first_structural_difference": first_structural_difference,
        "actor_checkpoint_differences": checkpoint_differences,
        "baseline_extension_actor_identity": baseline_extension,
        "verified_cause_boundary": "The process topology/env_workers change is the only recorded effective training difference before U20 beyond total horizon/output path; the specific internal mechanism remains unresolved.",
    },
    "target_kl": {
        "tkl002_early_stop_updates": sum(row["early_stop"] for row in target_rows if row["run"] == tkl002),
        "tkl002_steps_completed": sum(row["steps_completed"] for row in target_rows if row["run"] == tkl002),
        "tkl004_early_stop_updates": sum(row["early_stop"] for row in target_rows if row["run"] == tkl004),
        "tkl004_steps_completed": sum(row["steps_completed"] for row in target_rows if row["run"] == tkl004),
        "tkl004_eval": tkl_eval_rows,
        "mechanism": "KL is checked before the current minibatch update. Overshoot was created by earlier completed minibatches; the gate stops later minibatches but cannot undo those parameter steps. Irregular completed-step counts alter the actor path and future minibatch RNG/rollout distribution while critic updates continue.",
        "removal_recommendation": "Disable in the selected recipe by leaving target_kl=None and remove the Group 6 command from the future run matrix. Retain the optional implementation and telemetry because this experiment does not prove the feature is universally harmful.",
    },
    "paired_tests": paired_rows,
    "final_checkpoint_scenarios": {
        "ppo_run_count": len(final_sets),
        "union": len(final_union),
        "intersection": len(final_intersection),
        "persistent": persistent_rows,
    },
    "claims": claim_rows,
}
(OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({
    "output_dir": str(OUT),
    "claim_count": len(claim_rows),
    "high_severity_issues": sum(row["severity"] == "high" for row in claim_rows),
    "g3_g5_first_structural_difference": first_structural_difference,
    "tkl002_early_stops": summary["target_kl"]["tkl002_early_stop_updates"],
    "tkl004_early_stops": summary["target_kl"]["tkl004_early_stop_updates"],
}, indent=2, ensure_ascii=False))

#!/usr/bin/env python3
"""Build a portable technical report for the clip-extension decision."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_results" / "clip_extension_decision"


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str, integer: bool = False):
    return int(float(value)) if integer else float(value)


summary = json.loads((OUT / "decision_summary.json").read_text())
eval_rows = [
    {
        "clip": row["clip"],
        "update": number(row["update"], True),
        "collisions": number(row["collisions"], True),
        "success_rate": number(row["success_rate"]),
        "wall_like_collisions": number(row["wall_like_collisions"], True),
        "overtakes": number(row["overtakes"], True),
        "follows": number(row["follows"], True),
        "mean_speed_mps": number(row["mean_speed_mps"]),
        "mean_min_surface_m": number(row["mean_min_surface_m"]),
        "scenario_count": number(row["scenario_count"], True),
    }
    for row in read_csv("checkpoint_eval.csv")
]
paired = [
    {
        "update": number(row["update"], True),
        "clip015_collisions": number(row["clip015_collisions"], True),
        "clip020_collisions": number(row["clip020_collisions"], True),
        "shared": number(row["shared"], True),
        "resolved_by_clip020": number(row["resolved_by_clip020"], True),
        "created_by_clip020": number(row["created_by_clip020"], True),
        "net_collision_change": number(row["net_collision_change"], True),
        "paired_exact_p_unadjusted": number(row["paired_exact_p_unadjusted"]),
    }
    for row in read_csv("paired_by_checkpoint.csv")
]
telemetry = [
    {
        "clip": row["clip"],
        "updates": number(row["updates"], True),
        "mean_clip_fraction": number(row["mean_clip_fraction"]),
        "median_clip_fraction": number(row["median_clip_fraction"]),
        "mean_approx_kl": number(row["mean_approx_kl"]),
        "updates_kl_max_gt_0_5": number(row["updates_kl_max_gt_0_5"], True),
        "updates_kl_max_gt_1": number(row["updates_kl_max_gt_1"], True),
        "largest_kl_max": number(row["largest_kl_max"]),
    }
    for row in read_csv("training_telemetry.csv")
]
late_training = [
    {
        "update": number(row["update"], True),
        "rollout_policy_update": number(row["rollout_policy_update"], True),
        "approx_kl_mean": number(row["approx_kl_mean"]),
        "approx_kl_max": number(row["approx_kl_max"]),
        "clip_fraction_mean": number(row["clip_fraction_mean"]),
        "explained_variance_post": number(row["explained_variance_post"]),
        "rollout_collision_count": number(row["rollout_collision_count"], True),
        "rollout_episode_count": number(row["rollout_episode_count"], True),
        "actor_step_l2": number(row["actor_step_l2"]),
        "actor_step_relative_l2": number(row["actor_step_relative_l2"]),
    }
    for row in read_csv("late_training_u20_u30.csv")
]
discount_horizons = [
    {
        "gamma": number(row["gamma"]),
        "gae_lambda": number(row["gae_lambda"]),
        "gamma_times_lambda": number(row["gamma_times_lambda"]),
        "td_error_half_life_seconds": number(row["td_error_half_life_seconds"]),
        "geometric_horizon_seconds": number(row["geometric_horizon_seconds"]),
        "weight_after_1s": number(row["weight_after_1s"]),
        "weight_after_2s": number(row["weight_after_2s"]),
        "weight_after_4s": number(row["weight_after_4s"]),
        "weight_after_8s": number(row["weight_after_8s"]),
    }
    for row in read_csv("discount_horizons.csv")
]
hard_neighbor = [
    {
        "base_candidate_count": number(row["base_candidate_count"], True),
        "base_collision_count": number(row["base_collision_count"], True),
        "base_valid_collision_rate": number(row["base_valid_collision_rate"]),
        "probe_neighbor_count": number(row["probe_neighbor_count"], True),
        "probe_neighbor_collisions": number(row["probe_neighbor_collisions"], True),
        "probe_neighbor_collision_rate": number(row["probe_neighbor_collision_rate"]),
        "probe_enrichment": number(row["probe_enrichment"]),
        "outcome_flip_boundary_pairs": number(row["outcome_flip_boundary_pairs"], True),
        "boundary_unique_scenarios": number(row["boundary_unique_scenarios"], True),
        "boundary_collision_side_scenarios": number(row["boundary_collision_side_scenarios"], True),
        "boundary_other_side_scenarios": number(row["boundary_other_side_scenarios"], True),
        "pipeline_integrated": row["pipeline_integrated"].lower() == "true",
    }
    for row in read_csv("hard_neighbor_evidence.csv")
]
roadmap = [
    {
        "priority": number(row["priority"], True),
        "axis": row["axis"],
        "treatment": row["treatment"],
        "comparator": row["comparator"],
        "decision": row["decision"],
    }
    for row in read_csv("experiment_roadmap.csv")
]
evaluation_panel_audit = [
    {
        "ego_idx_offset": number(row["ego_idx_offset"], True),
        "raceline_waypoints": number(row["raceline_waypoints"], True),
        "nominal_startpoints": number(row["nominal_startpoints"], True),
        "unique_effective_startpoints": number(row["unique_effective_startpoints"], True),
        "duplicated_startpoint_slots": number(row["duplicated_startpoint_slots"], True),
        "nominal_scenarios": number(row["nominal_scenarios"], True),
        "unique_physical_scenarios": number(row["unique_physical_scenarios"], True),
        "duplicated_physical_scenario_slots": number(row["duplicated_physical_scenario_slots"], True),
        "effective_start_overlap_with_offset0": number(row["effective_start_overlap_with_offset0"], True),
    }
    for row in read_csv("evaluation_panel_audit.csv")
]
paired_stat_examples = [
    {
        "comparison": row["comparison"],
        "left_collisions": number(row["left_collisions"], True),
        "right_collisions": number(row["right_collisions"], True),
        "net_reduction": number(row["net_reduction"], True),
        "resolved": number(row["resolved"], True),
        "created": number(row["created"], True),
        "discordant_pairs": number(row["discordant_pairs"], True),
        "paired_exact_p_unadjusted": number(row["paired_exact_p_unadjusted"]),
    }
    for row in read_csv("paired_stat_examples.csv")
]

headline = [{
    "best_u30_collisions": summary["clip020_u30_collisions"],
    "u30_paired_p": summary["u30_paired_exact_p_unadjusted"],
    "clip020_mean_clip_fraction": summary["clip020_mean_clip_fraction_u2_u30"],
    "hard_neighbor_boundary_pairs": summary["hard_neighbor_boundary_pairs"],
}]

sources = [
    {
        "id": "checkpoint_eval",
        "label": "Group 5 Austin600 checkpoint evaluation",
        "path": "analysis_results/clip_extension_decision/checkpoint_eval.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/clip_extension_decision/checkpoint_eval.csv', header = true);",
            "description": "Loads the seven matched Austin600 checkpoints for the clean 12-worker clip 0.15/0.20 comparison.",
            "tables_used": ["eval_results/ppo_privilege_gru_0722_long_clip015_u*/multiagents/results_multi.json", "eval_results/ppo_privilege_gru_0722_long_clip020_u*/multiagents/results_multi.json"],
            "filters": ["Austin", "600 fixed scenario ids", "U1/U5/U10/U15/U20/U25/U30"],
            "metric_definitions": ["collision count is the number of scenarios with ego_collision_occurred=true", "wall-like means ego collision true and opponent collision false"],
        },
    },
    {
        "id": "paired_eval",
        "label": "Paired Austin600 clip comparison",
        "path": "analysis_results/clip_extension_decision/paired_by_checkpoint.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/clip_extension_decision/paired_by_checkpoint.csv', header = true);",
            "description": "Loads shared, resolved and created collision identities plus exact paired p-values.",
            "tables_used": ["analysis_results/clip_extension_decision/paired_by_checkpoint.csv"],
            "metric_definitions": ["paired exact p is a two-sided exact binomial test on resolved versus created scenario pairs and is unadjusted for checkpoint selection"],
        },
    },
    {
        "id": "training_telemetry",
        "label": "Group 5 PPO clipping and KL telemetry",
        "path": "analysis_results/clip_extension_decision/training_telemetry.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/clip_extension_decision/training_telemetry.csv', header = true);",
            "description": "Loads bounded U2-U30 clipping and approximate-KL summaries.",
            "tables_used": ["post-trained/ppo_privilege_gru_0722_long_clip015/metrics.jsonl", "post-trained/ppo_privilege_gru_0722_long_clip020/metrics.jsonl"],
            "filters": ["phase=formal", "update>=2"],
            "metric_definitions": ["mean clip fraction averages the recorded per-update clip_fraction_mean", "KL tail counts updates whose recorded approx_kl_max exceeds the named threshold"],
        },
    },
    {
        "id": "late_training",
        "label": "Clip 0.20 late-training optimization telemetry",
        "path": "analysis_results/clip_extension_decision/late_training_u20_u30.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/clip_extension_decision/late_training_u20_u30.csv', header = true);",
            "description": "Loads U20-U30 KL, critic fit, rollout labels, and sequential actor checkpoint norms.",
            "tables_used": ["post-trained/ppo_privilege_gru_0722_long_clip020/metrics.jsonl", "post-trained/ppo_privilege_gru_0722_long_clip020/checkpoints/actor_u*.pth"],
            "filters": ["phase=formal", "update between 20 and 30"],
            "metric_definitions": ["actor step relative L2 is the L2 norm of checkpoint U minus U-1 divided by the L2 norm of checkpoint U-1", "rollout collision count in formal row U belongs to rollout_policy_update U-1"],
        },
    },
    {
        "id": "discount_horizons",
        "label": "Gamma and GAE temporal-weight calculations",
        "path": "analysis_results/clip_extension_decision/discount_horizons.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/clip_extension_decision/discount_horizons.csv', header = true);",
            "description": "Loads deterministic gamma-times-lambda temporal weights at 100 Hz.",
            "tables_used": ["train_ppo.py", "ppo/reward.py", "analysis_results/clip_extension_decision/discount_horizons.csv"],
            "filters": ["gamma fixed at 0.999", "100 simulator steps per second"],
            "metric_definitions": ["half life is log(0.5)/log(gamma*lambda)/100 seconds", "geometric horizon is 1/(1-gamma*lambda)/100 seconds"],
        },
    },
    {
        "id": "hard_neighbor",
        "label": "Hard-neighbor probe and cache-boundary evidence",
        "path": "analysis_results/clip_extension_decision/hard_neighbor_evidence.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/clip_extension_decision/hard_neighbor_evidence.csv', header = true);",
            "description": "Loads the standalone hard-neighbor probe and recomputed adjacent outcome-flip boundary counts.",
            "tables_used": ["analysis_results/hard_neighbor_probe/summary.json", "post-trained/collision-cache/default/candidate_outcomes.jsonl", "post-trained/collision-cache/default/classification_summary.json"],
            "filters": ["training candidate grid only", "invalid outcomes excluded from boundary edges", "one configured interval or speed adjacency at a time"],
            "metric_definitions": ["probe enrichment is descriptive because three dense source families were deliberately selected", "boundary pair has one ego_collision and one other endpoint"],
        },
    },
    {
        "id": "evaluation_panel_audit",
        "label": "Austin evaluation panel construction audit",
        "path": "analysis_results/clip_extension_decision/evaluation_panel_audit.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/clip_extension_decision/evaluation_panel_audit.csv', header = true);",
            "description": "Audits modulo-reduced Austin ego starts for the current 50-startpoint panel and an offset-1 panel.",
            "tables_used": ["evaluate.sh", "eval_multiagent.py", "f1tenth_racetracks/Austin/raceline1.csv"],
            "metric_definitions": ["physical scenario identity is effective ego index, opponent raceline, and opponent speed scale"],
        },
    },
    {
        "id": "paired_stat_examples",
        "label": "Paired fixed-panel statistical counterexamples",
        "path": "analysis_results/clip_extension_decision/paired_stat_examples.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/clip_extension_decision/paired_stat_examples.csv', header = true);",
            "description": "Shows why net collision difference alone does not determine paired-panel evidence strength.",
            "tables_used": ["eval_results/end2race_Austin/multiagents/results_multi.json", "eval_results/ppo_privilege_gru_0721_base_u0020_Austin/multiagents/results_multi.json", "analysis_results/clip_extension_decision/paired_by_checkpoint.csv"],
            "metric_definitions": ["paired exact p is a two-sided exact binomial test on resolved versus created fixed scenarios"],
        },
    },
    {
        "id": "experiment_roadmap",
        "label": "Recommended controlled experiment order",
        "path": "analysis_results/clip_extension_decision/experiment_roadmap.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/clip_extension_decision/experiment_roadmap.csv', header = true);",
            "description": "Loads the evidence-ranked next-experiment matrix.",
            "tables_used": ["analysis_results/clip_extension_decision/experiment_roadmap.csv"],
        },
    },
]

cards = [
    {
        "id": "best_card",
        "description": "Current best recorded Group 5 checkpoint on Austin600.",
        "dataset": "headline",
        "sourceId": "checkpoint_eval",
        "metrics": [{"label": "clip 0.20 U30 collisions", "field": "best_u30_collisions", "format": "number"}],
    },
    {
        "id": "paired_card",
        "description": "Exact paired Austin600 result at the selected U30 checkpoint; unadjusted for checkpoint selection.",
        "dataset": "headline",
        "sourceId": "paired_eval",
        "metrics": [{"label": "U30 paired p", "field": "u30_paired_p", "format": "number"}],
    },
    {
        "id": "clipfrac_card",
        "description": "Average fraction outside the PPO 0.20 ratio interval over formal U2-U30.",
        "dataset": "headline",
        "sourceId": "training_telemetry",
        "metrics": [{"label": "clip 0.20 mean clipped", "field": "clip020_mean_clip_fraction", "format": "percent"}],
    },
    {
        "id": "boundary_card",
        "description": "Adjacent collision/other outcome-flip edges in the independent training candidate lattice.",
        "dataset": "headline",
        "sourceId": "hard_neighbor",
        "metrics": [{"label": "hard-neighbor boundary pairs", "field": "hard_neighbor_boundary_pairs", "format": "number"}],
    },
]

charts = [{
    "id": "collision_chart",
    "title": "Group 5 checkpoint collisions by clip range",
    "subtitle": "Austin600; matched 12-worker runs at seven planned checkpoints",
    "type": "bar",
    "dataset": "checkpoint_eval",
    "sourceId": "checkpoint_eval",
    "valueFormat": "number",
    "encodings": {
        "x": {"field": "update", "type": "nominal", "label": "Formal update"},
        "y": {"field": "collisions", "type": "quantitative", "label": "Collisions / 600"},
        "color": {"field": "clip", "type": "nominal", "label": "Clip range"},
        "tooltip": [
            {"field": "wall_like_collisions", "type": "quantitative", "label": "Wall-like collisions"},
            {"field": "overtakes", "type": "quantitative", "label": "Overtakes"},
            {"field": "mean_speed_mps", "type": "quantitative", "label": "Mean speed m/s"},
            {"field": "mean_min_surface_m", "type": "quantitative", "label": "Mean min surface m"},
        ],
    },
}]

tables = [
    {
        "id": "paired_table",
        "title": "Paired scenario comparison by checkpoint",
        "subtitle": "Positive net change means clip 0.20 created more collisions; p-values are unadjusted",
        "dataset": "paired",
        "sourceId": "paired_eval",
        "defaultSort": {"field": "update", "direction": "asc"},
        "columns": [
            {"field": "update", "label": "U", "format": "number"},
            {"field": "clip015_collisions", "label": "clip .15", "format": "number"},
            {"field": "clip020_collisions", "label": "clip .20", "format": "number"},
            {"field": "shared", "label": "shared", "format": "number"},
            {"field": "resolved_by_clip020", "label": "resolved", "format": "number"},
            {"field": "created_by_clip020", "label": "created", "format": "number"},
            {"field": "net_collision_change", "label": "net", "format": "number", "movement": True},
            {"field": "paired_exact_p_unadjusted", "label": "paired p", "format": "number"},
        ],
    },
    {
        "id": "telemetry_table",
        "title": "PPO clipping and KL telemetry",
        "subtitle": "Formal U2-U30; U1 warm-up transient excluded",
        "dataset": "telemetry",
        "sourceId": "training_telemetry",
        "defaultSort": {"field": "clip", "direction": "asc"},
        "columns": [
            {"field": "clip", "label": "clip"},
            {"field": "updates", "label": "updates", "format": "number"},
            {"field": "mean_clip_fraction", "label": "mean clipped", "format": "percent"},
            {"field": "median_clip_fraction", "label": "median clipped", "format": "percent"},
            {"field": "mean_approx_kl", "label": "mean approx KL", "format": "number"},
            {"field": "updates_kl_max_gt_0_5", "label": "KL max > .5", "format": "number"},
            {"field": "updates_kl_max_gt_1", "label": "KL max > 1", "format": "number"},
            {"field": "largest_kl_max", "label": "largest KL max", "format": "number"},
        ],
    },
    {
        "id": "late_training_table",
        "title": "Clip 0.20 late-training movement",
        "subtitle": "Formal U20-U30; rollout collisions are labeled by their actual policy update",
        "dataset": "late_training",
        "sourceId": "late_training",
        "defaultSort": {"field": "update", "direction": "asc"},
        "columns": [
            {"field": "update", "label": "update", "format": "number"},
            {"field": "rollout_policy_update", "label": "rollout policy", "format": "number"},
            {"field": "approx_kl_mean", "label": "KL mean", "format": "number"},
            {"field": "approx_kl_max", "label": "KL max", "format": "number"},
            {"field": "clip_fraction_mean", "label": "clipped", "format": "percent"},
            {"field": "explained_variance_post", "label": "EV post", "format": "number"},
            {"field": "rollout_collision_count", "label": "rollout collisions", "format": "number"},
            {"field": "rollout_episode_count", "label": "episodes", "format": "number"},
            {"field": "actor_step_relative_l2", "label": "actor relative step", "format": "number"},
        ],
    },
    {
        "id": "discount_table",
        "title": "GAE temporal reach at gamma 0.999",
        "subtitle": "100 Hz simulator; weights describe TD-error contribution, not episode-return discount alone",
        "dataset": "discount_horizons",
        "sourceId": "discount_horizons",
        "defaultSort": {"field": "gae_lambda", "direction": "asc"},
        "columns": [
            {"field": "gae_lambda", "label": "lambda", "format": "number"},
            {"field": "gamma_times_lambda", "label": "gamma*lambda", "format": "number"},
            {"field": "td_error_half_life_seconds", "label": "half-life s", "format": "number"},
            {"field": "geometric_horizon_seconds", "label": "geometric horizon s", "format": "number"},
            {"field": "weight_after_1s", "label": "weight @1s", "format": "number"},
            {"field": "weight_after_2s", "label": "weight @2s", "format": "number"},
            {"field": "weight_after_4s", "label": "weight @4s", "format": "number"},
            {"field": "weight_after_8s", "label": "weight @8s", "format": "number"},
        ],
    },
    {
        "id": "hard_neighbor_table",
        "title": "Hard-neighbor evidence boundary",
        "subtitle": "Standalone dense-family probe plus full independent training-cache lattice",
        "dataset": "hard_neighbor",
        "sourceId": "hard_neighbor",
        "defaultSort": {"field": "outcome_flip_boundary_pairs", "direction": "desc"},
        "columns": [
            {"field": "base_candidate_count", "label": "base candidates", "format": "number"},
            {"field": "base_collision_count", "label": "base collisions", "format": "number"},
            {"field": "base_valid_collision_rate", "label": "base rate", "format": "percent"},
            {"field": "probe_neighbor_collisions", "label": "probe collisions", "format": "number"},
            {"field": "probe_neighbor_count", "label": "probe total", "format": "number"},
            {"field": "probe_neighbor_collision_rate", "label": "probe rate", "format": "percent"},
            {"field": "outcome_flip_boundary_pairs", "label": "boundary pairs", "format": "number"},
            {"field": "boundary_unique_scenarios", "label": "boundary scenarios", "format": "number"},
            {"field": "pipeline_integrated", "label": "integrated"},
        ],
    },
    {
        "id": "panel_audit_table",
        "title": "Austin panel construction audit",
        "subtitle": "Current endpoint formula followed by modulo reduction; offset 1 is shifted but retains the internal endpoint alias",
        "dataset": "evaluation_panel_audit",
        "sourceId": "evaluation_panel_audit",
        "defaultSort": {"field": "ego_idx_offset", "direction": "asc"},
        "columns": [
            {"field": "ego_idx_offset", "label": "offset", "format": "number"},
            {"field": "nominal_startpoints", "label": "nominal starts", "format": "number"},
            {"field": "unique_effective_startpoints", "label": "unique starts", "format": "number"},
            {"field": "nominal_scenarios", "label": "nominal scenarios", "format": "number"},
            {"field": "unique_physical_scenarios", "label": "unique scenarios", "format": "number"},
            {"field": "duplicated_physical_scenario_slots", "label": "duplicate slots", "format": "number"},
            {"field": "effective_start_overlap_with_offset0", "label": "start overlap vs base", "format": "number"},
        ],
    },
    {
        "id": "paired_stat_examples_table",
        "title": "Net counts do not define paired-panel significance",
        "subtitle": "Unadjusted exact paired tests; shown as methodological counterexamples, not final model-selection claims",
        "dataset": "paired_stat_examples",
        "sourceId": "paired_stat_examples",
        "defaultSort": {"field": "comparison", "direction": "asc"},
        "columns": [
            {"field": "comparison", "label": "comparison"},
            {"field": "left_collisions", "label": "left", "format": "number"},
            {"field": "right_collisions", "label": "right", "format": "number"},
            {"field": "net_reduction", "label": "net reduction", "format": "number"},
            {"field": "resolved", "label": "resolved", "format": "number"},
            {"field": "created", "label": "created", "format": "number"},
            {"field": "paired_exact_p_unadjusted", "label": "paired p", "format": "number"},
        ],
    },
    {
        "id": "roadmap_table",
        "title": "Evidence-ranked experiment order",
        "subtitle": "Each stage freezes all earlier choices before opening the next axis",
        "dataset": "roadmap",
        "sourceId": "experiment_roadmap",
        "defaultSort": {"field": "priority", "direction": "asc"},
        "columns": [
            {"field": "priority", "label": "priority", "format": "number"},
            {"field": "axis", "label": "axis"},
            {"field": "treatment", "label": "treatment"},
            {"field": "comparator", "label": "comparator"},
            {"field": "decision", "label": "decision"},
        ],
    },
]

blocks = [
    {"id": "title", "type": "markdown", "layout": "full", "body": "# End2Race PPO 下一轮实验路线：Clip、45U、GAE 与 Hard Neighbors"},
    {
        "id": "summary",
        "type": "markdown",
        "body": """## 技术摘要

**把 clip 搜索上限设为0.25是合理的，当前没有依据测试0.30。** 下一轮最有信息量的无代码语义变更实验，是 fresh-start 运行 `clip=0.20/0.25 × 45 updates`：0.20负责回答U30以后是否形成稳定晚期改善，0.25负责关闭clip搜索边界。`gamma` 应保持0.999，因为当前实现同时把它用于PPO折扣和risk potential reward；`gae_lambda` 可研究但优先级较低。Hard neighbors值得加入，但必须作为独立的固定 boundary-aware collision-cache A/B，不能把当前11条probe场景直接混入最终配方，也不能与clip、训练长度或reward同时变化。""",
    },
    {"id": "metrics", "type": "metric-strip", "cardIds": ["best_card", "paired_card", "clipfrac_card", "boundary_card"]},
    {
        "id": "claude_review",
        "type": "markdown",
        "sourceId": "paired_stat_examples",
        "body": """## 对所给 Claude 评审的复核：大方向正确，但不能原样执行

暂时排除seed问题后，bit-repro、45U fresh-start、复用U1-U30 eval、冻结gamma、GAE诊断先行、hard-neighbor独立A/B以及关闭既有低价值参数轴，均与代码和现有证据一致。需要修正四点：固定Austin面板不能使用通用 `±4.2` 或“差异必须超过8”的独立二项噪声规则；当前holdout生成式内部有端点重复；若0.25仍待决，不能先打开hard-neighbor轴；advantage telemetry必须证明不消耗训练RNG、不调用会变形buffer的 `get()`，collision-lag还需要跨rollout episode对齐定义。""",
    },
    {
        "id": "paired_stat_review",
        "type": "markdown",
        "sourceId": "paired_stat_examples",
        "body": """## 统计功效判断的核心错误

Austin比较是同一组固定场景上的**配对结果**，证据强弱由 `resolved` 与 `created` 的不对称性决定，而不是两次独立抽样的碰撞总数标准差。BC到base U20只净减少8次，但为10 resolved / 2 created，未校正exact p=0.0386；long clip0.15到0.20 U30净减少9次，却为15 / 6，未校正p=0.0784。故“0.25单run必须改善超过约8次才有辨识力”不成立。0.25优先级可低，但理由应是现有clip梯度不稳定、U30选择偏差和边界收益不确定。""",
    },
    {"id": "paired_stat_examples_block", "type": "table", "tableId": "paired_stat_examples_table"},
    {
        "id": "panel_audit_review",
        "type": "markdown",
        "sourceId": "evaluation_panel_audit",
        "body": """## Holdout建议正确，但当前脚本的600并非600个唯一物理场景

Austin raceline1有2,096个waypoints。当前公式 `i*2096/(50-1)` 同时生成0和2096；eval端再做modulo后，2096回到0。因此名义50个start只有49个唯一start，名义600条只有588个唯一 `(ego, opponent raceline, speed)` 组合，12条是端点重复。非零offset可以与base start集合错开，但仍保留内部12条重复。终审前应修正分母或按物理键去重，并核验offset面板与选择面板、最好也与训练起点集合不重叠；它应被称为Austin域内shifted holdout，而非跨地图独立holdout。""",
    },
    {"id": "panel_audit_block", "type": "table", "tableId": "panel_audit_table"},
    {
        "id": "evidence",
        "type": "markdown",
        "sourceId": "checkpoint_eval",
        "body": """## 0.20 的 U30 很好，但轨迹不足以支持单调外推

两条12-worker、30-update轨迹是当前最干净的 clip 对照。0.15 为 `16→18→18→14→14→17→20`，0.20 为 `21→18→16→13→13→17→11`。七个 checkpoints 的平均碰撞仅由16.71降到15.57；真正大的差异只出现在U30。因此“0.20比0.15更好”可作为候选方向，但不能外推出“0.25/0.30会继续更好”。下图用离散更新点的分组柱图，避免把七个选定 checkpoints 误读成连续时间趋势。""",
    },
    {"id": "collision_chart_block", "type": "chart", "chartId": "collision_chart"},
    {
        "id": "paired_section",
        "type": "markdown",
        "sourceId": "paired_eval",
        "body": """## 场景配对证据仍不足以确认 U30 优势稳定

U30 中 0.20 解决了15个 0.15 的碰撞场景，同时新增6个，未校正配对 p=0.078。这个结果比只看9次净下降更有信息，但尚未跨过常用0.05门槛，并且 U30 是查看多个 checkpoints 后选出的最优点，存在选择偏差。最合理动作是验证，而不是立即扩大搜索。""",
    },
    {"id": "paired_block", "type": "table", "tableId": "paired_table"},
    {
        "id": "telemetry_section",
        "type": "markdown",
        "sourceId": "training_telemetry",
        "body": """## 0.20 并未严重受 clipping 限制，但 KL 尾部已经较宽

在 U2-U30，0.20 的平均 clip fraction 约5.3%，中位数约4.5%。这意味着把阈值提高到0.25只会直接改变一小部分当前超出 ±20% 的 ratio；潜在收益存在，但预计不会是线性放大。同时，29个 updates 中已有15个出现 `approx_kl_max>0.5`、5个超过1.0。clip objective 本身不是硬 KL 约束，更大的 clip 会进一步放宽 policy ratio，必须用 eval 安全指标约束，而不能根据训练 return 决策。""",
    },
    {"id": "telemetry_block", "type": "table", "tableId": "telemetry_table"},
    {
        "id": "horizon_section",
        "type": "markdown",
        "sourceId": "late_training",
        "body": """## 45 updates 值得做，但目的是判断晚期稳定性，不是追逐更低单点

U20/U25/U30 的 eval 碰撞为 `13→17→11`，因此不能称为持续下降；同时 U20-U30 的 actor 相邻 checkpoint relative-L2 仍在约 `6.95e-5–1.30e-4` 之间，没有收缩到零，update KL 也继续出现0.443等尖峰。模型仍在移动，但是否向更优策略移动尚未确定。

建议 fresh-start 跑到45U并评估U35/U40/U45。当前实现不保存actor optimizer、critic optimizer、RNG、scenario queues和完整环境状态，不能把U30 actor作为“精确续训”。由于LR与clip schedule均为常量，同源同worker的45U `clip=0.20` 理应复现前30U；可用U30 actor tensor/hash作为复现检查。收敛判据应是U35/U40/U45的晚期均值、范围和碰撞场景churn，而不是45个checkpoints中的最小值。""",
    },
    {"id": "late_training_block", "type": "table", "tableId": "late_training_table"},
    {
        "id": "gamma_gae_section",
        "type": "markdown",
        "sourceId": "discount_horizons",
        "body": """## Gamma 先冻结；GAE lambda 可以研究但排在后面

`gamma=0.999` 同时进入PPO return/GAE和 `gamma*Phi(next)-Phi(current)` risk shaping。修改它会同时改变优化目标和reward credit redistribution，不是干净的折扣轴，因此本轮保持0.999。

`gae_lambda` 不改变reward，理论上更适合单轴实验。当前0.995在100Hz下使TD residual权重半衰期约1.15秒；降到0.99缩短到0.63秒，可能在高EV critic下减少方差，但也更依赖critic bootstrap。现有日志没有按collision/ordinary记录advantage mean/std或collision前时间滞后，尚不能判断该向上还是向下调。若后续只开一条稳定性ablation，可先试0.99；在此之前不建议同时测试0.9975/1.0。""",
    },
    {"id": "discount_block", "type": "table", "tableId": "discount_table"},
    {
        "id": "hard_neighbor_section",
        "type": "markdown",
        "sourceId": "hard_neighbor",
        "body": """## Hard neighbors 值得做，但必须是固定 boundary-aware cache

standalone probe的18条neighbors中有11条BC碰撞，描述性碰撞率61.1%，相对完整训练候选的4.45%明显富集；但它刻意选择了3个碰撞密集家族，且 `pipeline_integration=false`。直接把11条加入479条基础collision pool只占collision pool约2.24%，折算全部transitions约1.12%，信息弱且容易记住三个位置。

更干净的方案是从10,800条独立训练候选自动发现1,042条 `collision↔other` 邻接边，在边界内部生成新候选，再由同一冻结BC完整分类；最终固定cache只保留确认的ego collision。第一次A/B保持50/50 collision/ordinary transitions、reward、clip、horizon、seed和workers不变，并从同一BC fresh-start。""",
    },
    {"id": "hard_neighbor_block", "type": "table", "tableId": "hard_neighbor_table"},
    {
        "id": "roadmap_section",
        "type": "markdown",
        "sourceId": "experiment_roadmap",
        "body": """## 最小实验矩阵先关闭训练轴，再打开数据轴

优先级按信息增益与归因能力排列：先用现有base cache确定clip和训练长度，再单独评估boundary-aware cache；GAE只在补足advantage telemetry后考虑，gamma继续冻结。这样每一步只有一个解释变量，不会把“更长训练”“更宽clip”和“更难场景”混成一个不可解释的结果。""",
    },
    {"id": "roadmap_block", "type": "table", "tableId": "roadmap_table"},
    {
        "id": "scope",
        "type": "markdown",
        "body": """## 范围、数据与定义

- 比较对象：Group 5 的 `long_clip015` 与 `long_clip020`，共同为12 workers、30 updates、batch 12,800、中档 actor LR、target-KL关闭。
- Eval：同一 Austin600 场景集合，U1/U5/U10/U15/U20/U25/U30。
- `resolved/created`：0.20 相对0.15消除/新增的碰撞 scenario IDs。
- `clip fraction`：PPO importance ratio 超出当前 `[1−clip,1+clip]` 的记录比例，不等于发生参数更新的比例，也不能直接预测0.25结果。
- 训练 telemetry U2-U30；排除两组共同的U1 warm-up后瞬态。
- `gamma*lambda` 表示GAE中TD residual随时间的几何权重；模拟器为100Hz。
- hard-neighbor证据只来自独立训练候选与冻结BC，不读取Austin600失败场景。""",
    },
    {
        "id": "methodology",
        "type": "markdown",
        "body": """## 方法

1. 从两条 run 的原始 `results_multi.json` 重建每个 checkpoint 的600个场景结果。
2. 按 scenario ID 配对碰撞集合，计算 shared/resolved/created 和双侧 exact McNemar/binomial p。
3. 从原始 `metrics.jsonl` 汇总 U2-U30 的 clip fraction 与 approximate-KL 尾部。
4. 对U19-U30 actor checkpoints计算相邻参数relative-L2，判断优化是否静止。
5. 在固定gamma0.999、100Hz下计算GAE temporal weights，明确lambda改动的credit范围。
6. 从10,800条cache outcomes重建一轴相邻的collision/other outcome-flip边，并核对standalone hard-neighbor probe。
7. 按“一次一个轴”设计训练长度、clip、collision curriculum与GAE的实验顺序。""",
    },
    {
        "id": "limitations",
        "type": "markdown",
        "body": """## 局限与稳健性边界

- 每个 clip 只有 seed 42；无法估计跨 seed 方差。
- Austin600 同时承担 checkpoint 选择，U30配对 p 未校正多重查看，不能当作确认性结果。
- 当前Austin600只有588个唯一物理场景；端点别名造成12个重复槽位，offset holdout也继承该问题。
- 训练日志可以描述0.15/0.20发生了什么，但不能反事实预测0.25。
- `approx_kl_max` 对少量样本敏感；这里只把它当漂移风险信号，不当作性能代理。
- 增大 clip 可能改变后续采样分布，因此不能根据当前约5.3%的 clip fraction线性估算收益。
- actor参数仍移动不等于eval还会继续改善；45U的收益必须实测。
- hard-neighbor probe有选择偏差；1,042条boundary pairs只是候选生成依据，不是可学习性证明。
- GAE方向没有advantage distribution telemetry支持，0.99只是一条后续稳定性假设。""",
    },
    {
        "id": "next_steps",
        "type": "markdown",
        "body": """## 推荐的最小实验顺序

1. **先修正/审计评估面板：** 消除端点重复，验证shifted holdout与选择面板的有效ego indices不重叠。
2. **运行A：** fresh-start `clip=0.20, 45U`。若episode流和U30 actor/critic/metrics精确复现long020，则复用已有U1/U5/U10/U15/U20/U25/U30结果，只评U35/U40/U45；这可省7次而不是5次600场景eval。
3. **关闭clip轴：** 如果仍认为0.25待决，就在hard-neighbor之前运行matched `clip=0.25,45U`；否则明确冻结0.20并取消0.25。
4. **再打开hard-neighbor轴：** 只有clip和horizon已经冻结，才做base-cache vs schema-2 boundary-cache fresh-start A/B。
5. **暂不动gamma；延后GAE：** role级advantage统计可用不消耗RNG的raw-buffer只读实现搭载；collision前信用滞后须另行定义和验证，不能默认“免费”。""",
    },
    {
        "id": "questions",
        "type": "markdown",
        "body": """## 仍待回答的问题

1. U35/U40/U45的晚期均值是否低于U20/U25/U30，而不只是出现另一个单点低谷？
2. 0.25相对0.20是否在相同45U路径中持续减少同一批困难场景？
3. boundary-aware cache增加的是可改善边界碰撞，还是更多深碰撞/近不可解场景？
4. 按role与collision时间分解advantage后，lambda0.995的问题更像高方差还是credit过短？
5. 如果hard-neighbor有效，下一步应增加唯一边界场景覆盖，还是调整collision-role内部权重？""",
    },
]

generated = datetime.now(timezone.utc).isoformat()
artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "End2Race PPO 下一轮实验路线：Clip、45U、GAE 与 Hard Neighbors",
        "description": "基于 Group 5 训练 telemetry、Austin600 配对场景、actor checkpoint movement 与 collision-cache boundary evidence 的实验路线。",
        "generatedAt": generated,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    },
    "snapshot": {
        "version": 1,
        "generatedAt": generated,
        "status": "ready",
        "datasets": {
            "headline": headline,
            "checkpoint_eval": eval_rows,
            "paired": paired,
            "telemetry": telemetry,
            "late_training": late_training,
            "discount_horizons": discount_horizons,
            "hard_neighbor": hard_neighbor,
            "evaluation_panel_audit": evaluation_panel_audit,
            "paired_stat_examples": paired_stat_examples,
            "roadmap": roadmap,
        },
        "accessIssues": [],
    },
    "sources": sources,
}

(OUT / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
(OUT / "REPORT_NOTES.md").write_text(
    """# PPO experiment-roadmap report notes

## Required structure map

Title; technical summary; key findings with visual evidence; scope and definitions; methodology;
limitations and robustness; recommended next steps; further questions.

## Chart map

| Section | Question | Family | Fields | Supported claim | Palette |
|---|---|---|---|---|---|
| Checkpoint evidence | How do 0.15 and 0.20 compare across seven planned eval points? | Grouped bar | update, collisions, clip | 0.20's large advantage is concentrated at U30 | Hard two-root comparator |

## Visualization rationale

Seven selected checkpoints are too sparse for a continuous line-trend claim, so the report uses grouped bars.
Paired identities, late-training movement, discount horizons, hard-neighbor evidence, and the experiment roadmap
use exact tables because lookup across multiple measures and caveats is the point. No forecast chart is shown for
U35-U45 because those outcomes have not been observed.

## Delivery

Portable HTML, technical audience, single report surface.
"""
)
print(f"Wrote {OUT / 'artifact.json'}")

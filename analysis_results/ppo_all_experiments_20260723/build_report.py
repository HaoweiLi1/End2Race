#!/usr/bin/env python3
"""Build the canonical portable technical report for all End2Race PPO evals."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def clean(value):
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    return [{key: clean(value) for key, value in row.items()} for row in frame.to_dict("records")]


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False)


def source(source_id: str, label: str, path: str, description: str, *, tables: list[str],
           filters: list[str] | None = None, definitions: list[str] | None = None) -> dict:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        sql = f"SELECT * FROM read_csv_auto('{path}', header = true);"
    elif suffix == ".json":
        sql = f"SELECT * FROM read_json_auto('{path}');"
    else:
        sql = f"SELECT '{path}' AS artifact_path;"
    query = {
        "engine": "duckdb",
        "language": "sql",
        "sql": sql,
        "description": description,
        "tables_used": tables,
    }
    if filters:
        query["filters"] = filters
    if definitions:
        query["metric_definitions"] = definitions
    return {"id": source_id, "label": label, "path": path, "query": query}


summary = json.loads((OUT / "analysis_summary.json").read_text())
validation = json.loads((OUT / "validation_receipt.json").read_text())
npz_audit = json.loads((OUT / "npz_audit.json").read_text())
architecture = json.loads((OUT / "model_architecture.json").read_text())

panels = pd.read_csv(OUT / "eval_panels.csv")
episodes = pd.read_csv(OUT / "eval_episode_outcomes.csv")
paired = pd.read_csv(OUT / "paired_vs_bc.csv")
group_summary = pd.read_csv(OUT / "group_summary.csv")
group_control = pd.read_csv(OUT / "group_control_audit.csv")
run_inventory = pd.read_csv(OUT / "run_inventory.csv")
run_eval = pd.read_csv(OUT / "run_eval_summary.csv")
actor_delta = pd.read_csv(OUT / "actor_parameter_deltas.csv")
critic_summary = pd.read_csv(OUT / "critic_parameter_summary.csv")
kinematics = pd.read_csv(OUT / "collision_episode_kinematics.csv")
frequency = pd.read_csv(OUT / "scenario_frequency_unique_policies.csv")
bc_tail_outcomes = pd.read_csv(OUT / "bc_tail_scenario_outcomes.csv")
correlations = pd.read_csv(OUT / "training_eval_correlations.csv")


RUN_LABELS = {
    "ppo_independent_gru_0721_base": "独立 GRU critic",
    "ppo_privilege_mlp_0721_base": "特权 MLP critic",
    "ppo_privilege_gru_0721_base": "P20 privilege_gru 基线",
    "ppo_privilege_gru_0721_bs25600": "batch 25,600",
    "ppo_privilege_gru_0721_bs51200": "batch 51,200",
    "ppo_privilege_gru_0721_clip010": "clip .10（8 workers）",
    "ppo_privilege_gru_0721_clip020": "clip .20（8 workers）",
    "ppo_privilege_gru_0722_clip015_tkl002": "target-KL .02（8 workers）",
    "ppo_privilege_gru_0722_clip015_tkl004": "target-KL .04",
    "ppo_privilege_gru_0722_long_clip015": "clip .15 / 30U",
    "ppo_privilege_gru_0722_long_clip020": "clip .20 / 30U",
    "ppo_privilege_gru_0722_long45_clip020": "clip .20 / 45U",
    "ppo_privilege_gru_0722_long45_clip025": "clip .25 / 45U",
    "ppo_privilege_gru_0722_long45_clip020_hard": "clip .20 / hard-neighbor / 45U",
    "ppo_privilege_gru_0722_lr1_tkloff": "actor LR 1x",
    "ppo_privilege_gru_0722_lr5_tkloff": "actor LR 5x",
}

PANEL_LABELS = {
    "BC": "BC",
    "ppo_privilege_gru_0721_base_u0020": "P20 base U20",
    "ppo_privilege_gru_0722_long_clip020_u0030": "clip .20 U30",
    "ppo_privilege_gru_0722_long45_clip020_u0040": "clip .20 U40",
    "ppo_privilege_gru_0722_long45_clip020_u0045": "clip .20 U45",
    "ppo_privilege_gru_0722_long45_clip025_u0045": "clip .25 U45",
    "ppo_privilege_gru_0722_long45_clip020_hard_u0035": "hard U35",
    "ppo_privilege_gru_0722_long45_clip020_hard_u0045": "hard U45",
}


# Bounded report datasets ---------------------------------------------------
headline = pd.DataFrame([{
    "bc_collisions": 22,
    "best_collisions": 11,
    "collision_reduction_rate": 0.5,
    "bc_tail": 11,
    "best_tail": 4,
    "tail_reduction_rate": 7 / 11,
    "best_overtakes": 358,
    "valid_panels": 92,
    "episode_npz_records": 55_799,
}])

selected_names = list(PANEL_LABELS)
selected_eval = panels[panels.panel.isin(selected_names)].copy()
selected_eval["checkpoint"] = selected_eval.panel.map(PANEL_LABELS)
selected_eval = selected_eval[[
    "checkpoint", "panel", "collision_count", "overtake_count", "follow_count",
    "collision_with_opponent_count", "wall_like_collision_count",
    "merge_tail_relaxed_count", "merge_tail_primary_count", "merge_tail_strict_count",
    "merge_tail_primary_share_of_collisions", "valid",
]]

selected_pair = paired[paired.panel.isin(selected_names[1:])].copy()
selected_pair["checkpoint"] = selected_pair.panel.map(PANEL_LABELS)
selected_pair = selected_pair[[
    "checkpoint", "panel", "collision_resolved", "collision_shared", "collision_created",
    "collision_net_reduction", "collision_exact_p",
    "collision_cluster_bootstrap_diff_ci_low", "collision_cluster_bootstrap_diff_ci_high",
    "tail_resolved", "tail_shared", "tail_created", "tail_net_reduction", "tail_exact_p",
    "tail_cluster_bootstrap_diff_ci_low", "tail_cluster_bootstrap_diff_ci_high",
    "bc_tail_scenarios_still_any_collision", "bc_tail_scenarios_now_overtake",
    "collision_exact_p_holm_all_valid_panels", "tail_exact_p_holm_all_valid_panels",
]]

current_runs = {
    "ppo_privilege_gru_0722_long45_clip020": "clip .20",
    "ppo_privilege_gru_0722_long45_clip025": "clip .25",
    "ppo_privilege_gru_0722_long45_clip020_hard": "clip .20 + hard",
}
current_eval_runs = {
    **current_runs,
    # U1-U30 evals were saved under the 30U run; actor checkpoints are byte-identical
    # to the corresponding checkpoints in the continued 45U run.
    "ppo_privilege_gru_0722_long_clip020": "clip .20",
}
current_paths = panels[(panels.run.isin(current_eval_runs)) & (panels.valid)].copy()
current_paths["arm"] = current_paths.run.map(current_eval_runs)
current_paths = current_paths[[
    "arm", "run", "update", "collision_count", "merge_tail_primary_count",
    "merge_tail_relaxed_count", "merge_tail_strict_count", "overtake_count", "follow_count",
]]

class_labels = {
    "post_overtake_merge_rear_sweep": "超车后并线尾部扫碰（主判据）",
    "post_overtake_rear_contact_without_merge_threshold": "超车后后部接触（未达并线阈值）",
    "prepass_or_front_side_opponent_contact": "超车前/前侧 opponent 接触",
    "ego_only_wall_like": "ego-only / 墙类",
}
mix = episodes[(episodes.panel.isin(selected_names)) & (episodes.ego_collision)].copy()
mix = mix.groupby(["panel", "collision_class"], as_index=False).size().rename(columns={"size": "events"})
mix["checkpoint"] = mix.panel.map(PANEL_LABELS)
mix["collision_class_label"] = mix.collision_class.map(class_labels)

top_frequency = frequency.sort_values(
    ["collision_panels", "merge_tail_panels", "scenario_id"], ascending=[False, False, True]
).head(15).copy()
top_frequency["scenario_short"] = (
    top_frequency.scenario_id.str.replace("evaluation-", "", regex=False)
    .str.replace("raceline", "r", regex=False)
)

formal_updates = {1, 5, 10, 15, 20, 25, 30, 35, 40, 45}
delta_paths = actor_delta[(actor_delta.run.isin(current_runs)) & (actor_delta["update"].isin(formal_updates))].copy()
delta_paths["arm"] = delta_paths.run.map(current_runs)

selected_delta_names = [
    "ppo_privilege_gru_0721_base_u0020",
    "ppo_privilege_gru_0722_long_clip020_u0030",
    "ppo_privilege_gru_0722_long45_clip020_u0045",
    "ppo_privilege_gru_0722_long45_clip025_u0045",
    "ppo_privilege_gru_0722_long45_clip020_hard_u0045",
]
selected_delta = actor_delta[actor_delta.panel.isin(selected_delta_names)].copy()
selected_delta["checkpoint"] = selected_delta.panel.map(PANEL_LABELS)
selected_delta = selected_delta[[
    "checkpoint", "update", "trainable_actor_parameter_count", "fixed_actor_parameter_count",
    "actor_relative_l2_from_bc", "actor_delta_rms_from_bc", "actor_max_abs_delta_from_bc",
    "gru_relative_l2_from_bc", "head_relative_l2_from_bc", "fixed_frontend_delta_l2",
    "policy_sha256",
]]

model_contract = pd.DataFrame([
    {"section": "Actor", "item": "Observation", "value": "361D", "definition": "360 lidar + previous speed"},
    {"section": "Actor", "item": "Action", "value": "2D", "definition": "steering latent + physical speed"},
    {"section": "Actor", "item": "GRU", "value": "420 → 1680", "definition": "speed embedding concatenated with lidar"},
    {"section": "Actor", "item": "Parameters", "value": f"{architecture['actor_total_parameter_count']:,}", "definition": f"trainable {architecture['actor_trainable_parameter_count']:,}; fixed {architecture['actor_fixed_parameter_count']:,}"},
    {"section": "Actor", "item": "Trainable", "value": "GRU + output head", "definition": "BC initialization; frontend frozen"},
    {"section": "Actor", "item": "Exploration std", "value": "0.03 / 0.15", "definition": "steering latent / physical speed"},
    {"section": "Critic", "item": "P20 observation", "value": "381D", "definition": "361 actor input + 20 privileged features"},
    {"section": "Critic", "item": "privilege_gru parameters", "value": f"{architecture['critic_parameter_counts_by_observed_variant']['privilege_gru']:,}", "definition": "independent BC-initialized GRU + late P20 projection"},
    {"section": "PPO", "item": "Discount / GAE", "value": "0.999 / 0.995", "definition": "100 Hz, 8 s horizon"},
    {"section": "PPO", "item": "Epochs", "value": "2 actor / 5 critic", "definition": "per formal update"},
    {"section": "PPO", "item": "Selected batch / clip", "value": "12,800 / 0.20", "definition": "current observed best-total arm"},
    {"section": "Vehicle", "item": "Geometry", "value": "0.58 m × 0.31 m", "definition": "length × width"},
    {"section": "Simulator", "item": "Step / horizon", "value": "0.01 s / 8 s", "definition": "post-step-v2 normally has 802 trace rows"},
])

run_table = run_inventory.merge(run_eval, on="run", how="left", validate="one_to_one")
run_table["run_label"] = run_table.run.map(RUN_LABELS).fillna(run_table.run)
run_table["target_kl_display"] = run_table.target_kl.apply(lambda x: "off" if pd.isna(x) else f"{x:.2f}")
run_table["collision_pool"] = run_table.apply(
    lambda row: f"{int(row.collision_pool_count)} ({'hard' if row.hard_neighbors else 'base'})", axis=1
)
run_table = run_table[[
    "run_label", "run", "critic", "env_workers", "batch_size", "num_updates", "gru_learning_rate",
    "head_learning_rate", "clip_range", "target_kl_display", "collision_pool", "early_stop_updates",
    "mean_approx_kl", "max_approx_kl", "final_explained_variance_post", "final_value_loss_post",
    "mean_training_collision_rate", "collision_path", "merge_tail_path", "overtake_path",
    "latest_valid_collision_count", "latest_valid_tail_count", "latest_valid_overtake_count",
]]

group_table = group_summary.copy()
group_table = group_table[[
    "group", "arm", "updates", "collision_path", "merge_tail_path", "overtake_path",
    "mean_collisions_all_valid", "mean_tail_all_valid", "mean_collisions_u25_plus",
    "mean_tail_u25_plus", "best_update", "best_collision_count", "best_tail_count",
    "final_update", "final_collision_count", "final_tail_count", "final_overtake_count",
    "mean_adjacent_collision_flips", "mean_adjacent_tail_flips", "unique_policy_count",
]]

all_panels = panels.copy()
all_panels["checkpoint"] = all_panels.apply(
    lambda row: "BC" if row.panel == "BC" else f"{RUN_LABELS.get(row['run'], row['run'])} U{int(row['update'])}", axis=1
)
all_panels = all_panels[[
    "checkpoint", "panel", "valid", "episode_rows", "collision_count", "overtake_count", "follow_count",
    "merge_tail_relaxed_count", "merge_tail_primary_count", "merge_tail_strict_count",
    "collision_with_opponent_count", "wall_like_collision_count", "median_collision_time_s",
]]

bc_tail = kinematics[(kinematics.panel == "BC") & (kinematics.merge_tail_primary)].copy()
bc_episode_table = bc_tail[[
    "scenario_id", "collision_time_s", "pass_time_s", "pass_to_collision_s",
    "post_pass_lateral_convergence_m", "terminal_opponent_body_x_m", "terminal_opponent_body_y_m",
    "terminal_heading_difference_rad", "terminal_abs_kinematic_slip_proxy_rad",
]].sort_values("scenario_id")

transition_panels = [
    "ppo_privilege_gru_0722_long_clip020_u0030",
    "ppo_privilege_gru_0722_long45_clip020_u0045",
    "ppo_privilege_gru_0722_long45_clip020_hard_u0035",
    "ppo_privilege_gru_0722_long45_clip020_hard_u0045",
]


def outcome_label(row: pd.Series) -> str:
    if bool(row.merge_tail_primary):
        return "主判据尾扫碰撞"
    if bool(row.ego_collision):
        return "其他碰撞"
    if row.outcome == "overtake":
        return "超车完成"
    if row.outcome == "follow":
        return "跟随"
    return str(row.outcome)


tail_transitions = []
for scenario in sorted(bc_tail.scenario_id):
    parts = scenario.split("-")
    row = {"scenario_id": scenario, "start": parts[1], "ego": parts[2], "raceline": parts[3], "speed": parts[4]}
    for panel_name in transition_panels:
        match = bc_tail_outcomes[(bc_tail_outcomes.scenario_id == scenario) & (bc_tail_outcomes.panel == panel_name)]
        row[PANEL_LABELS[panel_name]] = outcome_label(match.iloc[0])
    tail_transitions.append(row)
tail_transitions = pd.DataFrame(tail_transitions)

quality = pd.DataFrame([
    {"check": "原始 eval JSON 对账", "result": "93/93 panels pass", "interpretation": "episode、scenario、collision/overtake/follow 逐项重算一致"},
    {"check": "有效性边界", "result": "92 valid; 1 invalid", "interpretation": "lr5 U20 仅 599 rows，排除排序与推断"},
    {"check": "episode 粒度", "result": "55,799 unique panel-scenario rows", "interpretation": "无重复键"},
    {"check": "NPZ 数值/数组对齐", "result": "55,799/55,799", "interpretation": "所有数值有限且 leading dimensions 对齐"},
    {"check": "post-step-v2", "result": "34,799/34,799 terminal valid", "interpretation": "碰撞 marker 与 JSON 完全一致"},
    {"check": "legacy traces", "result": "21,000; 736 expected marker gaps", "interpretation": "旧格式漏 terminal post-step；JSON 控制碰撞真值"},
    {"check": "Actor policy 去重", "result": "91 valid PPO panels → 86 unique policies", "interpretation": "5 对 byte-identical actor 的 eval 结果也完全一致"},
    {"check": "训练 metrics/checkpoints", "result": "415 formal rows; 16/16 runs complete", "interpretation": "updates 连续、数值有限、actor/critic checkpoints 完整"},
    {"check": "历史 provenance", "result": "seed / source commit not recorded", "interpretation": "不能从当前工作树回填历史 run 的 seed 或 commit"},
])

control_table = group_control.copy()
control_table["strict_single_axis"] = control_table.strict_single_axis.map({True: "是", False: "否"})
control_table["confounds"] = control_table.confounds.fillna("无")

correlation_table = correlations.copy()

# Write bounded, reviewable report sources.
write_csv(selected_eval, "report_selected_eval.csv")
write_csv(selected_pair, "report_selected_paired.csv")
write_csv(current_paths, "report_current_paths.csv")
write_csv(mix, "report_collision_mix.csv")
write_csv(top_frequency, "report_top_scenarios.csv")
write_csv(delta_paths, "report_actor_delta_paths.csv")
write_csv(selected_delta, "report_selected_actor_deltas.csv")
write_csv(model_contract, "report_model_contract.csv")
write_csv(run_table, "report_training_runs.csv")
write_csv(group_table, "report_group_summary.csv")
write_csv(all_panels, "report_all_eval_panels.csv")
write_csv(bc_episode_table, "report_bc_tail_episodes.csv")
write_csv(tail_transitions, "report_bc_tail_transitions.csv")
write_csv(quality, "report_quality.csv")


sources = [
    source(
        "analysis_summary", "All-experiment consolidated analysis",
        "analysis_results/ppo_all_experiments_20260723/analysis_summary.json",
        "Consolidates model, training, eval, pairing, trace geometry, robustness and selection results.",
        tables=["post-trained/*", "eval_results/*/multiagents/results_multi.json", "eval_results/*/multiagents/traces/*.npz"],
        filters=["Austin600", "valid panels for ranking and inference", "duplicate actor policies removed for cross-policy frequency"],
        definitions=["collision truth comes from results_multi.json", "primary tail signature requires pass lead time >=0.10 s, opponent behind, opponent collision, and lateral convergence >=0.10 m"],
    ),
    source(
        "selected_eval", "Selected checkpoint evaluation summary",
        "analysis_results/ppo_all_experiments_20260723/report_selected_eval.csv",
        "Loads BC and decision-relevant PPO checkpoint totals and tail-signature sensitivity counts.",
        tables=["analysis_results/ppo_all_experiments_20260723/eval_panels.csv"],
        filters=["BC plus seven decision-relevant checkpoints"],
        definitions=["collision_count, overtake_count, and follow_count partition each valid 600-scenario panel", "relaxed/primary/strict are threshold sensitivity variants"],
    ),
    source(
        "paired_bc", "Episode-paired PPO versus BC transitions",
        "analysis_results/ppo_all_experiments_20260723/report_selected_paired.csv",
        "Loads scenario-identity resolved/shared/created transitions and paired fixed-panel uncertainty.",
        tables=["analysis_results/ppo_all_experiments_20260723/paired_vs_bc.csv", "analysis_results/ppo_all_experiments_20260723/eval_episode_outcomes.csv"],
        filters=["selected valid PPO checkpoints", "same 600 scenario IDs as BC"],
        definitions=["paired exact p is a two-sided exact binomial test on resolved versus created scenarios", "cluster bootstrap CI resamples 50 startpoint clusters", "Holm adjustment covers all 91 valid PPO panel comparisons"],
    ),
    source(
        "current_paths", "Current 45-update checkpoint paths",
        "analysis_results/ppo_all_experiments_20260723/report_current_paths.csv",
        "Loads total-collision and tail-signature paths for clip .20, clip .25 and hard-neighbor arms.",
        tables=["analysis_results/ppo_all_experiments_20260723/eval_panels.csv"],
        filters=["valid checkpoints U1/U5/U10/U15/U20/U25/U30/U35/U40/U45"],
    ),
    source(
        "collision_mix", "Selected-checkpoint collision mechanism mix",
        "analysis_results/ppo_all_experiments_20260723/report_collision_mix.csv",
        "Counts mutually exclusive collision geometry classes for selected checkpoints.",
        tables=["analysis_results/ppo_all_experiments_20260723/eval_episode_outcomes.csv", "eval_results/*/multiagents/traces/*.npz"],
        filters=["ego_collision=true", "selected checkpoints"],
        definitions=["classes partition all ego-collision episodes; primary tail class is the user's target mechanism"],
    ),
    source(
        "model_contract", "Actor, critic and PPO model contract",
        "analysis_results/ppo_all_experiments_20260723/report_model_contract.csv",
        "Presents source-verified actor/critic dimensions, parameter counts and selected PPO contract.",
        tables=["analysis_results/ppo_all_experiments_20260723/model_architecture.json", "train_ppo.py", "ppo/*.py", "post-trained/*/run_config.json"],
    ),
    source(
        "actor_deltas", "Selected actor checkpoint parameter deltas",
        "analysis_results/ppo_all_experiments_20260723/report_selected_actor_deltas.csv",
        "Loads tensor-level actor changes relative to the BC checkpoint.",
        tables=["analysis_results/ppo_all_experiments_20260723/actor_parameter_deltas.csv", "pretrained/end2race.pth", "post-trained/*/checkpoints/actor_u*.pth"],
        definitions=["relative L2 is norm(checkpoint-BC)/norm(BC)", "fixed frontend delta covers k, speed_mlp and dummy_embedding"],
    ),
    source(
        "actor_delta_paths", "Actor movement over evaluated checkpoints",
        "analysis_results/ppo_all_experiments_20260723/report_actor_delta_paths.csv",
        "Loads BC-relative actor tensor movement for current 45-update arms.",
        tables=["analysis_results/ppo_all_experiments_20260723/actor_parameter_deltas.csv"],
        filters=["formal evaluated checkpoints", "current three 45-update arms"],
    ),
    source(
        "training_runs", "Training configuration and metrics by run",
        "analysis_results/ppo_all_experiments_20260723/report_training_runs.csv",
        "Joins all 16 recorded run configurations, optimization telemetry and evaluation paths.",
        tables=["analysis_results/ppo_all_experiments_20260723/run_inventory.csv", "analysis_results/ppo_all_experiments_20260723/run_eval_summary.csv", "post-trained/*/metrics.jsonl"],
        definitions=["training collision rate is from sampled PPO rollout episodes, not Austin600", "latest valid excludes the 599-row lr5 U20 eval"],
    ),
    source(
        "group_summary", "All controlled experiment groups",
        "analysis_results/ppo_all_experiments_20260723/report_group_summary.csv",
        "Loads all declared experiment-arm paths, late means and checkpoint identity churn.",
        tables=["analysis_results/ppo_all_experiments_20260723/group_summary.csv"],
        definitions=["U25+ mean uses U25/U30/U35/U40/U45 where present", "adjacent flips count scenario IDs whose collision state changes between evaluated checkpoints"],
    ),
    source(
        "control_audit", "Recorded control-axis audit",
        "analysis_results/ppo_all_experiments_20260723/group_control_audit.csv",
        "Flags comparisons whose recorded configs changed beyond the intended parameter axis.",
        tables=["post-trained/*/run_config.json", "analysis_results/ppo_all_experiments_20260723/group_control_audit.csv"],
    ),
    source(
        "all_eval_panels", "All BC and PPO evaluation panels",
        "analysis_results/ppo_all_experiments_20260723/report_all_eval_panels.csv",
        "Loads every one of the 93 discovered panels, including the isolated invalid panel.",
        tables=["analysis_results/ppo_all_experiments_20260723/eval_panels.csv", "eval_results/*/multiagents/results_multi.json"],
        definitions=["valid requires complete 600-scenario identity, trace coverage, finite metrics and aggregate reconciliation"],
    ),
    source(
        "bc_tail_episodes", "BC target-mechanism episode kinematics",
        "analysis_results/ppo_all_experiments_20260723/report_bc_tail_episodes.csv",
        "Loads all 11 BC episodes satisfying the primary post-overtake merge/rear-sweep geometry.",
        tables=["analysis_results/ppo_all_experiments_20260723/collision_episode_kinematics.csv", "eval_results/end2race_Austin/multiagents/traces/*.npz"],
        filters=["panel=BC", "merge_tail_primary=true"],
    ),
    source(
        "bc_tail_transitions", "Original BC tail-scenario outcomes under selected PPO policies",
        "analysis_results/ppo_all_experiments_20260723/report_bc_tail_transitions.csv",
        "Shows the fate of every original BC target-mechanism scenario at selected PPO checkpoints.",
        tables=["analysis_results/ppo_all_experiments_20260723/bc_tail_scenario_outcomes.csv"],
        filters=["the 11 BC primary-signature scenarios", "four selected PPO checkpoints"],
    ),
    source(
        "top_scenarios", "Collision commonality across unique PPO policies",
        "analysis_results/ppo_all_experiments_20260723/report_top_scenarios.csv",
        "Ranks recurrent scenario failures after byte-level actor policy deduplication.",
        tables=["analysis_results/ppo_all_experiments_20260723/scenario_frequency_unique_policies.csv"],
        filters=["86 unique valid PPO actor policies", "top 15 by collision frequency"],
        definitions=["collision_rate denominator is 86 unique actor policies", "merge_tail_rate uses the same denominator"],
    ),
    source(
        "quality", "Independent analysis validation receipt",
        "analysis_results/ppo_all_experiments_20260723/report_quality.csv",
        "Summarizes independent raw-JSON, pairing, geometry and NPZ audit checks.",
        tables=["analysis_results/ppo_all_experiments_20260723/validation_receipt.json", "analysis_results/ppo_all_experiments_20260723/npz_audit.json"],
    ),
    source(
        "correlations", "Training-to-eval descriptive correlations",
        "analysis_results/ppo_all_experiments_20260723/training_eval_correlations.csv",
        "Loads pooled descriptive correlations across 86 unique policy checkpoints.",
        tables=["analysis_results/ppo_all_experiments_20260723/training_metrics.csv", "analysis_results/ppo_all_experiments_20260723/eval_panels.csv"],
        filters=["valid panels", "duplicate actor policies removed"],
        definitions=["pooled across heterogeneous runs; diagnostic only and not causal"],
    ),
]


cards = [
    {
        "id": "total_collision_card", "description": "Observed best-total checkpoint on the fixed Austin600 panel.",
        "dataset": "headline", "sourceId": "selected_eval",
        "metrics": [
            {"label": "clip .20 U30 总碰撞", "field": "best_collisions", "format": "number"},
            {"label": "BC", "field": "bc_collisions", "format": "number"},
            {"label": "降幅", "field": "collision_reduction_rate", "format": "percent"},
        ],
    },
    {
        "id": "tail_card", "description": "Primary post-overtake merge/rear-sweep signature at the same checkpoint.",
        "dataset": "headline", "sourceId": "selected_eval",
        "metrics": [
            {"label": "clip .20 U30 主判据尾扫", "field": "best_tail", "format": "number"},
            {"label": "BC", "field": "bc_tail", "format": "number"},
            {"label": "降幅", "field": "tail_reduction_rate", "format": "percent"},
        ],
    },
    {
        "id": "coverage_card", "description": "Validated panels and episode/trace records included in the analysis.",
        "dataset": "headline", "sourceId": "quality",
        "metrics": [
            {"label": "有效 eval panels", "field": "valid_panels", "format": "number"},
            {"label": "episode / NPZ", "field": "episode_npz_records", "format": "number"},
        ],
    },
]


charts = [
    {
        "id": "current_total_path", "title": "45U 三个当前实验臂的总碰撞路径",
        "subtitle": "Austin600 evaluated checkpoints; 越低越好，BC=22 仅作文字基准",
        "type": "line", "dataset": "current_paths", "sourceId": "current_paths", "valueFormat": "number",
        "encodings": {
            "x": {"field": "update", "type": "quantitative", "label": "Formal update"},
            "y": {"field": "collision_count", "type": "quantitative", "label": "碰撞 / 600"},
            "color": {"field": "arm", "type": "nominal", "label": "实验臂"},
            "tooltip": [
                {"field": "merge_tail_primary_count", "type": "quantitative", "label": "主判据尾扫"},
                {"field": "overtake_count", "type": "quantitative", "label": "超车"},
                {"field": "follow_count", "type": "quantitative", "label": "跟随"},
            ],
        },
    },
    {
        "id": "current_tail_path", "title": "同一批 checkpoint 的主判据尾扫碰撞路径",
        "subtitle": "最低点为 hard U35=2，但该点总碰撞为20；宽松判据仍为7",
        "type": "line", "dataset": "current_paths", "sourceId": "current_paths", "valueFormat": "number",
        "encodings": {
            "x": {"field": "update", "type": "quantitative", "label": "Formal update"},
            "y": {"field": "merge_tail_primary_count", "type": "quantitative", "label": "主判据事件 / 600"},
            "color": {"field": "arm", "type": "nominal", "label": "实验臂"},
            "tooltip": [
                {"field": "merge_tail_relaxed_count", "type": "quantitative", "label": "宽松判据"},
                {"field": "merge_tail_strict_count", "type": "quantitative", "label": "严格判据"},
                {"field": "collision_count", "type": "quantitative", "label": "总碰撞"},
            ],
        },
    },
    {
        "id": "collision_mix_chart", "title": "关键 checkpoint 的碰撞几何分类",
        "subtitle": "四类互斥并覆盖所有 ego 碰撞；主判据事件在所有候选中仍存在",
        "type": "bar", "dataset": "collision_mix", "sourceId": "collision_mix", "valueFormat": "number",
        "encodings": {
            "x": {"field": "checkpoint", "type": "nominal", "label": "Checkpoint"},
            "y": {"field": "events", "type": "quantitative", "label": "Episode 数"},
            "color": {"field": "collision_class_label", "type": "nominal", "label": "碰撞几何类"},
            "tooltip": [
                {"field": "collision_class_label", "type": "nominal", "label": "分类"},
                {"field": "events", "type": "quantitative", "label": "事件"},
            ],
        },
    },
    {
        "id": "actor_delta_chart", "title": "Actor 相对 BC 的参数移动",
        "subtitle": "三条当前 45U 轨迹；固定前端参数在所有 checkpoint 的 L2 变化均为0",
        "type": "line", "dataset": "actor_delta_paths", "sourceId": "actor_delta_paths", "valueFormat": "number",
        "encodings": {
            "x": {"field": "update", "type": "quantitative", "label": "Formal update"},
            "y": {"field": "actor_relative_l2_from_bc", "type": "quantitative", "label": "Actor relative L2 vs BC"},
            "color": {"field": "arm", "type": "nominal", "label": "实验臂"},
            "tooltip": [
                {"field": "actor_delta_rms_from_bc", "type": "quantitative", "label": "Delta RMS"},
                {"field": "actor_max_abs_delta_from_bc", "type": "quantitative", "label": "Max abs delta"},
                {"field": "head_relative_l2_from_bc", "type": "quantitative", "label": "Head relative L2"},
            ],
        },
    },
    {
        "id": "persistent_chart", "title": "跨 86 个唯一 PPO actor 的最持续碰撞场景",
        "subtitle": "按 collision rate 排序；颜色不重复编码，尾扫频率见 tooltip 与下表",
        "type": "bar", "dataset": "top_scenarios", "sourceId": "top_scenarios", "valueFormat": "percent",
        "encodings": {
            "x": {"field": "scenario_short", "type": "nominal", "label": "Scenario"},
            "y": {"field": "collision_rate", "type": "quantitative", "label": "碰撞 policy 比例"},
            "tooltip": [
                {"field": "scenario_id", "type": "nominal", "label": "完整 ID"},
                {"field": "collision_panels", "type": "quantitative", "label": "碰撞 policies"},
                {"field": "merge_tail_panels", "type": "quantitative", "label": "尾扫 policies"},
                {"field": "merge_tail_rate", "type": "quantitative", "label": "尾扫比例", "format": "percent"},
            ],
        },
    },
]


tables = [
    {
        "id": "selected_eval_table", "title": "关键 checkpoint 的 eval 与阈值敏感性",
        "subtitle": "同一 Austin600；relaxed / primary / strict 同时报告",
        "dataset": "selected_eval", "sourceId": "selected_eval",
        "defaultSort": {"field": "collision_count", "direction": "asc"},
        "columns": [
            {"field": "checkpoint", "label": "checkpoint"},
            {"field": "collision_count", "label": "碰撞", "format": "number"},
            {"field": "overtake_count", "label": "超车", "format": "number"},
            {"field": "follow_count", "label": "跟随", "format": "number"},
            {"field": "merge_tail_relaxed_count", "label": "尾扫 relaxed", "format": "number"},
            {"field": "merge_tail_primary_count", "label": "尾扫 primary", "format": "number"},
            {"field": "merge_tail_strict_count", "label": "尾扫 strict", "format": "number"},
            {"field": "merge_tail_primary_share_of_collisions", "label": "尾扫占碰撞", "format": "percent"},
        ],
    },
    {
        "id": "paired_table", "title": "PPO 对 BC 的逐 scenario 配对转移",
        "subtitle": "raw exact p 与 startpoint-cluster bootstrap CI；Holm(91 panels) 均未通过",
        "dataset": "selected_pair", "sourceId": "paired_bc",
        "defaultSort": {"field": "collision_net_reduction", "direction": "desc"},
        "columns": [
            {"field": "checkpoint", "label": "checkpoint"},
            {"field": "collision_resolved", "label": "总碰撞消除", "format": "number"},
            {"field": "collision_shared", "label": "总碰撞共同", "format": "number"},
            {"field": "collision_created", "label": "总碰撞新增", "format": "number"},
            {"field": "collision_net_reduction", "label": "净减少", "format": "number"},
            {"field": "collision_exact_p", "label": "总碰撞 raw p", "format": "number"},
            {"field": "tail_resolved", "label": "尾扫消除", "format": "number"},
            {"field": "tail_shared", "label": "尾扫共同", "format": "number"},
            {"field": "tail_created", "label": "尾扫新增", "format": "number"},
            {"field": "tail_net_reduction", "label": "尾扫净减少", "format": "number"},
            {"field": "tail_exact_p", "label": "尾扫 raw p", "format": "number"},
            {"field": "bc_tail_scenarios_now_overtake", "label": "原11中变超车", "format": "number"},
        ],
    },
    {
        "id": "model_contract_table", "title": "模型与 PPO 合同",
        "subtitle": "当前源代码与保存 checkpoint 的联合核对",
        "dataset": "model_contract", "sourceId": "model_contract",
        "defaultSort": {"field": "section", "direction": "asc"},
        "columns": [
            {"field": "section", "label": "部分"}, {"field": "item", "label": "项目"},
            {"field": "value", "label": "值"}, {"field": "definition", "label": "定义"},
        ],
    },
    {
        "id": "actor_delta_table", "title": "关键 actor checkpoint 的参数变化",
        "subtitle": "相对 BC；fixed frontend delta 应严格为0",
        "dataset": "selected_delta", "sourceId": "actor_deltas",
        "defaultSort": {"field": "update", "direction": "asc"},
        "columns": [
            {"field": "checkpoint", "label": "checkpoint"},
            {"field": "update", "label": "update", "format": "number"},
            {"field": "actor_relative_l2_from_bc", "label": "actor rel L2", "format": "number"},
            {"field": "actor_delta_rms_from_bc", "label": "delta RMS", "format": "number"},
            {"field": "actor_max_abs_delta_from_bc", "label": "max abs delta", "format": "number"},
            {"field": "gru_relative_l2_from_bc", "label": "GRU rel L2", "format": "number"},
            {"field": "head_relative_l2_from_bc", "label": "head rel L2", "format": "number"},
            {"field": "fixed_frontend_delta_l2", "label": "fixed delta", "format": "number"},
        ],
    },
    {
        "id": "run_table", "title": "16 个训练 run：参数、训练 metrics 与 eval 路径",
        "subtitle": "training collision rate 来自 PPO rollout，不等于 Austin600 eval 碰撞率",
        "dataset": "training_runs", "sourceId": "training_runs",
        "defaultSort": {"field": "run_label", "direction": "asc"},
        "columns": [
            {"field": "run_label", "label": "run"}, {"field": "critic", "label": "critic"},
            {"field": "env_workers", "label": "workers", "format": "number"},
            {"field": "batch_size", "label": "batch", "format": "number"},
            {"field": "num_updates", "label": "updates", "format": "number"},
            {"field": "clip_range", "label": "clip", "format": "number"},
            {"field": "target_kl_display", "label": "target KL"},
            {"field": "collision_pool", "label": "collision pool"},
            {"field": "early_stop_updates", "label": "early-stop U", "format": "number"},
            {"field": "mean_approx_kl", "label": "mean KL", "format": "number"},
            {"field": "final_explained_variance_post", "label": "final critic EV", "format": "number"},
            {"field": "collision_path", "label": "eval collision path"},
            {"field": "merge_tail_path", "label": "eval tail path"},
            {"field": "latest_valid_collision_count", "label": "latest valid collision", "format": "number"},
            {"field": "latest_valid_tail_count", "label": "latest valid tail", "format": "number"},
        ],
    },
    {
        "id": "group_table", "title": "所有实验组的结果与 checkpoint 稳定性",
        "subtitle": "含早期 critic/batch/LR/KL 对照与当前 clip/hard 45U 对照",
        "dataset": "group_summary", "sourceId": "group_summary",
        "defaultSort": {"field": "group", "direction": "asc"},
        "columns": [
            {"field": "group", "label": "组"}, {"field": "arm", "label": "实验臂"},
            {"field": "collision_path", "label": "碰撞路径"}, {"field": "merge_tail_path", "label": "尾扫路径"},
            {"field": "mean_collisions_all_valid", "label": "碰撞均值", "format": "number"},
            {"field": "mean_tail_all_valid", "label": "尾扫均值", "format": "number"},
            {"field": "mean_collisions_u25_plus", "label": "U25+碰撞", "format": "number"},
            {"field": "mean_tail_u25_plus", "label": "U25+尾扫", "format": "number"},
            {"field": "final_collision_count", "label": "最终碰撞", "format": "number"},
            {"field": "final_tail_count", "label": "最终尾扫", "format": "number"},
            {"field": "mean_adjacent_collision_flips", "label": "相邻碰撞翻转", "format": "number"},
            {"field": "mean_adjacent_tail_flips", "label": "相邻尾扫翻转", "format": "number"},
        ],
    },
    {
        "id": "control_table", "title": "实验控制轴审计",
        "subtitle": "worker 数变化使部分历史比较不是严格单轴；G8/G9 是严格对照",
        "dataset": "control_audit", "sourceId": "control_audit",
        "defaultSort": {"field": "group", "direction": "asc"},
        "columns": [
            {"field": "group", "label": "组"}, {"field": "baseline_run", "label": "baseline"},
            {"field": "arm_run", "label": "arm"}, {"field": "intended_differences", "label": "预期变化"},
            {"field": "recorded_differences", "label": "实际变化"}, {"field": "confounds", "label": "混杂"},
            {"field": "strict_single_axis", "label": "严格单轴"},
        ],
    },
    {
        "id": "all_panels_table", "title": "完整 eval panel 清单（BC + 92 PPO）",
        "subtitle": "唯一无效 panel 保留展示但不进入排序/推断",
        "dataset": "all_eval_panels", "sourceId": "all_eval_panels",
        "defaultSort": {"field": "collision_count", "direction": "asc"},
        "columns": [
            {"field": "checkpoint", "label": "checkpoint"}, {"field": "valid", "label": "valid"},
            {"field": "episode_rows", "label": "rows", "format": "number"},
            {"field": "collision_count", "label": "碰撞", "format": "number"},
            {"field": "overtake_count", "label": "超车", "format": "number"},
            {"field": "follow_count", "label": "跟随", "format": "number"},
            {"field": "merge_tail_relaxed_count", "label": "relaxed", "format": "number"},
            {"field": "merge_tail_primary_count", "label": "primary", "format": "number"},
            {"field": "merge_tail_strict_count", "label": "strict", "format": "number"},
            {"field": "median_collision_time_s", "label": "碰撞时间中位(s)", "format": "number"},
        ],
    },
    {
        "id": "bc_tail_table", "title": "BC 的 11 个目标机制 episode",
        "subtitle": "pass-to-collision 与横向收敛直接来自 pose/raceline 投影",
        "dataset": "bc_tail_episodes", "sourceId": "bc_tail_episodes",
        "defaultSort": {"field": "scenario_id", "direction": "asc"},
        "columns": [
            {"field": "scenario_id", "label": "scenario"},
            {"field": "collision_time_s", "label": "碰撞(s)", "format": "number"},
            {"field": "pass_time_s", "label": "超车(s)", "format": "number"},
            {"field": "pass_to_collision_s", "label": "超车→碰撞(s)", "format": "number"},
            {"field": "post_pass_lateral_convergence_m", "label": "横向收敛(m)", "format": "number"},
            {"field": "terminal_opponent_body_x_m", "label": "终止 opp body-x(m)", "format": "number"},
            {"field": "terminal_opponent_body_y_m", "label": "终止 opp body-y(m)", "format": "number"},
            {"field": "terminal_abs_kinematic_slip_proxy_rad", "label": "slip proxy(rad)", "format": "number"},
        ],
    },
    {
        "id": "transition_table", "title": "原 11 个 BC 尾扫场景在 PPO 下的去向",
        "subtitle": "解决原场景并不等于机制归零；新增尾扫事件见配对表",
        "dataset": "bc_tail_transitions", "sourceId": "bc_tail_transitions",
        "defaultSort": {"field": "scenario_id", "direction": "asc"},
        "columns": [
            {"field": "scenario_id", "label": "scenario"}, {"field": "clip .20 U30", "label": "clip .20 U30"},
            {"field": "clip .20 U45", "label": "clip .20 U45"}, {"field": "hard U35", "label": "hard U35"},
            {"field": "hard U45", "label": "hard U45"},
        ],
    },
    {
        "id": "top_scenario_table", "title": "碰撞 episode 的跨 policy 共性",
        "subtitle": "86 个唯一 actor policy；同一 actor 的重复 eval 已去重",
        "dataset": "top_scenarios", "sourceId": "top_scenarios",
        "defaultSort": {"field": "collision_panels", "direction": "desc"},
        "columns": [
            {"field": "scenario_id", "label": "scenario"}, {"field": "opponent_raceline", "label": "raceline"},
            {"field": "opponent_speed_scale", "label": "speed", "format": "number"},
            {"field": "collision_panels", "label": "碰撞 policies", "format": "number"},
            {"field": "collision_rate", "label": "碰撞率", "format": "percent"},
            {"field": "merge_tail_panels", "label": "尾扫 policies", "format": "number"},
            {"field": "merge_tail_rate", "label": "尾扫率", "format": "percent"},
            {"field": "overtake_panels", "label": "超车 policies", "format": "number"},
            {"field": "follow_panels", "label": "跟随 policies", "format": "number"},
        ],
    },
    {
        "id": "correlation_table", "title": "训练 metrics 对固定 eval 碰撞的描述性相关",
        "subtitle": "86 个唯一 actor policy；跨异质 run pooled，不能作因果解释",
        "dataset": "correlations", "sourceId": "correlations",
        "defaultSort": {"field": "spearman_with_eval_collision_count", "direction": "desc"},
        "columns": [
            {"field": "metric", "label": "训练 metric"},
            {"field": "unique_policy_panels", "label": "policies", "format": "number"},
            {"field": "pearson_with_eval_collision_count", "label": "Pearson", "format": "number"},
            {"field": "spearman_with_eval_collision_count", "label": "Spearman", "format": "number"},
        ],
    },
    {
        "id": "quality_table", "title": "数据质量与独立复核",
        "subtitle": "结论评级：可分享但必须带单 seed、选择与 slip-state 局限",
        "dataset": "quality", "sourceId": "quality",
        "defaultSort": {"field": "check", "direction": "asc"},
        "columns": [
            {"field": "check", "label": "检查"}, {"field": "result", "label": "结果"},
            {"field": "interpretation", "label": "解释"},
        ],
    },
]


blocks = [
    {"id": "title", "type": "markdown", "layout": "full", "body": "# End2Race PPO 全实验：训练、Eval 与超车后尾扫碰撞分析"},
    {
        "id": "technical_summary", "type": "markdown", "layout": "full", "sourceId": "analysis_summary",
        "body": """## 技术摘要

**直接结论：PPO 显著缓解，但没有解决 BC 从 lattice expert 继承的“超车后并线、车尾扫到 opponent”问题。** BC 在固定 Austin600 上有22次碰撞，其中11次满足主判据。观察到的整体最佳折中是 `privilege_gru + batch 12,800 + clip 0.20 + target-KL off` 的U30：11次总碰撞、358次超车、231次跟随，主判据事件降至4次。逐场景配对显示，它消除10个BC原尾扫场景，但仍保留1个，并在另外3个场景新产生同类尾扫。

hard-neighbor U35把11个BC原尾扫场景全部变为超车，却仍有2个新的主判据事件、7个宽松判据事件和20次总碰撞。全部91个有效PPO panel中，没有任何 checkpoint 让 relaxed 或 primary 判据归零。因此证据支持“原问题身份可以被修复”，不支持“失败机制已被消灭”。""",
    },
    {"id": "headline_cards", "type": "metric-strip", "layout": "full", "cardIds": ["total_collision_card", "tail_card", "coverage_card"]},
    {
        "id": "key_findings", "type": "markdown", "layout": "full", "sourceId": "analysis_summary",
        "body": """## 四个决定性发现

1. **总碰撞最优与目标机制最优不是同一个 checkpoint。** clip .20 U30/U40 各11次总碰撞；hard U35 的主判据最低为2，但总碰撞20。
2. **尾扫机制会迁移到新 scenario。** clip .20 U30 对BC尾扫是10 resolved / 1 shared / 3 created；hard U45是11 / 0 / 4。
3. **训练继续并未形成稳定单调收敛。** clip .20 的U25+总碰撞路径为17/11/14/11/12，相邻 checkpoint 的碰撞身份仍大量翻转。
4. **两个失败族必须分开。** 两个 v0.5 场景在86/86个唯一PPO policy都碰撞，但不是尾扫类；另有 sp5-r0-v0.7 在69/86个 policy出现尾扫，是最持续的目标机制场景。""",
    },
    {"id": "selected_eval_block", "type": "table", "layout": "full", "tableId": "selected_eval_table"},
    {"id": "current_total_chart_block", "type": "chart", "layout": "full", "chartId": "current_total_path"},
    {"id": "current_tail_chart_block", "type": "chart", "layout": "full", "chartId": "current_tail_path"},
    {"id": "mix_chart_block", "type": "chart", "layout": "full", "chartId": "collision_mix_chart"},
    {
        "id": "model_spec", "type": "markdown", "layout": "full", "sourceId": "model_contract",
        "body": """## 模型参数：PPO 只微调 actor GRU/head，P20 只进入 critic

Actor 输入361D、输出2D，11,301,482个参数中11,300,942个可训练；`k`、speed MLP、dummy embedding 共540个参数固定。P20 critic 输入381D，`privilege_gru` 有11,309,401个参数：它使用独立的BC初始化GRU，并把20维privileged特征通过后投影融合，actor在eval时不依赖P20。

选定U30 actor相对BC的参数relative-L2为0.000801，U45 clip .20为0.000969；固定前端变化严格为0。变化数值虽小，但相同actor字节哈希的重复eval结果完全一致，而不同checkpoint的scenario结果会改变，说明不是汇总噪声。""",
    },
    {"id": "model_contract_block", "type": "table", "layout": "full", "tableId": "model_contract_table"},
    {"id": "actor_delta_chart_block", "type": "chart", "layout": "full", "chartId": "actor_delta_chart"},
    {"id": "actor_delta_table_block", "type": "table", "layout": "full", "tableId": "actor_delta_table"},
    {
        "id": "training_metrics", "type": "markdown", "layout": "full", "sourceId": "group_summary",
        "body": """## 训练 metrics：优化过程有效，但不能替代固定面板 eval

16个run共415个formal update，metrics数值有限、序列连续、actor/critic checkpoint完整。早期严格 critic 对照中，U20总碰撞为 independent GRU 34、privileged MLP 25、`privilege_gru` 14，支持P20共享GRU critic。batch 12,800优于25,600/51,200的后期结果；LR 3x优于1x/5x。target-KL .04在12/20个update早停却出现45/70/58/33等高碰撞点，说明当前门控不是安全保证。

训练rollout collision rate、return、value loss、explained variance、mean KL与clip fraction，对86个唯一policy的eval碰撞只有弱到中等的pooled描述相关；最高Spearman为clip fraction的0.336。训练metrics适合发现过弱、过激、早停和critic拟合问题，最终安全排序仍必须用逐scenario eval。""",
    },
    {"id": "run_table_block", "type": "table", "layout": "full", "tableId": "run_table"},
    {"id": "correlation_table_block", "type": "table", "layout": "full", "tableId": "correlation_table"},
    {
        "id": "experiment_comparisons", "type": "markdown", "layout": "full", "sourceId": "group_summary",
        "body": """## 全部实验对照：clip .20 保住总碰撞，hard 只改善目标机制均值

当前45U严格对照的U25+均值为：clip .20总碰撞13.0、主判据6.6；clip .25为16.8/5.6；hard为21.4/5.6。也就是说，.25与hard都略降尾扫均值，却以更多其他碰撞为代价。最终U45为clip .20 12/7、clip .25 16/7、hard 17/4（总碰撞/主判据）。

历史Group 3与target-KL .02比较存在env workers混杂，只能描述；clip .20 vs .25与baseline vs hard均为严格单轴。下表保留所有路径及相邻checkpoint身份翻转，避免只看最终点。""",
    },
    {"id": "group_table_block", "type": "table", "layout": "full", "tableId": "group_table"},
    {"id": "control_table_block", "type": "table", "layout": "full", "tableId": "control_table"},
    {
        "id": "all_eval", "type": "markdown", "layout": "full", "sourceId": "all_eval_panels",
        "body": """## 全部 eval：92个有效面板，1个599-row面板隔离

共发现BC 1个、PPO 92个panel；其中91个PPO panel有效。`ppo_privilege_gru_0722_lr5_tkloff_u0020` 只有599个episode/trace并记录1个error，因此保留在完整清单中，但不参加最佳checkpoint排序、配对统计或跨policy频率。91个有效PPO panel包含5对byte-identical actor重复eval；结果也完全一致，所以共性统计使用86个唯一actor policy。""",
    },
    {"id": "all_panels_block", "type": "table", "layout": "full", "tableId": "all_panels_table"},
    {
        "id": "mechanism_definition", "type": "markdown", "layout": "full", "sourceId": "bc_tail_episodes",
        "body": """## 目标机制定义：证据证明后部扫碰几何，不证明真实轮胎侧偏因果

主判据要求：(1) ego相对赛道进度从不领先穿越到领先，且至少0.10秒后才碰撞；(2) opponent也碰撞；(3)终止前opponent位于ego车体坐标后方；(4)超车后双方赛道横向间距至少收敛0.10米。BC的11个事件中，超车到碰撞中位0.39秒（0.19–1.17），横向收敛中位0.313米（0.200–0.563）；7个在raceline0、4个在raceline2，速度只出现在0.7/0.8。

保存轨迹没有车辆动力学内部tire slip angle。pose-derived绝对slip proxy在BC目标事件的中位数为0.081 rad，其他BC碰撞为0.071 rad，没有形成清晰区分。因此可严谨回答“超车后并线导致车尾与opp接触是否仍存在”，不能把数据升级成“已经证明轮胎甩尾物理原因”。""",
    },
    {"id": "bc_tail_block", "type": "table", "layout": "full", "tableId": "bc_tail_table"},
    {
        "id": "episode_transitions", "type": "markdown", "layout": "full", "sourceId": "paired_bc",
        "body": """## Episode 配对：原问题可被修复，但同类问题会在新场景再生

clip .20 U30相对BC总碰撞17 resolved / 5 shared / 6 created，净减少11；目标机制10 / 1 / 3，净减少7。hard U35把原11个目标机制场景全部变为超车，却新增2个目标机制事件；其总碰撞是18 resolved / 4 shared / 16 created，只净减少2。hard U45也让原11个全部超车，但新增4个目标机制事件。

这正是“解决原episode”和“解决机制”的差别：前者对hard checkpoint成立，后者对任何有效checkpoint都不成立。""",
    },
    {"id": "transition_table_block", "type": "table", "layout": "full", "tableId": "transition_table"},
    {"id": "paired_table_block", "type": "table", "layout": "full", "tableId": "paired_table"},
    {
        "id": "commonality", "type": "markdown", "layout": "full", "sourceId": "top_scenarios",
        "body": """## 碰撞共性：两个必碰场景与两个高频尾扫场景

去重后的86个PPO policy共有1,785个碰撞episode：778个主判据尾扫、76个超车后后部接触但未达并线阈值、784个超车前/前侧opponent接触、147个ego-only/墙类。碰撞时间10%/中位/90%分位为2.35/4.65/7.22秒。

`sp17-ego727-r2-v0.5` 与 `sp35-ego1497-r1-v0.5` 在86/86个policy都碰撞，但0次被判为尾扫；它们是另一类固定困难样本。目标机制最持续的是 `sp5-ego213-r0-v0.7`（73/86碰撞、69/86尾扫）和 `sp15-ego641-r2-v0.7`（60/86碰撞、58/86尾扫）。""",
    },
    {"id": "persistent_chart_block", "type": "chart", "layout": "full", "chartId": "persistent_chart"},
    {"id": "top_scenario_table_block", "type": "table", "layout": "full", "tableId": "top_scenario_table"},
    {
        "id": "statistical_evidence", "type": "markdown", "layout": "full", "sourceId": "paired_bc",
        "body": """## 统计证据：固定面板内有改善信号，但不足以宣布泛化解决

clip .20 U30的总碰撞配对exact p=0.0347，startpoint-cluster bootstrap净减少95% CI为[2,21]；目标机制p=0.0923、CI为[0,14]。hard U35的目标机制p=0.0225、CI为[2,16]，但总碰撞p=0.864。以上是同一Austin600固定面板上的配对描述。

由于观察了91个PPO panel并从中挑checkpoint，所有panel与BC比较的Holm校正后结果均未通过。再加上训练run没有保存seed/source commit、没有独立重复训练，这些p值不能解释为跨seed或新场景泛化概率。""",
    },
    {
        "id": "data_quality", "type": "markdown", "layout": "full", "sourceId": "quality",
        "body": """## 数据质量：原始结果完整，旧NPZ终止语义有已知边界

独立验证从93个原始`results_multi.json`重新计算所有结果，逐panel与汇总完全一致；所有有效PPO↔BC set transition也独立重算一致。55,799个NPZ全部numeric且数组对齐。34,799个post-step-v2轨迹终止标记完整并与JSON一致；21,000个legacy轨迹漏terminal post-step，其中736个碰撞轨迹的最后marker因此与JSON不一致。这是可解释的格式边界，旧格式碰撞真值一律取JSON。""",
    },
    {"id": "quality_table_block", "type": "table", "layout": "full", "tableId": "quality_table"},
    {
        "id": "scope", "type": "markdown", "layout": "full",
        "body": """## 范围、数据与指标口径

- **训练来源**：`post-trained/*/run_config.json`、`metrics.jsonl`、actor/critic checkpoints。
- **评估来源**：`eval_results/*/multiagents/results_multi.json` 与所有NPZ traces。
- **面板**：固定Austin600；每个有效panel包含同一600个scenario ID。
- **主指标**：ego collision episode数；overtake/follow与collision互斥并覆盖有效panel。
- **目标机制**：超车领先时间、opponent后方接触和赛道横向收敛的几何复合判据；relaxed/primary/strict用于阈值敏感性。
- **共性统计**：相同actor SHA重复eval先去重，分母为86个唯一PPO actor。
- **训练与eval边界**：训练rollout样本分布与Austin600不同，训练collision rate不直接替代eval。""",
    },
    {
        "id": "methodology", "type": "markdown", "layout": "full",
        "body": """## 方法与复现路径

1. 从源代码、run config和checkpoint张量重建actor/critic结构、固定/可训练参数与BC-relative变化。
2. 解析全部formal `metrics.jsonl`，核验update连续性、有限值、optimizer steps、KL/clip、value loss/EV和checkpoint完整性。
3. 逐个原始eval JSON重算episode结果与scenario集合，并将唯一缺失episode的panel隔离。
4. 对所有碰撞NPZ进行赛道弧长与横向投影，计算超车穿越、碰撞间隔、opponent车体相对位置和横向收敛；结果JSON控制碰撞真值。
5. 同scenario ID做BC↔PPO resolved/shared/created配对，使用exact binomial检验与50-startpoint cluster bootstrap；同时报告全91-panel Holm校正。
6. 相同actor SHA去重后计算跨policy碰撞共性；relaxed/primary/strict三套阈值做稳健性检查。

全量入口：`analyze_all.py`；独立QA：`validate_analysis.py`；执行型伴随文档：`ppo_all_experiments_tail_analysis.ipynb`。""",
    },
    {
        "id": "limitations", "type": "markdown", "layout": "full", "sourceId": "quality",
        "body": """## 局限、不确定性与稳健性

- **单训练实现、无持久化seed/source commit**：历史run provenance不完整，不能从当前dirty worktree回填。
- **checkpoint选择与多重比较**：U30是观察多个点后的最优总碰撞点；raw p值必须保留选择偏差。
- **固定Austin600**：配对能力强，但不直接证明跨地图、跨场景分布泛化。
- **真实slip state缺失**：只能用pose-derived proxy验证几何，不可声称已证明轮胎侧偏因果。
- **阈值依赖**：最低relaxed/primary/strict分别为3/2/0；主结论依赖“不以strict=0单独宣布解决”，并由relaxed与primary仍非零支持。
- **策略后期身份漂移**：相邻checkpoint总数相近时，碰撞scenario集合仍大量变化，尚无稳定收敛证据。""",
    },
    {
        "id": "recommendations", "type": "markdown", "layout": "full", "sourceId": "analysis_summary",
        "body": """## 建议：冻结U30候选，下一步只针对机制做最小受控修复

1. 当前候选冻结为clip .20 U30：它是已观察数据中总碰撞与目标机制的最佳折中，不把hard-neighbor当最终模型。
2. 把四个场景作为最小诊断集：两个86/86非尾扫必碰场景，以及sp5/sp15两个高频尾扫场景；不要用总碰撞数替代逐episode检查。
3. 下一次eval额外保存真实tire slip angle、接触对象/接触点和完整post-step状态，以区分控制侧偏与纯几何并线接触。
4. 若实现尾部安全/并线处理，保留当前baseline、reward、超参、schedule和Austin600，只加一个显式开关做单轴A/B；预先固定读取checkpoint，仍报告resolved/shared/created。
5. 暂不扩展成多seed或新面板大工程；先确认目标机制能在relaxed/primary口径下接近归零，且不把总碰撞转移到其他类别。""",
    },
    {
        "id": "further_questions", "type": "markdown", "layout": "full",
        "body": """## 进一步问题

1. sp5/sp15的动作在超车后多久开始向raceline收敛，是否存在可预警的rear-clearance阈值？
2. 两个86/86必碰场景属于几何不可达、BC感知别名，还是奖励/终止策略缺陷？
3. hard-neighbor为何能清除原11个尾扫身份，却在后期把总碰撞推高到21.4均值？
4. U30→U45期间碰撞身份持续翻转，主要来自actor head、GRU，还是特定训练scenario簇的梯度主导？""",
    },
]


generated_at = datetime.now(timezone.utc).isoformat()
artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "End2Race PPO 全实验：训练、Eval 与超车后尾扫碰撞分析",
        "description": "基于16个训练run、93个eval panel和55,799个episode/NPZ的模型、训练、评估与碰撞机制技术报告。",
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    },
    "snapshot": {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "headline": records(headline),
            "selected_eval": records(selected_eval),
            "selected_pair": records(selected_pair),
            "current_paths": records(current_paths),
            "collision_mix": records(mix),
            "top_scenarios": records(top_frequency),
            "actor_delta_paths": records(delta_paths),
            "selected_delta": records(selected_delta),
            "model_contract": records(model_contract),
            "training_runs": records(run_table),
            "group_summary": records(group_table),
            "control_audit": records(control_table),
            "all_eval_panels": records(all_panels),
            "bc_tail_episodes": records(bc_episode_table),
            "bc_tail_transitions": records(tail_transitions),
            "correlations": records(correlation_table),
            "quality": records(quality),
        },
        "accessIssues": [],
    },
    "sources": sources,
}

(OUT / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

notes = """# Report notes

## Source inventory

- Raw model/training: `train_ppo.py`, `ppo/*.py`, `post-trained/*/run_config.json`, `metrics.jsonl`, checkpoints.
- Raw evaluation: 93 `results_multi.json` files and 55,799 NPZ traces under `eval_results/`.
- Core outputs: `eval_panels.csv`, `eval_episode_outcomes.csv`, `collision_episode_kinematics.csv`,
  `paired_vs_bc.csv`, `group_summary.csv`, `actor_parameter_deltas.csv`, `npz_audit.json`.
- Reproducibility: `analyze_all.py`, `validate_analysis.py`, and
  `ppo_all_experiments_tail_analysis.ipynb`.

## Chart map

| Chart | Question | Form | Reason |
|---|---|---|---|
| Current total path | Which current arm controls total collisions over training? | line | 10 ordered checkpoints across 45 updates |
| Current tail path | Does the target mechanism approach zero? | line | same ordered checkpoints and denominator |
| Collision mix | Which failure modes replace one another? | grouped bar | exact discrete class counts per checkpoint |
| Actor delta | Did policy tensors move and stay within the controlled frontend boundary? | line | ordered BC-relative parameter movement |
| Persistent scenarios | Which scenario identities fail across policies? | ranked bar | common denominator of 86 unique actor policies |

## Evidence boundary

- The report answers fixed-panel descriptive and paired questions, not cross-seed or cross-map generalization.
- Primary/relaxed/strict sensitivity is visible; no valid checkpoint reaches zero on relaxed or primary.
- Pose-derived slip proxy is not simulator tire slip angle.
- The isolated 599-row eval remains visible but is excluded from inference.
"""
(OUT / "REPORT_NOTES.md").write_text(notes)
print(f"wrote {OUT / 'artifact.json'}")
print(f"wrote {OUT / 'REPORT_NOTES.md'}")

#!/usr/bin/env python3
"""Build a portable, source-backed report for all PPO experiments through G9."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def sql(path: str, where: str = "") -> str:
    suffix = Path(path).suffix
    reader = "read_json_auto" if suffix == ".json" else "read_csv_auto"
    header = "" if suffix == ".json" else ", header = true"
    return f"SELECT * FROM {reader}('{path}'{header}){where};"


summary = json.loads((OUT / "analysis_summary.json").read_text())
configs = pd.read_csv(OUT / "run_config_matrix.csv")
train_summary = pd.read_csv(OUT / "training_run_summary.csv")
panels = pd.read_csv(OUT / "eval_panels.csv")
tail = pd.read_csv(OUT / "tail_issue_by_panel.csv")
frequency = pd.read_csv(OUT / "collision_scenario_frequency.csv")
paired = pd.read_csv(OUT / "selected_paired_comparisons.csv")
logical = pd.read_csv(OUT / "logical_clip020_45u_path.csv")
source_updates = pd.read_csv(OUT / "training_collision_source_by_update.csv")
repro = pd.read_csv(OUT / "checkpoint_reproducibility.csv")

BC = summary["current_snapshot"]["BC"]


def valid_run_path(run: str) -> pd.DataFrame:
    if run == "ppo_privilege_gru_0722_long45_clip020":
        prefix = panels[
            (panels.run == "ppo_privilege_gru_0722_long_clip020")
            & (panels["update"] <= 30)
            & panels.summary_valid
        ]
        suffix = panels[(panels.run == run) & panels.summary_valid]
        return pd.concat([prefix, suffix], ignore_index=True).sort_values("update")
    return panels[(panels.run == run) & panels.summary_valid].sort_values("update")


path_rows: list[dict] = []
for row in configs.itertuples(index=False):
    frame = valid_run_path(row.run)
    if frame.empty:
        continue
    collision_path = "/".join(f"U{int(x.update)}={int(x.collision_count)}" for x in frame.itertuples())
    tail_path = "/".join(f"U{int(x.update)}={int(x.high_sideslip_tail_5deg_count)}" for x in frame.itertuples())
    best = frame.sort_values(["collision_count", "update"]).iloc[0]
    final = frame.iloc[-1]
    path_rows.append(
        {
            "group": row.group,
            "arm": row.arm,
            "run": row.run,
            "workers": int(row.env_workers),
            "clip": float(row.clip_range),
            "target_kl": None if pd.isna(row.target_kl) else float(row.target_kl),
            "hard": bool(row.hard_neighbors),
            "collision_path": collision_path,
            "tail5_path": tail_path,
            "best_update": int(best["update"]),
            "best_collisions": int(best.collision_count),
            "final_update": int(final["update"]),
            "final_collisions": int(final.collision_count),
            "final_tail5": int(final.high_sideslip_tail_5deg_count),
            "final_overtakes": int(final.overtake_count),
            "final_success_rate": float(final.success_rate),
        }
    )
arm_paths = pd.DataFrame(path_rows).sort_values(["group", "arm"])


run_labels = {
    "BC": "BC",
    "ppo_privilege_gru_0722_long_clip020": "G5 clip .20 U30",
    "ppo_privilege_gru_0722_long45_clip020": "G7 clip .20 U40",
    "ppo_privilege_gru_0722_long45_clip025": "G8 clip .25 U45",
    "ppo_privilege_gru_0722_long45_clip020_hard": "G9 hard U45",
}
selected_keys = [
    ("BC", 0),
    ("ppo_privilege_gru_0722_long_clip020", 30),
    ("ppo_privilege_gru_0722_long45_clip020", 40),
    ("ppo_privilege_gru_0722_long45_clip025", 45),
    ("ppo_privilege_gru_0722_long45_clip020_hard", 45),
]
selected_rows = []
for run, update in selected_keys:
    row = panels[(panels.run == run) & (panels["update"] == update)].iloc[0]
    selected_rows.append(
        {
            "checkpoint": run_labels[run],
            "collisions": int(row.collision_count),
            "success_rate": float(row.success_rate),
            "overtakes": int(row.overtake_count),
            "follows": int(row.follow_count),
            "vehicle_collisions": int(row.vehicle_collision_count),
            "wall_collisions": int(row.wall_only_collision_count),
            "structural_tail": int(row.post_overtake_rear_contact_count),
            "strict_tail_3deg": int(row.high_sideslip_tail_3deg_count),
            "strict_tail_5deg": int(row.high_sideslip_tail_5deg_count),
            "strict_tail_8deg": int(row.high_sideslip_tail_8deg_count),
            "bc_collisions_resolved": int(row.bc_collisions_resolved),
            "bc_collisions_persistent": int(row.bc_collisions_persistent),
            "new_collisions": int(row.new_collisions_created),
            "bc_strict_tail_resolved": int(row.bc_strict_5deg_tail_scenarios_resolved),
            "new_strict_tail": int(row.new_strict_5deg_tail_scenarios_vs_bc),
            "min_surface_dist_m": float(row.mean_global_min_surface_dist_m),
            "mean_speed_mps": float(row.mean_avg_speed_mps),
        }
    )
selected = pd.DataFrame(selected_rows)

candidate_runs = {
    "ppo_privilege_gru_0722_long45_clip020": "clip .20",
    "ppo_privilege_gru_0722_long45_clip025": "clip .25",
    "ppo_privilege_gru_0722_long45_clip020_hard": "hard-neighbor",
}
candidate_path_rows = []
for run, label in candidate_runs.items():
    frame = valid_run_path(run)
    for row in frame.itertuples():
        candidate_path_rows.append(
            {
                "arm": label,
                "update": int(row.update),
                "checkpoint": f"U{int(row.update)}",
                "collisions": int(row.collision_count),
                "strict_tail_5deg": int(row.high_sideslip_tail_5deg_count),
                "overtakes": int(row.overtake_count),
            }
        )
candidate_paths = pd.DataFrame(candidate_path_rows)

new_train = train_summary[train_summary.group.isin(["G7", "G8", "G9"])].copy()
new_train = new_train[
    [
        "group",
        "arm",
        "mean_rollout_collision_rate",
        "final_rollout_collision_rate",
        "mean_episode_return",
        "final_episode_return",
        "median_approx_kl_mean",
        "max_approx_kl_max",
        "updates_approx_kl_mean_gt_0p05",
        "updates_approx_kl_max_gt_0p5",
        "mean_clip_fraction",
        "median_actor_grad_norm",
        "max_actor_grad_norm",
        "final_explained_variance",
    ]
]

hard_source = source_updates[source_updates.group.eq("G9")]
hard_source_summary = []
for label, count_col, rate_col in [
    ("base collision cache", "base_collision_source_episodes", "base_source_realized_collision_rate"),
    ("boundary extension", "boundary_collision_source_episodes", "boundary_source_realized_collision_rate"),
    ("ordinary pool", "ordinary_role_episodes", "ordinary_role_realized_collision_rate"),
]:
    count = int(hard_source[count_col].sum())
    realized = float((hard_source[count_col] * hard_source[rate_col]).sum() / count)
    hard_source_summary.append({"source": label, "episodes": count, "realized_collision_rate": realized})

repro_summary = (
    repro.groupby("comparison")
    .agg(
        updates=("update", "count"),
        exact_actor_updates=("actor_tensors_exact", "sum"),
        exact_metric_updates=("non_walltime_metrics_exact", "sum"),
        max_parameter_difference=("max_abs_actor_parameter_difference", "max"),
    )
    .reset_index()
)

top_scenarios = frequency.sort_values(
    ["ppo_collision_panel_count", "scenario_id"], ascending=[False, True]
).head(12).copy()
top_scenarios["scenario"] = top_scenarios.scenario_id.str.replace("evaluation-", "", regex=False)
top_scenarios = top_scenarios[
    [
        "scenario",
        "bc_collision",
        "ppo_collision_panel_count",
        "ppo_observed_panel_count",
        "ppo_collision_panel_rate",
        "ppo_final_collision_count",
        "ppo_structural_tail_panel_count",
        "ppo_strict_5deg_tail_panel_count",
    ]
]

strict_bc = pd.read_csv(OUT / "bc_tail_scenario_outcomes.csv")
bc_ids = strict_bc[(strict_bc.run == "BC") & (strict_bc["update"] == 0)].scenario_id.tolist()
strict_outcome_rows = []
for scenario_id in bc_ids:
    row = {"scenario": scenario_id.replace("evaluation-", "")}
    for run, update, label in [
        ("ppo_privilege_gru_0722_long_clip020", 30, "G5_U30"),
        ("ppo_privilege_gru_0722_long45_clip020", 40, "G7_U40"),
        ("ppo_privilege_gru_0722_long45_clip020_hard", 45, "G9_U45"),
    ]:
        q = strict_bc[(strict_bc.scenario_id == scenario_id) & (strict_bc.run == run) & (strict_bc["update"] == update)].iloc[0]
        if bool(q.high_sideslip_tail_5deg):
            status = "strict tail collision"
        elif bool(q.ego_collision):
            status = "other collision"
        else:
            status = str(q.outcome)
        row[label] = status
    strict_outcome_rows.append(row)
strict_outcomes = pd.DataFrame(strict_outcome_rows)

quality = pd.DataFrame(
    [
        ["训练 run", "16/16 完成", "415 个 formal update；每 run checkpoint 数与配置一致"],
        ["eval 面板", "92/93 完整有效", "高 actor-LR U20 缺 1 episode/trace，完整面板结论排除该点"],
        ["场景 ID", "600/panel", "所有有效面板可与 BC 逐 ID 配对"],
        ["唯一物理初态", "592/panel", "闭环端点 2096 与 0 重合；跨 raceline 组合中有 8 组物理重复"],
        ["NPZ 碰撞 trace", "1913/1913 可读取", "相对 results_multi.json 没有碰撞 trace 缺失"],
        ["旧版 0721 terminal", "终止前约 0.01s", "碰撞真值/时刻使用 JSON；NPZ 只做碰撞前运动分析"],
        ["G7 复现", "U1-U30 逐 tensor 相同", "与 G5 clip .20 的 30 个 actor checkpoint 和非 wall-time metrics 完全一致"],
    ],
    columns=["check", "result", "interpretation"],
)

headline = pd.DataFrame(
    [
        {
            "bc_collisions": 22,
            "best_collisions": 11,
            "collision_reduction": 0.5,
            "best_success": 589 / 600,
            "bc_strict_tail": 8,
            "hard_u45_strict_tail": 1,
            "valid_panels": 92,
            "episode_pairs": 55200,
        }
    ]
)

base = "analysis_results/ppo_all_experiments_0723"
sources = [
    {
        "id": "summary",
        "label": "Consolidated all-experiment audit",
        "path": f"{base}/analysis_summary.json",
        "query": {
            "language": "sql",
            "engine": "duckdb",
            "sql": sql(f"{base}/analysis_summary.json"),
            "description": "Audited counts and selected checkpoints derived from configs, training metrics, evaluation JSON, and NPZ traces.",
            "metric_definitions": [
                "collision count = eval episodes with ego_collision=true",
                "success rate = (600 - collision count) / 600",
                "fixed-panel paired transitions compare identical scenario IDs against BC",
            ],
        },
    },
    {
        "id": "configs",
        "label": "Recorded experiment configuration matrix",
        "path": f"{base}/run_config_matrix.csv",
        "query": {"language": "sql", "engine": "duckdb", "sql": sql(f"{base}/run_config_matrix.csv"), "description": "One row per PPO run, using its recorded run configuration and checkpoint tensors."},
    },
    {
        "id": "training",
        "label": "Training metrics summary",
        "path": f"{base}/training_run_summary.csv",
        "query": {"language": "sql", "engine": "duckdb", "sql": sql(f"{base}/training_run_summary.csv"), "description": "Aggregates formal-update metrics.jsonl without interpreting rollout k as checkpoint k performance."},
    },
    {
        "id": "panels",
        "label": "Evaluation panel and trace metrics",
        "path": f"{base}/eval_panels.csv",
        "query": {"language": "sql", "engine": "duckdb", "sql": sql(f"{base}/eval_panels.csv"), "description": "One row per BC/PPO eval panel, reconciled with episode rows, aggregate results, scenario identity, and collision traces."},
    },
    {
        "id": "episodes",
        "label": "Every PPO eval episode paired to BC",
        "path": f"{base}/eval_episode_comparison.csv",
        "query": {"language": "sql", "engine": "duckdb", "sql": sql(f"{base}/eval_episode_comparison.csv"), "description": "55,200 PPO-vs-BC scenario-paired rows plus 600 BC self-reference rows, with collision transitions and KPI deltas."},
    },
    {
        "id": "tail",
        "label": "Collision trace mechanism features",
        "path": f"{base}/collision_episode_features.csv",
        "query": {"language": "sql", "engine": "duckdb", "sql": sql(f"{base}/collision_episode_features.csv"), "description": "Trace-derived progress, ego-frame geometry, lateral closure, yaw-rate, and pose-derived sideslip for 1,913 collision episodes."},
    },
    {
        "id": "frequency",
        "label": "Collision scenario recurrence across all PPO panels",
        "path": f"{base}/collision_scenario_frequency.csv",
        "query": {"language": "sql", "engine": "duckdb", "sql": sql(f"{base}/collision_scenario_frequency.csv", " ORDER BY ppo_collision_panel_count DESC"), "description": "Per-scenario collision and tail-mechanism frequency over 92 PPO panels and 16 final checkpoints."},
    },
    {
        "id": "paired",
        "label": "Selected exact episode-paired comparisons",
        "path": f"{base}/selected_paired_comparisons.csv",
        "query": {"language": "sql", "engine": "duckdb", "sql": sql(f"{base}/selected_paired_comparisons.csv"), "description": "Shared, resolved, and created collision cases on the same scenario IDs; p-values are unadjusted descriptive McNemar exact tests."},
    },
]

cards = [
    {"id": "safety", "description": "Best observed fixed-panel safety result.", "dataset": "headline", "sourceId": "summary", "metrics": [{"label": "BC → best collisions", "field": "best_collisions", "format": "number"}, {"label": "Collision reduction", "field": "collision_reduction", "format": "percent"}]},
    {"id": "tail_card", "description": "Strict 5-degree tail-swing proxy.", "dataset": "headline", "sourceId": "tail", "metrics": [{"label": "BC strict tail", "field": "bc_strict_tail", "format": "number"}, {"label": "Hard U45 strict tail", "field": "hard_u45_strict_tail", "format": "number"}]},
    {"id": "evidence", "description": "Audited evidence volume.", "dataset": "headline", "sourceId": "summary", "metrics": [{"label": "Valid eval panels", "field": "valid_panels", "format": "number"}, {"label": "BC-paired episodes", "field": "episode_pairs", "format": "number"}]},
]

charts = [
    {
        "id": "candidate_paths",
        "title": "45-update candidates: collision path",
        "subtitle": "G7 U1-U30 reuses the tensor-identical G5 clip .20 path; BC reference is 22",
        "type": "bar",
        "dataset": "candidate_paths",
        "sourceId": "panels",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "checkpoint", "type": "nominal", "label": "Checkpoint"},
            "y": {"field": "collisions", "type": "quantitative", "label": "Collisions / 600"},
            "color": {"field": "arm", "type": "nominal", "label": "Arm"},
            "tooltip": [{"field": "strict_tail_5deg", "type": "quantitative", "label": "Strict tail"}, {"field": "overtakes", "type": "quantitative", "label": "Overtakes"}],
        },
    },
    {
        "id": "tail_compare",
        "title": "Total collision and strict tail-swing counts",
        "subtitle": "Strict = post-overtake rear merge contact + pose-derived sideslip >= 5 degrees",
        "type": "bar",
        "dataset": "selected",
        "sourceId": "tail",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "checkpoint", "type": "nominal", "label": "Checkpoint"},
            "y": {"field": "strict_tail_5deg", "type": "quantitative", "label": "Strict tail episodes"},
            "color": {"field": "checkpoint", "type": "nominal", "label": "Checkpoint"},
            "tooltip": [{"field": "collisions", "type": "quantitative", "label": "All collisions"}, {"field": "structural_tail", "type": "quantitative", "label": "Structural tail"}, {"field": "strict_tail_3deg", "type": "quantitative", "label": "3 degree"}, {"field": "strict_tail_8deg", "type": "quantitative", "label": "8 degree"}],
        },
    },
    {
        "id": "persistent",
        "title": "Most persistent collision scenarios",
        "subtitle": "Fraction of 92 PPO panels; one panel is partial, so per-scenario observed denominators are retained",
        "type": "bar",
        "dataset": "top_scenarios",
        "sourceId": "frequency",
        "valueFormat": "percent",
        "encodings": {
            "x": {"field": "scenario", "type": "nominal", "label": "Scenario"},
            "y": {"field": "ppo_collision_panel_rate", "type": "quantitative", "label": "PPO panel collision rate"},
            "tooltip": [{"field": "ppo_collision_panel_count", "type": "quantitative", "label": "Collision panels"}, {"field": "ppo_strict_5deg_tail_panel_count", "type": "quantitative", "label": "Strict tail panels"}],
        },
    },
]

tables = [
    {"id": "paths", "title": "All experiment arms", "subtitle": "G7 displays its tensor-identical U1-U30 history plus U35/U40/U45 extension", "dataset": "arm_paths", "sourceId": "panels", "columns": [{"field": "group", "label": "Group"}, {"field": "arm", "label": "Arm"}, {"field": "workers", "label": "Workers", "format": "number"}, {"field": "collision_path", "label": "Collision path"}, {"field": "tail5_path", "label": "Strict-tail path"}, {"field": "best_collisions", "label": "Best", "format": "number"}, {"field": "final_collisions", "label": "Final", "format": "number"}, {"field": "final_overtakes", "label": "Final overtakes", "format": "number"}]},
    {"id": "selected_table", "title": "Representative checkpoint KPIs", "subtitle": "All values recomputed from 600 episode rows and collision traces", "dataset": "selected", "sourceId": "panels", "columns": [{"field": "checkpoint", "label": "Checkpoint"}, {"field": "collisions", "label": "Collisions", "format": "number"}, {"field": "success_rate", "label": "Success", "format": "percent"}, {"field": "overtakes", "label": "Overtakes", "format": "number"}, {"field": "structural_tail", "label": "Structural tail", "format": "number"}, {"field": "strict_tail_5deg", "label": "Strict tail", "format": "number"}, {"field": "bc_collisions_resolved", "label": "BC resolved", "format": "number"}, {"field": "new_collisions", "label": "New vs BC", "format": "number"}, {"field": "min_surface_dist_m", "label": "Mean min-surface (m)", "format": "number"}]},
    {"id": "training_table", "title": "G7-G9 training telemetry", "subtitle": "Rollout metrics diagnose optimization but do not rank fixed-panel safety", "dataset": "new_train", "sourceId": "training", "columns": [{"field": "group", "label": "Group"}, {"field": "arm", "label": "Arm"}, {"field": "mean_rollout_collision_rate", "label": "Mean rollout collision", "format": "percent"}, {"field": "final_rollout_collision_rate", "label": "Final rollout collision", "format": "percent"}, {"field": "final_episode_return", "label": "Final return", "format": "number"}, {"field": "median_approx_kl_mean", "label": "Median mean-KL", "format": "number"}, {"field": "max_approx_kl_max", "label": "Max minibatch KL", "format": "number"}, {"field": "mean_clip_fraction", "label": "Clip fraction", "format": "percent"}, {"field": "max_actor_grad_norm", "label": "Max pre-clip grad", "format": "number"}, {"field": "final_explained_variance", "label": "Final EV", "format": "number"}]},
    {"id": "paired_table", "title": "Exact scenario-paired collision transitions", "subtitle": "Unadjusted exact p is descriptive; all-panel BH correction removes conventional significance", "dataset": "paired", "sourceId": "paired", "columns": [{"field": "comparison", "label": "Comparison"}, {"field": "left_collisions", "label": "Left", "format": "number"}, {"field": "right_collisions", "label": "Right", "format": "number"}, {"field": "shared_collisions", "label": "Shared", "format": "number"}, {"field": "resolved_left_collisions", "label": "Resolved", "format": "number"}, {"field": "created_right_collisions", "label": "Created", "format": "number"}, {"field": "exact_paired_p_unadjusted", "label": "Exact p", "format": "number"}]},
    {"id": "hard_source", "title": "Hard-neighbor training pool realized sampling", "subtitle": "45 updates; source tags from episodes.jsonl", "dataset": "hard_source", "sourceId": "training", "columns": [{"field": "source", "label": "Source"}, {"field": "episodes", "label": "Episodes", "format": "number"}, {"field": "realized_collision_rate", "label": "Realized collision rate", "format": "percent"}]},
    {"id": "strict_outcomes", "title": "Eight BC strict tail-swing scenarios", "subtitle": "Per-scenario outcome at representative checkpoints", "dataset": "strict_outcomes", "sourceId": "tail", "columns": [{"field": "scenario", "label": "Scenario"}, {"field": "G5_U30", "label": "G5 U30"}, {"field": "G7_U40", "label": "G7 U40"}, {"field": "G9_U45", "label": "G9 U45"}]},
    {"id": "top_table", "title": "Collision episode commonality", "subtitle": "Recurrence over every valid PPO checkpoint and final-run snapshots", "dataset": "top_scenarios", "sourceId": "frequency", "columns": [{"field": "scenario", "label": "Scenario"}, {"field": "bc_collision", "label": "BC collision"}, {"field": "ppo_collision_panel_count", "label": "PPO panels", "format": "number"}, {"field": "ppo_collision_panel_rate", "label": "Panel rate", "format": "percent"}, {"field": "ppo_final_collision_count", "label": "Final runs", "format": "number"}, {"field": "ppo_structural_tail_panel_count", "label": "Structural tail", "format": "number"}, {"field": "ppo_strict_5deg_tail_panel_count", "label": "Strict tail", "format": "number"}]},
    {"id": "quality", "title": "Data quality and reproducibility", "subtitle": "Claims below respect these boundaries", "dataset": "quality", "sourceId": "summary", "columns": [{"field": "check", "label": "Check"}, {"field": "result", "label": "Result"}, {"field": "interpretation", "label": "Interpretation"}]},
    {"id": "repro", "title": "Checkpoint reproducibility", "subtitle": "Exact tensor comparison; wall-time fields excluded from metrics equality", "dataset": "repro_summary", "sourceId": "summary", "columns": [{"field": "comparison", "label": "Comparison"}, {"field": "updates", "label": "Updates", "format": "number"}, {"field": "exact_actor_updates", "label": "Actor exact", "format": "number"}, {"field": "exact_metric_updates", "label": "Metrics exact", "format": "number"}, {"field": "max_parameter_difference", "label": "Max abs diff", "format": "number"}]},
]

blocks = [
    {"id": "title", "type": "markdown", "layout": "full", "body": "# End2Race PPO 全实验审计：BC 至 Group 9"},
    {"id": "summary_text", "type": "markdown", "sourceId": "summary", "body": """## 结论先行

固定 Austin600 面板上，**总体安全最优仍是 clip 0.20 的普通训练路径**：G5 U30 与逐 tensor 复现后延长的 G7 U40 都为 **11/600 碰撞（98.17% 成功）**；G7 U45 为 12。训练延长到 45U 没有继续单调下降，而是进入 11–14 次碰撞的平台。clip 0.25 U45 为 16，未显示继续放宽 clip 的收益。

对于 BC 从 lattice expert 继承的“超车后并线、尾部高侧滑撞对手”，答案是：**PPO 明显缓解，但没有从策略机制上彻底解决。** BC 有 12 个结构性超车后尾部接触，其中 8 个满足 5° 严格侧滑判据；G5 U30 降至 5/3，G7 U40 为 8/5。它们各自解决了 BC 严格 8 例中的 7 例，但又产生 2 或 4 个新严格案例，所以不是全局消除。

G9 hard-neighbor U45 把严格甩尾压到 **1 例**，且没有新严格甩尾；但总碰撞仍为 17，并相对 BC 新增 10 个其他碰撞场景。它证明 boundary-aware 数据能定向改善该机制，同时也证明单纯增加 hard neighbors 会发生失败类型迁移，当前版本不能替代普通 clip 0.20 winner。"""},
    {"id": "headline", "type": "metric-strip", "cardIds": ["safety", "tail_card", "evidence"]},
    {"id": "quality_text", "type": "markdown", "sourceId": "summary", "body": """## 证据范围与数据质量

本审计覆盖 16 个 PPO run、415 个 formal updates、92 个 PPO eval 面板（另有 1 个 BC 面板）、55,200 条 PPO-vs-BC episode 配对与 1,913 条碰撞 trace。全体 93 个面板中 92 个完整有效；唯一无效面板是高 actor-LR U20（599 JSON 行、599 NPZ、1 error），不进入需要完整面板的结论。逐 episode CSV 另保留 600 条 BC 自参考行，因此文件总行为 55,800。

名义 Austin600 每面板有 600 个唯一 scenario ID，但仅 **592 个唯一物理初态**：raceline 闭环索引 2096 与 0 是相同 waypoint，跨-raceline 的 8 个组合完全重复。固定 ID 仍允许严格配对，但有效独立场景数不能写成 600。0721 老 trace 缺 terminal post-step；其碰撞标签和时刻使用 `results_multi.json`，NPZ 只用于碰撞前运动重建。"""},
    {"id": "quality_table", "type": "table", "tableId": "quality"},
    {"id": "params", "type": "markdown", "sourceId": "configs", "body": """## 模型参数与实验控制

所有 run 的 actor 架构固定为 **11,301,482 参数**；`privilege_gru` critic 为 **11,309,401 参数**。共享训练参数为 16 envs × 6,400 steps、batch 12,800、actor/critic epochs 2/5、GRU/head/critic LR `3e-6/3e-5/3e-4`、gamma 0.999、GAE 0.995、动作 std 0.03/0.15、seed 42。G1 改 critic；G2 改 batch；G3 改 clip 但 0.10/0.20 旧 run 同时是 8 workers；G4 改 actor LR；G5 做 30U 干净 clip；G6 开 target-KL；G7 延长 clip .20 到 45U；G8 改 clip .25；G9 只打开 805 条 collision cache 的 hard-neighbor。

G7 U1–U30 与 G5 clip .20 的 30 个 actor checkpoint **每个 tensor 完全相同**，非 wall-time metrics 也完全相同，因此前 30U 可以合法复用，不是缺失评估。"""},
    {"id": "repro_table", "type": "table", "tableId": "repro"},
    {"id": "all_paths", "type": "table", "tableId": "paths"},
    {"id": "training_text", "type": "markdown", "sourceId": "training", "body": """## 训练 Metrics：能诊断优化，不能代替 Eval

G7/G8/G9 的最终 rollout collision rate 为 19.0%/17.7%/18.6%，最终 return 为 0.133/0.190/0.173；若只看训练，G8/G9 并不差。但 U45 eval 碰撞却是 12/16/17。G9 的全程平均 rollout collision rate 还是三者最低（25.2%），却没有成为 eval winner。

三条新 run 均未使用 target-KL；mean-KL 超过 0.05 的 updates 分别为 23/22/23，最大单 minibatch KL 为 3.83/2.43/2.74。actor gradient 是裁剪前范数；G9 最大 852 是警报信号，不等价于实际参数步长。训练日志 update k 的 rollout 由 policy k-1 生成，而 eval Uk 使用更新后的 checkpoint k，这个一拍错位也禁止简单逐行相关解释。"""},
    {"id": "training_table_block", "type": "table", "tableId": "training_table"},
    {"id": "candidate_chart", "type": "chart", "chartId": "candidate_paths"},
    {"id": "groups_text", "type": "markdown", "sourceId": "panels", "body": """## Group-by-group 结论

- **G1 critic：** `privilege_gru` U20=14，优于 privilege MLP=25 与 independent GRU=34；critic 自身 EV 高并不保证 advantage 对 eval 有益。
- **G2 batch：** 12,800 最稳；增大 batch 同时把每 update actor steps 从 16 降至 8/4，所以不是纯粹的样本方差实验。
- **G3 clip：** 旧 8-worker 0.20 优于 0.10，但不能与 12-worker baseline 纯归因；干净结论由 G5/G7 提供。
- **G4 LR：** 中档 `3e-6/3e-5` 最好；高 LR 训练 return/rollout collision 改善而 eval 恶化，是明确的分布内外背离。
- **G5/G7 长训：** clip .20 路径 U1–U45 为 21/18/16/13/13/17/11/14/11/12；30 后不再持续下降。
- **G6 target-KL：** 0.02 路径 20/17/19/18/20；0.04 为 23/45/70/58/33。早停减少 optimizer steps，却没有保护 eval，关闭是正确选择。
- **G8 clip .25：** 18/23/15/15/21/14/18/18/18/16；没有越过 .20 的最优 11。
- **G9 hard-neighbor：** 19/18/19/15/14/22/26/20/22/17；U20 尚可，之后明显漂移。45U 全程实际抽到 2,198 条 base-cache、1,490 条 boundary-extension 与 2,925 条 ordinary episodes。"""},
    {"id": "hard_source_block", "type": "table", "tableId": "hard_source"},
    {"id": "selected_block", "type": "table", "tableId": "selected_table"},
    {"id": "paired_text", "type": "markdown", "sourceId": "paired", "body": """## 与 BC 的逐 episode 对比

只看碰撞总数会掩盖策略漂移。BC→G5 U30 为共同 5、解决 17、新增 6；BC→G7 U40 为共同 7、解决 15、新增 4；BC→G9 U45 则共同 7、解决 15、新增 10。对应未校正 exact paired p 为 0.0347/0.0192/0.4244。

由于同一面板上看了 92 个 PPO checkpoints，应用全体比较的 BH 校正后，单个 checkpoint 均未达到 0.05；例如 G7 U40 q=0.413。这里的 exact p 只能描述固定面板上的不对称转换，不能替代新 seed 或 holdout。"""},
    {"id": "paired_table_block", "type": "table", "tableId": "paired_table"},
    {"id": "tail_method", "type": "markdown", "sourceId": "tail", "body": """## “超车后甩尾”判据与敏感性

trace 没保存模拟器真实 `slip_angle` 或碰撞接触点，因此报告不把主观轨迹截图当证据。结构性判据要求：车车碰撞；碰撞前相对赛道进度曾越过 0；参考帧对手位于 ego 后方且几何上更靠近车尾；前 0.5 秒横向中心距至少闭合 0.10m。严格判据再要求由 5-step 位置差分航向与车身 yaw 的差得到的 0.5 秒最大绝对侧滑角 ≥5°。

BC 的严格数量在 3°/5°/8° 下为 10/8/7，G5 U30 为 3/3/2，G7 U40 为 5/5/3，G9 U45 为 1/1/1。阈值变化不改变“普通 PPO 部分缓解、hard-neighbor 定向最强但总体退化”的排序。"""},
    {"id": "tail_chart", "type": "chart", "chartId": "tail_compare"},
    {"id": "strict_outcomes_block", "type": "table", "tableId": "strict_outcomes"},
    {"id": "commonality_text", "type": "markdown", "sourceId": "frequency", "body": """## 碰撞 episode 的共性

`sp17-ego727-raceline2-v0.5` 与 `sp35-ego1497-raceline1-v0.5` 在 **92/92** 个 PPO 面板都碰撞，BC 也失败；它们是稳定的早期/前向接触，并非甩尾。`sp5-ego213-raceline0-v0.7` 是最顽固的甩尾场景：79/92 面板碰撞，而且每次碰撞 79/79 都满足结构性尾部接触与严格 5° 判据。

另一个关键共性是 failure-set churn：普通 winner 与 BC 共享的碰撞只占 5–7 个，改善同时伴随新失败。G9 U45 虽解决 BC 15 个碰撞，却新增 10 个，且 17 次全是车车接触。这说明当前主要瓶颈不是单一墙撞或单一出生位，而是策略在不同 overtaking/merging 几何间迁移风险。"""},
    {"id": "persistent_chart", "type": "chart", "chartId": "persistent"},
    {"id": "top_table_block", "type": "table", "tableId": "top_table"},
    {"id": "answer", "type": "markdown", "sourceId": "summary", "body": """## 最终判断与行动建议

**PPO 是否解决 BC 的超车后甩尾？** 若“解决”指原有 8 个严格场景多数不再失败，答案是 **基本解决：普通 clip .20 的 G5 U30/G7 U40 都解决 7/8**。若“解决”指策略不再产生该类机制，答案是 **没有：它们分别新增 2/4 个严格甩尾，最终仍有 3/5 个严格案例**。G9 hard U45 接近机制消除（1 例、0 新增），但它以更多其他碰撞为代价，不是总体解。

当前应保留 **G5 U30 或 G7 U40 的 clip .20 checkpoint** 作为总体候选，不选 clip .25，也不选 G9 U45。下一步应先修正 50 点闭环采样，冻结 checkpoint 选择规则，在不重叠 holdout 上同时报告：总碰撞、BC resolved/new、结构性尾部接触、3°/5°/8°严格甩尾、超车与 min-surface。hard-neighbor 若继续，应调整采样/训练停止点，重点比较 U15/U20，而不是默认延长到 U45。"""},
    {"id": "limitations", "type": "markdown", "sourceId": "summary", "body": """## 局限

- 单 seed、固定 Austin 域内面板；不能宣称跨 seed 或跨地图泛化。
- 600 个 scenario ID 中只有 592 个唯一物理初态，Wilson 区间只作描述，不能当独立抽样置信区间。
- 多 checkpoint 选择存在 winner's curse；全体比较校正后无单点达到常规显著性。
- 旧 0721 NPZ 缺终止帧；终止碰撞真值来自 JSON。
- 甩尾使用 pose-derived sideslip 与中心几何代理；没有真实 tire slip/contact point，故结论限定为“与超车后尾部漂移机制一致”。"""},
]

generated = datetime.now(timezone.utc).isoformat()
artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "End2Race PPO 全实验审计：BC 至 Group 9",
        "description": "基于训练配置、415 个 update metrics、92 个 PPO eval 面板、55,200 条 PPO-vs-BC episode 配对与 NPZ trace 的技术分析。",
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
            "headline": records(headline),
            "arm_paths": records(arm_paths),
            "selected": records(selected),
            "candidate_paths": records(candidate_paths),
            "new_train": records(new_train),
            "hard_source": hard_source_summary,
            "repro_summary": records(repro_summary),
            "top_scenarios": records(top_scenarios),
            "strict_outcomes": records(strict_outcomes),
            "quality": records(quality),
            "paired": records(paired),
        },
        "accessIssues": [],
    },
    "sources": sources,
}

(OUT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")

notes = """# Report notes

## Source inventory

- Training source: `post-trained/*/run_config.json`, `metrics.jsonl`, `episodes.jsonl`, actor/critic checkpoints.
- Evaluation source: `eval_results/*/results_multi.json` and per-episode NPZ traces.
- Bounded audit tables: parameter matrix, update metrics, panel metrics, 55,200-row PPO-vs-BC comparison plus 600 BC self-reference rows,
  collision feature table, tail-mechanism table, scenario frequency, and paired comparisons.

## Structure

1. Answer-first safety and tail-swing conclusion.
2. Data-quality and exact-reproduction boundary.
3. Recorded parameters, full experiment paths, and training telemetry.
4. Exact scenario-paired evaluation and collision commonality.
5. Auditable tail-swing classifier, threshold sensitivity, conclusion, and limitations.

## Visualization choices

| View | Question | Form | Reason |
|---|---|---|---|
| Candidate path | Do 45U, clip .25, or hard-neighbor improve? | grouped bar | Exact checkpoint values and discrete arms |
| Tail comparison | Is the target failure mechanism reduced? | bar | Small exact counts across representative checkpoints |
| Persistent scenarios | Which episodes fail repeatedly? | ranked bar | Shared 92-panel denominator |

## Exclusions

- No wall-time comparison because runs may overlap and hardware utilization is uncontrolled.
- No inferential error bars because there is one training seed and the panel includes duplicated physical starts.
- No true slip-angle/contact-point claim because those simulator fields were not saved in trace NPZ.
"""
(OUT / "REPORT_NOTES.md").write_text(notes)
print(OUT / "artifact.json")
print(OUT / "REPORT_NOTES.md")

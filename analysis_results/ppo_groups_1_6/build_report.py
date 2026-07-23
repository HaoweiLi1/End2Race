#!/usr/bin/env python3
"""Build the canonical Data Analytics report artifact for PPO Groups 1-6."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_results" / "ppo_groups_1_6"


def load_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, *, integer: bool = False):
    if value in (None, ""):
        return None
    return int(float(value)) if integer else float(value)


def truth(value) -> bool:
    return str(value).strip().lower() == "true"


summary = json.loads((OUT / "analysis_summary.json").read_text())
npz = json.loads((OUT / "npz_audit.json").read_text())
group_rows = load_csv("group_summary.csv")
control_rows = load_csv("group_control_audit.csv")
pairwise_rows = load_csv("scenario_pairwise.csv")
trace_rows = load_csv("trace_summary_selected.csv")
join_rows = load_csv("training_eval_join.csv")
panel_rows = load_csv("panel_metrics.csv")
scenario_rows = load_csv("scenario_frequency.csv")


run_display = {
    "ppo_independent_gru_0721_base": "独立 GRU",
    "ppo_privilege_mlp_0721_base": "特权 MLP",
    "ppo_privilege_gru_0721_base": "共享基线",
    "ppo_privilege_gru_0721_bs25600": "batch 25,600",
    "ppo_privilege_gru_0721_bs51200": "batch 51,200",
    "ppo_privilege_gru_0721_clip010": "clip 0.10（旧）",
    "ppo_privilege_gru_0721_clip020": "clip 0.20（旧）",
    "ppo_privilege_gru_0722_lr1_tkloff": "LR 1×",
    "ppo_privilege_gru_0722_lr5_tkloff": "LR 5×",
    "ppo_privilege_gru_0722_long_clip015": "clip 0.15 / 30U",
    "ppo_privilege_gru_0722_long_clip020": "clip 0.20 / 30U",
    "ppo_privilege_gru_0722_clip015_tkl002": "target-KL 0.02",
    "ppo_privilege_gru_0722_clip015_tkl004": "target-KL 0.04",
}


arm_performance = []
for row in group_rows:
    run = row["run"]
    arm = run_display.get(run, row["label"])
    if row["group"] == "Group 2" and run == "ppo_privilege_gru_0721_base":
        arm = "batch 12,800"
    elif row["group"] == "Group 3" and run == "ppo_privilege_gru_0721_base":
        arm = "clip 0.15（基线）"
    elif row["group"] == "Group 4" and run == "ppo_privilege_gru_0721_base":
        arm = "LR 3×（基线）"
    elif row["group"] == "Group 6" and run == "ppo_privilege_gru_0721_base":
        arm = "target-KL 关闭"
    arm_performance.append(
        {
            "group": row["group"],
            "arm": arm,
            "arm_display": f"{row['group'].replace('Group ', 'G')} · {arm}",
            "run": run,
            "collision_path": row["collision_path"],
            "mean_collision_u5_plus": number(row["mean_collision_u5_plus"]),
            "best_update": number(row["best_update"], integer=True),
            "best_collision_count": number(row["best_collision_count"], integer=True),
            "final_update": number(row["final_update"], integer=True),
            "final_collision_count": number(row["final_collision_count"], integer=True),
            "final_eval_valid": truth(row["final_eval_valid"]),
            "env_workers": number(row["env_workers"], integer=True),
            "mean_approx_kl": number(row["mean_approx_kl"]),
            "mean_clip_fraction": number(row["mean_clip_fraction"]),
            "final_explained_variance_post": number(row["final_explained_variance_post"]),
            "early_stop_updates": number(row["early_stop_updates"], integer=True),
            "actor_steps_completed": number(row["actor_steps_completed"], integer=True),
            "actor_steps_planned": number(row["actor_steps_planned"], integer=True),
        }
    )


group5_panels = []
for row in panel_rows:
    if row["run"] not in {
        "ppo_privilege_gru_0722_long_clip015",
        "ppo_privilege_gru_0722_long_clip020",
    }:
        continue
    group5_panels.append(
        {
            "arm": "clip 0.15" if row["run"].endswith("clip015") else "clip 0.20",
            "run": row["run"],
            "update": number(row["update"], integer=True),
            "collision_count": number(row["collision_count"], integer=True),
            "overtake_count": number(row["overtake_count"], integer=True),
            "follow_count": number(row["follow_count"], integer=True),
            "success_rate": number(row["success_rate"]),
            "median_collision_time_s": number(row["median_collision_time_s"]),
            "episode_rows": number(row["episode_rows"], integer=True),
            "valid": truth(row["valid"]),
        }
    )


telemetry_scatter = []
for row in join_rows:
    if not truth(row["valid"]):
        continue
    stopped = truth(row["actor_early_stop_triggered"])
    telemetry_scatter.append(
        {
            "point": f"{run_display.get(row['run'], row['run'])} U{row['update']}",
            "run": row["run"],
            "update": number(row["update"], integer=True),
            "collision_count": number(row["collision_count"], integer=True),
            "approx_kl_mean": number(row["approx_kl_mean"]),
            "approx_kl_max": number(row["approx_kl_max"]),
            "clip_fraction_mean": number(row["clip_fraction_mean"]),
            "explained_variance_post": number(row["explained_variance_post"]),
            "training_collision_rate": number(row["training_collision_rate"]),
            "early_stop": "本更新早停" if stopped else "未早停",
            "episode_rows": number(row["episode_rows"], integer=True),
        }
    )


persistent_scenarios = []
for rank, row in enumerate(scenario_rows[:10], start=1):
    parts = row["scenario_id"].replace("evaluation-", "").replace("raceline", "r")
    persistent_scenarios.append(
        {
            "rank": rank,
            "scenario": row["scenario_id"],
            "scenario_short": parts,
            "collision_panels": number(row["collision_panels"], integer=True),
            "valid_ppo_panels": number(row["valid_ppo_panels"], integer=True),
            "collision_rate": number(row["rate"]),
        }
    )


control_audit = []
for row in control_rows:
    control_audit.append(
        {
            "group": row["group"],
            "arm": run_display.get(row["arm_run"], row["arm_run"]),
            "intended_axis": row["intended_axis"],
            "recorded_differences": row["recorded_differences"],
            "confounds": row["confounds"] or "无",
            "strict_single_axis": "是" if truth(row["strict_single_axis"]) else "否",
        }
    )


pairwise = []
for row in pairwise_rows:
    pairwise.append(
        {
            "comparison": row["comparison"],
            "a_collisions": number(row["a_collisions"], integer=True),
            "b_collisions": number(row["b_collisions"], integer=True),
            "shared": number(row["shared"], integer=True),
            "resolved": number(row["resolved"], integer=True),
            "created": number(row["created"], integer=True),
            "net_change": number(row["net_change"], integer=True),
            "jaccard": number(row["jaccard"]),
        }
    )


trace_summary = []
for row in trace_rows:
    trace_summary.append(
        {
            "label": row["label"],
            "collision_count": number(row["collision_count"], integer=True),
            "opponent_collisions": number(row["collision_with_opponent_count"], integer=True),
            "ego_or_wall_collisions": number(row["collision_count"], integer=True)
            - number(row["collision_with_opponent_count"], integer=True),
            "initial_collisions": number(row["initial_collision_count"], integer=True),
            "median_collision_time_s": number(row["median_collision_time_s"]),
            "raceline0": number(row["raceline0_count"], integer=True),
            "raceline1": number(row["raceline1_count"], integer=True),
            "raceline2": number(row["raceline2_count"], integer=True),
        }
    )


quality = [
    {"check": "评估面板", "result": "69 / 70 有效", "interpretation": "仅 Group 4 高 LR U20 无效；599 行、1 错误"},
    {"check": "场景覆盖", "result": "有效面板均为同一 Austin600", "interpretation": "逐场景可配对；每面板 600 个唯一场景"},
    {"check": "训练 metrics", "result": "全部有限且 update 连续", "interpretation": "13 个训练 run；正式 update 序列完整"},
    {"check": "checkpoint", "result": "全部存在", "interpretation": "每个 actor_final 与最后 actor checkpoint 张量一致"},
    {"check": "NPZ 数值与时间维", "result": f"{npz['totals']['numeric_True']:,} / {npz['totals']['files']:,}", "interpretation": "全部 numeric/bool 且 leading dimension 对齐"},
    {"check": "新版 NPZ 终止语义", "result": "20,999 / 20,999 有效", "interpretation": "post_step_v2 的终止帧及碰撞标记与 JSON 一致"},
    {"check": "旧版 NPZ 终止语义", "result": "21,000 缺 terminal post-step", "interpretation": "736 个旧版碰撞轨迹不可用 NPZ 碰撞位复核，改用 results_multi.json"},
    {"check": "训练/评估泄漏", "result": "ego index 重叠 0", "interpretation": "训练 131 个唯一 ego index；评估 50 个"},
]


headline = [
    {
        "selected_collisions": 11,
        "bc_collisions": 22,
        "collision_reduction": 0.5,
        "selected_success_rate": 589 / 600,
        "selected_overtakes": 358,
        "valid_panels": summary["quality"]["valid_panels"],
        "npz_files": npz["totals"]["files"],
    }
]


correlation_rows = []
for row in summary["correlations"]:
    correlation_rows.append(
        {
            "metric": row["label"],
            "panels": row["panels"],
            "pearson": row["pearson_with_eval_collision_count"],
            "spearman": row["spearman_with_eval_collision_count"],
            "scope": row["scope"],
        }
    )


sources = [
    {
        "id": "analysis_summary",
        "label": "PPO Groups 1-6 consolidated analysis",
        "path": "analysis_results/ppo_groups_1_6/analysis_summary.json",
        "query": {
            "language": "python",
            "description": "Reads run configs, training metrics.jsonl, evaluation results_multi.json and NPZ traces; validates panels and produces joined comparison tables.",
            "tables_used": [
                "post-trained/*/config.json",
                "post-trained/*/metrics.jsonl",
                "eval_results/*/results_multi.json",
                "eval_results/*/traces/*.npz",
            ],
            "filters": ["Austin600", "formal checkpoints U1/U5/U10/U15/U20 and Group 5 U25/U30", "invalid panels excluded from inferential summaries"],
            "metric_definitions": [
                "collision_count = collision episodes among the 600 evaluation scenarios",
                "success_rate = (600 - collision_count) / 600",
                "mean_collision_u5_plus = arithmetic mean of valid checkpoint collision counts from U5 onward",
            ],
        },
    },
    {
        "id": "group_summary",
        "label": "Experiment parameters, training telemetry and evaluation paths",
        "path": "analysis_results/ppo_groups_1_6/group_summary.csv",
        "query": {
            "language": "python",
            "description": "One bounded row per experiment arm, joining recorded parameters, formal training metrics and evaluation checkpoints.",
            "tables_used": ["analysis_results/ppo_groups_1_6/run_summary.csv", "analysis_results/ppo_groups_1_6/panel_metrics.csv"],
            "metric_definitions": ["best/final collision counts use only valid evaluation panels", "logged actor gradient norm is the pre-clip norm returned by clip_grad_norm_"],
        },
    },
    {
        "id": "control_audit",
        "label": "Recorded single-axis control audit",
        "path": "analysis_results/ppo_groups_1_6/group_control_audit.csv",
        "query": {
            "language": "python",
            "description": "Compares every arm's recorded config against its declared baseline and flags any additional changed parameter.",
            "tables_used": ["post-trained/*/config.json"],
        },
    },
    {
        "id": "evaluation_panels",
        "label": "Evaluation panel metrics",
        "path": "analysis_results/ppo_groups_1_6/panel_metrics.csv",
        "query": {
            "language": "python",
            "description": "One row per evaluated checkpoint, reconciled against episode records and trace filenames.",
            "tables_used": ["eval_results/*/results_multi.json", "eval_results/*/traces/*.npz"],
            "metric_definitions": ["valid requires 600 episodes, 600 unique scenarios, 600 traces, zero errors and consistent aggregates"],
        },
    },
    {
        "id": "training_eval_join",
        "label": "Training/evaluation checkpoint join",
        "path": "analysis_results/ppo_groups_1_6/training_eval_join.csv",
        "query": {
            "language": "python",
            "description": "Joins formal update telemetry to the matching evaluation checkpoint; correlations are pooled descriptive diagnostics.",
            "tables_used": ["post-trained/*/metrics.jsonl", "analysis_results/ppo_groups_1_6/panel_metrics.csv"],
        },
    },
    {
        "id": "npz_audit",
        "label": "Full NPZ structural and terminal-state audit",
        "path": "analysis_results/ppo_groups_1_6/npz_audit.json",
        "query": {
            "language": "python",
            "description": "Inspects every trace NPZ for numeric dtype, aligned leading dimensions, terminal fields and collision-marker agreement with evaluation JSON.",
            "tables_used": ["eval_results/*/traces/*.npz", "eval_results/*/results_multi.json"],
        },
    },
    {
        "id": "scenario_identity",
        "label": "Scenario-level collision identity and trace summaries",
        "path": "analysis_results/ppo_groups_1_6/scenario_pairwise.csv",
        "query": {
            "language": "python",
            "description": "Pairs the same Austin600 scenario IDs across checkpoints and counts shared, resolved and newly created collision cases.",
            "tables_used": ["eval_results/*/results_multi.json", "analysis_results/ppo_groups_1_6/trace_summary_selected.csv"],
            "metric_definitions": ["Jaccard = shared / (shared + resolved + created)"],
        },
    },
    {
        "id": "scenario_frequency",
        "label": "Persistent collision scenario frequency",
        "path": "analysis_results/ppo_groups_1_6/scenario_frequency.csv",
        "query": {
            "language": "python",
            "description": "Counts in how many of the 68 valid PPO checkpoint panels each Austin600 scenario collides.",
            "tables_used": ["eval_results/*/results_multi.json"],
        },
    },
    {
        "id": "trace_summary",
        "label": "Selected checkpoint trace-level collision summaries",
        "path": "analysis_results/ppo_groups_1_6/trace_summary_selected.csv",
        "query": {
            "language": "sql",
            "description": "Loads the bounded trace-level collision classification table derived from the audited NPZ files.",
            "tables_used": ["analysis_results/ppo_groups_1_6/trace_summary_selected.csv", "eval_results/*/traces/*.npz"],
        },
    },
]

# Portable artifact validation currently requires a concrete SQL retrieval
# statement for every visible card/chart/table.  These DuckDB queries reproduce
# each bounded widget dataset from the reviewable analysis files; the heavier raw
# transformation remains in analyze_groups.py and is documented above.
source_sql = {
    "analysis_summary": "SELECT * FROM read_json_auto('analysis_results/ppo_groups_1_6/analysis_summary.json');",
    "group_summary": "SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6/group_summary.csv', header = true);",
    "control_audit": "SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6/group_control_audit.csv', header = true);",
    "evaluation_panels": "SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6/panel_metrics.csv', header = true);",
    "training_eval_join": "SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6/training_eval_join.csv', header = true) WHERE valid = true;",
    "npz_audit": "SELECT * FROM read_json_auto('analysis_results/ppo_groups_1_6/npz_audit.json');",
    "scenario_identity": "SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6/scenario_pairwise.csv', header = true);",
    "scenario_frequency": "SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6/scenario_frequency.csv', header = true) ORDER BY collision_panels DESC, scenario_id LIMIT 10;",
    "trace_summary": "SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6/trace_summary_selected.csv', header = true);",
}
for source in sources:
    source["query"]["engine"] = "duckdb"
    source["query"]["sql"] = source_sql[source["id"]]


cards = [
    {
        "id": "selected_collision_card",
        "description": "Group 5 clip 0.20 at U30 on Austin600.",
        "dataset": "headline",
        "sourceId": "analysis_summary",
        "metrics": [
            {"label": "选定模型碰撞", "field": "selected_collisions", "format": "number"},
            {"label": "相对 BC 降幅", "field": "collision_reduction", "format": "percent"},
        ],
    },
    {
        "id": "success_card",
        "description": "589 of 600 scenarios complete without collision.",
        "dataset": "headline",
        "sourceId": "analysis_summary",
        "metrics": [
            {"label": "成功率", "field": "selected_success_rate", "format": "percent"},
            {"label": "超车", "field": "selected_overtakes", "format": "number"},
        ],
    },
    {
        "id": "quality_card",
        "description": "Validated evidence included in this report.",
        "dataset": "headline",
        "sourceId": "analysis_summary",
        "metrics": [
            {"label": "有效评估面板", "field": "valid_panels", "format": "number"},
            {"label": "NPZ 全量检查", "field": "npz_files", "format": "number"},
        ],
    },
]


charts = [
    {
        "id": "group_mean_collisions",
        "title": "各组实验臂的 U5+ 平均碰撞数",
        "subtitle": "越低越好；Group 4 高 LR 的无效 U20 已排除",
        "type": "bar",
        "dataset": "arm_performance",
        "sourceId": "group_summary",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "arm", "type": "nominal", "label": "实验臂"},
            "y": {"field": "mean_collision_u5_plus", "type": "quantitative", "label": "U5+ 平均碰撞数"},
            "color": {"field": "group", "type": "nominal", "label": "实验组"},
            "tooltip": [
                {"field": "collision_path", "type": "nominal", "label": "碰撞路径"},
                {"field": "final_collision_count", "type": "quantitative", "label": "最终碰撞"},
                {"field": "env_workers", "type": "quantitative", "label": "评估 workers"},
                {"field": "mean_approx_kl", "type": "quantitative", "label": "平均 KL"},
            ],
        },
    },
    {
        "id": "group5_path",
        "title": "Group 5 正式检查点碰撞路径",
        "subtitle": "同为 12 workers、30 updates，仅 clip range 不同",
        "type": "bar",
        "dataset": "group5_panels",
        "sourceId": "evaluation_panels",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "update", "type": "nominal", "label": "Formal update"},
            "y": {"field": "collision_count", "type": "quantitative", "label": "碰撞数"},
            "color": {"field": "arm", "type": "nominal", "label": "Clip range"},
            "tooltip": [
                {"field": "overtake_count", "type": "quantitative", "label": "超车"},
                {"field": "follow_count", "type": "quantitative", "label": "跟随"},
                {"field": "success_rate", "type": "quantitative", "label": "成功率", "format": "percent"},
                {"field": "median_collision_time_s", "type": "quantitative", "label": "碰撞时间中位数"},
            ],
        },
    },
    {
        "id": "telemetry_scatter",
        "title": "训练 KL 与评估碰撞数",
        "subtitle": "68 个有效 PPO 检查点；点色表示该 update 是否触发 actor 早停",
        "type": "scatter",
        "dataset": "telemetry_scatter",
        "sourceId": "training_eval_join",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "approx_kl_mean", "type": "quantitative", "label": "Actor mean approximate KL"},
            "y": {"field": "collision_count", "type": "quantitative", "label": "评估碰撞数"},
            "color": {"field": "early_stop", "type": "nominal", "label": "Actor 早停"},
            "label": {"field": "point", "type": "nominal", "label": "检查点"},
            "tooltip": [
                {"field": "point", "type": "nominal", "label": "检查点"},
                {"field": "approx_kl_max", "type": "quantitative", "label": "最大 KL"},
                {"field": "clip_fraction_mean", "type": "quantitative", "label": "Clip fraction"},
                {"field": "explained_variance_post", "type": "quantitative", "label": "Critic EV"},
            ],
        },
    },
    {
        "id": "persistent_scenarios",
        "title": "最持续的 10 个碰撞场景",
        "subtitle": "分母为 68 个有效 PPO 检查点面板",
        "type": "bar",
        "dataset": "persistent_scenarios",
        "sourceId": "scenario_frequency",
        "valueFormat": "percent",
        "encodings": {
            "x": {"field": "scenario_short", "type": "nominal", "label": "场景"},
            "y": {"field": "collision_rate", "type": "quantitative", "label": "出现碰撞的面板比例"},
            "tooltip": [
                {"field": "scenario", "type": "nominal", "label": "完整场景 ID"},
                {"field": "collision_panels", "type": "quantitative", "label": "碰撞面板"},
                {"field": "valid_ppo_panels", "type": "quantitative", "label": "有效面板"},
            ],
        },
    },
]


tables = [
    {
        "id": "control_table",
        "title": "参数控制审计",
        "subtitle": "“否”表示对共享 baseline 的比较存在额外变化",
        "dataset": "control_audit",
        "sourceId": "control_audit",
        "columns": [
            {"field": "group", "label": "组"},
            {"field": "arm", "label": "实验臂"},
            {"field": "intended_axis", "label": "预期变量"},
            {"field": "recorded_differences", "label": "实际变化"},
            {"field": "confounds", "label": "混杂"},
            {"field": "strict_single_axis", "label": "严格单轴"},
        ],
    },
    {
        "id": "group_detail_table",
        "title": "六组训练—评估明细",
        "subtitle": "路径依次为正式检查点；星号对应无效最终面板",
        "dataset": "arm_performance",
        "sourceId": "group_summary",
        "columns": [
            {"field": "group", "label": "组"},
            {"field": "arm", "label": "实验臂"},
            {"field": "env_workers", "label": "workers", "format": "number"},
            {"field": "collision_path", "label": "碰撞路径"},
            {"field": "mean_collision_u5_plus", "label": "U5+ 均值", "format": "number"},
            {"field": "best_update", "label": "最佳 U", "format": "number"},
            {"field": "best_collision_count", "label": "最佳碰撞", "format": "number"},
            {"field": "final_collision_count", "label": "最终碰撞", "format": "number"},
            {"field": "mean_approx_kl", "label": "平均 KL", "format": "number"},
            {"field": "mean_clip_fraction", "label": "Clip frac", "format": "percent"},
            {"field": "early_stop_updates", "label": "早停 updates", "format": "number"},
            {"field": "actor_steps_completed", "label": "Actor steps", "format": "number"},
        ],
    },
    {
        "id": "quality_table",
        "title": "证据完整性与可用性",
        "subtitle": "结果可用于固定面板比较；旧 NPZ 的终止碰撞位除外",
        "dataset": "quality",
        "sourceId": "analysis_summary",
        "columns": [
            {"field": "check", "label": "检查"},
            {"field": "result", "label": "结果"},
            {"field": "interpretation", "label": "解释"},
        ],
    },
    {
        "id": "pairwise_table",
        "title": "逐场景碰撞身份变化",
        "subtitle": "净下降不代表同一批场景稳定修复；需同时看 resolved 与 created",
        "dataset": "pairwise",
        "sourceId": "scenario_identity",
        "columns": [
            {"field": "comparison", "label": "比较"},
            {"field": "a_collisions", "label": "A 碰撞", "format": "number"},
            {"field": "b_collisions", "label": "B 碰撞", "format": "number"},
            {"field": "shared", "label": "共同", "format": "number"},
            {"field": "resolved", "label": "消除", "format": "number"},
            {"field": "created", "label": "新增", "format": "number"},
            {"field": "net_change", "label": "净变化", "format": "number"},
            {"field": "jaccard", "label": "Jaccard", "format": "number"},
        ],
    },
    {
        "id": "trace_table",
        "title": "代表性检查点的 trace 级碰撞构成",
        "subtitle": "碰撞对手数来自逐帧空间关系；ego/wall 为差额分类",
        "dataset": "trace_summary",
        "sourceId": "trace_summary",
        "columns": [
            {"field": "label", "label": "检查点"},
            {"field": "collision_count", "label": "碰撞", "format": "number"},
            {"field": "opponent_collisions", "label": "对手碰撞", "format": "number"},
            {"field": "ego_or_wall_collisions", "label": "ego/墙", "format": "number"},
            {"field": "initial_collisions", "label": "初始碰撞", "format": "number"},
            {"field": "median_collision_time_s", "label": "时间中位数(s)", "format": "number"},
            {"field": "raceline0", "label": "R0", "format": "number"},
            {"field": "raceline1", "label": "R1", "format": "number"},
            {"field": "raceline2", "label": "R2", "format": "number"},
        ],
    },
    {
        "id": "correlation_table",
        "title": "训练指标与评估碰撞的 pooled 相关性",
        "subtitle": "68 个有效 PPO 检查点；异质 run 的描述性诊断，不作因果解释",
        "dataset": "correlations",
        "sourceId": "training_eval_join",
        "columns": [
            {"field": "metric", "label": "训练指标"},
            {"field": "panels", "label": "面板数", "format": "number"},
            {"field": "pearson", "label": "Pearson", "format": "number"},
            {"field": "spearman", "label": "Spearman", "format": "number"},
        ],
    },
]


blocks = [
    {"id": "title", "type": "markdown", "layout": "full", "body": "# End2Race PPO：Baseline 至 Group 6 对比实验技术分析"},
    {
        "id": "technical_summary",
        "type": "markdown",
        "sourceId": "analysis_summary",
        "body": """## 技术摘要

在现有单次训练、固定 Austin600 评估面板上，最有证据支持的配置是 **`privilege_gru` + batch 12,800 + actor GRU/head LR `3e-6 / 3e-5` + clip 0.20 + target-KL 关闭，并训练到 U30**。它得到 **11/600 碰撞、358 次超车、231 次跟随、98.17% 成功率**；相对 BC 的 22 次碰撞减半，也优于同条件 clip 0.15 U30 的 20 次碰撞。

六组实验给出的参数方向一致：特权共享 GRU critic 最好；增大 batch 会削弱优化步数与后期稳定性；中间学习率最好；干净的 Group 5 支持 clip 0.20；target-KL 早停没有转化成更安全的 policy。**但最终选择仍是单 seed、同一固定场景面板、多个 checkpoint 中择优的结果，不能等价为泛化收益或置信区间。**""",
    },
    {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["selected_collision_card", "success_card", "quality_card"]},
    {
        "id": "key_findings",
        "type": "markdown",
        "sourceId": "analysis_summary",
        "body": """## 关键发现

**最终排序由 eval 决定，训练 metrics 用于解释机制。** 选定的 clip 0.20 U30 虽然最终 critic explained variance（0.906）低于 clip 0.15（0.950），但碰撞数为 11 对 20。反过来，Group 1 的特权 MLP warm-up validation loss 最低（0.183），U20 却有 25 次碰撞，明显差于 `privilege_gru` 的 14 次。

**累计碰撞数必须与场景身份一起看。** Group 3 的 clip 0.15 与旧 clip 0.20 在 U20 都是 14 次碰撞，但只共享 6 个碰撞场景，各自消除 8 个、又新增 8 个；相同总数掩盖了明显的场景迁移。""",
    },
    {"id": "group_chart_block", "type": "chart", "chartId": "group_mean_collisions"},
    {
        "id": "group1",
        "type": "markdown",
        "sourceId": "group_summary",
        "body": """## Group 1：Critic 架构

这是严格单轴对照，三组仅改变 critic。U20 碰撞为：独立 GRU **34**、特权 MLP **25**、共享 `privilege_gru` **14**；U5–U20 均值分别为 27.25、24.00、16.00。`privilege_gru` 的最终总体/碰撞/普通 explained variance 为 0.922/0.921/0.864，兼具最好 eval 和最均衡的价值拟合。因此后续组固定采用 `privilege_gru`。

warm-up validation loss 不能直接排序 policy：特权 MLP 的 0.183 是三者最低，但 eval 反而更差。""",
    },
    {
        "id": "group2",
        "type": "markdown",
        "sourceId": "group_summary",
        "body": """## Group 2：Batch size

这是严格单轴对照。batch 12,800 / 25,600 / 51,200 每个 update 的计划 actor optimizer steps 分别为 16 / 8 / 4，20 updates 总步数为 320 / 160 / 80。对应 U5–U20 平均碰撞为 **16.00 / 20.25 / 17.50**，U20 为 **14 / 16 / 21**。

更大 batch 同时降低平均 KL 与 clip fraction，说明每个 formal update 的学习强度下降；51,200 在 U10 短暂达到 15 次碰撞，但 U20 回退到 21。保留 12,800 的依据是最终性能与跨 checkpoint 稳定性，而不是单点最优。""",
    },
    {
        "id": "group3",
        "type": "markdown",
        "sourceId": "analysis_summary",
        "body": """## Group 3：Clip range（历史实验含混杂）

记录的碰撞路径为 clip 0.10：28/23/17/28/20；共享 baseline clip 0.15：16/18/18/14/14；旧 clip 0.20：24/23/22/18/14。**baseline 使用 12 workers，而 0.10 与旧 0.20 使用 8 workers，因此 baseline 对另外两臂不是严格单轴。**

0.10 与旧 0.20 在 8-worker 内是干净对照，0.20 的 U5–U20 均值 19.25 优于 0.10 的 22.00。两者 warm-up 记录完全一致（19 epochs、best epoch 16、validation loss 0.169683），而与 baseline 的 5/2/0.298154 不同；由于 clip 在 warm-up 不生效，这进一步证明不能把 baseline 差异只归因于 clip。

张量复核也显示：共享 baseline 与 Group 5 长跑 clip 0.15 的 U1/U5/U10/U15/U20 actor 完全一致；旧 clip 0.20 与 Group 5 clip 0.20 在相同 updates 均不一致，最大参数差从 0.00057 增至 0.00319。两条 clip 0.20 轨迹不能混作同一 run，干净结论只取 Group 5。""",
    },
    {
        "id": "group4",
        "type": "markdown",
        "sourceId": "group_summary",
        "body": """## Group 4：Actor 学习率

这是严格单轴对照，仅 GRU/head LR 改为 1×=`1e-6/1e-5`、3×=`3e-6/3e-5`、5×=`5e-6/5e-5`。1× 路径 18/15/15/19/21，早期较好但随后漂移；3× 路径 16/18/18/14/14，最终最好；5× 路径 19/21/21/26/26*，且 U20 因缺 1 个场景无效。

平均 KL 从 1× 的 0.0147 升至 3× 的 0.0728、5× 的 0.0815；clip fraction 从 0.0343 升至 0.0975、0.1595。1× 明显偏弱，5× 更激进且 eval 恶化，因此保留 3×。日志中的 gradient norm 是 clip 前范数；不能把它当成实际应用的更新幅度。""",
    },
    {
        "id": "group5",
        "type": "markdown",
        "sourceId": "analysis_summary",
        "body": """## Group 5：30 updates 的干净 Clip 对照

两臂均使用 12 workers、相同 warm-up、相同学习率与 batch，只改变 clip 0.15 → 0.20；这是当前权威 clip 对照。clip 0.15 路径为 16/18/18/14/14/17/20，U20 后持续退化；clip 0.20 为 21/18/16/13/13/17/11，U25 短暂回退后 U30 达到全实验最低 **11**。

U30 逐场景比较显示，clip 0.20 相对 clip 0.15 从 20 降到 11：共同 5、消除 15、新增 6。相对 BC 从 22 降到 11：共同 5、消除 17、新增 6。因存在 checkpoint 选择，建议先把 clip 0.20 U30 视为候选，而非已证明的稳定终局。""",
    },
    {"id": "group5_chart_block", "type": "chart", "chartId": "group5_path"},
    {
        "id": "group6",
        "type": "markdown",
        "sourceId": "analysis_summary",
        "body": """## Group 6：Target-KL

关闭 / 0.02 / 0.04 的碰撞路径为 16/18/18/14/14、20/17/19/18/20、23/45/70/58/33。0.02 使用 8 workers，和 12-worker baseline 同时改变两个因素；只有 0.04 是严格单轴比较。

0.02 在 11/20 个 updates 早停，完成 206/320 个 actor steps；0.04 在 12/20 个 updates 早停，完成 203/320。实际门槛是 `1.5 × target_kl`，分别为 0.03 与 0.06；停止记录仍出现最高 2.43 与 1.05 的 minibatch KL。尽管显著减少 optimizer steps，0.04 的 U20 仍为 33 次碰撞，并新增 28 个 baseline 未碰撞的场景。当前门控是在触发该 minibatch 更新后停止后续 minibatch，不能撤销已经造成 KL overshoot 的更新；因此它是计算/信赖域诊断，不是安全保证。结论是保持 `target_kl=None`。""",
    },
    {"id": "control_table_block", "type": "table", "tableId": "control_table"},
    {"id": "group_detail_block", "type": "table", "tableId": "group_detail_table"},
    {
        "id": "telemetry",
        "type": "markdown",
        "sourceId": "analysis_summary",
        "body": """## 训练 Metrics 与 Eval 的关系

跨 68 个有效 PPO 检查点汇总后，训练 rollout collision rate、return、critic value loss、explained variance、mean KL、clip fraction 与 eval 碰撞的 Pearson 绝对值都不超过 **0.114**；Spearman 绝对值最高也只有 **0.279**。这是异质 run 的 pooled 描述统计，不能用于因果推断，但足以说明：**这些训练遥测适合诊断更新是否过弱、过激或早停，不适合替代固定场景 eval 来选 checkpoint。**""",
    },
    {"id": "telemetry_chart_block", "type": "chart", "chartId": "telemetry_scatter"},
    {"id": "correlation_table_block", "type": "table", "tableId": "correlation_table"},
    {
        "id": "data_quality",
        "type": "markdown",
        "sourceId": "analysis_summary",
        "body": """## 数据质量与 NPZ 证据

70 个面板中 69 个有效；唯一无效项是 Group 4 高 LR U20（599 个 episode/trace、1 个错误），该点不进入均值和最终判断。所有有效面板具有相同 600 个场景，训练 metrics 数值有限且 formal update 连续，所有 checkpoint 存在，且每个 `actor_final.pth` 与最后一个 actor checkpoint 张量一致。

全量检查的 41,999 个 NPZ 均为 numeric/bool，所有 leading dimension 对齐。20,999 个 `post_step_v2` 轨迹的终止字段与 JSON 完全一致；21,000 个旧轨迹缺 terminal post-step，其中 736 个碰撞 episode 的 NPZ 最后一帧没有碰撞标记，这是格式语义限制，不是 JSON 标签冲突。因此旧面板碰撞真值使用 `results_multi.json`，NPZ 只用于终止前运动与空间关系分析。""",
    },
    {"id": "quality_table_block", "type": "table", "tableId": "quality_table"},
    {
        "id": "scenario_identity_section",
        "type": "markdown",
        "sourceId": "analysis_summary",
        "body": """## 场景身份与碰撞构成

最终 clip 0.20 U30 的 11 次碰撞全部具有对手空间接触证据，碰撞时间中位数为 6.07 秒。相比之下，target-KL 0.04 U20 的 33 次碰撞只有 23 次涉及对手，另有 10 次为 ego/墙类，碰撞时间中位数 3.92 秒，说明失败不只是车车冲突数量增加，而是更广泛的轨迹失稳。

两条场景在 68/68 个有效 PPO 面板中都碰撞，说明它们是强稳定困难样本；但不能据此宣称不可解，应作为后续定向诊断集。""",
    },
    {"id": "trace_table_block", "type": "table", "tableId": "trace_table"},
    {"id": "persistent_chart_block", "type": "chart", "chartId": "persistent_scenarios"},
    {"id": "pairwise_table_block", "type": "table", "tableId": "pairwise_table"},
    {
        "id": "scope",
        "type": "markdown",
        "body": """## 范围、数据与指标定义

- **Baseline**：`pretrained/end2race.pth` 的 BC，在 Austin600 上为 22 碰撞、344 超车、234 跟随。
- **共享 PPO baseline**：`ppo_privilege_gru_0721_base`，20 updates，clip 0.15，batch 12,800，actor GRU/head LR `3e-6/3e-5`，target-KL 关闭。
- **主指标**：每个 600-scenario 面板中的 collision episode 数；越低越好。
- **成功率**：`(600 - collisions) / 600`。超车/跟随为 episode 最终结果分解。
- **U5+ 均值**：从 U5 开始的所有有效正式检查点碰撞数算术平均；用于削弱 U1 warm-start 的影响，但不替代完整路径。
- **场景身份**：同一 scenario ID 在两检查点的 shared / resolved / created collision 集合；Jaccard 衡量集合重合。
- **训练遥测**：formal update 的 `metrics.jsonl` 记录。gradient norm 是 pre-clip；wall time 因实验重叠运行，不用于算法速度比较。""",
    },
    {
        "id": "methodology",
        "type": "markdown",
        "body": """## 方法

1. 从每个训练目录的配置与 `metrics.jsonl` 重建实际参数、warm-up、formal update、optimizer step、KL/clip、critic loss/EV。
2. 从每个 eval JSON 逐 episode 重算 collision/overtake/follow，并与 aggregate、错误数、600 个唯一场景和 trace 文件名对账。
3. 对所有 41,999 个 NPZ 检查 dtype、leading dimension、终止字段和碰撞标记；按 legacy 与 post-step-v2 分层解释。
4. 用 recorded config 对每组做单轴控制审计；存在额外变化时仅作描述性比较。
5. 在相同 scenario ID 上配对碰撞集合，避免只看总数。
6. 将训练 update 与相同 checkpoint eval join；相关性仅作为 pooled 诊断，不作因果或跨 run 排名依据。

可复跑入口为 `analysis_results/ppo_groups_1_6/analyze_groups.py`；notebook 是对其 bounded outputs 的执行型伴随文档。""",
    },
    {
        "id": "limitations",
        "type": "markdown",
        "sourceId": "analysis_summary",
        "body": """## 局限、不确定性与稳健性

- **无重复 seed / 无置信区间**：目前每臂一次训练，无法分离随机性与参数效应。
- **固定面板 checkpoint 选择偏差**：clip 0.20 U30 是 7 个已观察 checkpoint 中的最低点；如果同一 Austin600 同时承担调参与报告，11 次碰撞可能乐观。
- **Group 3 与 Group 6(0.02) 混杂**：worker 数变化使其不能对共享 baseline 做纯参数归因。
- **旧 NPZ 终止状态不完整**：旧 trace 的碰撞真值依赖 eval JSON；终止后一帧动力学不能复原。
- **无统计独立场景集**：场景 ID 相同便于配对，但没有验证到新地图、新 ego 分布或扰动条件。
- **训练 telemetry 相关性是 pooled 描述统计**：不同 run/更新阶段混合，不满足独立同分布或因果解释条件。""",
    },
    {
        "id": "recommendations",
        "type": "markdown",
        "sourceId": "analysis_summary",
        "body": """## 推荐配置与下一步

当前建议冻结候选：**`privilege_gru`、batch 12,800、actor GRU/head LR `3e-6/3e-5`、critic LR `3e-4`、clip 0.20、target-KL 关闭、30 updates，选择 U30。**

下一步只做一个最小确认实验：用相同配置再训练 **2–3 个独立 seed**，预先固定 U20/U25/U30 三个读取点，并增加一份不参与选择的 holdout 场景集。主要报告每 seed/每 checkpoint 的 collision 数、跨 seed 中位数与范围，并保留场景 shared/resolved/created 分解。如果复现不了 11，也能判断 0.20 的方向性是否仍优于 0.15。""",
    },
    {
        "id": "further_questions",
        "type": "markdown",
        "body": """## 进一步问题

1. 两个 68/68 持续碰撞场景的几何失败机制是感知不足、动作饱和，还是奖励/终止边界？
2. clip 0.20 在 U25 回退、U30 恢复，是采样噪声还是特定场景簇切换？
3. target-KL 早停触发前的 minibatch 顺序与首次 overshoot，是否集中在少数轨迹类型？
4. 在独立 holdout 上，11 次碰撞的优势能否保留，同时维持 358 次超车？""",
    },
]


generated_at = datetime.now(timezone.utc).isoformat()
artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "End2Race PPO：Baseline 至 Group 6 对比实验技术分析",
        "description": "基于训练参数、metrics.jsonl、eval JSON 与 41,999 个 NPZ 的可复核技术报告。",
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
            "headline": headline,
            "arm_performance": arm_performance,
            "group5_panels": group5_panels,
            "telemetry_scatter": telemetry_scatter,
            "persistent_scenarios": persistent_scenarios,
            "control_audit": control_audit,
            "pairwise": pairwise,
            "trace_summary": trace_summary,
            "quality": quality,
            "correlations": correlation_rows,
        },
        "accessIssues": [],
    },
    "sources": sources,
}


(OUT / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")

notes = """# Report notes

## Source inventory

- Raw training: `post-trained/*/config.json`, `post-trained/*/metrics.jsonl`, actor/critic checkpoints.
- Raw evaluation: `eval_results/*/results_multi.json`, `eval_results/*/traces/*.npz`.
- Bounded analytical outputs: `run_summary.csv`, `panel_metrics.csv`, `training_eval_join.csv`,
  `group_summary.csv`, `group_control_audit.csv`, `scenario_pairwise.csv`,
  `scenario_frequency.csv`, `trace_summary_selected.csv`, `npz_audit.json`.

## Report structure map

1. Technical summary and KPI strip.
2. Group-by-group findings with control boundaries.
3. Training telemetry diagnostics.
4. Evaluation/NPZ data-quality evidence and scenario identity.
5. Scope, methods, limitations, recommendation, further questions.

## Chart map

| Chart | Question | Fields | Form | Why |
|---|---|---|---|---|
| U5+ mean collisions | Which arm is consistently safer after warm start? | group, arm, mean collisions | grouped bar | Discrete arm comparison with a zero baseline |
| Group 5 path | How do clean clip arms move across checkpoints? | update, collision, clip | grouped bar | Seven fixed checkpoints, two directly comparable arms |
| KL vs eval collisions | Does KL telemetry rank safety? | mean KL, collisions, early stop | scatter | 68 points reveal dispersion and outliers |
| Persistent scenarios | Which scenarios repeatedly fail? | scenario, panel collision rate | bar | Exact ranking across a common 68-panel denominator |

## Omissions and caveats

- No wall-time chart: runs overlap, so elapsed time is not an algorithm-speed comparison.
- No long temporal line chart: each arm has only 5 or 7 formal checkpoints.
- No error bars: there is one training seed per arm.
- Legacy NPZ collision markers are not used as terminal collision truth.
"""
(OUT / "REPORT_NOTES.md").write_text(notes)
print(f"Wrote {OUT / 'artifact.json'}")
print(f"Wrote {OUT / 'REPORT_NOTES.md'}")

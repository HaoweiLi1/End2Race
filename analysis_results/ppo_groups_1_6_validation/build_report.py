#!/usr/bin/env python3
"""Build the portable technical validation report for the Claude analysis."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_results" / "ppo_groups_1_6_validation"


def load_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def num(value, integer: bool = False):
    if value in (None, "", "None"):
        return None
    return int(float(value)) if integer else float(value)


def boolean(value) -> bool:
    return str(value).lower() == "true"


summary = json.loads((OUT / "validation_summary.json").read_text())
claims_raw = load_csv("claim_review.csv")
target_raw = load_csv("target_kl_steps.csv")
target_eval_raw = load_csv("target_kl_eval.csv")
paired_raw = load_csv("paired_scenario_tests.csv")
old_actor_raw = load_csv("old_vs_long_actor_diff.csv")
base_actor_raw = load_csv("base_vs_long015_actor_diff.csv")

claims = [
    {
        "claim": row["claim"],
        "assessment": row["assessment"],
        "severity": row["severity"],
        "evidence": row["evidence"],
        "required_revision": row["required_revision"],
    }
    for row in claims_raw
]

target_steps = [
    {
        "label": row["label"],
        "run": row["run"],
        "update": num(row["update"], True),
        "threshold": num(row["threshold"]),
        "steps_completed": num(row["steps_completed"], True),
        "steps_planned": num(row["steps_planned"], True),
        "early_stop": "早停" if boolean(row["early_stop"]) else "完整",
        "trigger_kl": num(row["trigger_kl"]),
        "approx_kl_mean": num(row["approx_kl_mean"]),
        "approx_kl_max": num(row["approx_kl_max"]),
        "rollout_policy_update": num(row["rollout_policy_update"], True),
        "checkpoint_update": num(row["checkpoint_update"], True),
        "rollout_collision_count": num(row["rollout_collision_count"], True),
        "rollout_episode_count": num(row["rollout_episode_count"], True),
    }
    for row in target_raw
]

target_eval = [
    {
        "update": num(row["update"], True),
        "collisions": num(row["collisions"], True),
        "opponent_collisions": num(row["opponent_collisions"], True),
        "ego_or_wall_collisions": num(row["ego_or_wall_collisions"], True),
        "mean_min_surface_distance_m": num(row["mean_min_surface_distance_m"]),
        "mean_speed_mps": num(row["mean_speed_mps"]),
        "scenarios": 600,
    }
    for row in target_eval_raw
]

paired = [
    {
        "comparison": row["comparison"],
        "before": num(row["before_collisions"], True),
        "after": num(row["after_collisions"], True),
        "shared": num(row["shared"], True),
        "resolved": num(row["resolved"], True),
        "created": num(row["created"], True),
        "net_change": num(row["net_change"], True),
        "p_unadjusted": num(row["paired_exact_p_unadjusted"]),
    }
    for row in paired_raw
]

actor_diff = []
for row in old_actor_raw:
    actor_diff.append(
        {
            "comparison": "旧 G3 clip 0.20 vs G5 long clip 0.20",
            "update": num(row["update"], True),
            "tensor_equal": "是" if boolean(row["tensor_equal"]) else "否",
            "max_abs_difference": num(row["max_abs_parameter_difference"]),
        }
    )
for row in base_actor_raw:
    actor_diff.append(
        {
            "comparison": "共享 baseline vs G5 long clip 0.15",
            "update": num(row["update"], True),
            "tensor_equal": "是" if boolean(row["tensor_equal"]) else "否",
            "max_abs_difference": num(row["max_abs_parameter_difference"]),
        }
    )

g3 = summary["g3_vs_g5"]
g3_evidence = [
    {"check": "env_workers", "旧 G3": g3["config_differences"]["env_workers"]["old_g3"], "G5 long": g3["config_differences"]["env_workers"]["g5_long"], "meaning": "进程拓扑不同"},
    {"check": "num_updates", "旧 G3": g3["config_differences"]["num_updates"]["old_g3"], "G5 long": g3["config_differences"]["num_updates"]["g5_long"], "meaning": "仅总训练长度不同；固定 schedule 下不改变前 20U"},
    {"check": "warm-up epochs", "旧 G3": g3["old_warmup"]["epochs"], "G5 long": g3["long_warmup"]["epochs"], "meaning": "actor 更新前 critic 数据已分叉"},
    {"check": "warm-up best epoch", "旧 G3": g3["old_warmup"]["best_epoch"], "G5 long": g3["long_warmup"]["best_epoch"], "meaning": "16 vs 2"},
    {"check": "warm-up best validation loss", "旧 G3": g3["old_warmup"]["best_validation_loss"], "G5 long": g3["long_warmup"]["best_validation_loss"], "meaning": "0.1697 vs 0.2982"},
    {"check": "warm-up rollout 1 episodes", "旧 G3": g3["old_warmup"]["rollout1_episode_count"], "G5 long": g3["long_warmup"]["rollout1_episode_count"], "meaning": "152 vs 153"},
    {"check": "first structural difference", "旧 G3": g3["first_structural_difference"]["old_steps"], "G5 long": g3["first_structural_difference"]["long_steps"], "meaning": "同场景第 4 条 episode：623 vs 628 steps"},
]

headline = [{
    "high_severity_issues": sum(row["severity"] == "high" for row in claims),
    "tkl004_steps_completed": summary["target_kl"]["tkl004_steps_completed"],
    "tkl004_steps_planned": 320,
    "tkl004_peak_collisions": max(row["collisions"] for row in target_eval),
    "tkl004_final_collisions": target_eval[-1]["collisions"],
    "tkl002_early_stops": summary["target_kl"]["tkl002_early_stop_updates"],
}]

sources = [
    {
        "id": "validation_summary",
        "label": "Claude analysis validation summary",
        "path": "analysis_results/ppo_groups_1_6_validation/validation_summary.json",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_json_auto('analysis_results/ppo_groups_1_6_validation/validation_summary.json');",
            "description": "Loads bounded results produced by the raw-artifact validation script.",
            "tables_used": ["post-trained/*/run_config.json", "post-trained/*/metrics.jsonl", "post-trained/*/episodes.jsonl", "eval_results/*/results_multi.json", "post-trained/*/checkpoints/*.pth"],
            "filters": ["Austin600", "formal checkpoints through U20; G5 through U30"],
            "metric_definitions": ["paired p-value is an exact two-sided binomial test on resolved versus created collision scenarios", "ego/wall-like collision means ego_collision=true and opponent_collision=false"],
        },
    },
    {
        "id": "claim_review",
        "label": "Claim-level validation table",
        "path": "analysis_results/ppo_groups_1_6_validation/claim_review.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6_validation/claim_review.csv', header = true);",
            "description": "Loads the reviewed Claude claims, severity and required wording corrections.",
            "tables_used": ["analysis_results/ppo_groups_1_6_validation/claim_review.csv"],
        },
    },
    {
        "id": "target_steps",
        "label": "Target-KL recorded optimizer steps",
        "path": "analysis_results/ppo_groups_1_6_validation/target_kl_steps.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6_validation/target_kl_steps.csv', header = true);",
            "description": "Loads the per-update target-KL threshold, completed steps, early-stop location and rollout/checkpoint labels.",
            "tables_used": ["post-trained/ppo_privilege_gru_0722_clip015_tkl002/metrics.jsonl", "post-trained/ppo_privilege_gru_0722_clip015_tkl004/metrics.jsonl"],
        },
    },
    {
        "id": "target_eval",
        "label": "Target-KL 0.04 Austin600 checkpoint results",
        "path": "analysis_results/ppo_groups_1_6_validation/target_kl_eval.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6_validation/target_kl_eval.csv', header = true);",
            "description": "Loads collision count, opponent/ego-wall split, proximity and speed at five formal checkpoints.",
            "tables_used": ["eval_results/ppo_privilege_gru_0722_clip015_tkl004_u*/multiagents/results_multi.json"],
        },
    },
    {
        "id": "paired_tests",
        "label": "Paired Austin600 scenario comparisons",
        "path": "analysis_results/ppo_groups_1_6_validation/paired_scenario_tests.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6_validation/paired_scenario_tests.csv', header = true);",
            "description": "Loads shared, resolved and created collision sets plus unadjusted exact paired p-values.",
            "tables_used": ["eval_results/*/multiagents/results_multi.json"],
        },
    },
    {
        "id": "actor_diff",
        "label": "Actor checkpoint tensor comparisons",
        "path": "analysis_results/ppo_groups_1_6_validation/old_vs_long_actor_diff.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6_validation/old_vs_long_actor_diff.csv', header = true) UNION ALL SELECT * FROM read_csv_auto('analysis_results/ppo_groups_1_6_validation/base_vs_long015_actor_diff.csv', header = true);",
            "description": "Loads tensor equality and maximum absolute actor-parameter differences at common checkpoints.",
            "tables_used": ["post-trained/*/checkpoints/actor_u*.pth"],
        },
    },
]

cards = [
    {
        "id": "issues_card",
        "description": "Methodology issues that materially affect decision wording.",
        "dataset": "headline",
        "sourceId": "validation_summary",
        "metrics": [{"label": "高严重度修正", "field": "high_severity_issues", "format": "number"}],
    },
    {
        "id": "steps_card",
        "description": "Target-KL 0.04 completed actor steps across 20 updates.",
        "dataset": "headline",
        "sourceId": "target_steps",
        "metrics": [
            {"label": "T-KL 0.04 actor steps", "field": "tkl004_steps_completed", "format": "number"},
            {"label": "计划", "field": "tkl004_steps_planned", "format": "number"},
        ],
    },
    {
        "id": "collapse_card",
        "description": "Austin600 target-KL 0.04 collision path.",
        "dataset": "headline",
        "sourceId": "target_eval",
        "metrics": [
            {"label": "峰值碰撞", "field": "tkl004_peak_collisions", "format": "number"},
            {"label": "U20", "field": "tkl004_final_collisions", "format": "number"},
        ],
    },
]

charts = [
    {
        "id": "target_step_chart",
        "title": "Target-KL 每个 update 的 actor optimizer steps",
        "subtitle": "每臂计划 16 steps/update；0.02 与 0.04 分别完成 206 与 203/320",
        "type": "bar",
        "dataset": "target_steps",
        "sourceId": "target_steps",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "update", "type": "nominal", "label": "Formal update"},
            "y": {"field": "steps_completed", "type": "quantitative", "label": "完成 steps"},
            "color": {"field": "label", "type": "nominal", "label": "Target-KL"},
            "tooltip": [
                {"field": "early_stop", "type": "nominal", "label": "状态"},
                {"field": "trigger_kl", "type": "quantitative", "label": "触发 KL"},
                {"field": "rollout_policy_update", "type": "quantitative", "label": "Rollout policy U"},
                {"field": "checkpoint_update", "type": "quantitative", "label": "Checkpoint U"},
            ],
        },
    },
    {
        "id": "target_eval_chart",
        "title": "Target-KL 0.04 的 Austin600 碰撞路径",
        "subtitle": "U10 达到 70 次碰撞，其中 17 次为 ego/wall-like；U20 仅部分恢复",
        "type": "bar",
        "dataset": "target_eval",
        "sourceId": "target_eval",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "update", "type": "nominal", "label": "Formal update"},
            "y": {"field": "collisions", "type": "quantitative", "label": "碰撞 / 600"},
            "tooltip": [
                {"field": "opponent_collisions", "type": "quantitative", "label": "车车碰撞"},
                {"field": "ego_or_wall_collisions", "type": "quantitative", "label": "ego/wall-like"},
                {"field": "mean_min_surface_distance_m", "type": "quantitative", "label": "平均 min surface"},
                {"field": "mean_speed_mps", "type": "quantitative", "label": "平均速度"},
            ],
        },
    },
]

tables = [
    {
        "id": "claim_table",
        "title": "Claude 关键结论核验",
        "subtitle": "按对最终参数决策的影响分级",
        "dataset": "claims",
        "sourceId": "claim_review",
        "columns": [
            {"field": "claim", "label": "原结论"},
            {"field": "assessment", "label": "核验"},
            {"field": "severity", "label": "严重度"},
            {"field": "evidence", "label": "证据"},
            {"field": "required_revision", "label": "建议改写"},
        ],
    },
    {
        "id": "g3_table",
        "title": "旧 G3 与 G5 long clip 0.20 的 update 前证据",
        "subtitle": "同一初始 actor；warm-up 阶段 clip 尚未参与 actor 更新",
        "dataset": "g3_evidence",
        "sourceId": "validation_summary",
        "columns": [
            {"field": "check", "label": "检查"},
            {"field": "旧 G3", "label": "旧 G3"},
            {"field": "G5 long", "label": "G5 long"},
            {"field": "meaning", "label": "解释"},
        ],
    },
    {
        "id": "actor_table",
        "title": "共同 checkpoints 的 actor 张量比较",
        "subtitle": "旧/新 clip 0.20 从 U1 已不同；baseline 与 long clip 0.15 到 U20 完全一致",
        "dataset": "actor_diff",
        "sourceId": "actor_diff",
        "columns": [
            {"field": "comparison", "label": "比较"},
            {"field": "update", "label": "U", "format": "number"},
            {"field": "tensor_equal", "label": "完全一致"},
            {"field": "max_abs_difference", "label": "最大参数差", "format": "number"},
        ],
    },
    {
        "id": "paired_table",
        "title": "Austin600 配对场景比较",
        "subtitle": "p 值未校正 checkpoint 选择，只用于替代错误的通用碰撞阈值",
        "dataset": "paired",
        "sourceId": "paired_tests",
        "columns": [
            {"field": "comparison", "label": "比较"},
            {"field": "before", "label": "前", "format": "number"},
            {"field": "after", "label": "后", "format": "number"},
            {"field": "shared", "label": "共同", "format": "number"},
            {"field": "resolved", "label": "消除", "format": "number"},
            {"field": "created", "label": "新增", "format": "number"},
            {"field": "p_unadjusted", "label": "配对 p（未校正）", "format": "number"},
        ],
    },
]

blocks = [
    {"id": "title", "type": "markdown", "layout": "full", "body": "# 六组 PPO 对照分析：Claude 结论核验与原因诊断"},
    {
        "id": "summary",
        "type": "markdown",
        "sourceId": "validation_summary",
        "body": """## 技术摘要

Claude 的**参数方向和大部分原始数字是正确的**，尤其是 `privilege_gru`、batch 12,800、中档 LR、Group 5 clip 0.20 和关闭 target-KL 的选择。但当前文本还不适合作为最终决策记录：它有两个会改变证据强度的高严重度方法问题——把固定 Austin600 当作独立二项采样并设置“差 8 才显著”的通用门槛，以及把 formal metrics 同一行的 rollout KPI 与 update 后 checkpoint eval 当成同一个 policy。

G3/G5 分叉可高置信归因到不同 worker 进程拓扑造成的训练数据流变化，但浅拷贝 planner 只是候选机制。target-KL 0.04 对关闭组是干净单轴失败；应从当前训练配方移除，但不需要删除可选实现。""",
    },
    {"id": "metrics", "type": "metric-strip", "cardIds": ["issues_card", "steps_card", "collapse_card"]},
    {
        "id": "key_findings",
        "type": "markdown",
        "sourceId": "claim_review",
        "body": """## 哪些结论可保留，哪些必须改写

可保留的是六组的方向性参数结论、G3/tkl002 的 worker 混杂、Group 5 的权威 clip 对照，以及 target-KL 0.04 的失败。必须改写的是“连续改善”“随机步长”“有害 advantage 方向”“保守跟车换 reward”“近不可解出生位”等机制性措辞；这些都超过了现有日志直接证明的范围。tkl002 早停数还应从 10/20 改为 **11/20**。""",
    },
    {"id": "claims", "type": "table", "tableId": "claim_table"},
    {
        "id": "g3_g5",
        "type": "markdown",
        "sourceId": "validation_summary",
        "body": """## G3 与 G5 的 clip 0.20 为什么前 20U 对不上

两条 run 不是同一条训练轨迹的 20U/30U 版本。旧 G3 使用 8 workers，G5 使用 12 workers；在 warm-up 第一轮、任何 actor 更新之前，同一初始 actor 和同一第一个场景的浮点结果已经不同，第 4 条同场景 episode 进一步出现 **623 vs 628 steps**。warm-up critic 因而训练 19 vs 5 epochs、best epoch 16 vs 2；U1 advantage 已不相同，之后差异累积。

`num_updates=20/30` 在当前常量 LR/clip schedule 下不会改变前 20U，真正可观察的有效差异是 worker/process 拓扑。共享 baseline 与 long clip 0.15 的 U1–U20 actor 张量完全一致，证明 20U→30U 延长本身可复现；旧/新 clip 0.20 在 U1 起就不一致。""",
    },
    {"id": "g3_evidence_block", "type": "table", "tableId": "g3_table"},
    {"id": "actor_block", "type": "table", "tableId": "actor_table"},
    {
        "id": "g3_mechanism",
        "type": "markdown",
        "body": """## Worker 敏感性的根因仍未定位

`CentralScheduleSubprocVecEnv` 会随 worker 数改变“哪些逻辑 env 同处一个进程”：8 workers 时 16 个 env 全部成对，12 workers 时只有 4 个进程承载两个 env。进程内确有 planner template cache、浅拷贝 planner 和 F110 `RaceCar` 类级共享扫描器，因此共享状态污染是合理排查方向；但静态阅读尚未找到必然被交叉修改的对象，已有同进程交替 step 探针也未复现。因此只能确认 **worker 数改变了数据流**，不能把根因写死为 planner 浅拷贝。""",
    },
    {
        "id": "target_mechanism",
        "type": "markdown",
        "sourceId": "validation_summary",
        "body": """## Target-KL 恶化来自粗粒度、路径依赖的优化截断

0.04 组与关闭组 warm-up 完全一致，是干净对照。门控在当前 minibatch 更新前计算 KL；超阈值表示**前面已经完成的 minibatch steps**把 policy 推过阈值，门控只能停止后续 steps，不能回滚。0.04 组在 U1/U2 都只完成 1/16 step，U8/U14 只完成 2/16；20U 合计 203/320。actor 更新预算因此高度不规则，同时 critic 仍完整训练五个 epochs，未来 rollout 与 minibatch RNG 路径也随之改变。

这解释了为什么它不是平滑的“小步保护”。它没有证明每次早停必然导致碰撞，但提供了最贴近代码与日志的机制链：少数已完成 step 产生大 KL → 门控截断且不回滚 → actor/critic 与未来采样路径分叉 → 固定 eval 面板暴露安全退化。""",
    },
    {"id": "target_steps_chart", "type": "chart", "chartId": "target_step_chart"},
    {
        "id": "target_eval_section",
        "type": "markdown",
        "sourceId": "target_eval",
        "body": """## Target-KL 0.04 的退化是 eval 中的真实结构变化

碰撞路径为 23→45→70→58→33。峰值 U10 的 70 次碰撞包含 17 次 ego/wall-like，平均最小表面距离降至 0.266 m；U20 仍有 33 次碰撞和 10 次 ego/wall-like。这里“贴墙”是失败症状，不是 target-KL 的直接物理原因。训练 rollout 也不能与同编号 eval 直接对齐：formal row U 的 rollout 来自 policy U−1，而 checkpoint 是 policy U。""",
    },
    {"id": "target_eval_chart_block", "type": "chart", "chartId": "target_eval_chart"},
    {
        "id": "statistics",
        "type": "markdown",
        "sourceId": "paired_tests",
        "body": """## 固定 Austin600 应按场景配对，而不是套通用 ±4.2 门槛

如果把 600 场景假设为 IID Bernoulli 样本，p≈3% 时计数标准差约 4.2；但这里每个模型面对的是同一组确定场景，真正的信息是哪些场景被消除、哪些被新增。base U20→long clip 0.20 U30 为 9 个消除、6 个新增，未校正配对 p=0.607，确实不足以证明两者有稳定差异；BC→long U30 为 17 个消除、6 个新增，未校正 p=0.0347，但还需要考虑多 checkpoint 选择与单 seed。结论可以是“候选更好”，不能写成通用的“差少于 8 一律不显著”。""",
    },
    {"id": "paired_block", "type": "table", "tableId": "paired_table"},
    {
        "id": "scope",
        "type": "markdown",
        "body": """## 范围、数据与定义

- 核验对象：用户粘贴的 Claude 六组综合分析。
- 主数据：13 个 PPO run 的配置、训练 metrics、episode records、actor checkpoints，以及 BC/PPO Austin600 eval JSON。
- `rollout_policy_update`：生成该 formal row rollout 数据的 actor 版本。
- `checkpoint_update`：该 formal row 完成 actor/critic 训练后保存的 checkpoint 版本。
- `resolved/created`：固定 scenario ID 在前后模型中的碰撞集合差。
- target-KL 门槛：`1.5 × target_kl`，0.02/0.04 对应 0.03/0.06。""",
    },
    {
        "id": "methodology",
        "type": "markdown",
        "body": """## 核验方法

1. 对 recorded config 做键级比较，不从 `run.sh` 注释推断实际参数。
2. 比较 warm-up 第一轮 episode records，确认分叉发生在 actor 更新之前。
3. 对共同 checkpoints 逐 tensor 比较 actor 权重。
4. 读取每个 target-KL formal row 的 threshold、触发位置和 completed/planned steps，并对照源代码门控顺序。
5. 由 eval JSON 重建固定场景碰撞集合、车车/ego-wall-like 分类及配对 exact test。
6. 所有机制结论按“已验证事实 / 最可能解释 / 未解决假设”分级。""",
    },
    {
        "id": "limitations",
        "type": "markdown",
        "body": """## 局限与稳健性边界

- 每臂仍只有一个 seed；target-KL 的失败只对当前配置、阈值与训练路径成立。
- 配对 p 值未校正多个 checkpoint 的选择，也不代表新地图/新 ego 分布。
- 旧 0721 NPZ 缺终止 post-step；碰撞真值取 `results_multi.json`。
- `ego_collision && !opp_collision` 是 wall-like 分类，未进一步做地图边界几何重建。
- run config 未记录训练时 Git SHA；Git 历史显示期间主要加入 target-KL/telemetry，target off 的 actor 更新语义应保持，但这不是完整 source snapshot 证明。
- worker 根因仍未通过能稳定复现的最小测试定位。""",
    },
    {
        "id": "recommendation",
        "type": "markdown",
        "sourceId": "validation_summary",
        "body": """## 是否移除 Target-KL

**从当前正式配方和后续对照矩阵移除：可以，而且应该。** 保持 `target_kl=None`，并停止继续运行当前 0.02/0.04 arms。

**从代码中删除可选能力：不建议。** 默认值已经是 `None`，关闭时没有行为影响；保留实现与 telemetry 便于以后在更小 actor LR、回滚式 KL 约束或不同 epoch/minibatch 设计下重新研究。现有证据证明的是这两个 target 在当前 PPO 配方中无收益，不能证明 target-KL 机制在所有设置下都有害。""",
    },
    {
        "id": "questions",
        "type": "markdown",
        "body": """## 下一步问题

1. worker 敏感性究竟来自 planner/template、F110 类级共享对象，还是底层数值/进程初始化？
2. 若要保留 KL 约束，应否改为 update 后检测并回滚、KL penalty，或缩小 actor LR，而不是只截断剩余 minibatches？
3. 将 rollout KPI 按 `rollout_policy_update` 重新对齐后，训练与 eval 背离是否仍同样明显？
4. long clip 0.20 在独立 seed 与 holdout 场景中能否保持对 clip 0.15 的优势？""",
    },
]

generated = datetime.now(timezone.utc).isoformat()
artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "六组 PPO 对照分析：Claude 结论核验与原因诊断",
        "description": "基于 recorded configs、metrics、episodes、actor tensors 与 Austin600 场景配对的技术核验。",
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
            "claims": claims,
            "target_steps": target_steps,
            "target_eval": target_eval,
            "paired": paired,
            "actor_diff": actor_diff,
            "g3_evidence": g3_evidence,
        },
        "accessIssues": [],
    },
    "sources": sources,
}

(OUT / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
(OUT / "REPORT_NOTES.md").write_text(
    """# Validation report notes

## Required structure map

Title; technical summary; key findings with chart/table evidence; scope and definitions;
methodology; limitations and robustness; recommendation; further questions.

## Chart map

| Section | Question | Family | Fields | Supported claim |
|---|---|---|---|---|
| Target-KL mechanism | How irregular is the optimization budget? | Grouped bar | update, steps, target | Both targets create state-dependent truncation |
| Target-KL eval | Where and how large is the safety collapse? | Bar | update, collision count | 0.04 peaks at 70 and only partly recovers |

## Omitted visuals

- G3/G5 uses tables because there are only two warm-up arms and five tensor checkpoints.
- Paired scenario comparisons use a table because shared/resolved/created and unadjusted p-values require exact lookup.
"""
)
print(f"Wrote {OUT / 'artifact.json'}")

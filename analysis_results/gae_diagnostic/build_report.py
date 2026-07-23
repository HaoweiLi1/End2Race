#!/usr/bin/env python3
"""Build the canonical portable report artifact for the GAE diagnosis."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_results" / "gae_diagnostic"


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str, integer: bool = False):
    return int(float(value)) if integer else float(value)


summary = json.loads((OUT / "diagnosis_summary.json").read_text())
windows = [
    {
        "window": row["window"],
        "updates": number(row["updates"], True),
        "overall_a2_proxy": number(row["overall_advantage_second_moment_proxy"]),
        "overall_rms_proxy": number(row["overall_advantage_rms_proxy"]),
        "collision_a2_proxy": number(row["collision_advantage_second_moment_proxy"]),
        "collision_rms_proxy": number(row["collision_advantage_rms_proxy"]),
        "ordinary_a2_proxy": number(row["ordinary_advantage_second_moment_proxy"]),
        "ordinary_rms_proxy": number(row["ordinary_advantage_rms_proxy"]),
        "collision_to_ordinary_ratio": number(row["collision_to_ordinary_second_moment_ratio"]),
        "overall_ev_pre": number(row["overall_ev_pre_mean"]),
        "overall_ev_post": number(row["overall_ev_post_mean"]),
        "collision_ev_pre": number(row["collision_ev_pre_mean"]),
        "collision_ev_post": number(row["collision_ev_post_mean"]),
        "ordinary_ev_pre": number(row["ordinary_ev_pre_mean"]),
        "ordinary_ev_post": number(row["ordinary_ev_post_mean"]),
        "mean_approx_kl": number(row["mean_approx_kl"]),
        "max_approx_kl": number(row["max_approx_kl"]),
        "mean_actor_grad_norm": number(row["mean_actor_grad_norm"]),
    }
    for row in read_csv("window_diagnostics.csv")
]
role_proxy = [
    {
        "window": row["window"],
        "role": row["role"],
        "advantage_second_moment_proxy": number(row["advantage_second_moment_proxy"]),
        "advantage_rms_proxy": number(row["advantage_rms_proxy"]),
    }
    for row in read_csv("role_advantage_proxy.csv")
]
lambda_summary = [
    {
        "gae_lambda": number(row["gae_lambda"]),
        "gamma_times_lambda": number(row["gamma_times_lambda"]),
        "half_life_s": number(row["td_residual_half_life_s"]),
        "geometric_horizon_s": number(row["geometric_horizon_s"]),
        "weight_1s": number(row["weight_1s"]),
        "weight_2s": number(row["weight_2s"]),
        "weight_4s": number(row["weight_4s"]),
        "weight_6s": number(row["weight_6s"]),
        "weight_8s": number(row["weight_8s"]),
    }
    for row in read_csv("lambda_sensitivity.csv")
]
lambda_curve = [
    {
        "gae_lambda": row["gae_lambda"],
        "seconds_before_td_residual": number(row["seconds_before_td_residual"]),
        "relative_weight": number(row["relative_weight"]),
    }
    for row in read_csv("lambda_weight_curve.csv")
]
collision_times = [
    {
        "window": row["window"],
        "collision_episodes": number(row["collision_episodes"], True),
        "mean_s": number(row["mean_collision_time_s"]),
        "p05_s": number(row["p05_collision_time_s"]),
        "p25_s": number(row["p25_collision_time_s"]),
        "median_s": number(row["median_collision_time_s"]),
        "p75_s": number(row["p75_collision_time_s"]),
        "p95_s": number(row["p95_collision_time_s"]),
    }
    for row in read_csv("collision_time_distributions.csv")
]
correlations = [
    {
        "updates": row["updates"],
        "source": row["source"],
        "target": row["target"],
        "pearson_r": number(row["pearson_r"]),
        "spearman_rho": number(row["spearman_rho"]),
    }
    for row in read_csv("proxy_correlations.csv")
]
availability = read_csv("data_availability.csv")

headline = [{
    "role_ratio": summary["u20_u30_collision_to_ordinary_second_moment_ratio"],
    "post_ev": summary["u20_u30_overall_ev_post"],
    "half_life_s": summary["current_half_life_s"],
    "weight_2s": summary["current_weight_2s"],
}]

sources = [
    {
        "id": "gae_code",
        "label": "GAE, recurrent timeout, and environment termination code",
        "path": "ppo/algorithm.py; ppo/environment.py; ppo/vec_env.py; installed sb3_contrib recurrent PPO",
        "query": {
            "description": "Static audit of GAE recursion inputs, episode boundaries, recurrent timeout bootstrap, and terminal/truncated flags.",
            "tables_used": ["ppo/algorithm.py", "ppo/environment.py", "ppo/vec_env.py", "stable_baselines3/common/buffers.py", "sb3_contrib/ppo_recurrent/ppo_recurrent.py"],
            "metric_definitions": ["GAE decay is gamma times lambda", "true terminals cut recursion", "time-limit truncations bootstrap the recurrent terminal value"],
        },
    },
    {
        "id": "window_diagnostics",
        "label": "Long clip0.20 update-level GAE proxies",
        "path": "analysis_results/gae_diagnostic/window_diagnostics.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/gae_diagnostic/window_diagnostics.csv', header = true);",
            "description": "Aggregates persisted U2-U30, U10-U30, and U20-U30 value-fit and actor telemetry.",
            "tables_used": ["post-trained/ppo_privilege_gru_0722_long_clip020/metrics.jsonl"],
            "filters": ["phase=formal", "gamma=0.999", "gae_lambda=0.995"],
            "metric_definitions": ["pre-update value loss is used as a second-moment proxy for raw advantage", "role ratio divides mean collision-role proxy by mean ordinary-role proxy"],
        },
    },
    {
        "id": "role_proxy",
        "label": "Role-level advantage second-moment proxy",
        "path": "analysis_results/gae_diagnostic/role_advantage_proxy.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/gae_diagnostic/role_advantage_proxy.csv', header = true);",
            "description": "Tidy role view derived from the same formal update telemetry.",
            "tables_used": ["post-trained/ppo_privilege_gru_0722_long_clip020/metrics.jsonl"],
            "filters": ["collision and ordinary transition roles", "equal role transition allocation"],
        },
    },
    {
        "id": "lambda_sensitivity",
        "label": "Deterministic gamma-lambda sensitivity calculation",
        "path": "analysis_results/gae_diagnostic/lambda_sensitivity.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/gae_diagnostic/lambda_sensitivity.csv', header = true);",
            "description": "Computes TD-residual weight decay at 100 simulator steps per second with gamma fixed at 0.999.",
            "tables_used": ["post-trained/ppo_privilege_gru_0722_long_clip020/run_config.json", "ppo/ppo_config.yaml"],
            "metric_definitions": ["half-life is log(0.5)/log(gamma*lambda)/100", "relative weight at d seconds is (gamma*lambda)^(100*d)"],
        },
    },
    {
        "id": "lambda_curve",
        "label": "GAE temporal-weight curve",
        "path": "analysis_results/gae_diagnostic/lambda_weight_curve.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/gae_diagnostic/lambda_weight_curve.csv', header = true);",
            "description": "Loads the half-second temporal-weight curve for the four inspected lambda values.",
            "tables_used": ["analysis_results/gae_diagnostic/lambda_weight_curve.csv"],
            "filters": ["gamma=0.999", "100 simulator steps per second"],
        },
    },
    {
        "id": "episode_timing",
        "label": "Completed training-episode collision timing",
        "path": "analysis_results/gae_diagnostic/collision_time_distributions.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/gae_diagnostic/collision_time_distributions.csv', header = true);",
            "description": "Summarizes persisted completed episodes by formal rollout window.",
            "tables_used": ["post-trained/ppo_privilege_gru_0722_long_clip020/episodes.jsonl"],
            "filters": ["phase=formal", "ego_collision=true"],
            "metric_definitions": ["completed episodes omit partial episodes crossing a rollout boundary"],
        },
    },
    {
        "id": "proxy_correlations",
        "label": "Advantage-proxy association with actor instability telemetry",
        "path": "analysis_results/gae_diagnostic/proxy_correlations.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/gae_diagnostic/proxy_correlations.csv', header = true);",
            "description": "Pearson and Spearman associations across 29 formal updates, used only as non-causal diagnostics.",
            "tables_used": ["post-trained/ppo_privilege_gru_0722_long_clip020/metrics.jsonl"],
            "filters": ["U2-U30"],
        },
    },
    {
        "id": "availability",
        "label": "GAE diagnostic evidence availability audit",
        "path": "analysis_results/gae_diagnostic/data_availability.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('analysis_results/gae_diagnostic/data_availability.csv', header = true);",
            "description": "Separates direct evidence, indirect proxies, and missing transition-grain fields.",
            "tables_used": ["analysis_results/gae_diagnostic/data_availability.csv"],
        },
    },
]

cards = [
    {
        "id": "ratio_card",
        "description": "U20-U30 collision-role versus ordinary-role raw advantage second-moment proxy.",
        "dataset": "headline",
        "sourceId": "window_diagnostics",
        "metrics": [{"label": "Collision / ordinary A2 proxy", "field": "role_ratio", "format": "number"}],
    },
    {
        "id": "ev_card",
        "description": "Mean post-update critic explained variance across U20-U30 on current lambda-return targets.",
        "dataset": "headline",
        "sourceId": "window_diagnostics",
        "metrics": [{"label": "Post-update EV", "field": "post_ev", "format": "percent"}],
    },
    {
        "id": "half_life_card",
        "description": "TD-residual weight half-life at gamma 0.999 and lambda 0.995 in the 100 Hz simulator.",
        "dataset": "headline",
        "sourceId": "lambda_sensitivity",
        "metrics": [{"label": "Current GAE half-life", "field": "half_life_s", "format": "number", "unit": "s"}],
    },
    {
        "id": "two_second_card",
        "description": "Relative contribution of a TD residual to an advantage two seconds earlier at current lambda.",
        "dataset": "headline",
        "sourceId": "lambda_sensitivity",
        "metrics": [{"label": "Current weight at 2 s", "field": "weight_2s", "format": "percent"}],
    },
]

charts = [
    {
        "id": "role_proxy_chart",
        "title": "Raw advantage energy is concentrated in collision-role transitions",
        "subtitle": "Pre-critic-update value-loss proxy across three late-training windows",
        "type": "bar",
        "dataset": "role_proxy",
        "sourceId": "role_proxy",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "window", "type": "nominal", "label": "Training window"},
            "y": {"field": "advantage_second_moment_proxy", "type": "quantitative", "label": "Advantage second-moment proxy"},
            "color": {"field": "role", "type": "nominal", "label": "Transition role"},
            "tooltip": [{"field": "advantage_rms_proxy", "type": "quantitative", "label": "RMS proxy"}],
        },
    },
    {
        "id": "lambda_decay_chart",
        "title": "Lower lambda sharply shortens backward TD-residual credit",
        "subtitle": "Relative GAE weight by seconds before a residual; gamma 0.999 at 100 Hz",
        "type": "line",
        "dataset": "lambda_curve",
        "sourceId": "lambda_curve",
        "valueFormat": "percent",
        "encodings": {
            "x": {"field": "seconds_before_td_residual", "type": "quantitative", "label": "Seconds before TD residual"},
            "y": {"field": "relative_weight", "type": "quantitative", "label": "Relative weight"},
            "color": {"field": "gae_lambda", "type": "nominal", "label": "GAE lambda"},
            "tooltip": [{"field": "relative_weight", "type": "quantitative", "label": "Relative weight"}],
        },
    },
]

tables = [
    {
        "id": "window_table",
        "title": "Late-training GAE proxy and critic fit",
        "subtitle": "Means over each update window; A2 is an indirect second-moment proxy",
        "dataset": "windows",
        "sourceId": "window_diagnostics",
        "defaultSort": {"field": "window", "direction": "asc"},
        "columns": [
            {"field": "window", "label": "window"},
            {"field": "updates", "label": "updates", "format": "number"},
            {"field": "collision_a2_proxy", "label": "collision A2", "format": "number"},
            {"field": "ordinary_a2_proxy", "label": "ordinary A2", "format": "number"},
            {"field": "collision_to_ordinary_ratio", "label": "role ratio", "format": "number"},
            {"field": "overall_ev_pre", "label": "EV pre", "format": "percent"},
            {"field": "overall_ev_post", "label": "EV post", "format": "percent"},
            {"field": "collision_ev_post", "label": "collision EV post", "format": "percent"},
            {"field": "ordinary_ev_post", "label": "ordinary EV post", "format": "percent"},
        ],
    },
    {
        "id": "lambda_table",
        "title": "GAE lambda sensitivity at gamma 0.999",
        "subtitle": "100 Hz temporal weights; higher values retain more distant TD residuals",
        "dataset": "lambda_summary",
        "sourceId": "lambda_sensitivity",
        "defaultSort": {"field": "gae_lambda", "direction": "asc"},
        "columns": [
            {"field": "gae_lambda", "label": "lambda", "format": "number"},
            {"field": "half_life_s", "label": "half-life s", "format": "number"},
            {"field": "geometric_horizon_s", "label": "horizon s", "format": "number"},
            {"field": "weight_1s", "label": "weight 1s", "format": "percent"},
            {"field": "weight_2s", "label": "weight 2s", "format": "percent"},
            {"field": "weight_4s", "label": "weight 4s", "format": "percent"},
            {"field": "weight_8s", "label": "weight 8s", "format": "percent"},
        ],
    },
    {
        "id": "collision_time_table",
        "title": "Completed collision timing remains broad",
        "subtitle": "Training collision episodes only; seconds from episode start",
        "dataset": "collision_times",
        "sourceId": "episode_timing",
        "defaultSort": {"field": "window", "direction": "asc"},
        "columns": [
            {"field": "window", "label": "window"},
            {"field": "collision_episodes", "label": "collisions", "format": "number"},
            {"field": "p05_s", "label": "p05 s", "format": "number"},
            {"field": "p25_s", "label": "p25 s", "format": "number"},
            {"field": "median_s", "label": "median s", "format": "number"},
            {"field": "p75_s", "label": "p75 s", "format": "number"},
            {"field": "p95_s", "label": "p95 s", "format": "number"},
        ],
    },
    {
        "id": "correlation_table",
        "title": "Advantage-energy proxies do not track actor spikes",
        "subtitle": "U2-U30 descriptive correlations; no causal interpretation",
        "dataset": "correlations",
        "sourceId": "proxy_correlations",
        "defaultSort": {"field": "source", "direction": "asc"},
        "columns": [
            {"field": "source", "label": "proxy"},
            {"field": "target", "label": "actor metric"},
            {"field": "pearson_r", "label": "Pearson r", "format": "number"},
            {"field": "spearman_rho", "label": "Spearman rho", "format": "number"},
        ],
    },
    {
        "id": "availability_table",
        "title": "Evidence coverage determines what can be concluded",
        "subtitle": "Direct, indirect, and missing evidence for a lambda decision",
        "dataset": "availability",
        "sourceId": "availability",
        "defaultSort": {"field": "status", "direction": "asc"},
        "columns": [
            {"field": "evidence", "label": "evidence"},
            {"field": "status", "label": "status"},
            {"field": "grain", "label": "grain"},
            {"field": "diagnostic_use", "label": "diagnostic use"},
        ],
    },
]

blocks = [
    {"id": "title", "type": "markdown", "layout": "full", "body": "# End2Race PPO GAE 诊断：当前不应直接修改 lambda"},
    {
        "id": "summary",
        "type": "markdown",
        "body": """## 技术结论

**没有发现GAE实现或timeout处理错误，也没有足够证据认定 `lambda=0.995` 是当前不稳定性的原因。** U20-U30中，collision-role原始advantage二阶矩代理为0.1435，ordinary-role为0.0246，相差5.83倍，说明碰撞数据确实承载更强或更分散的策略信号；但该代理与KL和actor梯度尖峰没有正相关。critic对当前lambda-return的U20-U30 post-update EV为0.925，collision-role为0.913，拟合并未失效。

因此本轮保持0.995。下一次rollout应在任何buffer重排之前只读记录raw advantages，并用同一批TD residual离线重算0.99/0.995/0.9975的counterfactual advantages；只有0.99明显削弱异常尾部、又不破坏碰撞前1-2秒的信用方向时，才值得开一条matched 0.99 ablation。""",
    },
    {"id": "metrics", "type": "metric-strip", "cardIds": ["ratio_card", "ev_card", "half_life_card", "two_second_card"]},
    {
        "id": "role_finding",
        "type": "markdown",
        "sourceId": "window_diagnostics",
        "body": """## 碰撞role的advantage能量显著更大，但不是恶化证据

formal update开始时，rollout buffer满足 `return = stored value + advantage`；critic尚未更新，重新计算的value应接近stored value，所以pre-update value MSE可作为 `E[A^2]` 的代理。U20-U30 collision/ordinary代理均值为0.1435/0.0246，RMS代理为0.379/0.157。

这说明50/50 transition mix中，collision-role会主导合并minibatch的advantage尺度与尾部，但actor随后按minibatch标准化advantage，所以二阶矩较大不会机械地造成更大梯度。没有raw mean、quantile和tail-energy前，不能把该差异等同于“GAE方差过高”。""",
    },
    {"id": "role_chart_block", "type": "chart", "chartId": "role_proxy_chart"},
    {"id": "window_table_block", "type": "table", "tableId": "window_table"},
    {
        "id": "credit_finding",
        "type": "markdown",
        "sourceId": "lambda_sensitivity",
        "body": """## 从0.995降到0.99不是小改动：2秒信用降至约三分之一

当前 `gamma*lambda=0.994005`，TD residual权重半衰期1.15秒：向前2秒仍保留30.0%，4秒保留9.0%。改为0.99后，半衰期缩至0.63秒，对应2秒10.97%、4秒1.20%。它可能降低碰撞role的长尾方差，但也会明显削弱驾驶动作对稍后碰撞的直接Monte Carlo信用。

privilege GRU critic可以通过bootstrap补偿部分远期信用，因此0.99并非不合理；只是不能根据高EV直接断言它更好，因为该EV本身是对当前0.995 lambda-return的拟合指标。""",
    },
    {"id": "lambda_chart_block", "type": "chart", "chartId": "lambda_decay_chart"},
    {"id": "lambda_table_block", "type": "table", "tableId": "lambda_table"},
    {
        "id": "correctness",
        "type": "markdown",
        "sourceId": "gae_code",
        "body": """## GAE递推、真终止和8秒timeout处理均正确

rollout使用标准反向递推 `A_t = delta_t + gamma*lambda*nonterminal*A_(t+1)`；下一episode通过episode-start标志切断。ego collision被标为terminated，不bootstrap；8秒结束被标为truncated，vector env保留terminal observation，recurrent PPO使用对应critic hidden state计算terminal value并加入timeout reward。3,204个formal完成episode走timeout路径，1,327个碰撞episode走真终止路径。

本次静态与日志核对没有发现GAE边界泄漏、把timeout误当真终止或碰撞后继续传播的问题。""",
    },
    {
        "id": "timing",
        "type": "markdown",
        "sourceId": "episode_timing",
        "body": """## 碰撞发生时间跨度宽，单看episode时刻无法选择lambda

全部formal碰撞episode的中位碰撞时刻为3.49秒，U20-U30为3.20秒，5%-95%约0.94-7.43秒。该分布说明既有近出生位碰撞，也有晚期交互失败；但GAE真正相关的是每个决策距离终止还剩多久，而现有episode日志没有transition索引，无法得到“碰撞前0.5/1/2/4秒”的advantage形状。""",
    },
    {"id": "collision_time_block", "type": "table", "tableId": "collision_time_table"},
    {
        "id": "instability",
        "type": "markdown",
        "sourceId": "proxy_correlations",
        "body": """## 现有代理不支持“0.995导致KL/梯度尖峰”

U2-U30中，overall advantage二阶矩代理与mean KL的Pearson r=-0.204、与mean actor gradient norm的r=0.071；collision-role代理对应为-0.148和0.020。样本仅29个updates、指标又经过minibatch normalization，不能证明无关系，但至少不存在应当驱动立即降lambda的正向共振证据。""",
    },
    {"id": "correlation_block", "type": "table", "tableId": "correlation_table"},
    {
        "id": "scope",
        "type": "markdown",
        "body": """## 诊断范围与定义

- 焦点run：`ppo_privilege_gru_0722_long_clip020`，30 updates、12 workers、gamma0.999、lambda0.995、clip0.20、target-KL关闭。
- 更新证据：30行formal metrics；主要晚期窗口U20-U30有11个updates。
- episode证据：4,531个formal完成episode；未完成并跨rollout的episode片段不在episodes.jsonl中，但仍存在于训练buffer。
- `A2 proxy`：critic更新前full-buffer value MSE。它近似raw advantage二阶矩，不提供均值、标准差、分位数、偏度或尾部集中度。
- 所有相关系数均为update级描述性诊断，不作因果解释。""",
    },
    {
        "id": "methodology",
        "type": "markdown",
        "body": """## 方法与稳健性检查

1. 从run config确认gamma、lambda、100Hz timestep和训练配置。
2. 审计环境terminated/truncated语义、vector-env terminal observation以及recurrent timeout bootstrap。
3. 用formal metrics重建U2/U10/U20开始的value-fit、role MSE、KL和gradient窗口统计。
4. 将pre-update role value loss作为 `E[A^2]` 代理，并检查它与KL、clip fraction、actor gradient的Pearson/Spearman关系。
5. 从完成episode计算return与碰撞时刻分布，明确它不能替代transition-level credit诊断。
6. 固定gamma0.999，在100Hz上精确计算候选lambda的TD-residual衰减曲线。""",
    },
    {
        "id": "limitations",
        "type": "markdown",
        "sourceId": "availability",
        "body": """## 当前证据足以排除实现错误，不足以选择新lambda

现有产物没有保存raw transition advantages、rewards、values、TD residuals，也没有把terminal outcome与buffer timestep/env rank对齐。因此无法直接测量tail ratio、sign balance、collision-lag或同rollout counterfactual lambda差异。post-update EV高只代表critic能拟合当前lambda目标；pre-update loss代理还混入recurrent replay的微小数值误差。

结论置信度为中等：保持0.995是有证据边界的保守决策，但它不是对0.99性能的否定。""",
    },
    {"id": "availability_block", "type": "table", "tableId": "availability_table"},
    {
        "id": "telemetry",
        "type": "markdown",
        "body": """## 下一条run应加入的最小只读诊断

在任何 `rollout_buffer.get()`、swap或flatten之前读取原始 `[time, env]` 数组，且不调用model forward、不消耗训练RNG：

1. 按collision/ordinary记录raw advantage的mean/std/RMS、p01/p05/p50/p95/p99、abs max、正负比例和top-1% squared-energy share。
2. 从当前advantage反推出同一rollout的TD residual：`delta_t = A_t - gamma*lambda*nonterminal*A_(t+1)`；episode末端和rollout最后一步令后项为0。
3. 用同一组delta离线递推0.99/0.995/0.9975 counterfactual advantages，比较RMS、尾部能量、rank correlation、sign flips和归一化后的cosine similarity。
4. 额外保存terminal transition的buffer timestep、env rank和outcome，才能按碰撞前0.5/1/2/4秒切片。
5. instrumentation加入后必须重新验证U1-U30 actor逐tensor复现；遥测不能成为45U A run的隐含第二变量。""",
    },
    {
        "id": "next_steps",
        "type": "markdown",
        "body": """## 决策

1. 45U复现run继续使用 `gae_lambda=0.995`。
2. 若严格bit-repro优先，先不改训练代码；若加入上述只读诊断，则把前30U逐tensor复现作为硬门槛。
3. 只有counterfactual 0.99显著压低collision-role异常尾部，同时不大面积改变归一化advantage符号、并保留碰撞前1-2秒方向时，才运行一条单轴0.99 ablation。
4. 暂不测试0.9975或1.0：它们把4秒TD residual权重提高到24.6%/67.0%，当前没有信用过短的证据。""",
    },
    {
        "id": "questions",
        "type": "markdown",
        "body": """## 仍待回答

1. collision-role的高二阶矩来自均值偏移、双峰混合，还是少量极端advantage？
2. 0.99重算后，变化集中在深碰撞尾部还是普遍改变正常超车决策？
3. KL尖峰updates的normalized advantage尾部、action log-ratio和场景组成是否同时异常？
4. hard-neighbor cache启用后，collision-role advantage分布是否改变，从而需要重新判断lambda？""",
    },
]

generated = datetime.now(timezone.utc).isoformat()
artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "End2Race PPO GAE 诊断：当前不应直接修改 lambda",
        "description": "基于long clip0.20训练日志、GAE代码路径、role value-fit代理和时间信用敏感性的技术诊断。",
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
            "windows": windows,
            "role_proxy": role_proxy,
            "lambda_summary": lambda_summary,
            "lambda_curve": lambda_curve,
            "collision_times": collision_times,
            "correlations": correlations,
            "availability": availability,
        },
        "accessIssues": [],
    },
    "sources": sources,
}

(OUT / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
(OUT / "REPORT_NOTES.md").write_text(
    """# GAE diagnostic report notes

## Required structure map

Title; technical conclusion; key findings with role-proxy and temporal-decay visuals; scope and definitions;
methodology; limitations and evidence availability; recommended telemetry and decision; further questions.

## Chart map

| Section | Question | Family | Fields | Supported claim | Palette |
|---|---|---|---|---|---|
| Role signal | How different is raw advantage energy by transition role and window? | Grouped bar | window, role, A2 proxy | Collision role carries materially larger raw advantage energy | Hard two-root comparator |
| Credit span | How strongly does lambda change backward TD-residual weight? | Multi-series line | seconds, relative weight, lambda | 0.99 sharply shortens 1-4 second credit versus 0.995 | Ordered categorical series |

## Caveat

The role chart uses an indirect pre-update value-loss proxy rather than persisted raw advantages. The line chart is a
deterministic sensitivity calculation, not observed performance. Browser QA may be structural-only if Chromium is absent.
"""
)
print(f"Wrote {OUT / 'artifact.json'}")

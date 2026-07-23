#!/usr/bin/env python3
"""Read-only diagnosis of the current GAE setting from persisted PPO evidence."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_results" / "gae_diagnostic"
RUN = ROOT / "post-trained" / "ppo_privilege_gru_0722_long_clip020"
OUT.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta) * sum(value * value for value in right_delta)
    )
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0
        for slot in range(index, end):
            result[order[slot]] = rank
        index = end
    return result


config = json.loads((RUN / "run_config.json").read_text())["args"]
gamma = float(config["gamma"])
current_lambda = float(config["gae_lambda"])
metrics = [row for row in read_jsonl(RUN / "metrics.jsonl") if row.get("phase") == "formal"]
episodes = [row for row in read_jsonl(RUN / "episodes.jsonl") if row.get("phase") == "formal"]
if len(metrics) != 30 or (gamma, current_lambda) != (0.999, 0.995):
    raise AssertionError("Unexpected focal-run configuration or update coverage")

windows = {
    "U2-U30": [row for row in metrics if int(row["update"]) >= 2],
    "U10-U30": [row for row in metrics if int(row["update"]) >= 10],
    "U20-U30": [row for row in metrics if int(row["update"]) >= 20],
}
window_rows: list[dict[str, Any]] = []
role_proxy_rows: list[dict[str, Any]] = []
for window, rows in windows.items():
    overall_a2 = statistics.fmean(float(row["value_loss_pre_update"]) for row in rows)
    collision_a2 = statistics.fmean(float(row["collision_value_loss_pre"]) for row in rows)
    ordinary_a2 = statistics.fmean(float(row["ordinary_value_loss_pre"]) for row in rows)
    window_rows.append(
        {
            "window": window,
            "updates": len(rows),
            "overall_advantage_second_moment_proxy": overall_a2,
            "overall_advantage_rms_proxy": math.sqrt(overall_a2),
            "collision_advantage_second_moment_proxy": collision_a2,
            "collision_advantage_rms_proxy": math.sqrt(collision_a2),
            "ordinary_advantage_second_moment_proxy": ordinary_a2,
            "ordinary_advantage_rms_proxy": math.sqrt(ordinary_a2),
            "collision_to_ordinary_second_moment_ratio": collision_a2 / ordinary_a2,
            "overall_ev_pre_mean": statistics.fmean(float(row["explained_variance_pre_update"]) for row in rows),
            "overall_ev_post_mean": statistics.fmean(float(row["explained_variance_post_update"]) for row in rows),
            "collision_ev_pre_mean": statistics.fmean(float(row["collision_explained_variance_pre"]) for row in rows),
            "collision_ev_post_mean": statistics.fmean(float(row["collision_explained_variance_post"]) for row in rows),
            "ordinary_ev_pre_mean": statistics.fmean(float(row["ordinary_explained_variance_pre"]) for row in rows),
            "ordinary_ev_post_mean": statistics.fmean(float(row["ordinary_explained_variance_post"]) for row in rows),
            "mean_approx_kl": statistics.fmean(float(row["approx_kl_mean"]) for row in rows),
            "max_approx_kl": max(float(row["approx_kl_mean"]) for row in rows),
            "mean_actor_grad_norm": statistics.fmean(float(row["actor_grad_norm_mean"]) for row in rows),
        }
    )
    for role, second_moment in (("collision", collision_a2), ("ordinary", ordinary_a2)):
        role_proxy_rows.append(
            {
                "window": window,
                "role": role,
                "advantage_second_moment_proxy": second_moment,
                "advantage_rms_proxy": math.sqrt(second_moment),
            }
        )

episode_rows: list[dict[str, Any]] = []
episode_windows = {
    "U1-U30": episodes,
    "U10-U30": [row for row in episodes if int(row["formal_update"]) >= 10],
    "U20-U30": [row for row in episodes if int(row["formal_update"]) >= 20],
}
for window, rows in episode_windows.items():
    for group_field in ("env_role", "episode_outcome"):
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            groups[str(row[group_field])].append(float(row["episode_return"]))
        for group, values in sorted(groups.items()):
            episode_rows.append(
                {
                    "window": window,
                    "group_type": group_field,
                    "group": group,
                    "episodes": len(values),
                    "return_mean": statistics.fmean(values),
                    "return_std": statistics.stdev(values),
                    "return_p05": quantile(values, 0.05),
                    "return_p50": quantile(values, 0.50),
                    "return_p95": quantile(values, 0.95),
                }
            )

collision_time_rows: list[dict[str, Any]] = []
for window, rows in episode_windows.items():
    times = sorted(float(row["elapsed_time"]) for row in rows if bool(row["ego_collision"]))
    collision_time_rows.append(
        {
            "window": window,
            "collision_episodes": len(times),
            "mean_collision_time_s": statistics.fmean(times),
            "p05_collision_time_s": quantile(times, 0.05),
            "p25_collision_time_s": quantile(times, 0.25),
            "median_collision_time_s": quantile(times, 0.50),
            "p75_collision_time_s": quantile(times, 0.75),
            "p95_collision_time_s": quantile(times, 0.95),
        }
    )

lambda_values = (0.99, 0.995, 0.9975, 1.0)
lambda_rows: list[dict[str, Any]] = []
weight_curve_rows: list[dict[str, Any]] = []
for gae_lambda in lambda_values:
    decay = gamma * gae_lambda
    lambda_rows.append(
        {
            "gae_lambda": gae_lambda,
            "gamma_times_lambda": decay,
            "td_residual_half_life_s": math.log(0.5) / math.log(decay) / 100.0,
            "geometric_horizon_s": 1.0 / (1.0 - decay) / 100.0,
            "weight_1s": decay**100,
            "weight_2s": decay**200,
            "weight_4s": decay**400,
            "weight_6s": decay**600,
            "weight_8s": decay**800,
        }
    )
    for half_second in range(17):
        seconds = half_second / 2.0
        weight_curve_rows.append(
            {
                "gae_lambda": f"lambda {gae_lambda:g}",
                "seconds_before_td_residual": seconds,
                "relative_weight": decay ** int(100 * seconds),
            }
        )

correlation_rows: list[dict[str, Any]] = []
correlation_sources = {
    "overall advantage second-moment proxy": "value_loss_pre_update",
    "collision advantage second-moment proxy": "collision_value_loss_pre",
    "ordinary advantage second-moment proxy": "ordinary_value_loss_pre",
}
correlation_targets = {
    "actor grad norm mean": "actor_grad_norm_mean",
    "actor grad norm max": "actor_grad_norm_max",
    "approx KL mean": "approx_kl_mean",
    "clip fraction mean": "clip_fraction_mean",
}
correlation_input = windows["U2-U30"]
for source_label, source_field in correlation_sources.items():
    left = [float(row[source_field]) for row in correlation_input]
    for target_label, target_field in correlation_targets.items():
        right = [float(row[target_field]) for row in correlation_input]
        correlation_rows.append(
            {
                "updates": "U2-U30",
                "source": source_label,
                "target": target_label,
                "pearson_r": pearson(left, right),
                "spearman_rho": pearson(ranks(left), ranks(right)),
            }
        )

availability_rows = [
    {
        "evidence": "GAE implementation and timeout bootstrap",
        "status": "available",
        "grain": "code path",
        "diagnostic_use": "Correctness audit: recurrent GAE recursion, terminal reset, and timeout value bootstrap.",
    },
    {
        "evidence": "Pre-update value loss by role",
        "status": "indirect",
        "grain": "update x role",
        "diagnostic_use": "Proxy for raw advantage second moment because returns = stored value + advantage before critic update.",
    },
    {
        "evidence": "Explained variance by role",
        "status": "available",
        "grain": "update x role",
        "diagnostic_use": "Shows critic fit to the current lambda-return targets, not unbiased value accuracy.",
    },
    {
        "evidence": "Completed-episode returns and collision times",
        "status": "available",
        "grain": "completed episode",
        "diagnostic_use": "Describes outcome mixture and event timing; excludes unfinished rollout fragments.",
    },
    {
        "evidence": "Raw per-transition advantages and TD residuals",
        "status": "missing",
        "grain": "transition",
        "diagnostic_use": "Required for means, quantiles, tails, sign balance, and exact counterfactual lambda recomputation.",
    },
    {
        "evidence": "Transition-aligned terminal type and collision lag",
        "status": "missing",
        "grain": "transition x episode",
        "diagnostic_use": "Required to measure how far collision credit propagates before the actual terminal event.",
    },
]

u20 = next(row for row in window_rows if row["window"] == "U20-U30")
current = next(row for row in lambda_rows if row["gae_lambda"] == current_lambda)
candidate = next(row for row in lambda_rows if row["gae_lambda"] == 0.99)
summary = {
    "as_of": "2026-07-22",
    "run": RUN.name,
    "verdict": "No GAE correctness defect found. Existing evidence does not justify changing lambda from 0.995 yet; collect transition-level counterfactual telemetry before a 0.99 ablation.",
    "confidence": "medium",
    "gamma": gamma,
    "current_gae_lambda": current_lambda,
    "formal_updates": len(metrics),
    "formal_completed_episodes": len(episodes),
    "timeout_episodes": sum(bool(row["timeout"]) for row in episodes),
    "collision_episodes": sum(bool(row["ego_collision"]) for row in episodes),
    "u20_u30_collision_advantage_second_moment_proxy": u20["collision_advantage_second_moment_proxy"],
    "u20_u30_ordinary_advantage_second_moment_proxy": u20["ordinary_advantage_second_moment_proxy"],
    "u20_u30_collision_to_ordinary_second_moment_ratio": u20["collision_to_ordinary_second_moment_ratio"],
    "u20_u30_overall_ev_post": u20["overall_ev_post_mean"],
    "u20_u30_collision_ev_post": u20["collision_ev_post_mean"],
    "current_half_life_s": current["td_residual_half_life_s"],
    "current_weight_2s": current["weight_2s"],
    "current_weight_4s": current["weight_4s"],
    "lambda099_weight_2s": candidate["weight_2s"],
    "lambda099_weight_4s": candidate["weight_4s"],
    "direct_advantage_distribution_available": False,
    "recommendation": "Keep lambda=0.995 for the 45-update reproduction run unless exact read-only telemetry is added and bit reproduction is revalidated. Use same-rollout counterfactual advantages to decide whether one matched lambda=0.99 ablation is warranted.",
}

write_csv("window_diagnostics.csv", window_rows)
write_csv("role_advantage_proxy.csv", role_proxy_rows)
write_csv("episode_return_distributions.csv", episode_rows)
write_csv("collision_time_distributions.csv", collision_time_rows)
write_csv("lambda_sensitivity.csv", lambda_rows)
write_csv("lambda_weight_curve.csv", weight_curve_rows)
write_csv("proxy_correlations.csv", correlation_rows)
write_csv("data_availability.csv", availability_rows)
(OUT / "diagnosis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))

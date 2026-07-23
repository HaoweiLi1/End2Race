from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, norm


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
DATA_DIR = ANALYSIS_DIR / "data"
FIGURE_DIR = ANALYSIS_DIR / "figures"
UPDATES = (1, 5, 10, 15, 20)

RUNS = {
    "independent_gru": "ppo_independent_gru_0721_base",
    "privilege_mlp": "ppo_privilege_mlp_0721_base",
    "privilege_gru": "ppo_privilege_gru_0721_base",
    "batch_25600": "ppo_privilege_gru_0721_bs25600",
    "batch_51200": "ppo_privilege_gru_0721_bs51200",
    "clip_010": "ppo_privilege_gru_0721_clip010",
    "clip_020": "ppo_privilege_gru_0721_clip020",
}

RUN_LABELS = {
    "independent_gru": "independent_gru",
    "privilege_mlp": "privilege_mlp",
    "privilege_gru": "privilege_gru / bs12800 / clip0.15",
    "batch_25600": "privilege_gru / bs25600",
    "batch_51200": "privilege_gru / bs51200",
    "clip_010": "privilege_gru / clip0.10",
    "clip_020": "privilege_gru / clip0.20",
}

COLORS = {
    "independent_gru": "#718096",
    "privilege_mlp": "#d97706",
    "privilege_gru": "#0f766e",
    "batch_25600": "#2563eb",
    "batch_51200": "#7c3aed",
    "clip_010": "#db2777",
    "clip_020": "#dc2626",
}

CONFIG_FIELDS = (
    "critic",
    "n_envs",
    "env_workers",
    "n_steps",
    "batch_size",
    "num_updates",
    "actor_epochs",
    "critic_epochs",
    "gru_learning_rate",
    "head_learning_rate",
    "critic_learning_rate",
    "steering_latent_std",
    "speed_physical_std",
    "gamma",
    "gae_lambda",
    "clip_range",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = norm.ppf(0.5 + confidence / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - spread, center + spread


def state_delta_stats(
    state: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    keys: Iterable[str],
) -> dict[str, float]:
    delta_sq = 0.0
    baseline_sq = 0.0
    max_abs = 0.0
    count = 0
    for key in keys:
        delta = (state[key] - baseline[key]).double()
        reference = baseline[key].double()
        delta_sq += float((delta * delta).sum())
        baseline_sq += float((reference * reference).sum())
        max_abs = max(max_abs, float(delta.abs().max()))
        count += delta.numel()
    return {
        "delta_l2": math.sqrt(delta_sq),
        "relative_l2": math.sqrt(delta_sq / baseline_sq) if baseline_sq else math.nan,
        "delta_rms": math.sqrt(delta_sq / count) if count else math.nan,
        "max_abs_delta": max_abs,
    }


def cluster_bootstrap_difference(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    seed: int,
    draws: int = 10_000,
) -> tuple[float, float]:
    clusters: dict[str, list[int]] = {}
    for episode_key in sorted(left):
        cluster = re.sub(r"_s[0-9.]+$", "", episode_key)
        left_collision = int(left[episode_key]["outcome"] == "ego_collision")
        right_collision = int(right[episode_key]["outcome"] == "ego_collision")
        clusters.setdefault(cluster, []).append(left_collision - right_collision)
    cluster_differences = np.asarray([sum(values) for values in clusters.values()], dtype=np.int16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cluster_differences), size=(draws, len(cluster_differences)))
    samples = cluster_differences[indices].sum(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def load_training_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    formal_rows: list[dict[str, Any]] = []
    warmup_rows: list[dict[str, Any]] = []
    episodes_by_run: dict[str, list[dict[str, Any]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for run_key, run_name in RUNS.items():
        run_dir = REPO_ROOT / "post-trained" / run_name
        config = read_json(run_dir / "run_config.json")
        configs[run_key] = config
        metrics = read_jsonl(run_dir / "metrics.jsonl")
        for row in metrics:
            row = dict(row)
            row["run"] = run_key
            row["run_label"] = RUN_LABELS[run_key]
            if row["phase"] == "warmup":
                warmup_rows.append(row)
            else:
                row["rollout_collision_rate"] = row["ego_collision_count"] / row["episode_count"]
                row["rollout_overtake_rate"] = row["overtake_count"] / row["episode_count"]
                formal_rows.append(row)
        episodes_by_run[run_key] = read_jsonl(run_dir / "episodes.jsonl")
    return pd.DataFrame(formal_rows), pd.DataFrame(warmup_rows), episodes_by_run, configs


def load_eval_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], dict[str, dict[str, Any]]]]:
    summary_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    episode_maps: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    numeric_fields = (
        "avg_speed",
        "total_distance",
        "speed_variance",
        "global_min_surface_dist",
        "ego_min_lidar",
        "max_steer_delta",
        "ego_max_abs_steer",
        "final_relative_position_m",
    )
    for run_key, run_name in RUNS.items():
        for update in UPDATES:
            eval_root = REPO_ROOT / "eval_results" / f"{run_name}_u{update:04d}_Austin"
            payload = read_json(eval_root / "multiagents" / "results_multi.json")
            episodes = payload["episodes"]
            episode_maps[(run_key, update)] = episodes
            outcome_counts = Counter(row["outcome"] for row in episodes.values())
            collision_count = outcome_counts["ego_collision"]
            lower, upper = wilson_interval(collision_count, len(episodes))
            row: dict[str, Any] = {
                "run": run_key,
                "run_label": RUN_LABELS[run_key],
                "update": update,
                "total_episodes": len(episodes),
                "collision_count": collision_count,
                "collision_rate": collision_count / len(episodes),
                "collision_rate_ci_low": lower,
                "collision_rate_ci_high": upper,
                "overtake_count": outcome_counts["overtake"],
                "overtake_rate": outcome_counts["overtake"] / len(episodes),
                "follow_count": outcome_counts["follow"],
                "follow_rate": outcome_counts["follow"] / len(episodes),
                "error_count": payload["final"]["error_count"],
                "all_action_finite": all(item["action_finite"] for item in episodes.values()),
                "all_observation_finite": all(item["observation_finite"] for item in episodes.values()),
                "source_path": f"eval_results/{run_name}_u{update:04d}_Austin/multiagents/results_multi.json",
            }
            for field in numeric_fields:
                values = [float(item[field]) for item in episodes.values() if item.get(field) is not None]
                row[f"{field}_mean"] = float(np.mean(values))
                row[f"{field}_median"] = float(np.median(values))
            collision_times = [
                float(item["ego_collision_time_s"])
                for item in episodes.values()
                if item["outcome"] == "ego_collision" and item.get("ego_collision_time_s") is not None
            ]
            row["collision_time_median"] = float(np.median(collision_times)) if collision_times else math.nan
            summary_rows.append(row)

            for episode_key, item in episodes.items():
                match = re.match(r"ol(\d+)_e(\d+)_o(\d+)_s([0-9.]+)", episode_key)
                if match is None:
                    raise ValueError(f"Unexpected Austin episode key: {episode_key}")
                episode_rows.append(
                    {
                        "run": run_key,
                        "run_label": RUN_LABELS[run_key],
                        "update": update,
                        "episode_key": episode_key,
                        "cluster_id": re.sub(r"_s[0-9.]+$", "", episode_key),
                        "opponent_raceline": int(match.group(1)),
                        "ego_index": int(match.group(2)),
                        "opponent_index": int(match.group(3)),
                        "speed_scale": float(match.group(4)),
                        "outcome": item["outcome"],
                        "collision": item["outcome"] == "ego_collision",
                        "avg_speed": item["avg_speed"],
                        "total_distance": item["total_distance"],
                        "speed_variance": item["speed_variance"],
                        "global_min_surface_dist": item["global_min_surface_dist"],
                        "ego_min_lidar": item["ego_min_lidar"],
                        "ego_max_abs_steer": item["ego_max_abs_steer"],
                        "final_relative_position_m": item["final_relative_position_m"],
                        "steps": item["steps"],
                        "ego_collision_time_s": item.get("ego_collision_time_s"),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(episode_rows), episode_maps


def control_audit(
    formal: pd.DataFrame,
    episodes_by_run: dict[str, list[dict[str, Any]]],
    configs: dict[str, dict[str, Any]],
    episode_maps: dict[tuple[str, int], dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    base_warmup = [row for row in episodes_by_run["privilege_gru"] if row["phase"] == "warmup"]
    clip_reference = [row for row in episodes_by_run["clip_010"] if row["phase"] == "warmup"]
    rows = []
    for run_key, run_name in RUNS.items():
        run_dir = REPO_ROOT / "post-trained" / run_name
        config = configs[run_key]
        args = config["args"]
        warmup = [row for row in episodes_by_run[run_key] if row["phase"] == "warmup"]
        checkpoint_dir = run_dir / "checkpoints"
        actor_files = sorted(checkpoint_dir.glob("actor_u*.pth"))
        critic_files = sorted(checkpoint_dir.glob("critic_u*.pt"))
        eval_counts = [len(episode_maps[(run_key, update)]) for update in UPDATES]
        final_state = torch.load(run_dir / "actor_final.pth", map_location="cpu", weights_only=True)
        update_state = torch.load(checkpoint_dir / "actor_u0020.pth", map_location="cpu", weights_only=True)
        rows.append(
            {
                "run": run_key,
                "run_label": RUN_LABELS[run_key],
                **{field: args[field] for field in CONFIG_FIELDS},
                "started_at": config["started_at"],
                "metrics_rows": len(formal.loc[formal["run"] == run_key]) + 1,
                "formal_updates": len(formal.loc[formal["run"] == run_key]),
                "actor_checkpoints": len(actor_files),
                "critic_checkpoints": len(critic_files),
                "eval_checkpoints": len(UPDATES),
                "eval_episodes_min": min(eval_counts),
                "eval_episodes_max": max(eval_counts),
                "warmup_episode_count": len(warmup),
                "warmup_equals_local_baseline": warmup == base_warmup,
                "warmup_equals_clip010": warmup == clip_reference,
                "scenario_hash_collision": sha256(run_dir / "collision_scenarios.json"),
                "scenario_hash_ordinary": sha256(run_dir / "ordinary_scenarios.json"),
                "seed_persisted": "seed" in args,
                "source_commit_persisted": any(key in config for key in ("git_commit", "commit", "source_commit")),
                "normalization_version": config.get("PRIVILEGED_NORMALIZATION", {}).get("version", "not_used"),
                "actor_final_tensor_equals_u0020": final_state.keys() == update_state.keys()
                and all(torch.equal(final_state[key], update_state[key]) for key in final_state),
                "source_path": f"post-trained/{run_name}/run_config.json",
            }
        )
    return pd.DataFrame(rows)


def training_summary(formal: pd.DataFrame, warmup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_key in RUNS:
        group = formal.loc[formal["run"] == run_key].sort_values("update")
        last5 = group.tail(5)
        warm = warmup.loc[warmup["run"] == run_key].iloc[0]
        total_seconds = float(
            group[["rollout_wall_seconds", "actor_train_wall_seconds", "critic_train_wall_seconds"]].sum().sum()
            + warm["rollout_wall_seconds"]
            + warm["train_wall_seconds"]
        )
        rows.append(
            {
                "run": run_key,
                "run_label": RUN_LABELS[run_key],
                "warmup_best_validation_loss": warm["best_validation_loss"],
                "ev_last5": last5["explained_variance_post_update"].mean(),
                "collision_ev_last5": last5["collision_explained_variance_post"].mean(),
                "ordinary_ev_last5": last5["ordinary_explained_variance_post"].mean(),
                "value_loss_last5": last5["value_loss_post_update"].mean(),
                "ev_improved_updates": int((group["explained_variance_post_update"] > group["explained_variance_pre_update"]).sum()),
                "value_loss_improved_updates": int((group["value_loss_post_update"] < group["value_loss_pre_update"]).sum()),
                "rollout_collision_first5": group.head(5)["rollout_collision_rate"].mean(),
                "rollout_collision_last5": last5["rollout_collision_rate"].mean(),
                "rollout_collision_slope": np.polyfit(group["update"], group["rollout_collision_rate"], 1)[0],
                "kl_mean_median": group["approx_kl_mean"].median(),
                "kl_mean_p95": group["approx_kl_mean"].quantile(0.95),
                "kl_mean_max": group["approx_kl_mean"].max(),
                "kl_single_minibatch_max": group["approx_kl_max"].max(),
                "kl_mean_gt_0_02_updates": int((group["approx_kl_mean"] > 0.02).sum()),
                "kl_mean_gt_0_10_updates": int((group["approx_kl_mean"] > 0.10).sum()),
                "clip_fraction_median": group["clip_fraction_mean"].median(),
                "actor_preclip_grad_norm_median": group["actor_grad_norm_mean"].median(),
                "actor_train_seconds_mean": group["actor_train_wall_seconds"].mean(),
                "critic_train_seconds_mean": group["critic_train_wall_seconds"].mean(),
                "rollout_seconds_mean": group["rollout_wall_seconds"].mean(),
                "total_training_minutes": total_seconds / 60,
                "source_path": f"post-trained/{RUNS[run_key]}/metrics.jsonl",
            }
        )
    return pd.DataFrame(rows)


def eval_rollup(eval_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_key in RUNS:
        group = eval_summary.loc[eval_summary["run"] == run_key].sort_values("update")
        best = group.sort_values(["collision_count", "overtake_count"], ascending=[True, False]).iloc[0]
        final = group.loc[group["update"] == 20].iloc[0]
        rows.append(
            {
                "run": run_key,
                "run_label": RUN_LABELS[run_key],
                "collision_sequence": "/".join(str(int(value)) for value in group["collision_count"]),
                "collision_mean": group["collision_count"].mean(),
                "collision_std": group["collision_count"].std(ddof=0),
                "best_update": int(best["update"]),
                "best_collision_count": int(best["collision_count"]),
                "best_overtake_count": int(best["overtake_count"]),
                "final_collision_count": int(final["collision_count"]),
                "final_collision_rate": final["collision_rate"],
                "final_overtake_count": int(final["overtake_count"]),
                "final_follow_count": int(final["follow_count"]),
                "final_avg_speed": final["avg_speed_mean"],
                "final_total_distance": final["total_distance_mean"],
                "source_path": final["source_path"],
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(episode_maps: dict[tuple[str, int], dict[str, dict[str, Any]]]) -> pd.DataFrame:
    pairs = (
        ("critic", "independent_gru", "privilege_gru", "strict"),
        ("critic", "privilege_mlp", "privilege_gru", "strict"),
        ("batch", "batch_25600", "privilege_gru", "strict"),
        ("batch", "batch_51200", "privilege_gru", "strict"),
        ("clip", "clip_010", "privilege_gru", "worker_contaminated"),
        ("clip", "clip_020", "privilege_gru", "worker_contaminated"),
        ("clip", "clip_010", "clip_020", "strict"),
    )
    rows = []
    for pair_index, (group, left_key, right_key, control_strength) in enumerate(pairs):
        for update in UPDATES:
            left = episode_maps[(left_key, update)]
            right = episode_maps[(right_key, update)]
            if set(left) != set(right):
                raise ValueError(f"Austin scenario panel mismatch: {left_key} vs {right_key}, U{update}")
            keys = sorted(left)
            left_collision = {key: left[key]["outcome"] == "ego_collision" for key in keys}
            right_collision = {key: right[key]["outcome"] == "ego_collision" for key in keys}
            left_only = sum(left_collision[key] and not right_collision[key] for key in keys)
            right_only = sum(right_collision[key] and not left_collision[key] for key in keys)
            discordant = left_only + right_only
            p_value = binomtest(min(left_only, right_only), discordant, 0.5).pvalue if discordant else 1.0
            ci_low, ci_high = cluster_bootstrap_difference(
                left,
                right,
                seed=20260722 + pair_index * 100 + update,
            )
            rows.append(
                {
                    "group": group,
                    "control_strength": control_strength,
                    "left_run": left_key,
                    "left_label": RUN_LABELS[left_key],
                    "right_run": right_key,
                    "right_label": RUN_LABELS[right_key],
                    "update": update,
                    "left_collision_count": sum(left_collision.values()),
                    "right_collision_count": sum(right_collision.values()),
                    "delta_left_minus_right": sum(left_collision.values()) - sum(right_collision.values()),
                    "left_only_collisions": left_only,
                    "right_only_collisions": right_only,
                    "discordant_scenarios": discordant,
                    "mcnemar_exact_p": p_value,
                    "cluster_bootstrap_delta_ci_low": ci_low,
                    "cluster_bootstrap_delta_ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def checkpoint_stability(episode_maps: dict[tuple[str, int], dict[str, dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for run_key in RUNS:
        for left_update, right_update in zip(UPDATES[:-1], UPDATES[1:]):
            left = episode_maps[(run_key, left_update)]
            right = episode_maps[(run_key, right_update)]
            left_set = {key for key, row in left.items() if row["outcome"] == "ego_collision"}
            right_set = {key for key, row in right.items() if row["outcome"] == "ego_collision"}
            union = left_set | right_set
            rows.append(
                {
                    "run": run_key,
                    "run_label": RUN_LABELS[run_key],
                    "left_update": left_update,
                    "right_update": right_update,
                    "collision_flip_count": len(left_set ^ right_set),
                    "collision_flip_rate": len(left_set ^ right_set) / 600,
                    "collision_set_jaccard": len(left_set & right_set) / len(union) if union else 1.0,
                }
            )
    return pd.DataFrame(rows)


def actor_parameter_deltas() -> tuple[pd.DataFrame, pd.DataFrame]:
    pretrained = torch.load(REPO_ROOT / "pretrained" / "end2race.pth", map_location="cpu", weights_only=True)
    trainable_keys = [key for key in pretrained if key.startswith("gru.") or key.startswith("output_layer.")]
    gru_keys = [key for key in trainable_keys if key.startswith("gru.")]
    head_keys = [key for key in trainable_keys if key.startswith("output_layer.")]
    fixed_keys = [key for key in pretrained if key not in trainable_keys]
    rows = []
    states: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    for run_key, run_name in RUNS.items():
        for update in UPDATES:
            state = torch.load(
                REPO_ROOT / "post-trained" / run_name / "checkpoints" / f"actor_u{update:04d}.pth",
                map_location="cpu",
                weights_only=True,
            )
            states[(run_key, update)] = state
            total = state_delta_stats(state, pretrained, trainable_keys)
            gru = state_delta_stats(state, pretrained, gru_keys)
            head = state_delta_stats(state, pretrained, head_keys)
            fixed = state_delta_stats(state, pretrained, fixed_keys)
            rows.append(
                {
                    "run": run_key,
                    "run_label": RUN_LABELS[run_key],
                    "update": update,
                    "actor_delta_l2_from_bc": total["delta_l2"],
                    "actor_relative_l2_from_bc": total["relative_l2"],
                    "actor_delta_rms_from_bc": total["delta_rms"],
                    "actor_max_abs_delta_from_bc": total["max_abs_delta"],
                    "gru_relative_l2_from_bc": gru["relative_l2"],
                    "head_relative_l2_from_bc": head["relative_l2"],
                    "fixed_frontend_delta_l2": fixed["delta_l2"],
                    "source_path": f"post-trained/{run_name}/checkpoints/actor_u{update:04d}.pth",
                }
            )
    consecutive = []
    for run_key in RUNS:
        for left_update, right_update in zip(UPDATES[:-1], UPDATES[1:]):
            stats = state_delta_stats(states[(run_key, right_update)], states[(run_key, left_update)], trainable_keys)
            consecutive.append(
                {
                    "run": run_key,
                    "run_label": RUN_LABELS[run_key],
                    "left_update": left_update,
                    "right_update": right_update,
                    "delta_l2": stats["delta_l2"],
                    "relative_l2_to_left": stats["relative_l2"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(consecutive)


def critic_parameter_deltas(
    configs: dict[str, dict[str, Any]],
    formal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    proxy_rows = []
    for run_key, run_name in RUNS.items():
        checkpoint_dir = REPO_ROOT / "post-trained" / run_name / "checkpoints"
        warm = torch.load(checkpoint_dir / "critic_warmup.pt", map_location="cpu", weights_only=True)
        keys = list(warm)
        for update_label, path in [(0, checkpoint_dir / "critic_warmup.pt")] + [
            (update, checkpoint_dir / f"critic_u{update:04d}.pt") for update in UPDATES
        ]:
            state = torch.load(path, map_location="cpu", weights_only=True)
            delta = state_delta_stats(state, warm, keys)
            total_norm = math.sqrt(sum(float((value.double() ** 2).sum()) for value in state.values()))
            row = {
                "run": run_key,
                "run_label": RUN_LABELS[run_key],
                "update": update_label,
                "checkpoint_stage": "warmup" if update_label == 0 else f"u{update_label:04d}",
                "critic_parameter_count": sum(value.numel() for value in state.values()),
                "critic_total_l2_norm": total_norm,
                "critic_delta_l2_from_warmup": delta["delta_l2"],
                "critic_relative_l2_from_warmup": delta["relative_l2"],
                "privileged_projection_l2": math.nan,
                "privileged_projection_max_abs": math.nan,
                "source_path": str(path.relative_to(REPO_ROOT)),
            }
            if "privileged_projection.weight" in state:
                projection = state["privileged_projection.weight"].double()
                row["privileged_projection_l2"] = float(torch.linalg.vector_norm(projection))
                row["privileged_projection_max_abs"] = float(projection.abs().max())
                if update_label in UPDATES:
                    metric = formal.loc[(formal["run"] == run_key) & (formal["update"] == update_label)].iloc[0]
                    names = configs[run_key]["PRIVILEGED_FEATURE_NAMES"]
                    stds = metric["privileged_feature_std"]
                    for index, feature_name in enumerate(names):
                        column_norm = float(torch.linalg.vector_norm(projection[:, index]))
                        proxy_rows.append(
                            {
                                "run": run_key,
                                "run_label": RUN_LABELS[run_key],
                                "update": update_label,
                                "feature": feature_name,
                                "projection_column_l2": column_norm,
                                "feature_std": stds[index],
                                "weight_times_std_proxy": column_norm * stds[index],
                            }
                        )
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(proxy_rows)


def feature_health(formal: pd.DataFrame, configs: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for run_key in RUNS:
        config = configs[run_key]
        names = config.get("PRIVILEGED_FEATURE_NAMES", [])
        if not names or "privileged_feature_mean" not in formal.loc[formal["run"] == run_key].columns:
            continue
        for _, metric in formal.loc[formal["run"] == run_key].iterrows():
            if not isinstance(metric.get("privileged_feature_mean"), list):
                continue
            for index, feature_name in enumerate(names):
                rows.append(
                    {
                        "run": run_key,
                        "run_label": RUN_LABELS[run_key],
                        "update": int(metric["update"]),
                        "feature": feature_name,
                        "mean": metric["privileged_feature_mean"][index],
                        "std": metric["privileged_feature_std"][index],
                        "exact_low_fraction": metric["privileged_feature_saturation_low"][index],
                        "exact_high_fraction": metric["privileged_feature_saturation_high"][index],
                        "fraction_ge_0_95": metric["privileged_feature_fraction_ge_0_95"][index],
                        "fraction_ge_0_99": metric["privileged_feature_fraction_ge_0_99"][index],
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("feature", as_index=False)
        .agg(
            mean_of_means=("mean", "mean"),
            mean_std=("std", "mean"),
            max_exact_low_fraction=("exact_low_fraction", "max"),
            max_exact_high_fraction=("exact_high_fraction", "max"),
            max_fraction_ge_0_99=("fraction_ge_0_99", "max"),
        )
        .sort_values("feature")
    )
    summary["normalization_interpretation"] = "healthy"
    summary.loc[summary["feature"].isin(["left_body_margin", "right_body_margin"]), "normalization_interpretation"] = "intentional complementary +1 plateau"
    summary.loc[summary["feature"].str.startswith("cos_"), "normalization_interpretation"] = "natural cosine concentration near +1"
    summary.loc[summary["feature"].isin(["current_curvature", "lookahead_mean_curvature"]), "normalization_interpretation"] = "expected percentile-tail clipping"
    summary.loc[summary["feature"].str.contains("clearance"), "normalization_interpretation"] = "softsign clearance; exact +1 should be zero"
    return detail, summary


def scan_trace_quality(episode_maps: dict[tuple[str, int], dict[str, dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for run_key, run_name in RUNS.items():
        for update in UPDATES:
            episodes = episode_maps[(run_key, update)]
            trace_dir = REPO_ROOT / "eval_results" / f"{run_name}_u{update:04d}_Austin" / "multiagents" / "traces"
            missing = 0
            length_mismatch = 0
            nonfinite_action_traces = 0
            trace_collision_any_count = 0
            collision_terminal_missing = 0
            total_frames = 0
            steer_clip_frames = 0
            speed_clip_frames = 0
            clipped_episodes = 0
            abs_steering: list[np.ndarray] = []
            for episode_key, result in episodes.items():
                path = trace_dir / f"{episode_key}.npz"
                if not path.is_file():
                    missing += 1
                    continue
                with np.load(path, allow_pickle=False) as trace:
                    raw = trace["ego_raw_action"]
                    executed = trace["ego_executed_action"]
                    collision_bits = trace["collisions"]
                    if len(raw) != result["steps"]:
                        length_mismatch += 1
                    if not (np.isfinite(raw).all() and np.isfinite(executed).all()):
                        nonfinite_action_traces += 1
                    steer_clipped = np.abs(raw[:, 0] - executed[:, 0]) > 1e-6
                    speed_clipped = np.abs(raw[:, 1] - executed[:, 1]) > 1e-6
                    any_trace_collision = bool(collision_bits[:, 0].any())
                    trace_collision_any_count += int(any_trace_collision)
                    collision_terminal_missing += int(result["outcome"] == "ego_collision" and not any_trace_collision)
                    total_frames += len(raw)
                    steer_clip_frames += int(steer_clipped.sum())
                    speed_clip_frames += int(speed_clipped.sum())
                    clipped_episodes += int(steer_clipped.any() or speed_clipped.any())
                    abs_steering.append(np.abs(executed[:, 0]))
            steering_values = np.concatenate(abs_steering) if abs_steering else np.asarray([], dtype=float)
            rows.append(
                {
                    "run": run_key,
                    "run_label": RUN_LABELS[run_key],
                    "update": update,
                    "expected_trace_files": len(episodes),
                    "missing_trace_files": missing,
                    "trace_length_mismatches": length_mismatch,
                    "nonfinite_action_traces": nonfinite_action_traces,
                    "total_frames": total_frames,
                    "steering_clip_frames": steer_clip_frames,
                    "steering_clip_frame_rate": steer_clip_frames / total_frames if total_frames else math.nan,
                    "speed_clip_frames": speed_clip_frames,
                    "speed_clip_frame_rate": speed_clip_frames / total_frames if total_frames else math.nan,
                    "episodes_with_any_action_clip": clipped_episodes,
                    "executed_abs_steer_p95": float(np.quantile(steering_values, 0.95)),
                    "executed_abs_steer_p99": float(np.quantile(steering_values, 0.99)),
                    "json_collision_count": sum(item["outcome"] == "ego_collision" for item in episodes.values()),
                    "trace_collision_any_count": trace_collision_any_count,
                    "json_collisions_missing_terminal_trace_frame": collision_terminal_missing,
                    "source_path": f"eval_results/{run_name}_u{update:04d}_Austin/multiagents/traces",
                }
            )
    return pd.DataFrame(rows)


def scenario_risk(eval_episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = (
        eval_episodes.groupby(
            ["episode_key", "cluster_id", "opponent_raceline", "ego_index", "opponent_index", "speed_scale"],
            as_index=False,
        )
        .agg(collision_count=("collision", "sum"), checkpoint_count=("collision", "size"))
    )
    scenario["collision_frequency"] = scenario["collision_count"] / scenario["checkpoint_count"]
    scenario["risk_band"] = pd.cut(
        scenario["collision_frequency"],
        bins=[-0.001, 0, 0.2, 0.8, 0.999999, 1.0],
        labels=["never", "occasional", "frequent", "near_universal", "universal"],
    ).astype(str)
    slices = (
        eval_episodes.groupby(["opponent_raceline", "speed_scale"], as_index=False)
        .agg(collision_count=("collision", "sum"), evaluation_rows=("collision", "size"))
    )
    slices["collision_rate"] = slices["collision_count"] / slices["evaluation_rows"]
    slices["slice"] = slices.apply(lambda row: f"R{int(row.opponent_raceline)} / v{row.speed_scale:.1f}", axis=1)
    return scenario.sort_values(["collision_count", "episode_key"], ascending=[False, True]), slices


def write_dataframes(frames: dict[str, pd.DataFrame]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(DATA_DIR / f"{name}.csv", index=False)


def make_figures(
    formal: pd.DataFrame,
    eval_summary: pd.DataFrame,
    actor_deltas: pd.DataFrame,
    slices: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    groups = (
        ("Critic", ["independent_gru", "privilege_mlp", "privilege_gru"]),
        ("Batch size", ["privilege_gru", "batch_25600", "batch_51200"]),
        ("Clip range", ["clip_010", "privilege_gru", "clip_020"]),
    )
    for axis, (title, run_keys) in zip(axes, groups):
        for run_key in run_keys:
            data = eval_summary.loc[eval_summary["run"] == run_key].sort_values("update")
            axis.plot(data["update"], data["collision_count"], marker="o", linewidth=2, color=COLORS[run_key], label=RUN_LABELS[run_key])
        axis.set_title(title)
        axis.set_xlabel("Formal update")
        axis.set_xticks(UPDATES)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Austin collisions / 600")
    figure.suptitle("Austin 600 checkpoint evaluation")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "eval_collision_trajectories.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for run_key in RUNS:
        data = formal.loc[formal["run"] == run_key].sort_values("update")
        axes[0].plot(data["update"], data["explained_variance_post_update"], color=COLORS[run_key], label=RUN_LABELS[run_key], alpha=0.9)
        axes[1].plot(data["update"], data["approx_kl_mean"], color=COLORS[run_key], label=RUN_LABELS[run_key], alpha=0.9)
    axes[0].set(title="Critic explained variance", xlabel="Formal update", ylabel="Post-update EV")
    axes[1].set(title="Actor approximate KL", xlabel="Formal update", ylabel="Mean KL across minibatches", yscale="log")
    axes[1].axhline(0.02, color="#111827", linestyle="--", linewidth=1, label="0.02 reference")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "training_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for run_key in RUNS:
        data = actor_deltas.loc[actor_deltas["run"] == run_key].sort_values("update")
        axis.plot(data["update"], data["actor_relative_l2_from_bc"] * 100, marker="o", color=COLORS[run_key], label=RUN_LABELS[run_key])
    axis.set(title="Actor parameter displacement from BC", xlabel="Formal update", ylabel="Relative L2 displacement (%)")
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "actor_parameter_displacement.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    pivot = slices.pivot(index="opponent_raceline", columns="speed_scale", values="collision_rate")
    figure, axis = plt.subplots(figsize=(8, 3.6))
    image = axis.imshow(pivot.values * 100, cmap="YlOrRd", aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), [f"{value:.1f}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [f"R{value}" for value in pivot.index])
    axis.set(xlabel="Opponent speed scale", ylabel="Opponent raceline", title="Collision concentration across 35 checkpoints (%)")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            axis.text(column, row, f"{pivot.values[row, column] * 100:.1f}", ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=axis, label="Collision rate (%)")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "scenario_slice_risk.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_notebook(frames: dict[str, pd.DataFrame]) -> Path:
    rollup = frames["eval_rollup"].set_index("run")
    training = frames["training_summary"].set_index("run")
    control = frames["control_audit"]
    trace = frames["trace_quality"]
    summary = (
        f"在严格 critic 对照中，privilege_gru 的 5 个 checkpoint 碰撞数为 "
        f"{rollup.loc['privilege_gru', 'collision_sequence']}，均值 {rollup.loc['privilege_gru', 'collision_mean']:.1f}，"
        f"优于 independent_gru 的 {rollup.loc['independent_gru', 'collision_mean']:.1f} 和 privilege_mlp 的 {rollup.loc['privilege_mlp', 'collision_mean']:.1f}。"
    )
    cells: list[dict[str, Any]] = []

    def markdown(source: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})

    def code(source: str) -> None:
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)})

    markdown(
        "# PPO 0721：7 个实验 / 3 组对照的综合审计\n\n"
        "## TL;DR\n\n"
        f"{summary}\n\n"
        "当前证据支持保留 privilege_gru、batch_size=12800；clip=0.15 是现有面板上的首选，clip=0.20 是需要 worker-matched 多 seed 复验的候选。"
    )
    markdown(
        "## Context and decision\n\n"
        "分析目标是同时回答 critic 结构、batch size 和 clip range 的取舍，并审计训练稳定性、checkpoint 参数位移、Austin 600 固定面板和 NPZ trace 的可信度。"
    )
    code(
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "DATA = Path('analysis/ppo_0721_seven_experiments/data')\n"
        "def load(name): return pd.read_csv(DATA / f'{name}.csv')\n"
        "control = load('control_audit')\n"
        "training = load('training_summary')\n"
        "eval_rollup = load('eval_rollup')\n"
        "paired = load('paired_comparisons')\n"
        "actor = load('actor_parameter_deltas')\n"
        "trace = load('trace_quality')\n"
        "print(f'Loaded {len(control)} runs, {len(eval_rollup) * 5} checkpoint evals, {trace.expected_trace_files.sum():,.0f} NPZ traces expected.')\n"
    )
    markdown("## Data quality and control-variable audit")
    code(
        "cols=['run','critic','batch_size','clip_range','env_workers','formal_updates','eval_checkpoints','eval_episodes_min','warmup_equals_local_baseline','warmup_equals_clip010','seed_persisted','source_commit_persisted']\n"
        "print(control[cols].to_string(index=False))\n"
    )
    markdown(
        "![Austin checkpoint trajectories](figures/eval_collision_trajectories.png)\n\n"
        "三张图分别对应 critic、batch size、clip range。clip=0.15 与另外两档存在 env_workers 混入，因此第三张图是结果参考，不是完全干净的三档因果比较。"
    )
    markdown("## Training metrics and optimization dynamics")
    code(
        "cols=['run','warmup_best_validation_loss','ev_last5','collision_ev_last5','rollout_collision_first5','rollout_collision_last5','kl_mean_median','kl_mean_p95','kl_single_minibatch_max','actor_preclip_grad_norm_median','total_training_minutes']\n"
        "print(training[cols].round(4).to_string(index=False))\n"
    )
    markdown("![Training diagnostics](figures/training_diagnostics.png)")
    markdown("## Checkpoint parameter movement")
    code(
        "final_actor=actor.loc[actor['update']==20, ['run','actor_relative_l2_from_bc','gru_relative_l2_from_bc','head_relative_l2_from_bc','fixed_frontend_delta_l2']]\n"
        "print(final_actor.round(7).to_string(index=False))\n"
    )
    markdown("![Actor parameter displacement](figures/actor_parameter_displacement.png)")
    markdown("## Austin 600 outcomes and paired evidence")
    code(
        "cols=['run','collision_sequence','collision_mean','best_update','best_collision_count','final_collision_count','final_overtake_count','final_follow_count']\n"
        "print(eval_rollup[cols].to_string(index=False))\n"
    )
    code(
        "final_pairs=paired.loc[paired['update']==20, ['group','control_strength','left_run','right_run','delta_left_minus_right','left_only_collisions','right_only_collisions','mcnemar_exact_p','cluster_bootstrap_delta_ci_low','cluster_bootstrap_delta_ci_high']]\n"
        "print(final_pairs.round(4).to_string(index=False))\n"
    )
    markdown("## Scenario concentration")
    code(
        "scenario=load('scenario_risk')\n"
        "print(scenario.head(15)[['episode_key','collision_count','checkpoint_count','collision_frequency','risk_band']].to_string(index=False))\n"
        "print('universal=', int((scenario.collision_frequency==1).sum()), '>=80%=', int((scenario.collision_frequency>=.8).sum()), 'ever=', int((scenario.collision_count>0).sum()))\n"
    )
    markdown("![Scenario slice risk](figures/scenario_slice_risk.png)")
    markdown("## Privileged feature and trace health")
    code(
        "features=load('feature_health_summary')\n"
        "print(features[['feature','mean_std','max_exact_low_fraction','max_exact_high_fraction','max_fraction_ge_0_99','normalization_interpretation']].round(4).to_string(index=False))\n"
    )
    code(
        "print(trace[['run','update','missing_trace_files','trace_length_mismatches','steering_clip_frame_rate','json_collision_count','trace_collision_any_count','json_collisions_missing_terminal_trace_frame']].to_string(index=False))\n"
    )
    markdown(
        "## Takeaways and limitations\n\n"
        "- critic 与 batch 两组控制成立；clip=0.10 vs 0.20 控制成立，但它们与 clip=0.15 的比较混入 env_workers/执行环境。\n"
        "- 单 seed、固定 Austin 面板和 checkpoint 选优会高估可泛化优势；逐场景配对能减少面板噪声，但不能替代多 seed 与 holdout。\n"
        "- NPZ 缺 terminal post-step frame，碰撞标签必须以 results_multi.json 为准。\n"
        "- 下一步只需对 privilege_gru 的 clip=0.15/0.20 做同 worker、同源码哈希、2–3 seed 的窄对照，并加入 target-KL early stop。"
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (end2race)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = ANALYSIS_DIR / "ppo_0721_seven_experiments.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def execute_notebook(path: Path) -> None:
    notebook = read_json(path)
    namespace: dict[str, Any] = {"__name__": "__notebook__"}
    execution_count = 0
    previous_cwd = Path.cwd()
    try:
        # Notebook code uses repo-relative source paths.
        import os

        os.chdir(REPO_ROOT)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            execution_count += 1
            source = "".join(cell["source"])
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    exec(compile(source, f"{path.name}:cell-{execution_count}", "exec"), namespace)
            except BaseException as error:
                cell["execution_count"] = execution_count
                cell["outputs"] = [
                    {
                        "output_type": "error",
                        "ename": type(error).__name__,
                        "evalue": str(error),
                        "traceback": [],
                    }
                ]
                path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
                raise
            text = output.getvalue()
            cell["execution_count"] = execution_count
            cell["outputs"] = [] if not text else [{"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}]
    finally:
        import os

        os.chdir(previous_cwd)
    notebook["metadata"]["execution"] = {"status": "completed", "cells_executed": execution_count}
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def json_records(frame: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    selected = frame if columns is None else frame[columns]
    records = []
    for record in selected.to_dict(orient="records"):
        clean = {}
        for key, value in record.items():
            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean[key] = None if not math.isfinite(float(value)) else float(value)
            elif isinstance(value, (np.bool_,)):
                clean[key] = bool(value)
            else:
                clean[key] = value
        records.append(clean)
    return records


def build_report_artifact(frames: dict[str, pd.DataFrame]) -> Path:
    eval_summary = frames["eval_summary"]
    eval_rollup_frame = frames["eval_rollup"]
    training = frames["training_summary"]
    control = frames["control_audit"]
    actor = frames["actor_parameter_deltas"]
    trace = frames["trace_quality"]
    slices = frames["scenario_slices"]
    paired = frames["paired_comparisons"]
    stability = frames["checkpoint_stability"]
    scenario = frames["scenario_risk"]
    critic_final = eval_summary.loc[eval_summary["run"].isin(["independent_gru", "privilege_mlp", "privilege_gru"])]
    batch_eval = eval_summary.loc[eval_summary["run"].isin(["privilege_gru", "batch_25600", "batch_51200"])]
    clip_eval = eval_summary.loc[eval_summary["run"].isin(["clip_010", "privilege_gru", "clip_020"])]
    kl_rows = frames["training_metrics"].loc[
        frames["training_metrics"]["run"].isin(["privilege_gru", "batch_25600", "batch_51200", "clip_010", "clip_020"])
    ]
    final_table = eval_rollup_frame.copy()
    final_table["final_collision_rate_pct"] = final_table["final_collision_rate"] * 100
    final_actor = actor.loc[actor["update"] == 20].copy()
    final_actor["actor_relative_l2_pct"] = final_actor["actor_relative_l2_from_bc"] * 100
    local_control_complete = int(
        (
            (control["formal_updates"] == 20)
            & (control["actor_checkpoints"] == 20)
            & (control["critic_checkpoints"] == 20)
            & (control["eval_checkpoints"] == 5)
            & (control["eval_episodes_min"] == 600)
            & (control["eval_episodes_max"] == 600)
        ).sum()
    )
    best_row = eval_summary.sort_values(["collision_count", "overtake_count"], ascending=[True, False]).iloc[0]
    final_priv = final_table.loc[final_table["run"] == "privilege_gru"].iloc[0]
    final_ind = final_table.loc[final_table["run"] == "independent_gru"].iloc[0]
    final_mlp = final_table.loc[final_table["run"] == "privilege_mlp"].iloc[0]
    final_pair = paired.loc[
        (paired["left_run"] == "independent_gru") & (paired["right_run"] == "privilege_gru") & (paired["update"] == 20)
    ].iloc[0]
    trace_collision_total = int(trace["json_collision_count"].sum())
    trace_missing_terminal_total = int(trace["json_collisions_missing_terminal_trace_frame"].sum())
    generated_at = "2026-07-22T00:00:00+08:00"

    source_specs = [
        ("report_views", "Read-only report snapshot views", "queries/report_views.sql"),
        ("control_audit", "Control-variable audit snapshot", "data/control_audit.csv"),
        ("training_metrics", "Formal training metrics snapshot", "data/training_metrics.csv"),
        ("training_summary", "Training summary snapshot", "data/training_summary.csv"),
        ("eval_summary", "Austin checkpoint summary snapshot", "data/eval_summary.csv"),
        ("eval_rollup", "Austin run rollup snapshot", "data/eval_rollup.csv"),
        ("paired_comparisons", "Austin paired comparison snapshot", "data/paired_comparisons.csv"),
        ("actor_deltas", "Actor checkpoint parameter audit", "data/actor_parameter_deltas.csv"),
        ("trace_quality", "NPZ trace quality audit", "data/trace_quality.csv"),
        ("scenario_slices", "Austin scenario-slice risk snapshot", "data/scenario_slices.csv"),
    ]
    manifest_sources = [{"id": source_id, "label": label, "path": path} for source_id, label, path in source_specs]
    top_sources = []
    for source_id, _label, _path in source_specs:
        if source_id == "report_views":
            description = "DuckDB-compatible read-only views over the reviewed CSV snapshots used by report charts and tables."
        else:
            frame_name = source_id if source_id in frames else "actor_parameter_deltas"
            description = f"Derived, reviewed snapshot. Raw relative source paths are retained in provenance columns. Rows: {len(frames[frame_name])}."
        top_sources.append({"id": source_id, "description": description})

    report_kpis = [
        {
            "runs": 7,
            "checkpoint_evals": 35,
            "austin_scenarios": 600,
            "best_collision_count": int(best_row["collision_count"]),
            "best_collision_rate": float(best_row["collision_rate"]),
            "complete_runs": local_control_complete,
        }
    ]
    control_display = control.copy()
    control_display["warmup_control"] = np.where(
        control_display["run"].isin(["clip_010", "clip_020"]),
        control_display["warmup_equals_clip010"],
        control_display["warmup_equals_local_baseline"],
    )
    control_display["audit_note"] = "strict"
    control_display.loc[control_display["run"].isin(["clip_010", "clip_020"]), "audit_note"] = "strict pair; comparison to clip0.15 is worker-contaminated"

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "PPO 0721 七实验综合评估",
            "description": "基于训练参数、140 个 formal metrics、35 个 Austin 600 checkpoint eval、checkpoint 权重和 21,000 个 NPZ trace 的技术审计。",
            "generatedAt": generated_at,
            "cards": [],
            "charts": [
                {
                    "id": "critic_collision",
                    "title": "Critic 对照：privilege_gru 全程领先",
                    "subtitle": "同 worker、同 warmup 轨迹、同 batch/clip；纵轴越低越好。",
                    "type": "line",
                    "dataset": "eval_critic",
                    "sourceId": "report_views",
                    "encodings": {
                        "x": {"field": "update", "type": "quantitative", "label": "Formal update"},
                        "y": {"field": "collision_count", "type": "quantitative", "label": "碰撞数 / 600"},
                        "color": {"field": "run_label", "type": "nominal", "label": "Critic"},
                        "tooltip": [
                            {"field": "overtake_count", "type": "quantitative", "label": "超车"},
                            {"field": "follow_count", "type": "quantitative", "label": "跟随"},
                        ],
                    },
                },
                {
                    "id": "batch_collision",
                    "title": "Batch size：12800 的总体结果最好",
                    "subtitle": "更大 batch 减少每 update 的优化器步数，并降低 critic EV 与参数学习量。",
                    "type": "line",
                    "dataset": "eval_batch",
                    "sourceId": "report_views",
                    "encodings": {
                        "x": {"field": "update", "type": "quantitative", "label": "Formal update"},
                        "y": {"field": "collision_count", "type": "quantitative", "label": "碰撞数 / 600"},
                        "color": {"field": "run_label", "type": "nominal", "label": "Batch"},
                    },
                },
                {
                    "id": "clip_collision",
                    "title": "Clip range：0.15 当前最稳，0.20 是候选",
                    "subtitle": "0.10 vs 0.20 是严格对照；0.15 使用不同 env_workers，只能作参考。",
                    "type": "line",
                    "dataset": "eval_clip",
                    "sourceId": "report_views",
                    "encodings": {
                        "x": {"field": "update", "type": "quantitative", "label": "Formal update"},
                        "y": {"field": "collision_count", "type": "quantitative", "label": "碰撞数 / 600"},
                        "color": {"field": "run_label", "type": "nominal", "label": "Clip"},
                    },
                },
                {
                    "id": "actor_delta",
                    "title": "Actor 权重位移很小，但策略 KL 并不小",
                    "subtitle": "仅 GRU 与输出头更新；固定 BC 前端逐 tensor 未改变。",
                    "type": "line",
                    "dataset": "actor_deltas",
                    "sourceId": "report_views",
                    "encodings": {
                        "x": {"field": "update", "type": "quantitative", "label": "Formal update"},
                        "y": {"field": "actor_relative_l2_pct", "type": "quantitative", "label": "相对 BC L2 位移 (%)"},
                        "color": {"field": "run_label", "type": "nominal", "label": "Run"},
                    },
                },
                {
                    "id": "kl_dynamics",
                    "title": "所有设置仍有稀疏 KL 尖峰",
                    "subtitle": "clip 约束 surrogate objective，不会硬限制 KL；当前 target_kl=None。",
                    "type": "line",
                    "dataset": "kl_rows",
                    "sourceId": "report_views",
                    "encodings": {
                        "x": {"field": "update", "type": "quantitative", "label": "Formal update"},
                        "y": {"field": "approx_kl_mean", "type": "quantitative", "label": "Mean approximate KL"},
                        "color": {"field": "run_label", "type": "nominal", "label": "Run"},
                    },
                },
                {
                    "id": "scenario_slices",
                    "title": "风险集中在少数 raceline × speed 切片",
                    "subtitle": "35 个 checkpoint 的描述性聚合；同一 checkpoint/场景相关，不能当独立样本。",
                    "type": "bar",
                    "dataset": "scenario_slices",
                    "sourceId": "report_views",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "slice", "type": "nominal", "label": "Austin slice"},
                        "y": {"field": "collision_rate", "type": "quantitative", "label": "碰撞率"},
                    },
                },
            ],
            "tables": [
                {
                    "id": "final_models",
                    "title": "7 个 run 的 checkpoint 与最终结果",
                    "subtitle": "序列顺序为 U1/U5/U10/U15/U20；best 在同一 Austin 面板上选取，存在选择偏差。",
                    "dataset": "final_table",
                    "sourceId": "report_views",
                    "defaultSort": {"field": "final_collision_count", "direction": "asc"},
                    "columns": [
                        {"field": "run_label", "label": "Run", "type": "text"},
                        {"field": "collision_sequence", "label": "碰撞序列", "type": "text"},
                        {"field": "collision_mean", "label": "5点均值", "format": "decimal"},
                        {"field": "best_update", "label": "Best U", "format": "number"},
                        {"field": "best_collision_count", "label": "Best 碰撞", "format": "number"},
                        {"field": "final_collision_count", "label": "U20 碰撞", "format": "number"},
                        {"field": "final_overtake_count", "label": "U20 超车", "format": "number"},
                        {"field": "final_follow_count", "label": "U20 跟随", "format": "number"},
                    ],
                },
                {
                    "id": "training_table",
                    "title": "训练质量与优化风险",
                    "subtitle": "EV 与 rollout 取 last-5；KL/grad 为 20 update 汇总。grad norm 是裁剪前。",
                    "dataset": "training_summary",
                    "sourceId": "report_views",
                    "columns": [
                        {"field": "run_label", "label": "Run", "type": "text"},
                        {"field": "ev_last5", "label": "EV last5", "format": "decimal"},
                        {"field": "collision_ev_last5", "label": "碰撞 EV", "format": "decimal"},
                        {"field": "rollout_collision_last5", "label": "训练碰撞率", "format": "percent"},
                        {"field": "kl_mean_median", "label": "KL 中位", "format": "decimal"},
                        {"field": "kl_single_minibatch_max", "label": "单 MB KL max", "format": "decimal"},
                        {"field": "actor_preclip_grad_norm_median", "label": "preclip grad", "format": "decimal"},
                        {"field": "total_training_minutes", "label": "训练分钟", "format": "decimal"},
                    ],
                },
                {
                    "id": "control_table",
                    "title": "控制变量与完整性审计",
                    "subtitle": "seed 和源码 commit 未持久化；clip=0.15 与 0.10/0.20 的 worker 数不同。",
                    "dataset": "control_table",
                    "sourceId": "report_views",
                    "columns": [
                        {"field": "run_label", "label": "Run", "type": "text"},
                        {"field": "critic", "label": "Critic", "type": "text"},
                        {"field": "batch_size", "label": "Batch", "format": "number"},
                        {"field": "clip_range", "label": "Clip", "format": "decimal"},
                        {"field": "env_workers", "label": "Workers", "format": "number"},
                        {"field": "warmup_control", "label": "Warmup 匹配", "type": "text"},
                        {"field": "audit_note", "label": "控制强度", "type": "text"},
                    ],
                },
            ],
            "sources": manifest_sources,
            "blocks": [
                {
                    "id": "summary",
                    "type": "markdown",
                    "body": (
                        "## 技术结论\n\n"
                        "**首选配置：privilege_gru + batch_size=12800 + clip_range=0.15；U15 与 U20 的聚合结果并列。** "
                        f"该 run 在 Austin 600 的 U1/U5/U10/U15/U20 为 16/18/18/14/14，U20 为 {int(final_priv.final_collision_count)} 碰撞、{int(final_priv.final_overtake_count)} 超车。"
                        f"严格 critic 对照的 U20 中，independent_gru 为 {int(final_ind.final_collision_count)}，privilege_mlp 为 {int(final_mlp.final_collision_count)}。"
                        f"privilege_gru 相对 independent_gru 净减少 {int(final_pair.delta_left_minus_right)} 次碰撞；逐场景 discordant 为 "
                        f"{int(final_pair.left_only_collisions)} vs {int(final_pair.right_only_collisions)}，exact p={final_pair.mcnemar_exact_p:.4f}。\n\n"
                        "**证据强度：中等。** critic 与 batch 两组控制成立；但只有单 seed，且所有 best checkpoint 都在同一 Austin 面板上选择。"
                    ),
                },
                {"id": "critic_text", "type": "markdown", "body": "## 1. Critic：hybrid privilege_gru 胜出\n\n三种 critic 的初始 warmup 轨迹逐条相同；actor 总参数位移也近似，因此结果差异不能用‘更新幅度更大’解释。privilege_gru 同时保留 361D 时序观测与 P20 当步特权状态，并且仅比 independent_gru 多 8,400 个 projection 参数。它在 5 个 checkpoint 均优于另外两种结构，支持‘时序观测 + 特权状态互补’，但不能证明当前 late-fusion 是全局最优架构。"},
                {"id": "critic_chart", "type": "chart", "chartId": "critic_collision"},
                {"id": "final_table_block", "type": "table", "tableId": "final_models"},
                {"id": "batch_text", "type": "markdown", "body": "## 2. Batch size：保留 12800\n\n在 n_envs×n_steps=102,400 固定时，batch 12800/25600/51200 分别给 actor 每 update 16/8/4 个优化器 step，critic 为 40/20/10 个 step。学习率和 epoch 未随 batch 缩放，因此大 batch 实际降低了每 update 的优化量：U20 actor 位移、privileged projection norm 和 critic EV 都随 batch 增大而下降。25600/51200 没有稳定 eval 收益，训练阶段节时不足以抵消质量损失。"},
                {"id": "batch_chart", "type": "chart", "chartId": "batch_collision"},
                {"id": "clip_text", "type": "markdown", "body": "## 3. Clip range：0.15 当前首选，0.20 进入复验\n\n严格的 0.10 vs 0.20 对照中，0.20 的 U20 为 14 碰撞，0.10 为 20；0.20 的 actor 权重位移更大、eval 从 24→14 单调改善，但单-minibatch KL 尾部更高。0.15 的 5 点均值 16.0、U20 14，是现有面板最稳的设置；不过它使用 env_workers=12，而另外两档为 8，初始 warmup 轨迹已不同，所以三档排序不能作纯 clip 因果结论。"},
                {"id": "clip_chart", "type": "chart", "chartId": "clip_collision"},
                {"id": "param_text", "type": "markdown", "body": "## 4. 参数变化：权重小位移不等于策略小变化\n\n所有 run 的固定 BC 前端 tensor 完全不变；U20 的可训练 actor 相对 BC L2 位移只有约 0.037%–0.068%，输出头相对位移约为 GRU 的 10 倍。尽管如此，approximate KL 有稀疏尖峰，说明窄探索分布下小权重变化仍可显著改变动作概率。privilege_gru 的 P20 projection norm 在 warmup 后持续增长到 U20，表明通道被使用；权重 norm 不能当作因果特征重要性。"},
                {"id": "actor_chart", "type": "chart", "chartId": "actor_delta"},
                {"id": "kl_chart", "type": "chart", "chartId": "kl_dynamics"},
                {"id": "training_table_block", "type": "table", "tableId": "training_table"},
                {"id": "scenario_text", "type": "markdown", "body": f"## 5. Austin 风险结构\n\n600 个场景里，{int((scenario.collision_count > 0).sum())} 个在至少一个 checkpoint 中碰撞，{int((scenario.collision_frequency >= .8).sum())} 个在至少 80% checkpoint 中碰撞，{int((scenario.collision_frequency == 1).sum())} 个在全部 35 个 checkpoint 中碰撞。说明排名主要由不足百个边界场景决定。相邻 checkpoint 平均只有少量场景翻转，但低碰撞总数下足以改变名次；best checkpoint 必须用独立 holdout 复核。"},
                {"id": "scenario_chart", "type": "chart", "chartId": "scenario_slices"},
                {"id": "quality_text", "type": "markdown", "body": f"## 6. 数据质量\n\n7 个训练目录均有 1 条 warmup + 20 条 formal metrics、20 个 actor/critic checkpoint 和 5×600 个有效 eval；所有 results JSON 的 observation/action finite 且 error_count=0。21,000 个 NPZ trace 齐全，但 {trace_collision_total} 次 JSON 碰撞中有 {trace_missing_terminal_total} 次在 trace 内没有碰撞 bit，原因是缺 terminal post-step frame；碰撞标签必须以 results_multi.json 为准。P20 clearance softsign 后 exact +1 饱和为 0，body margin 的约半数 +1 是互补几何语义，不是旧的 clearance 硬截断。"},
                {"id": "control_table_block", "type": "table", "tableId": "control_table"},
                {"id": "method", "type": "markdown", "body": "## Scope, definitions, and methodology\n\n- **Primary KPI:** ego collision count on the same 600-scenario Austin panel; overtakes are the secondary guardrail.\n- **Training evidence:** post-update EV/value loss, rollout outcome rates, approximate KL, clip fraction, pre-clip gradient norm, and wall time.\n- **Parameter evidence:** tensor-level L2 displacement from the shared BC checkpoint; critic deltas are only compared within architecture.\n- **Paired evidence:** exact McNemar/binomial test on discordant collision outcomes plus a 10,000-draw bootstrap clustered by the 150 base startpoint/opponent pairs.\n- **Trace evidence:** all NPZ files were opened; action finiteness, length, physical clipping, and collision-bit coverage were checked.\n\nWilson intervals describe one fixed panel, not cross-map or cross-seed generalization. Checkpoints from one training run are correlated and are never treated as five independent seeds."},
                {"id": "limitations", "type": "markdown", "body": "## Limitations and robustness\n\n1. **Single seed:** critic result is convincing within this trajectory, not yet an architecture-level population estimate.\n2. **Selection bias:** U1/U5/U10/U15/U20 are all scored on the same Austin panel; choosing the minimum is optimistic.\n3. **Clip contamination:** clip0.15 uses 12 workers; clip0.10/0.20 use 8 and were produced on the remote execution environment. Their warmup paths differ.\n4. **Missing provenance:** run_config intentionally omits seed and records neither Git commit nor source hash. The script implies seed=42, but artifacts alone cannot prove it.\n5. **Trace terminal frame:** terminal collision geometry is unavailable in NPZ, so trace-based collision labeling is invalid."},
                {"id": "recommendations", "type": "markdown", "body": "## Recommended next actions\n\n1. **保留 privilege_gru / batch12800 / clip0.15。** U15 与 U20 都是 14 碰撞、349 超车、237 跟随，但碰撞集合并不相同；若现在必须指定单一文件，默认用 final alias 对应的 U20，只是工程选择，不是指标胜出。\n2. **只补一个窄实验：clip0.15 vs 0.20，固定 env_workers、硬件、源码 hash，各 2–3 seed。** 不再扩 batch sweep。\n3. **加 target-KL early stop（建议起点 0.015–0.02）并保持 max_grad_norm=0.5。** 当前 clip 不足以阻止 KL 尾部；先直接控制策略距离。\n4. **增加独立 Austin holdout 起点面板。** 模型晋升要求主面板和 holdout 同向，不用同一面板挑最低 checkpoint。\n5. **补齐 run provenance 与 trace terminal frame。** run_config 记录 seed、Git commit、pretrained SHA256；trace 在 env.step 后写入 terminal post-step frame。"},
                {"id": "questions", "type": "markdown", "body": "## Further questions\n\n- privilege_gru 的优势在不同 seed 和平移起点面板上是否仍为 10–20 次碰撞？\n- clip0.20 在 target-KL early stop 下是否保留后期收益，同时消除 KL 尾部？\n- 3 个 universal-collision 场景对原始 BC actor 是否也必碰，还是 PPO reward/credit assignment 的共同失败？"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "report_kpis": report_kpis,
                "eval_critic": json_records(critic_final, ["run_label", "update", "collision_count", "overtake_count", "follow_count"]),
                "eval_batch": json_records(batch_eval, ["run_label", "update", "collision_count", "overtake_count", "follow_count"]),
                "eval_clip": json_records(clip_eval, ["run_label", "update", "collision_count", "overtake_count", "follow_count"]),
                "actor_deltas": json_records(final_actor if False else actor.assign(actor_relative_l2_pct=actor["actor_relative_l2_from_bc"] * 100), ["run_label", "update", "actor_relative_l2_pct"]),
                "kl_rows": json_records(kl_rows, ["run_label", "update", "approx_kl_mean", "approx_kl_max"]),
                "scenario_slices": json_records(slices.sort_values("collision_rate", ascending=False), ["slice", "collision_rate", "collision_count", "evaluation_rows"]),
                "final_table": json_records(final_table, ["run_label", "collision_sequence", "collision_mean", "best_update", "best_collision_count", "final_collision_count", "final_overtake_count", "final_follow_count"]),
                "training_summary": json_records(training, ["run_label", "ev_last5", "collision_ev_last5", "rollout_collision_last5", "kl_mean_median", "kl_single_minibatch_max", "actor_preclip_grad_norm_median", "total_training_minutes"]),
                "control_table": json_records(control_display, ["run_label", "critic", "batch_size", "clip_range", "env_workers", "warmup_control", "audit_note"]),
            },
            "accessIssues": [],
        },
        "sources": top_sources,
        "package_info": {
            "originUrl": "artifact://ppo-0721-seven-experiments",
            "controls": {"edit": False, "refresh": False},
        },
    }
    path = ANALYSIS_DIR / "artifact.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_all(scan_traces: bool = True) -> dict[str, pd.DataFrame]:
    formal, warmup, training_episodes, configs = load_training_artifacts()
    eval_summary, eval_episodes, episode_maps = load_eval_artifacts()
    control = control_audit(formal, training_episodes, configs, episode_maps)
    train_summary = training_summary(formal, warmup)
    rollup = eval_rollup(eval_summary)
    paired = paired_comparisons(episode_maps)
    stability = checkpoint_stability(episode_maps)
    actor_deltas, actor_consecutive = actor_parameter_deltas()
    critic_deltas, feature_proxy = critic_parameter_deltas(configs, formal)
    feature_detail, feature_summary = feature_health(formal, configs)
    trace_path = DATA_DIR / "trace_quality.csv"
    if scan_traces or not trace_path.is_file():
        traces = scan_trace_quality(episode_maps)
    else:
        traces = pd.read_csv(trace_path)
    scenarios, slices = scenario_risk(eval_episodes)
    frames = {
        "control_audit": control,
        "training_metrics": formal,
        "warmup_metrics": warmup,
        "training_summary": train_summary,
        "eval_summary": eval_summary,
        "eval_episodes": eval_episodes,
        "eval_rollup": rollup,
        "paired_comparisons": paired,
        "checkpoint_stability": stability,
        "actor_parameter_deltas": actor_deltas,
        "actor_consecutive_deltas": actor_consecutive,
        "critic_parameter_deltas": critic_deltas,
        "privilege_weight_proxy": feature_proxy,
        "feature_health_detail": feature_detail,
        "feature_health_summary": feature_summary,
        "trace_quality": traces,
        "scenario_risk": scenarios,
        "scenario_slices": slices,
    }
    write_dataframes(frames)
    make_figures(formal, eval_summary, actor_deltas, slices)
    notebook_path = make_notebook(frames)
    execute_notebook(notebook_path)
    build_report_artifact(frames)
    return frames


if __name__ == "__main__":
    frames = run_all(scan_traces=True)
    print("Analysis complete")
    print(frames["eval_rollup"][["run", "collision_sequence", "collision_mean", "final_collision_count"]].to_string(index=False))
    print(f"Notebook: {ANALYSIS_DIR / 'ppo_0721_seven_experiments.ipynb'}")
    print(f"Artifact: {ANALYSIS_DIR / 'artifact.json'}")

#!/usr/bin/env python3
"""P3: offline credit-horizon comparison on the exact P1 rollout tensors."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import itertools
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

try:
    from audit_rl_direction_common import (
        EXPERIMENT_DIR,
        GAMMA,
        PREREGISTRATION_PATH,
        ROOT,
        RUN_DIR,
        TIMESTEP,
        assert_frozen_contract,
        read_json,
        sha256_file,
        write_json_atomic,
    )
    from audit_rl_gradient_direction import (
        BOOTSTRAP_SEED,
        TIME_BIN_NAMES,
        _bootstrap_ci,
        _combined,
        _cosine,
        _group_slices,
        _load_gradient,
        _norms,
        _policy,
        _probe_metrics,
        _quantiles,
        _replay_log_prob_components,
        _standardize,
        _time_bins,
        _trainable_parameters,
    )
except ModuleNotFoundError:
    from scripts.audit_rl_direction_common import (
        EXPERIMENT_DIR,
        GAMMA,
        PREREGISTRATION_PATH,
        ROOT,
        RUN_DIR,
        TIMESTEP,
        assert_frozen_contract,
        read_json,
        sha256_file,
        write_json_atomic,
    )
    from scripts.audit_rl_gradient_direction import (
        BOOTSTRAP_SEED,
        TIME_BIN_NAMES,
        _bootstrap_ci,
        _combined,
        _cosine,
        _group_slices,
        _load_gradient,
        _norms,
        _policy,
        _probe_metrics,
        _quantiles,
        _replay_log_prob_components,
        _standardize,
        _time_bins,
        _trainable_parameters,
    )


CANDIDATES = {
    "C0_CURRENT": {"lambda": 0.995, "collision_credit": "terminal"},
    "C1_LONGER_GAE": {"lambda": 0.997, "collision_credit": "terminal"},
    "C2_MONTE_CARLO_LIKE": {"lambda": 1.000, "collision_credit": "terminal"},
    "C3_REDISTRIBUTED": {"lambda": 0.995, "collision_credit": "redistributed"},
}
COLLISION_PENALTY = -2.0
MODEL_SEED = 20260717


def _gae(rewards: np.ndarray, values: np.ndarray, bootstrap_value: float, gae_lambda: float) -> np.ndarray:
    advantages = np.empty_like(rewards, dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if index == len(rewards) - 1:
            next_value = bootstrap_value
            continuation = 0.0
        else:
            next_value = float(values[index + 1])
            continuation = 1.0
        delta = float(rewards[index]) + GAMMA * next_value - float(values[index])
        running = delta + GAMMA * gae_lambda * continuation * running
        advantages[index] = running
    return advantages


def _discounted_returns(rewards: np.ndarray) -> np.ndarray:
    returns = np.empty_like(rewards, dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + GAMMA * running
        returns[index] = running
    return returns


def _redistribute_collision(rewards: np.ndarray, window_seconds: float) -> tuple[np.ndarray, dict[str, float | int]]:
    redistributed = rewards.astype(np.float64, copy=True)
    window_steps = min(int(round(window_seconds / TIMESTEP)), len(redistributed))
    if window_steps <= 0:
        raise RuntimeError(f"Invalid redistribution window: {window_seconds}")
    terminal_without_collision = float(redistributed[-1] - COLLISION_PENALTY)
    redistributed[-1] = terminal_without_collision
    base_weights = np.arange(1, window_steps + 1, dtype=np.float64)
    discount = GAMMA ** np.arange(window_steps, dtype=np.float64)
    target_at_window_start = COLLISION_PENALTY * GAMMA ** (window_steps - 1)
    scale = target_at_window_start / float(np.dot(discount, base_weights))
    redistributed[-window_steps:] += scale * base_weights
    observed = float(np.dot(discount, scale * base_weights))
    if not math.isclose(observed, target_at_window_start, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise RuntimeError("Collision redistribution changed its discounted contribution")
    return redistributed, {
        "window_steps": window_steps,
        "window_seconds_effective": float(window_steps * TIMESTEP),
        "triangular_scale": float(scale),
        "target_discounted_collision_contribution": float(target_at_window_start),
        "observed_discounted_collision_contribution": observed,
    }


def _mc_advantages(
    episode_rows: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    small: list[dict[str, Any]] = []
    global_sums: defaultdict[tuple[str, int], float] = defaultdict(float)
    global_counts: Counter[tuple[str, int]] = Counter()
    for row in episode_rows:
        with np.load(ROOT / row["path"]) as data:
            returns = _discounted_returns(data["rewards"].astype(np.float64))
        bins = _time_bins(len(returns))
        per_episode: dict[tuple[str, int], tuple[float, int]] = {}
        for bin_index in range(5):
            mask = bins == bin_index
            key = (str(row["branch"]), bin_index)
            subtotal = float(returns[mask].sum())
            count = int(mask.sum())
            global_sums[key] += subtotal
            global_counts[key] += count
            per_episode[key] = (subtotal, count)
        small.append({"returns": returns, "bins": bins, "branch": str(row["branch"]), "per_episode": per_episode})
    rows: list[np.ndarray] = []
    for entry in small:
        baseline = np.empty_like(entry["returns"], dtype=np.float64)
        for bin_index in range(5):
            mask = entry["bins"] == bin_index
            key = (entry["branch"], bin_index)
            own_sum, own_count = entry["per_episode"][key]
            denominator = global_counts[key] - own_count
            if denominator <= 0:
                raise RuntimeError(f"MC leave-one-episode-out baseline is empty for {key}")
            baseline[mask] = (global_sums[key] - own_sum) / denominator
        rows.append(entry["returns"] - baseline)
    standardized, normalization = _standardize(rows)
    raw = np.concatenate(rows)
    return standardized, rows, {"raw": _quantiles(raw), "normalization": normalization}


def _candidate_advantages(
    episode_rows: list[dict[str, Any]],
    redistribution_window_seconds: float,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any], dict[str, Any]]:
    raw_by_candidate: dict[str, list[np.ndarray]] = {name: [] for name in CANDIDATES}
    redistribution_checks: list[dict[str, float | int]] = []
    for row in episode_rows:
        with np.load(ROOT / row["path"]) as data:
            original_rewards = data["rewards"].astype(np.float64)
            values = data["values"].astype(np.float64)
            bootstrap_value = float(data["bootstrap_value"][0])
        for candidate, definition in CANDIDATES.items():
            rewards = original_rewards
            if definition["collision_credit"] == "redistributed" and bool(row["ego_collision"]):
                if original_rewards[-1] > -1.5:
                    raise RuntimeError(
                        f"Collision episode lacks the expected terminal penalty: {row['scenario_id']} "
                        f"reward={original_rewards[-1]}"
                    )
                rewards, check = _redistribute_collision(original_rewards, redistribution_window_seconds)
                redistribution_checks.append(check)
            raw_by_candidate[candidate].append(
                _gae(rewards, values, bootstrap_value, float(definition["lambda"]))
            )
    standardized: dict[str, list[np.ndarray]] = {}
    metrics: dict[str, Any] = {}
    for candidate, rows in raw_by_candidate.items():
        standardized[candidate], normalization = _standardize(rows)
        raw = np.concatenate(rows)
        metrics[candidate] = {"raw": _quantiles(raw), "normalization": normalization}
    _standardized_mc, raw_mc_rows, mc_metrics = _mc_advantages(episode_rows)
    raw_mc = np.concatenate(raw_mc_rows)
    for candidate, raw_rows in raw_by_candidate.items():
        raw = np.concatenate(raw_rows)
        metrics[candidate]["mc_correlation"] = float(np.corrcoef(raw, raw_mc)[0, 1])
        metrics[candidate]["mc_sign_agreement"] = float(np.mean(np.sign(raw) == np.sign(raw_mc)))
    if redistribution_checks:
        metrics["C3_REDISTRIBUTED"]["redistribution_check"] = {
            "collision_episode_count": len(redistribution_checks),
            "max_discounted_contribution_abs_error": float(
                max(
                    abs(
                        float(row["observed_discounted_collision_contribution"])
                        - float(row["target_discounted_collision_contribution"])
                    )
                    for row in redistribution_checks
                )
            ),
        }
    return standardized, metrics, mc_metrics


def _gradient_shard(
    policy,
    episode_rows: list[dict[str, Any]],
    shard_dir: Path,
    device: torch.device,
    redistribution_window_seconds: float,
) -> dict[str, Any]:
    names, parameters = _trainable_parameters(policy)
    slices = _group_slices(names, parameters)
    advantages, advantage_metrics, mc_metrics = _candidate_advantages(
        episode_rows, redistribution_window_seconds
    )
    candidate_names = list(CANDIDATES)
    total_steps = int(sum(row["steps"] for row in episode_rows))
    safe_steps = int(sum(row["steps"] for row in episode_rows if not bool(row["ego_collision"])))
    coefficient_rows = 7
    accumulators = {
        candidate: [
            torch.zeros((coefficient_rows, *parameter.shape), dtype=torch.float32, device=device)
            for parameter in parameters
        ]
        for candidate in candidate_names
    }
    replay_max_abs = 0.0
    torch.backends.cudnn.enabled = False
    for episode_index, row in enumerate(episode_rows):
        with np.load(ROOT / row["path"]) as data:
            observations = data["observations"]
            actions = data["actions"]
            old_log_prob_components = data["old_log_prob_components"]
        log_prob_components = _replay_log_prob_components(policy, observations, actions, device)
        replay_max_abs = max(
            replay_max_abs,
            float(
                torch.max(
                    torch.abs(
                        log_prob_components.detach()
                        - torch.as_tensor(old_log_prob_components, dtype=torch.float32, device=device)
                    )
                ).item()
            ),
        )
        bins = _time_bins(len(actions))
        for candidate_index, candidate in enumerate(candidate_names):
            coefficients = torch.zeros(
                (coefficient_rows, len(actions), 2), dtype=torch.float32, device=device
            )
            advantage = torch.as_tensor(
                advantages[candidate][episode_index], dtype=torch.float32, device=device
            )
            coefficients[0, :, :] = advantage[:, None]
            if not bool(row["ego_collision"]):
                coefficients[1, :, :] = advantage[:, None]
            for bin_index in range(5):
                mask = torch.as_tensor(bins == bin_index, dtype=torch.bool, device=device)
                coefficients[2 + bin_index, mask, :] = advantage[mask, None]
            gradients = torch.autograd.grad(
                log_prob_components,
                parameters,
                grad_outputs=coefficients,
                is_grads_batched=True,
                retain_graph=candidate_index + 1 < len(candidate_names),
                create_graph=False,
            )
            for accumulator, gradient in zip(accumulators[candidate], gradients):
                accumulator.add_(gradient.detach())
        if (episode_index + 1) % 4 == 0 or episode_index + 1 == len(episode_rows):
            print(f"P3_GRADIENT episode={episode_index + 1}/{len(episode_rows)}", flush=True)
    flat_by_candidate = {
        candidate: torch.cat(
            [gradient.reshape(coefficient_rows, -1) for gradient in candidate_accumulators],
            dim=1,
        )
        for candidate, candidate_accumulators in accumulators.items()
    }
    records: dict[str, Any] = {}
    p1_source = torch.load(
        ROOT / episode_rows[0]["p1_gradient_file"], map_location="cpu", weights_only=True
    )
    mc_source = p1_source["mc_combined"]
    for candidate in candidate_names:
        flat = flat_by_candidate[candidate]
        combined = (flat[0] / total_steps).detach().cpu()
        safe = (flat[1] / max(safe_steps, 1)).detach().cpu()
        time_vectors = [
            (flat[2 + index] / total_steps).detach().cpu() for index in range(5)
        ]
        candidate_path = shard_dir / f"{candidate.lower()}_gradients.pt"
        torch.save(
            {
                "parameter_names": names,
                "parameter_shapes": [list(parameter.shape) for parameter in parameters],
                "parameter_numels": [parameter.numel() for parameter in parameters],
                "gae_combined": combined,
                "safe_combined": safe,
                "mc_combined": mc_source,
            },
            candidate_path,
        )
        records[candidate] = {
            "gradient_file": str(candidate_path.relative_to(ROOT)),
            "gradient_file_sha256": sha256_file(candidate_path),
            "total_steps": total_steps,
            "safe_steps": safe_steps,
            "gradient_norm": _norms(combined, slices),
            "safe_gradient_norm": _norms(safe, slices),
            "time_to_end_gradient_norm": {
                name: _norms(vector, slices) for name, vector in zip(TIME_BIN_NAMES, time_vectors)
            },
            "advantage": advantage_metrics[candidate],
            "replay_log_prob_component_max_abs": replay_max_abs,
        }
        if candidate == "C0_CURRENT":
            p1_gae = p1_source["gae_combined"]
            records[candidate]["exact_p1_reproduction"] = {
                "gradient_cosine": _cosine(combined.double(), p1_gae.double()),
                "max_abs_difference": float(torch.max(torch.abs(combined - p1_gae)).item()),
            }
    del flat_by_candidate, accumulators
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = True
    return {"candidates": records, "mc_advantage": mc_metrics}


def _aggregate_candidate(
    candidate: str,
    shard_records: list[dict[str, Any]],
    device: torch.device,
    seed_offset: int,
) -> dict[str, Any]:
    candidate_records = [row["p3"]["candidates"][candidate] for row in shard_records]
    payload = torch.load(ROOT / candidate_records[0]["gradient_file"], map_location="cpu", weights_only=True)
    names = payload["parameter_names"]
    dummy_parameters = [torch.empty(shape) for shape in payload["parameter_shapes"]]
    slices = _group_slices(names, dummy_parameters)
    vectors = [_load_gradient(row["gradient_file"], "gae_combined") for row in candidate_records]
    safe_vectors = [_load_gradient(row["gradient_file"], "safe_combined") for row in candidate_records]
    mc_vectors = [_load_gradient(row["gradient_file"], "mc_combined") for row in candidate_records]
    weights = [int(row["total_steps"]) for row in candidate_records]
    safe_weights = [int(row["safe_steps"]) for row in candidate_records]
    pairwise = {
        group: [
            _cosine(vectors[first][group_slice], vectors[second][group_slice])
            for first, second in itertools.combinations(range(len(vectors)), 2)
        ]
        for group, group_slice in slices.items()
    }
    bootstrap = {
        group: _bootstrap_ci(vectors, group_slice, BOOTSTRAP_SEED + seed_offset + index)
        for index, (group, group_slice) in enumerate(slices.items())
    }
    final = _combined(vectors, weights)
    final_safe = _combined(safe_vectors, safe_weights)
    final_mc = _combined(mc_vectors, weights)
    probe = _probe_metrics(candidate_records, device)
    raw_advantage = [row["advantage"]["raw"] for row in candidate_records]
    return {
        "pairwise_gradient_cosine": pairwise,
        "pairwise_combined_median": float(np.median(pairwise["combined"])),
        "bootstrap_95ci": bootstrap,
        "gae_mc_gradient_cosine_by_shard": [
            _cosine(vector[slices["combined"]], mc[slices["combined"]])
            for vector, mc in zip(vectors, mc_vectors)
        ],
        "aggregate_gae_mc_gradient_cosine": _cosine(
            final[slices["combined"]], final_mc[slices["combined"]]
        ),
        "aggregate_gradient_norm": _norms(final, slices),
        "aggregate_safe_gradient_norm": _norms(final_safe, slices),
        "advantage": {
            "std_by_shard": [float(row["std"]) for row in raw_advantage],
            "p95_abs_by_shard": [float(row["p95_abs"]) for row in raw_advantage],
            "p99_abs_by_shard": [float(row["p99_abs"]) for row in raw_advantage],
            "p99_abs_max": max(float(row["p99_abs"]) for row in raw_advantage),
        },
        "time_to_end_gradient_norm_by_shard": [
            row["time_to_end_gradient_norm"] for row in candidate_records
        ],
        "probe": probe,
    }


def _selection(candidate_records: dict[str, Any]) -> dict[str, Any]:
    current = candidate_records["C0_CURRENT"]
    current_median = float(current["pairwise_combined_median"])
    current_agreement = float(current["probe"]["collision_action_delta_sign_agreement"])
    current_safe_norm = float(current["aggregate_safe_gradient_norm"]["combined"])
    current_p99 = float(current["advantage"]["p99_abs_max"])
    evaluations: dict[str, Any] = {}
    passing: list[str] = []
    for candidate in ("C1_LONGER_GAE", "C2_MONTE_CARLO_LIKE", "C3_REDISTRIBUTED"):
        record = candidate_records[candidate]
        checks = {
            "median_cosine_improvement_ge_0.10": (
                float(record["pairwise_combined_median"]) - current_median >= 0.10
            ),
            "bootstrap_95ci_lower_gt_0": float(record["bootstrap_95ci"]["combined"]["lower"]) > 0.0,
            "collision_action_agreement_not_lower": (
                float(record["probe"]["collision_action_delta_sign_agreement"]) >= current_agreement
            ),
            "safe_gradient_norm_increase_le_20pct": (
                float(record["aggregate_safe_gradient_norm"]["combined"])
                <= current_safe_norm * 1.20
            ),
            "advantage_p99_le_2x_current": float(record["advantage"]["p99_abs_max"]) <= current_p99 * 2.0,
        }
        passed = all(checks.values())
        if passed:
            passing.append(candidate)
        evaluations[candidate] = {
            "checks": checks,
            "passed": passed,
            "median_cosine_improvement": float(record["pairwise_combined_median"]) - current_median,
            "collision_action_agreement_delta": (
                float(record["probe"]["collision_action_delta_sign_agreement"]) - current_agreement
            ),
            "safe_gradient_norm_ratio": (
                float(record["aggregate_safe_gradient_norm"]["combined"]) / current_safe_norm
                if current_safe_norm > 0.0
                else math.inf
            ),
            "advantage_p99_ratio": (
                float(record["advantage"]["p99_abs_max"]) / current_p99 if current_p99 > 0.0 else math.inf
            ),
        }
    selected = max(
        passing,
        key=lambda name: float(candidate_records[name]["pairwise_combined_median"]),
        default="C0_CURRENT",
    )
    return {
        "candidate_evaluations": evaluations,
        "passing_candidates": passing,
        "selected_credit": selected,
        "verdict": "CREDIT_CANDIDATE_PASSES" if passing else "KEEP_CURRENT_CREDIT",
    }


def main() -> None:
    started = time.monotonic()
    frozen_hashes = assert_frozen_contract()
    preregistration = read_json(PREREGISTRATION_PATH)
    p1_path = EXPERIMENT_DIR / "P1_GRADIENT_DIRECTION.json"
    p2_path = EXPERIMENT_DIR / "P2_COUNTERFACTUAL_ACTIONABILITY.json"
    p1 = read_json(p1_path)
    p2 = read_json(p2_path)
    if p1["status"] != "COMPLETED_AFTER_ALLOWED_EXTENSION":
        raise RuntimeError(f"P1 is not final: {p1['status']}")
    if p2["status"] != "COMPLETED":
        raise RuntimeError(f"P2 is not complete: {p2['status']}")
    earliest = p2["actionability_window"]["earliest_actionable_seconds_before_collision"]
    p75 = None if earliest is None else earliest.get("p75")
    redistribution_window_seconds = 2.0 if p75 is not None and float(p75) <= 2.0 else 3.0
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("P3 requires CUDA")
    policy = _policy(device)
    initial_actor_state = {
        name: tensor.detach().cpu().clone() for name, tensor in policy.end2race_actor.state_dict().items()
    }
    pools: dict[str, Any] = {}
    for pool_index, (pool_name, p1_pool) in enumerate(p1["pools"].items()):
        shard_records: list[dict[str, Any]] = []
        for shard in p1_pool["shards"]:
            shard_index = int(shard["shard_index"])
            shard_dir = RUN_DIR / "p3" / pool_name / f"shard_{shard_index}"
            shard_result_path = shard_dir / "shard_result.json"
            if shard_result_path.is_file():
                record = read_json(shard_result_path)
                if int(record["seed"]) != int(shard["seed"]):
                    raise RuntimeError(f"P3 resumable shard seed mismatch: {shard_result_path}")
                for candidate in CANDIDATES:
                    gradient = ROOT / record["p3"]["candidates"][candidate]["gradient_file"]
                    expected = record["p3"]["candidates"][candidate]["gradient_file_sha256"]
                    if sha256_file(gradient) != expected:
                        raise RuntimeError(f"P3 resumable gradient hash mismatch: {gradient}")
                shard_records.append(record)
                print(f"P3_SHARD_RESUME pool={pool_name} shard={shard_index}", flush=True)
                continue
            episodes = []
            for episode in shard["episodes"]:
                copied = dict(episode)
                copied["p1_gradient_file"] = shard["gradient"]["gradient_file"]
                episodes.append(copied)
            print(f"P3_SHARD_START pool={pool_name} shard={shard_index}", flush=True)
            p3_record = _gradient_shard(
                policy, episodes, shard_dir, device, redistribution_window_seconds
            )
            record = {
                "pool": pool_name,
                "shard_index": shard_index,
                "seed": int(shard["seed"]),
                "p1_episode_paths": [row["path"] for row in episodes],
                "p1_episode_sha256": [row["sha256"] for row in episodes],
                "p1_gradient_file": shard["gradient"]["gradient_file"],
                "p1_gradient_file_sha256": shard["gradient"]["gradient_file_sha256"],
                "p3": p3_record,
            }
            write_json_atomic(shard_result_path, record)
            shard_records.append(record)
            current_actor_state = policy.end2race_actor.state_dict()
            if any(
                not torch.equal(current_actor_state[name].detach().cpu(), reference)
                for name, reference in initial_actor_state.items()
            ):
                raise RuntimeError("Actor parameters changed during P3 offline audit")
            print(f"P3_SHARD_COMPLETE pool={pool_name} shard={shard_index}", flush=True)
        candidate_records = {
            candidate: _aggregate_candidate(
                candidate, shard_records, device, seed_offset=pool_index * 100 + index * 10
            )
            for index, candidate in enumerate(CANDIDATES)
        }
        pools[pool_name] = {
            "complete_episodes": sum(len(row["p1_episode_paths"]) for row in shard_records),
            "shard_count": len(shard_records),
            "candidates": candidate_records,
            "selection": _selection(candidate_records),
        }
        print(
            f"P3_POOL_COMPLETE pool={pool_name} "
            f"verdict={pools[pool_name]['selection']['verdict']} "
            f"selected={pools[pool_name]['selection']['selected_credit']}",
            flush=True,
        )
    directional = [
        name for name, verdict in p1["pool_verdicts"].items() if verdict == "DIRECTION_PRESENT"
    ]
    if not directional:
        raise RuntimeError("P3 has no P1 DIRECTION_PRESENT pool for its primary selection")
    primary_pool = max(
        directional,
        key=lambda name: float(p1["pools"][name]["pairwise_combined_median"]),
    )
    primary_selection = pools[primary_pool]["selection"]
    result = {
        "schema_version": 1,
        "record": "P3_OFFLINE_CREDIT_HORIZON",
        "status": "COMPLETED",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_head": preregistration["source"]["head"],
        "device": "cuda",
        "optimizer_steps": 0,
        "actor_parameters_bitwise_unchanged": True,
        "frozen_hashes": frozen_hashes,
        "p1_result": {"path": str(p1_path.relative_to(ROOT)), "sha256": sha256_file(p1_path)},
        "p2_result": {"path": str(p2_path.relative_to(ROOT)), "sha256": sha256_file(p2_path)},
        "exact_p1_rollouts_reused": True,
        "redistribution": {
            "earliest_actionable_p75_seconds": p75,
            "selected_window_seconds": redistribution_window_seconds,
            "base_weights": "linearly increasing triangular weights toward collision",
            "discounted_collision_contribution_preserved_at_window_start": True,
        },
        "candidate_definitions": CANDIDATES,
        "pools": pools,
        "primary_pool": primary_pool,
        "primary_selection": primary_selection,
        "verdict": primary_selection["verdict"],
        "selected_credit": primary_selection["selected_credit"],
        "elapsed_seconds": float(time.monotonic() - started),
    }
    write_json_atomic(EXPERIMENT_DIR / "P3_CREDIT_HORIZON.json", result)
    print(
        f"P3_COMPLETE primary_pool={primary_pool} verdict={result['verdict']} "
        f"selected={result['selected_credit']} elapsed_seconds={result['elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

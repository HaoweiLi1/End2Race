#!/usr/bin/env python3
"""P4: compare sequential, full-rollout, and transactional actor-only updates."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

try:
    from audit_credit_horizon import _candidate_advantages
    from audit_rl_direction_common import (
        EXPERIMENT_DIR,
        PREREGISTRATION_PATH,
        ROOT,
        RUN_DIR,
        SPEED_PHYSICAL_STD,
        STEERING_LATENT_STD,
        FixedScenarioProvider,
        assert_frozen_contract,
        make_env,
        read_json,
        run_deterministic_episode,
        sha256_file,
        write_json_atomic,
    )
    from audit_rl_gradient_direction import (
        _collect_episode,
        _combined,
        _cosine,
        _group_slices,
        _load_gradient,
        _perturbed_actor,
        _physical_actions,
        _policy,
        _replay_log_prob_components,
        _trainable_parameters,
    )
except ModuleNotFoundError:
    from scripts.audit_credit_horizon import _candidate_advantages
    from scripts.audit_rl_direction_common import (
        EXPERIMENT_DIR,
        PREREGISTRATION_PATH,
        ROOT,
        RUN_DIR,
        SPEED_PHYSICAL_STD,
        STEERING_LATENT_STD,
        FixedScenarioProvider,
        assert_frozen_contract,
        make_env,
        read_json,
        run_deterministic_episode,
        sha256_file,
        write_json_atomic,
    )
    from scripts.audit_rl_gradient_direction import (
        _collect_episode,
        _combined,
        _cosine,
        _group_slices,
        _load_gradient,
        _perturbed_actor,
        _physical_actions,
        _policy,
        _replay_log_prob_components,
        _trainable_parameters,
    )
from ppo.config import CLIP_RANGE, MAX_GRAD_NORM
from ppo.policy import EVALUATOR_STEER_BOUND
from ppo.scenarios import load_hard_pool, scenario_from_dict, training_scenarios


METHODS = (
    "S1_SEQUENTIAL_MINIBATCH",
    "S2_FULL_ROLLOUT_ONE_STEP",
    "S3_TRANSACTIONAL_BACKTRACKED_ONE_STEP",
)
P4_SEEDS = (20260761, 20260762, 20260763)
POOL_NAME = "H1_EXPANDED_DET"
BATCH_SIZE = 1600
GRU_LR = 1.0e-6
HEAD_LR = 1.0e-5
MEAN_KL_MAX = 0.005
P99_SEQUENCE_KL_MAX = 0.020
BACKTRACK_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625)


def _actor_state(policy) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in policy.end2race_actor.state_dict().items()}


def _actor_equal(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> bool:
    return first.keys() == second.keys() and all(torch.equal(first[name], second[name]) for name in first)


def _scenarios(seed: int, excluded: set[str]) -> list[tuple[Any, str]]:
    hard, _hard_ids, _manifest = load_hard_pool("h1_expanded_det")
    hard = [scenario for scenario in hard if scenario.scenario_id not in excluded]
    ordinary = [scenario for scenario in training_scenarios() if scenario.scenario_id not in excluded]
    rng = np.random.default_rng(seed)
    hard_indices = rng.choice(len(hard), size=64, replace=False)
    ordinary_indices = rng.choice(len(ordinary), size=64, replace=False)
    rows = [(hard[int(index)], "hard") for index in hard_indices]
    rows.extend((ordinary[int(index)], "ordinary_training") for index in ordinary_indices)
    order = rng.permutation(len(rows))
    return [rows[int(index)] for index in order]


def _collect_seed(seed: int, policy, excluded: set[str], device: torch.device) -> list[dict[str, Any]]:
    seed_dir = RUN_DIR / "p4" / f"seed_{seed}" / "rollout"
    result_path = seed_dir / "rollout_index.json"
    if result_path.is_file():
        result = read_json(result_path)
        if int(result["seed"]) != seed or len(result["episodes"]) != 128:
            raise RuntimeError(f"Invalid P4 resumable rollout: {result_path}")
        for row in result["episodes"]:
            if sha256_file(ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError(f"P4 rollout hash mismatch: {row['path']}")
        print(f"P4_ROLLOUT_RESUME seed={seed} episodes=128", flush=True)
        return result["episodes"]
    provider = FixedScenarioProvider()
    env = make_env(provider, seed)
    episodes: list[dict[str, Any]] = []
    try:
        for episode_index, (scenario, branch) in enumerate(_scenarios(seed, excluded)):
            destination = seed_dir / "episodes" / f"episode_{episode_index:03d}_{scenario.scenario_id}.npz"
            row = _collect_episode(
                env,
                provider,
                policy,
                scenario,
                branch=branch,
                pool_name=POOL_NAME,
                seed=seed,
                episode_index=episode_index,
                device=device,
                destination=destination,
            )
            episodes.append(row)
            print(
                f"P4_COLLECT seed={seed} episode={episode_index + 1}/128 outcome={row['outcome']}",
                flush=True,
            )
    finally:
        env.close()
    if sum(row["branch"] == "hard" for row in episodes) != 64:
        raise RuntimeError("P4 rollout must contain 64 hard episodes")
    result = {
        "seed": seed,
        "pool": POOL_NAME,
        "episodes": episodes,
        "excluded_panel_ids": sorted(excluded),
    }
    write_json_atomic(result_path, result)
    return episodes


def _raw_advantages(
    episodes: list[dict[str, Any]], selected_credit: str, redistribution_window_seconds: float
) -> list[np.ndarray]:
    standardized, metrics, _mc_metrics = _candidate_advantages(
        episodes, redistribution_window_seconds
    )
    normalization = metrics[selected_credit]["normalization"]
    mean = float(normalization["mean"])
    std = float(normalization["std"])
    return [row.astype(np.float64) * std + mean for row in standardized[selected_credit]]


def _minibatches(episodes: list[dict[str, Any]]) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    current_steps = 0
    for index, row in enumerate(episodes):
        steps = int(row["steps"])
        if steps > BATCH_SIZE:
            raise RuntimeError(f"P4 episode exceeds batch size: {steps}")
        if current and current_steps + steps > BATCH_SIZE:
            batches.append(current)
            current = []
            current_steps = 0
        current.append(index)
        current_steps += steps
    if current:
        batches.append(current)
    return batches


def _normalized_batch_advantages(
    indices: list[int], advantages: list[np.ndarray]
) -> dict[int, np.ndarray]:
    combined = np.concatenate([advantages[index] for index in indices]).astype(np.float64)
    mean = float(combined.mean())
    std = float(combined.std())
    if std <= 1.0e-12 or not np.isfinite(std):
        raise RuntimeError(f"Invalid P4 minibatch advantage std: {std}")
    return {index: ((advantages[index] - mean) / std).astype(np.float32) for index in indices}


def _batch_loss(policy, episodes, advantages, indices, device) -> tuple[torch.Tensor, dict[str, float]]:
    normalized = _normalized_batch_advantages(indices, advantages)
    losses: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    total_steps = sum(int(episodes[index]["steps"]) for index in indices)
    old_replay_error = 0.0
    for index in indices:
        row = episodes[index]
        with np.load(ROOT / row["path"]) as data:
            observations = data["observations"]
            actions = data["actions"]
            old_components = data["old_log_prob_components"]
        new_components = _replay_log_prob_components(policy, observations, actions, device)
        old_tensor = torch.as_tensor(old_components, dtype=torch.float32, device=device)
        if all(not state for state in policy.optimizer.state.values()):
            old_replay_error = max(
                old_replay_error,
                float(torch.max(torch.abs(new_components.detach() - old_tensor)).item()),
            )
        ratio = torch.exp(new_components.sum(dim=1) - old_tensor.sum(dim=1))
        advantage = torch.as_tensor(normalized[index], dtype=torch.float32, device=device)
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * advantage
        episode_loss = -torch.min(unclipped, clipped).mean()
        losses.append(episode_loss * (len(advantage) / total_steps))
        ratios.append(ratio.detach())
    all_ratios = torch.cat(ratios)
    return sum(losses), {
        "valid_steps": total_steps,
        "ratio_mean": float(all_ratios.mean().item()),
        "clip_fraction": float(torch.mean((torch.abs(all_ratios - 1.0) > CLIP_RANGE).float()).item()),
        "old_policy_log_prob_component_replay_max_abs": old_replay_error,
    }


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    values = [parameter.grad.detach().reshape(-1) for parameter in parameters if parameter.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(values)).item()) if values else 0.0


def _set_actor_lrs(policy, factor: float) -> None:
    for group in policy.optimizer.param_groups:
        if group["name"] == "gru":
            group["lr"] = GRU_LR * factor
        elif group["name"] == "head":
            group["lr"] = HEAD_LR * factor


def _full_rollout_gradient_step(policy, episodes, advantages, batches, device, lr_factor: float) -> dict[str, Any]:
    _set_actor_lrs(policy, lr_factor)
    policy.optimizer.zero_grad(set_to_none=True)
    total_steps = sum(int(row["steps"]) for row in episodes)
    telemetry: list[dict[str, float]] = []
    for indices in batches:
        loss, metrics = _batch_loss(policy, episodes, advantages, indices, device)
        weight = metrics["valid_steps"] / total_steps
        (loss * weight).backward()
        telemetry.append(metrics)
    _names, parameters = _trainable_parameters(policy)
    preclip_norm = _gradient_norm(parameters)
    torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
    policy.optimizer.step()
    return {
        "optimizer_steps": 1,
        "preclip_gradient_norm": preclip_norm,
        "minibatches_accumulated": len(batches),
        "minibatches": telemetry,
        "lr_factor": lr_factor,
    }


def _sequential_steps(policy, episodes, advantages, batches, device) -> dict[str, Any]:
    _set_actor_lrs(policy, 1.0)
    telemetry: list[dict[str, Any]] = []
    _names, parameters = _trainable_parameters(policy)
    for batch_index, indices in enumerate(batches):
        policy.optimizer.zero_grad(set_to_none=True)
        loss, metrics = _batch_loss(policy, episodes, advantages, indices, device)
        loss.backward()
        preclip_norm = _gradient_norm(parameters)
        torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
        policy.optimizer.step()
        telemetry.append(
            {"batch_index": batch_index, "preclip_gradient_norm": preclip_norm, **metrics}
        )
    return {
        "optimizer_steps": len(batches),
        "minibatches": telemetry,
        "preclip_gradient_norm_median": float(
            np.median([row["preclip_gradient_norm"] for row in telemetry])
        ),
    }


def _raw_means(actor, observations: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    hidden = torch.zeros((1, 1, actor.gru.hidden_size), dtype=torch.float32, device=device)
    means: list[np.ndarray] = []
    with torch.inference_mode():
        for observation in tensor:
            action, hidden = actor(
                observation[:360].reshape(1, 1, 360),
                observation[360:].reshape(1, 1, 1),
                hidden,
            )
            means.append(action[0, 0].detach().cpu().numpy().astype(np.float64))
    return np.asarray(means, dtype=np.float64)


def _latent_steer(means: np.ndarray) -> np.ndarray:
    normalized = np.clip(
        means[:, 0] / EVALUATOR_STEER_BOUND, -1.0 + 1.0e-6, 1.0 - 1.0e-6
    )
    return np.arctanh(normalized)


def _exact_kl(base_actor, actor, episodes, device) -> dict[str, Any]:
    sequence_values: list[float] = []
    step_values: list[np.ndarray] = []
    for row in episodes:
        with np.load(ROOT / row["path"]) as data:
            observations = data["observations"]
        old = _raw_means(base_actor, observations, device)
        new = _raw_means(actor, observations, device)
        kl = 0.5 * np.square((_latent_steer(old) - _latent_steer(new)) / STEERING_LATENT_STD)
        kl += 0.5 * np.square((old[:, 1] - new[:, 1]) / SPEED_PHYSICAL_STD)
        sequence_values.append(float(np.mean(kl)))
        step_values.append(kl)
    all_steps = np.concatenate(step_values)
    return {
        "mean_exact_kl": float(np.mean(all_steps)),
        "p95_per_sequence_kl": float(np.quantile(sequence_values, 0.95)),
        "p99_per_sequence_kl": float(np.quantile(sequence_values, 0.99)),
        "max_per_sequence_kl": float(np.max(sequence_values)),
        "sequence_count": len(sequence_values),
        "transition_count": len(all_steps),
        "passes_transactional_gate": (
            float(np.mean(all_steps)) <= MEAN_KL_MAX
            and float(np.quantile(sequence_values, 0.99)) <= P99_SEQUENCE_KL_MAX
        ),
    }


def _sampled_policy_metrics(policy, episodes, device) -> dict[str, float]:
    log_ratios: list[np.ndarray] = []
    for row in episodes:
        with np.load(ROOT / row["path"]) as data:
            observations = data["observations"]
            actions = data["actions"]
            old = data["old_log_prob_components"].sum(axis=1)
        new = _replay_log_prob_components(policy, observations, actions, device).sum(dim=1)
        log_ratios.append(new.detach().cpu().numpy().astype(np.float64) - old.astype(np.float64))
    log_ratio = np.concatenate(log_ratios)
    ratio = np.exp(log_ratio)
    return {
        "sampled_approx_kl": float(np.mean((ratio - 1.0) - log_ratio)),
        "sampled_clip_fraction": float(np.mean(np.abs(ratio - 1.0) > CLIP_RANGE)),
        "sampled_ratio_mean": float(np.mean(ratio)),
    }


def _parameter_delta(base_state, policy) -> tuple[torch.Tensor, dict[str, float]]:
    names, parameters = _trainable_parameters(policy)
    delta_parts: list[torch.Tensor] = []
    by_name = dict(policy.end2race_actor.named_parameters())
    for name in names:
        delta_parts.append(by_name[name].detach().cpu().reshape(-1) - base_state[name].reshape(-1))
    delta = torch.cat(delta_parts).double()
    slices = _group_slices(names, parameters)
    return delta, {
        group: float(torch.linalg.vector_norm(delta[group_slice]).item())
        for group, group_slice in slices.items()
    }


def _evaluate_panel(actor, rows: list[dict[str, Any]], seed: int, device: torch.device) -> list[dict[str, Any]]:
    provider = FixedScenarioProvider()
    env = make_env(provider, seed)
    results: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows):
            scenario = scenario_from_dict(row["scenario"])
            result, _trace = run_deterministic_episode(
                env, provider, actor, scenario, device, seed=seed, capture_trace=False
            )
            results.append({"source": row["source"], **result})
            if (index + 1) % 12 == 0:
                print(f"P4_PANEL seed={seed} case={index + 1}/96", flush=True)
    finally:
        env.close()
    return results


def _panel_delta(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    old = {row["scenario_id"]: row for row in before}
    new = {row["scenario_id"]: row for row in after}
    fixed = sorted(
        scenario_id for scenario_id in old if old[scenario_id]["ego_collision"] and not new[scenario_id]["ego_collision"]
    )
    introduced = sorted(
        scenario_id for scenario_id in old if not old[scenario_id]["ego_collision"] and new[scenario_id]["ego_collision"]
    )
    gained = sorted(
        scenario_id for scenario_id in old if old[scenario_id]["outcome"] != "overtake" and new[scenario_id]["outcome"] == "overtake"
    )
    lost = sorted(
        scenario_id for scenario_id in old if old[scenario_id]["outcome"] == "overtake" and new[scenario_id]["outcome"] != "overtake"
    )
    safe_new = sorted(
        scenario_id
        for scenario_id in old
        if old[scenario_id]["source"] == "SAFE_REFERENCE" and new[scenario_id]["ego_collision"]
    )
    return {
        "fixed_collision_count": len(fixed),
        "fixed_collision_ids": fixed,
        "new_collision_count": len(introduced),
        "new_collision_ids": introduced,
        "gained_overtake_count": len(gained),
        "gained_overtake_ids": gained,
        "lost_overtake_count": len(lost),
        "lost_overtake_ids": lost,
        "safe_reference_new_collision_count": len(safe_new),
        "safe_reference_new_collision_ids": safe_new,
    }


def _action_delta_metrics(base_actor, actor, predicted_actor, device) -> dict[str, Any]:
    trace_index = read_json(RUN_DIR / "p0" / "probe_trace_index.json")["rows"]
    actual_by_source: dict[str, list[np.ndarray]] = {"collision": [], "safe": []}
    predicted_by_source: dict[str, list[np.ndarray]] = {"collision": [], "safe": []}
    safe_actual: list[np.ndarray] = []
    for row in trace_index:
        with np.load(ROOT / row["path"]) as data:
            observations = data["observations"]
        base = _physical_actions(base_actor, observations, device)
        actual = _physical_actions(actor, observations, device) - base
        predicted = _physical_actions(predicted_actor, observations, device) - base
        if row["source"] == "H0" and row["outcome"] == "ego_collision":
            key = "collision"
            actual = actual[-min(300, len(actual)) :]
            predicted = predicted[-min(300, len(predicted)) :]
        elif row["source"] == "SAFE_REFERENCE":
            key = "safe"
            safe_actual.append(actual)
        else:
            continue
        actual_by_source[key].append(actual)
        predicted_by_source[key].append(predicted)
    cosine = {}
    for source in ("collision", "safe"):
        actual = torch.as_tensor(np.concatenate(actual_by_source[source]).reshape(-1), dtype=torch.float64)
        predicted = torch.as_tensor(np.concatenate(predicted_by_source[source]).reshape(-1), dtype=torch.float64)
        cosine[source] = _cosine(actual, predicted)
    safe = np.concatenate(safe_actual)
    absolute = np.abs(safe)
    return {
        "p1_predicted_vs_actual_action_delta_cosine": cosine,
        "safe_reference_action_drift": {
            "rms": float(np.sqrt(np.mean(np.square(safe)))),
            "p95_abs": float(np.quantile(absolute, 0.95)),
            "max_abs": float(np.max(absolute)),
        },
    }


def _p1_predicted_actor(p1: dict[str, Any], device: torch.device):
    shards = p1["pools"][POOL_NAME]["shards"]
    vectors = [_load_gradient(row["gradient"]["gradient_file"], "gae_combined") for row in shards]
    weights = [int(row["gradient"]["total_steps"]) for row in shards]
    gradient = _combined(vectors, weights)
    payload = torch.load(ROOT / shards[0]["gradient"]["gradient_file"], map_location="cpu", weights_only=True)
    return _perturbed_actor(
        gradient, payload["parameter_names"], payload["parameter_shapes"], device
    )


def _save_actor(policy, path: Path) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_actor_state(policy), path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def _run_method(
    method: str,
    episodes: list[dict[str, Any]],
    advantages: list[np.ndarray],
    batches: list[list[int]],
    baseline_panel: list[dict[str, Any]],
    panel_rows: list[dict[str, Any]],
    predicted_actor,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    policy = _policy(device)
    policy.set_training_mode(True)
    base_state = _actor_state(policy)
    base_actor = deepcopy(policy.end2race_actor).to(device).eval()
    if method == "S1_SEQUENTIAL_MINIBATCH":
        update = _sequential_steps(policy, episodes, advantages, batches, device)
        kl = _exact_kl(base_actor, policy.end2race_actor, episodes, device)
        transaction = {"committed": True, "attempts": [{"lr_factor": 1.0, "exact_kl": kl}]}
    elif method == "S2_FULL_ROLLOUT_ONE_STEP":
        update = _full_rollout_gradient_step(policy, episodes, advantages, batches, device, 1.0)
        kl = _exact_kl(base_actor, policy.end2race_actor, episodes, device)
        transaction = {"committed": True, "attempts": [{"lr_factor": 1.0, "exact_kl": kl}]}
    else:
        initial_optimizer = deepcopy(policy.optimizer.state_dict())
        attempts: list[dict[str, Any]] = []
        committed = False
        update = {}
        kl = {}
        for factor in BACKTRACK_FACTORS:
            policy.end2race_actor.load_state_dict(base_state, strict=True)
            policy.optimizer.load_state_dict(deepcopy(initial_optimizer))
            update = _full_rollout_gradient_step(
                policy, episodes, advantages, batches, device, factor
            )
            kl = _exact_kl(base_actor, policy.end2race_actor, episodes, device)
            attempts.append({"lr_factor": factor, "exact_kl": kl})
            if kl["passes_transactional_gate"]:
                committed = True
                break
        if not committed:
            policy.end2race_actor.load_state_dict(base_state, strict=True)
            policy.optimizer.load_state_dict(initial_optimizer)
            kl = _exact_kl(base_actor, policy.end2race_actor, episodes, device)
        transaction = {"committed": committed, "attempts": attempts}
    policy.set_training_mode(False)
    sampled = _sampled_policy_metrics(policy, episodes, device)
    after_panel = _evaluate_panel(policy.end2race_actor, panel_rows, seed, device)
    panel = _panel_delta(baseline_panel, after_panel)
    delta, delta_norms = _parameter_delta(base_state, policy)
    action = _action_delta_metrics(
        base_actor, policy.end2race_actor, predicted_actor, device
    )
    checkpoint = _save_actor(
        policy, RUN_DIR / "p4" / f"seed_{seed}" / method / "actor.pth"
    )
    return {
        "method": method,
        "update": update,
        "transaction": transaction,
        "exact_kl": kl,
        "sampled_policy_metrics": sampled,
        "parameter_delta_norm": delta_norms,
        "panel": panel,
        "action_delta": action,
        "actor_checkpoint": checkpoint,
        "after_panel": after_panel,
    }, delta


def main() -> None:
    started = time.monotonic()
    frozen_hashes = assert_frozen_contract()
    preregistration = read_json(PREREGISTRATION_PATH)
    p1 = read_json(EXPERIMENT_DIR / "P1_GRADIENT_DIRECTION.json")
    p2 = read_json(EXPERIMENT_DIR / "P2_COUNTERFACTUAL_ACTIONABILITY.json")
    p3 = read_json(EXPERIMENT_DIR / "P3_CREDIT_HORIZON.json")
    panel = read_json(EXPERIMENT_DIR / "P4_PANEL.json")
    if p1["pool_verdicts"].get(POOL_NAME) != "DIRECTION_PRESENT":
        raise RuntimeError("P4 kill gate: H1 has no stable P1 gradient")
    if p2["verdict"] in {"REWARD_MISALIGNED", "LOCAL_ACTION_NOT_FOUND"}:
        raise RuntimeError(f"P4 kill gate from P2: {p2['verdict']}")
    if p3["status"] != "COMPLETED":
        raise RuntimeError("P4 requires a completed P3")
    if panel["status"] != "FROZEN_BEFORE_P4_ROLLOUT" or len(panel["rows"]) != 96:
        raise RuntimeError("P4 fixed panel is invalid")
    if tuple(preregistration["seeds"]["p4"]) != P4_SEEDS:
        raise RuntimeError("P4 seed contract drifted")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("P4 requires CUDA")
    selected_credit = str(p3["selected_credit"])
    redistribution_window = float(p3["redistribution"]["selected_window_seconds"])
    excluded = {str(row["scenario_id"]) for row in panel["rows"]}
    predicted_actor = _p1_predicted_actor(p1, device)
    seed_records: dict[str, Any] = {}
    delta_vectors: dict[str, list[torch.Tensor]] = {method: [] for method in METHODS}
    collection_policy = _policy(device)
    collection_actor_state = _actor_state(collection_policy)
    for seed in P4_SEEDS:
        episodes = _collect_seed(seed, collection_policy, excluded, device)
        if not _actor_equal(collection_actor_state, _actor_state(collection_policy)):
            raise RuntimeError("P4 collection changed the actor")
        advantages = _raw_advantages(episodes, selected_credit, redistribution_window)
        batches = _minibatches(episodes)
        baseline_actor = deepcopy(collection_policy.end2race_actor).to(device).eval()
        baseline_panel = _evaluate_panel(baseline_actor, panel["rows"], seed, device)
        methods: dict[str, Any] = {}
        for method in METHODS:
            print(f"P4_METHOD_START seed={seed} method={method}", flush=True)
            record, delta = _run_method(
                method,
                episodes,
                advantages,
                batches,
                baseline_panel,
                panel["rows"],
                predicted_actor,
                seed,
                device,
            )
            methods[method] = record
            delta_vectors[method].append(delta)
            print(
                f"P4_METHOD_COMPLETE seed={seed} method={method} "
                f"mean_kl={record['exact_kl']['mean_exact_kl']:.6g} "
                f"fixed={record['panel']['fixed_collision_count']} "
                f"new={record['panel']['new_collision_count']}",
                flush=True,
            )
        seed_record = {
            "seed": seed,
            "complete_episodes": len(episodes),
            "hard_episodes": sum(row["branch"] == "hard" for row in episodes),
            "ordinary_episodes": sum(row["branch"] == "ordinary_training" for row in episodes),
            "actual_ego_collisions": sum(bool(row["ego_collision"]) for row in episodes),
            "rollout_episode_paths": [row["path"] for row in episodes],
            "minibatch_count": len(batches),
            "minibatch_valid_steps": [sum(int(episodes[index]["steps"]) for index in batch) for batch in batches],
            "baseline_panel": baseline_panel,
            "methods": methods,
        }
        seed_path = RUN_DIR / "p4" / f"seed_{seed}" / "seed_result.json"
        write_json_atomic(seed_path, seed_record)
        seed_records[str(seed)] = {
            **seed_record,
            "raw_record": {"path": str(seed_path.relative_to(ROOT)), "sha256": sha256_file(seed_path)},
        }
    direction_consistency = {
        method: [
            _cosine(delta_vectors[method][first], delta_vectors[method][second])
            for first, second in itertools.combinations(range(len(P4_SEEDS)), 2)
        ]
        for method in METHODS
    }
    method_gates: dict[str, Any] = {}
    for method in ("S2_FULL_ROLLOUT_ONE_STEP", "S3_TRANSACTIONAL_BACKTRACKED_ONE_STEP"):
        passing_seeds: list[int] = []
        checks_by_seed: dict[str, Any] = {}
        for seed in P4_SEEDS:
            candidate = seed_records[str(seed)]["methods"][method]
            sequential = seed_records[str(seed)]["methods"]["S1_SEQUENTIAL_MINIBATCH"]
            checks = {
                "fixed_collision_gt_new_collision": (
                    candidate["panel"]["fixed_collision_count"] > candidate["panel"]["new_collision_count"]
                ),
                "safe_reference_new_collision_le_1": candidate["panel"]["safe_reference_new_collision_count"] <= 1,
                "overtake_lost_le_2": candidate["panel"]["lost_overtake_count"] <= 2,
                "mean_exact_kl_smaller_than_s1": (
                    candidate["exact_kl"]["mean_exact_kl"] < sequential["exact_kl"]["mean_exact_kl"]
                ),
                "p99_sequence_exact_kl_smaller_than_s1": (
                    candidate["exact_kl"]["p99_per_sequence_kl"]
                    < sequential["exact_kl"]["p99_per_sequence_kl"]
                ),
                "transaction_committed": candidate["transaction"]["committed"],
            }
            if all(checks.values()):
                passing_seeds.append(seed)
            checks_by_seed[str(seed)] = checks
        method_gates[method] = {
            "checks_by_seed": checks_by_seed,
            "passing_seeds": passing_seeds,
            "passes_two_of_three": len(passing_seeds) >= 2,
        }
    geometry_confirmed = any(row["passes_two_of_three"] for row in method_gates.values())
    verdict = "UPDATE_GEOMETRY_CONFIRMED" if geometry_confirmed else "CONTROLLED_STEP_INSUFFICIENT"
    result = {
        "schema_version": 1,
        "record": "P4_CONTROLLED_ACTOR_STEP",
        "status": "COMPLETED",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_head": preregistration["source"]["head"],
        "device": "cuda",
        "frozen_hashes": frozen_hashes,
        "pool": POOL_NAME,
        "selected_credit": selected_credit,
        "redistribution_window_seconds": redistribution_window,
        "same_rollout_buffer_per_method": True,
        "actor_only_updates": True,
        "critic_optimizer_steps": 0,
        "minibatch_contract": (
            "collection-order complete recurrent sequences greedily packed to at most 1600 valid steps; "
            "zero padding contributes no loss; the identical partition and per-minibatch advantage "
            "normalization are reused by S1, S2, and S3"
        ),
        "panel": {"path": "ppo_experiments/rl_direction_audit/P4_PANEL.json", "sha256": sha256_file(EXPERIMENT_DIR / "P4_PANEL.json")},
        "seeds": seed_records,
        "parameter_delta_pairwise_cosine_across_seeds": direction_consistency,
        "method_gates": method_gates,
        "verdict": verdict,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    write_json_atomic(EXPERIMENT_DIR / "P4_CONTROLLED_STEP.json", result)
    print(f"P4_COMPLETE verdict={verdict} elapsed_seconds={result['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()

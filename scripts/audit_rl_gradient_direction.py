#!/usr/bin/env python3
"""P1: collect fixed no-update rollouts and measure actor-gradient direction stability."""

from __future__ import annotations

from collections import Counter, defaultdict
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
from gymnasium import spaces

try:
    from audit_rl_direction_common import (
        BC_PATH,
        EXPERIMENT_DIR,
        GAMMA,
        PREREGISTRATION_PATH,
        ROOT,
        RUN_DIR,
        SIM_DURATION,
        SPEED_PHYSICAL_STD,
        STEERING_LATENT_STD,
        TIMESTEP,
        FixedScenarioProvider,
        assert_frozen_contract,
        make_env,
        read_json,
        set_determinism,
        sha256_file,
        write_json_atomic,
    )
except ModuleNotFoundError:
    from scripts.audit_rl_direction_common import (
        BC_PATH,
        EXPERIMENT_DIR,
        GAMMA,
        PREREGISTRATION_PATH,
        ROOT,
        RUN_DIR,
        SIM_DURATION,
        SPEED_PHYSICAL_STD,
        STEERING_LATENT_STD,
        TIMESTEP,
        FixedScenarioProvider,
        assert_frozen_contract,
        make_env,
        read_json,
        set_determinism,
        sha256_file,
        write_json_atomic,
    )
from ppo.policy import EVALUATOR_STEER_BOUND, End2RaceGRUPolicy
from ppo.scenarios import ScenarioSpec, load_hard_pool, scenario_from_dict, training_scenarios


POOL_NAMES = (
    "H0_CURRENT_DET",
    "H1_EXPANDED_DET",
    "H2_STOCH_CORE",
    "H3_UNION_CORE",
)
POOL_FILES = {name: name.lower() for name in POOL_NAMES}
TIME_BIN_NAMES = ("0_0.5s", "0.5_1s", "1_2s", "2_3s", "gt_3s")
GAE_LAMBDA = 0.995
MODEL_SEED = 20260717
BOOTSTRAP_SEED = 20260770
FINITE_DIFFERENCE_EPSILON = 1.0e-3
SIGN_ZERO_TOLERANCE = 1.0e-8


def _policy(device: torch.device) -> End2RaceGRUPolicy:
    observation_space = spaces.Box(
        low=np.full((361,), -np.inf, dtype=np.float32),
        high=np.full((361,), np.inf, dtype=np.float32),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=np.asarray((-EVALUATOR_STEER_BOUND, -np.finfo(np.float32).max), dtype=np.float32),
        high=np.asarray((EVALUATOR_STEER_BOUND, np.finfo(np.float32).max), dtype=np.float32),
        dtype=np.float32,
    )
    set_determinism(MODEL_SEED)
    policy = End2RaceGRUPolicy(
        observation_space,
        action_space,
        lambda _progress: 1.0,
        checkpoint_path=BC_PATH,
        critic_profile="C0_RAW_SINGLE_FRAME",
        steering_distribution="squashed_latent",
        steering_latent_std=STEERING_LATENT_STD,
        speed_physical_std=SPEED_PHYSICAL_STD,
    ).to(device)
    policy.end2race_actor.eval()
    policy.value_net.eval()
    return policy


def _trainable_parameters(policy: End2RaceGRUPolicy) -> tuple[list[str], list[torch.nn.Parameter]]:
    rows = [
        (name, parameter)
        for name, parameter in policy.end2race_actor.named_parameters()
        if name.startswith(("gru.", "output_layer."))
    ]
    names = [name for name, _parameter in rows]
    parameters = [parameter for _name, parameter in rows]
    if len(names) != 8 or not all(parameter.requires_grad for parameter in parameters):
        raise RuntimeError(f"Unexpected trainable actor contract: {names}")
    return names, parameters


def _atanh(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(value) - torch.log1p(-value))


def _sample_action(
    raw_mean: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_mean = (raw_mean[0] / EVALUATOR_STEER_BOUND).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    latent_mean = _atanh(normalized_mean)
    steer_noise = torch.randn((), generator=generator, device=raw_mean.device)
    speed_noise = torch.randn((), generator=generator, device=raw_mean.device)
    latent_steer = latent_mean + STEERING_LATENT_STD * steer_noise
    speed = raw_mean[1] + SPEED_PHYSICAL_STD * speed_noise
    steering = EVALUATOR_STEER_BOUND * torch.tanh(latent_steer)
    action = torch.stack((steering, speed))
    normalized_action = (steering / EVALUATOR_STEER_BOUND).clamp(
        -1.0 + torch.finfo(raw_mean.dtype).eps,
        1.0 - torch.finfo(raw_mean.dtype).eps,
    )
    jacobian = math.log(EVALUATOR_STEER_BOUND) + torch.log1p(-normalized_action.square())
    normal_constant = 0.5 * math.log(2.0 * math.pi)
    steer_log_prob = (
        -0.5 * ((latent_steer - latent_mean) / STEERING_LATENT_STD).square()
        - math.log(STEERING_LATENT_STD)
        - normal_constant
        - jacobian
    )
    speed_log_prob = (
        -0.5 * ((speed - raw_mean[1]) / SPEED_PHYSICAL_STD).square()
        - math.log(SPEED_PHYSICAL_STD)
        - normal_constant
    )
    return action, torch.stack((steer_log_prob, speed_log_prob))


def _value(policy: End2RaceGRUPolicy, observation: np.ndarray, device: torch.device) -> float:
    tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).reshape(1, -1)
    with torch.inference_mode():
        value = policy.value_net(tensor).reshape(-1)[0]
    return float(value.detach().cpu().item())


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _collect_episode(
    env,
    provider: FixedScenarioProvider,
    policy: End2RaceGRUPolicy,
    scenario: ScenarioSpec,
    *,
    branch: str,
    pool_name: str,
    seed: int,
    episode_index: int,
    device: torch.device,
    destination: Path,
) -> dict[str, Any]:
    provider.set(scenario, sampler_branch=branch, hard_pool_id=pool_name)
    observation, _reset_info = env.reset(seed=seed)
    hidden = torch.zeros((1, 1, policy.actor_hidden_size), dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 1000 + episode_index)
    observations: list[np.ndarray] = []
    pre_action_hidden: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    old_log_prob_components: list[np.ndarray] = []
    rewards: list[float] = []
    values: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    action_saturation = 0
    info: dict[str, Any] | None = None

    while True:
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        pre_action_hidden.append(hidden[0, 0].detach().cpu().numpy().astype(np.float32, copy=True))
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        lidar = observation_tensor[:360].reshape(1, 1, 360)
        previous_speed = observation_tensor[360:].reshape(1, 1, 1)
        with torch.inference_mode():
            raw_action, next_hidden = policy.end2race_actor(lidar, previous_speed, hidden)
            action, log_prob_components = _sample_action(raw_action[0, 0], generator)
            critic_value = policy.value_net(observation_tensor.reshape(1, -1)).reshape(-1)[0]
        action_numpy = action.detach().cpu().numpy().astype(np.float32, copy=False)
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        action_saturation += int(abs(float(action_numpy[0])) >= EVALUATOR_STEER_BOUND - 1.0e-7)
        actions.append(action_numpy.copy())
        old_log_prob_components.append(log_prob_components.detach().cpu().numpy().astype(np.float32, copy=False))
        rewards.append(float(reward))
        values.append(float(critic_value.detach().cpu().item()))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        hidden = next_hidden
        if terminated or truncated:
            break
        if len(rewards) > 1000:
            raise RuntimeError(f"P1 episode exceeded 1000 steps: {scenario.scenario_id}")
    if info is None:
        raise RuntimeError("P1 episode produced no transition")
    bootstrap_value = _value(policy, observation, device) if truncated_flags[-1] else 0.0
    ego_collision = bool(info["ego_collision"])
    relative_position = float(info["relative_position_m"])
    outcome = "ego_collision" if ego_collision else ("overtake" if relative_position > 0.0 else "follow")
    _atomic_savez(
        destination,
        observations=np.asarray(observations, dtype=np.float32),
        pre_action_hidden=np.asarray(pre_action_hidden, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        old_log_prob_components=np.asarray(old_log_prob_components, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=np.bool_),
        truncated=np.asarray(truncated_flags, dtype=np.bool_),
        bootstrap_value=np.asarray([bootstrap_value], dtype=np.float32),
    )
    return {
        "scenario_id": scenario.scenario_id,
        "scenario": scenario.to_dict(),
        "branch": branch,
        "outcome": outcome,
        "ego_collision": ego_collision,
        "steps": len(rewards),
        "elapsed_time": float(info["elapsed_time"]),
        "final_relative_position_m": relative_position,
        "steering_latent_saturation_steps": action_saturation,
        "path": str(destination.relative_to(ROOT)),
        "sha256": sha256_file(destination),
    }


def _time_bins(length: int) -> np.ndarray:
    time_to_end = (length - np.arange(length, dtype=np.float64)) * TIMESTEP
    return np.select(
        [time_to_end <= 0.5, time_to_end <= 1.0, time_to_end <= 2.0, time_to_end <= 3.0],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int64)


def _discounted_returns(rewards: np.ndarray) -> np.ndarray:
    returns = np.empty_like(rewards, dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + GAMMA * running
        returns[index] = running
    return returns


def _gae(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_value: float,
) -> np.ndarray:
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
        running = delta + GAMMA * GAE_LAMBDA * continuation * running
        advantages[index] = running
    return advantages


def _standardize(rows: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, float]]:
    combined = np.concatenate(rows).astype(np.float64, copy=False)
    mean = float(combined.mean())
    std = float(combined.std())
    if not np.isfinite(std) or std <= 1.0e-12:
        raise RuntimeError(f"Advantage standard deviation is invalid: {std}")
    return [((row - mean) / std).astype(np.float32) for row in rows], {"mean": mean, "std": std}


def _quantiles(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "std": float(np.std(values)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
    }


def _advantage_rows(episode_rows: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    small: list[dict[str, Any]] = []
    global_sums: defaultdict[tuple[str, int], float] = defaultdict(float)
    global_counts: Counter[tuple[str, int]] = Counter()
    for row in episode_rows:
        with np.load(ROOT / row["path"]) as data:
            rewards = data["rewards"].astype(np.float64)
            values = data["values"].astype(np.float64)
            bootstrap_value = float(data["bootstrap_value"][0])
        returns = _discounted_returns(rewards)
        gae = _gae(rewards, values, bootstrap_value)
        bins = _time_bins(len(rewards))
        per_episode: dict[tuple[str, int], tuple[float, int]] = {}
        for bin_index in range(5):
            mask = bins == bin_index
            key = (str(row["branch"]), bin_index)
            subtotal = float(returns[mask].sum())
            count = int(mask.sum())
            global_sums[key] += subtotal
            global_counts[key] += count
            per_episode[key] = (subtotal, count)
        small.append(
            {
                "returns": returns,
                "gae": gae,
                "bins": bins,
                "per_episode": per_episode,
                "branch": str(row["branch"]),
            }
        )
    mc_rows: list[np.ndarray] = []
    gae_rows = [entry["gae"] for entry in small]
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
        mc_rows.append(entry["returns"] - baseline)
    standardized_gae, gae_normalization = _standardize(gae_rows)
    standardized_mc, mc_normalization = _standardize(mc_rows)
    raw_gae = np.concatenate(gae_rows)
    raw_mc = np.concatenate(mc_rows)
    correlation = float(np.corrcoef(raw_gae, raw_mc)[0, 1])
    sign_agreement = float(np.mean(np.sign(raw_gae) == np.sign(raw_mc)))
    return standardized_gae, standardized_mc, {
        "gae_raw": _quantiles(raw_gae),
        "mc_raw": _quantiles(raw_mc),
        "gae_normalization": gae_normalization,
        "mc_normalization": mc_normalization,
        "gae_mc_correlation": correlation,
        "gae_mc_sign_agreement": sign_agreement,
    }


def _replay_log_prob_components(
    policy: End2RaceGRUPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    hidden = torch.zeros((1, 1, policy.actor_hidden_size), dtype=torch.float32, device=device)
    means: list[torch.Tensor] = []
    for observation in observation_tensor:
        raw_action, hidden = policy.end2race_actor(
            observation[:360].reshape(1, 1, 360),
            observation[360:].reshape(1, 1, 1),
            hidden,
        )
        means.append(raw_action[0, 0])
    mean = torch.stack(means)
    normalized_mean = (mean[:, 0] / EVALUATOR_STEER_BOUND).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    latent_mean = _atanh(normalized_mean)
    normalized_action = (action_tensor[:, 0] / EVALUATOR_STEER_BOUND).clamp(
        -1.0 + torch.finfo(action_tensor.dtype).eps,
        1.0 - torch.finfo(action_tensor.dtype).eps,
    )
    latent_action = _atanh(normalized_action)
    jacobian = math.log(EVALUATOR_STEER_BOUND) + torch.log1p(-normalized_action.square())
    constant = 0.5 * math.log(2.0 * math.pi)
    steer = (
        -0.5 * ((latent_action - latent_mean) / STEERING_LATENT_STD).square()
        - math.log(STEERING_LATENT_STD)
        - constant
        - jacobian
    )
    speed = (
        -0.5 * ((action_tensor[:, 1] - mean[:, 1]) / SPEED_PHYSICAL_STD).square()
        - math.log(SPEED_PHYSICAL_STD)
        - constant
    )
    return torch.stack((steer, speed), dim=1)


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(first).item() * torch.linalg.vector_norm(second).item())
    if denominator <= 0.0:
        return 0.0
    return float(torch.dot(first, second).item() / denominator)


def _group_slices(names: list[str], parameters: list[torch.nn.Parameter]) -> dict[str, slice]:
    offsets = [0]
    for parameter in parameters:
        offsets.append(offsets[-1] + parameter.numel())
    gru_end = max(offsets[index + 1] for index, name in enumerate(names) if name.startswith("gru."))
    return {"gru": slice(0, gru_end), "head": slice(gru_end, offsets[-1]), "combined": slice(0, offsets[-1])}


def _norms(vector: torch.Tensor, slices: dict[str, slice]) -> dict[str, float]:
    return {name: float(torch.linalg.vector_norm(vector[group_slice]).item()) for name, group_slice in slices.items()}


def _gradient_shard(
    policy: End2RaceGRUPolicy,
    episode_rows: list[dict[str, Any]],
    destination: Path,
    device: torch.device,
) -> dict[str, Any]:
    names, parameters = _trainable_parameters(policy)
    slices = _group_slices(names, parameters)
    gae_rows, mc_rows, advantage_metrics = _advantage_rows(episode_rows)
    total_steps = int(sum(row["steps"] for row in episode_rows))
    batched_accumulators = [torch.zeros((8, *parameter.shape), dtype=torch.float32, device=device) for parameter in parameters]
    replay_max_abs = 0.0
    torch.backends.cudnn.enabled = False
    for episode_index, (row, gae, mc) in enumerate(zip(episode_rows, gae_rows, mc_rows), start=1):
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
        bins = _time_bins(len(gae))
        coefficients = torch.zeros((8, len(gae), 2), dtype=torch.float32, device=device)
        gae_tensor = torch.as_tensor(gae, dtype=torch.float32, device=device)
        mc_tensor = torch.as_tensor(mc, dtype=torch.float32, device=device)
        coefficients[0, :, :] = gae_tensor[:, None]
        coefficients[1, :, :] = mc_tensor[:, None]
        coefficients[2, :, 0] = gae_tensor
        for bin_index in range(5):
            mask = torch.as_tensor(bins == bin_index, dtype=torch.bool, device=device)
            coefficients[3 + bin_index, mask, :] = gae_tensor[mask, None]
        gradients = torch.autograd.grad(
            log_prob_components,
            parameters,
            grad_outputs=coefficients,
            is_grads_batched=True,
            retain_graph=False,
            create_graph=False,
        )
        for accumulator, gradient in zip(batched_accumulators, gradients):
            accumulator.add_(gradient.detach())
        if episode_index % 4 == 0 or episode_index == len(episode_rows):
            print(f"P1_GRADIENT episode={episode_index}/{len(episode_rows)}", flush=True)
    flat = torch.cat(
        [(gradient / total_steps).reshape(8, -1) for gradient in batched_accumulators],
        dim=1,
    )
    gae_combined = flat[0].detach().cpu()
    mc_combined = flat[1].detach().cpu()
    gae_steering = flat[2].detach().cpu()
    gae_speed = gae_combined - gae_steering
    time_vectors = [flat[3 + index].detach().cpu() for index in range(5)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "parameter_names": names,
            "parameter_shapes": [list(parameter.shape) for parameter in parameters],
            "parameter_numels": [parameter.numel() for parameter in parameters],
            "gae_combined": gae_combined,
            "mc_combined": mc_combined,
        },
        destination,
    )
    metrics = {
        "total_steps": total_steps,
        "gradient_file": str(destination.relative_to(ROOT)),
        "gradient_file_sha256": sha256_file(destination),
        "parameter_elements": int(gae_combined.numel()),
        "gae_gradient_norm": _norms(gae_combined, slices),
        "mc_gradient_norm": _norms(mc_combined, slices),
        "gae_mc_gradient_cosine": {
            group: _cosine(gae_combined[group_slice], mc_combined[group_slice])
            for group, group_slice in slices.items()
        },
        "score_function_contribution": {
            "steering_norm": _norms(gae_steering, slices),
            "speed_norm": _norms(gae_speed, slices),
            "steering_speed_cosine": {
                group: _cosine(gae_steering[group_slice], gae_speed[group_slice])
                for group, group_slice in slices.items()
            },
        },
        "time_to_end_gradient_norm": {
            name: _norms(vector, slices) for name, vector in zip(TIME_BIN_NAMES, time_vectors)
        },
        "replay_log_prob_component_max_abs": replay_max_abs,
        "advantage": advantage_metrics,
    }
    del flat, batched_accumulators, time_vectors
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = True
    return metrics


def _load_gradient(path: str, estimator: str = "gae_combined") -> torch.Tensor:
    return torch.load(ROOT / path, map_location="cpu", weights_only=True)[estimator].double()


def _pairwise_metrics(vectors: list[torch.Tensor], slices: dict[str, slice]) -> dict[str, list[float]]:
    return {
        group: [
            _cosine(vectors[first][group_slice], vectors[second][group_slice])
            for first, second in itertools.combinations(range(len(vectors)), 2)
        ]
        for group, group_slice in slices.items()
    }


def _bootstrap_ci(vectors: list[torch.Tensor], group_slice: slice, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    cosine_matrix = np.eye(len(vectors), dtype=np.float64)
    for first, second in itertools.combinations(range(len(vectors)), 2):
        value = _cosine(vectors[first][group_slice], vectors[second][group_slice])
        cosine_matrix[first, second] = value
        cosine_matrix[second, first] = value
    values: list[float] = []
    for _replicate in range(10000):
        sampled = rng.integers(0, len(vectors), size=len(vectors))
        pair_values = [
            float(cosine_matrix[first, second])
            for first, second in itertools.combinations(sampled.tolist(), 2)
            if first != second
        ]
        if pair_values:
            values.append(float(np.median(pair_values)))
    if not values:
        raise RuntimeError("Bootstrap produced no distinct shard pairs")
    return {
        "replicates_requested": 10000,
        "replicates_valid": len(values),
        "lower": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper": float(np.quantile(values, 0.975)),
    }


def _combined(vectors: list[torch.Tensor], weights: list[int]) -> torch.Tensor:
    denominator = float(sum(weights))
    return sum(vector * weight for vector, weight in zip(vectors, weights)) / denominator


def _physical_actions(actor, observations: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=device).reshape(1, -1, 361)
    with torch.inference_mode():
        action, _hidden = actor(tensor[:, :, :360], tensor[:, :, 360:], None)
    physical = action[0].detach().cpu().numpy().astype(np.float64)
    physical[:, 0] = np.clip(physical[:, 0], -EVALUATOR_STEER_BOUND, EVALUATOR_STEER_BOUND)
    return physical


def _perturbed_actor(
    gradient: torch.Tensor,
    names: list[str],
    shapes: list[list[int]],
    device: torch.device,
):
    from model import End2Race

    actor = End2Race(mask_prob=0.0, hidden_scale=4).to(device)
    actor.load_state_dict(torch.load(BC_PATH, map_location=device, weights_only=True), strict=True)
    norm = float(torch.linalg.vector_norm(gradient).item())
    if norm <= 0.0:
        raise RuntimeError("Cannot perturb actor along a zero gradient")
    direction = gradient.float() / norm
    offset = 0
    by_name = dict(actor.named_parameters())
    with torch.no_grad():
        for name, shape in zip(names, shapes):
            count = int(np.prod(shape))
            by_name[name].add_(
                FINITE_DIFFERENCE_EPSILON
                * direction[offset : offset + count].reshape(shape).to(device)
            )
            offset += count
    if offset != direction.numel():
        raise RuntimeError("Gradient vector does not match actor parameter contract")
    actor.eval()
    return actor


def _probe_metrics(
    gradient_records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    index = read_json(RUN_DIR / "p0" / "probe_trace_index.json")["rows"]
    gradient_payloads = [
        torch.load(ROOT / record["gradient_file"], map_location="cpu", weights_only=True)
        for record in gradient_records
    ]
    names = gradient_payloads[0]["parameter_names"]
    shapes = gradient_payloads[0]["parameter_shapes"]
    from model import End2Race

    base_actor = End2Race(mask_prob=0.0, hidden_scale=4).to(device)
    base_actor.load_state_dict(torch.load(BC_PATH, map_location=device, weights_only=True), strict=True)
    base_actor.eval()
    base_actions: dict[str, np.ndarray] = {}
    observations_by_id: dict[str, np.ndarray] = {}
    for row in index:
        with np.load(ROOT / row["path"]) as data:
            observations = data["observations"]
        observations_by_id[row["scenario_id"]] = observations
        base_actions[row["scenario_id"]] = _physical_actions(base_actor, observations, device)

    collision_deltas: list[np.ndarray] = []
    safe_deltas: list[np.ndarray] = []
    safe_drift: list[dict[str, float]] = []
    for payload in gradient_payloads:
        actor = _perturbed_actor(payload["gae_combined"], names, shapes, device)
        collision_rows: list[np.ndarray] = []
        safe_rows: list[np.ndarray] = []
        for row in index:
            scenario_id = str(row["scenario_id"])
            perturbed = _physical_actions(actor, observations_by_id[scenario_id], device)
            delta = (perturbed - base_actions[scenario_id]) / FINITE_DIFFERENCE_EPSILON
            if row["source"] == "H0" and row["outcome"] == "ego_collision":
                collision_rows.append(delta[-min(300, len(delta)) :])
            elif row["source"] == "SAFE_REFERENCE":
                safe_rows.append(delta)
        collision_flat = np.concatenate(collision_rows, axis=0)
        safe_flat = np.concatenate(safe_rows, axis=0)
        collision_deltas.append(collision_flat)
        safe_deltas.append(safe_flat)
        absolute_safe = np.abs(safe_flat)
        safe_drift.append(
            {
                "rms_per_unit_parameter_l2": float(np.sqrt(np.mean(np.square(safe_flat)))),
                "p95_abs_per_unit_parameter_l2": float(np.quantile(absolute_safe, 0.95)),
                "max_abs_per_unit_parameter_l2": float(np.max(absolute_safe)),
            }
        )

    collision_stack = np.stack(collision_deltas, axis=0)
    sign_records: dict[str, Any] = {}
    dimension_agreements: list[float] = []
    for dimension, name in enumerate(("steering", "speed")):
        values = collision_stack[:, :, dimension]
        positive = np.sum(values > SIGN_ZERO_TOLERANCE, axis=0)
        negative = np.sum(values < -SIGN_ZERO_TOLERANCE, axis=0)
        nonzero = positive + negative
        valid = nonzero > 0
        agreement = np.maximum(positive[valid], negative[valid]) / nonzero[valid]
        mean_agreement = float(np.mean(agreement)) if agreement.size else 0.0
        sign_records[name] = {
            "mean_majority_sign_agreement": mean_agreement,
            "valid_state_fraction": float(np.mean(valid)),
            "state_count": int(values.shape[1]),
        }
        dimension_agreements.append(mean_agreement)

    def pairwise_action_cosine(rows: list[np.ndarray]) -> list[float]:
        tensors = [torch.as_tensor(row.reshape(-1), dtype=torch.float64) for row in rows]
        return [
            _cosine(tensors[first], tensors[second])
            for first, second in itertools.combinations(range(len(tensors)), 2)
        ]

    return {
        "finite_difference_epsilon": FINITE_DIFFERENCE_EPSILON,
        "gradient_normalization": "unit combined parameter L2",
        "collision_probe_reproduced_h0_count": 22,
        "collision_probe_window_seconds": 3.0,
        "collision_action_delta_sign_agreement": float(np.mean(dimension_agreements)),
        "sign_agreement_by_dimension": sign_records,
        "collision_action_delta_pairwise_cosine": pairwise_action_cosine(collision_deltas),
        "safe_action_delta_pairwise_cosine": pairwise_action_cosine(safe_deltas),
        "safe_predicted_drift": safe_drift,
    }


def _aggregate_pool(
    pool_name: str,
    shard_records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(ROOT / shard_records[0]["gradient"]["gradient_file"], map_location="cpu", weights_only=True)
    names = payload["parameter_names"]
    dummy_parameters = [torch.empty(shape) for shape in payload["parameter_shapes"]]
    slices = _group_slices(names, dummy_parameters)
    gae_vectors = [_load_gradient(record["gradient"]["gradient_file"], "gae_combined") for record in shard_records]
    mc_vectors = [_load_gradient(record["gradient"]["gradient_file"], "mc_combined") for record in shard_records]
    weights = [int(record["gradient"]["total_steps"]) for record in shard_records]
    pairwise = _pairwise_metrics(gae_vectors, slices)
    bootstrap = {
        group: _bootstrap_ci(gae_vectors, group_slice, BOOTSTRAP_SEED + list(slices).index(group))
        for group, group_slice in slices.items()
    }
    final = _combined(gae_vectors, weights)
    cumulative: dict[str, Any] = {}
    cumulative_levels = [(1, "32"), (2, "64"), (4, "128")]
    if len(gae_vectors) >= 8:
        cumulative_levels.append((8, "256"))
    for shard_count, label in cumulative_levels:
        partial = _combined(gae_vectors[:shard_count], weights[:shard_count])
        cumulative[label] = {
            group: _cosine(partial[group_slice], final[group_slice])
            for group, group_slice in slices.items()
        }
    probe = _probe_metrics([record["gradient"] for record in shard_records], device)
    median_pairwise = float(np.median(pairwise["combined"]))
    ci = bootstrap["combined"]
    gae_mc = [record["gradient"]["gae_mc_gradient_cosine"]["combined"] for record in shard_records]
    median_gae_mc = float(np.median(gae_mc))
    probe_agreement = float(probe["collision_action_delta_sign_agreement"])
    direction_present = (
        median_pairwise >= 0.15
        and float(ci["lower"]) > 0.0
        and probe_agreement >= 0.70
    )
    absent_reasons: list[str] = []
    if float(ci["upper"]) <= 0.05:
        absent_reasons.append("PAIRWISE_COSINE_CI_UPPER_LE_0.05")
    if probe_agreement <= 0.55:
        absent_reasons.append("PROBE_ACTION_DELTA_SIGN_AGREEMENT_LE_0.55")
    if median_gae_mc <= 0.0:
        absent_reasons.append("MEDIAN_GAE_MC_GRADIENT_COSINE_LE_0")
    if absent_reasons:
        verdict = "DIRECTION_ABSENT_OR_CONFLICTING"
    elif direction_present:
        verdict = "DIRECTION_PRESENT"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "pool": pool_name,
        "complete_episodes": sum(len(record["episodes"]) for record in shard_records),
        "hard_episodes": sum(sum(row["branch"] == "hard" for row in record["episodes"]) for record in shard_records),
        "ordinary_training_episodes": sum(
            sum(row["branch"] == "ordinary_training" for row in record["episodes"]) for record in shard_records
        ),
        "actual_ego_collisions": sum(
            sum(bool(row["ego_collision"]) for row in record["episodes"]) for record in shard_records
        ),
        "unique_collision_scenario_ids": sorted(
            {
                row["scenario_id"]
                for record in shard_records
                for row in record["episodes"]
                if row["ego_collision"]
            }
        ),
        "shards": shard_records,
        "pairwise_gradient_cosine": pairwise,
        "pairwise_combined_median": median_pairwise,
        "bootstrap_95ci": bootstrap,
        "cumulative_gradient_cosine_to_final": cumulative,
        "gae_mc_gradient_cosine_by_shard": gae_mc,
        "median_gae_mc_gradient_cosine": median_gae_mc,
        "probe": probe,
        "verdict": verdict,
        "absent_or_conflicting_reasons": absent_reasons,
        "extension_to_256_allowed": verdict == "INCONCLUSIVE",
    }


def _shard_scenarios(
    pool_name: str,
    seed: int,
) -> list[tuple[ScenarioSpec, str]]:
    hard_scenarios, _hard_ids, _manifest = load_hard_pool(POOL_FILES[pool_name])
    ordinary = training_scenarios()
    rng = np.random.default_rng(seed)
    hard_indices = rng.choice(len(hard_scenarios), size=16, replace=False)
    ordinary_indices = rng.choice(len(ordinary), size=16, replace=False)
    rows = [(hard_scenarios[int(index)], "hard") for index in hard_indices]
    rows.extend((ordinary[int(index)], "ordinary_training") for index in ordinary_indices)
    order = rng.permutation(len(rows))
    return [rows[int(index)] for index in order]


def _collection_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(str(row["outcome"]) for row in episodes)
    branches = Counter(str(row["branch"]) for row in episodes)
    visits = Counter(str(row["scenario_id"]) for row in episodes if row["branch"] == "hard")
    action_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    for row in episodes:
        with np.load(ROOT / row["path"]) as data:
            action_rows.append(data["actions"])
            reward_rows.append(data["rewards"])
    actions = np.concatenate(action_rows, axis=0).astype(np.float64)
    rewards = np.concatenate(reward_rows, axis=0).astype(np.float64)
    return {
        "complete_episodes": len(episodes),
        "branch_counts": dict(sorted(branches.items())),
        "outcomes": {name: int(outcomes[name]) for name in ("ego_collision", "follow", "overtake")},
        "actual_ego_collisions": int(outcomes["ego_collision"]),
        "unique_collision_scenario_count": len({row["scenario_id"] for row in episodes if row["ego_collision"]}),
        "hard_sampler_visit_counts": dict(sorted(visits.items())),
        "transitions": int(len(actions)),
        "action_statistics": {
            "steering": {
                "mean": float(actions[:, 0].mean()),
                "std": float(actions[:, 0].std()),
                "min": float(actions[:, 0].min()),
                "max": float(actions[:, 0].max()),
            },
            "speed": {
                "mean": float(actions[:, 1].mean()),
                "std": float(actions[:, 1].std()),
                "min": float(actions[:, 1].min()),
                "max": float(actions[:, 1].max()),
            },
        },
        "reward_statistics": {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "min": float(rewards.min()),
            "max": float(rewards.max()),
        },
        "steering_latent_saturation_steps": int(
            sum(row["steering_latent_saturation_steps"] for row in episodes)
        ),
    }


def main() -> None:
    started = time.monotonic()
    frozen_hashes = assert_frozen_contract()
    preregistration = read_json(PREREGISTRATION_PATH)
    safe_reference = read_json(EXPERIMENT_DIR / "SAFE_REFERENCE.json")
    if safe_reference["status"] != "FROZEN" or safe_reference["selected_count"] != 48:
        raise RuntimeError("SAFE_REFERENCE is not frozen at 48 cases")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("P1 requires CUDA")
    policy = _policy(device)
    initial_actor_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.end2race_actor.state_dict().items()
    }
    all_pool_records: dict[str, Any] = {}
    for pool_name in POOL_NAMES:
        pool_result_path = RUN_DIR / "p1" / pool_name / "pool_result.json"
        if pool_result_path.is_file():
            all_pool_records[pool_name] = read_json(pool_result_path)
            print(
                f"P1_POOL_RESUME pool={pool_name} "
                f"verdict={all_pool_records[pool_name]['verdict']}",
                flush=True,
            )
            continue
        shard_records: list[dict[str, Any]] = []
        seeds = preregistration["seeds"]["p1"][pool_name]
        for shard_index, seed in enumerate(seeds):
            shard_dir = RUN_DIR / "p1" / pool_name / f"shard_{shard_index}"
            shard_result_path = shard_dir / "shard_result.json"
            if shard_result_path.is_file():
                shard_record = read_json(shard_result_path)
                if len(shard_record["episodes"]) != 32 or int(shard_record["seed"]) != int(seed):
                    raise RuntimeError(f"Invalid resumable shard record: {shard_result_path}")
                gradient_path = ROOT / shard_record["gradient"]["gradient_file"]
                if sha256_file(gradient_path) != shard_record["gradient"]["gradient_file_sha256"]:
                    raise RuntimeError(f"Resumable shard gradient hash mismatch: {gradient_path}")
                shard_records.append(shard_record)
                print(
                    f"P1_SHARD_RESUME pool={pool_name} shard={shard_index} "
                    f"seed={seed} collisions={shard_record['collection']['actual_ego_collisions']}",
                    flush=True,
                )
                continue
            print(f"P1_SHARD_START pool={pool_name} shard={shard_index} seed={seed}", flush=True)
            provider = FixedScenarioProvider()
            env = make_env(provider, int(seed))
            episodes: list[dict[str, Any]] = []
            try:
                for episode_index, (scenario, branch) in enumerate(_shard_scenarios(pool_name, int(seed))):
                    destination = shard_dir / "episodes" / f"episode_{episode_index:02d}_{scenario.scenario_id}.npz"
                    row = _collect_episode(
                        env,
                        provider,
                        policy,
                        scenario,
                        branch=branch,
                        pool_name=pool_name,
                        seed=int(seed),
                        episode_index=episode_index,
                        device=device,
                        destination=destination,
                    )
                    episodes.append(row)
                    print(
                        f"P1_COLLECT pool={pool_name} shard={shard_index} "
                        f"episode={episode_index + 1}/32 outcome={row['outcome']} steps={row['steps']}",
                        flush=True,
                    )
            finally:
                env.close()
            collection = _collection_summary(episodes)
            gradient = _gradient_shard(
                policy,
                episodes,
                shard_dir / "gradients.pt",
                device,
            )
            shard_record = {
                "shard_index": shard_index,
                "seed": int(seed),
                "episodes": episodes,
                "collection": collection,
                "gradient": gradient,
            }
            write_json_atomic(shard_dir / "shard_result.json", shard_record)
            shard_records.append(shard_record)
            current_actor_state = policy.end2race_actor.state_dict()
            if any(
                not torch.equal(current_actor_state[name].detach().cpu(), reference)
                for name, reference in initial_actor_state.items()
            ):
                raise RuntimeError("Actor parameters changed during P1 no-update audit")
            print(
                f"P1_SHARD_COMPLETE pool={pool_name} shard={shard_index} "
                f"collisions={collection['actual_ego_collisions']} "
                f"gae_mc_cos={gradient['gae_mc_gradient_cosine']['combined']:.6f}",
                flush=True,
            )
        all_pool_records[pool_name] = _aggregate_pool(pool_name, shard_records, device)
        write_json_atomic(pool_result_path, all_pool_records[pool_name])
        print(
            f"P1_POOL_COMPLETE pool={pool_name} verdict={all_pool_records[pool_name]['verdict']} "
            f"median_cos={all_pool_records[pool_name]['pairwise_combined_median']:.6f}",
            flush=True,
        )

    verdicts = {name: record["verdict"] for name, record in all_pool_records.items()}
    status = "INCONCLUSIVE_EXTENSION_ALLOWED" if "INCONCLUSIVE" in verdicts.values() else "COMPLETED"
    result = {
        "schema_version": 1,
        "record": "P1_GRADIENT_DIRECTION_AND_SAMPLE_SCALING",
        "status": status,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_head": preregistration["source"]["head"],
        "preregistration_commit": "725e0b741e7342a2774f2e0250d7c654440cea02",
        "safe_reference_sha256": sha256_file(EXPERIMENT_DIR / "SAFE_REFERENCE.json"),
        "device": "cuda",
        "optimizer_steps": 0,
        "actor_parameters_bitwise_unchanged": True,
        "frozen_hashes": frozen_hashes,
        "pools": all_pool_records,
        "pool_verdicts": verdicts,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    write_json_atomic(EXPERIMENT_DIR / "P1_GRADIENT_DIRECTION.json", result)
    print(
        f"P1_COMPLETE status={status} elapsed_seconds={result['elapsed_seconds']:.1f} "
        f"verdicts={json.dumps(verdicts, sort_keys=True)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

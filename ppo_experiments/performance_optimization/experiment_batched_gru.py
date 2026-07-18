#!/usr/bin/env python3
"""Numerical experiments for batched End2Race GRU execution.

This file deliberately does not patch the production policy.  It measures the
three requested batching variants against the scalar-kernel reference on one
frozen formal rollout and can run the collection-time variant through the
existing closed-loop contract harness.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time
from types import MethodType
from typing import Any, Callable

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo import config as ppo_config
from ppo.policy import (
    END2RACE_ACTION_SIZE,
    END2RACE_LIDAR_SIZE,
    END2RACE_OBSERVATION_SIZE,
)
from train_ppo import PPOTrainingCallback, build_model, build_sampler, build_training_vector_env


ActorForward = Callable[..., tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("frozen-minibatch", "closed-loop-a"))
    parser.add_argument("--config", default="N1-H1F-p50", choices=ppo_config.CONFIGS)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--worker-count", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def tensor_difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    difference = (candidate.detach() - reference.detach()).abs().float().cpu().reshape(-1)
    if difference.numel() == 0:
        return {"count": 0, "nonzero": 0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "count": int(difference.numel()),
        "nonzero": int(torch.count_nonzero(difference).item()),
        "max": float(difference.max().item()),
        "mean": float(difference.mean().item()),
        "p50": float(torch.quantile(difference, 0.50).item()),
        "p90": float(torch.quantile(difference, 0.90).item()),
        "p99": float(torch.quantile(difference, 0.99).item()),
        "p999": float(torch.quantile(difference, 0.999).item()),
    }


@dataclass
class StreamingDifference:
    """Exact count/mean/max plus a deterministic sample for quantiles."""

    count: int = 0
    nonzero: int = 0
    total: float = 0.0
    maximum: float = 0.0
    samples: list[np.ndarray] = field(default_factory=list)

    def add(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        values = (candidate.detach() - reference.detach()).abs().float().cpu().numpy().reshape(-1)
        if values.size == 0:
            return
        self.count += int(values.size)
        self.nonzero += int(np.count_nonzero(values))
        self.total += float(values.sum(dtype=np.float64))
        self.maximum = max(self.maximum, float(values.max()))
        stride = max(1, values.size // 512)
        self.samples.append(values[::stride][:512].copy())

    def record(self) -> dict[str, Any]:
        sample = np.concatenate(self.samples) if self.samples else np.zeros(1, dtype=np.float32)
        return {
            "count": self.count,
            "nonzero": self.nonzero,
            "max": self.maximum,
            "mean": 0.0 if self.count == 0 else self.total / self.count,
            "sample_count_for_quantiles": int(sample.size),
            "p50_sampled": float(np.quantile(sample, 0.50)),
            "p90_sampled": float(np.quantile(sample, 0.90)),
            "p99_sampled": float(np.quantile(sample, 0.99)),
            "p999_sampled": float(np.quantile(sample, 0.999)),
        }


def _sequence_layout(policy: Any, obs: Any, states: Any, episode_starts: torch.Tensor):
    hidden, _dummy_cell = states
    actor_obs = policy._actor_observation(obs).float()
    if actor_obs.ndim == 1:
        actor_obs = actor_obs.unsqueeze(0)
    n_seq = hidden.shape[1]
    obs_sequence = actor_obs.reshape(n_seq, -1, END2RACE_OBSERVATION_SIZE).swapaxes(0, 1)
    start_sequence = episode_starts.float().reshape(n_seq, -1).swapaxes(0, 1)
    return hidden, actor_obs, n_seq, obs_sequence, start_sequence


def timestep_batched_actor_forward(
    policy: Any,
    obs: Any,
    states: Any,
    episode_starts: torch.Tensor,
    valid_by_timestep: tuple[tuple[bool, ...], ...] | None = None,
):
    """Variant A/B: one GRU call for all valid sequence slots per timestep."""

    hidden, actor_obs, n_seq, obs_sequence, start_sequence = _sequence_layout(
        policy, obs, states, episode_starts
    )
    means: list[torch.Tensor] = []
    timestep_hidden: list[torch.Tensor] = []
    for timestep, (step_obs, episode_start) in enumerate(zip(obs_sequence, start_sequence)):
        hidden = hidden * (1.0 - episode_start).view(1, n_seq, 1)
        if valid_by_timestep is None:
            valid_indices = list(range(n_seq))
        else:
            valid_indices = [index for index, valid in enumerate(valid_by_timestep[timestep]) if valid]
        valid_tensor = torch.as_tensor(valid_indices, dtype=torch.long, device=actor_obs.device)
        if valid_indices:
            action_sequence, next_valid_hidden = policy.end2race_actor(
                step_obs[valid_tensor, :END2RACE_LIDAR_SIZE].unsqueeze(1),
                step_obs[valid_tensor, END2RACE_LIDAR_SIZE:].unsqueeze(1),
                hidden[:, valid_tensor],
            )
            valid_means = action_sequence[:, -1, :]
            hidden_by_slot = []
            means_by_slot = []
            valid_position = {index: position for position, index in enumerate(valid_indices)}
            for index in range(n_seq):
                position = valid_position.get(index)
                if position is None:
                    hidden_by_slot.append(hidden[:, index : index + 1])
                    means_by_slot.append(torch.zeros((1, END2RACE_ACTION_SIZE), device=actor_obs.device))
                else:
                    hidden_by_slot.append(next_valid_hidden[:, position : position + 1])
                    means_by_slot.append(valid_means[position : position + 1])
            hidden = torch.cat(hidden_by_slot, dim=1)
            means.append(torch.cat(means_by_slot, dim=0))
        else:
            means.append(torch.zeros((n_seq, END2RACE_ACTION_SIZE), device=actor_obs.device))
        timestep_hidden.append(hidden.squeeze(0))
    mean_actions = torch.stack(means).transpose(0, 1).reshape(-1, END2RACE_ACTION_SIZE)
    actor_features = torch.stack(timestep_hidden).transpose(0, 1).reshape(-1, policy.actor_hidden_size)
    return mean_actions, (hidden, torch.zeros_like(hidden)), actor_features


def packed_actor_forward(
    policy: Any,
    obs: Any,
    states: Any,
    episode_starts: torch.Tensor,
    valid_by_timestep: tuple[tuple[bool, ...], ...] | None = None,
):
    """Variant C: one packed full-sequence GRU call, without regrouping data."""

    hidden, actor_obs, n_seq, obs_sequence, start_sequence = _sequence_layout(
        policy, obs, states, episode_starts
    )
    max_length = obs_sequence.shape[0]
    if valid_by_timestep is None:
        lengths = [max_length] * n_seq
    else:
        lengths = [sum(valid_by_timestep[timestep][index] for timestep in range(max_length)) for index in range(n_seq)]
    for index, length in enumerate(lengths):
        if torch.count_nonzero(start_sequence[1:length, index]).item() != 0:
            raise RuntimeError("Packed experiment requires the unchanged sequencer episode boundaries")
    hidden = hidden * (1.0 - start_sequence[0]).view(1, n_seq, 1)
    batch_obs = obs_sequence.swapaxes(0, 1)
    lidar = batch_obs[:, :, :END2RACE_LIDAR_SIZE]
    speed = batch_obs[:, :, END2RACE_LIDAR_SIZE:]
    processed_lidar = (-1.0 / (1.0 + torch.exp(-policy.end2race_actor.k * lidar)) + 1.0) * 2.0
    speed_embedding = policy.end2race_actor.speed_mlp(speed)
    features = torch.cat((processed_lidar, speed_embedding), dim=2)
    packed = pack_padded_sequence(
        features,
        torch.as_tensor(lengths, dtype=torch.long).cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    packed_output, next_hidden = policy.end2race_actor.gru(packed, hidden)
    actor_features, _ = pad_packed_sequence(
        packed_output,
        batch_first=True,
        total_length=max_length,
    )
    mean_actions = policy.end2race_actor.output_layer(actor_features)
    valid_mask = torch.arange(max_length, device=actor_obs.device).unsqueeze(0) < torch.as_tensor(
        lengths, device=actor_obs.device
    ).unsqueeze(1)
    mean_actions = torch.where(valid_mask.unsqueeze(-1), mean_actions, torch.zeros_like(mean_actions))
    return (
        mean_actions.reshape(-1, END2RACE_ACTION_SIZE),
        (next_hidden, torch.zeros_like(next_hidden)),
        actor_features.reshape(-1, policy.actor_hidden_size),
    )


def cuda_elapsed_ms(function: Callable[[], Any]) -> float:
    function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _parameter_vectors(policy: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }


def _gradient_vectors(policy: Any) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }


def _mapping_difference(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict[str, Any]:
    ref = torch.cat([reference[name].float().reshape(-1).cpu() for name in sorted(reference)])
    cand = torch.cat([candidate[name].float().reshape(-1).cpu() for name in sorted(reference)])
    return tensor_difference(ref, cand)


def evaluate_first_batch(
    model: Any,
    samples: Any,
    actor_forward: ActorForward,
    initial_policy: dict[str, torch.Tensor],
    initial_optimizer: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    policy = model.policy
    policy.load_state_dict(initial_policy, strict=True)
    policy.optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
    policy.optimizer.zero_grad(set_to_none=True)
    policy.set_training_mode(True)
    original_actor_forward = policy._actor_forward
    policy._actor_forward = MethodType(actor_forward, policy)
    actions = samples.actions
    mask = samples.mask > 1e-8
    means, next_states, actor_features = policy._actor_forward(
        samples.observations,
        samples.lstm_states.pi,
        samples.episode_starts,
        policy._actor_hidden_rollout_buffer.current_valid_by_timestep,
    )
    distribution = policy._distribution(means)
    log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()
    values = policy._critic_values(samples.observations, actor_features).flatten()
    advantages = samples.advantages
    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
    ratio = torch.exp(log_prob - samples.old_log_prob)
    clip_range = model.clip_range(model._current_progress_remaining)
    policy_loss = -torch.mean(
        torch.min(
            advantages * ratio,
            advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range),
        )[mask]
    )
    value_loss = torch.mean(((samples.returns - values) ** 2)[mask])
    entropy_loss = -torch.mean(entropy[mask]) if entropy is not None else torch.mean(log_prob[mask])
    loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss
    log_ratio = log_prob - samples.old_log_prob
    approx_kl = torch.mean(((torch.exp(log_ratio) - 1.0) - log_ratio)[mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
    gradients = _gradient_vectors(policy)
    before = _parameter_vectors(policy)
    policy.optimizer.step()
    after = _parameter_vectors(policy)
    deltas = {name: after[name] - before[name] for name in before}
    policy._actor_forward = original_actor_forward
    record = {
        "valid_samples": int(mask.sum().item()),
        "padded_samples": int(mask.numel()),
        "ratio": {
            "min": float(ratio[mask].min().item()),
            "max": float(ratio[mask].max().item()),
            "mean": float(ratio[mask].mean().item()),
        },
        "approx_kl": float(approx_kl.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy_loss": float(entropy_loss.item()),
        "total_loss": float(loss.item()),
    }
    outputs = {
        "means": means.detach(),
        "log_prob": log_prob.detach(),
        "values": values.detach(),
        "hidden": next_states[0].detach(),
    }
    return record, outputs, gradients, deltas


class RolloutCaptured(RuntimeError):
    pass


def frozen_minibatch_experiment(config_name: str, seed: int, worker_count: int) -> dict[str, Any]:
    config = ppo_config.get_config(config_name)
    sampler = build_sampler(config)
    vector_env = build_training_vector_env(sampler, config, seed, worker_count=worker_count)
    vector_env.seed(seed)
    model = build_model(vector_env, config, seed)
    callback = PPOTrainingCallback()
    result: dict[str, Any] = {}

    def capture_train() -> None:
        np.random.seed(51973)
        samples = next(model.rollout_buffer.get(config.batch_size))
        valid_by_timestep = model.rollout_buffer.current_valid_by_timestep
        initial_policy = copy.deepcopy(model.policy.state_dict())
        initial_optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
        variants: dict[str, ActorForward] = {
            "reference_batch1": type(model.policy)._actor_forward,
            "B_timestep_batch": timestep_batched_actor_forward,
            "C_full_sequence_packed": packed_actor_forward,
        }
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        gradients: dict[str, dict[str, torch.Tensor]] = {}
        deltas: dict[str, dict[str, torch.Tensor]] = {}
        records: dict[str, Any] = {}
        timings: dict[str, float] = {}
        for name, variant in variants.items():
            records[name], outputs[name], gradients[name], deltas[name] = evaluate_first_batch(
                model,
                samples,
                variant,
                initial_policy,
                initial_optimizer,
            )
            original_actor_forward = model.policy._actor_forward
            model.policy._actor_forward = MethodType(variant, model.policy)
            timings[name] = cuda_elapsed_ms(
                lambda: model.policy._actor_forward(
                    samples.observations,
                    samples.lstm_states.pi,
                    samples.episode_starts,
                    valid_by_timestep,
                )
            )
            model.policy._actor_forward = original_actor_forward
        reference = "reference_batch1"
        comparisons = {}
        mask = samples.mask > 1e-8
        for name in ("B_timestep_batch", "C_full_sequence_packed"):
            comparisons[name] = {
                "mean_action_abs_diff": tensor_difference(outputs[reference]["means"][mask], outputs[name]["means"][mask]),
                "logp_abs_diff": tensor_difference(outputs[reference]["log_prob"][mask], outputs[name]["log_prob"][mask]),
                "value_abs_diff": tensor_difference(outputs[reference]["values"][mask], outputs[name]["values"][mask]),
                "final_hidden_abs_diff": tensor_difference(outputs[reference]["hidden"], outputs[name]["hidden"]),
                "gradient_abs_diff": _mapping_difference(gradients[reference], gradients[name]),
                "parameter_delta_abs_diff": _mapping_difference(deltas[reference], deltas[name]),
                "first_batch_metric_delta": {
                    key: records[name][key] - records[reference][key]
                    for key in ("approx_kl", "policy_loss", "value_loss", "entropy_loss", "total_loss")
                },
            }
        result.update(
            {
                "schema_version": 1,
                "experiment": "phase5_training_b_c",
                "config": config_name,
                "seed": seed,
                "worker_count": worker_count,
                "rollout_transitions": ppo_config.N_ENVS * config.n_steps,
                "logical_minibatch": {
                    "valid_samples": int((samples.mask > 1e-8).sum().item()),
                    "padded_samples": int(samples.mask.numel()),
                    "sequence_count": len(valid_by_timestep[0]),
                    "max_length": len(valid_by_timestep),
                    "internal_episode_starts": int(
                        samples.episode_starts.reshape(len(valid_by_timestep[0]), -1)[:, 1:].sum().item()
                    ),
                },
                "metrics": records,
                "actor_forward_cuda_ms_warmup_then_repeat1": timings,
                "comparisons_to_reference": comparisons,
                "production_policy_modified": False,
                "merge_decision": "owner_required_for_any_nonzero_numerical_difference",
            }
        )
        raise RolloutCaptured

    model.train = capture_train
    vector_env.env_method("set_policy_update_index", 1)
    started = time.perf_counter()
    try:
        with torch.autograd.set_multithreading_enabled(config.autograd_multithreading):
            model.learn(
                total_timesteps=ppo_config.N_ENVS * config.n_steps,
                callback=callback,
                log_interval=None,
                reset_num_timesteps=True,
                progress_bar=False,
            )
    except RolloutCaptured:
        pass
    finally:
        vector_env.close()
    result["wall_s_including_formal_rollout"] = time.perf_counter() - started
    return result


def closed_loop_a_experiment(config_name: str, seed: int, worker_count: int) -> dict[str, Any]:
    """Run A through the complete contract harness, while measuring reference drift."""

    from ppo_experiments.performance_optimization import validate_pipeline

    original_build_model = validate_pipeline.build_model
    model_holder: dict[str, Any] = {}
    streams = {
        "sampled_action": StreamingDifference(),
        "old_logp": StreamingDifference(),
        "value": StreamingDifference(),
        "actor_hidden": StreamingDifference(),
    }
    first_snapshot: dict[str, Any] = {}

    def instrumented_build_model(vector_env: Any, config: Any, model_seed: int):
        model = original_build_model(vector_env, config, model_seed)
        model_holder["model"] = model
        policy = model.policy
        reference_forward = policy.forward

        def forward(observation, lstm_states, episode_starts, deterministic=False):
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all()
            reference = reference_forward(observation, lstm_states, episode_starts, deterministic)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            original_actor_forward = policy._actor_forward
            policy._actor_forward = MethodType(timestep_batched_actor_forward, policy)
            try:
                candidate = reference_forward(observation, lstm_states, episode_starts, deterministic)
            finally:
                policy._actor_forward = original_actor_forward
            streams["sampled_action"].add(reference[0], candidate[0])
            streams["value"].add(reference[1], candidate[1])
            streams["old_logp"].add(reference[2], candidate[2])
            streams["actor_hidden"].add(reference[3].pi[0], candidate[3].pi[0])
            if not first_snapshot:
                first_snapshot.update(
                    {
                        "observation": observation.detach().clone(),
                        "states": tuple(state.detach().clone() for state in lstm_states.pi),
                        "episode_starts": episode_starts.detach().clone(),
                    }
                )
            return candidate

        policy.forward = forward
        return model

    validate_pipeline.build_model = instrumented_build_model
    try:
        contract = validate_pipeline.capture(
            config_name,
            seed,
            "central_subproc",
            worker_count,
            False,
            "phase5_collection_batch_a",
        )
    finally:
        validate_pipeline.build_model = original_build_model
    timing_ms = None
    if first_snapshot and "model" in model_holder:
        policy = model_holder["model"].policy
        observation = first_snapshot["observation"]
        states = first_snapshot["states"]
        starts = first_snapshot["episode_starts"]
        reference_ms = cuda_elapsed_ms(lambda: policy._actor_forward(observation, states, starts))
        batched_ms = cuda_elapsed_ms(
            lambda: timestep_batched_actor_forward(policy, observation, states, starts)
        )
        timing_ms = {"reference_batch1": reference_ms, "A_collection_env_batch": batched_ms}
    return {
        "schema_version": 1,
        "experiment": "phase5_collection_a_closed_loop",
        "config": config_name,
        "seed": seed,
        "worker_count": worker_count,
        "numerical_differences": {name: stream.record() for name, stream in streams.items()},
        "single_collection_forward_cuda_ms_warmup_then_repeat1": timing_ms,
        "closed_loop_contract": contract,
        "production_policy_modified": False,
        "merge_decision": "owner_required_for_any_nonzero_numerical_difference",
    }


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.command == "frozen-minibatch":
        output = frozen_minibatch_experiment(arguments.config, arguments.seed, arguments.worker_count)
    else:
        output = closed_loop_a_experiment(arguments.config, arguments.seed, arguments.worker_count)
    write_json(arguments.output, output)
    print(json.dumps({"output": str(arguments.output), "experiment": output["experiment"]}, sort_keys=True))

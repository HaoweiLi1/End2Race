#!/usr/bin/env python3
"""Capture a stepwise numerical contract for one formal PPO update."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo import config as ppo_config
from train_ppo import (
    PPOTrainingCallback,
    build_model,
    build_sampler,
    make_training_env,
    make_subprocess_training_env,
    save_actor,
    write_json,
)


CONTRACT_INFO_KEYS = (
    "ego_collision",
    "opponent_collision",
    "opponent_collision_latched",
    "timeout",
    "termination_reason",
    "scenario_id",
    "sampler_branch",
    "env_role",
    "pair_group",
    "pair_member",
    "pair_episode_ordinal",
    "policy_update_index",
    "episode_outcome",
    "elapsed_time",
    "episode_return",
    "reward_progress",
    "reward_relative",
    "reward_margin",
    "reward_collision",
    "reward_total",
    "ego_progress_delta_m",
    "opponent_progress_delta_m",
    "relative_position_m",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="N1-H1F-p50", choices=ppo_config.CONFIGS)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--mode", default="dummy", choices=("dummy", "central_subproc"))
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--zero-lr", action="store_true")
    parser.add_argument("--disable-padding-skip", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-output", type=Path)
    return parser.parse_args()


def hash_array(value: Any) -> str:
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def hash_zero_array(shape: tuple[int, ...], dtype: np.dtype = np.dtype(np.float32)) -> str:
    digest = hashlib.sha256()
    digest.update(str(dtype).encode())
    digest.update(str(shape).encode())
    remaining = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    zeros = bytes(1024 * 1024)
    while remaining:
        size = min(remaining, len(zeros))
        digest.update(zeros[:size])
        remaining -= size
    return digest.hexdigest()


def hash_observation(value: torch.Tensor | dict[str, torch.Tensor] | np.ndarray | dict[str, np.ndarray]) -> str:
    if isinstance(value, dict):
        return hash_json({key: hash_array(value[key]) for key in sorted(value)})
    return hash_array(value)


def hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize_nested(value: Any) -> Any:
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return {"array_sha256": hash_array(value)}
    if isinstance(value, dict):
        return [
            [repr(key), normalize_nested(item)]
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        ]
    if isinstance(value, (list, tuple)):
        return [normalize_nested(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def reset_spec_record(spec: Any) -> dict[str, Any]:
    return {
        "poses": np.asarray(spec.poses, dtype=np.float64).tolist(),
        "initial_speed_feature": float(spec.initial_speed_feature),
        "scenario": dict(spec.scenario),
    }


def instrument_reset_specs(base_factory):
    """Attach reset specs to validation-only infos without changing production."""

    def factory():
        env = base_factory()
        pending: list[Any] = []
        original_provider = env.reset_provider

        def provider(rng):
            spec = original_provider(rng)
            pending.append(spec)
            return spec

        env.reset_provider = provider
        original_reset = env.reset

        def reset(*args, **kwargs):
            options = kwargs.get("options")
            external = None if options is None else options.get("end2race_episode_reset_spec")
            result = original_reset(*args, **kwargs)
            spec = external if external is not None else pending.pop(0)
            result[1]["_validation_reset_spec"] = reset_spec_record(spec)
            return result

        env.reset = reset
        return env

    return factory


def parameter_record(
    policy: torch.nn.Module,
    initial: dict[str, torch.Tensor],
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for name, parameter in policy.named_parameters():
        current = parameter.detach().cpu()
        baseline = initial[name]
        delta = current - baseline
        gradient = None if parameter.grad is None else parameter.grad.detach().cpu()
        parameters[name] = {
            "parameter_sha256": hash_array(current),
            "delta_sha256": hash_array(delta),
            "delta_max_abs": float(delta.abs().max().item()),
            "gradient_sha256": None if gradient is None else hash_array(gradient),
            "gradient_max_abs": None if gradient is None else float(gradient.abs().max().item()),
        }
    return parameters


def build_vector_env(
    mode: str,
    config: ppo_config.PPOConfig,
    sampler: Any,
    seed: int,
    worker_count: int | None,
):
    if mode == "dummy":
        factories = [
            instrument_reset_specs(make_training_env(rank, sampler, config, seed))
            for rank in range(ppo_config.N_ENVS)
        ]
        return DummyVecEnv(factories)
    from ppo.vec_env import CentralScheduleSubprocVecEnv

    factories = [
        instrument_reset_specs(make_subprocess_training_env(rank, config, seed))
        for rank in range(ppo_config.N_ENVS)
    ]
    return CentralScheduleSubprocVecEnv(
        factories,
        sampler=sampler,
        config=config,
        seed=seed,
        worker_count=worker_count,
    )


def capture(
    config_name: str,
    seed: int,
    mode: str,
    worker_count: int | None,
    zero_lr: bool,
    label: str,
    action_output: Path | None = None,
    disable_padding_skip: bool = False,
) -> dict[str, Any]:
    config = ppo_config.get_config(config_name)
    sampler = build_sampler(config)
    vector_env = build_vector_env(mode, config, sampler, seed, worker_count)
    vector_env.seed(seed)
    model = build_model(vector_env, config, seed)
    if disable_padding_skip:
        model.policy._actor_hidden_rollout_buffer = None
    callback = PPOTrainingCallback()
    initial_parameters = {
        name: parameter.detach().cpu().clone() for name, parameter in model.policy.named_parameters()
    }
    optimizer_initial_sha256 = hash_json(normalize_nested(model.policy.optimizer.state_dict()))
    if zero_lr:
        for group in model.policy.optimizer.param_groups:
            group["lr"] = 0.0
            group["base_lr"] = 0.0

    step_hashes: dict[str, list[str]] = {
        key: []
        for key in (
            "policy_observation",
            "policy_episode_starts",
            "policy_action",
            "policy_value",
            "policy_old_logp",
            "policy_pi_hidden_in",
            "policy_pi_cell_in",
            "policy_vf_hidden_in",
            "policy_vf_cell_in",
            "policy_pi_hidden_out",
            "policy_pi_cell_out",
            "policy_vf_hidden_out",
            "policy_vf_cell_out",
            "executed_action",
            "next_observation",
            "reward",
            "done",
            "info_contract",
        )
    }
    reset_order: list[tuple[Any, ...]] = []
    reset_specs: list[dict[str, Any]] = []
    events: Counter[str] = Counter()
    executed_actions: list[np.ndarray] = []

    original_reset = vector_env.reset

    def append_resets(indices: list[int]) -> None:
        for index in indices:
            info = vector_env.reset_infos[index]
            reset_order.append(
                (
                    index,
                    str(info["scenario_id"]),
                    str(info["env_role"]),
                    info.get("pair_group"),
                    info.get("pair_member"),
                    info.get("pair_episode_ordinal"),
                )
            )
            reset_specs.append(
                {
                    "env_rank": index,
                    **info["_validation_reset_spec"],
                }
            )

    def reset():
        observation = original_reset()
        append_resets(list(range(ppo_config.N_ENVS)))
        return observation

    vector_env.reset = reset
    original_step_async = vector_env.step_async

    def step_async(actions: np.ndarray) -> None:
        step_hashes["executed_action"].append(hash_array(actions))
        executed_actions.append(np.asarray(actions).copy())
        original_step_async(actions)

    vector_env.step_async = step_async
    original_step_wait = vector_env.step_wait

    def step_wait():
        observation, rewards, dones, infos = original_step_wait()
        step_hashes["next_observation"].append(hash_observation(observation))
        step_hashes["reward"].append(hash_array(rewards))
        step_hashes["done"].append(hash_array(dones))
        info_contract = []
        reset_indices = []
        for index, (done, info) in enumerate(zip(dones, infos)):
            contract = {key: info.get(key) for key in CONTRACT_INFO_KEYS}
            contract["terminated"] = bool(done and not info["TimeLimit.truncated"])
            contract["truncated"] = bool(info["TimeLimit.truncated"])
            info_contract.append(contract)
            events["terminated"] += int(contract["terminated"])
            events["truncated"] += int(contract["truncated"])
            events["ego_collision"] += int(bool(info["ego_collision"]))
            events["opponent_collision"] += int(bool(info["opponent_collision"]))
            events["timeout"] += int(bool(info["timeout"]))
            if info["episode_outcome"] is not None:
                events[str(info["episode_outcome"])] += 1
            if done:
                reset_indices.append(index)
        step_hashes["info_contract"].append(hash_json(info_contract))
        append_resets(reset_indices)
        return observation, rewards, dones, infos

    vector_env.step_wait = step_wait
    original_forward = model.policy.forward

    def policy_forward(observation, lstm_states, episode_starts, deterministic=False):
        step_hashes["policy_observation"].append(hash_observation(observation))
        step_hashes["policy_episode_starts"].append(hash_array(episode_starts))
        step_hashes["policy_pi_hidden_in"].append(hash_array(lstm_states.pi[0]))
        step_hashes["policy_pi_cell_in"].append(hash_array(lstm_states.pi[1]))
        step_hashes["policy_vf_hidden_in"].append(hash_array(lstm_states.vf[0]))
        step_hashes["policy_vf_cell_in"].append(hash_array(lstm_states.vf[1]))
        actions, values, log_prob, next_states = original_forward(
            observation, lstm_states, episode_starts, deterministic
        )
        step_hashes["policy_action"].append(hash_array(actions))
        step_hashes["policy_value"].append(hash_array(values))
        step_hashes["policy_old_logp"].append(hash_array(log_prob))
        step_hashes["policy_pi_hidden_out"].append(hash_array(next_states.pi[0]))
        step_hashes["policy_pi_cell_out"].append(hash_array(next_states.pi[1]))
        step_hashes["policy_vf_hidden_out"].append(hash_array(next_states.vf[0]))
        step_hashes["policy_vf_cell_out"].append(hash_array(next_states.vf[1]))
        return actions, values, log_prob, next_states

    model.policy.forward = policy_forward

    # Episode-start resets must make arbitrary incoming actor hidden irrelevant.
    actor_observation = torch.zeros((ppo_config.N_ENVS, 361), device=model.device)
    random_hidden = torch.arange(
        ppo_config.N_ENVS * model.policy.actor_hidden_size,
        dtype=torch.float32,
        device=model.device,
    ).reshape(1, ppo_config.N_ENVS, model.policy.actor_hidden_size)
    zeros = torch.zeros_like(random_hidden)
    starts = torch.ones(ppo_config.N_ENVS, device=model.device)
    with torch.no_grad():
        reset_mean, reset_state = model.policy.actor_mean(actor_observation, (random_hidden, zeros), starts)
        zero_mean, zero_state = model.policy.actor_mean(actor_observation, (zeros, zeros), starts)
    episode_start_reset = {
        "mean_exact": bool(torch.equal(reset_mean, zero_mean)),
        "hidden_exact": bool(torch.equal(reset_state[0], zero_state[0])),
        "cell_exact": bool(torch.equal(reset_state[1], zero_state[1])),
    }

    frozen_rollout: dict[str, str] = {}
    replay_identity: dict[str, Any] = {}
    original_train = model.train

    def train():
        for name in (
            "observations",
            "actions",
            "rewards",
            "episode_starts",
            "values",
            "log_probs",
            "advantages",
            "returns",
            "hidden_states_pi",
            "cell_states_pi",
            "hidden_states_vf",
            "cell_states_vf",
        ):
            if hasattr(model.rollout_buffer, name):
                frozen_rollout[name] = hash_array(getattr(model.rollout_buffer, name))
            else:
                frozen_rollout[name] = hash_zero_array(model.rollout_buffer.hidden_state_shape)
        numpy_state = np.random.get_state()
        samples = next(model.rollout_buffer.get(config.batch_size))
        model.policy.set_training_mode(True)
        with torch.no_grad():
            values, log_prob, _entropy = model.policy.evaluate_actions(
                samples.observations,
                samples.actions,
                samples.lstm_states,
                samples.episode_starts,
            )
        mask = samples.mask > 1e-8
        replay_identity.update(
            {
                "valid_samples": int(mask.sum().item()),
                "old_logp_exact": bool(torch.equal(log_prob[mask], samples.old_log_prob[mask])),
                "old_logp_max_abs": float((log_prob[mask] - samples.old_log_prob[mask]).abs().max().item()),
                "value_exact": bool(torch.equal(values.flatten()[mask], samples.old_values[mask])),
                "value_max_abs": float((values.flatten()[mask] - samples.old_values[mask]).abs().max().item()),
                "action_sha256": hash_array(samples.actions[mask]),
            }
        )
        np.random.set_state(numpy_state)
        return original_train()

    model.train = train
    vector_env.env_method("set_policy_update_index", 1)
    start = time.perf_counter()
    with torch.autograd.set_multithreading_enabled(config.autograd_multithreading):
        model.learn(
            total_timesteps=ppo_config.N_ENVS * config.n_steps,
            callback=callback,
            log_interval=None,
            reset_num_timesteps=True,
            progress_bar=False,
        )
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - start
    parameter_results = parameter_record(model.policy, initial_parameters)
    parameters_unchanged = all(record["delta_max_abs"] == 0.0 for record in parameter_results.values())
    with tempfile.TemporaryDirectory(prefix=f"end2race_contract_{label}_") as directory:
        checkpoint = Path(directory) / "actor.pth"
        checkpoint_sha256 = save_actor(model, checkpoint)
        checkpoint_bytes = checkpoint.stat().st_size
        checkpoint_keys = sorted(
            torch.load(checkpoint, map_location="cpu", weights_only=True).keys()
        )
    action_array = np.stack(executed_actions)
    if action_output is not None:
        np.save(action_output, action_array)
    result = {
        "schema_version": 1,
        "label": label,
        "head": torch_version_safe_git_head(),
        "config": config_name,
        "seed": seed,
        "mode": mode,
        "worker_count": worker_count,
        "zero_lr": zero_lr,
        "disable_padding_skip": disable_padding_skip,
        "wall_s": wall_s,
        "scenario_manifest_sha256": hashlib.sha256(
            (PROJECT_ROOT / "ppo" / "hard_pools" / "h1_expanded_det.json").read_bytes()
        ).hexdigest(),
        "reset_order": reset_order,
        "reset_order_sha256": hash_json(reset_order),
        "reset_specs": reset_specs,
        "reset_specs_sha256": hash_json(reset_specs),
        "sampler_visit_sha256": hash_json(sampler.visit_counts),
        "step_hashes": step_hashes,
        "step_contract_sha256": {key: hash_json(value) for key, value in step_hashes.items()},
        "saved_action_sequence_sha256": hash_array(action_array),
        "frozen_rollout_sha256": frozen_rollout,
        "replay_identity": replay_identity,
        "episode_start_hidden_reset": episode_start_reset,
        "events": dict(sorted(events.items())),
        "outcomes": callback.latest["completed_episodes"],
        "roles": callback.latest.get("env_role_transitions"),
        "logger": {
            key: float(value)
            for key, value in model.logger.name_to_value.items()
            if key.startswith("train/") and isinstance(value, (float, int, np.floating, np.integer))
        },
        "parameters": parameter_results,
        "parameters_unchanged": parameters_unchanged,
        "optimizer_state": {
            "initial_sha256": optimizer_initial_sha256,
            "final_sha256": hash_json(normalize_nested(model.policy.optimizer.state_dict())),
        },
        "checkpoint": {
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_bytes,
            "keys": checkpoint_keys,
            "strict_12_key_load": len(checkpoint_keys) == 12,
        },
    }
    vector_env.close()
    return result


def torch_version_safe_git_head() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    arguments = parse_arguments()
    contract = capture(
        arguments.config,
        arguments.seed,
        arguments.mode,
        arguments.worker_count,
        arguments.zero_lr,
        arguments.label,
        arguments.action_output,
        arguments.disable_padding_skip,
    )
    write_json(arguments.output, contract)
    print(
        "CONTRACT_RESULT "
        + json.dumps(
            {
                "label": contract["label"],
                "wall_s": contract["wall_s"],
                "reset_order_sha256": contract["reset_order_sha256"],
                "frozen_rollout_sha256": contract["frozen_rollout_sha256"],
                "step_contract_sha256": contract["step_contract_sha256"],
                "replay_identity": contract["replay_identity"],
                "episode_start_hidden_reset": contract["episode_start_hidden_reset"],
                "events": contract["events"],
                "outcomes": contract["outcomes"],
                "roles": contract["roles"],
                "parameters_unchanged": contract["parameters_unchanged"],
                "checkpoint": contract["checkpoint"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

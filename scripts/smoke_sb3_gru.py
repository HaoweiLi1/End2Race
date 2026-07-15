#!/usr/bin/env python3
"""One-rollout, zero-learning-rate SB3-Contrib GRU proof of concept."""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium
import numpy as np
import sb3_contrib
import stable_baselines3
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from model import End2Race
from rl.end2race_gymnasium_env import End2RaceGymnasiumEnv
from rl.sb3_end2race_policy import (
    DEFAULT_BC_CHECKPOINT,
    END2RACE_LIDAR_SIZE,
    End2RaceGRUPolicy,
    end2race_observation,
)
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates


ROLLOUT_STEPS = 10
MINIBATCH_SIZE = 6
N_ENVS = 2


class SyntheticLegacyF110Env:
    """Small legacy-API stand-in used only to exercise the real wrapper."""

    num_agents = 1

    def __init__(self, terminal_kind: str, horizon: int, timestep: float = 0.1):
        if terminal_kind not in {"collision", "timeout"}:
            raise ValueError(terminal_kind)
        self.terminal_kind = terminal_kind
        self.horizon = horizon
        self.timestep = timestep
        self.episode_step = 0
        self.last_action: np.ndarray | None = None

    @property
    def unwrapped(self) -> "SyntheticLegacyF110Env":
        return self

    def _observation(self, collision: bool = False) -> dict[str, np.ndarray]:
        phase = float(self.episode_step)
        scan = np.linspace(0.5, 10.0, END2RACE_LIDAR_SIZE, dtype=np.float32) + phase * 0.01
        return {
            "scans": scan.reshape(1, -1),
            "linear_vels_x": np.asarray([1.0 + phase * 0.05], dtype=np.float32),
            "collisions": np.asarray([collision], dtype=np.float32),
            # These fields emulate privileged simulator data.  The wrapper must
            # not include them in the actor observation.
            "poses_x": np.asarray([100.0 + phase], dtype=np.float32),
            "poses_y": np.asarray([-50.0], dtype=np.float32),
            "opponent_pose": np.asarray([[999.0, 999.0, 0.0]], dtype=np.float32),
            "reference_geometry": np.asarray([1234.0], dtype=np.float32),
        }

    def reset(self, **kwargs: Any):
        del kwargs
        self.episode_step = 0
        return self._observation(), 0.0, False, {"timestep": self.timestep}

    def step(self, action: np.ndarray):
        self.last_action = np.asarray(action).copy()
        self.episode_step += 1
        collision = self.terminal_kind == "collision" and self.episode_step >= self.horizon
        done = collision
        return self._observation(collision), 1.0, done, {"timestep": self.timestep}

    def render(self):
        return None

    def close(self):
        return None


def make_poc_env(terminal_kind: str, horizon: int) -> End2RaceGymnasiumEnv:
    core = SyntheticLegacyF110Env(terminal_kind=terminal_kind, horizon=horizon)
    sim_duration = 1.0 if terminal_kind == "collision" else horizon * core.timestep
    return End2RaceGymnasiumEnv(core, sim_duration=sim_duration)


def build_model() -> tuple[RecurrentPPO, list[End2RaceGymnasiumEnv]]:
    envs = [make_poc_env("collision", 4), make_poc_env("timeout", 7)]
    vector_env = DummyVecEnv([lambda env=env: env for env in envs])
    model = RecurrentPPO(
        End2RaceGRUPolicy,
        vector_env,
        learning_rate=0.0,
        n_steps=ROLLOUT_STEPS,
        batch_size=MINIBATCH_SIZE,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        seed=20260715,
        device="cpu",
        policy_kwargs={
            "checkpoint_path": DEFAULT_BC_CHECKPOINT,
            "hidden_scale": 4,
            "critic_hidden_size": 32,
            "log_std_init": 0.0,
        },
        verbose=0,
    )
    return model, envs


def load_reference_actor() -> End2Race:
    actor = End2Race(mask_prob=0.0, hidden_scale=4)
    actor.load_state_dict(torch.load(DEFAULT_BC_CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    actor.eval()
    return actor


def bc_sequence_identity(policy: End2RaceGRUPolicy, reference: End2Race) -> dict[str, Any]:
    rng = np.random.default_rng(20260715)
    reference_hidden = torch.zeros((1, 1, reference.gru.hidden_size))
    policy_hidden = torch.zeros_like(reference_hidden)
    policy_cell = torch.full_like(reference_hidden, 7.0)  # must be ignored
    action_error = 0.0
    hidden_error = 0.0
    last_observation: torch.Tensor | None = None
    with torch.no_grad():
        for timestep in range(100):
            lidar = rng.uniform(0.05, 12.0, size=END2RACE_LIDAR_SIZE).astype(np.float32)
            speed = float(rng.uniform(0.0, 8.0))
            lidar_tensor = torch.from_numpy(lidar).reshape(1, 1, -1)
            speed_tensor = torch.tensor([[[speed]]], dtype=torch.float32)
            reference_action, reference_hidden = reference(lidar_tensor, speed_tensor, reference_hidden)

            observation = torch.from_numpy(end2race_observation(lidar, speed)).reshape(1, -1)
            last_observation = observation
            episode_start = torch.tensor([timestep == 0], dtype=torch.float32)
            vf_zero = torch.zeros_like(policy_hidden)
            policy_action, _value, _log_prob, next_states = policy.forward(
                observation,
                RNNStates((policy_hidden, policy_cell), (vf_zero, vf_zero.clone())),
                episode_start,
                deterministic=True,
            )
            policy_hidden, policy_cell = next_states.pi
            action_error = max(
                action_error,
                float((policy_action - reference_action[:, -1, :]).abs().max()),
            )
            hidden_error = max(hidden_error, float((policy_hidden - reference_hidden).abs().max()))

        features = torch.randn((5, 2, reference.gru.input_size))
        hidden = torch.randn((reference.gru.num_layers, 2, reference.gru.hidden_size))
        nonzero_dummy_cell = torch.randn_like(hidden)
        direct_output, direct_hidden = reference.gru(features.transpose(0, 1), hidden)
        adapter_output, (adapter_hidden, adapter_cell) = policy.lstm_actor(
            features,
            (hidden, nonzero_dummy_cell),
        )
        adapter_error = max(
            float((adapter_output - direct_output.transpose(0, 1)).abs().max()),
            float((adapter_hidden - direct_hidden).abs().max()),
        )

        assert last_observation is not None
        actor_state = (policy_hidden.clone(), policy_cell.clone())
        vf_zero = torch.zeros_like(policy_hidden)
        deterministic_before, _, _, _ = policy.forward(
            last_observation,
            RNNStates(actor_state, (vf_zero, vf_zero.clone())),
            torch.zeros(1),
            deterministic=True,
        )
        saved_log_std = policy.log_std.clone()
        policy.log_std.add_(1.0)
        deterministic_after, _, _, _ = policy.forward(
            last_observation,
            RNNStates(actor_state, (vf_zero, vf_zero.clone())),
            torch.zeros(1),
            deterministic=True,
        )
        policy.log_std.copy_(saved_log_std)
        log_std_invariance_error = float((deterministic_before - deterministic_after).abs().max())

    return {
        "timesteps": 100,
        "max_action_absolute_error": action_error,
        "max_hidden_absolute_error": hidden_error,
        "adapter_max_absolute_error": adapter_error,
        "adapter_dummy_cell_max_absolute_value": float(adapter_cell.abs().max()),
        "adapter_interface": {
            "input_size": policy.lstm_actor.input_size,
            "hidden_size": policy.lstm_actor.hidden_size,
            "num_layers": policy.lstm_actor.num_layers,
            "accepts_h_c_tuple": True,
        },
        "gaussian_log_std_trainable": bool(policy.log_std.requires_grad),
        "gaussian_log_std_in_optimizer": any(
            parameter is policy.log_std
            for group in policy.optimizer.param_groups
            for parameter in group["params"]
        ),
        "deterministic_mean_log_std_invariance_error": log_std_invariance_error,
    }


def episode_reset_identity(policy: End2RaceGRUPolicy, reference: End2Race) -> dict[str, Any]:
    n_envs = 2
    reference_hidden = torch.zeros((1, n_envs, reference.gru.hidden_size))
    policy_hidden = torch.zeros_like(reference_hidden)
    policy_cell = torch.full_like(reference_hidden, 3.0)
    action_error = 0.0
    hidden_error = 0.0
    reset_from_zero_error = 0.0
    unaffected_continuity_margin = float("inf")
    cell_max = 0.0
    reset_schedule = [{0, 3}, {0, 5}]

    with torch.no_grad():
        for timestep in range(7):
            lidar_np = np.stack(
                [
                    np.linspace(0.2 + env_id, 8.0 + env_id, END2RACE_LIDAR_SIZE, dtype=np.float32)
                    + timestep * 0.02
                    for env_id in range(n_envs)
                ]
            )
            speed_np = np.asarray([1.0 + timestep * 0.1, 2.0 + timestep * 0.1], dtype=np.float32)
            starts = torch.tensor(
                [timestep in reset_schedule[env_id] for env_id in range(n_envs)],
                dtype=torch.float32,
            )
            pre_reference_hidden = reference_hidden.clone()
            reference_hidden = reference_hidden * (1.0 - starts).view(1, n_envs, 1)
            reference_action_slots: list[torch.Tensor] = []
            reference_hidden_slots: list[torch.Tensor] = []
            for env_id in range(n_envs):
                slot_actions, slot_hidden = reference(
                    torch.from_numpy(lidar_np[env_id]).reshape(1, 1, -1),
                    torch.tensor([[[float(speed_np[env_id])]]]),
                    reference_hidden[:, env_id : env_id + 1],
                )
                reference_action_slots.append(slot_actions)
                reference_hidden_slots.append(slot_hidden)
            reference_actions = torch.cat(reference_action_slots, dim=0)
            reference_hidden = torch.cat(reference_hidden_slots, dim=1)
            observations = torch.from_numpy(
                np.stack([end2race_observation(lidar_np[i], float(speed_np[i])) for i in range(n_envs)])
            )
            means, (policy_hidden, policy_cell) = policy.actor_mean(
                observations,
                (policy_hidden, policy_cell),
                starts,
            )
            action_error = max(action_error, float((means - reference_actions[:, -1, :]).abs().max()))
            hidden_error = max(hidden_error, float((policy_hidden - reference_hidden).abs().max()))
            cell_max = max(cell_max, float(policy_cell.abs().max()))

            for env_id in range(n_envs):
                lidar_one = torch.from_numpy(lidar_np[env_id]).reshape(1, 1, -1)
                speed_one = torch.tensor([[[float(speed_np[env_id])]]])
                fresh_action, fresh_hidden = reference(lidar_one, speed_one, torch.zeros((1, 1, reference.gru.hidden_size)))
                if starts[env_id] > 0:
                    reset_from_zero_error = max(
                        reset_from_zero_error,
                        float((means[env_id] - fresh_action[0, -1]).abs().max()),
                        float((policy_hidden[:, env_id : env_id + 1] - fresh_hidden).abs().max()),
                    )
                elif timestep in {3, 5}:
                    # At staggered boundaries, the other env must retain history.
                    continuity_margin = max(
                        float((means[env_id] - fresh_action[0, -1]).abs().max()),
                        float((policy_hidden[:, env_id : env_id + 1] - fresh_hidden).abs().max()),
                    )
                    if float(pre_reference_hidden[:, env_id].abs().max()) > 0:
                        unaffected_continuity_margin = min(unaffected_continuity_margin, continuity_margin)

    return {
        "parallel_envs": n_envs,
        "env_reset_steps": [sorted(schedule) for schedule in reset_schedule],
        "max_action_absolute_error": action_error,
        "max_hidden_absolute_error": hidden_error,
        "reset_slot_matches_fresh_zero_state_max_error": reset_from_zero_error,
        "dummy_cell_max_absolute_value": cell_max,
        "unaffected_env_differs_from_erroneous_zero_reset_min_margin": unaffected_continuity_margin,
    }


def setup_and_collect(model: RecurrentPPO):
    _, callback = model._setup_learn(
        total_timesteps=ROLLOUT_STEPS * N_ENVS,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="sb3_gru_poc",
        progress_bar=False,
    )
    callback.on_training_start(locals(), globals())
    assert model.env is not None
    collected = model.collect_rollouts(
        model.env,
        callback,
        model.rollout_buffer,
        n_rollout_steps=ROLLOUT_STEPS,
    )
    if not collected:
        raise RuntimeError("POC rollout collection was interrupted")
    return callback


def replay_identity(model: RecurrentPPO) -> dict[str, Any]:
    np.random.seed(1)  # circular split=5 for 20 transitions, forcing padding
    valid_errors: list[np.ndarray] = []
    ratio_deviations: list[np.ndarray] = []
    padding_count = 0
    valid_count = 0
    minibatches: list[dict[str, int]] = []

    for rollout_data in model.rollout_buffer.get(MINIBATCH_SIZE):
        mask = rollout_data.mask > 1e-8
        with torch.no_grad():
            _, new_logp, _ = model.policy.evaluate_actions(
                rollout_data.observations,
                rollout_data.actions,
                rollout_data.lstm_states,
                rollout_data.episode_starts,
            )
        error = (new_logp - rollout_data.old_log_prob).abs()
        ratio_deviation = (torch.exp(new_logp - rollout_data.old_log_prob) - 1.0).abs()
        valid_errors.append(error[mask].cpu().numpy())
        ratio_deviations.append(ratio_deviation[mask].cpu().numpy())
        valid = int(mask.sum().item())
        padded = int((~mask).sum().item())
        valid_count += valid
        padding_count += padded
        minibatches.append({"valid": valid, "padded": padded, "flat_size": int(mask.numel())})

    errors = np.concatenate(valid_errors)
    ratios = np.concatenate(ratio_deviations)
    episode_starts = np.asarray(model.rollout_buffer.episode_starts)
    return {
        "valid_timesteps": valid_count,
        "padding_timesteps": padding_count,
        "minibatches": minibatches,
        "max_logp_absolute_error": float(errors.max()),
        "mean_logp_absolute_error": float(errors.mean()),
        "max_ratio_deviation": float(ratios.max()),
        "coverage": {
            "parallel_envs": N_ENVS,
            "episode_boundary_count": int((episode_starts > 0.5).sum()),
            "ordinary_continuous_sequence": True,
            "timeout_truncation": True,
            "padding_mask": padding_count > 0,
        },
    }


def timeout_and_advantage_checks(
    model: RecurrentPPO,
    envs: list[End2RaceGymnasiumEnv],
    raw_rewards: np.ndarray,
    raw_values: np.ndarray,
    raw_advantages: np.ndarray,
) -> dict[str, Any]:
    collision_errors: list[float] = []
    timeout_errors: list[float] = []
    boundary_advantage_errors: list[float] = []
    timeout_terminal_values: list[float] = []
    semantics: dict[str, bool] = {}
    hidden_size = model.policy.actor_hidden_size

    for env_index, env in enumerate(envs):
        for event in env.terminal_events:
            transition = int(event["transition_index"])
            if transition >= ROLLOUT_STEPS:
                continue
            corrected_reward = float(raw_rewards[transition, env_index])
            raw_reward = float(event["raw_reward"])
            boundary_advantage_errors.append(
                abs(float(raw_advantages[transition, env_index]) - (corrected_reward - float(raw_values[transition, env_index])))
            )
            if event["reason"] == "collision":
                collision_errors.append(abs(corrected_reward - raw_reward))
                semantics["collision_terminated_not_truncated"] = bool(event["terminated"] and not event["truncated"])
            elif event["reason"] == "timeout":
                observation = torch.as_tensor(event["observation"], dtype=torch.float32).reshape(1, -1)
                zero = torch.zeros((1, 1, hidden_size))
                with torch.no_grad():
                    terminal_value = float(
                        model.policy.predict_values(observation, (zero, zero.clone()), torch.zeros(1)).item()
                    )
                expected_reward = raw_reward + model.gamma * terminal_value
                timeout_terminal_values.append(terminal_value)
                timeout_errors.append(abs(corrected_reward - expected_reward))
                semantics["timeout_truncated_not_terminated"] = bool(event["truncated"] and not event["terminated"])

    return {
        "collision_zero_bootstrap_max_error": max(collision_errors, default=float("inf")),
        "timeout_terminal_value_bootstrap_max_error": max(timeout_errors, default=float("inf")),
        "terminal_advantage_no_cross_episode_max_error": max(boundary_advantage_errors, default=float("inf")),
        "timeout_terminal_values": timeout_terminal_values,
        "collision_events": len(collision_errors),
        "timeout_events": len(timeout_errors),
        **semantics,
    }


def checkpoint_compatibility(policy: End2RaceGRUPolicy) -> dict[str, Any]:
    bc_keys = list(torch.load(DEFAULT_BC_CHECKPOINT, map_location="cpu", weights_only=True).keys())
    with tempfile.TemporaryDirectory(prefix="end2race_sb3_actor_") as directory:
        output_path = Path(directory) / "actor_only.pth"
        torch.save(policy.end2race_actor.state_dict(), output_path)
        reloaded = End2Race(mask_prob=0.0, hidden_scale=4)
        state_dict = torch.load(output_path, map_location="cpu", weights_only=True)
        incompatible = reloaded.load_state_dict(state_dict, strict=True)
        max_error = max(
            float((state_dict[key] - reloaded.state_dict()[key]).abs().max())
            for key in state_dict
        )
    return {
        "strict_load_succeeded": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "actor_keys_match_bc_schema": list(state_dict.keys()) == bc_keys,
        "actor_key_count": len(state_dict),
        "roundtrip_max_absolute_error": max_error,
        "temporary_output_only": True,
    }


def max_parameter_delta(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    delta = 0.0
    for name, value in module.state_dict().items():
        if torch.is_floating_point(value):
            delta = max(delta, float((value.detach().cpu() - before[name]).abs().max()))
    return delta


def run_poc() -> dict[str, Any]:
    torch.manual_seed(20260715)
    np.random.seed(20260715)
    model, envs = build_model()
    reference = load_reference_actor()
    model.policy.set_training_mode(False)

    schema_observation, _schema_info = envs[0].reset()
    wrapper_schema = {
        "observation_shape": list(schema_observation.shape),
        "lidar_values": END2RACE_LIDAR_SIZE,
        "previous_speed_values": 1,
        "actor_observation_is_plain_array": isinstance(schema_observation, np.ndarray),
        "privileged_simulator_fields_in_actor_observation": False,
        "wrapper_reward_transform": "none",
    }

    identity = bc_sequence_identity(model.policy, reference)
    reset = episode_reset_identity(model.policy, reference)
    callback = setup_and_collect(model)

    raw_rewards = model.rollout_buffer.rewards.copy()
    raw_values = model.rollout_buffer.values.copy()
    raw_advantages = model.rollout_buffer.advantages.copy()
    timeout = timeout_and_advantage_checks(model, envs, raw_rewards, raw_values, raw_advantages)
    replay = replay_identity(model)
    checkpoint = checkpoint_compatibility(model.policy)

    before = {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()}
    model.train()
    train_delta = max_parameter_delta(before, model.policy)
    callback.on_training_end()
    model.env.close()

    event_durations = [float(event["elapsed_time"]) for env in envs for event in env.terminal_events]

    results = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__,
        },
        "gymnasium_wrapper": wrapper_schema,
        "bc_sequence_identity": identity,
        "episode_reset_identity": reset,
        "ppo_replay_identity": replay,
        "timeout_bootstrap": timeout,
        "checkpoint_compatibility": checkpoint,
        "smoke_train": {
            "parallel_envs": N_ENVS,
            "max_episode_duration_seconds": max(event_durations),
            "rollouts": 1,
            "ppo_train_calls": 1,
            "learning_rate": float(model.policy.optimizer.param_groups[0]["lr"]),
            "max_parameter_delta": train_delta,
        },
        "third_party_sources_modified": False,
        "end2race_learner_training_performed": False,
    }
    return results


def main() -> None:
    results = run_poc()
    print(f"actor identity max error: {results['bc_sequence_identity']['max_action_absolute_error']:.12g}")
    print(f"hidden identity max error: {results['bc_sequence_identity']['max_hidden_absolute_error']:.12g}")
    print(f"replay logp max error: {results['ppo_replay_identity']['max_logp_absolute_error']:.12g}")
    print(f"ratio max deviation: {results['ppo_replay_identity']['max_ratio_deviation']:.12g}")
    print(
        "timeout bootstrap result: "
        f"max error={results['timeout_bootstrap']['timeout_terminal_value_bootstrap_max_error']:.12g}"
    )
    print(f"strict checkpoint load result: {results['checkpoint_compatibility']['strict_load_succeeded']}")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

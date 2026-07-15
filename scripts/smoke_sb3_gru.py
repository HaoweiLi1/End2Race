#!/usr/bin/env python3
"""Strict, removable SB3-Contrib GRU integration contract smoke.

Only synthetic contract tests and one optional real-F110 reset/step smoke are
performed.  The sole PPO train call uses learning_rate=0.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import inspect
import json
import platform
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium
import numpy as np
import sb3_contrib
import stable_baselines3
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from model import End2Race
from rl.end2race_gymnasium_env import (
    End2RaceGymnasiumEnv,
    EpisodeResetSpec,
    LatticePlannerOpponentController,
)
from rl.sb3_end2race_policy import (
    DEFAULT_BC_CHECKPOINT,
    END2RACE_LIDAR_SIZE,
    EVALUATOR_STEER_BOUND,
    EvaluatorCompatibleJointDistribution,
    End2RaceGRUPolicy,
    end2race_observation,
)
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates


ROLLOUT_STEPS = 10
MINIBATCH_SIZE = 6
N_ENVS = 2


class SyntheticResetProvider:
    """Stateful provider whose scenario values are driven only by its RNG."""

    def __init__(self, env_id: int):
        self.env_id = env_id
        self.call_count = 0
        self.samples: list[EpisodeResetSpec] = []

    def __call__(self, rng: np.random.Generator) -> EpisodeResetSpec:
        self.call_count += 1
        offset = float(rng.uniform(0.0, 0.25))
        poses = np.asarray(
            [
                [10.0 * self.env_id + offset, 0.0, 0.0],
                [10.0 * self.env_id + 2.0 + offset, 0.5, 0.1],
            ],
            dtype=np.float64,
        )
        spec = EpisodeResetSpec(
            poses=poses,
            initial_speed_feature=1.1 + 0.1 * self.env_id + offset,
            scenario={
                "map_name": "Synthetic",
                "opp_raceline": f"lane{self.env_id}",
                "opp_speedscale": 0.5 + 0.1 * self.env_id,
                "provider_env_id": self.env_id,
            },
        )
        self.samples.append(deepcopy(spec))
        return spec


class FakeTracker:
    def __init__(self):
        self.prev_error = 0.0
        self.calls = 0

    def plan(self, pose_x: float, pose_y: float, pose_theta: float, speed: float, trajectory: np.ndarray):
        del pose_y, pose_theta
        self.calls += 1
        steering = 0.02 * pose_x + 0.01 * float(trajectory[0, 0])
        desired_speed = 2.0 + 0.1 * speed
        self.prev_error = steering
        return steering, desired_speed


class FakeLatticePlanner:
    """Small deterministic stand-in with the same state fields as LatticePlanner."""

    def __init__(self, factory_id: int, planner_serial: int):
        self.factory_id = factory_id
        self.planner_serial = planner_serial
        self.conf = SimpleNamespace(tracker_steps=2)
        self.tracker = FakeTracker()
        self.step = 0
        self.prev_traj_local = np.zeros((3, 2), dtype=np.float64)
        self.prev_opp_pose = np.zeros((1, 2), dtype=np.float64)
        self.plan_calls = 0

    def plan(self, pose_x: float, pose_y: float, pose_theta: float, opponent_poses: np.ndarray, speed: float):
        del pose_theta, speed
        self.step += 1
        self.plan_calls += 1
        trajectory = np.asarray([[pose_x + 1.0, pose_y, 2.5], [pose_x + 2.0, pose_y, 2.5]], dtype=np.float64)
        self.prev_traj_local = trajectory[:, :2].copy()
        self.prev_opp_pose = np.asarray(opponent_poses[:, :2], dtype=np.float64).copy()
        return trajectory


class FakePlannerFactory:
    def __init__(self, factory_id: int):
        self.factory_id = factory_id
        self.created: list[FakeLatticePlanner] = []

    def __call__(self, map_name: str, raceline: str) -> FakeLatticePlanner:
        del map_name, raceline
        planner = FakeLatticePlanner(self.factory_id, len(self.created))
        self.created.append(planner)
        return planner


class NoOpOpponentController:
    def __init__(self):
        self.num_agents = 0
        self.ego_index = 0
        self.reset_count = 0

    def reset(self, spec: EpisodeResetSpec, num_agents: int, ego_index: int) -> None:
        del spec
        self.num_agents = num_agents
        self.ego_index = ego_index
        self.reset_count += 1

    def actions(self, raw_observation: dict[str, Any]) -> np.ndarray:
        del raw_observation
        return np.zeros((self.num_agents, 2), dtype=np.float32)

    def state_snapshot(self) -> dict[str, Any]:
        return {"reset_count": self.reset_count, "planners": {}}


class SyntheticLegacyF110Env:
    """Strict legacy API core with action-sensitive transitions and rewards."""

    num_agents = 2

    def __init__(self, env_id: int, terminal_kind: str, horizon: int, timestep: float = 0.1):
        if terminal_kind not in {"ego_collision", "timeout"}:
            raise ValueError(terminal_kind)
        self.env_id = env_id
        self.terminal_kind = terminal_kind
        self.horizon = horizon
        self.timestep = timestep
        self.episode_step = 0
        self.lifetime_step = 0
        self.reset_poses: list[np.ndarray] = []
        self.received_joint_actions: list[np.ndarray] = []
        self.returned_rewards: list[float] = []
        self.observation_history: list[dict[str, np.ndarray]] = []
        self.step_observations: list[dict[str, np.ndarray]] = []
        self._last_ego_action = np.zeros(2, dtype=np.float32)

    @property
    def unwrapped(self) -> "SyntheticLegacyF110Env":
        return self

    def _observation(self, collisions: tuple[bool, bool] = (False, False)) -> dict[str, np.ndarray]:
        marker = float(self.env_id * 0.1 + self.lifetime_step * 0.001)
        action_effect = float(self._last_ego_action[0] * 0.001 + self._last_ego_action[1] * 0.0001)
        ego_scan = np.linspace(0.05, 1.0, 720, dtype=np.float32) + marker + action_effect
        opponent_scan = np.linspace(0.1, 1.1, 720, dtype=np.float32) + marker
        measured_ego_speed = 2.1 + 0.1 * self.env_id + 0.05 * self.lifetime_step + 0.01 * float(self._last_ego_action[1])
        obs = {
            "scans": np.stack((ego_scan, opponent_scan)),
            "linear_vels_x": np.asarray([measured_ego_speed, 3.0 + 0.1 * self.episode_step], dtype=np.float32),
            "collisions": np.asarray(collisions, dtype=np.float32),
            "poses_x": np.asarray([self.lifetime_step * 0.001, 2.0 + self.lifetime_step * 0.001], dtype=np.float32),
            "poses_y": np.asarray([0.0, 0.5], dtype=np.float32),
            "poses_theta": np.asarray([0.0, 0.1], dtype=np.float32),
            "opponent_pose": np.asarray([[999.0, 999.0, 0.0]], dtype=np.float32),
            "reference_geometry": np.asarray([1234.0], dtype=np.float32),
        }
        self.observation_history.append(deepcopy(obs))
        return obs

    def reset(self, *, poses: np.ndarray):
        poses = np.asarray(poses)
        if poses.shape != (self.num_agents, 3):
            raise ValueError(f"strict reset expected {(self.num_agents, 3)}, got {poses.shape}")
        self.reset_poses.append(poses.copy())
        self.episode_step = 0
        self._last_ego_action = np.zeros(2, dtype=np.float32)
        return self._observation(), 0.0, False, {"timestep": self.timestep}

    def step(self, action: np.ndarray):
        joint_action = np.asarray(action, dtype=np.float32)
        if joint_action.shape != (self.num_agents, 2):
            raise ValueError(joint_action.shape)
        self.received_joint_actions.append(joint_action.copy())
        self._last_ego_action = joint_action[0].copy()
        self.episode_step += 1
        self.lifetime_step += 1
        opponent_collision = self.episode_step == 2
        ego_collision = self.terminal_kind == "ego_collision" and self.episode_step >= self.horizon
        reward = 1.0 + 0.2 * float(joint_action[0, 0]) + 0.03 * float(joint_action[0, 1])
        self.returned_rewards.append(reward)
        done = ego_collision
        observation = self._observation((ego_collision, opponent_collision))
        self.step_observations.append(deepcopy(observation))
        return observation, reward, done, {"timestep": self.timestep}

    def render(self):
        return None

    def close(self):
        return None


class ActionTraceCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.raw_actions: list[np.ndarray] = []
        self.clipped_actions: list[np.ndarray] = []
        self.raw_actor_means: list[np.ndarray] = []

    def _on_step(self) -> bool:
        self.raw_actions.append(np.asarray(self.locals["actions"]).copy())
        self.clipped_actions.append(np.asarray(self.locals["clipped_actions"]).copy())
        raw_mean = self.model.policy.last_raw_actor_mean
        if raw_mean is None:
            raise RuntimeError("Policy did not expose its diagnostic raw actor mean")
        self.raw_actor_means.append(raw_mean.detach().cpu().numpy().copy())
        return True


def make_poc_env(env_id: int, terminal_kind: str, horizon: int):
    core = SyntheticLegacyF110Env(env_id=env_id, terminal_kind=terminal_kind, horizon=horizon)
    provider = SyntheticResetProvider(env_id)
    planner_factory = FakePlannerFactory(env_id)
    opponent_controller = LatticePlannerOpponentController(planner_factory=planner_factory)
    sim_duration = 10.0 if terminal_kind == "ego_collision" else horizon * core.timestep
    env = End2RaceGymnasiumEnv(
        core,
        sim_duration=sim_duration,
        reset_provider=provider,
        ego_index=0,
        opponent_controller=opponent_controller,
    )
    return env, core, provider, opponent_controller, planner_factory


def build_model():
    bundles = [
        make_poc_env(0, "ego_collision", 4),
        make_poc_env(1, "timeout", 7),
    ]
    envs = [bundle[0] for bundle in bundles]
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
            "steer_log_std_init": -2.0,
            "speed_log_std_init": 0.0,
        },
        verbose=0,
    )
    return model, bundles


def load_reference_actor() -> End2Race:
    actor = End2Race(mask_prob=0.0, hidden_scale=4)
    actor.load_state_dict(torch.load(DEFAULT_BC_CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    actor.eval()
    return actor


def bc_sequence_identity(policy: End2RaceGRUPolicy, reference: End2Race) -> dict[str, Any]:
    rng = np.random.default_rng(20260715)
    reference_hidden = torch.zeros((1, 1, reference.gru.hidden_size))
    policy_hidden = torch.zeros_like(reference_hidden)
    policy_cell = torch.full_like(reference_hidden, 7.0)
    action_error = 0.0
    hidden_error = 0.0
    last_observation: torch.Tensor | None = None
    with torch.no_grad():
        for timestep in range(100):
            lidar = rng.uniform(0.05, 12.0, size=END2RACE_LIDAR_SIZE).astype(np.float32)
            speed = float(rng.uniform(0.0, 8.0))
            lidar_tensor = torch.from_numpy(lidar).reshape(1, 1, -1)
            speed_tensor = torch.tensor([[[speed]]], dtype=torch.float32)
            raw_reference_action, reference_hidden = reference(lidar_tensor, speed_tensor, reference_hidden)
            expected_action = raw_reference_action[:, -1, :].clone()
            expected_action[:, 0].clamp_(-EVALUATOR_STEER_BOUND, EVALUATOR_STEER_BOUND)

            observation = torch.from_numpy(end2race_observation(lidar, speed)).reshape(1, -1)
            last_observation = observation
            vf_zero = torch.zeros_like(policy_hidden)
            policy_action, _value, _log_prob, next_states = policy.forward(
                observation,
                RNNStates((policy_hidden, policy_cell), (vf_zero, vf_zero.clone())),
                torch.tensor([timestep == 0], dtype=torch.float32),
                deterministic=True,
            )
            policy_hidden, policy_cell = next_states.pi
            action_error = max(action_error, float((policy_action - expected_action).abs().max()))
            hidden_error = max(hidden_error, float((policy_hidden - reference_hidden).abs().max()))

        features = torch.randn((5, 2, reference.gru.input_size))
        hidden = torch.randn((reference.gru.num_layers, 2, reference.gru.hidden_size))
        nonzero_dummy_cell = torch.randn_like(hidden)
        direct_output, direct_hidden = reference.gru(features.transpose(0, 1), hidden)
        adapter_output, (adapter_hidden, adapter_cell) = policy.lstm_actor(features, (hidden, nonzero_dummy_cell))
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

    optimizer_ids = {id(parameter) for group in policy.optimizer.param_groups for parameter in group["params"]}
    # Independent scalar oracle for the transformed steering density, including
    # inverse tanh and both the tanh and 0.52 scale Jacobians.
    oracle_raw_mean = torch.tensor([[0.13, 2.4]], dtype=torch.float64)
    oracle_log_std = torch.tensor([-1.2, -0.3], dtype=torch.float64)
    oracle_action = torch.tensor([[0.21, 2.8]], dtype=torch.float64)
    oracle_distribution = EvaluatorCompatibleJointDistribution().proba_distribution(
        oracle_raw_mean, oracle_log_std
    )
    implementation_logp = float(oracle_distribution.log_prob(oracle_action))
    normalized_action = float(oracle_action[0, 0] / EVALUATOR_STEER_BOUND)
    latent_action = np.arctanh(normalized_action)
    normalized_mode = float(oracle_raw_mean[0, 0] / EVALUATOR_STEER_BOUND)
    latent_mode = np.arctanh(normalized_mode)
    steer_std = float(np.exp(oracle_log_std[0]))
    speed_std = float(np.exp(oracle_log_std[1]))
    steer_base_logp = -0.5 * ((latent_action - latent_mode) / steer_std) ** 2 - np.log(
        steer_std * np.sqrt(2.0 * np.pi)
    )
    steer_log_jacobian = np.log(EVALUATOR_STEER_BOUND) + np.log(1.0 - normalized_action**2)
    speed_base_logp = -0.5 * ((float(oracle_action[0, 1] - oracle_raw_mean[0, 1])) / speed_std) ** 2 - np.log(
        speed_std * np.sqrt(2.0 * np.pi)
    )
    oracle_logp = steer_base_logp - steer_log_jacobian + speed_base_logp
    return {
        "timesteps": 100,
        "max_deterministic_action_error": action_error,
        "max_hidden_absolute_error": hidden_error,
        "adapter_max_absolute_error": adapter_error,
        "adapter_dummy_cell_max_absolute_value": float(adapter_cell.abs().max()),
        "deterministic_action_log_std_invariance_error": float((deterministic_before - deterministic_after).abs().max()),
        "distribution_log_std_trainable": bool(policy.log_std.requires_grad),
        "distribution_log_std_in_optimizer": id(policy.log_std) in optimizer_ids,
        "joint_log_prob_independent_oracle_error": abs(implementation_logp - float(oracle_logp)),
    }


def episode_reset_identity(policy: End2RaceGRUPolicy, reference: End2Race) -> dict[str, Any]:
    n_envs = 2
    reference_hidden = torch.zeros((1, n_envs, reference.gru.hidden_size))
    policy_hidden = torch.zeros_like(reference_hidden)
    policy_cell = torch.full_like(reference_hidden, 3.0)
    action_error = hidden_error = reset_from_zero_error = cell_max = 0.0
    unaffected_continuity_margin = float("inf")
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
            starts = torch.tensor([timestep in reset_schedule[i] for i in range(n_envs)], dtype=torch.float32)
            pre_reference_hidden = reference_hidden.clone()
            reference_hidden = reference_hidden * (1.0 - starts).view(1, n_envs, 1)
            raw_actions: list[torch.Tensor] = []
            next_h: list[torch.Tensor] = []
            for env_id in range(n_envs):
                action, hidden = reference(
                    torch.from_numpy(lidar_np[env_id]).reshape(1, 1, -1),
                    torch.tensor([[[float(speed_np[env_id])]]]),
                    reference_hidden[:, env_id : env_id + 1],
                )
                raw_actions.append(action)
                next_h.append(hidden)
            reference_raw = torch.cat(raw_actions, dim=0)[:, -1, :]
            reference_hidden = torch.cat(next_h, dim=1)
            observations = torch.from_numpy(
                np.stack([end2race_observation(lidar_np[i], float(speed_np[i])) for i in range(n_envs)])
            )
            raw_means, (policy_hidden, policy_cell) = policy.actor_mean(
                observations, (policy_hidden, policy_cell), starts
            )
            action_error = max(action_error, float((raw_means - reference_raw).abs().max()))
            hidden_error = max(hidden_error, float((policy_hidden - reference_hidden).abs().max()))
            cell_max = max(cell_max, float(policy_cell.abs().max()))
            for env_id in range(n_envs):
                fresh_action, fresh_hidden = reference(
                    torch.from_numpy(lidar_np[env_id]).reshape(1, 1, -1),
                    torch.tensor([[[float(speed_np[env_id])]]]),
                    torch.zeros((1, 1, reference.gru.hidden_size)),
                )
                if starts[env_id] > 0:
                    reset_from_zero_error = max(
                        reset_from_zero_error,
                        float((raw_means[env_id] - fresh_action[0, -1]).abs().max()),
                        float((policy_hidden[:, env_id : env_id + 1] - fresh_hidden).abs().max()),
                    )
                elif timestep in {3, 5} and float(pre_reference_hidden[:, env_id].abs().max()) > 0:
                    unaffected_continuity_margin = min(
                        unaffected_continuity_margin,
                        max(
                            float((raw_means[env_id] - fresh_action[0, -1]).abs().max()),
                            float((policy_hidden[:, env_id : env_id + 1] - fresh_hidden).abs().max()),
                        ),
                    )
    return {
        "parallel_envs": n_envs,
        "env_reset_steps": [sorted(schedule) for schedule in reset_schedule],
        "max_raw_action_error": action_error,
        "max_hidden_absolute_error": hidden_error,
        "reset_slot_matches_fresh_zero_state_max_error": reset_from_zero_error,
        "dummy_cell_max_absolute_value": cell_max,
        "unaffected_env_differs_from_erroneous_zero_reset_min_margin": unaffected_continuity_margin,
    }


def setup_and_collect(model: RecurrentPPO):
    trace_callback = ActionTraceCallback()
    _, callback = model._setup_learn(
        total_timesteps=ROLLOUT_STEPS * N_ENVS,
        callback=trace_callback,
        reset_num_timesteps=True,
        tb_log_name="sb3_gru_repair",
        progress_bar=False,
    )
    callback.on_training_start(locals(), globals())
    assert model.env is not None
    if not model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=ROLLOUT_STEPS):
        raise RuntimeError("POC rollout collection was interrupted")
    return callback, trace_callback


def action_contract(model: RecurrentPPO, bundles: list[tuple[Any, ...]], trace: ActionTraceCallback) -> dict[str, Any]:
    buffer_actions = np.asarray(model.rollout_buffer.actions)
    sb3_actions = np.stack(trace.raw_actions)
    clipped_actions = np.stack(trace.clipped_actions)
    raw_means = np.stack(trace.raw_actor_means)
    wrapper_actions = np.stack(
        [[bundles[env][0].action_trace[step]["ego_action"] for env in range(N_ENVS)] for step in range(ROLLOUT_STEPS)]
    )
    core_actions = np.stack(
        [[bundles[env][1].received_joint_actions[step][0] for env in range(N_ENVS)] for step in range(ROLLOUT_STEPS)]
    )
    opponent_actions = np.stack(
        [[bundles[env][1].received_joint_actions[step][1] for env in range(N_ENVS)] for step in range(ROLLOUT_STEPS)]
    )
    clip_delta = np.abs(sb3_actions - clipped_actions)
    buffer_core_error = np.abs(buffer_actions - core_actions)
    sb3_core_error = np.abs(sb3_actions - core_actions)
    rewards_depend_on_action = all(
        abs(
            bundles[env][1].returned_rewards[step]
            - (
                1.0
                + 0.2 * float(core_actions[step, env, 0])
                + 0.03 * float(core_actions[step, env, 1])
            )
        )
        <= 1e-7
        for env in range(N_ENVS)
        for step in range(ROLLOUT_STEPS)
    )
    observation_action_errors = [
        abs(
            float(bundles[env][1].step_observations[step]["scans"][0, 0])
            - (
                0.05
                + env * 0.1
                + (step + 1) * 0.001
                + float(core_actions[step, env, 0]) * 0.001
                + float(core_actions[step, env, 1]) * 0.0001
            )
        )
        for env in range(N_ENVS)
        for step in range(ROLLOUT_STEPS)
    ]
    return {
        "max_buffer_ego_action_vs_core_error": float(buffer_core_error.max()),
        "max_sb3_ego_action_vs_core_error": float(sb3_core_error.max()),
        "max_wrapper_ego_action_vs_core_error": float(np.abs(wrapper_actions - core_actions).max()),
        "sb3_pre_env_action_clipping_count": int(np.count_nonzero(clip_delta > 0)),
        "sb3_pre_env_action_max_clip_delta": float(clip_delta.max()),
        "steering_out_of_bound_count": int(np.count_nonzero(np.abs(buffer_actions[:, :, 0]) > EVALUATOR_STEER_BOUND)),
        "action_sensitive_reward_verified": rewards_depend_on_action,
        "action_sensitive_observation_max_error": max(observation_action_errors),
        "action_sensitive_observation_verified": max(observation_action_errors) <= 1e-7,
        "ppo_buffer_action_width": int(buffer_actions.shape[-1]),
        "opponent_action_present_in_ppo_buffer": bool(buffer_actions.shape[-1] != 2),
        "fully_traced_ego_transition_count": int(buffer_actions.shape[0] * buffer_actions.shape[1]),
        "trace_example": {
            "raw_end2race_mean": raw_means[0, 0].tolist(),
            "distribution_physical_sample": sb3_actions[0, 0].tolist(),
            "buffer_action": buffer_actions[0, 0].tolist(),
            "sb3_action_passed_to_wrapper": clipped_actions[0, 0].tolist(),
            "wrapper_ego_action": wrapper_actions[0, 0].tolist(),
            "core_received_ego_action": core_actions[0, 0].tolist(),
            "opponent_lattice_action": opponent_actions[0, 0].tolist(),
            "old_logp": float(model.rollout_buffer.log_probs[0, 0]),
        },
    }


def _oracle_hidden_at_transition(
    reference: End2Race,
    observations: np.ndarray,
    episode_starts: np.ndarray,
    env_index: int,
    transition: int,
) -> torch.Tensor:
    episode_start = transition
    while episode_start > 0 and episode_starts[episode_start, env_index] < 0.5:
        episode_start -= 1
    hidden = torch.zeros((1, 1, reference.gru.hidden_size))
    with torch.no_grad():
        for step in range(episode_start, transition):
            obs = observations[step, env_index]
            lidar = torch.from_numpy(obs[:END2RACE_LIDAR_SIZE]).reshape(1, 1, -1)
            speed = torch.tensor([[[float(obs[END2RACE_LIDAR_SIZE])]]])
            _action, hidden = reference(lidar, speed, hidden)
    return hidden


def replay_identity(
    model: RecurrentPPO,
    bundles: list[tuple[Any, ...]],
    reference: End2Race,
) -> dict[str, Any]:
    buffer = model.rollout_buffer
    raw_observations = np.asarray(buffer.observations).copy()
    raw_episode_starts = np.asarray(buffer.episode_starts).copy()
    raw_hidden = np.asarray(buffer.hidden_states_pi).copy()
    expected_ids = [float(value) for value in raw_observations[:, :, 0].reshape(-1)]

    np.random.seed(1)
    valid_errors: list[np.ndarray] = []
    ratio_deviations: list[np.ndarray] = []
    sampled_ids: list[float] = []
    padding_count = valid_count = continuation_sequences = nonzero_continuation_sequences = 0
    hidden_oracle_errors: list[float] = []
    minibatches: list[dict[str, int]] = []
    for rollout_data in buffer.get(MINIBATCH_SIZE):
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
        sampled_ids.extend(float(value) for value in rollout_data.observations[mask, 0].cpu().numpy())
        valid = int(mask.sum().item())
        padded = int((~mask).sum().item())
        valid_count += valid
        padding_count += padded
        n_seq = rollout_data.lstm_states.pi[0].shape[1]
        max_length = int(mask.numel() // n_seq)
        sequence_mask = mask.reshape(n_seq, max_length)
        sequence_starts = rollout_data.episode_starts.reshape(n_seq, max_length)
        sequence_observations = rollout_data.observations.reshape(n_seq, max_length, -1)
        for sequence_index in range(n_seq):
            if not bool(sequence_mask[sequence_index, 0]):
                continue
            initial_h = rollout_data.lstm_states.pi[0][:, sequence_index : sequence_index + 1]
            if float(sequence_starts[sequence_index, 0]) < 0.5:
                continuation_sequences += 1
                if float(initial_h.abs().max()) > 0:
                    nonzero_continuation_sequences += 1
                    observation_id = float(sequence_observations[sequence_index, 0, 0])
                    matches = np.argwhere(raw_observations[:, :, 0] == observation_id)
                    if matches.shape[0] != 1:
                        raise AssertionError(f"Expected one raw transition for observation id {observation_id}")
                    transition, env_index = (int(value) for value in matches[0])
                    oracle_h = _oracle_hidden_at_transition(
                        reference, raw_observations, raw_episode_starts, env_index, transition
                    )
                    hidden_oracle_errors.append(float((initial_h.cpu() - oracle_h).abs().max()))
        minibatches.append({"valid": valid, "padded": padded, "sequences": n_seq})

    errors = np.concatenate(valid_errors)
    ratios = np.concatenate(ratio_deviations)
    timeout_events = sum(event["reason"] == "timeout" for bundle in bundles for event in bundle[0].terminal_events)
    ego_terminal_events = sum(event["ego_collision"] for bundle in bundles for event in bundle[0].terminal_events)
    opponent_continuations = 0
    for bundle in bundles:
        events = bundle[0].step_events
        opponent_continuations += sum(
            event["opponent_collision"]
            and not event["terminated"]
            and not event["truncated"]
            and next_event["episode_index"] == event["episode_index"]
            for event, next_event in zip(events, events[1:])
        )
    return {
        "valid_timesteps": valid_count,
        "expected_valid_timesteps": ROLLOUT_STEPS * N_ENVS,
        "every_transition_counted_once": Counter(sampled_ids) == Counter(expected_ids),
        "padding_timesteps": padding_count,
        "minibatches": minibatches,
        "max_logp_absolute_error": float(errors.max()),
        "mean_logp_absolute_error": float(errors.mean()),
        "max_ratio_deviation": float(ratios.max()),
        "continuation_sequence_count": continuation_sequences,
        "nonzero_pre_action_hidden_continuation_count": nonzero_continuation_sequences,
        "continuation_hidden_oracle_max_error": max(hidden_oracle_errors, default=float("inf")),
        "coverage": {
            "parallel_envs": N_ENVS,
            "episode_boundary_count": int((raw_episode_starts > 0.5).sum()),
            "ordinary_continuous_sequence": continuation_sequences > 0,
            "timeout_event_count": timeout_events,
            "ego_true_terminal_event_count": ego_terminal_events,
            "opponent_only_collision_continuation_count": opponent_continuations,
            "padding_mask": padding_count > 0,
        },
        "raw_nonzero_pre_action_state_count": int(
            ((np.max(np.abs(raw_hidden), axis=(1, 3)) > 0) & (raw_episode_starts < 0.5)).sum()
        ),
    }


def timeout_and_advantage_checks(
    model: RecurrentPPO,
    bundles: list[tuple[Any, ...]],
    rewards: np.ndarray,
    values: np.ndarray,
    advantages: np.ndarray,
) -> dict[str, Any]:
    collision_errors: list[float] = []
    timeout_errors: list[float] = []
    boundary_advantage_errors: list[float] = []
    hidden_size = model.policy.actor_hidden_size
    for env_index, bundle in enumerate(bundles):
        for event in bundle[0].terminal_events:
            transition = int(event["transition_index"])
            if transition >= ROLLOUT_STEPS:
                continue
            corrected_reward = float(rewards[transition, env_index])
            raw_reward = float(event["raw_reward"])
            boundary_advantage_errors.append(
                abs(float(advantages[transition, env_index]) - (corrected_reward - float(values[transition, env_index])))
            )
            if event["ego_collision"]:
                collision_errors.append(abs(corrected_reward - raw_reward))
            elif event["reason"] == "timeout":
                observation = torch.as_tensor(event["observation"], dtype=torch.float32).reshape(1, -1)
                zero = torch.zeros((1, 1, hidden_size))
                with torch.no_grad():
                    terminal_value = float(
                        model.policy.predict_values(observation, (zero, zero.clone()), torch.zeros(1)).item()
                    )
                timeout_errors.append(abs(corrected_reward - (raw_reward + model.gamma * terminal_value)))
    return {
        "ego_collision_zero_bootstrap_max_error": max(collision_errors, default=float("inf")),
        "timeout_terminal_value_bootstrap_max_error": max(timeout_errors, default=float("inf")),
        "terminal_advantage_no_cross_episode_max_error": max(boundary_advantage_errors, default=float("inf")),
        "ego_collision_events": len(collision_errors),
        "timeout_events": len(timeout_errors),
    }


def reset_and_opponent_evidence(bundles: list[tuple[Any, ...]]) -> dict[str, Any]:
    envs = [bundle[0] for bundle in bundles]
    cores = [bundle[1] for bundle in bundles]
    providers = [bundle[2] for bundle in bundles]
    controllers = [bundle[3] for bundle in bundles]
    planner_factories = [bundle[4] for bundle in bundles]
    total_auto_resets = sum(max(0, len(env.reset_history) - 1) for env in envs)
    pose_shapes = [list(record["poses"].shape) for env in envs for record in env.reset_history]
    all_initial_states_clean = all(
        not planner["cached_trajectory"]
        and planner["tracker_step_counter"] == 0
        and planner["planner_step_counter"] == 0
        and planner["tracker_previous_error"] == 0.0
        and planner["previous_opponent_pose_max_abs"] == 0.0
        and planner["previous_local_trajectory_max_abs"] == 0.0
        for controller in controllers
        for snapshot in controller.reset_snapshots
        for planner in snapshot["planners"].values()
    )
    planner_ids_by_env = [
        {id(planner) for planner in factory.created}
        for factory in planner_factories
    ]
    first_rng_offsets = [
        float(provider.samples[0].poses[0, 0] - 10.0 * provider.env_id)
        for provider in providers
    ]

    # Re-seeding resets the wrapper-owned RNG deterministically even though the
    # provider object continues to count calls.
    deterministic_env, deterministic_core, deterministic_provider, _, _ = make_poc_env(9, "timeout", 2)
    deterministic_env.reset(seed=1234)
    first_seeded_poses = deterministic_core.reset_poses[-1].copy()
    deterministic_env.reset(seed=1234)
    second_seeded_poses = deterministic_core.reset_poses[-1].copy()
    calls_before_override = deterministic_provider.call_count
    explicit = EpisodeResetSpec(
        poses=np.asarray([[1.0, 2.0, 0.0], [3.0, 4.0, 0.1]]),
        initial_speed_feature=17.0,
        scenario={"map_name": "Fixed", "opp_raceline": "fixed", "opp_speedscale": 0.75},
    )
    observation, _ = deterministic_env.reset(options={"reset_spec": explicit})
    override_provider_called = deterministic_provider.call_count == calls_before_override + 1
    options_override_pose_error = float(np.max(np.abs(deterministic_core.reset_poses[-1] - explicit.poses)))
    options_override_speed_error = abs(float(observation[-1]) - explicit.initial_speed_feature)
    deterministic_env.reset(options={"scenario": {"scenario_override_marker": 123}})
    scenario_override_applied = deterministic_env.reset_history[-1]["scenario"]["scenario_override_marker"] == 123
    try:
        deterministic_core.reset()  # type: ignore[call-arg]
    except TypeError:
        strict_core_rejects_missing_poses = True
    else:
        strict_core_rejects_missing_poses = False
    deterministic_env.close()

    # Two fresh controllers with identical raw state/scenario must produce the
    # same opponent action; PPO ego actions are not arguments to this function.
    spec = providers[0].samples[0]
    raw = cores[0].observation_history[0]
    factory_a, factory_b = FakePlannerFactory(100), FakePlannerFactory(100)
    controller_a = LatticePlannerOpponentController(factory_a)
    controller_b = LatticePlannerOpponentController(factory_b)
    controller_a.reset(spec, 2, 0)
    controller_b.reset(spec, 2, 0)
    opponent_a = controller_a.actions(raw)[1]
    opponent_b = controller_b.actions(raw)[1]
    controller_a.actions(raw)
    controller_a.actions(raw)
    replan_frequency_correct = factory_a.created[0].plan_calls == 2 and factory_a.created[0].tracker.calls == 3

    return {
        "parallel_envs": len(envs),
        "episode_lengths": [cores[0].horizon, cores[1].horizon],
        "total_auto_resets": total_auto_resets,
        "reset_counts": [len(env.reset_history) for env in envs],
        "all_reset_pose_shapes": pose_shapes,
        "all_reset_poses_reached_strict_core": all(
            len(core.reset_poses) == len(env.reset_history) for core, env in zip(cores, envs)
        ),
        "provider_instances_independent": len({id(provider) for provider in providers}) == len(providers),
        "parallel_env_rng_samples_differ": len(set(first_rng_offsets)) == len(first_rng_offsets),
        "controller_instances_independent": len({id(controller) for controller in controllers}) == len(controllers),
        "planner_instances_disjoint_between_envs": planner_ids_by_env[0].isdisjoint(planner_ids_by_env[1]),
        "opponent_state_clean_after_every_reset": all_initial_states_clean,
        "seeded_reset_poses_max_error": float(np.max(np.abs(first_seeded_poses - second_seeded_poses))),
        "options_override_still_called_provider": override_provider_called,
        "options_override_pose_max_error": options_override_pose_error,
        "options_override_initial_speed_error": options_override_speed_error,
        "scenario_only_override_applied": scenario_override_applied,
        "strict_core_rejects_reset_without_poses": strict_core_rejects_missing_poses,
        "opponent_determinism_max_error": float(np.max(np.abs(opponent_a - opponent_b))),
        "opponent_action_has_no_direct_ego_action_argument": list(
            inspect.signature(controller_a.actions).parameters
        ) == ["raw_observation"],
        "opponent_replan_and_tracker_frequency_correct": replan_frequency_correct,
    }


class ContractCore:
    """One-step strict core used by speed, LiDAR and termination table tests."""

    def __init__(
        self,
        scans: np.ndarray,
        measured_speeds: list[float],
        collisions: tuple[bool, ...] = (False,),
        base_terminated: bool = False,
        base_truncated: bool = False,
        timestep: float = 1.0,
        privileged_offset: float = 0.0,
    ):
        self.num_agents = len(collisions)
        self.scans = np.asarray(scans)
        self.measured_speeds = measured_speeds
        self.collisions = collisions
        self.base_terminated = base_terminated
        self.base_truncated = base_truncated
        self.timestep = timestep
        self.privileged_offset = privileged_offset
        self.step_index = 0
        self.reset_poses: list[np.ndarray] = []

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        speed = self.measured_speeds[min(self.step_index, len(self.measured_speeds) - 1)]
        return {
            "scans": np.stack([self.scans.copy() for _ in range(self.num_agents)]),
            "linear_vels_x": np.asarray([speed] + [2.0] * (self.num_agents - 1), dtype=np.float32),
            "collisions": np.asarray(self.collisions, dtype=np.float32),
            "poses_x": np.arange(self.num_agents, dtype=np.float32) + self.privileged_offset,
            "poses_y": np.zeros(self.num_agents, dtype=np.float32) - self.privileged_offset,
            "poses_theta": np.zeros(self.num_agents, dtype=np.float32),
            "opponent_pose": np.full((max(1, self.num_agents - 1), 3), self.privileged_offset, dtype=np.float32),
            "reference_geometry": np.asarray([self.privileged_offset], dtype=np.float32),
        }

    def reset(self, *, poses: np.ndarray):
        if np.asarray(poses).shape != (self.num_agents, 3):
            raise ValueError("poses required")
        self.reset_poses.append(np.asarray(poses).copy())
        self.step_index = 0
        return self._obs(), 0.0, False, {"timestep": self.timestep}

    def step(self, action: np.ndarray):
        del action
        self.step_index += 1
        return self._obs(), 2.0, self.base_terminated, self.base_truncated, {"timestep": self.timestep}

    def close(self):
        return None


def _fixed_provider(num_agents: int, initial_speed: float = 11.0):
    poses = np.zeros((num_agents, 3), dtype=np.float64)

    def provider(rng: np.random.Generator) -> EpisodeResetSpec:
        del rng
        return EpisodeResetSpec(poses.copy(), initial_speed, {"map_name": "Fixed"})

    return provider


def speed_and_lidar_contracts() -> dict[str, Any]:
    scan = np.arange(720, dtype=np.float64)
    core = ContractCore(scan, measured_speeds=[21.0, 22.0, 23.0])
    env = End2RaceGymnasiumEnv(core, 10.0, _fixed_provider(1, 11.0))
    observations = [env.reset()[0]]
    observations.append(env.step(np.asarray([0.1, 12.0], dtype=np.float32))[0])
    observations.append(env.step(np.asarray([-0.1, 13.0], dtype=np.float32))[0])
    actual_speed_features = [float(obs[-1]) for obs in observations]
    independent_speed_oracle = [11.0, 21.0, 22.0]
    expected_indices = np.linspace(0, scan.size - 1, END2RACE_LIDAR_SIZE, dtype=int)
    beam_error = float(np.max(np.abs(observations[0][:-1] - scan[expected_indices].astype(np.float32))))
    env.close()

    fail_fast_results: dict[str, bool] = {}
    bad_scans = {
        "short_scan": np.arange(359, dtype=np.float64),
        "nan_scan": np.concatenate((np.arange(719, dtype=np.float64), [np.nan])),
        "inf_scan": np.concatenate((np.arange(719, dtype=np.float64), [np.inf])),
    }
    for name, bad_scan in bad_scans.items():
        bad_core = ContractCore(bad_scan, measured_speeds=[1.0])
        bad_env = End2RaceGymnasiumEnv(bad_core, 1.0, _fixed_provider(1))
        try:
            bad_env.reset()
        except ValueError:
            fail_fast_results[name] = True
        else:
            fail_fast_results[name] = False
        finally:
            bad_env.close()

    privileged_core_a = ContractCore(scan, measured_speeds=[4.0], collisions=(False, False), privileged_offset=0.0)
    privileged_core_b = ContractCore(scan, measured_speeds=[4.0], collisions=(False, False), privileged_offset=999.0)
    privileged_env_a = End2RaceGymnasiumEnv(
        privileged_core_a, 1.0, _fixed_provider(2, 5.0), opponent_controller=NoOpOpponentController()
    )
    privileged_env_b = End2RaceGymnasiumEnv(
        privileged_core_b, 1.0, _fixed_provider(2, 5.0), opponent_controller=NoOpOpponentController()
    )
    privileged_observation_a = privileged_env_a.reset()[0]
    privileged_observation_b = privileged_env_b.reset()[0]
    privileged_error = float(np.max(np.abs(privileged_observation_a - privileged_observation_b)))
    privileged_env_a.close()
    privileged_env_b.close()
    return {
        "speed_feature_trace": actual_speed_features,
        "independent_evaluator_oracle": independent_speed_oracle,
        "speed_feature_max_error": float(np.max(np.abs(np.asarray(actual_speed_features) - independent_speed_oracle))),
        "previous_desired_commands_not_used": [12.0, 13.0] != actual_speed_features[1:],
        "current_measured_speeds_not_used": [22.0, 23.0] != actual_speed_features[1:],
        "lidar_source_beams": int(scan.size),
        "lidar_beam_index_max_error": beam_error,
        "lidar_expected_first_last_indices": [int(expected_indices[0]), int(expected_indices[-1])],
        "fail_fast": fail_fast_results,
        "privileged_field_metamorphic_observation_error": privileged_error,
    }


def termination_semantics_table() -> dict[str, Any]:
    cases = [
        ("ego_collision_only", (True, False), False, False, False, True, False),
        ("opponent_collision_only", (False, True), False, False, False, False, False),
        ("base_terminal_only", (False, False), True, False, False, True, False),
        ("timeout_only", (False, False), False, False, True, False, True),
        ("base_truncation_only", (False, False), False, True, False, False, True),
        ("ego_collision_timeout", (True, False), False, False, True, True, False),
        ("opponent_collision_timeout", (False, True), False, False, True, False, True),
        ("base_terminal_timeout", (False, False), True, False, True, True, False),
        ("base_truncation_timeout", (False, False), False, True, True, False, True),
        ("ego_and_opponent_collision", (True, True), False, False, False, True, False),
    ]
    rows: list[dict[str, Any]] = []
    max_reward_error = 0.0
    for name, collisions, base_term, base_trunc, timeout, expected_term, expected_trunc in cases:
        core = ContractCore(
            np.arange(360, dtype=np.float32),
            measured_speeds=[1.0, 2.0],
            collisions=collisions,
            base_terminated=base_term,
            base_truncated=base_trunc,
            timestep=1.0,
        )
        env = End2RaceGymnasiumEnv(
            core,
            sim_duration=0.5 if timeout else 10.0,
            reset_provider=_fixed_provider(2),
            opponent_controller=NoOpOpponentController(),
        )
        env.reset()
        _obs, reward, terminated, truncated, info = env.step(np.asarray([0.0, 3.0], dtype=np.float32))
        max_reward_error = max(max_reward_error, abs(reward - 2.0))
        rows.append(
            {
                "case": name,
                "terminated": terminated,
                "truncated": truncated,
                "expected_terminated": expected_term,
                "expected_truncated": expected_trunc,
                "matches": terminated == expected_term and truncated == expected_trunc,
                "ego_collision": info["ego_collision"],
                "opponent_collision": info["opponent_collision"],
                "base_terminated": info["base_terminated"],
                "base_truncated": info["base_truncated"],
                "timeout": info["timeout"],
                "termination_reason": info["termination_reason"],
            }
        )
        env.close()
    return {
        "rows": rows,
        "all_cases_match": all(row["matches"] for row in rows),
        "base_reward_max_error": max_reward_error,
        "opponent_only_collision_continues": next(
            not row["terminated"] and not row["truncated"]
            for row in rows
            if row["case"] == "opponent_collision_only"
        ),
        "opponent_collision_timeout_is_truncation": next(
            not row["terminated"] and row["truncated"]
            for row in rows
            if row["case"] == "opponent_collision_timeout"
        ),
        "true_terminal_beats_timeout": all(
            row["terminated"] and not row["truncated"]
            for row in rows
            if row["case"] in {"ego_collision_timeout", "base_terminal_timeout"}
        ),
    }


def gradient_and_optimizer_contract(model: RecurrentPPO, bundles: list[tuple[Any, ...]]) -> dict[str, Any]:
    policy = model.policy
    actor_named = list(policy.end2race_actor.named_parameters())
    critic_named = list(policy.value_net.named_parameters())
    distribution_named = [("log_std", policy.log_std)]
    actor_ids = {id(parameter) for _, parameter in actor_named}
    critic_ids = {id(parameter) for _, parameter in critic_named}
    distribution_ids = {id(policy.log_std)}
    optimizer_parameters = [parameter for group in policy.optimizer.param_groups for parameter in group["params"]]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]

    np.random.seed(3)
    rollout_data = next(model.rollout_buffer.get(MINIBATCH_SIZE))
    mask = rollout_data.mask > 1e-8
    values, log_prob, _entropy = policy.evaluate_actions(
        rollout_data.observations,
        rollout_data.actions,
        rollout_data.lstm_states,
        rollout_data.episode_starts,
    )
    advantages = rollout_data.advantages
    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
    policy_loss = -torch.mean(
        torch.min(advantages * ratio, advantages * torch.clamp(ratio, 0.8, 1.2))[mask]
    )
    value_loss = torch.mean(((rollout_data.returns - values.flatten()) ** 2)[mask])

    def backward_result(loss: torch.Tensor) -> dict[str, Any]:
        policy.optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)

        def stats(named_parameters: list[tuple[str, torch.nn.Parameter]]) -> dict[str, Any]:
            finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for _, parameter in named_parameters
            )
            nonzero = [
                name
                for name, parameter in named_parameters
                if parameter.grad is not None and bool(torch.any(parameter.grad != 0))
            ]
            return {"finite": finite, "nonzero": nonzero, "nonzero_count": len(nonzero)}

        return {
            "actor": stats(actor_named),
            "critic": stats(critic_named),
            "distribution": stats(distribution_named),
        }

    policy_gradients = backward_result(policy_loss)
    value_gradients = backward_result(value_loss)
    expected_optimizer_ids = actor_ids | critic_ids | distribution_ids
    opponent_parameter_count = 0
    for bundle in bundles:
        controller = bundle[3]
        for planner in controller._planners.values():
            if isinstance(planner, torch.nn.Module):
                opponent_parameter_count += sum(parameter.numel() for parameter in planner.parameters())
            else:
                opponent_parameter_count += sum(
                    value.numel()
                    for value in vars(planner).values()
                    if isinstance(value, torch.nn.Parameter)
                )
    return {
        "policy_loss": policy_gradients,
        "value_loss": value_gradients,
        "active_actor_parameter_count": len(actor_named) - 1,
        "dummy_embedding_policy_gradient_absent": "dummy_embedding" not in policy_gradients["actor"]["nonzero"],
        "optimizer_group_count": len(policy.optimizer.param_groups),
        "optimizer_tensor_count": len(optimizer_parameters),
        "optimizer_scalar_parameter_count": sum(parameter.numel() for parameter in optimizer_parameters),
        "optimizer_parameter_identities_unique": len(optimizer_ids) == len(set(optimizer_ids)),
        "optimizer_parameters_fully_classified": set(optimizer_ids) == expected_optimizer_ids,
        "unused_inherited_action_net_parameter_count": sum(parameter.numel() for parameter in policy.action_net.parameters()),
        "opponent_planner_parameter_count": opponent_parameter_count,
    }


def checkpoint_compatibility(policy: End2RaceGRUPolicy) -> dict[str, Any]:
    bc_keys = list(torch.load(DEFAULT_BC_CHECKPOINT, map_location="cpu", weights_only=True).keys())
    with tempfile.TemporaryDirectory(prefix="end2race_sb3_actor_") as directory:
        output_path = Path(directory) / "actor_only.pth"
        torch.save(policy.end2race_actor.state_dict(), output_path)
        reloaded = End2Race(mask_prob=0.0, hidden_scale=4)
        state_dict = torch.load(output_path, map_location="cpu", weights_only=True)
        incompatible = reloaded.load_state_dict(state_dict, strict=True)
        max_error = max(float((state_dict[key] - reloaded.state_dict()[key]).abs().max()) for key in state_dict)
    return {
        "strict_load_succeeded": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "actor_keys_match_bc_schema": list(state_dict.keys()) == bc_keys,
        "actor_key_count": len(state_dict),
        "roundtrip_max_absolute_error": max_error,
        "temporary_output_only": True,
    }


def real_f110_contract_smoke() -> dict[str, Any]:
    map_image = ROOT / "f1tenth_racetracks" / "Austin" / "Austin_map.png"
    if not map_image.exists():
        return {"resources_available": False, "passed": False, "reason": "Austin map resources unavailable"}
    env = None
    try:
        import gym
        from f110_gym.envs.base_classes import Integrator
        from utils import load_positions_and_speeds_from_params

        scenario = {
            "map_name": "Austin",
            "ego_raceline": "raceline1",
            "opp_raceline": "raceline1",
            "opp_speedscale": 0.5,
            "ego_idx": 0,
            "opp_idx": 15,
        }
        poses, initial_speeds = load_positions_and_speeds_from_params(scenario, "Austin")
        spec = EpisodeResetSpec(poses, float(initial_speeds[0] * 0.9), scenario)

        def provider(rng: np.random.Generator) -> EpisodeResetSpec:
            del rng
            return deepcopy(spec)

        core = gym.make(
            "f110-v0",
            map=str(ROOT / "f1tenth_racetracks" / "Austin" / "Austin_map"),
            map_ext=".png",
            num_agents=2,
            timestep=0.01,
            integrator=Integrator.RK4,
        )
        wrapped = End2RaceGymnasiumEnv(
            core,
            sim_duration=0.01,
            reset_provider=provider,
            opponent_controller=LatticePlannerOpponentController(),
        )
        env = DummyVecEnv([lambda: wrapped])
        observation = env.reset()
        new_observation, _reward, done, infos = env.step(
            np.asarray([[0.0, float(initial_speeds[0])]], dtype=np.float32)
        )
        result = {
            "resources_available": True,
            "passed": bool(
                observation.shape == (1, END2RACE_LIDAR_SIZE + 1)
                and new_observation.shape == observation.shape
                and done[0]
                and infos[0].get("TimeLimit.truncated", False)
                and infos[0].get("terminal_observation") is not None
                and len(wrapped.reset_history) >= 2
                and all(record["poses"].shape == (2, 3) for record in wrapped.reset_history)
            ),
            "initial_observation_shape": list(observation.shape),
            "auto_reset_observation_shape": list(new_observation.shape),
            "auto_reset_count": len(wrapped.reset_history) - 1,
            "terminal_observation_present": infos[0].get("terminal_observation") is not None,
            "time_limit_truncated": bool(infos[0].get("TimeLimit.truncated", False)),
            "reset_pose_shapes": [list(record["poses"].shape) for record in wrapped.reset_history],
        }
        return result
    except Exception as exc:
        return {
            "resources_available": True,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if env is not None:
            env.close()


def max_parameter_delta(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    delta = 0.0
    for name, value in module.state_dict().items():
        if torch.is_floating_point(value):
            delta = max(delta, float((value.detach().cpu() - before[name]).abs().max()))
    return delta


def run_poc(include_real_f110: bool = True) -> dict[str, Any]:
    torch.manual_seed(20260715)
    np.random.seed(20260715)
    model, bundles = build_model()
    reference = load_reference_actor()
    model.policy.set_training_mode(False)

    identity = bc_sequence_identity(model.policy, reference)
    hidden_reset = episode_reset_identity(model.policy, reference)
    speed_lidar = speed_and_lidar_contracts()
    termination = termination_semantics_table()
    callback, trace_callback = setup_and_collect(model)
    action = action_contract(model, bundles, trace_callback)
    reset_opponent = reset_and_opponent_evidence(bundles)

    rewards = model.rollout_buffer.rewards.copy()
    values = model.rollout_buffer.values.copy()
    advantages = model.rollout_buffer.advantages.copy()
    timeout = timeout_and_advantage_checks(model, bundles, rewards, values, advantages)
    replay = replay_identity(model, bundles, reference)
    gradients = gradient_and_optimizer_contract(model, bundles)
    checkpoint = checkpoint_compatibility(model.policy)

    before = {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()}
    model.train()
    train_delta = max_parameter_delta(before, model.policy)
    callback.on_training_end()
    model.env.close()

    real_f110 = real_f110_contract_smoke() if include_real_f110 else {
        "resources_available": (ROOT / "f1tenth_racetracks" / "Austin" / "Austin_map.png").exists(),
        "passed": None,
        "skipped": True,
    }
    results = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__,
        },
        "bc_sequence_identity": identity,
        "episode_hidden_reset_identity": hidden_reset,
        "reset_and_opponent_contract": reset_opponent,
        "speed_and_lidar_contract": speed_lidar,
        "termination_semantics": termination,
        "ego_action_contract": action,
        "ppo_replay_identity": replay,
        "timeout_bootstrap": timeout,
        "gradient_and_optimizer_contract": gradients,
        "checkpoint_compatibility": checkpoint,
        "real_f110_contract_smoke": real_f110,
        "zero_lr_api_smoke": {
            "parallel_envs": N_ENVS,
            "rollouts": 1,
            "ppo_train_calls": 1,
            "learning_rate": float(model.policy.optimizer.param_groups[0]["lr"]),
            "max_parameter_delta": train_delta,
        },
        "end2race_learner_training_performed": False,
    }
    return results


def main() -> None:
    results = run_poc(include_real_f110=True)
    print(f"deterministic actor max error: {results['bc_sequence_identity']['max_deterministic_action_error']:.12g}")
    print(f"hidden identity max error: {results['bc_sequence_identity']['max_hidden_absolute_error']:.12g}")
    print(f"buffer/core ego action max error: {results['ego_action_contract']['max_buffer_ego_action_vs_core_error']:.12g}")
    print(f"SB3 pre-env clipping count: {results['ego_action_contract']['sb3_pre_env_action_clipping_count']}")
    print(f"replay logp max error: {results['ppo_replay_identity']['max_logp_absolute_error']:.12g}")
    print(f"ratio max deviation: {results['ppo_replay_identity']['max_ratio_deviation']:.12g}")
    print(f"real F110 contract smoke: {results['real_f110_contract_smoke']['passed']}")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

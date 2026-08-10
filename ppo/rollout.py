"""End2Race recurrent PPO rollout storage and training algorithm."""

from __future__ import annotations

from collections.abc import Generator
import copy as copy_module
import math
from typing import Optional

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer, create_sequencers
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RecurrentRolloutBufferSamples, RNNStates
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.utils import FloatSchedule, explained_variance, safe_mean
from stable_baselines3.common.vec_env import VecNormalize

from ppo.collision_anchor import CollisionBCAnchor
from ppo.policy import (
    BASELINE_EXPLORATION_MODE,
    END2RACE_OBSERVATION_SIZE,
    EXPLORATION_GATE_INFO_KEY,
    PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE,
    P20_CRITIC_VARIANTS,
    PRIVILEGED_FEATURE_HIGHS,
    PRIVILEGED_FEATURE_LOWS,
)
from utils import TrainingRecorder, require_finite_number, require_finite_tensor


WARMUP_MAX_EPOCHS = 30
WARMUP_PATIENCE = 3
WARMUP_TRAIN_FRACTION = 0.8
VALUE_LOSS_COEFFICIENT = 0.5
MAX_GRAD_NORM = 0.5


class End2RaceRolloutBuffer(RecurrentRolloutBuffer):
    """Store the real actor and independent-critic GRU streams."""

    def __init__(self, *args, store_independent_gru_hidden: bool = False, **kwargs):
        self.store_independent_gru_hidden = bool(store_independent_gru_hidden)
        self.joint_context_carry: dict[int, dict] = {}
        self.joint_context_next_carry: dict[int, dict] = {}
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        self.joint_context_carry = self.joint_context_next_carry
        self.joint_context_next_carry = {}
        RolloutBuffer.reset(self)
        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.recurrent_resets = np.zeros(
            (self.buffer_size, self.n_envs), dtype=bool
        )
        self.exploration_speed_log_stds = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.exploration_danger_gates = np.zeros(
            (self.buffer_size, self.n_envs), dtype=bool
        )
        self.exploration_temporal_active = np.zeros(
            (self.buffer_size, self.n_envs), dtype=bool
        )
        self.exploration_block_ids = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.int64
        )
        self.exploration_standard_residuals = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.joint_temporal_active = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.joint_temporal_block_uids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.joint_temporal_block_positions = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self.joint_temporal_prefix_steps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.joint_temporal_collision_sources = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.joint_temporal_standard_residuals = np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32)
        if self.store_independent_gru_hidden:
            self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self._staged_exploration: tuple[np.ndarray, ...] | None = None
        self._staged_recurrent_resets: np.ndarray | None = None
        self.current_valid_by_timestep: tuple[tuple[bool, ...], ...] | None = None
        self.current_collision_mask: torch.Tensor | None = None
        self.current_speed_log_stds: torch.Tensor | None = None
        self.current_danger_gates: torch.Tensor | None = None
        self.current_temporal_active: torch.Tensor | None = None
        self.current_block_ids: torch.Tensor | None = None
        self.current_standard_residuals: torch.Tensor | None = None
        self.current_joint_temporal_active: torch.Tensor | None = None
        self.current_joint_temporal_block_uids: torch.Tensor | None = None
        self.current_joint_temporal_block_positions: torch.Tensor | None = None
        self.current_joint_temporal_prefix_steps: torch.Tensor | None = None
        self.current_joint_temporal_collision_sources: torch.Tensor | None = None
        self.current_joint_temporal_standard_residuals: torch.Tensor | None = None
        self.current_joint_temporal_contexts: list[dict | None] | None = None

    def stage_exploration(
        self,
        *,
        speed_log_std: np.ndarray,
        danger_gate: np.ndarray,
        temporal_active: np.ndarray,
        block_id: np.ndarray,
        standard_residual: np.ndarray,
        joint_active: np.ndarray,
        joint_block_uid: np.ndarray,
        joint_block_position: np.ndarray,
        joint_prefix_step: np.ndarray,
        joint_collision_source: np.ndarray,
        joint_standard_residual: np.ndarray,
    ) -> None:
        arrays = (
            np.asarray(speed_log_std, dtype=np.float32).reshape(-1),
            np.asarray(danger_gate, dtype=bool).reshape(-1),
            np.asarray(temporal_active, dtype=bool).reshape(-1),
            np.asarray(block_id, dtype=np.int64).reshape(-1),
            np.asarray(standard_residual, dtype=np.float32).reshape(-1),
            np.asarray(joint_active, dtype=bool).reshape(-1),
            np.asarray(joint_block_uid, dtype=np.int64).reshape(-1),
            np.asarray(joint_block_position, dtype=np.int64).reshape(-1),
            np.asarray(joint_prefix_step, dtype=np.int64).reshape(-1),
            np.asarray(joint_collision_source, dtype=bool).reshape(-1),
        )
        if any(array.shape != (self.n_envs,) for array in arrays):
            raise RuntimeError(
                f"Exploration transition fields must have shape {(self.n_envs,)}"
            )
        residuals = np.asarray(joint_standard_residual, dtype=np.float32)
        if residuals.shape != (self.n_envs, 2):
            raise RuntimeError(f"Joint-temporal residuals must have shape {(self.n_envs, 2)}")
        if not np.isfinite(arrays[0]).all() or not np.isfinite(arrays[4]).all() or not np.isfinite(residuals).all():
            raise ValueError("Exploration transition fields must be finite")
        self._staged_exploration = (*arrays, residuals)

    def stage_recurrent_resets(self, recurrent_resets: np.ndarray) -> None:
        resets = np.asarray(recurrent_resets, dtype=bool).reshape(-1)
        if resets.shape != (self.n_envs,):
            raise RuntimeError(
                f"Recurrent reset fields must have shape {(self.n_envs,)}"
            )
        self._staged_recurrent_resets = resets

    def add(self, obs, action, reward, episode_start, value, log_prob, *, lstm_states: RNNStates) -> None:
        if self._staged_exploration is None:
            raise RuntimeError(
                "Exploration distribution fields were not staged before add"
            )
        (
            self.exploration_speed_log_stds[self.pos],
            self.exploration_danger_gates[self.pos],
            self.exploration_temporal_active[self.pos],
            self.exploration_block_ids[self.pos],
            self.exploration_standard_residuals[self.pos],
            self.joint_temporal_active[self.pos],
            self.joint_temporal_block_uids[self.pos],
            self.joint_temporal_block_positions[self.pos],
            self.joint_temporal_prefix_steps[self.pos],
            self.joint_temporal_collision_sources[self.pos],
            self.joint_temporal_standard_residuals[self.pos],
        ) = self._staged_exploration
        self._staged_exploration = None
        recurrent_resets = (
            np.asarray(episode_start, dtype=bool).reshape(-1)
            if self._staged_recurrent_resets is None
            else self._staged_recurrent_resets
        )
        if recurrent_resets.shape != (self.n_envs,):
            raise RuntimeError(
                f"Recurrent reset fields must have shape {(self.n_envs,)}"
            )
        self.recurrent_resets[self.pos] = recurrent_resets
        self._staged_recurrent_resets = None
        self.hidden_states_pi[self.pos] = np.asarray(lstm_states.pi[0].cpu().numpy())
        if self.store_independent_gru_hidden:
            self.hidden_states_vf[self.pos] = np.asarray(lstm_states.vf[0].cpu().numpy())
        RolloutBuffer.add(self, obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None, *, rng: np.random.Generator) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        if not self.full:
            raise RuntimeError("Rollout buffer must be full before training")
        if not self.generator_ready:
            self.hidden_states_pi = self.hidden_states_pi.swapaxes(1, 2)
            names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "episode_starts",
                "recurrent_resets",
                "exploration_speed_log_stds",
                "exploration_danger_gates",
                "exploration_temporal_active",
                "exploration_block_ids",
                "exploration_standard_residuals",
                "joint_temporal_active",
                "joint_temporal_block_uids",
                "joint_temporal_block_positions",
                "joint_temporal_prefix_steps",
                "joint_temporal_collision_sources",
                "joint_temporal_standard_residuals",
            ]
            if self.store_independent_gru_hidden:
                self.hidden_states_vf = self.hidden_states_vf.swapaxes(1, 2)
                names.append("hidden_states_vf")
            for name in names:
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True
        total = self.buffer_size * self.n_envs
        batch_size = total if batch_size is None else batch_size
        split_index = int(rng.integers(total))
        indices = np.concatenate((np.arange(total)[split_index:], np.arange(total)[:split_index]))
        env_change = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)
        for start in range(0, total, batch_size):
            yield self._get_samples(indices[start : start + batch_size], env_change)

    def finalize_joint_context_carry(self) -> None:
        if self.generator_ready:
            raise RuntimeError("Joint-temporal carry must be finalized before buffer flattening")
        carry = {}
        last = self.buffer_size - 1
        for env_index in range(self.n_envs):
            if not bool(self.joint_temporal_active[last, env_index]):
                continue
            position = int(self.joint_temporal_block_positions[last, env_index])
            uid = int(self.joint_temporal_block_uids[last, env_index])
            if position < 0 or position >= 49 or uid <= 0:
                continue
            start = last - position
            if start < 0:
                raise RuntimeError("A joint-temporal block crossed more than one rollout boundary")
            positions = np.asarray(self.joint_temporal_block_positions[start : last + 1, env_index], dtype=np.int64).reshape(-1)
            uids = np.asarray(self.joint_temporal_block_uids[start : last + 1, env_index], dtype=np.int64).reshape(-1)
            if not np.array_equal(positions, np.arange(position + 1, dtype=np.int64)) or not np.all(uids == uid):
                raise RuntimeError("Joint-temporal rollout carry is not one complete block prefix")
            carry[env_index] = {
                "block_uid": uid,
                "positions": positions.copy(),
                "standard_residuals": self.joint_temporal_standard_residuals[start : last + 1, env_index].copy(),
            }
        self.joint_context_next_carry = carry

    def _joint_context_for_sequence(self, flat_index: int) -> dict | None:
        active_rows = np.asarray(self.joint_temporal_active).reshape(-1)
        position_rows = np.asarray(self.joint_temporal_block_positions, dtype=np.int64).reshape(-1)
        uid_rows = np.asarray(self.joint_temporal_block_uids, dtype=np.int64).reshape(-1)
        if not bool(active_rows[flat_index]):
            return None
        position = int(position_rows[flat_index])
        uid = int(uid_rows[flat_index])
        if position == 0:
            return None
        if position < 0 or position >= 50 or uid <= 0:
            raise RuntimeError("Joint-temporal sequence start has invalid block metadata")
        env_index = flat_index // self.buffer_size
        time_index = flat_index % self.buffer_size
        current_count = min(position, time_index)
        current_start = flat_index - current_count
        fixed_residual_sum = np.zeros(2, dtype=np.float32)
        fixed_count = 0
        if position > current_count:
            required = position - current_count
            previous = self.joint_context_carry.get(env_index)
            if previous is None or int(previous["block_uid"]) != uid or len(previous["positions"]) < required:
                raise RuntimeError("Joint-temporal cross-rollout context is missing")
            fixed_positions = np.asarray(previous["positions"][-required:], dtype=np.int64)
            if not np.array_equal(fixed_positions, np.arange(required, dtype=np.int64)):
                raise RuntimeError("Joint-temporal cross-rollout positions are incomplete")
            fixed_residuals = np.asarray(previous["standard_residuals"][-required:], dtype=np.float32)
            if fixed_residuals.shape != (required, 2) or not np.isfinite(fixed_residuals).all():
                raise RuntimeError("Joint-temporal cross-rollout residual state is invalid")
            fixed_residual_sum = fixed_residuals.sum(axis=0, dtype=np.float32)
            fixed_count = required
        hidden_start = np.asarray(self.hidden_states_pi[current_start], dtype=np.float32)
        observations = np.asarray(self.observations[current_start:flat_index])
        actions = np.asarray(self.actions[current_start:flat_index])
        context_positions = position_rows[current_start:flat_index]
        if current_count:
            indices = np.arange(current_start, flat_index, dtype=np.int64)
            if not np.all(uid_rows[indices] == uid):
                raise RuntimeError("Joint-temporal within-rollout context changed block UID")
        if len(context_positions) != current_count or not np.array_equal(context_positions, np.arange(fixed_count, position, dtype=np.int64)):
            raise RuntimeError("Joint-temporal context positions are incomplete")
        return {
            "block_uid": uid,
            "fixed_count": fixed_count,
            "fixed_residual_sum": self.to_torch(fixed_residual_sum),
            "positions": self.to_torch(context_positions),
            "observations": self.to_torch(observations),
            "actions": self.to_torch(actions),
            "hidden_start": self.to_torch(hidden_start[:, None, :]).contiguous(),
        }

    def _get_samples(self, batch_inds: np.ndarray, env_change: np.ndarray, env: Optional[VecNormalize] = None) -> RecurrentRolloutBufferSamples:
        del env
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(self.episode_starts[batch_inds], env_change[batch_inds], self.device)
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        sequence_lengths = np.diff(np.concatenate((self.seq_start_indices, np.asarray([len(batch_inds)]))))
        self.current_valid_by_timestep = tuple(tuple(step < int(length) for length in sequence_lengths) for step in range(max_length))
        collision_by_transition = ((batch_inds // self.buffer_size) % 2 == 0).astype(np.float32)
        self.current_collision_mask = self.to_torch(self.pad_and_flatten(collision_by_transition)) > 0.5
        self.current_speed_log_stds = self.to_torch(
            self.pad_and_flatten(
                self.exploration_speed_log_stds[batch_inds]
            )
        )
        self.current_danger_gates = self.to_torch(
            self.pad_and_flatten(
                self.exploration_danger_gates[batch_inds].astype(np.float32)
            )
        ) > 0.5
        self.current_temporal_active = self.to_torch(
            self.pad_and_flatten(
                self.exploration_temporal_active[batch_inds].astype(np.float32)
            )
        ) > 0.5
        self.current_block_ids = self.to_torch(
            self.pad_and_flatten(
                self.exploration_block_ids[batch_inds].astype(np.float32)
            )
        )
        self.current_standard_residuals = self.to_torch(
            self.pad_and_flatten(
                self.exploration_standard_residuals[batch_inds]
            )
        )
        self.current_joint_temporal_active = self.to_torch(self.pad_and_flatten(self.joint_temporal_active[batch_inds].astype(np.float32))) > 0.5
        self.current_joint_temporal_block_uids = torch.as_tensor(self.pad_and_flatten(self.joint_temporal_block_uids[batch_inds]), dtype=torch.int64, device=self.device)
        self.current_joint_temporal_block_positions = torch.as_tensor(self.pad_and_flatten(self.joint_temporal_block_positions[batch_inds]), dtype=torch.int64, device=self.device)
        self.current_joint_temporal_prefix_steps = torch.as_tensor(self.pad_and_flatten(self.joint_temporal_prefix_steps[batch_inds]), dtype=torch.int64, device=self.device)
        self.current_joint_temporal_collision_sources = self.to_torch(self.pad_and_flatten(self.joint_temporal_collision_sources[batch_inds].astype(np.float32))) > 0.5
        self.current_joint_temporal_standard_residuals = self.to_torch(self.pad(self.joint_temporal_standard_residuals[batch_inds]).reshape((padded_batch_size, 2)))
        self.current_joint_temporal_contexts = [self._joint_context_for_sequence(int(batch_inds[index])) for index in self.seq_start_indices]
        actor_hidden = self.to_torch(self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1)).contiguous()
        actor_cell = torch.zeros_like(actor_hidden)
        if self.store_independent_gru_hidden:
            critic_hidden = self.to_torch(self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1)).contiguous()
        else:
            critic_hidden = torch.zeros_like(actor_hidden)
        critic_cell = torch.zeros_like(actor_hidden)
        return RecurrentRolloutBufferSamples(
            observations=self.pad(self.observations[batch_inds]).reshape((padded_batch_size, *self.obs_shape)),
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size, *self.actions.shape[1:])),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates((actor_hidden, actor_cell), (critic_hidden, critic_cell)),
            episode_starts=self.pad_and_flatten(self.recurrent_resets[batch_inds].astype(np.float32)),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )


class End2RaceRecurrentPPO(RecurrentPPO):
    """Run critic warm-up, then separate actor and critic PPO phases."""

    def __init__(
        self,
        *args,
        actor_epochs: int,
        critic_epochs: int,
        recorder: TrainingRecorder,
        collision_bc_anchor_dataset: str = "",
        collision_bc_anchor_beta: float = 0.0,
        **kwargs,
    ):
        self.actor_epochs = actor_epochs
        self.critic_epochs = critic_epochs
        self.recorder = recorder
        self.rollout_speed_physical_std: float | None = None
        self.warmup_completed = False
        self.rollout_index = 0
        self.current_phase = "warmup"
        self.rollout_for_update = 0
        self.rollout_policy_update = 0
        self._rollout_episode_records: list[dict] = []
        self._last_exploration_gates: np.ndarray | None = None
        self._last_recurrent_resets: np.ndarray | None = None
        self._last_prefix_active: np.ndarray | None = None
        self._last_prefix_steps: np.ndarray | None = None
        self._last_prefix_keys: list[str | None] | None = None
        self._last_prefix_strata: list[str | None] | None = None
        self.last_prefix_transition_mask: np.ndarray | None = None
        self.last_prefix_window_mask: np.ndarray | None = None
        self.last_prefix_step_indices: np.ndarray | None = None
        self.last_prefix_key_rows: list[list[str | None]] | None = None
        self.last_rollout_final_values: np.ndarray | None = None
        self.last_rollout_dones: np.ndarray | None = None
        self.last_joint_action_identity_checked_count = 0
        self.collision_bc_anchor_beta = float(collision_bc_anchor_beta)
        self.collision_bc_anchor_dataset = str(collision_bc_anchor_dataset)
        self.collision_bc_anchor = None
        kwargs["n_epochs"] = actor_epochs
        super().__init__(*args, **kwargs)
        if not math.isfinite(self.collision_bc_anchor_beta) or self.collision_bc_anchor_beta < 0.0:
            raise ValueError("Collision BC anchor beta must be finite and nonnegative")
        if self.collision_bc_anchor_beta > 0.0:
            if not self.collision_bc_anchor_dataset:
                raise ValueError("Positive collision BC anchor beta requires a dataset")
            self.collision_bc_anchor = CollisionBCAnchor(self.collision_bc_anchor_dataset, self.policy, self.device)

    @staticmethod
    def _gradient_norm(gradients) -> float:
        squared = sum(float(torch.sum(gradient.detach().double().square()).cpu().item()) for gradient in gradients)
        return math.sqrt(squared)

    def _actor_step_space_norm(self, gradients) -> float:
        gru_count = len(tuple(self.policy.end2race_actor.gru.parameters()))
        gru_lr = float(self.policy.actor_optimizer.param_groups[0]["lr"])
        head_lr = float(self.policy.actor_optimizer.param_groups[1]["lr"])
        squared = 0.0
        for index, gradient in enumerate(gradients):
            learning_rate = gru_lr if index < gru_count else head_lr
            squared += learning_rate * learning_rate * float(torch.sum(gradient.detach().double().square()).cpu().item())
        return math.sqrt(squared)

    def _actor_parameter_gradient_norm(self) -> float:
        gradients = []
        for parameter in self.policy.actor_parameters:
            if parameter.grad is None:
                raise RuntimeError("Actor parameter has no gradient")
            gradients.append(parameter.grad)
        return self._gradient_norm(gradients)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs).to(self.device)
        if not isinstance(self.policy, RecurrentActorCriticPolicy) or not self.policy.supports_end2race_rollout_buffer():
            raise TypeError("End2Race PPO requires the End2Race GRU policy")
        self.rollout_speed_physical_std = self.policy.speed_physical_std()

        lstm = self.policy.lstm_actor
        single_hidden_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        self._last_lstm_states = RNNStates(
            (torch.zeros(single_hidden_shape, device=self.device), torch.zeros(single_hidden_shape, device=self.device)),
            (torch.zeros(single_hidden_shape, device=self.device), torch.zeros(single_hidden_shape, device=self.device)),
        )
        hidden_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
        minibatch_root = np.random.SeedSequence([self.seed, 2])
        warmup_split_seed, warmup_shuffle_seed, actor_minibatch_seed, critic_minibatch_seed = minibatch_root.spawn(4)
        self.warmup_split_rng = np.random.default_rng(warmup_split_seed)  # Warm-up train/validation sequence split only.
        self.warmup_shuffle_rng = np.random.default_rng(warmup_shuffle_seed)  # Warm-up critic epoch shuffle only.
        self.actor_minibatch_rng = np.random.default_rng(actor_minibatch_seed)  # Formal actor minibatch splits only.
        self.critic_minibatch_rng = np.random.default_rng(critic_minibatch_seed)  # Formal critic minibatch splits only.
        self.telemetry_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 4]))  # Full-buffer value-loss telemetry only.
        self.ratio_identity_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 5])
        )
        self.rollout_buffer = End2RaceRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            store_independent_gru_hidden=self.policy.critic_is_independent_gru,
        )
        self.policy._end2race_rollout_buffer = self.rollout_buffer
        if self.policy.speed_exploration_mode == PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE:
            self.policy.configure_joint_temporal_generators(self.seed, self.n_envs)
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

        action_seed = int(np.random.SeedSequence([self.seed, 3]).generate_state(1)[0])
        torch.manual_seed(action_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(action_seed)

    def _prefix_reset_state(self, env, episode_starts: np.ndarray, states: RNNStates) -> tuple[RNNStates, np.ndarray, np.ndarray, np.ndarray, list[str | None], list[str | None]]:
        recurrent_resets = np.asarray(episode_starts, dtype=bool).copy()
        active = np.zeros(self.n_envs, dtype=bool)
        steps = np.zeros(self.n_envs, dtype=np.int64)
        keys: list[str | None] = [None] * self.n_envs
        strata: list[str | None] = [None] * self.n_envs
        actor_hidden = states.pi[0].clone()
        actor_cell = states.pi[1].clone()
        critic_hidden = states.vf[0].clone()
        critic_cell = states.vf[1].clone()
        for index, started in enumerate(episode_starts):
            if not started:
                continue
            info = env.reset_infos[index]
            if not bool(info.get("prefix_reset", False)):
                continue
            prefix = np.asarray(info["prefix_observations"], dtype=np.float32)
            if prefix.shape != (int(info["prefix_length"]), self.observation_space.shape[0]) or not np.isfinite(prefix).all():
                raise RuntimeError("Prefix-reset burn-in observations are invalid")
            actor_state = (torch.zeros_like(actor_hidden[:, index : index + 1]), torch.zeros_like(actor_cell[:, index : index + 1]))
            critic_state = (torch.zeros_like(critic_hidden[:, index : index + 1]), torch.zeros_like(critic_cell[:, index : index + 1]))
            zero_start = torch.zeros(1, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                for observation in prefix:
                    observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
                    _mean, actor_state = self.policy._actor_forward(observation_tensor, actor_state, zero_start)
                    if self.policy.critic_is_independent_gru:
                        _value, critic_state = self.policy._independent_gru_forward_collection(observation_tensor, critic_state, zero_start)
            actor_hidden[:, index : index + 1] = actor_state[0]
            actor_cell[:, index : index + 1] = actor_state[1]
            critic_hidden[:, index : index + 1] = critic_state[0]
            critic_cell[:, index : index + 1] = critic_state[1]
            recurrent_resets[index] = False
            active[index] = True
            keys[index] = str(info["prefix_reset_key"])
            stratum = str(info.get("prefix_reset_stratum", ""))
            if stratum not in ("collision", "lost_overtake"):
                raise RuntimeError("Prefix-reset source stratum is missing or invalid")
            strata[index] = stratum
        return RNNStates((actor_hidden, actor_cell), (critic_hidden, critic_cell)), recurrent_resets, active, steps, keys, strata

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        self.rollout_index += 1
        self.current_phase = "warmup" if not self.warmup_completed else "formal"
        self.rollout_for_update = 0 if not self.warmup_completed else self._n_updates + 1
        self.rollout_policy_update = self._n_updates
        if self.rollout_speed_physical_std is None:
            raise RuntimeError("Speed exploration std was not initialized")
        self._rollout_episode_records = []
        print(
            f"Rollout {self.rollout_index} start: phase={self.current_phase}, "
            f"rollout_policy_update={self.rollout_policy_update}, "
            f"rollout_for_update={self.rollout_for_update}, "
            f"speed_physical_std={self.rollout_speed_physical_std:.9f}",
            flush=True,
        )
        if bool(getattr(env, "prefix_reset_enabled", False)):
            completed = self._collect_prefix_reset_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        elif self.policy.speed_exploration_mode == BASELINE_EXPLORATION_MODE:
            completed = super().collect_rollouts(
                env, callback, rollout_buffer, n_rollout_steps
            )
        else:
            completed = self._collect_structured_exploration_rollouts(
                env, callback, rollout_buffer, n_rollout_steps
            )
        print(f"Rollout {self.rollout_index} complete", flush=True)
        return completed

    def _collect_prefix_reset_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        if not isinstance(rollout_buffer, End2RaceRolloutBuffer):
            raise TypeError("Prefix reset requires End2RaceRolloutBuffer")
        if self._last_obs is None:
            raise RuntimeError("No previous observation was provided")
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        lstm_states = copy_module.deepcopy(self._last_lstm_states)
        if self._last_recurrent_resets is None:
            lstm_states, recurrent_resets, prefix_active, prefix_steps, prefix_keys, prefix_strata = self._prefix_reset_state(env, self._last_episode_starts, lstm_states)
        else:
            recurrent_resets = self._last_recurrent_resets.copy()
            prefix_active = self._last_prefix_active.copy()
            prefix_steps = self._last_prefix_steps.copy()
            prefix_keys = list(self._last_prefix_keys)
            prefix_strata = list(self._last_prefix_strata)
        current_gates = np.asarray([bool(info.get(EXPLORATION_GATE_INFO_KEY, False)) for info in env.reset_infos], dtype=bool)
        prefix_transition_rows = []
        prefix_window_rows = []
        prefix_step_rows = []
        prefix_key_rows = []
        self.last_joint_action_identity_checked_count = 0

        while n_steps < n_rollout_steps:
            self.policy.prepare_rollout_exploration(current_gates, self._last_episode_starts, prefix_active, prefix_steps, np.asarray([stratum == "collision" for stratum in prefix_strata], dtype=bool))
            rollout_buffer.stage_recurrent_resets(recurrent_resets)
            prefix_transition_rows.append(prefix_active.copy())
            prefix_window_rows.append(prefix_active & (prefix_steps < 150))
            prefix_step_rows.append(prefix_steps.copy())
            prefix_key_rows.append(list(prefix_keys))
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                recurrent_starts_tensor = torch.as_tensor(recurrent_resets, dtype=torch.float32, device=self.device)
                actions, values, log_probs, next_lstm_states = self.policy.forward(obs_tensor, lstm_states, recurrent_starts_tensor)
            actions = actions.cpu().numpy()
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high) if isinstance(self.action_space, spaces.Box) else actions
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            if self.policy.speed_exploration_mode == PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE:
                if not np.array_equal(actions, clipped_actions):
                    raise RuntimeError("Prefix joint-temporal action was clipped before the environment")
                executed_actions = np.asarray([info.get("executed_ego_action") for info in infos], dtype=np.float32)
                if executed_actions.shape != clipped_actions.shape or not np.array_equal(executed_actions, clipped_actions):
                    raise RuntimeError("Stored PPO action, wrapper action, and simulator ego action diverged")
                self.last_joint_action_identity_checked_count += self.n_envs
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            for index, done in enumerate(dones):
                if done and infos[index].get("terminal_observation") is not None and infos[index].get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(infos[index]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_state = (next_lstm_states.vf[0][:, index : index + 1].contiguous(), next_lstm_states.vf[1][:, index : index + 1].contiguous())
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_state, torch.zeros(1, dtype=torch.float32, device=self.device))[0]
                    rewards[index] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, lstm_states=lstm_states)
            prefix_steps[prefix_active] += 1
            next_states, next_recurrent_resets, new_prefix_active, new_prefix_steps, new_prefix_keys, new_prefix_strata = self._prefix_reset_state(env, dones, next_lstm_states)
            continuing = ~dones
            new_prefix_active[continuing] = prefix_active[continuing]
            new_prefix_steps[continuing] = prefix_steps[continuing]
            for index in np.flatnonzero(continuing):
                new_prefix_keys[int(index)] = prefix_keys[int(index)]
                new_prefix_strata[int(index)] = prefix_strata[int(index)]
            self._last_obs = new_obs
            self._last_episode_starts = dones
            lstm_states = next_states
            recurrent_resets = next_recurrent_resets
            prefix_active = new_prefix_active
            prefix_steps = new_prefix_steps
            prefix_keys = new_prefix_keys
            prefix_strata = new_prefix_strata
            current_gates = np.asarray([bool((env.reset_infos[index] if done else infos[index]).get(EXPLORATION_GATE_INFO_KEY, False)) for index, done in enumerate(dones)], dtype=bool)

        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, torch.as_tensor(recurrent_resets, dtype=torch.float32, device=self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        rollout_buffer.finalize_joint_context_carry()
        self._last_lstm_states = lstm_states
        self._last_recurrent_resets = recurrent_resets.copy()
        self._last_prefix_active = prefix_active.copy()
        self._last_prefix_steps = prefix_steps.copy()
        self._last_prefix_keys = list(prefix_keys)
        self._last_prefix_strata = list(prefix_strata)
        self.last_prefix_transition_mask = np.asarray(prefix_transition_rows, dtype=bool)
        self.last_prefix_window_mask = np.asarray(prefix_window_rows, dtype=bool)
        self.last_prefix_step_indices = np.asarray(prefix_step_rows, dtype=np.int64)
        self.last_prefix_key_rows = prefix_key_rows
        self.last_rollout_final_values = values.detach().cpu().numpy().reshape(-1).astype(np.float32)
        self.last_rollout_dones = np.asarray(dones, dtype=bool).copy()
        callback.on_rollout_end()
        return True

    def _collect_structured_exploration_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """SB3 recurrent collection with one extra causal gate side channel."""

        if not isinstance(rollout_buffer, End2RaceRolloutBuffer):
            raise TypeError("Structured exploration requires End2RaceRolloutBuffer")
        if self._last_obs is None:
            raise RuntimeError("No previous observation was provided")
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        lstm_states = copy_module.deepcopy(self._last_lstm_states)
        current_gates = (
            np.asarray(self._last_exploration_gates, dtype=bool).copy()
            if self._last_exploration_gates is not None
            else np.asarray(
                [
                    bool(info.get(EXPLORATION_GATE_INFO_KEY, False))
                    for info in env.reset_infos
                ],
                dtype=bool,
            )
        )

        while n_steps < n_rollout_steps:
            self.policy.prepare_rollout_exploration(
                current_gates,
                self._last_episode_starts,
            )
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = torch.as_tensor(
                    self._last_episode_starts,
                    dtype=torch.float32,
                    device=self.device,
                )
                actions, values, log_probs, lstm_states = self.policy.forward(
                    obs_tensor,
                    lstm_states,
                    episode_starts,
                )
            actions = actions.cpu().numpy()
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            for index, done in enumerate(dones):
                if (
                    done
                    and infos[index].get("terminal_observation") is not None
                    and infos[index].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[index]["terminal_observation"]
                    )[0]
                    with torch.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, index : index + 1, :].contiguous(),
                            lstm_states.vf[1][:, index : index + 1, :].contiguous(),
                        )
                        terminal_starts = torch.as_tensor(
                            [False], dtype=torch.float32, device=self.device
                        )
                        terminal_value = self.policy.predict_values(
                            terminal_obs,
                            terminal_lstm_state,
                            terminal_starts,
                        )[0]
                    rewards[index] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states
            current_gates = np.asarray(
                [
                    bool(
                        (
                            env.reset_infos[index]
                            if done
                            else infos[index]
                        ).get(EXPLORATION_GATE_INFO_KEY, False)
                    )
                    for index, done in enumerate(dones)
                ],
                dtype=bool,
            )
            self._last_exploration_gates = current_gates.copy()

        with torch.no_grad():
            final_starts = torch.as_tensor(
                dones, dtype=torch.float32, device=self.device
            )
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device),
                lstm_states.vf,
                final_starts,
            )
        rollout_buffer.compute_returns_and_advantage(
            last_values=values,
            dones=dones,
        )
        callback.on_rollout_end()
        return True

    def dump_logs(self, iteration: int = 0) -> None:
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if self.ep_info_buffer and self.ep_info_buffer[0]:
            self.logger.record("rollout/ep_rew_mean", safe_mean([info["r"] for info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([info["l"] for info in self.ep_info_buffer]))
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.ep_success_buffer:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def _update_info_buffer(self, infos: list[dict], dones: Optional[np.ndarray] = None) -> None:
        super()._update_info_buffer(infos, dones)
        if dones is None:
            dones = np.zeros(len(infos), dtype=bool)
        for info, done in zip(infos, dones):
            if not done:
                continue
            record = {
                "phase": self.current_phase,
                "rollout_index": self.rollout_index,
                "formal_update": self.rollout_for_update,
                "rollout_for_update": self.rollout_for_update,
                "rollout_policy_update": self.rollout_policy_update,
                "scenario_id": str(info["scenario_id"]),
                "scenario_pool": str(info["scenario"]["pool"]),
                "env_role": str(info["env_role"]),
                "sampler_branch": str(info["sampler_branch"]),
                "episode_outcome": str(info["episode_outcome"]),
                "episode_return": float(info["episode_return"]),
                "episode_steps": int(info["episode_steps"]),
                "elapsed_time": float(info["elapsed_time"]),
                "ego_collision": bool(info["ego_collision"]),
                "relative_position_m": float(info["relative_position_m"]),
                "episode_reward_progress": float(info["episode_reward_progress"]),
                "episode_reward_relative": float(info["episode_reward_relative"]),
                "episode_reward_collision": float(info["episode_reward_collision"]),
                "episode_reward_risk": float(info["episode_reward_risk"]),
                "episode_abs_reward_risk": float(info["episode_abs_reward_risk"]),
                "episode_min_obb_clearance_m": float(info["episode_min_obb_clearance_m"]),
                "episode_min_wall_clearance_m": float(info["episode_min_wall_clearance_m"]),
                "episode_risk_active_fraction": float(info["episode_risk_active_fraction"]),
                "termination_reason": str(info["termination_reason"]),
                "timeout": bool(info["timeout"]),
                "opponent_collision": bool(info["opponent_collision"]),
            }
            self.recorder.record_episode(record)
            self._rollout_episode_records.append(record)

    def _warmup_split(self) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        starts = self.rollout_buffer.episode_starts
        sequences = {"collision": [], "ordinary": []}
        for env_index in range(self.n_envs):
            if starts[0, env_index] <= 0.5:
                raise RuntimeError("Warm-up rollout must begin with freshly reset environments")
            role = "collision" if env_index % 2 == 0 else "ordinary"
            boundaries = np.flatnonzero(starts[:, env_index] > 0.5).tolist()
            boundaries.append(self.n_steps)
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end == self.n_steps and not bool(self._last_episode_starts[env_index]):
                    continue
                if end > start:
                    sequences[role].append((env_index, start, end))

        train_sequences: list[tuple[int, int, int]] = []
        validation_sequences: list[tuple[int, int, int]] = []
        for role in ("collision", "ordinary"):
            role_sequences = sequences[role]
            if len(role_sequences) < 2:
                raise RuntimeError(f"Warm-up requires at least two {role} recurrent sequences")
            order = self.warmup_split_rng.permutation(len(role_sequences))
            train_count = min(max(int(len(order) * WARMUP_TRAIN_FRACTION), 1), len(order) - 1)
            for destination, selected in ((train_sequences, order[:train_count]), (validation_sequences, order[train_count:])):
                for sequence_index in selected:
                    destination.append(role_sequences[int(sequence_index)])
        return train_sequences, validation_sequences

    def _flat_sequence_indices(self, sequences: list[tuple[int, int, int]]) -> np.ndarray:
        parts = [np.arange(start, end, dtype=np.int64) * self.n_envs + env_index for env_index, start, end in sequences]
        return np.concatenate(parts) if parts else np.asarray([], dtype=np.int64)

    def _flat_critic_inputs(self) -> np.ndarray:
        return self.rollout_buffer.observations.reshape(-1, *self.rollout_buffer.obs_shape)

    def _critic_batch_loss(self, flat_inputs: np.ndarray, flat_returns: np.ndarray, indices: np.ndarray) -> torch.Tensor:
        inputs = torch.as_tensor(flat_inputs[indices], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(flat_returns[indices], dtype=torch.float32, device=self.device)
        return torch.nn.functional.mse_loss(self.policy.evaluate_values(inputs).flatten(), returns)

    def _validation_loss(self, flat_inputs: np.ndarray, flat_returns: np.ndarray, indices: np.ndarray) -> float:
        losses = []
        with torch.no_grad():
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                losses.append((float(self._critic_batch_loss(flat_inputs, flat_returns, batch).item()), len(batch)))
        return sum(loss * count for loss, count in losses) / sum(count for _loss, count in losses)

    def _pack_sequences(self, sequences: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
        """Group sequences so each recurrent minibatch holds about batch_size transitions."""

        groups: list[list[tuple[int, int, int]]] = []
        current: list[tuple[int, int, int]] = []
        transitions = 0
        for sequence in sequences:
            length = sequence[2] - sequence[1]
            if current and transitions + length > self.batch_size:
                groups.append(current)
                current, transitions = [], 0
            current.append(sequence)
            transitions += length
        if current:
            groups.append(current)
        return groups

    def _independent_gru_sequence_batch(self, sequences: list[tuple[int, int, int]]):
        """Build a padded recurrent critic batch from time-major rollout arrays."""

        buffer = self.rollout_buffer
        n_seq = len(sequences)
        max_length = max(end - start for _env_index, start, end in sequences)
        observation_size = buffer.obs_shape[0]
        num_layers, hidden_size = buffer.hidden_state_shape[1], buffer.hidden_state_shape[3]
        observations = np.zeros((n_seq, max_length, observation_size), dtype=np.float32)
        episode_starts = np.zeros((n_seq, max_length), dtype=np.float32)
        returns = np.zeros((n_seq, max_length), dtype=np.float32)
        valid = np.zeros((n_seq, max_length), dtype=bool)
        hidden = np.zeros((num_layers, n_seq, hidden_size), dtype=np.float32)
        for slot, (env_index, start, end) in enumerate(sequences):
            length = end - start
            observations[slot, :length] = buffer.observations[start:end, env_index]
            episode_starts[slot, :length] = buffer.recurrent_resets[start:end, env_index]
            returns[slot, :length] = buffer.returns[start:end, env_index]
            valid[slot, :length] = True
            hidden[:, slot] = buffer.hidden_states_vf[start, :, env_index]
        valid_by_timestep = tuple(tuple(bool(flag) for flag in valid[:, timestep]) for timestep in range(max_length))
        observations_tensor = torch.as_tensor(observations.reshape(n_seq * max_length, observation_size), device=self.device)
        starts_tensor = torch.as_tensor(episode_starts.reshape(-1), device=self.device)
        returns_tensor = torch.as_tensor(returns.reshape(-1), device=self.device)
        mask_tensor = torch.as_tensor(valid.reshape(-1), device=self.device)
        hidden_tensor = torch.as_tensor(hidden, device=self.device)
        states = (hidden_tensor, torch.zeros_like(hidden_tensor))
        return observations_tensor, states, starts_tensor, valid_by_timestep, returns_tensor, mask_tensor

    def _independent_gru_sequence_loss(self, sequences: list[tuple[int, int, int]]) -> tuple[torch.Tensor, int]:
        observations, states, starts, valid_by_timestep, returns, mask = self._independent_gru_sequence_batch(sequences)
        values = self.policy.evaluate_values_independent_gru(observations, states, starts, valid_by_timestep).flatten()
        return torch.nn.functional.mse_loss(values[mask], returns[mask]), int(mask.sum().item())

    def _independent_gru_validation_loss(self, validation_sequences: list[tuple[int, int, int]]) -> float:
        losses = []
        with torch.no_grad():
            for group in self._pack_sequences(validation_sequences):
                loss, count = self._independent_gru_sequence_loss(group)
                losses.append((float(loss.item()), count))
        return sum(loss * count for loss, count in losses) / sum(count for _loss, count in losses)

    def _apply_critic_gradient(self, loss: torch.Tensor, loss_name: str, critic_grad_norms: list[float]) -> None:
        require_finite_tensor(loss_name, loss)
        self.policy.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters, MAX_GRAD_NORM)
        require_finite_tensor(f"{loss_name} gradient norm", grad_norm)
        critic_grad_norms.append(float(grad_norm.detach().cpu().item()))
        self.policy.critic_optimizer.step()

    def _warmup_critic(self) -> None:
        train_sequences, validation_sequences = self._warmup_split()
        independent_gru = self.policy.critic_is_independent_gru
        if not independent_gru:
            flat_inputs = self._flat_critic_inputs()
            flat_returns = self.rollout_buffer.returns.reshape(-1)
            train_indices = self._flat_sequence_indices(train_sequences)
            validation_indices = self._flat_sequence_indices(validation_sequences)
        best_loss = float("inf")
        best_critic = None
        best_optimizer = None
        best_epoch = 0
        stale_epochs = 0
        critic_grad_norms = []
        warmup_train_losses: list[float] = []
        warmup_validation_losses: list[float] = []
        for epoch in range(WARMUP_MAX_EPOCHS):
            epoch_train_losses: list[tuple[float, int]] = []
            if independent_gru:
                order = self.warmup_shuffle_rng.permutation(len(train_sequences))
                shuffled_sequences = [train_sequences[int(index)] for index in order]
                for group in self._pack_sequences(shuffled_sequences):
                    group_loss, count = self._independent_gru_sequence_loss(group)
                    self._apply_critic_gradient(VALUE_LOSS_COEFFICIENT * group_loss, "Warm-up loss", critic_grad_norms)
                    epoch_train_losses.append((float(group_loss.item()), count))
                validation_loss = self._independent_gru_validation_loss(validation_sequences)
            else:
                shuffled = self.warmup_shuffle_rng.permutation(train_indices)
                for start in range(0, len(shuffled), self.batch_size):
                    batch = shuffled[start : start + self.batch_size]
                    batch_loss = self._critic_batch_loss(flat_inputs, flat_returns, batch)
                    self._apply_critic_gradient(VALUE_LOSS_COEFFICIENT * batch_loss, "Warm-up loss", critic_grad_norms)
                    epoch_train_losses.append((float(batch_loss.item()), len(batch)))
                validation_loss = self._validation_loss(flat_inputs, flat_returns, validation_indices)
            train_loss = sum(loss * count for loss, count in epoch_train_losses) / sum(
                count for _loss, count in epoch_train_losses
            )
            require_finite_number("Warm-up train loss", train_loss)
            require_finite_number("Warm-up validation loss", validation_loss)
            warmup_train_losses.append(train_loss)
            warmup_validation_losses.append(validation_loss)
            print(
                f"Warm-up epoch {epoch + 1}: train_loss={train_loss:.6f}, validation_loss={validation_loss:.6f}",
                flush=True,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_critic = copy_module.deepcopy(self.policy.value_net.state_dict())
                best_optimizer = copy_module.deepcopy(self.policy.critic_optimizer.state_dict())
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= WARMUP_PATIENCE:
                    break
        if best_critic is None or best_optimizer is None:
            raise RuntimeError("Critic warm-up did not produce a valid checkpoint")
        self.policy.value_net.load_state_dict(best_critic)
        self.policy.critic_optimizer.load_state_dict(best_optimizer)
        self.warmup_completed = True
        metrics = {
            "phase": "warmup",
            "critic_variant": self.policy.critic_variant,
            "rollout_speed_physical_std": self.rollout_speed_physical_std,
            "epochs": epoch + 1,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "warmup_train_losses": warmup_train_losses,
            "warmup_validation_losses": warmup_validation_losses,
            "critic_grad_norm_mean": float(np.mean(critic_grad_norms)),
            "critic_grad_norm_max": float(np.max(critic_grad_norms)),
            **self._prefix_reset_statistics(),
        }
        checkpoint_path = self.recorder.save_warmup_critic(self.policy.value_net.state_dict())
        metrics["critic_checkpoint"] = str(checkpoint_path)
        self.recorder.record_metrics(metrics)
        self.logger.record("warmup/epochs", epoch + 1)
        self.logger.record("warmup/best_validation_loss", best_loss)
        print(f"Warm-up complete: best_validation_loss={best_loss:.6f}, checkpoint={checkpoint_path}", flush=True)

    def _batch_values(self, rollout_data) -> torch.Tensor:
        """Critic values for one recurrent minibatch, dispatched by critic variant."""

        if self.policy.critic_is_independent_gru:
            critic_states = (rollout_data.lstm_states.vf[0], rollout_data.lstm_states.vf[1])
            return self.policy.evaluate_values_independent_gru(
                rollout_data.observations, critic_states, rollout_data.episode_starts, self.rollout_buffer.current_valid_by_timestep
            ).flatten()
        return self.policy.evaluate_values(rollout_data.observations).flatten()

    def _full_buffer_value_statistics(self) -> dict[str, float]:
        """Value-fit statistics over one partition of every transition in the rollout."""

        predictions = []
        returns = []
        role_predictions = {"collision": [], "ordinary": []}
        role_returns = {"collision": [], "ordinary": []}
        with torch.no_grad():
            for rollout_data in self.rollout_buffer.get(self.batch_size, rng=self.telemetry_rng):
                mask = rollout_data.mask > 1e-8
                values = self._batch_values(rollout_data)
                predictions.append(values[mask].detach().cpu().numpy())
                returns.append(rollout_data.returns[mask].detach().cpu().numpy())
                collision_mask = self.rollout_buffer.current_collision_mask
                if collision_mask is None:
                    raise RuntimeError("Rollout role mask is missing for value-fit statistics")
                for role, role_mask in (
                    ("collision", mask & collision_mask),
                    ("ordinary", mask & ~collision_mask),
                ):
                    role_predictions[role].append(values[role_mask].detach().cpu().numpy())
                    role_returns[role].append(rollout_data.returns[role_mask].detach().cpu().numpy())
        prediction_array = np.concatenate(predictions).astype(np.float64, copy=False)
        return_array = np.concatenate(returns).astype(np.float64, copy=False)
        statistics = {
            "value_loss": float(np.mean(np.square(prediction_array - return_array))),
            "explained_variance": float(explained_variance(prediction_array, return_array)),
            "value_prediction_mean": float(prediction_array.mean()),
            "value_prediction_std": float(prediction_array.std()),
            "return_mean": float(return_array.mean()),
            "return_std": float(return_array.std()),
        }
        for role in ("collision", "ordinary"):
            role_prediction_array = np.concatenate(role_predictions[role]).astype(np.float64, copy=False)
            role_return_array = np.concatenate(role_returns[role]).astype(np.float64, copy=False)
            statistics[f"{role}_value_loss"] = float(
                np.mean(np.square(role_prediction_array - role_return_array))
            )
            statistics[f"{role}_explained_variance"] = float(
                explained_variance(role_prediction_array, role_return_array)
            )
        for name, value in statistics.items():
            require_finite_number(f"Full-buffer {name}", value)
        return statistics

    @staticmethod
    def _mean_episode_metric(records: list[dict], name: str) -> float | None:
        if not records:
            return None
        return float(np.mean([record[name] for record in records]))

    @classmethod
    def _episode_metrics(cls, episodes: list[dict]) -> dict[str, float | int | None]:
        metrics: dict[str, float | int | None] = {
            "episode_count": len(episodes),
            "mean_episode_steps": cls._mean_episode_metric(episodes, "episode_steps"),
            "mean_episode_return": cls._mean_episode_metric(episodes, "episode_return"),
            "mean_relative_position_m": cls._mean_episode_metric(episodes, "relative_position_m"),
            "mean_episode_min_obb_clearance_m": cls._mean_episode_metric(
                episodes,
                "episode_min_obb_clearance_m",
            ),
            "mean_episode_min_wall_clearance_m": cls._mean_episode_metric(
                episodes,
                "episode_min_wall_clearance_m",
            ),
            "mean_episode_risk_active_fraction": cls._mean_episode_metric(
                episodes,
                "episode_risk_active_fraction",
            ),
            "mean_episode_abs_reward_risk": cls._mean_episode_metric(
                episodes,
                "episode_abs_reward_risk",
            ),
        }
        for component in ("progress", "relative", "collision", "risk"):
            metrics[f"mean_episode_reward_{component}"] = cls._mean_episode_metric(
                episodes,
                f"episode_reward_{component}",
            )
        for role in ("collision", "ordinary"):
            role_episodes = [record for record in episodes if record["env_role"] == role]
            metrics[f"{role}_role_episode_count"] = len(role_episodes)
            metrics[f"mean_{role}_episode_return"] = cls._mean_episode_metric(role_episodes, "episode_return")
            metrics[f"mean_{role}_relative_position_m"] = cls._mean_episode_metric(role_episodes, "relative_position_m")
        return metrics

    def _critic_input_statistics(self) -> dict:
        """Per-rollout critic input telemetry for privileged variants."""

        if self.policy.critic_variant in P20_CRITIC_VARIANTS:
            features = self.rollout_buffer.observations.reshape(-1, self.rollout_buffer.obs_shape[0])[:, END2RACE_OBSERVATION_SIZE:]
            lows = np.asarray(PRIVILEGED_FEATURE_LOWS, dtype=np.float32)
            highs = np.asarray(PRIVILEGED_FEATURE_HIGHS, dtype=np.float32)
            return {
                "privileged_feature_min": [float(value) for value in features.min(axis=0)],
                "privileged_feature_max": [float(value) for value in features.max(axis=0)],
                "privileged_feature_mean": [float(value) for value in features.mean(axis=0)],
                "privileged_feature_std": [float(value) for value in features.std(axis=0)],
                "privileged_feature_saturation_low": [float(value) for value in (features <= lows + 1e-6).mean(axis=0)],
                "privileged_feature_saturation_high": [float(value) for value in (features >= highs - 1e-6).mean(axis=0)],
                "privileged_feature_fraction_ge_0_95": [float(value) for value in (features >= 0.95).mean(axis=0)],
                "privileged_feature_fraction_ge_0_99": [float(value) for value in (features >= 0.99).mean(axis=0)],
            }
        return {}

    def _exploration_statistics(self) -> dict[str, float | int | str]:
        speed_log_std = np.asarray(
            self.rollout_buffer.exploration_speed_log_stds
        )
        danger = np.asarray(self.rollout_buffer.exploration_danger_gates)
        temporal = np.asarray(
            self.rollout_buffer.exploration_temporal_active
        )
        block_ids = np.asarray(self.rollout_buffer.exploration_block_ids)
        residuals = np.asarray(
            self.rollout_buffer.exploration_standard_residuals
        )
        joint_active = np.asarray(self.rollout_buffer.joint_temporal_active)
        joint_uids = np.asarray(self.rollout_buffer.joint_temporal_block_uids)
        joint_positions = np.asarray(self.rollout_buffer.joint_temporal_block_positions)
        joint_prefix_steps = np.asarray(self.rollout_buffer.joint_temporal_prefix_steps)
        joint_collision_sources = np.asarray(self.rollout_buffer.joint_temporal_collision_sources)
        joint_residuals = np.asarray(self.rollout_buffer.joint_temporal_standard_residuals)
        expected_shape = (
            self.rollout_buffer.buffer_size,
            self.rollout_buffer.n_envs,
        )
        if any(
            array.shape != expected_shape
            for array in (speed_log_std, danger, temporal, block_ids, residuals, joint_active, joint_uids, joint_positions, joint_prefix_steps, joint_collision_sources)
        ):
            raise RuntimeError("Exploration telemetry lost its step/env layout")
        if joint_residuals.shape != (*expected_shape, 2):
            raise RuntimeError("Joint-temporal residual telemetry lost its step/env layout")
        same_block = (
            temporal[1:]
            & temporal[:-1]
            & (block_ids[1:] == block_ids[:-1])
            & (block_ids[1:] > 0)
        )
        residual_difference = np.abs(residuals[1:] - residuals[:-1])
        metrics = {
            "speed_exploration_mode": self.policy.speed_exploration_mode,
            "exploration_danger_gate_fraction": float(danger.mean()),
            "exploration_temporal_active_fraction": float(temporal.mean()),
            "exploration_speed_std_mean": float(np.exp(speed_log_std).mean()),
            "exploration_speed_std_min": float(np.exp(speed_log_std).min()),
            "exploration_speed_std_max": float(np.exp(speed_log_std).max()),
            "exploration_standard_residual_mean": float(residuals.mean()),
            "exploration_standard_residual_std": float(residuals.std()),
            "exploration_temporal_same_block_pairs": int(same_block.sum()),
            "exploration_temporal_same_block_max_residual_error": (
                float(residual_difference[same_block].max())
                if np.any(same_block)
                else 0.0
            ),
        }
        joint_count = int(joint_active.sum())
        joint_values = joint_residuals[joint_active]
        joint_blocks = np.unique(joint_uids[joint_active]) if joint_count else np.empty(0, dtype=np.int64)
        joint_leak = joint_active & (~joint_collision_sources | (joint_prefix_steps < 0) | (joint_prefix_steps >= 150) | (joint_positions != joint_prefix_steps % 50))
        metrics.update({
            "joint_temporal_active_count": joint_count,
            "joint_temporal_active_fraction": float(joint_count / np.prod(expected_shape)),
            "joint_temporal_block_count": int(len(joint_blocks)),
            "joint_temporal_unique_block_uid_count": int(len(joint_blocks)),
            "joint_temporal_treatment_leak_count": int(joint_leak.sum()),
            "joint_temporal_steering_residual_mean": float(joint_values[:, 0].mean()) if joint_count else 0.0,
            "joint_temporal_steering_residual_std": float(joint_values[:, 0].std()) if joint_count else 0.0,
            "joint_temporal_speed_residual_mean": float(joint_values[:, 1].mean()) if joint_count else 0.0,
            "joint_temporal_speed_residual_std": float(joint_values[:, 1].std()) if joint_count else 0.0,
            "joint_temporal_cross_correlation": float(np.corrcoef(joint_values.T)[0, 1]) if joint_count > 1 and np.all(joint_values.std(axis=0) > 0.0) else 0.0,
            "joint_temporal_steering_abs_ge_0p95_bound_fraction": float((np.abs(self.rollout_buffer.actions[..., 0][joint_active] / 0.52) >= 0.95).mean()) if joint_count else 0.0,
            "joint_temporal_steering_abs_ge_0p99_bound_fraction": float((np.abs(self.rollout_buffer.actions[..., 0][joint_active] / 0.52) >= 0.99).mean()) if joint_count else 0.0,
            "joint_temporal_speed_min": float(self.rollout_buffer.actions[..., 1][joint_active].min()) if joint_count else 0.0,
            "joint_temporal_speed_max": float(self.rollout_buffer.actions[..., 1][joint_active].max()) if joint_count else 0.0,
            "joint_temporal_action_identity_checked_count": int(self.last_joint_action_identity_checked_count),
        })
        return metrics

    def _prefix_reset_statistics(self) -> dict[str, float | int]:
        if self.last_prefix_transition_mask is None or self.last_prefix_window_mask is None:
            return {
                "prefix_reset_transition_count": 0,
                "prefix_reset_transition_fraction": 0.0,
                "prefix_reset_window_transition_count": 0,
                "prefix_reset_window_transition_fraction": 0.0,
                "prefix_reset_boundary_count": 0,
            }
        transition_count = int(self.last_prefix_transition_mask.sum())
        window_count = int(self.last_prefix_window_mask.sum())
        total = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
        boundaries = np.asarray(self.rollout_buffer.episode_starts, dtype=bool) & ~np.asarray(self.rollout_buffer.recurrent_resets, dtype=bool)
        return {
            "prefix_reset_transition_count": transition_count,
            "prefix_reset_transition_fraction": float(transition_count / total),
            "prefix_reset_window_transition_count": window_count,
            "prefix_reset_window_transition_fraction": float(window_count / total),
            "prefix_reset_boundary_count": int(boundaries.sum()),
        }

    def _assert_full_buffer_ratio_identity(self) -> dict[str, float]:
        """Fail before optimization if rollout/replay distributions differ."""

        def measure(collection_equivalent: bool) -> tuple[float, float]:
            maximum_log_ratio_error = 0.0
            maximum_ratio_error = 0.0
            valid_count = 0
            with torch.no_grad():
                for rollout_data in self.rollout_buffer.get(
                    self.batch_size,
                    rng=self.ratio_identity_rng,
                ):
                    mask = rollout_data.mask > 1e-8
                    log_prob, _entropy = self.policy.evaluate_actor_actions(
                        rollout_data.observations,
                        rollout_data.actions,
                        rollout_data.lstm_states,
                        rollout_data.episode_starts,
                        collection_equivalent=collection_equivalent,
                    )
                    log_ratio = log_prob - rollout_data.old_log_prob
                    ratio = torch.exp(log_ratio)
                    maximum_log_ratio_error = max(
                        maximum_log_ratio_error,
                        float(torch.abs(log_ratio[mask]).max().cpu().item()),
                    )
                    maximum_ratio_error = max(
                        maximum_ratio_error,
                        float(
                            torch.abs(ratio[mask] - 1.0).max().cpu().item()
                        ),
                    )
                    valid_count += int(mask.sum().item())
            expected = (
                self.rollout_buffer.buffer_size
                * self.rollout_buffer.n_envs
            )
            if valid_count != expected:
                raise RuntimeError(
                    f"Ratio identity covered {valid_count} transitions, "
                    f"expected {expected}"
                )
            return maximum_log_ratio_error, maximum_ratio_error

        rng_state = copy_module.deepcopy(
            self.ratio_identity_rng.bit_generator.state
        )
        maximum_log_ratio_error, maximum_ratio_error = measure(False)
        # Collection intentionally preserves one actor call per logical slot,
        # while replay batches valid slots per timestep. Across a full recurrent
        # episode that legacy FP32 accumulation-order difference reaches about
        # 3.5e-3 with 16 envs. A missing 0.15/0.25/0.50 per-transition std
        # produces errors orders of magnitude larger. When the batched replay
        # leaves its 1e-2 legacy envelope, audit the exact one-slot collection
        # path rather than weakening the training/replay contract.
        exact_log_ratio_error = 0.0
        exact_ratio_error = 0.0
        exact_measurement_required = bool(
            self.policy.speed_exploration_mode == PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE
            or maximum_log_ratio_error > 1e-2
            or maximum_ratio_error > 1e-2
        )
        exact_fallback_used = bool(
            maximum_log_ratio_error > 1e-2
            or maximum_ratio_error > 1e-2
        )
        if exact_measurement_required:
            self.ratio_identity_rng.bit_generator.state = rng_state
            exact_log_ratio_error, exact_ratio_error = measure(True)
            if exact_log_ratio_error > 5e-5 or exact_ratio_error > 5e-5:
                raise RuntimeError(
                    "Collection-equivalent likelihood mismatch before PPO "
                    f"update: max_log_ratio_error={exact_log_ratio_error:.9g}, "
                    f"max_ratio_error={exact_ratio_error:.9g}"
                )
        return {
            "preupdate_max_abs_log_ratio": maximum_log_ratio_error,
            "preupdate_max_abs_ratio_minus_one": maximum_ratio_error,
            "preupdate_exact_ratio_measured": exact_measurement_required,
            "preupdate_exact_ratio_fallback_used": exact_fallback_used,
            "preupdate_exact_max_abs_log_ratio": exact_log_ratio_error,
            "preupdate_exact_max_abs_ratio_minus_one": exact_ratio_error,
        }

    def _dry_actor_gradient(self, collection_equivalent, rng_state):
        gradients = [torch.zeros_like(parameter, device="cpu") for parameter in self.policy.actor_parameters]
        losses = []
        valid_counts = []
        old_log_prob_sums = []
        approximate_kl_sum = 0.0
        clip_count = 0
        valid_total = 0
        self.actor_minibatch_rng.bit_generator.state = copy_module.deepcopy(rng_state)
        for rollout_data in self.rollout_buffer.get(self.batch_size, rng=self.actor_minibatch_rng):
            mask = rollout_data.mask > 1e-8
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                valid_advantages = advantages[mask]
                advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
            log_prob, _entropy = self.policy.evaluate_actor_actions(rollout_data.observations, rollout_data.actions, rollout_data.lstm_states, rollout_data.episode_starts, collection_equivalent=collection_equivalent)
            log_ratio = log_prob - rollout_data.old_log_prob
            ratio = torch.exp(log_ratio)
            loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 1 - self.clip_range(self._current_progress_remaining), 1 + self.clip_range(self._current_progress_remaining)))[mask].mean()
            require_finite_tensor("Dry actor policy loss", loss)
            self.policy.actor_optimizer.zero_grad()
            loss.backward()
            for index, parameter in enumerate(self.policy.actor_parameters):
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
                    raise RuntimeError("Dry actor gradient is missing or non-finite")
                gradients[index] += parameter.grad.detach().cpu()
            valid_count = int(mask.sum().item())
            valid_log_ratio = log_ratio[mask].detach()
            approximate_kl_sum += float(((torch.exp(valid_log_ratio) - 1.0) - valid_log_ratio).sum().cpu().item())
            clip_count += int((torch.abs(torch.exp(valid_log_ratio) - 1.0) > self.clip_range(self._current_progress_remaining)).sum().cpu().item())
            valid_total += valid_count
            valid_counts.append(valid_count)
            old_log_prob_sums.append(float(rollout_data.old_log_prob[mask].double().sum().cpu().item()))
            losses.append(float(loss.detach().cpu().item()))
        self.policy.actor_optimizer.zero_grad()
        return {
            "gradients": gradients,
            "losses": losses,
            "valid_counts": valid_counts,
            "old_log_prob_sums": old_log_prob_sums,
            "minibatches": len(losses),
            "valid_total": valid_total,
            "clip_fraction": float(clip_count / valid_total),
            "mean_approximate_kl": float(approximate_kl_sum / valid_total),
        }

    @staticmethod
    def _compare_dry_gradients(batched, exact):
        dot = 0.0
        batched_squared = 0.0
        exact_squared = 0.0
        difference_squared = 0.0
        for left, right in zip(batched, exact):
            left_double = left.double()
            right_double = right.double()
            dot += float(torch.sum(left_double * right_double).item())
            batched_squared += float(torch.sum(left_double * left_double).item())
            exact_squared += float(torch.sum(right_double * right_double).item())
            difference_squared += float(torch.sum((left_double - right_double) ** 2).item())
        batched_norm = math.sqrt(batched_squared)
        exact_norm = math.sqrt(exact_squared)
        if batched_norm <= 0.0 or exact_norm <= 0.0:
            raise RuntimeError("Dry actor gradient norm must be positive")
        return {
            "cosine": dot / (batched_norm * exact_norm),
            "batched_l2_norm": batched_norm,
            "exact_l2_norm": exact_norm,
            "difference_l2_norm": math.sqrt(difference_squared),
            "relative_l2_difference_over_exact": math.sqrt(difference_squared) / exact_norm,
        }

    def _adjudicate_batched_replay(self):
        actor_state_before = [parameter.detach().cpu().clone() for parameter in self.policy.actor_parameters]
        rng_state = copy_module.deepcopy(self.actor_minibatch_rng.bit_generator.state)
        batched = self._dry_actor_gradient(False, rng_state)
        exact = self._dry_actor_gradient(True, rng_state)
        self.actor_minibatch_rng.bit_generator.state = rng_state
        parameters_unchanged = all(torch.equal(before, parameter.detach().cpu()) for before, parameter in zip(actor_state_before, self.policy.actor_parameters))
        comparison = self._compare_dry_gradients(batched["gradients"], exact["gradients"])
        expected_minibatches = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs // self.batch_size
        maximum_loss_difference = float(np.max(np.abs(np.asarray(batched["losses"], dtype=np.float64) - np.asarray(exact["losses"], dtype=np.float64))))
        criteria = {
            "gradient_cosine": comparison["cosine"] >= 0.999,
            "gradient_relative_l2": comparison["relative_l2_difference_over_exact"] <= 0.02,
            "clip_fraction_batched_zero": batched["clip_fraction"] == 0.0,
            "clip_fraction_exact_zero": exact["clip_fraction"] == 0.0,
            "mean_approximate_kl_batched": batched["mean_approximate_kl"] <= 1.0e-4,
            "mean_approximate_kl_exact": exact["mean_approximate_kl"] <= 1.0e-4,
            "minibatches_complete": batched["minibatches"] == exact["minibatches"] == expected_minibatches == 8,
            "valid_transition_counts_identical": batched["valid_counts"] == exact["valid_counts"] and batched["valid_total"] == exact["valid_total"] == self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs,
            "minibatch_identity_identical": batched["old_log_prob_sums"] == exact["old_log_prob_sums"],
            "parameters_unchanged": parameters_unchanged,
        }
        return {
            "verdict": "pass" if all(criteria.values()) else "fail",
            "criteria": criteria,
            "gradient_comparison": comparison,
            "maximum_abs_policy_loss_difference": maximum_loss_difference,
            "batched_policy_losses": batched["losses"],
            "exact_policy_losses": exact["losses"],
            "batched_minibatches": batched["minibatches"],
            "exact_minibatches": exact["minibatches"],
            "batched_valid_counts": batched["valid_counts"],
            "exact_valid_counts": exact["valid_counts"],
            "batched_clip_fraction": batched["clip_fraction"],
            "exact_clip_fraction": exact["clip_fraction"],
            "batched_mean_approximate_kl": batched["mean_approximate_kl"],
            "exact_mean_approximate_kl": exact["mean_approximate_kl"],
        }

    def train(self) -> None:
        self.policy.set_training_mode(True)
        if not self.warmup_completed:
            self._warmup_critic()
            return

        clip_range = self.clip_range(self._current_progress_remaining)
        policy_losses = []
        clip_fractions = []
        approximate_kls = []
        actor_grad_norms = []
        critic_grad_norms = []
        anchor_losses = []
        anchor_steering_losses = []
        anchor_speed_losses = []
        ppo_gradient_norms = []
        anchor_gradient_norms = []
        ppo_step_space_norms = []
        anchor_step_space_norms = []
        combined_gradient_norms = []
        clipped_gradient_norms = []
        update = self._n_updates + 1
        actor_optimizer_steps_planned = (
            self.actor_epochs
            * self.rollout_buffer.buffer_size
            * self.rollout_buffer.n_envs
            // self.batch_size
        )
        actor_optimizer_steps_completed = 0
        critic_input_stats = self._critic_input_statistics()
        exploration_stats = self._exploration_statistics()
        telemetry_rng_state = copy_module.deepcopy(self.telemetry_rng.bit_generator.state)
        value_statistics_pre_update = self._full_buffer_value_statistics()
        ratio_identity_stats = self._assert_full_buffer_ratio_identity()
        batched_replay_adjudication = None
        exact_actor_replay = bool(
            self.policy.speed_exploration_mode
            == PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE
        )
        anchor_loss_pre_update = None
        if self.collision_bc_anchor is not None:
            with torch.no_grad():
                anchor_loss_pre_update = float(self.collision_bc_anchor.loss()[0].detach().cpu().item())
        if bool(getattr(self.env, "prefix_reset_enabled", False)) and max(ratio_identity_stats["preupdate_max_abs_log_ratio"], ratio_identity_stats["preupdate_max_abs_ratio_minus_one"]) >= 0.02:
            if self.policy.speed_exploration_mode != PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE:
                raise RuntimeError("Prefix-reset batched replay exceeded the adjudicated 0.02 causal guardrail")
            if not bool(ratio_identity_stats["preupdate_exact_ratio_measured"]) or max(ratio_identity_stats["preupdate_exact_max_abs_log_ratio"], ratio_identity_stats["preupdate_exact_max_abs_ratio_minus_one"]) > 5e-5:
                raise RuntimeError("Prefix joint-temporal exact replay failed before batched adjudication")
            batched_replay_adjudication = {
                "verdict": "not_applicable_exact_actor_replay",
                "reason": "Formal prefix joint-temporal actor updates use the collection-equivalent replay path directly",
            }

        print(f"Formal update {update}: actor phase start", flush=True)
        for parameter in self.policy.critic_parameters:
            parameter.requires_grad_(False)
        for epoch in range(self.actor_epochs):
            for minibatch, rollout_data in enumerate(
                self.rollout_buffer.get(self.batch_size, rng=self.actor_minibatch_rng),
                start=1,
            ):
                mask = rollout_data.mask > 1e-8
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    valid_advantages = advantages[mask]
                    advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
                log_prob, _entropy = self.policy.evaluate_actor_actions(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                    collection_equivalent=exact_actor_replay,
                )
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approximate_kl = float(
                        ((torch.exp(log_ratio) - 1) - log_ratio)[mask].mean().cpu().item()
                    )
                    require_finite_number("Approximate KL", approximate_kl)
                    approximate_kls.append(approximate_kl)
                    clip_fractions.append(float((torch.abs(ratio - 1) > clip_range)[mask].float().mean().cpu().item()))
                policy_loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range))[mask].mean()
                require_finite_tensor("Policy loss", policy_loss)
                self.policy.actor_optimizer.zero_grad()
                if self.collision_bc_anchor is None:
                    policy_loss.backward()
                else:
                    anchor_loss, anchor_steering_loss, anchor_speed_loss = self.collision_bc_anchor.loss()
                    require_finite_tensor("Collision BC anchor loss", anchor_loss)
                    ppo_gradients = torch.autograd.grad(policy_loss, self.policy.actor_parameters, retain_graph=True)
                    anchor_gradients = torch.autograd.grad(anchor_loss, self.policy.actor_parameters, retain_graph=True)
                    ppo_gradient_norms.append(self._gradient_norm(ppo_gradients))
                    anchor_gradient_norms.append(self._gradient_norm(anchor_gradients))
                    ppo_step_space_norms.append(self._actor_step_space_norm(ppo_gradients))
                    anchor_step_space_norms.append(self._actor_step_space_norm(anchor_gradients))
                    combined_loss = policy_loss + self.collision_bc_anchor_beta * anchor_loss
                    require_finite_tensor("Combined PPO and collision BC anchor loss", combined_loss)
                    combined_loss.backward()
                    anchor_losses.append(float(anchor_loss.detach().cpu().item()))
                    anchor_steering_losses.append(float(anchor_steering_loss.detach().cpu().item()))
                    anchor_speed_losses.append(float(anchor_speed_loss.detach().cpu().item()))
                    combined_gradient_norms.append(self._actor_parameter_gradient_norm())
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters, MAX_GRAD_NORM)
                require_finite_tensor("Actor gradient norm", grad_norm)
                actor_grad_norms.append(float(grad_norm.detach().cpu().item()))
                if self.collision_bc_anchor is not None:
                    clipped_gradient_norms.append(self._actor_parameter_gradient_norm())
                self.policy.actor_optimizer.step()
                actor_optimizer_steps_completed += 1
                policy_losses.append(float(policy_loss.item()))
        for parameter in self.policy.critic_parameters:
            parameter.requires_grad_(True)
        print(f"Formal update {update}: actor phase complete", flush=True)

        print(f"Formal update {update}: critic phase start", flush=True)
        for parameter in self.policy.actor_parameters:
            parameter.requires_grad_(False)
        value_loss_samples: list[tuple[float, int]] = []
        critic_epoch_value_losses: list[float] = []
        for _epoch in range(self.critic_epochs):
            epoch_value_losses: list[tuple[float, int]] = []
            for rollout_data in self.rollout_buffer.get(self.batch_size, rng=self.critic_minibatch_rng):
                mask = rollout_data.mask > 1e-8
                values = self._batch_values(rollout_data)
                value_loss = torch.nn.functional.mse_loss(values[mask], rollout_data.returns[mask])
                self._apply_critic_gradient(VALUE_LOSS_COEFFICIENT * value_loss, "Value loss", critic_grad_norms)
                sample = (float(value_loss.item()), int(mask.sum().item()))
                epoch_value_losses.append(sample)
                value_loss_samples.append(sample)
            critic_epoch_value_losses.append(
                sum(loss * count for loss, count in epoch_value_losses)
                / sum(count for _loss, count in epoch_value_losses)
            )
        for parameter in self.policy.actor_parameters:
            parameter.requires_grad_(True)
        self.telemetry_rng.bit_generator.state = telemetry_rng_state
        value_statistics_post_update = self._full_buffer_value_statistics()
        anchor_loss_post_update = None
        if self.collision_bc_anchor is not None:
            with torch.no_grad():
                anchor_loss_post_update = float(self.collision_bc_anchor.loss()[0].detach().cpu().item())
        print(f"Formal update {update}: critic phase complete", flush=True)

        self._n_updates += 1
        policy_gradient_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
        value_loss_mean = sum(loss * count for loss, count in value_loss_samples) / sum(
            count for _loss, count in value_loss_samples
        )
        approximate_kl_mean = float(np.mean(approximate_kls))
        approximate_kl_max = float(np.max(approximate_kls))
        clip_fraction_mean = float(np.mean(clip_fractions))
        clip_fraction_max = float(np.max(clip_fractions))
        episodes = self._rollout_episode_records
        collision_times = [record["elapsed_time"] for record in episodes if record["episode_outcome"] == "ego_collision"]
        episode_metrics = self._episode_metrics(episodes)
        metrics = {
            "phase": "formal",
            "update": update,
            "critic_variant": self.policy.critic_variant,
            "rollout_speed_physical_std": self.rollout_speed_physical_std,
            "rollout_policy_update": update - 1,
            "checkpoint_update": update,
            "num_timesteps": self.num_timesteps,
            "total_collected_timesteps": self.num_timesteps,
            "formal_training_timesteps": update * self.n_envs * self.n_steps,
            "policy_gradient_loss": policy_gradient_loss,
            "value_loss": value_loss_mean,
            "critic_epoch_value_losses": critic_epoch_value_losses,
            "value_loss_pre_update": value_statistics_pre_update["value_loss"],
            "value_loss_post_update": value_statistics_post_update["value_loss"],
            "explained_variance_pre_update": value_statistics_pre_update["explained_variance"],
            "explained_variance_post_update": value_statistics_post_update["explained_variance"],
            "collision_value_loss_pre": value_statistics_pre_update["collision_value_loss"],
            "collision_value_loss_post": value_statistics_post_update["collision_value_loss"],
            "ordinary_value_loss_pre": value_statistics_pre_update["ordinary_value_loss"],
            "ordinary_value_loss_post": value_statistics_post_update["ordinary_value_loss"],
            "collision_explained_variance_pre": value_statistics_pre_update["collision_explained_variance"],
            "collision_explained_variance_post": value_statistics_post_update["collision_explained_variance"],
            "ordinary_explained_variance_pre": value_statistics_pre_update["ordinary_explained_variance"],
            "ordinary_explained_variance_post": value_statistics_post_update["ordinary_explained_variance"],
            "value_prediction_post_mean": value_statistics_post_update["value_prediction_mean"],
            "value_prediction_post_std": value_statistics_post_update["value_prediction_std"],
            "return_mean": value_statistics_post_update["return_mean"],
            "return_std": value_statistics_post_update["return_std"],
            **critic_input_stats,
            **exploration_stats,
            **self._prefix_reset_statistics(),
            **ratio_identity_stats,
            "preupdate_batched_replay_adjudication": batched_replay_adjudication,
            "actor_replay_mode": (
                "collection_equivalent"
                if exact_actor_replay
                else "batched"
            ),
            "approx_kl_mean": approximate_kl_mean,
            "approx_kl_max": approximate_kl_max,
            "clip_fraction_mean": clip_fraction_mean,
            "clip_fraction_max": clip_fraction_max,
            "actor_optimizer_steps_planned": actor_optimizer_steps_planned,
            "actor_optimizer_steps_completed": actor_optimizer_steps_completed,
            "actor_grad_norm_mean": float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0,
            "actor_grad_norm_max": float(np.max(actor_grad_norms)) if actor_grad_norms else 0.0,
            "critic_grad_norm_mean": float(np.mean(critic_grad_norms)),
            "critic_grad_norm_max": float(np.max(critic_grad_norms)),
            "ego_collision_count": sum(record["episode_outcome"] == "ego_collision" for record in episodes),
            "overtake_count": sum(record["episode_outcome"] == "overtake" for record in episodes),
            "follow_count": sum(record["episode_outcome"] == "follow" for record in episodes),
            "mean_ego_collision_time": float(np.mean(collision_times)) if collision_times else None,
            **episode_metrics,
        }
        if self.collision_bc_anchor is not None:
            metrics.update({
                "collision_bc_anchor_beta": self.collision_bc_anchor_beta,
                "collision_bc_anchor_episode_count": len(self.collision_bc_anchor.episodes),
                "collision_bc_anchor_loss_mean": float(np.mean(anchor_losses)),
                "collision_bc_anchor_steering_loss_mean": float(np.mean(anchor_steering_losses)),
                "collision_bc_anchor_speed_loss_mean": float(np.mean(anchor_speed_losses)),
                "collision_bc_anchor_loss_pre_update": anchor_loss_pre_update,
                "collision_bc_anchor_loss_post_update": anchor_loss_post_update,
                "collision_bc_anchor_functional_drift": anchor_loss_post_update - anchor_loss_pre_update,
                "collision_bc_anchor_ppo_gradient_norm_mean": float(np.mean(ppo_gradient_norms)),
                "collision_bc_anchor_gradient_norm_mean": float(np.mean(anchor_gradient_norms)),
                "collision_bc_anchor_ppo_step_space_norm_mean": float(np.mean(ppo_step_space_norms)),
                "collision_bc_anchor_step_space_norm_mean": float(np.mean(anchor_step_space_norms)),
                "collision_bc_anchor_combined_gradient_norm_mean": float(np.mean(combined_gradient_norms)),
                "collision_bc_anchor_clipped_gradient_norm_mean": float(np.mean(clipped_gradient_norms)),
            })
        actor_path, critic_path = self.recorder.save_formal_checkpoints(update, self.policy.actor_checkpoint_state_dict(), self.policy.value_net.state_dict())
        metrics["actor_checkpoint"] = str(actor_path)
        metrics["critic_checkpoint"] = str(critic_path)
        self.recorder.record_metrics(metrics)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_gradient_loss", policy_gradient_loss)
        self.logger.record("train/value_loss", value_loss_mean)
        self.logger.record("train/approx_kl", approximate_kl_mean)
        self.logger.record("train/clip_fraction", clip_fraction_mean)
        self.logger.record("train/actor_optimizer_steps", actor_optimizer_steps_completed)
        self.logger.record("train/explained_variance", value_statistics_post_update["explained_variance"])
        print(
            f"Formal update {update}: policy_gradient_loss={policy_gradient_loss:.6f}, value_loss={value_loss_mean:.6f}, "
            f"approx_kl={approximate_kl_mean:.6f}, clip_fraction={clip_fraction_mean:.6f}, "
            f"explained_variance_post={value_statistics_post_update['explained_variance']:.6f}",
            flush=True,
        )
        print(f"Formal update {update}: actor_checkpoint={actor_path}, critic_checkpoint={critic_path}", flush=True)

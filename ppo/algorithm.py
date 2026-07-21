"""End2Race recurrent PPO rollout storage and training algorithm."""

from __future__ import annotations

from collections.abc import Generator
import copy as copy_module
import time
from typing import Optional

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer, create_sequencers
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RecurrentRolloutBufferSamples, RNNStates
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import FloatSchedule, explained_variance
from stable_baselines3.common.vec_env import VecNormalize

from ppo.policy import END2RACE_OBSERVATION_SIZE, P20_CRITIC_VARIANTS
from ppo.privileged import PRIVILEGED_FEATURE_HIGHS, PRIVILEGED_FEATURE_LOWS
from ppo.training_records import TrainingRecorder, require_finite_number, require_finite_tensor


WARMUP_MAX_EPOCHS = 30
WARMUP_PATIENCE = 3
WARMUP_TRAIN_FRACTION = 0.8
VALUE_LOSS_COEFFICIENT = 0.5
MAX_GRAD_NORM = 0.5


class End2RaceRolloutBuffer(RecurrentRolloutBuffer):
    """Store the real actor GRU hidden state and materialize dummy states per batch.

    Optionally stores detached-GRU features or the independent-GRU hidden stream.
    """

    def __init__(self, *args, detached_gru_feature_size: int = 0, store_independent_gru_hidden: bool = False, **kwargs):
        self.detached_gru_feature_size = int(detached_gru_feature_size)
        self.store_independent_gru_hidden = bool(store_independent_gru_hidden)
        if self.detached_gru_feature_size < 0 or (self.detached_gru_feature_size and self.store_independent_gru_hidden):
            raise ValueError("Critic feature storage and critic hidden storage are mutually exclusive")
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        RolloutBuffer.reset(self)
        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        if self.store_independent_gru_hidden:
            self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        if self.detached_gru_feature_size:
            self.detached_gru_features = np.zeros((self.buffer_size, self.n_envs, self.detached_gru_feature_size), dtype=np.float32)
        self._staged_detached_gru_features: np.ndarray | None = None
        self.current_valid_by_timestep: tuple[tuple[bool, ...], ...] | None = None
        self.current_detached_gru_features: torch.Tensor | None = None
        self.current_collision_mask: torch.Tensor | None = None

    def stage_detached_gru_features(self, features: np.ndarray) -> None:
        """Receive the per-step detached critic features computed inside policy.forward."""

        if not self.detached_gru_feature_size:
            return
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (self.n_envs, self.detached_gru_feature_size):
            raise RuntimeError(f"Staged detached-GRU features must have shape {(self.n_envs, self.detached_gru_feature_size)}, got {features.shape}")
        self._staged_detached_gru_features = features

    def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
        self.hidden_states_pi[self.pos] = np.asarray(lstm_states.pi[0].cpu().numpy())
        if self.store_independent_gru_hidden:
            self.hidden_states_vf[self.pos] = np.asarray(lstm_states.vf[0].cpu().numpy())
        if self.detached_gru_feature_size:
            if self._staged_detached_gru_features is None:
                raise RuntimeError("Detached-GRU features were not staged before adding a transition")
            self.detached_gru_features[self.pos] = self._staged_detached_gru_features
            self._staged_detached_gru_features = None
        RolloutBuffer.add(self, *args, **kwargs)

    def get(self, batch_size: Optional[int] = None, *, rng: np.random.Generator) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        if not self.full:
            raise RuntimeError("Rollout buffer must be full before training")
        if not self.generator_ready:
            self.hidden_states_pi = self.hidden_states_pi.swapaxes(1, 2)
            names = ["observations", "actions", "values", "log_probs", "advantages", "returns", "hidden_states_pi", "episode_starts"]
            if self.store_independent_gru_hidden:
                self.hidden_states_vf = self.hidden_states_vf.swapaxes(1, 2)
                names.append("hidden_states_vf")
            if self.detached_gru_feature_size:
                names.append("detached_gru_features")
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
        if self.detached_gru_feature_size:
            self.current_detached_gru_features = self.to_torch(self.pad(self.detached_gru_features[batch_inds])).reshape(padded_batch_size, self.detached_gru_feature_size)
        else:
            self.current_detached_gru_features = None
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
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )


class End2RaceRecurrentPPO(RecurrentPPO):
    """Run critic warm-up, then separate actor and critic PPO phases."""

    def __init__(self, *args, actor_epochs: int, critic_epochs: int, recorder: TrainingRecorder, **kwargs):
        self.actor_epochs = actor_epochs
        self.critic_epochs = critic_epochs
        self.recorder = recorder
        self.warmup_completed = False
        self.rollout_index = 0
        self.current_phase = "warmup"
        self.rollout_for_update = 0
        self.rollout_policy_update = 0
        self.rollout_wall_seconds = 0.0
        self._rollout_episode_records: list[dict] = []
        kwargs["n_epochs"] = actor_epochs
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs).to(self.device)
        if not isinstance(self.policy, RecurrentActorCriticPolicy) or not self.policy.supports_end2race_rollout_buffer():
            raise TypeError("End2Race PPO requires the End2Race GRU policy")

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
        self.rollout_buffer = End2RaceRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            detached_gru_feature_size=self.policy.detached_gru_feature_size,
            store_independent_gru_hidden=self.policy.critic_is_independent_gru,
        )
        self.policy._end2race_rollout_buffer = self.rollout_buffer
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

        action_seed = int(np.random.SeedSequence([self.seed, 3]).generate_state(1)[0])
        torch.manual_seed(action_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(action_seed)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        self.rollout_index += 1
        self.current_phase = "warmup" if not self.warmup_completed else "formal"
        self.rollout_for_update = 0 if not self.warmup_completed else self._n_updates + 1
        self.rollout_policy_update = self._n_updates
        self._rollout_episode_records = []
        print(
            f"Rollout {self.rollout_index} start: phase={self.current_phase}, "
            f"rollout_policy_update={self.rollout_policy_update}, rollout_for_update={self.rollout_for_update}",
            flush=True,
        )
        started_at = time.perf_counter()
        completed = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        self.rollout_wall_seconds = time.perf_counter() - started_at
        print(f"Rollout {self.rollout_index} complete: {self.rollout_wall_seconds:.2f}s", flush=True)
        return completed

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
                "env_role": str(info["env_role"]),
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
        if self.policy.detached_gru_feature_size:
            return self.rollout_buffer.detached_gru_features.reshape(-1, self.policy.detached_gru_feature_size)
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
            episode_starts[slot, :length] = buffer.episode_starts[start:end, env_index]
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
        train_started_at = time.perf_counter()
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
        train_wall_seconds = time.perf_counter() - train_started_at
        metrics = {
            "phase": "warmup",
            "critic_variant": self.policy.critic_variant,
            "epochs": epoch + 1,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "warmup_train_losses": warmup_train_losses,
            "warmup_validation_losses": warmup_validation_losses,
            "critic_grad_norm_mean": float(np.mean(critic_grad_norms)),
            "critic_grad_norm_max": float(np.max(critic_grad_norms)),
            "rollout_wall_seconds": self.rollout_wall_seconds,
            "train_wall_seconds": train_wall_seconds,
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
        if self.policy.detached_gru_feature_size:
            features = self.rollout_buffer.current_detached_gru_features
            if features is None:
                raise RuntimeError("Stored critic features are missing for the current minibatch")
            return self.policy.evaluate_values(features).flatten()
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
        """Per-rollout critic input telemetry for the stored-feature and privileged variants."""

        if self.policy.detached_gru_feature_size:
            features = self.rollout_buffer.detached_gru_features
            return {
                "detached_gru_feature_mean": float(features.mean()),
                "detached_gru_feature_std": float(features.std()),
                "detached_gru_feature_abs_max": float(np.abs(features).max()),
            }
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
        update = self._n_updates + 1
        critic_input_stats = self._critic_input_statistics()
        telemetry_rng_state = copy_module.deepcopy(self.telemetry_rng.bit_generator.state)
        value_statistics_pre_update = self._full_buffer_value_statistics()

        print(f"Formal update {update}: actor phase start", flush=True)
        actor_started_at = time.perf_counter()
        for parameter in self.policy.critic_parameters:
            parameter.requires_grad_(False)
        for _epoch in range(self.actor_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size, rng=self.actor_minibatch_rng):
                mask = rollout_data.mask > 1e-8
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    valid_advantages = advantages[mask]
                    advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
                log_prob, _entropy = self.policy.evaluate_actor_actions(rollout_data.observations, rollout_data.actions, rollout_data.lstm_states, rollout_data.episode_starts)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range))[mask].mean()
                require_finite_tensor("Policy loss", policy_loss)
                self.policy.actor_optimizer.zero_grad()
                policy_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters, MAX_GRAD_NORM)
                require_finite_tensor("Actor gradient norm", grad_norm)
                actor_grad_norms.append(float(grad_norm.detach().cpu().item()))
                self.policy.actor_optimizer.step()
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approximate_kls.append(float(((torch.exp(log_ratio) - 1) - log_ratio)[mask].mean().cpu().item()))
                    clip_fractions.append(float((torch.abs(ratio - 1) > clip_range)[mask].float().mean().cpu().item()))
                policy_losses.append(float(policy_loss.item()))
        for parameter in self.policy.critic_parameters:
            parameter.requires_grad_(True)
        actor_train_wall_seconds = time.perf_counter() - actor_started_at
        print(f"Formal update {update}: actor phase complete in {actor_train_wall_seconds:.2f}s", flush=True)

        print(f"Formal update {update}: critic phase start", flush=True)
        critic_started_at = time.perf_counter()
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
        critic_train_wall_seconds = time.perf_counter() - critic_started_at
        print(f"Formal update {update}: critic phase complete in {critic_train_wall_seconds:.2f}s", flush=True)

        self._n_updates += 1
        policy_gradient_loss = float(np.mean(policy_losses))
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
            "approx_kl_mean": approximate_kl_mean,
            "approx_kl_max": approximate_kl_max,
            "clip_fraction_mean": clip_fraction_mean,
            "clip_fraction_max": clip_fraction_max,
            "actor_grad_norm_mean": float(np.mean(actor_grad_norms)),
            "actor_grad_norm_max": float(np.max(actor_grad_norms)),
            "critic_grad_norm_mean": float(np.mean(critic_grad_norms)),
            "critic_grad_norm_max": float(np.max(critic_grad_norms)),
            "rollout_wall_seconds": self.rollout_wall_seconds,
            "actor_train_wall_seconds": actor_train_wall_seconds,
            "critic_train_wall_seconds": critic_train_wall_seconds,
            "ego_collision_count": sum(record["episode_outcome"] == "ego_collision" for record in episodes),
            "overtake_count": sum(record["episode_outcome"] == "overtake" for record in episodes),
            "follow_count": sum(record["episode_outcome"] == "follow" for record in episodes),
            "mean_ego_collision_time": float(np.mean(collision_times)) if collision_times else None,
            **episode_metrics,
        }
        actor_path, critic_path = self.recorder.save_formal_checkpoints(update, self.policy.actor_checkpoint_state_dict(), self.policy.value_net.state_dict())
        metrics["actor_checkpoint"] = str(actor_path)
        metrics["critic_checkpoint"] = str(critic_path)
        self.recorder.record_metrics(metrics)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_gradient_loss", policy_gradient_loss)
        self.logger.record("train/value_loss", value_loss_mean)
        self.logger.record("train/approx_kl", approximate_kl_mean)
        self.logger.record("train/clip_fraction", clip_fraction_mean)
        self.logger.record("train/explained_variance", value_statistics_post_update["explained_variance"])
        print(
            f"Formal update {update}: policy_gradient_loss={policy_gradient_loss:.6f}, value_loss={value_loss_mean:.6f}, "
            f"approx_kl={approximate_kl_mean:.6f}, clip_fraction={clip_fraction_mean:.6f}, "
            f"explained_variance_post={value_statistics_post_update['explained_variance']:.6f}",
            flush=True,
        )
        print(f"Formal update {update}: actor_checkpoint={actor_path}, critic_checkpoint={critic_path}", flush=True)

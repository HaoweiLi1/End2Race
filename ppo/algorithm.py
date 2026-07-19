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

from ppo.training_records import TrainingRecorder, require_finite_number, require_finite_tensor


WARMUP_MAX_EPOCHS = 20
WARMUP_PATIENCE = 3
WARMUP_TRAIN_FRACTION = 0.8
VALUE_LOSS_COEFFICIENT = 0.5
MAX_GRAD_NORM = 0.5


class ActorHiddenRolloutBuffer(RecurrentRolloutBuffer):
    """Store the real actor GRU hidden state and materialize dummy states per batch."""

    def reset(self) -> None:
        RolloutBuffer.reset(self)
        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.current_valid_by_timestep: tuple[tuple[bool, ...], ...] | None = None

    def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
        self.hidden_states_pi[self.pos] = np.asarray(lstm_states.pi[0].cpu().numpy())
        RolloutBuffer.add(self, *args, **kwargs)

    def get(self, batch_size: Optional[int] = None, *, rng: np.random.Generator) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        if not self.full:
            raise RuntimeError("Rollout buffer must be full before training")
        if not self.generator_ready:
            self.hidden_states_pi = self.hidden_states_pi.swapaxes(1, 2)
            for name in ("observations", "actions", "values", "log_probs", "advantages", "returns", "hidden_states_pi", "episode_starts"):
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
        actor_hidden = self.to_torch(self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1)).contiguous()
        actor_cell = torch.zeros_like(actor_hidden)
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
        if not isinstance(self.policy, RecurrentActorCriticPolicy) or not self.policy.supports_actor_hidden_only_buffer():
            raise TypeError("End2Race PPO requires the actor-hidden-only recurrent policy")

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
        self.rollout_buffer = ActorHiddenRolloutBuffer(self.n_steps, self.observation_space, self.action_space, hidden_buffer_shape, self.device, gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=self.n_envs)
        self.policy._actor_hidden_rollout_buffer = self.rollout_buffer
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
                "termination_reason": str(info["termination_reason"]),
                "timeout": bool(info["timeout"]),
                "opponent_collision": bool(info["opponent_collision"]),
            }
            self.recorder.record_episode(record)
            self._rollout_episode_records.append(record)

    def _warmup_split(self) -> tuple[np.ndarray, np.ndarray]:
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

        train_indices: list[int] = []
        validation_indices: list[int] = []
        for role in ("collision", "ordinary"):
            role_sequences = sequences[role]
            if len(role_sequences) < 2:
                raise RuntimeError(f"Warm-up requires at least two {role} recurrent sequences")
            order = self.warmup_split_rng.permutation(len(role_sequences))
            train_count = min(max(int(len(order) * WARMUP_TRAIN_FRACTION), 1), len(order) - 1)
            for destination, selected in ((train_indices, order[:train_count]), (validation_indices, order[train_count:])):
                for sequence_index in selected:
                    env_index, start, end = role_sequences[int(sequence_index)]
                    destination.extend(np.arange(start, end) * self.n_envs + env_index)
        return np.asarray(train_indices, dtype=np.int64), np.asarray(validation_indices, dtype=np.int64)

    def _critic_batch_loss(self, flat_observations: np.ndarray, flat_returns: np.ndarray, indices: np.ndarray) -> torch.Tensor:
        observations = torch.as_tensor(flat_observations[indices], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(flat_returns[indices], dtype=torch.float32, device=self.device)
        return torch.nn.functional.mse_loss(self.policy.evaluate_values(observations).flatten(), returns)

    def _validation_loss(self, flat_observations: np.ndarray, flat_returns: np.ndarray, indices: np.ndarray) -> float:
        losses = []
        with torch.no_grad():
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                losses.append((float(self._critic_batch_loss(flat_observations, flat_returns, batch).item()), len(batch)))
        return sum(loss * count for loss, count in losses) / sum(count for _loss, count in losses)

    def _warmup_critic(self) -> None:
        train_started_at = time.perf_counter()
        observations = self.rollout_buffer.observations.reshape(-1, *self.rollout_buffer.obs_shape)
        returns = self.rollout_buffer.returns.reshape(-1)
        train_indices, validation_indices = self._warmup_split()
        best_loss = float("inf")
        best_critic = None
        best_optimizer = None
        stale_epochs = 0
        critic_grad_norms = []
        for epoch in range(WARMUP_MAX_EPOCHS):
            shuffled = self.warmup_shuffle_rng.permutation(train_indices)
            for start in range(0, len(shuffled), self.batch_size):
                loss = VALUE_LOSS_COEFFICIENT * self._critic_batch_loss(observations, returns, shuffled[start : start + self.batch_size])
                require_finite_tensor("Warm-up loss", loss)
                self.policy.critic_optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters, MAX_GRAD_NORM)
                require_finite_tensor("Warm-up critic gradient norm", grad_norm)
                critic_grad_norms.append(float(grad_norm.detach().cpu().item()))
                self.policy.critic_optimizer.step()
            validation_loss = self._validation_loss(observations, returns, validation_indices)
            require_finite_number("Warm-up validation loss", validation_loss)
            print(f"Warm-up epoch {epoch + 1}: validation_loss={validation_loss:.6f}", flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_critic = copy_module.deepcopy(self.policy.value_net.state_dict())
                best_optimizer = copy_module.deepcopy(self.policy.critic_optimizer.state_dict())
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
            "epochs": epoch + 1,
            "best_validation_loss": best_loss,
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

    def train(self) -> None:
        self.policy.set_training_mode(True)
        if not self.warmup_completed:
            self._warmup_critic()
            return

        clip_range = self.clip_range(self._current_progress_remaining)
        policy_losses = []
        value_losses = []
        clip_fractions = []
        approximate_kls = []
        actor_grad_norms = []
        critic_grad_norms = []
        update = self._n_updates + 1

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
        for _epoch in range(self.critic_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size, rng=self.critic_minibatch_rng):
                mask = rollout_data.mask > 1e-8
                values = self.policy.evaluate_values(rollout_data.observations).flatten()
                value_loss = torch.nn.functional.mse_loss(values[mask], rollout_data.returns[mask])
                require_finite_tensor("Value loss", value_loss)
                self.policy.critic_optimizer.zero_grad()
                (VALUE_LOSS_COEFFICIENT * value_loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters, MAX_GRAD_NORM)
                require_finite_tensor("Critic gradient norm", grad_norm)
                critic_grad_norms.append(float(grad_norm.detach().cpu().item()))
                self.policy.critic_optimizer.step()
                value_losses.append(float(value_loss.item()))
        for parameter in self.policy.actor_parameters:
            parameter.requires_grad_(True)
        critic_train_wall_seconds = time.perf_counter() - critic_started_at
        print(f"Formal update {update}: critic phase complete in {critic_train_wall_seconds:.2f}s", flush=True)

        self._n_updates += 1
        policy_gradient_loss = float(np.mean(policy_losses))
        value_loss_mean = float(np.mean(value_losses))
        approximate_kl = float(np.mean(approximate_kls))
        clip_fraction = float(np.mean(clip_fractions))
        explained_variance_value = float(explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))
        episodes = self._rollout_episode_records
        collision_times = [record["elapsed_time"] for record in episodes if record["episode_outcome"] == "ego_collision"]
        metrics = {
            "phase": "formal",
            "update": update,
            "rollout_policy_update": update - 1,
            "checkpoint_update": update,
            "num_timesteps": self.num_timesteps,
            "total_collected_timesteps": self.num_timesteps,
            "formal_training_timesteps": update * self.n_envs * self.n_steps,
            "policy_gradient_loss": policy_gradient_loss,
            "value_loss": value_loss_mean,
            "approx_kl": approximate_kl,
            "clip_fraction": clip_fraction,
            "explained_variance": explained_variance_value,
            "actor_grad_norm_mean": float(np.mean(actor_grad_norms)),
            "actor_grad_norm_max": float(np.max(actor_grad_norms)),
            "critic_grad_norm_mean": float(np.mean(critic_grad_norms)),
            "critic_grad_norm_max": float(np.max(critic_grad_norms)),
            "rollout_wall_seconds": self.rollout_wall_seconds,
            "actor_train_wall_seconds": actor_train_wall_seconds,
            "critic_train_wall_seconds": critic_train_wall_seconds,
            "collision_episode_count": sum(record["env_role"] == "collision" for record in episodes),
            "ordinary_episode_count": sum(record["env_role"] == "ordinary" for record in episodes),
            "ego_collision_count": sum(record["episode_outcome"] == "ego_collision" for record in episodes),
            "overtake_count": sum(record["episode_outcome"] == "overtake" for record in episodes),
            "follow_count": sum(record["episode_outcome"] == "follow" for record in episodes),
            "mean_episode_return": float(np.mean([record["episode_return"] for record in episodes])) if episodes else 0.0,
            "mean_collision_time": float(np.mean(collision_times)) if collision_times else 0.0,
        }
        actor_path, critic_path = self.recorder.save_formal_checkpoints(update, self.policy.actor_checkpoint_state_dict(), self.policy.value_net.state_dict())
        metrics["actor_checkpoint"] = str(actor_path)
        metrics["critic_checkpoint"] = str(critic_path)
        self.recorder.record_metrics(metrics)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_gradient_loss", policy_gradient_loss)
        self.logger.record("train/value_loss", value_loss_mean)
        self.logger.record("train/approx_kl", approximate_kl)
        self.logger.record("train/clip_fraction", clip_fraction)
        self.logger.record("train/explained_variance", explained_variance_value)
        print(
            f"Formal update {update}: policy_gradient_loss={policy_gradient_loss:.6f}, value_loss={value_loss_mean:.6f}, "
            f"approx_kl={approximate_kl:.6f}, clip_fraction={clip_fraction:.6f}, explained_variance={explained_variance_value:.6f}",
            flush=True,
        )
        print(f"Formal update {update}: actor_checkpoint={actor_path}, critic_checkpoint={critic_path}", flush=True)

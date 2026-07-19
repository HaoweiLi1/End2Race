#!/usr/bin/env python3
"""Train the fixed End2Race PPO pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
import copy as copy_module
import multiprocessing as mp
from pathlib import Path
import random
import time
from typing import Optional

import numpy as np
import torch
import yaml
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer, create_sequencers
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RecurrentRolloutBufferSamples, RNNStates
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import FloatSchedule, explained_variance
from stable_baselines3.common.vec_env import VecNormalize

from model import End2Race
from ppo.environment import EXTERNAL_RESET_OPTION, make_environment
from ppo.policy import END2RACE_LIDAR_SIZE, SPEED_PHYSICAL_STD, STEERING_BOUND, STEERING_LATENT_STD, End2RaceGRUPolicy
from ppo.scenarios import ScenarioSpec, collision_cache_exists, collision_classification_config, expanded_scenarios, load_collision_cache, ordinary_scenarios, write_collision_cache
from ppo.training_records import TrainingRecorder, require_finite_number, require_finite_tensor
from ppo.vec_env import CentralScheduleSubprocVecEnv, _limit_worker_threads


WARMUP_MAX_EPOCHS = 20
WARMUP_PATIENCE = 3
WARMUP_TRAIN_FRACTION = 0.8
VALUE_LOSS_COEFFICIENT = 0.5
MAX_GRAD_NORM = 0.5
PPO_CONFIG_PATH = Path(__file__).resolve().parent / "ppo" / "ppo_config.yaml"
with PPO_CONFIG_PATH.open("r", encoding="utf-8") as file:
    PPO_CONFIG = yaml.safe_load(file)
START_METHOD = str(PPO_CONFIG["start_method"])


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train End2Race PPO")

    # Model paths
    parser.add_argument("--pretrained_model_path", type=str, default="pretrained/end2race.pth")
    parser.add_argument("--output_dir", type=str, default="post-trained/ppo_run")

    # Model configuration
    parser.add_argument("--hidden_scale", type=int, default=4)

    # Environment configuration
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--env_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--collision_cache_dir", type=str, default="post-trained/collision-cache/default")
    parser.add_argument("--reclassify_collisions", action="store_true")

    # Rollout configuration
    parser.add_argument("--n_steps", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=12800)
    parser.add_argument("--num_updates", type=int, default=20)

    # Training configuration
    parser.add_argument("--actor_epochs", type=int, default=3)
    parser.add_argument("--critic_epochs", type=int, default=8)
    parser.add_argument("--gru_learning_rate", type=float, default=1.0e-6)
    parser.add_argument("--head_learning_rate", type=float, default=1.0e-5)
    parser.add_argument("--critic_learning_rate", type=float, default=3.0e-4)

    # PPO configuration
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.995)
    parser.add_argument("--clip_range", type=float, default=0.10)
    return parser.parse_args()


def configure_training_numerics() -> None:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False


_COLLISION_ENV = None
_COLLISION_ACTOR = None


def _collision_worker_init(pretrained_model_path: str, hidden_scale: int, map_name: str) -> None:
    global _COLLISION_ENV, _COLLISION_ACTOR
    _limit_worker_threads()
    _COLLISION_ENV = make_environment(0, map_name)()
    _COLLISION_ACTOR = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)
    _COLLISION_ACTOR.load_state_dict(torch.load(pretrained_model_path, map_location="cpu", weights_only=True), strict=True)
    _COLLISION_ACTOR.eval()


def _classify_collision_candidate(task: tuple[int, ScenarioSpec]) -> tuple[int, str]:
    index, scenario = task
    if _COLLISION_ENV is None or _COLLISION_ACTOR is None:
        raise RuntimeError("Collision classification worker is not initialized")
    try:
        observation, _info = _COLLISION_ENV.reset(options={EXTERNAL_RESET_OPTION: scenario.to_reset_spec("collision")})
        raw = _COLLISION_ENV._raw_observation
        finite = np.isfinite(observation).all() and all(np.isfinite(np.asarray(value)).all() for value in raw.values() if isinstance(value, (list, tuple, np.ndarray)))
        if not finite or np.asarray(raw["collisions"], dtype=bool).any():
            return index, "invalid"
        hidden = None
        while True:
            actor_observation = torch.as_tensor(observation, dtype=torch.float32)
            with torch.no_grad():
                actions, hidden = _COLLISION_ACTOR(actor_observation[:END2RACE_LIDAR_SIZE].reshape(1, 1, -1), actor_observation[END2RACE_LIDAR_SIZE:].reshape(1, 1, 1), hidden)
            action = actions[0, -1].numpy().copy()
            action[0] = np.clip(action[0], -STEERING_BOUND, STEERING_BOUND)
            if not np.isfinite(action).all():
                raise RuntimeError("actor produced a non-finite action")
            observation, _reward, terminated, truncated, info = _COLLISION_ENV.step(action)
            if terminated or truncated:
                return index, "ego_collision" if info["ego_collision"] else "other"
    except Exception as error:
        raise RuntimeError(f"Collision classification failed for {scenario.scenario_id}") from error


def classify_collision_scenarios(
    pretrained_model_path: str | Path,
    hidden_scale: int,
    map_name: str,
    env_workers: int,
    candidates: tuple[ScenarioSpec, ...],
) -> tuple[tuple[ScenarioSpec, ...], list[dict], dict]:
    candidate_count = len(candidates)
    context = mp.get_context(START_METHOD)
    collisions = []
    outcomes = []
    collision_count = 0
    invalid_count = 0
    started_at = time.perf_counter()
    with ProcessPoolExecutor(max_workers=env_workers, mp_context=context, initializer=_collision_worker_init, initargs=(str(Path(pretrained_model_path).expanduser().resolve()), hidden_scale, map_name)) as executor:
        for completed, (index, outcome) in enumerate(executor.map(_classify_collision_candidate, enumerate(candidates), chunksize=4), start=1):
            if index != completed - 1 or outcome not in {"ego_collision", "other", "invalid"}:
                raise RuntimeError(f"Invalid classification result at candidate {completed - 1}/{candidate_count}")
            outcomes.append({"candidate_index": index, "scenario_id": candidates[index].scenario_id, "outcome": outcome})
            if outcome == "ego_collision":
                collisions.append(candidates[index])
                collision_count += 1
            elif outcome == "invalid":
                invalid_count += 1
            if completed % 100 == 0 or completed == candidate_count:
                elapsed = time.perf_counter() - started_at
                rate = completed / elapsed
                eta = (candidate_count - completed) / rate
                print(f"Collision classification: {completed}/{candidate_count}, collision={collision_count}, invalid={invalid_count}, rate={rate:.2f}/s, ETA={eta:.1f}s", flush=True)
    if not collisions:
        raise RuntimeError(f"The pretrained model produced no ego-collision scenarios from {candidate_count} candidates")
    wall_seconds = time.perf_counter() - started_at
    summary = {
        "candidate_count": candidate_count,
        "collision_count": collision_count,
        "other_count": candidate_count - collision_count - invalid_count,
        "invalid_count": invalid_count,
        "env_workers": env_workers,
        "wall_seconds": wall_seconds,
        "scenarios_per_second": candidate_count / wall_seconds,
    }
    return tuple(collisions), outcomes, summary


def resolve_collision_scenarios(args, candidates: tuple[ScenarioSpec, ...]) -> tuple[tuple[ScenarioSpec, ...], bool, bool]:
    candidate_count = len(candidates)
    if candidate_count == 0:
        raise RuntimeError("Collision candidate set is empty")
    cache_dir = Path(args.collision_cache_dir).expanduser().resolve()
    current_config = collision_classification_config(args, candidate_count)
    if args.reclassify_collisions:
        print(f"Rebuilding collision classification cache for {candidate_count} candidates", flush=True)
    elif collision_cache_exists(cache_dir):
        collision_scenarios = load_collision_cache(cache_dir, current_config, candidates)
        print(f"Collision cache hit: loaded {len(collision_scenarios)} collision scenarios from {candidate_count} candidates", flush=True)
        return collision_scenarios, True, False
    else:
        print(f"Collision cache miss: classifying {candidate_count} candidates", flush=True)
    collision_scenarios, outcomes, summary = classify_collision_scenarios(args.pretrained_model_path, args.hidden_scale, args.map_name, args.env_workers, candidates)
    write_collision_cache(cache_dir, current_config, outcomes, collision_scenarios, summary)
    return collision_scenarios, False, bool(args.reclassify_collisions)


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
        self.current_formal_update = 0
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
        self.current_formal_update = 0 if not self.warmup_completed else self._n_updates + 1
        self._rollout_episode_records = []
        print(f"Rollout {self.rollout_index} start: phase={self.current_phase}, formal_update={self.current_formal_update}", flush=True)
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
                "formal_update": self.current_formal_update,
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
        self.recorder.record_metrics(metrics)
        checkpoint_path = self.recorder.save_warmup_critic(self.policy.value_net.state_dict())
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
            "update": update,
            "num_timesteps": self.num_timesteps,
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
        self.recorder.record_metrics(metrics)
        actor_path, critic_path = self.recorder.save_formal_checkpoints(update, self.policy.actor_checkpoint_state_dict(), self.policy.value_net.state_dict())
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


def validate_arguments(args) -> None:
    pretrained_path = Path(args.pretrained_model_path).expanduser().resolve()
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"Pretrained model does not exist: {pretrained_path}")
    if not args.output_dir.strip():
        raise ValueError("output_dir must not be empty")
    if not args.map_name.strip():
        raise ValueError("map_name must not be empty")
    if not args.collision_cache_dir.strip():
        raise ValueError("collision_cache_dir must not be empty")
    if args.env_workers <= 0 or args.n_envs < args.env_workers or args.n_envs % 2 != 0:
        raise ValueError("n_envs must be even and at least env_workers, and env_workers must be positive")
    for name in ("hidden_scale", "n_steps", "batch_size", "num_updates", "actor_epochs", "critic_epochs"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.n_envs * args.n_steps % args.batch_size != 0:
        raise ValueError("n_envs * n_steps must be divisible by batch_size")
    if args.batch_size % (2 * args.n_steps) != 0:
        raise ValueError("batch_size must be divisible by 2 * n_steps so each env-major recurrent minibatch has equal collision and ordinary transitions")
    for name in ("gru_learning_rate", "head_learning_rate", "critic_learning_rate", "gamma", "gae_lambda", "clip_range"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")


def build_model(vector_env, args, device, recorder: TrainingRecorder) -> End2RaceRecurrentPPO:
    return End2RaceRecurrentPPO(
        End2RaceGRUPolicy,
        vector_env,
        actor_epochs=args.actor_epochs,
        critic_epochs=args.critic_epochs,
        recorder=recorder,
        learning_rate=1.0,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=VALUE_LOSS_COEFFICIENT,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=None,
        seed=args.seed,
        device=device,
        policy_kwargs={
            "checkpoint_path": args.pretrained_model_path,
            "hidden_scale": args.hidden_scale,
            "gru_learning_rate": args.gru_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "critic_learning_rate": args.critic_learning_rate,
        },
        verbose=1,
    )


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    configure_training_numerics()
    random.seed(args.seed)
    np.random.seed(args.seed)

    recorder = TrainingRecorder(args.output_dir, args.hidden_scale)
    print(
        f"PPO training configuration: output_dir={recorder.output_dir}, pretrained_model_path={Path(args.pretrained_model_path).expanduser().resolve()}, "
        f"map={args.map_name}, n_envs={args.n_envs}, env_workers={args.env_workers}, n_steps={args.n_steps}, "
        f"batch_size={args.batch_size}, num_updates={args.num_updates}, seed={args.seed}",
        flush=True,
    )
    print("[1/5] Building collision candidates", flush=True)
    candidates = expanded_scenarios(args.map_name)
    candidate_count = len(candidates)
    print("[2/5] Loading or classifying collision pool", flush=True)
    collision_scenarios, cache_hit, reclassified = resolve_collision_scenarios(args, candidates)
    print("[3/5] Building ordinary scenarios", flush=True)
    ordinary_scenario_set = ordinary_scenarios(args.map_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    recorder.write_run_config(
        args,
        PPO_CONFIG,
        device,
        {
            "WARMUP_MAX_EPOCHS": WARMUP_MAX_EPOCHS,
            "WARMUP_PATIENCE": WARMUP_PATIENCE,
            "WARMUP_TRAIN_FRACTION": WARMUP_TRAIN_FRACTION,
            "VALUE_LOSS_COEFFICIENT": VALUE_LOSS_COEFFICIENT,
            "MAX_GRAD_NORM": MAX_GRAD_NORM,
            "STEERING_LATENT_STD": STEERING_LATENT_STD,
            "SPEED_PHYSICAL_STD": SPEED_PHYSICAL_STD,
        },
    )
    recorder.write_scenario_pools(
        collision_scenarios,
        ordinary_scenario_set,
        {
            "cache_dir": str(Path(args.collision_cache_dir).expanduser().resolve()),
            "cache_hit": cache_hit,
            "reclassified": reclassified,
            "candidate_count": candidate_count,
            "collision_count": len(collision_scenarios),
        },
    )
    print("[4/5] Creating vector environments", flush=True)
    vector_env = CentralScheduleSubprocVecEnv(args.n_envs, args.env_workers, START_METHOD, args.seed, args.map_name, collision_scenarios, ordinary_scenario_set)
    try:
        print("[5/5] Building PPO model", flush=True)
        model = build_model(vector_env, args, device, recorder)
        total_rollouts = args.num_updates + 1
        model.learn(total_timesteps=args.n_envs * args.n_steps * total_rollouts, log_interval=1, progress_bar=False)
        final_actor_path = recorder.save_final_actor(model.policy.actor_checkpoint_state_dict())
        print(f"PPO final actor saved: {final_actor_path}", flush=True)
    finally:
        vector_env.close()


if __name__ == "__main__":
    main()

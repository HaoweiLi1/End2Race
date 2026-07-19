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
from ppo.policy import END2RACE_LIDAR_SIZE, STEERING_BOUND, End2RaceGRUPolicy
from ppo.scenarios import ScenarioSpec, expanded_scenarios, ordinary_scenarios
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
    parser.add_argument("--ppo_model_path", type=str, default="end2race_ppo.pth")

    # Model configuration
    parser.add_argument("--hidden_scale", type=int, default=4)

    # Environment configuration
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--env_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

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


def classify_collision_scenarios(pretrained_model_path: str | Path, hidden_scale: int, map_name: str, env_workers: int) -> tuple[ScenarioSpec, ...]:
    candidates = expanded_scenarios(map_name)
    context = mp.get_context(START_METHOD)
    collisions = []
    invalid_count = 0
    with ProcessPoolExecutor(max_workers=env_workers, mp_context=context, initializer=_collision_worker_init, initargs=(str(Path(pretrained_model_path).expanduser().resolve()), hidden_scale, map_name)) as executor:
        for completed, (index, outcome) in enumerate(executor.map(_classify_collision_candidate, enumerate(candidates), chunksize=4), start=1):
            if outcome == "ego_collision":
                collisions.append(candidates[index])
            elif outcome == "invalid":
                invalid_count += 1
            if completed % 100 == 0 or completed == len(candidates):
                print(f"Collision classification: {completed}/{len(candidates)}")
    if not collisions:
        raise RuntimeError("The pretrained model produced no ego-collision scenarios")
    print(f"Collision classification complete: {len(collisions)} collisions, {invalid_count} invalid")
    return tuple(collisions)


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

    def __init__(self, *args, actor_epochs: int, critic_epochs: int, **kwargs):
        self.actor_epochs = actor_epochs
        self.critic_epochs = critic_epochs
        self.warmup_completed = False
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
        observations = self.rollout_buffer.observations.reshape(-1, *self.rollout_buffer.obs_shape)
        returns = self.rollout_buffer.returns.reshape(-1)
        train_indices, validation_indices = self._warmup_split()
        best_loss = float("inf")
        best_critic = None
        best_optimizer = None
        stale_epochs = 0
        for epoch in range(WARMUP_MAX_EPOCHS):
            shuffled = self.warmup_shuffle_rng.permutation(train_indices)
            for start in range(0, len(shuffled), self.batch_size):
                loss = VALUE_LOSS_COEFFICIENT * self._critic_batch_loss(observations, returns, shuffled[start : start + self.batch_size])
                self.policy.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters, MAX_GRAD_NORM)
                self.policy.critic_optimizer.step()
            validation_loss = self._validation_loss(observations, returns, validation_indices)
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
        self.logger.record("warmup/epochs", epoch + 1)
        self.logger.record("warmup/best_validation_loss", best_loss)

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
                self.policy.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters, MAX_GRAD_NORM)
                self.policy.actor_optimizer.step()
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approximate_kls.append(float(((torch.exp(log_ratio) - 1) - log_ratio)[mask].mean().cpu().item()))
                    clip_fractions.append(float((torch.abs(ratio - 1) > clip_range)[mask].float().mean().cpu().item()))
                policy_losses.append(float(policy_loss.item()))
        for parameter in self.policy.critic_parameters:
            parameter.requires_grad_(True)

        for parameter in self.policy.actor_parameters:
            parameter.requires_grad_(False)
        for _epoch in range(self.critic_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size, rng=self.critic_minibatch_rng):
                mask = rollout_data.mask > 1e-8
                values = self.policy.evaluate_values(rollout_data.observations).flatten()
                value_loss = torch.nn.functional.mse_loss(values[mask], rollout_data.returns[mask])
                self.policy.critic_optimizer.zero_grad()
                (VALUE_LOSS_COEFFICIENT * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters, MAX_GRAD_NORM)
                self.policy.critic_optimizer.step()
                value_losses.append(float(value_loss.item()))
        for parameter in self.policy.actor_parameters:
            parameter.requires_grad_(True)

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_gradient_loss", np.mean(policy_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approximate_kls))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/explained_variance", explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))


def validate_arguments(args) -> None:
    pretrained_path = Path(args.pretrained_model_path).expanduser().resolve()
    ppo_path = Path(args.ppo_model_path).expanduser().resolve()
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"Pretrained model does not exist: {pretrained_path}")
    if pretrained_path == ppo_path:
        raise ValueError("PPO model path must not overwrite the pretrained model")
    if not args.map_name.strip():
        raise ValueError("map_name must not be empty")
    if args.env_workers <= 0 or args.n_envs < args.env_workers or args.n_envs % 2 != 0:
        raise ValueError("n_envs must be even and at least env_workers, and env_workers must be positive")
    for name in ("hidden_scale", "n_steps", "batch_size", "num_updates", "actor_epochs", "critic_epochs"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.n_envs * args.n_steps % args.batch_size != 0:
        raise ValueError("n_envs * n_steps must be divisible by batch_size")
    for name in ("gru_learning_rate", "head_learning_rate", "critic_learning_rate", "gamma", "gae_lambda", "clip_range"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")


def build_model(vector_env, args, device) -> End2RaceRecurrentPPO:
    return End2RaceRecurrentPPO(
        End2RaceGRUPolicy,
        vector_env,
        actor_epochs=args.actor_epochs,
        critic_epochs=args.critic_epochs,
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


def save_actor(model: End2RaceRecurrentPPO, path: Path, hidden_scale: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {name: tensor.detach().cpu() for name, tensor in model.policy.actor_checkpoint_state_dict().items()}
    if len(state_dict) != 12:
        raise RuntimeError(f"Expected a 12-key actor checkpoint, got {len(state_dict)} keys")
    torch.save(state_dict, path)
    actor = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    configure_training_numerics()
    random.seed(args.seed)
    np.random.seed(args.seed)

    collision_scenarios = classify_collision_scenarios(args.pretrained_model_path, args.hidden_scale, args.map_name, args.env_workers)
    ordinary_scenario_set = ordinary_scenarios(args.map_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    vector_env = CentralScheduleSubprocVecEnv(args.n_envs, args.env_workers, START_METHOD, args.seed, args.map_name, collision_scenarios, ordinary_scenario_set)
    try:
        model = build_model(vector_env, args, device)
        total_rollouts = args.num_updates + 1
        model.learn(total_timesteps=args.n_envs * args.n_steps * total_rollouts, log_interval=1, progress_bar=False)
        output_path = Path(args.ppo_model_path).expanduser().resolve()
        save_actor(model, output_path, args.hidden_scale)
        print(f"PPO model saved: {output_path}")
    finally:
        vector_env.close()


if __name__ == "__main__":
    main()

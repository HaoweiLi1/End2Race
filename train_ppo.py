#!/usr/bin/env python3
"""Train the fixed End2Race PPO pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
import copy as copy_module
from dataclasses import asdict
import json
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


def collision_classification_config(args, candidate_count: int) -> dict:
    return {
        "classification_schema": 1,
        "pretrained_model_path": str(Path(args.pretrained_model_path).expanduser().resolve()),
        "hidden_scale": int(args.hidden_scale),
        "map_name": str(args.map_name),
        "ego_raceline": str(PPO_CONFIG["ego_raceline"]),
        "opponent_racelines": [str(value) for value in PPO_CONFIG["opponent_racelines"]],
        "collision_startpoint_count": int(PPO_CONFIG["collision_startpoint_count"]),
        "collision_interval_indices": [int(value) for value in PPO_CONFIG["collision_interval_indices"]],
        "collision_speed_scales": [float(value) for value in PPO_CONFIG["collision_speed_scales"]],
        "collision_startpoint_min_distance": float(PPO_CONFIG["collision_startpoint_min_distance"]),
        "simulator_timestep": float(PPO_CONFIG["simulator_timestep"]),
        "episode_horizon": float(PPO_CONFIG["episode_horizon"]),
        "candidate_count": candidate_count,
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _collision_cache_exists(cache_dir: Path) -> bool:
    required_paths = (
        cache_dir / "classification_config.json",
        cache_dir / "candidate_outcomes.jsonl",
        cache_dir / "collision_scenarios.json",
        cache_dir / "classification_summary.json",
    )
    existing_count = sum(path.exists() for path in required_paths)
    if existing_count not in (0, len(required_paths)):
        raise RuntimeError("Collision classification cache is incomplete; use --reclassify_collisions")
    return existing_count == len(required_paths)


def write_collision_cache(cache_dir: Path, config: dict, outcomes: list[dict], collision_scenarios: tuple[ScenarioSpec, ...], summary: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cache_dir / "classification_config.json", config)
    with (cache_dir / "candidate_outcomes.jsonl").open("w", encoding="utf-8") as file:
        for outcome in outcomes:
            file.write(json.dumps(outcome) + "\n")
    _write_json(cache_dir / "collision_scenarios.json", [asdict(scenario) for scenario in collision_scenarios])
    _write_json(cache_dir / "classification_summary.json", summary)


def _load_candidate_outcomes(path: Path, candidates: tuple[ScenarioSpec, ...]) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        outcomes = [json.loads(line) for line in file]
    candidate_count = len(candidates)
    if len(outcomes) != candidate_count:
        raise RuntimeError(f"Collision cache has {len(outcomes)} outcomes for {candidate_count} candidates")
    expected_keys = {"candidate_index", "scenario_id", "outcome"}
    for candidate_index, (outcome, candidate) in enumerate(zip(outcomes, candidates)):
        if set(outcome) != expected_keys or type(outcome["candidate_index"]) is not int or outcome["candidate_index"] != candidate_index:
            raise RuntimeError(f"Collision cache candidate_index must be 0 through {candidate_count - 1}")
        if outcome["scenario_id"] != candidate.scenario_id:
            raise RuntimeError(f"Collision cache scenario_id mismatch at candidate {candidate_index}/{candidate_count}")
        if outcome["outcome"] not in {"ego_collision", "other", "invalid"}:
            raise RuntimeError(f"Collision cache has an invalid outcome at candidate {candidate_index}/{candidate_count}")
    return outcomes


def _load_collision_scenarios(path: Path, candidates: tuple[ScenarioSpec, ...], outcomes: list[dict]) -> tuple[ScenarioSpec, ...]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise RuntimeError("Collision cache must contain at least one collision ScenarioSpec")
    candidate_by_id = {candidate.scenario_id: candidate for candidate in candidates}
    expected_ids = [outcome["scenario_id"] for outcome in outcomes if outcome["outcome"] == "ego_collision"]
    actual_ids = [record.get("scenario_id") for record in records if isinstance(record, dict)]
    if len(actual_ids) != len(records) or len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError("Collision cache collision scenario IDs must be unique")
    if actual_ids != expected_ids:
        raise RuntimeError("Collision cache collision scenarios do not match ego_collision outcomes")
    collision_scenarios = []
    expected_fields = set(asdict(candidates[0]))
    for record in records:
        if set(record) != expected_fields:
            raise RuntimeError(f"Collision cache ScenarioSpec fields are invalid for {record['scenario_id']}")
        scenario = ScenarioSpec(**record)
        current_candidate = candidate_by_id.get(scenario.scenario_id)
        if current_candidate is None:
            raise RuntimeError(f"Collision cache ScenarioSpec does not match current candidate {scenario.scenario_id}")
        current_record = asdict(current_candidate)
        if any(type(record[name]) is not type(current_record[name]) or record[name] != current_record[name] for name in expected_fields):
            raise RuntimeError(f"Collision cache ScenarioSpec does not match current candidate {scenario.scenario_id}")
        collision_scenarios.append(scenario)
    return tuple(collision_scenarios)


def _validate_classification_summary(path: Path, outcomes: list[dict], candidate_count: int) -> None:
    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    collision_count = sum(outcome["outcome"] == "ego_collision" for outcome in outcomes)
    invalid_count = sum(outcome["outcome"] == "invalid" for outcome in outcomes)
    expected_counts = {
        "candidate_count": candidate_count,
        "collision_count": collision_count,
        "other_count": candidate_count - collision_count - invalid_count,
        "invalid_count": invalid_count,
    }
    expected_keys = set(expected_counts) | {"env_workers", "wall_seconds", "scenarios_per_second"}
    if not isinstance(summary, dict) or set(summary) != expected_keys or any(type(summary[name]) is not int or summary[name] != value for name, value in expected_counts.items()):
        raise RuntimeError(f"Collision cache summary does not match {candidate_count} candidate outcomes")
    if type(summary["env_workers"]) is not int or summary["env_workers"] <= 0:
        raise RuntimeError("Collision cache summary has invalid env_workers")
    for name in ("wall_seconds", "scenarios_per_second"):
        value = summary[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise RuntimeError(f"Collision cache summary has invalid {name}")


def load_collision_cache(cache_dir: Path, current_config: dict, candidates: tuple[ScenarioSpec, ...]) -> tuple[ScenarioSpec, ...]:
    with (cache_dir / "classification_config.json").open("r", encoding="utf-8") as file:
        cached_config = json.load(file)
    candidate_count = len(candidates)
    if json.dumps(cached_config, sort_keys=True) != json.dumps(current_config, sort_keys=True):
        raise RuntimeError(f"Collision cache configuration does not match the current {candidate_count} candidates; use --reclassify_collisions")
    outcomes = _load_candidate_outcomes(cache_dir / "candidate_outcomes.jsonl", candidates)
    collision_scenarios = _load_collision_scenarios(cache_dir / "collision_scenarios.json", candidates, outcomes)
    _validate_classification_summary(cache_dir / "classification_summary.json", outcomes, candidate_count)
    return collision_scenarios


def resolve_collision_scenarios(args, candidates: tuple[ScenarioSpec, ...]) -> tuple[tuple[ScenarioSpec, ...], bool, bool]:
    candidate_count = len(candidates)
    if candidate_count == 0:
        raise RuntimeError("Collision candidate set is empty")
    cache_dir = Path(args.collision_cache_dir).expanduser().resolve()
    current_config = collision_classification_config(args, candidate_count)
    if args.reclassify_collisions:
        print(f"Rebuilding collision classification cache for {candidate_count} candidates", flush=True)
    elif _collision_cache_exists(cache_dir):
        collision_scenarios = load_collision_cache(cache_dir, current_config, candidates)
        print(f"Collision cache hit: loaded {len(collision_scenarios)} collision scenarios from {candidate_count} candidates", flush=True)
        return collision_scenarios, True, False
    else:
        print(f"Collision cache miss: classifying {candidate_count} candidates", flush=True)
    collision_scenarios, outcomes, summary = classify_collision_scenarios(args.pretrained_model_path, args.hidden_scale, args.map_name, args.env_workers, candidates)
    write_collision_cache(cache_dir, current_config, outcomes, collision_scenarios, summary)
    return collision_scenarios, False, bool(args.reclassify_collisions)


def _training_run_directory(ppo_model_path: str | Path) -> Path:
    output_path = Path(ppo_model_path).expanduser().resolve()
    post_trained_dir = (Path.cwd() / "post-trained").resolve()
    try:
        relative_output = output_path.relative_to(post_trained_dir)
    except ValueError:
        run_name = output_path.stem
    else:
        run_name = relative_output.parts[0] if len(relative_output.parts) > 1 else output_path.stem
    return post_trained_dir / run_name


def write_run_collision_records(args, collision_scenarios: tuple[ScenarioSpec, ...], cache_hit: bool, reclassified: bool, candidate_count: int) -> None:
    run_dir = _training_run_directory(args.ppo_model_path)
    _write_json(run_dir / "collision_scenarios.json", [asdict(scenario) for scenario in collision_scenarios])
    _write_json(
        run_dir / "collision_cache_info.json",
        {
            "cache_dir": str(Path(args.collision_cache_dir).expanduser().resolve()),
            "cache_hit": cache_hit,
            "reclassified": reclassified,
            "candidate_count": candidate_count,
            "collision_count": len(collision_scenarios),
        },
    )


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
    if not args.collision_cache_dir.strip():
        raise ValueError("collision_cache_dir must not be empty")
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

    candidates = expanded_scenarios(args.map_name)
    candidate_count = len(candidates)
    collision_scenarios, cache_hit, reclassified = resolve_collision_scenarios(args, candidates)
    write_run_collision_records(args, collision_scenarios, cache_hit, reclassified, candidate_count)
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

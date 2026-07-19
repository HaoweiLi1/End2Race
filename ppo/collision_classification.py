"""Collision candidate classification and cache selection."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from model import End2Race
from ppo.environment import EXTERNAL_RESET_OPTION, make_environment
from ppo.policy import END2RACE_LIDAR_SIZE, STEERING_BOUND
from ppo.scenarios import (
    ScenarioSpec,
    collision_cache_exists,
    collision_classification_config,
    load_collision_cache,
    write_collision_cache,
)
from ppo.vec_env import _limit_worker_threads


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
    start_method: str,
) -> tuple[tuple[ScenarioSpec, ...], list[dict], dict]:
    candidate_count = len(candidates)
    context = mp.get_context(start_method)
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


def resolve_collision_scenarios(args, candidates: tuple[ScenarioSpec, ...], start_method: str) -> tuple[tuple[ScenarioSpec, ...], bool, bool]:
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
    collision_scenarios, outcomes, summary = classify_collision_scenarios(args.pretrained_model_path, args.hidden_scale, args.map_name, args.env_workers, candidates, start_method)
    write_collision_cache(cache_dir, current_config, outcomes, collision_scenarios, summary)
    return collision_scenarios, False, bool(args.reclassify_collisions)

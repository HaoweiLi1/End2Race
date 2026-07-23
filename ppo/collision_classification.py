"""Collision candidate classification and cache selection."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from model import End2Race
from ppo.environment import EXTERNAL_RESET_OPTION, make_environment
from ppo.policy import END2RACE_LIDAR_SIZE, STEERING_BOUND
from ppo.scenarios import (
    COLLISION_INTERVAL_INDICES,
    COLLISION_SPEED_SCALES,
    COLLISION_STARTPOINT_COUNT,
    COLLISION_STARTPOINT_MIN_DISTANCE,
    EGO_RACELINE,
    OPPONENT_RACELINES,
    SIM_DURATION,
    TIMESTEP,
    ScenarioSpec,
)
from ppo.vec_env import limit_worker_threads


_COLLISION_ENV = None
_COLLISION_ACTOR = None
COLLISION_CLASSIFICATION_SCHEMA = 1


def collision_classification_config(args, candidate_count: int) -> dict:
    return {
        "classification_schema": COLLISION_CLASSIFICATION_SCHEMA,
        "pretrained_model_path": str(Path(args.pretrained_model_path).expanduser().resolve()),
        "hidden_scale": int(args.hidden_scale),
        "map_name": str(args.map_name),
        "ego_raceline": EGO_RACELINE,
        "opponent_racelines": list(OPPONENT_RACELINES),
        "collision_startpoint_count": COLLISION_STARTPOINT_COUNT,
        "collision_interval_indices": list(COLLISION_INTERVAL_INDICES),
        "collision_speed_scales": list(COLLISION_SPEED_SCALES),
        "collision_startpoint_min_distance": COLLISION_STARTPOINT_MIN_DISTANCE,
        "simulator_timestep": TIMESTEP,
        "episode_horizon": SIM_DURATION,
        "candidate_count": candidate_count,
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def collision_cache_exists(cache_dir: Path) -> bool:
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


def write_collision_cache(
    cache_dir: Path,
    config: dict,
    outcomes: list[dict],
    collision_scenarios: tuple[ScenarioSpec, ...],
    summary: dict,
) -> None:
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
        if (
            set(outcome) != expected_keys
            or type(outcome["candidate_index"]) is not int
            or outcome["candidate_index"] != candidate_index
        ):
            raise RuntimeError(f"Collision cache candidate_index must be 0 through {candidate_count - 1}")
        if outcome["scenario_id"] != candidate.scenario_id:
            raise RuntimeError(f"Collision cache scenario_id mismatch at candidate {candidate_index}/{candidate_count}")
        if outcome["outcome"] not in {"ego_collision", "other", "invalid"}:
            raise RuntimeError(f"Collision cache has an invalid outcome at candidate {candidate_index}/{candidate_count}")
    return outcomes


def _load_collision_scenarios(
    path: Path,
    candidates: tuple[ScenarioSpec, ...],
    outcomes: list[dict],
) -> tuple[ScenarioSpec, ...]:
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
        if any(
            type(record[name]) is not type(current_record[name]) or record[name] != current_record[name]
            for name in expected_fields
        ):
            raise RuntimeError(f"Collision cache ScenarioSpec does not match current candidate {scenario.scenario_id}")
        collision_scenarios.append(scenario)
    return tuple(collision_scenarios)


def _validate_classification_summary(path: Path, outcomes: list[dict], candidate_count: int) -> dict:
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
    if (
        not isinstance(summary, dict)
        or set(summary) != expected_keys
        or any(type(summary[name]) is not int or summary[name] != value for name, value in expected_counts.items())
    ):
        raise RuntimeError(f"Collision cache summary does not match {candidate_count} candidate outcomes")
    if type(summary["env_workers"]) is not int or summary["env_workers"] <= 0:
        raise RuntimeError("Collision cache summary has invalid env_workers")
    for name in ("wall_seconds", "scenarios_per_second"):
        value = summary[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise RuntimeError(f"Collision cache summary has invalid {name}")
    return summary


def load_collision_cache_artifacts(
    cache_dir: Path,
    current_config: dict,
    candidates: tuple[ScenarioSpec, ...],
) -> tuple[tuple[ScenarioSpec, ...], list[dict], dict]:
    """Strictly load the collision pool together with its validated evidence."""

    with (cache_dir / "classification_config.json").open("r", encoding="utf-8") as file:
        cached_config = json.load(file)
    candidate_count = len(candidates)
    if json.dumps(cached_config, sort_keys=True) != json.dumps(current_config, sort_keys=True):
        raise RuntimeError(
            f"Collision cache configuration does not match the current {candidate_count} candidates; "
            "use --reclassify_collisions"
        )
    outcomes = _load_candidate_outcomes(cache_dir / "candidate_outcomes.jsonl", candidates)
    collision_scenarios = _load_collision_scenarios(cache_dir / "collision_scenarios.json", candidates, outcomes)
    summary = _validate_classification_summary(cache_dir / "classification_summary.json", outcomes, candidate_count)
    return collision_scenarios, outcomes, summary


def load_collision_cache(
    cache_dir: Path,
    current_config: dict,
    candidates: tuple[ScenarioSpec, ...],
) -> tuple[ScenarioSpec, ...]:
    collision_scenarios, _outcomes, _summary = load_collision_cache_artifacts(
        cache_dir,
        current_config,
        candidates,
    )
    return collision_scenarios


def _collision_worker_init(pretrained_model_path: str, hidden_scale: int, map_name: str) -> None:
    global _COLLISION_ENV, _COLLISION_ACTOR
    limit_worker_threads()
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

#!/usr/bin/env python3
"""Replay one saved action sequence through Dummy and parent-scheduled workers."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo import config as ppo_config
from ppo.vec_env import CentralScheduleSubprocVecEnv
from train_ppo import (
    build_sampler,
    make_subprocess_training_env,
    make_training_env,
    write_json,
)


INFO_KEYS = (
    "ego_collision",
    "opponent_collision",
    "opponent_collision_latched",
    "timeout",
    "termination_reason",
    "scenario_id",
    "env_role",
    "episode_outcome",
    "reward_total",
    "ego_progress_delta_m",
    "opponent_progress_delta_m",
    "relative_position_m",
)


def hash_array(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def reset_record(rank: int, info: dict[str, Any]) -> tuple[Any, ...]:
    return (
        rank,
        str(info["scenario_id"]),
        str(info["env_role"]),
        info.get("pair_group"),
        info.get("pair_member"),
        info.get("pair_episode_ordinal"),
    )


def replay(
    mode: str,
    actions: np.ndarray,
    config: ppo_config.PPOConfig,
    seed: int,
    worker_count: int,
) -> dict[str, Any]:
    sampler = build_sampler(config)
    if mode == "dummy":
        vector_env = DummyVecEnv(
            [
                make_training_env(rank, sampler, config, seed)
                for rank in range(ppo_config.N_ENVS)
            ]
        )
    else:
        vector_env = CentralScheduleSubprocVecEnv(
            [
                make_subprocess_training_env(rank, config, seed)
                for rank in range(ppo_config.N_ENVS)
            ],
            sampler=sampler,
            config=config,
            seed=seed,
            worker_count=worker_count,
        )
    vector_env.seed(seed)
    vector_env.env_method("set_policy_update_index", 1)
    observation = vector_env.reset()
    reset_order = [reset_record(rank, info) for rank, info in enumerate(vector_env.reset_infos)]
    hashes: dict[str, list[str]] = {
        key: []
        for key in (
            "observation",
            "executed_action",
            "reward",
            "terminated",
            "truncated",
            "episode_outcome",
            "info_contract",
        )
    }
    events: Counter[str] = Counter()
    for step_actions in actions:
        hashes["observation"].append(hash_array(observation))
        hashes["executed_action"].append(hash_array(step_actions))
        observation, rewards, dones, infos = vector_env.step(step_actions)
        truncated = np.asarray([bool(info["TimeLimit.truncated"]) for info in infos])
        terminated = np.logical_and(dones, np.logical_not(truncated))
        outcomes = [info["episode_outcome"] for info in infos]
        contract = [{key: info.get(key) for key in INFO_KEYS} for info in infos]
        hashes["reward"].append(hash_array(rewards))
        hashes["terminated"].append(hash_array(terminated))
        hashes["truncated"].append(hash_array(truncated))
        hashes["episode_outcome"].append(hash_json(outcomes))
        hashes["info_contract"].append(hash_json(contract))
        for rank, (done, info) in enumerate(zip(dones, infos)):
            events["ego_collision"] += int(bool(info["ego_collision"]))
            events["opponent_collision"] += int(bool(info["opponent_collision"]))
            events["timeout"] += int(bool(info["timeout"]))
            if info["episode_outcome"] is not None:
                events[str(info["episode_outcome"])] += 1
            if done:
                reset_order.append(reset_record(rank, vector_env.reset_infos[rank]))
    vector_env.close()
    return {
        "mode": mode,
        "reset_order": reset_order,
        "reset_order_sha256": hash_json(reset_order),
        "sampler_visit_sha256": hash_json(sampler.visit_counts),
        "step_hashes": hashes,
        "step_contract_sha256": {key: hash_json(value) for key, value in hashes.items()},
        "events": dict(sorted(events.items())),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="N1-H1F-p50", choices=ppo_config.CONFIGS)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--worker-count", type=int, default=6)
    arguments = parser.parse_args()
    action_array = np.load(arguments.actions, allow_pickle=False)
    config = ppo_config.get_config(arguments.config)
    baseline = replay("dummy", action_array, config, arguments.seed, arguments.worker_count)
    candidate = replay("central_subproc", action_array, config, arguments.seed, arguments.worker_count)
    compared = (
        "reset_order",
        "reset_order_sha256",
        "sampler_visit_sha256",
        "step_hashes",
        "step_contract_sha256",
        "events",
    )
    checks = {key: baseline[key] == candidate[key] for key in compared}
    result = {
        "schema_version": 1,
        "config": arguments.config,
        "seed": arguments.seed,
        "worker_count": arguments.worker_count,
        "actions_shape": list(action_array.shape),
        "actions_sha256": hash_array(action_array),
        "baseline": baseline,
        "candidate": candidate,
        "checks": checks,
        "all_pass": all(checks.values()),
    }
    write_json(arguments.output, result)
    print(json.dumps({"checks": checks, "all_pass": result["all_pass"]}, sort_keys=True))
    raise SystemExit(0 if result["all_pass"] else 1)

#!/usr/bin/env python3
"""Exercise normal and exceptional worker cleanup without a training update."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo import config as ppo_config
from ppo.vec_env import CentralScheduleSubprocVecEnv
from train_ppo import build_sampler, write_json


class LifecycleEnv(gym.Env):
    render_mode = None

    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {"options_present": options is not None}

    def step(self, action):
        if self.fail:
            raise RuntimeError("intentional lifecycle validation failure")
        return np.asarray(action, dtype=np.float32), 0.0, False, False, {}


def factory(fail: bool):
    return lambda: LifecycleEnv(fail)


def run_case(fail: bool) -> dict[str, object]:
    config = ppo_config.get_config("N1-H1F-p50")
    vector_env = CentralScheduleSubprocVecEnv(
        [factory(fail), factory(fail)],
        sampler=build_sampler(config),
        config=config,
        seed=20260917,
        worker_count=2,
    )
    vector_env.seed(20260917)
    vector_env.reset()
    error = None
    try:
        vector_env.step(np.zeros((2, 1), dtype=np.float32))
    except RuntimeError as caught:
        error = str(caught)
    if not vector_env.closed:
        vector_env.close()
    workers = [
        {
            "pid": process.pid,
            "alive": process.is_alive(),
            "exitcode": process.exitcode,
        }
        for process in vector_env.processes
    ]
    return {
        "intentional_failure": fail,
        "error_seen": error is not None,
        "expected_error_text": error is not None and "intentional lifecycle validation failure" in error,
        "workers": workers,
        "all_workers_exited": all(not worker["alive"] for worker in workers),
    }


if __name__ == "__main__":
    output = Path(sys.argv[1])
    normal = run_case(False)
    exceptional = run_case(True)
    checks = {
        "normal_has_no_error": not normal["error_seen"],
        "normal_workers_exited": normal["all_workers_exited"],
        "exception_propagated": exceptional["expected_error_text"],
        "exception_workers_exited": exceptional["all_workers_exited"],
    }
    result = {"normal": normal, "exceptional": exceptional, "checks": checks, "all_pass": all(checks.values())}
    write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["all_pass"] else 1)

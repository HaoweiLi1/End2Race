#!/usr/bin/env python3
"""Shared, audit-only helpers for the PPO RL direction mechanism audit."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import End2Race  # noqa: E402
from ppo.environment import (  # noqa: E402
    End2RaceGymnasiumEnv,
    LatticePlannerOpponentController,
    oriented_rectangle_clearance,
)
from ppo.policy import EVALUATOR_STEER_BOUND  # noqa: E402
from ppo.reward import PPOTransitionReward, ProgressProjector  # noqa: E402
from ppo.scenarios import ScenarioSpec  # noqa: E402


EXPERIMENT_DIR = ROOT / "ppo_experiments" / "rl_direction_audit"
RUN_DIR = ROOT / "runs" / "ppo" / "RL_DIRECTION_AUDIT_20260717"
PREREGISTRATION_PATH = EXPERIMENT_DIR / "AUDIT_PREREGISTRATION.json"
BC_PATH = ROOT / "pretrained" / "end2race.pth"
GAMMA = 0.999
STEERING_LATENT_STD = 0.05
SPEED_PHYSICAL_STD = 0.15
SIM_DURATION = 8.0
TIMESTEP = 0.01


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def assert_frozen_contract() -> dict[str, str]:
    preregistration = read_json(PREREGISTRATION_PATH)
    observed: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in preregistration["frozen_files"].items():
        digest = sha256_file(ROOT / relative)
        observed[relative] = digest
        if digest != expected:
            mismatches[relative] = {"expected": expected, "observed": digest}
    if mismatches:
        raise RuntimeError(f"Frozen audit contract drifted: {json.dumps(mismatches, sort_keys=True)}")
    return observed


def load_actor(device: torch.device) -> End2Race:
    actor = End2Race(mask_prob=0.0, hidden_scale=4).to(device)
    state = torch.load(BC_PATH, map_location=device, weights_only=True)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


class FixedScenarioProvider:
    """Mutable provider used only to reset one reusable simulator to an exact case."""

    def __init__(self) -> None:
        self._scenario: ScenarioSpec | None = None
        self._sampler_branch = "reference"
        self._hard_pool_id = "REFERENCE"

    def set(self, scenario: ScenarioSpec, *, sampler_branch: str, hard_pool_id: str) -> None:
        self._scenario = scenario
        self._sampler_branch = str(sampler_branch)
        self._hard_pool_id = str(hard_pool_id)

    def __call__(self, rng: np.random.Generator):
        del rng
        if self._scenario is None:
            raise RuntimeError("FixedScenarioProvider requires set() before reset()")
        spec = deepcopy(self._scenario.to_reset_spec(sampler_branch=self._sampler_branch))
        spec.scenario["hard_pool_id"] = self._hard_pool_id
        spec.scenario["hard_sampling_mode"] = "audit_fixed"
        return spec


def make_env(provider: FixedScenarioProvider, seed: int) -> End2RaceGymnasiumEnv:
    import gym
    from f110_gym.envs.base_classes import Integrator

    core = gym.make(
        "f110-v0",
        map=str(ROOT / "f1tenth_racetracks" / "Austin" / "Austin_map"),
        map_ext=".png",
        num_agents=2,
        timestep=TIMESTEP,
        integrator=Integrator.RK4,
        seed=seed,
    )
    return End2RaceGymnasiumEnv(
        core,
        sim_duration=SIM_DURATION,
        reset_provider=provider,
        ego_index=0,
        opponent_controller=LatticePlannerOpponentController(),
        transition_reward=PPOTransitionReward(ProgressProjector.from_csv()),
        privileged_critic=False,
    )


def _actor_step(
    actor: End2Race,
    observation: np.ndarray,
    hidden: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
    lidar = observation_tensor[:360].reshape(1, 1, 360)
    previous_speed = observation_tensor[360:].reshape(1, 1, 1)
    with torch.inference_mode():
        raw_action, next_hidden = actor(lidar, previous_speed, hidden)
    raw = raw_action[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    physical = raw.copy()
    physical[0] = np.clip(physical[0], -EVALUATOR_STEER_BOUND, EVALUATOR_STEER_BOUND)
    return physical, next_hidden, raw


def _poses(raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(
        [raw["poses_x"][0], raw["poses_y"][0], raw["poses_theta"][0]],
        dtype=np.float64,
    )
    second = np.asarray(
        [raw["poses_x"][1], raw["poses_y"][1], raw["poses_theta"][1]],
        dtype=np.float64,
    )
    return first, second


def run_deterministic_episode(
    env: End2RaceGymnasiumEnv,
    provider: FixedScenarioProvider,
    actor: End2Race,
    scenario: ScenarioSpec,
    device: torch.device,
    *,
    seed: int,
    capture_trace: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    provider.set(scenario, sampler_branch="reference", hard_pool_id="REFERENCE")
    observation, _reset_info = env.reset(seed=seed)
    hidden = torch.zeros((1, 1, actor.gru.hidden_size), dtype=torch.float32, device=device)
    minimum_clearance = float("inf")
    discounted_return = 0.0
    step_index = 0
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    raw_means: list[np.ndarray] = []
    rewards: list[float] = []
    clearances: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []

    while True:
        if capture_trace:
            observations.append(np.asarray(observation, dtype=np.float32).copy())
        action, hidden, raw_mean = _actor_step(actor, observation, hidden, device)
        observation, reward, terminated, truncated, info = env.step(action)
        first_pose, second_pose = _poses(env._raw_observation)
        clearance = oriented_rectangle_clearance(first_pose, second_pose)
        minimum_clearance = min(minimum_clearance, clearance)
        discounted_return += (GAMMA**step_index) * float(reward)
        if capture_trace:
            actions.append(action.copy())
            raw_means.append(raw_mean.copy())
            rewards.append(float(reward))
            clearances.append(float(clearance))
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
        step_index += 1
        if terminated or truncated:
            break
        if step_index > 1000:
            raise RuntimeError(f"Episode exceeded 1000 steps: {scenario.scenario_id}")

    ego_collision = bool(info["ego_collision"])
    relative_position = float(info["relative_position_m"])
    outcome = "ego_collision" if ego_collision else ("overtake" if relative_position > 0.0 else "follow")
    result = {
        "scenario_id": scenario.scenario_id,
        "scenario": scenario.to_dict(),
        "outcome": outcome,
        "ego_collision": ego_collision,
        "opponent_collision": bool(info["opponent_collision"]),
        "elapsed_time": float(info["elapsed_time"]),
        "steps": step_index,
        "discounted_return": float(discounted_return),
        "final_relative_position_m": relative_position,
        "min_oriented_clearance_m": float(minimum_clearance),
        "termination_reason": info["termination_reason"],
    }
    trace = None
    if capture_trace:
        trace = {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "raw_means": np.asarray(raw_means, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "clearances": np.asarray(clearances, dtype=np.float32),
            "terminated": np.asarray(terminated_flags, dtype=np.bool_),
            "truncated": np.asarray(truncated_flags, dtype=np.bool_),
        }
    return result, trace


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def summarize_outcomes(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    summary = {"ego_collision": 0, "follow": 0, "overtake": 0}
    for row in rows:
        summary[str(row["outcome"])] += 1
    return summary


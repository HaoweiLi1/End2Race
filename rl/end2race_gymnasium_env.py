"""Gymnasium adapter for legacy F1Tenth environments used by End2Race.

The actor observation contains only a 360D ego LiDAR scan and previous ego
speed.  F1Tenth source code is not modified.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.sb3_end2race_policy import END2RACE_LIDAR_SIZE, end2race_observation


OpponentActionFn = Callable[[dict[str, Any], np.ndarray], np.ndarray]


class End2RaceGymnasiumEnv(gym.Env[np.ndarray, np.ndarray]):
    """Convert legacy F1Tenth reset/step results to Gymnasium termination rules."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        f110_env: Any,
        sim_duration: float,
        ego_index: int = 0,
        opponent_action_fn: OpponentActionFn | None = None,
        action_low: tuple[float, float] = (-0.52, 0.0),
        action_high: tuple[float, float] = (0.52, 20.0),
    ) -> None:
        super().__init__()
        if sim_duration <= 0:
            raise ValueError("sim_duration must be positive")
        self.f110_env = f110_env
        self.sim_duration = float(sim_duration)
        self.ego_index = int(ego_index)
        self.opponent_action_fn = opponent_action_fn
        self.observation_space = spaces.Box(
            low=np.full((END2RACE_LIDAR_SIZE + 1,), -np.inf, dtype=np.float32),
            high=np.full((END2RACE_LIDAR_SIZE + 1,), np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.asarray(action_low, dtype=np.float32),
            high=np.asarray(action_high, dtype=np.float32),
            dtype=np.float32,
        )
        self._elapsed_time = 0.0
        self._previous_ego_speed = 0.0
        self._raw_observation: dict[str, Any] | None = None
        self._lifetime_steps = 0
        self.terminal_events: list[dict[str, Any]] = []

    @property
    def num_agents(self) -> int:
        unwrapped = getattr(self.f110_env, "unwrapped", self.f110_env)
        return int(getattr(unwrapped, "num_agents", getattr(self.f110_env, "num_agents", 1)))

    @staticmethod
    def _legacy_reset_result(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(result, tuple):
            return result, {}
        if len(result) == 2 and isinstance(result[1], dict):
            return result[0], result[1]
        if len(result) == 4:
            return result[0], result[3] if isinstance(result[3], dict) else {}
        raise ValueError(f"Unsupported F1Tenth reset result with {len(result)} entries")

    @staticmethod
    def _step_result(result: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not isinstance(result, tuple):
            raise TypeError("F1Tenth step must return a tuple")
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, float(reward), bool(terminated), bool(truncated), dict(info)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, float(reward), bool(done), False, dict(info)
        raise ValueError(f"Unsupported F1Tenth step result with {len(result)} entries")

    def _ego_lidar(self, raw_observation: dict[str, Any]) -> np.ndarray:
        scan = np.asarray(raw_observation["scans"][self.ego_index], dtype=np.float32).reshape(-1)
        if scan.size != END2RACE_LIDAR_SIZE:
            indices = np.linspace(0, scan.size - 1, END2RACE_LIDAR_SIZE, dtype=int)
            scan = scan[indices]
        return scan

    def _ego_speed(self, raw_observation: dict[str, Any]) -> float:
        return float(np.asarray(raw_observation["linear_vels_x"])[self.ego_index])

    def _actor_observation(self, raw_observation: dict[str, Any]) -> np.ndarray:
        return end2race_observation(self._ego_lidar(raw_observation), self._previous_ego_speed)

    def _base_reset(self, seed: int | None, options: dict[str, Any] | None) -> Any:
        reset_kwargs: dict[str, Any] = {}
        if options and "poses" in options:
            reset_kwargs["poses"] = options["poses"]
        if seed is not None:
            try:
                return self.f110_env.reset(seed=seed, **reset_kwargs)
            except TypeError:
                pass
        return self.f110_env.reset(**reset_kwargs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        raw_observation, base_info = self._legacy_reset_result(self._base_reset(seed, options))
        self._elapsed_time = 0.0
        self._raw_observation = raw_observation
        self._previous_ego_speed = self._ego_speed(raw_observation)
        info = {
            "collision": False,
            "elapsed_time": 0.0,
            "termination_reason": None,
            "base_info": base_info,
        }
        return self._actor_observation(raw_observation), info

    def _joint_action(self, ego_action: np.ndarray) -> np.ndarray:
        ego_action = np.asarray(ego_action, dtype=np.float32).reshape(2)
        if self.num_agents == 1:
            return ego_action.reshape(1, 2)
        joint_action = np.zeros((self.num_agents, 2), dtype=np.float32)
        joint_action[self.ego_index] = ego_action
        if self.opponent_action_fn is not None:
            if self._raw_observation is None:
                raise RuntimeError("Environment must be reset before step")
            opponent_actions = np.asarray(self.opponent_action_fn(self._raw_observation, ego_action), dtype=np.float32)
            if opponent_actions.shape == joint_action.shape:
                joint_action = opponent_actions
                joint_action[self.ego_index] = ego_action
            elif opponent_actions.shape == (self.num_agents - 1, 2):
                joint_action[np.arange(self.num_agents) != self.ego_index] = opponent_actions
            else:
                raise ValueError(f"Unexpected opponent action shape {opponent_actions.shape}")
        return joint_action

    def _step_duration(self, reward: float, info: dict[str, Any]) -> float:
        if "timestep" in info:
            return float(info["timestep"])
        unwrapped = getattr(self.f110_env, "unwrapped", self.f110_env)
        configured = getattr(unwrapped, "timestep", getattr(self.f110_env, "timestep", None))
        if configured is not None:
            return float(configured)
        # Legacy F1Tenth uses its timestep as the returned reward.
        if reward > 0:
            return float(reward)
        raise ValueError("Cannot infer simulation timestep; provide info['timestep']")

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_observation, reward, base_terminated, base_truncated, base_info = self._step_result(
            self.f110_env.step(self._joint_action(action))
        )
        self._lifetime_steps += 1
        self._elapsed_time += self._step_duration(reward, base_info)
        self._raw_observation = raw_observation
        self._previous_ego_speed = self._ego_speed(raw_observation)
        actor_observation = self._actor_observation(raw_observation)

        collisions = np.asarray(raw_observation.get("collisions", []), dtype=bool)
        collision = bool(collisions.size > self.ego_index and collisions[self.ego_index])
        timeout = self._elapsed_time + 1e-12 >= self.sim_duration
        if collision:
            terminated, truncated, reason = True, False, "collision"
        elif timeout:
            terminated, truncated, reason = False, True, "timeout"
        else:
            terminated = bool(base_terminated)
            truncated = bool(base_truncated)
            reason = "base_terminated" if terminated else "base_truncated" if truncated else None

        info = {
            "collision": collision,
            "elapsed_time": self._elapsed_time,
            "termination_reason": reason,
            "base_info": base_info,
        }
        if terminated or truncated:
            self.terminal_events.append(
                {
                    "transition_index": self._lifetime_steps - 1,
                    "reason": reason,
                    "observation": actor_observation.copy(),
                    "raw_reward": reward,
                    "elapsed_time": self._elapsed_time,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
        return actor_observation, reward, terminated, truncated, info

    def render(self) -> Any:
        return self.f110_env.render()

    def close(self) -> None:
        close = getattr(self.f110_env, "close", None)
        if close is not None:
            close()

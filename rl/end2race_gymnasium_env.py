"""Gymnasium integration for the legacy multi-agent F1Tenth environment.

Only the ego action belongs to PPO.  Opponent actions are produced by fixed,
episode-local controllers and never enter the actor observation or optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.sb3_end2race_policy import (
    END2RACE_LIDAR_SIZE,
    EVALUATOR_STEER_BOUND,
    NOOP_SPEED_BOUND,
    end2race_observation,
)


@dataclass
class EpisodeResetSpec:
    """Complete scenario information needed for one legacy F1Tenth reset."""

    poses: np.ndarray
    initial_speed_feature: float
    scenario: dict[str, Any]


EpisodeResetProvider = Callable[[np.random.Generator], EpisodeResetSpec]


class OpponentController(Protocol):
    """Non-learning opponent controller owned by one environment instance."""

    def reset(self, spec: EpisodeResetSpec, num_agents: int, ego_index: int) -> None: ...

    def actions(self, raw_observation: dict[str, Any]) -> np.ndarray: ...

    def state_snapshot(self) -> dict[str, Any]: ...


class LatticePlannerOpponentController:
    """Run fresh fixed Lattice Planners at the original evaluator frequency."""

    def __init__(self, planner_factory: Callable[[str, str], Any] | None = None) -> None:
        self._planner_factory = planner_factory
        self._planners: dict[int, Any] = {}
        self._trajectories: dict[int, np.ndarray | None] = {}
        self._tracker_counts: dict[int, int] = {}
        self._speed_scales: dict[int, float] = {}
        self._ego_index = 0
        self._num_agents = 0
        self.reset_count = 0
        self.action_history: list[np.ndarray] = []
        self.reset_snapshots: list[dict[str, Any]] = []

    def _create_planner(self, map_name: str, raceline: str) -> Any:
        if self._planner_factory is not None:
            return self._planner_factory(map_name, raceline)
        # Lazy import keeps the basic wrapper usable without importing the old
        # Gym stack until a real lattice-planner opponent is requested.
        from demonstration import setup_opp_planner

        return setup_opp_planner(map_name, raceline)

    @staticmethod
    def _per_opponent_value(
        scenario: dict[str, Any],
        singular_key: str,
        plural_key: str,
        opponent_index: int,
        opponent_position: int,
        default: Any,
    ) -> Any:
        if plural_key not in scenario:
            return scenario.get(singular_key, default)
        values = scenario[plural_key]
        if isinstance(values, dict):
            return values.get(opponent_index, values.get(str(opponent_index), default))
        return values[opponent_position]

    def reset(self, spec: EpisodeResetSpec, num_agents: int, ego_index: int) -> None:
        self._planners = {}
        self._trajectories = {}
        self._tracker_counts = {}
        self._speed_scales = {}
        self._ego_index = int(ego_index)
        self._num_agents = int(num_agents)
        scenario = dict(spec.scenario)
        map_name = str(scenario.get("map_name", "Austin"))
        opponent_indices = [index for index in range(num_agents) if index != ego_index]
        for position, opponent_index in enumerate(opponent_indices):
            raceline = str(
                self._per_opponent_value(
                    scenario,
                    "opp_raceline",
                    "opponent_racelines",
                    opponent_index,
                    position,
                    "raceline1",
                )
            )
            speed_scale = float(
                self._per_opponent_value(
                    scenario,
                    "opp_speedscale",
                    "opponent_speed_scales",
                    opponent_index,
                    position,
                    1.0,
                )
            )
            planner = self._create_planner(map_name, raceline)
            self._planners[opponent_index] = planner
            self._trajectories[opponent_index] = None
            self._tracker_counts[opponent_index] = 0
            self._speed_scales[opponent_index] = speed_scale
        self.reset_count += 1
        self.reset_snapshots.append(self.state_snapshot())

    def actions(self, raw_observation: dict[str, Any]) -> np.ndarray:
        from latticeplanner.utils import obsDict2oppoArray

        joint_actions = np.zeros((self._num_agents, 2), dtype=np.float32)
        for opponent_index, planner in self._planners.items():
            pose_x = float(np.asarray(raw_observation["poses_x"])[opponent_index])
            pose_y = float(np.asarray(raw_observation["poses_y"])[opponent_index])
            pose_theta = float(np.asarray(raw_observation["poses_theta"])[opponent_index])
            speed = float(np.asarray(raw_observation["linear_vels_x"])[opponent_index])
            if self._tracker_counts[opponent_index] == 0 or self._trajectories[opponent_index] is None:
                opponent_poses = obsDict2oppoArray(raw_observation, opponent_index)
                self._trajectories[opponent_index] = planner.plan(
                    pose_x,
                    pose_y,
                    pose_theta,
                    opponent_poses,
                    speed,
                )
            steering, desired_speed = planner.tracker.plan(
                pose_x,
                pose_y,
                pose_theta,
                speed,
                self._trajectories[opponent_index],
            )
            joint_actions[opponent_index] = (
                np.clip(steering, -EVALUATOR_STEER_BOUND, EVALUATOR_STEER_BOUND),
                desired_speed * self._speed_scales[opponent_index],
            )
            tracker_steps = int(planner.conf.tracker_steps)
            self._tracker_counts[opponent_index] = (self._tracker_counts[opponent_index] + 1) % tracker_steps
        self.action_history.append(joint_actions.copy())
        return joint_actions

    def state_snapshot(self) -> dict[str, Any]:
        planners: dict[int, dict[str, Any]] = {}
        for opponent_index, planner in self._planners.items():
            planners[opponent_index] = {
                "planner_identity": id(planner),
                "tracker_identity": id(planner.tracker),
                "tracker_previous_error": float(getattr(planner.tracker, "prev_error", 0.0)),
                "cached_trajectory": self._trajectories[opponent_index] is not None,
                "tracker_step_counter": self._tracker_counts[opponent_index],
                "planner_step_counter": int(getattr(planner, "step", 0)),
                "previous_opponent_pose_max_abs": float(np.max(np.abs(getattr(planner, "prev_opp_pose", 0.0)))),
                "previous_local_trajectory_max_abs": float(np.max(np.abs(getattr(planner, "prev_traj_local", 0.0)))),
            }
        return {"reset_count": self.reset_count, "planners": planners}


class End2RaceGymnasiumEnv(gym.Env[np.ndarray, np.ndarray]):
    """Convert legacy F1Tenth results to the exact ego deployment contract."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        f110_env: Any,
        sim_duration: float,
        reset_provider: EpisodeResetProvider,
        ego_index: int = 0,
        opponent_controller: OpponentController | None = None,
        transition_reward: Any | None = None,
    ) -> None:
        super().__init__()
        if sim_duration <= 0:
            raise ValueError("sim_duration must be positive")
        if not callable(reset_provider):
            raise TypeError("reset_provider must be callable")
        self.f110_env = f110_env
        self.sim_duration = float(sim_duration)
        self.ego_index = int(ego_index)
        self.reset_provider = reset_provider
        self.opponent_controller = opponent_controller
        self.transition_reward = transition_reward
        if transition_reward is not None and not all(
            callable(getattr(transition_reward, name, None)) for name in ("reset", "step")
        ):
            raise TypeError("transition_reward must provide callable reset() and step() methods")
        self.observation_space = spaces.Box(
            low=np.full((END2RACE_LIDAR_SIZE + 1,), -np.inf, dtype=np.float32),
            high=np.full((END2RACE_LIDAR_SIZE + 1,), np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.asarray((-EVALUATOR_STEER_BOUND, -NOOP_SPEED_BOUND), dtype=np.float32),
            high=np.asarray((EVALUATOR_STEER_BOUND, NOOP_SPEED_BOUND), dtype=np.float32),
            dtype=np.float32,
        )
        self._reset_rng = np.random.default_rng()
        self._elapsed_time = 0.0
        self._previous_ego_speed = 0.0
        self._raw_observation: dict[str, Any] | None = None
        self._lifetime_steps = 0
        self._episode_index = -1
        self._current_spec: EpisodeResetSpec | None = None
        self.terminal_events: list[dict[str, Any]] = []
        self.step_events: list[dict[str, Any]] = []
        self.action_trace: list[dict[str, Any]] = []
        self.reset_history: list[dict[str, Any]] = []

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
        scan = np.asarray(raw_observation["scans"][self.ego_index]).reshape(-1)
        if scan.size < END2RACE_LIDAR_SIZE:
            raise ValueError(f"LiDAR scan has {scan.size} beams; at least {END2RACE_LIDAR_SIZE} are required")
        if not np.isfinite(scan).all():
            raise ValueError("LiDAR scan contains NaN or Inf")
        if scan.size > END2RACE_LIDAR_SIZE:
            indices = np.linspace(0, scan.size - 1, END2RACE_LIDAR_SIZE, dtype=int)
            scan = scan[indices]
        return np.asarray(scan, dtype=np.float32)

    def _ego_speed(self, raw_observation: dict[str, Any]) -> float:
        speed = float(np.asarray(raw_observation["linear_vels_x"])[self.ego_index])
        if not np.isfinite(speed):
            raise ValueError("Ego measured speed must be finite")
        return speed

    def _actor_observation(self, raw_observation: dict[str, Any]) -> np.ndarray:
        if not np.isfinite(self._previous_ego_speed):
            raise ValueError("Previous ego speed feature must be finite")
        return end2race_observation(self._ego_lidar(raw_observation), self._previous_ego_speed)

    def _resolve_reset_spec(self, options: dict[str, Any] | None) -> EpisodeResetSpec:
        provided = self.reset_provider(self._reset_rng)
        if not isinstance(provided, EpisodeResetSpec):
            raise TypeError("reset_provider must return EpisodeResetSpec")
        poses = np.asarray(provided.poses, dtype=np.float64).copy()
        initial_speed_feature = float(provided.initial_speed_feature)
        scenario = dict(provided.scenario)
        if options:
            explicit = options.get("reset_spec")
            if explicit is not None:
                if not isinstance(explicit, EpisodeResetSpec):
                    raise TypeError("options['reset_spec'] must be EpisodeResetSpec")
                poses = np.asarray(explicit.poses, dtype=np.float64).copy()
                initial_speed_feature = float(explicit.initial_speed_feature)
                scenario = dict(explicit.scenario)
            else:
                if "poses" in options:
                    poses = np.asarray(options["poses"], dtype=np.float64).copy()
                if "initial_speed_feature" in options:
                    initial_speed_feature = float(options["initial_speed_feature"])
                if "scenario" in options:
                    scenario.update(dict(options["scenario"]))
        expected_shape = (self.num_agents, 3)
        if poses.shape != expected_shape:
            raise ValueError(f"Reset poses must have shape {expected_shape}, got {poses.shape}")
        if not np.isfinite(poses).all() or not np.isfinite(initial_speed_feature):
            raise ValueError("Reset poses and initial speed feature must be finite")
        return EpisodeResetSpec(poses=poses, initial_speed_feature=initial_speed_feature, scenario=scenario)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._reset_rng = np.random.default_rng(seed)
        spec = self._resolve_reset_spec(options)
        # Always use the real legacy API.  DummyVecEnv auto-reset therefore does
        # not need options and cannot accidentally omit poses.
        raw_observation, base_info = self._legacy_reset_result(self.f110_env.reset(poses=spec.poses.copy()))
        self._elapsed_time = 0.0
        self._raw_observation = raw_observation
        self._previous_ego_speed = float(spec.initial_speed_feature)
        self._current_spec = spec
        self._episode_index += 1
        scenario_id = str(spec.scenario.get("scenario_id", f"episode-{self._episode_index}"))
        if self.transition_reward is not None:
            self.transition_reward.reset(raw_observation, scenario_id=scenario_id, ego_index=self.ego_index)
        if self.num_agents > 1:
            if self.opponent_controller is None:
                raise RuntimeError("A fixed opponent controller is required for multi-agent F1Tenth")
            self.opponent_controller.reset(spec, self.num_agents, self.ego_index)
        reset_record = {
            "episode_index": self._episode_index,
            "poses": spec.poses.copy(),
            "initial_speed_feature": spec.initial_speed_feature,
            "scenario": dict(spec.scenario),
            "opponent_state": self.opponent_controller.state_snapshot() if self.opponent_controller else None,
        }
        self.reset_history.append(reset_record)
        info = {
            "ego_collision": False,
            "opponent_collision": False,
            "base_terminated": False,
            "base_truncated": False,
            "timeout": False,
            "elapsed_time": 0.0,
            "termination_reason": None,
            "scenario": dict(spec.scenario),
            "scenario_id": scenario_id,
            "sampler_branch": spec.scenario.get("sampler_branch"),
            "base_info": base_info,
        }
        return self._actor_observation(raw_observation), info

    def _joint_action(self, ego_action: np.ndarray) -> np.ndarray:
        ego_action = np.asarray(ego_action, dtype=np.float32).reshape(2)
        if not np.isfinite(ego_action).all():
            raise ValueError("Ego action must be finite")
        if abs(float(ego_action[0])) > EVALUATOR_STEER_BOUND + 1e-7:
            raise ValueError(f"Ego steering {ego_action[0]} is outside evaluator bounds")
        if self.num_agents == 1:
            return ego_action.reshape(1, 2)
        if self._raw_observation is None or self.opponent_controller is None:
            raise RuntimeError("Environment and opponent controller must be reset before step")
        joint_action = np.asarray(self.opponent_controller.actions(self._raw_observation), dtype=np.float32)
        if joint_action.shape != (self.num_agents, 2):
            raise ValueError(f"Opponent controller returned {joint_action.shape}, expected {(self.num_agents, 2)}")
        joint_action = joint_action.copy()
        joint_action[self.ego_index] = ego_action
        return joint_action

    def _step_duration(self, reward: float, info: dict[str, Any]) -> float:
        if "timestep" in info:
            return float(info["timestep"])
        unwrapped = getattr(self.f110_env, "unwrapped", self.f110_env)
        configured = getattr(unwrapped, "timestep", getattr(self.f110_env, "timestep", None))
        if configured is not None:
            return float(configured)
        if reward > 0:
            return float(reward)
        raise ValueError("Cannot infer simulation timestep; provide info['timestep']")

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._raw_observation is None:
            raise RuntimeError("Environment must be reset before step")
        if self._current_spec is None:
            raise RuntimeError("Environment has no active reset specification")
        # Deployment evaluator updates prev_speed from the decision observation
        # before stepping, then pairs it with the next LiDAR observation.
        previous_raw_observation = self._raw_observation
        pre_step_ego_speed = self._ego_speed(previous_raw_observation)
        joint_action = self._joint_action(action)
        result = self.f110_env.step(joint_action)
        raw_observation, simulator_reward, base_terminated, base_truncated, base_info = self._step_result(result)
        self._lifetime_steps += 1
        self._elapsed_time += self._step_duration(simulator_reward, base_info)

        collisions = np.asarray(raw_observation.get("collisions", []), dtype=bool).reshape(-1)
        ego_collision = bool(collisions.size > self.ego_index and collisions[self.ego_index])
        opponent_collision = bool(
            any(bool(collisions[index]) for index in range(collisions.size) if index != self.ego_index)
        )
        timeout = self._elapsed_time + 1e-12 >= self.sim_duration
        if ego_collision or base_terminated:
            terminated, truncated = True, False
            reason = "ego_collision" if ego_collision else "base_terminated"
        elif base_truncated or timeout:
            terminated, truncated = False, True
            reason = "base_truncated" if base_truncated else "timeout"
        else:
            terminated, truncated, reason = False, False, None

        scenario = dict(self._current_spec.scenario)
        scenario_id = str(scenario.get("scenario_id", f"episode-{self._episode_index}"))
        reward_info: dict[str, Any] = {}
        if self.transition_reward is None:
            reward = float(simulator_reward)
        else:
            reward_result = self.transition_reward.step(
                previous_raw_observation,
                raw_observation,
                ego_collision=ego_collision,
                opponent_collision=opponent_collision,
                scenario_id=scenario_id,
                ego_index=self.ego_index,
            )
            reward_info = dict(reward_result.to_info())
            reward = float(reward_info["reward_total"])

        self._raw_observation = raw_observation
        self._previous_ego_speed = pre_step_ego_speed
        actor_observation = self._actor_observation(raw_observation)

        info = {
            "ego_collision": ego_collision,
            "opponent_collision": opponent_collision,
            "base_terminated": bool(base_terminated),
            "base_truncated": bool(base_truncated),
            "timeout": timeout,
            "elapsed_time": self._elapsed_time,
            "termination_reason": reason,
            "scenario": scenario,
            "scenario_id": scenario_id,
            "sampler_branch": scenario.get("sampler_branch"),
            "simulator_reward": float(simulator_reward),
            "base_info": base_info,
            **reward_info,
        }
        event = {
            "transition_index": self._lifetime_steps - 1,
            "episode_index": self._episode_index,
            "reason": reason,
            "observation": actor_observation.copy(),
            "raw_reward": simulator_reward,
            "reward": reward,
            "elapsed_time": self._elapsed_time,
            "terminated": terminated,
            "truncated": truncated,
            **{key: info[key] for key in ("ego_collision", "opponent_collision", "base_terminated", "base_truncated", "timeout")},
        }
        self.step_events.append(event)
        if terminated or truncated:
            self.terminal_events.append(event.copy())
        self.action_trace.append(
            {
                "transition_index": self._lifetime_steps - 1,
                "episode_index": self._episode_index,
                "ego_action": np.asarray(action, dtype=np.float32).reshape(2).copy(),
                "joint_action": joint_action.copy(),
                "opponent_actions": np.delete(joint_action, self.ego_index, axis=0),
            }
        )
        return actor_observation, float(reward), terminated, truncated, info

    def render(self) -> Any:
        return self.f110_env.render()

    def close(self) -> None:
        close = getattr(self.f110_env, "close", None)
        if close is not None:
            close()

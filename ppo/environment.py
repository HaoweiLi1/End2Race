"""Gymnasium integration for the legacy multi-agent F1Tenth environment.

Only the ego action belongs to PPO.  Opponent actions are produced by fixed,
episode-local controllers and never enter the actor observation or optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ppo.policy import (
    END2RACE_LIDAR_SIZE,
    EVALUATOR_STEER_BOUND,
    NOOP_SPEED_BOUND,
    end2race_observation,
)
from ppo.reward import ProgressProjector, wrapped_progress_delta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUSTIN_DIRECTORY = PROJECT_ROOT / "f1tenth_racetracks" / "Austin"
VEHICLE_LENGTH_M = 0.58
VEHICLE_WIDTH_M = 0.31


def _rectangle_vertices(x: float, y: float, heading: float) -> np.ndarray:
    local = np.asarray(
        [
            [VEHICLE_LENGTH_M / 2, VEHICLE_WIDTH_M / 2],
            [VEHICLE_LENGTH_M / 2, -VEHICLE_WIDTH_M / 2],
            [-VEHICLE_LENGTH_M / 2, -VEHICLE_WIDTH_M / 2],
            [-VEHICLE_LENGTH_M / 2, VEHICLE_WIDTH_M / 2],
        ],
        dtype=np.float64,
    )
    rotation = np.asarray([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    return local @ rotation.T + np.asarray([x, y])


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    denominator = float(np.dot(vector, vector))
    fraction = 0.0 if denominator == 0.0 else float(np.clip(np.dot(point - start, vector) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * vector)))


def oriented_rectangle_clearance(first_pose: np.ndarray, second_pose: np.ndarray) -> float:
    """Return zero for overlap, otherwise the exact minimum edge distance."""

    first = _rectangle_vertices(*map(float, first_pose))
    second = _rectangle_vertices(*map(float, second_pose))
    for polygon in (first, second):
        for edge_index in range(4):
            edge = polygon[(edge_index + 1) % 4] - polygon[edge_index]
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float64)
            axis /= np.linalg.norm(axis)
            projection_first = first @ axis
            projection_second = second @ axis
            if projection_first.max() < projection_second.min() or projection_second.max() < projection_first.min():
                break
        else:
            continue
        break
    else:
        return 0.0
    distances = []
    for polygon_a, polygon_b in ((first, second), (second, first)):
        for point in polygon_a:
            for index in range(4):
                distances.append(_point_segment_distance(point, polygon_b[index], polygon_b[(index + 1) % 4]))
    return float(min(distances))


@dataclass(frozen=True)
class PrivilegedFeatureManifest:
    curvature_scale: float
    curvature_statistic: str = "p95_abs_austin_raceline1_unique"
    vehicle_length_m: float = VEHICLE_LENGTH_M
    vehicle_width_m: float = VEHICLE_WIDTH_M

    def to_dict(self) -> dict[str, float | str]:
        return {
            "curvature_scale": self.curvature_scale,
            "curvature_statistic": self.curvature_statistic,
            "vehicle_length_m": self.vehicle_length_m,
            "vehicle_width_m": self.vehicle_width_m,
        }


class AustinPrivilegedFeatureExtractor:
    """Build the fixed 12D feature from one current pre-action simulator state."""

    def __init__(self) -> None:
        center = np.loadtxt(AUSTIN_DIRECTORY / "raceline1.csv", delimiter=";", comments="#", dtype=np.float64)
        inner = np.loadtxt(AUSTIN_DIRECTORY / "raceline0.csv", delimiter=";", comments="#", dtype=np.float64)
        outer = np.loadtxt(AUSTIN_DIRECTORY / "raceline2.csv", delimiter=";", comments="#", dtype=np.float64)
        self.center = center[:-1]
        self.inner_xy = inner[:-1, 1:3]
        self.outer_xy = outer[:-1, 1:3]
        self.projector = ProgressProjector(center[:, 0], center[:, 1:3], float(center[-1, 0]))
        scale = float(np.percentile(np.abs(self.center[:, 4]), 95.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Austin curvature scale must be finite and positive")
        self.manifest = PrivilegedFeatureManifest(curvature_scale=scale)

    @staticmethod
    def _array(raw: dict[str, Any], name: str) -> np.ndarray:
        return np.asarray(raw[name], dtype=np.float64).reshape(-1)

    def __call__(self, raw: dict[str, Any], ego_index: int = 0) -> np.ndarray:
        if ego_index != 0:
            raise ValueError("Privileged physical features require ego index 0")
        opponent_index = 1
        x = self._array(raw, "poses_x")
        y = self._array(raw, "poses_y")
        heading = self._array(raw, "poses_theta")
        speed = self._array(raw, "linear_vels_x")
        yaw_rate = self._array(raw, "ang_vels_z")

        ego_xy = np.asarray([x[ego_index], y[ego_index]])
        opponent_xy = np.asarray([x[opponent_index], y[opponent_index]])
        ego_progress = self.projector.project(ego_xy)
        opponent_progress = self.projector.project(opponent_xy)
        relative_progress = wrapped_progress_delta(opponent_progress, ego_progress, self.projector.track_length)

        delta_xy = opponent_xy - ego_xy
        cosine, sine = np.cos(heading[ego_index]), np.sin(heading[ego_index])
        relative_longitudinal = cosine * delta_xy[0] + sine * delta_xy[1]
        relative_lateral = -sine * delta_xy[0] + cosine * delta_xy[1]
        ego_velocity = speed[ego_index] * np.asarray([cosine, sine])
        opponent_velocity = speed[opponent_index] * np.asarray([np.cos(heading[opponent_index]), np.sin(heading[opponent_index])])
        relative_velocity = opponent_velocity - ego_velocity
        relative_longitudinal_velocity = cosine * relative_velocity[0] + sine * relative_velocity[1]
        relative_lateral_velocity = -sine * relative_velocity[0] + cosine * relative_velocity[1]
        relative_heading = heading[opponent_index] - heading[ego_index]

        center_index = int(np.argmin(np.linalg.norm(self.center[:, 1:3] - ego_xy, axis=1)))
        center_xy = self.center[center_index, 1:3]
        track_heading = float(self.center[center_index, 3])
        normal = np.asarray([-np.sin(track_heading), np.cos(track_heading)])
        signed_offset = float(np.dot(ego_xy - center_xy, normal))
        inner_distance = float(np.min(np.linalg.norm(self.inner_xy - center_xy, axis=1)))
        outer_distance = float(np.min(np.linalg.norm(self.outer_xy - center_xy, axis=1)))
        local_half_width = max(0.5 * (inner_distance + outer_distance), 1e-6)
        normalized_lateral_offset = signed_offset / local_half_width
        curvature = float(self.center[center_index, 4])
        clearance = oriented_rectangle_clearance(
            np.asarray([x[ego_index], y[ego_index], heading[ego_index]]),
            np.asarray([x[opponent_index], y[opponent_index], heading[opponent_index]]),
        )

        features = np.asarray(
            [
                np.clip(relative_progress / 10.0, -1.0, 1.0),
                np.clip(relative_lateral / 2.0, -1.0, 1.0),
                np.clip(relative_longitudinal_velocity / 10.0, -1.0, 1.0),
                np.clip(relative_lateral_velocity / 5.0, -1.0, 1.0),
                np.sin(relative_heading),
                np.cos(relative_heading),
                np.clip(speed[ego_index] / 10.0, 0.0, 1.0),
                np.clip(yaw_rate[ego_index] / 5.0, -1.0, 1.0),
                np.clip(yaw_rate[opponent_index] / 5.0, -1.0, 1.0),
                np.clip(clearance / 2.0, 0.0, 1.0),
                np.clip(normalized_lateral_offset, -1.0, 1.0),
                np.tanh(curvature / self.manifest.curvature_scale),
            ],
            dtype=np.float32,
        )
        return features


@dataclass
class EpisodeResetSpec:
    """Complete scenario information needed for one legacy F1Tenth reset."""

    poses: np.ndarray
    initial_speed_feature: float
    scenario: dict[str, Any]


EpisodeResetProvider = Callable[[np.random.Generator], EpisodeResetSpec]


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
    ) -> Any:
        if plural_key not in scenario:
            return scenario[singular_key]
        values = scenario[plural_key]
        if isinstance(values, dict):
            if opponent_index in values:
                return values[opponent_index]
            return values[str(opponent_index)]
        return values[opponent_position]

    def reset(self, spec: EpisodeResetSpec, num_agents: int, ego_index: int) -> None:
        self._planners = {}
        self._trajectories = {}
        self._tracker_counts = {}
        self._speed_scales = {}
        self._ego_index = int(ego_index)
        self._num_agents = int(num_agents)
        scenario = dict(spec.scenario)
        map_name = str(scenario["map_name"])
        opponent_indices = [index for index in range(num_agents) if index != ego_index]
        for position, opponent_index in enumerate(opponent_indices):
            raceline = str(
                self._per_opponent_value(
                    scenario,
                    "opp_raceline",
                    "opponent_racelines",
                    opponent_index,
                    position,
                )
            )
            speed_scale = float(
                self._per_opponent_value(
                    scenario,
                    "opp_speedscale",
                    "opponent_speed_scales",
                    opponent_index,
                    position,
                )
            )
            planner = self._create_planner(map_name, raceline)
            self._planners[opponent_index] = planner
            self._trajectories[opponent_index] = None
            self._tracker_counts[opponent_index] = 0
            self._speed_scales[opponent_index] = speed_scale

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
        return joint_actions


class End2RaceGymnasiumEnv(gym.Env):
    """Convert legacy F1Tenth results to the exact ego deployment contract."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        f110_env: Any,
        sim_duration: float,
        reset_provider: EpisodeResetProvider,
        ego_index: int = 0,
        opponent_controller: Any | None = None,
        transition_reward: Any | None = None,
        privileged_critic: bool = False,
        privileged_feature_extractor: AustinPrivilegedFeatureExtractor | None = None,
    ) -> None:
        super().__init__()
        self.f110_env = f110_env
        self.sim_duration = float(sim_duration)
        self.ego_index = int(ego_index)
        self.reset_provider = reset_provider
        self.opponent_controller = opponent_controller
        self.transition_reward = transition_reward
        self.privileged_critic = bool(privileged_critic)
        self.privileged_feature_extractor = (
            privileged_feature_extractor
            if privileged_feature_extractor is not None
            else (AustinPrivilegedFeatureExtractor() if self.privileged_critic else None)
        )
        actor_space = spaces.Box(
            low=np.full((END2RACE_LIDAR_SIZE + 1,), -np.inf, dtype=np.float32),
            high=np.full((END2RACE_LIDAR_SIZE + 1,), np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = (
            spaces.Dict(
                {
                    "actor": actor_space,
                    "critic": spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32),
                }
            )
            if self.privileged_critic
            else actor_space
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
        self._current_spec: EpisodeResetSpec | None = None

    @property
    def num_agents(self) -> int:
        return int(getattr(self.f110_env, "unwrapped", self.f110_env).num_agents)

    def _ego_lidar(self, raw_observation: dict[str, Any]) -> np.ndarray:
        scan = np.asarray(raw_observation["scans"][self.ego_index]).reshape(-1)
        if scan.size > END2RACE_LIDAR_SIZE:
            indices = np.linspace(0, scan.size - 1, END2RACE_LIDAR_SIZE, dtype=int)
            scan = scan[indices]
        return np.asarray(scan, dtype=np.float32)

    def _ego_speed(self, raw_observation: dict[str, Any]) -> float:
        return float(np.asarray(raw_observation["linear_vels_x"])[self.ego_index])

    def _actor_observation(self, raw_observation: dict[str, Any]) -> np.ndarray:
        return end2race_observation(self._ego_lidar(raw_observation), self._previous_ego_speed)

    def _observation(self, raw_observation: dict[str, Any]) -> np.ndarray | dict[str, np.ndarray]:
        actor = self._actor_observation(raw_observation)
        if not self.privileged_critic:
            return actor
        return {"actor": actor, "critic": self.privileged_feature_extractor(raw_observation, self.ego_index)}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._reset_rng = np.random.default_rng(seed)
        # The fixed sampler fully specifies every reset, so DummyVecEnv auto-reset
        # never needs options and cannot accidentally omit poses.
        spec = self.reset_provider(self._reset_rng)
        raw_observation, _, _, base_info = self.f110_env.reset(poses=spec.poses.copy())
        self._elapsed_time = 0.0
        self._raw_observation = raw_observation
        self._previous_ego_speed = float(spec.initial_speed_feature)
        self._current_spec = spec
        scenario_id = str(spec.scenario["scenario_id"])
        if self.transition_reward is not None:
            self.transition_reward.reset(raw_observation, scenario_id=scenario_id, ego_index=self.ego_index)
        if self.num_agents > 1:
            self.opponent_controller.reset(spec, self.num_agents, self.ego_index)
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
            "sampler_branch": spec.scenario["sampler_branch"],
            "hard_pool_id": spec.scenario["hard_pool_id"],
            "hard_sampling_mode": spec.scenario["hard_sampling_mode"],
            "base_info": base_info,
        }
        return self._observation(raw_observation), info

    def _joint_action(self, ego_action: np.ndarray) -> np.ndarray:
        ego_action = np.asarray(ego_action, dtype=np.float32).reshape(2)
        if self.num_agents == 1:
            return ego_action.reshape(1, 2)
        joint_action = np.asarray(self.opponent_controller.actions(self._raw_observation), dtype=np.float32).copy()
        joint_action[self.ego_index] = ego_action
        return joint_action

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Deployment evaluator updates prev_speed from the decision observation
        # before stepping, then pairs it with the next LiDAR observation.
        previous_raw_observation = self._raw_observation
        pre_step_ego_speed = self._ego_speed(previous_raw_observation)
        joint_action = self._joint_action(action)
        # The legacy F1Tenth step returns (obs, reward, done, info); its reward is
        # the physics timestep, so it doubles as the elapsed-time increment.
        raw_observation, simulator_reward, base_terminated, base_info = self.f110_env.step(joint_action)
        base_truncated = False
        self._elapsed_time += float(simulator_reward)

        collisions = np.asarray(raw_observation["collisions"], dtype=bool).reshape(-1)
        ego_collision = bool(collisions[self.ego_index])
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
        scenario_id = str(scenario["scenario_id"])
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
        observation = self._observation(raw_observation)

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
            "sampler_branch": scenario["sampler_branch"],
            "hard_pool_id": scenario["hard_pool_id"],
            "hard_sampling_mode": scenario["hard_sampling_mode"],
            "simulator_reward": float(simulator_reward),
            "base_info": base_info,
            **reward_info,
        }
        return observation, float(reward), terminated, truncated, info

    def render(self) -> Any:
        return self.f110_env.render()

    def close(self) -> None:
        self.f110_env.close()

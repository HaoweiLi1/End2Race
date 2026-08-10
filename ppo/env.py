"""End2Race environment and vector-environment execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import copy as copy_module
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import traceback
from typing import Any
import warnings

from gym_notices import notices as gym_notices

gym_notices.notices.clear()

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.spatial import cKDTree
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from threadpoolctl import threadpool_limits
import torch

from latticeplanner.utils import load_config
from ppo.policy import (
    END2RACE_LIDAR_SIZE,
    END2RACE_OBSERVATION_SIZE,
    NOOP_SPEED_BOUND,
    PRIVILEGED_FEATURE_SIZE,
    PrivilegedStateExtractor,
    end2race_observation,
)
from ppo.reward import OccupancyMapClearance, PPOTransitionReward
from ppo.scenarios import EpisodeResetSpec, ScenarioScheduler, ScenarioSpec

CONFIG = load_config("ppo/ppo_config.yaml")

EGO_INDEX = 0
OPPONENT_INDEX = 1
NUM_AGENTS = 2
EXTERNAL_RESET_OPTION = "end2race_episode_reset_spec"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PLANNER_TEMPLATE_CACHE: dict[tuple[str, str], Any] = {}
RUNTIME_SNAPSHOT_REWARD_FIELDS = (
    "_previous_ego_progress",
    "_previous_opponent_progress",
    "_relative_position_m",
    "_opponent_collision_latched",
    "_ego_collision_penalty_applied",
    "_scenario_id",
    "_previous_risk_potential",
    "current_clearances",
)
RUNTIME_SNAPSHOT_WRAPPER_FIELDS = (
    "_elapsed_time",
    "_previous_ego_speed",
    "_raw_observation",
    "_current_spec",
    "_episode_return",
    "_episode_steps",
)
RUNTIME_SNAPSHOT_CORE_FIELDS = (
    "poses_x",
    "poses_y",
    "poses_theta",
    "collisions",
    "near_start",
    "num_toggles",
    "lap_times",
    "lap_counts",
    "current_time",
    "near_starts",
    "toggle_list",
    "start_xs",
    "start_ys",
    "start_thetas",
    "start_rot",
    "render_obs",
)
RUNTIME_SNAPSHOT_PLANNER_FIELDS = (
    "best_traj",
    "best_traj_ref_v",
    "best_traj_idx",
    "prev_traj_local",
    "prev_opp_pose",
    "goal_grid",
    "state_i",
    "state_t",
    "step_all_cost",
    "all_costs",
    "last_s",
    "step",
)


@dataclass(frozen=True)
class FrontCorridorGateConfig:
    maximum_front_gap_m: float = CONFIG.front_corridor_gate_maximum_gap_m
    maximum_abs_opponent_lateral_d_m: float = CONFIG.front_corridor_gate_maximum_abs_opponent_lateral_d_m
    require_positive_lateral_overlap: bool = CONFIG.front_corridor_gate_require_positive_lateral_overlap

    def validate(self) -> None:
        values = np.asarray(
            (self.maximum_front_gap_m, self.maximum_abs_opponent_lateral_d_m),
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("Front-corridor gate thresholds must be finite and positive")
        if not self.require_positive_lateral_overlap:
            raise ValueError("Front-corridor temporal exploration requires positive lateral OBB overlap")


@dataclass(frozen=True)
class _Projection:
    progress_m: float
    lateral_d_m: float
    tangent_heading_rad: float


class _FrenetProjector:

    def __init__(self, path: Path) -> None:
        reference = np.loadtxt(path, delimiter=";", comments="#", dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] < 3:
            raise ValueError(f"Invalid raceline CSV: {path}")
        self.track_length_m = float(reference[-1, 0])
        progress = reference[:, 0]
        points = reference[:, 1:3]
        if np.linalg.norm(points[-1] - points[0]) <= 1e-9:
            progress = progress[:-1]
            points = points[:-1]
        if (
            len(points) < 3
            or not np.isfinite(points).all()
            or not np.isfinite(progress).all()
            or np.any(np.diff(progress) <= 0.0)
            or self.track_length_m <= progress[-1]
        ):
            raise ValueError(f"Invalid cyclic raceline geometry: {path}")
        self.progress_m = progress
        self.points_xy = points
        self.segment_xy = np.roll(points, -1, axis=0) - points
        self.segment_norm_sq = np.einsum("ij,ij->i", self.segment_xy, self.segment_xy)
        if np.any(self.segment_norm_sq <= 0.0):
            raise ValueError("Raceline contains a zero-length segment")
        self.segment_length_m = np.sqrt(self.segment_norm_sq)
        self.segment_progress_m = np.concatenate((np.diff(progress), np.asarray([self.track_length_m - progress[-1]])))
        self.tree = cKDTree(points)

    def project(self, point_xy: np.ndarray) -> _Projection:
        point = np.asarray(point_xy, dtype=np.float64).reshape(-1)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Projection point must be one finite XY pair")
        _distance, nearest = self.tree.query(point)
        candidates = np.asarray((nearest, (nearest - 1) % len(self.points_xy)), dtype=np.int64)
        starts = self.points_xy[candidates]
        vectors = self.segment_xy[candidates]
        offsets = point - starts
        fractions = np.clip(
            np.einsum("ci,ci->c", offsets, vectors) / self.segment_norm_sq[candidates],
            0.0,
            1.0,
        )
        closest = starts + fractions[:, None] * vectors
        distance_sq = np.einsum("ci,ci->c", point - closest, point - closest)
        choice = int(np.argmin(distance_sq))
        segment = int(candidates[choice])
        fraction = float(fractions[choice])
        tangent = self.segment_xy[segment] / self.segment_length_m[segment]
        normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
        progress = (self.progress_m[segment] + fraction * self.segment_progress_m[segment]) % self.track_length_m
        lateral = float(np.dot(point - closest[choice], normal))
        heading = float(np.arctan2(tangent[1], tangent[0]))
        return _Projection(float(progress), lateral, heading)


def _wrap_angle(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


class FrontCorridorGate:

    def __init__(
        self,
        map_name: str,
        ego_raceline: str,
        *,
        vehicle_length_m: float,
        vehicle_width_m: float,
        config: FrontCorridorGateConfig | None = None,
    ) -> None:
        self.config = config or FrontCorridorGateConfig()
        self.config.validate()
        dimensions = np.asarray((vehicle_length_m, vehicle_width_m), dtype=np.float64)
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0.0):
            raise ValueError("Vehicle dimensions must be finite and positive")
        self.vehicle_length_m = float(vehicle_length_m)
        self.vehicle_width_m = float(vehicle_width_m)
        self.projector = _FrenetProjector(
            PROJECT_ROOT / "f1tenth_racetracks" / map_name / f"{ego_raceline}.csv"
        )
        self.current_gate = False

    @staticmethod
    def _position(raw_observation: dict[str, Any], index: int) -> np.ndarray:
        return np.asarray(
            (
                np.asarray(raw_observation["poses_x"])[index],
                np.asarray(raw_observation["poses_y"])[index],
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _heading(raw_observation: dict[str, Any], index: int) -> float:
        return float(np.asarray(raw_observation["poses_theta"])[index])

    def _evaluate(self, raw_observation: dict[str, Any], *, ego_index: int, opponent_index: int) -> bool:
        ego = self.projector.project(self._position(raw_observation, ego_index))
        opponent = self.projector.project(self._position(raw_observation, opponent_index))
        raw_relative_m = float(
            (ego.progress_m - opponent.progress_m + 0.5 * self.projector.track_length_m)
            % self.projector.track_length_m
            - 0.5 * self.projector.track_length_m
        )
        opponent_ahead_center_m = -raw_relative_m

        def extents(heading_error: float) -> tuple[float, float]:
            cosine = abs(float(np.cos(heading_error)))
            sine = abs(float(np.sin(heading_error)))
            return (
                0.5 * (self.vehicle_length_m * cosine + self.vehicle_width_m * sine),
                0.5 * (self.vehicle_length_m * sine + self.vehicle_width_m * cosine),
            )

        ego_longitudinal, ego_lateral = extents(
            _wrap_angle(self._heading(raw_observation, ego_index) - ego.tangent_heading_rad)
        )
        opponent_longitudinal, opponent_lateral = extents(
            _wrap_angle(self._heading(raw_observation, opponent_index) - opponent.tangent_heading_rad)
        )
        front_gap_m = opponent_ahead_center_m - ego_longitudinal - opponent_longitudinal
        ego_low = ego.lateral_d_m - ego_lateral
        ego_high = ego.lateral_d_m + ego_lateral
        opponent_low = opponent.lateral_d_m - opponent_lateral
        opponent_high = opponent.lateral_d_m + opponent_lateral
        lateral_overlap_m = min(ego_high, opponent_high) - max(ego_low, opponent_low)
        self.current_gate = bool(
            opponent_ahead_center_m > 0.0
            and opponent_ahead_center_m < 0.5 * self.projector.track_length_m
            and front_gap_m > 0.0
            and front_gap_m < self.config.maximum_front_gap_m
            and abs(opponent.lateral_d_m) < self.config.maximum_abs_opponent_lateral_d_m
            and lateral_overlap_m > 0.0
        )
        return self.current_gate

    def reset(
        self,
        raw_observation: dict[str, Any],
        *,
        elapsed_time_s: float = 0.0,
        ego_index: int = 0,
        opponent_index: int = 1,
    ) -> bool:
        del elapsed_time_s
        self.current_gate = False
        return self._evaluate(raw_observation, ego_index=ego_index, opponent_index=opponent_index)

    def step(
        self,
        raw_observation: dict[str, Any],
        *,
        elapsed_time_s: float,
        ego_index: int = 0,
        opponent_index: int = 1,
    ) -> bool:
        del elapsed_time_s
        return self._evaluate(raw_observation, ego_index=ego_index, opponent_index=opponent_index)


class LatticePlannerOpponentController:

    def __init__(self) -> None:
        self.planner = None
        self.trajectory = None
        self.tracker_count = 0
        self.speed_scale = 1.0

    def _create_planner(self, map_name: str, raceline: str) -> Any:
        from demonstration import setup_opp_planner

        key = (map_name, raceline)
        template = _PLANNER_TEMPLATE_CACHE.get(key)
        if template is None:
            template = setup_opp_planner(map_name, raceline)
            _PLANNER_TEMPLATE_CACHE[key] = template
        planner = copy_module.copy(template)
        planner.best_traj = None
        planner.best_traj_ref_v = 0.0
        planner.best_traj_idx = 0
        planner.prev_traj_local = np.zeros((planner.traj_points, 2))
        planner.prev_opp_pose = np.array([0, 0])
        planner.goal_grid = None
        planner.state_i = None
        planner.state_t = None
        planner.step_all_cost = {}
        planner.all_costs = None
        planner.last_s = 0.0
        planner.selection_func = None
        planner.step = 0
        planner.tracker = copy_module.copy(template.tracker)
        planner.tracker.drawn_waypoints = []
        planner.tracker.prev_error = 0.0
        return planner

    def reset(self, spec: EpisodeResetSpec) -> None:
        self.planner = self._create_planner(str(spec.scenario["map_name"]), str(spec.scenario["opp_raceline"]))
        self.trajectory = None
        self.tracker_count = 0
        self.speed_scale = float(spec.scenario["opp_speedscale"])

    def action(self, raw_observation: dict[str, Any]) -> np.ndarray:
        from latticeplanner.utils import obsDict2oppoArray

        pose_x = float(np.asarray(raw_observation["poses_x"])[OPPONENT_INDEX])
        pose_y = float(np.asarray(raw_observation["poses_y"])[OPPONENT_INDEX])
        pose_theta = float(np.asarray(raw_observation["poses_theta"])[OPPONENT_INDEX])
        speed = float(np.asarray(raw_observation["linear_vels_x"])[OPPONENT_INDEX])
        if self.tracker_count == 0 or self.trajectory is None:
            opponent_poses = obsDict2oppoArray(raw_observation, OPPONENT_INDEX)
            self.trajectory = self.planner.plan(pose_x, pose_y, pose_theta, opponent_poses, speed)
        steering, desired_speed = self.planner.tracker.plan(
            pose_x, pose_y, pose_theta, speed, self.trajectory
        )
        self.tracker_count = (self.tracker_count + 1) % int(self.planner.conf.tracker_steps)
        return np.asarray(
            (np.clip(steering, -CONFIG.steering_bound, CONFIG.steering_bound), desired_speed * self.speed_scale),
            dtype=np.float32,
        )


class End2RaceGymnasiumEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        f110_env: Any,
        reset_provider: Callable[[np.random.Generator], EpisodeResetSpec],
        map_name: str,
        ego_raceline: str,
        privileged: bool = False,
        reward_gamma: float = 0.999,
        speed_exploration_mode: str = "baseline",
    ) -> None:
        super().__init__()
        self.f110_env = f110_env
        self.reset_provider = reset_provider
        self.opponent_controller = LatticePlannerOpponentController()
        core = f110_env.unwrapped
        core_params = core.params
        vehicle_length = float(core_params["length"])
        vehicle_width = float(core_params["width"])
        scan_simulator = core.sim.agents[EGO_INDEX].scan_simulator
        map_clearance = OccupancyMapClearance(
            scan_simulator.dt,
            scan_simulator.map_resolution,
            scan_simulator.origin,
        )
        self.transition_reward = PPOTransitionReward(
            map_name,
            ego_raceline,
            gamma=reward_gamma,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            map_clearance=map_clearance,
            risk_longitudinal_clearance_m=CONFIG.risk_longitudinal_clearance_m,
            risk_lateral_clearance_m=CONFIG.risk_lateral_clearance_m,
            risk_wall_clearance_m=CONFIG.risk_wall_clearance_m,
            risk_potential_maximum=CONFIG.risk_potential_maximum,
        )
        if speed_exploration_mode == "corridor_temporal":
            corridor_config = FrontCorridorGateConfig(
                maximum_front_gap_m=float(
                    CONFIG.front_corridor_gate_maximum_gap_m
                )
            )
            self.corridor_gate_config = corridor_config
            self.following_danger_gate = FrontCorridorGate(
                map_name,
                ego_raceline,
                vehicle_length_m=vehicle_length,
                vehicle_width_m=vehicle_width,
                config=corridor_config,
            )
        else:
            self.following_danger_gate = None
        if privileged:
            self.privileged_extractor = PrivilegedStateExtractor(
                map_name,
                ego_raceline,
                self.transition_reward.projector,
                vehicle_length,
                vehicle_width,
                steering_min_rad=float(core_params["s_min"]),
                steering_max_rad=float(core_params["s_max"]),
            )
        else:
            self.privileged_extractor = None
        observation_size = END2RACE_OBSERVATION_SIZE + (PRIVILEGED_FEATURE_SIZE if privileged else 0)
        self.observation_space = spaces.Box(
            low=np.full((observation_size,), -np.inf, dtype=np.float32),
            high=np.full((observation_size,), np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.asarray((-CONFIG.steering_bound, -NOOP_SPEED_BOUND), dtype=np.float32),
            high=np.asarray((CONFIG.steering_bound, NOOP_SPEED_BOUND), dtype=np.float32),
            dtype=np.float32,
        )
        self._reset_rng = np.random.default_rng()
        self._elapsed_time = 0.0
        self._previous_ego_speed = 0.0
        self._raw_observation = None
        self._current_spec = None
        self._episode_return = 0.0
        self._episode_steps = 0

    def _ego_lidar(self, raw_observation: dict[str, Any]) -> np.ndarray:
        scan = np.asarray(raw_observation["scans"][EGO_INDEX]).reshape(-1)
        if scan.size > END2RACE_LIDAR_SIZE:
            scan = scan[np.linspace(0, scan.size - 1, END2RACE_LIDAR_SIZE, dtype=int)]
        return np.asarray(scan, dtype=np.float32)

    def _ego_speed(self, raw_observation: dict[str, Any]) -> float:
        return float(np.asarray(raw_observation["linear_vels_x"])[EGO_INDEX])

    def _privileged_physical_state(self) -> tuple[float, float, float]:
        agents = self.f110_env.unwrapped.sim.agents
        if len(agents) != NUM_AGENTS:
            raise RuntimeError("Privileged critic requires simulator agent states")
        ego_state = np.asarray(agents[EGO_INDEX].state, dtype=np.float64).reshape(-1)
        opponent_state = np.asarray(agents[OPPONENT_INDEX].state, dtype=np.float64).reshape(-1)
        if ego_state.size <= 6 or opponent_state.size <= 6:
            raise RuntimeError("Privileged critic requires steering and slip in simulator agent states")
        physical_state = (float(ego_state[2]), float(ego_state[6]), float(opponent_state[6]))
        if not np.isfinite(physical_state).all():
            raise ValueError("Simulator steering and slip angles must be finite")
        return physical_state

    def _observation(self, raw_observation: dict[str, Any]) -> np.ndarray:
        observation = end2race_observation(self._ego_lidar(raw_observation), self._previous_ego_speed)
        if self.privileged_extractor is None:
            return observation
        ego_steering_angle, ego_slip_angle, opponent_slip_angle = self._privileged_physical_state()
        if self.transition_reward.current_clearances is None:
            raise RuntimeError("Reward current-state geometry must exist before privileged observation")
        features = self.privileged_extractor.features(
            raw_observation,
            ego_index=EGO_INDEX,
            opponent_index=OPPONENT_INDEX,
            ego_steering_angle=ego_steering_angle,
            ego_slip_angle=ego_slip_angle,
            opponent_slip_angle=opponent_slip_angle,
            clearances=self.transition_reward.current_clearances,
        )
        return np.concatenate((observation, features))

    def privileged_normalization_metadata(self) -> dict[str, Any]:
        if self.privileged_extractor is None:
            return {}
        return self.privileged_extractor.normalization_metadata()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._reset_rng = np.random.default_rng(seed)
        spec = None if options is None else options.get(EXTERNAL_RESET_OPTION)
        if spec is not None and not isinstance(spec, EpisodeResetSpec):
            raise TypeError(f"{EXTERNAL_RESET_OPTION} must contain an EpisodeResetSpec")
        if spec is None:
            spec = self.reset_provider(self._reset_rng)
        raw_observation, _, _, _ = self.f110_env.reset(poses=spec.poses.copy())
        if int(self.f110_env.unwrapped.num_agents) != NUM_AGENTS:
            raise RuntimeError("PPO environment requires exactly two agents")
        self._elapsed_time = 0.0
        self._episode_return = 0.0
        self._episode_steps = 0
        self._raw_observation = raw_observation
        self._previous_ego_speed = float(spec.initial_speed_feature)
        self._current_spec = spec
        scenario_id = str(spec.scenario["scenario_id"])
        self.transition_reward.reset(raw_observation, scenario_id=scenario_id, ego_index=EGO_INDEX)
        if self.following_danger_gate is not None:
            self.following_danger_gate.reset(
                raw_observation,
                elapsed_time_s=0.0,
                ego_index=EGO_INDEX,
                opponent_index=OPPONENT_INDEX,
            )
        self.opponent_controller.reset(spec)
        info = self._info(False, False, False, False, None, None)
        return self._observation(raw_observation), info

    @staticmethod
    def _copied_fields(obj: Any, names: Sequence[str]) -> dict[str, Any]:
        return {name: copy_module.deepcopy(getattr(obj, name)) for name in names}

    def _capture_opponent_controller_state(self) -> dict[str, Any]:
        controller = self.opponent_controller
        planner = controller.planner
        tracker = planner.tracker
        if planner.selection_func not in (None, np.argmin):
            raise RuntimeError("Unsupported LatticePlanner selection function in runtime snapshot")
        if tracker.drawn_waypoints:
            raise RuntimeError("Runtime snapshot does not support rendered PurePursuit state")
        return {
            "trajectory": copy_module.deepcopy(controller.trajectory),
            "tracker_count": int(controller.tracker_count),
            "speed_scale": float(controller.speed_scale),
            "planner": self._copied_fields(planner, RUNTIME_SNAPSHOT_PLANNER_FIELDS),
            "selection_func": "none" if planner.selection_func is None else "numpy_argmin",
            "tracker_prev_error": float(tracker.prev_error),
            "tracker_has_nearest_dist": hasattr(tracker, "nearest_dist"),
            "tracker_nearest_dist": None if not hasattr(tracker, "nearest_dist") else float(tracker.nearest_dist),
        }

    def capture_runtime_snapshot(self) -> dict[str, Any]:
        core = self.f110_env.unwrapped
        agents = []
        for agent in core.sim.agents:
            agents.append({
                "state": np.asarray(agent.state).copy(),
                "opp_poses": copy_module.deepcopy(agent.opp_poses),
                "accel": float(agent.accel),
                "steer_angle_vel": float(agent.steer_angle_vel),
                "steer_buffer": np.asarray(agent.steer_buffer).copy(),
                "in_collision": bool(agent.in_collision),
                "scan_rng_state": copy_module.deepcopy(agent.scan_rng.bit_generator.state),
            })
        state = {
            "schema_version": 1,
            "order_enforcing_has_reset": bool(self.f110_env._has_reset),
            "racecars": agents,
            "simulator": {
                "agent_poses": np.asarray(core.sim.agent_poses).copy(),
                "collisions": np.asarray(core.sim.collisions).copy(),
                "collision_idx": np.asarray(core.sim.collision_idx).copy(),
            },
            "f110_core": self._copied_fields(core, RUNTIME_SNAPSHOT_CORE_FIELDS),
            "f110_current_obs": copy_module.deepcopy(type(core).current_obs),
            "opponent_controller": self._capture_opponent_controller_state(),
            "reward": self._copied_fields(self.transition_reward, RUNTIME_SNAPSHOT_REWARD_FIELDS),
            "wrapper": self._copied_fields(self, RUNTIME_SNAPSHOT_WRAPPER_FIELDS),
            "reset_rng_state": copy_module.deepcopy(self._reset_rng.bit_generator.state),
            "corridor_gate_current": None if self.following_danger_gate is None else bool(self.following_danger_gate.current_gate),
        }
        return {
            "schema_version": 1,
            "environment": state,
            "observation": np.asarray(self._observation(self._raw_observation), dtype=np.float32).copy(),
        }

    def _restore_environment_state(self, state: dict[str, Any]) -> None:
        required = {"schema_version", "order_enforcing_has_reset", "racecars", "simulator", "f110_core", "f110_current_obs", "opponent_controller", "reward", "wrapper", "reset_rng_state", "corridor_gate_current"}
        if state.get("schema_version") != 1 or set(state) != required:
            raise RuntimeError("Environment snapshot state is incomplete or unsupported")
        core = self.f110_env.unwrapped
        if len(state["racecars"]) != len(core.sim.agents):
            raise RuntimeError("Environment snapshot RaceCar count changed")
        for agent, saved in zip(core.sim.agents, state["racecars"]):
            agent.state = np.asarray(saved["state"]).copy()
            agent.opp_poses = copy_module.deepcopy(saved["opp_poses"])
            agent.accel = float(saved["accel"])
            agent.steer_angle_vel = float(saved["steer_angle_vel"])
            agent.steer_buffer = np.asarray(saved["steer_buffer"]).copy()
            agent.in_collision = bool(saved["in_collision"])
            agent.scan_rng.bit_generator.state = copy_module.deepcopy(saved["scan_rng_state"])
        core.sim.agent_poses = np.asarray(state["simulator"]["agent_poses"]).copy()
        core.sim.collisions = np.asarray(state["simulator"]["collisions"]).copy()
        core.sim.collision_idx = np.asarray(state["simulator"]["collision_idx"]).copy()
        for name, value in state["f110_core"].items():
            setattr(core, name, copy_module.deepcopy(value))
        type(core).current_obs = copy_module.deepcopy(state["f110_current_obs"])
        self.f110_env._has_reset = bool(state["order_enforcing_has_reset"])
        controller = self.opponent_controller
        controller_state = state["opponent_controller"]
        controller.trajectory = copy_module.deepcopy(controller_state["trajectory"])
        controller.tracker_count = int(controller_state["tracker_count"])
        controller.speed_scale = float(controller_state["speed_scale"])
        planner = controller.planner
        for name, value in controller_state["planner"].items():
            setattr(planner, name, copy_module.deepcopy(value))
        planner.selection_func = None if controller_state["selection_func"] == "none" else np.argmin
        tracker = planner.tracker
        tracker.prev_error = float(controller_state["tracker_prev_error"])
        tracker.drawn_waypoints = []
        if controller_state["tracker_has_nearest_dist"]:
            tracker.nearest_dist = float(controller_state["tracker_nearest_dist"])
        elif hasattr(tracker, "nearest_dist"):
            delattr(tracker, "nearest_dist")
        for name, value in state["reward"].items():
            setattr(self.transition_reward, name, copy_module.deepcopy(value))
        for name, value in state["wrapper"].items():
            setattr(self, name, copy_module.deepcopy(value))
        self._reset_rng.bit_generator.state = copy_module.deepcopy(state["reset_rng_state"])
        if self.following_danger_gate is not None:
            if state["corridor_gate_current"] is None:
                raise RuntimeError("Environment snapshot corridor gate state is missing")
            self.following_danger_gate.current_gate = bool(state["corridor_gate_current"])

    def restore_runtime_snapshot(self, snapshot: dict[str, Any]) -> np.ndarray:
        if snapshot.get("schema_version") != 1 or set(snapshot) != {"schema_version", "environment", "observation"}:
            raise RuntimeError("Runtime snapshot schema is incomplete or unsupported")
        self._restore_environment_state(snapshot["environment"])
        observation = self._observation(self._raw_observation)
        if not np.array_equal(observation, np.asarray(snapshot["observation"], dtype=np.float32)):
            raise RuntimeError("Runtime snapshot restored observation changed")
        return observation

    def _info(
        self,
        ego_collision: bool,
        opponent_collision: bool,
        base_terminated: bool,
        timeout: bool,
        reason: str | None,
        outcome: str | None,
    ) -> dict[str, Any]:
        scenario = dict(self._current_spec.scenario)
        return {
            "ego_collision": ego_collision,
            "opponent_collision": opponent_collision,
            "base_terminated": base_terminated,
            "base_truncated": False,
            "timeout": timeout,
            "elapsed_time": self._elapsed_time,
            "termination_reason": reason,
            "scenario": scenario,
            "scenario_id": str(scenario["scenario_id"]),
            "sampler_branch": str(scenario["env_role"]),
            "env_role": str(scenario["env_role"]),
            "episode_outcome": outcome,
            "episode_return": self._episode_return,
            "episode_steps": self._episode_steps,
            CONFIG.exploration_gate_info_key: bool(
                self.following_danger_gate.current_gate
                if self.following_danger_gate is not None
                else False
            ),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        previous_raw_observation = self._raw_observation
        previous_ego_speed = self._ego_speed(previous_raw_observation)
        joint_action = np.stack(
            (np.asarray(action, dtype=np.float32).reshape(2), self.opponent_controller.action(previous_raw_observation))
        )
        raw_observation, simulator_reward, base_terminated, _ = self.f110_env.step(joint_action)
        self._elapsed_time += float(simulator_reward)
        collisions = np.asarray(raw_observation["collisions"], dtype=bool).reshape(-1)
        ego_collision = bool(collisions[EGO_INDEX])
        opponent_collision = bool(collisions[OPPONENT_INDEX])
        timeout = self._elapsed_time + 1e-12 >= CONFIG.episode_horizon
        if ego_collision or (base_terminated and not opponent_collision):
            terminated, truncated = True, False
            reason = "ego_collision" if ego_collision else "base_terminated"
        elif timeout:
            terminated, truncated, reason = False, True, "timeout"
        else:
            terminated, truncated, reason = False, False, None
        scenario_id = str(self._current_spec.scenario["scenario_id"])
        reward_result = self.transition_reward.step(
            previous_raw_observation,
            raw_observation,
            ego_collision=ego_collision,
            opponent_collision=opponent_collision,
            terminated=terminated,
            scenario_id=scenario_id,
            ego_index=EGO_INDEX,
        )
        reward = reward_result.reward_total
        self._episode_return += reward
        self._episode_steps += 1
        outcome = None
        if terminated or truncated:
            if ego_collision:
                outcome = "ego_collision"
            elif reward_result.relative_position_m > 0.0:
                outcome = "overtake"
            else:
                outcome = "follow"
        self._raw_observation = raw_observation
        self._previous_ego_speed = previous_ego_speed
        if self.following_danger_gate is not None:
            self.following_danger_gate.step(
                raw_observation,
                elapsed_time_s=self._elapsed_time,
                ego_index=EGO_INDEX,
                opponent_index=OPPONENT_INDEX,
            )
        info = self._info(
            ego_collision,
            opponent_collision,
            bool(base_terminated),
            timeout,
            reason,
            outcome,
        )
        info["executed_ego_action"] = np.asarray(action, dtype=np.float32).reshape(2).copy()
        return self._observation(raw_observation), reward, terminated, truncated, info

    def render(self) -> Any:
        return self.f110_env.render()

    def close(self) -> None:
        self.f110_env.close()


def _external_reset_required(_rng: np.random.Generator) -> EpisodeResetSpec:
    raise RuntimeError("Subprocess resets must be supplied by the parent scheduler")


def make_environment(
    seed: int,
    map_name: str,
    privileged: bool = False,
    reward_gamma: float = 0.999,
    speed_exploration_mode: str = "baseline",
) -> Callable[[], End2RaceGymnasiumEnv]:

    def factory() -> End2RaceGymnasiumEnv:
        import gym
        from f110_gym.envs.base_classes import Integrator

        warnings.filterwarnings("ignore", message="Chosen integrator is RK4.*", category=UserWarning, module="f110_gym.envs.base_classes")
        core = gym.make(
            "f110-v0",
            map=str(PROJECT_ROOT / "f1tenth_racetracks" / map_name / f"{map_name}_map"),
            map_ext=".png",
            num_agents=NUM_AGENTS,
            timestep=CONFIG.simulator_timestep,
            integrator=Integrator.RK4,
            seed=seed,
        )
        return End2RaceGymnasiumEnv(
            core,
            _external_reset_required,
            map_name,
            CONFIG.ego_raceline,
            privileged=privileged,
            reward_gamma=reward_gamma,
            speed_exploration_mode=speed_exploration_mode,
        )

    return factory


def limit_worker_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _worker(remote: Any, parent_remote: Any, env_fn_wrapper: CloudpickleWrapper, rank: int) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = None
    try:
        limit_worker_threads()
        env = _patch_env(env_fn_wrapper.var())
        while True:
            command, data = remote.recv()
            if command == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                remote.send(("ok", (rank, observation, reward, terminated, truncated, info)))
            elif command == "reset":
                seed, spec = data
                observation, reset_info = env.reset(
                    seed=seed,
                    options={EXTERNAL_RESET_OPTION: spec},
                )
                remote.send(("ok", (rank, observation, reset_info)))
            elif command == "render":
                remote.send(("ok", env.render()))
            elif command == "get_spaces":
                remote.send(("ok", (env.observation_space, env.action_space)))
            elif command == "env_method":
                method_name, method_args, method_kwargs = data
                method = env.get_wrapper_attr(method_name)
                remote.send(("ok", method(*method_args, **method_kwargs)))
            elif command == "get_attr":
                remote.send(("ok", env.get_wrapper_attr(data)))
            elif command == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    result = True
                except AttributeError:
                    result = False
                remote.send(("ok", result))
            elif command == "set_attr":
                name, value = data
                setattr(env, name, value)
                remote.send(("ok", None))
            elif command == "is_wrapped":
                remote.send(("ok", is_wrapped(env, data)))
            elif command == "close":
                break
            else:
                raise NotImplementedError(command)
    except (EOFError, KeyboardInterrupt):
        pass
    except BaseException:
        try:
            remote.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        remote.close()


class CentralScheduleSubprocVecEnv(VecEnv):

    def __init__(
        self,
        n_envs: int,
        start_method: str,
        seed: int,
        map_name: str,
        collision_scenarios: Sequence[ScenarioSpec],
        ordinary_scenarios: Sequence[ScenarioSpec],
        privileged: bool = False,
        reward_gamma: float = 0.999,
        speed_exploration_mode: str = "baseline",
    ) -> None:
        if n_envs <= 0 or n_envs % 2 != 0:
            raise ValueError("n_envs must be positive and even")
        if torch.cuda.is_initialized():
            raise RuntimeError("Subprocess environments must be created before CUDA initialization")
        self.waiting = False
        self.closed = False
        self.actions = None
        self.scheduler = ScenarioScheduler(seed, collision_scenarios, ordinary_scenarios)
        logical_seeds = [
            int(np.random.SeedSequence([seed, 1, rank % 2, rank // 2]).generate_state(1)[0])
            for rank in range(n_envs)
        ]
        env_fns = [
            make_environment(
                logical_seeds[rank],
                map_name,
                privileged=privileged,
                reward_gamma=reward_gamma,
                speed_exploration_mode=speed_exploration_mode,
            )
            for rank in range(n_envs)
        ]
        context = mp.get_context(start_method)
        self.remotes, work_remotes = zip(*[context.Pipe() for _ in range(n_envs)])
        self.processes = []
        previous = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        for name in previous:
            os.environ[name] = "1"
        try:
            for rank, (work_remote, remote, env_fn) in enumerate(zip(work_remotes, self.remotes, env_fns)):
                process = context.Process(
                    target=_worker,
                    args=(work_remote, remote, CloudpickleWrapper(env_fn), rank),
                    daemon=True,
                )
                process.start()
                self.processes.append(process)
                work_remote.close()
            self.remotes[0].send(("get_spaces", None))
            observation_space, action_space = self._recv_checked(0)
            super().__init__(n_envs, observation_space, action_space)
            self.seed(seed)
        except BaseException:
            self._terminate_workers()
            raise
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def seed(self, seed: int) -> list[int]:
        self._seeds = [
            int(np.random.SeedSequence([seed, 1, rank % 2, rank // 2]).generate_state(1)[0])
            for rank in range(self.num_envs)
        ]
        return self._seeds

    def _recv_checked(self, rank: int) -> Any:
        try:
            status, payload = self.remotes[rank].recv()
        except (EOFError, BrokenPipeError, OSError) as error:
            self._terminate_workers()
            raise RuntimeError(f"environment worker {rank} exited unexpectedly") from error
        if status != "ok":
            self._terminate_workers()
            raise RuntimeError(f"environment worker {rank} failed:\n{payload}")
        return payload

    def _terminate_workers(self) -> None:
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        self.closed = True

    def _reset_round(self, indices: list[int], seeds: list[int | None]) -> list[VecEnvObs]:
        for rank, seed in zip(indices, seeds):
            self.remotes[rank].send(("reset", (seed, self.scheduler.next(rank))))
        observations = []
        for rank in indices:
            returned_rank, observation, reset_info = self._recv_checked(rank)
            if returned_rank != rank:
                raise RuntimeError(f"Environment worker rank mismatch: expected {rank}, got {returned_rank}")
            self.reset_infos[rank] = reset_info
            observations.append(observation)
        return observations

    def reset(self) -> VecEnvObs:
        indices = list(range(self.num_envs))
        observations = self._reset_round(indices, list(self._seeds))
        self._reset_seeds()
        self._reset_options()
        return np.stack(observations)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        for rank, action in enumerate(actions):
            self.remotes[rank].send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        rows = [self._recv_checked(rank) for rank in range(self.num_envs)]
        self.waiting = False
        by_rank = {int(row[0]): row for row in rows}
        observations = [by_rank[rank][1] for rank in range(self.num_envs)]
        rewards = np.asarray([by_rank[rank][2] for rank in range(self.num_envs)], dtype=np.float32)
        terminated = np.asarray([by_rank[rank][3] for rank in range(self.num_envs)], dtype=bool)
        truncated = np.asarray([by_rank[rank][4] for rank in range(self.num_envs)], dtype=bool)
        infos = [by_rank[rank][5] for rank in range(self.num_envs)]
        dones = np.logical_or(terminated, truncated)
        reset_indices = []
        for rank, done in enumerate(dones):
            infos[rank]["TimeLimit.truncated"] = bool(truncated[rank] and not terminated[rank])
            if done:
                infos[rank]["terminal_observation"] = observations[rank]
                reset_indices.append(rank)
        if reset_indices:
            reset_observations = self._reset_round(reset_indices, [None] * len(reset_indices))
            for rank, observation in zip(reset_indices, reset_observations):
                observations[rank] = observation
        return np.stack(observations), rewards, dones, tuple(infos)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for rank in range(self.num_envs):
                self._recv_checked(rank)
            self.waiting = False
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5.0)
        if any(process.is_alive() for process in self.processes):
            self._terminate_workers()
            raise RuntimeError("Environment workers did not exit normally")
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            warnings.warn(f"render_mode is {self.render_mode}, not rgb_array")
            return [None] * self.num_envs
        for remote in self.remotes:
            remote.send(("render", None))
        return [self._recv_checked(rank) for rank in range(self.num_envs)]

    def has_attr(self, attr_name: str) -> bool:
        for remote in self.remotes:
            remote.send(("has_attr", attr_name))
        return all(self._recv_checked(rank) for rank in range(self.num_envs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        ranks = self._get_indices(indices)
        for rank in ranks:
            self.remotes[rank].send(("get_attr", attr_name))
        return [self._recv_checked(rank) for rank in ranks]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        ranks = self._get_indices(indices)
        for rank in ranks:
            self.remotes[rank].send(("set_attr", (attr_name, value)))
        for rank in ranks:
            self._recv_checked(rank)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        ranks = self._get_indices(indices)
        for rank in ranks:
            self.remotes[rank].send(("env_method", (method_name, method_args, method_kwargs)))
        return [self._recv_checked(rank) for rank in ranks]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        ranks = self._get_indices(indices)
        for rank in ranks:
            self.remotes[rank].send(("is_wrapped", wrapper_class))
        return [self._recv_checked(rank) for rank in ranks]

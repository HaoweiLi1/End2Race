"""End2Race environment and vector-environment execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import copy as copy_module
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import traceback
from typing import Any
import warnings

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.spatial import cKDTree
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from threadpoolctl import threadpool_limits
import torch
import yaml

from ppo.policy import (
    BASELINE_EXPLORATION_MODE,
    CORRIDOR_TEMPORAL_EXPLORATION_MODE,
    END2RACE_LIDAR_SIZE,
    END2RACE_OBSERVATION_SIZE,
    EXPLORATION_GATE_INFO_KEY,
    NOOP_SPEED_BOUND,
    PRIVILEGED_FEATURE_SIZE,
    STEERING_BOUND,
    PrivilegedStateExtractor,
    end2race_observation,
    exploration_uses_gate,
)
from ppo.reward import OccupancyMapClearance, PPOTransitionReward
from ppo.scenarios import EpisodeResetSpec, ScenarioScheduler, ScenarioSpec


with Path(__file__).with_name("ppo_config.yaml").open("r", encoding="utf-8") as file:
    PPO_CONFIG = yaml.safe_load(file)

SIMULATOR_TIMESTEP = float(PPO_CONFIG["simulator_timestep"])
EPISODE_HORIZON = float(PPO_CONFIG["episode_horizon"])
EGO_RACELINE = str(PPO_CONFIG["ego_raceline"])
EGO_INDEX = 0
OPPONENT_INDEX = 1
NUM_AGENTS = 2
EXTERNAL_RESET_OPTION = "end2race_episode_reset_spec"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PLANNER_TEMPLATE_CACHE: dict[tuple[str, str], Any] = {}
PREFIX_RESET_SEED_TAG = 0x50524658


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prefix_reset_panel(path: str | Path) -> tuple[dict[str, Any], ...]:
    root = Path(path).expanduser().resolve()
    manifest_path = root / "prefix_reset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("panel_id") != "prefix_reset_consensus_v1":
        raise RuntimeError("Prefix-reset panel manifest is unsupported")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 28 or len({task["episode_key"] for task in tasks}) != 28:
        raise RuntimeError("Prefix-reset panel must contain 28 unique tasks")
    loaded = []
    total_prefix_rows = 0
    for task in tasks:
        snapshot_path = root / task["snapshot_file"]
        prefix_path = root / task["prefix_file"]
        if _sha256_file(snapshot_path) != task["snapshot_sha256"] or _sha256_file(prefix_path) != task["prefix_sha256"]:
            raise RuntimeError(f"Prefix-reset panel content changed: {task['episode_key']}")
        with snapshot_path.open("rb") as stream:
            snapshot = pickle.load(stream)
        with np.load(prefix_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"prefix_observations", "window_observation"}:
                raise RuntimeError(f"Prefix-reset array schema changed: {task['episode_key']}")
            prefix = np.asarray(arrays["prefix_observations"], dtype=np.float32)
            window_observation = np.asarray(arrays["window_observation"], dtype=np.float32)
        expected_length = int(task["prefix_length"])
        if prefix.shape != (expected_length, 381) or window_observation.shape != (381,) or not np.isfinite(prefix).all() or not np.isfinite(window_observation).all():
            raise RuntimeError(f"Prefix-reset observation contract failed: {task['episode_key']}")
        if not np.array_equal(window_observation, np.asarray(snapshot["observation"], dtype=np.float32)):
            raise RuntimeError(f"Prefix-reset window observation changed: {task['episode_key']}")
        total_prefix_rows += expected_length
        loaded.append({"episode_key": str(task["episode_key"]), "snapshot": snapshot, "prefix_observations": prefix, "prefix_length": expected_length})
    if total_prefix_rows != 9589:
        raise RuntimeError(f"Prefix-reset panel must contain 9,589 prefix rows, got {total_prefix_rows}")
    return tuple(loaded)


@dataclass(frozen=True)
class FrontCorridorGateConfig:
    maximum_front_gap_m: float = 2.0
    maximum_abs_opponent_lateral_d_m: float = 0.25
    require_positive_lateral_overlap: bool = True

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
            (np.clip(steering, -STEERING_BOUND, STEERING_BOUND), desired_speed * self.speed_scale),
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
        speed_exploration_mode: str = BASELINE_EXPLORATION_MODE,
    ) -> None:
        super().__init__()
        self.f110_env = f110_env
        self.reset_provider = reset_provider
        self.opponent_controller = LatticePlannerOpponentController()
        core = getattr(f110_env, "unwrapped", f110_env)
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
            risk_longitudinal_clearance_m=float(PPO_CONFIG["risk_longitudinal_clearance_m"]),
            risk_lateral_clearance_m=float(PPO_CONFIG["risk_lateral_clearance_m"]),
            risk_wall_clearance_m=float(PPO_CONFIG["risk_wall_clearance_m"]),
            risk_potential_maximum=float(PPO_CONFIG["risk_potential_maximum"]),
        )
        if speed_exploration_mode == CORRIDOR_TEMPORAL_EXPLORATION_MODE:
            corridor_config = FrontCorridorGateConfig(
                maximum_front_gap_m=float(
                    PPO_CONFIG["front_corridor_gate_maximum_gap_m"]
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
            low=np.asarray((-STEERING_BOUND, -NOOP_SPEED_BOUND), dtype=np.float32),
            high=np.asarray((STEERING_BOUND, NOOP_SPEED_BOUND), dtype=np.float32),
            dtype=np.float32,
        )
        self._reset_rng = np.random.default_rng()
        self._elapsed_time = 0.0
        self._previous_ego_speed = 0.0
        self._raw_observation = None
        self._current_spec = None
        self._episode_return = 0.0
        self._episode_steps = 0
        self._episode_reward_progress = 0.0
        self._episode_reward_relative = 0.0
        self._episode_reward_collision = 0.0
        self._episode_reward_risk = 0.0
        self._episode_abs_reward_risk = 0.0
        self._episode_min_obb_clearance_m = float("inf")
        self._episode_min_wall_clearance_m = float("inf")
        self._episode_risk_active_steps = 0

    def _ego_lidar(self, raw_observation: dict[str, Any]) -> np.ndarray:
        scan = np.asarray(raw_observation["scans"][EGO_INDEX]).reshape(-1)
        if scan.size > END2RACE_LIDAR_SIZE:
            scan = scan[np.linspace(0, scan.size - 1, END2RACE_LIDAR_SIZE, dtype=int)]
        return np.asarray(scan, dtype=np.float32)

    def _ego_speed(self, raw_observation: dict[str, Any]) -> float:
        return float(np.asarray(raw_observation["linear_vels_x"])[EGO_INDEX])

    def _privileged_physical_state(self) -> tuple[float, float, float]:
        core = getattr(self.f110_env, "unwrapped", self.f110_env)
        agents = getattr(getattr(core, "sim", None), "agents", None)
        if agents is None or len(agents) != NUM_AGENTS:
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
        raw_observation, _, _, base_info = self.f110_env.reset(poses=spec.poses.copy())
        if int(getattr(getattr(self.f110_env, "unwrapped", self.f110_env), "num_agents")) != NUM_AGENTS:
            raise RuntimeError("PPO environment requires exactly two agents")
        self._elapsed_time = 0.0
        self._episode_return = 0.0
        self._episode_steps = 0
        self._episode_reward_progress = 0.0
        self._episode_reward_relative = 0.0
        self._episode_reward_collision = 0.0
        self._episode_reward_risk = 0.0
        self._episode_abs_reward_risk = 0.0
        self._episode_risk_active_steps = 0
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
        self._episode_min_obb_clearance_m = float(self.transition_reward.current_obb_clearance_m)
        self._episode_min_wall_clearance_m = float(self.transition_reward.current_wall_clearance_m)
        self.opponent_controller.reset(spec)
        info = self._info(False, False, False, False, None, None, base_info, {})
        return self._observation(raw_observation), info

    def restore_prefix_snapshot(self, snapshot: dict[str, Any], episode_key: str, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if snapshot.get("schema_version") != 1 or set(snapshot) != {"schema_version", "environment", "observation", "actor_hidden", "critic_hidden"}:
            raise RuntimeError("Prefix-reset snapshot schema is incomplete or unsupported")
        state = snapshot["environment"]
        required = {"schema_version", "order_enforcing_has_reset", "racecars", "simulator", "f110_core", "f110_current_obs", "opponent_controller", "reward", "wrapper", "reset_rng_state", "corridor_gate_current"}
        if state.get("schema_version") != 1 or set(state) != required:
            raise RuntimeError("Prefix-reset environment state is incomplete or unsupported")
        spec = copy_module.deepcopy(state["wrapper"]["_current_spec"])
        self.reset(seed=seed, options={EXTERNAL_RESET_OPTION: spec})
        core = self.f110_env.unwrapped
        if len(state["racecars"]) != len(core.sim.agents):
            raise RuntimeError("Prefix-reset RaceCar count changed")
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
        if hasattr(self.f110_env, "_has_reset"):
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
                raise RuntimeError("Prefix-reset corridor gate state is missing")
            self.following_danger_gate.current_gate = bool(state["corridor_gate_current"])
        scenario = dict(self._current_spec.scenario)
        scenario["pool"] = "prefix_reset_consensus_v1"
        scenario["sampler_branch"] = "prefix_reset"
        scenario["env_role"] = "collision"
        self._current_spec.scenario = scenario
        observation = self._observation(self._raw_observation)
        if not np.array_equal(observation, np.asarray(snapshot["observation"], dtype=np.float32)):
            raise RuntimeError(f"Prefix-reset restored observation changed: {episode_key}")
        info = self._info(False, False, False, False, None, None, {}, {})
        info["prefix_reset"] = True
        info["prefix_reset_key"] = str(episode_key)
        return observation, info

    def _info(
        self,
        ego_collision: bool,
        opponent_collision: bool,
        base_terminated: bool,
        timeout: bool,
        reason: str | None,
        outcome: str | None,
        base_info: dict[str, Any],
        reward_info: dict[str, Any],
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
            "episode_reward_progress": self._episode_reward_progress,
            "episode_reward_relative": self._episode_reward_relative,
            "episode_reward_collision": self._episode_reward_collision,
            "episode_reward_risk": self._episode_reward_risk,
            "episode_abs_reward_risk": self._episode_abs_reward_risk,
            "episode_min_obb_clearance_m": self._episode_min_obb_clearance_m,
            "episode_min_wall_clearance_m": self._episode_min_wall_clearance_m,
            "episode_risk_active_fraction": (
                self._episode_risk_active_steps / self._episode_steps
                if self._episode_steps > 0
                else 0.0
            ),
            EXPLORATION_GATE_INFO_KEY: bool(
                self.following_danger_gate.current_gate
                if self.following_danger_gate is not None
                else False
            ),
            "base_info": base_info,
            **reward_info,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        previous_raw_observation = self._raw_observation
        previous_ego_speed = self._ego_speed(previous_raw_observation)
        joint_action = np.stack(
            (np.asarray(action, dtype=np.float32).reshape(2), self.opponent_controller.action(previous_raw_observation))
        )
        raw_observation, simulator_reward, base_terminated, base_info = self.f110_env.step(joint_action)
        self._elapsed_time += float(simulator_reward)
        collisions = np.asarray(raw_observation["collisions"], dtype=bool).reshape(-1)
        ego_collision = bool(collisions[EGO_INDEX])
        opponent_collision = bool(collisions[OPPONENT_INDEX])
        timeout = self._elapsed_time + 1e-12 >= EPISODE_HORIZON
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
        reward_info = reward_result.to_info()
        reward = float(reward_info["reward_total"])
        self._episode_return += reward
        self._episode_steps += 1
        self._episode_reward_progress += float(reward_info["reward_progress"])
        self._episode_reward_relative += float(reward_info["reward_relative"])
        self._episode_reward_collision += float(reward_info["reward_collision"])
        self._episode_reward_risk += float(reward_info["reward_risk"])
        self._episode_abs_reward_risk += abs(float(reward_info["reward_risk"]))
        obb_clearance_m = float(reward_info["obb_clearance_m"])
        self._episode_min_obb_clearance_m = min(self._episode_min_obb_clearance_m, obb_clearance_m)
        wall_clearance_m = float(reward_info["wall_clearance_m"])
        self._episode_min_wall_clearance_m = min(self._episode_min_wall_clearance_m, wall_clearance_m)
        if bool(reward_info["risk_active"]):
            self._episode_risk_active_steps += 1
        outcome = None
        if terminated or truncated:
            if ego_collision:
                outcome = "ego_collision"
            elif float(reward_info["relative_position_m"]) > 0.0:
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
            base_info,
            {"simulator_reward": float(simulator_reward), **reward_info},
        )
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
    speed_exploration_mode: str = BASELINE_EXPLORATION_MODE,
) -> Callable[[], End2RaceGymnasiumEnv]:

    def factory() -> End2RaceGymnasiumEnv:
        import gym
        from f110_gym.envs.base_classes import Integrator

        core = gym.make(
            "f110-v0",
            map=str(PROJECT_ROOT / "f1tenth_racetracks" / map_name / f"{map_name}_map"),
            map_ext=".png",
            num_agents=NUM_AGENTS,
            timestep=SIMULATOR_TIMESTEP,
            integrator=Integrator.RK4,
            seed=seed,
        )
        return End2RaceGymnasiumEnv(
            core,
            _external_reset_required,
            map_name,
            EGO_RACELINE,
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
            elif command == "reset_prefix":
                seed, snapshot, episode_key = data
                observation, reset_info = env.restore_prefix_snapshot(snapshot, episode_key, seed=seed)
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
        speed_exploration_mode: str = BASELINE_EXPLORATION_MODE,
        prefix_reset_inputs: Sequence[dict[str, Any]] = (),
        prefix_reset_interval: int = 0,
    ) -> None:
        if n_envs <= 0 or n_envs % 2 != 0:
            raise ValueError("n_envs must be positive and even")
        if torch.cuda.is_initialized():
            raise RuntimeError("Subprocess environments must be created before CUDA initialization")
        self.waiting = False
        self.closed = False
        self.actions = None
        self.scheduler = ScenarioScheduler(seed, collision_scenarios, ordinary_scenarios)
        self.prefix_reset_inputs = tuple(prefix_reset_inputs)
        self.prefix_reset_interval = int(prefix_reset_interval)
        if bool(self.prefix_reset_inputs) != (self.prefix_reset_interval > 0):
            raise ValueError("Prefix-reset inputs and interval must be enabled together")
        if self.prefix_reset_inputs and (len(self.prefix_reset_inputs) != 28 or len({item["episode_key"] for item in self.prefix_reset_inputs}) != 28):
            raise ValueError("Prefix-reset integration requires 28 unique inputs")
        self.prefix_reset_enabled = bool(self.prefix_reset_inputs)
        self.prefix_reset_rng = np.random.default_rng(np.random.SeedSequence([seed, PREFIX_RESET_SEED_TAG]))
        self.prefix_reset_order = np.asarray(self.prefix_reset_rng.permutation(len(self.prefix_reset_inputs)), dtype=np.int64) if self.prefix_reset_inputs else np.empty(0, dtype=np.int64)
        self.prefix_reset_cursor = 0
        self.prefix_reset_cycle = 1 if self.prefix_reset_inputs else 0
        self.collision_reset_count = 0
        self.reset_history: list[dict[str, Any]] = []
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

    def seed(self, seed: int | None = None) -> list[int | None]:
        if seed is None:
            seed = 0
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
        for process in getattr(self, "processes", []):
            if process.is_alive():
                process.terminate()
        for process in getattr(self, "processes", []):
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        self.closed = True

    def _next_prefix_reset(self) -> dict[str, Any]:
        if self.prefix_reset_cursor == len(self.prefix_reset_order):
            self.prefix_reset_order = np.asarray(self.prefix_reset_rng.permutation(len(self.prefix_reset_inputs)), dtype=np.int64)
            self.prefix_reset_cursor = 0
            self.prefix_reset_cycle += 1
        item = self.prefix_reset_inputs[int(self.prefix_reset_order[self.prefix_reset_cursor])]
        self.prefix_reset_cursor += 1
        return item

    def _reset_round(self, indices: list[int], seeds: list[int | None]) -> list[VecEnvObs]:
        requests = {}
        for rank, seed in zip(indices, seeds):
            use_prefix = False
            if rank % 2 == 0:
                self.collision_reset_count += 1
                use_prefix = self.prefix_reset_enabled and self.collision_reset_count % self.prefix_reset_interval == 0
            if use_prefix:
                item = self._next_prefix_reset()
                requests[rank] = ("prefix_reset", item)
                self.remotes[rank].send(("reset_prefix", (seed, item["snapshot"], item["episode_key"])))
            else:
                requests[rank] = ("standard", None)
                self.remotes[rank].send(("reset", (seed, self.scheduler.next(rank))))
        observations = []
        for rank in indices:
            returned_rank, observation, reset_info = self._recv_checked(rank)
            if returned_rank != rank:
                raise RuntimeError(f"Environment worker rank mismatch: expected {rank}, got {returned_rank}")
            source, item = requests[rank]
            if source == "prefix_reset":
                reset_info["prefix_reset"] = True
                reset_info["prefix_reset_key"] = item["episode_key"]
                reset_info["prefix_observations"] = item["prefix_observations"]
                reset_info["prefix_length"] = int(item["prefix_length"])
            else:
                reset_info["prefix_reset"] = False
                reset_info["prefix_reset_key"] = None
                reset_info["prefix_observations"] = np.empty((0, self.observation_space.shape[0]), dtype=np.float32)
                reset_info["prefix_length"] = 0
            self.reset_infos[rank] = reset_info
            self.reset_history.append({"rank": int(rank), "env_role": str(reset_info["env_role"]), "source": source, "prefix_reset_key": reset_info["prefix_reset_key"], "prefix_length": int(reset_info["prefix_length"]), "scenario_id": str(reset_info["scenario_id"])})
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

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler.state_dict(),
            "prefix_reset": {
                "enabled": self.prefix_reset_enabled,
                "interval": self.prefix_reset_interval,
                "order": self.prefix_reset_order.copy(),
                "cursor": self.prefix_reset_cursor,
                "cycle": self.prefix_reset_cycle,
                "collision_reset_count": self.collision_reset_count,
                "rng_state": copy_module.deepcopy(self.prefix_reset_rng.bit_generator.state),
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state["scheduler"])
        prefix = state["prefix_reset"]
        if bool(prefix["enabled"]) != self.prefix_reset_enabled or int(prefix["interval"]) != self.prefix_reset_interval:
            raise ValueError("Prefix-reset scheduler configuration does not match")
        order = np.asarray(prefix["order"], dtype=np.int64)
        if sorted(order.tolist()) != list(range(len(self.prefix_reset_inputs))):
            raise ValueError("Prefix-reset scheduler order is invalid")
        self.prefix_reset_order = order.copy()
        self.prefix_reset_cursor = int(prefix["cursor"])
        self.prefix_reset_cycle = int(prefix["cycle"])
        self.collision_reset_count = int(prefix["collision_reset_count"])
        self.prefix_reset_rng.bit_generator.state = copy_module.deepcopy(prefix["rng_state"])

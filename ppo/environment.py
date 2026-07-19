"""Single logical End2Race F110 environment."""

from __future__ import annotations

from collections.abc import Callable
import copy as copy_module
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from ppo.policy import END2RACE_LIDAR_SIZE, END2RACE_OBSERVATION_SIZE, NOOP_SPEED_BOUND, STEERING_BOUND, end2race_observation
from ppo.reward import PPOTransitionReward
from ppo.scenarios import EpisodeResetSpec


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

    def __init__(self, f110_env: Any, reset_provider: Callable[[np.random.Generator], EpisodeResetSpec], map_name: str, ego_raceline: str) -> None:
        super().__init__()
        self.f110_env = f110_env
        self.reset_provider = reset_provider
        self.opponent_controller = LatticePlannerOpponentController()
        self.transition_reward = PPOTransitionReward(map_name, ego_raceline)
        self.observation_space = spaces.Box(
            low=np.full((END2RACE_OBSERVATION_SIZE,), -np.inf, dtype=np.float32),
            high=np.full((END2RACE_OBSERVATION_SIZE,), np.inf, dtype=np.float32),
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

    def _ego_lidar(self, raw_observation: dict[str, Any]) -> np.ndarray:
        scan = np.asarray(raw_observation["scans"][EGO_INDEX]).reshape(-1)
        if scan.size > END2RACE_LIDAR_SIZE:
            scan = scan[np.linspace(0, scan.size - 1, END2RACE_LIDAR_SIZE, dtype=int)]
        return np.asarray(scan, dtype=np.float32)

    def _ego_speed(self, raw_observation: dict[str, Any]) -> float:
        return float(np.asarray(raw_observation["linear_vels_x"])[EGO_INDEX])

    def _observation(self, raw_observation: dict[str, Any]) -> np.ndarray:
        return end2race_observation(self._ego_lidar(raw_observation), self._previous_ego_speed)

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
        self._raw_observation = raw_observation
        self._previous_ego_speed = float(spec.initial_speed_feature)
        self._current_spec = spec
        scenario_id = str(spec.scenario["scenario_id"])
        self.transition_reward.reset(raw_observation, scenario_id=scenario_id, ego_index=EGO_INDEX)
        self.opponent_controller.reset(spec)
        info = self._info(False, False, False, False, None, None, base_info, {})
        return self._observation(raw_observation), info

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
            scenario_id=scenario_id,
            ego_index=EGO_INDEX,
        )
        reward_info = reward_result.to_info()
        reward = float(reward_info["reward_total"])
        self._episode_return += reward
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


def make_environment(seed: int, map_name: str) -> Callable[[], End2RaceGymnasiumEnv]:

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
        return End2RaceGymnasiumEnv(core, _external_reset_required, map_name, EGO_RACELINE)

    return factory

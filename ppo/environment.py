"""Fixed Austin simulator environment and subprocess vector environment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
import multiprocessing as mp
import os
from pathlib import Path
import traceback
from typing import Any
import warnings

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env

from ppo.policy import *
from ppo.reward import *
from ppo.scenarios import *


ENV_WORKERS = 6
SIMULATOR_TIMESTEP = 0.01
EPISODE_HORIZON = 8.0
EGO_INDEX = 0
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
        planner = copy.copy(template)
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
        planner.tracker = copy.copy(template.tracker)
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

        opponent_index = 1
        pose_x = float(np.asarray(raw_observation["poses_x"])[opponent_index])
        pose_y = float(np.asarray(raw_observation["poses_y"])[opponent_index])
        pose_theta = float(np.asarray(raw_observation["poses_theta"])[opponent_index])
        speed = float(np.asarray(raw_observation["linear_vels_x"])[opponent_index])
        if self.tracker_count == 0 or self.trajectory is None:
            opponent_poses = obsDict2oppoArray(raw_observation, opponent_index)
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

    def __init__(self, f110_env: Any, reset_provider: Callable[[np.random.Generator], EpisodeResetSpec]) -> None:
        super().__init__()
        self.f110_env = f110_env
        self.reset_provider = reset_provider
        self.opponent_controller = LatticePlannerOpponentController()
        self.transition_reward = PPOTransitionReward()
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
        opponent_collision = bool(collisions[1])
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


def make_environment(seed: int) -> Callable[[], End2RaceGymnasiumEnv]:

    def factory() -> End2RaceGymnasiumEnv:
        import gym
        from f110_gym.envs.base_classes import Integrator

        core = gym.make(
            "f110-v0",
            map=str(PROJECT_ROOT / "f1tenth_racetracks" / "Austin" / "Austin_map"),
            map_ext=".png",
            num_agents=NUM_AGENTS,
            timestep=SIMULATOR_TIMESTEP,
            integrator=Integrator.RK4,
            seed=seed,
        )
        return End2RaceGymnasiumEnv(core, _external_reset_required)

    return factory


def _limit_worker_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except ImportError:
        pass
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def _worker(
    remote: Any,
    parent_remote: Any,
    env_fn_wrappers: list[CloudpickleWrapper],
    env_indices: list[int],
) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    envs = []
    try:
        _limit_worker_threads()
        envs = [_patch_env(wrapper.var()) for wrapper in env_fn_wrappers]
        by_rank = dict(zip(env_indices, envs))
        while True:
            command, data = remote.recv()
            if command == "step":
                results = []
                for rank, action in data:
                    observation, reward, terminated, truncated, info = by_rank[rank].step(action)
                    results.append((rank, observation, reward, terminated, truncated, info))
                remote.send(("ok", results))
            elif command == "reset":
                results = []
                for rank, seed, spec in data:
                    observation, reset_info = by_rank[rank].reset(
                        seed=seed,
                        options={EXTERNAL_RESET_OPTION: spec},
                    )
                    results.append((rank, observation, reset_info))
                remote.send(("ok", results))
            elif command == "render":
                remote.send(("ok", [(rank, by_rank[rank].render()) for rank in data]))
            elif command == "close":
                break
            elif command == "get_spaces":
                remote.send(("ok", (envs[0].observation_space, envs[0].action_space)))
            elif command == "env_method":
                results = []
                for rank, method_name, method_args, method_kwargs in data:
                    method = by_rank[rank].get_wrapper_attr(method_name)
                    results.append((rank, method(*method_args, **method_kwargs)))
                remote.send(("ok", results))
            elif command == "get_attr":
                remote.send(("ok", [(rank, by_rank[rank].get_wrapper_attr(name)) for rank, name in data]))
            elif command == "has_attr":
                results = []
                for rank, name in data:
                    try:
                        by_rank[rank].get_wrapper_attr(name)
                        result = True
                    except AttributeError:
                        result = False
                    results.append((rank, result))
                remote.send(("ok", results))
            elif command == "set_attr":
                for rank, name, value in data:
                    setattr(by_rank[rank], name, value)
                remote.send(("ok", [(rank, None) for rank, _name, _value in data]))
            elif command == "is_wrapped":
                remote.send(("ok", [(rank, is_wrapped(by_rank[rank], wrapper)) for rank, wrapper in data]))
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
        for env in envs:
            try:
                env.close()
            except BaseException:
                pass
        remote.close()


class CentralScheduleSubprocVecEnv(VecEnv):

    def __init__(self, n_envs: int, seed: int) -> None:
        if n_envs < ENV_WORKERS or n_envs % 2 != 0:
            raise ValueError(f"n_envs must be even and at least {ENV_WORKERS}")
        if torch.cuda.is_initialized():
            raise RuntimeError("Subprocess environments must be created before CUDA initialization")
        self.waiting = False
        self.closed = False
        self.actions = None
        self.worker_count = ENV_WORKERS
        self.scheduler = ScenarioScheduler(seed)
        logical_seeds = [
            int(np.random.SeedSequence([seed, 1, rank % 2, rank // 2]).generate_state(1)[0])
            for rank in range(n_envs)
        ]
        env_fns = [make_environment(logical_seeds[rank]) for rank in range(n_envs)]
        self.worker_env_indices = [[] for _ in range(self.worker_count)]
        for rank in range(n_envs):
            self.worker_env_indices[rank % self.worker_count].append(rank)
        self.env_to_worker = {
            rank: worker_index
            for worker_index, ranks in enumerate(self.worker_env_indices)
            for rank in ranks
        }
        context = mp.get_context("forkserver")
        self.remotes, work_remotes = zip(*[context.Pipe() for _ in range(self.worker_count)])
        self.processes = []
        previous = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        for name in previous:
            os.environ[name] = "1"
        try:
            for worker_index, (work_remote, remote) in enumerate(zip(work_remotes, self.remotes)):
                ranks = self.worker_env_indices[worker_index]
                wrappers = [CloudpickleWrapper(env_fns[rank]) for rank in ranks]
                process = context.Process(target=_worker, args=(work_remote, remote, wrappers, ranks), daemon=True)
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

    def _recv_checked(self, worker_index: int) -> Any:
        try:
            status, payload = self.remotes[worker_index].recv()
        except (EOFError, BrokenPipeError, OSError) as error:
            self._terminate_workers()
            raise RuntimeError(f"environment worker {worker_index} exited unexpectedly") from error
        if status != "ok":
            self._terminate_workers()
            raise RuntimeError(f"environment worker {worker_index} failed:\n{payload}")
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

    def _group_entries(self, entries: list[tuple[Any, ...]]) -> dict[int, list[tuple[Any, ...]]]:
        grouped = {}
        for entry in entries:
            grouped.setdefault(self.env_to_worker[int(entry[0])], []).append(entry)
        return grouped

    def _reset_round(self, indices: list[int], seeds: list[int | None]) -> list[VecEnvObs]:
        entries = [(rank, seed, self.scheduler.next(rank)) for rank, seed in zip(indices, seeds)]
        grouped = self._group_entries(entries)
        for worker_index, worker_entries in grouped.items():
            self.remotes[worker_index].send(("reset", worker_entries))
        results = {
            worker_index: self._recv_checked(worker_index)
            for worker_index in grouped
        }
        by_rank = {int(row[0]): row for rows in results.values() for row in rows}
        observations = []
        for rank in indices:
            _rank, observation, reset_info = by_rank[rank]
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
        grouped = self._group_entries([(rank, action) for rank, action in enumerate(actions)])
        for worker_index, worker_entries in grouped.items():
            self.remotes[worker_index].send(("step", worker_entries))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        worker_results = [self._recv_checked(index) for index in range(self.worker_count)]
        self.waiting = False
        by_rank = {int(row[0]): row for rows in worker_results for row in rows}
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
            for worker_index in range(self.worker_count):
                self._recv_checked(worker_index)
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

    def _request_by_rank(self, command: str, entries: list[tuple[Any, ...]], ranks: list[int]) -> list[Any]:
        grouped = self._group_entries(entries)
        for worker_index, worker_entries in grouped.items():
            self.remotes[worker_index].send((command, worker_entries))
        flattened = [row for worker_index in grouped for row in self._recv_checked(worker_index)]
        by_rank = {int(rank): value for rank, value in flattened}
        return [by_rank[rank] for rank in ranks]

    def get_images(self) -> Sequence[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            warnings.warn(f"render_mode is {self.render_mode}, not rgb_array")
            return [None] * self.num_envs
        ranks = list(range(self.num_envs))
        grouped = self._group_entries([(rank,) for rank in ranks])
        for worker_index, entries in grouped.items():
            self.remotes[worker_index].send(("render", [entry[0] for entry in entries]))
        flattened = [row for worker_index in grouped for row in self._recv_checked(worker_index)]
        by_rank = dict(flattened)
        return [by_rank[rank] for rank in ranks]

    def has_attr(self, attr_name: str) -> bool:
        ranks = list(range(self.num_envs))
        return all(self._request_by_rank("has_attr", [(rank, attr_name) for rank in ranks], ranks))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        ranks = self._get_indices(indices)
        return self._request_by_rank("get_attr", [(rank, attr_name) for rank in ranks], ranks)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        ranks = self._get_indices(indices)
        self._request_by_rank("set_attr", [(rank, attr_name, value) for rank in ranks], ranks)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        ranks = self._get_indices(indices)
        entries = [(rank, method_name, method_args, method_kwargs) for rank in ranks]
        return self._request_by_rank("env_method", entries, ranks)

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        ranks = self._get_indices(indices)
        return self._request_by_rank("is_wrapped", [(rank, wrapper_class) for rank in ranks], ranks)

    def state_dict(self) -> dict[str, Any]:
        return {"scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state["scheduler"])

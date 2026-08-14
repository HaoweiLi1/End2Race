import copy as copy_module
import multiprocessing as mp
import os
import traceback
import warnings

from gym_notices import notices as gym_notices

gym_notices.notices.clear()

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from threadpoolctl import threadpool_limits
import torch

from latticeplanner.utils import TrackProjector, get_map_paths
from ppo.policy import (
    END2RACE_LIDAR_SIZE,
    END2RACE_OBSERVATION_SIZE,
    NOOP_SPEED_BOUND,
    PRIVILEGED_FEATURE_SIZE,
    PrivilegedStateExtractor,
    end2race_observation,
    wrap_to_pi,
)
from ppo.reward import ClearanceCalculator, collision_reward, progress_reward, relative_reward, risk_potential, risk_reward, wrapped_progress_delta
from ppo.scenarios import ScenarioScheduler


class End2RaceGymnasiumEnv(gym.Env):

    def __init__(self, f110_env, map_name, ego_raceline, config, reward_weights, privileged=False, reward_gamma=0.999, front_corridor_speed_noise_hold_steps=0):
        super().__init__()
        self.f110_env = f110_env
        self.config = config
        core = f110_env.unwrapped
        core_params = core.params
        self.vehicle_length = float(core_params["length"])
        self.vehicle_width = float(core_params["width"])
        scan_simulator = core.sim.agents[0].scan_simulator
        self.clearance_calculator = ClearanceCalculator(
            scan_simulator.dt,
            scan_simulator.map_resolution,
            scan_simulator.origin,
            self.vehicle_length,
            self.vehicle_width,
        )
        reference_path = os.path.join(get_map_paths(map_name)[0], f"{ego_raceline}.csv")
        self.projector = TrackProjector.from_csv(reference_path)
        self.reward_gamma = float(reward_gamma)
        self.reward_weights = np.asarray(reward_weights, dtype=np.float64)
        self.use_front_corridor_gate = front_corridor_speed_noise_hold_steps > 0
        self.front_corridor_gate = False
        self._planner_templates = {}
        self._opponent_planner = None
        self._opponent_trajectory = None
        self._opponent_tracker_count = 0
        self._opponent_speed_scale = 1.0
        if privileged:
            self.privileged_extractor = PrivilegedStateExtractor(
                map_name,
                ego_raceline,
                self.projector,
                self.vehicle_length,
                self.vehicle_width,
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
            low=np.asarray((-config.steering_bound, -NOOP_SPEED_BOUND), dtype=np.float32),
            high=np.asarray((config.steering_bound, NOOP_SPEED_BOUND), dtype=np.float32),
            dtype=np.float32,
        )
        self._elapsed_time = 0.0
        self._previous_ego_speed = 0.0
        self._raw_observation = None
        self._current_spec = None
        self._episode_return = 0.0
        self._episode_steps = 0

    def _front_corridor_gate(self, raw_observation):
        # K50 holds speed exploration while an overlapping opponent is close ahead.
        ego_pose = np.asarray((raw_observation["poses_x"][0], raw_observation["poses_y"][0], raw_observation["poses_theta"][0]), dtype=np.float64)
        opponent_pose = np.asarray((raw_observation["poses_x"][1], raw_observation["poses_y"][1], raw_observation["poses_theta"][1]), dtype=np.float64)
        ego_progress, ego_lateral_d, ego_tangent = self.projector.frenet(ego_pose[:2])
        opponent_progress, opponent_lateral_d, opponent_tangent = self.projector.frenet(opponent_pose[:2])
        opponent_ahead_center_m = -wrapped_progress_delta(ego_progress, opponent_progress, self.projector.track_length)

        ego_heading_error = wrap_to_pi(ego_pose[2] - ego_tangent)
        ego_cosine = abs(float(np.cos(ego_heading_error)))
        ego_sine = abs(float(np.sin(ego_heading_error)))
        ego_longitudinal = 0.5 * (self.vehicle_length * ego_cosine + self.vehicle_width * ego_sine)
        ego_lateral = 0.5 * (self.vehicle_length * ego_sine + self.vehicle_width * ego_cosine)
        opponent_heading_error = wrap_to_pi(opponent_pose[2] - opponent_tangent)
        opponent_cosine = abs(float(np.cos(opponent_heading_error)))
        opponent_sine = abs(float(np.sin(opponent_heading_error)))
        opponent_longitudinal = 0.5 * (self.vehicle_length * opponent_cosine + self.vehicle_width * opponent_sine)
        opponent_lateral = 0.5 * (self.vehicle_length * opponent_sine + self.vehicle_width * opponent_cosine)

        front_gap_m = opponent_ahead_center_m - ego_longitudinal - opponent_longitudinal
        lateral_overlap_m = min(ego_lateral_d + ego_lateral, opponent_lateral_d + opponent_lateral) - max(ego_lateral_d - ego_lateral, opponent_lateral_d - opponent_lateral)
        self.front_corridor_gate = bool(
            opponent_ahead_center_m > 0.0
            and opponent_ahead_center_m < 0.5 * self.projector.track_length
            and front_gap_m > 0.0
            and front_gap_m < self.config.front_corridor_gate_maximum_gap_m
            and abs(opponent_lateral_d) < self.config.front_corridor_gate_maximum_abs_opponent_lateral_d_m
            and lateral_overlap_m > 0.0
        )
        return self.front_corridor_gate

    def _clearance_risk(self, ego_pose, opponent_pose):
        (
            _,
            self.current_obb_longitudinal_clearance_m,
            self.current_obb_lateral_clearance_m,
            self.current_wall_clearance_m,
        ) = self.clearance_calculator.calculate(ego_pose, opponent_pose)
        return risk_potential(
            self.current_obb_longitudinal_clearance_m,
            self.current_obb_lateral_clearance_m,
            self.current_wall_clearance_m,
            longitudinal_safe_m=self.config.risk_longitudinal_clearance_m,
            lateral_safe_m=self.config.risk_lateral_clearance_m,
            wall_safe_m=self.config.risk_wall_clearance_m,
            maximum_magnitude=1.0,
        )

    def _transition_reward(self, raw_observation, ego_collision, opponent_collision, terminated):
        # Update track progress before calculating the four reward terms.
        ego_pose = np.asarray((raw_observation["poses_x"][0], raw_observation["poses_y"][0], raw_observation["poses_theta"][0]), dtype=np.float64)
        opponent_pose = np.asarray((raw_observation["poses_x"][1], raw_observation["poses_y"][1], raw_observation["poses_theta"][1]), dtype=np.float64)
        ego_progress = self.projector.progress_at(ego_pose[:2])
        opponent_progress = self.projector.progress_at(opponent_pose[:2])
        ego_delta = wrapped_progress_delta(ego_progress, self._previous_ego_progress, self.projector.track_length)
        opponent_delta = wrapped_progress_delta(opponent_progress, self._previous_opponent_progress, self.projector.track_length)
        self._previous_ego_progress = ego_progress
        self._previous_opponent_progress = opponent_progress
        self.relative_position_m += ego_delta - opponent_delta
        self._opponent_collision_latched = self._opponent_collision_latched or opponent_collision

        reward_progress = progress_reward(ego_delta, self.reward_weights[0])
        reward_relative = relative_reward(ego_delta, opponent_delta, self._opponent_collision_latched, self.reward_weights[1])
        reward_collision = collision_reward(ego_collision, self.reward_weights[2])
        physical_risk_potential = self._clearance_risk(ego_pose, opponent_pose)
        reward_risk, self._previous_risk_potential = risk_reward(self._previous_risk_potential, physical_risk_potential, self.reward_gamma, terminated, self.reward_weights[3])
        return float(reward_progress + reward_relative + reward_collision + reward_risk)

    def _observation(self, raw_observation):
        lidar = np.asarray(raw_observation["scans"][0]).reshape(-1)
        if lidar.size > END2RACE_LIDAR_SIZE:
            lidar = lidar[np.linspace(0, lidar.size - 1, END2RACE_LIDAR_SIZE, dtype=int)]
        observation = end2race_observation(np.asarray(lidar, dtype=np.float32), self._previous_ego_speed)
        if self.privileged_extractor is None:
            return observation
        agents = self.f110_env.unwrapped.sim.agents
        ego_state = np.asarray(agents[0].state, dtype=np.float64).reshape(-1)
        opponent_state = np.asarray(agents[1].state, dtype=np.float64).reshape(-1)
        features = self.privileged_extractor.features(
            raw_observation,
            ego_index=0,
            opponent_index=1,
            ego_progress=self._previous_ego_progress,
            opponent_progress=self._previous_opponent_progress,
            ego_steering_angle=float(ego_state[2]),
            ego_slip_angle=float(ego_state[6]),
            opponent_slip_angle=float(opponent_state[6]),
            obb_longitudinal_clearance_m=self.current_obb_longitudinal_clearance_m,
            obb_lateral_clearance_m=self.current_obb_lateral_clearance_m,
            wall_clearance_m=self.current_wall_clearance_m,
        )
        return np.concatenate((observation, features))

    def privileged_normalization_metadata(self):
        return self.privileged_extractor.normalization_metadata()

    def reset(self, *, seed=None, options=None):
        from demonstration import setup_opp_planner

        super().reset(seed=seed)
        spec = options["end2race_episode_reset_spec"]
        raw_observation, _, _, _ = self.f110_env.reset(poses=spec["poses"].copy())
        self._elapsed_time = 0.0
        self._episode_return = 0.0
        self._episode_steps = 0
        self._raw_observation = raw_observation
        self._previous_ego_speed = float(spec["initial_speed_feature"])
        self._current_spec = spec

        # Initialize the previous-state values used by transition rewards.
        ego_pose = np.asarray((raw_observation["poses_x"][0], raw_observation["poses_y"][0], raw_observation["poses_theta"][0]), dtype=np.float64)
        opponent_pose = np.asarray((raw_observation["poses_x"][1], raw_observation["poses_y"][1], raw_observation["poses_theta"][1]), dtype=np.float64)
        self._previous_ego_progress = self.projector.progress_at(ego_pose[:2])
        self._previous_opponent_progress = self.projector.progress_at(opponent_pose[:2])
        self.relative_position_m = wrapped_progress_delta(self._previous_ego_progress, self._previous_opponent_progress, self.projector.track_length)
        self._opponent_collision_latched = False
        self._previous_risk_potential = self._clearance_risk(ego_pose, opponent_pose)
        if self.use_front_corridor_gate:
            self._front_corridor_gate(raw_observation)

        # Reuse a planner template, then reset episode-local tracking state.
        scenario = spec["scenario"]
        key = (str(scenario["map_name"]), str(scenario["opp_raceline"]))
        template = self._planner_templates.get(key)
        if template is None:
            template = setup_opp_planner(*key)
            self._planner_templates[key] = template
        self._opponent_planner = copy_module.copy(template)
        self._opponent_planner.tracker = copy_module.copy(template.tracker)
        self._opponent_planner.reset()
        self._opponent_trajectory = None
        self._opponent_tracker_count = 0
        self._opponent_speed_scale = float(scenario["opp_speedscale"])
        info = self._info(False, False, None, None)
        return self._observation(raw_observation), info

    def _info(self, opponent_collision, timeout, reason, outcome):
        scenario = self._current_spec["scenario"]
        return {
            "opponent_collision": opponent_collision,
            "timeout": timeout,
            "elapsed_time": self._elapsed_time,
            "termination_reason": reason,
            "scenario_id": str(scenario["scenario_id"]),
            "env_role": str(scenario["env_role"]),
            "episode_outcome": outcome,
            "episode_return": self._episode_return,
            "episode_steps": self._episode_steps,
            self.config.exploration_gate_info_key: self.front_corridor_gate,
        }

    def step(self, action):
        from demonstration import lattice_opponent_action

        previous_raw_observation = self._raw_observation
        previous_ego_speed = float(np.asarray(previous_raw_observation["linear_vels_x"])[0])
        opponent_steering, opponent_speed, self._opponent_trajectory, self._opponent_tracker_count = lattice_opponent_action(
            self._opponent_planner,
            previous_raw_observation,
            self._opponent_trajectory,
            self._opponent_tracker_count,
            self._opponent_speed_scale,
            steering_bound=self.config.steering_bound,
            opponent_index=1,
        )
        joint_action = np.stack(
            (np.asarray(action, dtype=np.float32).reshape(2), np.asarray((opponent_steering, opponent_speed), dtype=np.float32))
        )
        raw_observation, simulator_reward, base_terminated, _ = self.f110_env.step(joint_action)
        self._elapsed_time += float(simulator_reward)
        collisions = np.asarray(raw_observation["collisions"], dtype=bool).reshape(-1)
        ego_collision = bool(collisions[0])
        opponent_collision = bool(collisions[1])
        timeout = self._elapsed_time + 1e-12 >= self.config.episode_horizon
        if ego_collision or (base_terminated and not opponent_collision):
            terminated, truncated = True, False
            reason = "ego_collision" if ego_collision else "base_terminated"
        elif timeout:
            terminated, truncated, reason = False, True, "timeout"
        else:
            terminated, truncated, reason = False, False, None
        reward = self._transition_reward(raw_observation, ego_collision, opponent_collision, terminated)
        self._episode_return += reward
        self._episode_steps += 1
        outcome = None
        if terminated or truncated:
            if ego_collision:
                outcome = "ego_collision"
            elif self.relative_position_m > 0.0:
                outcome = "overtake"
            else:
                outcome = "follow"
        self._raw_observation = raw_observation
        self._previous_ego_speed = previous_ego_speed
        if self.use_front_corridor_gate:
            self._front_corridor_gate(raw_observation)
        info = self._info(
            opponent_collision,
            timeout,
            reason,
            outcome,
        )
        return self._observation(raw_observation), reward, terminated, truncated, info

    def close(self):
        self.f110_env.close()


def make_environment(seed, map_name, config, privileged=False, reward_gamma=0.999, reward_weights=None, front_corridor_speed_noise_hold_steps=0):

    if reward_weights is None:
        reward_weights = (
            config.progress_weight,
            config.relative_weight,
            config.collision_penalty,
            config.risk_potential_maximum,
        )

    def factory():
        import gym
        from f110_gym.envs.base_classes import Integrator

        map_path = get_map_paths(map_name)[1]
        warnings.filterwarnings("ignore", message="Chosen integrator is RK4.*", category=UserWarning, module="f110_gym.envs.base_classes")
        core = gym.make(
            "f110-v0",
            map=map_path,
            map_ext=".png",
            num_agents=2,
            timestep=config.simulator_timestep,
            integrator=Integrator.RK4,
            seed=seed,
        )
        return End2RaceGymnasiumEnv(
            core,
            map_name,
            config.ego_raceline,
            config,
            reward_weights,
            privileged=privileged,
            reward_gamma=reward_gamma,
            front_corridor_speed_noise_hold_steps=front_corridor_speed_noise_hold_steps,
        )

    return factory


def _worker(remote, parent_remote, env_fn_wrapper):
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    env = None
    try:
        env = _patch_env(env_fn_wrapper.var())
        while True:
            command, data = remote.recv()
            if command == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                remote.send(("ok", (observation, reward, terminated, truncated, info)))
            elif command == "reset":
                seed, spec = data
                observation, reset_info = env.reset(
                    seed=seed,
                    options={"end2race_episode_reset_spec": spec},
                )
                remote.send(("ok", (observation, reset_info)))
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

    def __init__(self, n_envs, start_method, seed, map_name, config, collision_scenarios, ordinary_scenarios, privileged=False, reward_gamma=0.999, reward_weights=None, front_corridor_speed_noise_hold_steps=0):
        self.waiting = False
        self.closed = False
        self.scheduler = ScenarioScheduler(seed, collision_scenarios, ordinary_scenarios)
        logical_seeds = [
            int(np.random.SeedSequence([seed, 1, rank % 2, rank // 2]).generate_state(1)[0])
            for rank in range(n_envs)
        ]
        env_fns = [
            make_environment(
                logical_seeds[rank],
                map_name,
                config,
                privileged=privileged,
                reward_gamma=reward_gamma,
                reward_weights=reward_weights,
                front_corridor_speed_noise_hold_steps=front_corridor_speed_noise_hold_steps,
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
            for work_remote, remote, env_fn in zip(work_remotes, self.remotes, env_fns):
                process = context.Process(
                    target=_worker,
                    args=(work_remote, remote, CloudpickleWrapper(env_fn)),
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

    def seed(self, seed):
        self._seeds = [
            int(np.random.SeedSequence([seed, 1, rank % 2, rank // 2]).generate_state(1)[0])
            for rank in range(self.num_envs)
        ]
        return self._seeds

    def _recv_checked(self, rank):
        try:
            status, payload = self.remotes[rank].recv()
        except (EOFError, BrokenPipeError, OSError) as error:
            self._terminate_workers()
            raise RuntimeError(f"environment worker {rank} exited unexpectedly") from error
        if status != "ok":
            self._terminate_workers()
            raise RuntimeError(f"environment worker {rank} failed:\n{payload}")
        return payload

    def _terminate_workers(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        self.closed = True

    def _reset_round(self, indices, seeds):
        for rank, seed in zip(indices, seeds):
            self.remotes[rank].send(("reset", (seed, self.scheduler.next(rank))))
        observations = []
        for rank in indices:
            observation, reset_info = self._recv_checked(rank)
            self.reset_infos[rank] = reset_info
            observations.append(observation)
        return observations

    def reset(self):
        indices = list(range(self.num_envs))
        observations = self._reset_round(indices, list(self._seeds))
        self._reset_seeds()
        self._reset_options()
        return np.stack(observations)

    def step_async(self, actions):
        for rank, action in enumerate(actions):
            self.remotes[rank].send(("step", action))
        self.waiting = True

    def step_wait(self):
        rows = [self._recv_checked(rank) for rank in range(self.num_envs)]
        self.waiting = False
        observations, rewards, terminated, truncated, infos = map(list, zip(*rows))
        rewards = np.asarray(rewards, dtype=np.float32)
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)
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

    def close(self):
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

    def has_attr(self, attr_name):
        for remote in self.remotes:
            remote.send(("has_attr", attr_name))
        return all(self._recv_checked(rank) for rank in range(self.num_envs))

    def get_attr(self, attr_name, indices=None):
        ranks = self._get_indices(indices)
        for rank in ranks:
            self.remotes[rank].send(("get_attr", attr_name))
        return [self._recv_checked(rank) for rank in ranks]

    def set_attr(self, attr_name, value, indices=None):
        ranks = self._get_indices(indices)
        for rank in ranks:
            self.remotes[rank].send(("set_attr", (attr_name, value)))
        for rank in ranks:
            self._recv_checked(rank)

    def env_method(
        self,
        method_name,
        *method_args,
        indices=None,
        **method_kwargs,
    ):
        ranks = self._get_indices(indices)
        for rank in ranks:
            self.remotes[rank].send(("env_method", (method_name, method_args, method_kwargs)))
        return [self._recv_checked(rank) for rank in ranks]

    def env_is_wrapped(self, wrapper_class, indices=None):
        ranks = self._get_indices(indices)
        for rank in ranks:
            self.remotes[rank].send(("is_wrapped", wrapper_class))
        return [self._recv_checked(rank) for rank in ranks]

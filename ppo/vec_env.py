"""Multiprocess vector environment and parent-side scenario scheduling."""

from __future__ import annotations

from collections.abc import Sequence
import multiprocessing as mp
import os
import traceback
from typing import Any
import warnings

import gymnasium as gym
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env

from ppo.environment import EXTERNAL_RESET_OPTION, make_environment
from ppo.scenarios import ScenarioScheduler, ScenarioSpec


ENV_WORKERS = 6


def _limit_worker_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


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

    def __init__(self, n_envs: int, seed: int, hard_scenarios: Sequence[ScenarioSpec]) -> None:
        if n_envs < ENV_WORKERS or n_envs % 2 != 0:
            raise ValueError(f"n_envs must be even and at least {ENV_WORKERS}")
        if torch.cuda.is_initialized():
            raise RuntimeError("Subprocess environments must be created before CUDA initialization")
        self.waiting = False
        self.closed = False
        self.actions = None
        self.worker_count = ENV_WORKERS
        self.scheduler = ScenarioScheduler(seed, hard_scenarios)
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

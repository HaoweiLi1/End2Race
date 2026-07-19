"""Process-parallel VecEnv with deterministic parent-side episode scheduling.

The pipe/cloudpickle lifecycle follows SB3's validated ``SubprocVecEnv``
design, but reset selection is deliberately kept in the parent process.  A
worker can host one or more environments so the process count can be tuned
without changing the fixed 16-env rollout configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import multiprocessing as mp
import os
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

from ppo.config import PPOConfig
from ppo.environment import EXTERNAL_RESET_OPTION, EpisodeResetSpec
from ppo.scenarios import FixedMixtureScenarioSampler


def _limit_worker_threads() -> None:
    """Prevent simulator workers from oversubscribing BLAS/OpenMP pools."""

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_info, threadpool_limits

        threadpool_limits(limits=1)
    except ImportError:
        threadpool_info = None
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Simulator worker Torch thread pools must both be limited to one")
    if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise RuntimeError("Simulator worker OpenMP/BLAS thread environment must be limited to one")
    if threadpool_info is not None:
        oversized = [
            pool for pool in threadpool_info()
            if pool.get("num_threads") not in (None, 1)
        ]
        if oversized:
            raise RuntimeError(f"Simulator worker native thread pools exceed one thread: {oversized}")


def _worker(
    remote: Any,
    parent_remote: Any,
    env_fn_wrappers: list[CloudpickleWrapper],
    env_indices: list[int],
) -> None:
    """Execute simulator calls only; reset selection remains in the parent."""

    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    envs: list[gym.Env] = []
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
                env = envs[0]
                remote.send(("ok", (env.observation_space, env.action_space)))
            elif command == "env_method":
                results = []
                for rank, method_name, method_args, method_kwargs in data:
                    method = by_rank[rank].get_wrapper_attr(method_name)
                    results.append((rank, method(*method_args, **method_kwargs)))
                remote.send(("ok", results))
            elif command == "get_attr":
                remote.send(
                    ("ok", [(rank, by_rank[rank].get_wrapper_attr(attr_name)) for rank, attr_name in data])
                )
            elif command == "has_attr":
                results = []
                for rank, attr_name in data:
                    try:
                        by_rank[rank].get_wrapper_attr(attr_name)
                        result = True
                    except AttributeError:
                        result = False
                    results.append((rank, result))
                remote.send(("ok", results))
            elif command == "set_attr":
                results = []
                for rank, attr_name, value in data:
                    setattr(by_rank[rank], attr_name, value)
                    results.append((rank, None))
                remote.send(("ok", results))
            elif command == "is_wrapped":
                remote.send(
                    ("ok", [(rank, is_wrapped(by_rank[rank], wrapper_class)) for rank, wrapper_class in data])
                )
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


class CentralEpisodeScheduler:
    """Reproduce the existing per-env RNG streams in deterministic rank order."""

    def __init__(
        self,
        sampler: FixedMixtureScenarioSampler,
        config: PPOConfig,
        master_seed: int,
        n_envs: int,
    ) -> None:
        self.sampler = sampler
        self.config = config
        self.master_seed = int(master_seed)
        self.rngs: list[np.random.Generator | None] = [None] * n_envs
        self.pair_episode_ordinals = [0] * n_envs
        self.reset_history: list[tuple[int, str, str, int | None, int | None, int | None]] = []

    def _role(self, rank: int) -> str | None:
        if self.config.fixed_hard_env_count is None:
            return None
        return "hard" if rank < self.config.fixed_hard_env_count else "ordinary"

    def reseed(self, rank: int, seed: int) -> None:
        self.rngs[rank] = np.random.default_rng(seed)
        self.pair_episode_ordinals[rank] = 0

    def next(self, rank: int, seed: int | None = None) -> EpisodeResetSpec:
        if seed is not None:
            self.reseed(rank, seed)
        if self.rngs[rank] is None:
            self.rngs[rank] = np.random.default_rng()
        rng = self.rngs[rank]
        assert rng is not None
        role = self._role(rank)
        if self.config.paired_hard_sampling and role == "hard":
            ordinal = self.pair_episode_ordinals[rank]
            spec = self.sampler.reset_spec(
                rng,
                env_role="hard",
                pair_seed=self.master_seed,
                pair_group=rank // self.config.hard_pair_size,
                pair_member=rank % self.config.hard_pair_size,
                pair_episode_ordinal=ordinal,
            )
            self.pair_episode_ordinals[rank] += 1
        else:
            spec = self.sampler.reset_spec(rng, env_role=role)
        scenario = spec.scenario
        self.reset_history.append(
            (
                rank,
                str(scenario["scenario_id"]),
                str(scenario["env_role"]),
                scenario.get("pair_group"),
                scenario.get("pair_member"),
                scenario.get("pair_episode_ordinal"),
            )
        )
        return spec


class CentralScheduleSubprocVecEnv(VecEnv):
    """Parallel environments with all scenario decisions made in the parent."""

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        *,
        sampler: FixedMixtureScenarioSampler,
        config: PPOConfig,
        seed: int,
        worker_count: int | None = None,
        start_method: str | None = None,
    ) -> None:
        if torch.cuda.is_initialized():
            raise RuntimeError("Subprocess environments must be created before CUDA initialization")
        n_envs = len(env_fns)
        count = n_envs if worker_count is None else int(worker_count)
        if not 1 <= count <= n_envs:
            raise ValueError(f"worker_count must be in [1, {n_envs}], got {count}")
        self.waiting = False
        self.closed = False
        self.actions: np.ndarray | None = None
        self.worker_count = count
        self.scheduler = CentralEpisodeScheduler(sampler, config, seed, n_envs)
        if start_method is None:
            start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        self.start_method = start_method
        context = mp.get_context(start_method)
        self.worker_env_indices = [[] for _ in range(count)]
        for rank in range(n_envs):
            self.worker_env_indices[rank % count].append(rank)
        self.env_to_worker = {
            rank: worker_index
            for worker_index, ranks in enumerate(self.worker_env_indices)
            for rank in ranks
        }
        self.remotes, work_remotes = zip(*[context.Pipe() for _ in range(count)])
        self.processes: list[mp.Process] = []
        thread_env = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        previous_thread_env = {name: os.environ.get(name) for name in thread_env}
        for name in thread_env:
            os.environ[name] = "1"
        try:
            for worker_index, (work_remote, remote) in enumerate(zip(work_remotes, self.remotes)):
                ranks = self.worker_env_indices[worker_index]
                wrappers = [CloudpickleWrapper(env_fns[rank]) for rank in ranks]
                process = context.Process(
                    target=_worker,
                    args=(work_remote, remote, wrappers, ranks),
                    daemon=True,
                )
                process.start()
                self.processes.append(process)
                work_remote.close()
            for name, value in previous_thread_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            self.remotes[0].send(("get_spaces", None))
            observation_space, action_space = self._recv_checked(0)
            super().__init__(n_envs, observation_space, action_space)
        except BaseException:
            for name, value in previous_thread_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            self._terminate_workers()
            raise

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
        grouped: dict[int, list[tuple[Any, ...]]] = {}
        for entry in entries:
            grouped.setdefault(self.env_to_worker[int(entry[0])], []).append(entry)
        return grouped

    def _reset_round(self, indices: list[int], seeds: list[int | None]) -> list[VecEnvObs]:
        entries = [(rank, seed, self.scheduler.next(rank, seed)) for rank, seed in zip(indices, seeds)]
        grouped = self._group_entries(entries)
        for worker_index, worker_entries in grouped.items():
            self.remotes[worker_index].send(("reset", worker_entries))
        worker_results = {
            worker_index: self._recv_checked(worker_index)
            for worker_index in grouped
        }
        flattened = [result for results in worker_results.values() for result in results]
        by_rank = {int(result[0]): result for result in flattened}
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
        return _stack_observations(observations, self.observation_space)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        entries = [(rank, action) for rank, action in enumerate(actions)]
        grouped = self._group_entries(entries)
        for worker_index, worker_entries in grouped.items():
            self.remotes[worker_index].send(("step", worker_entries))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        worker_results = [self._recv_checked(index) for index in range(self.worker_count)]
        self.waiting = False
        flattened = [result for results in worker_results for result in results]
        by_rank = {int(result[0]): result for result in flattened}
        observations = [by_rank[rank][1] for rank in range(self.num_envs)]
        rewards = np.asarray([by_rank[rank][2] for rank in range(self.num_envs)], dtype=np.float32)
        terminated = np.asarray([by_rank[rank][3] for rank in range(self.num_envs)], dtype=bool)
        truncated = np.asarray([by_rank[rank][4] for rank in range(self.num_envs)], dtype=bool)
        infos = [by_rank[rank][5] for rank in range(self.num_envs)]
        dones = np.logical_or(terminated, truncated)

        reset_indices: list[int] = []
        for index, done in enumerate(dones):
            infos[index]["TimeLimit.truncated"] = bool(truncated[index] and not terminated[index])
            if done:
                infos[index]["terminal_observation"] = observations[index]
                reset_indices.append(index)
        if reset_indices:
            reset_observations = self._reset_round(reset_indices, [None] * len(reset_indices))
            for index, observation in zip(reset_indices, reset_observations):
                observations[index] = observation
        return (
            _stack_observations(observations, self.observation_space),
            rewards,
            dones,
            tuple(infos),
        )

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            try:
                for worker_index in range(self.worker_count):
                    self._recv_checked(worker_index)
            finally:
                self.waiting = False
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5.0)
        alive = [process for process in self.processes if process.is_alive()]
        if alive:
            self._terminate_workers()
            raise RuntimeError(f"{len(alive)} environment workers did not exit normally")
        self.closed = True

    def _request_by_rank(
        self,
        command: str,
        entries: list[tuple[Any, ...]],
        ordered_ranks: list[int],
    ) -> list[Any]:
        grouped = self._group_entries(entries)
        for worker_index, worker_entries in grouped.items():
            self.remotes[worker_index].send((command, worker_entries))
        flattened = [
            result
            for worker_index in grouped
            for result in self._recv_checked(worker_index)
        ]
        by_rank = {int(rank): value for rank, value in flattened}
        return [by_rank[rank] for rank in ordered_ranks]

    def get_images(self) -> Sequence[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            warnings.warn(f"render_mode is {self.render_mode}, not rgb_array")
            return [None] * self.num_envs
        ranks = list(range(self.num_envs))
        grouped = self._group_entries([(rank,) for rank in ranks])
        for worker_index, entries in grouped.items():
            self.remotes[worker_index].send(("render", [entry[0] for entry in entries]))
        flattened = [
            result
            for worker_index in grouped
            for result in self._recv_checked(worker_index)
        ]
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


def _stack_observations(observations: Sequence[VecEnvObs], space: spaces.Space) -> VecEnvObs:
    if isinstance(space, spaces.Dict):
        return {
            key: np.stack([observation[key] for observation in observations])
            for key in space.spaces
        }
    if isinstance(space, spaces.Tuple):
        return tuple(
            np.stack([observation[index] for observation in observations])
            for index in range(len(space.spaces))
        )
    return np.stack(observations)

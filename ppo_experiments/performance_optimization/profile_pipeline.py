#!/usr/bin/env python3
"""Profile one fresh End2Race PPO update without changing training semantics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import functools
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import types
from typing import Any

import numpy as np
import psutil
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo import config as ppo_config
from train_ppo import (
    PPOTrainingCallback,
    _actor_delta_record,
    _optimizer_step,
    build_model,
    build_sampler,
    make_training_env,
    make_subprocess_training_env,
    save_actor,
    write_json,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="N1-H1F-p50", choices=ppo_config.CONFIGS)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--mode", default="dummy", choices=("dummy", "central_subproc"))
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


class ResourceMonitor:
    """Sample process-tree CPU/RSS and device-level NVML metrics out of band."""

    def __init__(self) -> None:
        self.root = psutil.Process()
        self.samples: list[dict[str, float]] = []
        self.gpu_samples: list[dict[str, float]] = []
        self.stop_event = threading.Event()
        self.sample_thread: threading.Thread | None = None
        self.dmon_thread: threading.Thread | None = None
        self.dmon: subprocess.Popen[str] | None = None
        self.context_baseline: dict[int, tuple[int, int]] = {}

    def _training_processes(self) -> list[psutil.Process]:
        processes = [self.root]
        try:
            children = self.root.children(recursive=True)
        except psutil.Error:
            children = []
        dmon_pid = None if self.dmon is None else self.dmon.pid
        processes.extend(child for child in children if child.pid != dmon_pid)
        return processes

    def start(self) -> None:
        psutil.cpu_percent(None)
        for process in self._training_processes():
            try:
                process.cpu_percent(None)
            except psutil.Error:
                pass
        self.dmon = subprocess.Popen(
            ["nvidia-smi", "dmon", "-i", "0", "-s", "pucm", "-d", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        def read_gpu() -> None:
            assert self.dmon is not None and self.dmon.stdout is not None
            for line in self.dmon.stdout:
                if self.stop_event.is_set():
                    break
                fields = line.split()
                if fields and fields[0] == "0" and len(fields) >= 15:
                    try:
                        self.gpu_samples.append(
                            {
                                "power_w": float(fields[1]),
                                "sm_pct": float(fields[4]),
                                "memory_activity_pct": float(fields[5]),
                                "framebuffer_mib": float(fields[12]),
                            }
                        )
                    except ValueError:
                        pass

        self.dmon_thread = threading.Thread(target=read_gpu, daemon=True)
        self.dmon_thread.start()

        def sample_resources() -> None:
            known_pids: set[int] = set()
            while not self.stop_event.wait(0.2):
                processes = self._training_processes()
                cpu_pct = 0.0
                rss_bytes = 0
                pss_bytes = 0
                thread_count = 0
                alive = 0
                voluntary_context_switches = 0
                involuntary_context_switches = 0
                for process in processes:
                    try:
                        if process.pid not in known_pids:
                            process.cpu_percent(None)
                            known_pids.add(process.pid)
                        cpu_pct += process.cpu_percent(None)
                        rss_bytes += process.memory_info().rss
                        full_memory = process.memory_full_info()
                        pss_bytes += getattr(full_memory, "pss", process.memory_info().rss)
                        thread_count += process.num_threads()
                        context = process.num_ctx_switches()
                        baseline = self.context_baseline.setdefault(
                            process.pid,
                            (context.voluntary, context.involuntary),
                        )
                        voluntary_context_switches += max(0, context.voluntary - baseline[0])
                        involuntary_context_switches += max(0, context.involuntary - baseline[1])
                        alive += 1
                    except psutil.Error:
                        pass
                self.samples.append(
                    {
                        "process_cpu_pct": cpu_pct,
                        "system_cpu_pct": psutil.cpu_percent(None),
                        "rss_mib": rss_bytes / 2**20,
                        "pss_mib": pss_bytes / 2**20,
                        "threads": float(thread_count),
                        "processes": float(alive),
                        "voluntary_context_switches": float(voluntary_context_switches),
                        "involuntary_context_switches": float(involuntary_context_switches),
                    }
                )

        self.sample_thread = threading.Thread(target=sample_resources, daemon=True)
        self.sample_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.sample_thread is not None:
            self.sample_thread.join(timeout=2)
        if self.dmon is not None and self.dmon.poll() is None:
            self.dmon.terminate()
            try:
                self.dmon.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.dmon.kill()
        if self.dmon_thread is not None:
            self.dmon_thread.join(timeout=2)

    @staticmethod
    def _statistics(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None, "max": None}
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    def summary(self) -> dict[str, dict[str, float | None]]:
        keys = (
            "process_cpu_pct",
            "system_cpu_pct",
            "rss_mib",
            "pss_mib",
            "threads",
            "processes",
            "voluntary_context_switches",
            "involuntary_context_switches",
        )
        result = {key: self._statistics([sample[key] for sample in self.samples]) for key in keys}
        result.update(
            {
                "gpu_sm_pct_device": self._statistics([sample["sm_pct"] for sample in self.gpu_samples]),
                "gpu_memory_activity_pct_device": self._statistics(
                    [sample["memory_activity_pct"] for sample in self.gpu_samples]
                ),
                "gpu_framebuffer_mib_device": self._statistics(
                    [sample["framebuffer_mib"] for sample in self.gpu_samples]
                ),
                "gpu_power_w": self._statistics([sample["power_w"] for sample in self.gpu_samples]),
            }
        )
        return result


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def install_read_only_detail_timers(times: defaultdict[str, float]) -> None:
    """Monkeypatch methods only inside this profiler process; source is untouched."""

    from f110_gym.envs.base_classes import RaceCar, Simulator
    from f110_gym.envs.f110_env import F110Env
    from f110_gym.envs.laser_models import ScanSimulator2D
    from latticeplanner.lattice_planner import LatticePlanner
    from latticeplanner.pure_pursuit import PurePursuitPlanner
    from ppo.environment import End2RaceGymnasiumEnv, LatticePlannerOpponentController
    from ppo.reward import ProgressProjector

    def patch(target: Any, method_name: str, metric: str) -> None:
        original = getattr(target, method_name)

        @functools.wraps(original)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                times[f"{metric}_s"] += time.perf_counter() - start
                times[f"{metric}_calls"] += 1

        setattr(target, method_name, timed)

    patch(RaceCar, "update_pose", "sim_update_pose_including_map_lidar")
    patch(RaceCar, "check_ttc", "sim_environment_collision")
    patch(RaceCar, "ray_cast_agents", "sim_agent_lidar_ray_cast")
    patch(Simulator, "check_collision", "sim_agent_collision")
    patch(Simulator, "step", "simulator_step")
    patch(ScanSimulator2D, "scan", "sim_map_lidar")
    patch(F110Env, "reset", "f110_reset")
    patch(LatticePlanner, "__init__", "planner_static_asset_load")
    patch(LatticePlanner, "plan", "opponent_replan")
    patch(PurePursuitPlanner, "plan", "opponent_tracking")
    patch(LatticePlannerOpponentController, "actions", "opponent_controller")
    patch(ProgressProjector, "project", "progress_projection")
    patch(End2RaceGymnasiumEnv, "_observation", "observation_construction")


def profile_one(
    config_name: str,
    seed: int,
    label: str,
    mode: str = "dummy",
    worker_count: int | None = None,
) -> dict[str, Any]:
    startup_start = time.perf_counter()
    config = ppo_config.get_config(config_name)
    times: defaultdict[str, float] = defaultdict(float)
    reset_order: list[tuple[Any, ...]] = []
    event_counts: Counter[str] = Counter()
    sampler_start = time.perf_counter()
    sampler = build_sampler(config)
    sampler_startup_s = time.perf_counter() - sampler_start
    install_read_only_detail_timers(times)
    factories = []

    for rank in range(ppo_config.N_ENVS):
        base_factory = (
            make_training_env(rank, sampler, config, seed)
            if mode == "dummy"
            else make_subprocess_training_env(rank, config, seed)
        )

        def profiled_factory(base_factory=base_factory, rank=rank):
            env = base_factory()
            original_step = env.step
            original_reset = env.reset

            def step(this, *args, **kwargs):
                start = time.perf_counter()
                result = original_step(*args, **kwargs)
                times["env_step_s"] += time.perf_counter() - start
                times["env_step_calls"] += 1
                _observation, _reward, terminated, truncated, info = result
                event_counts["terminated"] += int(terminated)
                event_counts["truncated"] += int(truncated)
                event_counts["ego_collision"] += int(info["ego_collision"])
                event_counts["opponent_collision"] += int(info["opponent_collision"])
                event_counts["timeout"] += int(info["timeout"])
                if info["episode_outcome"] is not None:
                    event_counts[str(info["episode_outcome"])] += 1
                return result

            def reset(this, *args, **kwargs):
                start = time.perf_counter()
                result = original_reset(*args, **kwargs)
                times["env_reset_s"] += time.perf_counter() - start
                times["env_reset_calls"] += 1
                info = result[1]
                reset_order.append(
                    (
                        rank,
                        str(info["scenario_id"]),
                        str(info["env_role"]),
                        info.get("pair_group"),
                        info.get("pair_member"),
                        info.get("pair_episode_ordinal"),
                    )
                )
                return result

            env.step = types.MethodType(step, env)
            env.reset = types.MethodType(reset, env)
            return env

        factories.append(profiled_factory if mode == "dummy" else base_factory)

    vector_start = time.perf_counter()
    if mode == "dummy":
        vector_env = DummyVecEnv(factories)
    else:
        from ppo.vec_env import CentralScheduleSubprocVecEnv

        vector_env = CentralScheduleSubprocVecEnv(
            factories,
            sampler=sampler,
            config=config,
            seed=seed,
            worker_count=worker_count,
        )
    vector_startup_s = time.perf_counter() - vector_start
    vector_env.seed(seed)
    if mode == "central_subproc":
        original_vector_reset = vector_env.reset

        def vector_reset():
            observation = original_vector_reset()
            for rank, info in enumerate(vector_env.reset_infos):
                reset_order.append(
                    (
                        rank,
                        str(info["scenario_id"]),
                        str(info["env_role"]),
                        info.get("pair_group"),
                        info.get("pair_member"),
                        info.get("pair_episode_ordinal"),
                    )
                )
            return observation

        vector_env.reset = vector_reset
        original_vector_step_wait = vector_env.step_wait

        def vector_step_wait():
            observation, rewards, dones, infos = original_vector_step_wait()
            for rank, (done, info) in enumerate(zip(dones, infos)):
                event_counts["terminated"] += int(bool(done) and not info["TimeLimit.truncated"])
                event_counts["truncated"] += int(bool(info["TimeLimit.truncated"]))
                event_counts["ego_collision"] += int(bool(info["ego_collision"]))
                event_counts["opponent_collision"] += int(bool(info["opponent_collision"]))
                event_counts["timeout"] += int(bool(info["timeout"]))
                if info["episode_outcome"] is not None:
                    event_counts[str(info["episode_outcome"])] += 1
                if done:
                    reset_info = vector_env.reset_infos[rank]
                    reset_order.append(
                        (
                            rank,
                            str(reset_info["scenario_id"]),
                            str(reset_info["env_role"]),
                            reset_info.get("pair_group"),
                            reset_info.get("pair_member"),
                            reset_info.get("pair_episode_ordinal"),
                        )
                    )
            return observation, rewards, dones, infos

        vector_env.step_wait = vector_step_wait
    model_start = time.perf_counter()
    model = build_model(vector_env, config, seed)
    model_startup_s = time.perf_counter() - model_start
    startup_s = time.perf_counter() - startup_start
    callback = PPOTrainingCallback()
    rollout_buffer = model.rollout_buffer

    def time_bound_method(target: Any, method_name: str, metric: str, *, synchronize: bool = False) -> None:
        original = getattr(target, method_name)

        @functools.wraps(original)
        def timed(*args, **kwargs):
            if synchronize:
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            if synchronize:
                torch.cuda.synchronize()
            times[f"{metric}_s"] += time.perf_counter() - start
            times[f"{metric}_calls"] += 1
            return result

        setattr(target, method_name, timed)

    time_bound_method(model.policy, "_actor_forward", "policy_actor_total", synchronize=True)
    time_bound_method(model.policy.end2race_actor, "forward", "policy_actor_module_enqueue")
    time_bound_method(model.policy.end2race_actor.gru, "forward", "policy_gru_enqueue")
    time_bound_method(model.policy, "_critic_values", "policy_critic_enqueue")

    original_optimizer_step = model.policy.optimizer.step

    def optimizer_step(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_optimizer_step(*args, **kwargs)
        torch.cuda.synchronize()
        times["optimizer_step_s"] += time.perf_counter() - start
        times["optimizer_step_calls"] += 1
        return result

    model.policy.optimizer.step = optimizer_step

    original_backward = torch.autograd.backward

    def backward(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_backward(*args, **kwargs)
        torch.cuda.synchronize()
        times["backward_s"] += time.perf_counter() - start
        times["backward_calls"] += 1
        return result

    torch.autograd.backward = backward

    original_setup_learn = model._setup_learn

    def setup_learn(*args, **kwargs):
        result = original_setup_learn(*args, **kwargs)
        original_logger_record = model.logger.record
        original_logger_dump = model.logger.dump

        def logger_record(*record_args, **record_kwargs):
            start = time.perf_counter()
            record_result = original_logger_record(*record_args, **record_kwargs)
            times["logger_record_s"] += time.perf_counter() - start
            times["logger_record_calls"] += 1
            return record_result

        def logger_dump(*dump_args, **dump_kwargs):
            start = time.perf_counter()
            dump_result = original_logger_dump(*dump_args, **dump_kwargs)
            times["logger_dump_s"] += time.perf_counter() - start
            times["logger_dump_calls"] += 1
            return dump_result

        model.logger.record = logger_record
        model.logger.dump = logger_dump
        return result

    model._setup_learn = setup_learn

    from sb3_contrib.ppo_recurrent import ppo_recurrent as recurrent_module

    original_obs_as_tensor = recurrent_module.obs_as_tensor

    def timed_obs_as_tensor(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_obs_as_tensor(*args, **kwargs)
        torch.cuda.synchronize()
        times["collection_observation_h2d_s"] += time.perf_counter() - start
        times["collection_observation_h2d_calls"] += 1
        return result

    recurrent_module.obs_as_tensor = timed_obs_as_tensor
    time_bound_method(model.policy, "obs_to_tensor", "terminal_observation_h2d", synchronize=True)

    original_add = rollout_buffer.add

    def add(*args, **kwargs):
        start = time.perf_counter()
        result = original_add(*args, **kwargs)
        times["buffer_add_s"] += time.perf_counter() - start
        times["buffer_add_calls"] += 1
        return result

    rollout_buffer.add = add
    original_gae = rollout_buffer.compute_returns_and_advantage

    def compute_gae(*args, **kwargs):
        start = time.perf_counter()
        result = original_gae(*args, **kwargs)
        times["gae_s"] += time.perf_counter() - start
        times["gae_calls"] += 1
        return result

    rollout_buffer.compute_returns_and_advantage = compute_gae
    original_get = rollout_buffer.get
    batches: list[dict[str, int]] = []

    def get_batches(batch_size=None):
        iterator = iter(original_get(batch_size))
        while True:
            start = time.perf_counter()
            try:
                data = next(iterator)
            except StopIteration:
                times["sequence_padding_s"] += time.perf_counter() - start
                break
            torch.cuda.synchronize()
            times["sequence_padding_s"] += time.perf_counter() - start
            times["sequence_padding_calls"] += 1
            batches.append(
                {
                    "valid": int(data.mask.sum().item()),
                    "padded": int(data.mask.numel()),
                    "n_seq": int(data.lstm_states.pi[0].shape[1]),
                }
            )
            yield data

    rollout_buffer.get = get_batches
    original_forward = model.policy.forward

    def policy_forward(*args, **kwargs):
        start = time.perf_counter()
        result = original_forward(*args, **kwargs)
        torch.cuda.synchronize()
        times["policy_forward_s"] += time.perf_counter() - start
        times["policy_forward_calls"] += 1
        return result

    model.policy.forward = policy_forward
    original_predict_values = model.policy.predict_values

    def predict_values(*args, **kwargs):
        start = time.perf_counter()
        result = original_predict_values(*args, **kwargs)
        torch.cuda.synchronize()
        times["predict_values_s"] += time.perf_counter() - start
        times["predict_values_calls"] += 1
        return result

    model.policy.predict_values = predict_values
    original_evaluate_actions = model.policy.evaluate_actions

    def evaluate_actions(*args, **kwargs):
        start = time.perf_counter()
        result = original_evaluate_actions(*args, **kwargs)
        torch.cuda.synchronize()
        times["evaluate_actions_s"] += time.perf_counter() - start
        times["evaluate_actions_calls"] += 1
        return result

    model.policy.evaluate_actions = evaluate_actions
    original_train = model.train

    def ppo_train():
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_train()
        torch.cuda.synchronize()
        times["ppo_train_s"] += time.perf_counter() - start
        times["ppo_train_calls"] += 1
        return result

    model.train = ppo_train

    bc_state = torch.load(ppo_config.BC_CHECKPOINT, map_location="cpu", weights_only=True)
    previous_actor_state = {name: tensor.clone() for name, tensor in bc_state.items()}
    initial_log_std = model.policy.log_std.detach().cpu().clone()
    vector_env.env_method("set_policy_update_index", 1)
    torch.cuda.reset_peak_memory_stats()
    monitor = ResourceMonitor()
    monitor.start()
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    learn_start = time.perf_counter()
    with torch.autograd.set_multithreading_enabled(config.autograd_multithreading):
        model.learn(
            total_timesteps=ppo_config.N_ENVS * config.n_steps,
            callback=callback,
            log_interval=None,
            reset_num_timesteps=True,
            progress_bar=False,
        )
    torch.cuda.synchronize()
    learn_s = time.perf_counter() - learn_start

    post_start = time.perf_counter()
    optimizer_step = _optimizer_step(model, require_initialized=True)
    actor_delta = _actor_delta_record(model, bc_state, previous_actor_state, initial_log_std)
    torch.cuda.synchronize()
    post_update_s = time.perf_counter() - post_start
    with tempfile.TemporaryDirectory(prefix=f"end2race_ppo_profile_{label}_") as directory:
        checkpoint = Path(directory) / "actor.pth"
        checkpoint_start = time.perf_counter()
        checkpoint_hash = save_actor(model, checkpoint)
        torch.cuda.synchronize()
        checkpoint_io_s = time.perf_counter() - checkpoint_start
        checkpoint_bytes = checkpoint.stat().st_size
    total_update_s = time.perf_counter() - total_start
    monitor.stop()

    pipeline_profile: dict[str, Any] = {}
    if hasattr(vector_env, "get_pipeline_profile"):
        pipeline_profile = vector_env.get_pipeline_profile()
        times["env_step_s"] = pipeline_profile["step_send_s"] + pipeline_profile["step_wait_s"]
        times["env_step_calls"] = pipeline_profile["step_calls"]
        times["env_reset_s"] = (
            pipeline_profile["reset_schedule_s"]
            + pipeline_profile["reset_send_s"]
            + pipeline_profile["reset_wait_s"]
        )
        times["env_reset_calls"] = pipeline_profile["reset_calls"]

    transitions = ppo_config.N_ENVS * config.n_steps
    episodes = sum(callback.latest["completed_episodes"].values())
    padding_valid = sum(batch["valid"] for batch in batches)
    padding_computed = sum(batch["padded"] for batch in batches)
    rollout_s = learn_s - times["ppo_train_s"]
    actor_hidden_size = model.policy.actor_hidden_size
    stored_recurrent_state_count = sum(
        hasattr(rollout_buffer, name)
        for name in (
            "hidden_states_pi",
            "cell_states_pi",
            "hidden_states_vf",
            "cell_states_vf",
        )
    )
    buffer_bytes = sum(
        value.nbytes for value in vars(rollout_buffer).values() if isinstance(value, np.ndarray)
    )
    detailed_breakdown = {
        "f110_physics_s": max(
            0.0,
            times["sim_update_pose_including_map_lidar_s"] - times["sim_map_lidar_s"],
        ),
        "collision_checking_s": times["sim_agent_collision_s"] + times["sim_environment_collision_s"],
        "lidar_s": times["sim_map_lidar_s"] + times["sim_agent_lidar_ray_cast_s"],
        "observation_construction_s": times["observation_construction_s"],
        "progress_projection_s": times["progress_projection_s"],
        "opponent_replan_s": times["opponent_replan_s"],
        "opponent_tracking_s": times["opponent_tracking_s"],
        "reset_s": times["env_reset_s"],
        "reset_static_asset_load_s": times["planner_static_asset_load_s"],
        "policy_actor_total_s": times["policy_actor_total_s"],
        "policy_gru_cpu_enqueue_s": times["policy_gru_enqueue_s"],
        "policy_preprocess_and_head_cpu_enqueue_s": max(
            0.0,
            times["policy_actor_module_enqueue_s"] - times["policy_gru_enqueue_s"],
        ),
        "critic_cpu_enqueue_s": times["policy_critic_enqueue_s"],
        "buffer_add_s": times["buffer_add_s"],
        "gae_s": times["gae_s"],
        "sequence_generation_padding_s": times["sequence_padding_s"],
        "ppo_forward_s": times["evaluate_actions_s"],
        "ppo_backward_s": times["backward_s"],
        "ppo_optimizer_s": times["optimizer_step_s"],
        "checkpoint_io_s": checkpoint_io_s,
        "logger_io_s": times["logger_record_s"] + times["logger_dump_s"],
    }
    result = {
        "schema_version": 1,
        "label": label,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": config_name,
        "mode": mode,
        "worker_count": 0 if mode == "dummy" else vector_env.worker_count,
        "seed": seed,
        "n_envs": ppo_config.N_ENVS,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "transitions": transitions,
        "learn_s": learn_s,
        "rollout_s": rollout_s,
        "ppo_train_s": times["ppo_train_s"],
        "post_update_s": post_update_s,
        "checkpoint_io_s": checkpoint_io_s,
        "total_update_s": total_update_s,
        "rollout_transitions_per_s": transitions / rollout_s,
        "train_valid_transitions_per_s": transitions / times["ppo_train_s"],
        "total_transitions_per_s": transitions / total_update_s,
        "completed_episodes": episodes,
        "completed_episodes_per_rollout_s": episodes / rollout_s,
        "outcomes": callback.latest["completed_episodes"],
        "roles": callback.latest.get("env_role_transitions"),
        "event_counts": dict(sorted(event_counts.items())),
        "padding": {
            "valid": padding_valid,
            "computed": padding_computed,
            "ratio": padding_computed / padding_valid,
            "batches": batches,
        },
        "timings": dict(times),
        "detailed_breakdown": detailed_breakdown,
        "startup": {
            "total_s": startup_s,
            "sampler_s": sampler_startup_s,
            "vector_env_s": vector_startup_s,
            "model_s": model_startup_s,
        },
        "pipeline_profile": pipeline_profile,
        "reset_order_count": len(reset_order),
        "reset_order_sha256": _hash_json(reset_order),
        "sampler_visit_sha256": _hash_json(sampler.visit_counts),
        "rollout_buffer_mib": buffer_bytes / 2**20,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "estimated_copy_bytes": {
            "collection_observation_h2d": transitions * 361 * 4,
            "collection_action_d2h": transitions * 2 * 4,
            "collection_value_logp_d2h": transitions * 2 * 4,
            "collection_recurrent_states_d2h": (
                transitions * actor_hidden_size * stored_recurrent_state_count * 4
            ),
            "stored_recurrent_state_count": stored_recurrent_state_count,
            "training_padded_observation_h2d": padding_computed * 361 * 4,
        },
        "copy_timings": {
            "collection_observation_h2d_s": times["collection_observation_h2d_s"],
            "terminal_observation_h2d_s": times["terminal_observation_h2d_s"],
            "buffer_add_including_recurrent_d2h_s": times["buffer_add_s"],
        },
        "resources": monitor.summary(),
        "optimizer_step": optimizer_step,
        "actor_frozen_max_abs_delta": actor_delta["frozen_actor"]["max_abs_delta_from_bc"],
        "log_std_max_abs_delta": actor_delta["log_std_max_abs_delta_from_initial"],
        "checkpoint": {
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_hash,
            "strict_12_key_load": True,
        },
        "runtime": {
            "python": os.sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
    }
    vector_env.close()
    return result


if __name__ == "__main__":
    arguments = parse_arguments()
    profile = profile_one(
        arguments.config,
        arguments.seed,
        arguments.label,
        arguments.mode,
        arguments.worker_count,
    )
    json_start = time.perf_counter()
    write_json(arguments.output, profile)
    profile["json_io_s"] = time.perf_counter() - json_start
    write_json(arguments.output, profile)
    print("PROFILE_RESULT " + json.dumps(profile, sort_keys=True), flush=True)

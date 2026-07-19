#!/usr/bin/env python3
"""Boundary-synchronized warm-up/repeat1 full-update performance runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for path in (PROJECT_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_common import (
    CONFIG_NAME,
    OUTPUT_DIR,
    SEED,
    WORKER_COUNT,
    assert_locked_sources,
    backend_flags,
    provenance,
    sha256_file,
    write_json,
)
from batched_backends import install_dispatch
from ppo import config as ppo_config
from ppo_experiments.performance_optimization.profile_pipeline import ResourceMonitor
from train_ppo import PPOTrainingCallback, build_model, build_sampler, build_training_vector_env, save_actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("R0", "R1", "A", "B", "C", "AB", "AC"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    assert_locked_sources()
    if torch.cuda.is_initialized():
        raise RuntimeError("performance process initialized CUDA before creating simulator workers")
    bundle = json.loads((OUTPUT_DIR / "REFERENCE_BUNDLE.json").read_text())
    config = ppo_config.get_config(CONFIG_NAME)
    tf32_off = args.backend != "R0"
    with backend_flags(tf32_off) as flags:
        startup_start = time.perf_counter()
        sampler = build_sampler(config)
        vector_env = build_training_vector_env(
            sampler,
            config,
            SEED,
            worker_count=WORKER_COUNT,
        )
        vector_env.seed(SEED)
        model = build_model(vector_env, config, SEED)
        if args.backend not in ("R0", "R1"):
            install_dispatch(model.policy, args.backend)
        callback = PPOTrainingCallback()
        startup_s = time.perf_counter() - startup_start
        actor_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            "collection": [],
            "training": [],
        }
        actor_forward = model.policy._actor_forward

        def timed_actor_forward(obs, states, episode_starts, valid_by_timestep=None):
            stage = "collection" if valid_by_timestep is None else "training"
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = actor_forward(obs, states, episode_starts, valid_by_timestep)
            end.record()
            actor_events[stage].append((start, end))
            return result

        model.policy._actor_forward = timed_actor_forward
        stage_times: dict[str, float] = {}
        original_collect = model.collect_rollouts

        def collect_rollouts(*collect_args, **collect_kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original_collect(*collect_args, **collect_kwargs)
            torch.cuda.synchronize()
            stage_times["rollout_s"] = stage_times.get("rollout_s", 0.0) + time.perf_counter() - start
            return result

        model.collect_rollouts = collect_rollouts
        original_train = model.train

        def train():
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original_train()
            torch.cuda.synchronize()
            stage_times["ppo_train_s"] = stage_times.get("ppo_train_s", 0.0) + time.perf_counter() - start
            return result

        model.train = train
        vector_env.env_method("set_policy_update_index", 1)
        monitor = ResourceMonitor()
        monitor.start()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        total_start = time.perf_counter()
        try:
            with torch.autograd.set_multithreading_enabled(config.autograd_multithreading):
                model.learn(
                    total_timesteps=ppo_config.N_ENVS * config.n_steps,
                    callback=callback,
                    log_interval=None,
                    reset_num_timesteps=True,
                    progress_bar=False,
                )
            torch.cuda.synchronize()
            total_update_s = time.perf_counter() - total_start
            peak_allocated = torch.cuda.max_memory_allocated() / 2**20
            peak_reserved = torch.cuda.max_memory_reserved() / 2**20
            actor_cuda_ms = {
                stage: sum(float(start.elapsed_time(end)) for start, end in events)
                for stage, events in actor_events.items()
            }
            checkpoint = None
            if not args.warmup:
                checkpoint_path = OUTPUT_DIR / f"PERF_{args.backend}_actor_checkpoint.pth"
                checkpoint_hash = save_actor(model, checkpoint_path)
                checkpoint = {
                    "path": checkpoint_path.name,
                    "sha256": checkpoint_hash,
                    "file_sha256": sha256_file(checkpoint_path),
                    "bytes": checkpoint_path.stat().st_size,
                    "strict_12_key_load": True,
                }
        finally:
            monitor.stop()
            vector_env.close()
        transitions = ppo_config.N_ENVS * config.n_steps
        rollout_s = stage_times["rollout_s"]
        train_s = stage_times["ppo_train_s"]
        completed = sum(callback.latest["completed_episodes"].values())
        timing = {
            "startup_s": startup_s,
            "rollout_wall_s": rollout_s,
            "ppo_train_wall_s": train_s,
            "total_update_wall_s": total_update_s,
            "rollout_transitions_per_s": transitions / rollout_s,
            "train_valid_transitions_per_s": transitions / train_s,
            "total_transitions_per_s": transitions / total_update_s,
            "completed_episodes": completed,
            "completed_episodes_per_rollout_s": completed / rollout_s,
            "actor_forward_cuda_ms": actor_cuda_ms,
            "actor_forward_calls": {stage: len(events) for stage, events in actor_events.items()},
            "torch_peak_allocated_mib": peak_allocated,
            "torch_peak_reserved_mib": peak_reserved,
            "resources": monitor.summary(),
        }
        result = {
            "schema_version": 1,
            **provenance(f"{args.backend} full update performance", args.backend, flags, bundle["rollout_hash"]),
            "model_initial_hash": bundle["model_initial_hash"],
            "optimizer_initial_hash": bundle["optimizer_initial_hash"],
            "rng_initial_hash": bundle["initial_rng_hashes"],
            "minibatch_order_hash": bundle["minibatch_order_hash"],
            "backend": args.backend,
            "batch_or_microbatch": 16 if args.backend in ("A", "AB", "AC") else "all active" if args.backend in ("B",) else "packed" if args.backend in ("C",) else 1,
            "warmup": args.warmup,
            "numerical_metrics": {},
            "timing_metrics": timing,
            "outcomes": callback.latest["completed_episodes"],
            "roles": callback.latest.get("env_role_transitions"),
            "checkpoint_hash": None if checkpoint is None else checkpoint["sha256"],
            "checkpoint": checkpoint,
            "verdict": "WARMUP_COMPLETE" if args.warmup else "PERFORMANCE_REPEAT1_COMPLETE",
        }
        write_json(args.output, result)
    assert_locked_sources()
    print(json.dumps({"backend": args.backend, "warmup": args.warmup, **timing}, sort_keys=True))


if __name__ == "__main__":
    main()

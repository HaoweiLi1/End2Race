#!/usr/bin/env python3
"""Capture one immutable current-HEAD rollout and its exact training order."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
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
    canonical_hash,
    object_hash,
    provenance,
    rng_hashes,
    rng_state,
    sha256_file,
    state_dict_hash,
    tensor_hash,
    write_json,
)
from ppo import config as ppo_config
from train_ppo import PPOTrainingCallback, build_model, build_sampler, build_training_vector_env


class Captured(RuntimeError):
    pass


def cpu_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [cpu_clone(item) for item in value]
    return copy.deepcopy(value)


def main() -> None:
    assert_locked_sources()
    config = ppo_config.get_config(CONFIG_NAME)
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "frozen_rollout_current.npz"
    minibatch_path = output_dir / "frozen_minibatches_current.pt"
    state_path = output_dir / "initial_state_and_rng_current.pt"
    for path in (raw_path, minibatch_path, state_path):
        if path.exists():
            raise FileExistsError(path)

    with backend_flags(True) as flags:
        sampler = build_sampler(config)
        vector_env = build_training_vector_env(sampler, config, SEED, worker_count=WORKER_COUNT)
        vector_env.seed(SEED)
        model = build_model(vector_env, config, SEED)
        callback = PPOTrainingCallback()
        initial_model = cpu_clone(model.policy.state_dict())
        initial_optimizer = cpu_clone(model.policy.optimizer.state_dict())
        initial_rng = rng_state()
        training_rng: dict[str, Any] = {}
        raw_hashes: dict[str, str] = {}
        minibatches: list[dict[str, Any]] = []
        captured_batch_indices: list[np.ndarray] = []
        original_get_samples = model.rollout_buffer._get_samples

        def get_samples(batch_inds, env_change, env=None):
            captured_batch_indices.append(np.asarray(batch_inds, dtype=np.int64).copy())
            return original_get_samples(batch_inds, env_change, env)

        model.rollout_buffer._get_samples = get_samples

        def capture_train() -> None:
            nonlocal training_rng
            arrays = {}
            for name in (
                "observations",
                "actions",
                "rewards",
                "episode_starts",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
            ):
                array = np.asarray(getattr(model.rollout_buffer, name)).copy()
                arrays[name] = array
                raw_hashes[name] = tensor_hash(array)
            np.savez(raw_path, **arrays)
            training_rng = rng_state()
            for batch_index, samples in enumerate(model.rollout_buffer.get(config.batch_size)):
                valid_by_timestep = model.rollout_buffer.current_valid_by_timestep
                sequence_lengths = [
                    sum(int(valid_by_timestep[step][sequence]) for step in range(len(valid_by_timestep)))
                    for sequence in range(len(valid_by_timestep[0]))
                ]
                record = {
                    "batch_index": batch_index,
                    "batch_inds": torch.from_numpy(captured_batch_indices[-1].copy()),
                    "seq_start_indices": torch.from_numpy(model.rollout_buffer.seq_start_indices.copy()),
                    "sequence_lengths": torch.as_tensor(sequence_lengths, dtype=torch.int64),
                    "valid_by_timestep": torch.as_tensor(valid_by_timestep, dtype=torch.bool),
                    "observations": cpu_clone(samples.observations),
                    "actions": cpu_clone(samples.actions),
                    "old_values": cpu_clone(samples.old_values),
                    "old_log_prob": cpu_clone(samples.old_log_prob),
                    "advantages": cpu_clone(samples.advantages),
                    "returns": cpu_clone(samples.returns),
                    "episode_starts": cpu_clone(samples.episode_starts),
                    "mask": cpu_clone(samples.mask),
                    "pi_hidden": cpu_clone(samples.lstm_states.pi[0]),
                    "pi_cell": cpu_clone(samples.lstm_states.pi[1]),
                    "vf_hidden": cpu_clone(samples.lstm_states.vf[0]),
                    "vf_cell": cpu_clone(samples.lstm_states.vf[1]),
                }
                minibatches.append(record)
            raise Captured

        model.train = capture_train
        vector_env.env_method("set_policy_update_index", 1)
        started = time.perf_counter()
        try:
            with torch.autograd.set_multithreading_enabled(config.autograd_multithreading):
                model.learn(
                    total_timesteps=ppo_config.N_ENVS * config.n_steps,
                    callback=callback,
                    log_interval=None,
                    reset_num_timesteps=True,
                    progress_bar=False,
                )
        except Captured:
            pass
        finally:
            vector_env.close()
        capture_wall_s = time.perf_counter() - started

        torch.save(minibatches, minibatch_path)
        torch.save(
            {
                "model_state": initial_model,
                "optimizer_state": initial_optimizer,
                "initial_rng": initial_rng,
                "training_rng": training_rng,
            },
            state_path,
        )
        metadata = []
        for batch in minibatches:
            metadata.append(
                {
                    "batch_index": batch["batch_index"],
                    "batch_inds_sha256": tensor_hash(batch["batch_inds"]),
                    "seq_start_indices": batch["seq_start_indices"].tolist(),
                    "sequence_lengths": batch["sequence_lengths"].tolist(),
                    "valid": int(batch["mask"].sum().item()),
                    "padded": int(batch["mask"].numel()),
                    "n_seq": int(batch["pi_hidden"].shape[1]),
                    "max_length": int(batch["valid_by_timestep"].shape[0]),
                    "sample_hashes": {
                        name: tensor_hash(batch[name])
                        for name in (
                            "observations",
                            "actions",
                            "old_values",
                            "old_log_prob",
                            "advantages",
                            "returns",
                            "episode_starts",
                            "mask",
                            "pi_hidden",
                            "pi_cell",
                            "vf_hidden",
                            "vf_cell",
                        )
                    },
                }
            )
        rollout_hash = canonical_hash(raw_hashes)
        metadata_hash = canonical_hash(metadata)
        result = {
            "schema_version": 1,
            **provenance("R1 frozen rollout and fixed minibatch order", 1, flags, rollout_hash),
            "capture_wall_s": capture_wall_s,
            "transitions": ppo_config.N_ENVS * config.n_steps,
            "model_initial_hash": state_dict_hash(initial_model),
            "optimizer_initial_hash": object_hash(initial_optimizer),
            "initial_rng_hashes": rng_hashes(initial_rng),
            "training_rng_hashes": rng_hashes(training_rng),
            "rollout_tensor_hashes": raw_hashes,
            "rollout_hash": rollout_hash,
            "minibatch_order_hash": metadata_hash,
            "minibatches": metadata,
            "files": {
                "raw_rollout": {"path": raw_path.name, "sha256": sha256_file(raw_path)},
                "minibatches": {"path": minibatch_path.name, "sha256": sha256_file(minibatch_path)},
                "state_and_rng": {"path": state_path.name, "sha256": sha256_file(state_path)},
            },
            "dtype_assertions": {
                "model_float32": all(
                    not torch.is_floating_point(value) or value.dtype == torch.float32
                    for value in initial_model.values()
                ),
                "observation_float32": all(batch["observations"].dtype == torch.float32 for batch in minibatches),
                "hidden_float32": all(batch["pi_hidden"].dtype == torch.float32 for batch in minibatches),
                "actions_float32": all(batch["actions"].dtype == torch.float32 for batch in minibatches),
            },
            "checkpoint_hash": None,
            "numerical_metrics": {},
            "timing_metrics": {"capture_wall_s": capture_wall_s},
            "verdict": "CURRENT_HEAD_REFERENCE_BUNDLE_GENERATED",
        }
    assert_locked_sources()
    write_json(output_dir / "REFERENCE_BUNDLE.json", result)
    print(json.dumps({"rollout_hash": result["rollout_hash"], "minibatch_order_hash": metadata_hash}, sort_keys=True))


if __name__ == "__main__":
    main()

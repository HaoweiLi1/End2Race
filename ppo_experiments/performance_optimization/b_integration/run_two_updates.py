#!/usr/bin/env python3
"""Run two formal PPO updates and capture U1 per-minibatch numerics.

Runs against whichever project tree is given via --project-root, so the same
script drives both the detached reference worktree (batch-1 replay) and the
integrated Phase 5B tree.  Captures, for update 1 only: per-minibatch mask,
advantages, old/new log-probs, raw and latent actor means, post-clip
gradients; plus the update-1 parameter delta, rollout-buffer hashes for both
updates, and the final U2 strict 12-key actor checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--config", default="N1-H1F-p50")
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve()
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    import numpy as np
    import torch

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False

    from ppo import config as ppo_config
    import train_ppo

    config = ppo_config.get_config(arguments.config)
    sampler = train_ppo.build_sampler(config)
    train_ppo.set_random_seed(arguments.seed)
    vector_env = train_ppo.build_training_vector_env(sampler, config, arguments.seed)
    vector_env.seed(arguments.seed)
    model = train_ppo.build_model(vector_env, config, arguments.seed)
    callback = train_ppo.PPOTrainingCallback()
    policy = model.policy

    parameter_names = sorted(
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    )

    def parameter_vector() -> np.ndarray:
        state = dict(policy.named_parameters())
        return np.concatenate(
            [state[name].detach().cpu().numpy().reshape(-1) for name in parameter_names]
        )

    def gradient_vector() -> np.ndarray:
        state = dict(policy.named_parameters())
        return np.concatenate(
            [
                np.zeros(state[name].numel(), dtype=np.float32)
                if state[name].grad is None
                else state[name].grad.detach().cpu().numpy().reshape(-1)
                for name in parameter_names
            ]
        )

    capture: dict[str, list[np.ndarray]] = {
        "mask": [], "advantages": [], "old_log_prob": [], "log_prob": [],
        "raw_mean": [], "latent_mean": [], "gradient": [],
    }
    capture_enabled = {"on": False}
    rollout_hashes: list[str] = []

    original_get = model.rollout_buffer.get

    def traced_get(batch_size=None):
        for samples in original_get(batch_size):
            if capture_enabled["on"]:
                capture["mask"].append(samples.mask.detach().cpu().numpy())
                capture["advantages"].append(samples.advantages.detach().cpu().numpy())
                capture["old_log_prob"].append(samples.old_log_prob.detach().cpu().numpy())
            yield samples

    model.rollout_buffer.get = traced_get

    original_evaluate = policy.evaluate_actions

    def traced_evaluate(observation, actions, lstm_states, episode_starts):
        values, log_prob, entropy = original_evaluate(
            observation, actions, lstm_states, episode_starts
        )
        if capture_enabled["on"]:
            capture["log_prob"].append(log_prob.detach().cpu().numpy())
            capture["raw_mean"].append(policy.action_dist.raw_mean_actions.detach().cpu().numpy())
            capture["latent_mean"].append(policy.action_dist.latent_steer_mean.detach().cpu().numpy())
        return values, log_prob, entropy

    policy.evaluate_actions = traced_evaluate

    original_step = policy.optimizer.step

    def traced_step(*step_args, **step_kwargs):
        if capture_enabled["on"]:
            capture["gradient"].append(gradient_vector())
        return original_step(*step_args, **step_kwargs)

    policy.optimizer.step = traced_step

    original_train = model.train

    def traced_train():
        digest = hashlib.sha256()
        buffer = model.rollout_buffer
        for name in (
            "observations", "actions", "log_probs", "values", "advantages",
            "returns", "episode_starts", "hidden_states_pi",
        ):
            digest.update(np.ascontiguousarray(getattr(buffer, name)).tobytes())
        rollout_hashes.append(digest.hexdigest())
        return original_train()

    model.train = traced_train

    meta: dict[str, object] = {
        "label": arguments.label,
        "project_root": str(project_root),
        "seed": arguments.seed,
        "config": arguments.config,
        "tf32": [
            torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.benchmark,
        ],
        "updates": {},
    }
    parameters_before = parameter_vector()
    for update in (1, 2):
        vector_env.env_method("set_policy_update_index", update)
        capture_enabled["on"] = update == 1
        start = time.perf_counter()
        with torch.autograd.set_multithreading_enabled(config.autograd_multithreading):
            model.learn(
                total_timesteps=ppo_config.N_ENVS * config.n_steps,
                callback=callback,
                log_interval=None,
                reset_num_timesteps=update == 1,
                progress_bar=False,
            )
        capture_enabled["on"] = False
        record = {
            "wall_s": time.perf_counter() - start,
            "rollout_sha256": rollout_hashes[update - 1],
            "outcomes": dict(callback.latest["completed_episodes"]),
            "optimizer_step": train_ppo._optimizer_step(model, require_initialized=True),
        }
        for key in ("loss", "policy_gradient_loss", "value_loss", "approx_kl", "clip_fraction"):
            logger_key = f"train/{key}"
            if logger_key in model.logger.name_to_value:
                record[key] = float(model.logger.name_to_value[logger_key])
        meta["updates"][str(update)] = record
        if update == 1:
            parameters_after_u1 = parameter_vector()

    arguments.output.mkdir(parents=True, exist_ok=True)
    checkpoint = arguments.output / f"{arguments.label}_u2_actor.pth"
    meta["u2_checkpoint_sha256"] = train_ppo.save_actor(model, checkpoint)
    meta["u2_checkpoint"] = str(checkpoint)
    meta["parameter_names"] = parameter_names

    arrays: dict[str, np.ndarray] = {
        "parameters_before": parameters_before,
        "parameters_after_u1": parameters_after_u1,
    }
    for name, values in capture.items():
        for index, value in enumerate(values):
            arrays[f"{name}_{index}"] = value
    meta["minibatches"] = len(capture["gradient"])
    np.savez_compressed(arguments.output / f"{arguments.label}_capture.npz", **arrays)
    (arguments.output / f"{arguments.label}_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    vector_env.close()
    print(f"DONE {arguments.label} minibatches={meta['minibatches']}")


if __name__ == "__main__":
    main()

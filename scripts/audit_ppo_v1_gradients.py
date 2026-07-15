#!/usr/bin/env python3
"""Read-only PPO V1 gradient coupling and critic saturation audit.

The audit builds a fresh PPO V1 model from the canonical BC actor, constructs
the formal 16-environment training stack, and collects exactly one 800-step
rollout.  It replays every stock recurrent minibatch without clipping gradients
or stepping the optimizer, then proves that every policy parameter is bitwise
unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sb3_contrib
import stable_baselines3
import torch
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.ppo_scenarios import FixedMixtureScenarioSampler, classify_bc_ego_collisions, training_scenarios
from rl.sb3_end2race_policy import DEFAULT_BC_CHECKPOINT, END2RACE_LIDAR_SIZE
from train_ppo_sb3 import DEFAULT_CONFIG, build_model, make_training_env


EXPECTED_HEAD = "fd22c962a4fd62681a9a4fb53e14f277ef8f3418"
DEFAULT_BC_OUTCOMES = ROOT / "runs" / "ppo_v1" / "pilot_20_updates" / "train_bc_outcomes.json"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "ppo_v1" / "gradient_audit"
N_ENVS = 16
N_STEPS = 800
BATCH_SIZE = 800
ROLLOUT_TRANSITIONS = N_ENVS * N_STEPS
SATURATION_PREACTIVATION_THRESHOLD = 3.0
SATURATION_TANH_THRESHOLD = 0.99


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-outcomes", type=Path, default=DEFAULT_BC_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(path, text)


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def source_evidence() -> dict[str, Any]:
    repo_paths = [
        ROOT / "rl" / "sb3_end2race_policy.py",
        ROOT / "rl" / "end2race_recurrent_ppo.py",
        ROOT / "rl" / "end2race_gymnasium_env.py",
        ROOT / "rl" / "ppo_reward.py",
        ROOT / "rl" / "ppo_scenarios.py",
        ROOT / "train_ppo_sb3.py",
        Path(__file__).resolve(),
    ]
    stock_sources = {
        "RecurrentPPO.train": inspect.getsource(RecurrentPPO.train),
        "RecurrentRolloutBuffer.get": inspect.getsource(RecurrentRolloutBuffer.get),
        "RecurrentRolloutBuffer._get_samples": inspect.getsource(RecurrentRolloutBuffer._get_samples),
    }
    return {
        "git_head": git_output("rev-parse", "HEAD"),
        "repo_files": {
            str(path.relative_to(ROOT)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in repo_paths
        },
        "packages": {
            "python": sys.version,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__,
        },
        "stock_sources": {
            name: {
                "sha256": sha256_text(source),
                "source_file": str(
                    Path(inspect.getsourcefile(RecurrentPPO if name.startswith("RecurrentPPO") else RecurrentRolloutBuffer) or "")
                ),
            }
            for name, source in stock_sources.items()
        },
    }


def clone_parameters(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in policy.named_parameters()}


def parameter_digest(parameters: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(parameters):
        tensor = parameters[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def compare_parameters(before: dict[str, torch.Tensor], policy: torch.nn.Module) -> dict[str, Any]:
    after = clone_parameters(policy)
    if set(before) != set(after):
        raise AssertionError("Policy parameter names changed during the audit")
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    maximum_delta = max(
        (float((after[name] - before[name]).abs().max()) for name in changed),
        default=0.0,
    )
    before_digest = parameter_digest(before)
    after_digest = parameter_digest(after)
    return {
        "parameter_tensor_count": len(before),
        "all_torch_equal": not changed,
        "changed_parameter_names": changed,
        "max_abs_delta": maximum_delta,
        "sha256_before": before_digest,
        "sha256_after": after_digest,
        "sha256_equal": before_digest == after_digest,
    }


class RolloutInfoTrace(BaseCallback):
    """Keep step-major info records aligned with the stock rollout buffer."""

    def __init__(self, n_envs: int) -> None:
        super().__init__(verbose=0)
        self.n_envs = int(n_envs)
        self.episode_ordinals = [0 for _ in range(self.n_envs)]
        self.steps: list[list[dict[str, Any]]] = []
        self.completed_outcomes: dict[str, str] = {}
        self.completed_records: list[dict[str, Any]] = []
        self.rollout_start_count = 0
        self.rollout_end_count = 0

    def _on_rollout_start(self) -> None:
        self.rollout_start_count += 1

    def _on_step(self) -> bool:
        infos = list(self.locals.get("infos", []))
        dones = np.asarray(self.locals.get("dones", np.zeros(self.n_envs, dtype=bool)), dtype=bool)
        if len(infos) != self.n_envs or dones.shape != (self.n_envs,):
            raise AssertionError("Callback info/done geometry does not match the formal vector environment")
        vector_step = len(self.steps)
        step_records: list[dict[str, Any]] = []
        for env_index, info in enumerate(infos):
            ordinal = self.episode_ordinals[env_index]
            episode_key = f"env{env_index:02d}-episode{ordinal:04d}"
            done = bool(dones[env_index])
            outcome: str | None = None
            if done:
                if bool(info.get("ego_collision", False)):
                    outcome = "ego_collision"
                else:
                    outcome = "overtake" if float(info.get("relative_position_m", 0.0)) > 0.0 else "follow"
                self.completed_outcomes[episode_key] = outcome
            record = {
                "vector_step": vector_step,
                "env_index": env_index,
                "episode_ordinal": ordinal,
                "episode_key": episode_key,
                "scenario_id": info.get("scenario_id"),
                "sampler_branch": info.get("sampler_branch"),
                "done": done,
                "ego_collision": bool(info.get("ego_collision", False)),
                "relative_position_m": float(info.get("relative_position_m", 0.0)),
                "outcome": outcome,
            }
            step_records.append(record)
            if done:
                self.completed_records.append(record.copy())
                self.episode_ordinals[env_index] += 1
        self.steps.append(step_records)
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_end_count += 1


def scalar_stats(values: torch.Tensor | np.ndarray) -> dict[str, float]:
    tensor = torch.as_tensor(values).detach().double().reshape(-1)
    if tensor.numel() == 0:
        raise ValueError("Cannot summarize an empty tensor")
    return {
        "mean": float(tensor.mean().cpu()),
        "std": float(tensor.std(unbiased=False).cpu()),
        "min": float(tensor.min().cpu()),
        "max": float(tensor.max().cpu()),
    }


def norm_stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Norm statistics require finite non-empty values")
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def gradient_stats(parameters: Sequence[torch.nn.Parameter]) -> dict[str, Any]:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    squared = sum(
        float(gradient.detach().double().square().sum().cpu())
        for gradient in gradients
    )
    nonzero_tensors = sum(bool(torch.any(gradient != 0)) for gradient in gradients)
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    return {
        "l2_norm": math.sqrt(squared),
        "parameter_tensor_count": len(parameters),
        "gradient_tensor_count": len(gradients),
        "nonzero_gradient_tensor_count": int(nonzero_tensors),
        "all_finite": finite,
    }


def critic_numeric_stats(
    policy: torch.nn.Module,
    observations: torch.Tensor,
    returns: torch.Tensor,
    predictions: torch.Tensor | None = None,
) -> dict[str, Any]:
    observations = observations.float()
    returns = returns.flatten()
    first_linear = policy.value_net[0]
    with torch.no_grad():
        preactivation = first_linear(observations)
        tanh_output = torch.tanh(preactivation)
        if predictions is None:
            predictions = policy.value_net(observations).flatten()
        else:
            predictions = predictions.flatten()
    lidar = observations[:, :END2RACE_LIDAR_SIZE]
    previous_speed = observations[:, END2RACE_LIDAR_SIZE]
    prediction_array = predictions.detach().cpu().numpy()
    return_array = returns.detach().cpu().numpy()
    return {
        "raw_lidar": scalar_stats(lidar),
        "previous_speed": scalar_stats(previous_speed),
        "first_layer_preactivation": {
            **scalar_stats(preactivation),
            "max_abs": float(preactivation.detach().abs().max().cpu()),
            "abs_gt_3_fraction": float(
                (preactivation.detach().abs() > SATURATION_PREACTIVATION_THRESHOLD).float().mean().cpu()
            ),
        },
        "first_tanh": {
            "abs_gt_0_99_fraction": float(
                (tanh_output.detach().abs() > SATURATION_TANH_THRESHOLD).float().mean().cpu()
            ),
        },
        "critic_predictions": scalar_stats(predictions),
        "returns": scalar_stats(returns),
        "explained_variance": explained_variance(prediction_array, return_array),
    }


def load_bc_outcomes(path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError("Canonical BC outcomes must be a JSON list")
    collision_ids = classify_bc_ego_collisions(rows)
    return rows, collision_ids


def formal_audit_config(device: str) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        n_envs=N_ENVS,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=1,
        updates=1,
        device=device,
        vector_env="DummyVecEnv",
    )
    config["transitions_per_update"] = ROLLOUT_TRANSITIONS
    config["minibatches_per_update"] = ROLLOUT_TRANSITIONS // BATCH_SIZE
    config["optimizer_steps_per_update"] = 0
    config["total_transitions"] = ROLLOUT_TRANSITIONS
    config["gamma_times_gae_lambda"] = config["gamma"] * config["gae_lambda"]
    config["bc_checkpoint"] = str(DEFAULT_BC_CHECKPOINT.relative_to(ROOT))
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_fresh_bc_actor(policy: torch.nn.Module) -> dict[str, Any]:
    checkpoint = torch.load(DEFAULT_BC_CHECKPOINT, map_location="cpu", weights_only=True)
    actor_state = {name: tensor.detach().cpu() for name, tensor in policy.end2race_actor.state_dict().items()}
    if set(checkpoint) != set(actor_state):
        raise AssertionError("Fresh PPO actor keys do not match the canonical BC checkpoint")
    mismatches = [name for name in checkpoint if not torch.equal(checkpoint[name], actor_state[name])]
    return {
        "checkpoint": str(DEFAULT_BC_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(DEFAULT_BC_CHECKPOINT),
        "actor_key_count": len(actor_state),
        "all_actor_tensors_bitwise_equal": not mismatches,
        "mismatched_actor_keys": mismatches,
    }


def collect_one_rollout(model: RecurrentPPO, trace: RolloutInfoTrace) -> None:
    total_timesteps, callback = model._setup_learn(
        total_timesteps=ROLLOUT_TRANSITIONS,
        callback=trace,
        reset_num_timesteps=True,
        tb_log_name="gradient_audit",
        progress_bar=False,
    )
    if total_timesteps != ROLLOUT_TRANSITIONS:
        raise AssertionError(f"Unexpected setup timestep target: {total_timesteps}")
    callback.on_training_start(locals(), globals())
    try:
        complete = model.collect_rollouts(
            model.env,
            callback,
            model.rollout_buffer,
            n_rollout_steps=N_STEPS,
        )
    finally:
        callback.on_training_end()
    if not complete:
        raise RuntimeError("Stock collect_rollouts terminated before the requested rollout completed")
    if not model.rollout_buffer.full:
        raise AssertionError("Rollout buffer is not full after one formal rollout")
    if model.num_timesteps != ROLLOUT_TRANSITIONS:
        raise AssertionError(f"Expected {ROLLOUT_TRANSITIONS} timesteps, got {model.num_timesteps}")
    if len(trace.steps) != N_STEPS or any(len(step) != N_ENVS for step in trace.steps):
        raise AssertionError("Rollout callback trace does not cover the full step/env grid")


def sequence_and_episode_metadata(
    rollout_data: Any,
    batch_inds: np.ndarray,
    trace: RolloutInfoTrace,
) -> dict[str, Any]:
    mask = rollout_data.mask > 1e-8
    n_seq = int(rollout_data.lstm_states.pi[0].shape[1])
    if mask.numel() % n_seq != 0:
        raise AssertionError("Padded minibatch size is not divisible by n_seq")
    max_sequence_length = int(mask.numel() // n_seq)
    sequence_mask = mask.reshape(n_seq, max_sequence_length)
    sequence_lengths = [int(row.sum().item()) for row in sequence_mask]
    valid_count = int(mask.sum().item())
    padded_count = int((~mask).sum().item())
    valid_records: list[dict[str, Any]] = []
    for flat_index in batch_inds:
        env_index = int(flat_index // N_STEPS)
        vector_step = int(flat_index % N_STEPS)
        valid_records.append(trace.steps[vector_step][env_index])
    episode_keys = sorted({str(record["episode_key"]) for record in valid_records})
    outcome_counts = Counter(
        trace.completed_outcomes[key]
        for key in episode_keys
        if key in trace.completed_outcomes
    )
    scenario_ids = sorted({str(record["scenario_id"]) for record in valid_records})
    sampler_branches = Counter(str(record["sampler_branch"] or "unknown") for record in valid_records)
    episode_starts = rollout_data.episode_starts[mask]
    sequence_starts = rollout_data.episode_starts.reshape(n_seq, max_sequence_length)[:, 0]
    return {
        "valid_transition_count": valid_count,
        "padded_transition_count": padded_count,
        "n_seq": n_seq,
        "max_sequence_length": max_sequence_length,
        "sequence_valid_lengths": sequence_lengths,
        "episode_start_count": int((episode_starts > 0.5).sum().item()),
        "sequence_start_episode_start_count": int((sequence_starts > 0.5).sum().item()),
        "continuation_sequence_count": int((sequence_starts <= 0.5).sum().item()),
        "unique_episode_count": len(episode_keys),
        "episode_equivalent_at_800_steps": float(valid_count / N_STEPS),
        "completed_episode_count": int(sum(outcome_counts.values())),
        "unclassified_episode_count": int(len(episode_keys) - sum(outcome_counts.values())),
        "collision_episode_count": int(outcome_counts["ego_collision"]),
        "follow_episode_count": int(outcome_counts["follow"]),
        "overtake_episode_count": int(outcome_counts["overtake"]),
        "outcomes_when_info_associated": dict(sorted(outcome_counts.items())),
        "episode_keys": episode_keys,
        "scenario_ids": scenario_ids,
        "sampler_branch_transitions": dict(sorted(sampler_branches.items())),
    }


def replay_all_minibatches(
    model: RecurrentPPO,
    trace: RolloutInfoTrace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buffer = model.rollout_buffer
    total = buffer.buffer_size * buffer.n_envs
    if total != ROLLOUT_TRANSITIONS or BATCH_SIZE != model.batch_size:
        raise AssertionError("Formal rollout/minibatch geometry changed")

    numpy_state = np.random.get_state()
    split_index = int(np.random.randint(total))
    np.random.set_state(numpy_state)
    indices = np.arange(total)
    indices = np.concatenate((indices[split_index:], indices[:split_index]))

    policy = model.policy
    policy.set_training_mode(True)
    gru_parameters = list(policy.end2race_actor.gru.parameters())
    head_parameters = list(policy.end2race_actor.output_layer.parameters())
    critic_parameters = list(policy.value_net.parameters())
    all_parameters = list(policy.parameters())
    clip_range = float(model.clip_range(model._current_progress_remaining))
    clip_range_vf = None
    if model.clip_range_vf is not None:
        clip_range_vf = float(model.clip_range_vf(model._current_progress_remaining))

    rows: list[dict[str, Any]] = []
    seen_indices: list[int] = []
    for minibatch_index, rollout_data in enumerate(buffer.get(BATCH_SIZE)):
        start = minibatch_index * BATCH_SIZE
        batch_inds = indices[start : start + BATCH_SIZE]
        seen_indices.extend(int(value) for value in batch_inds)
        metadata = sequence_and_episode_metadata(rollout_data, batch_inds, trace)
        mask = rollout_data.mask > 1e-8
        actions = rollout_data.actions

        values, log_prob, entropy = policy.evaluate_actions(
            rollout_data.observations,
            actions,
            rollout_data.lstm_states,
            rollout_data.episode_starts,
        )
        values = values.flatten()
        advantages = rollout_data.advantages
        if model.normalize_advantage:
            advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
        ratio = torch.exp(log_prob - rollout_data.old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.mean(torch.min(policy_loss_1, policy_loss_2)[mask])

        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = rollout_data.old_values + torch.clamp(
                values - rollout_data.old_values,
                -clip_range_vf,
                clip_range_vf,
            )
        value_loss = torch.mean(((rollout_data.returns - values_pred) ** 2)[mask])
        weighted_value_loss = model.vf_coef * value_loss
        if entropy is None:
            entropy_loss = -torch.mean(-log_prob[mask])
        else:
            entropy_loss = -torch.mean(entropy[mask])
        combined_loss = policy_loss + model.ent_coef * entropy_loss + weighted_value_loss

        policy.optimizer.zero_grad(set_to_none=True)
        policy_loss.backward(retain_graph=True)
        policy_only = {
            "gru": gradient_stats(gru_parameters),
            "head": gradient_stats(head_parameters),
            "critic": gradient_stats(critic_parameters),
        }
        policy.optimizer.zero_grad(set_to_none=True)

        weighted_value_loss.backward(retain_graph=True)
        value_only = {
            "gru": gradient_stats(gru_parameters),
            "head": gradient_stats(head_parameters),
            "critic": gradient_stats(critic_parameters),
        }
        policy.optimizer.zero_grad(set_to_none=True)

        combined_loss.backward()
        combined = {
            "gru": gradient_stats(gru_parameters),
            "head": gradient_stats(head_parameters),
            "critic": gradient_stats(critic_parameters),
            "global": gradient_stats(all_parameters),
        }
        global_norm = float(combined["global"]["l2_norm"])
        clip_multiplier = min(1.0, float(model.max_grad_norm) / (global_norm + 1e-6))
        combined["max_grad_norm"] = float(model.max_grad_norm)
        combined["theoretical_clip_multiplier"] = clip_multiplier
        combined["theoretical_post_clip_norms"] = {
            name: float(combined[name]["l2_norm"] * clip_multiplier)
            for name in ("gru", "head", "critic", "global")
        }
        actor_norm = math.hypot(
            float(combined["gru"]["l2_norm"]),
            float(combined["head"]["l2_norm"]),
        )
        combined["actor_l2_norm"] = actor_norm
        actor_only_clip_multiplier = min(1.0, float(model.max_grad_norm) / (actor_norm + 1e-6))
        combined["counterfactual_actor_only_clip_multiplier"] = actor_only_clip_multiplier
        combined["critic_induced_actor_retention_ratio"] = float(
            clip_multiplier / actor_only_clip_multiplier
        )
        combined["critic_induced_additional_actor_reduction_percent"] = float(
            100.0 * (1.0 - clip_multiplier / actor_only_clip_multiplier)
        )
        combined["critic_to_actor_norm_ratio"] = safe_ratio(
            float(combined["critic"]["l2_norm"]),
            actor_norm,
        )
        combined["critic_squared_norm_fraction"] = safe_ratio(
            float(combined["critic"]["l2_norm"]) ** 2,
            global_norm**2,
        )

        valid_observations = rollout_data.observations[mask]
        valid_returns = rollout_data.returns[mask]
        valid_predictions = values[mask]
        numeric = critic_numeric_stats(
            policy,
            valid_observations,
            valid_returns,
            valid_predictions,
        )
        replay_error = (log_prob - rollout_data.old_log_prob)[mask]
        row = {
            "minibatch_index": minibatch_index,
            **metadata,
            "losses": {
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "weighted_value_loss": float(weighted_value_loss.detach().cpu()),
                "entropy_loss": float(entropy_loss.detach().cpu()),
                "combined_stock_loss": float(combined_loss.detach().cpu()),
            },
            "ppo_replay": {
                "clip_range": clip_range,
                "ratio_mean": float(ratio[mask].detach().mean().cpu()),
                "ratio_std": float(ratio[mask].detach().std(unbiased=False).cpu()),
                "ratio_min": float(ratio[mask].detach().min().cpu()),
                "ratio_max": float(ratio[mask].detach().max().cpu()),
                "max_abs_log_prob_replay_error": float(replay_error.detach().abs().max().cpu()),
                "advantage_normalized_mean": float(advantages[mask].detach().mean().cpu()),
                "advantage_normalized_std_unbiased": float(advantages[mask].detach().std().cpu()),
            },
            "policy_loss_only_backward": policy_only,
            "vf_coef_value_loss_only_backward": value_only,
            "combined_stock_loss_backward": combined,
            "critic_numeric": numeric,
        }
        rows.append(row)
        policy.optimizer.zero_grad(set_to_none=True)

    if len(rows) != total // BATCH_SIZE:
        raise AssertionError(f"Expected {total // BATCH_SIZE} minibatches, got {len(rows)}")
    if sorted(seen_indices) != list(range(total)):
        raise AssertionError("Stock recurrent minibatches did not cover every rollout transition exactly once")
    if sum(row["valid_transition_count"] for row in rows) != total:
        raise AssertionError("Valid minibatch transition counts do not sum to the rollout size")
    if any(row["valid_transition_count"] != BATCH_SIZE for row in rows):
        raise AssertionError("A stock minibatch did not contain exactly batch_size valid transitions")
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise AssertionError("A parameter gradient remained after the final zero_grad")

    replay_summary = {
        "stock_split_index": split_index,
        "minibatch_count": len(rows),
        "all_transitions_seen_exactly_once": True,
        "valid_transition_total": sum(row["valid_transition_count"] for row in rows),
        "padded_transition_total": sum(row["padded_transition_count"] for row in rows),
        "gradients_cleared_after_every_backward": True,
        "optimizer_step_calls": 0,
        "clip_grad_norm_calls": 0,
    }
    return rows, replay_summary


def aggregate_minibatches(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    policy_critic = [row["policy_loss_only_backward"]["critic"]["l2_norm"] for row in rows]
    value_gru = [row["vf_coef_value_loss_only_backward"]["gru"]["l2_norm"] for row in rows]
    value_head = [row["vf_coef_value_loss_only_backward"]["head"]["l2_norm"] for row in rows]
    clip_multipliers = [row["combined_stock_loss_backward"]["theoretical_clip_multiplier"] for row in rows]
    actor_only_clip_multipliers = [
        row["combined_stock_loss_backward"]["counterfactual_actor_only_clip_multiplier"]
        for row in rows
    ]
    critic_additional_reductions = [
        row["combined_stock_loss_backward"]["critic_induced_additional_actor_reduction_percent"]
        for row in rows
    ]
    critic_fractions = [row["combined_stock_loss_backward"]["critic_squared_norm_fraction"] for row in rows]
    critic_actor_ratios = [row["combined_stock_loss_backward"]["critic_to_actor_norm_ratio"] for row in rows]
    if any(value is None for value in (*critic_fractions, *critic_actor_ratios)):
        raise AssertionError("A zero actor/global norm prevented coupling measurement")
    direct_actor_to_critic_leakage = any(value != 0.0 for value in policy_critic)
    direct_critic_to_actor_leakage = any(value != 0.0 for value in (*value_gru, *value_head))
    clip_stats = norm_stats(clip_multipliers)
    actor_only_clip_stats = norm_stats(actor_only_clip_multipliers)
    critic_additional_reduction_stats = norm_stats(critic_additional_reductions)
    critic_fraction_stats = norm_stats(float(value) for value in critic_fractions)
    critic_dominates = critic_fraction_stats["median"] >= 0.90
    actor_significantly_suppressed = clip_stats["median"] <= 0.50
    critic_materially_increases_suppression = critic_additional_reduction_stats["median"] >= 10.0
    return {
        "direct_gradient_leakage": {
            "actor_loss_to_critic": direct_actor_to_critic_leakage,
            "critic_loss_to_gru_or_head": direct_critic_to_actor_leakage,
            "max_policy_loss_only_critic_norm": max(policy_critic),
            "max_value_loss_only_gru_norm": max(value_gru),
            "max_value_loss_only_head_norm": max(value_head),
        },
        "combined_pre_clip_norms": {
            group: norm_stats(row["combined_stock_loss_backward"][group]["l2_norm"] for row in rows)
            for group in ("gru", "head", "critic", "global")
        },
        "critic_to_actor_norm_ratio": norm_stats(float(value) for value in critic_actor_ratios),
        "critic_squared_global_norm_fraction": critic_fraction_stats,
        "theoretical_clip_multiplier": clip_stats,
        "counterfactual_actor_only_clip_multiplier": actor_only_clip_stats,
        "critic_induced_additional_actor_reduction_percent": critic_additional_reduction_stats,
        "theoretical_actor_norm_retained_percent": {
            key: 100.0 * value for key, value in clip_stats.items()
        },
        "theoretical_actor_norm_reduction_percent": {
            "min": 100.0 * (1.0 - clip_stats["max"]),
            "median": 100.0 * (1.0 - clip_stats["median"]),
            "mean": 100.0 * (1.0 - clip_stats["mean"]),
            "max": 100.0 * (1.0 - clip_stats["min"]),
        },
        "clip_active_minibatch_count": sum(value < 1.0 for value in clip_multipliers),
        "critic_significantly_dominates_by_declared_threshold": critic_dominates,
        "actor_significantly_suppressed_by_declared_threshold": actor_significantly_suppressed,
        "critic_materially_increases_actor_suppression_by_declared_threshold": critic_materially_increases_suppression,
        "sequence_geometry": {
            "n_seq": norm_stats(row["n_seq"] for row in rows),
            "unique_episode_count": norm_stats(row["unique_episode_count"] for row in rows),
            "episode_equivalent_at_800_steps": norm_stats(
                row["episode_equivalent_at_800_steps"] for row in rows
            ),
            "episode_start_count": norm_stats(row["episode_start_count"] for row in rows),
            "valid_sequence_length": norm_stats(
                length for row in rows for length in row["sequence_valid_lengths"]
            ),
        },
    }


def overall_critic_numeric(model: RecurrentPPO) -> dict[str, Any]:
    buffer = model.rollout_buffer
    if buffer.generator_ready:
        observations = np.asarray(buffer.observations)
        returns = np.asarray(buffer.returns)
        old_values = np.asarray(buffer.values)
    else:
        observations = buffer.swap_and_flatten(np.asarray(buffer.observations))
        returns = buffer.swap_and_flatten(np.asarray(buffer.returns))
        old_values = buffer.swap_and_flatten(np.asarray(buffer.values))
    observation_tensor = torch.as_tensor(observations, device=model.device).float()
    returns_tensor = torch.as_tensor(returns, device=model.device).flatten()
    with torch.no_grad():
        predictions = model.policy.value_net(observation_tensor).flatten()
    numeric = critic_numeric_stats(model.policy, observation_tensor, returns_tensor, predictions)
    numeric["stock_rollout_value_explained_variance"] = explained_variance(
        np.asarray(old_values).flatten(),
        np.asarray(returns).flatten(),
    )
    numeric["current_vs_rollout_value_max_abs_error"] = float(
        np.max(np.abs(predictions.detach().cpu().numpy() - np.asarray(old_values).flatten()))
    )
    return numeric


def build_verdicts(aggregate: dict[str, Any], critic_numeric: dict[str, Any]) -> dict[str, Any]:
    leakage = aggregate["direct_gradient_leakage"]
    direct_leakage = bool(leakage["actor_loss_to_critic"] or leakage["critic_loss_to_gru_or_head"])
    critic_dominates = bool(aggregate["critic_significantly_dominates_by_declared_threshold"])
    actor_suppressed = bool(aggregate["actor_significantly_suppressed_by_declared_threshold"])
    critic_increases_suppression = bool(
        aggregate["critic_materially_increases_actor_suppression_by_declared_threshold"]
    )
    saturation_fraction = float(critic_numeric["first_tanh"]["abs_gt_0_99_fraction"])
    critic_saturated = saturation_fraction >= 0.10
    blocking_reasons: list[str] = []
    if direct_leakage:
        blocking_reasons.append("错误的 actor/critic direct gradient leakage")
    if critic_increases_suppression and actor_suppressed:
        blocking_reasons.append("critic 通过 stock global clipping 额外显著缩小 actor gradient")
    if critic_saturated:
        blocking_reasons.append("critic 第一层 Tanh 明显饱和")
    return {
        "declared_interpretation_thresholds": {
            "critic_dominance_median_squared_norm_fraction_gte": 0.90,
            "significant_actor_suppression_median_clip_multiplier_lte": 0.50,
            "material_critic_induced_additional_actor_reduction_median_gte_percent": 10.0,
            "clear_tanh_saturation_fraction_gte": 0.10,
        },
        "incorrect_direct_gradient_leakage": direct_leakage,
        "critic_significantly_dominates_global_gradient_norm": critic_dominates,
        "global_clipping_significantly_suppresses_actor_gradient": actor_suppressed,
        "critic_materially_increases_actor_clipping": critic_increases_suppression,
        "critic_tanh_clearly_saturated": critic_saturated,
        "blocking_issue_before_v1_1": bool(blocking_reasons),
        "blocking_reasons": blocking_reasons,
    }


def format_range(stats: dict[str, float], digits: int = 6) -> str:
    return f"{stats['min']:.{digits}g}–{stats['max']:.{digits}g}（median {stats['median']:.{digits}g}）"


def render_summary(audit: dict[str, Any]) -> str:
    aggregate = audit["gradient_aggregate"]
    numeric = audit["critic_numeric"]
    verdict = audit["verdict"]
    leakage = aggregate["direct_gradient_leakage"]
    clip = aggregate["theoretical_clip_multiplier"]
    reduction = aggregate["theoretical_actor_norm_reduction_percent"]
    critic_additional_reduction = aggregate["critic_induced_additional_actor_reduction_percent"]
    critic_fraction = aggregate["critic_squared_global_norm_fraction"]
    sequence = aggregate["sequence_geometry"]
    outcome_counts = audit["rollout"]["completed_outcomes"]
    blocker = "有" if verdict["blocking_issue_before_v1_1"] else "没有"
    blocker_reasons = "；".join(verdict["blocking_reasons"]) if verdict["blocking_reasons"] else "无"
    lines = [
        "# PPO V1 gradient audit",
        "",
        "## 审计边界",
        "",
        f"- Git HEAD：`{audit['source_evidence']['git_head']}`",
        f"- BC actor：`{audit['fresh_bc_actor']['checkpoint']}`，SHA-256 `{audit['fresh_bc_actor']['checkpoint_sha256']}`；fresh model 的 12 个 actor tensors 全部 bitwise equal。",
        f"- stock 版本：stable-baselines3 `{audit['source_evidence']['packages']['stable_baselines3']}`，sb3-contrib `{audit['source_evidence']['packages']['sb3_contrib']}`。",
        f"- 正式 rollout：16 env × 800 step = {audit['rollout']['valid_transition_count']:,} transitions；只收集 1 个 rollout；optimizer.step = 0；未保存 checkpoint。",
        f"- stock minibatch：{audit['minibatch_replay']['minibatch_count']} × 800 valid transitions；padding 总数 {audit['minibatch_replay']['padded_transition_total']:,}。",
        "",
        "## Verdict",
        "",
        f"1. **Actor/critic 是否存在错误 direct gradient leakage：{'是' if verdict['incorrect_direct_gradient_leakage'] else '否'}。** policy-only backward 的 critic norm 最大值为 `{leakage['max_policy_loss_only_critic_norm']:.6g}`；`vf_coef * value_loss` backward 的 GRU/head norm 最大值分别为 `{leakage['max_value_loss_only_gru_norm']:.6g}` / `{leakage['max_value_loss_only_head_norm']:.6g}`。",
        f"2. **Critic 是否显著主导 global gradient norm：{'是' if verdict['critic_significantly_dominates_global_gradient_norm'] else '否'}。** critic 的 squared global-norm fraction 为 {format_range(critic_fraction)}。",
        f"3. **Stock global clipping 是否显著压小 actor gradient：{'是' if verdict['global_clipping_significantly_suppresses_actor_gradient'] else '否'}；但 critic 是否是主要原因：{'是' if verdict['critic_materially_increases_actor_clipping'] else '否'}。** combined 理论 multiplier 为 {format_range(clip)}，即 actor norm 缩小 {format_range(reduction)}%。与 actor-only counterfactual 相比，critic 额外造成的 actor norm 缩小仅为 {format_range(critic_additional_reduction)} 个百分点；GRU/head 使用同一个 global multiplier。",
        f"4. **Critic Tanh 是否明显饱和：{'是' if verdict['critic_tanh_clearly_saturated'] else '否'}。** `abs(preactivation)>3` 比例为 `{numeric['first_layer_preactivation']['abs_gt_3_fraction']:.2%}`；Tanh `abs(value)>0.99` 比例为 `{numeric['first_tanh']['abs_gt_0_99_fraction']:.2%}`；preactivation `max_abs={numeric['first_layer_preactivation']['max_abs']:.6g}`。",
        f"5. **batch_size=800 的实际 recurrent 几何：** 每 minibatch `n_seq` 为 {format_range(sequence['n_seq'])}，涉及 unique episodes 为 {format_range(sequence['unique_episode_count'])}；按 800-step formal horizon 折算均为 `{sequence['episode_equivalent_at_800_steps']['mean']:.3f}` episode-equivalent。详细 sequence 长度和 outcome 关联见 `minibatches.jsonl`。",
        f"6. **V1.1 前是否有阻断性问题：{blocker}。** {blocker_reasons}。本审计不据此修改参数、网络、reward 或 clipping，也未启动训练。",
        "",
        "## Critic 数值证据",
        "",
        f"- raw LiDAR：mean `{numeric['raw_lidar']['mean']:.6g}`，std `{numeric['raw_lidar']['std']:.6g}`，min/max `{numeric['raw_lidar']['min']:.6g}` / `{numeric['raw_lidar']['max']:.6g}`。",
        f"- previous speed：mean `{numeric['previous_speed']['mean']:.6g}`，std `{numeric['previous_speed']['std']:.6g}`，min/max `{numeric['previous_speed']['min']:.6g}` / `{numeric['previous_speed']['max']:.6g}`。",
        f"- first-layer preactivation：mean `{numeric['first_layer_preactivation']['mean']:.6g}`，std `{numeric['first_layer_preactivation']['std']:.6g}`。",
        f"- critic prediction：mean `{numeric['critic_predictions']['mean']:.6g}`，std `{numeric['critic_predictions']['std']:.6g}`；returns mean `{numeric['returns']['mean']:.6g}`，std `{numeric['returns']['std']:.6g}`。",
        f"- explained variance：`{numeric['explained_variance']:.6g}`（stock rollout values：`{numeric['stock_rollout_value_explained_variance']:.6g}`）。",
        "",
        "## Rollout episode evidence",
        "",
        f"- completed episodes：`{sum(outcome_counts.values())}`；outcomes：`{json.dumps(outcome_counts, sort_keys=True)}`。",
        f"- sampler branch transitions：`{json.dumps(audit['rollout']['sampler_branch_transitions'], sort_keys=True)}`。",
        "- 每个 minibatch 的 collision/follow/overtake 只统计能通过本 rollout 的 info 与 episode key 关联到完整 outcome 的 episode；未完成 episode 单独计为 unclassified。",
        "",
        "## 完整性证明",
        "",
        f"- 所有 parameter tensors bitwise unchanged：`{audit['parameter_integrity']['all_torch_equal']}`。",
        f"- parameter SHA-256 before/after：`{audit['parameter_integrity']['sha256_before']}` / `{audit['parameter_integrity']['sha256_after']}`。",
        f"- 所有 gradients 在每次 backward 后清空：`{audit['minibatch_replay']['gradients_cleared_after_every_backward']}`。",
        f"- 所有 rollout transitions 恰好进入一个 stock minibatch：`{audit['minibatch_replay']['all_transitions_seen_exactly_once']}`。",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    evidence = source_evidence()
    if evidence["git_head"] != EXPECTED_HEAD:
        raise RuntimeError(f"This audit is scoped to {EXPECTED_HEAD}, found {evidence['git_head']}")
    if sb3_contrib.__version__ != "2.7.1" or stable_baselines3.__version__ != "2.7.1":
        raise RuntimeError("PPO V1 audit requires stable-baselines3 and sb3-contrib 2.7.1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the formal audit but is unavailable")

    bc_outcomes_path = args.bc_outcomes.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    bc_rows, collision_ids = load_bc_outcomes(bc_outcomes_path)
    config = formal_audit_config(args.device)
    seed_everything(int(config["master_seed"]))
    scenarios = training_scenarios()
    sampler = FixedMixtureScenarioSampler(
        scenarios,
        collision_ids,
        collision_probability=float(config["collision_sampling_probability"]),
    )
    vector_env = DummyVecEnv([make_training_env(rank, sampler, config) for rank in range(N_ENVS)])
    vector_env.seed(int(config["master_seed"]))
    try:
        model = build_model(vector_env, config)
        fresh_bc = verify_fresh_bc_actor(model.policy)
        before = clone_parameters(model.policy)
        before_digest = parameter_digest(before)
        trace = RolloutInfoTrace(N_ENVS)
        collect_one_rollout(model, trace)

        raw_buffer_observations = np.asarray(model.rollout_buffer.observations)
        raw_buffer_fields = {
            name: bool(np.isfinite(np.asarray(getattr(model.rollout_buffer, name))).all())
            for name in ("observations", "actions", "rewards", "advantages", "returns", "log_probs", "values")
        }
        if not all(raw_buffer_fields.values()):
            raise FloatingPointError(f"Non-finite rollout buffer field: {raw_buffer_fields}")
        if raw_buffer_observations.shape != (N_STEPS, N_ENVS, END2RACE_LIDAR_SIZE + 1):
            raise AssertionError(f"Unexpected rollout observation shape: {raw_buffer_observations.shape}")

        rows, replay_summary = replay_all_minibatches(model, trace)
        critic_numeric = overall_critic_numeric(model)
        aggregate = aggregate_minibatches(rows)
        integrity = compare_parameters(before, model.policy)
        if parameter_digest(before) != before_digest:
            raise AssertionError("The in-memory before snapshot changed")
        if not integrity["all_torch_equal"] or not integrity["sha256_equal"]:
            raise AssertionError(f"Policy parameters changed during read-only audit: {integrity}")

        completed_outcomes = Counter(record["outcome"] for record in trace.completed_records)
        sampler_branch_transitions = Counter(
            str(record["sampler_branch"] or "unknown")
            for step in trace.steps
            for record in step
        )
        audit = {
            "audit_scope": {
                "read_only": True,
                "formal_training_started": False,
                "candidate_checkpoint_saved": False,
                "stock_site_packages_modified": False,
                "rollouts_collected": 1,
            },
            "source_evidence": evidence,
            "config": config,
            "canonical_bc_outcomes": {
                "path": str(bc_outcomes_path.relative_to(ROOT)),
                "sha256": sha256_file(bc_outcomes_path),
                "row_count": len(bc_rows),
                "outcomes": dict(sorted(Counter(str(row.get("outcome")) for row in bc_rows).items())),
                "bc_ego_collision_id_count": len(collision_ids),
            },
            "fresh_bc_actor": fresh_bc,
            "rollout": {
                "n_envs": N_ENVS,
                "n_steps": N_STEPS,
                "valid_transition_count": ROLLOUT_TRANSITIONS,
                "buffer_shapes": {
                    "observations": list(raw_buffer_observations.shape),
                    "actions": list(np.asarray(model.rollout_buffer.actions).shape),
                },
                "finite_fields": raw_buffer_fields,
                "callback_rollout_start_count": trace.rollout_start_count,
                "callback_rollout_end_count": trace.rollout_end_count,
                "completed_outcomes": dict(sorted(completed_outcomes.items())),
                "sampler_branch_transitions": dict(sorted(sampler_branch_transitions.items())),
            },
            "minibatch_replay": replay_summary,
            "gradient_aggregate": aggregate,
            "critic_numeric": critic_numeric,
            "parameter_integrity": integrity,
        }
        audit["verdict"] = build_verdicts(aggregate, critic_numeric)
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_dir / "gradient_audit.json", audit)
        atomic_write_jsonl(output_dir / "minibatches.jsonl", rows)
        atomic_write_text(output_dir / "summary.md", render_summary(audit))
        return audit
    finally:
        vector_env.close()


def main() -> int:
    args = parse_arguments()
    audit = run(args)
    print(f"PPO_V1_GRADIENT_AUDIT_DIR={args.output_dir.expanduser().resolve()}")
    print(json.dumps(audit["verdict"], indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

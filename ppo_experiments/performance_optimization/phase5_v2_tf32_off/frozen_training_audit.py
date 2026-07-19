#!/usr/bin/env python3
"""All-four-minibatch R1/B/C numerical and performance audit on one bundle."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for path in (PROJECT_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_common import (
    CONFIG_NAME,
    OUTPUT_DIR,
    assert_locked_sources,
    backend_flags,
    diff_metrics,
    flatten_mapping,
    object_hash,
    provenance,
    restore_rng_state,
    sha256_file,
    state_dict_hash,
    write_json,
)
from batched_backends import REFERENCE_ACTOR_FORWARD, packed_actor_forward, timestep_batched_actor_forward
from model import End2Race
from ppo import config as ppo_config
from ppo.policy import EVALUATOR_STEER_BOUND, NOOP_SPEED_BOUND, End2RaceGRUPolicy
from ppo_experiments.performance_optimization.profile_pipeline import ResourceMonitor


BUNDLE_PATH = OUTPUT_DIR / "REFERENCE_BUNDLE.json"
MINIBATCH_PATH = OUTPUT_DIR / "frozen_minibatches_current.pt"
STATE_PATH = OUTPUT_DIR / "initial_state_and_rng_current.pt"
BACKENDS = ("R1", "B", "C")


def create_policy(state: dict[str, Any]) -> End2RaceGRUPolicy:
    config = ppo_config.get_config(CONFIG_NAME)
    observation_space = spaces.Box(
        low=np.full((361,), -np.inf, dtype=np.float32),
        high=np.full((361,), np.inf, dtype=np.float32),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=np.asarray((-EVALUATOR_STEER_BOUND, -NOOP_SPEED_BOUND), dtype=np.float32),
        high=np.asarray((EVALUATOR_STEER_BOUND, NOOP_SPEED_BOUND), dtype=np.float32),
        dtype=np.float32,
    )
    policy = End2RaceGRUPolicy(
        observation_space,
        action_space,
        lambda _: 1.0,
        checkpoint_path=ppo_config.BC_CHECKPOINT,
        hidden_scale=4,
        critic_hidden_size=64,
        critic_profile=config.critic_profile,
        gru_lr=config.gru_lr,
        head_lr=config.head_lr,
        steering_distribution=config.steering_distribution,
        steering_latent_std=config.steering_latent_std,
        speed_physical_std=config.speed_physical_std,
    ).to("cuda")
    policy.load_state_dict(state["model_state"], strict=True)
    policy.optimizer.load_state_dict(copy.deepcopy(state["optimizer_state"]))
    policy.set_training_mode(True)
    return policy


def cuda_batch(batch: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value.to("cuda", non_blocking=False) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    result["valid_tuple"] = tuple(
        tuple(bool(value) for value in row)
        for row in batch["valid_by_timestep"].tolist()
    )
    return result


def actor_forward(policy: Any, batch: dict[str, Any], backend: str):
    states = (batch["pi_hidden"], batch["pi_cell"])
    arguments = (
        policy,
        batch["observations"],
        states,
        batch["episode_starts"],
        batch["valid_tuple"],
    )
    if backend == "R1":
        return REFERENCE_ACTOR_FORWARD(*arguments)
    if backend == "B":
        return timestep_batched_actor_forward(*arguments)
    if backend == "C":
        return packed_actor_forward(*arguments)
    raise ValueError(backend)


def trainable_mapping(policy: Any, kind: str) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if kind == "parameter":
            output[name] = parameter.detach().cpu().clone()
        elif kind == "gradient":
            output[name] = (
                torch.zeros_like(parameter, device="cpu")
                if parameter.grad is None
                else parameter.grad.detach().cpu().clone()
            )
        else:
            raise ValueError(kind)
    return output


def optimizer_tensor_mapping(policy: Any) -> tuple[dict[str, torch.Tensor], list[int]]:
    state = policy.optimizer.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    steps: list[int] = []
    for parameter_id, parameter_state in sorted(state["state"].items()):
        for key, value in sorted(parameter_state.items()):
            if torch.is_tensor(value):
                tensors[f"{parameter_id}:{key}"] = value.detach().cpu().clone()
                if key == "step":
                    steps.append(int(value.item()))
    return tensors, steps


def forward_loss(policy: Any, batch: dict[str, Any], backend: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    config = ppo_config.get_config(CONFIG_NAME)
    means, next_states, actor_features = actor_forward(policy, batch, backend)
    distribution = policy._distribution(means)
    log_prob = distribution.log_prob(batch["actions"])
    values = policy._critic_values(batch["observations"], actor_features).flatten()
    mask = batch["mask"] > 1e-8
    advantages = batch["advantages"]
    if ppo_config.NORMALIZE_ADVANTAGE:
        advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
    ratio = torch.exp(log_prob - batch["old_log_prob"])
    policy_loss = -torch.mean(
        torch.min(
            advantages * ratio,
            advantages * torch.clamp(
                ratio,
                1.0 - ppo_config.CLIP_RANGE,
                1.0 + ppo_config.CLIP_RANGE,
            ),
        )[mask]
    )
    values_pred = values
    value_loss = torch.mean(((batch["returns"] - values_pred) ** 2)[mask])
    entropy_surrogate = -log_prob
    entropy_loss = -torch.mean(entropy_surrogate[mask])
    total_loss = policy_loss + ppo_config.ENT_COEF * entropy_loss + ppo_config.VF_COEF * value_loss
    log_ratio = log_prob - batch["old_log_prob"]
    approx_kl = torch.mean(((torch.exp(log_ratio) - 1.0) - log_ratio)[mask])
    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > ppo_config.CLIP_RANGE).float()[mask])
    outputs = {
        "actor_mean": means,
        "hidden": next_states[0],
        "latent_mean": distribution.latent_steer_mean,
        "new_logp": log_prob,
        "entropy_surrogate": entropy_surrogate,
        "value": values,
    }
    metrics = {
        "valid_samples": int(mask.sum().item()),
        "padded_samples": int(mask.numel()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "entropy_loss": float(entropy_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "approx_kl": float(approx_kl.detach().item()),
        "clip_fraction": float(clip_fraction.detach().item()),
        "ratio_min": float(ratio[mask].detach().min().item()),
        "ratio_max": float(ratio[mask].detach().max().item()),
        "finite": bool(
            all(torch.isfinite(value).all().item() for value in outputs.values())
            and torch.isfinite(total_loss).item()
        ),
        "mask": mask,
        "total_loss_tensor": total_loss,
    }
    return outputs, metrics


def one_optimizer_step(policy: Any, batch: dict[str, Any], backend: str) -> dict[str, Any]:
    policy.optimizer.zero_grad()
    before = trainable_mapping(policy, "parameter")
    outputs, metrics = forward_loss(policy, batch, backend)
    metrics.pop("total_loss_tensor").backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo_config.MAX_GRAD_NORM)
    gradients = trainable_mapping(policy, "gradient")
    policy.optimizer.step()
    after = trainable_mapping(policy, "parameter")
    deltas = {name: after[name] - before[name] for name in before}
    optimizer_tensors, optimizer_steps = optimizer_tensor_mapping(policy)
    mask = metrics.pop("mask")
    return {
        "metrics": metrics,
        "mask": mask.detach().cpu(),
        "outputs": {name: value.detach().cpu() for name, value in outputs.items()},
        "gradients": gradients,
        "deltas": deltas,
        "parameters_after": after,
        "optimizer_tensors": optimizer_tensors,
        "optimizer_steps": optimizer_steps,
        "optimizer_hash": object_hash(policy.optimizer.state_dict()),
    }


def run_numerical(state: dict[str, Any], batches: list[dict[str, Any]], backend: str):
    restore_rng_state(state["training_rng"])
    policy = create_policy(state)
    records = [one_optimizer_step(policy, batch, backend) for batch in batches]
    return policy, records


def group_metric(mapping_ref: dict[str, torch.Tensor], mapping_candidate: dict[str, torch.Tensor], prefix: str):
    names = [name for name in sorted(mapping_ref) if name.startswith(prefix)]
    ref = torch.cat([mapping_ref[name].reshape(-1) for name in names])
    candidate = torch.cat([mapping_candidate[name].reshape(-1) for name in names])
    return diff_metrics(ref, candidate)


def mapping_metrics(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]):
    return {
        "full": diff_metrics(flatten_mapping(reference), flatten_mapping(candidate)),
        "gru": group_metric(reference, candidate, "lstm_actor.gru."),
        "output_head": group_metric(reference, candidate, "end2race_actor.output_layer."),
        "critic": group_metric(reference, candidate, "value_net."),
        "per_parameter": {
            name: diff_metrics(reference[name], candidate[name])
            for name in sorted(reference)
        },
    }


def large_vector_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """Exact aggregate norms with deterministic quantile sampling for huge vectors."""

    if reference.numel() <= 8_000_000:
        return diff_metrics(reference, candidate)
    ref = reference.detach().double().cpu().reshape(-1)
    cand = candidate.detach().double().cpu().reshape(-1)
    difference = (cand - ref).abs()
    stride = max(1, (difference.numel() + 999_999) // 1_000_000)
    sample = difference[::stride]
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    cand_norm = float(torch.linalg.vector_norm(cand).item())
    diff_norm = float(torch.linalg.vector_norm(cand - ref).item())
    cosine = 1.0 if ref_norm == 0.0 and cand_norm == 0.0 else float(
        torch.dot(ref, cand).item() / max(ref_norm * cand_norm, torch.finfo(torch.float64).tiny)
    )
    return {
        "count": int(ref.numel()),
        "finite": bool(torch.isfinite(ref).all() and torch.isfinite(cand).all()),
        "mean_abs": float(difference.mean().item()),
        "p50_abs": float(torch.quantile(sample, 0.50).item()),
        "p95_abs": float(torch.quantile(sample, 0.95).item()),
        "p99_abs": float(torch.quantile(sample, 0.99).item()),
        "max_abs": float(difference.max().item()),
        "relative_l2": diff_norm / max(ref_norm, torch.finfo(torch.float64).tiny),
        "cosine": cosine,
        "quantile_sample_count": int(sample.numel()),
        "quantile_stride": stride,
    }


def compare_minibatch(reference: dict[str, Any], candidate: dict[str, Any], policy: Any) -> dict[str, Any]:
    mask = reference["mask"].bool()
    outputs = {
        name: diff_metrics(reference["outputs"][name][mask] if reference["outputs"][name].shape[0] == mask.shape[0] else reference["outputs"][name],
                           candidate["outputs"][name][mask] if candidate["outputs"][name].shape[0] == mask.shape[0] else candidate["outputs"][name])
        for name in reference["outputs"]
    }
    ref_means = reference["outputs"]["actor_mean"][mask]
    cand_means = candidate["outputs"]["actor_mean"][mask]
    ref_latent = reference["outputs"]["latent_mean"][mask]
    cand_latent = candidate["outputs"]["latent_mean"][mask]
    steer_std = float(ppo_config.get_config(CONFIG_NAME).steering_latent_std)
    speed_std = float(ppo_config.get_config(CONFIG_NAME).speed_physical_std)
    policy_kl = torch.mean(
        0.5 * ((cand_latent - ref_latent) / steer_std).square()
        + 0.5 * ((cand_means[:, 1] - ref_means[:, 1]) / speed_std).square()
    )
    losses = {
        name: {
            "reference": reference["metrics"][name],
            "candidate": candidate["metrics"][name],
            "absolute_difference": abs(candidate["metrics"][name] - reference["metrics"][name]),
        }
        for name in (
            "policy_loss",
            "value_loss",
            "entropy_loss",
            "total_loss",
            "approx_kl",
            "clip_fraction",
        )
    }
    return {
        "valid_samples": int(mask.sum().item()),
        "padded_samples": int(mask.numel()),
        "finite": bool(reference["metrics"]["finite"] and candidate["metrics"]["finite"]),
        "forward": outputs,
        "reference_vs_candidate_policy_kl": float(policy_kl.item()),
        "loss": losses,
        "gradient": mapping_metrics(reference["gradients"], candidate["gradients"]),
        "parameter_delta": mapping_metrics(reference["deltas"], candidate["deltas"]),
        "parameter_after": mapping_metrics(reference["parameters_after"], candidate["parameters_after"]),
        "optimizer_state": {
            "tensor_difference": large_vector_metrics(
                flatten_mapping(reference["optimizer_tensors"]),
                flatten_mapping(candidate["optimizer_tensors"]),
            ),
            "reference_step_counts": reference["optimizer_steps"],
            "candidate_step_counts": candidate["optimizer_steps"],
            "step_counts_equal": reference["optimizer_steps"] == candidate["optimizer_steps"],
            "reference_hash": reference["optimizer_hash"],
            "candidate_hash": candidate["optimizer_hash"],
        },
    }


def actor_forward_timing(policy: Any, batches: list[dict[str, Any]], backend: str) -> dict[str, float]:
    with torch.no_grad():
        actor_forward(policy, batches[0], backend)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        actor_forward(policy, batches[0], backend)
        end.record()
        torch.cuda.synchronize()
        first_ms = float(start.elapsed_time(end))
        for batch in batches:
            actor_forward(policy, batch, backend)
        torch.cuda.synchronize()
        start_all = torch.cuda.Event(enable_timing=True)
        end_all = torch.cuda.Event(enable_timing=True)
        start_all.record()
        for batch in batches:
            actor_forward(policy, batch, backend)
        end_all.record()
        torch.cuda.synchronize()
        all_ms = float(start_all.elapsed_time(end_all))
    return {"first_logical_minibatch_actor_cuda_ms": first_ms, "all_four_actor_cuda_ms": all_ms}


def timed_full_train(state: dict[str, Any], batches: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    # One complete warm-up from the same immutable state, then one measured run.
    warm_policy, warm_records = run_numerical(state, batches, backend)
    del warm_policy, warm_records
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    policy = create_policy(state)
    restore_rng_state(state["training_rng"])
    torch.cuda.reset_peak_memory_stats()
    monitor = ResourceMonitor()
    monitor.start()
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        for batch in batches:
            one_optimizer_step(policy, batch, backend)
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - start
    finally:
        monitor.stop()
    return {
        "ppo_train_wall_s": wall_s,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "resources": monitor.summary(),
    }


def save_checkpoint(policy: Any, backend: str) -> dict[str, Any]:
    destination = OUTPUT_DIR / f"{backend}_frozen_actor_checkpoint.pth"
    state = {name: tensor.detach().cpu() for name, tensor in policy.actor_checkpoint_state_dict().items()}
    torch.save(state, destination)
    fresh = End2Race(mask_prob=0.0, hidden_scale=4)
    fresh.load_state_dict(torch.load(destination, map_location="cpu", weights_only=True), strict=True)
    return {
        "path": destination.name,
        "sha256": sha256_file(destination),
        "keys": sorted(state),
        "strict_12_key_load": len(state) == 12,
    }


def candidate_gate(minibatches: list[dict[str, Any]], checkpoint: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "finite": all(batch["finite"] for batch in minibatches),
        "policy_kl_le_1e-6": all(batch["reference_vs_candidate_policy_kl"] <= 1e-6 for batch in minibatches),
        "gradient_cosine_ge_0.99999": all(batch["gradient"]["full"]["cosine"] >= 0.99999 for batch in minibatches),
        "delta_cosine_ge_0.99999": all(batch["parameter_delta"]["full"]["cosine"] >= 0.99999 for batch in minibatches),
        "gradient_relative_l2_le_1e-3": all(batch["gradient"]["full"]["relative_l2"] <= 1e-3 for batch in minibatches),
        "delta_relative_l2_le_1e-3": all(batch["parameter_delta"]["full"]["relative_l2"] <= 1e-3 for batch in minibatches),
        "policy_loss_abs_le_1e-5": all(batch["loss"]["policy_loss"]["absolute_difference"] <= 1e-5 for batch in minibatches),
        "optimizer_steps_equal": all(batch["optimizer_state"]["step_counts_equal"] for batch in minibatches),
        "strict_12_key_load": bool(checkpoint["strict_12_key_load"]),
    }
    return all(checks.values()), checks


def main() -> None:
    assert_locked_sources()
    bundle = json.loads(BUNDLE_PATH.read_text())
    state = torch.load(STATE_PATH, map_location="cpu", weights_only=False)
    cpu_batches = torch.load(MINIBATCH_PATH, map_location="cpu", weights_only=False)
    batches = [cuda_batch(batch) for batch in cpu_batches]
    if state_dict_hash(state["model_state"]) != bundle["model_initial_hash"]:
        raise RuntimeError("model hash mismatch")
    if object_hash(state["optimizer_state"]) != bundle["optimizer_initial_hash"]:
        raise RuntimeError("optimizer hash mismatch")
    with backend_flags(True) as flags:
        reference_policy, reference = run_numerical(state, batches, "R1")
        reference_checkpoint = save_checkpoint(reference_policy, "R1")
        timing: dict[str, Any] = {}
        timing["R1"] = {
            **actor_forward_timing(create_policy(state), batches, "R1"),
            **timed_full_train(state, batches, "R1"),
        }
        reference_record = {
            "schema_version": 1,
            **provenance("R1 frozen all-four-minibatch reference", 1, flags, bundle["rollout_hash"]),
            "model_initial_hash": bundle["model_initial_hash"],
            "optimizer_initial_hash": bundle["optimizer_initial_hash"],
            "rng_initial_hash": bundle["training_rng_hashes"],
            "minibatch_order_hash": bundle["minibatch_order_hash"],
            "backend": "R1 batch-1 recurrent replay",
            "batch_or_microbatch": 1,
            "numerical_metrics": {
                "minibatches": [record["metrics"] for record in reference],
            },
            "timing_metrics": timing["R1"],
            "checkpoint_hash": reference_checkpoint["sha256"],
            "checkpoint": reference_checkpoint,
            "verdict": "R1_FROZEN_TRAINING_REFERENCE",
        }
        write_json(OUTPUT_DIR / "R1_FROZEN_TRAINING.json", reference_record)

        for backend, filename, label in (
            ("B", "B_TIMESTEP.json", "B training-time active timestep batching TF32 off"),
            ("C", "C_PACKED.json", "C packed full-sequence GRU TF32 off"),
        ):
            candidate_policy, candidate = run_numerical(state, batches, backend)
            comparisons = [
                compare_minibatch(reference[index], candidate[index], candidate_policy)
                for index in range(4)
            ]
            checkpoint = save_checkpoint(candidate_policy, backend)
            timing[backend] = {
                **actor_forward_timing(create_policy(state), batches, backend),
                **timed_full_train(state, batches, backend),
            }
            passed, checks = candidate_gate(comparisons, checkpoint)
            record = {
                "schema_version": 1,
                **provenance(label, "all active" if backend == "B" else "packed", flags, bundle["rollout_hash"]),
                "model_initial_hash": bundle["model_initial_hash"],
                "optimizer_initial_hash": bundle["optimizer_initial_hash"],
                "rng_initial_hash": bundle["training_rng_hashes"],
                "minibatch_order_hash": bundle["minibatch_order_hash"],
                "reference_bundle_ref": BUNDLE_PATH.name,
                "backend": label,
                "batch_or_microbatch": "all active sequences" if backend == "B" else "full logical minibatch sequences",
                "numerical_metrics": {
                    "all_four_minibatches": comparisons,
                    "gate_checks": checks,
                    "gate_pass": passed,
                },
                "timing_metrics": timing[backend],
                "checkpoint_hash": checkpoint["sha256"],
                "checkpoint": checkpoint,
                "verdict": f"{backend}_NUMERIC_PASS" if passed else f"{backend}_NUMERIC_FAIL",
            }
            write_json(OUTPUT_DIR / filename, record)
    assert_locked_sources()
    print(json.dumps({"R1": timing["R1"], "B": timing["B"], "C": timing["C"]}, sort_keys=True))


if __name__ == "__main__":
    main()

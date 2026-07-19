#!/usr/bin/env python3
"""A batch-16 open-loop, 1,400-step teacher-forced, and closed-loop audit."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType
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
    diff_metrics,
    provenance,
    restore_rng_state,
    write_json,
)
from batched_backends import REFERENCE_ACTOR_FORWARD, install_dispatch, timestep_batched_actor_forward
from frozen_training_audit import create_policy
from ppo.policy import END2RACE_LIDAR_SIZE


CHECKPOINT_STEPS = (1, 10, 100, 400, 800, 1400)


def tensor_summary(values: list[torch.Tensor]) -> dict[str, float | int | bool]:
    count = 0
    total = 0.0
    maximum = 0.0
    finite = True
    samples: list[torch.Tensor] = []
    for value in values:
        tensor = value.detach().double().cpu().reshape(-1)
        count += int(tensor.numel())
        total += float(tensor.sum().item())
        maximum = max(maximum, float(tensor.max().item()))
        finite = finite and bool(torch.isfinite(tensor).all())
        stride = max(1, (tensor.numel() + 1023) // 1024)
        samples.append(tensor[::stride][:1024])
    sample = torch.cat(samples)
    return {
        "count": count,
        "finite": finite,
        "mean_abs": total / count,
        "p50_abs": float(torch.quantile(sample, 0.50).item()),
        "p95_abs": float(torch.quantile(sample, 0.95).item()),
        "p99_abs": float(torch.quantile(sample, 0.99).item()),
        "max_abs": maximum,
        "quantile_sample_count": int(sample.numel()),
    }


def sampled_actions_and_logp(policy: Any, means: torch.Tensor, rng: tuple[torch.Tensor, list[torch.Tensor]]):
    torch.set_rng_state(rng[0])
    torch.cuda.set_rng_state_all(rng[1])
    distribution = policy._distribution(means)
    actions = distribution.get_actions(deterministic=False)
    return actions, distribution.log_prob(actions), distribution.latent_steer_mean.detach().clone()


def actor_features(policy: Any, observations: torch.Tensor, scalar: bool) -> torch.Tensor:
    actor = policy.end2race_actor
    pieces = []
    indices = range(observations.shape[0]) if scalar else (slice(None),)
    for index in indices:
        value = observations[index : index + 1] if isinstance(index, int) else observations[index]
        lidar = value[:, :END2RACE_LIDAR_SIZE]
        speed = value[:, END2RACE_LIDAR_SIZE:]
        processed = (-1.0 / (1.0 + torch.exp(-actor.k * lidar)) + 1.0) * 2.0
        pieces.append(torch.cat((processed, actor.speed_mlp(speed)), dim=1))
    return torch.cat(pieces, dim=0)


def teacher_forced(bundle: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    raw = np.load(OUTPUT_DIR / "frozen_rollout_current.npz")
    observations = raw["observations"]
    actions = raw["actions"]
    episode_starts = raw["episode_starts"]
    hidden_states = raw["hidden_states_pi"]
    if observations.shape[:2] != (1600, 16):
        raise RuntimeError(f"unexpected rollout shape: {observations.shape}")
    restore_rng_state(state["training_rng"])
    policy = create_policy(state)
    ref_hidden = torch.as_tensor(hidden_states[0], device="cuda")
    candidate_hidden = ref_hidden.clone()
    zero = torch.zeros_like(ref_hidden)
    aggregate: dict[str, list[torch.Tensor]] = defaultdict(list)
    checkpoints: dict[str, Any] = {}
    first_step: dict[str, int | None] = {
        "sampled_action_gt_1e-5": None,
        "sampled_action_gt_1e-4": None,
        "sampled_action_gt_1e-3": None,
    }
    a1_feature = None
    with torch.no_grad():
        for step in range(1400):
            obs = torch.as_tensor(observations[step], device="cuda")
            starts = torch.as_tensor(episode_starts[step], device="cuda")
            ref_means, ref_states, _ = REFERENCE_ACTOR_FORWARD(
                policy, obs, (ref_hidden, zero), starts, None
            )
            candidate_means, candidate_states, _ = timestep_batched_actor_forward(
                policy, obs, (candidate_hidden, zero), starts, None
            )
            if step == 0:
                a1_feature = diff_metrics(actor_features(policy, obs, True), actor_features(policy, obs, False))
            rng = (torch.get_rng_state().clone(), [value.clone() for value in torch.cuda.get_rng_state_all()])
            ref_action, ref_logp, ref_latent = sampled_actions_and_logp(policy, ref_means, rng)
            candidate_action, candidate_logp, candidate_latent = sampled_actions_and_logp(policy, candidate_means, rng)
            fixed_action = torch.as_tensor(actions[step], device="cuda")
            ref_replay_logp = policy._distribution(ref_means).log_prob(fixed_action)
            candidate_replay_logp = policy._distribution(candidate_means).log_prob(fixed_action)
            differences = {
                "hidden": (candidate_states[0] - ref_states[0]).abs(),
                "physical_mean": (candidate_means - ref_means).abs(),
                "latent_mean": (candidate_latent - ref_latent).abs(),
                "sampled_action": (candidate_action - ref_action).abs(),
                "sampled_logp": (candidate_logp - ref_logp).abs(),
                "replay_logp": (candidate_replay_logp - ref_replay_logp).abs(),
            }
            for name, difference in differences.items():
                if name == "hidden":
                    # Exact checkpoint distributions; bounded deterministic sample for the aggregate.
                    aggregate[name].append(difference.reshape(-1)[::32].cpu())
                else:
                    aggregate[name].append(difference.cpu())
            maximum = float(differences["sampled_action"].max().item())
            for threshold, key in (
                (1e-5, "sampled_action_gt_1e-5"),
                (1e-4, "sampled_action_gt_1e-4"),
                (1e-3, "sampled_action_gt_1e-3"),
            ):
                if first_step[key] is None and maximum > threshold:
                    first_step[key] = step + 1
            if step + 1 in CHECKPOINT_STEPS:
                checkpoints[str(step + 1)] = {
                    "hidden": diff_metrics(ref_states[0], candidate_states[0]),
                    "physical_mean": diff_metrics(ref_means, candidate_means),
                    "latent_mean": diff_metrics(ref_latent, candidate_latent),
                    "sampled_action": diff_metrics(ref_action, candidate_action),
                    "sampled_logp": diff_metrics(ref_logp, candidate_logp),
                    "replay_logp": diff_metrics(ref_replay_logp, candidate_replay_logp),
                }
            ref_hidden = ref_states[0]
            candidate_hidden = candidate_states[0]
    a1 = checkpoints["1"]
    a1["processed_feature"] = a1_feature
    a1_checks = {
        "hidden_max_le_5e-6": a1["hidden"]["max_abs"] <= 5e-6,
        "physical_mean_max_le_5e-6": a1["physical_mean"]["max_abs"] <= 5e-6,
        "latent_mean_p99_le_5e-6": a1["latent_mean"]["p99_abs"] <= 5e-6,
        "logp_p99_le_1e-5": a1["sampled_logp"]["p99_abs"] <= 1e-5,
        "finite": all(record["finite"] for record in a1.values()),
    }
    final = checkpoints["1400"]
    # Exponential divergence is conservatively rejected if the last checkpoint is
    # over 16x the 400-step error while also exceeding the absolute gate.
    exponential = (
        final["hidden"]["max_abs"] > 16.0 * max(checkpoints["400"]["hidden"]["max_abs"], 1e-12)
        and final["hidden"]["max_abs"] > 2e-5
    )
    a2_checks = {
        "hidden_max_le_2e-5": final["hidden"]["max_abs"] <= 2e-5,
        "physical_mean_p99_le_2e-5": final["physical_mean"]["p99_abs"] <= 2e-5,
        "no_exponential_divergence": not exponential,
        "finite": all(record["finite"] for record in final.values()),
    }
    return {
        "steps": 1400,
        "A1_open_loop": {"metrics": a1, "gate_checks": a1_checks, "gate_pass": all(a1_checks.values())},
        "A2_teacher_forced": {
            "checkpoints": checkpoints,
            "aggregate": {name: tensor_summary(values) for name, values in aggregate.items()},
            "first_threshold_step": first_step,
            "gate_checks": a2_checks,
            "gate_pass": all(a2_checks.values()),
        },
    }


def first_array_difference(reference: list[np.ndarray], candidate: list[np.ndarray], threshold: float = 0.0):
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if float(np.max(np.abs(np.asarray(left) - np.asarray(right)))) > threshold:
            return index + 1
    return None


def run_closed_loop(backend: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from ppo_experiments.performance_optimization import validate_pipeline

    original_build_model = validate_pipeline.build_model
    trace: dict[str, Any] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "positions": [],
        "episodes": [],
        "within_trajectory": defaultdict(list),
        "within_first_threshold": {"action_gt_1e-5": None, "action_gt_1e-4": None, "action_gt_1e-3": None},
    }

    def instrumented_build_model(vector_env: Any, config: Any, seed: int):
        model = original_build_model(vector_env, config, seed)
        policy = model.policy
        if backend == "A":
            install_dispatch(policy, "A")
        base_forward = policy.forward
        dispatch = policy._actor_forward

        def traced_forward(observation, lstm_states, episode_starts, deterministic=False):
            trace["observations"].append(observation.detach().cpu().numpy().copy())
            if backend == "A":
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
                policy._actor_forward = MethodType(REFERENCE_ACTOR_FORWARD, policy)
                reference = base_forward(observation, lstm_states, episode_starts, deterministic)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                policy._actor_forward = dispatch
                candidate = base_forward(observation, lstm_states, episode_starts, deterministic)
                comparisons = {
                    "action": (candidate[0] - reference[0]).abs(),
                    "logp": (candidate[2] - reference[2]).abs(),
                    "hidden": (candidate[3].pi[0] - reference[3].pi[0]).abs(),
                }
                for name, difference in comparisons.items():
                    trace["within_trajectory"][name].append(difference.detach().cpu())
                action_max = float(comparisons["action"].max().item())
                step = len(trace["observations"])
                for threshold, key in ((1e-5, "action_gt_1e-5"), (1e-4, "action_gt_1e-4"), (1e-3, "action_gt_1e-3")):
                    if trace["within_first_threshold"][key] is None and action_max > threshold:
                        trace["within_first_threshold"][key] = step
                result = candidate
            else:
                result = base_forward(observation, lstm_states, episode_starts, deterministic)
            trace["actions"].append(result[0].detach().cpu().numpy().copy())
            return result

        policy.forward = traced_forward
        original_step_wait = vector_env.step_wait
        ego_indices = vector_env.get_attr("ego_index")

        def traced_step_wait():
            observation, rewards, dones, infos = original_step_wait()
            trace["rewards"].append(np.asarray(rewards).copy())
            raw_observations = vector_env.get_attr("_raw_observation")
            positions = []
            for raw, ego_index in zip(raw_observations, ego_indices):
                positions.append((float(raw["poses_x"][ego_index]), float(raw["poses_y"][ego_index])))
            trace["positions"].append(np.asarray(positions, dtype=np.float64))
            for rank, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    trace["episodes"].append({
                        "step": len(trace["rewards"]),
                        "env_rank": rank,
                        "scenario_id": str(info["scenario_id"]),
                        "outcome": info["episode_outcome"],
                        "elapsed_time": float(info["elapsed_time"]),
                        "ego_collision": bool(info["ego_collision"]),
                        "opponent_collision": bool(info["opponent_collision"]),
                        "timeout": bool(info["timeout"]),
                    })
            return observation, rewards, dones, infos

        vector_env.step_wait = traced_step_wait
        return model

    validate_pipeline.build_model = instrumented_build_model
    try:
        contract = validate_pipeline.capture(
            CONFIG_NAME,
            SEED,
            "central_subproc",
            WORKER_COUNT,
            False,
            f"phase5_v2_closed_loop_{backend}",
        )
    finally:
        validate_pipeline.build_model = original_build_model
    compact = {
        "observations": trace["observations"],
        "actions": trace["actions"],
        "rewards": trace["rewards"],
        "positions": trace["positions"],
        "episodes": trace["episodes"],
        "within_first_threshold": trace["within_first_threshold"],
        "within_trajectory": {
            name: tensor_summary(values)
            for name, values in trace["within_trajectory"].items()
        },
    }
    return contract, compact


def closed_loop_comparison(reference_contract: dict[str, Any], reference: dict[str, Any], candidate_contract: dict[str, Any], candidate: dict[str, Any]):
    positions_ref = np.asarray(reference["positions"])
    positions_candidate = np.asarray(candidate["positions"])
    displacement = np.linalg.norm(positions_candidate - positions_ref, axis=-1)
    reward_ref = np.asarray(reference["rewards"])
    reward_candidate = np.asarray(candidate["rewards"])
    reward_difference = np.abs(reward_candidate - reward_ref)
    first_hash_divergence = {}
    for key in reference_contract["step_hashes"]:
        first_hash_divergence[key] = next(
            (
                index + 1
                for index, (left, right) in enumerate(
                    zip(reference_contract["step_hashes"][key], candidate_contract["step_hashes"][key])
                )
                if left != right
            ),
            None,
        )
    return {
        "steps": len(reference["observations"]),
        "first_observation_divergence_step": first_array_difference(reference["observations"], candidate["observations"]),
        "first_action_difference_step": {
            "gt_1e-5": first_array_difference(reference["actions"], candidate["actions"], 1e-5),
            "gt_1e-4": first_array_difference(reference["actions"], candidate["actions"], 1e-4),
            "gt_1e-3": first_array_difference(reference["actions"], candidate["actions"], 1e-3),
        },
        "first_contract_hash_divergence": first_hash_divergence,
        "trajectory_displacement_m": {
            "mean": float(displacement.mean()),
            "p95": float(np.quantile(displacement, 0.95)),
            "p99": float(np.quantile(displacement, 0.99)),
            "max": float(displacement.max()),
        },
        "rollout_reward_abs_difference": {
            "mean": float(reward_difference.mean()),
            "p99": float(np.quantile(reward_difference, 0.99)),
            "max": float(reward_difference.max()),
            "sum_reference": float(reward_ref.sum()),
            "sum_candidate": float(reward_candidate.sum()),
        },
        "reference_events": reference_contract["events"],
        "candidate_events": candidate_contract["events"],
        "reference_outcomes": reference_contract["outcomes"],
        "candidate_outcomes": candidate_contract["outcomes"],
        "reference_episodes": reference["episodes"],
        "candidate_episodes": candidate["episodes"],
        "candidate_within_trajectory": {
            "first_threshold_step": candidate["within_first_threshold"],
            "differences": candidate["within_trajectory"],
        },
        "reset_order_equal": reference_contract["reset_order"] == candidate_contract["reset_order"],
        "reset_specs_equal": reference_contract["reset_specs"] == candidate_contract["reset_specs"],
        "scenario_manifest_equal": reference_contract["scenario_manifest_sha256"] == candidate_contract["scenario_manifest_sha256"],
    }


def save_closed_loop_artifacts(backend: str, contract: dict[str, Any], trace: dict[str, Any]) -> None:
    np.savez(
        OUTPUT_DIR / f"CLOSED_{backend}_TRACE.npz",
        observations=np.asarray(trace["observations"]),
        actions=np.asarray(trace["actions"]),
        rewards=np.asarray(trace["rewards"]),
        positions=np.asarray(trace["positions"]),
    )
    write_json(OUTPUT_DIR / f"CLOSED_{backend}_CONTRACT.json", contract)
    write_json(
        OUTPUT_DIR / f"CLOSED_{backend}_META.json",
        {
            "episodes": trace["episodes"],
            "within_first_threshold": trace["within_first_threshold"],
            "within_trajectory": trace["within_trajectory"],
        },
    )


def load_closed_loop_artifacts(backend: str) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = np.load(OUTPUT_DIR / f"CLOSED_{backend}_TRACE.npz")
    metadata = json.loads((OUTPUT_DIR / f"CLOSED_{backend}_META.json").read_text())
    trace = {
        "observations": arrays["observations"],
        "actions": arrays["actions"],
        "rewards": arrays["rewards"],
        "positions": arrays["positions"],
        **metadata,
    }
    return json.loads((OUTPUT_DIR / f"CLOSED_{backend}_CONTRACT.json").read_text()), trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closed-backend", choices=("R1", "A"))
    parser.add_argument("--reuse-closed", action="store_true")
    args = parser.parse_args()
    assert_locked_sources()
    if args.closed_backend is not None:
        if torch.cuda.is_initialized():
            raise RuntimeError("closed-loop worker entered with CUDA already initialized")
        with backend_flags(True):
            contract, trace = run_closed_loop(args.closed_backend)
            save_closed_loop_artifacts(args.closed_backend, contract, trace)
        assert_locked_sources()
        print(json.dumps({"closed_backend": args.closed_backend, "wall_s": contract["wall_s"]}, sort_keys=True))
        return

    bundle = json.loads((OUTPUT_DIR / "REFERENCE_BUNDLE.json").read_text())
    state = torch.load(OUTPUT_DIR / "initial_state_and_rng_current.pt", map_location="cpu", weights_only=False)
    if not args.reuse_closed:
        for backend in ("R1", "A"):
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--closed-backend", backend],
                cwd=PROJECT_ROOT,
                check=True,
            )
    with backend_flags(True) as flags:
        numerical = teacher_forced(bundle, state)
        reference_contract, reference_trace = load_closed_loop_artifacts("R1")
        candidate_contract, candidate_trace = load_closed_loop_artifacts("A")
        closed_loop = closed_loop_comparison(reference_contract, reference_trace, candidate_contract, candidate_trace)
        a_pass = numerical["A1_open_loop"]["gate_pass"] and numerical["A2_teacher_forced"]["gate_pass"]
        result = {
            "schema_version": 1,
            **provenance("A collection-time env batching TF32 off", 16, flags, bundle["rollout_hash"]),
            "model_initial_hash": bundle["model_initial_hash"],
            "optimizer_initial_hash": bundle["optimizer_initial_hash"],
            "rng_initial_hash": bundle["training_rng_hashes"],
            "minibatch_order_hash": bundle["minibatch_order_hash"],
            "reference_bundle_ref": "REFERENCE_BUNDLE.json",
            "backend": "A collection-time env batching TF32 off",
            "batch_or_microbatch": 16,
            "numerical_metrics": {**numerical, "A3_closed_loop": closed_loop},
            "timing_metrics": {},
            "checkpoint_hash": candidate_contract["checkpoint"]["sha256"],
            "closed_loop_checkpoint": candidate_contract["checkpoint"],
            "verdict": "A_NUMERIC_PASS_DISTRIBUTIONAL_ONLY" if a_pass else "A_NUMERIC_FAIL",
        }
        write_json(OUTPUT_DIR / "A_BATCH16.json", result)
    assert_locked_sources()
    print(json.dumps({"verdict": result["verdict"], "A1": numerical["A1_open_loop"]["gate_checks"], "A2": numerical["A2_teacher_forced"]["gate_checks"]}, sort_keys=True))


if __name__ == "__main__":
    main()

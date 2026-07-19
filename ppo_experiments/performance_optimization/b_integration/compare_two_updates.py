#!/usr/bin/env python3
"""Compute the Phase 5B integration numeric gates from the two-update captures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parent
STEER_STD = 0.05
SPEED_STD = 0.15
CLIP_RANGE = 0.10


def vector_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    difference = candidate - reference
    reference_norm = float(np.linalg.norm(reference))
    cosine = float(
        np.dot(reference, candidate)
        / max(np.linalg.norm(reference) * np.linalg.norm(candidate), np.finfo(np.float64).tiny)
    )
    return {
        "max_abs": float(np.abs(difference).max()),
        "relative_l2": float(np.linalg.norm(difference) / max(reference_norm, np.finfo(np.float64).tiny)),
        "cosine": cosine,
    }


def policy_loss(batch: dict[str, np.ndarray], log_prob: np.ndarray) -> float:
    mask = batch["mask"] > 1e-8
    advantages = batch["advantages"].astype(np.float64)
    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std(ddof=1) + 1e-8)
    ratio = np.exp(log_prob.astype(np.float64) - batch["old_log_prob"].astype(np.float64))
    unclipped = advantages * ratio
    clipped = advantages * np.clip(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
    return float(-np.minimum(unclipped, clipped)[mask].mean())


reference = np.load(DIRECTORY / "reference_capture.npz")
candidate = np.load(DIRECTORY / "b_integrated_capture.npz")
reference_meta = json.load(open(DIRECTORY / "reference_meta.json"))
candidate_meta = json.load(open(DIRECTORY / "b_integrated_meta.json"))

assert reference_meta["parameter_names"] == candidate_meta["parameter_names"]
result: dict[str, object] = {
    "u1_rollout_identical": reference_meta["updates"]["1"]["rollout_sha256"]
    == candidate_meta["updates"]["1"]["rollout_sha256"],
    "u2_rollout_identical": reference_meta["updates"]["2"]["rollout_sha256"]
    == candidate_meta["updates"]["2"]["rollout_sha256"],
    "u1_outcomes": [reference_meta["updates"]["1"]["outcomes"], candidate_meta["updates"]["1"]["outcomes"]],
    "u2_outcomes": [reference_meta["updates"]["2"]["outcomes"], candidate_meta["updates"]["2"]["outcomes"]],
    "optimizer_steps": [
        reference_meta["updates"]["2"]["optimizer_step"],
        candidate_meta["updates"]["2"]["optimizer_step"],
    ],
    "u2_checkpoints": {
        "reference": reference_meta["u2_checkpoint_sha256"],
        "b_integrated": candidate_meta["u2_checkpoint_sha256"],
    },
    "minibatches": [],
}

worst = {
    "policy_kl": 0.0,
    "gradient_cosine": 1.0,
    "gradient_relative_l2": 0.0,
    "policy_loss_abs": 0.0,
}
for index in range(4):
    for name in ("mask", "old_log_prob", "advantages"):
        assert np.array_equal(reference[f"{name}_{index}"], candidate[f"{name}_{index}"]), (
            f"minibatch {index} field {name} differs: rollout/minibatch order not aligned"
        )
    mask = reference[f"mask_{index}"] > 1e-8
    delta_latent = (
        candidate[f"latent_mean_{index}"].astype(np.float64)
        - reference[f"latent_mean_{index}"].astype(np.float64)
    )[mask]
    delta_speed = (
        candidate[f"raw_mean_{index}"].astype(np.float64)
        - reference[f"raw_mean_{index}"].astype(np.float64)
    )[mask][:, 1]
    kl = float(np.mean(0.5 * (delta_latent / STEER_STD) ** 2 + 0.5 * (delta_speed / SPEED_STD) ** 2))
    gradient = vector_metrics(reference[f"gradient_{index}"], candidate[f"gradient_{index}"])
    loss_reference = policy_loss(
        {name: reference[f"{name}_{index}"] for name in ("mask", "advantages", "old_log_prob")},
        reference[f"log_prob_{index}"],
    )
    loss_candidate = policy_loss(
        {name: candidate[f"{name}_{index}"] for name in ("mask", "advantages", "old_log_prob")},
        candidate[f"log_prob_{index}"],
    )
    loss_abs = abs(loss_candidate - loss_reference)
    result["minibatches"].append(
        {
            "policy_kl": kl,
            "gradient": gradient,
            "policy_loss": {"reference": loss_reference, "candidate": loss_candidate, "abs": loss_abs},
        }
    )
    worst["policy_kl"] = max(worst["policy_kl"], kl)
    worst["gradient_cosine"] = min(worst["gradient_cosine"], gradient["cosine"])
    worst["gradient_relative_l2"] = max(worst["gradient_relative_l2"], gradient["relative_l2"])
    worst["policy_loss_abs"] = max(worst["policy_loss_abs"], loss_abs)

delta_reference = reference["parameters_after_u1"].astype(np.float64) - reference["parameters_before"].astype(np.float64)
delta_candidate = candidate["parameters_after_u1"].astype(np.float64) - candidate["parameters_before"].astype(np.float64)
assert np.array_equal(reference["parameters_before"], candidate["parameters_before"])
delta = vector_metrics(delta_reference, delta_candidate)
result["u1_parameter_delta"] = delta

gates = {
    "policy_kl_le_1e-8": worst["policy_kl"] <= 1e-8,
    "gradient_cosine_ge_0.999999": worst["gradient_cosine"] >= 0.999999,
    "delta_cosine_ge_0.999999": delta["cosine"] >= 0.999999,
    "gradient_relative_l2_le_1e-4": worst["gradient_relative_l2"] <= 1e-4,
    "delta_relative_l2_le_1e-4": delta["relative_l2"] <= 1e-4,
    "policy_loss_abs_le_1e-6": worst["policy_loss_abs"] <= 1e-6,
    "u1_rollout_identical": bool(result["u1_rollout_identical"]),
}
result["worst"] = worst
result["gates"] = gates
result["all_gates_pass"] = all(gates.values())

(DIRECTORY / "TWO_UPDATE_COMPARISON.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({"worst": worst, "u1_parameter_delta": delta, "gates": gates,
                  "u1_rollout_identical": result["u1_rollout_identical"],
                  "u2_rollout_identical": result["u2_rollout_identical"]}, indent=2))

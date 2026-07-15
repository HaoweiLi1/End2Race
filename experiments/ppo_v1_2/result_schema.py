"""Strict run and evaluation result validation for PPO V1.2."""

from __future__ import annotations

import math
from typing import Any

from .config_schema import LEGAL_STATUSES
from .experiment_spec import canonical_hash
from .selectors import checkpoint_flags


def validate_evaluation(metrics: dict[str, Any]) -> dict[str, Any]:
    required = ("ego_collision", "follow", "overtake", "opponent_only_collision", "error", "total")
    if any(key not in metrics for key in required):
        raise ValueError("Evaluation metrics omit required fields")
    if any(not isinstance(metrics[key], int) or metrics[key] < 0 for key in required):
        raise ValueError("Evaluation counts must be non-negative integers")
    if metrics["total"] != 600 or metrics["error"] != 0:
        raise ValueError(f"Evaluation is not a complete error-free 600-case panel: {metrics}")
    if metrics["ego_collision"] + metrics["follow"] + metrics["overtake"] != 600:
        raise ValueError("Evaluation outcomes must be mutually exclusive and sum to 600")
    numeric_values = [value for value in metrics.values() if isinstance(value, (int, float))]
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("Evaluation metrics contain non-finite values")
    expected_flags = checkpoint_flags(metrics)
    for key, expected in expected_flags.items():
        if key in metrics and bool(metrics[key]) != expected:
            raise ValueError(f"Incorrect checkpoint classification field {key}")
    return {**metrics, **expected_flags}


def validate_run_result(result: dict[str, Any], manifest_arm: dict[str, Any]) -> None:
    if result.get("arm_id") != manifest_arm.get("arm_id"):
        raise ValueError("run_result arm_id differs from manifest")
    if result.get("status") not in LEGAL_STATUSES:
        raise ValueError("run_result contains an illegal status")
    if result.get("config_hash") != manifest_arm.get("config_hash"):
        raise ValueError("run_result config hash differs from manifest")
    if canonical_hash(result["resolved_config"]) != result["config_hash"]:
        raise ValueError("run_result resolved config hash is invalid")
    if not 1 <= int(result.get("attempt", 0)) <= 2:
        raise ValueError("run_result attempt must be 1 or 2")
    if result["status"] == "COMPLETED":
        checkpoints = result.get("checkpoints", [])
        if not checkpoints:
            raise ValueError("A completed arm must contain checkpoint evaluations")
        for checkpoint in checkpoints:
            checkpoint["metrics"] = validate_evaluation(checkpoint["metrics"])

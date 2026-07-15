"""Only the preregistered PPO V1.2 selection tuples."""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence


def checkpoint_flags(metrics: dict[str, Any]) -> dict[str, bool]:
    collision = int(metrics["ego_collision"])
    overtake = int(metrics["overtake"])
    return {
        "eligible": overtake >= 329,
        "beats_bc_collision": collision < 21,
        "keeps_or_beats_bc_overtake": overtake >= 346,
        "beats_bc_both": collision < 21 and overtake >= 346,
        "beats_v1_best": collision < 17 or (collision == 17 and overtake > 347),
        "beats_v1_1_best": collision < 15 or (collision == 15 and overtake > 353),
        "dominates_v1_1_best": collision <= 15 and overtake >= 353 and (collision < 15 or overtake > 353),
    }


def select_checkpoint(checkpoints: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = []
    for checkpoint in checkpoints:
        metrics = checkpoint["metrics"]
        flags = checkpoint_flags(metrics)
        if bool(checkpoint.get("valid", True)) and flags["eligible"]:
            eligible.append({**checkpoint, "metrics": {**metrics, **flags}})
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (int(row["metrics"]["ego_collision"]), -int(row["metrics"]["overtake"]), int(row["update"])),
    )


def arm_rank_tuple(result: dict[str, Any]) -> tuple[Any, ...]:
    selected = result.get("selected_checkpoint")
    evaluations = [row["metrics"] for row in result.get("checkpoints", []) if row.get("valid", True)]
    if selected is None:
        selected_collision, selected_overtake = math.inf, -math.inf
    else:
        selected_collision = int(selected["metrics"]["ego_collision"])
        selected_overtake = int(selected["metrics"]["overtake"])
    median_collision = statistics.median([int(row["ego_collision"]) for row in evaluations]) if evaluations else math.inf
    median_overtake = statistics.median([int(row["overtake"]) for row in evaluations]) if evaluations else -math.inf
    return (
        int(selected is None),
        selected_collision,
        -selected_overtake,
        median_collision,
        -median_overtake,
        str(result["arm_id"]),
    )


def rank_arms(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in results if row.get("status") == "COMPLETED"]
    return sorted(completed, key=arm_rank_tuple)


def select_top(results: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ranked = [row for row in rank_arms(results) if row.get("selected_checkpoint") is not None]
    return ranked[:count]

"""Deterministic stage/global aggregation for PPO V1.2."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence

from utils import atomic_write_json

from .config_schema import STAGES
from .selectors import arm_rank_tuple, rank_arms, select_top


TOP_COUNTS = {"C": 2, "H": 2, "B": 2, "R": 1, "K": 2, "E": 1, "G": 1, "W": 1, "X": 3, "S": 0}


def _write_tsv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    temporary.replace(path)


def _flat_result(result: dict[str, Any]) -> dict[str, Any]:
    selected = result.get("selected_checkpoint")
    metrics = selected.get("metrics", {}) if selected else {}
    return {
        "arm_id": result["arm_id"],
        "stage": result["stage"],
        "status": result["status"],
        "attempt": result.get("attempt"),
        "config_hash": result.get("config_hash"),
        "selected_checkpoint_missing": int(selected is None),
        "selected_update": selected.get("update") if selected else None,
        "selected_checkpoint": selected.get("checkpoint") if selected else None,
        "selected_ego_collision": metrics.get("ego_collision"),
        "selected_follow": metrics.get("follow"),
        "selected_overtake": metrics.get("overtake"),
        "selected_opponent_only_collision": metrics.get("opponent_only_collision"),
        "actual_optimizer_steps": result.get("actual_optimizer_steps"),
    }


def stage_aggregate(stage: str, results: Sequence[dict[str, Any]], stage_dir: Path) -> dict[str, Any]:
    if any(result.get("stage") != stage for result in results):
        raise ValueError(f"Non-{stage} result supplied to stage aggregator")
    flattened = [_flat_result(result) for result in sorted(results, key=lambda row: row["arm_id"])]
    atomic_write_json(stage_dir / "stage_results.json", list(results))
    _write_tsv(stage_dir / "stage_results.tsv", flattened, tuple(flattened[0]) if flattened else ("arm_id", "status"))
    ranked = rank_arms(results)
    rank_rows = []
    for index, result in enumerate(ranked, start=1):
        row = _flat_result(result)
        rank_tuple = arm_rank_tuple(result)
        row.update(
            rank=index,
            median_eval_ego_collision=None if math.isinf(float(rank_tuple[3])) else rank_tuple[3],
            median_eval_overtake=None if math.isinf(float(-rank_tuple[4])) else -rank_tuple[4],
        )
        rank_rows.append(row)
    atomic_write_json(stage_dir / "stage_rank.json", rank_rows)
    selected_results = select_top(results, TOP_COUNTS[stage])
    selection = {
        "stage": stage,
        "selector": "(selected_checkpoint_missing, selected_ego_collision, -selected_overtake, median_eval_ego_collision, -median_eval_overtake, arm_id)",
        "selected_arm_ids": [row["arm_id"] for row in selected_results],
        "selected": [
            {"rank": index, "arm_id": row["arm_id"], "resolved_config": row["resolved_config"], "config_hash": row["config_hash"]}
            for index, row in enumerate(selected_results, start=1)
        ],
    }
    atomic_write_json(stage_dir / "stage_selection.json", selection)
    failures = [_flat_result(row) for row in results if row.get("status") != "COMPLETED"]
    _write_tsv(stage_dir / "stage_failures.tsv", failures, ("arm_id", "stage", "status", "attempt", "config_hash"))
    return selection


def repeatability_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    for result in results:
        if result.get("stage") == "S" and result.get("status") == "COMPLETED":
            grouped[int(result["metadata"]["x_rank"])].append(result)
    rows: list[dict[str, Any]] = []
    for rank, group in grouped.items():
        if not group:
            continue
        collisions = [int(row["selected_checkpoint"]["metrics"]["ego_collision"]) for row in group]
        overtakes = [int(row["selected_checkpoint"]["metrics"]["overtake"]) for row in group]
        rows.append(
            {
                "x_rank": rank,
                "seed_count": len(group),
                "best_collision_per_seed": json.dumps(collisions, separators=(",", ":")),
                "best_overtake_per_seed": json.dumps(overtakes, separators=(",", ":")),
                "median_collision": statistics.median(collisions),
                "mean_collision": statistics.mean(collisions),
                "collision_std": statistics.pstdev(collisions),
                "median_overtake": statistics.median(overtakes),
                "mean_overtake": statistics.mean(overtakes),
                "overtake_std": statistics.pstdev(overtakes),
                "beats_bc_both_seed_count": sum(bool(row["selected_checkpoint"]["metrics"]["beats_bc_both"]) for row in group),
                "beats_v1_1_best_seed_count": sum(bool(row["selected_checkpoint"]["metrics"]["beats_v1_1_best"]) for row in group),
            }
        )
    return rows


def global_aggregate(root: Path, results: Sequence[dict[str, Any]], selections: dict[str, Any]) -> dict[str, Any]:
    flattened = [_flat_result(row) for row in sorted(results, key=lambda item: item["arm_id"])]
    _write_tsv(root / "GLOBAL_RUNS.tsv", flattened, tuple(flattened[0]) if flattened else ("arm_id", "status"))
    checkpoints = []
    for result in results:
        for checkpoint in result.get("checkpoints", []):
            checkpoints.append(
                {
                    "arm_id": result["arm_id"],
                    "stage": result["stage"],
                    "update": checkpoint["update"],
                    "checkpoint": checkpoint.get("checkpoint"),
                    **checkpoint["metrics"],
                }
            )
    checkpoint_columns = tuple(checkpoints[0]) if checkpoints else ("arm_id", "stage", "update")
    _write_tsv(root / "GLOBAL_CHECKPOINTS.tsv", checkpoints, checkpoint_columns)
    failures = [row for row in flattened if row["status"] != "COMPLETED"]
    _write_tsv(root / "GLOBAL_FAILURES.tsv", failures, tuple(flattened[0]) if flattened else ("arm_id", "status"))
    atomic_write_json(root / "GLOBAL_SELECTIONS.json", selections)
    repeats = repeatability_rows(results)
    _write_tsv(root / "FINAL_REPEATABILITY.tsv", repeats, tuple(repeats[0]) if repeats else ("x_rank", "seed_count"))
    status_counts = {status: sum(row["status"] == status for row in results) for status in sorted({row["status"] for row in results})}
    completion = {
        "training_arm_count": 125,
        "terminal_arm_count": len(results),
        "status_counts": status_counts,
        "all_arms_terminal": len(results) == 125 and not any(row["status"] in {"PENDING", "RUNNING"} for row in results),
        "stage_selection_files": {stage: f"{stage}/stage_selection.json" for stage in STAGES},
    }
    atomic_write_json(root / "EXPERIMENT_COMPLETION.json", completion)
    return completion

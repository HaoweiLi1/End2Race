"""Single-candidate 288-row opened-development evaluation for B7."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from bplus_v22.b4_direct import load_strict_plain_actor
from bplus_v22.b4_eval import B4DeterministicActor
from bplus_v22.ppo_eval import physical_shard_rows, validate_task8_rows


B7_EVAL_SHARD_SCHEMA = "end2race-b7-eval-shard-1"
B7_EVAL_MERGE_SCHEMA = "end2race-b7-eval-merge-1"
B7_VARIANT = "seed1_iter10"
EXPECTED_SCENARIOS = 288
EXPECTED_BC_COLLISION = 24
EXPECTED_BC_OVERTAKE = 138
SHARD_COUNT = 4


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid canonical bool text: {value!r}")


def load_immutable_bc_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        raw = [row for row in csv.DictReader(handle, delimiter="\t") if row["variant"] == "BC"]
    rows: dict[str, dict[str, Any]] = {}
    for row in raw:
        index = int(row["task8_row_index"])
        if index in rows:
            raise ValueError("B7 immutable BC rows are duplicated")
        rows[index] = {
            "task8_row_index": index,
            "l2_id": row["l2_id"],
            "l4_id": row["l4_id"],
            "manifest_order": row["manifest_order"],
            "collision_any": _bool(row["collision_any"]),
            "terminal_overtake": _bool(row["terminal_overtake"]),
            "four_state": row["four_state"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "scenario_manifest_sha256": row["scenario_manifest_sha256"],
            "trajectory_sha256": row["trajectory_sha256"],
        }
    if set(rows) != set(range(EXPECTED_SCENARIOS)):
        raise ValueError("B7 immutable BC row inventory is not exactly 288")
    if sum(row["collision_any"] for row in rows.values()) != EXPECTED_BC_COLLISION:
        raise ValueError("B7 immutable BC collision count drift")
    if sum(row["terminal_overtake"] for row in rows.values()) != EXPECTED_BC_OVERTAKE:
        raise ValueError("B7 immutable BC overtake count drift")
    return rows


@dataclass(frozen=True)
class B7EvaluationShard:
    shard_index: int
    shard_count: int
    candidate_checkpoint_sha256: str
    training_run_plan_sha256: str
    baseline_rows_sha256: str
    rows: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = B7_EVAL_SHARD_SCHEMA
        value["rows"] = list(self.rows)
        return value


def evaluate_candidate_shard(
    *,
    task8_rows: Sequence[Mapping[str, str]],
    baseline_rows_path: str | Path,
    candidate_path: str | Path,
    training_run_plan_sha256: str,
    device: torch.device,
    shard_index: int,
    shard_count: int = SHARD_COUNT,
) -> B7EvaluationShard:
    from d25.oracle import simulate_episode
    from d25.search import trajectory_digest

    validate_task8_rows(task8_rows)
    if int(shard_count) != SHARD_COUNT or not 0 <= int(shard_index) < SHARD_COUNT:
        raise ValueError("B7 evaluation shard topology drift")
    baseline = load_immutable_bc_rows(baseline_rows_path)
    candidate_sha = file_sha256(candidate_path)
    actor = load_strict_plain_actor(candidate_path, device)
    rows: list[dict[str, Any]] = []
    for physical_index, case in physical_shard_rows(task8_rows, shard_index, shard_count):
        before = baseline[physical_index]
        if (
            before["l2_id"] != case["l2_id"]
            or before["l4_id"] != case["l4_id"]
            or before["manifest_order"] != case["manifest_order"]
        ):
            raise ValueError("B7 immutable BC row/scenario identity drift")
        adapter = B4DeterministicActor(actor)
        result = simulate_episode(adapter, device, case)
        accounting = adapter.accounting()
        collision = bool(result.outcome.collision_any)
        overtake = result.outcome.corrected_outcome3 == "overtake"
        rows.append(
            {
                "task8_row_index": physical_index,
                "manifest_order": str(case["manifest_order"]),
                "l2_id": str(case["l2_id"]),
                "l4_id": str(case["l4_id"]),
                "map_name": str(case["map_name"]),
                "skill": str(case["skill"]),
                "opponent_raceline": str(case["opponent_raceline"]),
                "speedscale_hex": str(case["speedscale_hex"]),
                "variant": B7_VARIANT,
                "collision_any": collision,
                "terminal_overtake": overtake,
                "four_state": str(result.outcome.four_state),
                "ego_collision": bool(result.outcome.ego_collision),
                "opp_collision": bool(result.outcome.opp_collision),
                "fixed_collision": bool(before["collision_any"]) and not collision,
                "new_collision": not bool(before["collision_any"]) and collision,
                "gained_overtake": not bool(before["terminal_overtake"]) and overtake,
                "lost_overtake": bool(before["terminal_overtake"]) and not overtake,
                "bc_collision_any": bool(before["collision_any"]),
                "bc_terminal_overtake": bool(before["terminal_overtake"]),
                "bc_four_state": str(before["four_state"]),
                "candidate_checkpoint_sha256": candidate_sha,
                "training_run_plan_sha256": training_run_plan_sha256,
                "trajectory_sha256": trajectory_digest(result.arrays),
                **accounting,
            }
        )
    return B7EvaluationShard(
        shard_index=int(shard_index),
        shard_count=int(shard_count),
        candidate_checkpoint_sha256=candidate_sha,
        training_run_plan_sha256=training_run_plan_sha256,
        baseline_rows_sha256=file_sha256(baseline_rows_path),
        rows=tuple(rows),
    )


def exact_cluster_signflip_one_sided(effects: Sequence[int]) -> float:
    """Exact conditional one-sided sign-flip test over nonzero L4 effects."""

    nonzero = [abs(int(value)) for value in effects if int(value) != 0]
    observed = sum(int(value) for value in effects)
    if not nonzero:
        return 1.0
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in nonzero:
        updated: Counter[int] = Counter()
        for total, count in distribution.items():
            updated[total + magnitude] += count
            updated[total - magnitude] += count
        distribution = updated
    favorable = sum(count for total, count in distribution.items() if total >= observed)
    return favorable / sum(distribution.values())


def merge_candidate_shards(
    *,
    shards: Sequence[B7EvaluationShard],
    task8_rows: Sequence[Mapping[str, str]],
    baseline_rows_path: str | Path,
    candidate_path: str | Path,
    training_run_plan_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_task8_rows(task8_rows)
    if len(shards) != SHARD_COUNT or {row.shard_index for row in shards} != set(range(SHARD_COUNT)):
        raise ValueError("B7 evaluation shard inventory is incomplete")
    candidate_sha = file_sha256(candidate_path)
    baseline_sha = file_sha256(baseline_rows_path)
    rows: list[dict[str, Any]] = []
    for shard in shards:
        if (
            shard.shard_count != SHARD_COUNT
            or shard.candidate_checkpoint_sha256 != candidate_sha
            or shard.training_run_plan_sha256 != training_run_plan_sha256
            or shard.baseline_rows_sha256 != baseline_sha
        ):
            raise ValueError("B7 evaluation shard provenance drift")
        for row in shard.rows:
            if int(row["task8_row_index"]) % SHARD_COUNT != shard.shard_index:
                raise ValueError("B7 evaluation row came from the wrong shard")
            rows.append(dict(row))
    if len(rows) != EXPECTED_SCENARIOS or {
        int(row["task8_row_index"]) for row in rows
    } != set(range(EXPECTED_SCENARIOS)):
        raise ValueError("B7 candidate evaluation is not exactly 288 unique rows")
    baseline = load_immutable_bc_rows(baseline_rows_path)
    for row in rows:
        before = baseline[int(row["task8_row_index"])]
        expected = {
            "fixed_collision": before["collision_any"] and not row["collision_any"],
            "new_collision": not before["collision_any"] and row["collision_any"],
            "gained_overtake": not before["terminal_overtake"] and row["terminal_overtake"],
            "lost_overtake": before["terminal_overtake"] and not row["terminal_overtake"],
        }
        if any(bool(row[name]) != bool(value) for name, value in expected.items()):
            raise ValueError("B7 paired transition diagnostic drift")
    collision = sum(bool(row["collision_any"]) for row in rows)
    overtake = sum(bool(row["terminal_overtake"]) for row in rows)
    fixed = sum(bool(row["fixed_collision"]) for row in rows)
    new = sum(bool(row["new_collision"]) for row in rows)
    gained = sum(bool(row["gained_overtake"]) for row in rows)
    lost = sum(bool(row["lost_overtake"]) for row in rows)
    speed_projection = sum(int(row["deterministic_speed_projection_count"]) for row in rows)
    by_l4: dict[str, int] = {}
    for row in rows:
        by_l4.setdefault(str(row["l4_id"]), 0)
        by_l4[str(row["l4_id"])] += int(bool(row["fixed_collision"]))
        by_l4[str(row["l4_id"])] -= int(bool(row["new_collision"]))
    cluster_p = exact_cluster_signflip_one_sided(list(by_l4.values()))
    checks = {
        "overtake_ge_132": overtake >= 132,
        "fixed_minus_new_ge_6": fixed - new >= 6,
        "collision_le_18": collision <= 18,
        "l4_cluster_signflip_one_sided_le_0_10": cluster_p <= 0.10,
        "zero_deterministic_speed_projection": speed_projection == 0,
    }
    pass_gate = all(checks.values())
    rows.sort(key=lambda row: int(row["task8_row_index"]))
    summary = {
        "schema": B7_EVAL_MERGE_SCHEMA,
        "integrity_passed": True,
        "panel_status": "opened-development 288 regression panel",
        "fresh_or_final_confirmation": False,
        "scenario_count": EXPECTED_SCENARIOS,
        "variant_count": 1,
        "candidate_checkpoint_sha256": candidate_sha,
        "training_run_plan_sha256": training_run_plan_sha256,
        "baseline_rows_sha256": baseline_sha,
        "bc": {
            "collision": EXPECTED_BC_COLLISION,
            "terminal_overtake": EXPECTED_BC_OVERTAKE,
        },
        "candidate": {
            "collision": collision,
            "terminal_overtake": overtake,
            "fixed_collision": fixed,
            "new_collision": new,
            "fixed_minus_new": fixed - new,
            "gained_overtake": gained,
            "lost_overtake": lost,
            "deterministic_speed_projection_count": speed_projection,
            "l4_cluster_count": len(by_l4),
            "l4_cluster_signflip_one_sided_p": cluster_p,
        },
        "checks": checks,
        "seed1_minimum_continue_gate_pass": pass_gate,
        "opened_development_target_collision_le_16": pass_gate and collision <= 16,
        "verdict": "B7_SEED1_CONTINUE" if pass_gate else "B7_SEED1_SUBSTANTIVE_NEGATIVE",
        "seed0_automatically_started": False,
        "austin600_opened": False,
        "sealed_pool_opened": False,
    }
    return rows, summary

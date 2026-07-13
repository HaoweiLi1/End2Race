#!/usr/bin/env python3
"""Exact 288x4 Cartesian merge and seed-1 snapshot selection regression."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from bplus_v22.b4_eval import (
    B4CheckpointSpec,
    B4EvaluationShard,
    B4_VARIANT_BC,
    b4_variant,
    merge_shards,
    paired_row,
)
from bplus_v22.ppo_eval import read_task8_development


REPO = Path(__file__).resolve().parent.parent
DEVELOPMENT = (
    REPO
    / "Experiments/B1_route_r2_scaffold/artifacts/task8_manifests_20260712_113241/development_scenarios.tsv"
)


def sha_bytes(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class Outcome:
    four_state: str
    collision_any: bool
    ego_collision: bool
    opp_collision: bool
    corrected_outcome3: str
    confirmed_safe_pass: bool
    interaction_attempt: bool


def outcome(collision: bool, overtake: bool) -> Outcome:
    if collision:
        return Outcome("collision", True, True, False, "collision", False, True)
    if overtake:
        return Outcome(
            "terminal_overtake_only", False, False, False, "overtake", False, True
        )
    return Outcome("safe_follow", False, False, False, "follow", False, False)


def main() -> None:
    manifest_sha = hashlib.sha256(DEVELOPMENT.read_bytes()).hexdigest()
    cases = read_task8_development(DEVELOPMENT, manifest_sha)
    checkpoint_manifest_sha = "a" * 64
    bc_sha = "b" * 64
    training_sha = "c" * 64
    parent_sha = "d" * 64
    specs = tuple(
        B4CheckpointSpec(
            seed=seed,
            iteration=iteration,
            checkpoint_path=f"/unused/seed{seed}_iter{iteration}.pth",
            checkpoint_sha256=sha_bytes(f"seed{seed}-iter{iteration}"),
            training_manifest_sha256=training_sha,
            training_run_plan_sha256=parent_sha,
        )
        for seed in (1,)
        for iteration in (10, 20, 30)
    )
    spec_by_variant = {spec.variant: spec for spec in specs}
    inventory = {B4_VARIANT_BC: bc_sha} | {
        spec.variant: spec.checkpoint_sha256 for spec in specs
    }
    baseline_collision = set(range(24))
    baseline_overtake = set(range(24, 162))
    contracts = {
        (1, 10): (21, 135, 5, 2),
        (1, 20): (16, 133, 9, 1),
        (1, 30): (15, 131, 10, 1),
    }
    rows_by_shard: list[list[dict[str, object]]] = [[] for _ in range(4)]
    for index, case in enumerate(cases):
        baseline = outcome(index in baseline_collision, index in baseline_overtake)
        rows_by_shard[index % 4].append(
            paired_row(
                physical_index=index,
                case=case,
                variant=B4_VARIANT_BC,
                seed=-1,
                iteration=0,
                checkpoint_sha256=bc_sha,
                scenario_manifest_sha256=manifest_sha,
                checkpoint_manifest_sha256=checkpoint_manifest_sha,
                outcome=baseline,
                baseline_outcome=baseline,
                trajectory_sha256=sha_bytes(f"BC:{index}"),
                accounting=None,
            )
        )
        for (seed, iteration), (collision_count, overtake_count, fixed, new) in contracts.items():
            fixed_indices = set(range(fixed))
            new_indices = set(range(200, 200 + new))
            retained_needed = collision_count - new
            collision_indices = set(range(fixed, fixed + retained_needed)) | new_indices
            overtake_indices = set(range(24, 24 + overtake_count))
            candidate = outcome(index in collision_indices, index in overtake_indices)
            variant = b4_variant(seed, iteration)
            rows_by_shard[index % 4].append(
                paired_row(
                    physical_index=index,
                    case=case,
                    variant=variant,
                    seed=seed,
                    iteration=iteration,
                    checkpoint_sha256=spec_by_variant[variant].checkpoint_sha256,
                    scenario_manifest_sha256=manifest_sha,
                    checkpoint_manifest_sha256=checkpoint_manifest_sha,
                    outcome=candidate,
                    baseline_outcome=baseline,
                    trajectory_sha256=sha_bytes(f"{variant}:{index}"),
                    accounting={
                        "deterministic_steps": 800,
                        "deterministic_speed_projection_count": 0,
                        "deterministic_steer_projection_count": 0,
                        "max_abs_deterministic_speed_projection_delta": 0.0,
                        "max_abs_deterministic_steer_projection_delta": 0.0,
                    },
                )
            )
    shards = tuple(
        B4EvaluationShard(
            shard_index=index,
            shard_count=4,
            scenario_manifest_sha256=manifest_sha,
            checkpoint_manifest_sha256=checkpoint_manifest_sha,
            bc_checkpoint_sha256=bc_sha,
            checkpoint_sha256_by_variant=inventory,
            rows=tuple(rows),
        )
        for index, rows in enumerate(rows_by_shard)
    )
    merged, summary = merge_shards(
        shards=shards,
        task8_rows=cases,
        scenario_manifest_sha256=manifest_sha,
        checkpoint_manifest_sha256=checkpoint_manifest_sha,
        bc_checkpoint_sha256=bc_sha,
        checkpoints=specs,
    )
    assert len(merged) == 1152
    assert summary["scenario_count"] == 288
    assert summary["variant_count"] == 4
    assert summary["selected_iteration"] == 20
    assert summary["same_iteration_snapshots"]["iter20"]["terminal_overtake"] == 133
    assert summary["same_iteration_snapshots"]["iter20"]["collision"] == 16
    assert summary["same_iteration_snapshots"]["iter20"]["checks"] == {
        "seed1_overtake_ge_132": True,
        "seed1_collision_le_24": True,
        "seed1_collision_strict_improve": True,
        "fixed_gt_new": True,
        "zero_deterministic_speed_projection": True,
    }
    assert summary["same_iteration_snapshots"]["iter30"]["feasible"] is False

    broken_rows = [list(shard.rows) for shard in shards]
    broken_rows[0][0] = dict(broken_rows[0][0])
    broken_rows[0][0]["l2_id"] = "L2:" + "f" * 64
    broken = tuple(
        B4EvaluationShard(
            shard_index=index,
            shard_count=4,
            scenario_manifest_sha256=manifest_sha,
            checkpoint_manifest_sha256=checkpoint_manifest_sha,
            bc_checkpoint_sha256=bc_sha,
            checkpoint_sha256_by_variant=inventory,
            rows=tuple(rows),
        )
        for index, rows in enumerate(broken_rows)
    )
    try:
        merge_shards(
            shards=broken,
            task8_rows=cases,
            scenario_manifest_sha256=manifest_sha,
            checkpoint_manifest_sha256=checkpoint_manifest_sha,
            bc_checkpoint_sha256=bc_sha,
            checkpoints=specs,
        )
        raise RuntimeError("B4 merge accepted a broken paired L2 identity")
    except ValueError as error:
        assert "identity" in str(error) or "baseline" in str(error)

    diagnostic_rows = [list(shard.rows) for shard in shards]
    candidate_index = next(
        index
        for index, row in enumerate(diagnostic_rows[0])
        if row["variant"] != B4_VARIANT_BC
    )
    diagnostic_rows[0][candidate_index] = dict(
        diagnostic_rows[0][candidate_index]
    )
    diagnostic_rows[0][candidate_index]["fixed_collision"] = not bool(
        diagnostic_rows[0][candidate_index]["fixed_collision"]
    )
    diagnostic_broken = tuple(
        B4EvaluationShard(
            shard_index=index,
            shard_count=4,
            scenario_manifest_sha256=manifest_sha,
            checkpoint_manifest_sha256=checkpoint_manifest_sha,
            bc_checkpoint_sha256=bc_sha,
            checkpoint_sha256_by_variant=inventory,
            rows=tuple(rows),
        )
        for index, rows in enumerate(diagnostic_rows)
    )
    try:
        merge_shards(
            shards=diagnostic_broken,
            task8_rows=cases,
            scenario_manifest_sha256=manifest_sha,
            checkpoint_manifest_sha256=checkpoint_manifest_sha,
            bc_checkpoint_sha256=bc_sha,
            checkpoints=specs,
        )
        raise RuntimeError("B4 merge accepted a corrupted paired diagnostic")
    except ValueError as error:
        assert "diagnostic" in str(error)

    print("B4 seed-1 288x4 paired evaluation contracts passed")


if __name__ == "__main__":
    main()

"""Frozen 288x7 paired evaluator and same-iteration B4 selection contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from bplus_v22.b4_direct import load_strict_plain_actor
from bplus_v22.ppo_eval import (
    EXPECTED_BC_COLLISIONS,
    EXPECTED_BC_OVERTAKES,
    EXPECTED_SCENARIOS,
    physical_shard_rows,
    validate_task8_rows,
)


B4_EVAL_SHARD_SCHEMA = "end2race-b4-eval-shard-1"
B4_EVAL_ROW_SCHEMA = "end2race-b4-eval-row-1"
B4_EVAL_MERGE_SCHEMA = "end2race-b4-eval-merge-1"
B4_VARIANT_BC = "BC"
B4_SEEDS = (0, 1)
B4_ITERATIONS = (10, 20, 30)
B4_CANDIDATE_COUNT = len(B4_SEEDS) * len(B4_ITERATIONS)
B4_VARIANT_COUNT = 1 + B4_CANDIDATE_COUNT
B4_EXPECTED_RESULTS = EXPECTED_SCENARIOS * B4_VARIANT_COUNT
B4_OVERTAKE_GATE = 132
B4_COLLISION_FEASIBILITY = 24
B4_COLLISION_PRODUCT_PER_SEED = 16
B4_COLLISION_PRODUCT_POOLED = 33


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def b4_variant(seed: int, iteration: int) -> str:
    if int(seed) not in B4_SEEDS or int(iteration) not in B4_ITERATIONS:
        raise ValueError("B4 evaluation seed/iteration is invalid")
    return f"seed{int(seed)}_iter{int(iteration)}"


@dataclass(frozen=True)
class B4CheckpointSpec:
    seed: int
    iteration: int
    checkpoint_path: str
    checkpoint_sha256: str
    training_manifest_sha256: str
    training_run_plan_sha256: str

    def __post_init__(self) -> None:
        b4_variant(self.seed, self.iteration)
        if not self.checkpoint_path:
            raise ValueError("B4 evaluation checkpoint path is empty")
        if not all(
            _is_sha256(value)
            for value in (
                self.checkpoint_sha256,
                self.training_manifest_sha256,
                self.training_run_plan_sha256,
            )
        ):
            raise ValueError("B4 evaluation checkpoint identity is invalid")

    @property
    def variant(self) -> str:
        return b4_variant(self.seed, self.iteration)


def validate_checkpoint_specs(
    specs: Sequence[B4CheckpointSpec], checkpoint_manifest_sha256: str
) -> tuple[B4CheckpointSpec, ...]:
    if not _is_sha256(checkpoint_manifest_sha256):
        raise ValueError("B4 checkpoint manifest SHA256 is invalid")
    expected = {(seed, iteration) for seed in B4_SEEDS for iteration in B4_ITERATIONS}
    observed = {(int(spec.seed), int(spec.iteration)) for spec in specs}
    if len(specs) != B4_CANDIDATE_COUNT or observed != expected:
        raise ValueError("B4 evaluation requires exactly six seed/iteration snapshots")
    if len({spec.variant for spec in specs}) != B4_CANDIDATE_COUNT:
        raise ValueError("B4 evaluation variant identity is duplicated")
    if len({spec.training_manifest_sha256 for spec in specs}) != 1:
        raise ValueError("B4 evaluation snapshots use different training manifests")
    if len({spec.training_run_plan_sha256 for spec in specs}) != 1:
        raise ValueError("B4 evaluation snapshots use different training RunPlans")
    return tuple(sorted(specs, key=lambda spec: (spec.iteration, spec.seed)))


class B4DeterministicActor(nn.Module):
    """Plain actor adapter that records deterministic projection diagnostics."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor
        self.reset_runtime()

    @property
    def gru(self):
        return self.actor.gru

    def reset_runtime(self) -> None:
        self.steps = 0
        self.speed_projection_count = 0
        self.steer_projection_count = 0
        self.max_abs_speed_projection_delta = 0.0
        self.max_abs_steer_projection_delta = 0.0
        self._requested: tuple[float, float] | None = None

    def forward(self, lidar, previous_speed, hidden):
        action, next_hidden = self.actor(lidar, previous_speed, hidden)
        requested = action[:, -1, :]
        if requested.shape != (1, 2) or not torch.all(torch.isfinite(requested)):
            raise ValueError("B4 deterministic actor output is invalid")
        self._requested = (float(requested[0, 0].item()), float(requested[0, 1].item()))
        return action, next_hidden

    def observe_applied_command(self, steer: float, speed: float) -> None:
        if self._requested is None:
            raise RuntimeError("B4 evaluator applied a command without a model request")
        raw_steer, raw_speed = self._requested
        projected_steer = float(np.clip(raw_steer, -0.52, 0.52))
        projected_speed = float(np.clip(raw_speed, 0.0, 20.0))
        steer_delta = abs(projected_steer - raw_steer)
        speed_delta = abs(projected_speed - raw_speed)
        self.steer_projection_count += int(steer_delta > 0.0)
        self.speed_projection_count += int(speed_delta > 0.0)
        self.max_abs_steer_projection_delta = max(
            self.max_abs_steer_projection_delta, steer_delta
        )
        self.max_abs_speed_projection_delta = max(
            self.max_abs_speed_projection_delta, speed_delta
        )
        # The helper simulator clamps negative speed, while the original BC
        # evaluator does not.  B4 promotion forbids all such speed cases, so a
        # nonzero count is retained as an integrity failure at merge time.
        if float(steer) != projected_steer:
            raise AssertionError("B4 evaluator steering projection contract drift")
        if raw_speed >= 0.0 and float(speed) != raw_speed:
            raise AssertionError("B4 evaluator changed an in-range requested speed")
        self.steps += 1
        self._requested = None

    def accounting(self) -> dict[str, int | float]:
        if self._requested is not None or self.steps <= 0:
            raise RuntimeError("B4 evaluator command accounting is incomplete")
        return {
            "deterministic_steps": self.steps,
            "deterministic_speed_projection_count": self.speed_projection_count,
            "deterministic_steer_projection_count": self.steer_projection_count,
            "max_abs_deterministic_speed_projection_delta": self.max_abs_speed_projection_delta,
            "max_abs_deterministic_steer_projection_delta": self.max_abs_steer_projection_delta,
        }


def _outcome(outcome) -> dict[str, object]:
    return {
        "four_state": str(outcome.four_state),
        "collision_any": bool(outcome.collision_any),
        "ego_collision": bool(outcome.ego_collision),
        "opp_collision": bool(
            getattr(
                outcome,
                "opp_collision",
                bool(outcome.collision_any) and not bool(outcome.ego_collision),
            )
        ),
        "terminal_overtake": outcome.corrected_outcome3 == "overtake",
        "confirmed_safe_pass": outcome.confirmed_safe_pass is True,
        "interaction_attempt": outcome.interaction_attempt is True,
    }


def paired_row(
    *,
    physical_index: int,
    case: Mapping[str, str],
    variant: str,
    seed: int,
    iteration: int,
    checkpoint_sha256: str,
    scenario_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    outcome,
    baseline_outcome,
    trajectory_sha256: str,
    accounting: Mapping[str, int | float] | None,
) -> dict[str, object]:
    candidate = _outcome(outcome)
    baseline = _outcome(baseline_outcome)
    metrics = accounting or {
        "deterministic_steps": 0,
        "deterministic_speed_projection_count": 0,
        "deterministic_steer_projection_count": 0,
        "max_abs_deterministic_speed_projection_delta": 0.0,
        "max_abs_deterministic_steer_projection_delta": 0.0,
    }
    return {
        "schema": B4_EVAL_ROW_SCHEMA,
        "task8_row_index": int(physical_index),
        "manifest_order": str(case["manifest_order"]),
        "panel": str(case["panel"]),
        "l2_id": str(case["l2_id"]),
        "l4_id": str(case["l4_id"]),
        "map_name": str(case["map_name"]),
        "skill": str(case["skill"]),
        "opponent_raceline": str(case["opponent_raceline"]),
        "speedscale_hex": str(case["speedscale_hex"]),
        "variant": variant,
        "seed": int(seed),
        "iteration": int(iteration),
        "checkpoint_sha256": checkpoint_sha256,
        "scenario_manifest_sha256": scenario_manifest_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "trajectory_sha256": trajectory_sha256,
        **candidate,
        "bc_four_state": baseline["four_state"],
        "bc_collision_any": baseline["collision_any"],
        "bc_terminal_overtake": baseline["terminal_overtake"],
        "bc_interaction_attempt": baseline["interaction_attempt"],
        "transition": f"{baseline['four_state']}->{candidate['four_state']}",
        "fixed_collision": baseline["collision_any"] and not candidate["collision_any"],
        "new_collision": not baseline["collision_any"] and candidate["collision_any"],
        "gained_overtake": not baseline["terminal_overtake"] and candidate["terminal_overtake"],
        "lost_overtake": baseline["terminal_overtake"] and not candidate["terminal_overtake"],
        "collision_to_confirmed_pass": baseline["collision_any"]
        and candidate["confirmed_safe_pass"],
        **metrics,
    }


@dataclass(frozen=True)
class B4EvaluationShard:
    shard_index: int
    shard_count: int
    scenario_manifest_sha256: str
    checkpoint_manifest_sha256: str
    bc_checkpoint_sha256: str
    checkpoint_sha256_by_variant: Mapping[str, str]
    rows: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["schema"] = B4_EVAL_SHARD_SCHEMA
        value["checkpoint_sha256_by_variant"] = dict(self.checkpoint_sha256_by_variant)
        value["rows"] = list(self.rows)
        return value


def evaluate_shard(
    *,
    task8_rows: Sequence[Mapping[str, str]],
    scenario_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    bc_checkpoint_path: str | Path,
    bc_checkpoint_sha256: str,
    checkpoints: Sequence[B4CheckpointSpec],
    device: torch.device,
    shard_index: int,
    shard_count: int,
    simulator: Callable | None = None,
    trajectory_digest_fn: Callable[[Mapping], str] | None = None,
) -> B4EvaluationShard:
    validate_task8_rows(task8_rows)
    if simulator is None:
        from d25.oracle import simulate_episode

        simulator = simulate_episode
    if trajectory_digest_fn is None:
        from d25.search import trajectory_digest

        trajectory_digest_fn = trajectory_digest
    if not _is_sha256(scenario_manifest_sha256) or not _is_sha256(bc_checkpoint_sha256):
        raise ValueError("B4 evaluation manifest/BC identity is invalid")
    specs = validate_checkpoint_specs(checkpoints, checkpoint_manifest_sha256)
    bc_path = Path(bc_checkpoint_path)
    if file_sha256(bc_path) != bc_checkpoint_sha256:
        raise ValueError("B4 evaluation BC checkpoint hash mismatch")
    bc_model = load_strict_plain_actor(bc_path, device)
    policies: dict[str, nn.Module] = {}
    for spec in specs:
        path = Path(spec.checkpoint_path)
        if file_sha256(path) != spec.checkpoint_sha256:
            raise ValueError(f"B4 candidate checkpoint hash mismatch: {spec.variant}")
        policies[spec.variant] = load_strict_plain_actor(path, device)
    assigned = physical_shard_rows(task8_rows, shard_index, shard_count)
    inventory = {B4_VARIANT_BC: bc_checkpoint_sha256} | {
        spec.variant: spec.checkpoint_sha256 for spec in specs
    }
    rows: list[dict[str, object]] = []
    for physical_index, case in assigned:
        baseline = simulator(bc_model, device, case)
        baseline_digest = trajectory_digest_fn(baseline.arrays)
        rows.append(
            paired_row(
                physical_index=physical_index,
                case=case,
                variant=B4_VARIANT_BC,
                seed=-1,
                iteration=0,
                checkpoint_sha256=bc_checkpoint_sha256,
                scenario_manifest_sha256=scenario_manifest_sha256,
                checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                outcome=baseline.outcome,
                baseline_outcome=baseline.outcome,
                trajectory_sha256=baseline_digest,
                accounting=None,
            )
        )
        for spec in specs:
            actor = B4DeterministicActor(policies[spec.variant])
            result = simulator(actor, device, case)
            rows.append(
                paired_row(
                    physical_index=physical_index,
                    case=case,
                    variant=spec.variant,
                    seed=spec.seed,
                    iteration=spec.iteration,
                    checkpoint_sha256=spec.checkpoint_sha256,
                    scenario_manifest_sha256=scenario_manifest_sha256,
                    checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                    outcome=result.outcome,
                    baseline_outcome=baseline.outcome,
                    trajectory_sha256=trajectory_digest_fn(result.arrays),
                    accounting=actor.accounting(),
                )
            )
    return B4EvaluationShard(
        shard_index=int(shard_index),
        shard_count=int(shard_count),
        scenario_manifest_sha256=scenario_manifest_sha256,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        bc_checkpoint_sha256=bc_checkpoint_sha256,
        checkpoint_sha256_by_variant=inventory,
        rows=tuple(rows),
    )


def _counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {
        "episodes": len(rows),
        "collision": sum(bool(row["collision_any"]) for row in rows),
        "terminal_overtake": sum(bool(row["terminal_overtake"]) for row in rows),
        "fixed_collision": sum(bool(row["fixed_collision"]) for row in rows),
        "new_collision": sum(bool(row["new_collision"]) for row in rows),
        "gained_overtake": sum(bool(row["gained_overtake"]) for row in rows),
        "lost_overtake": sum(bool(row["lost_overtake"]) for row in rows),
        "collision_to_confirmed_pass": sum(
            bool(row["collision_to_confirmed_pass"]) for row in rows
        ),
        "deterministic_speed_projection_count": sum(
            int(row["deterministic_speed_projection_count"]) for row in rows
        ),
        "deterministic_steer_projection_count": sum(
            int(row["deterministic_steer_projection_count"]) for row in rows
        ),
    }


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    bc_rows = [row for row in rows if row["variant"] == B4_VARIANT_BC]
    if len(bc_rows) != EXPECTED_SCENARIOS:
        raise ValueError("B4 merged baseline row count drift")
    bc = _counts(bc_rows)
    if bc["collision"] != EXPECTED_BC_COLLISIONS or bc["terminal_overtake"] != EXPECTED_BC_OVERTAKES:
        raise ValueError("B4 frozen BC development baseline drift")
    variants: dict[str, dict[str, object]] = {}
    for seed in B4_SEEDS:
        for iteration in B4_ITERATIONS:
            variant = b4_variant(seed, iteration)
            selected = [row for row in rows if row["variant"] == variant]
            if len(selected) != EXPECTED_SCENARIOS:
                raise ValueError(f"B4 candidate row count drift: {variant}")
            count = _counts(selected)
            count.update(
                {
                    "seed": seed,
                    "iteration": iteration,
                    "overtake_gate_pass": count["terminal_overtake"] >= B4_OVERTAKE_GATE,
                    "collision_feasibility_pass": count["collision"] <= B4_COLLISION_FEASIBILITY,
                    "collision_strict_improve": count["collision"] < EXPECTED_BC_COLLISIONS,
                    "deterministic_speed_projection_pass": count[
                        "deterministic_speed_projection_count"
                    ]
                    == 0,
                    "product_collision_target_pass": count["collision"]
                    <= B4_COLLISION_PRODUCT_PER_SEED,
                }
            )
            variants[variant] = count

    pairs: dict[str, dict[str, object]] = {}
    feasible: list[dict[str, object]] = []
    for iteration in B4_ITERATIONS:
        names = [b4_variant(seed, iteration) for seed in B4_SEEDS]
        selected = [row for row in rows if row["variant"] in names]
        pooled = _counts(selected)
        per_seed = [variants[name] for name in names]
        checks = {
            "both_seed_overtake_ge_132": all(
                bool(value["overtake_gate_pass"]) for value in per_seed
            ),
            "both_seed_collision_le_24": all(
                bool(value["collision_feasibility_pass"]) for value in per_seed
            ),
            "both_seed_collision_strict_improve": all(
                bool(value["collision_strict_improve"]) for value in per_seed
            ),
            "pooled_fixed_gt_new": pooled["fixed_collision"] > pooled["new_collision"],
            "zero_deterministic_speed_projection": pooled[
                "deterministic_speed_projection_count"
            ]
            == 0,
        }
        is_feasible = all(checks.values())
        product_hit = is_feasible and all(
            bool(value["product_collision_target_pass"]) for value in per_seed
        ) and pooled["collision"] <= B4_COLLISION_PRODUCT_POOLED
        pair = {
            "iteration": iteration,
            "seed_variants": names,
            **pooled,
            "reported_pooled_overtake_floor": 264,
            "checks": checks,
            "feasible": is_feasible,
            "product_collision_target_pass": product_hit,
            "verdict_label": (
                "OPENED_DEVELOPMENT_PRODUCT_TARGET_HIT"
                if product_hit
                else (
                    "OPENED_DEVELOPMENT_DIRECTIONAL_SURVIVOR"
                    if is_feasible
                    else "INFEASIBLE"
                )
            ),
        }
        pairs[f"iter{iteration}"] = pair
        if is_feasible:
            feasible.append(pair)
    selected_pair = None
    if feasible:
        selected_pair = min(
            feasible,
            key=lambda value: (
                int(value["collision"]),
                -int(value["terminal_overtake"]),
                int(value["iteration"]),
            ),
        )
    return {
        "schema": B4_EVAL_MERGE_SCHEMA,
        "integrity_passed": True,
        "opened_development_only": True,
        "fresh_pool_opened": False,
        "bc_baseline": bc,
        "variants": variants,
        "same_iteration_pairs": pairs,
        "selected_iteration": (
            None if selected_pair is None else int(selected_pair["iteration"])
        ),
        "selected_pair_verdict": (
            "B4_SUBSTANTIVE_NEGATIVE"
            if selected_pair is None
            else str(selected_pair["verdict_label"])
        ),
        "automatic_b3_fallback_authorized": False,
        "snapshot_pair_selection_performed": selected_pair is not None,
        "architecture_arm_selection_performed": False,
    }


def merge_shards(
    *,
    shards: Sequence[B4EvaluationShard],
    task8_rows: Sequence[Mapping[str, str]],
    scenario_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    bc_checkpoint_sha256: str,
    checkpoints: Sequence[B4CheckpointSpec],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    validate_task8_rows(task8_rows)
    specs = validate_checkpoint_specs(checkpoints, checkpoint_manifest_sha256)
    inventory = {B4_VARIANT_BC: bc_checkpoint_sha256} | {
        spec.variant: spec.checkpoint_sha256 for spec in specs
    }
    if not shards:
        raise ValueError("B4 evaluation merge requires shards")
    shard_count = shards[0].shard_count
    if shard_count != len(shards) or {shard.shard_index for shard in shards} != set(
        range(shard_count)
    ):
        raise ValueError("B4 evaluation shard inventory is incomplete")
    rows: list[dict[str, object]] = []
    for shard in shards:
        if (
            shard.shard_count != shard_count
            or shard.scenario_manifest_sha256 != scenario_manifest_sha256
            or shard.checkpoint_manifest_sha256 != checkpoint_manifest_sha256
            or shard.bc_checkpoint_sha256 != bc_checkpoint_sha256
            or dict(shard.checkpoint_sha256_by_variant) != inventory
        ):
            raise ValueError("B4 evaluation shard identity mismatch")
        for row in shard.rows:
            if row.get("schema") != B4_EVAL_ROW_SCHEMA:
                raise ValueError("B4 evaluation row schema mismatch")
            if int(row["task8_row_index"]) % shard_count != shard.shard_index:
                raise ValueError("B4 evaluation row came from wrong physical shard")
            rows.append(dict(row))
    expected_variants = set(inventory)
    observed = {(int(row["task8_row_index"]), str(row["variant"])) for row in rows}
    expected = {
        (index, variant)
        for index in range(EXPECTED_SCENARIOS)
        for variant in expected_variants
    }
    if len(rows) != B4_EXPECTED_RESULTS or observed != expected:
        raise ValueError("B4 288x7 Cartesian product is incomplete or duplicated")
    by_key = {(int(row["task8_row_index"]), str(row["variant"])): row for row in rows}
    spec_by_variant = {spec.variant: spec for spec in specs}
    for index, case in enumerate(task8_rows):
        baseline = by_key[(index, B4_VARIANT_BC)]
        for variant in expected_variants:
            row = by_key[(index, variant)]
            if (
                row["l2_id"] != case["l2_id"]
                or row["manifest_order"] != case["manifest_order"]
                or row["scenario_manifest_sha256"] != scenario_manifest_sha256
                or row["checkpoint_manifest_sha256"] != checkpoint_manifest_sha256
                or row["checkpoint_sha256"] != inventory[variant]
            ):
                raise ValueError("B4 evaluation row identity/checkpoint mismatch")
            if variant == B4_VARIANT_BC:
                if int(row["seed"]) != -1 or int(row["iteration"]) != 0:
                    raise ValueError("B4 baseline row identity mismatch")
            else:
                spec = spec_by_variant[variant]
                if int(row["seed"]) != spec.seed or int(row["iteration"]) != spec.iteration:
                    raise ValueError("B4 candidate row seed/iteration mismatch")
            paired_fields = (
                ("bc_four_state", "four_state"),
                ("bc_collision_any", "collision_any"),
                ("bc_terminal_overtake", "terminal_overtake"),
                ("bc_interaction_attempt", "interaction_attempt"),
            )
            if any(row[name] != baseline[base_name] for name, base_name in paired_fields):
                raise ValueError("B4 paired baseline fields drift")
            expected_diagnostics = {
                "transition": f"{baseline['four_state']}->{row['four_state']}",
                "fixed_collision": bool(baseline["collision_any"])
                and not bool(row["collision_any"]),
                "new_collision": not bool(baseline["collision_any"])
                and bool(row["collision_any"]),
                "gained_overtake": not bool(baseline["terminal_overtake"])
                and bool(row["terminal_overtake"]),
                "lost_overtake": bool(baseline["terminal_overtake"])
                and not bool(row["terminal_overtake"]),
                "collision_to_confirmed_pass": bool(baseline["collision_any"])
                and bool(row["confirmed_safe_pass"]),
            }
            if any(row.get(name) != value for name, value in expected_diagnostics.items()):
                raise ValueError("B4 paired transition diagnostic drift")
    rows.sort(key=lambda row: (int(row["task8_row_index"]), str(row["variant"])))
    summary = summarize(rows)
    summary.update(
        {
            "scenario_manifest_sha256": scenario_manifest_sha256,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "scenario_count": EXPECTED_SCENARIOS,
            "variant_count": B4_VARIANT_COUNT,
            "result_count": B4_EXPECTED_RESULTS,
            "shard_count": shard_count,
        }
    )
    return rows, summary

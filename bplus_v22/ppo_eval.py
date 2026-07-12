"""B2 deterministic paired evaluator, independent of warm-start releases.

The evaluator consumes generic injected PPO checkpoint loaders.  It executes
the primary centered policy with zero behavior offsets and records the standard
Bernoulli decision only as a same-state diagnostic.  Scenario sharding uses
the physical row index in the corrected Task-8 TSV, never ``manifest_order``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from bplus_v22 import ARMS, HISTORY_OFFSETS, LIDAR_BEAMS, MACRO_STEPS, PILOT_SEEDS
from bplus_v22.exploration import DETERMINISTIC_CENTERED, DETERMINISTIC_STANDARD
from bplus_v22.remediated_model import (
    HierarchicalResidualAction,
    HierarchicalResidualDistribution,
)
from d25.oracle import simulate_episode
from d25.search import trajectory_digest


PPO_EVAL_SCHEMA = "bplus-v2.2-ppo-eval-shard-1"
PPO_EVAL_RESULT_SCHEMA = "bplus-v2.2-ppo-eval-result-1"
PPO_EVAL_MERGE_SCHEMA = "bplus-v2.2-ppo-eval-merge-1"
BC_VARIANT = "BC"
EXPECTED_SCENARIOS = 288
EXPECTED_VARIANTS = 1 + len(ARMS) * len(PILOT_SEEDS)
EXPECTED_RESULTS = EXPECTED_SCENARIOS * EXPECTED_VARIANTS
EXPECTED_BC_COLLISIONS = 24
EXPECTED_BC_OVERTAKES = 138
CLUSTER_BOOTSTRAP_REPLICATES = 10000
CLUSTER_BOOTSTRAP_DOMAIN = b"end2race:bplus-v2.2:b2-l4-bootstrap:v1\0"

TASK8_REQUIRED_FIELDS = (
    "manifest_order",
    "panel",
    "l2_id",
    "l4_id",
    "map_name",
    "skill",
    "opponent_raceline",
    "speedscale_hex",
    "resolved_ego_idx",
)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_variant(arm: str, seed: int) -> str:
    if arm not in ARMS or int(seed) != seed or int(seed) not in PILOT_SEEDS:
        raise ValueError("invalid B2 candidate arm/seed")
    return f"{arm}::seed{int(seed)}"


@dataclass(frozen=True)
class CandidateCheckpoint:
    """Expected checkpoint identity; the injected loader must prove it loaded this."""

    arm: str
    seed: int
    checkpoint_id: str
    checkpoint_sha256: str
    training_manifest_sha256: str

    def __post_init__(self) -> None:
        candidate_variant(self.arm, self.seed)
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("candidate checkpoint_id must be non-empty")
        if not _is_sha256(self.checkpoint_sha256):
            raise ValueError("candidate checkpoint SHA256 is invalid")
        if not _is_sha256(self.training_manifest_sha256):
            raise ValueError("candidate training-manifest SHA256 is invalid")

    @property
    def variant(self) -> str:
        return candidate_variant(self.arm, self.seed)


@dataclass(frozen=True)
class LoadedCandidatePolicy:
    """Generic loader result with independently observed envelope identities."""

    policy: nn.Module
    checkpoint_id: str
    checkpoint_sha256: str
    training_manifest_sha256: str


PolicyLoader = Callable[[CandidateCheckpoint, torch.device], LoadedCandidatePolicy]
ActorFactory = Callable[[nn.Module], nn.Module]
Simulator = Callable[..., object]


def validate_checkpoint_specs(
    checkpoints: Sequence[CandidateCheckpoint], training_manifest_sha256: str
) -> tuple[CandidateCheckpoint, ...]:
    if not _is_sha256(training_manifest_sha256):
        raise ValueError("expected training-manifest SHA256 is invalid")
    ordered = tuple(checkpoints)
    expected = {(arm, seed) for arm in ARMS for seed in PILOT_SEEDS}
    observed = {(item.arm, item.seed) for item in ordered}
    if observed != expected or len(ordered) != len(expected):
        raise ValueError("candidate checkpoint inventory must be exact 3 arms x 2 seeds")
    if len({item.variant for item in ordered}) != len(ordered):
        raise ValueError("duplicate candidate checkpoint variant")
    if any(item.training_manifest_sha256 != training_manifest_sha256 for item in ordered):
        raise ValueError("candidate checkpoint/training-manifest mismatch")
    return tuple(sorted(ordered, key=lambda item: (ARMS.index(item.arm), item.seed)))


def load_candidate_policies(
    checkpoints: Sequence[CandidateCheckpoint],
    training_manifest_sha256: str,
    loader: PolicyLoader,
    device: torch.device,
) -> dict[str, nn.Module]:
    policies: dict[str, nn.Module] = {}
    for expected in validate_checkpoint_specs(checkpoints, training_manifest_sha256):
        loaded = loader(expected, device)
        if not isinstance(loaded, LoadedCandidatePolicy):
            raise TypeError("candidate loader must return LoadedCandidatePolicy")
        if (
            loaded.checkpoint_id != expected.checkpoint_id
            or loaded.checkpoint_sha256 != expected.checkpoint_sha256
            or loaded.training_manifest_sha256 != expected.training_manifest_sha256
        ):
            raise ValueError(f"loaded checkpoint/envelope mismatch: {expected.variant}")
        policy = loaded.policy
        if getattr(policy, "arm", None) != expected.arm:
            raise ValueError(f"loaded checkpoint arm mismatch: {expected.variant}")
        offset = getattr(policy, "intervention_logit_offset", None)
        if not isinstance(offset, torch.Tensor) or not torch.equal(
            offset.detach(), torch.zeros_like(offset.detach())
        ):
            raise ValueError(f"candidate has nonzero persistent exploration offset: {expected.variant}")
        policy.eval()
        policies[expected.variant] = policy
    return policies


def read_task8_development(
    path: str | Path, expected_sha256: str
) -> list[dict[str, str]]:
    source = Path(path)
    if not _is_sha256(expected_sha256) or _file_sha256(source) != expected_sha256:
        raise ValueError("Task-8 development manifest SHA256 mismatch")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    validate_task8_rows(rows)
    return rows


def validate_task8_rows(rows: Sequence[Mapping[str, str]]) -> None:
    if len(rows) != EXPECTED_SCENARIOS:
        raise ValueError(f"Task-8 development manifest must contain {EXPECTED_SCENARIOS} rows")
    l2_ids: list[str] = []
    for physical_index, row in enumerate(rows):
        missing = set(TASK8_REQUIRED_FIELDS) - set(row)
        if missing:
            raise ValueError(f"Task-8 physical row {physical_index} lacks {sorted(missing)}")
        if not row["l2_id"] or not row["map_name"] or not row["opponent_raceline"]:
            raise ValueError(f"Task-8 physical row {physical_index} has empty identity")
        try:
            int(row["resolved_ego_idx"])
            speedscale = float.fromhex(row["speedscale_hex"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Task-8 physical row {physical_index} scenario is invalid") from error
        if not np.isfinite(speedscale) or speedscale <= 0.0:
            raise ValueError(f"Task-8 physical row {physical_index} speedscale is invalid")
        l2_ids.append(row["l2_id"])
    if len(set(l2_ids)) != EXPECTED_SCENARIOS:
        raise ValueError("Task-8 development manifest contains duplicate L2")


def physical_shard_rows(
    rows: Sequence[Mapping[str, str]], shard_index: int, shard_count: int
) -> tuple[tuple[int, Mapping[str, str]], ...]:
    """Assign by physical TSV position; ``manifest_order`` is provenance only."""

    validate_task8_rows(rows)
    if int(shard_count) != shard_count or int(shard_count) <= 0:
        raise ValueError("evaluation shard_count must be positive")
    if int(shard_index) != shard_index or not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("evaluation shard_index is invalid")
    return tuple(
        (physical_index, row)
        for physical_index, row in enumerate(rows)
        if physical_index % int(shard_count) == int(shard_index)
    )


class B2DeterministicActor(nn.Module):
    """Centered primary actor plus same-state standard-mode decision diagnostics."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        self.reset_runtime()

    @property
    def gru(self):
        return self.policy.bc.gru

    def reset_runtime(self) -> None:
        self.micro_steps = 0
        self.macro_decisions = 0
        self.primary_intervention_decisions = 0
        self.primary_brake_decisions = 0
        self.standard_intervention_decisions = 0
        self.standard_brake_decisions = 0
        self.external_clip_micro_steps = 0
        self.sum_abs_applied_steer_delta = 0.0
        self.sum_applied_speed_delta = 0.0
        self.max_abs_applied_steer_delta = 0.0
        self.max_brake_delta = 0.0
        self._records: list[dict[str, object]] = []
        self._held_action: HierarchicalResidualAction | None = None
        self._lidar_history: list[torch.Tensor] = []
        self._speed_history: list[torch.Tensor] = []
        self._steer_history: list[torch.Tensor] = []
        self._command_speed_history: list[torch.Tensor] = []
        self._last_applied_command: tuple[float, float] | None = None
        self._pending_actual_speed: float | None = None
        self._requested_command: tuple[float, float] | None = None
        self._awaiting_applied_command = False

    def observe_actual_speed(self, value: float) -> None:
        speed = float(value)
        if not np.isfinite(speed) or self._pending_actual_speed is not None:
            raise ValueError("B2 evaluator actual-speed sequence invalid")
        self._pending_actual_speed = speed

    def observe_applied_command(self, steer: float, speed: float) -> None:
        command = (float(steer), float(speed))
        if (
            not all(np.isfinite(value) for value in command)
            or not self._awaiting_applied_command
            or self._requested_command is None
        ):
            raise RuntimeError("B2 evaluator applied-command sequence invalid")
        if command != self._requested_command:
            self.external_clip_micro_steps += 1
            raise AssertionError("B2 evaluator changed a bound-preserving command")
        self._last_applied_command = command
        self._awaiting_applied_command = False

    def _history(self) -> tuple[torch.Tensor, torch.Tensor]:
        current = len(self._lidar_history) - 1
        indices = [max(0, current - offset) for offset in HISTORY_OFFSETS]
        lidar = torch.stack([self._lidar_history[index] for index in indices], dim=1)
        scalar = torch.cat(
            [
                torch.stack([self._speed_history[index] for index in indices], dim=1).squeeze(-1),
                torch.stack([self._steer_history[index] for index in indices], dim=1).squeeze(-1),
                torch.stack(
                    [self._command_speed_history[index] for index in indices], dim=1
                ).squeeze(-1),
            ],
            dim=1,
        )
        if lidar.shape != (1, len(HISTORY_OFFSETS), LIDAR_BEAMS) or scalar.shape != (1, 24):
            raise AssertionError("B2 evaluator history shape drift")
        return lidar, scalar

    def forward(self, lidar, previous_speed, hidden):
        if lidar.shape != (1, 1, LIDAR_BEAMS) or previous_speed.shape != (1, 1, 1):
            raise ValueError("B2 evaluator requires canonical batch-one evaluator tensors")
        if self._pending_actual_speed is None or self._awaiting_applied_command:
            raise RuntimeError("B2 evaluator observation/command sequence incomplete")
        actual_speed = torch.full_like(previous_speed[:, -1], self._pending_actual_speed)
        self._pending_actual_speed = None
        base, bc_feature, next_hidden = self.policy.bc_step(lidar, previous_speed, hidden)
        prior = (
            torch.zeros_like(base)
            if self._last_applied_command is None
            else base.new_tensor(self._last_applied_command).reshape(1, 2)
        )
        self._lidar_history.append((torch.clamp(lidar[:, -1], 0.0, 30.0) / 30.0).detach())
        self._speed_history.append((actual_speed / 10.0).detach())
        self._steer_history.append((prior[:, 0:1] / 0.52).detach())
        self._command_speed_history.append((prior[:, 1:2] / 10.0).detach())

        if self.micro_steps % MACRO_STEPS == 0:
            lidar_history, scalar_history = self._history()
            primary = self.policy.deterministic_action(
                bc_feature, lidar_history, scalar_history, DETERMINISTIC_CENTERED
            )
            standard = self.policy.deterministic_action(
                bc_feature, lidar_history, scalar_history, DETERMINISTIC_STANDARD
            )
            self._held_action = primary
            self._records.append(
                {"primary": primary, "standard": standard, "micro_count": 0}
            )
            self.macro_decisions += 1
            self.primary_intervention_decisions += int(primary.intervention_gate.item())
            self.primary_brake_decisions += int(primary.brake_gate.item())
            self.standard_intervention_decisions += int(standard.intervention_gate.item())
            self.standard_brake_decisions += int(standard.brake_gate.item())
        if self._held_action is None:
            raise AssertionError("B2 evaluator lacks held macro latent")

        ledger = HierarchicalResidualDistribution.compose(base, self._held_action)
        if torch.any(ledger.external_clip_would_change):
            raise AssertionError("B2 evaluator composition requires external clipping")
        steer_delta = float(ledger.applied_residual[0, 0].item())
        speed_delta = float(ledger.applied_residual[0, 1].item())
        self.sum_abs_applied_steer_delta += abs(steer_delta)
        self.sum_applied_speed_delta += speed_delta
        self.max_abs_applied_steer_delta = max(
            self.max_abs_applied_steer_delta, abs(steer_delta)
        )
        self.max_brake_delta = max(self.max_brake_delta, -speed_delta)
        self._records[-1]["micro_count"] = int(self._records[-1]["micro_count"]) + 1
        command = ledger.command
        self._requested_command = (float(command[0, 0].item()), float(command[0, 1].item()))
        self._awaiting_applied_command = True
        self.micro_steps += 1
        return command.unsqueeze(1), next_hidden

    def accounting(self) -> dict[str, object]:
        lengths = [int(record["micro_count"]) for record in self._records]
        if (
            self.micro_steps <= 0
            or sum(lengths) != self.micro_steps
            or any(not 1 <= length <= MACRO_STEPS for length in lengths)
            or any(length != MACRO_STEPS for length in lengths[:-1])
            or self.external_clip_micro_steps != 0
        ):
            raise AssertionError("B2 evaluator macro/composition accounting failed")
        return {
            "micro_steps": self.micro_steps,
            "macro_decisions": self.macro_decisions,
            "macro_lengths": lengths,
            "short_terminal_macro": lengths[-1] < MACRO_STEPS,
            "primary_intervention_decisions": self.primary_intervention_decisions,
            "primary_brake_decisions": self.primary_brake_decisions,
            "standard_intervention_decisions": self.standard_intervention_decisions,
            "standard_brake_decisions": self.standard_brake_decisions,
            "mean_abs_applied_steer_delta": self.sum_abs_applied_steer_delta / self.micro_steps,
            "max_abs_applied_steer_delta": self.max_abs_applied_steer_delta,
            "mean_brake_delta": -self.sum_applied_speed_delta / self.micro_steps,
            "max_brake_delta": self.max_brake_delta,
            "external_clip_micro_steps": self.external_clip_micro_steps,
        }


def _outcome_values(outcome) -> dict[str, object]:
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


def paired_result_row(
    *,
    physical_index: int,
    case: Mapping[str, str],
    variant: str,
    arm: str,
    seed: int,
    checkpoint_sha256: str,
    scenario_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    outcome,
    baseline_outcome,
    trajectory_sha256: str,
    accounting: Mapping[str, object] | None,
) -> dict[str, object]:
    candidate = _outcome_values(outcome)
    baseline = _outcome_values(baseline_outcome)
    metrics = accounting or {
        "micro_steps": 0,
        "macro_decisions": 0,
        "macro_lengths": [],
        "short_terminal_macro": False,
        "primary_intervention_decisions": 0,
        "primary_brake_decisions": 0,
        "standard_intervention_decisions": 0,
        "standard_brake_decisions": 0,
        "mean_abs_applied_steer_delta": 0.0,
        "max_abs_applied_steer_delta": 0.0,
        "mean_brake_delta": 0.0,
        "max_brake_delta": 0.0,
        "external_clip_micro_steps": 0,
    }
    return {
        "schema": PPO_EVAL_RESULT_SCHEMA,
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
        "arm": arm,
        "seed": int(seed),
        "deterministic_mode": "bc" if variant == BC_VARIANT else DETERMINISTIC_CENTERED,
        "standard_mode_is_diagnostic_only": variant != BC_VARIANT,
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
class EvaluationShard:
    shard_index: int
    shard_count: int
    scenario_manifest_sha256: str
    checkpoint_manifest_sha256: str
    bc_checkpoint_sha256: str
    checkpoint_sha256_by_variant: Mapping[str, str]
    rows: tuple[dict[str, object], ...]
    schema: str = PPO_EVAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PPO_EVAL_SCHEMA:
            raise ValueError("B2 evaluation shard schema mismatch")
        if (
            int(self.shard_count) != self.shard_count
            or int(self.shard_count) <= 0
            or int(self.shard_index) != self.shard_index
            or not 0 <= int(self.shard_index) < int(self.shard_count)
        ):
            raise ValueError("B2 evaluation shard identity is invalid")
        if not all(
            _is_sha256(value)
            for value in (
                self.scenario_manifest_sha256,
                self.checkpoint_manifest_sha256,
                self.bc_checkpoint_sha256,
                *self.checkpoint_sha256_by_variant.values(),
            )
        ):
            raise ValueError("B2 evaluation shard contains invalid SHA256")


def evaluate_shard(
    *,
    task8_rows: Sequence[Mapping[str, str]],
    scenario_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    bc_model: nn.Module,
    bc_checkpoint_sha256: str,
    checkpoints: Sequence[CandidateCheckpoint],
    policy_loader: PolicyLoader,
    device: torch.device,
    shard_index: int,
    shard_count: int,
    actor_factory: ActorFactory = B2DeterministicActor,
    simulator: Simulator = simulate_episode,
    trajectory_digest_fn: Callable[[Mapping], str] = trajectory_digest,
) -> EvaluationShard:
    if not _is_sha256(scenario_manifest_sha256) or not _is_sha256(bc_checkpoint_sha256):
        raise ValueError("B2 evaluation manifest/BC identity is invalid")
    specs = validate_checkpoint_specs(checkpoints, checkpoint_manifest_sha256)
    policies = load_candidate_policies(specs, checkpoint_manifest_sha256, policy_loader, device)
    assigned = physical_shard_rows(task8_rows, shard_index, shard_count)
    bc_model.eval()
    checkpoint_inventory = {BC_VARIANT: bc_checkpoint_sha256} | {
        spec.variant: spec.checkpoint_sha256 for spec in specs
    }
    rows: list[dict[str, object]] = []
    for physical_index, case in assigned:
        baseline = simulator(bc_model, device, case)
        if bool(baseline.action_clipped):
            raise AssertionError(f"BC evaluator clipped action: {case['l2_id']}")
        baseline_digest = trajectory_digest_fn(baseline.arrays)
        rows.append(
            paired_result_row(
                physical_index=physical_index,
                case=case,
                variant=BC_VARIANT,
                arm=BC_VARIANT,
                seed=-1,
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
            actor = actor_factory(policies[spec.variant])
            result = simulator(actor, device, case)
            accounting = actor.accounting()
            if bool(result.action_clipped) or int(accounting["external_clip_micro_steps"]) != 0:
                raise AssertionError(f"candidate evaluator clipped action: {spec.variant}/{case['l2_id']}")
            rows.append(
                paired_result_row(
                    physical_index=physical_index,
                    case=case,
                    variant=spec.variant,
                    arm=spec.arm,
                    seed=spec.seed,
                    checkpoint_sha256=spec.checkpoint_sha256,
                    scenario_manifest_sha256=scenario_manifest_sha256,
                    checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                    outcome=result.outcome,
                    baseline_outcome=baseline.outcome,
                    trajectory_sha256=trajectory_digest_fn(result.arrays),
                    accounting=accounting,
                )
            )
    return EvaluationShard(
        shard_index=int(shard_index),
        shard_count=int(shard_count),
        scenario_manifest_sha256=scenario_manifest_sha256,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        bc_checkpoint_sha256=bc_checkpoint_sha256,
        checkpoint_sha256_by_variant=checkpoint_inventory,
        rows=tuple(rows),
    )


def evaluate_bc_baseline_preflight(
    *,
    task8_rows: Sequence[Mapping[str, str]],
    scenario_manifest_sha256: str,
    bc_model: nn.Module,
    bc_checkpoint_sha256: str,
    device: torch.device,
    simulator: Simulator = simulate_episode,
    trajectory_digest_fn: Callable[[Mapping], str] = trajectory_digest,
) -> dict[str, object]:
    """Reproduce the frozen 288-row BC baseline before any learner is launched."""

    validate_task8_rows(task8_rows)
    if not _is_sha256(scenario_manifest_sha256) or not _is_sha256(
        bc_checkpoint_sha256
    ):
        raise ValueError("B2 BC preflight manifest/checkpoint identity is invalid")
    bc_model.eval()
    rows: list[dict[str, object]] = []
    for physical_index, case in enumerate(task8_rows):
        result = simulator(bc_model, device, case)
        if bool(result.action_clipped):
            raise AssertionError(f"B2 BC preflight clipped action: {case['l2_id']}")
        rows.append(
            {
                "task8_row_index": physical_index,
                "l2_id": str(case["l2_id"]),
                "l4_id": str(case["l4_id"]),
                "map_name": str(case["map_name"]),
                "trajectory_sha256": trajectory_digest_fn(result.arrays),
                **_outcome_values(result.outcome),
            }
        )
    collision = sum(bool(row["collision_any"]) for row in rows)
    overtake = sum(bool(row["terminal_overtake"]) for row in rows)
    if collision != EXPECTED_BC_COLLISIONS or overtake != EXPECTED_BC_OVERTAKES:
        raise ValueError(
            f"B2 BC preflight drift: collision={collision}, overtake={overtake}"
        )
    return {
        "schema": "bplus-v2.2-b2-bc-baseline-preflight-1",
        "passed": True,
        "scenario_manifest_sha256": scenario_manifest_sha256,
        "bc_checkpoint_sha256": bc_checkpoint_sha256,
        "scenario_count": len(rows),
        "collision": collision,
        "terminal_overtake": overtake,
        "rows": rows,
    }


def _expected_checkpoint_inventory(
    checkpoints: Sequence[CandidateCheckpoint],
    checkpoint_manifest_sha256: str,
    bc_checkpoint_sha256: str,
) -> dict[str, str]:
    specs = validate_checkpoint_specs(checkpoints, checkpoint_manifest_sha256)
    return {BC_VARIANT: bc_checkpoint_sha256} | {
        spec.variant: spec.checkpoint_sha256 for spec in specs
    }


def _check_row_pairing(row: Mapping[str, object], baseline: Mapping[str, object]) -> None:
    collision = bool(row["collision_any"])
    overtake = bool(row["terminal_overtake"])
    bc_collision = bool(baseline["collision_any"])
    bc_overtake = bool(baseline["terminal_overtake"])
    expected = {
        "bc_four_state": baseline["four_state"],
        "bc_collision_any": bc_collision,
        "bc_terminal_overtake": bc_overtake,
        "bc_interaction_attempt": bool(baseline["interaction_attempt"]),
        "transition": f"{baseline['four_state']}->{row['four_state']}",
        "fixed_collision": bc_collision and not collision,
        "new_collision": not bc_collision and collision,
        "gained_overtake": not bc_overtake and overtake,
        "lost_overtake": bc_overtake and not overtake,
        "collision_to_confirmed_pass": bc_collision and bool(row["confirmed_safe_pass"]),
    }
    if any(row.get(name) != value for name, value in expected.items()):
        raise ValueError(f"paired transition metric mismatch: {row.get('variant')}/{row.get('l2_id')}")


def _count_summary(
    selected: Sequence[Mapping[str, object]], baseline_collision: int
) -> dict[str, object]:
    transitions: dict[str, int] = {}
    for row in selected:
        key = str(row["transition"])
        transitions[key] = transitions.get(key, 0) + 1
    collision = sum(bool(row["collision_any"]) for row in selected)
    micro_steps = sum(int(row["micro_steps"]) for row in selected)
    fixed_with_attempt = sum(
        bool(row["fixed_collision"]) and bool(row["interaction_attempt"])
        for row in selected
    )
    return {
        "episodes": len(selected),
        "collision": collision,
        "collision_rr_vs_bc": (
            None if baseline_collision == 0 else collision / baseline_collision
        ),
        "ego_collision": sum(bool(row["ego_collision"]) for row in selected),
        "opp_collision": sum(bool(row["opp_collision"]) for row in selected),
        "terminal_overtake": sum(bool(row["terminal_overtake"]) for row in selected),
        "confirmed_safe_pass": sum(
            bool(row["confirmed_safe_pass"]) for row in selected
        ),
        "interaction_attempt": sum(
            bool(row["interaction_attempt"]) for row in selected
        ),
        "bc_interaction_attempt": sum(
            bool(row["bc_interaction_attempt"]) for row in selected
        ),
        "interaction_attempt_change": sum(
            bool(row["interaction_attempt"]) for row in selected
        )
        - sum(bool(row["bc_interaction_attempt"]) for row in selected),
        "fixed_collision": sum(bool(row["fixed_collision"]) for row in selected),
        "new_collision": sum(bool(row["new_collision"]) for row in selected),
        "gained_overtake": sum(bool(row["gained_overtake"]) for row in selected),
        "lost_overtake": sum(bool(row["lost_overtake"]) for row in selected),
        "collision_to_confirmed_pass": sum(
            bool(row["collision_to_confirmed_pass"]) for row in selected
        ),
        "collision_to_safe_follow": sum(
            row["bc_four_state"] == "collision" and row["four_state"] == "safe_follow"
            for row in selected
        ),
        "overtake_to_follow": sum(
            bool(row["bc_terminal_overtake"]) and row["four_state"] == "safe_follow"
            for row in selected
        ),
        "safe_to_new_collision": sum(bool(row["new_collision"]) for row in selected),
        "fixed_collision_with_interaction_attempt": fixed_with_attempt,
        "primary_intervention_decisions": sum(
            int(row["primary_intervention_decisions"]) for row in selected
        ),
        "primary_brake_decisions": sum(
            int(row["primary_brake_decisions"]) for row in selected
        ),
        "standard_intervention_decisions": sum(
            int(row["standard_intervention_decisions"]) for row in selected
        ),
        "standard_brake_decisions": sum(
            int(row["standard_brake_decisions"]) for row in selected
        ),
        "micro_steps": micro_steps,
        "mean_abs_applied_steer_delta": (
            0.0
            if micro_steps == 0
            else sum(
                float(row["mean_abs_applied_steer_delta"])
                * int(row["micro_steps"])
                for row in selected
            )
            / micro_steps
        ),
        "mean_brake_delta": (
            0.0
            if micro_steps == 0
            else sum(
                float(row["mean_brake_delta"]) * int(row["micro_steps"])
                for row in selected
            )
            / micro_steps
        ),
        "external_clip_micro_steps": sum(
            int(row["external_clip_micro_steps"]) for row in selected
        ),
        "transitions": dict(sorted(transitions.items())),
    }


def _cluster_bootstrap(
    selected: Sequence[Mapping[str, object]], variant: str
) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in selected:
        groups.setdefault(str(row["l4_id"]), []).append(row)
    ordered = sorted(groups)
    if not ordered:
        raise ValueError("B2 cluster bootstrap has no L4 groups")
    seed_digest = hashlib.sha256(CLUSTER_BOOTSTRAP_DOMAIN + variant.encode("utf-8")).digest()
    seed = int.from_bytes(seed_digest[:8], "big")
    rng = np.random.default_rng(seed)
    aggregates = np.asarray(
        [
            [
                len(groups[key]),
                sum(bool(row["bc_collision_any"]) for row in groups[key]),
                sum(bool(row["collision_any"]) for row in groups[key]),
                sum(bool(row["bc_terminal_overtake"]) for row in groups[key]),
                sum(bool(row["terminal_overtake"]) for row in groups[key]),
            ]
            for key in ordered
        ],
        dtype=np.float64,
    )
    sampled = rng.integers(
        0,
        len(ordered),
        size=(CLUSTER_BOOTSTRAP_REPLICATES, len(ordered)),
    )
    totals = aggregates[sampled].sum(axis=1)
    valid = totals[:, 1] > 0
    rr = totals[valid, 2] / totals[valid, 1]
    overtake_rd = totals[:, 4] / totals[:, 0] - totals[:, 3] / totals[:, 0]
    return {
        "unit": "L4",
        "cluster_count": len(ordered),
        "replicates": CLUSTER_BOOTSTRAP_REPLICATES,
        "seed": seed,
        "collision_rr_valid_replicates": int(len(rr)),
        "collision_rr_one_sided_95_upper": (
            None if len(rr) < int(0.95 * CLUSTER_BOOTSTRAP_REPLICATES)
            else float(np.quantile(rr, 0.95))
        ),
        "overtake_rate_difference_one_sided_95_lower": float(
            np.quantile(overtake_rd, 0.05)
        ),
        "opened_development_only": True,
    }


def _slice_summary(
    selected: Sequence[Mapping[str, object]], baseline_by_l2: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    output: dict[str, object] = {}
    for field in ("map_name", "skill", "opponent_raceline", "speedscale_hex", "l4_id"):
        values: dict[str, object] = {}
        for value in sorted({str(row[field]) for row in selected}):
            rows = [row for row in selected if str(row[field]) == value]
            baseline_collision = sum(
                bool(baseline_by_l2[str(row["l2_id"])]["collision_any"])
                for row in rows
            )
            values[value] = _count_summary(rows, baseline_collision)
        output[field] = values
    return output


def _direction_verdict(summary: Mapping[str, object], episode_count: int) -> dict[str, object]:
    allowed_loss = int(np.floor(0.01 * int(episode_count) + 1e-12))
    checks = {
        "overtake_noninferiority_1pp": int(summary["lost_overtake"])
        - int(summary["gained_overtake"])
        <= allowed_loss,
        "fixed_collision_gt_new_collision": int(summary["fixed_collision"])
        > int(summary["new_collision"]),
        "collision_to_confirmed_pass_present": int(
            summary["collision_to_confirmed_pass"]
        )
        >= 1,
        "collision_improvement_not_only_attempt_suppression": int(
            summary["fixed_collision_with_interaction_attempt"]
        )
        >= 1,
    }
    return {
        "allowed_net_overtake_loss": allowed_loss,
        "checks": checks,
        "direction_pass": all(checks.values()),
    }


def summarize_paired_results(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    variants: dict[str, dict[str, object]] = {}
    bc_rows = [row for row in rows if row["variant"] == BC_VARIANT]
    if len(bc_rows) != EXPECTED_SCENARIOS:
        raise ValueError("B2 merged BC baseline row count drift")
    bc_collision = sum(bool(row["collision_any"]) for row in bc_rows)
    bc_overtake = sum(bool(row["terminal_overtake"]) for row in bc_rows)
    if bc_collision != EXPECTED_BC_COLLISIONS or bc_overtake != EXPECTED_BC_OVERTAKES:
        raise ValueError(
            f"B2 frozen BC baseline drift: collision={bc_collision}, overtake={bc_overtake}"
        )
    baseline_by_l2 = {str(row["l2_id"]): row for row in bc_rows}
    for variant in sorted({str(row["variant"]) for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        summary = _count_summary(selected, bc_collision)
        if variant != BC_VARIANT:
            direction = _direction_verdict(summary, len(selected))
            target = bool(direction["direction_pass"]) and (
                int(summary["collision"]) <= 16
                and int(summary["terminal_overtake"]) >= EXPECTED_BC_OVERTAKES
            )
            summary.update(
                {
                    "direction_verdict": direction,
                    "per_seed_point_target_pass": target,
                    "verdict_label": (
                        "OPENED_DEV_KPI_POINT_TARGET_HIT"
                        if target
                        else (
                            "DEVELOPMENT_SURVIVOR"
                            if direction["direction_pass"]
                            else "FAILED_DIRECTION_GATE"
                        )
                    ),
                    "l4_cluster_bootstrap": _cluster_bootstrap(selected, variant),
                    "slices": _slice_summary(selected, baseline_by_l2),
                }
            )
        variants[variant] = summary

    arms: dict[str, object] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        pooled = _count_summary(arm_rows, 2 * bc_collision)
        direction = _direction_verdict(pooled, len(arm_rows))
        seed_variants = [candidate_variant(arm, seed) for seed in PILOT_SEEDS]
        same_direction = all(
            int(variants[variant]["fixed_collision"])
            > int(variants[variant]["new_collision"])
            for variant in seed_variants
        )
        direction["checks"]["both_seeds_net_collision_improve"] = same_direction
        direction["direction_pass"] = all(direction["checks"].values())
        target = (
            bool(direction["direction_pass"])
            and int(pooled["collision"]) <= 33
            and int(pooled["terminal_overtake"]) >= 276
            and all(bool(variants[variant]["per_seed_point_target_pass"]) for variant in seed_variants)
        )
        pooled.update(
            {
                "seed_variants": seed_variants,
                "direction_verdict": direction,
                "opened_dev_point_target_pass": target,
                "verdict_label": (
                    "OPENED_DEV_KPI_POINT_TARGET_HIT"
                    if target
                    else (
                        "DEVELOPMENT_SURVIVOR"
                        if direction["direction_pass"]
                        else "FAILED_DIRECTION_GATE"
                    )
                ),
                "l4_cluster_bootstrap": _cluster_bootstrap(arm_rows, f"{arm}::pooled"),
            }
        )
        arms[arm] = pooled
    return {
        "schema": PPO_EVAL_MERGE_SCHEMA,
        "integrity_passed": True,
        "opened_development_only": True,
        "fresh_pool_opened": False,
        "bc_baseline": {
            "episodes": EXPECTED_SCENARIOS,
            "collision": bc_collision,
            "terminal_overtake": bc_overtake,
        },
        "variants": variants,
        "arms_pooled": arms,
        "any_opened_dev_point_target_hit": any(
            bool(value["opened_dev_point_target_pass"]) for value in arms.values()
        ),
        "arm_selection_performed": False,
    }


def merge_evaluation_shards(
    *,
    shards: Sequence[EvaluationShard],
    task8_rows: Sequence[Mapping[str, str]],
    scenario_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    bc_checkpoint_sha256: str,
    checkpoints: Sequence[CandidateCheckpoint],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Strictly merge the complete 288 x 7 paired evaluation Cartesian product."""

    validate_task8_rows(task8_rows)
    expected_checkpoints = _expected_checkpoint_inventory(
        checkpoints, checkpoint_manifest_sha256, bc_checkpoint_sha256
    )
    if not shards:
        raise ValueError("B2 evaluation merge requires shards")
    shard_count = shards[0].shard_count
    if shard_count != len(shards) or {shard.shard_index for shard in shards} != set(
        range(shard_count)
    ):
        raise ValueError("B2 evaluation shard inventory is incomplete or duplicated")
    rows: list[dict[str, object]] = []
    source_shard: dict[tuple[int, str], int] = {}
    for shard in shards:
        if (
            shard.shard_count != shard_count
            or shard.scenario_manifest_sha256 != scenario_manifest_sha256
            or shard.checkpoint_manifest_sha256 != checkpoint_manifest_sha256
            or shard.bc_checkpoint_sha256 != bc_checkpoint_sha256
            or dict(shard.checkpoint_sha256_by_variant) != expected_checkpoints
        ):
            raise ValueError("B2 evaluation shard manifest/checkpoint mismatch")
        for row in shard.rows:
            if row.get("schema") != PPO_EVAL_RESULT_SCHEMA:
                raise ValueError("B2 evaluation result schema mismatch")
            key = (int(row["task8_row_index"]), str(row["variant"]))
            if key in source_shard:
                raise ValueError(f"duplicate B2 evaluation result: {key}")
            source_shard[key] = shard.shard_index
            if key[0] % shard_count != shard.shard_index:
                raise ValueError("B2 evaluation row came from the wrong physical shard")
            rows.append(dict(row))

    expected_variants = set(expected_checkpoints)
    expected_keys = {
        (physical_index, variant)
        for physical_index in range(EXPECTED_SCENARIOS)
        for variant in expected_variants
    }
    observed_keys = {(int(row["task8_row_index"]), str(row["variant"])) for row in rows}
    if len(rows) != EXPECTED_RESULTS or observed_keys != expected_keys:
        raise ValueError("B2 evaluation Cartesian product is missing or extra")

    by_key = {
        (int(row["task8_row_index"]), str(row["variant"])): row for row in rows
    }
    specs = {spec.variant: spec for spec in validate_checkpoint_specs(checkpoints, checkpoint_manifest_sha256)}
    for physical_index, case in enumerate(task8_rows):
        baseline = by_key[(physical_index, BC_VARIANT)]
        for variant in expected_variants:
            row = by_key[(physical_index, variant)]
            if (
                row["l2_id"] != case["l2_id"]
                or row["manifest_order"] != case["manifest_order"]
                or row["scenario_manifest_sha256"] != scenario_manifest_sha256
                or row["checkpoint_manifest_sha256"] != checkpoint_manifest_sha256
                or row["checkpoint_sha256"] != expected_checkpoints[variant]
            ):
                raise ValueError("B2 evaluation row identity/checkpoint mismatch")
            if variant == BC_VARIANT:
                if row["arm"] != BC_VARIANT or int(row["seed"]) != -1:
                    raise ValueError("B2 BC row identity mismatch")
            else:
                spec = specs[variant]
                if row["arm"] != spec.arm or int(row["seed"]) != spec.seed:
                    raise ValueError("B2 candidate row arm/seed mismatch")
                if row["deterministic_mode"] != DETERMINISTIC_CENTERED:
                    raise ValueError("B2 primary result is not centered deterministic")
                if row["standard_mode_is_diagnostic_only"] is not True:
                    raise ValueError("B2 standard mode was not marked diagnostic-only")
            if int(row["external_clip_micro_steps"]) != 0:
                raise ValueError("B2 evaluation contains external action clipping")
            _check_row_pairing(row, baseline)

    rows.sort(key=lambda row: (int(row["task8_row_index"]), str(row["variant"])))
    summary = summarize_paired_results(rows)
    summary.update(
        {
            "scenario_manifest_sha256": scenario_manifest_sha256,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "scenario_count": EXPECTED_SCENARIOS,
            "variant_count": EXPECTED_VARIANTS,
            "result_count": EXPECTED_RESULTS,
            "shard_count": shard_count,
        }
    )
    return rows, summary

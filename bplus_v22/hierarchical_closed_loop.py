"""Replacement Task-10 hierarchical warm-start closed-loop diagnostics.

The historical failed Task-10 artifact remains immutable.  This runner uses a
new Task-6 hierarchical checkpoint release and evaluates BC plus full,
steer-off, and brake-off executions for every arm.  A macro latent is held for
K=10, while bound-preserving composition is recomputed against the current BC
command at every 100 Hz micro-step.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from bplus_v22 import ARMS, HISTORY_OFFSETS, LIDAR_BEAMS, MACRO_STEPS, OWNER_DECISION
from bplus_v22.closed_loop import (
    OOFAlarmBank,
    OOF_OUTPUT_MANIFEST_SHA256,
    OOF_RELPATH,
)
from bplus_v22.hierarchical_checkpoint_preflight import (
    BC_MODEL_RELPATH,
    _validate_output_inventory,
    load_hierarchical_warmstart_release,
)
from bplus_v22.manifests import (
    METADATA_RELPATH,
    METADATA_SHA256,
    REGISTRY_RELPATH,
    validate_manifest_release,
)
from bplus_v22.release import file_sha256, validate_source_preflight
from bplus_v22.remediated_model import (
    ACTION_SCHEMA,
    CHECKPOINT_SCHEMA,
    HierarchicalResidualAction,
    HierarchicalResidualDistribution,
    RemediatedV22Policy,
)
from d0.outcomes import OutcomeRecord
from d25.oracle import ARRAY_KEYS, compare_archived, load_bc_model, simulate_episode
from d25.search import trajectory_digest


MODE_FULL = "full"
MODE_STEER_OFF = "steer_off"
MODE_BRAKE_OFF = "brake_off"
MODES = (MODE_FULL, MODE_STEER_OFF, MODE_BRAKE_OFF)
TASK10_SCHEMA = "bplus-v2.2-hierarchical-task10-config-1"
TASK10_SUMMARY_SCHEMA = "bplus-v2.2-hierarchical-task10-summary-1"
TASK10_VALIDATION_SCHEMA = "bplus-v2.2-hierarchical-task10-validation-1"

RESULT_FIELDS = (
    "manifest_order",
    "panel",
    "l2_id",
    "l4_id",
    "map_name",
    "skill",
    "outer_fold",
    "arm",
    "mode",
    "variant",
    "checkpoint_sha256",
    "trajectory_sha256",
    "four_state",
    "collision_any",
    "ego_collision",
    "terminal_overtake",
    "confirmed_safe_pass",
    "interaction_attempt",
    "baseline_oof_alarm_100hz",
    "bc_four_state",
    "transition",
    "fixed_collision",
    "new_collision",
    "gained_overtake",
    "lost_overtake",
    "collision_to_confirmed_pass",
    "macro_decisions",
    "intervention_decisions",
    "intervention_decision_rate_hex",
    "episode_intervened",
    "brake_decisions",
    "brake_decision_rate_hex",
    "episode_braked",
    "mean_brake_delta_hex",
    "max_brake_delta_hex",
    "mean_abs_steer_delta_hex",
    "max_abs_steer_delta_hex",
    "external_clip_micro_steps",
    "episode_external_clipped",
    "oof_alarm_decisions",
    "episode_oof_alarm",
    "alarm_intervention_cell",
    "baseline_alarm_intervention_cell",
    "shadow_alarm_0p5_decisions",
    "episode_shadow_alarm_0p5",
    "micro_steps",
    "short_terminal_macro",
)

DECISION_FIELDS = (
    "l2_id",
    "arm",
    "mode",
    "variant",
    "macro_index",
    "micro_start",
    "natural_intervention_gate",
    "natural_steer_latent_hex",
    "natural_brake_gate",
    "natural_brake_latent_hex",
    "effective_intervention_gate",
    "effective_steer_latent_hex",
    "effective_brake_gate",
    "effective_brake_latent_hex",
    "intervention_probability_hex",
    "conditional_brake_probability_hex",
    "natural_log_prob_hex",
    "requested_steer_delta_hex",
    "requested_speed_delta_hex",
    "micro_count",
    "composition_sha256",
    "mean_applied_steer_delta_hex",
    "mean_abs_applied_steer_delta_hex",
    "max_abs_applied_steer_delta_hex",
    "mean_applied_speed_delta_hex",
    "max_brake_delta_hex",
    "min_negative_steer_headroom_hex",
    "min_positive_steer_headroom_hex",
    "min_brake_headroom_hex",
    "external_clip_micro_steps",
    "shadow_ego_1s_raw_hex",
    "shadow_ttc_hex",
    "shadow_alarm_0p5",
    "oof_ego_1s_raw_hex",
    "oof_ego_1s_calibrated_hex",
    "oof_threshold_hex",
    "oof_alarm",
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_output_manifest(directory: Path) -> None:
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    )
    (directory / "output_manifest.sha256").write_text(
        "\n".join(
            f"{file_sha256(path)}  {path.relative_to(directory).as_posix()}" for path in paths
        ) + "\n",
        encoding="utf-8",
    )


def _bool(value) -> str:
    return str(bool(value)).lower()


def _variant(arm: str, mode: str) -> str:
    return f"{arm}::{mode}"


def _ablate(action: HierarchicalResidualAction, mode: str) -> HierarchicalResidualAction:
    zero = torch.zeros_like(action.steer_latent)
    if mode == MODE_FULL:
        return action
    if mode == MODE_STEER_OFF:
        return HierarchicalResidualAction(
            action.intervention_gate, zero, action.brake_gate, action.brake_latent
        )
    if mode == MODE_BRAKE_OFF:
        return HierarchicalResidualAction(action.intervention_gate, action.steer_latent, zero, zero)
    raise ValueError(f"unknown hierarchical Task-10 ablation: {mode}")


class HierarchicalClosedLoopActor(nn.Module):
    """Deterministic hierarchical actor with per-micro headroom projection."""

    def __init__(self, policy: RemediatedV22Policy, alarm_bank: nn.Module, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown hierarchical Task-10 mode: {mode}")
        self.policy = policy
        self.alarm_bank = alarm_bank
        self.mode = mode
        self.reset_runtime()

    @property
    def gru(self):
        return self.policy.bc.gru

    def reset_runtime(self, outer_fold: int | None = None) -> None:
        self.outer_fold = outer_fold
        self.micro_steps = 0
        self.macro_decisions = 0
        self.external_clip_micro_steps = 0
        self.records: list[dict] = []
        self._held_action: HierarchicalResidualAction | None = None
        self._lidar_history: list[torch.Tensor] = []
        self._speed_history: list[torch.Tensor] = []
        self._steer_history: list[torch.Tensor] = []
        self._command_speed_history: list[torch.Tensor] = []
        self._last_applied_command: tuple[float, float] | None = None
        self._pending_actual_speed: float | None = None
        self._awaiting_applied_command = False
        self._requested_command: tuple[float, float] | None = None

    def observe_actual_speed(self, value: float) -> None:
        speed = float(value)
        if not np.isfinite(speed) or self._pending_actual_speed is not None:
            raise ValueError("hierarchical Task-10 actual-speed sequence invalid")
        self._pending_actual_speed = speed

    def observe_applied_command(self, steer: float, speed: float) -> None:
        command = (float(steer), float(speed))
        if (
            not all(np.isfinite(value) for value in command)
            or not self._awaiting_applied_command
            or self._requested_command is None
        ):
            raise RuntimeError("hierarchical Task-10 applied-command sequence invalid")
        if command != self._requested_command:
            self.external_clip_micro_steps += 1
            self.records[-1]["external_clip_micro_steps"] += 1
            raise AssertionError("hierarchical Task-10 evaluator changed a composed command")
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
            raise AssertionError("hierarchical Task-10 history shape drift")
        return lidar, scalar

    @staticmethod
    def _digest_tensor(digest, name: str, value: torch.Tensor) -> None:
        tensor = value.detach().cpu().contiguous()
        if not torch.all(torch.isfinite(tensor)):
            raise ValueError("hierarchical Task-10 composition contains nonfinite tensor")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())

    def _new_macro_record(
        self,
        distribution,
        natural: HierarchicalResidualAction,
        effective: HierarchicalResidualAction,
        shadow: Mapping[str, torch.Tensor],
        oof: Mapping,
    ) -> dict:
        requested = distribution.requested_residual(effective)
        return {
            "macro_index": self.macro_decisions,
            "micro_start": self.micro_steps,
            "natural": natural,
            "effective": effective,
            "intervention_probability": float(distribution.intervention_probability.item()),
            "conditional_brake_probability": float(distribution.brake_probability.item()),
            "natural_log_prob": float(distribution.log_prob(natural).item()),
            "requested_steer_delta": float(requested[0, 0].item()),
            "requested_speed_delta": float(requested[0, 1].item()),
            "micro_count": 0,
            "_composition_digest": hashlib.sha256(
                b"end2race:bplus-v2.2:hier-task10:composition:v1\0"
            ),
            "sum_applied_steer_delta": 0.0,
            "sum_abs_applied_steer_delta": 0.0,
            "max_abs_applied_steer_delta": 0.0,
            "sum_applied_speed_delta": 0.0,
            "max_brake_delta": 0.0,
            "min_negative_steer_headroom": float("inf"),
            "min_positive_steer_headroom": float("inf"),
            "min_brake_headroom": float("inf"),
            "external_clip_micro_steps": 0,
            "shadow_ego_1s_raw": float(shadow["collision_probability"][0, 1].item()),
            "shadow_ttc": float(shadow["ttc"][0].item()),
            "shadow_alarm_0p5": bool(shadow["collision_probability"][0, 1].item() >= 0.5),
            "oof_raw": float(oof["raw"]),
            "oof_calibrated": float(oof["calibrated"]),
            "oof_threshold": float(oof["threshold"]),
            "oof_alarm": bool(oof["alarm"]),
        }

    def forward(self, lidar, previous_speed, hidden):
        if lidar.shape != (1, 1, LIDAR_BEAMS) or previous_speed.shape != (1, 1, 1):
            raise ValueError("hierarchical Task-10 adapter requires evaluator batch one")
        if self.outer_fold is None:
            raise RuntimeError("hierarchical Task-10 outer fold unset")
        if self._pending_actual_speed is None or self._awaiting_applied_command:
            raise RuntimeError("hierarchical Task-10 observation/command sequence incomplete")
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
            distribution = self.policy.distribution(bc_feature, lidar_history, scalar_history)
            natural = distribution.deterministic()
            effective = _ablate(natural, self.mode)
            shadow = self.policy.diagnostic(bc_feature, lidar_history, scalar_history)
            oof = self.alarm_bank.evaluate(
                int(self.outer_fold), lidar_history, bc_feature, scalar_history
            )
            self._held_action = effective
            self.records.append(self._new_macro_record(distribution, natural, effective, shadow, oof))
            self.macro_decisions += 1
        if self._held_action is None:
            raise AssertionError("hierarchical Task-10 lacks held macro latent")
        ledger = HierarchicalResidualDistribution.compose(base, self._held_action)
        if torch.any(ledger.external_clip_would_change):
            raise AssertionError("hierarchical Task-10 composition requests external clipping")
        record = self.records[-1]
        digest = record["_composition_digest"]
        for name in (
            "raw_base",
            "deployed_base",
            "requested_residual",
            "negative_steer_headroom",
            "positive_steer_headroom",
            "brake_headroom",
            "applied_residual",
            "command",
        ):
            self._digest_tensor(digest, name, getattr(ledger, name))
        steer_delta = float(ledger.applied_residual[0, 0].item())
        speed_delta = float(ledger.applied_residual[0, 1].item())
        record["micro_count"] += 1
        record["sum_applied_steer_delta"] += steer_delta
        record["sum_abs_applied_steer_delta"] += abs(steer_delta)
        record["max_abs_applied_steer_delta"] = max(
            record["max_abs_applied_steer_delta"], abs(steer_delta)
        )
        record["sum_applied_speed_delta"] += speed_delta
        record["max_brake_delta"] = max(record["max_brake_delta"], -speed_delta)
        record["min_negative_steer_headroom"] = min(
            record["min_negative_steer_headroom"],
            float(ledger.negative_steer_headroom.item()),
        )
        record["min_positive_steer_headroom"] = min(
            record["min_positive_steer_headroom"],
            float(ledger.positive_steer_headroom.item()),
        )
        record["min_brake_headroom"] = min(
            record["min_brake_headroom"], float(ledger.brake_headroom.item())
        )
        command = ledger.command
        self._requested_command = (float(command[0, 0].item()), float(command[0, 1].item()))
        self._awaiting_applied_command = True
        self.micro_steps += 1
        return command.unsqueeze(1), next_hidden

    def accounting(self) -> dict:
        full, remainder = divmod(self.micro_steps, MACRO_STEPS)
        lengths = [MACRO_STEPS] * full + ([remainder] if remainder else [])
        if (
            self.micro_steps <= 0
            or len(lengths) != self.macro_decisions
            or [row["micro_count"] for row in self.records] != lengths
            or self.external_clip_micro_steps != 0
        ):
            raise AssertionError("hierarchical Task-10 macro/composition accounting failed")
        return {
            "micro_steps": self.micro_steps,
            "macro_decisions": self.macro_decisions,
            "macro_lengths": lengths,
            "external_clip_micro_steps": self.external_clip_micro_steps,
        }


def _decision_row(l2_id: str, arm: str, mode: str, row: Mapping) -> dict[str, str]:
    natural = row["natural"]
    effective = row["effective"]
    count = int(row["micro_count"])
    return {
        "l2_id": l2_id,
        "arm": arm,
        "mode": mode,
        "variant": _variant(arm, mode),
        "macro_index": str(row["macro_index"]),
        "micro_start": str(row["micro_start"]),
        "natural_intervention_gate": str(int(natural.intervention_gate.item())),
        "natural_steer_latent_hex": float(natural.steer_latent.item()).hex(),
        "natural_brake_gate": str(int(natural.brake_gate.item())),
        "natural_brake_latent_hex": float(natural.brake_latent.item()).hex(),
        "effective_intervention_gate": str(int(effective.intervention_gate.item())),
        "effective_steer_latent_hex": float(effective.steer_latent.item()).hex(),
        "effective_brake_gate": str(int(effective.brake_gate.item())),
        "effective_brake_latent_hex": float(effective.brake_latent.item()).hex(),
        "intervention_probability_hex": float(row["intervention_probability"]).hex(),
        "conditional_brake_probability_hex": float(row["conditional_brake_probability"]).hex(),
        "natural_log_prob_hex": float(row["natural_log_prob"]).hex(),
        "requested_steer_delta_hex": float(row["requested_steer_delta"]).hex(),
        "requested_speed_delta_hex": float(row["requested_speed_delta"]).hex(),
        "micro_count": str(count),
        "composition_sha256": row["_composition_digest"].hexdigest(),
        "mean_applied_steer_delta_hex": float(row["sum_applied_steer_delta"] / count).hex(),
        "mean_abs_applied_steer_delta_hex": float(
            row["sum_abs_applied_steer_delta"] / count
        ).hex(),
        "max_abs_applied_steer_delta_hex": float(row["max_abs_applied_steer_delta"]).hex(),
        "mean_applied_speed_delta_hex": float(row["sum_applied_speed_delta"] / count).hex(),
        "max_brake_delta_hex": float(row["max_brake_delta"]).hex(),
        "min_negative_steer_headroom_hex": float(row["min_negative_steer_headroom"]).hex(),
        "min_positive_steer_headroom_hex": float(row["min_positive_steer_headroom"]).hex(),
        "min_brake_headroom_hex": float(row["min_brake_headroom"]).hex(),
        "external_clip_micro_steps": str(row["external_clip_micro_steps"]),
        "shadow_ego_1s_raw_hex": float(row["shadow_ego_1s_raw"]).hex(),
        "shadow_ttc_hex": float(row["shadow_ttc"]).hex(),
        "shadow_alarm_0p5": _bool(row["shadow_alarm_0p5"]),
        "oof_ego_1s_raw_hex": float(row["oof_raw"]).hex(),
        "oof_ego_1s_calibrated_hex": float(row["oof_calibrated"]).hex(),
        "oof_threshold_hex": float(row["oof_threshold"]).hex(),
        "oof_alarm": _bool(row["oof_alarm"]),
    }


def _outcome_fields(outcome: OutcomeRecord) -> dict[str, str]:
    return {
        "four_state": outcome.four_state,
        "collision_any": _bool(outcome.collision_any),
        "ego_collision": _bool(outcome.ego_collision),
        "terminal_overtake": _bool(outcome.corrected_outcome3 == "overtake"),
        "confirmed_safe_pass": _bool(outcome.confirmed_safe_pass is True),
        "interaction_attempt": _bool(outcome.interaction_attempt is True),
    }


def _result_row(
    case: Mapping[str, str],
    outer_fold: int,
    arm: str,
    mode: str,
    checkpoint_sha: str,
    outcome: OutcomeRecord,
    digest: str,
    baseline: OutcomeRecord,
    baseline_oof_alarm: bool,
    decisions: list[Mapping] | None,
    micro_steps: int,
) -> dict[str, str]:
    bc_overtake = baseline.corrected_outcome3 == "overtake"
    candidate_overtake = outcome.corrected_outcome3 == "overtake"
    if decisions:
        intervention = int(sum(int(row["effective"].intervention_gate.item()) for row in decisions))
        brake_decisions = int(sum(int(row["effective"].brake_gate.item()) for row in decisions))
        total_micro = sum(int(row["micro_count"]) for row in decisions)
        mean_brake = sum(-float(row["sum_applied_speed_delta"]) for row in decisions) / total_micro
        max_brake = max(float(row["max_brake_delta"]) for row in decisions)
        mean_abs_steer = sum(
            float(row["sum_abs_applied_steer_delta"]) for row in decisions
        ) / total_micro
        max_abs_steer = max(float(row["max_abs_applied_steer_delta"]) for row in decisions)
        clips = sum(int(row["external_clip_micro_steps"]) for row in decisions)
        oof_alarms = sum(bool(row["oof_alarm"]) for row in decisions)
        shadow_alarms = sum(bool(row["shadow_alarm_0p5"]) for row in decisions)
        lengths = [int(row["micro_count"]) for row in decisions]
    else:
        intervention = brake_decisions = clips = oof_alarms = shadow_alarms = 0
        mean_brake = max_brake = mean_abs_steer = max_abs_steer = 0.0
        lengths = []
    episode_intervened = intervention > 0
    episode_braked = brake_decisions > 0
    episode_alarm = oof_alarms > 0
    variant = "BC" if arm == "BC" else _variant(arm, mode)
    return {
        "manifest_order": case["manifest_order"],
        "panel": case["panel"],
        "l2_id": case["l2_id"],
        "l4_id": case["l4_id"],
        "map_name": case["map_name"],
        "skill": case["skill"],
        "outer_fold": str(outer_fold),
        "arm": arm,
        "mode": mode,
        "variant": variant,
        "checkpoint_sha256": checkpoint_sha,
        "trajectory_sha256": digest,
        **_outcome_fields(outcome),
        "baseline_oof_alarm_100hz": _bool(baseline_oof_alarm),
        "bc_four_state": baseline.four_state,
        "transition": f"{baseline.four_state}->{outcome.four_state}",
        "fixed_collision": _bool(baseline.collision_any and not outcome.collision_any),
        "new_collision": _bool(not baseline.collision_any and outcome.collision_any),
        "gained_overtake": _bool(not bc_overtake and candidate_overtake),
        "lost_overtake": _bool(bc_overtake and not candidate_overtake),
        "collision_to_confirmed_pass": _bool(
            baseline.collision_any and outcome.confirmed_safe_pass is True
        ),
        "macro_decisions": str(len(decisions or [])),
        "intervention_decisions": str(intervention),
        "intervention_decision_rate_hex": float(
            intervention / len(decisions) if decisions else 0.0
        ).hex(),
        "episode_intervened": _bool(episode_intervened),
        "brake_decisions": str(brake_decisions),
        "brake_decision_rate_hex": float(
            brake_decisions / len(decisions) if decisions else 0.0
        ).hex(),
        "episode_braked": _bool(episode_braked),
        "mean_brake_delta_hex": float(mean_brake).hex(),
        "max_brake_delta_hex": float(max_brake).hex(),
        "mean_abs_steer_delta_hex": float(mean_abs_steer).hex(),
        "max_abs_steer_delta_hex": float(max_abs_steer).hex(),
        "external_clip_micro_steps": str(clips),
        "episode_external_clipped": _bool(clips > 0),
        "oof_alarm_decisions": str(oof_alarms),
        "episode_oof_alarm": _bool(episode_alarm),
        "alarm_intervention_cell": f"alarm{int(episode_alarm)}_intervene{int(episode_intervened)}",
        "baseline_alarm_intervention_cell": (
            f"alarm{int(baseline_oof_alarm)}_intervene{int(episode_intervened)}"
        ),
        "shadow_alarm_0p5_decisions": str(shadow_alarms),
        "episode_shadow_alarm_0p5": _bool(shadow_alarms > 0),
        "micro_steps": str(micro_steps),
        "short_terminal_macro": _bool(bool(lengths and lengths[-1] < MACRO_STEPS)),
    }


def _summarize(rows: list[dict[str, str]]) -> dict:
    variants: dict[str, dict] = {}
    for arm in ARMS:
        for mode in MODES:
            name = _variant(arm, mode)
            selected = [row for row in rows if row["variant"] == name]
            transitions: dict[str, int] = {}
            for row in selected:
                transitions[row["transition"]] = transitions.get(row["transition"], 0) + 1
            macro = sum(int(row["macro_decisions"]) for row in selected)
            interventions = sum(int(row["intervention_decisions"]) for row in selected)
            brakes = sum(int(row["brake_decisions"]) for row in selected)
            variants[name] = {
                "arm": arm,
                "mode": mode,
                "episodes": len(selected),
                "collision": sum(row["collision_any"] == "true" for row in selected),
                "terminal_overtake": sum(row["terminal_overtake"] == "true" for row in selected),
                "confirmed_safe_pass": sum(
                    row["confirmed_safe_pass"] == "true" for row in selected
                ),
                "fixed_collision": sum(row["fixed_collision"] == "true" for row in selected),
                "new_collision": sum(row["new_collision"] == "true" for row in selected),
                "gained_overtake": sum(row["gained_overtake"] == "true" for row in selected),
                "lost_overtake": sum(row["lost_overtake"] == "true" for row in selected),
                "collision_to_confirmed_pass": sum(
                    row["collision_to_confirmed_pass"] == "true" for row in selected
                ),
                "episodes_intervened": sum(
                    row["episode_intervened"] == "true" for row in selected
                ),
                "episode_intervention_rate": sum(
                    row["episode_intervened"] == "true" for row in selected
                ) / len(selected),
                "intervention_decisions": interventions,
                "intervention_decision_rate": interventions / macro,
                "episodes_braked": sum(row["episode_braked"] == "true" for row in selected),
                "brake_decisions": brakes,
                "brake_decision_rate": brakes / macro,
                "safe_bc_episodes": sum(row["bc_four_state"] != "collision" for row in selected),
                "safe_bc_intervened_episodes": sum(
                    row["bc_four_state"] != "collision" and row["episode_intervened"] == "true"
                    for row in selected
                ),
                "safe_bc_intervened_then_lost_overtake": sum(
                    row["bc_four_state"] != "collision"
                    and row["episode_intervened"] == "true"
                    and row["lost_overtake"] == "true"
                    for row in selected
                ),
                "external_clip_micro_steps": sum(
                    int(row["external_clip_micro_steps"]) for row in selected
                ),
                "globally_intervenes": all(
                    row["episode_intervened"] == "true" for row in selected
                ),
                "net_overtake_loss": sum(row["lost_overtake"] == "true" for row in selected)
                > sum(row["gained_overtake"] == "true" for row in selected),
                "transitions": dict(sorted(transitions.items())),
            }
    full = [variants[_variant(arm, MODE_FULL)] for arm in ARMS]
    shared_stop = {
        "any_external_action_clipping": any(
            value["external_clip_micro_steps"] > 0 for value in variants.values()
        ),
        "all_full_arms_globally_intervene": all(value["globally_intervenes"] for value in full),
        "all_full_arms_net_overtake_loss": all(value["net_overtake_loss"] for value in full),
    }
    return {
        "schema": TASK10_SUMMARY_SCHEMA,
        "interpretation": "mechanism_and_within_opened_development_only_no_l4_generalization",
        "arm_selection_performed": False,
        "ppo_authorized": False,
        "variants": variants,
        "shared_stop": shared_stop,
        "task10_mechanism_passed": not any(shared_stop.values()),
    }


def _baseline_oof_alarms(root: Path, cases: list[dict[str, str]], alarm_bank: OOFAlarmBank):
    metadata = _read_tsv(root / METADATA_RELPATH)
    if file_sha256(root / METADATA_RELPATH) != METADATA_SHA256:
        raise ValueError("hierarchical Task-10 metadata hash drift")
    by_l2 = {row["l2_id"]: row for row in metadata}
    dataset_manifest = json.loads(
        (root / METADATA_RELPATH).with_name("dataset_manifest.json").read_text(encoding="utf-8")
    )
    valid_entry = dataset_manifest["arrays"]["ego_valid_100"]
    valid_path = (root / METADATA_RELPATH).parent / valid_entry["relpath"]
    if file_sha256(valid_path) != valid_entry["sha256"]:
        raise ValueError("hierarchical Task-10 ego-valid array hash drift")
    ego_valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
    predictions = np.load(root / OOF_RELPATH / "oof_predictions.npy", mmap_mode="r", allow_pickle=False)
    alarms = {}
    for case in cases:
        row = by_l2[case["l2_id"]]
        start = int(row["frame_start"])
        stop = start + int(row["frame_count"])
        fold = int(row["outer_fold"])
        valid = np.asarray(ego_valid[start:stop], dtype=bool)
        probability = np.asarray(predictions[start:stop, 1])
        alarms[case["l2_id"]] = bool(
            np.any(valid & (probability >= alarm_bank.thresholds[fold]))
        )
    return by_l2, alarms


def run_hierarchical_closed_loop(
    repo_root: str | Path,
    source_preflight_dir: str | Path,
    manifest_release_dir: str | Path,
    warmstart_release_dir: str | Path,
    warmstart_output_manifest_sha256: str,
    output_dir: str | Path,
    created_at: str,
    device_name: str = "cuda:0",
) -> dict:
    root = Path(repo_root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("hierarchical Task-10 runner must execute from repository root")
    cache = os.environ.get("NUMBA_CACHE_DIR")
    if not cache or not Path(cache).is_absolute():
        raise ValueError("hierarchical Task-10 requires an isolated absolute NUMBA_CACHE_DIR")
    if not validate_source_preflight(source_preflight_dir, root)["passed"]:
        raise ValueError("hierarchical Task-10 source preflight invalid")
    if not validate_manifest_release(manifest_release_dir, root)["passed"]:
        raise ValueError("hierarchical Task-10 manifest release invalid")
    payloads, checkpoint_sha, warmstart_config = load_hierarchical_warmstart_release(
        warmstart_release_dir, warmstart_output_manifest_sha256, root
    )
    registry_sha256 = str(warmstart_config.get("registry", {}).get("after_sha256", ""))
    if (
        len(registry_sha256) != 64
        or any(character not in "0123456789abcdef" for character in registry_sha256)
        or file_sha256(root / REGISTRY_RELPATH) != registry_sha256
    ):
        raise ValueError("hierarchical Task-10 registry hash drift")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("hierarchical Task-10 CUDA requested but unavailable")
    device = torch.device(device_name)
    cases = _read_tsv(Path(manifest_release_dir) / "development_scenarios.tsv")
    if len(cases) != 288:
        raise ValueError("hierarchical Task-10 requires exactly 288 development scenarios")
    alarm_bank = OOFAlarmBank(root, device)
    _validate_output_inventory(root / OOF_RELPATH)
    metadata_by_l2, baseline_alarm = _baseline_oof_alarms(root, cases, alarm_bank)

    output = Path(output_dir)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("hierarchical Task-10 output/partial exists")
    partial.mkdir(parents=True)
    try:
        rows: list[dict[str, str]] = []
        decisions: list[dict[str, str]] = []
        baseline_outcomes: dict[str, OutcomeRecord] = {}
        bc = load_bc_model(str(root / BC_MODEL_RELPATH), device)
        for index, case in enumerate(cases):
            source_path = root / case["npz_relpath"]
            if file_sha256(source_path) != case["npz_sha256"]:
                raise ValueError(f"hierarchical Task-10 archive hash drift: {case['l2_id']}")
            with np.load(source_path, allow_pickle=False) as archive:
                archived = {name: np.asarray(archive[name]) for name in archive.files}
            result = simulate_episode(bc, device, case)
            if not compare_archived(result.arrays, archived)["passed"] or result.action_clipped:
                raise AssertionError(f"hierarchical Task-10 BC replay mismatch: {case['l2_id']}")
            baseline_outcomes[case["l2_id"]] = result.outcome
            fold = int(metadata_by_l2[case["l2_id"]]["outer_fold"])
            rows.append(
                _result_row(
                    case,
                    fold,
                    "BC",
                    "baseline",
                    "NA",
                    result.outcome,
                    trajectory_digest(result.arrays),
                    result.outcome,
                    baseline_alarm[case["l2_id"]],
                    None,
                    len(result.arrays["time"]),
                )
            )
            if (index + 1) % 24 == 0:
                print(f"HIER_TASK10_PROGRESS variant=BC episodes={index + 1}/288", flush=True)
        del bc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for arm in ARMS:
            policy = RemediatedV22Policy(arm).to(device)
            policy.load_hierarchical_state_dict(payloads[arm]["state_dict"])
            policy.eval()
            for mode in MODES:
                actor = HierarchicalClosedLoopActor(policy, alarm_bank, mode).to(device).eval()
                for index, case in enumerate(cases):
                    fold = int(metadata_by_l2[case["l2_id"]]["outer_fold"])
                    actor.reset_runtime(fold)
                    result = simulate_episode(actor, device, case)
                    accounting = actor.accounting()
                    if (
                        accounting["micro_steps"] != len(result.arrays["time"])
                        or accounting["external_clip_micro_steps"] != 0
                        or result.action_clipped
                    ):
                        raise AssertionError("hierarchical Task-10 composition continuity failed")
                    rows.append(
                        _result_row(
                            case,
                            fold,
                            arm,
                            mode,
                            checkpoint_sha[arm],
                            result.outcome,
                            trajectory_digest(result.arrays),
                            baseline_outcomes[case["l2_id"]],
                            baseline_alarm[case["l2_id"]],
                            actor.records,
                            accounting["micro_steps"],
                        )
                    )
                    decisions.extend(
                        _decision_row(case["l2_id"], arm, mode, record)
                        for record in actor.records
                    )
                    if (index + 1) % 24 == 0:
                        print(
                            f"HIER_TASK10_PROGRESS variant={_variant(arm, mode)} "
                            f"episodes={index + 1}/288",
                            flush=True,
                        )
                del actor
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del policy
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        _write_tsv(partial / "episode_results.tsv", rows, RESULT_FIELDS)
        _write_tsv(partial / "macro_decisions.tsv", decisions, DECISION_FIELDS)
        summary = _summarize(rows)
        _write_json(partial / "task10_summary.json", summary)
        _write_json(
            partial / "config.json",
            {
                "schema": TASK10_SCHEMA,
                "created_at": str(created_at),
                "owner_decision": OWNER_DECISION,
                "action_schema": ACTION_SCHEMA,
                "checkpoint_schema": CHECKPOINT_SCHEMA,
                "interpretation": "mechanism_and_within_opened_development_only_no_l4_generalization",
                "held_out_policy_generalization": False,
                "arm_selection_performed": False,
                "ppo_authorized": False,
                "policy_training_started": False,
                "ppo_training_started": False,
                "test_opened": False,
                "final_pool": False,
                "device": str(device),
                "numba_cache_dir": cache,
                "cases": len(cases),
                "modes": list(MODES),
                "variants": ["BC"] + [_variant(arm, mode) for arm in ARMS for mode in MODES],
                "episode_rows": len(rows),
                "macro_decision_rows": len(decisions),
                "source_preflight_relpath": str(Path(source_preflight_dir).resolve()),
                "source_preflight_output_manifest_sha256": file_sha256(
                    Path(source_preflight_dir) / "output_manifest.sha256"
                ),
                "manifest_release_relpath": str(Path(manifest_release_dir).resolve()),
                "manifest_release_output_manifest_sha256": file_sha256(
                    Path(manifest_release_dir) / "output_manifest.sha256"
                ),
                "warmstart_release_relpath": str(Path(warmstart_release_dir).resolve()),
                "warmstart_release_output_manifest_sha256": warmstart_output_manifest_sha256,
                "checkpoint_sha256": checkpoint_sha,
                "oof_diagnostic_release_relpath": OOF_RELPATH,
                "oof_diagnostic_output_manifest_sha256": OOF_OUTPUT_MANIFEST_SHA256,
                "composition_semantics": "one natural 4D macro latent held for K=10; same latent projected against each current 100Hz BC base headroom",
                "ablation_semantics": {
                    MODE_FULL: "natural deterministic hierarchical action",
                    MODE_STEER_OFF: "same intervention/brake latent with steer latent forced exactly zero",
                    MODE_BRAKE_OFF: "same intervention/steer latent with brake gate and latent forced exactly zero",
                },
                "registry_sha256": registry_sha256,
            },
        )
        _write_json(
            partial / "validation.json",
            {
                "schema": TASK10_VALIDATION_SCHEMA,
                "passed": summary["task10_mechanism_passed"],
                "integrity_passed": True,
                "ppo_authorized": False,
                "shared_stop": summary["shared_stop"],
                "violations": [],
            },
        )
        _write_output_manifest(partial)
        (partial / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    validation = validate_hierarchical_closed_loop(output)
    return validation | {"output_manifest_sha256": file_sha256(output / "output_manifest.sha256")}


def validate_hierarchical_closed_loop(release_dir: str | Path) -> dict:
    release = Path(release_dir)
    violations: list[str] = []
    rows: list[dict[str, str]] = []
    decisions: list[dict[str, str]] = []
    mechanism_passed = False
    try:
        if not (release / "COMPLETE").is_file():
            raise ValueError("hierarchical Task-10 release lacks COMPLETE")
        _validate_output_inventory(release)
        config = json.loads((release / "config.json").read_text(encoding="utf-8"))
        checkpoint_sha = config.get("checkpoint_sha256")
        registry_sha256 = str(config.get("registry_sha256", ""))
        if (
            config.get("schema") != TASK10_SCHEMA
            or config.get("owner_decision") != OWNER_DECISION
            or config.get("action_schema") != ACTION_SCHEMA
            or config.get("checkpoint_schema") != CHECKPOINT_SCHEMA
            or config.get("held_out_policy_generalization") is not False
            or config.get("arm_selection_performed") is not False
            or config.get("ppo_authorized") is not False
            or config.get("modes") != list(MODES)
            or not isinstance(checkpoint_sha, dict)
            or set(checkpoint_sha) != set(ARMS)
            or len(registry_sha256) != 64
            or any(character not in "0123456789abcdef" for character in registry_sha256)
            or any(
                config.get(name) is not False
                for name in ("policy_training_started", "ppo_training_started", "test_opened", "final_pool")
            )
        ):
            raise ValueError("hierarchical Task-10 authority/scope mismatch")
        with (release / "episode_results.tsv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
                raise ValueError("hierarchical Task-10 episode header drift")
            rows = list(reader)
        with (release / "macro_decisions.tsv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != DECISION_FIELDS:
                raise ValueError("hierarchical Task-10 decision header drift")
            decisions = list(reader)
        expected_variants = ["BC"] + [_variant(arm, mode) for arm in ARMS for mode in MODES]
        if len(rows) != 288 * len(expected_variants):
            raise ValueError("hierarchical Task-10 episode Cartesian product incomplete")
        manifest_rows = _read_tsv(Path(config["manifest_release_relpath"]) / "development_scenarios.tsv")
        ordered_l2 = [row["l2_id"] for row in manifest_rows]
        expected = {(l2_id, variant) for l2_id in ordered_l2 for variant in expected_variants}
        if {(row["l2_id"], row["variant"]) for row in rows} != expected:
            raise ValueError("hierarchical Task-10 scenario/variant Cartesian mismatch")
        for variant in expected_variants:
            if [row["l2_id"] for row in rows if row["variant"] == variant] != ordered_l2:
                raise ValueError(f"hierarchical Task-10 ordered L2 mismatch: {variant}")
        decision_counts: dict[tuple[str, str], int] = {}
        decision_micro: dict[tuple[str, str], int] = {}
        for row in decisions:
            key = (row["l2_id"], row["variant"])
            decision_counts[key] = decision_counts.get(key, 0) + 1
            decision_micro[key] = decision_micro.get(key, 0) + int(row["micro_count"])
            if row["variant"] != _variant(row["arm"], row["mode"]):
                raise ValueError("hierarchical Task-10 decision variant mismatch")
            natural_gates = (
                int(row["natural_intervention_gate"]),
                int(row["natural_brake_gate"]),
            )
            effective_gates = (
                int(row["effective_intervention_gate"]),
                int(row["effective_brake_gate"]),
            )
            if (
                natural_gates[0] not in (0, 1)
                or natural_gates[1] not in (0, 1)
                or natural_gates[1] > natural_gates[0]
                or effective_gates[0] not in (0, 1)
                or effective_gates[1] not in (0, 1)
                or effective_gates[1] > effective_gates[0]
                or not 1 <= int(row["micro_count"]) <= MACRO_STEPS
                or len(row["composition_sha256"]) != 64
                or int(row["external_clip_micro_steps"]) != 0
            ):
                raise ValueError("hierarchical Task-10 decision accounting invalid")
            values = [float.fromhex(row[name]) for name in DECISION_FIELDS if name.endswith("_hex")]
            if not all(np.isfinite(values)):
                raise ValueError("hierarchical Task-10 decision has nonfinite value")
            if row["mode"] == MODE_STEER_OFF and (
                float.fromhex(row["effective_steer_latent_hex"]) != 0.0
                or float.fromhex(row["requested_steer_delta_hex"]) != 0.0
                or float.fromhex(row["max_abs_applied_steer_delta_hex"]) != 0.0
            ):
                raise ValueError("hierarchical Task-10 steer-off ablation leaked steering")
            if row["mode"] == MODE_BRAKE_OFF and (
                row["effective_brake_gate"] != "0"
                or float.fromhex(row["effective_brake_latent_hex"]) != 0.0
                or float.fromhex(row["requested_speed_delta_hex"]) != 0.0
                or float.fromhex(row["max_brake_delta_hex"]) != 0.0
            ):
                raise ValueError("hierarchical Task-10 brake-off ablation leaked braking")
        for row in rows:
            key = (row["l2_id"], row["variant"])
            if row["variant"] == "BC":
                if row["checkpoint_sha256"] != "NA" or row["macro_decisions"] != "0":
                    raise ValueError("hierarchical Task-10 BC row has policy accounting")
            else:
                if (
                    row["checkpoint_sha256"] != checkpoint_sha[row["arm"]]
                    or decision_counts.get(key, 0) != int(row["macro_decisions"])
                    or decision_micro.get(key, 0) != int(row["micro_steps"])
                    or row["episode_external_clipped"] != "false"
                    or int(row["external_clip_micro_steps"]) != 0
                ):
                    raise ValueError("hierarchical Task-10 episode/decision continuity mismatch")
        summary = json.loads((release / "task10_summary.json").read_text(encoding="utf-8"))
        if summary != _summarize(rows):
            raise ValueError("hierarchical Task-10 summary recomputation mismatch")
        saved = json.loads((release / "validation.json").read_text(encoding="utf-8"))
        mechanism_passed = bool(summary["task10_mechanism_passed"])
        if (
            saved.get("schema") != TASK10_VALIDATION_SCHEMA
            or saved.get("integrity_passed") is not True
            or saved.get("ppo_authorized") is not False
            or saved.get("shared_stop") != summary["shared_stop"]
            or saved.get("passed") is not mechanism_passed
        ):
            raise ValueError("hierarchical Task-10 saved validation mismatch")
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": TASK10_VALIDATION_SCHEMA,
        "passed": not violations and mechanism_passed,
        "integrity_passed": not violations,
        "task10_mechanism_passed": mechanism_passed,
        "ppo_authorized": False,
        "episode_rows": len(rows),
        "macro_decision_rows": len(decisions),
        "violations": violations,
    }

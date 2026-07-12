"""Task-10 deterministic warm-start closed-loop diagnostic evaluation."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn as nn

from bplus_v22 import ARMS, LIDAR_BEAMS, MACRO_STEPS, OWNER_DECISION
from bplus_v22.checkpoint_preflight import (
    CHECKPOINT_SHA256,
    WARMSTART_OUTPUT_MANIFEST_SHA256,
    WARMSTART_RELEASE_RELPATH,
    _validate_historical_warmstart,
    _validate_output_inventory,
)
from bplus_v22.identity import ZeroResidualActor
from bplus_v22.manifests import (
    METADATA_RELPATH,
    METADATA_SHA256,
    REGISTRY_RELPATH,
    REGISTRY_SHA256,
    validate_manifest_release,
)
from bplus_v22.model import V22Policy
from bplus_v22.release import file_sha256, validate_source_preflight
from d0.outcomes import OutcomeRecord
from d2.models import apply_platt_calibrator
from d2r.model import D2RGeometryNet, decode_ttc_logits
from d25.oracle import ARRAY_KEYS, classify_trajectory, simulate_episode
from d25.search import trajectory_digest


OOF_RELPATH = "logs/d2r_geometry_20260711/artifacts/full_oof_20260711_210200"
OOF_OUTPUT_MANIFEST_SHA256 = (
    "be7936acc95b9a98a3a97d4248d94b11ea8c4ed8adacc82a3dde513323b7c057"
)
RESULT_FIELDS = (
    "manifest_order",
    "panel",
    "l2_id",
    "l4_id",
    "map_name",
    "skill",
    "outer_fold",
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
    "brake_decisions",
    "brake_decision_rate_hex",
    "episode_braked",
    "mean_brake_delta_hex",
    "max_brake_delta_hex",
    "mean_abs_steer_delta_hex",
    "max_abs_steer_delta_hex",
    "clip_micro_steps",
    "episode_clipped",
    "oof_alarm_decisions",
    "episode_oof_alarm",
    "alarm_action_cell",
    "baseline_alarm_action_cell",
    "shadow_alarm_0p5_decisions",
    "episode_shadow_alarm_0p5",
    "micro_steps",
    "short_terminal_macro",
)
DECISION_FIELDS = (
    "l2_id",
    "variant",
    "macro_index",
    "micro_start",
    "brake_gate",
    "gate_probability_hex",
    "brake_delta_hex",
    "steer_delta_hex",
    "base_steer_hex",
    "base_speed_hex",
    "composed_steer_hex",
    "composed_speed_hex",
    "clip_micro_steps",
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


def _bool(value) -> str:
    return str(bool(value)).lower()


def _outcome_fields(outcome: OutcomeRecord) -> dict[str, str]:
    return {
        "four_state": outcome.four_state,
        "collision_any": _bool(outcome.collision_any),
        "ego_collision": _bool(outcome.ego_collision),
        "terminal_overtake": _bool(outcome.corrected_outcome3 == "overtake"),
        "confirmed_safe_pass": _bool(outcome.confirmed_safe_pass is True),
        "interaction_attempt": _bool(outcome.interaction_attempt is True),
    }


class OOFAlarmBank(nn.Module):
    """Five immutable outer-refit heads with their own calibration/threshold."""

    def __init__(self, root: Path, device: torch.device):
        super().__init__()
        if file_sha256(root / OOF_RELPATH / "output_manifest.sha256") != OOF_OUTPUT_MANIFEST_SHA256:
            raise ValueError("Task-10 OOF diagnostic release hash drift")
        self.models = nn.ModuleList()
        self.register_buffer("means", torch.empty(5, 1680))
        self.register_buffer("stds", torch.empty(5, 1680))
        self.calibrators: list[dict] = []
        self.thresholds: list[float] = []
        for fold in range(5):
            payload = torch.load(
                root / OOF_RELPATH / "models" / f"outer{fold}_refit.pt",
                map_location="cpu",
                weights_only=False,
            )
            model = D2RGeometryNet()
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
            self.models.append(model)
            self.means[fold].copy_(torch.as_tensor(payload["normalization_mean"]))
            self.stds[fold].copy_(torch.as_tensor(payload["normalization_std"]))
            report = json.loads(
                (root / OOF_RELPATH / f"outer{fold}_report.json").read_text(encoding="utf-8")
            )
            self.calibrators.append(report["calibrators"]["ego_probability_100"])
            self.thresholds.append(float(report["thresholds"]["ego_probability_100"]["threshold"]))
        self.to(device).eval()

    @torch.no_grad()
    def evaluate(
        self,
        fold: int,
        lidar_history: torch.Tensor,
        bc_feature: torch.Tensor,
        scalar_history: torch.Tensor,
    ) -> dict[str, float | bool]:
        if fold not in range(5):
            raise ValueError("Task-10 outer fold must be in 0..4")
        output = self.models[fold](
            lidar_history,
            (bc_feature - self.means[fold]) / self.stds[fold],
            scalar_history,
        )
        raw = float(torch.sigmoid(output["collision_logits"])[0, 1].item())
        calibrated = float(
            apply_platt_calibrator(np.asarray([raw]), self.calibrators[fold])[0]
        )
        threshold = self.thresholds[fold]
        return {
            "raw": raw,
            "calibrated": calibrated,
            "threshold": threshold,
            "alarm": calibrated >= threshold,
        }


class NaturalResidualActor(ZeroResidualActor):
    """Hold the warm-start checkpoint's natural deterministic residual for K=10."""

    def __init__(self, policy: V22Policy, alarm_bank: OOFAlarmBank):
        self.outer_fold: int | None = None
        super().__init__(policy, require_natural_zero=False)
        self.alarm_bank = alarm_bank

    def reset_runtime(self, outer_fold: int | None = None) -> None:
        super().reset_runtime()
        self.outer_fold = outer_fold
        self._held_delta: torch.Tensor | None = None
        self._requested_command: tuple[float, float] | None = None
        self.decision_records: list[dict] = []
        self.clip_micro_steps = 0

    def observe_applied_command(self, steer: float, speed: float) -> None:
        command = (float(steer), float(speed))
        if not all(np.isfinite(value) for value in command):
            raise ValueError("Task-10 applied command is nonfinite")
        if not self._awaiting_applied_command or self._requested_command is None:
            raise RuntimeError("Task-10 received an unexpected applied command")
        clipped = command != self._requested_command
        if clipped:
            self.clip_micro_steps += 1
            self.decision_records[-1]["clip_micro_steps"] += 1
        self._last_applied_command = command
        self._awaiting_applied_command = False

    def forward(self, lidar, previous_speed, hidden):
        if lidar.shape != (1, 1, LIDAR_BEAMS) or previous_speed.shape != (1, 1, 1):
            raise ValueError("Task-10 adapter supports exact batch-1 evaluator replay")
        if self.outer_fold is None:
            raise RuntimeError("Task-10 adapter outer fold is unset")
        if self._pending_actual_speed is None or self._awaiting_applied_command:
            raise RuntimeError("Task-10 observation/command sequence is incomplete")
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
            action = distribution.deterministic()
            delta = distribution.physical_delta(action)
            shadow = self.policy.diagnostic(bc_feature, lidar_history, scalar_history)
            oof = self.alarm_bank.evaluate(
                self.outer_fold, lidar_history, bc_feature, scalar_history
            )
            self._held_action = action
            self._held_delta = delta
            self.macro_decisions += 1
            self.natural_brake_decisions += int(action.brake_gate.item())
            self.max_abs_natural_residual = max(
                self.max_abs_natural_residual, float(delta.abs().max().item())
            )
            self._update_diagnostic_digest(shadow)
            composed = base + delta
            self.decision_records.append({
                "macro_index": self.macro_decisions - 1,
                "micro_start": self.micro_steps,
                "brake_gate": int(action.brake_gate.item()),
                "gate_probability": float(distribution.brake_probability.item()),
                "brake_delta": float(delta[0, 1].item()),
                "steer_delta": float(delta[0, 0].item()),
                "base_steer": float(base[0, 0].item()),
                "base_speed": float(base[0, 1].item()),
                "composed_steer": float(composed[0, 0].item()),
                "composed_speed": float(composed[0, 1].item()),
                "clip_micro_steps": 0,
                "shadow_ego_1s_raw": float(shadow["collision_probability"][0, 1].item()),
                "shadow_ttc": float(shadow["ttc"][0].item()),
                "shadow_alarm_0p5": bool(shadow["collision_probability"][0, 1].item() >= 0.5),
                "oof_raw": oof["raw"],
                "oof_calibrated": oof["calibrated"],
                "oof_threshold": oof["threshold"],
                "oof_alarm": oof["alarm"],
            })
        if self._held_delta is None:
            raise AssertionError("Task-10 adapter lacks held residual")
        composed = base + self._held_delta
        self._requested_command = (
            float(composed[0, 0].item()),
            float(composed[0, 1].item()),
        )
        self._awaiting_applied_command = True
        self.micro_steps += 1
        return composed.unsqueeze(1), next_hidden


def _decision_row(l2_id: str, arm: str, row: dict) -> dict[str, str]:
    return {
        "l2_id": l2_id,
        "variant": arm,
        "macro_index": str(row["macro_index"]),
        "micro_start": str(row["micro_start"]),
        "brake_gate": str(row["brake_gate"]),
        "gate_probability_hex": float(row["gate_probability"]).hex(),
        "brake_delta_hex": float(row["brake_delta"]).hex(),
        "steer_delta_hex": float(row["steer_delta"]).hex(),
        "base_steer_hex": float(row["base_steer"]).hex(),
        "base_speed_hex": float(row["base_speed"]).hex(),
        "composed_steer_hex": float(row["composed_steer"]).hex(),
        "composed_speed_hex": float(row["composed_speed"]).hex(),
        "clip_micro_steps": str(row["clip_micro_steps"]),
        "shadow_ego_1s_raw_hex": float(row["shadow_ego_1s_raw"]).hex(),
        "shadow_ttc_hex": float(row["shadow_ttc"]).hex(),
        "shadow_alarm_0p5": _bool(row["shadow_alarm_0p5"]),
        "oof_ego_1s_raw_hex": float(row["oof_raw"]).hex(),
        "oof_ego_1s_calibrated_hex": float(row["oof_calibrated"]).hex(),
        "oof_threshold_hex": float(row["oof_threshold"]).hex(),
        "oof_alarm": _bool(row["oof_alarm"]),
    }


def _result_row(
    case: dict[str, str],
    outer_fold: int,
    variant: str,
    checkpoint_sha: str,
    outcome: OutcomeRecord,
    digest: str,
    baseline: OutcomeRecord,
    baseline_oof_alarm: bool,
    decisions: list[dict] | None,
    micro_steps: int,
) -> dict[str, str]:
    bc_overtake = baseline.corrected_outcome3 == "overtake"
    candidate_overtake = outcome.corrected_outcome3 == "overtake"
    if decisions:
        brake = np.asarray([-float(row["brake_delta"]) for row in decisions])
        steer = np.asarray([abs(float(row["steer_delta"])) for row in decisions])
        brakes = int(sum(row["brake_gate"] for row in decisions))
        oof_alarms = int(sum(row["oof_alarm"] for row in decisions))
        shadow_alarms = int(sum(row["shadow_alarm_0p5"] for row in decisions))
        clips = int(sum(row["clip_micro_steps"] for row in decisions))
        lengths = [MACRO_STEPS] * (micro_steps // MACRO_STEPS)
        if micro_steps % MACRO_STEPS:
            lengths.append(micro_steps % MACRO_STEPS)
    else:
        brake = steer = np.asarray([], dtype=np.float64)
        brakes = oof_alarms = shadow_alarms = clips = 0
        lengths = []
    episode_braked = brakes > 0
    episode_alarm = oof_alarms > 0
    return {
        "manifest_order": case["manifest_order"],
        "panel": case["panel"],
        "l2_id": case["l2_id"],
        "l4_id": case["l4_id"],
        "map_name": case["map_name"],
        "skill": case["skill"],
        "outer_fold": str(outer_fold),
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
        "brake_decisions": str(brakes),
        "brake_decision_rate_hex": float(brakes / len(decisions)).hex() if decisions else float(0).hex(),
        "episode_braked": _bool(episode_braked),
        "mean_brake_delta_hex": float(np.mean(brake)).hex() if len(brake) else float(0).hex(),
        "max_brake_delta_hex": float(np.max(brake)).hex() if len(brake) else float(0).hex(),
        "mean_abs_steer_delta_hex": float(np.mean(steer)).hex() if len(steer) else float(0).hex(),
        "max_abs_steer_delta_hex": float(np.max(steer)).hex() if len(steer) else float(0).hex(),
        "clip_micro_steps": str(clips),
        "episode_clipped": _bool(clips > 0),
        "oof_alarm_decisions": str(oof_alarms),
        "episode_oof_alarm": _bool(episode_alarm),
        "alarm_action_cell": f"alarm{int(episode_alarm)}_brake{int(episode_braked)}",
        "baseline_alarm_action_cell": f"alarm{int(baseline_oof_alarm)}_brake{int(episode_braked)}",
        "shadow_alarm_0p5_decisions": str(shadow_alarms),
        "episode_shadow_alarm_0p5": _bool(shadow_alarms > 0),
        "micro_steps": str(micro_steps),
        "short_terminal_macro": _bool(bool(lengths and lengths[-1] < MACRO_STEPS)),
    }


def _summarize(rows: list[dict[str, str]]) -> dict:
    summary = {}
    for arm in ARMS:
        selected = [row for row in rows if row["variant"] == arm]
        transitions: dict[str, int] = {}
        joins: dict[str, int] = {}
        baseline_joins: dict[str, int] = {}
        for row in selected:
            transitions[row["transition"]] = transitions.get(row["transition"], 0) + 1
            key = f"{row['alarm_action_cell']}->{row['four_state']}"
            joins[key] = joins.get(key, 0) + 1
            baseline_key = f"{row['baseline_alarm_action_cell']}->{row['four_state']}"
            baseline_joins[baseline_key] = baseline_joins.get(baseline_key, 0) + 1
        macro = sum(int(row["macro_decisions"]) for row in selected)
        brakes = sum(int(row["brake_decisions"]) for row in selected)
        summary[arm] = {
            "episodes": len(selected),
            "collision": sum(row["collision_any"] == "true" for row in selected),
            "terminal_overtake": sum(row["terminal_overtake"] == "true" for row in selected),
            "confirmed_safe_pass": sum(row["confirmed_safe_pass"] == "true" for row in selected),
            "fixed_collision": sum(row["fixed_collision"] == "true" for row in selected),
            "new_collision": sum(row["new_collision"] == "true" for row in selected),
            "gained_overtake": sum(row["gained_overtake"] == "true" for row in selected),
            "lost_overtake": sum(row["lost_overtake"] == "true" for row in selected),
            "collision_to_confirmed_pass": sum(row["collision_to_confirmed_pass"] == "true" for row in selected),
            "episodes_braked": sum(row["episode_braked"] == "true" for row in selected),
            "episode_brake_rate": sum(row["episode_braked"] == "true" for row in selected) / len(selected),
            "brake_decisions": brakes,
            "macro_decisions": macro,
            "brake_decision_rate": brakes / macro,
            "episodes_oof_alarm": sum(row["episode_oof_alarm"] == "true" for row in selected),
            "baseline_oof_alarm_100hz_episodes": sum(row["baseline_oof_alarm_100hz"] == "true" for row in selected),
            "baseline_safe_oof_false_alarm_episodes": sum(
                row["baseline_oof_alarm_100hz"] == "true"
                and row["bc_four_state"] != "collision"
                for row in selected
            ),
            "baseline_safe_oof_false_alarm_then_brake": sum(
                row["baseline_oof_alarm_100hz"] == "true"
                and row["bc_four_state"] != "collision"
                and row["episode_braked"] == "true"
                for row in selected
            ),
            "baseline_safe_oof_false_alarm_brake_then_lost_overtake": sum(
                row["baseline_oof_alarm_100hz"] == "true"
                and row["bc_four_state"] != "collision"
                and row["episode_braked"] == "true"
                and row["lost_overtake"] == "true"
                for row in selected
            ),
            "clip_micro_steps": sum(int(row["clip_micro_steps"]) for row in selected),
            "globally_brakes": all(row["episode_braked"] == "true" for row in selected),
            "net_overtake_loss": sum(row["lost_overtake"] == "true" for row in selected) > sum(row["gained_overtake"] == "true" for row in selected),
            "transitions": dict(sorted(transitions.items())),
            "alarm_action_outcome": dict(sorted(joins.items())),
            "baseline_oof_alarm_action_outcome": dict(sorted(baseline_joins.items())),
        }
    shared_stop = {
        "all_arms_globally_brake": all(summary[arm]["globally_brakes"] for arm in ARMS),
        "all_arms_net_overtake_loss": all(summary[arm]["net_overtake_loss"] for arm in ARMS),
        "any_action_clipping": any(summary[arm]["clip_micro_steps"] > 0 for arm in ARMS),
    }
    return {
        "schema": "bplus-v2.2-task10-summary-1",
        "interpretation": "mechanism_and_within_opened_development_only_no_l4_generalization",
        "arm_selection_performed": False,
        "arms": summary,
        "shared_stop": shared_stop,
        "task10_passed": not any(shared_stop.values()),
    }


def _write_output_manifest(directory: Path) -> None:
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    )
    (directory / "output_manifest.sha256").write_text(
        "\n".join(f"{file_sha256(path)}  {path.relative_to(directory).as_posix()}" for path in paths) + "\n",
        encoding="utf-8",
    )


def run_closed_loop_warmstart(
    repo_root: str | Path,
    source_preflight_dir: str | Path,
    manifest_release_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
    device_name: str = "cuda:0",
) -> dict:
    root = Path(repo_root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("Task-10 runner must execute from repository root")
    cache = os.environ.get("NUMBA_CACHE_DIR")
    if not cache or not Path(cache).is_absolute():
        raise ValueError("Task-10 requires an isolated absolute NUMBA_CACHE_DIR")
    if not validate_source_preflight(source_preflight_dir, root)["passed"]:
        raise ValueError("Task-10 source preflight invalid")
    manifest_validation = validate_manifest_release(manifest_release_dir, root)
    if not manifest_validation["passed"]:
        raise ValueError(f"Task-10 manifest invalid: {manifest_validation}")
    if file_sha256(root / REGISTRY_RELPATH) != REGISTRY_SHA256:
        raise ValueError("Task-10 registry hash drift")
    _, checkpoints = _validate_historical_warmstart(root)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Task-10 CUDA requested but unavailable")
    device = torch.device(device_name)
    cases = _read_tsv(Path(manifest_release_dir) / "development_scenarios.tsv")
    metadata = _read_tsv(root / METADATA_RELPATH)
    if file_sha256(root / METADATA_RELPATH) != METADATA_SHA256:
        raise ValueError("Task-10 metadata hash drift")
    metadata_by_l2 = {row["l2_id"]: row for row in metadata}
    alarm_bank = OOFAlarmBank(root, device)
    _validate_output_inventory(root / OOF_RELPATH)
    dataset_manifest = json.loads(
        (root / METADATA_RELPATH).with_name("dataset_manifest.json").read_text(encoding="utf-8")
    )
    valid_entry = dataset_manifest["arrays"]["ego_valid_100"]
    valid_path = (root / METADATA_RELPATH).parent / valid_entry["relpath"]
    if file_sha256(valid_path) != valid_entry["sha256"]:
        raise ValueError("Task-10 D2 ego-valid-100 array hash drift")
    ego_valid_100 = np.load(valid_path, mmap_mode="r", allow_pickle=False)
    oof_predictions = np.load(
        root / OOF_RELPATH / "oof_predictions.npy", mmap_mode="r", allow_pickle=False
    )
    baseline_oof_alarm: dict[str, bool] = {}
    for case in cases:
        metadata_row = metadata_by_l2[case["l2_id"]]
        start = int(metadata_row["frame_start"])
        stop = start + int(metadata_row["frame_count"])
        fold = int(metadata_row["outer_fold"])
        valid = np.asarray(ego_valid_100[start:stop], dtype=bool)
        probability = np.asarray(oof_predictions[start:stop, 1])
        baseline_oof_alarm[case["l2_id"]] = bool(
            np.any(valid & (probability >= alarm_bank.thresholds[fold]))
        )

    output = Path(output_dir)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("Task-10 output/partial exists")
    partial.mkdir(parents=True)
    try:
        baseline_outcomes: dict[str, OutcomeRecord] = {}
        baseline_digests: dict[str, str] = {}
        rows: list[dict[str, str]] = []
        decisions: list[dict[str, str]] = []
        for case in cases:
            with np.load(root / case["npz_relpath"], allow_pickle=False) as source:
                arrays = {key: np.asarray(source[key]) for key in ARRAY_KEYS}
            outcome = classify_trajectory(arrays, case["map_name"])
            baseline_outcomes[case["l2_id"]] = outcome
            baseline_digests[case["l2_id"]] = trajectory_digest(arrays)
            fold = int(metadata_by_l2[case["l2_id"]]["outer_fold"])
            rows.append(_result_row(
                case, fold, "BC", "NA", outcome, baseline_digests[case["l2_id"]],
                outcome, baseline_oof_alarm[case["l2_id"]], None, len(arrays["time"])
            ))

        for arm in ARMS:
            policy = V22Policy(arm).to(device)
            policy.load_state_dict(checkpoints[arm]["state_dict"], strict=True)
            policy.eval()
            adapter = NaturalResidualActor(policy, alarm_bank).to(device).eval()
            for index, case in enumerate(cases):
                fold = int(metadata_by_l2[case["l2_id"]]["outer_fold"])
                adapter.reset_runtime(fold)
                result = simulate_episode(adapter, device, case)
                accounting = adapter.accounting()
                if accounting["micro_steps"] != len(result.arrays["time"]):
                    raise AssertionError("Task-10 micro-step accounting mismatch")
                rows.append(_result_row(
                    case,
                    fold,
                    arm,
                    CHECKPOINT_SHA256[arm],
                    result.outcome,
                    trajectory_digest(result.arrays),
                    baseline_outcomes[case["l2_id"]],
                    baseline_oof_alarm[case["l2_id"]],
                    adapter.decision_records,
                    accounting["micro_steps"],
                ))
                decisions.extend(
                    _decision_row(case["l2_id"], arm, record)
                    for record in adapter.decision_records
                )
                if (index + 1) % 24 == 0:
                    print(f"TASK10_PROGRESS arm={arm} episodes={index + 1}/288", flush=True)
            del adapter, policy
            gc.collect()
            torch.cuda.empty_cache()

        _write_tsv(partial / "episode_results.tsv", rows, RESULT_FIELDS)
        _write_tsv(partial / "macro_decisions.tsv", decisions, DECISION_FIELDS)
        summary = _summarize(rows)
        (partial / "task10_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        config = {
            "schema": "bplus-v2.2-task10-config-1",
            "created_at": str(created_at),
            "owner_decision": OWNER_DECISION,
            "interpretation": "mechanism_and_within_opened_development_only_no_l4_generalization",
            "held_out_policy_generalization": False,
            "arm_selection_performed": False,
            "policy_training_started": False,
            "ppo_training_started": False,
            "test_opened": False,
            "final_pool": False,
            "device": str(device),
            "numba_cache_dir": cache,
            "cases": len(cases),
            "variants": ["BC", *ARMS],
            "episode_rows": len(rows),
            "macro_decision_rows": len(decisions),
            "source_preflight_relpath": str(Path(source_preflight_dir)),
            "source_preflight_output_manifest_sha256": file_sha256(Path(source_preflight_dir) / "output_manifest.sha256"),
            "manifest_release_relpath": str(Path(manifest_release_dir)),
            "manifest_release_output_manifest_sha256": file_sha256(Path(manifest_release_dir) / "output_manifest.sha256"),
            "warmstart_release_relpath": WARMSTART_RELEASE_RELPATH,
            "warmstart_release_output_manifest_sha256": WARMSTART_OUTPUT_MANIFEST_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "oof_diagnostic_release_relpath": OOF_RELPATH,
            "oof_diagnostic_output_manifest_sha256": OOF_OUTPUT_MANIFEST_SHA256,
            "alarm_semantics": "outer-refit raw ego-1s probability with same-fold Platt calibration and locked OOF threshold; candidate-trajectory mechanism diagnostic, not a generalization estimate",
            "shadow_alarm_semantics": "full-non-test immutable shadow raw ego-1s probability >=0.5, uncalibrated diagnostic only",
            "global_brake_definition": "at least one deterministic brake decision in every one of 288 episodes",
            "registry_sha256": REGISTRY_SHA256,
        }
        (partial / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (partial / "validation.json").write_text(json.dumps({
            "schema": "bplus-v2.2-task10-validation-1",
            "passed": summary["task10_passed"],
            "integrity_passed": True,
            "shared_stop": summary["shared_stop"],
            "violations": [],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_output_manifest(partial)
        (partial / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    validation = validate_closed_loop_release(output)
    return validation | {"output_manifest_sha256": file_sha256(output / "output_manifest.sha256")}


def validate_closed_loop_release(release_dir: str | Path) -> dict:
    release = Path(release_dir)
    violations: list[str] = []
    rows: list[dict[str, str]] = []
    decisions: list[dict[str, str]] = []
    task10_passed = False
    try:
        if not (release / "COMPLETE").is_file():
            raise ValueError("Task-10 release lacks COMPLETE")
        entries = {}
        for line in (release / "output_manifest.sha256").read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            entries[relative] = digest
        observed = {
            path.relative_to(release).as_posix() for path in release.rglob("*")
            if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
        }
        if set(entries) != observed or any(file_sha256(release / name) != digest for name, digest in entries.items()):
            raise ValueError("Task-10 output manifest mismatch")
        config = json.loads((release / "config.json").read_text(encoding="utf-8"))
        if (
            config["schema"] != "bplus-v2.2-task10-config-1"
            or config["owner_decision"] != OWNER_DECISION
            or config["held_out_policy_generalization"] is not False
            or config["arm_selection_performed"] is not False
            or config["checkpoint_sha256"] != CHECKPOINT_SHA256
            or config["warmstart_release_output_manifest_sha256"] != WARMSTART_OUTPUT_MANIFEST_SHA256
            or config["oof_diagnostic_output_manifest_sha256"] != OOF_OUTPUT_MANIFEST_SHA256
            or any(config[name] is not False for name in (
                "policy_training_started", "ppo_training_started", "test_opened", "final_pool"
            ))
        ):
            raise ValueError("Task-10 authority/scope mismatch")
        with (release / "episode_results.tsv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
                raise ValueError("Task-10 episode header drift")
            rows = list(reader)
        with (release / "macro_decisions.tsv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != DECISION_FIELDS:
                raise ValueError("Task-10 decision header drift")
            decisions = list(reader)
        if len(rows) != 288 * 4:
            raise ValueError("Task-10 episode Cartesian product incomplete")
        manifest_rows = _read_tsv(Path(config["manifest_release_relpath"]) / "development_scenarios.tsv")
        ordered_l2 = [row["l2_id"] for row in manifest_rows]
        expected = {(l2_id, variant) for l2_id in ordered_l2 for variant in ("BC", *ARMS)}
        if {(row["l2_id"], row["variant"]) for row in rows} != expected:
            raise ValueError("Task-10 scenario/variant Cartesian mismatch")
        for variant in ("BC", *ARMS):
            observed_order = [row["l2_id"] for row in rows if row["variant"] == variant]
            if observed_order != ordered_l2:
                raise ValueError(f"Task-10 ordered L2 sequence mismatch: {variant}")
        if any(row["variant"] != "BC" and row["checkpoint_sha256"] != CHECKPOINT_SHA256[row["variant"]] for row in rows):
            raise ValueError("Task-10 checkpoint continuity mismatch")
        decision_counts: dict[tuple[str, str], int] = {}
        for row in decisions:
            key = (row["l2_id"], row["variant"])
            decision_counts[key] = decision_counts.get(key, 0) + 1
            values = [
                float.fromhex(row[name]) for name in DECISION_FIELDS if name.endswith("_hex")
            ]
            if not all(np.isfinite(values)):
                raise ValueError("Task-10 decision contains nonfinite value")
        for row in rows:
            if row["variant"] == "BC":
                if row["macro_decisions"] != "0" or row["checkpoint_sha256"] != "NA":
                    raise ValueError("Task-10 BC row has policy accounting")
            elif decision_counts.get((row["l2_id"], row["variant"]), 0) != int(row["macro_decisions"]):
                raise ValueError("Task-10 macro decision completeness mismatch")
        summary = json.loads((release / "task10_summary.json").read_text(encoding="utf-8"))
        if summary != _summarize(rows):
            raise ValueError("Task-10 summary recomputation mismatch")
        validation = json.loads((release / "validation.json").read_text(encoding="utf-8"))
        if validation["integrity_passed"] is not True or validation["shared_stop"] != summary["shared_stop"]:
            raise ValueError("Task-10 validation summary mismatch")
        task10_passed = bool(summary["task10_passed"])
        if validation["passed"] is not task10_passed:
            raise ValueError("Task-10 pass flag mismatch")
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": "bplus-v2.2-task10-validation-1",
        "passed": not violations and task10_passed,
        "integrity_passed": not violations,
        "task10_passed": task10_passed,
        "episode_rows": len(rows),
        "macro_decision_rows": len(decisions),
        "violations": violations,
    }

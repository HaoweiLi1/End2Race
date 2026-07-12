"""Replacement Task-9 preflight for hierarchical warm-start checkpoints.

This module is deliberately separate from the historical Task-9 runner.  It
accepts the prospective Task-6 release and its manifest hash as arguments,
rejects the old single-gate checkpoint schema, records the checkpoint's natural
four-coordinate macro latent, and then forces an exact physical zero residual.
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
from bplus_v22.manifests import REGISTRY_RELPATH, SMOKE_FIELDS, validate_manifest_release
from bplus_v22.hierarchical_warmstart import validate_hierarchical_warmstart_release
from bplus_v22.release import file_sha256, validate_source_preflight
from bplus_v22.remediated_model import (
    ACTION_SCHEMA,
    CHECKPOINT_SCHEMA,
    HierarchicalResidualDistribution,
    RemediatedV22Policy,
)
from bplus_v22.sidecar import _tensor_digest
from d25.oracle import ARRAY_KEYS, compare_archived, load_bc_model, simulate_episode
from d25.search import trajectory_digest


BC_MODEL_RELPATH = "pretrained/end2race.pth"
TASK9_SCHEMA = "bplus-v2.2-hierarchical-task9-checkpoint-preflight-1"
TASK9_VALIDATION_SCHEMA = "bplus-v2.2-hierarchical-task9-validation-1"

RESULT_FIELDS = (
    "smoke_order",
    "l2_id",
    "map_name",
    "variant",
    "checkpoint_sha256",
    "run1_trajectory_sha256",
    "run2_trajectory_sha256",
    "baseline_trajectory_sha256",
    "run1_matches_baseline",
    "run2_matches_run1",
    "four_state",
    "action_clipped",
    "micro_steps",
    "macro_decisions",
    "macro_lengths_json",
    "short_terminal_macro",
    "forced_max_abs_residual_hex",
    "natural_intervention_decisions",
    "natural_brake_decisions",
    "natural_max_abs_requested_residual_hex",
    "run1_natural_latent_sha256",
    "run2_natural_latent_sha256",
    "diagnostic_sha256",
    "trajectory_relpath",
)

LATENT_FIELDS = (
    "smoke_order",
    "l2_id",
    "arm",
    "macro_index",
    "micro_start",
    "intervention_gate",
    "steer_latent_hex",
    "brake_gate",
    "brake_latent_hex",
    "intervention_probability_hex",
    "conditional_brake_probability_hex",
    "natural_log_prob_hex",
    "requested_steer_delta_hex",
    "requested_speed_delta_hex",
    "forced_intervention_gate",
    "forced_steer_latent_hex",
    "forced_brake_gate",
    "forced_brake_latent_hex",
    "forced_max_abs_residual_hex",
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


def _validate_output_inventory(directory: Path) -> None:
    entries: dict[str, str] = {}
    for line in (directory / "output_manifest.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in entries:
            raise ValueError("duplicate output-manifest path")
        entries[relative] = digest
    observed = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    }
    if set(entries) != observed:
        raise ValueError("output-manifest inventory mismatch")
    for relative, digest in entries.items():
        if file_sha256(directory / relative) != digest:
            raise ValueError(f"output hash mismatch: {relative}")


def _require_sha256(value: str, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return digest


def load_hierarchical_warmstart_release(
    release_dir: str | Path,
    expected_output_manifest_sha256: str,
    repo_root: str | Path,
) -> tuple[dict[str, dict], dict[str, str], dict]:
    """Fail-closed loader shared by replacement Task 9 and Task 10.

    The prospective Task-6 writer must use ``CHECKPOINT_SCHEMA`` and
    ``ACTION_SCHEMA`` in every payload.  The release config may add fields, but
    cannot omit the acceptance and no-PPO scope fields checked here.
    """

    release = Path(release_dir).resolve()
    root = Path(repo_root).resolve()
    expected = _require_sha256(expected_output_manifest_sha256, "warm-start manifest hash")
    independent = validate_hierarchical_warmstart_release(release, root)
    if (
        independent.get("integrity_passed") is not True
        or independent.get("task6_acceptance_passed") is not True
        or independent.get("passed") is not True
    ):
        raise ValueError(f"hierarchical warm-start independent validation failed: {independent}")
    if file_sha256(release / "output_manifest.sha256") != expected:
        raise ValueError("hierarchical warm-start output-manifest hash drift")
    _validate_output_inventory(release)
    config = json.loads((release / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("task6_acceptance_passed") is not True
        or config.get("ppo_checkpoint_eligible") is not True
        or config.get("action_schema") != ACTION_SCHEMA
        or config.get("checkpoint_schema") != CHECKPOINT_SCHEMA
        or config.get("ppo_training_started") is not False
        or config.get("closed_loop_evaluation_started") is not False
        or config.get("arm_selection_performed") is not False
        or config.get("test_opened") is not False
        or config.get("final_pool") is not False
    ):
        raise ValueError("hierarchical warm-start acceptance/scope mismatch")
    reports = config.get("reports")
    if not isinstance(reports, dict) or set(reports) != set(ARMS):
        raise ValueError("hierarchical warm-start reports are incomplete")

    payloads: dict[str, dict] = {}
    checkpoint_sha: dict[str, str] = {}
    for arm in ARMS:
        path = release / "checkpoints" / f"{arm}.pt"
        digest = file_sha256(path)
        report = reports[arm]
        if not isinstance(report, dict) or report.get("checkpoint_sha256") != digest:
            raise ValueError(f"hierarchical checkpoint report hash mismatch: {arm}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict")
        if (
            set(payload) != {
                "schema",
                "action_schema",
                "release_label",
                "arm",
                "manifest_output_sha256",
                "task6_acceptance_passed",
                "calibration_offset_float32",
                "state_dict",
                "state_dict_sha256",
            }
            or
            payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("action_schema") != ACTION_SCHEMA
            or payload.get("arm") != arm
            or payload.get("task6_acceptance_passed") is not True
            or payload.get("manifest_output_sha256") != config.get("manifest_output_sha256")
            or not isinstance(state, dict)
            or payload.get("state_dict_sha256") != _tensor_digest(state.items())
        ):
            raise ValueError(f"hierarchical checkpoint envelope/schema mismatch: {arm}")
        required = {
            "intervention_gate.weight",
            "intervention_gate.bias",
            "intervention_logit_offset",
        }
        if not required.issubset(state):
            raise ValueError(f"hierarchical checkpoint lacks intervention state: {arm}")
        if any(
            not isinstance(value, torch.Tensor)
            or (value.is_floating_point() and not torch.all(torch.isfinite(value)))
            for value in state.values()
        ):
            raise ValueError(f"hierarchical checkpoint state is nonfinite/non-tensor: {arm}")
        if (
            float(state["intervention_logit_offset"].item())
            != float(payload["calibration_offset_float32"])
        ):
            raise ValueError(f"hierarchical checkpoint calibration continuity mismatch: {arm}")
        payloads[arm] = payload
        checkpoint_sha[arm] = digest
    return payloads, checkpoint_sha, config


def _arrays_equal(left: Mapping, right: Mapping) -> bool:
    return all(
        key in left
        and key in right
        and np.asarray(left[key]).dtype == np.asarray(right[key]).dtype
        and np.asarray(left[key]).shape == np.asarray(right[key]).shape
        and np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
        for key in ARRAY_KEYS
    )


class HierarchicalForcedZeroActor(nn.Module):
    """Record the natural 4D macro latent, then deploy exact BC actions."""

    def __init__(self, policy: RemediatedV22Policy):
        super().__init__()
        self.policy = policy
        self.reset_runtime()

    @property
    def gru(self):
        return self.policy.bc.gru

    def reset_runtime(self) -> None:
        self.micro_steps = 0
        self.macro_decisions = 0
        self.natural_intervention_decisions = 0
        self.natural_brake_decisions = 0
        self.max_abs_requested_residual = 0.0
        self.max_abs_forced_residual = 0.0
        self.records: list[dict] = []
        self._lidar_history: list[torch.Tensor] = []
        self._speed_history: list[torch.Tensor] = []
        self._steer_history: list[torch.Tensor] = []
        self._command_speed_history: list[torch.Tensor] = []
        self._last_applied_command: tuple[float, float] | None = None
        self._pending_actual_speed: float | None = None
        self._awaiting_applied_command = False
        self._diagnostic_digest = hashlib.sha256(b"end2race:bplus-v2.2:hier-task9:diagnostic:v1\0")
        self._latent_digest = hashlib.sha256(b"end2race:bplus-v2.2:hier-task9:latent:v1\0")

    def observe_actual_speed(self, value: float) -> None:
        speed = float(value)
        if not np.isfinite(speed) or self._pending_actual_speed is not None:
            raise ValueError("hierarchical Task-9 actual-speed sequence invalid")
        self._pending_actual_speed = speed

    def observe_applied_command(self, steer: float, speed: float) -> None:
        command = (float(steer), float(speed))
        if not all(np.isfinite(value) for value in command) or not self._awaiting_applied_command:
            raise RuntimeError("hierarchical Task-9 applied-command sequence invalid")
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
            raise AssertionError("hierarchical Task-9 history shape drift")
        return lidar, scalar

    @staticmethod
    def _update_tensor_digest(digest: "hashlib._Hash", name: str, value: torch.Tensor) -> None:
        tensor = value.detach().cpu().contiguous()
        if not torch.all(torch.isfinite(tensor)):
            raise ValueError("hierarchical Task-9 tensor is nonfinite")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())

    def forward(self, lidar, previous_speed, hidden):
        if lidar.shape != (1, 1, LIDAR_BEAMS) or previous_speed.shape != (1, 1, 1):
            raise ValueError("hierarchical Task-9 adapter requires evaluator batch one")
        if self._pending_actual_speed is None or self._awaiting_applied_command:
            raise RuntimeError("hierarchical Task-9 observation/command sequence incomplete")
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
            requested = distribution.requested_residual(natural)
            latent = natural.as_tensor()
            self._update_tensor_digest(self._latent_digest, "latent", latent)
            self._update_tensor_digest(
                self._latent_digest, "intervention_probability", distribution.intervention_probability
            )
            self._update_tensor_digest(
                self._latent_digest, "conditional_brake_probability", distribution.brake_probability
            )
            natural_log_prob = distribution.log_prob(natural)
            self._update_tensor_digest(self._latent_digest, "natural_log_prob", natural_log_prob)
            diagnostic = self.policy.diagnostic(bc_feature, lidar_history, scalar_history)
            for name in sorted(diagnostic):
                self._update_tensor_digest(self._diagnostic_digest, name, diagnostic[name])
            self.natural_intervention_decisions += int(natural.intervention_gate.item())
            self.natural_brake_decisions += int(natural.brake_gate.item())
            self.max_abs_requested_residual = max(
                self.max_abs_requested_residual, float(requested.abs().max().item())
            )
            self.records.append(
                {
                    "macro_index": self.macro_decisions,
                    "micro_start": self.micro_steps,
                    "intervention_gate": int(natural.intervention_gate.item()),
                    "steer_latent": float(natural.steer_latent.item()),
                    "brake_gate": int(natural.brake_gate.item()),
                    "brake_latent": float(natural.brake_latent.item()),
                    "intervention_probability": float(distribution.intervention_probability.item()),
                    "conditional_brake_probability": float(distribution.brake_probability.item()),
                    "natural_log_prob": float(natural_log_prob.item()),
                    "requested_steer_delta": float(requested[0, 0].item()),
                    "requested_speed_delta": float(requested[0, 1].item()),
                }
            )
            self.macro_decisions += 1
        # Task 9 tests forced physical zero, not natural NO_OP.  Returning the
        # exact BC tensor is load-bearing for array-level bit identity.
        forced = torch.zeros_like(base)
        self.max_abs_forced_residual = max(
            self.max_abs_forced_residual, float(forced.abs().max().item())
        )
        command = base + forced
        if not torch.equal(command, base):
            raise AssertionError("hierarchical Task-9 forced zero changed BC action")
        self._awaiting_applied_command = True
        self.micro_steps += 1
        return command.unsqueeze(1), next_hidden

    def accounting(self) -> dict:
        full, remainder = divmod(self.micro_steps, MACRO_STEPS)
        lengths = [MACRO_STEPS] * full + ([remainder] if remainder else [])
        if self.micro_steps <= 0 or len(lengths) != self.macro_decisions:
            raise AssertionError("hierarchical Task-9 macro accounting mismatch")
        return {
            "micro_steps": self.micro_steps,
            "macro_decisions": self.macro_decisions,
            "macro_lengths": lengths,
            "natural_intervention_decisions": self.natural_intervention_decisions,
            "natural_brake_decisions": self.natural_brake_decisions,
            "max_abs_requested_residual": self.max_abs_requested_residual,
            "max_abs_forced_residual": self.max_abs_forced_residual,
            "natural_latent_sha256": self._latent_digest.hexdigest(),
            "diagnostic_sha256": self._diagnostic_digest.hexdigest(),
        }


def _latent_row(case: Mapping[str, str], arm: str, row: Mapping) -> dict[str, str]:
    zero = float(0.0).hex()
    return {
        "smoke_order": case["smoke_order"],
        "l2_id": case["l2_id"],
        "arm": arm,
        "macro_index": str(row["macro_index"]),
        "micro_start": str(row["micro_start"]),
        "intervention_gate": str(row["intervention_gate"]),
        "steer_latent_hex": float(row["steer_latent"]).hex(),
        "brake_gate": str(row["brake_gate"]),
        "brake_latent_hex": float(row["brake_latent"]).hex(),
        "intervention_probability_hex": float(row["intervention_probability"]).hex(),
        "conditional_brake_probability_hex": float(row["conditional_brake_probability"]).hex(),
        "natural_log_prob_hex": float(row["natural_log_prob"]).hex(),
        "requested_steer_delta_hex": float(row["requested_steer_delta"]).hex(),
        "requested_speed_delta_hex": float(row["requested_speed_delta"]).hex(),
        "forced_intervention_gate": "0",
        "forced_steer_latent_hex": zero,
        "forced_brake_gate": "0",
        "forced_brake_latent_hex": zero,
        "forced_max_abs_residual_hex": zero,
    }


def _latent_digest_from_rows(rows: list[dict[str, str]]) -> str:
    """Reconstruct the exact float32 decision digest from the TSV ledger."""

    digest = hashlib.sha256(b"end2race:bplus-v2.2:hier-task9:latent:v1\0")

    def update(name: str, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=np.float32)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(b"torch.float32\0")
        digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes())

    for row in sorted(rows, key=lambda item: int(item["macro_index"])):
        update(
            "latent",
            np.asarray(
                [[
                    int(row["intervention_gate"]),
                    float.fromhex(row["steer_latent_hex"]),
                    int(row["brake_gate"]),
                    float.fromhex(row["brake_latent_hex"]),
                ]],
                dtype=np.float32,
            ),
        )
        update(
            "intervention_probability",
            np.asarray([[float.fromhex(row["intervention_probability_hex"])]], dtype=np.float32),
        )
        update(
            "conditional_brake_probability",
            np.asarray(
                [[float.fromhex(row["conditional_brake_probability_hex"])]], dtype=np.float32
            ),
        )
        update(
            "natural_log_prob",
            np.asarray([float.fromhex(row["natural_log_prob_hex"])], dtype=np.float32),
        )
    return digest.hexdigest()


def run_hierarchical_checkpoint_preflight(
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
        raise ValueError("hierarchical Task-9 runner must execute from repository root")
    cache = os.environ.get("NUMBA_CACHE_DIR")
    if not cache or not Path(cache).is_absolute():
        raise ValueError("hierarchical Task-9 requires an isolated absolute NUMBA_CACHE_DIR")
    if not validate_source_preflight(source_preflight_dir, root)["passed"]:
        raise ValueError("hierarchical Task-9 source preflight invalid")
    if not validate_manifest_release(manifest_release_dir, root)["passed"]:
        raise ValueError("hierarchical Task-9 manifest release invalid")
    payloads, checkpoint_sha, warmstart_config = load_hierarchical_warmstart_release(
        warmstart_release_dir, warmstart_output_manifest_sha256, root
    )
    registry_sha256 = _require_sha256(
        warmstart_config.get("registry", {}).get("after_sha256", ""),
        "hierarchical Task-9 registry hash",
    )
    if file_sha256(root / REGISTRY_RELPATH) != registry_sha256:
        raise ValueError("hierarchical Task-9 registry hash drift")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("hierarchical Task-9 CUDA requested but unavailable")
    device = torch.device(device_name)
    cases = _read_tsv(Path(manifest_release_dir) / "no_learning_smoke.tsv")
    if len(cases) != 8 or tuple(cases[0]) != SMOKE_FIELDS:
        raise ValueError("hierarchical Task-9 smoke manifest shape drift")

    output = Path(output_dir)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("hierarchical Task-9 output/partial exists")
    partial.mkdir(parents=True)
    try:
        (partial / "trajectories").mkdir()
        bc = load_bc_model(str(root / BC_MODEL_RELPATH), device)
        baselines = {}
        results: list[dict[str, str]] = []
        latent_rows: list[dict[str, str]] = []
        for case in cases:
            source_path = root / case["npz_relpath"]
            if file_sha256(source_path) != case["npz_sha256"]:
                raise ValueError(f"hierarchical Task-9 source archive drift: {case['l2_id']}")
            with np.load(source_path, allow_pickle=False) as archive:
                archived = {name: np.asarray(archive[name]) for name in archive.files}
            first = simulate_episode(bc, device, case)
            second = simulate_episode(bc, device, case)
            if (
                not _arrays_equal(first.arrays, second.arrays)
                or not compare_archived(first.arrays, archived)["passed"]
                or first.action_clipped
                or second.action_clipped
            ):
                raise AssertionError(f"hierarchical Task-9 BC replay mismatch: {case['l2_id']}")
            baselines[case["l2_id"]] = first
            relative = f"trajectories/{case['l2_id'][3:]}__BC.npz"
            np.savez_compressed(partial / relative, **{key: first.arrays[key] for key in ARRAY_KEYS})
            digest = trajectory_digest(first.arrays)
            results.append(
                {
                    "smoke_order": case["smoke_order"],
                    "l2_id": case["l2_id"],
                    "map_name": case["map_name"],
                    "variant": "BC",
                    "checkpoint_sha256": "NA",
                    "run1_trajectory_sha256": digest,
                    "run2_trajectory_sha256": trajectory_digest(second.arrays),
                    "baseline_trajectory_sha256": digest,
                    "run1_matches_baseline": "true",
                    "run2_matches_run1": "true",
                    "four_state": first.outcome.four_state,
                    "action_clipped": "false",
                    "micro_steps": str(len(first.arrays["time"])),
                    "macro_decisions": "0",
                    "macro_lengths_json": "[]",
                    "short_terminal_macro": "false",
                    "forced_max_abs_residual_hex": float(0).hex(),
                    "natural_intervention_decisions": "0",
                    "natural_brake_decisions": "0",
                    "natural_max_abs_requested_residual_hex": float(0).hex(),
                    "run1_natural_latent_sha256": "NA",
                    "run2_natural_latent_sha256": "NA",
                    "diagnostic_sha256": "NA",
                    "trajectory_relpath": relative,
                }
            )
        del bc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for arm in ARMS:
            policy = RemediatedV22Policy(arm).to(device)
            policy.load_hierarchical_state_dict(payloads[arm]["state_dict"])
            policy.eval()
            actor = HierarchicalForcedZeroActor(policy).to(device).eval()
            for case in cases:
                baseline = baselines[case["l2_id"]]
                actor.reset_runtime()
                first = simulate_episode(actor, device, case)
                first_accounting = actor.accounting()
                first_records = [dict(row) for row in actor.records]
                actor.reset_runtime()
                second = simulate_episode(actor, device, case)
                second_accounting = actor.accounting()
                if (
                    not _arrays_equal(first.arrays, baseline.arrays)
                    or not _arrays_equal(first.arrays, second.arrays)
                    or first_accounting != second_accounting
                    or first_records != actor.records
                    or first_accounting["max_abs_forced_residual"] != 0.0
                    or first.action_clipped
                    or second.action_clipped
                ):
                    raise AssertionError(
                        f"hierarchical Task-9 forced-zero identity mismatch: {arm}/{case['l2_id']}"
                    )
                latent_rows.extend(_latent_row(case, arm, row) for row in first_records)
                relative = f"trajectories/{case['l2_id'][3:]}__{arm}.npz"
                np.savez_compressed(
                    partial / relative, **{key: first.arrays[key] for key in ARRAY_KEYS}
                )
                lengths = first_accounting["macro_lengths"]
                results.append(
                    {
                        "smoke_order": case["smoke_order"],
                        "l2_id": case["l2_id"],
                        "map_name": case["map_name"],
                        "variant": arm,
                        "checkpoint_sha256": checkpoint_sha[arm],
                        "run1_trajectory_sha256": trajectory_digest(first.arrays),
                        "run2_trajectory_sha256": trajectory_digest(second.arrays),
                        "baseline_trajectory_sha256": trajectory_digest(baseline.arrays),
                        "run1_matches_baseline": "true",
                        "run2_matches_run1": "true",
                        "four_state": first.outcome.four_state,
                        "action_clipped": "false",
                        "micro_steps": str(first_accounting["micro_steps"]),
                        "macro_decisions": str(first_accounting["macro_decisions"]),
                        "macro_lengths_json": json.dumps(lengths),
                        "short_terminal_macro": str(bool(lengths and lengths[-1] < MACRO_STEPS)).lower(),
                        "forced_max_abs_residual_hex": float(
                            first_accounting["max_abs_forced_residual"]
                        ).hex(),
                        "natural_intervention_decisions": str(
                            first_accounting["natural_intervention_decisions"]
                        ),
                        "natural_brake_decisions": str(
                            first_accounting["natural_brake_decisions"]
                        ),
                        "natural_max_abs_requested_residual_hex": float(
                            first_accounting["max_abs_requested_residual"]
                        ).hex(),
                        "run1_natural_latent_sha256": first_accounting[
                            "natural_latent_sha256"
                        ],
                        "run2_natural_latent_sha256": second_accounting[
                            "natural_latent_sha256"
                        ],
                        "diagnostic_sha256": first_accounting["diagnostic_sha256"],
                        "trajectory_relpath": relative,
                    }
                )
            del actor, policy
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        _write_tsv(partial / "task9_results.tsv", results, RESULT_FIELDS)
        _write_tsv(partial / "natural_macro_latents.tsv", latent_rows, LATENT_FIELDS)
        _write_json(
            partial / "config.json",
            {
                "schema": TASK9_SCHEMA,
                "created_at": str(created_at),
                "owner_decision": OWNER_DECISION,
                "action_schema": ACTION_SCHEMA,
                "checkpoint_schema": CHECKPOINT_SCHEMA,
                "mode": "record_natural_4d_latent_then_force_exact_physical_zero",
                "natural_noop_required": False,
                "forced_physical_residual_required": True,
                "policy_training_started": False,
                "closed_loop_evaluation_started": False,
                "ppo_training_started": False,
                "arm_selection_performed": False,
                "test_opened": False,
                "final_pool": False,
                "device": str(device),
                "numba_cache_dir": cache,
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
                "registry_sha256": registry_sha256,
                "cases": len(cases),
                "variants": ["BC", *ARMS],
                "reruns_per_variant": 2,
            },
        )
        _write_json(
            partial / "validation.json",
            {"schema": TASK9_VALIDATION_SCHEMA, "passed": True, "violations": []},
        )
        _write_output_manifest(partial)
        (partial / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    validation = validate_hierarchical_checkpoint_preflight(output)
    if not validation["passed"]:
        raise AssertionError(f"created invalid hierarchical Task-9 release: {validation}")
    return validation | {"output_manifest_sha256": file_sha256(output / "output_manifest.sha256")}


def validate_hierarchical_checkpoint_preflight(release_dir: str | Path) -> dict:
    release = Path(release_dir)
    violations: list[str] = []
    rows: list[dict[str, str]] = []
    latent_rows: list[dict[str, str]] = []
    try:
        if not (release / "COMPLETE").is_file():
            raise ValueError("hierarchical Task-9 release lacks COMPLETE")
        _validate_output_inventory(release)
        config = json.loads((release / "config.json").read_text(encoding="utf-8"))
        checkpoint_sha = config.get("checkpoint_sha256")
        _require_sha256(config.get("registry_sha256", ""), "Task-9 registry hash")
        if (
            config.get("schema") != TASK9_SCHEMA
            or config.get("owner_decision") != OWNER_DECISION
            or config.get("action_schema") != ACTION_SCHEMA
            or config.get("checkpoint_schema") != CHECKPOINT_SCHEMA
            or config.get("natural_noop_required") is not False
            or config.get("forced_physical_residual_required") is not True
            or not isinstance(checkpoint_sha, dict)
            or set(checkpoint_sha) != set(ARMS)
            or any(
                config.get(name) is not False
                for name in (
                    "policy_training_started",
                    "closed_loop_evaluation_started",
                    "ppo_training_started",
                    "arm_selection_performed",
                    "test_opened",
                    "final_pool",
                )
            )
        ):
            raise ValueError("hierarchical Task-9 authority/scope mismatch")
        with (release / "task9_results.tsv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
                raise ValueError("hierarchical Task-9 result header drift")
            rows = list(reader)
        with (release / "natural_macro_latents.tsv").open(
            newline="", encoding="utf-8"
        ) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != LATENT_FIELDS:
                raise ValueError("hierarchical Task-9 latent header drift")
            latent_rows = list(reader)
        if len(rows) != 32:
            raise ValueError("hierarchical Task-9 Cartesian product incomplete")
        expected = {(str(index), variant) for index in range(8) for variant in ("BC", *ARMS)}
        if {(row["smoke_order"], row["variant"]) for row in rows} != expected:
            raise ValueError("hierarchical Task-9 case/variant Cartesian mismatch")
        baseline = {row["l2_id"]: row for row in rows if row["variant"] == "BC"}
        latent_counts: dict[tuple[str, str], int] = {}
        latent_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        for latent in latent_rows:
            key = (latent["l2_id"], latent["arm"])
            latent_counts[key] = latent_counts.get(key, 0) + 1
            latent_groups.setdefault(key, []).append(latent)
            gates = (int(latent["intervention_gate"]), int(latent["brake_gate"]))
            if gates[0] not in (0, 1) or gates[1] not in (0, 1) or gates[1] > gates[0]:
                raise ValueError("hierarchical Task-9 latent gate invalid")
            if (
                latent["forced_intervention_gate"] != "0"
                or latent["forced_brake_gate"] != "0"
                or float.fromhex(latent["forced_steer_latent_hex"]) != 0.0
                or float.fromhex(latent["forced_brake_latent_hex"]) != 0.0
                or float.fromhex(latent["forced_max_abs_residual_hex"]) != 0.0
            ):
                raise ValueError("hierarchical Task-9 forced-zero ledger drift")
            values = [
                float.fromhex(latent[name]) for name in LATENT_FIELDS if name.endswith("_hex")
            ]
            if not all(np.isfinite(values)):
                raise ValueError("hierarchical Task-9 latent contains nonfinite value")
        short_terminal_seen = False
        for row in rows:
            if (
                row["run1_matches_baseline"] != "true"
                or row["run2_matches_run1"] != "true"
                or row["run1_trajectory_sha256"] != row["run2_trajectory_sha256"]
                or row["run1_trajectory_sha256"] != row["baseline_trajectory_sha256"]
                or row["action_clipped"] != "false"
            ):
                raise ValueError("hierarchical Task-9 identity/repeat gate failed")
            path = release / row["trajectory_relpath"]
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != set(ARRAY_KEYS) or trajectory_digest(arrays) != row[
                    "run1_trajectory_sha256"
                ]:
                    raise ValueError("hierarchical Task-9 saved trajectory mismatch")
            if row["variant"] == "BC":
                if row["checkpoint_sha256"] != "NA" or row["macro_decisions"] != "0":
                    raise ValueError("hierarchical Task-9 BC accounting drift")
            else:
                if row["checkpoint_sha256"] != checkpoint_sha[row["variant"]]:
                    raise ValueError("hierarchical Task-9 checkpoint continuity mismatch")
                lengths = json.loads(row["macro_lengths_json"])
                if (
                    sum(lengths) != int(row["micro_steps"])
                    or len(lengths) != int(row["macro_decisions"])
                    or any(not 1 <= length <= MACRO_STEPS for length in lengths)
                    or float.fromhex(row["forced_max_abs_residual_hex"]) != 0.0
                    or row["run1_natural_latent_sha256"] != row["run2_natural_latent_sha256"]
                    or len(row["run1_natural_latent_sha256"]) != 64
                    or latent_counts.get((row["l2_id"], row["variant"]), 0)
                    != int(row["macro_decisions"])
                    or _latent_digest_from_rows(
                        latent_groups.get((row["l2_id"], row["variant"]), [])
                    )
                    != row["run1_natural_latent_sha256"]
                ):
                    raise ValueError("hierarchical Task-9 macro/latent accounting failed")
                short_terminal_seen = short_terminal_seen or row["short_terminal_macro"] == "true"
                if row["four_state"] != baseline[row["l2_id"]]["four_state"]:
                    raise ValueError("hierarchical Task-9 outcome mismatch")
        if not short_terminal_seen:
            raise ValueError("hierarchical Task-9 did not exercise a short terminal macro")
        saved_validation = json.loads((release / "validation.json").read_text(encoding="utf-8"))
        if saved_validation != {
            "schema": TASK9_VALIDATION_SCHEMA,
            "passed": True,
            "violations": [],
        }:
            raise ValueError("hierarchical Task-9 saved validation drift")
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": TASK9_VALIDATION_SCHEMA,
        "passed": not violations,
        "rows": len(rows),
        "natural_macro_rows": len(latent_rows),
        "violations": violations,
    }

"""Fresh natural-NO_OP identity gate for the remediated hierarchical policy.

This release is deliberately separate from the historical forced-zero identity
and Task-9 checkpoint preflight.  It proves that a freshly constructed
``RemediatedV22Policy`` naturally chooses the exact hierarchical NO_OP and that
composing that held action against every deployed 100 Hz BC command preserves
the complete simulator trajectory bit-for-bit.
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
from bplus_v22.identity import (
    BC_MODEL_RELPATH,
    EXPECTED_MAPS,
    _arrays_equal,
    _read_cases,
    _save_trajectory,
)
from bplus_v22.manifests import REGISTRY_RELPATH, REGISTRY_SHA256
from bplus_v22.release import (
    file_sha256,
    validate_pinned_inputs,
    validate_source_preflight,
)
from bplus_v22.remediated_model import (
    ACTION_SCHEMA,
    CHECKPOINT_SCHEMA,
    INITIAL_INTERVENTION_LOGIT,
    HierarchicalResidualAction,
    HierarchicalResidualDistribution,
    RemediatedV22Policy,
)
from bplus_v22.sidecar import load_sidecar_bundle, validate_sidecar_release
from d25.oracle import ARRAY_KEYS, compare_archived, load_bc_model, simulate_episode
from d25.search import trajectory_digest


ARTIFACT_SCHEMA = "bplus-v2.2-hierarchical-fresh-identity-1"
VALIDATION_SCHEMA = "bplus-v2.2-hierarchical-fresh-identity-validation-1"
MODEL_SCHEMA = "bplus-v2.2-hierarchical-fresh-identity-models-1"
LEGACY_CHECKPOINT_SCHEMAS = (
    "bplus-v2.2-warmstart-remediation-checkpoint-2",
    "bplus-v2.2-warmstart-checkpoint-1",
)

RESULT_FIELDS = (
    "case_order",
    "l2_id",
    "map_name",
    "variant",
    "run1_trajectory_sha256",
    "run2_trajectory_sha256",
    "baseline_trajectory_sha256",
    "run1_matches_baseline",
    "run2_matches_run1",
    "run1_matches_archive",
    "four_state",
    "action_clipped",
    "micro_steps",
    "macro_decisions",
    "macro_lengths_json",
    "natural_intervention_decisions",
    "natural_brake_decisions",
    "max_abs_requested_residual_hex",
    "max_abs_applied_residual_hex",
    "natural_action_sequence_sha256",
    "diagnostic_sha256",
    "trajectory_relpath",
)


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_tsv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_output_manifest(directory: Path) -> None:
    relpaths = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    )
    (directory / "output_manifest.sha256").write_text(
        "\n".join(
            f"{file_sha256(directory / relpath)}  {relpath}" for relpath in relpaths
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_output_manifest(directory: Path) -> None:
    entries: dict[str, str] = {}
    for line in (directory / "output_manifest.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, relpath = line.split("  ", 1)
        if relpath in entries:
            raise ValueError("hierarchical identity duplicate output path")
        entries[relpath] = digest
    observed = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    }
    if set(entries) != observed:
        raise ValueError("hierarchical identity output inventory mismatch")
    for relpath, digest in entries.items():
        if file_sha256(directory / relpath) != digest:
            raise ValueError(f"hierarchical identity output hash mismatch: {relpath}")


def load_hierarchical_checkpoint(
    policy: RemediatedV22Policy,
    payload: Mapping,
    *,
    expected_arm: str,
) -> None:
    """Load only the new four-dimensional checkpoint envelope, fail closed.

    Fresh identity itself never calls this function.  Keeping the strict loader
    here makes it impossible for callers to silently reinterpret a historical
    three-dimensional/single-gate checkpoint as a fresh hierarchical policy.
    """

    schema = payload.get("schema")
    if schema != CHECKPOINT_SCHEMA:
        legacy = " (legacy checkpoint rejected)" if schema in LEGACY_CHECKPOINT_SCHEMAS else ""
        raise ValueError(
            f"hierarchical checkpoint schema mismatch: {schema!r}{legacy}"
        )
    if payload.get("action_schema") != ACTION_SCHEMA:
        raise ValueError("hierarchical checkpoint action schema mismatch")
    if payload.get("arm") != expected_arm or policy.arm != expected_arm:
        raise ValueError("hierarchical checkpoint arm mismatch")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("hierarchical checkpoint lacks state_dict")
    policy.load_hierarchical_state_dict(state)


class HierarchicalIdentityActor(nn.Module):
    """Simulator adapter holding a natural hierarchical action for K=10."""

    def __init__(self, policy: RemediatedV22Policy):
        super().__init__()
        if not isinstance(policy, RemediatedV22Policy):
            raise TypeError("hierarchical identity requires RemediatedV22Policy")
        self.policy = policy
        self._assert_fresh_intervention_head()
        self.reset_runtime()

    @property
    def gru(self):
        return self.policy.bc.gru

    def _assert_fresh_intervention_head(self) -> None:
        weight = self.policy.intervention_gate.weight.detach()
        bias = self.policy.intervention_gate.bias.detach()
        offset = self.policy.intervention_logit_offset.detach()
        if not torch.equal(weight, torch.zeros_like(weight)):
            raise ValueError("hierarchical identity policy is not fresh: intervention weight")
        if not torch.equal(
            bias, torch.full_like(bias, float(INITIAL_INTERVENTION_LOGIT))
        ):
            raise ValueError("hierarchical identity policy is not fresh: intervention bias")
        if not torch.equal(offset, torch.zeros_like(offset)):
            raise ValueError("hierarchical identity policy is calibrated/warm-started")

    def reset_runtime(self) -> None:
        self.micro_steps = 0
        self.macro_decisions = 0
        self.natural_intervention_decisions = 0
        self.natural_brake_decisions = 0
        self.max_abs_requested_residual = 0.0
        self.max_abs_applied_residual = 0.0
        self._lidar_history: list[torch.Tensor] = []
        self._speed_history: list[torch.Tensor] = []
        self._steer_history: list[torch.Tensor] = []
        self._command_speed_history: list[torch.Tensor] = []
        self._last_applied_command: tuple[float, float] | None = None
        self._awaiting_applied_command = False
        self._pending_actual_speed: float | None = None
        self._held_action: HierarchicalResidualAction | None = None
        self._diagnostic_digest = hashlib.sha256(
            b"end2race:bplus-v2.2:hierarchical-fresh-diagnostic:v1\0"
        )
        self._action_digest = hashlib.sha256(
            b"end2race:bplus-v2.2:hierarchical-natural-action-sequence:v1\0"
        )

    def observe_actual_speed(self, value: float) -> None:
        speed = float(value)
        if not np.isfinite(speed):
            raise ValueError("hierarchical identity actual speed is nonfinite")
        if self._pending_actual_speed is not None:
            raise RuntimeError("hierarchical identity actual speed was not consumed")
        self._pending_actual_speed = speed

    def observe_applied_command(self, steer: float, speed: float) -> None:
        command = (float(steer), float(speed))
        if not all(np.isfinite(value) for value in command):
            raise ValueError("hierarchical identity applied command is nonfinite")
        if not self._awaiting_applied_command:
            raise RuntimeError("hierarchical identity received unexpected applied command")
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
        if lidar.shape != (1, len(HISTORY_OFFSETS), LIDAR_BEAMS):
            raise AssertionError("hierarchical identity LiDAR history shape drift")
        if scalar.shape != (1, 24):
            raise AssertionError("hierarchical identity scalar history shape drift")
        return lidar, scalar

    def _update_diagnostic_digest(self, output: dict[str, torch.Tensor]) -> None:
        for name in sorted(output):
            value = output[name].detach().cpu().contiguous()
            if not torch.all(torch.isfinite(value)):
                raise ValueError("hierarchical identity diagnostic is nonfinite")
            self._diagnostic_digest.update(name.encode("utf-8") + b"\0")
            self._diagnostic_digest.update(value.numpy().tobytes())

    def forward(self, lidar, previous_speed, hidden):
        if lidar.shape != (1, 1, LIDAR_BEAMS) or previous_speed.shape != (1, 1, 1):
            raise ValueError("hierarchical identity supports exact batch-1 replay")
        if self._pending_actual_speed is None:
            raise RuntimeError("hierarchical identity lacks current actual speed")
        if self._awaiting_applied_command:
            raise RuntimeError("hierarchical identity lacks prior applied command")
        actual_speed = torch.full_like(previous_speed[:, -1], self._pending_actual_speed)
        self._pending_actual_speed = None
        base, bc_feature, next_hidden = self.policy.bc_step(lidar, previous_speed, hidden)
        if self._last_applied_command is None:
            prior_command = torch.zeros_like(base)
        else:
            prior_command = base.new_tensor(self._last_applied_command).reshape(1, 2)
        self._lidar_history.append(
            (torch.clamp(lidar[:, -1], 0.0, 30.0) / 30.0).detach()
        )
        self._speed_history.append((actual_speed / 10.0).detach())
        self._steer_history.append((prior_command[:, 0:1] / 0.52).detach())
        self._command_speed_history.append((prior_command[:, 1:2] / 10.0).detach())

        if self.micro_steps % MACRO_STEPS == 0:
            lidar_history, scalar_history = self._history()
            distribution = self.policy.distribution(
                bc_feature, lidar_history, scalar_history
            )
            action = distribution.deterministic()
            requested = distribution.requested_residual(action)
            canonical = torch.zeros_like(action.as_tensor())
            if not torch.equal(action.as_tensor(), canonical):
                raise AssertionError("fresh hierarchical policy did not select exact NO_OP")
            if not torch.equal(requested, torch.zeros_like(requested)):
                raise AssertionError("fresh hierarchical NO_OP requested a residual")
            self.natural_intervention_decisions += int(action.intervention_gate.item())
            self.natural_brake_decisions += int(action.brake_gate.item())
            self.max_abs_requested_residual = max(
                self.max_abs_requested_residual,
                float(requested.abs().max().item()),
            )
            tensor = action.as_tensor().detach().cpu().contiguous()
            self._action_digest.update(tensor.numpy().tobytes())
            self._held_action = action
            self.macro_decisions += 1
            self._update_diagnostic_digest(
                self.policy.diagnostic(bc_feature, lidar_history, scalar_history)
            )
        if self._held_action is None:
            raise AssertionError("hierarchical identity lacks held action")

        ledger = HierarchicalResidualDistribution.compose(base, self._held_action)
        if not torch.equal(ledger.command, ledger.deployed_base):
            raise AssertionError("hierarchical NO_OP changed deployed BC action")
        if not torch.equal(
            ledger.applied_residual, torch.zeros_like(ledger.applied_residual)
        ):
            raise AssertionError("hierarchical NO_OP applied a residual")
        if torch.any(ledger.external_clip_would_change):
            raise AssertionError("hierarchical NO_OP requires downstream clipping")
        self.max_abs_applied_residual = max(
            self.max_abs_applied_residual,
            float(ledger.applied_residual.abs().max().item()),
        )
        self._awaiting_applied_command = True
        self.micro_steps += 1
        return ledger.command.unsqueeze(1), next_hidden

    def accounting(self) -> dict:
        if self.micro_steps <= 0:
            raise ValueError("hierarchical identity executed no micro-step")
        full, remainder = divmod(self.micro_steps, MACRO_STEPS)
        lengths = [MACRO_STEPS] * full + ([remainder] if remainder else [])
        if len(lengths) != self.macro_decisions or sum(lengths) != self.micro_steps:
            raise AssertionError("hierarchical identity macro accounting mismatch")
        return {
            "micro_steps": self.micro_steps,
            "macro_decisions": self.macro_decisions,
            "macro_lengths": lengths,
            "natural_intervention_decisions": self.natural_intervention_decisions,
            "natural_brake_decisions": self.natural_brake_decisions,
            "max_abs_requested_residual": self.max_abs_requested_residual,
            "max_abs_applied_residual": self.max_abs_applied_residual,
            "natural_action_sequence_sha256": self._action_digest.hexdigest(),
            "diagnostic_sha256": self._diagnostic_digest.hexdigest(),
        }


def _model_initialization(policy: RemediatedV22Policy) -> dict:
    return {
        "arm": policy.arm,
        "policy_class": type(policy).__name__,
        "action_schema": ACTION_SCHEMA,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "policy_checkpoint_loaded": False,
        "initialization_phase": "FRESH_NATURAL_NO_OP",
        "initial_intervention_logit_hex": float(
            policy.intervention_gate.bias.detach().cpu().item()
        ).hex(),
        "intervention_offset_hex": float(
            policy.intervention_logit_offset.detach().cpu().item()
        ).hex(),
        "policy_sidecar_encoder_sha256": policy.policy_sidecar_encoder_sha256(),
        "shadow_sidecar_sha256": policy.shadow_sha256(),
    }


def run_hierarchical_identity(
    repo_root: str | Path,
    source_preflight_dir: str | Path,
    sidecar_release_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
    device_name: str = "cuda:0",
) -> dict:
    """Create an atomic fresh fitted-sidecar hierarchical identity artifact."""

    root = Path(repo_root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("hierarchical identity runner must execute from repo root")
    cache = os.environ.get("NUMBA_CACHE_DIR")
    if not cache or not Path(cache).is_absolute():
        raise ValueError("hierarchical identity requires isolated absolute NUMBA_CACHE_DIR")
    source_validation = validate_source_preflight(source_preflight_dir, root)
    if not source_validation["passed"]:
        raise ValueError(f"hierarchical identity source preflight failed: {source_validation}")
    inputs = validate_pinned_inputs(root)
    if not inputs["passed"]:
        raise ValueError(f"hierarchical identity pinned inputs failed: {inputs}")
    sidecar_release = Path(sidecar_release_dir).resolve()
    sidecar_validation = validate_sidecar_release(sidecar_release, root)
    if not sidecar_validation["passed"]:
        raise ValueError(f"hierarchical identity fitted sidecar failed: {sidecar_validation}")
    sidecar_state, sidecar_mean, sidecar_std, _ = load_sidecar_bundle(sidecar_release)
    registry = root / REGISTRY_RELPATH
    registry_before = file_sha256(registry)
    if registry_before != REGISTRY_SHA256:
        raise ValueError("hierarchical identity registry hash drift")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("hierarchical identity CUDA requested but unavailable")
    device = torch.device(device_name)

    output = Path(output_dir)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("hierarchical identity output/partial exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    try:
        cases = _read_cases(root)
        _write_tsv(
            partial / "case_manifest.tsv",
            [
                {
                    "case_order": str(index),
                    "l2_id": row["l2_id"],
                    "map_name": row["map_name"],
                    "npz_relpath": row["npz_relpath"],
                    "npz_sha256": row["npz_sha256"],
                }
                for index, row in enumerate(cases)
            ],
            ("case_order", "l2_id", "map_name", "npz_relpath", "npz_sha256"),
        )
        (partial / "trajectories").mkdir()
        bc = load_bc_model(str(root / BC_MODEL_RELPATH), device)
        bc_state = {
            name: value.detach().cpu().clone() for name, value in bc.state_dict().items()
        }
        baselines = {}
        rows: list[dict[str, str]] = []
        for case_order, case in enumerate(cases):
            source_path = root / case["npz_relpath"]
            if file_sha256(source_path) != case["npz_sha256"]:
                raise ValueError(f"hierarchical identity archive drift: {case['l2_id']}")
            with np.load(source_path, allow_pickle=False) as source:
                archived = {key: np.asarray(source[key]) for key in source.files}
            first = simulate_episode(bc, device, case)
            second = simulate_episode(bc, device, case)
            archive_match = compare_archived(first.arrays, archived)["passed"]
            if not archive_match or not _arrays_equal(first.arrays, second.arrays):
                raise AssertionError(f"hierarchical identity BC replay failed: {case['l2_id']}")
            baselines[case["l2_id"]] = first
            digest = trajectory_digest(first.arrays)
            relpath = f"trajectories/{case['l2_id'][3:]}__BC.npz"
            _save_trajectory(partial / relpath, first.arrays)
            rows.append(
                {
                    "case_order": str(case_order),
                    "l2_id": case["l2_id"],
                    "map_name": case["map_name"],
                    "variant": "BC",
                    "run1_trajectory_sha256": digest,
                    "run2_trajectory_sha256": trajectory_digest(second.arrays),
                    "baseline_trajectory_sha256": digest,
                    "run1_matches_baseline": "true",
                    "run2_matches_run1": str(_arrays_equal(first.arrays, second.arrays)).lower(),
                    "run1_matches_archive": str(archive_match).lower(),
                    "four_state": first.outcome.four_state,
                    "action_clipped": "false",
                    "micro_steps": str(len(first.arrays["time"])),
                    "macro_decisions": "0",
                    "macro_lengths_json": "[]",
                    "natural_intervention_decisions": "0",
                    "natural_brake_decisions": "0",
                    "max_abs_requested_residual_hex": float(0.0).hex(),
                    "max_abs_applied_residual_hex": float(0.0).hex(),
                    "natural_action_sequence_sha256": "NA",
                    "diagnostic_sha256": "NA",
                    "trajectory_relpath": relpath,
                }
            )
        del bc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        initializations = []
        sequence_by_case: dict[str, str] = {}
        for arm in ARMS:
            policy = RemediatedV22Policy(
                arm,
                bc_state_dict=bc_state,
                sidecar_state_dict=sidecar_state,
                sidecar_bc_mean=sidecar_mean,
                sidecar_bc_std=sidecar_std,
            ).to(device).eval()
            initializations.append(_model_initialization(policy))
            actor = HierarchicalIdentityActor(policy).to(device).eval()
            for case_order, case in enumerate(cases):
                actor.reset_runtime()
                first = simulate_episode(actor, device, case)
                first_accounting = actor.accounting()
                actor.reset_runtime()
                second = simulate_episode(actor, device, case)
                second_accounting = actor.accounting()
                baseline = baselines[case["l2_id"]]
                if (
                    not _arrays_equal(first.arrays, baseline.arrays)
                    or not _arrays_equal(first.arrays, second.arrays)
                    or first_accounting != second_accounting
                    or first_accounting["natural_intervention_decisions"] != 0
                    or first_accounting["natural_brake_decisions"] != 0
                    or first_accounting["max_abs_requested_residual"] != 0.0
                    or first_accounting["max_abs_applied_residual"] != 0.0
                    or first.action_clipped
                    or second.action_clipped
                ):
                    raise AssertionError(
                        f"hierarchical fresh identity replay failed: {arm}/{case['l2_id']}"
                    )
                action_digest = first_accounting["natural_action_sequence_sha256"]
                expected_digest = sequence_by_case.setdefault(case["l2_id"], action_digest)
                if action_digest != expected_digest:
                    raise AssertionError(
                        f"A/B/C fresh action sequence mismatch: {case['l2_id']}"
                    )
                relpath = f"trajectories/{case['l2_id'][3:]}__{arm}.npz"
                _save_trajectory(partial / relpath, first.arrays)
                rows.append(
                    {
                        "case_order": str(case_order),
                        "l2_id": case["l2_id"],
                        "map_name": case["map_name"],
                        "variant": arm,
                        "run1_trajectory_sha256": trajectory_digest(first.arrays),
                        "run2_trajectory_sha256": trajectory_digest(second.arrays),
                        "baseline_trajectory_sha256": trajectory_digest(baseline.arrays),
                        "run1_matches_baseline": "true",
                        "run2_matches_run1": "true",
                        "run1_matches_archive": "true",
                        "four_state": first.outcome.four_state,
                        "action_clipped": "false",
                        "micro_steps": str(first_accounting["micro_steps"]),
                        "macro_decisions": str(first_accounting["macro_decisions"]),
                        "macro_lengths_json": json.dumps(first_accounting["macro_lengths"]),
                        "natural_intervention_decisions": "0",
                        "natural_brake_decisions": "0",
                        "max_abs_requested_residual_hex": float(
                            first_accounting["max_abs_requested_residual"]
                        ).hex(),
                        "max_abs_applied_residual_hex": float(
                            first_accounting["max_abs_applied_residual"]
                        ).hex(),
                        "natural_action_sequence_sha256": action_digest,
                        "diagnostic_sha256": first_accounting["diagnostic_sha256"],
                        "trajectory_relpath": relpath,
                    }
                )
            del actor, policy
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        by_arm = {row["arm"]: row for row in initializations}
        if (
            by_arm["SIDECAR_FROZEN"]["policy_sidecar_encoder_sha256"]
            != by_arm["SIDECAR_FINETUNE"]["policy_sidecar_encoder_sha256"]
        ):
            raise AssertionError("hierarchical identity B/C sidecar initialization mismatch")
        if len({row["shadow_sidecar_sha256"] for row in initializations}) != 1:
            raise AssertionError("hierarchical identity shadow sidecar mismatch")

        _write_tsv(partial / "identity_results.tsv", rows, RESULT_FIELDS)
        _write_json(
            partial / "model_initializations.json",
            {
                "schema": MODEL_SCHEMA,
                "sidecar_role": "FULL_NON_TEST_INITIALIZATION",
                "sidecar_release_relpath": str(sidecar_release),
                "sidecar_release_output_manifest_sha256": file_sha256(
                    sidecar_release / "output_manifest.sha256"
                ),
                "models": initializations,
            },
        )
        registry_after = file_sha256(registry)
        if registry_after != registry_before:
            raise AssertionError("hierarchical identity mutated registry")
        _write_json(
            partial / "config.json",
            {
                "schema": ARTIFACT_SCHEMA,
                "created_at": str(created_at),
                "owner_decision": OWNER_DECISION,
                "action_schema": ACTION_SCHEMA,
                "checkpoint_schema": CHECKPOINT_SCHEMA,
                "legacy_checkpoint_schemas_rejected": list(LEGACY_CHECKPOINT_SCHEMAS),
                "policy_class": "RemediatedV22Policy",
                "policy_initialization_phase": "FRESH_NATURAL_NO_OP",
                "policy_checkpoint_loaded": False,
                "identity_mechanism": "NATURAL_INTERVENTION_GATE_NO_OP",
                "composition_base": "DEPLOYED_BC_BASE_EVERY_MICRO_STEP",
                "macro_steps": MACRO_STEPS,
                "sidecar_role": "FULL_NON_TEST_INITIALIZATION",
                "sidecar_release_relpath": str(sidecar_release),
                "sidecar_release_output_manifest_sha256": file_sha256(
                    sidecar_release / "output_manifest.sha256"
                ),
                "source_preflight_relpath": str(Path(source_preflight_dir).resolve()),
                "source_preflight_output_manifest_sha256": file_sha256(
                    Path(source_preflight_dir) / "output_manifest.sha256"
                ),
                "registry_mutated": False,
                "registry_before_sha256": registry_before,
                "registry_after_sha256": registry_after,
                "policy_training_started": False,
                "warmstart_started": False,
                "ppo_training_started": False,
                "device": str(device),
                "numba_cache_dir": cache,
                "cases": len(cases),
                "variants": ["BC", *ARMS],
                "reruns_per_variant": 2,
                "saved_trajectories": len(rows),
                "fresh_natural_no_op_passed": True,
                "abc_initial_action_sequences_identical": True,
            },
        )
        _write_json(
            partial / "validation.json",
            {
                "schema": VALIDATION_SCHEMA,
                "passed": True,
                "cases": len(cases),
                "identity_rows": len(rows),
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

    validation = validate_hierarchical_identity(output, repo_root=root)
    if not validation["passed"]:
        raise AssertionError(f"created invalid hierarchical identity: {validation}")
    return {
        "passed": True,
        "cases": validation["cases"],
        "identity_rows": validation["identity_rows"],
        "output_manifest_sha256": file_sha256(output / "output_manifest.sha256"),
        "registry_sha256": registry_before,
    }


def validate_hierarchical_identity(
    release_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict:
    release = Path(release_dir)
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    violations: list[str] = []
    cases: list[dict[str, str]] = []
    rows: list[dict[str, str]] = []
    try:
        if not (release / "COMPLETE").is_file():
            raise ValueError("hierarchical identity lacks COMPLETE")
        config = json.loads((release / "config.json").read_text(encoding="utf-8"))
        if (
            config.get("schema") != ARTIFACT_SCHEMA
            or config.get("owner_decision") != OWNER_DECISION
            or config.get("action_schema") != ACTION_SCHEMA
            or config.get("checkpoint_schema") != CHECKPOINT_SCHEMA
        ):
            raise ValueError("hierarchical identity schema/authority mismatch")
        if config.get("legacy_checkpoint_schemas_rejected") != list(
            LEGACY_CHECKPOINT_SCHEMAS
        ):
            raise ValueError("hierarchical identity legacy checkpoint policy drift")
        required_false = (
            "policy_checkpoint_loaded",
            "registry_mutated",
            "policy_training_started",
            "warmstart_started",
            "ppo_training_started",
        )
        if any(config.get(name) is not False for name in required_false):
            raise ValueError("hierarchical identity fresh/no-training scope mismatch")
        if (
            config.get("policy_class") != "RemediatedV22Policy"
            or config.get("policy_initialization_phase") != "FRESH_NATURAL_NO_OP"
            or config.get("identity_mechanism") != "NATURAL_INTERVENTION_GATE_NO_OP"
            or config.get("composition_base") != "DEPLOYED_BC_BASE_EVERY_MICRO_STEP"
            or config.get("macro_steps") != MACRO_STEPS
            or config.get("fresh_natural_no_op_passed") is not True
            or config.get("abc_initial_action_sequences_identical") is not True
        ):
            raise ValueError("hierarchical identity mechanism mismatch")
        if not Path(config.get("numba_cache_dir", "")).is_absolute():
            raise ValueError("hierarchical identity Numba cache missing")

        sidecar_release = Path(config["sidecar_release_relpath"])
        sidecar_validation = validate_sidecar_release(sidecar_release, root)
        if not sidecar_validation["passed"]:
            raise ValueError("hierarchical identity fitted sidecar invalid")
        if file_sha256(sidecar_release / "output_manifest.sha256") != config[
            "sidecar_release_output_manifest_sha256"
        ]:
            raise ValueError("hierarchical identity sidecar manifest mismatch")
        if config["registry_before_sha256"] != config["registry_after_sha256"]:
            raise ValueError("hierarchical identity registry hash mismatch")
        if config["registry_before_sha256"] != REGISTRY_SHA256:
            raise ValueError("hierarchical identity pinned registry mismatch")

        with (release / "case_manifest.tsv").open(
            newline="", encoding="utf-8"
        ) as handle:
            cases = list(csv.DictReader(handle, delimiter="\t"))
        if len(cases) != 4 or tuple(row["map_name"] for row in cases) != EXPECTED_MAPS:
            raise ValueError("hierarchical identity case/map mismatch")

        models = json.loads(
            (release / "model_initializations.json").read_text(encoding="utf-8")
        )
        if models.get("schema") != MODEL_SCHEMA:
            raise ValueError("hierarchical identity model schema mismatch")
        records = models.get("models", [])
        if [record.get("arm") for record in records] != list(ARMS):
            raise ValueError("hierarchical identity model arm mismatch")
        if any(
            record.get("policy_class") != "RemediatedV22Policy"
            or record.get("action_schema") != ACTION_SCHEMA
            or record.get("checkpoint_schema") != CHECKPOINT_SCHEMA
            or record.get("policy_checkpoint_loaded") is not False
            or record.get("initialization_phase") != "FRESH_NATURAL_NO_OP"
            or float.fromhex(record["initial_intervention_logit_hex"])
            != INITIAL_INTERVENTION_LOGIT
            or float.fromhex(record["intervention_offset_hex"]) != 0.0
            for record in records
        ):
            raise ValueError("hierarchical identity model initialization mismatch")
        by_arm = {record["arm"]: record for record in records}
        if (
            by_arm["SIDECAR_FROZEN"]["policy_sidecar_encoder_sha256"]
            != by_arm["SIDECAR_FINETUNE"]["policy_sidecar_encoder_sha256"]
        ):
            raise ValueError("hierarchical identity persisted B/C sidecar mismatch")
        if len({record["shadow_sidecar_sha256"] for record in records}) != 1:
            raise ValueError("hierarchical identity persisted shadow mismatch")

        with (release / "identity_results.tsv").open(
            newline="", encoding="utf-8"
        ) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
                raise ValueError("hierarchical identity result header mismatch")
            rows = list(reader)
        if len(rows) != 16:
            raise ValueError("hierarchical identity result count mismatch")
        expected = {
            (case["l2_id"], variant)
            for case in cases
            for variant in ("BC", *ARMS)
        }
        if {(row["l2_id"], row["variant"]) for row in rows} != expected:
            raise ValueError("hierarchical identity Cartesian product mismatch")
        action_digests: dict[str, set[str]] = {case["l2_id"]: set() for case in cases}
        for row in rows:
            if (
                row["run1_matches_baseline"] != "true"
                or row["run2_matches_run1"] != "true"
                or row["run1_matches_archive"] != "true"
                or row["action_clipped"] != "false"
                or row["run1_trajectory_sha256"] != row["run2_trajectory_sha256"]
                or row["run1_trajectory_sha256"] != row["baseline_trajectory_sha256"]
            ):
                raise ValueError("hierarchical identity trajectory gate failed")
            path = release / row["trajectory_relpath"]
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != set(ARRAY_KEYS):
                    raise ValueError("hierarchical identity trajectory inventory mismatch")
                if trajectory_digest(arrays) != row["run1_trajectory_sha256"]:
                    raise ValueError("hierarchical identity trajectory digest mismatch")
            if row["variant"] == "BC":
                if (
                    row["macro_decisions"] != "0"
                    or row["macro_lengths_json"] != "[]"
                    or row["natural_action_sequence_sha256"] != "NA"
                ):
                    raise ValueError("hierarchical identity BC accounting mismatch")
                continue
            lengths = json.loads(row["macro_lengths_json"])
            if (
                len(lengths) != int(row["macro_decisions"])
                or sum(lengths) != int(row["micro_steps"])
                or any(not 1 <= value <= MACRO_STEPS for value in lengths)
            ):
                raise ValueError("hierarchical identity macro accounting mismatch")
            if (
                row["natural_intervention_decisions"] != "0"
                or row["natural_brake_decisions"] != "0"
                or float.fromhex(row["max_abs_requested_residual_hex"]) != 0.0
                or float.fromhex(row["max_abs_applied_residual_hex"]) != 0.0
                or len(row["natural_action_sequence_sha256"]) != 64
                or len(row["diagnostic_sha256"]) != 64
            ):
                raise ValueError("hierarchical identity natural NO_OP gate failed")
            action_digests[row["l2_id"]].add(
                row["natural_action_sequence_sha256"]
            )
        if any(len(digests) != 1 for digests in action_digests.values()):
            raise ValueError("hierarchical identity A/B/C action sequence mismatch")
        _validate_output_manifest(release)
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": VALIDATION_SCHEMA,
        "passed": not violations,
        "cases": len(cases),
        "identity_rows": len(rows),
        "violations": violations,
    }

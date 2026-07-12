"""No-learning BC/A/B/C zero-residual simulator identity release."""

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

from bplus_v22 import ARMS, HISTORY_OFFSETS, LIDAR_BEAMS, MACRO_STEPS, OWNER_DECISION
from bplus_v22.model import MacroResidualAction, V22Policy
from bplus_v22.release import (
    file_sha256,
    validate_pinned_inputs,
    validate_source_preflight,
)
from bplus_v22.sidecar import load_sidecar_bundle, validate_sidecar_release
from d25.oracle import ARRAY_KEYS, compare_archived, load_bc_model, simulate_episode
from d25.search import trajectory_digest


EXPECTED_MAPS = ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")
EXPECTED_REGISTRY_SHA256 = "59c8967034e12dbcbcc57f776b6ff246c5a313c9b1ec58641d7eba151c4b4663"
REGISTRY_RELPATH = "Experiments/A0_project_registry/opened_registry.tsv"
CASE_SOURCE_RELPATH = (
    "Experiments/A4_d25_counterfactual/artifacts/full_oracle_20260711_185500/"
    "case_manifest.tsv"
)
BC_MODEL_RELPATH = "pretrained/end2race.pth"

IDENTITY_FIELDS = (
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
    "max_abs_residual_hex",
    "diagnostic_sha256",
    "trajectory_relpath",
)


class ZeroResidualActor(nn.Module):
    """Runs frozen BC each micro-step and a deterministic zero macro residual."""

    def __init__(self, policy: V22Policy, *, require_natural_zero: bool = True):
        super().__init__()
        self.policy = policy
        self.require_natural_zero = bool(require_natural_zero)
        self.reset_runtime()

    @property
    def gru(self):
        return self.policy.bc.gru

    def reset_runtime(self) -> None:
        self.micro_steps = 0
        self.macro_decisions = 0
        self.max_abs_residual = 0.0
        self.max_abs_natural_residual = 0.0
        self.natural_brake_decisions = 0
        self._lidar_history: list[torch.Tensor] = []
        self._speed_history: list[torch.Tensor] = []
        self._steer_history: list[torch.Tensor] = []
        self._command_speed_history: list[torch.Tensor] = []
        self._last_applied_command: tuple[float, float] | None = None
        self._awaiting_applied_command = False
        self._held_action: MacroResidualAction | None = None
        self._pending_actual_speed: float | None = None
        self._diagnostic_digest = hashlib.sha256(b"end2race:bplus-v2.2:diagnostic:v1\0")
        self._policy_decision_digest = hashlib.sha256(
            b"end2race:bplus-v2.2:forced-zero-policy-decision:v1\0"
        )

    def observe_actual_speed(self, value: float) -> None:
        """Receive current deployable speed without changing the BC interface."""

        speed = float(value)
        if not np.isfinite(speed):
            raise ValueError("zero-identity actual speed is nonfinite")
        if self._pending_actual_speed is not None:
            raise RuntimeError("zero-identity actual speed was not consumed")
        self._pending_actual_speed = speed

    def observe_applied_command(self, steer: float, speed: float) -> None:
        """Receive the command after the evaluator's deployment clipping."""

        command = (float(steer), float(speed))
        if not all(np.isfinite(value) for value in command):
            raise ValueError("zero-identity applied command is nonfinite")
        if not self._awaiting_applied_command:
            raise RuntimeError("zero-identity received an unexpected applied command")
        self._last_applied_command = command
        self._awaiting_applied_command = False

    def _history(self) -> tuple[torch.Tensor, torch.Tensor]:
        current = len(self._lidar_history) - 1
        indices = [max(0, current - offset) for offset in HISTORY_OFFSETS]
        lidar = torch.stack([self._lidar_history[index] for index in indices], dim=1)
        scalar = torch.cat(
            [
                torch.stack(
                    [self._speed_history[index] for index in indices], dim=1
                ).squeeze(-1),
                torch.stack(
                    [self._steer_history[index] for index in indices], dim=1
                ).squeeze(-1),
                torch.stack(
                    [self._command_speed_history[index] for index in indices], dim=1
                ).squeeze(-1),
            ],
            dim=1,
        )
        if lidar.shape != (1, len(HISTORY_OFFSETS), LIDAR_BEAMS):
            raise AssertionError("zero-identity LiDAR history shape drift")
        if scalar.shape != (1, 24):
            raise AssertionError("zero-identity scalar history shape drift")
        return lidar, scalar

    def _update_diagnostic_digest(self, output: dict[str, torch.Tensor]) -> None:
        for name in sorted(output):
            value = output[name].detach().cpu().contiguous()
            if not torch.all(torch.isfinite(value)):
                raise ValueError("zero-identity diagnostic contains nonfinite value")
            self._diagnostic_digest.update(name.encode("utf-8") + b"\0")
            self._diagnostic_digest.update(value.numpy().tobytes())

    def forward(self, lidar, previous_speed, hidden):
        if lidar.shape != (1, 1, LIDAR_BEAMS) or previous_speed.shape != (1, 1, 1):
            raise ValueError("zero-identity adapter supports exact batch-1 evaluator replay")
        if self._pending_actual_speed is None:
            raise RuntimeError("zero-identity adapter lacks current actual speed")
        if self._awaiting_applied_command:
            raise RuntimeError("zero-identity adapter lacks prior applied command")
        actual_speed = torch.full_like(
            previous_speed[:, -1], self._pending_actual_speed
        )
        self._pending_actual_speed = None
        base, bc_feature, next_hidden = self.policy.bc_step(lidar, previous_speed, hidden)
        if self._last_applied_command is None:
            prior_command = torch.zeros_like(base)
        else:
            prior_command = base.new_tensor(self._last_applied_command).reshape(1, 2)
        self._lidar_history.append((torch.clamp(lidar[:, -1], 0.0, 30.0) / 30.0).detach())
        self._speed_history.append((actual_speed / 10.0).detach())
        self._steer_history.append((prior_command[:, 0:1] / 0.52).detach())
        self._command_speed_history.append((prior_command[:, 1:2] / 10.0).detach())
        if self.micro_steps % MACRO_STEPS == 0:
            lidar_history, scalar_history = self._history()
            distribution = self.policy.distribution(bc_feature, lidar_history, scalar_history)
            action = distribution.deterministic()
            delta = distribution.physical_delta(action)
            self.max_abs_natural_residual = max(
                self.max_abs_natural_residual, float(delta.abs().max().item())
            )
            self.natural_brake_decisions += int(action.brake_gate.item())
            for value in (action.as_tensor(), delta, distribution.brake_probability):
                tensor = value.detach().cpu().contiguous()
                self._policy_decision_digest.update(tensor.numpy().tobytes())
            if self.require_natural_zero and not torch.equal(delta, torch.zeros_like(delta)):
                raise AssertionError("zero-identity policy emitted nonzero residual")
            if self.require_natural_zero and not torch.equal(
                action.brake_gate, torch.zeros_like(action.brake_gate)
            ):
                raise AssertionError("zero-identity policy selected BRAKE")
            self._held_action = action
            self.macro_decisions += 1
            self._update_diagnostic_digest(
                self.policy.diagnostic(bc_feature, lidar_history, scalar_history)
            )
        if self._held_action is None:
            raise AssertionError("zero-identity adapter lacks held action")
        delta = torch.zeros_like(base)
        # The held latent action was already proven to map to exact zero at its
        # macro boundary. Avoid recomputing the sidecar at 100 Hz; compose the
        # same physical zero while BC continues to update every micro-step.
        self.max_abs_residual = max(self.max_abs_residual, float(delta.abs().max().item()))
        action = base + delta
        if not torch.equal(action, base):
            raise AssertionError("zero-identity composition changed BC action")
        self._awaiting_applied_command = True
        self.micro_steps += 1
        return action.unsqueeze(1), next_hidden

    def accounting(self) -> dict:
        if self.micro_steps <= 0:
            raise ValueError("zero-identity adapter executed no micro-step")
        full, remainder = divmod(self.micro_steps, MACRO_STEPS)
        lengths = [MACRO_STEPS] * full + ([remainder] if remainder else [])
        if len(lengths) != self.macro_decisions or sum(lengths) != self.micro_steps:
            raise AssertionError("zero-identity macro accounting mismatch")
        return {
            "micro_steps": self.micro_steps,
            "macro_decisions": self.macro_decisions,
            "macro_lengths": lengths,
            "max_abs_residual": self.max_abs_residual,
            "max_abs_natural_residual": self.max_abs_natural_residual,
            "natural_brake_decisions": self.natural_brake_decisions,
            "diagnostic_sha256": self._diagnostic_digest.hexdigest(),
            "policy_decision_sha256": self._policy_decision_digest.hexdigest(),
        }


def _read_cases(repo_root: Path) -> list[dict[str, str]]:
    with (repo_root / CASE_SOURCE_RELPATH).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = []
    for map_name in EXPECTED_MAPS:
        candidates = sorted(
            (row for row in rows if row["map_name"] == map_name), key=lambda row: row["l2_id"]
        )
        if not candidates:
            raise ValueError(f"zero-identity source has no case for {map_name}")
        selected.append(dict(candidates[0]))
    return selected


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_output_manifest(directory: Path) -> None:
    relpaths = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    )
    (directory / "output_manifest.sha256").write_text(
        "\n".join(f"{file_sha256(directory / relpath)}  {relpath}" for relpath in relpaths)
        + "\n",
        encoding="utf-8",
    )


def _save_trajectory(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: arrays[key] for key in ARRAY_KEYS})


def _arrays_equal(left: dict, right: dict) -> bool:
    return all(
        key in left
        and key in right
        and np.asarray(left[key]).dtype == np.asarray(right[key]).dtype
        and np.asarray(left[key]).shape == np.asarray(right[key]).shape
        and np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
        for key in ARRAY_KEYS
    )


def run_zero_identity(
    repo_root: str | Path,
    source_preflight_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
    device_name: str = "cuda:0",
    sidecar_release_dir: str | Path | None = None,
) -> dict:
    root = Path(repo_root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("zero-identity runner must execute from repo root")
    numba_cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    if not numba_cache_dir or not Path(numba_cache_dir).is_absolute():
        raise ValueError("zero-identity requires an isolated absolute NUMBA_CACHE_DIR")
    source_validation = validate_source_preflight(source_preflight_dir, root)
    if not source_validation["passed"]:
        raise ValueError(f"zero-identity source preflight failed: {source_validation}")
    inputs = validate_pinned_inputs(root)
    if not inputs["passed"]:
        raise ValueError(f"zero-identity pinned inputs failed: {inputs}")
    registry = root / REGISTRY_RELPATH
    registry_before = file_sha256(registry)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("zero-identity CUDA requested but unavailable")
    device = torch.device(device_name)
    if sidecar_release_dir is None:
        sidecar_state = None
        sidecar_mean = None
        sidecar_std = None
        sidecar_role = "UNFITTED_STRUCTURAL_PLACEHOLDER"
        sidecar_fit_started = False
        pretrained_sidecar_gate_satisfied = False
        scope = "STRUCTURAL_WIRING_IDENTITY_ONLY"
        sidecar_release_relpath = None
        sidecar_release_manifest_sha256 = None
        expected_registry_sha256 = EXPECTED_REGISTRY_SHA256
    else:
        sidecar_validation = validate_sidecar_release(sidecar_release_dir, root)
        if not sidecar_validation["passed"]:
            raise ValueError(f"zero-identity fitted sidecar failed: {sidecar_validation}")
        sidecar_state, sidecar_mean, sidecar_std, _ = load_sidecar_bundle(
            sidecar_release_dir
        )
        sidecar_role = "FULL_NON_TEST_INITIALIZATION"
        sidecar_fit_started = True
        pretrained_sidecar_gate_satisfied = True
        scope = "FITTED_SIDECAR_CHECKPOINT_IDENTITY"
        sidecar_release_relpath = str(Path(sidecar_release_dir))
        sidecar_release_manifest_sha256 = file_sha256(
            Path(sidecar_release_dir) / "output_manifest.sha256"
        )
        sidecar_config = json.loads(
            (Path(sidecar_release_dir) / "config.json").read_text(encoding="utf-8")
        )
        expected_registry_sha256 = sidecar_config["registry_after_sha256"]
    if registry_before != expected_registry_sha256:
        raise ValueError("zero-identity registry hash drift for selected sidecar scope")
    output = Path(output_dir)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("zero-identity output/partial already exists")
    partial.mkdir(parents=True)
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
        bc_model = load_bc_model(str(root / BC_MODEL_RELPATH), device)
        bc_state = {name: value.detach().cpu().clone() for name, value in bc_model.state_dict().items()}
        baselines = {}
        rows: list[dict[str, str]] = []
        for case_order, case in enumerate(cases):
            source_path = root / case["npz_relpath"]
            if file_sha256(source_path) != case["npz_sha256"]:
                raise ValueError(f"zero-identity source archive hash drift: {case['l2_id']}")
            with np.load(source_path, allow_pickle=False) as source:
                archived = {key: np.asarray(source[key]) for key in source.files}
            first = simulate_episode(bc_model, device, case)
            second = simulate_episode(bc_model, device, case)
            archive_comparison = compare_archived(first.arrays, archived)
            passed = archive_comparison["passed"] and _arrays_equal(first.arrays, second.arrays)
            if not passed or first.action_clipped or second.action_clipped:
                raise AssertionError(f"zero-identity BC replay failed: {case['l2_id']}")
            digest = trajectory_digest(first.arrays)
            baselines[case["l2_id"]] = first
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
                    "run1_matches_archive": str(archive_comparison["passed"]).lower(),
                    "four_state": first.outcome.four_state,
                    "action_clipped": "false",
                    "micro_steps": str(len(first.arrays["time"])),
                    "macro_decisions": "0",
                    "macro_lengths_json": "[]",
                    "max_abs_residual_hex": float(0.0).hex(),
                    "diagnostic_sha256": "NA",
                    "trajectory_relpath": relpath,
                }
            )

        del bc_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model_initializations = []
        for arm in ARMS:
            policy = V22Policy(
                arm,
                bc_state_dict=bc_state,
                sidecar_state_dict=sidecar_state,
                sidecar_bc_mean=sidecar_mean,
                sidecar_bc_std=sidecar_std,
            ).to(device).eval()
            model_initializations.append(
                {
                    "arm": arm,
                    "sidecar_role": sidecar_role,
                    "policy_sidecar_encoder_sha256": policy.policy_sidecar_encoder_sha256(),
                    "shadow_sidecar_sha256": policy.shadow_sha256(),
                }
            )
            adapter = ZeroResidualActor(policy).to(device).eval()
            for case_order, case in enumerate(cases):
                baseline = baselines[case["l2_id"]]
                adapter.reset_runtime()
                first = simulate_episode(adapter, device, case)
                first_accounting = adapter.accounting()
                adapter.reset_runtime()
                second = simulate_episode(adapter, device, case)
                second_accounting = adapter.accounting()
                matches_baseline = _arrays_equal(first.arrays, baseline.arrays)
                rerun_match = _arrays_equal(first.arrays, second.arrays)
                if (
                    not matches_baseline
                    or not rerun_match
                    or first_accounting != second_accounting
                    or first_accounting["max_abs_residual"] != 0.0
                    or first.action_clipped
                    or second.action_clipped
                ):
                    raise AssertionError(f"zero-identity arm replay failed: {arm}/{case['l2_id']}")
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
                        "run1_matches_baseline": str(matches_baseline).lower(),
                        "run2_matches_run1": str(rerun_match).lower(),
                        "run1_matches_archive": "true",
                        "four_state": first.outcome.four_state,
                        "action_clipped": "false",
                        "micro_steps": str(first_accounting["micro_steps"]),
                        "macro_decisions": str(first_accounting["macro_decisions"]),
                        "macro_lengths_json": json.dumps(first_accounting["macro_lengths"]),
                        "max_abs_residual_hex": float(
                            first_accounting["max_abs_residual"]
                        ).hex(),
                        "diagnostic_sha256": first_accounting["diagnostic_sha256"],
                        "trajectory_relpath": relpath,
                    }
                )
            del adapter, policy
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        initialization_by_arm = {row["arm"]: row for row in model_initializations}
        if (
            initialization_by_arm["SIDECAR_FROZEN"]["policy_sidecar_encoder_sha256"]
            != initialization_by_arm["SIDECAR_FINETUNE"][
                "policy_sidecar_encoder_sha256"
            ]
        ):
            raise AssertionError("zero-identity B/C sidecar initialization mismatch")
        if len({row["shadow_sidecar_sha256"] for row in model_initializations}) != 1:
            raise AssertionError("zero-identity shadow initialization mismatch")

        _write_tsv(partial / "identity_results.tsv", rows, IDENTITY_FIELDS)
        _write_json(
            partial / "model_initializations.json",
            {
                "schema": "bplus-v2.2-zero-identity-models-1",
                "sidecar_fit_started": sidecar_fit_started,
                "pretrained_sidecar_gate_satisfied": pretrained_sidecar_gate_satisfied,
                "sidecar_release_relpath": sidecar_release_relpath,
                "sidecar_release_output_manifest_sha256": sidecar_release_manifest_sha256,
                "models": model_initializations,
            },
        )
        registry_after = file_sha256(registry)
        if registry_after != registry_before:
            raise AssertionError("zero-identity mutated registry")
        config = {
            "schema": "bplus-v2.2-zero-identity-config-1",
            "created_at": str(created_at),
            "owner_decision": OWNER_DECISION,
            "old_d2r_gate_passed": False,
            "ttc_role": "diagnostic_only",
            "policy_training_started": False,
            "sidecar_fit_started": sidecar_fit_started,
            "pretrained_sidecar_gate_satisfied": pretrained_sidecar_gate_satisfied,
            "sidecar_role": sidecar_role,
            "scope": scope,
            "sidecar_release_relpath": sidecar_release_relpath,
            "sidecar_release_output_manifest_sha256": sidecar_release_manifest_sha256,
            "registry_mutated": False,
            "registry_before_sha256": registry_before,
            "registry_after_sha256": registry_after,
            "source_preflight_relpath": str(Path(source_preflight_dir)),
            "source_preflight_output_manifest_sha256": file_sha256(
                Path(source_preflight_dir) / "output_manifest.sha256"
            ),
            "device": str(device),
            "numba_cache_dir": numba_cache_dir,
            "cases": len(cases),
            "variants": ("BC", *ARMS),
            "reruns_per_variant": 2,
            "saved_trajectories": len(rows),
        }
        _write_json(partial / "config.json", config)
        _write_json(
            partial / "validation.json",
            {
                "schema": "bplus-v2.2-zero-identity-validation-1",
                "passed": True,
                "cases": len(cases),
                "variants": 4,
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
    validation = validate_zero_identity(output)
    if not validation["passed"]:
        raise AssertionError(f"created invalid zero-identity release: {validation}")
    return {
        "passed": True,
        "cases": validation["cases"],
        "identity_rows": validation["identity_rows"],
        "output_manifest_sha256": file_sha256(output / "output_manifest.sha256"),
        "registry_sha256": registry_before,
    }


def validate_zero_identity(release_dir: str | Path) -> dict:
    release = Path(release_dir)
    violations = []
    rows = []
    cases = []
    try:
        if not (release / "COMPLETE").is_file():
            raise ValueError("zero-identity release lacks COMPLETE")
        config = json.loads((release / "config.json").read_text(encoding="utf-8"))
        if config["owner_decision"] != OWNER_DECISION or config["old_d2r_gate_passed"] is not False:
            raise ValueError("zero-identity authority mismatch")
        if config["policy_training_started"] is not False or config["registry_mutated"] is not False:
            raise ValueError("zero-identity claims training/registry mutation")
        fitted = config.get("sidecar_fit_started") is True
        if fitted:
            if (
                config.get("pretrained_sidecar_gate_satisfied") is not True
                or config.get("sidecar_role") != "FULL_NON_TEST_INITIALIZATION"
                or config.get("scope") != "FITTED_SIDECAR_CHECKPOINT_IDENTITY"
                or not config.get("sidecar_release_relpath")
            ):
                raise ValueError("zero-identity fitted-sidecar scope mismatch")
            sidecar_release = Path(config["sidecar_release_relpath"])
            sidecar_validation = validate_sidecar_release(sidecar_release)
            if not sidecar_validation["passed"]:
                raise ValueError("zero-identity referenced fitted sidecar is invalid")
            if file_sha256(sidecar_release / "output_manifest.sha256") != config[
                "sidecar_release_output_manifest_sha256"
            ]:
                raise ValueError("zero-identity fitted-sidecar manifest mismatch")
        elif (
            config.get("pretrained_sidecar_gate_satisfied") is not False
            or config.get("sidecar_role") != "UNFITTED_STRUCTURAL_PLACEHOLDER"
            or config.get("scope") != "STRUCTURAL_WIRING_IDENTITY_ONLY"
            or config.get("sidecar_release_relpath") is not None
        ):
            raise ValueError("zero-identity structural scope mismatch")
        if not Path(config.get("numba_cache_dir", "")).is_absolute():
            raise ValueError("zero-identity Numba cache identity is missing")
        expected_registry = EXPECTED_REGISTRY_SHA256
        if fitted:
            sidecar_config = json.loads(
                (sidecar_release / "config.json").read_text(encoding="utf-8")
            )
            expected_registry = sidecar_config["registry_after_sha256"]
        if config["registry_before_sha256"] != expected_registry:
            raise ValueError("zero-identity registry-before hash mismatch")
        if config["registry_after_sha256"] != expected_registry:
            raise ValueError("zero-identity registry-after hash mismatch")
        with (release / "case_manifest.tsv").open(newline="", encoding="utf-8") as handle:
            cases = list(csv.DictReader(handle, delimiter="\t"))
        if len(cases) != 4 or tuple(row["map_name"] for row in cases) != EXPECTED_MAPS:
            raise ValueError("zero-identity case/map accounting mismatch")
        model_initializations = json.loads(
            (release / "model_initializations.json").read_text(encoding="utf-8")
        )
        if (
            model_initializations.get("sidecar_fit_started") is not fitted
            or model_initializations.get("pretrained_sidecar_gate_satisfied")
            is not fitted
        ):
            raise ValueError("zero-identity model initialization scope mismatch")
        models = model_initializations.get("models", [])
        if [row.get("arm") for row in models] != list(ARMS):
            raise ValueError("zero-identity model initialization arm mismatch")
        if any(row.get("sidecar_role") != config["sidecar_role"] for row in models):
            raise ValueError("zero-identity model sidecar role mismatch")
        by_arm = {row["arm"]: row for row in models}
        if (
            by_arm["SIDECAR_FROZEN"]["policy_sidecar_encoder_sha256"]
            != by_arm["SIDECAR_FINETUNE"]["policy_sidecar_encoder_sha256"]
        ):
            raise ValueError("zero-identity persisted B/C sidecar mismatch")
        if len({row["shadow_sidecar_sha256"] for row in models}) != 1:
            raise ValueError("zero-identity persisted shadow mismatch")
        with (release / "identity_results.tsv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != IDENTITY_FIELDS:
                raise ValueError("zero-identity result header mismatch")
            rows = list(reader)
        if len(rows) != 16:
            raise ValueError("zero-identity result count mismatch")
        expected_pairs = {(case["l2_id"], variant) for case in cases for variant in ("BC", *ARMS)}
        if {(row["l2_id"], row["variant"]) for row in rows} != expected_pairs:
            raise ValueError("zero-identity case/variant Cartesian product mismatch")
        for row in rows:
            if row["run1_matches_baseline"] != "true" or row["run2_matches_run1"] != "true":
                raise ValueError("zero-identity contains trajectory mismatch")
            if row["run1_matches_archive"] != "true" or row["action_clipped"] != "false":
                raise ValueError("zero-identity archive/clipping gate failed")
            if row["run1_trajectory_sha256"] != row["run2_trajectory_sha256"]:
                raise ValueError("zero-identity rerun digest mismatch")
            if row["run1_trajectory_sha256"] != row["baseline_trajectory_sha256"]:
                raise ValueError("zero-identity baseline digest mismatch")
            path = release / row["trajectory_relpath"]
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != set(ARRAY_KEYS) or trajectory_digest(arrays) != row["run1_trajectory_sha256"]:
                    raise ValueError("zero-identity saved trajectory mismatch")
            if row["variant"] == "BC":
                if row["macro_decisions"] != "0" or row["macro_lengths_json"] != "[]":
                    raise ValueError("zero-identity BC row has macro accounting")
            else:
                lengths = json.loads(row["macro_lengths_json"])
                if len(lengths) != int(row["macro_decisions"]):
                    raise ValueError("zero-identity macro decision count mismatch")
                if sum(lengths) != int(row["micro_steps"]):
                    raise ValueError("zero-identity macro lengths do not cover trajectory")
                if any(not 1 <= value <= MACRO_STEPS for value in lengths):
                    raise ValueError("zero-identity macro length outside 1..10")
                if float.fromhex(row["max_abs_residual_hex"]) != 0.0:
                    raise ValueError("zero-identity nonzero residual")
                if len(row["diagnostic_sha256"]) != 64:
                    raise ValueError("zero-identity diagnostic digest missing")
        manifest = {}
        for line in (release / "output_manifest.sha256").read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            if relative in manifest:
                raise ValueError("zero-identity duplicate output manifest path")
            manifest[relative] = digest
        observed = {
            path.relative_to(release).as_posix()
            for path in release.rglob("*")
            if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
        }
        if set(manifest) != observed:
            raise ValueError("zero-identity output inventory mismatch")
        for relative, digest in manifest.items():
            if file_sha256(release / relative) != digest:
                raise ValueError(f"zero-identity output hash mismatch: {relative}")
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": "bplus-v2.2-zero-identity-validation-1",
        "passed": not violations,
        "cases": len(cases),
        "identity_rows": len(rows),
        "violations": violations,
    }

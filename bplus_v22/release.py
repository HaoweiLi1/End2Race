"""Atomic source/input preflight release for B+ v2.2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import torch

from bplus_v22 import (
    BC_CHECKPOINT_SHA256,
    D25_OUTPUT_MANIFEST_SHA256,
    D2R_SIGNALS_MANIFEST_SHA256,
    D2_DATASET_MANIFEST_SHA256,
    D2_SPLIT_MANIFEST_SHA256,
    D2_TEST_SEAL_SHA256,
    LOCKED_CONFIG,
    OWNER_DECISION,
    V22_PLAN_SHA256,
    V22_SPEC_SHA256,
)


PINNED_INPUTS = {
    "bc_checkpoint": (
        "pretrained/end2race.pth",
        BC_CHECKPOINT_SHA256,
    ),
    "d2_dataset_manifest": (
        "Experiments/A3_d2_representation/artifacts/"
        "non_test_full_20260711_175713/dataset_manifest.json",
        D2_DATASET_MANIFEST_SHA256,
    ),
    "d2_scenario_split": (
        "Experiments/A3_d2_representation/artifacts/split_lock/scenario_split.tsv",
        D2_SPLIT_MANIFEST_SHA256,
    ),
    "d2_test_seal": (
        "Experiments/A3_d2_representation/artifacts/split_lock/test_seal.json",
        D2_TEST_SEAL_SHA256,
    ),
    "d2r_signals_manifest": (
        "Experiments/A3_d2_representation/artifacts/"
        "deployable_signals_20260711_182229/signals_manifest.json",
        D2R_SIGNALS_MANIFEST_SHA256,
    ),
    "d25_output_manifest": (
        "Experiments/A4_d25_counterfactual/artifacts/"
        "full_oracle_20260711_185500/output_manifest.sha256",
        D25_OUTPUT_MANIFEST_SHA256,
    ),
    "v22_spec": (
        "docs/superpowers/specs/2026-07-11-ppo-safety-first-bplus-v2.2.md",
        V22_SPEC_SHA256,
    ),
    "v22_plan": (
        "docs/superpowers/plans/2026-07-11-bplus-v2.2-d3r2-implementation-plan.md",
        V22_PLAN_SHA256,
    ),
}

SOURCE_PATHS = (
    "bplus_v22/__init__.py",
    "bplus_v22/macro.py",
    "bplus_v22/buffer.py",
    "bplus_v22/model.py",
    "bplus_v22/objective.py",
    "bplus_v22/sidecar.py",
    "bplus_v22/warmstart.py",
    "bplus_v22/identity.py",
    "bplus_v22/manifests.py",
    "bplus_v22/checkpoint_preflight.py",
    "bplus_v22/closed_loop.py",
    "bplus_v22/remediated_model.py",
    "bplus_v22/hierarchical_identity.py",
    "bplus_v22/hierarchical_warmstart.py",
    "bplus_v22/hierarchical_checkpoint_preflight.py",
    "bplus_v22/hierarchical_closed_loop.py",
    "bplus_v22/release.py",
    "bplus_v22/cli.py",
    "model.py",
    "ppo_utils.py",
    "train_ppo.py",
    "d25/oracle.py",
    "d25/search.py",
    "d25/__init__.py",
    "d0/outcomes.py",
    "d0/identity.py",
    "eval_multiagent.py",
    "demonstration.py",
    "latticeplanner/utils.py",
    "utils.py",
    "d2r/__init__.py",
    "d2/dataset.py",
    "d2/metrics.py",
    "d2/models.py",
    "d2/probe.py",
    "d2/release.py",
    "d2/replay.py",
    "d2r/data.py",
    "d2r/model.py",
    "d2r/train.py",
    "tests/test_bplus_v22_config.py",
    "tests/test_bplus_v22_macro.py",
    "tests/test_bplus_v22_buffer.py",
    "tests/test_bplus_v22_model.py",
    "tests/test_bplus_v22_objective.py",
    "tests/test_bplus_v22_sidecar.py",
    "tests/test_bplus_v22_warmstart.py",
    "tests/test_bplus_v22_identity.py",
    "tests/test_bplus_v22_manifests.py",
    "tests/test_bplus_v22_closed_loop.py",
    "tests/test_bplus_v22_remediated_model.py",
    "tests/test_bplus_v22_hierarchical_identity.py",
    "tests/test_bplus_v22_hierarchical_warmstart.py",
    "tests/test_bplus_v22_hierarchical_eval.py",
    "tests/test_bplus_v22_release.py",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_manifest(directory: Path) -> None:
    files = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != "COMPLETE"
    )
    if any(path.name == "output_manifest.sha256" for path in files):
        raise ValueError("output manifest already exists before inventory")
    lines = [f"{file_sha256(path)}  {path.relative_to(directory).as_posix()}" for path in files]
    (directory / "output_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_pinned_inputs(repo_root: str | Path) -> dict:
    root = Path(repo_root).resolve()
    observed = {}
    violations = []
    for name, (relative, expected) in PINNED_INPUTS.items():
        path = root / relative
        if not path.is_file():
            violations.append(f"missing pinned input: {relative}")
            continue
        actual = file_sha256(path)
        observed[name] = {"relpath": relative, "sha256": actual}
        if actual != expected:
            violations.append(f"pinned input hash drift: {relative}")
    return {"passed": not violations, "inputs": observed, "violations": violations}


def _source_inventory(repo_root: Path) -> list[dict]:
    rows = []
    for relative in SOURCE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"missing v2.2 source: {relative}")
        rows.append(
            {"relpath": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        )
    return rows


def create_source_preflight(
    output_dir: str | Path,
    created_at: str,
    repo_root: str | Path = ".",
) -> dict:
    root = Path(repo_root).resolve()
    output = Path(output_dir)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("v2.2 preflight output/partial already exists")
    partial.mkdir(parents=True)
    try:
        inputs = validate_pinned_inputs(root)
        if not inputs["passed"]:
            raise ValueError(f"v2.2 pinned input failure: {inputs['violations']}")
        sources = _source_inventory(root)
        authority = {
            "schema": "bplus-v2.2-authority-1",
            "created_at": str(created_at),
            "owner_decision": OWNER_DECISION,
            "old_d2r_decision": "STOP_D3_TEST_UNOPENED_D2R_G_FAILED_TTC_AND_2S_FA",
            "old_d2r_gate_passed": False,
            "ttc_role": "diagnostic_only",
            "d2_test_opened": False,
            "policy_training_started": False,
            "locked_config": LOCKED_CONFIG.__dict__,
        }
        environment = {
            "schema": "bplus-v2.2-environment-1",
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
        }
        _write_json(partial / "authority.json", authority)
        _write_json(partial / "pinned_inputs.json", inputs)
        _write_json(partial / "source_inventory.json", {"schema": "bplus-v2.2-sources-1", "sources": sources})
        _write_json(partial / "environment.json", environment)
        _write_json(
            partial / "validation.json",
            {
                "schema": "bplus-v2.2-source-preflight-validation-1",
                "passed": True,
                "pinned_inputs": len(inputs["inputs"]),
                "source_files": len(sources),
                "violations": [],
            },
        )
        _write_manifest(partial)
        (partial / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    validation = validate_source_preflight(output, root)
    if not validation["passed"]:
        raise AssertionError(f"created invalid v2.2 preflight: {validation}")
    return {
        "passed": True,
        "output_manifest_sha256": file_sha256(output / "output_manifest.sha256"),
        "source_files": validation["source_files"],
        "pinned_inputs": validation["pinned_inputs"],
    }


def validate_source_preflight(release_dir: str | Path, repo_root: str | Path = ".") -> dict:
    release = Path(release_dir)
    root = Path(repo_root).resolve()
    violations = []
    required = {
        "COMPLETE",
        "authority.json",
        "environment.json",
        "output_manifest.sha256",
        "pinned_inputs.json",
        "source_inventory.json",
        "validation.json",
    }
    observed_files = {path.name for path in release.iterdir() if path.is_file()} if release.is_dir() else set()
    if observed_files != required:
        violations.append("source preflight output inventory mismatch")
    manifest_entries = {}
    manifest = release / "output_manifest.sha256"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            parts = line.split("  ", 1)
            if len(parts) != 2 or parts[1] in manifest_entries:
                violations.append("source preflight manifest format/duplicate error")
                continue
            manifest_entries[parts[1]] = parts[0]
        expected_manifest = required - {"COMPLETE", "output_manifest.sha256"}
        if set(manifest_entries) != expected_manifest:
            violations.append("source preflight manifest inventory mismatch")
        for relative, expected in manifest_entries.items():
            path = release / relative
            if not path.is_file() or file_sha256(path) != expected:
                violations.append(f"source preflight output hash mismatch: {relative}")
    else:
        violations.append("source preflight lacks output manifest")

    authority = {}
    sources = []
    inputs = {}
    try:
        authority = json.loads((release / "authority.json").read_text(encoding="utf-8"))
        if authority.get("owner_decision") != OWNER_DECISION:
            violations.append("source preflight owner decision mismatch")
        if authority.get("old_d2r_gate_passed") is not False:
            violations.append("source preflight rewrites old D2R result")
        if authority.get("ttc_role") != "diagnostic_only":
            violations.append("source preflight TTC role mismatch")
        if authority.get("d2_test_opened") is not False:
            violations.append("source preflight claims D2 test opened")
        sources = json.loads((release / "source_inventory.json").read_text(encoding="utf-8"))["sources"]
        if [row["relpath"] for row in sources] != list(SOURCE_PATHS):
            violations.append("source preflight source ordering/inventory mismatch")
        for row in sources:
            path = root / row["relpath"]
            if not path.is_file() or file_sha256(path) != row["sha256"]:
                violations.append(f"live source hash drift: {row['relpath']}")
        inputs = json.loads((release / "pinned_inputs.json").read_text(encoding="utf-8"))
        live_inputs = validate_pinned_inputs(root)
        if not live_inputs["passed"] or inputs != live_inputs:
            violations.append("source preflight pinned input snapshot/live mismatch")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        violations.append(f"source preflight parse failure: {type(error).__name__}: {error}")
    return {
        "schema": "bplus-v2.2-source-preflight-validation-1",
        "passed": not violations,
        "source_files": len(sources),
        "pinned_inputs": len(inputs.get("inputs", {})) if isinstance(inputs, dict) else 0,
        "violations": violations,
    }

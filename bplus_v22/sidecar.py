"""Prospective registry plan and full-non-test sidecar initialization release."""

from __future__ import annotations

import csv
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Iterable, Mapping

import numpy as np
import torch

from bplus_v22 import (
    ARMS,
    D2_DATASET_MANIFEST_SHA256,
    D2R_SIGNALS_MANIFEST_SHA256,
    D2_SPLIT_MANIFEST_SHA256,
    D2_TEST_SEAL_SHA256,
    OWNER_DECISION,
)
from bplus_v22.release import (
    file_sha256,
    validate_pinned_inputs,
    validate_source_preflight,
)
from d0.identity import (
    REGISTRY_FIELDS,
    append_opened_registry,
    registry_row_id,
    validate_registry_row,
)
from d2r import LOCKED_CONFIG as D2R_LOCKED_CONFIG
from d2r import SEED as D2R_SEED
from d2r.data import D2RDataset, deterministic_fit_indices
from d2r.model import D2RGeometryNet
from d2r.train import PREDICTION_NAMES, predict_model, train_model


DATASET_RELPATH = (
    "logs/d2_representation_20260711_174039/artifacts/"
    "non_test_full_20260711_175713"
)
SPLIT_RELPATH = "logs/d2_representation_20260711_174039/artifacts/split_lock"
SIGNALS_RELPATH = (
    "logs/d2_representation_20260711_174039/artifacts/"
    "deployable_signals_20260711_182229"
)
REGISTRY_RELPATH = "logs/ppo_next_unattended_20260710_230212/opened_registry.tsv"
EVIDENCE_RELPATH = "logs/bplus_v22_d3r2_20260711"

EXPECTED_REGISTRY_BEFORE_SHA256 = (
    "59c8967034e12dbcbcc57f776b6ff246c5a313c9b1ec58641d7eba151c4b4663"
)
EPISODE_METADATA_SHA256 = (
    "468d8be50aecad19f89fbf2c35dc421acb4244a61f957f77dcfff1acd227eda3"
)
REGISTRY_OPENED_AT = "2026-07-12T07:50:00+08:00"
REGISTRY_STAGE = "D3-R2-v2.2"
REGISTRY_USE_CLASS = "actor_pretrain"
REGISTRY_DECISION_EFFECT = "model_choice"
REGISTRY_SOURCE_RUN_ID = "actor_pretrain_sidecar_initialization_v1"
REGISTRY_SPLIT_ID = f"d3r2_v22_actor_pretrain_{D2_DATASET_MANIFEST_SHA256[:16]}"

D2R_FULL_CONFIG_SHA256 = (
    "5713b1a2dd8686fed638249191c32402a135ecfaf1c036f5897bbcfc650dae6d"
)
D2R_FULL_OUTPUT_MANIFEST_SHA256 = (
    "be7936acc95b9a98a3a97d4248d94b11ea8c4ed8adacc82a3dde513323b7c057"
)
D2R_FULL_CONFIG_RELPATH = (
    "logs/d2r_geometry_20260711/artifacts/full_oof_20260711_210200/config.json"
)
D2R_FULL_OUTPUT_MANIFEST_RELPATH = (
    "logs/d2r_geometry_20260711/artifacts/full_oof_20260711_210200/"
    "output_manifest.sha256"
)
D2R_CORE_SOURCE_SHA256 = {
    "d2r/__init__.py": "1f59801ac8f2e65754c1906325c527461d76d47dba85ec284db49c469468b2e9",
    "d2r/data.py": "0adb6dac5cbe72b4049c63d647db46311937dbb5da6d60fd3d94af5435e73831",
    "d2r/model.py": "0cc40717259b056aacf6bf7ae4474352c4016272da46d4aaee5030d4a4476293",
    "d2r/train.py": "a0d49dae686d507ccdfef46951df70036742b0671086f28b587c1f3d3ad32732",
}

VALIDATION_FRAME_COUNT = 512
EXPECTED_MAPS = ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: Iterable[Mapping[str, str]], fields=REGISTRY_FIELDS) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            if set(row) != set(fields):
                raise ValueError(f"sidecar TSV field mismatch: {path.name}")
            writer.writerow({field: row[field] for field in fields})


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prepare_output(output: Path) -> Path:
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("sidecar output/partial already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    return partial


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
    manifest_path = directory / "output_manifest.sha256"
    if not manifest_path.is_file():
        raise ValueError("sidecar output manifest missing")
    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relpath = line.split("  ", 1)
        if relpath in expected:
            raise ValueError("sidecar output manifest has duplicate path")
        expected[relpath] = digest
    observed = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    }
    if set(expected) != observed:
        raise ValueError("sidecar output manifest inventory mismatch")
    for relpath, digest in expected.items():
        if file_sha256(directory / relpath) != digest:
            raise ValueError(f"sidecar output hash mismatch: {relpath}")


def _promote(partial: Path, output: Path) -> None:
    os.replace(partial, output)
    (output / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")


def _tensor_digest(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(items):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _array_digest(items: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(items):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_frozen_d2r_sources(root: Path) -> dict[str, str]:
    observed = {}
    for relpath, expected in D2R_CORE_SOURCE_SHA256.items():
        actual = file_sha256(root / relpath)
        observed[relpath] = actual
        if actual != expected:
            raise ValueError(f"sidecar frozen D2R source drift: {relpath}")
    if file_sha256(root / D2R_FULL_CONFIG_RELPATH) != D2R_FULL_CONFIG_SHA256:
        raise ValueError("sidecar frozen D2R config drift")
    if (
        file_sha256(root / D2R_FULL_OUTPUT_MANIFEST_RELPATH)
        != D2R_FULL_OUTPUT_MANIFEST_SHA256
    ):
        raise ValueError("sidecar frozen D2R artifact-manifest drift")
    return observed


def _episode_rows(root: Path) -> list[dict[str, str]]:
    metadata = root / DATASET_RELPATH / "episode_metadata.tsv"
    if file_sha256(metadata) != EPISODE_METADATA_SHA256:
        raise ValueError("sidecar episode metadata hash drift")
    rows = _read_tsv(metadata)
    if len(rows) != 1928 or len({row["l2_id"] for row in rows}) != 1928:
        raise ValueError("sidecar episode metadata population mismatch")
    if set(row["map_name"] for row in rows) != set(EXPECTED_MAPS):
        raise ValueError("sidecar episode metadata map mismatch")
    if any(int(row["outer_fold"]) not in range(5) for row in rows):
        raise ValueError("sidecar episode metadata outer-fold drift")
    split_rows = _read_tsv(root / SPLIT_RELPATH / "scenario_split.tsv")
    split = {row["l2_id"]: row for row in split_rows}
    if any(split.get(row["l2_id"], {}).get("split") != "non_test" for row in rows):
        raise ValueError("sidecar actor-pretrain population includes sealed test L2")
    return rows


def _validate_sealed_test_absence(
    root: Path,
    non_test_episodes: Iterable[Mapping[str, str]],
    registry_rows: Iterable[Mapping[str, str]],
) -> dict:
    seal = root / SPLIT_RELPATH / "test_seal.json"
    if file_sha256(seal) != D2_TEST_SEAL_SHA256:
        raise ValueError("sidecar D2 test-seal hash drift")
    split_rows = _read_tsv(root / SPLIT_RELPATH / "scenario_split.tsv")
    test_ids = {row["l2_id"] for row in split_rows if row["split"] == "test"}
    non_test_ids = {str(row["l2_id"]) for row in non_test_episodes}
    if len(test_ids) != 1108 or test_ids & non_test_ids:
        raise ValueError("sidecar non-test/test L2 boundary mismatch")
    forbidden_registry = [
        row
        for row in registry_rows
        if row["stage"] in {"D2", "D2R-G", REGISTRY_STAGE}
        and row["l2_id"] in test_ids
    ]
    if forbidden_registry:
        raise ValueError("sidecar registry contains sealed-test reuse")
    artifact_root = root / "logs/d2_representation_20260711_174039/artifacts"
    forbidden_artifacts = sorted(
        path.name
        for pattern in (
            "test_full_*",
            "test_dataset_*",
            "test_features_*",
            "test_predictions_*",
            "test_opening_*",
        )
        for path in artifact_root.glob(pattern)
    )
    if forbidden_artifacts:
        raise ValueError(f"sidecar found forbidden D2 test artifact: {forbidden_artifacts}")
    return {
        "sealed_test_l2_count": len(test_ids),
        "non_test_l2_count": len(non_test_ids),
        "forbidden_registry_rows": 0,
        "forbidden_test_artifacts": 0,
    }


def make_actor_pretrain_registry_rows(
    episodes: Iterable[Mapping[str, str]],
) -> list[dict[str, str]]:
    rows = []
    for episode in episodes:
        row = {
            "registry_schema": "bplus-opened-registry-1",
            "opened_at_utc": REGISTRY_OPENED_AT,
            "stage": REGISTRY_STAGE,
            "use_class": REGISTRY_USE_CLASS,
            "split_id": REGISTRY_SPLIT_ID,
            "l2_id": str(episode["l2_id"]),
            "l3_id": str(episode["l3_id"]),
            "l4_id": str(episode["l4_id"]),
            "map_name": str(episode["map_name"]),
            "source_manifest_sha256": D2_DATASET_MANIFEST_SHA256,
            "source_run_id": REGISTRY_SOURCE_RUN_ID,
            "decision_effect": REGISTRY_DECISION_EFFECT,
            "final_pool": "false",
            "evidence_relpath": EVIDENCE_RELPATH,
        }
        row["row_id"] = registry_row_id(row)
        rows.append(validate_registry_row(row))
    rows.sort(key=lambda row: row["row_id"])
    if len(rows) != 1928 or len({row["row_id"] for row in rows}) != 1928:
        raise ValueError("sidecar registry plan must contain 1,928 unique rows")
    return rows


def _read_registry(path: Path) -> list[dict[str, str]]:
    rows = _read_tsv(path)
    return [validate_registry_row(row) for row in rows]


def _registry_plan_live_state(
    registry: Path,
    planned_rows: list[dict[str, str]],
    before_sha256: str,
    after_sha256: str,
) -> str:
    actual_sha = file_sha256(registry)
    current = {row["row_id"]: row for row in _read_registry(registry)}
    present = [current.get(row["row_id"]) for row in planned_rows]
    if actual_sha == before_sha256 and all(value is None for value in present):
        return "ready"
    if actual_sha == after_sha256 and all(
        value == planned for value, planned in zip(present, planned_rows)
    ):
        return "already_appended"
    raise ValueError("sidecar registry is neither exact before nor exact planned-after state")


def create_registry_plan(
    repo_root: str | Path,
    source_preflight_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
) -> dict:
    root = Path(repo_root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("sidecar registry plan must run from repo root")
    source = validate_source_preflight(source_preflight_dir, root)
    if not source["passed"]:
        raise ValueError(f"sidecar source preflight failed: {source}")
    inputs = validate_pinned_inputs(root)
    if not inputs["passed"]:
        raise ValueError(f"sidecar pinned input failure: {inputs}")
    _validate_frozen_d2r_sources(root)
    episodes = _episode_rows(root)
    planned = make_actor_pretrain_registry_rows(episodes)
    registry = root / REGISTRY_RELPATH
    before_sha = file_sha256(registry)
    if before_sha != EXPECTED_REGISTRY_BEFORE_SHA256:
        raise ValueError("sidecar registry-plan input hash drift")
    existing_registry = _read_registry(registry)
    if any(row["stage"] == REGISTRY_STAGE for row in existing_registry):
        raise ValueError("sidecar registry-plan stage already exists")
    seal_audit = _validate_sealed_test_absence(root, episodes, existing_registry)

    output = Path(output_dir)
    partial = _prepare_output(output)
    try:
        shutil.copyfile(registry, partial / "registry_before.snapshot.tsv")
        shutil.copyfile(registry, partial / "registry_after.expected.tsv")
        append_result = append_opened_registry(
            partial / "registry_after.expected.tsv", planned
        )
        if append_result.appended != 1928 or append_result.skipped != 0:
            raise AssertionError("sidecar prospective registry append accounting failed")
        after_sha = file_sha256(partial / "registry_after.expected.tsv")
        _write_tsv(partial / "registry_rows.tsv", planned)
        config = {
            "schema": "bplus-v2.2-sidecar-registry-plan-1",
            "created_at": str(created_at),
            "owner_decision": OWNER_DECISION,
            "stage": REGISTRY_STAGE,
            "use_class": REGISTRY_USE_CLASS,
            "decision_effect": REGISTRY_DECISION_EFFECT,
            "purpose": "actor_pretrain",
            "opened_at_utc": REGISTRY_OPENED_AT,
            "rows": len(planned),
            "registry_before_sha256": before_sha,
            "registry_after_expected_sha256": after_sha,
            "source_preflight_relpath": str(Path(source_preflight_dir)),
            "source_preflight_output_manifest_sha256": file_sha256(
                Path(source_preflight_dir) / "output_manifest.sha256"
            ),
            "dataset_manifest_sha256": D2_DATASET_MANIFEST_SHA256,
            "episode_metadata_sha256": EPISODE_METADATA_SHA256,
            "split_manifest_sha256": D2_SPLIT_MANIFEST_SHA256,
            "test_seal_sha256": D2_TEST_SEAL_SHA256,
            "test_opened": False,
            "test_source_locators": 0,
            "final_pool": False,
            "sealed_test_audit": seal_audit,
        }
        _write_json(partial / "config.json", config)
        _write_json(
            partial / "validation.json",
            {
                "schema": "bplus-v2.2-sidecar-registry-plan-validation-1",
                "passed": True,
                "rows": len(planned),
                "live_state": "ready",
                "violations": [],
            },
        )
        _write_output_manifest(partial)
        _promote(partial, output)
    except BaseException as error:
        if partial.exists():
            _write_json(
                partial / "FAILED.json",
                {"type": type(error).__name__, "message": str(error)},
            )
        raise
    validation = validate_registry_plan(output, root, check_live=True)
    if not validation["passed"]:
        raise AssertionError(f"created invalid sidecar registry plan: {validation}")
    return {
        "passed": True,
        "rows": validation["rows"],
        "live_state": validation["live_state"],
        "registry_before_sha256": before_sha,
        "registry_after_expected_sha256": after_sha,
        "output_manifest_sha256": file_sha256(output / "output_manifest.sha256"),
    }


def validate_registry_plan(
    release_dir: str | Path,
    repo_root: str | Path = ".",
    *,
    check_live: bool = False,
) -> dict:
    release = Path(release_dir)
    root = Path(repo_root).resolve()
    violations = []
    rows: list[dict[str, str]] = []
    live_state = "not_checked"
    try:
        if not (release / "COMPLETE").is_file():
            raise ValueError("sidecar registry plan lacks COMPLETE")
        _validate_output_manifest(release)
        config = json.loads((release / "config.json").read_text(encoding="utf-8"))
        if (
            config["owner_decision"] != OWNER_DECISION
            or config["stage"] != REGISTRY_STAGE
            or config["purpose"] != "actor_pretrain"
            or config["rows"] != 1928
            or config["test_opened"] is not False
            or config["final_pool"] is not False
            or config["sealed_test_audit"]["sealed_test_l2_count"] != 1108
            or config["sealed_test_audit"]["forbidden_registry_rows"] != 0
            or config["sealed_test_audit"]["forbidden_test_artifacts"] != 0
        ):
            raise ValueError("sidecar registry-plan authority/scope mismatch")
        if config["registry_before_sha256"] != EXPECTED_REGISTRY_BEFORE_SHA256:
            raise ValueError("sidecar registry-plan before hash mismatch")
        if file_sha256(release / "registry_before.snapshot.tsv") != config[
            "registry_before_sha256"
        ]:
            raise ValueError("sidecar registry-plan before snapshot mismatch")
        if file_sha256(release / "registry_after.expected.tsv") != config[
            "registry_after_expected_sha256"
        ]:
            raise ValueError("sidecar registry-plan after snapshot mismatch")
        rows = _read_registry(release / "registry_rows.tsv")
        expected = make_actor_pretrain_registry_rows(_episode_rows(root))
        if rows != expected:
            raise ValueError("sidecar registry-plan row content mismatch")
        after_rows = _read_registry(release / "registry_after.expected.tsv")
        before_rows = _read_registry(release / "registry_before.snapshot.tsv")
        if len(after_rows) - len(before_rows) != 1928:
            raise ValueError("sidecar registry-plan append size mismatch")
        if after_rows[-1928:] != rows:
            raise ValueError("sidecar registry-plan append ordering mismatch")
        if check_live:
            live_state = _registry_plan_live_state(
                root / REGISTRY_RELPATH,
                rows,
                config["registry_before_sha256"],
                config["registry_after_expected_sha256"],
            )
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": "bplus-v2.2-sidecar-registry-plan-validation-1",
        "passed": not violations,
        "rows": len(rows),
        "live_state": live_state,
        "violations": violations,
    }


def _bundle_payload(
    model: D2RGeometryNet,
    mean: np.ndarray,
    std: np.ndarray,
    train_report: Mapping,
) -> dict:
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return {
        "schema": "bplus-v2.2-sidecar-bundle-1",
        "release_label": "SIDECAR_INITIALIZATION_ONLY",
        "family": D2R_LOCKED_CONFIG.family,
        "seed": D2R_SEED,
        "config": asdict(D2R_LOCKED_CONFIG),
        "state_dict": state,
        "state_dict_sha256": _tensor_digest(state.items()),
        "normalization_mean": mean,
        "normalization_std": std,
        "normalization_sha256": _array_digest(
            [("mean", mean), ("std", std)]
        ),
        "train_report": dict(train_report),
    }


def load_sidecar_bundle(
    release_dir: str | Path,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
    release = Path(release_dir)
    bundle = torch.load(
        release / "sidecar_bundle.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(bundle, dict):
        raise ValueError("sidecar bundle is not a mapping")
    state = bundle.get("state_dict", {})
    mean = torch.as_tensor(bundle.get("normalization_mean"), dtype=torch.float32)
    std = torch.as_tensor(bundle.get("normalization_std"), dtype=torch.float32)
    return state, mean, std, bundle


def _validate_bundle(release: Path) -> dict:
    state, mean, std, bundle = load_sidecar_bundle(release)
    if (
        bundle.get("schema") != "bplus-v2.2-sidecar-bundle-1"
        or bundle.get("release_label") != "SIDECAR_INITIALIZATION_ONLY"
        or bundle.get("family") != D2R_LOCKED_CONFIG.family
        or bundle.get("seed") != D2R_SEED
        or bundle.get("config") != asdict(D2R_LOCKED_CONFIG)
    ):
        raise ValueError("sidecar bundle authority/config mismatch")
    reference = D2RGeometryNet().state_dict()
    if set(state) != set(reference):
        raise ValueError("sidecar state-dict inventory mismatch")
    for name, expected in reference.items():
        value = torch.as_tensor(state[name])
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"sidecar tensor shape/dtype mismatch: {name}")
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"sidecar tensor is nonfinite: {name}")
    state_sha = _tensor_digest(state.items())
    if state_sha != bundle["state_dict_sha256"]:
        raise ValueError("sidecar state-dict digest mismatch")
    if mean.shape != (1680,) or std.shape != (1680,):
        raise ValueError("sidecar normalization shape mismatch")
    if (
        not torch.all(torch.isfinite(mean))
        or not torch.all(torch.isfinite(std))
        or torch.any(std <= 0.0)
    ):
        raise ValueError("sidecar normalization invalid")
    normalization_sha = _array_digest(
        [("mean", mean.numpy()), ("std", std.numpy())]
    )
    if normalization_sha != bundle["normalization_sha256"]:
        raise ValueError("sidecar normalization digest mismatch")
    report = bundle.get("train_report", {})
    history = report.get("history", [])
    if report.get("config") != asdict(D2R_LOCKED_CONFIG) or len(history) != 6:
        raise ValueError("sidecar train-report config/epoch mismatch")
    loss_names = (
        "loss",
        "classification_loss",
        "ttc_loss",
        "rel_loss",
        "lateral_loss",
        "closing_loss",
    )
    for epoch, row in enumerate(history):
        if row.get("epoch") != epoch or int(row.get("batches", 0)) <= 0:
            raise ValueError("sidecar train-report epoch accounting mismatch")
        if any(not math.isfinite(float(row[name])) for name in loss_names):
            raise ValueError("sidecar train-report contains nonfinite loss")
    return {
        "state_dict_sha256": state_sha,
        "normalization_sha256": normalization_sha,
        "parameters": sum(value.numel() for value in state.values()),
        "sampled_frames": int(report["sampled_frame_count"]),
        "epochs": len(history),
    }


def run_sidecar_initialization(
    repo_root: str | Path,
    source_preflight_dir: str | Path,
    registry_plan_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
    device_name: str = "cuda:0",
) -> dict:
    root = Path(repo_root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("sidecar initialization must run from repo root")
    source = validate_source_preflight(source_preflight_dir, root)
    if not source["passed"]:
        raise ValueError(f"sidecar source preflight failed: {source}")
    plan_validation = validate_registry_plan(
        registry_plan_dir, root, check_live=True
    )
    if not plan_validation["passed"]:
        raise ValueError(f"sidecar registry plan failed: {plan_validation}")
    inputs = validate_pinned_inputs(root)
    if not inputs["passed"]:
        raise ValueError(f"sidecar pinned inputs failed: {inputs}")
    core_sources = _validate_frozen_d2r_sources(root)
    if not os.environ.get("NUMBA_CACHE_DIR") or not Path(
        os.environ["NUMBA_CACHE_DIR"]
    ).is_absolute():
        raise ValueError("sidecar initialization requires isolated absolute NUMBA_CACHE_DIR")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("sidecar full fit requires available CUDA")

    output = Path(output_dir)
    partial = _prepare_output(output)
    try:
        plan = Path(registry_plan_dir)
        plan_config = json.loads((plan / "config.json").read_text(encoding="utf-8"))
        planned_rows = _read_registry(plan / "registry_rows.tsv")
        registry = root / REGISTRY_RELPATH
        before_sha = file_sha256(registry)
        seal_audit = _validate_sealed_test_absence(
            root, _episode_rows(root), _read_registry(registry)
        )
        append_result = append_opened_registry(registry, planned_rows)
        after_sha = file_sha256(registry)
        if after_sha != plan_config["registry_after_expected_sha256"]:
            raise AssertionError("sidecar live registry did not reach exact planned state")
        if (append_result.appended, append_result.skipped) not in {
            (1928, 0),
            (0, 1928),
        }:
            raise AssertionError("sidecar registry append/skip accounting mismatch")
        shutil.copyfile(registry, partial / "opened_registry.snapshot.tsv")

        dataset = D2RDataset(
            root / DATASET_RELPATH,
            root / SPLIT_RELPATH,
            root / SIGNALS_RELPATH,
        )
        if dataset.episode_count != 1928 or dataset.frame_count != 1505848:
            raise ValueError("sidecar full-non-test population shape drift")
        train_mask = np.ones(dataset.episode_count, dtype=bool)
        model, mean, std, train_report = train_model(
            dataset,
            train_mask,
            device,
            D2R_SEED,
            D2R_LOCKED_CONFIG,
        )
        bundle = _bundle_payload(model, mean, std, train_report)
        torch.save(bundle, partial / "sidecar_bundle.pt")
        _write_json(partial / "train_report.json", train_report)

        indices = np.unique(
            np.linspace(
                0,
                dataset.frame_count - 1,
                VALIDATION_FRAME_COUNT,
                dtype=np.int64,
            )
        )
        if len(indices) != VALIDATION_FRAME_COUNT:
            raise AssertionError("sidecar validation frame selection drift")
        np.save(partial / "validation_frame_indices.npy", indices)
        lidar, bc, scalar = dataset.input_batch(indices)
        input_sha = _array_digest(
            [("lidar", lidar), ("bc", bc), ("scalar", scalar)]
        )
        predictions = predict_model(model, dataset, indices, mean, std, device)
        rerun = predict_model(model, dataset, indices, mean, std, device)
        if not np.array_equal(predictions, rerun):
            raise AssertionError("sidecar same-device inference rerun mismatch")
        np.save(partial / "validation_predictions.npy", predictions)
        prediction_sha = _array_digest([("predictions", predictions)])
        history = dataset.history(indices)
        if np.any(history > indices[:, None]):
            raise AssertionError("sidecar validation history contains future frame")
        if np.any(dataset.base.episode_index[history] != dataset.base.episode_index[indices, None]):
            raise AssertionError("sidecar validation history crosses episode")

        gpu_name = torch.cuda.get_device_name(device)
        config = {
            "schema": "bplus-v2.2-sidecar-init-config-1",
            "release_label": "SIDECAR_INITIALIZATION_ONLY",
            "created_at": str(created_at),
            "owner_decision": OWNER_DECISION,
            "old_d2r_gate_passed": False,
            "ttc_role": "diagnostic_only",
            "d2_test_opened": False,
            "policy_training_started": False,
            "arm_selection_performed": False,
            "sidecar_fit_started": True,
            "sidecar_fit_completed": True,
            "sidecar_fit_count": 1,
            "initializes_arms": list(ARMS[1:]),
            "seed": D2R_SEED,
            "train_config": asdict(D2R_LOCKED_CONFIG),
            "dataset_manifest_sha256": D2_DATASET_MANIFEST_SHA256,
            "episode_metadata_sha256": EPISODE_METADATA_SHA256,
            "split_manifest_sha256": D2_SPLIT_MANIFEST_SHA256,
            "signals_manifest_sha256": D2R_SIGNALS_MANIFEST_SHA256,
            "test_seal_sha256": D2_TEST_SEAL_SHA256,
            "sealed_test_audit": seal_audit,
            "d2r_full_config_sha256": D2R_FULL_CONFIG_SHA256,
            "d2r_full_output_manifest_sha256": D2R_FULL_OUTPUT_MANIFEST_SHA256,
            "d2r_core_source_sha256": core_sources,
            "source_preflight_relpath": str(Path(source_preflight_dir)),
            "source_preflight_output_manifest_sha256": file_sha256(
                Path(source_preflight_dir) / "output_manifest.sha256"
            ),
            "registry_plan_relpath": str(Path(registry_plan_dir)),
            "registry_plan_output_manifest_sha256": file_sha256(
                Path(registry_plan_dir) / "output_manifest.sha256"
            ),
            "registry_before_observed_sha256": before_sha,
            "registry_after_sha256": after_sha,
            "registry_rows_appended": append_result.appended,
            "registry_rows_already_present": append_result.skipped,
            "registry_total_rows": append_result.total,
            "device": str(device),
            "gpu_name": gpu_name,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "numba_cache_dir": os.environ["NUMBA_CACHE_DIR"],
            "validation_frame_count": len(indices),
            "validation_input_sha256": input_sha,
            "validation_prediction_sha256": prediction_sha,
            "same_device_prediction_rerun_equal": True,
            "bundle_sha256": file_sha256(partial / "sidecar_bundle.pt"),
            "state_dict_sha256": bundle["state_dict_sha256"],
            "normalization_sha256": bundle["normalization_sha256"],
        }
        _write_json(partial / "config.json", config)
        _write_json(
            partial / "validation.json",
            {
                "schema": "bplus-v2.2-sidecar-init-validation-1",
                "passed": True,
                "mode": "same_device_full",
                "violations": [],
            },
        )
        _write_output_manifest(partial)
        preliminary = validate_sidecar_release(
            partial,
            root,
            dataset_dir=root / DATASET_RELPATH,
            split_dir=root / SPLIT_RELPATH,
            signals_dir=root / SIGNALS_RELPATH,
            device_name=device_name,
            require_live_registry=True,
            allow_partial=True,
        )
        if not preliminary["passed"]:
            raise AssertionError(f"sidecar preliminary validation failed: {preliminary}")
        _write_json(partial / "validation.json", preliminary)
        _write_output_manifest(partial)
        final = validate_sidecar_release(
            partial,
            root,
            dataset_dir=root / DATASET_RELPATH,
            split_dir=root / SPLIT_RELPATH,
            signals_dir=root / SIGNALS_RELPATH,
            device_name=device_name,
            require_live_registry=True,
            allow_partial=True,
        )
        if not final["passed"]:
            raise AssertionError(f"sidecar final validation failed: {final}")
        _promote(partial, output)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as error:
        if partial.exists():
            _write_json(
                partial / "FAILED.json",
                {"type": type(error).__name__, "message": str(error)},
            )
        raise
    validation = validate_sidecar_release(output, root)
    if not validation["passed"]:
        raise AssertionError(f"created invalid sidecar artifact: {validation}")
    return {
        "passed": True,
        "release_label": "SIDECAR_INITIALIZATION_ONLY",
        "epochs": validation["epochs"],
        "sampled_frames": validation["sampled_frames"],
        "state_dict_sha256": validation["state_dict_sha256"],
        "registry_after_sha256": after_sha,
        "output_manifest_sha256": file_sha256(output / "output_manifest.sha256"),
    }


def validate_sidecar_release(
    release_dir: str | Path,
    repo_root: str | Path = ".",
    *,
    dataset_dir: str | Path | None = None,
    split_dir: str | Path | None = None,
    signals_dir: str | Path | None = None,
    device_name: str | None = None,
    require_live_registry: bool = False,
    allow_partial: bool = False,
) -> dict:
    release = Path(release_dir)
    root = Path(repo_root).resolve()
    violations = []
    details = {
        "mode": "artifact_only",
        "epochs": 0,
        "sampled_frames": 0,
        "state_dict_sha256": "",
    }
    try:
        if not allow_partial and not (release / "COMPLETE").is_file():
            raise ValueError("sidecar initialization release lacks COMPLETE")
        _validate_output_manifest(release)
        config = json.loads((release / "config.json").read_text(encoding="utf-8"))
        if (
            config["release_label"] != "SIDECAR_INITIALIZATION_ONLY"
            or config["owner_decision"] != OWNER_DECISION
            or config["old_d2r_gate_passed"] is not False
            or config["d2_test_opened"] is not False
            or config["policy_training_started"] is not False
            or config["arm_selection_performed"] is not False
            or config["sidecar_fit_count"] != 1
            or config["initializes_arms"] != list(ARMS[1:])
            or config["sealed_test_audit"]["sealed_test_l2_count"] != 1108
            or config["sealed_test_audit"]["forbidden_registry_rows"] != 0
            or config["sealed_test_audit"]["forbidden_test_artifacts"] != 0
        ):
            raise ValueError("sidecar initialization authority/scope mismatch")
        if config["train_config"] != asdict(D2R_LOCKED_CONFIG):
            raise ValueError("sidecar initialization train-config drift")
        if config["d2r_core_source_sha256"] != D2R_CORE_SOURCE_SHA256:
            raise ValueError("sidecar initialization frozen-source record drift")
        fixed = {
            "dataset_manifest_sha256": D2_DATASET_MANIFEST_SHA256,
            "episode_metadata_sha256": EPISODE_METADATA_SHA256,
            "split_manifest_sha256": D2_SPLIT_MANIFEST_SHA256,
            "signals_manifest_sha256": D2R_SIGNALS_MANIFEST_SHA256,
            "test_seal_sha256": D2_TEST_SEAL_SHA256,
            "d2r_full_config_sha256": D2R_FULL_CONFIG_SHA256,
            "d2r_full_output_manifest_sha256": D2R_FULL_OUTPUT_MANIFEST_SHA256,
        }
        if any(config[name] != value for name, value in fixed.items()):
            raise ValueError("sidecar initialization pinned hash drift")
        source = root / config["source_preflight_relpath"]
        if file_sha256(source / "output_manifest.sha256") != config[
            "source_preflight_output_manifest_sha256"
        ]:
            raise ValueError("sidecar source-preflight manifest hash mismatch")
        bundle_details = _validate_bundle(release)
        details.update(bundle_details)
        if file_sha256(release / "sidecar_bundle.pt") != config["bundle_sha256"]:
            raise ValueError("sidecar bundle file hash mismatch")
        if details["state_dict_sha256"] != config["state_dict_sha256"]:
            raise ValueError("sidecar config/state digest mismatch")
        if details["normalization_sha256"] != config["normalization_sha256"]:
            raise ValueError("sidecar config/normalization digest mismatch")
        plan = root / config["registry_plan_relpath"]
        plan_validation = validate_registry_plan(plan, root, check_live=False)
        if not plan_validation["passed"]:
            raise ValueError(f"sidecar referenced registry plan invalid: {plan_validation}")
        plan_config = json.loads((plan / "config.json").read_text(encoding="utf-8"))
        if file_sha256(plan / "output_manifest.sha256") != config[
            "registry_plan_output_manifest_sha256"
        ]:
            raise ValueError("sidecar registry-plan manifest hash mismatch")
        if file_sha256(release / "opened_registry.snapshot.tsv") != plan_config[
            "registry_after_expected_sha256"
        ]:
            raise ValueError("sidecar registry snapshot is not exact planned-after state")
        if config["registry_after_sha256"] != plan_config[
            "registry_after_expected_sha256"
        ]:
            raise ValueError("sidecar config registry-after hash mismatch")
        snapshot_rows = _read_registry(release / "opened_registry.snapshot.tsv")
        if len(snapshot_rows) != 12019 or config["registry_total_rows"] != 12019:
            raise ValueError("sidecar registry total-row accounting mismatch")
        stage_rows = [row for row in snapshot_rows if row["stage"] == REGISTRY_STAGE]
        if len(stage_rows) != 1928 or any(
            row["final_pool"] != "false"
            or row["use_class"] != REGISTRY_USE_CLASS
            or row["decision_effect"] != REGISTRY_DECISION_EFFECT
            for row in stage_rows
        ):
            raise ValueError("sidecar registry snapshot stage semantics mismatch")
        observed_seal_audit = _validate_sealed_test_absence(
            root, _episode_rows(root), snapshot_rows
        )
        if observed_seal_audit != config["sealed_test_audit"]:
            raise ValueError("sidecar sealed-test audit recomputation mismatch")
        if require_live_registry and file_sha256(root / REGISTRY_RELPATH) != config[
            "registry_after_sha256"
        ]:
            raise ValueError("sidecar live registry hash mismatch")
        indices = np.load(
            release / "validation_frame_indices.npy", allow_pickle=False
        )
        predictions = np.load(
            release / "validation_predictions.npy", allow_pickle=False
        )
        if (
            indices.shape != (VALIDATION_FRAME_COUNT,)
            or indices.dtype != np.int64
            or predictions.shape != (VALIDATION_FRAME_COUNT, len(PREDICTION_NAMES))
            or predictions.dtype != np.float32
            or not np.all(np.isfinite(predictions))
        ):
            raise ValueError("sidecar validation array shape/dtype/finite mismatch")
        if _array_digest([("predictions", predictions)]) != config[
            "validation_prediction_sha256"
        ]:
            raise ValueError("sidecar validation prediction digest mismatch")

        full_args = (dataset_dir, split_dir, signals_dir, device_name)
        if any(value is not None for value in full_args):
            if any(value is None for value in full_args):
                raise ValueError("sidecar full validation requires all data/device arguments")
            device = torch.device(str(device_name))
            if device.type != "cuda" or not torch.cuda.is_available():
                raise ValueError("sidecar full numerical validation requires CUDA")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            dataset = D2RDataset(dataset_dir, split_dir, signals_dir)
            state, mean, std, _ = load_sidecar_bundle(release)
            model = D2RGeometryNet().to(device).eval()
            model.load_state_dict(state)
            lidar, bc, scalar = dataset.input_batch(indices)
            if _array_digest(
                [("lidar", lidar), ("bc", bc), ("scalar", scalar)]
            ) != config["validation_input_sha256"]:
                raise ValueError("sidecar full validation input digest mismatch")
            observed = predict_model(
                model,
                dataset,
                indices,
                mean.numpy(),
                std.numpy(),
                device,
            )
            rerun = predict_model(
                model,
                dataset,
                indices,
                mean.numpy(),
                std.numpy(),
                device,
            )
            if not np.array_equal(observed, predictions) or not np.array_equal(
                observed, rerun
            ):
                raise ValueError("sidecar same-device prediction recomputation mismatch")
            sampled = deterministic_fit_indices(
                dataset.base.arrays["episode_index"],
                np.ones(dataset.episode_count, dtype=bool),
                dataset.base.arrays["any_target_200"],
                dataset.base.arrays["corridor_ttc"],
            )
            if len(sampled) != details["sampled_frames"]:
                raise ValueError("sidecar sampled-frame accounting mismatch")
            history = dataset.history(indices)
            if np.any(history > indices[:, None]) or np.any(
                dataset.base.episode_index[history]
                != dataset.base.episode_index[indices, None]
            ):
                raise ValueError("sidecar causal-history validation failed")
            details["mode"] = "same_device_full"
            del model
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": "bplus-v2.2-sidecar-init-validation-1",
        "passed": not violations,
        **details,
        "violations": violations,
    }

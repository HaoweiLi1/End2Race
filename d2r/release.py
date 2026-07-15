"""Independent release validation for D2R-G OOF artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from d0.identity import REGISTRY_FIELDS, validate_registry_row
from d2.release import file_sha256
from d2r import EVIDENCE_RELPATH, FAMILY, REGISTRY_OPENED_AT
from d2r.data import D2RDataset
from d2r.train import PREDICTION_NAMES, _aggregate_oof


def _verify_manifest(directory: Path) -> None:
    manifest = directory / "output_manifest.sha256"
    if not manifest.is_file():
        raise ValueError("D2R output manifest missing")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relpath = line.split("  ", 1)
        if relpath in expected:
            raise ValueError("duplicate D2R output manifest path")
        expected[relpath] = digest
    observed = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    }
    if set(expected) != observed:
        raise ValueError("D2R output manifest file set mismatch")
    for relpath, digest in expected.items():
        if file_sha256(directory / relpath) != digest:
            raise ValueError(f"D2R output manifest hash mismatch: {relpath}")


def _validate_registry(path: Path, dataset: D2RDataset) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != REGISTRY_FIELDS:
            raise ValueError("D2R registry snapshot header mismatch")
        rows = [validate_registry_row(row) for row in reader]
    stage = [row for row in rows if row["stage"] == "D2R-G"]
    if len(stage) != 1928:
        raise ValueError("D2R registry snapshot does not have 1,928 stage rows")
    if {row["l2_id"] for row in stage} != {row["l2_id"] for row in dataset.base.episodes}:
        raise ValueError("D2R registry population mismatch")
    if any(
        row["opened_at_utc"] != REGISTRY_OPENED_AT
        or row["evidence_relpath"] != EVIDENCE_RELPATH
        or row["use_class"] != "probe_fit"
        or row["decision_effect"] != "representation_choice"
        or row["final_pool"] != "false"
        for row in stage
    ):
        raise ValueError("D2R registry semantics mismatch")
    return {"live_total": len(rows), "stage_rows": len(stage)}


def validate_release(
    release_dir: str | Path,
    dataset_dir: str | Path,
    split_dir: str | Path,
    signals_dir: str | Path,
    *,
    allow_partial: bool = False,
) -> dict:
    release_dir = Path(release_dir)
    violations = []
    details = {}
    try:
        if not allow_partial and not (release_dir / "COMPLETE").is_file():
            raise ValueError("D2R release lacks COMPLETE")
        _verify_manifest(release_dir)
        config = json.loads((release_dir / "config.json").read_text(encoding="utf-8"))
        if config["family"] != FAMILY or config["seed"] != 20260711:
            raise ValueError("D2R config family/seed drift")
        outer_folds = tuple(int(value) for value in config["outer_folds"])
        dataset = D2RDataset(dataset_dir, split_dir, signals_dir)
        if config["registry"] is not None:
            details["registry"] = _validate_registry(
                release_dir / "opened_registry.snapshot.tsv", dataset
            )
        expected_models = {
            f"outer{outer}_inner{inner}.pt" for outer in outer_folds for inner in range(3)
        } | {f"outer{outer}_refit.pt" for outer in outer_folds}
        observed_models = {path.name for path in (release_dir / "models").glob("*.pt")}
        if observed_models != expected_models:
            raise ValueError("D2R model bundle inventory mismatch")
        reports = []
        for outer in outer_folds:
            report = json.loads(
                (release_dir / f"outer{outer}_report.json").read_text(encoding="utf-8")
            )
            if int(report["outer_fold"]) != outer or len(report["inner_reports"]) != 3:
                raise ValueError("D2R outer report structure mismatch")
            if int(report["evaluation"]["frame_count"]) != len(
                dataset.base.frame_indices(dataset.base.outer_fold == outer)
            ):
                raise ValueError("D2R outer report frame count mismatch")
            reports.append(report)
        predictions = np.load(
            release_dir / "oof_predictions.npy", mmap_mode="r", allow_pickle=False
        )
        if predictions.shape != (dataset.frame_count, len(PREDICTION_NAMES)):
            raise ValueError("D2R OOF prediction shape mismatch")
        completed = np.isfinite(np.asarray(predictions[:, 0]))
        expected_completed = np.isin(
            dataset.base.outer_fold[dataset.base.episode_index], outer_folds
        )
        if not np.array_equal(completed, expected_completed):
            raise ValueError("D2R OOF prediction coverage mismatch")
        if np.any(~np.isfinite(np.asarray(predictions[completed]))):
            raise ValueError("D2R completed predictions contain nonfinite value")
        observed_report = json.loads(
            (release_dir / "oof_report.json").read_text(encoding="utf-8")
        )
        if outer_folds == (0, 1, 2, 3, 4):
            recomputed = _aggregate_oof(dataset, predictions, reports)
            if observed_report != recomputed:
                raise ValueError("D2R OOF report does not independently recompute")
        elif observed_report.get("complete_oof") is not False or observed_report.get("gate_passed") is not False:
            raise ValueError("D2R engineering smoke may not pass a gate")
        details.update(
            {
                "outer_folds": list(outer_folds),
                "completed_frames": int(np.count_nonzero(completed)),
                "model_bundles": len(observed_models),
                "complete_oof": outer_folds == (0, 1, 2, 3, 4),
                "gate_passed": bool(observed_report["gate_passed"]),
            }
        )
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": "d2r-validation-1",
        "passed": not violations,
        "violations": violations,
        **details,
    }

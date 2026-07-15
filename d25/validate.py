"""Independent structural validator for D2.5 oracle releases."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from d0.identity import validate_registry_row
from d2.release import file_sha256
from d25 import build_branch_specs
from d25.oracle import ARRAY_KEYS
from d25.search import (
    BASELINE_FIELDS,
    BRANCH_FIELDS,
    CASE_FIELDS,
    CASE_RESULT_FIELDS,
    EXPECTED_EGO_COLLISIONS,
    SMOKE_BRANCH_POSITIONS,
    route_summary,
    trajectory_digest,
)


def _read_tsv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing D2.5 table {path.name}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"D2.5 table header mismatch: {path.name}")
        return list(reader)


def _verify_output_manifest(directory: Path) -> None:
    path = directory / "output_manifest.sha256"
    if not path.is_file():
        raise ValueError("D2.5 output manifest missing")
    expected = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relpath = line.split("  ", 1)
        if relpath in expected:
            raise ValueError("duplicate D2.5 output manifest path")
        expected[relpath] = digest
    observed = {
        candidate.relative_to(directory).as_posix()
        for candidate in directory.rglob("*")
        if candidate.is_file() and candidate.name not in {"output_manifest.sha256", "COMPLETE"}
    }
    if set(expected) != observed:
        raise ValueError("D2.5 output manifest file set mismatch")
    for relpath, digest in expected.items():
        if file_sha256(directory / relpath) != digest:
            raise ValueError(f"D2.5 output manifest hash mismatch: {relpath}")


def validate_release(
    release_dir: str | Path,
    allow_partial: bool = False,
    verify_manifest: bool = True,
) -> dict:
    release_dir = Path(release_dir)
    violations = []
    try:
        if not allow_partial and not (release_dir / "COMPLETE").is_file():
            raise ValueError("D2.5 release lacks COMPLETE")
        config = json.loads((release_dir / "config.json").read_text(encoding="utf-8"))
        mode = config["mode"]
        cases = _read_tsv(release_dir / "case_manifest.tsv", CASE_FIELDS)
        if len(cases) != EXPECTED_EGO_COLLISIONS:
            raise ValueError("D2.5 case manifest is not 91 cases")
        if len({row["l2_id"] for row in cases}) != len(cases):
            raise ValueError("D2.5 case manifest has duplicate L2 IDs")
        smoke_ids = {row["l2_id"] for row in cases if row["smoke_selected"] == "true"}
        if len(smoke_ids) != 8 or smoke_ids != set(config["smoke_l2_ids"]):
            raise ValueError("D2.5 smoke IDs mismatch")

        baselines = _read_tsv(release_dir / "baseline_replay.tsv", BASELINE_FIELDS)
        expected_baselines = EXPECTED_EGO_COLLISIONS if mode == "full" else 8
        if len(baselines) != expected_baselines:
            raise ValueError("D2.5 baseline count mismatch")
        if any(row["passed"] != "true" for row in baselines):
            raise ValueError("D2.5 released baseline contains failure")
        expected_ids = {row["l2_id"] for row in cases} if mode == "full" else smoke_ids
        if {row["l2_id"] for row in baselines} != expected_ids:
            raise ValueError("D2.5 baseline population mismatch")

        with (release_dir / "opened_registry.snapshot.tsv").open(
            newline="", encoding="utf-8"
        ) as handle:
            registry = [validate_registry_row(row) for row in csv.DictReader(handle, delimiter="\t")]
        d25_rows = [row for row in registry if row["stage"] == "D2.5"]
        if len(d25_rows) != EXPECTED_EGO_COLLISIONS:
            raise ValueError("D2.5 registry snapshot does not contain exactly 91 stage rows")
        if {row["l2_id"] for row in d25_rows} != {row["l2_id"] for row in cases}:
            raise ValueError("D2.5 registry/case identity mismatch")
        if any(
            row["use_class"] != "oracle_search"
            or row["decision_effect"] != "action_choice"
            or row["final_pool"] != "false"
            for row in d25_rows
        ):
            raise ValueError("D2.5 registry semantics mismatch")

        branches = []
        results = []
        if mode in {"branch_smoke", "full"}:
            branches = _read_tsv(release_dir / "branch_results.tsv", BRANCH_FIELDS)
            results = _read_tsv(release_dir / "case_results.tsv", CASE_RESULT_FIELDS)
        if mode == "branch_smoke":
            if len(results) != 8 or {row["l2_id"] for row in results} != smoke_ids:
                raise ValueError("D2.5 branch-smoke case results mismatch")
            expected_count = len(SMOKE_BRANCH_POSITIONS) * 8
            if len(branches) != expected_count:
                raise ValueError("D2.5 branch-smoke branch count mismatch")
            if any(row["rerun_match"] != "true" for row in branches):
                raise ValueError("D2.5 branch smoke lacks deterministic rerun")
        elif mode == "full":
            if len(results) != EXPECTED_EGO_COLLISIONS:
                raise ValueError("D2.5 full case result count mismatch")
            by_case = {}
            for row in branches:
                by_case.setdefault(row["l2_id"], []).append(row)
            case_by_id = {row["l2_id"]: row for row in cases}
            for result in results:
                case = case_by_id[result["l2_id"]]
                rows = by_case.get(result["l2_id"], [])
                if [int(row["branch_sequence"]) for row in rows] != list(range(len(rows))):
                    raise ValueError("D2.5 branch sequence is noncontiguous")
                eligible = len(build_branch_specs(int(case["frame_count"])))
                if int(result["eligible_branch_count"]) != eligible:
                    raise ValueError("D2.5 eligible branch accounting mismatch")
                recovered = result["status"] == "recovered_confirmed_safe_pass"
                if recovered:
                    if not rows or rows[-1]["category"] != "collision_to_confirmed_safe_pass":
                        raise ValueError("D2.5 recovered case does not stop on witness")
                    if rows[-1]["action_clipped"] != "false" or rows[-1]["rerun_match"] != "true":
                        raise ValueError("D2.5 witness is clipped or not rerun-identical")
                elif len(rows) != eligible:
                    raise ValueError("D2.5 unrecovered case did not exhaust library")
            summary = json.loads((release_dir / "d25_summary.json").read_text(encoding="utf-8"))
            if summary["route_r2"] != route_summary(cases, results):
                raise ValueError("D2.5 Route-R2 summary mismatch")

            witness_rows = {
                row["trajectory_sha256"]: row
                for row in branches
                if row["category"] == "collision_to_confirmed_safe_pass"
            }
            witness_files = sorted((release_dir / "witnesses").glob("*.npz"))
            if len(witness_files) != len(witness_rows):
                raise ValueError("D2.5 witness file count mismatch")
            observed_digests = set()
            for path in witness_files:
                with np.load(path, allow_pickle=False) as arrays:
                    if set(arrays.files) != set(ARRAY_KEYS):
                        raise ValueError("D2.5 witness array key mismatch")
                    observed_digests.add(trajectory_digest(arrays))
            if observed_digests != set(witness_rows):
                raise ValueError("D2.5 witness trajectory digest mismatch")
        elif mode != "baseline_smoke":
            raise ValueError("unknown D2.5 release mode")

        if verify_manifest:
            _verify_output_manifest(release_dir)
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": "d25-validation-1",
        "passed": not violations,
        "violations": violations,
        "release_dir": str(release_dir),
    }

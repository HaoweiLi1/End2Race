"""Blocking and informational gates for the D0.1 canonical audit."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from d0.identity import SHA_RE, file_sha256
from d0.outcomes import EQUALITY_FIELDS


STATES = ("collision", "confirmed_pass", "terminal_overtake_only", "safe_follow")
BLOCKING_GATES = {"G1", "G2", "G3", "G5", "G6", "G7", "G8"}
ALL_GATES = BLOCKING_GATES | {"G4"}


@dataclass(frozen=True)
class GateResult:
    name: str
    blocking: bool
    passed: bool
    counts: dict
    violations: tuple[str, ...]


def _result(name: str, blocking: bool, counts: Mapping[str, Any], violations: Iterable[str]) -> GateResult:
    stable = tuple(sorted(set(str(item) for item in violations)))
    return GateResult(name=name, blocking=blocking, passed=not stable, counts=dict(counts), violations=stable)


def _value(record, key):
    if isinstance(record, Mapping):
        return record[key]
    return getattr(record, key)


def run_g1_duplicate_determinism(records: Iterable[Mapping]) -> GateResult:
    groups = defaultdict(list)
    violations = []
    total = 0
    for index, record in enumerate(records):
        total += 1
        try:
            key = (str(_value(record, "model_id")), str(_value(record, "l2_id")))
        except (KeyError, AttributeError) as exc:
            violations.append(f"record[{index}] missing grouping field: {exc}")
            continue
        groups[key].append(record)
    duplicate_groups = 0
    for key in sorted(groups):
        group = groups[key]
        if len(group) > 1:
            duplicate_groups += 1
        reference = group[0]
        for field in EQUALITY_FIELDS:
            try:
                expected = _value(reference, field)
            except (KeyError, AttributeError):
                violations.append(f"{key} missing field {field} in representative")
                continue
            for occurrence_index, candidate in enumerate(group[1:], start=1):
                try:
                    actual = _value(candidate, field)
                except (KeyError, AttributeError):
                    violations.append(f"{key} occurrence[{occurrence_index}] missing field {field}")
                    continue
                if actual != expected:
                    violations.append(
                        f"{key} field {field} disagreement: representative={expected!r} "
                        f"occurrence[{occurrence_index}]={actual!r}"
                    )
    return _result(
        "G1",
        True,
        {"records": total, "model_l2_groups": len(groups), "duplicate_groups": duplicate_groups},
        violations,
    )


def run_g2_inventory(expected_keys: set | frozenset, observed_records: Iterable[Mapping]) -> GateResult:
    expected = {str(key) for key in expected_keys}
    observed_records = list(observed_records)
    keys = [str(record.get("inventory_key", "")) for record in observed_records]
    counter = Counter(keys)
    observed = set(keys)
    violations = []
    for key in sorted(expected - observed):
        violations.append(f"missing inventory key {key}")
    for key in sorted(observed - expected):
        violations.append(f"extra inventory key {key}")
    for key, count in sorted(counter.items()):
        if count != 1:
            violations.append(f"duplicate inventory key {key}: count={count}")
    for index, record in enumerate(observed_records):
        key = keys[index]
        if not bool(record.get("validated", False)):
            violations.append(f"inventory key {key} unvalidated")
        if bool(record.get("stale", False)):
            violations.append(f"inventory key {key} stale")
    return _result(
        "G2",
        True,
        {
            "expected": len(expected),
            "observed_rows": len(observed_records),
            "observed_unique": len(observed),
            "missing": len(expected - observed),
            "extra": len(observed - expected),
        },
        violations,
    )


def run_g3_integrity(source_records: Iterable[Mapping]) -> GateResult:
    records = list(source_records)
    violations = []
    l1_counter = Counter(str(record.get("l1_id", "")) for record in records)
    for l1_id, count in sorted(l1_counter.items()):
        if not l1_id or count != 1:
            violations.append(f"duplicate or empty L1 ID {l1_id!r}: count={count}")
    checked = 0
    for index, record in enumerate(records):
        l1_id = str(record.get("l1_id", f"record[{index}]"))
        try:
            path = Path(record["path"]).resolve(strict=True)
            root = Path(record["root"]).resolve(strict=True)
        except (KeyError, FileNotFoundError, OSError) as exc:
            violations.append(f"{l1_id} missing/unreadable source: {exc}")
            continue
        try:
            path.relative_to(root)
        except ValueError:
            violations.append(f"{l1_id} source escapes root: {path} not under {root}")
            continue
        if not path.is_file():
            violations.append(f"{l1_id} source is not a file: {path}")
            continue
        if path.stat().st_size <= 0:
            violations.append(f"{l1_id} source is empty: {path}")
            continue
        expected = str(record.get("expected_sha256", ""))
        if not SHA_RE.fullmatch(expected):
            violations.append(f"{l1_id} expected SHA256 is invalid")
            continue
        actual = file_sha256(path)
        checked += 1
        if actual != expected:
            violations.append(f"{l1_id} SHA256 mismatch: expected={expected} actual={actual}")
    return _result(
        "G3",
        True,
        {"records": len(records), "hashes_checked": checked, "unique_l1": len(l1_counter)},
        violations,
    )


def run_g4_near_duplicate(pairs: Iterable[Mapping]) -> GateResult:
    compared = disagreements = ignored = 0
    violations = []
    condition_fields = (
        "map_name",
        "ego_raceline",
        "opponent_raceline",
        "speedscale",
        "interval_idx",
    )
    for index, pair in enumerate(pairs):
        try:
            left, right = pair["left"], pair["right"]
        except (KeyError, TypeError):
            violations.append(f"pair[{index}] malformed")
            continue
        matched = all(left.get(field) == right.get(field) for field in condition_fields)
        matched = matched and left.get("l4_id") == right.get("l4_id")
        matched = matched and left.get("l3_id") != right.get("l3_id")
        if not matched:
            ignored += 1
            continue
        compared += 1
        if left.get("outcome") != right.get("outcome"):
            disagreements += 1
    # Informational disagreements do not fail G4; malformed input does.
    return _result(
        "G4",
        False,
        {"compared": compared, "disagreements": disagreements, "ignored": ignored},
        violations,
    )


def run_g5_collision_floors(records: Iterable[Mapping], minimum: int = 30) -> GateResult:
    cases = {"skill_F": set(), "skill_S": set()}
    for record in records:
        skill = record.get("skill")
        if (
            record.get("model_id") == "bc"
            and skill in cases
            and bool(record.get("ego_collision"))
        ):
            cases[skill].add(str(record.get("l2_id")))
    violations = []
    for skill in ("skill_F", "skill_S"):
        if len(cases[skill]) < minimum:
            violations.append(f"{skill} ego-collision floor {len(cases[skill])} < {minimum}")
    return _result(
        "G5",
        True,
        {"minimum": minimum, "skill_F": len(cases["skill_F"]), "skill_S": len(cases["skill_S"])},
        violations,
    )


def run_g6_record_physics(records: Iterable[Mapping]) -> GateResult:
    records = list(records)
    checks = (
        "has_terminal_fields",
        "raw_label_ok",
        "json_npz_label_match",
        "collision_identity_ok",
        "equal_lengths",
        "finite_values",
        "exact_resolution",
        "terminal_gap_ok",
    )
    violations = []
    for index, record in enumerate(records):
        key = str(record.get("inventory_key", f"record[{index}]"))
        for field in checks:
            if not bool(record.get(field, False)):
                violations.append(f"{key} {field} failed")
        if record.get("frame_spacing_status") != "ok":
            violations.append(f"{key} frame_spacing_status failed: {record.get('frame_spacing_status')!r}")
    return _result("G6", True, {"records": len(records)}, violations)


def run_g7_unknown_censoring(records: Iterable[Mapping]) -> GateResult:
    records = list(records)
    alignment_failures = unknown_records = censored_records = 0
    violations = []
    for index, record in enumerate(records):
        key = f"{record.get('model_id', '?')}:{record.get('l2_id', index)}"
        if record.get("alignment_status") != "ok":
            alignment_failures += 1
            violations.append(f"{key} alignment failure")
        unknown_fields = [field for field in EQUALITY_FIELDS if record.get(field) == "unknown"]
        if unknown_fields:
            unknown_records += 1
            violations.append(f"{key} unknown fields {','.join(unknown_fields)}")
        if bool(record.get("censored")):
            censored_records += 1
            violations.append(f"{key} censored")
    return _result(
        "G7",
        True,
        {
            "records": len(records),
            "alignment_failures": alignment_failures,
            "unknown_records": unknown_records,
            "censored_records": censored_records,
        },
        violations,
    )


def _key_rows(rows: Iterable[Mapping], label: str, violations: list[str]) -> set[tuple[str, str]]:
    keys = []
    for index, row in enumerate(rows):
        try:
            keys.append((str(row["model_id"]), str(row["l2_id"])))
        except KeyError as exc:
            violations.append(f"{label} row[{index}] missing {exc}")
    counter = Counter(keys)
    for key, count in sorted(counter.items()):
        if count != 1:
            violations.append(f"duplicate {label} key {key}: count={count}")
    return set(keys)


def _compare_expected(actual: Any, expected: Any, path: str, violations: list[str]) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            violations.append(f"summary {path} expected mapping, got {type(actual).__name__}")
            return
        for key, value in expected.items():
            if key not in actual:
                violations.append(f"summary missing {path}.{key}")
            else:
                _compare_expected(actual[key], value, f"{path}.{key}", violations)
    elif actual != expected:
        violations.append(f"summary {path} mismatch: expected={expected!r} actual={actual!r}")


def run_g8_reconciliation(
    *,
    canonical_records: Iterable[Mapping],
    correction_rows: Iterable[Mapping],
    collision_rows: Iterable[Mapping],
    matrices: Mapping[str, Mapping],
    summary: Mapping,
    expected_summary: Mapping,
    expected_estimands: Mapping[str, set | frozenset],
    emitted_estimands: Mapping[str, Iterable[str]],
    required_manifest_files: set | frozenset,
    manifest_hashes: Mapping[str, str],
    registry_rows: Iterable[Mapping] = (),
    required_registry_rows: Iterable[Mapping] = (),
) -> GateResult:
    canonical_records = list(canonical_records)
    correction_rows = list(correction_rows)
    collision_rows = list(collision_rows)
    violations: list[str] = []

    canonical_keys = _key_rows(canonical_records, "canonical", violations)
    expected_corrections = {
        (str(row["model_id"]), str(row["l2_id"]))
        for row in canonical_records
        if row.get("archived_outcome3") != row.get("corrected_outcome3")
    }
    observed_corrections = _key_rows(correction_rows, "correction", violations)
    for key in sorted(expected_corrections - observed_corrections):
        violations.append(f"missing correction row {key}")
    for key in sorted(observed_corrections - expected_corrections):
        violations.append(f"extra correction row {key}")

    expected_collisions = {
        (str(row["model_id"]), str(row["l2_id"]))
        for row in canonical_records
        if bool(row.get("collision_any"))
    }
    observed_collisions = _key_rows(collision_rows, "collision", violations)
    for key in sorted(expected_collisions - observed_collisions):
        violations.append(f"missing collision row {key}")
    for key in sorted(observed_collisions - expected_collisions):
        violations.append(f"extra collision row {key}")

    expected_cells = {(left, right) for left in STATES for right in STATES}
    for matrix_id in sorted(matrices):
        matrix = matrices[matrix_id]
        rows = list(matrix.get("rows", []))
        cells = []
        total = 0
        for index, row in enumerate(rows):
            cell = (row.get("bc_state"), row.get("candidate_state"))
            cells.append(cell)
            count = row.get("count")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                violations.append(f"matrix {matrix_id} row[{index}] invalid count {count!r}")
            else:
                total += count
        counter = Counter(cells)
        if set(cells) != expected_cells or len(rows) != 16:
            violations.append(f"matrix {matrix_id} does not contain exactly 16 state cells")
        for cell, count in sorted(counter.items(), key=str):
            if count != 1:
                violations.append(f"matrix {matrix_id} duplicate cell {cell}: count={count}")
        expected_n = matrix.get("expected_n")
        if total != expected_n:
            violations.append(f"matrix {matrix_id} sum mismatch: expected={expected_n} actual={total}")

    _compare_expected(summary, expected_summary, "root", violations)

    if set(expected_estimands) != set(emitted_estimands):
        violations.append(
            f"estimand names mismatch: expected={sorted(expected_estimands)} "
            f"actual={sorted(emitted_estimands)}"
        )
    for name in sorted(set(expected_estimands) & set(emitted_estimands)):
        expected_ids = {str(x) for x in expected_estimands[name]}
        emitted_list = [str(x) for x in emitted_estimands[name]]
        emitted_ids = set(emitted_list)
        if len(emitted_list) != len(emitted_ids):
            violations.append(f"estimand {name} contains duplicate IDs")
        if emitted_ids != expected_ids:
            violations.append(
                f"estimand {name} ID drift: missing={sorted(expected_ids-emitted_ids)} "
                f"extra={sorted(emitted_ids-expected_ids)}"
            )

    required = {str(x) for x in required_manifest_files}
    covered = set(manifest_hashes)
    for name in sorted(required - covered):
        violations.append(f"output manifest missing {name}")
    for name, digest in sorted(manifest_hashes.items()):
        if not isinstance(digest, str) or not SHA_RE.fullmatch(digest):
            violations.append(f"output manifest invalid SHA256 for {name}")

    registry_rows = list(registry_rows)
    required_registry_rows = list(required_registry_rows)

    def registry_index(rows, label):
        indexed = {}
        for index, row in enumerate(rows):
            row_id = str(row.get("row_id", ""))
            if not row_id:
                violations.append(f"{label} registry row[{index}] missing row_id")
                continue
            normalized = {str(key): str(value) for key, value in row.items()}
            if row_id in indexed:
                violations.append(f"duplicate {label} registry row_id {row_id}")
            else:
                indexed[row_id] = normalized
        return indexed

    observed_registry = registry_index(registry_rows, "observed")
    required_registry = registry_index(required_registry_rows, "required")
    for row_id in sorted(set(required_registry) - set(observed_registry)):
        violations.append(f"opened registry missing required row {row_id}")
    for row_id in sorted(set(required_registry) & set(observed_registry)):
        if observed_registry[row_id] != required_registry[row_id]:
            violations.append(f"opened registry row mismatch {row_id}")

    return _result(
        "G8",
        True,
        {
            "canonical_rows": len(canonical_records),
            "canonical_keys": len(canonical_keys),
            "expected_corrections": len(expected_corrections),
            "observed_corrections": len(observed_corrections),
            "expected_collisions": len(expected_collisions),
            "observed_collisions": len(observed_collisions),
            "matrices": len(matrices),
            "manifest_required": len(required),
            "manifest_covered": len(required & covered),
            "registry_rows": len(observed_registry),
            "registry_required": len(required_registry),
        },
        violations,
    )


def release_verdict(results: Iterable[GateResult]) -> GateResult:
    results = list(results)
    by_name = defaultdict(list)
    for result in results:
        by_name[result.name].append(result)
    violations = []
    for name in sorted(ALL_GATES):
        if name not in by_name:
            violations.append(f"missing gate {name}")
        elif len(by_name[name]) != 1:
            violations.append(f"gate {name} appears {len(by_name[name])} times")
    for name in sorted(BLOCKING_GATES):
        for result in by_name.get(name, []):
            if not result.passed:
                violations.append(f"blocking gate {name} failed")
    return _result(
        "RELEASE",
        True,
        {"gate_results": len(results), "blocking_expected": len(BLOCKING_GATES)},
        violations,
    )

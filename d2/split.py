"""Outcome-blind D2 split construction and source redaction."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Iterable, Mapping

from d2 import INNER_FOLDS, OUTER_FOLDS, SPLIT_SCHEMA, SPLIT_SEED, TEST_FRACTION


PROJECTED_FIELDS = (
    "model_id",
    "l2_id",
    "l3_id",
    "l4_id",
    "map_name",
    "ego_raceline",
    "opponent_raceline",
    "speedscale_hex",
    "interval_idx",
    "skill",
    "representative_l1_id",
)
SPLIT_FIELDS = (
    "split_schema",
    "l2_id",
    "l3_id",
    "l4_id",
    "map_name",
    "ego_raceline",
    "opponent_raceline",
    "speedscale_hex",
    "interval_idx",
    "skill",
    "representative_l1_id",
    "split",
    "outer_fold",
    *(f"inner_fold_outer{outer}" for outer in range(OUTER_FOLDS)),
)
NON_TEST_SOURCE_FIELDS = (
    "source_schema",
    "l2_id",
    "representative_l1_id",
    "resolved_ego_idx",
    "npz_relpath",
    "npz_sha256",
)

_TEST_DOMAIN = b"end2race:d2:test-rank:v1\0"
_OUTER_DOMAIN = b"end2race:d2:outer-fold:v1\0"
_INNER_DOMAIN = b"end2race:d2:inner-fold:v1\0"
_MANIFEST_DOMAIN = b"end2race:d2:split-manifest:v1\0"
_TEST_IDS_DOMAIN = b"end2race:d2:test-ids:v1\0"


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _rank(domain: bytes, seed: int, *parts: str) -> str:
    payload = _canonical_json({"seed": int(seed), "parts": [str(part) for part in parts]})
    return hashlib.sha256(domain + payload).hexdigest()


def _project(rows: Iterable[Mapping]) -> list[dict]:
    projected = []
    for raw in rows:
        if str(raw.get("model_id", "")) != "bc":
            continue
        missing = [field for field in PROJECTED_FIELDS if field not in raw]
        if missing:
            raise ValueError(f"D2 split input missing fields: {missing}")
        row = {field: str(raw[field]) for field in PROJECTED_FIELDS}
        if not row["l2_id"] or not row["l3_id"] or not row["l4_id"]:
            raise ValueError("D2 identity fields must be nonempty")
        projected.append(row)
    projected.sort(key=lambda row: row["l2_id"])
    if len({row["l2_id"] for row in projected}) != len(projected):
        raise ValueError("duplicate BC L2 ID in D2 split input")
    return projected


def _block_table(rows: Iterable[Mapping]) -> dict[str, dict]:
    blocks: dict[str, dict] = {}
    for row in rows:
        block = blocks.setdefault(
            row["l4_id"],
            {"l4_id": row["l4_id"], "map_name": row["map_name"], "size": 0},
        )
        if block["map_name"] != row["map_name"]:
            raise ValueError("one L4 block spans multiple maps")
        block["size"] += 1
    return blocks


def _balanced_folds(blocks: Iterable[Mapping], n_folds: int, seed: int, domain: bytes, context: str) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("fold count must be at least two")
    by_map: dict[str, list[dict]] = defaultdict(list)
    for raw in blocks:
        block = dict(raw)
        by_map[block["map_name"]].append(block)
    assignment: dict[str, int] = {}
    for map_name, map_blocks in sorted(by_map.items()):
        if len(map_blocks) < n_folds:
            raise ValueError(f"map {map_name} has fewer blocks than folds")
        ordered = sorted(
            map_blocks,
            key=lambda block: (
                -int(block["size"]),
                _rank(domain, seed, context, map_name, block["l4_id"]),
                block["l4_id"],
            ),
        )
        scenario_load = [0] * n_folds
        block_load = [0] * n_folds
        for block in ordered:
            fold = min(range(n_folds), key=lambda value: (scenario_load[value], block_load[value], value))
            assignment[block["l4_id"]] = fold
            scenario_load[fold] += int(block["size"])
            block_load[fold] += 1
    return assignment


def build_split(
    rows: Iterable[Mapping],
    seed: int = SPLIT_SEED,
    test_fraction: float = TEST_FRACTION,
) -> list[dict]:
    """Build the byte-stable public split using projected, outcome-blind fields."""
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be inside (0, 1)")
    projected = _project(rows)
    if not projected:
        raise ValueError("no BC rows supplied for D2")
    blocks = _block_table(projected)
    by_map: dict[str, list[dict]] = defaultdict(list)
    for block in blocks.values():
        by_map[block["map_name"]].append(block)

    test_blocks: set[str] = set()
    for map_name, map_blocks in sorted(by_map.items()):
        ordered = sorted(
            map_blocks,
            key=lambda block: (
                _rank(_TEST_DOMAIN, seed, map_name, block["l4_id"]),
                block["l4_id"],
            ),
        )
        quota = int(math.ceil(float(test_fraction) * len(ordered)))
        if quota <= 0 or quota >= len(ordered):
            raise ValueError(f"invalid test quota for map {map_name}")
        test_blocks.update(block["l4_id"] for block in ordered[:quota])

    non_test_blocks = [block for block in blocks.values() if block["l4_id"] not in test_blocks]
    outer_assignment = _balanced_folds(
        non_test_blocks, OUTER_FOLDS, int(seed), _OUTER_DOMAIN, "outer"
    )
    inner_assignments: dict[int, dict[str, int]] = {}
    for outer in range(OUTER_FOLDS):
        inner_blocks = [
            block for block in non_test_blocks if outer_assignment[block["l4_id"]] != outer
        ]
        inner_assignments[outer] = _balanced_folds(
            inner_blocks,
            INNER_FOLDS,
            int(seed),
            _INNER_DOMAIN,
            f"outer={outer}",
        )

    manifest = []
    for source in projected:
        is_test = source["l4_id"] in test_blocks
        outer = "" if is_test else str(outer_assignment[source["l4_id"]])
        row = {
            "split_schema": SPLIT_SCHEMA,
            **{field: source[field] for field in PROJECTED_FIELDS if field != "model_id"},
            "split": "test" if is_test else "non_test",
            "outer_fold": outer,
        }
        for outer_index in range(OUTER_FOLDS):
            field = f"inner_fold_outer{outer_index}"
            if is_test or outer == str(outer_index):
                row[field] = ""
            else:
                row[field] = str(inner_assignments[outer_index][source["l4_id"]])
        row = {field: row[field] for field in SPLIT_FIELDS}
        manifest.append(row)
    manifest.sort(key=lambda row: row["l2_id"])
    validate_split(manifest, expected_l2=len(projected), expected_maps=set(by_map), test_fraction=test_fraction)
    return manifest


def validate_split(
    manifest: Iterable[Mapping],
    expected_l2: int | None = None,
    expected_maps: set[str] | None = None,
    test_fraction: float = TEST_FRACTION,
) -> dict:
    rows = [dict(row) for row in manifest]
    if not rows:
        raise ValueError("empty D2 split manifest")
    if expected_l2 is not None and len(rows) != int(expected_l2):
        raise ValueError("D2 L2 count mismatch")
    if any(tuple(row) != SPLIT_FIELDS for row in rows):
        raise ValueError("D2 split manifest field/order mismatch")
    if any(row["split_schema"] != SPLIT_SCHEMA for row in rows):
        raise ValueError("D2 split schema mismatch")
    if len({row["l2_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate L2 ID in D2 split")
    maps = {row["map_name"] for row in rows}
    if expected_maps is not None and maps != set(expected_maps):
        raise ValueError("D2 map set mismatch")

    by_l4: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["split"] not in {"non_test", "test"}:
            raise ValueError("invalid D2 split value")
        by_l4[row["l4_id"]].append(row)
    for l4_id, members in by_l4.items():
        if len({row["map_name"] for row in members}) != 1:
            raise ValueError(f"L4 {l4_id} spans maps")
        if len({row["split"] for row in members}) != 1:
            raise ValueError(f"L4 {l4_id} spans D2 splits")
        if len({row["outer_fold"] for row in members}) != 1:
            raise ValueError(f"L4 {l4_id} spans outer folds")
        for outer in range(OUTER_FOLDS):
            field = f"inner_fold_outer{outer}"
            if len({row[field] for row in members}) != 1:
                raise ValueError(f"L4 {l4_id} spans inner folds")

    for row in rows:
        if row["split"] == "test":
            if row["outer_fold"] or any(row[f"inner_fold_outer{i}"] for i in range(OUTER_FOLDS)):
                raise ValueError("test row carries development folds")
            continue
        if row["outer_fold"] not in {str(i) for i in range(OUTER_FOLDS)}:
            raise ValueError("invalid outer fold")
        for outer in range(OUTER_FOLDS):
            value = row[f"inner_fold_outer{outer}"]
            if row["outer_fold"] == str(outer):
                if value:
                    raise ValueError("outer-held row carries inner fold")
            elif value not in {str(i) for i in range(INNER_FOLDS)}:
                raise ValueError("invalid inner fold")

    for map_name in maps:
        map_blocks = {l4 for l4, members in by_l4.items() if members[0]["map_name"] == map_name}
        test_blocks = {l4 for l4 in map_blocks if by_l4[l4][0]["split"] == "test"}
        if len(test_blocks) != math.ceil(float(test_fraction) * len(map_blocks)):
            raise ValueError(f"test block quota mismatch for {map_name}")
        non_test_rows = [row for row in rows if row["map_name"] == map_name and row["split"] == "non_test"]
        if {row["outer_fold"] for row in non_test_rows} != {str(i) for i in range(OUTER_FOLDS)}:
            raise ValueError(f"outer folds incomplete for {map_name}")
        for outer in range(OUTER_FOLDS):
            values = {
                row[f"inner_fold_outer{outer}"]
                for row in non_test_rows
                if row["outer_fold"] != str(outer)
            }
            if values != {str(i) for i in range(INNER_FOLDS)}:
                raise ValueError(f"inner folds incomplete for {map_name}, outer {outer}")

    return {
        "l2_count": len(rows),
        "l4_count": len(by_l4),
        "test_l2_count": sum(row["split"] == "test" for row in rows),
        "test_l4_count": sum(members[0]["split"] == "test" for members in by_l4.values()),
        "maps": sorted(maps),
    }


def split_digest(manifest: Iterable[Mapping]) -> str:
    rows = [{field: str(row[field]) for field in SPLIT_FIELDS} for row in manifest]
    rows.sort(key=lambda row: row["l2_id"])
    return hashlib.sha256(_MANIFEST_DOMAIN + _canonical_json(rows)).hexdigest()


def test_seal(
    manifest: Iterable[Mapping],
    seed: int = SPLIT_SEED,
    test_fraction: float = TEST_FRACTION,
) -> dict:
    rows = [{field: str(row[field]) for field in SPLIT_FIELDS} for row in manifest]
    validate_split(rows, test_fraction=test_fraction)
    test_rows = [row for row in rows if row["split"] == "test"]
    test_ids = sorted(row["l2_id"] for row in test_rows)
    test_blocks = sorted({row["l4_id"] for row in test_rows})
    by_map_l2 = Counter(row["map_name"] for row in test_rows)
    by_map_l4 = Counter(
        next(row["map_name"] for row in test_rows if row["l4_id"] == block)
        for block in test_blocks
    )
    return {
        "schema": "d2-test-seal-1",
        "split_manifest_sha256": split_digest(rows),
        "split_seed": int(seed),
        "test_fraction_hex": float(test_fraction).hex(),
        "test_ids_sha256": hashlib.sha256(_TEST_IDS_DOMAIN + _canonical_json(test_ids)).hexdigest(),
        "test_blocks_sha256": hashlib.sha256(_TEST_IDS_DOMAIN + _canonical_json(test_blocks)).hexdigest(),
        "test_l2_count": len(test_ids),
        "test_l4_count": len(test_blocks),
        "test_l2_by_map": dict(sorted(by_map_l2.items())),
        "test_l4_by_map": dict(sorted(by_map_l4.items())),
    }


def build_non_test_sources(manifest: Iterable[Mapping], occurrences: Iterable[Mapping]) -> list[dict]:
    """Resolve source locators only for non-test representative L1 rows."""
    rows = [{field: str(row[field]) for field in SPLIT_FIELDS} for row in manifest]
    validate_split(rows)
    required = {row["representative_l1_id"]: row for row in rows if row["split"] == "non_test"}
    found: dict[str, dict] = {}
    for raw in occurrences:
        if str(raw.get("model_id", "")) != "bc":
            continue
        l1_id = str(raw.get("l1_id", ""))
        if l1_id not in required:
            continue
        missing = [field for field in ("l2_id", "resolved_ego_idx", "npz_relpath", "npz_sha256") if field not in raw]
        if missing:
            raise ValueError(f"non-test source missing fields: {missing}")
        expected = required[l1_id]
        if str(raw["l2_id"]) != expected["l2_id"]:
            raise ValueError("representative L1 resolves to wrong L2")
        row = {
            "source_schema": "d2-non-test-source-1",
            "l2_id": expected["l2_id"],
            "representative_l1_id": l1_id,
            "resolved_ego_idx": str(raw["resolved_ego_idx"]),
            "npz_relpath": str(raw["npz_relpath"]),
            "npz_sha256": str(raw["npz_sha256"]),
        }
        prior = found.get(l1_id)
        if prior is not None and prior != row:
            raise ValueError("representative L1 has conflicting source rows")
        found[l1_id] = row
    missing_ids = sorted(set(required) - set(found))
    if missing_ids:
        raise ValueError(f"missing {len(missing_ids)} non-test representative sources")
    output = [found[l1] for l1 in sorted(found, key=lambda value: found[value]["l2_id"])]
    if any(tuple(row) != NON_TEST_SOURCE_FIELDS for row in output):
        raise AssertionError("non-test source field order drift")
    return output


"""Atomic split-lock release and independent validation for D2."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from d2 import SPLIT_SEED, TEST_FRACTION
from d2.split import (
    NON_TEST_SOURCE_FIELDS,
    SPLIT_FIELDS,
    build_non_test_sources,
    build_split,
    split_digest,
    test_seal,
    validate_split,
)


D0_EXPECTED_HASHES = {
    "output_manifest.sha256": "425d62097b1463e72fca33f4e08690385bfbd21e6be3a91db900b92e4664bd89",
    "d0_summary.json": "56c9dcdc4af24afdd8b0f69a10e9b71487c75d23466bdde28a5090a214f92505",
    "d0_validation.json": "cf2a8165419bf49a1c3507eab2ca9cf9f0476aa125854374de62db999f5e4613",
}
RELEASE_FILES = (
    "config.json",
    "scenario_split.tsv",
    "non_test_sources.tsv",
    "test_seal.json",
    "fold_accounting.json",
    "validation.json",
)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: Iterable[Mapping], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if tuple(row) != fields:
                raise ValueError(f"field order mismatch while writing {path.name}")
            writer.writerow(row)


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_output(output_dir: Path) -> Path:
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"D2 output exists or is nonempty: {output_dir}")
        output_dir.rmdir()
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"D2 partial output exists: {partial}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    return partial


def _promote(partial: Path, output_dir: Path) -> None:
    for path in partial.iterdir():
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(partial)
    os.replace(partial, output_dir)
    _fsync_directory(output_dir.parent)
    complete = output_dir / "COMPLETE"
    with complete.open("w", encoding="utf-8") as handle:
        handle.write("COMPLETE\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(output_dir)


def _verify_d0(d0_dir: Path, expected_hashes: Mapping[str, str]) -> None:
    if not (d0_dir / "COMPLETE").is_file():
        raise ValueError("D0.1 release lacks COMPLETE")
    for relpath, expected in expected_hashes.items():
        path = d0_dir / relpath
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"D0.1 frozen hash mismatch: {relpath}")


def _load_d0_projection(d0_dir: Path) -> tuple[list[dict], list[dict]]:
    s0 = json.loads((d0_dir / "s0_manifest.json").read_text(encoding="utf-8"))
    primary_ids = set(s0["estimand_ids"]["primary"])
    if len(primary_ids) != 3036:
        raise ValueError("D0.1 Primary ID count is not 3036")
    canonical = _read_tsv(d0_dir / "canonical_episodes.tsv")
    rows = [
        row for row in canonical
        if row.get("model_id") == "bc" and row.get("l2_id") in primary_ids
    ]
    if len(rows) != 3036 or {row["l2_id"] for row in rows} != primary_ids:
        raise ValueError("D0.1 BC Primary projection mismatch")
    occurrences = _read_tsv(d0_dir / "episode_occurrences.tsv")
    return rows, occurrences


def _fold_accounting(manifest: list[dict]) -> dict:
    blocks: dict[str, dict] = {}
    for row in manifest:
        block = blocks.setdefault(
            row["l4_id"],
            {
                "map_name": row["map_name"],
                "split": row["split"],
                "outer_fold": row["outer_fold"],
                "size": 0,
            },
        )
        block["size"] += 1
    by_map = {}
    for map_name in sorted({row["map_name"] for row in manifest}):
        map_rows = [row for row in manifest if row["map_name"] == map_name]
        map_blocks = {key: value for key, value in blocks.items() if value["map_name"] == map_name}
        by_map[map_name] = {
            "l2_total": len(map_rows),
            "l4_total": len(map_blocks),
            "test_l2": sum(row["split"] == "test" for row in map_rows),
            "test_l4": sum(value["split"] == "test" for value in map_blocks.values()),
            "outer_l2": dict(sorted(Counter(row["outer_fold"] for row in map_rows if row["split"] == "non_test").items())),
            "outer_l4": dict(sorted(Counter(value["outer_fold"] for value in map_blocks.values() if value["split"] == "non_test").items())),
        }
    return {
        "schema": "d2-fold-accounting-1",
        "split_manifest_sha256": split_digest(manifest),
        "l2_total": len(manifest),
        "l4_total": len(blocks),
        "test_l2": sum(row["split"] == "test" for row in manifest),
        "test_l4": sum(value["split"] == "test" for value in blocks.values()),
        "by_map": by_map,
    }


def _write_output_manifest(directory: Path) -> None:
    lines = [f"{file_sha256(directory / name)}  {name}" for name in RELEASE_FILES]
    (directory / "output_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_split_release(
    d0_dir: str | Path,
    output_dir: str | Path,
    source_relpath: str,
    created_at: str,
    seed: int = SPLIT_SEED,
    test_fraction: float = TEST_FRACTION,
    expected_hashes: Mapping[str, str] = D0_EXPECTED_HASHES,
) -> dict:
    d0_dir = Path(d0_dir)
    output_dir = Path(output_dir)
    _verify_d0(d0_dir, expected_hashes)
    canonical, occurrences = _load_d0_projection(d0_dir)
    manifest = build_split(canonical, seed=seed, test_fraction=test_fraction)
    sources = build_non_test_sources(manifest, occurrences)
    validation = validate_split(
        manifest,
        expected_l2=3036,
        expected_maps={"Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring"},
        test_fraction=test_fraction,
    )
    seal = test_seal(manifest, seed=seed, test_fraction=test_fraction)
    accounting = _fold_accounting(manifest)
    config = {
        "schema": "d2-split-config-1",
        "analysis_version": "d2",
        "created_at": str(created_at),
        "source_d0_relpath": str(source_relpath),
        "source_d0_hashes": dict(sorted(expected_hashes.items())),
        "population": "D0.1 BC Primary",
        "expected_l2": 3036,
        "split_seed": int(seed),
        "test_fraction_hex": float(test_fraction).hex(),
        "outer_folds": 5,
        "inner_folds": 3,
        "test_opened": False,
    }
    validation = {
        "schema": "d2-split-validation-1",
        "passed": True,
        **validation,
        "non_test_source_count": len(sources),
        "split_manifest_sha256": split_digest(manifest),
        "test_ids_sha256": seal["test_ids_sha256"],
        "test_source_locators_emitted": 0,
        "outcome_fields_emitted": 0,
    }

    partial = _prepare_output(output_dir)
    try:
        _write_json(partial / "config.json", config)
        _write_tsv(partial / "scenario_split.tsv", manifest, SPLIT_FIELDS)
        _write_tsv(partial / "non_test_sources.tsv", sources, NON_TEST_SOURCE_FIELDS)
        _write_json(partial / "test_seal.json", seal)
        _write_json(partial / "fold_accounting.json", accounting)
        _write_json(partial / "validation.json", validation)
        _write_output_manifest(partial)
        independent = validate_split_release(d0_dir, partial, expected_hashes=expected_hashes, allow_partial=True)
        if not independent["passed"]:
            raise AssertionError("independent D2 split validation failed")
        _promote(partial, output_dir)
    except BaseException:
        # Preserve partial evidence for diagnosis; never silently reuse it.
        raise
    return validation


def validate_split_release(
    d0_dir: str | Path,
    release_dir: str | Path,
    expected_hashes: Mapping[str, str] = D0_EXPECTED_HASHES,
    allow_partial: bool = False,
) -> dict:
    d0_dir = Path(d0_dir)
    release_dir = Path(release_dir)
    if not allow_partial and not (release_dir / "COMPLETE").is_file():
        raise ValueError("D2 split release lacks COMPLETE")
    _verify_d0(d0_dir, expected_hashes)
    for name in (*RELEASE_FILES, "output_manifest.sha256"):
        if not (release_dir / name).is_file():
            raise ValueError(f"D2 split release missing {name}")
    expected_manifest_lines = {
        name: digest
        for digest, name in (
            line.split("  ", 1)
            for line in (release_dir / "output_manifest.sha256").read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    if set(expected_manifest_lines) != set(RELEASE_FILES):
        raise ValueError("D2 split output manifest inventory mismatch")
    for name in RELEASE_FILES:
        if file_sha256(release_dir / name) != expected_manifest_lines[name]:
            raise ValueError(f"D2 split output hash mismatch: {name}")

    config = json.loads((release_dir / "config.json").read_text(encoding="utf-8"))
    manifest = _read_tsv(release_dir / "scenario_split.tsv")
    sources = _read_tsv(release_dir / "non_test_sources.tsv")
    canonical, occurrences = _load_d0_projection(d0_dir)
    recomputed = build_split(
        canonical,
        seed=int(config["split_seed"]),
        test_fraction=float.fromhex(config["test_fraction_hex"]),
    )
    if manifest != recomputed:
        raise ValueError("emitted D2 split differs from outcome-blind recomputation")
    recomputed_sources = build_non_test_sources(manifest, occurrences)
    if sources != recomputed_sources:
        raise ValueError("emitted non-test source manifest differs from recomputation")
    emitted_seal = json.loads((release_dir / "test_seal.json").read_text(encoding="utf-8"))
    if emitted_seal != test_seal(
        manifest,
        seed=int(config["split_seed"]),
        test_fraction=float.fromhex(config["test_fraction_hex"]),
    ):
        raise ValueError("D2 test seal differs from recomputation")
    test_ids = {row["l2_id"] for row in manifest if row["split"] == "test"}
    if test_ids & {row["l2_id"] for row in sources}:
        raise ValueError("test source locator leaked before opening")
    validation = json.loads((release_dir / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("passed") or validation.get("split_manifest_sha256") != split_digest(manifest):
        raise ValueError("D2 split validation record mismatch")
    return {
        "passed": True,
        "split_manifest_sha256": split_digest(manifest),
        "l2_count": len(manifest),
        "non_test_source_count": len(sources),
        "test_l2_count": len(test_ids),
    }


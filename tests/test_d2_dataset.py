#!/usr/bin/env python3
"""Tests for D2 partition registry gates."""

import tempfile
from pathlib import Path

from d0.identity import append_opened_registry
from d2.dataset import _validate_registry_snapshot, make_registry_rows


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def expect_raises(name, exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"FAIL {name}: expected {exc_type.__name__}")


def selected(split="non_test"):
    return {
        "split": split,
        "l2_id": "L2:" + "1" * 64,
        "l3_id": "L3:" + "2" * 64,
        "l4_id": "L4:" + "3" * 64,
        "map_name": "Austin",
    }


def main():
    rows = make_registry_rows(
        [selected()],
        source_manifest_sha256="4" * 64,
        opened_at_utc="2026-07-11T17:40:39+08:00",
        evidence_relpath="logs/d2/split_lock",
    )
    check("one-row", len(rows) == 1)
    check("probe-fit", rows[0]["use_class"] == "probe_fit")
    check("not-final", rows[0]["final_pool"] == "false")
    check("representation-choice", rows[0]["decision_effect"] == "representation_choice")

    expect_raises(
        "test-registration-rejected",
        ValueError,
        lambda: make_registry_rows(
            [selected("test")],
            "4" * 64,
            "2026-07-11T17:40:39+08:00",
            "logs/d2/split_lock",
        ),
    )

    with tempfile.TemporaryDirectory() as td:
        registry = Path(td) / "opened_registry.tsv"
        result = append_opened_registry(registry, rows)
        check("registry-appended", result.appended == 1)
        _validate_registry_snapshot(registry, rows)

        missing = Path(td) / "missing.tsv"
        missing.write_text("\t".join(rows[0].keys()) + "\n", encoding="utf-8")
        expect_raises(
            "missing-required-row",
            ValueError,
            lambda: _validate_registry_snapshot(missing, rows),
        )

        corrupt = Path(td) / "corrupt.tsv"
        corrupt.write_text(
            registry.read_text(encoding="utf-8").replace("representation_choice", "model_choice"),
            encoding="utf-8",
        )
        expect_raises(
            "corrupt-required-row",
            ValueError,
            lambda: _validate_registry_snapshot(corrupt, rows),
        )

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

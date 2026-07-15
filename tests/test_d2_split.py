#!/usr/bin/env python3
"""Synthetic tests for the outcome-blind D2 split and test seal."""

import copy
import math

from d2.split import (
    SPLIT_FIELDS,
    build_non_test_sources,
    build_split,
    split_digest,
    test_seal,
    validate_split,
)


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def expect_raises(name, exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"FAIL {name}: expected {exc_type.__name__}")


def fixture_rows():
    rows = []
    counter = 0
    for map_index, map_name in enumerate(("MapA", "MapB")):
        for block_index in range(10):
            block_size = 2 + (block_index % 3)
            l4_id = f"L4:{map_index * 100 + block_index:064x}"
            for member in range(block_size):
                counter += 1
                rows.append(
                    {
                        "model_id": "bc",
                        "l2_id": f"L2:{counter:064x}",
                        "l3_id": f"L3:{counter:064x}",
                        "l4_id": l4_id,
                        "map_name": map_name,
                        "ego_raceline": "raceline1",
                        "opponent_raceline": f"raceline{member % 3}",
                        "speedscale_hex": float((5 + member % 4) / 10).hex(),
                        "interval_idx": "15",
                        "skill": "skill_F" if member % 2 else "other",
                        "representative_l1_id": f"L1:{counter:064x}",
                        # Forbidden fields deliberately exist in the source.
                        "collision_any": str(counter % 7 == 0),
                        "ego_collision": str(counter % 11 == 0),
                        "corrected_outcome3": "collision" if counter % 7 == 0 else "follow",
                        "collision_phase": "pre",
                    }
                )
    return rows


def occurrence_rows(rows):
    output = []
    for index, row in enumerate(rows):
        state = "collision" if index % 3 == 0 else "follow"
        output.append(
            {
                "model_id": "bc",
                "l1_id": row["representative_l1_id"],
                "l2_id": row["l2_id"],
                "resolved_ego_idx": str(index),
                "npz_relpath": f"eval_results/run/{state}/x{index}.npz",
                "npz_sha256": f"{index + 1:064x}",
            }
        )
    return output


def main():
    rows = fixture_rows()
    manifest = build_split(rows, seed=20260711, test_fraction=0.35)
    validate_split(manifest, expected_l2=len(rows), expected_maps={"MapA", "MapB"})

    check("exact-public-fields", all(tuple(row) == SPLIT_FIELDS for row in manifest))
    check("test-quota", len({r["l4_id"] for r in manifest if r["split"] == "test"}) == 8)
    for map_name in ("MapA", "MapB"):
        blocks = {r["l4_id"] for r in manifest if r["map_name"] == map_name}
        held = {r["l4_id"] for r in manifest if r["map_name"] == map_name and r["split"] == "test"}
        check(f"ceil-quota-{map_name}", len(held) == math.ceil(0.35 * len(blocks)))

    by_l4 = {}
    for row in manifest:
        by_l4.setdefault(row["l4_id"], set()).add((row["split"], row["outer_fold"]))
    check("l4-isolation", all(len(values) == 1 for values in by_l4.values()))
    check(
        "outer-five-folds",
        {row["outer_fold"] for row in manifest if row["split"] == "non_test"}
        == {"0", "1", "2", "3", "4"},
    )
    for outer in range(5):
        field = f"inner_fold_outer{outer}"
        check(
            f"inner-three-folds-{outer}",
            {row[field] for row in manifest if row["split"] == "non_test" and row["outer_fold"] != str(outer)}
            == {"0", "1", "2"},
        )
        check(
            f"outer-heldout-has-no-inner-{outer}",
            all(row[field] == "" for row in manifest if row["outer_fold"] == str(outer)),
        )

    reversed_manifest = build_split(list(reversed(rows)), seed=20260711, test_fraction=0.35)
    check("order-deterministic", manifest == reversed_manifest)

    poisoned = copy.deepcopy(rows)
    for index, row in enumerate(poisoned):
        row["collision_any"] = f"poison-{index}"
        row["ego_collision"] = f"secret-{index}"
        row["corrected_outcome3"] = "overtake" if index % 2 else "collision"
        row["collision_phase"] = "post"
        row["npz_relpath"] = f"secret/{index}"
    check("outcome-poison-invariant", build_split(poisoned) == manifest)
    check("digest-deterministic", split_digest(manifest) == split_digest(reversed_manifest))

    seal = test_seal(manifest, seed=20260711, test_fraction=0.35)
    check("seal-counts", seal["test_l2_count"] == sum(r["split"] == "test" for r in manifest))
    check("seal-no-labels", not any("collision" in key or "outcome" in key for key in seal))

    sources = build_non_test_sources(manifest, occurrence_rows(rows))
    non_test_ids = {r["l2_id"] for r in manifest if r["split"] == "non_test"}
    check("non-test-source-only", {r["l2_id"] for r in sources} == non_test_ids)
    check("no-test-source-locator", not ({r["l2_id"] for r in sources} & {r["l2_id"] for r in manifest if r["split"] == "test"}))

    corrupted = copy.deepcopy(manifest)
    same_block = corrupted[0]["l4_id"]
    peer = next(row for row in corrupted[1:] if row["l4_id"] == same_block)
    peer["split"] = "test" if corrupted[0]["split"] == "non_test" else "non_test"
    expect_raises("split-overlap-rejected", ValueError, lambda: validate_split(corrupted))

    duplicate = copy.deepcopy(manifest)
    duplicate.append(copy.deepcopy(duplicate[0]))
    expect_raises("duplicate-l2-rejected", ValueError, lambda: validate_split(duplicate))

    missing_occurrence = occurrence_rows(rows)[:-1]
    expect_raises(
        "missing-source-rejected",
        ValueError,
        lambda: build_non_test_sources(manifest, missing_occurrence),
    )

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

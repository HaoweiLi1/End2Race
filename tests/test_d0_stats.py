#!/usr/bin/env python3
"""Synthetic statistical pre-registration tests for D0.1."""

import copy

import numpy as np

from d0.stats import (
    paired_block_bootstrap,
    point_estimates,
    run_all_stats,
    stats_json_bytes,
)


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def record(map_name, block, collision=False, overtake=False):
    return {
        "map_name": map_name,
        "l4_id": block,
        "collision_any": bool(collision),
        "corrected_outcome3": "overtake" if overtake else "follow",
    }


def fixture():
    ids = ["s1", "s2", "s3", "s4", "s5"]
    bc = {
        "s1": record("A", "A1", True, False),
        "s2": record("A", "A1", False, True),
        "s3": record("A", "A2", False, True),
        "s4": record("B", "B1", True, False),
        "s5": record("B", "B1", False, False),
    }
    cand = {
        "s1": record("A", "A1", False, False),
        "s2": record("A", "A1", True, True),
        "s3": record("A", "A2", False, True),
        "s4": record("B", "B1", False, False),
        "s5": record("B", "B1", False, True),
    }
    blocks = {"A": {"A1": ["s1", "s2"], "A2": ["s3"]}, "B": {"B1": ["s4", "s5"]}}
    return ids, bc, cand, blocks


def main():
    ids, bc, cand, blocks = fixture()
    point = point_estimates(ids, bc, cand)
    check("point-N", point["N"] == 5)
    check("point-collision-counts", point["bc_collision"] == 2 and point["candidate_collision"] == 1)
    check("point-RR", point["rr_coll"] == 0.5)
    check("point-RD-coll", point["rd_coll"] == -0.2)
    check("point-RD-ot", point["rd_ot"] == 0.2)

    first = paired_block_bootstrap(ids, bc, cand, blocks, B=500, rng=np.random.default_rng(7))
    second = paired_block_bootstrap(ids, bc, cand, blocks, B=500, rng=np.random.default_rng(7))
    third = paired_block_bootstrap(ids, bc, cand, blocks, B=500, rng=np.random.default_rng(8))
    check("same-seed-bytes", stats_json_bytes(first) == stats_json_bytes(second))
    check("changed-seed-draws", first["draw_fingerprint"] != third["draw_fingerprint"])
    check("multiplicity-min", first["replicate_n"]["min"] >= 4)
    check("multiplicity-max", first["replicate_n"]["max"] <= 6)
    check("map-block-counts", first["block_counts"] == {"A": 2, "B": 1})
    check("ci-present", first["ci"]["rd_coll_95"] is not None and first["ci"]["rr_coll_upper_95"] is not None)

    identical = copy.deepcopy(bc)
    paired = paired_block_bootstrap(ids, bc, identical, blocks, B=200, rng=np.random.default_rng(9))
    check("paired-same-index-set", paired["diagnostics"]["rd_coll_min"] == 0.0 and paired["diagnostics"]["rd_coll_max"] == 0.0)

    zero_bc = {key: dict(value, collision_any=False) for key, value in bc.items()}
    zero = paired_block_bootstrap(ids, zero_bc, cand, blocks, B=200, rng=np.random.default_rng(10))
    check("bc-zero-rr", zero["point"]["rr_coll"] is None and zero["point"]["rr_status"] == "undefined_bc_zero")
    check("bc-zero-rd", zero["point"]["rd_coll"] is not None)
    check("bc-zero-degenerate", zero["rr_zero_denominator_fraction"] == 1.0)
    check("bc-zero-unstable", zero["rr_ci_status"] == "unstable")

    # Run-all uses the exact 3 x 3 x 3 child-stream order.
    mapped_bc = {}
    mapped_cand = {}
    mapped_blocks = {"Austin": {}, "Cross": {}}
    for index, scenario_id in enumerate(ids):
        target_map = "Austin" if bc[scenario_id]["map_name"] == "A" else "Cross"
        target_block = bc[scenario_id]["l4_id"]
        mapped_bc[scenario_id] = dict(bc[scenario_id], map_name=target_map)
        mapped_cand[scenario_id] = dict(cand[scenario_id], map_name=target_map)
        mapped_blocks[target_map].setdefault(target_block, []).append(scenario_id)
    records = {
        "bc": mapped_bc,
        "cand040": mapped_cand,
        "cand120": mapped_cand,
        "cand160": mapped_cand,
    }
    estimands = {"primary": set(ids), "sensA": set(ids[:-1]), "sensB": set(ids[1:])}
    run1 = run_all_stats(estimands, records, mapped_blocks, B=100, seed=20260710)
    run2 = run_all_stats(estimands, records, mapped_blocks, B=100, seed=20260710)
    run3 = run_all_stats(estimands, records, mapped_blocks, B=100, seed=20260711)
    check("runall-order-count", len(run1["child_order"]) == 27)
    expected_first = {"child_index": 0, "estimand": "primary", "candidate": "cand040", "pool": "all"}
    expected_last = {"child_index": 26, "estimand": "sensB", "candidate": "cand160", "pool": "cross"}
    check("runall-order-ends", run1["child_order"][0] == expected_first and run1["child_order"][-1] == expected_last)
    check("runall-byte-determinism", stats_json_bytes(run1) == stats_json_bytes(run2))
    fingerprints1 = [item["draw_fingerprint"] for item in run1["results"].values() if item["point"]["N"]]
    fingerprints3 = [item["draw_fingerprint"] for item in run3["results"].values() if item["point"]["N"]]
    check("runall-seed-sanity", fingerprints1 != fingerprints3)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

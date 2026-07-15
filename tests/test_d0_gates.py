#!/usr/bin/env python3
"""Synthetic negative and positive tests for D0.1 gates G1--G8."""

import copy
import hashlib
import tempfile
from pathlib import Path

from d0.gates import (
    release_verdict,
    run_g1_duplicate_determinism,
    run_g2_inventory,
    run_g3_integrity,
    run_g4_near_duplicate,
    run_g5_collision_floors,
    run_g6_record_physics,
    run_g7_unknown_censoring,
    run_g8_reconciliation,
)
from d0.outcomes import EQUALITY_FIELDS


STATES = ("collision", "confirmed_pass", "terminal_overtake_only", "safe_follow")


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def equality_record(model="bc", l2="L2:a", **changes):
    defaults = {
        "archived_outcome_raw": "following",
        "archived_outcome3": "follow",
        "ego_collision": False,
        "opp_collision": False,
        "collision_any": False,
        "alignment_status": "ok",
        "alignment_k": 0,
        "rel_start_hex": (-1.0).hex(),
        "rel_terminal_hex": (-0.5).hex(),
        "ego_wrap_count": 0,
        "opp_wrap_count": 0,
        "physics_status": "ok",
        "terminal_gap_hex": (0.01).hex(),
        "frame_spacing_status": "ok",
        "censored": False,
        "interaction_attempt": True,
        "confirmed_safe_pass": False,
        "attempted_follow_no_collision": True,
        "corrected_outcome3": "follow",
        "four_state": "safe_follow",
        "collision_involvement": "not_applicable",
        "collision_cause": "not_applicable",
        "collision_phase": "not_applicable",
        "collision_final_dist_hex": "not_applicable",
    }
    defaults.update(changes)
    return {"model_id": model, "l2_id": l2, **defaults}


def inventory_record(key, **changes):
    row = {"inventory_key": key, "validated": True, "stale": False}
    row.update(changes)
    return row


def integrity_row(l1, path, root, expected=None):
    data = Path(path).read_bytes() if Path(path).exists() else b""
    return {
        "l1_id": l1,
        "path": str(path),
        "root": str(root),
        "expected_sha256": expected or hashlib.sha256(data).hexdigest(),
    }


def g6_record(key="k", **changes):
    row = {
        "inventory_key": key,
        "has_terminal_fields": True,
        "raw_label_ok": True,
        "json_npz_label_match": True,
        "collision_identity_ok": True,
        "equal_lengths": True,
        "finite_values": True,
        "exact_resolution": True,
        "frame_spacing_status": "ok",
        "terminal_gap_ok": True,
    }
    row.update(changes)
    return row


def matrix(n=10):
    rows = []
    for left in STATES:
        for right in STATES:
            rows.append({"bc_state": left, "candidate_state": right, "count": 0})
    rows[0]["count"] = n
    return {"rows": rows, "expected_n": n}


def g8_args():
    canonical = [
        {"model_id": "bc", "l2_id": "L2:1", "archived_outcome3": "follow", "corrected_outcome3": "follow", "collision_any": False},
        {"model_id": "cand", "l2_id": "L2:1", "archived_outcome3": "overtake", "corrected_outcome3": "follow", "collision_any": False},
        {"model_id": "bc", "l2_id": "L2:2", "archived_outcome3": "collision", "corrected_outcome3": "collision", "collision_any": True},
    ]
    return {
        "canonical_records": canonical,
        "correction_rows": [{"model_id": "cand", "l2_id": "L2:1"}],
        "collision_rows": [{"model_id": "bc", "l2_id": "L2:2"}],
        "matrices": {"primary:cand:all": matrix(2)},
        "summary": {"counts": {"canonical": 2, "corrections": 1, "collisions": 1}},
        "expected_summary": {"counts": {"canonical": 2, "corrections": 1, "collisions": 1}},
        "expected_estimands": {"primary": {"L2:1", "L2:2"}},
        "emitted_estimands": {"primary": ["L2:1", "L2:2"]},
        "required_manifest_files": {"canonical.tsv", "summary.json"},
        "manifest_hashes": {"canonical.tsv": "a" * 64, "summary.json": "b" * 64},
        "registry_rows": [{"row_id": "registry-1", "value": "expected"}],
        "required_registry_rows": [{"row_id": "registry-1", "value": "expected"}],
    }


def main():
    base = equality_record()
    g1 = run_g1_duplicate_determinism([base, dict(base)])
    check("g1-happy", g1.passed)
    for field in EQUALITY_FIELDS:
        changed = dict(base)
        value = changed[field]
        if isinstance(value, bool):
            changed[field] = not value
        elif isinstance(value, int):
            changed[field] = value + 1
        else:
            changed[field] = str(value) + "_different"
        result = run_g1_duplicate_determinism([base, changed])
        check(f"g1-field-{field}", not result.passed and any(field in v for v in result.violations))

    expected = {"a", "b"}
    check("g2-happy", run_g2_inventory(expected, [inventory_record("a"), inventory_record("b")]).passed)
    check("g2-missing", not run_g2_inventory(expected, [inventory_record("a")]).passed)
    check("g2-extra", not run_g2_inventory(expected, [inventory_record("a"), inventory_record("b"), inventory_record("c")]).passed)
    check("g2-duplicate", not run_g2_inventory(expected, [inventory_record("a"), inventory_record("a"), inventory_record("b")]).passed)
    check("g2-unvalidated", not run_g2_inventory(expected, [inventory_record("a", validated=False), inventory_record("b")]).passed)
    check("g2-stale", not run_g2_inventory(expected, [inventory_record("a", stale=True), inventory_record("b")]).passed)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "root"
        root.mkdir()
        good = root / "good.npz"
        good.write_bytes(b"payload")
        empty = root / "empty.npz"
        empty.write_bytes(b"")
        outside = Path(td) / "outside.npz"
        outside.write_bytes(b"payload")
        happy = integrity_row("L1:" + "1" * 64, good, root)
        check("g3-happy", run_g3_integrity([happy]).passed)
        check("g3-empty", not run_g3_integrity([integrity_row("L1:" + "2" * 64, empty, root)]).passed)
        check("g3-badhash", not run_g3_integrity([integrity_row("L1:" + "3" * 64, good, root, "0" * 64)]).passed)
        check("g3-escape", not run_g3_integrity([integrity_row("L1:" + "4" * 64, outside, root)]).passed)
        check("g3-duplicate-l1", not run_g3_integrity([happy, happy]).passed)

    matched = {
        "left": {"map_name": "M", "ego_raceline": "r1", "opponent_raceline": "r1", "speedscale": 0.5, "interval_idx": 15, "l3_id": "a", "l4_id": "x", "outcome": "follow"},
        "right": {"map_name": "M", "ego_raceline": "r1", "opponent_raceline": "r1", "speedscale": 0.5, "interval_idx": 15, "l3_id": "b", "l4_id": "x", "outcome": "overtake"},
    }
    unmatched = copy.deepcopy(matched)
    unmatched["right"]["speedscale"] = 0.6
    g4 = run_g4_near_duplicate([matched, unmatched])
    check("g4-matched-only", g4.passed and g4.counts["compared"] == 1 and g4.counts["disagreements"] == 1)

    floor_records = []
    for skill in ("skill_F", "skill_S"):
        for i in range(30):
            floor_records.append({"model_id": "bc", "l2_id": f"{skill}-{i}", "skill": skill, "ego_collision": True})
    check("g5-30", run_g5_collision_floors(floor_records).passed)
    check("g5-29", not run_g5_collision_floors(floor_records[:-1]).passed)

    check("g6-happy", run_g6_record_physics([g6_record()]).passed)
    for field, value in (
        ("has_terminal_fields", False),
        ("raw_label_ok", False),
        ("json_npz_label_match", False),
        ("collision_identity_ok", False),
        ("equal_lengths", False),
        ("finite_values", False),
        ("exact_resolution", False),
        ("frame_spacing_status", "invalid"),
        ("terminal_gap_ok", False),
    ):
        result = run_g6_record_physics([g6_record(**{field: value})])
        check(f"g6-{field}", not result.passed and any(field in v for v in result.violations))

    check("g7-happy", run_g7_unknown_censoring([base]).passed)
    check("g7-alignment", not run_g7_unknown_censoring([equality_record(alignment_status="alignment_failure")]).passed)
    check("g7-unknown", not run_g7_unknown_censoring([equality_record(corrected_outcome3="unknown")]).passed)
    check("g7-censored", not run_g7_unknown_censoring([equality_record(censored=True)]).passed)

    args = g8_args()
    check("g8-happy", run_g8_reconciliation(**args).passed)
    omitted = copy.deepcopy(args)
    omitted["correction_rows"] = []
    check("g8-omitted-correction", not run_g8_reconciliation(**omitted).passed)
    extra = copy.deepcopy(args)
    extra["correction_rows"].append({"model_id": "bc", "l2_id": "L2:1"})
    check("g8-extra-correction", not run_g8_reconciliation(**extra).passed)
    duplicate = copy.deepcopy(args)
    duplicate["correction_rows"].append(dict(duplicate["correction_rows"][0]))
    check("g8-duplicate-correction", not run_g8_reconciliation(**duplicate).passed)
    corrupt_summary = copy.deepcopy(args)
    corrupt_summary["summary"]["counts"]["corrections"] = 99
    check("g8-corrupt-summary", not run_g8_reconciliation(**corrupt_summary).passed)
    corrupt_matrix = copy.deepcopy(args)
    corrupt_matrix["matrices"]["primary:cand:all"]["rows"][0]["count"] = 1
    check("g8-corrupt-matrix", not run_g8_reconciliation(**corrupt_matrix).passed)
    missing_manifest = copy.deepcopy(args)
    missing_manifest["manifest_hashes"].pop("summary.json")
    check("g8-missing-manifest", not run_g8_reconciliation(**missing_manifest).passed)
    drift = copy.deepcopy(args)
    drift["emitted_estimands"]["primary"] = ["L2:1"]
    check("g8-estimand-drift", not run_g8_reconciliation(**drift).passed)
    missing_registry = copy.deepcopy(args)
    missing_registry["registry_rows"] = []
    check("g8-missing-registry", not run_g8_reconciliation(**missing_registry).passed)
    corrupt_registry = copy.deepcopy(args)
    corrupt_registry["registry_rows"][0]["value"] = "corrupt"
    check("g8-corrupt-registry", not run_g8_reconciliation(**corrupt_registry).passed)
    duplicate_registry = copy.deepcopy(args)
    duplicate_registry["registry_rows"].append(dict(duplicate_registry["registry_rows"][0]))
    check("g8-duplicate-registry", not run_g8_reconciliation(**duplicate_registry).passed)

    all_gates = [
        run_g1_duplicate_determinism([base]),
        run_g2_inventory({"a"}, [inventory_record("a")]),
        run_g4_near_duplicate([]),
        run_g5_collision_floors(floor_records),
        run_g6_record_physics([g6_record()]),
        run_g7_unknown_censoring([base]),
        run_g8_reconciliation(**args),
    ]
    # G3 uses a real file and is exercised above; release requires it too.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x"
        p.write_bytes(b"x")
        all_gates.insert(2, run_g3_integrity([integrity_row("L1:" + "f" * 64, p, td)]))
        check("release-happy", release_verdict(all_gates).passed)
        bad = list(all_gates)
        bad[0] = run_g1_duplicate_determinism([base, equality_record(censored=True)])
        check("release-blocks", not release_verdict(bad).passed)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

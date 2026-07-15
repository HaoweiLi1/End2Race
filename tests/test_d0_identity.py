#!/usr/bin/env python3
"""Synthetic contract tests for D0.1 identity and registry semantics."""

import csv
import json
import tempfile
from pathlib import Path

from d0.identity import (
    REGISTRY_FIELDS,
    append_opened_registry,
    asset_namespace,
    asset_namespace_from_entries,
    build_l4_blocks,
    canonical_json,
    domain_id,
    make_l1_payload,
    make_l2_payload,
    make_l3_payload,
    registry_row_id,
    sensitivity_a_pairs,
    sensitivity_b_membership,
)
from d0 import default_runconfig


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def expect_raises(name, exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"FAIL {name}: expected {exc_type.__name__}")


def l2(map_name="Map", speed=0.5, opponent_x=2.0):
    return make_l2_payload(
        asset_namespace_sha256="a" * 64,
        map_name=map_name,
        ego_raceline="raceline1",
        opponent_raceline="raceline1",
        ego_start_pose=(0.0, 1.0, -0.5),
        opponent_start_pose=(opponent_x, 1.0, -0.5),
        ego_waypoint_speed=5.0,
        ego_prev_speed_input=4.5,
        ego_initial_actual_speed=0.0,
        opponent_initial_actual_speed=0.0,
        opponent_speedscale=speed,
        interval_idx=15,
        sim_dt=0.01,
        duration_ticks=800,
        noise_fraction=0.0,
        noise_seed=42,
    )


def main():
    payload = make_l3_payload(
        asset_namespace_sha256="a" * 64,
        map_name="Map",
        ego_raceline="raceline1",
        ego_start_pose=(0.0, 1.0, -0.5),
    )
    golden = (
        '{"asset_namespace_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"ego_raceline":"raceline1","ego_start_pose_hex":["0x0.0p+0",'
        '"0x1.0000000000000p+0","-0x1.0000000000000p-1"],'
        '"map_name":"Map","schema":"d0.1-l3-exact-start-1"}'
    )
    check("canonical-json-golden", canonical_json(payload) == golden.encode("ascii"))
    check(
        "l3-known-vector",
        domain_id("L3", payload)
        == "L3:a549725560427a41f54964e2d34019cc42c8c69f097c6c89a32b5ebed554b0a6",
    )
    check("domains-separated", domain_id("L2", l2()) != domain_id("L3", payload))

    missing = dict(payload)
    missing.pop("map_name")
    expect_raises("missing-key-rejected", ValueError, lambda: domain_id("L3", missing))
    extra = dict(payload, extra="x")
    expect_raises("extra-key-rejected", ValueError, lambda: domain_id("L3", extra))
    expect_raises(
        "nan-rejected",
        ValueError,
        lambda: make_l3_payload("a" * 64, "Map", "raceline1", (float("nan"), 0.0, 0.0)),
    )

    adjacent = float.fromhex("0x1.0000000000001p+0")
    p_adjacent = make_l3_payload("a" * 64, "Map", "raceline1", (adjacent, 1.0, -0.5))
    check("float-hex-adjacent", payload["ego_start_pose_hex"] != p_adjacent["ego_start_pose_hex"])

    entries = [
        {"relpath": "b.csv", "sha256": "b" * 64},
        {"relpath": "a.csv", "sha256": "a" * 64},
    ]
    ns1 = asset_namespace_from_entries(entries)
    ns2 = asset_namespace_from_entries(list(reversed(entries)))
    check("asset-order-independent", ns1 == ns2)
    production_config = default_runconfig()
    check(
        "asset-grid-map-dedup-precondition",
        len({item[0] for item in production_config["grids"]})
        < len(production_config["grids"]),
    )
    check("asset-grid-map-dedup", len(asset_namespace(production_config).entries) == 12)

    l2_id = domain_id("L2", l2())
    base_l1 = dict(
        source_run_id="20260710_121955",
        model_id="bc",
        model_relpath="pretrained/end2race.pth",
        checkpoint_sha256="b" * 64,
        map_name="Map",
        grid_id="Map_off0",
        offset=0,
        tag="tag",
        result_json_relpath="eval_results/tag_Map/results.json",
        result_json_sha256="c" * 64,
        episode_key="ol1_e0_o15_s0.5",
        npz_relpath="eval_results/tag_Map/follow/f_x.npz",
        npz_sha256="d" * 64,
        l2_id=l2_id,
    )
    p1 = make_l1_payload(**base_l1)
    p2 = make_l1_payload(**dict(base_l1, npz_sha256="e" * 64))
    p3 = make_l1_payload(**dict(base_l1, checkpoint_sha256="f" * 64))
    check("l1-binds-npz", domain_id("L1", p1) != domain_id("L1", p2))
    check("l1-binds-model", domain_id("L1", p1) != domain_id("L1", p3))

    nodes = [
        {"l3_id": "L3:" + "1" * 64, "asset_namespace_sha256": "a" * 64,
         "map_name": "Map", "ego_raceline": "raceline1", "x": 0.0, "y": 0.0, "is_dev": True},
        {"l3_id": "L3:" + "2" * 64, "asset_namespace_sha256": "a" * 64,
         "map_name": "Map", "ego_raceline": "raceline1", "x": 1.0, "y": 0.0, "is_dev": False},
        {"l3_id": "L3:" + "3" * 64, "asset_namespace_sha256": "a" * 64,
         "map_name": "Map", "ego_raceline": "raceline1", "x": 2.0000001, "y": 0.0, "is_dev": False},
    ]
    blocks1 = build_l4_blocks(nodes)
    blocks2 = build_l4_blocks(list(reversed(nodes)))
    check("block-order-independent", blocks1.l3_to_l4 == blocks2.l3_to_l4)
    check("one-meter-inclusive", blocks1.l3_to_l4[nodes[0]["l3_id"]] == blocks1.l3_to_l4[nodes[1]["l3_id"]])
    check("over-one-separated", blocks1.l3_to_l4[nodes[1]["l3_id"]] != blocks1.l3_to_l4[nodes[2]["l3_id"]])

    records = []
    for map_name in ("Nuerburgring", "MoscowRaceway", "Hockenheim"):
        for speed in (0.5, 0.6, 0.7, 0.8):
            keep_payload = l2(map_name, speed, opponent_x=1.0)
            drop_payload = l2(map_name, speed, opponent_x=2.0)
            records.extend(
                [
                    {"l2_id": domain_id("L2", keep_payload), "l2_payload": keep_payload,
                     "resolved_ego_indices": [0], "endpoint_ego_pose_speed_equal": True},
                    {"l2_id": domain_id("L2", drop_payload), "l2_payload": drop_payload,
                     "resolved_ego_indices": [99], "endpoint_ego_pose_speed_equal": True},
                ]
            )
    pairs = sensitivity_a_pairs(records, require_full_pattern=True)
    check("sensa-twelve", len(pairs) == 12)
    check("sensa-both-ids", len({p["retained_l2_id"] for p in pairs}) == 12 and len({p["excluded_l2_id"] for p in pairs}) == 12)
    check("sensa-min-index", all(p["retained_min_resolved_ego_idx"] < p["excluded_min_resolved_ego_idx"] for p in pairs))
    expect_raises("sensa-omission", ValueError, lambda: sensitivity_a_pairs(records[:-2], require_full_pattern=True))
    broken_endpoint = [dict(r) for r in records]
    broken_endpoint[0]["endpoint_ego_pose_speed_equal"] = False
    expect_raises("sensa-endpoint", ValueError, lambda: sensitivity_a_pairs(broken_endpoint, require_full_pattern=True))

    l2_to_l3 = {f"L2:{i:064x}": nodes[i % 3]["l3_id"] for i in range(5)}
    primary_excluded = {next(iter(l2_to_l3))}
    dev_l4s = {blocks1.l3_to_l4[nodes[0]["l3_id"]]}
    sb = sensitivity_b_membership(l2_to_l3, primary_excluded, blocks1.l3_to_l4, dev_l4s)
    check("sensb-from-exact", sb["sensB"] == set(l2_to_l3) - sb["excluded_from_exact_ids"])
    check("sensb-accounting", sb["excluded_from_exact"] - sb["already_excluded_by_primary"] == sb["additional_vs_primary"])
    check("sensb-primary-identity", len(l2_to_l3) - len(primary_excluded) - sb["additional_vs_primary"] == len(sb["sensB"]))

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "opened_registry.tsv"
        row = {
            "registry_schema": "bplus-opened-registry-1",
            "opened_at_utc": "2026-07-10T23:02:12+08:00",
            "stage": "D0.1",
            "use_class": "historical_analysis",
            "split_id": "p1",
            "l2_id": l2_id,
            "l3_id": domain_id("L3", payload),
            "l4_id": "L4:" + "4" * 64,
            "map_name": "Map",
            "source_manifest_sha256": "5" * 64,
            "source_run_id": "20260710_121955",
            "decision_effect": "historical_only",
            "final_pool": "false",
            "evidence_relpath": "logs/evidence",
        }
        row["row_id"] = registry_row_id(row)
        first = append_opened_registry(path, [row])
        second = append_opened_registry(path, [row])
        check("registry-append", first.appended == 1 and second.appended == 0)
        with path.open(newline="") as f:
            parsed = list(csv.DictReader(f, delimiter="\t"))
            check("registry-header", tuple(parsed[0]) == REGISTRY_FIELDS)
            check("registry-idempotent", len(parsed) == 1)
        text = path.read_text()
        path.write_text(text.replace("historical_only", "model_choice"))
        expect_raises("registry-conflict", ValueError, lambda: append_opened_registry(path, [row]))

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

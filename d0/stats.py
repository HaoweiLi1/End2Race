"""Scenario-weighted effects and paired map-stratified L4 bootstrap."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable, Mapping

import numpy as np


ESTIMAND_ORDER = ("primary", "sensA", "sensB")
CANDIDATE_ORDER = ("cand040", "cand120", "cand160")
POOL_ORDER = ("all", "austin", "cross")


def stats_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _validated_ids(pool: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(sorted(str(item) for item in pool))
    if len(ids) != len(set(ids)):
        raise ValueError("pool contains duplicate scenario IDs")
    return ids


def _collision(record: Mapping) -> int:
    return int(bool(record["collision_any"]))


def _overtake(record: Mapping) -> int:
    return int(record["corrected_outcome3"] == "overtake")


def point_estimates(
    pool: Iterable[str],
    bc_records: Mapping[str, Mapping],
    candidate_records: Mapping[str, Mapping],
) -> dict:
    ids = _validated_ids(pool)
    missing_bc = set(ids) - set(bc_records)
    missing_candidate = set(ids) - set(candidate_records)
    if missing_bc or missing_candidate:
        raise ValueError(
            f"point-estimate records missing: bc={sorted(missing_bc)} "
            f"candidate={sorted(missing_candidate)}"
        )
    n = len(ids)
    bc_collision = sum(_collision(bc_records[item]) for item in ids)
    candidate_collision = sum(_collision(candidate_records[item]) for item in ids)
    bc_overtake = sum(_overtake(bc_records[item]) for item in ids)
    candidate_overtake = sum(_overtake(candidate_records[item]) for item in ids)
    if n:
        rd_coll = (candidate_collision - bc_collision) / n
        rd_ot = (candidate_overtake - bc_overtake) / n
    else:
        rd_coll = None
        rd_ot = None
    if bc_collision:
        rr_coll = candidate_collision / bc_collision
        rr_status = "defined"
    else:
        rr_coll = None
        rr_status = "undefined_bc_zero"
    return {
        "N": n,
        "bc_collision": bc_collision,
        "candidate_collision": candidate_collision,
        "bc_overtake": bc_overtake,
        "candidate_overtake": candidate_overtake,
        "rr_coll": rr_coll,
        "rr_status": rr_status,
        "rd_coll": rd_coll,
        "rd_ot": rd_ot,
    }


def _interval(values: np.ndarray, lower: float, upper: float):
    if values.size == 0:
        return None
    result = np.percentile(values, [lower, upper], method="linear")
    return [float(result[0]), float(result[1])]


def _percentile(values: np.ndarray, q: float):
    if values.size == 0:
        return None
    return float(np.percentile(values, q, method="linear"))


def paired_block_bootstrap(
    pool: Iterable[str],
    bc_records: Mapping[str, Mapping],
    candidate_records: Mapping[str, Mapping],
    blocks_by_map: Mapping[str, Mapping[str, Iterable[str]]],
    *,
    B: int = 10000,
    rng,
) -> dict:
    ids = _validated_ids(pool)
    if not isinstance(B, int) or isinstance(B, bool) or B <= 0:
        raise ValueError("B must be a positive integer")
    point = point_estimates(ids, bc_records, candidate_records)
    if not ids:
        return {
            "schema": "d0.1-bootstrap-result-1",
            "B": B,
            "point": point,
            "block_counts": {},
            "replicate_n": {"min": 0, "max": 0},
            "rr_zero_denominator_fraction": 1.0,
            "rr_ci_status": "unstable",
            "ci": {"rr_coll_95": None, "rr_coll_upper_95": None, "rd_coll_95": None, "rd_ot_95": None},
            "draw_fingerprint": hashlib.sha256(b"empty-pool").hexdigest(),
            "diagnostics": {"rd_coll_min": None, "rd_coll_max": None, "rd_ot_min": None, "rd_ot_max": None},
        }

    pool_set = set(ids)
    ownership = {}
    block_aggregates: dict[str, list[dict]] = {}
    for map_name in sorted(blocks_by_map):
        entries = []
        for l4_id in sorted(blocks_by_map[map_name]):
            members = sorted(pool_set & {str(item) for item in blocks_by_map[map_name][l4_id]})
            if not members:
                continue
            for item in members:
                prior = ownership.get(item)
                if prior is not None:
                    raise ValueError(f"scenario {item} appears in blocks {prior} and {(map_name, l4_id)}")
                ownership[item] = (map_name, l4_id)
                if bc_records[item].get("map_name") != map_name or candidate_records[item].get("map_name") != map_name:
                    raise ValueError(f"scenario {item} map/block ownership mismatch")
                if bc_records[item].get("l4_id") != l4_id or candidate_records[item].get("l4_id") != l4_id:
                    raise ValueError(f"scenario {item} L4 ownership mismatch")
            entries.append(
                {
                    "l4_id": l4_id,
                    "n": len(members),
                    "bc_collision": sum(_collision(bc_records[item]) for item in members),
                    "candidate_collision": sum(_collision(candidate_records[item]) for item in members),
                    "bc_overtake": sum(_overtake(bc_records[item]) for item in members),
                    "candidate_overtake": sum(_overtake(candidate_records[item]) for item in members),
                }
            )
        if entries:
            block_aggregates[map_name] = entries
    if set(ownership) != pool_set:
        raise ValueError(f"pool scenarios without block ownership: {sorted(pool_set-set(ownership))}")

    total_n = np.zeros(B, dtype=np.int64)
    bc_collision = np.zeros(B, dtype=np.int64)
    candidate_collision = np.zeros(B, dtype=np.int64)
    bc_overtake = np.zeros(B, dtype=np.int64)
    candidate_overtake = np.zeros(B, dtype=np.int64)
    fingerprint = hashlib.sha256()
    block_counts = {}
    for map_name in sorted(block_aggregates):
        entries = block_aggregates[map_name]
        count = len(entries)
        block_counts[map_name] = count
        draws = rng.integers(0, count, size=(B, count), dtype=np.int64)
        fingerprint.update(map_name.encode("utf-8") + b"\0")
        fingerprint.update(draws.astype("<i8", copy=False).tobytes(order="C"))
        for target, field in (
            (total_n, "n"),
            (bc_collision, "bc_collision"),
            (candidate_collision, "candidate_collision"),
            (bc_overtake, "bc_overtake"),
            (candidate_overtake, "candidate_overtake"),
        ):
            values = np.asarray([entry[field] for entry in entries], dtype=np.int64)
            target += values[draws].sum(axis=1)
    if np.any(total_n <= 0):
        raise AssertionError("bootstrap produced an empty expanded sample")

    rd_coll = (candidate_collision - bc_collision) / total_n
    rd_ot = (candidate_overtake - bc_overtake) / total_n
    valid_rr = bc_collision > 0
    rr = candidate_collision[valid_rr] / bc_collision[valid_rr]
    zero_fraction = float(1.0 - np.mean(valid_rr))
    rr_status = "unstable" if zero_fraction > 0.01 or rr.size == 0 else "stable"
    return {
        "schema": "d0.1-bootstrap-result-1",
        "B": B,
        "point": point,
        "block_counts": block_counts,
        "replicate_n": {"min": int(total_n.min()), "max": int(total_n.max())},
        "rr_zero_denominator_fraction": zero_fraction,
        "rr_ci_status": rr_status,
        "ci": {
            "rr_coll_95": _interval(rr, 2.5, 97.5),
            "rr_coll_upper_95": _percentile(rr, 95.0),
            "rd_coll_95": _interval(rd_coll, 2.5, 97.5),
            "rd_ot_95": _interval(rd_ot, 2.5, 97.5),
        },
        "draw_fingerprint": fingerprint.hexdigest(),
        "diagnostics": {
            "rd_coll_min": float(rd_coll.min()),
            "rd_coll_max": float(rd_coll.max()),
            "rd_ot_min": float(rd_ot.min()),
            "rd_ot_max": float(rd_ot.max()),
        },
    }


def run_all_stats(
    estimands: Mapping[str, Iterable[str]],
    records: Mapping[str, Mapping[str, Mapping]],
    blocks_by_map: Mapping[str, Mapping[str, Iterable[str]]],
    *,
    B: int = 10000,
    seed: int = 20260710,
) -> dict:
    if set(estimands) != set(ESTIMAND_ORDER):
        raise ValueError(f"estimands must be exactly {ESTIMAND_ORDER}")
    if "bc" not in records or any(candidate not in records for candidate in CANDIDATE_ORDER):
        raise ValueError("records must contain BC and all three candidates")
    root = np.random.default_rng(seed)
    children = root.spawn(27)
    child_order = []
    results = {}
    child_index = 0
    for estimand in ESTIMAND_ORDER:
        estimand_ids = _validated_ids(estimands[estimand])
        for candidate in CANDIDATE_ORDER:
            for pool_name in POOL_ORDER:
                order_row = {
                    "child_index": child_index,
                    "estimand": estimand,
                    "candidate": candidate,
                    "pool": pool_name,
                }
                child_order.append(order_row)
                if pool_name == "all":
                    pool_ids = estimand_ids
                elif pool_name == "austin":
                    pool_ids = tuple(
                        item for item in estimand_ids if records["bc"][item]["map_name"] == "Austin"
                    )
                else:
                    pool_ids = tuple(
                        item for item in estimand_ids if records["bc"][item]["map_name"] != "Austin"
                    )
                result = paired_block_bootstrap(
                    pool_ids,
                    records["bc"],
                    records[candidate],
                    blocks_by_map,
                    B=B,
                    rng=children[child_index],
                )
                result["child_index"] = child_index
                result["estimand"] = estimand
                result["candidate"] = candidate
                result["pool"] = pool_name
                results[f"{estimand}|{candidate}|{pool_name}"] = result
                child_index += 1
    return {
        "schema": "d0.1-stats-all-1",
        "B": B,
        "seed": int(seed),
        "spawn_method": "numpy.random.default_rng(seed).spawn(27)",
        "child_order": child_order,
        "results": results,
    }

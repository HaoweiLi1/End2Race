#!/usr/bin/env python3
"""Unit tests for D2.5 selection, registry, digest, and Route-R2 gates."""

from __future__ import annotations

import random

import numpy as np

from d25.oracle import ARRAY_KEYS
from d25.search import (
    CASE_FIELDS,
    _case_manifest,
    branch_category,
    make_registry_rows,
    route_summary,
    select_smoke_cases,
    trajectory_digest,
)


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def case(index, map_name, skill):
    token = f"{index:064x}"
    return {
        "l2_id": f"L2:{token}",
        "l3_id": f"L3:{token}",
        "l4_id": f"L4:{index % 8:064x}",
        "map_name": map_name,
        "skill": skill,
    }


def arrays(seed=0):
    result = {}
    for index, key in enumerate(ARRAY_KEYS):
        if key == "state_label":
            result[key] = np.asarray("collision")
        elif key in {"collision", "ego_collision", "opp_collision"}:
            result[key] = np.asarray(key != "opp_collision", dtype=bool)
        elif key in {"final_time", "final_ego_progress", "final_opp_progress"}:
            result[key] = np.asarray(seed + index, dtype=np.float32)
        elif key in {"final_ego_pose", "final_opp_pose"}:
            result[key] = np.full(3, seed + index, dtype=np.float32)
        elif "lidar" in key:
            result[key] = np.full((2, 360), seed + index, dtype=np.float32)
        elif "pose" in key:
            result[key] = np.full((2, 3), seed + index, dtype=np.float32)
        else:
            result[key] = np.full(2, seed + index, dtype=np.float32)
    return result


class Outcome:
    def __init__(self, four_state):
        self.four_state = four_state


def main():
    maps = ["Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring"]
    cases = []
    index = 1
    for map_name in maps:
        for skill in ("skill_F", "skill_S", "other"):
            cases.extend([case(index, map_name, skill), case(index + 1, map_name, skill)])
            index += 2
    shuffled = list(cases)
    random.Random(20260711).shuffle(shuffled)
    selected = select_smoke_cases(shuffled)
    check("smoke-eight", len(selected) == 8)
    check("smoke-order-invariant", selected == select_smoke_cases(list(reversed(shuffled))))
    check(
        "smoke-two-per-map",
        all(sum(row["map_name"] == map_name for row in selected) == 2 for map_name in maps),
    )
    manifest_sources = []
    for row in cases:
        manifest_sources.append(
            {
                **row,
                "opponent_raceline": "raceline1",
                "speedscale_hex": float(0.6).hex(),
                "resolved_ego_idx": "0",
                "npz_relpath": "source.npz",
                "npz_sha256": "0" * 64,
                "frame_count": "200",
                "final_time_hex": float(2.0).hex(),
            }
        )
    manifest = _case_manifest(manifest_sources, {row["l2_id"] for row in selected})
    check("manifest-fields", all(tuple(row) == CASE_FIELDS for row in manifest))

    registry = make_registry_rows(cases)
    check("registry-count", len(registry) == len(cases))
    check("registry-semantics", all(row["use_class"] == "oracle_search" for row in registry))
    check("registry-final-false", all(row["final_pool"] == "false" for row in registry))

    digest = trajectory_digest(arrays())
    check("digest-repeat", digest == trajectory_digest(arrays()))
    check("digest-change", digest != trajectory_digest(arrays(1)))

    expected = {
        "collision": "still_collision",
        "confirmed_pass": "collision_to_confirmed_safe_pass",
        "terminal_overtake_only": "collision_to_terminal_overtake_only",
        "safe_follow": "collision_to_safe_abort_follow",
    }
    for state, category in expected.items():
        check(f"category-{state}", branch_category(Outcome(state), False) == category)
    check(
        "category-clipped",
        branch_category(Outcome("confirmed_pass"), True) == "invalid_or_action_clipped",
    )

    gate_cases = [case(i, maps[i % 4], "skill_S" if i < 20 else "skill_F") for i in range(40)]
    results = []
    for row in gate_cases:
        recovered = int(row["l2_id"][3:], 16) < 25
        results.append(
            {
                **row,
                "status": "recovered_confirmed_safe_pass" if recovered else "exhausted_no_confirmed_safe_pass",
                "witness_branch_id": "witness" if recovered else "NA",
            }
        )
    summary = route_summary(gate_cases, results)
    check("route-feasible", summary["route_r2_feasible"])
    check("route-25", summary["recovered_case_count"] == 25)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

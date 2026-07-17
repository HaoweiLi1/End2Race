#!/usr/bin/env python3
"""Freeze the P4 96-case diagnostic panel before any P4 rollout is collected."""

from __future__ import annotations

from datetime import datetime

try:
    from audit_rl_direction_common import (
        EXPERIMENT_DIR,
        PREREGISTRATION_PATH,
        assert_frozen_contract,
        read_json,
        write_json_atomic,
    )
except ModuleNotFoundError:
    from scripts.audit_rl_direction_common import (
        EXPERIMENT_DIR,
        PREREGISTRATION_PATH,
        assert_frozen_contract,
        read_json,
        write_json_atomic,
    )
from ppo.scenarios import load_hard_pool


def _rows(scenarios, source: str, excluded: set[str], count: int) -> list[dict]:
    eligible = sorted(
        (scenario for scenario in scenarios if scenario.scenario_id not in excluded),
        key=lambda scenario: scenario.scenario_id,
    )
    selected = eligible[:count]
    if len(selected) != count:
        raise RuntimeError(f"P4 panel source {source} has only {len(selected)} eligible cases")
    rows = [
        {"scenario_id": scenario.scenario_id, "source": source, "scenario": scenario.to_dict()}
        for scenario in selected
    ]
    excluded.update(row["scenario_id"] for row in rows)
    return rows


def main() -> None:
    frozen_hashes = assert_frozen_contract()
    preregistration = read_json(PREREGISTRATION_PATH)
    safe_reference = read_json(EXPERIMENT_DIR / "SAFE_REFERENCE.json")
    h0, _h0_ids, _h0_manifest = load_hard_pool("h0_current_det")
    h1, _h1_ids, _h1_manifest = load_hard_pool("h1_expanded_det")
    h3, _h3_ids, _h3_manifest = load_hard_pool("h3_union_core")
    excluded = {scenario.scenario_id for scenario in h0}
    excluded.update(str(row["scenario_id"]) for row in safe_reference["scenarios"])
    unseen = _rows(h1, "H1_UNSEEN", excluded, 12) + _rows(h3, "H3_UNSEEN", excluded, 12)
    rows = [
        {"scenario_id": scenario.scenario_id, "source": "H0", "scenario": scenario.to_dict()}
        for scenario in h0
    ]
    rows.extend(
        {
            "scenario_id": str(row["scenario_id"]),
            "source": "SAFE_REFERENCE",
            "selection_group": str(row["selection_group"]),
            "scenario": row["scenario"],
        }
        for row in safe_reference["scenarios"]
    )
    rows.extend(unseen)
    ids = [row["scenario_id"] for row in rows]
    if len(rows) != 96 or len(ids) != len(set(ids)):
        raise RuntimeError(f"P4 panel must contain 96 unique cases, got {len(rows)}/{len(set(ids))}")
    result = {
        "schema_version": 1,
        "record": "P4_FIXED_DIAGNOSTIC_PANEL",
        "status": "FROZEN_BEFORE_P4_ROLLOUT",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_head": preregistration["source"]["head"],
        "selection_rule": (
            "all 24 H0, all 48 SAFE_REFERENCE, then lexicographically first 12 unique non-H0/non-safe "
            "H1 cases and first 12 unique non-H0/non-safe/non-selected-H1 H3 cases"
        ),
        "counts": {"H0": 24, "SAFE_REFERENCE": 48, "H1_UNSEEN": 12, "H3_UNSEEN": 12},
        "frozen_hashes": frozen_hashes,
        "rows": rows,
    }
    write_json_atomic(EXPERIMENT_DIR / "P4_PANEL.json", result)
    print("P4_PANEL_FROZEN count=96", flush=True)


if __name__ == "__main__":
    main()

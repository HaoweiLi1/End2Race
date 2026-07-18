#!/usr/bin/env python3
"""Run the preregistered P0 training-600 reference scan and freeze SAFE_REFERENCE."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from audit_rl_direction_common import (
    EXPERIMENT_DIR,
    RUN_DIR,
    assert_frozen_contract,
    canonical_json_sha256,
    load_actor,
    make_env,
    read_json,
    run_deterministic_episode,
    save_npz,
    set_determinism,
    sha256_file,
    summarize_outcomes,
    write_json_atomic,
    FixedScenarioProvider,
)
from ppo.scenarios import load_hard_pool, training_scenarios


SEED = 20260717


def _select_stratified(
    candidates: list[dict[str, Any]],
    rng: np.random.Generator,
    outcome: str,
) -> list[dict[str, Any]]:
    eligible = sorted(
        (row for row in candidates if row["outcome"] == outcome),
        key=lambda row: (int(row["scenario"]["startpoint_ordinal"]), str(row["scenario_id"])),
    )
    if len(eligible) < 16:
        raise RuntimeError(f"Fewer than 16 safe {outcome} cases: {len(eligible)}")
    bins = np.array_split(np.asarray(eligible, dtype=object), 16)
    raceline_counts: Counter[str] = Counter()
    speed_counts: Counter[float] = Counter()
    selected: list[dict[str, Any]] = []
    for bin_index, raw_bin in enumerate(bins):
        choices = list(raw_bin)
        tie_break = {str(row["scenario_id"]): float(rng.random()) for row in choices}
        choice = min(
            choices,
            key=lambda row: (
                raceline_counts[str(row["scenario"]["opp_raceline"])],
                speed_counts[float(row["scenario"]["opp_speedscale"])],
                tie_break[str(row["scenario_id"])],
                str(row["scenario_id"]),
            ),
        )
        copied = dict(choice)
        copied["selection_group"] = f"safe_{outcome}"
        copied["selection_bin"] = bin_index
        selected.append(copied)
        raceline_counts[str(choice["scenario"]["opp_raceline"])] += 1
        speed_counts[float(choice["scenario"]["opp_speedscale"])] += 1
    return selected


def main() -> None:
    started = time.monotonic()
    frozen_hashes = assert_frozen_contract()
    preregistration = read_json(EXPERIMENT_DIR / "AUDIT_PREREGISTRATION.json")
    if preregistration["status"] != "PREREGISTERED_NOT_STARTED":
        raise RuntimeError(f"Unexpected preregistration status: {preregistration['status']}")
    set_determinism(SEED)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("P0 preregistration requires CUDA")
    actor = load_actor(device)
    provider = FixedScenarioProvider()
    env = make_env(provider, SEED)
    scenarios = training_scenarios()
    _h0_scenarios, h0_ids, _h0_manifest = load_hard_pool("h0_current_det")
    h0_set = set(h0_ids)

    scan_rows: list[dict[str, Any]] = []
    try:
        for index, scenario in enumerate(scenarios, start=1):
            row, _trace = run_deterministic_episode(
                env,
                provider,
                actor,
                scenario,
                device,
                seed=SEED,
                capture_trace=False,
            )
            scan_rows.append(row)
            if index % 25 == 0 or index == len(scenarios):
                print(
                    f"P0_REFERENCE {index}/{len(scenarios)} "
                    f"outcomes={json.dumps(summarize_outcomes(scan_rows), sort_keys=True)}",
                    flush=True,
                )
    finally:
        env.close()

    raw_record = {
        "schema_version": 1,
        "record": "P0_TRAINING_600_DETERMINISTIC_BC_REFERENCE_SCAN",
        "status": "COMPLETED",
        "seed": SEED,
        "device": "cuda",
        "source_head": preregistration["source"]["head"],
        "frozen_hashes": frozen_hashes,
        "scenario_count": len(scan_rows),
        "scenario_ids_unique": len({row["scenario_id"] for row in scan_rows}),
        "outcomes": summarize_outcomes(scan_rows),
        "rows": scan_rows,
    }
    raw_path = RUN_DIR / "p0" / "training_600_reference_scan.json"
    write_json_atomic(raw_path, raw_record)
    raw_sha256 = sha256_file(raw_path)

    candidates = [row for row in scan_rows if not row["ego_collision"] and row["scenario_id"] not in h0_set]
    rng = np.random.default_rng(SEED)
    selected_overtake = _select_stratified(candidates, rng, "overtake")
    selected_follow = _select_stratified(candidates, rng, "follow")
    selected_ids = {row["scenario_id"] for row in selected_overtake + selected_follow}
    remaining = sorted(
        (row for row in candidates if row["scenario_id"] not in selected_ids),
        key=lambda row: (float(row["min_oriented_clearance_m"]), str(row["scenario_id"])),
    )
    if len(remaining) < 16:
        raise RuntimeError(f"Fewer than 16 additional safe cases: {len(remaining)}")
    selected_clearance: list[dict[str, Any]] = []
    for rank, row in enumerate(remaining[:16]):
        copied = dict(row)
        copied["selection_group"] = "safe_minimum_clearance"
        copied["selection_rank"] = rank
        selected_clearance.append(copied)
    selected = selected_overtake + selected_follow + selected_clearance
    if len(selected) != 48 or len({row["scenario_id"] for row in selected}) != 48:
        raise RuntimeError("SAFE_REFERENCE selection did not produce 48 unique cases")
    selected_ids = {row["scenario_id"] for row in selected}

    safe_reference = {
        "schema_version": 1,
        "record": "END2RACE_PPO_RL_DIRECTION_SAFE_REFERENCE",
        "status": "FROZEN",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_head": preregistration["source"]["head"],
        "canonical_bc_sha256": preregistration["frozen_files"]["pretrained/end2race.pth"],
        "seed": SEED,
        "source_panel": "training_scenarios",
        "source_panel_count": 600,
        "excluded_h0_count": len(h0_ids),
        "reference_scan": {
            "path": "runs/ppo/RL_DIRECTION_AUDIT_20260717/p0/training_600_reference_scan.json",
            "sha256": raw_sha256,
            "outcomes": summarize_outcomes(scan_rows),
        },
        "selection_contract": preregistration["p0_reference_scan"]["safe_reference_selection"],
        "selected_count": 48,
        "selected_ids_unique": 48,
        "selected_outcomes": summarize_outcomes(selected),
        "selected_group_counts": dict(sorted(Counter(row["selection_group"] for row in selected).items())),
        "scenarios": selected,
    }
    safe_reference["selected_scenario_rows_sha256"] = canonical_json_sha256(
        [row["scenario"] for row in selected]
    )
    safe_path = EXPERIMENT_DIR / "SAFE_REFERENCE.json"
    write_json_atomic(safe_path, safe_reference)

    trace_scenarios = {scenario.scenario_id: scenario for scenario in scenarios if scenario.scenario_id in selected_ids}
    trace_scenarios.update({scenario.scenario_id: scenario for scenario in _h0_scenarios})
    trace_dir = RUN_DIR / "p0" / "probe_traces"
    trace_index: list[dict[str, Any]] = []
    provider = FixedScenarioProvider()
    env = make_env(provider, SEED)
    try:
        for index, scenario_id in enumerate(sorted(trace_scenarios), start=1):
            scenario = trace_scenarios[scenario_id]
            row, trace = run_deterministic_episode(
                env,
                provider,
                actor,
                scenario,
                device,
                seed=SEED,
                capture_trace=True,
            )
            if trace is None:
                raise RuntimeError("Trace capture unexpectedly returned None")
            relative = Path("p0") / "probe_traces" / f"{scenario_id}.npz"
            destination = RUN_DIR / relative
            save_npz(destination, trace)
            trace_index.append(
                {
                    "scenario_id": scenario_id,
                    "source": "H0" if scenario_id in h0_set else "SAFE_REFERENCE",
                    "outcome": row["outcome"],
                    "steps": row["steps"],
                    "path": str(Path("runs/ppo/RL_DIRECTION_AUDIT_20260717") / relative),
                    "sha256": sha256_file(destination),
                }
            )
            if index % 12 == 0 or index == len(trace_scenarios):
                print(f"P0_PROBE_TRACE {index}/{len(trace_scenarios)}", flush=True)
    finally:
        env.close()
    write_json_atomic(
        RUN_DIR / "p0" / "probe_trace_index.json",
        {
            "schema_version": 1,
            "count": len(trace_index),
            "rows": trace_index,
        },
    )
    print(
        f"P0_COMPLETE elapsed_seconds={time.monotonic() - started:.1f} "
        f"safe_reference_sha256={sha256_file(safe_path)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

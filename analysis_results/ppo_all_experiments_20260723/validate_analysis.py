#!/usr/bin/env python3
"""Independent QA checks for the 2026-07-23 End2Race PPO evaluation analysis.

This validator intentionally recomputes the highest-impact outcome and paired
counts from raw ``results_multi.json`` files instead of trusting the analyzer's
aggregate tables.  It also checks the saved collision-geometry predicates and
the full NPZ structural audit receipt.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EVAL_ROOT = REPO / "eval_results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    panels = pd.read_csv(OUT / "eval_panels.csv")
    episodes = pd.read_csv(OUT / "eval_episode_outcomes.csv")
    paired = pd.read_csv(OUT / "paired_vs_bc.csv")
    collision = pd.read_csv(OUT / "collision_episode_kinematics.csv")
    npz_audit = json.loads((OUT / "npz_audit.json").read_text())

    checks: list[dict] = []

    # 1. Recompute panel outcomes from the raw evaluator JSON files.
    raw_totals: dict[str, dict] = {}
    for row in panels.itertuples(index=False):
        result_path = EVAL_ROOT / row.eval_directory / "multiagents" / "results_multi.json"
        require(result_path.is_file(), f"missing raw result: {result_path}")
        raw = json.loads(result_path.read_text())
        raw_episodes = list(raw["episodes"].values())
        counts = Counter(item["outcome"] for item in raw_episodes)
        collision_count = sum(bool(item["ego_collision_occurred"]) for item in raw_episodes)
        scenario_count = len({item["scenario_id"] for item in raw_episodes})
        raw_totals[row.panel] = {
            "episodes": len(raw_episodes),
            "scenarios": scenario_count,
            "collisions": collision_count,
            "overtakes": counts["overtake"],
            "follows": counts["follow"],
        }
        require(len(raw_episodes) == int(row.episode_rows), f"episode count mismatch: {row.panel}")
        require(scenario_count == int(row.unique_scenarios), f"scenario count mismatch: {row.panel}")
        require(collision_count == int(row.collision_count), f"collision mismatch: {row.panel}")
        require(counts["overtake"] == int(row.overtake_count), f"overtake mismatch: {row.panel}")
        require(counts["follow"] == int(row.follow_count), f"follow mismatch: {row.panel}")
        require(collision_count + counts["overtake"] + counts["follow"] == len(raw_episodes),
                f"non-partitioning outcomes: {row.panel}")
    checks.append({"check": "raw_results_reconcile", "status": "pass", "panels": len(raw_totals)})

    # 2. Check analysis-table grain and panel validity boundary.
    require(not episodes.duplicated(["panel", "scenario_id"]).any(),
            "duplicate panel/scenario rows in eval_episode_outcomes.csv")
    require(len(episodes) == 55_799, "unexpected episode-analysis row count")
    valid_panels = panels[panels.valid.astype(bool)]
    invalid_panels = panels[~panels.valid.astype(bool)]
    require(len(valid_panels) == 92 and len(invalid_panels) == 1,
            "valid/invalid panel boundary changed")
    require(invalid_panels.iloc[0].panel == "ppo_privilege_gru_0722_lr5_tkloff_u0020",
            "unexpected invalid panel")
    require(int(invalid_panels.iloc[0].episode_rows) == 599, "invalid panel is not the known 599-row panel")
    checks.append({
        "check": "analysis_grain_and_validity",
        "status": "pass",
        "episode_rows": len(episodes),
        "valid_panels": len(valid_panels),
        "invalid_panels": len(invalid_panels),
    })

    # 3. Recompute all paired BC-vs-PPO set transitions independently.
    by_panel = {
        panel: frame.set_index("scenario_id")
        for panel, frame in episodes.groupby("panel", sort=False)
    }
    bc = by_panel["BC"]
    bc_collision = set(bc.index[bc.ego_collision.astype(bool)])
    bc_tail = set(bc.index[bc.merge_tail_primary.astype(bool)])
    for row in paired[paired.valid.astype(bool)].itertuples(index=False):
        current = by_panel[row.panel]
        ppo_collision = set(current.index[current.ego_collision.astype(bool)])
        ppo_tail = set(current.index[current.merge_tail_primary.astype(bool)])
        require(len(bc_collision - ppo_collision) == int(row.collision_resolved),
                f"paired collision_resolved mismatch: {row.panel}")
        require(len(bc_collision & ppo_collision) == int(row.collision_shared),
                f"paired collision_shared mismatch: {row.panel}")
        require(len(ppo_collision - bc_collision) == int(row.collision_created),
                f"paired collision_created mismatch: {row.panel}")
        require(len(bc_tail - ppo_tail) == int(row.tail_resolved),
                f"paired tail_resolved mismatch: {row.panel}")
        require(len(bc_tail & ppo_tail) == int(row.tail_shared),
                f"paired tail_shared mismatch: {row.panel}")
        require(len(ppo_tail - bc_tail) == int(row.tail_created),
                f"paired tail_created mismatch: {row.panel}")
    checks.append({
        "check": "paired_set_transitions",
        "status": "pass",
        "valid_ppo_panels": int(paired.valid.astype(bool).sum()),
        "bc_collision_scenarios": len(bc_collision),
        "bc_tail_scenarios": len(bc_tail),
    })

    # 4. Validate every primary mechanism label against its saved geometry predicates.
    primary = collision[collision.merge_tail_primary.astype(bool)]
    require(primary.opponent_collision.astype(bool).all(), "primary tail row without opponent collision")
    require(primary.pass_detected.astype(bool).all(), "primary tail row without detected pass")
    require((primary.pass_to_collision_s >= 0.10 - 1e-12).all(), "primary tail lead-time violation")
    require((primary.post_pass_lateral_convergence_m >= 0.10 - 1e-12).all(),
            "primary tail lateral-convergence violation")
    require((primary.terminal_opponent_body_x_m < 0.0).all(), "primary tail opponent not behind ego")
    require((primary.collision_class == "post_overtake_merge_rear_sweep").all(),
            "primary tail class-label mismatch")
    bc_primary = primary[primary.panel == "BC"]
    require(len(bc_primary) == 11, "BC primary mechanism count changed")
    require(np.isclose(bc_primary.pass_to_collision_s.median(), 0.39, atol=0.005),
            "BC median pass-to-collision drifted")
    require(np.isclose(bc_primary.post_pass_lateral_convergence_m.median(), 0.3130866825, atol=1e-8),
            "BC median lateral convergence drifted")
    checks.append({
        "check": "mechanism_geometry_predicates",
        "status": "pass",
        "primary_events_all_panels": len(primary),
        "bc_primary_events": len(bc_primary),
        "bc_median_pass_to_collision_s": float(bc_primary.pass_to_collision_s.median()),
        "bc_median_lateral_convergence_m": float(bc_primary.post_pass_lateral_convergence_m.median()),
    })

    # 5. Reconcile collision-class partitions after policy deduplication.
    valid_ppo = episodes[(episodes.panel_valid.astype(bool)) & (episodes.run != "BC")]
    dedup = valid_ppo.drop_duplicates(["policy_sha256", "scenario_id"])
    dedup_collision = dedup[dedup.ego_collision.astype(bool)]
    class_counts = dedup_collision.collision_class.value_counts().to_dict()
    require(len(dedup.policy_sha256.unique()) == 86, "unexpected unique PPO actor-policy count")
    require(len(dedup_collision) == 1_785, "unexpected deduplicated collision-event count")
    require(sum(class_counts.values()) == len(dedup_collision), "collision classes do not partition events")
    require(class_counts.get("post_overtake_merge_rear_sweep") == 778,
            "unexpected primary tail event count")
    checks.append({
        "check": "unique_policy_collision_partition",
        "status": "pass",
        "unique_policies": 86,
        "collision_events": len(dedup_collision),
        "class_counts": class_counts,
    })

    # 6. Consume the analyzer's full NPZ audit, checking the expected format boundary.
    totals = npz_audit["totals"]
    require(totals["files"] == 55_799, "NPZ audit file total mismatch")
    require(totals["numeric_True"] == 55_799, "non-numeric NPZ payload")
    require(totals["aligned_True"] == 55_799, "misaligned NPZ arrays")
    require(npz_audit["by_format"]["post_step_v2"]["terminal_valid_True"] == 34_799,
            "post-step terminal-marker count mismatch")
    require(npz_audit["by_format"]["legacy_pre_post_step"]["collision_marker_matches_False"] == 736,
            "legacy terminal-omission boundary changed")
    require(not npz_audit["structural_panel_issues"], "NPZ structural panel issues present")
    checks.append({
        "check": "full_npz_audit_receipt",
        "status": "pass",
        "files": totals["files"],
        "post_step_v2": npz_audit["by_format"]["post_step_v2"]["files"],
        "legacy_pre_post_step": npz_audit["by_format"]["legacy_pre_post_step"]["files"],
        "legacy_expected_terminal_marker_mismatches": 736,
    })

    receipt = {
        "assessment": "share_with_caveats",
        "as_of": "2026-07-23 Asia/Singapore",
        "checks": checks,
        "required_caveats": [
            "All experiments use one recorded training/evaluation realization; run_config.json does not persist seed or source commit.",
            "Checkpoint selection and 91 panel-wise comparisons create multiple-testing/selection risk; raw paired p-values do not generalize beyond Austin600.",
            "The evaluator saves pose-derived kinematics, not simulator tire slip angle; the analysis establishes rear-sweep merge geometry, not literal tire-side-slip causality.",
            "The 599-row lr5 update-20 panel is excluded from ranking and inference.",
        ],
    }
    json_dump(OUT / "validation_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

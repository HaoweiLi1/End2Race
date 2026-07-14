#!/usr/bin/env python3
"""Post-hoc B7 iter6 evaluation on the opened Austin 600-case grid.

The owner explicitly overrode the original iter10-only rule after B7 had
closed.  This wrapper keeps the original B4 simulator/evaluator byte-for-byte,
authorizes the immutable B7 training RunPlan, and merges one preselected iter6
actor against the immutable B4 canonical-BC rows.  The result is diagnostic
opened-development evidence and cannot retroactively promote B7.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from bplus_v22.b7_eval import exact_cluster_signflip_one_sided
from scripts import b4_product_eval as core
from scripts.b5_opened_product_eval import (
    B4_BASELINE_PLAN_SHA256,
    B4_BASELINE_SOURCE_COMMIT,
    CANONICAL_BC_SHA256,
)


SUMMARY_SCHEMA = "end2race-b7-posthoc-opened-product-eval-1"
VARIANT = "seed1_iter6_posthoc"


def _load_b7_training_plan(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != "end2race-b7-run-plan-1"
        or value.get("kind") != "b7_plain_recurrent_train"
        or value.get("primary_seed") != 1
    ):
        raise ValueError("B7 post-hoc evaluation requires the seed1 B7 RunPlan")
    observed = value.get("plan_sha256")
    unsigned = dict(value)
    unsigned.pop("plan_sha256", None)
    expected = hashlib.sha256(
        (json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    source_commit = value.get("source_commit")
    if (
        observed != expected
        or not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("B7 post-hoc RunPlan identity drift")
    return value


def run_shard(args: argparse.Namespace) -> int:
    core._load_training_plan = _load_b7_training_plan
    return core.run_shard(args)


def _counts(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    return {
        "episodes": len(rows),
        "collision": sum(row["outcome"] == "collision" for row in rows.values()),
        "overtake": sum(row["outcome"] == "overtaking" for row in rows.values()),
        "follow": sum(row["outcome"] == "following" for row in rows.values()),
        "ego_collision": sum(bool(row["ego_collision"]) for row in rows.values()),
        "opp_collision": sum(bool(row["opp_collision"]) for row in rows.values()),
        "speed_projection": sum(
            int(row["deterministic_speed_projection_count"]) for row in rows.values()
        ),
    }


def _mcnemar_two_sided(positive: int, negative: int) -> float:
    trials = positive + negative
    if trials == 0:
        return 1.0
    lower = min(positive, negative)
    probability = math.fsum(math.comb(trials, value) for value in range(lower + 1)) / (
        2**trials
    )
    return min(1.0, 2.0 * probability)


def merge(args: argparse.Namespace) -> int:
    if args.variant != VARIANT:
        raise ValueError(f"B7 post-hoc variant must be {VARIANT}")
    plan = _load_b7_training_plan(args.training_plan)
    baseline_rows, baseline_provenance = core._load_variant(
        Path(args.baseline_root).resolve(), "BC"
    )
    candidate_rows, candidate_provenance = core._load_variant(
        Path(args.root).resolve(), VARIANT
    )
    if baseline_provenance != (B4_BASELINE_PLAN_SHA256, B4_BASELINE_SOURCE_COMMIT):
        raise ValueError("B7 post-hoc immutable B4 baseline provenance drift")
    if {str(row["model_sha256"]) for row in baseline_rows.values()} != {
        CANONICAL_BC_SHA256
    }:
        raise ValueError("B7 post-hoc canonical BC checkpoint drift")
    if candidate_provenance != (plan["plan_sha256"], plan["source_commit"]):
        raise ValueError("B7 post-hoc candidate training provenance drift")
    keys = set(baseline_rows)
    if len(keys) != core.TOTAL_CASES or set(candidate_rows) != keys:
        raise ValueError("B7 post-hoc paired case inventory drift")

    baseline = _counts(baseline_rows)
    candidate = _counts(candidate_rows)
    if baseline["collision"] != 24 or baseline["overtake"] != 342:
        raise ValueError("B7 post-hoc immutable baseline count drift")
    fixed = new = gained = lost = 0
    startpoint_effect = {value: 0 for value in range(core.STARTPOINT_COUNT)}
    for key in keys:
        before = baseline_rows[key]
        after = candidate_rows[key]
        fixed_case = before["outcome"] == "collision" and after["outcome"] != "collision"
        new_case = before["outcome"] != "collision" and after["outcome"] == "collision"
        fixed += fixed_case
        new += new_case
        gained += before["outcome"] != "overtaking" and after["outcome"] == "overtaking"
        lost += before["outcome"] == "overtaking" and after["outcome"] != "overtaking"
        startpoint_effect[int(before["startpoint_ordinal"])] += int(fixed_case) - int(new_case)
    overtake_floor = math.ceil(0.95 * baseline["overtake"])
    feasible = (
        candidate["speed_projection"] == 0
        and candidate["overtake"] >= overtake_floor
        and candidate["collision"] < baseline["collision"]
        and fixed > new
    )
    candidate_summary = {
        **candidate,
        "fixed_collision": fixed,
        "new_collision": new,
        "gained_overtake": gained,
        "lost_overtake": lost,
        "collision_net_improvement": fixed - new,
        "collision_mcnemar_two_sided_p": _mcnemar_two_sided(fixed, new),
        "startpoint_cluster_signflip_one_sided_p": exact_cluster_signflip_one_sided(
            list(startpoint_effect.values())
        ),
        "overtake_floor": overtake_floor,
        "overtake_guardrail_pass": candidate["overtake"] >= overtake_floor,
        "collision_strict_improve": candidate["collision"] < baseline["collision"],
        "fixed_gt_new": fixed > new,
        "original_b4_feasibility_pass": feasible,
        "opened_development_target_collision_le_16": feasible
        and candidate["collision"] <= 16,
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "integrity_passed": True,
        "panel_status": "opened-development regression panel",
        "owner_authorized_posthoc_override": True,
        "retroactive_b7_promotion_allowed": False,
        "selection_rule": "last accepted actor iter6 fixed before evaluation",
        "grid": {
            "map_name": core.MAP_NAME,
            "opponent_racelines": list(core.OPP_RACELINES),
            "opponent_speed_scales": list(core.OPP_SPEED_SCALES),
            "startpoint_count": core.STARTPOINT_COUNT,
            "episode_count_per_variant": core.TOTAL_CASES,
        },
        "training_run_plan_sha256": plan["plan_sha256"],
        "training_source_commit": plan["source_commit"],
        "bc": baseline,
        "candidate": candidate_summary,
        "verdict": (
            "POSTHOC_OPENED_DEVELOPMENT_FEASIBLE"
            if feasible
            else "POSTHOC_OPENED_DEVELOPMENT_NEGATIVE"
        ),
    }

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    core._write_json(output / "summary.json", summary)
    fields = (
        "case_id",
        "startpoint_ordinal",
        "ego_idx",
        "opp_raceline",
        "opp_speedscale",
        "variant",
        "outcome",
        "ego_collision",
        "opp_collision",
        "deterministic_speed_projection_count",
        "model_sha256",
    )
    with (output / "paired_rows.tsv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in sorted(keys):
            for variant, rows in (("BC", baseline_rows), (VARIANT, candidate_rows)):
                writer.writerow(
                    {name: rows[key].get(name, "") for name in fields} | {"variant": variant}
                )
    report = f"""# B7 iter6 post-hoc opened-development evaluation

This owner-authorized evaluation is diagnostic and cannot retroactively promote B7.

- Grid: 3 racelines x 4 speeds x 50 startpoints = {core.TOTAL_CASES}
- BC: collision={baseline['collision']}, overtake={baseline['overtake']}, follow={baseline['follow']}
- iter6: collision={candidate['collision']}, overtake={candidate['overtake']}, follow={candidate['follow']}
- fixed/new collision: {fixed}/{new}
- gained/lost overtake: {gained}/{lost}
- collision McNemar two-sided p: {candidate_summary['collision_mcnemar_two_sided_p']:.6f}
- startpoint-cluster one-sided p: {candidate_summary['startpoint_cluster_signflip_one_sided_p']:.6f}
- 95% overtake floor: {overtake_floor}
- Verdict: **{summary['verdict']}**
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "COMPLETE").write_text(
        hashlib.sha256((output / "summary.json").read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--repo", default=".")
    run.add_argument("--model-path", required=True)
    run.add_argument("--variant", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--device", default="cpu")
    run.add_argument("--training-plan", required=True)
    run.add_argument("--producer-host-id", required=True)
    run.add_argument("--gpu-uuid", required=True)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--root", required=True)
    merge_parser.add_argument("--baseline-root", required=True)
    merge_parser.add_argument("--training-plan", required=True)
    merge_parser.add_argument("--variant", required=True)
    merge_parser.add_argument("--output", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return run_shard(args) if args.action == "run" else merge(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run B5 snapshots through the unchanged B4 600-case product evaluator.

The Austin grid is an opened-development regression panel.  Candidate shards
are produced by the exact B4 evaluator implementation; this wrapper only
authorizes a ``b5_train`` RunPlan and merges those rows against the immutable
B4 canonical-BC rows without pretending the panel is fresh confirmation.
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

from scripts import b4_product_eval as core


SUMMARY_SCHEMA = "end2race-b5-opened-development-product-eval-1"
B4_BASELINE_PLAN_SHA256 = (
    "08f0fe4275ae60928a6d5a6ce9704679bc91a624258bf5aef7f7a268b2c5e381"
)
B4_BASELINE_SOURCE_COMMIT = "9e5afdc9584343a163c4704597dad87487bd750a"
CANONICAL_BC_SHA256 = (
    "b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4"
)


def _load_b5_training_plan(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != "end2race-b2-run-plan-1"
        or value.get("kind") != "b5_train"
    ):
        raise ValueError("B5 opened-panel evaluation requires a b5_train RunPlan")
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
        raise ValueError("B5 opened-panel RunPlan identity drift")
    return value


def run_shard(args: argparse.Namespace) -> int:
    # Preserve the production evaluator byte-for-byte.  Only its RunPlan-kind
    # authorization is replaced for the new versioned training provenance.
    core._load_training_plan = _load_b5_training_plan
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
            int(row["deterministic_speed_projection_count"])
            for row in rows.values()
        ),
    }


def _iteration(name: str) -> int:
    try:
        return int(name.rsplit("iter", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def merge(args: argparse.Namespace) -> int:
    baseline_root = Path(args.baseline_root).resolve()
    candidate_root = Path(args.root).resolve()
    variants = tuple(args.variant)
    if variants != ("seed1_iter10", "seed1_iter20", "seed1_iter30"):
        raise ValueError("B5 merge requires the prospective iter10/20/30 inventory")
    baseline_rows, baseline_provenance = core._load_variant(baseline_root, "BC")
    loaded = {variant: core._load_variant(candidate_root, variant) for variant in variants}
    candidate_provenances = {value[1] for value in loaded.values()}
    if baseline_provenance != (B4_BASELINE_PLAN_SHA256, B4_BASELINE_SOURCE_COMMIT):
        raise ValueError("B5 immutable B4 baseline provenance drift")
    baseline_model_shas = {str(row["model_sha256"]) for row in baseline_rows.values()}
    if baseline_model_shas != {CANONICAL_BC_SHA256}:
        raise ValueError("B5 immutable canonical-BC checkpoint drift")
    if len(candidate_provenances) != 1:
        raise ValueError("B5 candidate training provenance drift")
    training_plan_sha256, training_source_commit = next(iter(candidate_provenances))
    training_plan = _load_b5_training_plan(args.training_plan)
    if (
        training_plan_sha256 != training_plan["plan_sha256"]
        or training_source_commit != training_plan["source_commit"]
    ):
        raise ValueError("B5 candidate shards do not match the supplied RunPlan")

    by_variant = {variant: value[0] for variant, value in loaded.items()}
    keys = set(baseline_rows)
    if len(keys) != core.TOTAL_CASES or any(set(rows) != keys for rows in by_variant.values()):
        raise ValueError("B5 opened-panel paired case inventory drift")
    baseline = _counts(baseline_rows)
    if baseline["collision"] != 24 or baseline["overtake"] != 342:
        raise ValueError("B5 opened-panel canonical-BC counts drift")
    overtake_floor = math.ceil(0.95 * baseline["overtake"])
    if overtake_floor != 325:
        raise AssertionError("B5 opened-panel overtake floor drift")

    candidates: dict[str, dict[str, Any]] = {}
    for variant in variants:
        rows = by_variant[variant]
        row_counts = _counts(rows)
        fixed = new = gained = lost = 0
        for key in keys:
            before = baseline_rows[key]["outcome"]
            after = rows[key]["outcome"]
            fixed += before == "collision" and after != "collision"
            new += before != "collision" and after == "collision"
            gained += before != "overtaking" and after == "overtaking"
            lost += before == "overtaking" and after != "overtaking"
        feasible = (
            row_counts["speed_projection"] == 0
            and row_counts["overtake"] >= overtake_floor
            and row_counts["collision"] < baseline["collision"]
            and fixed > new
        )
        candidates[variant] = {
            **row_counts,
            "fixed_collision": fixed,
            "new_collision": new,
            "gained_overtake": gained,
            "lost_overtake": lost,
            "overtake_floor": overtake_floor,
            "overtake_guardrail_pass": row_counts["overtake"] >= overtake_floor,
            "collision_strict_improve": row_counts["collision"] < baseline["collision"],
            "fixed_gt_new": fixed > new,
            "feasible": feasible,
            "opened_development_target_hit": feasible and row_counts["collision"] <= 16,
        }
    feasible = [name for name, value in candidates.items() if value["feasible"]]
    selected = (
        min(
            feasible,
            key=lambda name: (
                candidates[name]["collision"],
                -candidates[name]["overtake"],
                _iteration(name),
            ),
        )
        if feasible
        else None
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "integrity_passed": True,
        "panel_status": "opened-development regression panel",
        "fresh_or_final_confirmation": False,
        "grid": {
            "map_name": core.MAP_NAME,
            "opponent_racelines": list(core.OPP_RACELINES),
            "opponent_speed_scales": list(core.OPP_SPEED_SCALES),
            "startpoint_count": core.STARTPOINT_COUNT,
            "episode_count_per_variant": core.TOTAL_CASES,
        },
        "outcome_contract": "unchanged original eval_multiagent terminal state_label",
        "baseline_reused_from_b4": True,
        "baseline_run_plan_sha256": B4_BASELINE_PLAN_SHA256,
        "training_run_plan_sha256": training_plan_sha256,
        "training_source_commit": training_source_commit,
        "bc": baseline,
        "overtake_floor_95pct": overtake_floor,
        "candidates": candidates,
        "selected_variant": selected,
        "verdict": (
            "OPENED_DEVELOPMENT_SURVIVOR"
            if selected is not None
            else "B5_A_SUBSTANTIVE_NEGATIVE"
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
            for variant, rows in (("BC", baseline_rows), *by_variant.items()):
                writer.writerow(
                    {name: rows[key].get(name, "") for name in fields}
                    | {"variant": variant}
                )
    report = [
        "# B5-A opened-development regression evaluation",
        "",
        "This is not fresh or final confirmation.",
        "",
        f"- Grid: 3 racelines x 4 speeds x 50 startpoints = {core.TOTAL_CASES}",
        f"- BC (reused immutable B4 rows): collision={baseline['collision']}, overtake={baseline['overtake']}, follow={baseline['follow']}",
        f"- 95% overtake floor: {overtake_floor}",
        "",
        "| variant | collision | overtake | follow | fixed | new | gained | lost | speed projection | feasible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant, value in candidates.items():
        report.append(
            f"| {variant} | {value['collision']} | {value['overtake']} | {value['follow']} | "
            f"{value['fixed_collision']} | {value['new_collision']} | "
            f"{value['gained_overtake']} | {value['lost_overtake']} | "
            f"{value['speed_projection']} | {value['feasible']} |"
        )
    report.extend(["", f"Selected: `{selected}`", f"Verdict: **{summary['verdict']}**", ""])
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
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
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--training-plan", required=True)
    run.add_argument("--producer-host-id", required=True)
    run.add_argument("--gpu-uuid", required=True)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--root", required=True)
    merge_parser.add_argument("--baseline-root", required=True)
    merge_parser.add_argument("--training-plan", required=True)
    merge_parser.add_argument("--variant", action="append", required=True)
    merge_parser.add_argument("--output", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return run_shard(args) if args.action == "run" else merge(args)


if __name__ == "__main__":
    sys.exit(main())

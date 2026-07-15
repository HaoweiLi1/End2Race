#!/usr/bin/env python3
"""Execute and merge the four B7 opened-development 288 shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from bplus_v22.b7_eval import (
    B7_EVAL_SHARD_SCHEMA,
    B7EvaluationShard,
    evaluate_candidate_shard,
    merge_candidate_shards,
)
from scripts.run_b7_recurrent import load_plan


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _development_rows(task8: Path):
    with (task8 / "development_scenarios.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run_shard(args: argparse.Namespace) -> int:
    plan, paths = load_plan(args.plan, args.repo)
    if Path(args.baseline_rows).resolve() != paths["baseline"]:
        raise ValueError("B7 evaluation baseline path differs from the RunPlan")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    shard = evaluate_candidate_shard(
        task8_rows=_development_rows(paths["task8"]),
        baseline_rows_path=args.baseline_rows,
        candidate_path=args.candidate,
        training_run_plan_sha256=plan["plan_sha256"],
        device=device,
        shard_index=args.shard_index,
    )
    _atomic_json(output, shard.as_dict())
    print(json.dumps({"shard_index": args.shard_index, "rows": len(shard.rows)}, sort_keys=True))
    return 0


def _load_shard(path: str | Path) -> B7EvaluationShard:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.pop("schema", None) != B7_EVAL_SHARD_SCHEMA:
        raise ValueError("B7 evaluation shard schema drift")
    value["rows"] = tuple(value["rows"])
    return B7EvaluationShard(**value)


def merge(args: argparse.Namespace) -> int:
    plan, paths = load_plan(args.plan, args.repo)
    if Path(args.baseline_rows).resolve() != paths["baseline"]:
        raise ValueError("B7 evaluation baseline path differs from the RunPlan")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows, summary = merge_candidate_shards(
        shards=[_load_shard(path) for path in args.shard],
        task8_rows=_development_rows(paths["task8"]),
        baseline_rows_path=args.baseline_rows,
        candidate_path=args.candidate,
        training_run_plan_sha256=plan["plan_sha256"],
    )
    output.mkdir(parents=True)
    _atomic_json(output / "summary.json", summary)
    fields = tuple(rows[0])
    with (output / "paired_candidate_rows.tsv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# B7 seed1 opened-development 288 result",
        "",
        "This is opened-development evidence, not fresh/final confirmation.",
        "",
        f"- BC: collision={summary['bc']['collision']}, overtake={summary['bc']['terminal_overtake']}",
        f"- candidate: collision={summary['candidate']['collision']}, overtake={summary['candidate']['terminal_overtake']}",
        f"- fixed/new: {summary['candidate']['fixed_collision']}/{summary['candidate']['new_collision']}",
        f"- L4 cluster one-sided sign-flip p: {summary['candidate']['l4_cluster_signflip_one_sided_p']:.6f}",
        f"- verdict: **{summary['verdict']}**",
        "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    digest = hashlib.sha256((output / "summary.json").read_bytes()).hexdigest()
    (output / "COMPLETE").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="action", required=True)
    shard = sub.add_parser("shard")
    shard.add_argument("--repo", default=".")
    shard.add_argument("--plan", required=True)
    shard.add_argument("--candidate", required=True)
    shard.add_argument("--baseline-rows", required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--device", default="cuda:0")
    shard.add_argument("--output", required=True)
    shard.set_defaults(func=run_shard)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--repo", default=".")
    merge_parser.add_argument("--plan", required=True)
    merge_parser.add_argument("--candidate", required=True)
    merge_parser.add_argument("--baseline-rows", required=True)
    merge_parser.add_argument("--shard", action="append", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.set_defaults(func=merge)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Exact 3x4x50 grid, five-way sharding, and paired merge regression."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from scripts.b4_product_eval import (
    CASES_PER_SHARD,
    SCHEMA,
    SHARD_COUNT,
    TOTAL_CASES,
    _sha256,
    _load_training_plan,
    _write_json,
    enumerate_cases,
    merge,
)


REPO = Path(__file__).resolve().parent.parent


def outcome(variant: str, index: int) -> str:
    if variant == "BC":
        if index < 60:
            return "collision"
        if index < 360:
            return "overtaking"
        return "following"
    if index < 40:
        return "collision"
    if index < 330:
        return "overtaking"
    return "following"


def write_variant(root: Path, variant: str, cases) -> None:
    model_sha = hashlib.sha256(variant.encode()).hexdigest()
    indexed = {case["case_id"]: index for index, case in enumerate(cases)}
    for shard_index in range(SHARD_COUNT):
        shard = root / variant / f"shard{shard_index}"
        selected = [case for case in cases if case["shard_index"] == shard_index]
        assert len(selected) == CASES_PER_SHARD
        manifest = {
            "schema": SCHEMA,
            "variant": variant,
            "model_sha256": model_sha,
            "shard_index": shard_index,
            "case_count": len(selected),
            "cases": selected,
        }
        _write_json(shard / "manifest.json", manifest)
        counts = {"collision": 0, "following": 0, "overtaking": 0}
        for case in selected:
            case_outcome = outcome(variant, indexed[case["case_id"]])
            counts[case_outcome] += 1
            npz = shard / "npz" / f"{case['case_id']}.npz"
            npz.parent.mkdir(parents=True, exist_ok=True)
            npz.write_bytes(case["case_id"].encode())
            metric = {
                **case,
                "schema": SCHEMA,
                "variant": variant,
                "model_sha256": model_sha,
                "outcome": case_outcome,
                "ego_collision": case_outcome == "collision",
                "opp_collision": False,
                "deterministic_speed_projection_count": 0,
                "npz_path": str(npz),
                "npz_relpath": f"npz/{case['case_id']}.npz",
                "npz_sha256": _sha256(npz),
            }
            _write_json(shard / "metrics" / f"{case['case_id']}.json", metric)
        _write_json(
            shard / "summary.json",
            {
                "schema": SCHEMA,
                "passed": True,
                "variant": variant,
                "model_sha256": model_sha,
                "shard_index": shard_index,
                "case_count": len(selected),
                "counts": counts,
            },
        )
        (shard / "COMPLETE").write_text("test\n", encoding="utf-8")


def main() -> None:
    cases = enumerate_cases(REPO)
    assert len(cases) == TOTAL_CASES
    assert {case["shard_index"] for case in cases} == set(range(SHARD_COUNT))
    assert all(
        sum(case["shard_index"] == shard for case in cases) == CASES_PER_SHARD
        for shard in range(SHARD_COUNT)
    )
    assert len({case["startpoint_ordinal"] for case in cases}) == 50
    assert len({case["opp_raceline"] for case in cases}) == 3
    assert len({case["opp_speedscale"] for case in cases}) == 4

    with tempfile.TemporaryDirectory() as plan_directory:
        plan_path = Path(plan_directory) / "run_plan.json"
        unsigned = {
            "schema": "end2race-b2-run-plan-1",
            "kind": "b4_train",
            "source_commit": "1" * 40,
        }
        signed = {
            **unsigned,
            "plan_sha256": hashlib.sha256(
                (json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest(),
        }
        plan_path.write_text(json.dumps(signed), encoding="utf-8")
        assert _load_training_plan(plan_path) == signed
        signed["source_commit"] = "2" * 40
        plan_path.write_text(json.dumps(signed), encoding="utf-8")
        try:
            _load_training_plan(plan_path)
            raise RuntimeError("B4 product evaluator accepted a drifted RunPlan")
        except ValueError as error:
            assert "identity" in str(error)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_variant(root, "BC", cases)
        write_variant(root, "seed1_iter10", cases)
        output = root / "merged"
        assert (
            merge(
                argparse.Namespace(
                    root=str(root),
                    variant=["BC", "seed1_iter10"],
                    output=str(output),
                )
            )
            == 0
        )
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["grid"]["episode_count_per_variant"] == 600
        assert summary["bc"] == {
            "episodes": 600,
            "collision": 60,
            "overtake": 300,
            "follow": 240,
            "ego_collision": 60,
            "opp_collision": 0,
            "speed_projection": 0,
        }
        assert summary["overtake_floor_95pct"] == 285
        assert summary["selected_variant"] == "seed1_iter10"
        assert summary["candidates"]["seed1_iter10"]["collision"] == 40
        assert summary["candidates"]["seed1_iter10"]["overtake"] == 290
        assert summary["candidates"]["seed1_iter10"]["fixed_collision"] == 20
        assert summary["candidates"]["seed1_iter10"]["new_collision"] == 0
        assert sum(1 for _ in (output / "paired_rows.tsv").open(encoding="utf-8")) == 1201

    print("B4 3x4x50 product evaluation contracts passed")


if __name__ == "__main__":
    main()

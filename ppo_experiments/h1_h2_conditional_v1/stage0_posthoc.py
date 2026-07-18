#!/usr/bin/env python3
"""Record starting hashes and run the registered A1 U2 full-600 post-hoc once."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
BASELINE = ROOT / "ppo_experiments" / "quick_pool_3s_v2" / "FULL600_BC_RESULTS.json"
MANIFEST = (
    ROOT
    / "runs"
    / "ppo"
    / "quick_pool_3s_v2"
    / "QP3_A1_H1FULL_8S_seed20260718"
    / "checkpoint_manifest.json"
)
EVALUATOR = ROOT / "ppo_experiments" / "quick_pool_3s" / "evaluate_pool.py"
RAW_OUTPUT = OUTPUT / "posthoc_a1_u2_full600_raw.json"
FINAL_OUTPUT = OUTPUT / "POSTHOC_A1_U2_FULL600.json"
SOURCE_PATHS = (
    "ppo/config.py",
    "ppo/policy.py",
    "ppo/environment.py",
    "ppo/reward.py",
    "ppo/scenarios.py",
    "train_ppo.py",
    "eval_multiagent.py",
    "evaluate.sh",
    "utils.py",
    "pretrained/end2race.pth",
    "ppo_experiments/quick_pool_3s_v2/FULL600_BC_RESULTS.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, document: Any) -> None:
    from utils import atomic_write_json

    atomic_write_json(path, document)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_evaluation(document: dict[str, Any], expected_total: int = 600) -> None:
    rows = document.get("rows", [])
    ids = [str(row.get("scenario_id")) for row in rows]
    summary = document.get("summary", {})
    if (
        not document.get("complete")
        or len(rows) != expected_total
        or len(set(ids)) != expected_total
        or summary.get("total") != expected_total
        or summary.get("error") != 0
    ):
        raise RuntimeError(f"Evaluation coverage failure: rows={len(rows)}, unique={len(set(ids))}, summary={summary}")
    if sum(int(summary.get(key, 0)) for key in ("collision", "follow", "overtake")) != expected_total:
        raise RuntimeError(f"Evaluation outcome sum failure: {summary}")
    nonfinite = [
        scenario_id
        for scenario_id, row in zip(ids, rows)
        if not bool(row.get("observation_finite")) or not bool(row.get("action_finite"))
    ]
    if nonfinite:
        raise RuntimeError(f"Non-finite evaluation rows: {nonfinite[:5]}")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval total must be positive")
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return [center - margin, center + margin]


def paired_metrics(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_by_id = {str(row["scenario_id"]): str(row["outcome"]) for row in baseline["rows"]}
    candidate_by_id = {str(row["scenario_id"]): str(row["outcome"]) for row in candidate["rows"]}
    if set(baseline_by_id) != set(candidate_by_id):
        raise RuntimeError("Paired evaluation scenario IDs differ")
    fixed = sum(
        baseline_by_id[key] == "ego_collision" and candidate_by_id[key] != "ego_collision"
        for key in baseline_by_id
    )
    new = sum(
        baseline_by_id[key] != "ego_collision" and candidate_by_id[key] == "ego_collision"
        for key in baseline_by_id
    )
    gained = sum(
        baseline_by_id[key] != "overtake" and candidate_by_id[key] == "overtake"
        for key in baseline_by_id
    )
    lost = sum(
        baseline_by_id[key] == "overtake" and candidate_by_id[key] != "overtake"
        for key in baseline_by_id
    )
    budgets = {
        "new_budget_for_21": max(fixed - 1, 0),
        "new_budget_for_18": max(fixed - 4, 0),
        "new_budget_for_16": max(fixed - 6, 0),
    }
    return {
        "fixed_collision": fixed,
        "new_collision": new,
        "gained_overtake": gained,
        "lost_overtake": lost,
        "delta_collision": new - fixed,
        "repair_rate": fixed / 22.0,
        "repair_rate_wilson_95": wilson_interval(fixed, 22),
        "damage_rate": new / 578.0,
        "damage_rate_wilson_95": wilson_interval(new, 578),
        **budgets,
        "excess_new_for_21": max(new - budgets["new_budget_for_21"], 0),
        "excess_new_for_18": max(new - budgets["new_budget_for_18"], 0),
        "excess_new_for_16": max(new - budgets["new_budget_for_16"], 0),
    }


def resolve_checkpoint() -> tuple[Path, dict[str, Any]]:
    document = read_json(MANIFEST)
    matches = [row for row in document.get("checkpoints", []) if int(row.get("update", -1)) == 2]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one update-2 checkpoint in {MANIFEST}: {matches}")
    record = matches[0]
    checkpoint = MANIFEST.parent / str(record["path"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != str(record["sha256"]):
        raise RuntimeError("A1 update-2 checkpoint is missing or its manifest hash does not match")
    return checkpoint, record


def record_starting_hashes(checkpoint: Path, checkpoint_record: dict[str, Any]) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != "eb2ecef661e63dcf0a12fb7e7a9ffa8caa782ce3":
        raise RuntimeError(f"Starting reference mismatch: {head}")
    document = {
        "schema_version": 1,
        "recorded_at_utc": utc_now(),
        "git_head": head,
        "files": {relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS},
        "posthoc_checkpoint_manifest": str(MANIFEST.relative_to(ROOT)),
        "posthoc_checkpoint_manifest_sha256": sha256_file(MANIFEST),
        "posthoc_checkpoint": str(checkpoint.relative_to(ROOT)),
        "posthoc_checkpoint_sha256": sha256_file(checkpoint),
        "posthoc_checkpoint_manifest_record": checkpoint_record,
    }
    write_json(OUTPUT / "SOURCE_HASHES.json", document)


def run_evaluation_once(checkpoint: Path) -> dict[str, Any]:
    if RAW_OUTPUT.exists() or FINAL_OUTPUT.exists():
        raise RuntimeError("Post-hoc output already exists; refusing to evaluate A1 more than once")
    command = [
        sys.executable,
        "-u",
        str(EVALUATOR),
        "--model-path",
        str(checkpoint),
        "--output",
        str(RAW_OUTPUT),
        "--workers",
        "8",
        "--sim-duration",
        "8.0",
    ]
    command_record = {
        "schema_version": 1,
        "commands": [
            {
                "purpose": "Evaluate existing A1 H1-full update-2 checkpoint once on current CPU full-600",
                "argv": command,
                "started_at_utc": utc_now(),
            }
        ],
    }
    write_json(OUTPUT / "COMMANDS.json", command_record)
    subprocess.run(command, cwd=ROOT, check=True)
    command_record["commands"][0]["completed_at_utc"] = utc_now()
    command_record["commands"][0]["return_code"] = 0
    write_json(OUTPUT / "COMMANDS.json", command_record)
    return read_json(RAW_OUTPUT)


def main() -> int:
    sys.path.insert(0, str(ROOT))
    baseline = read_json(BASELINE)
    validate_evaluation(baseline)
    checkpoint, checkpoint_record = resolve_checkpoint()
    record_starting_hashes(checkpoint, checkpoint_record)
    candidate = run_evaluation_once(checkpoint)
    validate_evaluation(candidate)
    result = {
        "schema_version": 1,
        "experiment_id": "h1_h2_conditional_v1",
        "completed_at_utc": utc_now(),
        "post_hoc": True,
        "not_eligible_for_formal_selection": True,
        "question": "Does the existing full-H1 screen winner maintain low new collision on current CPU full-600?",
        "arm_id": "QP3_A1_H1FULL_8S",
        "seed": 20260718,
        "update": 2,
        "transitions": 25600,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_manifest": str(MANIFEST.relative_to(ROOT)),
        "baseline": {
            "path": str(BASELINE.relative_to(ROOT)),
            "sha256": sha256_file(BASELINE),
            "summary": baseline["summary"],
        },
        "candidate_raw": str(RAW_OUTPUT.relative_to(ROOT)),
        "candidate_raw_sha256": sha256_file(RAW_OUTPUT),
        "candidate_summary": candidate["summary"],
        "paired": paired_metrics(baseline, candidate),
        "evaluation_contract": candidate["evaluation_contract"],
        "coverage": {
            "rows": len(candidate["rows"]),
            "unique_scenario_ids": len({str(row["scenario_id"]) for row in candidate["rows"]}),
            "all_observations_finite": all(bool(row["observation_finite"]) for row in candidate["rows"]),
            "all_actions_finite": all(bool(row["action_finite"]) for row in candidate["rows"]),
        },
    }
    write_json(FINAL_OUTPUT, result)
    print(json.dumps({"candidate_summary": result["candidate_summary"], "paired": result["paired"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

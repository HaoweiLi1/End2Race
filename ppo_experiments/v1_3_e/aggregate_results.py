#!/usr/bin/env python3
"""Strict aggregation for the evaluation-only PPO V1.3-E continuation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ppo_experiments.v1_3_b.aggregate_results import (  # noqa: E402
    assert_finite,
    evaluation_summary,
    paired_comparison,
    read_json,
    relative,
    sha256_file,
    stripped_evaluation,
    validate_actor_checkpoint,
    write_json,
)


EXPERIMENT_DIR = ROOT / "ppo_experiments" / "v1_3_e"
RUN_ROOT = ROOT / "runs" / "ppo"
EVAL_ROOT = ROOT / "eval_results"
SEEDS = (20260735, 20260736, 20260737)
EXPECTED_BASELINE = {"collision": 21, "follow": 233, "overtake": 346}
EXPECTED_CHECKPOINTS = {
    20260735: "02351fc34fc6010dd0b2507bc11b25b1e1b10e6f0037646c549e968083ab5ce7",
    20260736: "e98976e42e8b55c1bfcfc637b43047efdbdf4c3288675c3040c20f5da91800d0",
    20260737: "b56f9998e7e2f149fbf19c9cb8eb7657c7f0baed29af1ef837896dfec28f68d5",
}


def source_checkpoint(seed: int) -> Path:
    return (
        RUN_ROOT
        / f"v1_3_d_seed{seed}"
        / "checkpoints"
        / f"end2race_ppo_v1_3_d_u0008_s{seed}.pth"
    )


def candidate_checkpoint(seed: int) -> Path:
    return RUN_ROOT / "v1_3_e_candidates" / f"end2race_ppo_v1_3_e_cpu_u0008_s{seed}.pth"


def evaluation_log(seed: int | None = None) -> dict[str, str]:
    name = "eval_bc_cpu.log" if seed is None else f"eval_seed{seed}_cpu.log"
    path = RUN_ROOT / "v1_3_e_logs" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    content = path.read_text(encoding="utf-8", errors="replace")
    for marker in (
        "EVAL_DEVICE=cpu",
        "Starting batch evaluation of 600 segments",
        "Noise level: 0.0",
        "error: 0",
    ):
        if marker not in content:
            raise ValueError(f"Evaluation log lacks marker {marker!r}: {path}")
    return {"path": relative(path), "sha256": sha256_file(path)}


def checkpoint_record(seed: int) -> dict[str, Any]:
    source = source_checkpoint(seed)
    candidate = candidate_checkpoint(seed)
    expected = EXPECTED_CHECKPOINTS[seed]
    source_hash = sha256_file(source)
    candidate_hash = sha256_file(candidate)
    if source_hash != expected or candidate_hash != expected:
        raise ValueError(f"Checkpoint hash mismatch for seed {seed}")
    source_keys = validate_actor_checkpoint(source)
    candidate_keys = validate_actor_checkpoint(candidate)
    if source_keys != 12 or candidate_keys != 12:
        raise ValueError(f"Non-12-key actor for seed {seed}")
    return {
        "seed": seed,
        "source_path": relative(source),
        "candidate_path": relative(candidate),
        "sha256": expected,
        "actor_key_count": 12,
    }


def build_report(results: dict[str, Any]) -> str:
    lines = [
        "# End2Race PPO V1.3-E CPU Evaluation Report",
        "",
        f"**Final verdict:** `{results['final_verdict']}`",
        "",
        "V1.3-E performs no training. It re-evaluates the three fixed V1.3-D U8 actors "
        "on CPU, matching the device used to generate the canonical development baseline.",
        "",
        "## CPU baseline",
        "",
    ]
    baseline = results["baseline"]
    lines.extend([
        f"Observed: `{baseline['counts']['collision']} collision / {baseline['counts']['follow']} follow / "
        f"{baseline['counts']['overtake']} overtake` over 600 rows.",
        "",
    ])
    if "evaluations" in results:
        lines.extend([
            "## Fixed U8 candidates",
            "",
            "| Seed | Collision | Follow | Overtake | Fixed | New | G | Speed ratio | Distance ratio | Pass |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ])
        for row in results["evaluations"]:
            counts = row["counts"]
            paired = row["paired"]
            lines.append(
                f"| {row['seed']} | {counts['collision']} | {counts['follow']} | {counts['overtake']} | "
                f"{paired['fixed_collision']} | {paired['new_collision']} | {paired['G']} | "
                f"{paired['speed_ratio_to_bc']:.4f} | {paired['distance_ratio_to_bc']:.4f} | "
                f"{'Y' if row['product_gate_pass'] else 'N'} |"
            )
        lines.append("")
    lines.extend([
        "## Conclusion",
        "",
        results["interpretation"],
        "",
        "This is exploratory development evidence. `selection_performed=false`, "
        "`holdout_performed=false`, and `promotion_performed=false`. Canonical BC remains "
        "the deployment recommendation.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    precheck = read_json(EXPERIMENT_DIR / "PRECHECK.json")
    preregistration = read_json(EXPERIMENT_DIR / "PREREGISTRATION.json")
    diagnosis = read_json(EXPERIMENT_DIR / "PROTOCOL_DIAGNOSIS.json")
    if precheck.get("status") != "PASS" or preregistration.get("status") != "PREREGISTERED":
        raise ValueError("V1.3-E is not locked")

    d_results = read_json(ROOT / "ppo_experiments" / "v1_3_d" / "RESULTS.json")
    if d_results.get("final_verdict") != "STOP_PROTOCOL_DRIFT":
        raise ValueError("V1.3-D terminal evidence changed")
    d_training = {int(row["seed"]): row for row in d_results["training"]}
    if set(d_training) != set(SEEDS) or not all(d_training[seed]["process_gate_pass"] for seed in SEEDS):
        raise ValueError("V1.3-D process gates are not 3/3 PASS")

    checkpoints = [checkpoint_record(seed) for seed in SEEDS]
    baseline_path = EVAL_ROOT / "end2race_bc_v1_3_e_cpu_Austin" / "multiagents" / "results_multi.json"
    baseline = evaluation_summary(baseline_path)
    baseline["log"] = evaluation_log()
    baseline_record = stripped_evaluation(baseline)
    baseline_record["expected_counts"] = EXPECTED_BASELINE
    baseline_record["protocol_match"] = baseline["counts"] == EXPECTED_BASELINE

    results: dict[str, Any] = {
        "schema_version": 1,
        "record": "PPO_V1_3_E_RESULTS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "evaluation_only_post_protocol_correction",
        "evaluation_device": "cpu",
        "source_v1_3_d_results": {
            "path": "ppo_experiments/v1_3_d/RESULTS.json",
            "sha256": sha256_file(ROOT / "ppo_experiments" / "v1_3_d" / "RESULTS.json"),
            "verdict": d_results["final_verdict"],
        },
        "protocol_diagnosis": {
            "path": relative(EXPERIMENT_DIR / "PROTOCOL_DIAGNOSIS.json"),
            "sha256": sha256_file(EXPERIMENT_DIR / "PROTOCOL_DIAGNOSIS.json"),
            "conclusion": diagnosis["conclusion"],
        },
        "checkpoints": checkpoints,
        "baseline": baseline_record,
        "selection_performed": False,
        "holdout_performed": False,
        "promotion_performed": False,
        "aggregation_script": {"path": relative(Path(__file__)), "sha256": sha256_file(Path(__file__))},
    }

    if not baseline_record["protocol_match"]:
        results["final_verdict"] = "STOP_CPU_BASELINE_MISMATCH"
        results["interpretation"] = (
            f"CPU BC exact-match gate failed: expected {EXPECTED_BASELINE}, observed {baseline['counts']}. "
            "Candidate evaluations were not run."
        )
    else:
        evaluations: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            seed = checkpoint["seed"]
            stem = Path(checkpoint["candidate_path"]).stem
            path = EVAL_ROOT / f"{stem}_Austin" / "multiagents" / "results_multi.json"
            summary = evaluation_summary(path, baseline["scenario_ids"])
            paired = paired_comparison(summary, baseline)
            gates = {
                "G_gte_5": paired["G"] >= 5,
                "collision_lte_16": summary["counts"]["collision"] <= 16,
                "overtake_gte_340": summary["counts"]["overtake"] >= 340,
                "speed_ratio_gte_0_99": paired["speed_ratio_to_bc"] >= 0.99,
                "distance_ratio_gte_0_99": paired["distance_ratio_to_bc"] >= 0.99,
            }
            evaluations.append({
                "seed": seed,
                "checkpoint": checkpoint,
                **stripped_evaluation(summary),
                "paired": paired,
                "product_gates": gates,
                "product_gate_pass": all(gates.values()),
                "log": evaluation_log(seed),
            })
        results["evaluations"] = evaluations
        if all(row["product_gate_pass"] for row in evaluations):
            results["final_verdict"] = "PASS_EXPLORATORY_CPU_DEVELOPMENT"
            results["interpretation"] = (
                "All three fixed U8 actors passed the unchanged development product gates on CPU. "
                "A separately preregistered fresh holdout would still be required before any promotion."
            )
        else:
            results["final_verdict"] = "FAIL_NO_STABLE_IMPROVEMENT"
            results["interpretation"] = (
                "The physical-Gaussian repair produced controlled updates, but the three fixed U8 actors "
                "did not produce the required cross-seed driving improvement under the corrected CPU contract."
            )

    assert_finite(results, "RESULTS")
    write_json(EXPERIMENT_DIR / "RESULTS.json", results)
    (EXPERIMENT_DIR / "FINAL_REPORT.md").write_text(build_report(results), encoding="utf-8")
    status = read_json(EXPERIMENT_DIR / "STATUS.json")
    status.update({
        "status": results["final_verdict"],
        "phase": "TERMINAL",
        "updated_at": results["generated_at"],
        "failure_reason": None if results["final_verdict"].startswith("PASS") else results["interpretation"],
    })
    write_json(EXPERIMENT_DIR / "STATUS.json", status)
    print(f"V1_3_E_VERDICT={results['final_verdict']}")


if __name__ == "__main__":
    main()

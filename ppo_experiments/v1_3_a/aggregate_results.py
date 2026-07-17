#!/usr/bin/env python3
"""Strict aggregation for the preregistered PPO V1.3-A experiment."""

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
    read_jsonl,
    relative,
    sha256_file,
    stripped_evaluation,
    validate_actor_checkpoint,
    write_json,
)


EXPERIMENT_DIR = ROOT / "ppo_experiments" / "v1_3_a"
RUN_ROOT = ROOT / "runs" / "ppo"
EVAL_ROOT = ROOT / "eval_results"
SEEDS = (20260718, 20260719, 20260720, 20260721, 20260722)
EXPECTED_BASELINE = {"collision": 21, "follow": 233, "overtake": 346}


def summarize_training(seed: int) -> dict[str, Any]:
    run_dir = RUN_ROOT / f"v1_3_a_seed{seed}"
    files = {
        "resolved_config": run_dir / "resolved_config.json",
        "run_status": run_dir / "run_status.json",
        "training_metrics": run_dir / "training_metrics.jsonl",
        "checkpoint_manifest": run_dir / "checkpoint_manifest.json",
        "sampler_summary": run_dir / "sampler_summary.json",
    }
    resolved = read_json(files["resolved_config"])
    status = read_json(files["run_status"])
    metrics = read_jsonl(files["training_metrics"])
    manifest = read_json(files["checkpoint_manifest"])
    sampler = read_json(files["sampler_summary"])
    assert_finite({"resolved": resolved, "metrics": metrics, "sampler": sampler}, f"training.seed{seed}")
    expected = {
        "name": "v1_3_a",
        "seed": seed,
        "n_envs": 16,
        "n_steps": 1600,
        "batch_size": 1600,
        "n_epochs": 1,
        "updates": 8,
        "checkpoint_updates": [8],
        "transitions_per_update": 25600,
        "minibatches_per_epoch": 16,
        "planned_optimizer_steps_per_update": 16,
        "total_optimizer_steps": 128,
        "gru_lr": 3e-6,
        "head_lr": 3e-5,
        "critic_lr": 3e-4,
        "target_kl": 0.01,
        "update_kl_guardrail": 0.02,
        "critic_profile": "C0_RAW_SINGLE_FRAME",
        "hard_pool": "h0_current_det",
        "hard_sampling_probability": 0.5,
        "hard_sampling_mode": "with_replacement",
        "steering_latent_std": 0.05,
        "speed_physical_std": 0.15,
        "margin_weight": 0.0,
        "margin_threshold": 0.0,
    }
    for key, value in expected.items():
        if resolved.get(key) != value:
            raise ValueError(f"Seed {seed} resolved {key}={resolved.get(key)!r}, expected {value!r}")
    for row in metrics:
        if not 0 < int(row["actual_optimizer_steps"]) <= 16:
            raise ValueError(f"Invalid optimizer step count at seed {seed} U{row['update']}")
        frozen = row["actor_delta_from_bc"]["frozen_actor"]
        if frozen["max_abs_delta_from_bc"] != 0.0:
            raise ValueError(f"Frozen actor drift at seed {seed} U{row['update']}")
        if row["actor_delta_from_bc"]["log_std_max_abs_delta_from_initial"] != 0.0:
            raise ValueError(f"Frozen log_std drift at seed {seed} U{row['update']}")
        if float(row["approx_kl"]) > 0.02 and status.get("status") != "STOPPED_KL_GUARDRAIL":
            raise ValueError(f"KL violation without guardrail stop at seed {seed} U{row['update']}")
    checkpoints = manifest.get("checkpoints", [])
    checkpoint: dict[str, Any] | None = None
    if status.get("status") == "COMPLETED":
        if status.get("last_completed_update") != 8 or [row.get("update") for row in metrics] != list(range(1, 9)):
            raise ValueError(f"Completed seed {seed} is not exactly U1..U8")
        if [row.get("update") for row in checkpoints] != [8]:
            raise ValueError(f"Seed {seed} does not have exactly one U8 checkpoint")
    if checkpoints:
        if len(checkpoints) != 1 or int(checkpoints[0]["update"]) != 8:
            raise ValueError(f"Unexpected checkpoint set for seed {seed}")
        checkpoint_path = run_dir / checkpoints[0]["path"]
        if sha256_file(checkpoint_path) != checkpoints[0]["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")
        checkpoint = {
            "path": relative(checkpoint_path),
            "stem": checkpoint_path.stem,
            "sha256": checkpoints[0]["sha256"],
            "actor_key_count": validate_actor_checkpoint(checkpoint_path),
        }
    in_window = sum(0.002 <= float(row["approx_kl"]) <= 0.010 for row in metrics)
    process_gate = (
        status.get("status") == "COMPLETED"
        and len(metrics) == 8
        and all(float(row["approx_kl"]) <= 0.020 for row in metrics)
        and in_window >= 6
    )
    return {
        "seed": seed,
        "status": status,
        "resolved_config": resolved,
        "metrics": metrics,
        "sampler_summary": sampler,
        "checkpoint": checkpoint,
        "updates_in_target_kl_window": in_window,
        "process_gate_pass": process_gate,
        "artifacts": {
            name: {"path": relative(path), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
    }


def evaluation_log(seed: int | None = None) -> dict[str, str]:
    filename = "eval_bc.log" if seed is None else f"eval_seed{seed}.log"
    path = RUN_ROOT / "v1_3_a_logs" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    content = path.read_text(encoding="utf-8", errors="replace")
    for marker in ("Starting batch evaluation of 600 segments", "Noise level: 0.0", "error: 0"):
        if marker not in content:
            raise ValueError(f"Evaluation log lacks completion marker {marker!r}: {path}")
    return {"path": relative(path), "sha256": sha256_file(path)}


def build_report(results: dict[str, Any]) -> str:
    lines = [
        "# End2Race PPO V1.3-A Final Report",
        "",
        f"**Final verdict:** `{results['final_verdict']}`",
        "",
        "V1.3-A keeps one PPO epoch and changes only the actor learning rates to 3x nominal. "
        "The only product checkpoint is U8 for every seed.",
        "",
        "## Frozen configuration",
        "",
        "```text",
        "n_envs=16, n_steps=1600, batch_size=1600, n_epochs=1, updates=8",
        "GRU LR=3e-6, head LR=3e-5, critic LR=3e-4",
        "target_kl=0.01, post-update guardrail=0.02",
        "C0 critic, H0 p0.50 with replacement, reward/exploration unchanged",
        "```",
        "",
        "## Training process",
        "",
        "| Seed | Status | Last U | KL sequence | Steps sequence | In [0.002,0.010] | Process pass |",
        "|---:|---|---:|---|---|---:|:---:|",
    ]
    for run in results["training"]:
        kls = ",".join(f"{row['approx_kl']:.4f}" for row in run["metrics"])
        steps = ",".join(str(row["actual_optimizer_steps"]) for row in run["metrics"])
        lines.append(
            f"| {run['seed']} | {run['status']['status']} | {run['status']['last_completed_update']} | "
            f"{kls} | {steps} | {run['updates_in_target_kl_window']} | "
            f"{'Y' if run['process_gate_pass'] else 'N'} |"
        )
    lines.append("")
    if "baseline" in results:
        baseline = results["baseline"]
        lines.extend([
            "## Paired development evaluation",
            "",
            f"BC: `{baseline['counts']['collision']} collision / {baseline['counts']['follow']} follow / "
            f"{baseline['counts']['overtake']} overtake`.",
            "",
            "| Seed | Collision | Follow | Overtake | Fixed | New | G | Speed ratio | Distance ratio | Product pass |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ])
        for row in results["evaluations"]:
            counts, paired = row["counts"], row["paired"]
            lines.append(
                f"| {row['seed']} | {counts['collision']} | {counts['follow']} | {counts['overtake']} | "
                f"{paired['fixed_collision']} | {paired['new_collision']} | {paired['G']} | "
                f"{paired['speed_ratio_to_bc']:.4f} | {paired['distance_ratio_to_bc']:.4f} | "
                f"{'Y' if row['product_gate_pass'] else 'N'} |"
            )
        lines.append("")
    elif results.get("not_started_seeds"):
        lines.extend([
            "## Fail-fast stop",
            "",
            f"Not started: `{results['not_started_seeds']}`. No candidate or BC evaluation was run.",
            "",
        ])
    lines.extend([
        "## Conclusion",
        "",
        results["interpretation"],
        "",
        "`selection_performed=false`, `holdout_performed=false`, and `promotion_performed=false`. "
        "Canonical BC remains the deployment recommendation.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    precheck = read_json(EXPERIMENT_DIR / "PRECHECK.json")
    preregistration = read_json(EXPERIMENT_DIR / "PREREGISTRATION.json")
    if precheck.get("status") != "PASS" or preregistration.get("status") != "PREREGISTERED":
        raise ValueError("V1.3-A precheck/preregistration is not locked")
    training: list[dict[str, Any]] = []
    not_started: list[int] = []
    for seed in SEEDS:
        path = RUN_ROOT / f"v1_3_a_seed{seed}"
        if path.exists():
            if not_started:
                raise ValueError("A later seed exists after an unstarted seed")
            training.append(summarize_training(seed))
        else:
            not_started.append(seed)
    if not training:
        raise ValueError("No formal V1.3-A run exists")
    results: dict[str, Any] = {
        "schema_version": 1,
        "record": "PPO_V1_3_A_RESULTS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": precheck["implementation_commit"],
        "training": training,
        "not_started_seeds": not_started,
        "selection_performed": False,
        "holdout_performed": False,
        "promotion_performed": False,
        "aggregation_script": {"path": relative(Path(__file__)), "sha256": sha256_file(Path(__file__))},
    }
    statuses = [run["status"]["status"] for run in training]
    if any(status == "STOPPED_KL_GUARDRAIL" for status in statuses):
        stop_index = statuses.index("STOPPED_KL_GUARDRAIL")
        if stop_index != len(training) - 1 or not_started != list(SEEDS[len(training):]):
            raise ValueError("Run layout violates fail-fast ordering")
        results["final_verdict"] = "FAIL_KL_UNSTABLE"
        results["interpretation"] = "A formal update exceeded the locked KL guardrail; fixed 3x actor LR is not stable."
    elif statuses != ["COMPLETED"] * len(SEEDS) or not_started:
        results["final_verdict"] = "INVALID_INFRASTRUCTURE"
        results["interpretation"] = "Formal training did not reach a registered terminal layout."
    elif not all(run["process_gate_pass"] for run in training):
        results["final_verdict"] = "FAIL_UPDATE_WINDOW_NOT_REACHED"
        results["interpretation"] = (
            "All seeds trained stably, but at least one seed had fewer than 6/8 updates in the preregistered "
            "KL window [0.002,0.010]. No product evaluation was consumed."
        )
    else:
        baseline_path = EVAL_ROOT / "end2race_bc_v1_3_a_Austin" / "multiagents" / "results_multi.json"
        baseline = evaluation_summary(baseline_path)
        if baseline["counts"] != EXPECTED_BASELINE:
            results["final_verdict"] = "STOP_PROTOCOL_DRIFT"
            results["interpretation"] = f"Paired BC drifted to {baseline['counts']}."
        else:
            baseline["log"] = evaluation_log()
            results["baseline"] = stripped_evaluation(baseline)
            evaluations: list[dict[str, Any]] = []
            for run in training:
                checkpoint = run["checkpoint"]
                result_path = EVAL_ROOT / f"{checkpoint['stem']}_Austin" / "multiagents" / "results_multi.json"
                summary = evaluation_summary(result_path, baseline["scenario_ids"])
                paired = paired_comparison(summary, baseline)
                gates = {
                    "G_gte_5": paired["G"] >= 5,
                    "collision_lte_16": summary["counts"]["collision"] <= 16,
                    "overtake_gte_340": summary["counts"]["overtake"] >= 340,
                    "speed_ratio_gte_0_99": paired["speed_ratio_to_bc"] >= 0.99,
                    "distance_ratio_gte_0_99": paired["distance_ratio_to_bc"] >= 0.99,
                }
                evaluations.append({
                    "seed": run["seed"],
                    "checkpoint": checkpoint,
                    **stripped_evaluation(summary),
                    "paired": paired,
                    "product_gates": gates,
                    "product_gate_pass": all(gates.values()),
                    "log": evaluation_log(run["seed"]),
                })
            results["evaluations"] = evaluations
            if all(row["product_gate_pass"] for row in evaluations):
                results["final_verdict"] = "PASS_STABLE_3X_LR_DEVELOPMENT"
                results["interpretation"] = (
                    "All five fixed U8 actors passed the locked process and product gates. A separately "
                    "preregistered new holdout is required before any deployment claim."
                )
            else:
                results["final_verdict"] = "FAIL_NO_STABLE_IMPROVEMENT"
                results["interpretation"] = (
                    "The fixed 3x LR reached the controlled update window, but the five U8 actors did not "
                    "produce a stable product improvement."
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
    print(f"V1_3_A_VERDICT={results['final_verdict']}")


if __name__ == "__main__":
    main()

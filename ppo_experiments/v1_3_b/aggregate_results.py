#!/usr/bin/env python3
"""Strictly aggregate the preregistered PPO V1.3-B artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "ppo_experiments" / "v1_3_b"
RUN_ROOT = ROOT / "runs" / "ppo"
EVAL_ROOT = ROOT / "eval_results"
SEEDS = (20260723, 20260724, 20260725, 20260726, 20260727)
CHECKPOINT_UPDATES = (2, 4, 8)
PRIMARY_UPDATE = 8
EXPECTED_BASELINE = {"collision": 21, "follow": 233, "overtake": 346}


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_actor_checkpoint(path: Path) -> int:
    """Validate the exact canonical 12-key actor-only checkpoint contract."""
    baseline = torch.load(ROOT / "pretrained" / "end2race.pth", map_location="cpu", weights_only=True)
    candidate = torch.load(path, map_location="cpu", weights_only=True)
    if set(candidate) != set(baseline):
        raise ValueError(f"Actor key mismatch: {path}")
    for name, tensor in candidate.items():
        if tensor.shape != baseline[name].shape or tensor.dtype != baseline[name].dtype:
            raise ValueError(f"Actor tensor mismatch for {name}: {path}")
    if len(candidate) != 12:
        raise ValueError(f"Expected 12 actor keys at {path}, got {len(candidate)}")
    return len(candidate)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def assert_finite(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite(child, f"{path}[{index}]")
    elif isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(f"Non-finite value at {path}: {value!r}")


def classify_episode(row: dict[str, Any]) -> str:
    if bool(row.get("ego_collision_occurred")):
        return "collision"
    state_label = row.get("state_label")
    if state_label in {"overtake", "overtaking"}:
        return "overtake"
    if state_label in {"follow", "following"}:
        return "follow"
    raise ValueError(f"Unknown non-collision state_label: {state_label!r}")


def evaluation_summary(path: Path, expected_scenarios: set[str] | None = None) -> dict[str, Any]:
    payload = read_json(path)
    assert_finite(payload, relative(path))
    if set(payload) != {"episodes", "final"} or not isinstance(payload["episodes"], dict):
        raise ValueError(f"Unexpected evaluator schema: {path}")
    episodes = payload["episodes"]
    if len(episodes) != 600:
        raise ValueError(f"Expected 600 episode rows at {path}, got {len(episodes)}")
    scenario_ids = [str(row.get("scenario_id")) for row in episodes.values()]
    if len(set(scenario_ids)) != 600:
        raise ValueError(f"Expected 600 unique scenario IDs at {path}")
    scenario_set = set(scenario_ids)
    if expected_scenarios is not None and scenario_set != expected_scenarios:
        raise ValueError(f"Scenario set differs from paired BC at {path}")
    if any(row.get("collision_scope") != "ego" for row in episodes.values()):
        raise ValueError(f"Non-ego collision scope at {path}")
    counts = Counter(classify_episode(row) for row in episodes.values())
    final = payload["final"]
    if int(final.get("error_count", -1)) != 0 or int(final.get("total_episodes", -1)) != 600:
        raise ValueError(f"Invalid final evaluator counts at {path}")
    expected_final = {
        "collision": int(final.get("collision_count", -1)),
        "follow": int(final.get("following_count", -1)),
        "overtake": int(final.get("overtaking_count", -1)),
    }
    observed = {name: int(counts[name]) for name in ("collision", "follow", "overtake")}
    if observed != expected_final or sum(observed.values()) != 600:
        raise ValueError(f"Episode/final outcome mismatch at {path}: {observed} vs {expected_final}")
    output_root = path.parents[1]
    generated_media = sorted(
        relative(candidate)
        for pattern in ("*.npz", "*.mp4")
        for candidate in output_root.rglob(pattern)
    )
    if generated_media:
        raise ValueError(f"Trace/render artifacts unexpectedly present for {path}: {generated_media[:3]}")
    outcomes = {
        str(row["scenario_id"]): classify_episode(row)
        for row in episodes.values()
    }
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "episode_rows": len(episodes),
        "unique_scenario_ids": len(scenario_set),
        "scenario_ids": scenario_set,
        "outcomes_by_scenario": outcomes,
        "counts": observed,
        "opponent_only_collision": sum(bool(row.get("opponent_only_collision")) for row in episodes.values()),
        "mean_avg_speed": sum(float(row["avg_speed"]) for row in episodes.values()) / len(episodes),
        "mean_total_distance": sum(float(row["total_distance"]) for row in episodes.values()) / len(episodes),
    }


def paired_comparison(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_outcomes = candidate["outcomes_by_scenario"]
    baseline_outcomes = baseline["outcomes_by_scenario"]
    fixed_collision = sum(
        baseline_outcomes[key] == "collision" and candidate_outcomes[key] != "collision"
        for key in baseline_outcomes
    )
    new_collision = sum(
        baseline_outcomes[key] != "collision" and candidate_outcomes[key] == "collision"
        for key in baseline_outcomes
    )
    gained_overtake = sum(
        baseline_outcomes[key] != "overtake" and candidate_outcomes[key] == "overtake"
        for key in baseline_outcomes
    )
    lost_overtake = sum(
        baseline_outcomes[key] == "overtake" and candidate_outcomes[key] != "overtake"
        for key in baseline_outcomes
    )
    return {
        "fixed_collision": fixed_collision,
        "new_collision": new_collision,
        "G": fixed_collision - new_collision,
        "gained_overtake": gained_overtake,
        "lost_overtake": lost_overtake,
        "speed_ratio_to_bc": candidate["mean_avg_speed"] / baseline["mean_avg_speed"],
        "distance_ratio_to_bc": candidate["mean_total_distance"] / baseline["mean_total_distance"],
    }


def summarize_training(seed: int) -> dict[str, Any]:
    run_dir = RUN_ROOT / f"v1_3_b_seed{seed}"
    paths = {
        name: run_dir / filename
        for name, filename in {
            "resolved_config": "resolved_config.json",
            "run_status": "run_status.json",
            "training_metrics": "training_metrics.jsonl",
            "checkpoint_manifest": "checkpoint_manifest.json",
            "sampler_summary": "sampler_summary.json",
        }.items()
    }
    resolved = read_json(paths["resolved_config"])
    status = read_json(paths["run_status"])
    metrics = read_jsonl(paths["training_metrics"])
    manifest = read_json(paths["checkpoint_manifest"])
    sampler = read_json(paths["sampler_summary"])
    assert_finite({"resolved": resolved, "metrics": metrics, "sampler": sampler}, f"training.seed{seed}")
    expected_config = {
        "name": "v1_3_b",
        "seed": seed,
        "n_epochs": 4,
        "updates": 8,
        "checkpoint_updates": [2, 4, 8],
        "target_kl": 0.01,
        "update_kl_guardrail": 0.02,
        "n_envs": 16,
        "n_steps": 1600,
        "batch_size": 1600,
        "transitions_per_update": 25600,
        "minibatches_per_epoch": 16,
        "planned_optimizer_steps_per_update": 64,
        "total_optimizer_steps": 512,
        "gru_lr": 1e-6,
        "head_lr": 1e-5,
        "critic_profile": "C0_RAW_SINGLE_FRAME",
        "hard_pool": "h0_current_det",
        "hard_sampling_probability": 0.5,
        "hard_sampling_mode": "with_replacement",
        "steering_latent_std": 0.05,
        "speed_physical_std": 0.15,
        "margin_weight": 0.0,
        "margin_threshold": 0.0,
    }
    for key, expected in expected_config.items():
        if resolved.get(key) != expected:
            raise ValueError(f"Resolved config mismatch for seed {seed}: {key}={resolved.get(key)!r}, expected {expected!r}")
    if status.get("status") == "COMPLETED":
        if status.get("last_completed_update") != 8:
            raise ValueError(f"Completed seed {seed} did not reach U8")
        if [row.get("update") for row in metrics] != list(range(1, 9)):
            raise ValueError(f"Training metrics are not exactly U1..U8 for seed {seed}")
        if [row.get("update") for row in manifest.get("checkpoints", [])] != [2, 4, 8]:
            raise ValueError(f"Checkpoint manifest is not exactly U2/U4/U8 for seed {seed}")
    for row in metrics:
        if not 0 < int(row["actual_optimizer_steps"]) <= 64:
            raise ValueError(f"Invalid actual optimizer steps at seed {seed} U{row['update']}")
        if float(row["approx_kl"]) > 0.02:
            if status.get("status") != "STOPPED_KL_GUARDRAIL":
                raise ValueError(f"KL guardrail violation without stop at seed {seed} U{row['update']}")
    checkpoints: dict[int, dict[str, Any]] = {}
    for row in manifest.get("checkpoints", []):
        update = int(row["update"])
        checkpoint_path = run_dir / row["path"]
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != row["sha256"]:
            raise ValueError(f"Missing or hash-mismatched checkpoint: {checkpoint_path}")
        checkpoints[update] = {
            "path": relative(checkpoint_path),
            "sha256": row["sha256"],
            "stem": checkpoint_path.stem,
            "actor_key_count": validate_actor_checkpoint(checkpoint_path),
        }
    return {
        "seed": seed,
        "status": status,
        "resolved_config": resolved,
        "metrics": metrics,
        "sampler_summary": sampler,
        "checkpoints": checkpoints,
        "artifacts": {
            name: {"path": relative(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }


def stripped_evaluation(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"scenario_ids", "outcomes_by_scenario"}
    }


def evaluation_log(stem: str, *, baseline: bool = False) -> dict[str, str]:
    path = RUN_ROOT / "v1_3_b_logs" / ("eval_bc.log" if baseline else f"eval_{stem}.log")
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    required = ("Starting batch evaluation of 600 segments", "Noise level: 0.0", "error: 0")
    if any(fragment not in text for fragment in required):
        raise ValueError(f"Evaluation log lacks frozen-protocol completion markers: {path}")
    return {"path": relative(path), "sha256": sha256_file(path)}


def build_report(results: dict[str, Any]) -> str:
    lines = [
        "# End2Race PPO V1.3-B Results",
        "",
        f"**Final verdict:** `{results['final_verdict']}`",
        "",
        "V1.3-B changes only PPO rollout reuse from one to four epochs at the original actor learning rates. "
        "All product decisions use the fixed U8 checkpoint for every seed.",
        "",
        "## Frozen configuration",
        "",
        "```text",
        "n_envs=16, n_steps=1600, batch_size=1600, updates=8",
        "n_epochs=4, target_kl=0.01, update_kl_guardrail=0.02",
        "GRU LR=1e-6, head LR=1e-5, critic=C0_RAW_SINGLE_FRAME",
        "H0 probability=0.50 with replacement, reward and exploration unchanged",
        "```",
        "",
    ]
    if "baseline" not in results:
        lines.extend([
            "## Training stop",
            "",
            "Formal evaluation was not started because at least one seed failed the frozen training-stability gate.",
            "",
        ])
    else:
        baseline = results["baseline"]
        lines.extend([
            "## Paired BC baseline",
            "",
            f"`{baseline['counts']['collision']} collision / {baseline['counts']['follow']} follow / "
            f"{baseline['counts']['overtake']} overtake`, 600 unique scenarios, 0 errors.",
            "",
            "## Checkpoint evaluation",
            "",
            "| Seed | Update | Collision | Follow | Overtake | Fixed | New | G | Speed ratio | Distance ratio |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in results["evaluations"]:
            counts, paired = row["counts"], row["paired"]
            lines.append(
                f"| {row['seed']} | {row['update']} | {counts['collision']} | {counts['follow']} | "
                f"{counts['overtake']} | {paired['fixed_collision']} | {paired['new_collision']} | "
                f"{paired['G']} | {paired['speed_ratio_to_bc']:.4f} | {paired['distance_ratio_to_bc']:.4f} |"
            )
        lines.extend([
            "",
            "## Fixed U8 decision gate",
            "",
            "| Seed | KL stable | G>=5 | Collision<=16 | Overtake>=340 | Speed>=0.99 | Distance>=0.99 | Pass |",
            "|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])
        for row in results["primary_u8_decisions"]:
            gate = row["gates"]
            mark = lambda value: "Y" if value else "N"
            lines.append(
                f"| {row['seed']} | {mark(gate['kl_stable'])} | {mark(gate['G_gte_5'])} | "
                f"{mark(gate['collision_lte_16'])} | {mark(gate['overtake_gte_340'])} | "
                f"{mark(gate['speed_ratio_gte_0_99'])} | {mark(gate['distance_ratio_gte_0_99'])} | "
                f"{mark(row['pass'])} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation boundary",
        "",
        results["interpretation"],
        "",
        "The canonical development panel is used only for this preregistered mechanism test. "
        "No checkpoint is promoted to `posttrained/`, and no deployment claim is made.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    precheck_path = EXPERIMENT_DIR / "PRECHECK.json"
    precheck = read_json(precheck_path)
    if precheck.get("status") != "PASS":
        raise ValueError("PRECHECK.json is not PASS")
    training = [summarize_training(seed) for seed in SEEDS]
    results: dict[str, Any] = {
        "schema_version": 1,
        "record": "PPO_V1_3_B_RESULTS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preregistration": "ppo_experiments/v1_3_b/IMPLEMENTATION_GUIDE.md",
        "precheck": {"path": relative(precheck_path), "sha256": sha256_file(precheck_path)},
        "implementation_commit": precheck.get("implementation_commit"),
        "seeds": list(SEEDS),
        "primary_update": PRIMARY_UPDATE,
        "training": training,
    }
    statuses = [run["status"]["status"] for run in training]
    if statuses != ["COMPLETED"] * len(SEEDS):
        if any(status == "STOPPED_KL_GUARDRAIL" for status in statuses):
            results["final_verdict"] = "FAIL_KL_UNSTABLE"
            results["interpretation"] = (
                "At least one formal seed crossed the preregistered post-update KL guardrail. "
                "The four-epoch update window is therefore not a controlled setting."
            )
        else:
            results["final_verdict"] = "INFRASTRUCTURE_FAILURE"
            results["interpretation"] = "At least one formal training run did not terminate under a registered V1.3-B status."
    else:
        baseline_path = EVAL_ROOT / "end2race_bc_v1_3_b_Austin" / "multiagents" / "results_multi.json"
        baseline = evaluation_summary(baseline_path)
        if baseline["counts"] != EXPECTED_BASELINE:
            raise ValueError(f"Canonical BC protocol drift: {baseline['counts']} != {EXPECTED_BASELINE}")
        baseline["log"] = evaluation_log("end2race_bc_v1_3_b", baseline=True)
        results["baseline"] = stripped_evaluation(baseline)
        evaluations: list[dict[str, Any]] = []
        for run in training:
            for update in CHECKPOINT_UPDATES:
                checkpoint = run["checkpoints"][update]
                result_path = EVAL_ROOT / f"{checkpoint['stem']}_Austin" / "multiagents" / "results_multi.json"
                summary = evaluation_summary(result_path, baseline["scenario_ids"])
                paired = paired_comparison(summary, baseline)
                row = {
                    "seed": run["seed"],
                    "update": update,
                    "checkpoint": checkpoint,
                    **stripped_evaluation(summary),
                    "paired": paired,
                    "log": evaluation_log(checkpoint["stem"]),
                }
                evaluations.append(row)
        results["evaluations"] = evaluations
        primary_decisions: list[dict[str, Any]] = []
        for run in training:
            evaluation = next(
                row for row in evaluations if row["seed"] == run["seed"] and row["update"] == PRIMARY_UPDATE
            )
            kl_stable = all(float(row["approx_kl"]) <= 0.02 for row in run["metrics"])
            gates = {
                "kl_stable": kl_stable,
                "G_gte_5": evaluation["paired"]["G"] >= 5,
                "collision_lte_16": evaluation["counts"]["collision"] <= 16,
                "overtake_gte_340": evaluation["counts"]["overtake"] >= 340,
                "speed_ratio_gte_0_99": evaluation["paired"]["speed_ratio_to_bc"] >= 0.99,
                "distance_ratio_gte_0_99": evaluation["paired"]["distance_ratio_to_bc"] >= 0.99,
            }
            primary_decisions.append({"seed": run["seed"], "gates": gates, "pass": all(gates.values())})
        results["primary_u8_decisions"] = primary_decisions
        if all(row["pass"] for row in primary_decisions):
            results["final_verdict"] = "PASS_STABLE_ACTOR_UPDATE"
            results["interpretation"] = (
                "All five fixed U8 actors passed the preregistered development-panel gate with controlled KL. "
                "This supports a new, independent holdout test but is not deployment evidence."
            )
        else:
            results["final_verdict"] = "FAIL_NO_STABLE_IMPROVEMENT"
            median_kl_values = sorted(
                float(row["approx_kl"])
                for run in training
                for row in run["metrics"]
            )
            median_kl = median_kl_values[len(median_kl_values) // 2]
            results["median_update_approx_kl"] = median_kl
            if median_kl < 0.002:
                results["interpretation"] = (
                    "The five fixed U8 actors did not pass consistently, and most updates remained in the low-movement "
                    "KL range. V1.3-B does not justify selecting intermediate checkpoints or automatically raising LR."
                )
            else:
                results["interpretation"] = (
                    "The five fixed U8 actors did not pass consistently despite controlled, material policy movement. "
                    "Increasing epochs or LR further is not supported by this result."
                )
    assert_finite(results, "RESULTS")
    results_path = EXPERIMENT_DIR / "RESULTS.json"
    report_path = EXPERIMENT_DIR / "REPORT.md"
    write_json(results_path, results)
    report_path.write_text(build_report(results), encoding="utf-8")
    print(f"V1_3_B_VERDICT={results['final_verdict']}")
    print(f"V1_3_B_RESULTS={relative(results_path)}")
    print(f"V1_3_B_REPORT={relative(report_path)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Execute the preregistered v2 quick-pool / mixed-horizon experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "ppo_experiments" / "quick_pool_3s_v2"
ATTEMPT1 = ROOT / "ppo_experiments" / "quick_pool_3s"
RUN_ROOT = ROOT / "runs" / "ppo" / "quick_pool_3s_v2"
PREREGISTRATION = OUTPUT / "PREREGISTRATION.json"
QUICK_PANEL = ATTEMPT1 / "QUICK_PANEL_120.json"
QUICK_BASELINE = ATTEMPT1 / "QUICK_PANEL_BC_RESULTS.json"
BASE_RUNNER = ATTEMPT1 / "run_experiment.py"


def load_base_runner() -> Any:
    spec = importlib.util.spec_from_file_location("quick_pool_3s_attempt1_runner", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import base runner: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_runner()
base.OUTPUT = OUTPUT
base.RUN_ROOT = RUN_ROOT
base.EVALUATOR = ATTEMPT1 / "evaluate_pool.py"
base.COMMANDS = []
base.PROCESSES = {}

ARMS = base.ARMS
RETENTION_ARMS = (
    "QP3_A0_H0_8S",
    "QP3_A2_H1EARLY_8S",
    "QP3_A3_H1EARLY_3S",
)
SOURCE_COMMIT = "566026ef60536346d412c094c9eac2986e5c8dd7"
IMMUTABLE_ATTEMPT1_HASHES = {
    "FINAL_VERDICT.md": "e67bbf2216357bed99267f23a60f3150cf07375f2b905ab589673874b31edd5f",
    "INTEGRITY_FAILURE.json": "28a222025dda20466d1aa21d3abb0b9fdc0b92d9e28c30ceba20f7c32623a649",
    "SCREEN_RESULTS.json": "079fe0dd1808fe959d7b34c9137829a977e07e443deee5df663ddecf266c78ce",
    "RETENTION_RESULTS.json": "2326e70403b9763be2351391e4129df1b15cac526612743306e15456e27eee62",
    "FULL600_RESULTS.json": "2326e70403b9763be2351391e4129df1b15cac526612743306e15456e27eee62",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return base.read_json(path)


def write_json(path: Path, document: Any) -> None:
    base.write_json(path, document)


def verify_sha256(path: Path, expected: str) -> None:
    observed = base.sha256_file(path)
    if observed != expected:
        raise base.IntegrityFailure(
            f"SHA-256 mismatch for {path.relative_to(ROOT)}: expected {expected}, got {observed}"
        )


def validate_evaluation(document: dict[str, Any], expected_total: int) -> None:
    rows = document.get("rows", [])
    summary = document.get("summary", {})
    ids = [str(row.get("scenario_id")) for row in rows]
    if not document.get("complete") or len(rows) != expected_total or len(set(ids)) != expected_total:
        raise base.IntegrityFailure(
            f"Evaluation coverage failure: total={len(rows)}, unique={len(set(ids))}, expected={expected_total}"
        )
    if summary.get("error") != 0 or summary.get("total") != expected_total:
        raise base.IntegrityFailure(f"Evaluation error/total failure: {summary}")
    if sum(int(summary.get(key, 0)) for key in ("collision", "follow", "overtake")) != expected_total:
        raise base.IntegrityFailure(f"Evaluation outcome sum failure: {summary}")
    nonfinite = [
        scenario_id
        for scenario_id, row in zip(ids, rows)
        if not bool(row.get("observation_finite")) or not bool(row.get("action_finite"))
    ]
    if nonfinite:
        raise base.IntegrityFailure(f"Evaluation has non-finite values: {nonfinite[:5]}")


def preflight() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    preregistration = read_json(PREREGISTRATION)
    if preregistration.get("candidate_results_observed") is not False:
        raise base.IntegrityFailure("Preregistration does not state that candidate results are unobserved")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != SOURCE_COMMIT or preregistration.get("source_commit") != SOURCE_COMMIT:
        raise base.IntegrityFailure(f"Source commit mismatch: HEAD={head}")
    for relative, expected in preregistration["source_hashes"].items():
        verify_sha256(ROOT / relative, expected)
    for relative, record in preregistration["reused_phase0"]["artifacts"].items():
        verify_sha256(ROOT / relative, record["sha256"])
    for name, expected in IMMUTABLE_ATTEMPT1_HASHES.items():
        verify_sha256(ATTEMPT1 / name, expected)
    if RUN_ROOT.exists():
        raise base.IntegrityFailure(f"QP3 v2 run directory already exists: {RUN_ROOT}")

    h1_summary = read_json(ATTEMPT1 / "H1_EARLY_3S_SUMMARY.json")
    if not (
        h1_summary.get("classification_total") == 482
        and h1_summary.get("classification_collision_within_3s") == 152
        and h1_summary.get("unique_scenario_count") == 138
        and h1_summary.get("filter_threshold_s") == 2.8
        and h1_summary.get("reproduced_collision_count") == 138
        and h1_summary.get("evaluation_error") == 0
    ):
        raise base.IntegrityFailure(f"Reused H1_EARLY_3S summary mismatch: {h1_summary}")

    panel = read_json(QUICK_PANEL)
    baseline = read_json(QUICK_BASELINE)
    validate_evaluation(baseline, 120)
    expected_summary = {"collision": 20, "follow": 49, "overtake": 51, "error": 0, "total": 120}
    if baseline["summary"] != expected_summary:
        raise base.IntegrityFailure(f"Binding quick baseline mismatch: {baseline['summary']}")
    expected_by_id = {str(key): str(value) for key, value in panel["baseline_outcomes"].items()}
    observed_by_id = {str(row["scenario_id"]): str(row["outcome"]) for row in baseline["rows"]}
    differences = [
        (scenario_id, expected_by_id[scenario_id], observed_by_id[scenario_id])
        for scenario_id in sorted(expected_by_id)
        if expected_by_id[scenario_id] != observed_by_id[scenario_id]
    ]
    expected_drift = [("evaluation-sp30-ego1283-raceline0-v070", "ego_collision", "overtake")]
    if differences != expected_drift:
        raise base.IntegrityFailure(f"Quick baseline drift set mismatch: {differences}")
    return h1_summary, panel, baseline


_base_training_summary = base.training_summary


def training_summary(arm: str, through_update: int) -> dict[str, Any]:
    summary = _base_training_summary(arm, through_update)
    rows = [
        json.loads(line)
        for line in (base.run_dir(arm) / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and int(json.loads(line)["update"]) <= through_update
    ]
    reward_keys = (
        "reward_progress",
        "reward_relative",
        "reward_margin",
        "reward_collision",
        "reward_total",
    )
    summary["reward_component_sums_by_role"] = {
        role: {
            key: sum(float(row["rollout"]["reward_component_sums_by_env_role"][role][key]) for row in rows)
            for key in reward_keys
        }
        for role in ("hard", "ordinary")
    }
    return summary


base.training_summary = training_summary


def evaluate_quick_stage(
    arms: tuple[str, ...],
    update: int,
    baseline: dict[str, Any],
    stage: str,
) -> list[dict[str, Any]]:
    results = []
    for arm in arms:
        raw = base.run_evaluation(
            base.checkpoint_path(arm, update),
            OUTPUT / f"{stage.lower()}_eval_{arm}.json",
            manifest=QUICK_PANEL,
            sim_duration=8.0,
            purpose=f"Evaluate {arm} update {update} on frozen QUICK_PANEL_120",
        )
        validate_evaluation(raw, 120)
        result = base.arm_result(arm, update, baseline, raw)
        evaluation = result["evaluation"]
        if stage == "SCREEN":
            result["screen_pass"] = bool(
                evaluation["error"] == 0
                and evaluation["collision"] <= 18
                and evaluation["net_collision_repair"] >= 2
                and evaluation["overtake"] >= 49
            )
        results.append(result)
    return results


def markdown_table(rows: list[dict[str, Any]], pass_key: str) -> list[str]:
    lines = [
        "| arm | collision | follow | overtake | fixed | new | net repair | gained | lost | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        evaluation = row["evaluation"]
        lines.append(
            f"| {row['arm_id']} | {evaluation['collision']} | {evaluation['follow']} | "
            f"{evaluation['overtake']} | {evaluation['fixed_collision']} | "
            f"{evaluation['new_collision']} | {evaluation['net_collision_repair']} | "
            f"{evaluation['gained_overtake']} | {evaluation['lost_overtake']} | "
            f"{'PASS' if row.get(pass_key, False) else 'FAIL'} |"
        )
    return lines


def write_report(
    verdict: str,
    h1_summary: dict[str, Any] | None,
    panel: dict[str, Any] | None,
    screen: list[dict[str, Any]] | None,
    retention: list[dict[str, Any]] | None,
    full_bc: dict[str, Any] | None,
    full: list[dict[str, Any]] | None,
    integrity_failure: str | None = None,
) -> None:
    lines = [f"# {verdict}", "", f"Completed: {utc_now()}", ""]
    if integrity_failure:
        lines.extend(("## Integrity failure", "", integrity_failure, ""))
    if h1_summary and panel:
        lines.extend(
            (
                "## Reused Phase 0",
                "",
                f"- Full H1: {h1_summary['classification_total']}",
                f"- H1 collision within 3 s: {h1_summary['classification_collision_within_3s']}",
                f"- H1_EARLY_3S: {h1_summary['unique_scenario_count']} at <= {h1_summary['filter_threshold_s']} s",
                f"- Reproduction: {h1_summary['reproduced_collision_count']}/{h1_summary['unique_scenario_count']}",
                "- Interpretation: early-failure subset, not full H1 coverage.",
                f"- Quick manifest hash: `{panel['manifest_hash']}`",
                "- Binding current CPU BC: collision 20, follow 49, overtake 51, error 0.",
                "",
            )
        )
    if screen:
        lines.extend(("## Screen", "", *markdown_table(screen, "screen_pass"), ""))
    if retention:
        lines.extend(("## Retention", "", *markdown_table(retention, "retention_pass"), ""))
    if full_bc:
        summary = full_bc["summary"]
        lines.extend(
            (
                "## Current CPU full-600 BC",
                "",
                f"- collision {summary['collision']}, follow {summary['follow']}, "
                f"overtake {summary['overtake']}, error {summary['error']}",
                "",
            )
        )
    if full:
        lines.extend(("## Full 600 candidates", "", *markdown_table(full, "full600_forward"), ""))
    lines.extend(
        (
            "## Scope",
            "",
            "This is a preregistered quick-experiment result, not a final PPO improvement claim or held-out proof.",
            "",
            "## Exact commands",
            "",
        )
    )
    for record in base.COMMANDS:
        lines.extend(("```text", " ".join(record["argv"]), "```", ""))
    payload = "\n".join(lines)
    (OUTPUT / "FINAL_VERDICT.md").write_text(payload, encoding="utf-8")
    (OUTPUT / "EXECUTION_LOG.md").write_text(payload, encoding="utf-8")


def write_not_run(path: Path, status: str) -> None:
    write_json(path, {"schema_version": 1, "status": status})


def main() -> int:
    os.environ.setdefault("NUMBA_CACHE_DIR", str(OUTPUT / ".numba_cache"))
    h1_summary: dict[str, Any] | None = None
    panel: dict[str, Any] | None = None
    screen: list[dict[str, Any]] | None = None
    retention: list[dict[str, Any]] | None = None
    full_bc: dict[str, Any] | None = None
    full: list[dict[str, Any]] | None = None
    try:
        h1_summary, panel, baseline = preflight()

        for arm in ARMS:
            base.launch_to_screen_pause(arm)

        screen = evaluate_quick_stage(ARMS, 2, baseline, "SCREEN")
        any_screen_pass = any(row["screen_pass"] for row in screen)
        write_json(
            OUTPUT / "SCREEN_RESULTS.json",
            {
                "schema_version": 1,
                "stage": "SCREEN",
                "baseline": baseline["summary"],
                "arms": screen,
                "any_screen_pass": any_screen_pass,
            },
        )
        if not any_screen_pass:
            for arm in ARMS:
                base.send_decision(arm, "stop", "STOPPED_SCREEN")
            write_not_run(OUTPUT / "RETENTION_RESULTS.json", "NOT_RUN_ALL_SCREEN_FAILED")
            write_not_run(OUTPUT / "FULL600_RESULTS.json", "NOT_RUN_ALL_SCREEN_FAILED")
            write_report("NO_QUICK_SIGNAL", h1_summary, panel, screen, None, None, None)
            return 0

        base.send_decision("QP3_A1_H1FULL_8S", "stop", "STOPPED_SCREEN")
        for arm in RETENTION_ARMS:
            base.send_decision(arm, "continue", "COMPLETED")

        retention = evaluate_quick_stage(RETENTION_ARMS, 4, baseline, "RETENTION")
        retention_by_arm = {row["arm_id"]: row for row in retention}
        screen_by_arm = {row["arm_id"]: row for row in screen}
        for arm in ("QP3_A2_H1EARLY_8S", "QP3_A3_H1EARLY_3S"):
            row = retention_by_arm[arm]
            evaluation = row["evaluation"]
            row["basic_retention_pass"] = bool(
                evaluation["error"] == 0
                and evaluation["collision"] <= 18
                and evaluation["fixed_collision"] > evaluation["new_collision"]
                and evaluation["overtake"] >= 49
            )
            row["retention_pass"] = row["basic_retention_pass"]
        retention_by_arm["QP3_A0_H0_8S"]["retention_pass"] = False
        a2 = retention_by_arm["QP3_A2_H1EARLY_8S"]
        a3 = retention_by_arm["QP3_A3_H1EARLY_3S"]
        e2 = a2["evaluation"]
        e3 = a3["evaluation"]
        a3["retention_pass"] = bool(
            a3["basic_retention_pass"]
            and (e3["collision"] <= e2["collision"] or e3["net_collision_repair"] >= e2["net_collision_repair"] + 2)
            and e3["collision"] <= screen_by_arm["QP3_A3_H1EARLY_3S"]["evaluation"]["collision"] + 2
        )
        write_json(
            OUTPUT / "RETENTION_RESULTS.json",
            {
                "schema_version": 1,
                "stage": "RETENTION",
                "baseline": baseline["summary"],
                "arms": retention,
                "a2_pass": a2["retention_pass"],
                "a3_pass": a3["retention_pass"],
            },
        )
        if not a2["retention_pass"] and not a3["retention_pass"]:
            write_not_run(OUTPUT / "FULL600_RESULTS.json", "NOT_RUN_RETENTION_FAILED")
            write_report("NO_QUICK_SIGNAL", h1_summary, panel, screen, retention, None, None)
            return 0

        full_bc = base.run_evaluation(
            base.BC_CHECKPOINT,
            OUTPUT / "FULL600_BC_RESULTS.json",
            purpose="Freeze canonical BC on the original 600-case panel with the current CPU protocol",
        )
        validate_evaluation(full_bc, 600)
        full600_overtake_floor = math.ceil(0.95 * int(full_bc["summary"]["overtake"]))

        full = []
        for arm in RETENTION_ARMS:
            raw = base.run_evaluation(
                base.checkpoint_path(arm, 4),
                OUTPUT / f"full600_eval_{arm}.json",
                purpose=f"Evaluate {arm} update 4 on the original 600-case panel",
            )
            validate_evaluation(raw, 600)
            row = base.arm_result(arm, 4, full_bc, raw)
            row["full600_overtake_floor"] = full600_overtake_floor
            row["full600_forward"] = False
            full.append(row)
        full_by_arm = {row["arm_id"]: row for row in full}
        a0_full = full_by_arm["QP3_A0_H0_8S"]["evaluation"]
        for arm in ("QP3_A2_H1EARLY_8S", "QP3_A3_H1EARLY_3S"):
            evaluation = full_by_arm[arm]["evaluation"]
            full_by_arm[arm]["full600_forward"] = bool(
                evaluation["error"] == 0
                and evaluation["collision"] < a0_full["collision"]
                and evaluation["fixed_collision"] > evaluation["new_collision"]
                and evaluation["overtake"] >= full600_overtake_floor
            )

        f2 = full_by_arm["QP3_A2_H1EARLY_8S"]
        f3 = full_by_arm["QP3_A3_H1EARLY_3S"]
        pool_and_3s_supported = bool(
            a3["retention_pass"]
            and f3["full600_forward"]
            and f3["evaluation"]["collision"] <= f2["evaluation"]["collision"]
            and f3["evaluation"]["overtake"] >= full600_overtake_floor
        )
        pool_supported = bool(a2["retention_pass"] and f2["full600_forward"])
        if pool_and_3s_supported:
            verdict = "POOL_AND_3S_SUPPORTED"
        elif pool_supported:
            verdict = "POOL_SUPPORTED_3S_NOT_SUPPORTED"
        else:
            verdict = "NO_QUICK_SIGNAL"
        write_json(
            OUTPUT / "FULL600_RESULTS.json",
            {
                "schema_version": 1,
                "stage": "FULL600",
                "baseline": full_bc["summary"],
                "full600_overtake_floor": full600_overtake_floor,
                "arms": full,
                "pool_and_3s_supported": pool_and_3s_supported,
                "pool_supported_3s_not_supported": pool_supported and not pool_and_3s_supported,
                "verdict": verdict,
            },
        )
        write_report(verdict, h1_summary, panel, screen, retention, full_bc, full)
        return 0
    except Exception as error:
        base.stop_live_processes()
        failure = f"{type(error).__name__}: {error}"
        if not (OUTPUT / "SCREEN_RESULTS.json").exists():
            write_not_run(OUTPUT / "SCREEN_RESULTS.json", "NOT_RUN_UPSTREAM_INVALID")
        if not (OUTPUT / "RETENTION_RESULTS.json").exists():
            write_not_run(OUTPUT / "RETENTION_RESULTS.json", "NOT_RUN_UPSTREAM_INVALID")
        if not (OUTPUT / "FULL600_RESULTS.json").exists():
            write_not_run(OUTPUT / "FULL600_RESULTS.json", "NOT_RUN_UPSTREAM_INVALID")
        write_json(
            OUTPUT / "INTEGRITY_FAILURE.json",
            {
                "schema_version": 1,
                "status": "INVALID_EXPERIMENT",
                "error": failure,
                "recorded_at": utc_now(),
            },
        )
        write_report(
            "INVALID_EXPERIMENT",
            h1_summary,
            panel,
            screen,
            retention,
            full_bc,
            full,
            integrity_failure=failure,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

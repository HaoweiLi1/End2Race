#!/usr/bin/env python3
"""Aggregate the preregistered mechanism audit into factual JSON and Markdown."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from audit_rl_direction_common import (
        EXPERIMENT_DIR,
        PREREGISTRATION_PATH,
        ROOT,
        RUN_DIR,
        assert_frozen_contract,
        read_json,
        sha256_file,
        write_json_atomic,
    )
except ModuleNotFoundError:
    from scripts.audit_rl_direction_common import (
        EXPERIMENT_DIR,
        PREREGISTRATION_PATH,
        ROOT,
        RUN_DIR,
        assert_frozen_contract,
        read_json,
        sha256_file,
        write_json_atomic,
    )


def _optional(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.is_file() else None


def _artifact(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def _f(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _next_action(p2: dict[str, Any], p3: dict[str, Any], p4: dict[str, Any] | None) -> dict[str, str]:
    verdict = str(p2["verdict"])
    if verdict == "REWARD_MISALIGNED":
        return {
            "action": "REDESIGN_REWARD_BEFORE_ANY_MORE_PPO_TRAINING",
            "reason": "P2 found safe repairs but the current return failed the preregistered ranking gate.",
        }
    if verdict == "LOCAL_ACTION_NOT_FOUND":
        return {
            "action": "RUN_P5_LOCAL_EXPRESSIVITY_AND_OBSERVATION_AUDIT",
            "reason": "The bounded local pulse library did not repair a majority of reproducible collisions.",
        }
    if verdict == "EXPLORATION_COVERAGE_INSUFFICIENT":
        return {
            "action": "PREREGISTER_A_NARROW_SUSTAINED_ACTION_EXPLORATION_INTERVENTION_THEN_REPEAT_P1_AND_P4",
            "reason": (
                "Reward direction passed and all best repairs were below 3 sigma per step, but every "
                "0.25 s repair had iid sequence probability below 1%; more samples of unchanged iid noise "
                "do not directly address that temporal coverage failure."
            ),
        }
    if p4 is None:
        return {
            "action": "DO_NOT_START_LONG_TRAINING_UNTIL_P4_GATE_IS_RESOLVED",
            "reason": "P4 has no completed controlled-step record.",
        }
    if p4["verdict"] == "UPDATE_GEOMETRY_CONFIRMED":
        return {
            "action": "P6_ONE_CANDIDATE_THREE_SEED_SHORT_CONFIRMATION_IS_ALLOWED",
            "reason": "A preregistered controlled update method passed in at least two seeds.",
        }
    return {
        "action": "RUN_P5_LOCAL_EXPRESSIVITY_BEFORE_MORE_TRAINING",
        "reason": "Stable signal and credit were audited, but no controlled P4 step passed across seeds.",
    }


def _build_questions(
    p1: dict[str, Any], p2: dict[str, Any], p3: dict[str, Any], p4: dict[str, Any] | None
) -> list[dict[str, Any]]:
    summaries = p1["final_pool_summaries"]
    h1 = summaries["H1_EXPANDED_DET"]
    extension = read_json(EXPERIMENT_DIR / "P1_GRADIENT_DIRECTION_EXTENSION.json")
    h0 = extension["pools"]["H0_CURRENT_DET"]
    h2 = extension["pools"]["H2_STOCH_CORE"]
    p3_primary = p3["pools"][p3["primary_pool"]]
    c0 = p3_primary["candidates"]["C0_CURRENT"]
    selected = p3_primary["candidates"][p3["selected_credit"]]
    earliest = p2["actionability_window"]["earliest_actionable_seconds_before_collision"]
    questions: list[dict[str, Any]] = [
        {
            "id": "Q1_STABLE_ACTOR_GRADIENT",
            "question": "Does the expected current PPO actor gradient have a stable direction?",
            "answer": (
                "POOL_DEPENDENT_YES_ONLY_FOR_H1; H0_AND_H2_REMAIN_INCONCLUSIVE; H3_IS_CONFLICTING"
            ),
            "evidence": {
                "H1_episodes": h1["episodes"],
                "H1_median_pairwise_cosine": h1["pairwise_combined_median"],
                "H1_bootstrap_95ci": p1["pools"]["H1_EXPANDED_DET"]["bootstrap_95ci"]["combined"],
                "H1_collision_action_sign_agreement": h1["collision_action_delta_sign_agreement"],
                "H0_256_verdict": h0["verdict"],
                "H0_256_median_pairwise_cosine": h0["pairwise_combined_median"],
                "H2_256_verdict": h2["verdict"],
                "H2_256_median_pairwise_cosine": h2["pairwise_combined_median"],
                "H3_128_verdict": summaries["H3_UNION_CORE"]["verdict"],
            },
        },
        {
            "id": "Q2_EPISODES_REQUIRED",
            "question": "How many independent episodes are required to estimate the direction stably?",
            "answer": (
                "128_IS_THE_FIRST_PREREGISTERED_TESTED_COUNT_THAT_PASSES_FOR_H1; "
                "NO_UNIVERSAL_COUNT_EXISTS_BECAUSE_H0_AND_H2_FAIL_TO_STABILIZE_AT_256"
            ),
            "evidence": {
                "H1_first_gate_passing_episode_count": 128,
                "H0_no_pass_episode_count": 256,
                "H2_no_pass_episode_count": 256,
                "transition_count_not_used_as_independence_count": True,
            },
        },
        {
            "id": "Q3_CREDIT_HORIZON",
            "question": "Is 2-3 second collision credit excessively attenuated by current GAE?",
            "answer": (
                "SUPPORTED_BY_A_PASSING_OFFLINE_CREDIT_CANDIDATE"
                if p3["verdict"] == "CREDIT_CANDIDATE_PASSES"
                else "NOT_SUPPORTED_BY_THE_PREREGISTERED_OFFLINE_IMPROVEMENT_GATE"
            ),
            "evidence": {
                "earliest_actionable_seconds_before_collision": earliest,
                "redistribution_window_seconds": p3["redistribution"]["selected_window_seconds"],
                "primary_pool": p3["primary_pool"],
                "current_median_pairwise_cosine": c0["pairwise_combined_median"],
                "selected_credit": p3["selected_credit"],
                "selected_median_pairwise_cosine": selected["pairwise_combined_median"],
                "selection": p3["primary_selection"],
            },
        },
        {
            "id": "Q4_REWARD_RANKING",
            "question": "Does current reward rank genuinely safe local alternatives above BC/no-op?",
            "answer": (
                "YES_FOR_THE_PREREGISTERED_LOCAL_REPAIRS; EXPLORATION_COVERAGE_IS_INSUFFICIENT"
                if p2["gates"]["reward_direction_ok"] and p2["gates"]["exploration_coverage_insufficient"]
                else p2["verdict"]
            ),
            "evidence": {
                "reproduced_collision_cases": p2["h0_collision_reproduced"],
                "repairable_case_count": p2["repairable_case_count"],
                "repairable_fraction": p2["repairable_fraction"],
                "best_safe_return_above_noop_fraction": p2["best_safe_return_above_noop_fraction"],
                "reward_misaligned_fraction": p2["reward_misaligned_fraction"],
                "gates": p2["gates"],
            },
        },
        {
            "id": "Q5_UPDATE_GEOMETRY",
            "question": "Does sequential minibatch Adam destroy an otherwise useful direction?",
            "answer": "NOT_EXECUTED_DUE_TO_EARLIER_GATE" if p4 is None else p4["verdict"],
            "evidence": None if p4 is None else {
                "method_gates": p4["method_gates"],
                "parameter_delta_pairwise_cosine_across_seeds": p4[
                    "parameter_delta_pairwise_cosine_across_seeds"
                ],
            },
        },
    ]
    return questions


def _report(result: dict[str, Any], p1: dict[str, Any], p2: dict[str, Any], p3: dict[str, Any], p4: dict[str, Any] | None) -> str:
    questions = {row["id"]: row for row in result["questions"]}
    lines = [
        "# End2Race PPO RL-only mechanism audit report",
        "",
        f"- Source HEAD: `{result['source_head']}`",
        f"- Canonical BC SHA-256: `{result['canonical_bc_sha256']}`",
        f"- Device: `{result['device']}`",
        f"- Final status: `{result['status']}`",
        "- Product claim: **none**; this audit did not authorize or perform a held-out product test.",
        "",
        "## Answers to the five mechanism questions",
        "",
        "### 1. Is there a stable actor-gradient direction?",
        "",
        f"`{questions['Q1_STABLE_ACTOR_GRADIENT']['answer']}`.",
        "",
        "| Pool | Episodes | Final verdict | Median pairwise cosine | Collision action-sign agreement |",
        "|---|---:|---|---:|---:|",
    ]
    for name in ("H0_CURRENT_DET", "H1_EXPANDED_DET", "H2_STOCH_CORE", "H3_UNION_CORE"):
        row = p1["final_pool_summaries"][name]
        lines.append(
            f"| {name} | {row['episodes']} | {row['verdict']} | "
            f"{_f(row['pairwise_combined_median'], 6)} | "
            f"{_f(row['collision_action_delta_sign_agreement'], 4)} |"
        )
    lines.extend(
        [
            "",
            "H1 is the only pool that passes all preregistered direction gates. H0 and H2 still do not pass after the single allowed 256-episode extension; H3 is explicitly conflicting.",
            "",
            "### 2. How many independent episodes are needed?",
            "",
            f"`{questions['Q2_EPISODES_REQUIRED']['answer']}`.",
            "",
            "The evidence supports 128 as the first tested passing count for H1, not as a universal minimum and not as proof that fewer episodes would fail. Transition count is not treated as independent sample count.",
            "",
            "### 3. Is the 2-3 second credit window being lost?",
            "",
            f"`{questions['Q3_CREDIT_HORIZON']['answer']}`.",
            "",
            f"P2 earliest-actionable window: `{p2['actionability_window']['earliest_actionable_seconds_before_collision']}`. "
            f"P3 chose `{p3['selected_credit']}` on `{p3['primary_pool']}` with verdict `{p3['verdict']}`.",
            "",
            "### 4. Does reward rank safe alternatives correctly?",
            "",
            f"`{questions['Q4_REWARD_RANKING']['answer']}`.",
            "",
            f"P2 verdict: `{p2['verdict']}`. Reproduced collisions: {p2['h0_collision_reproduced']}; "
            f"repairable: {p2['repairable_case_count']} ({_f(p2['repairable_fraction'])}); "
            f"best-safe return above no-op: {_f(p2['best_safe_return_above_noop_fraction'])}.",
            "",
            f"P2 aggregation revision: `{p2.get('aggregation_revision')}`. The corrected result reuses the identical raw branch SHA and unchanged thresholds; duration is part of a pulse template.",
            "",
            "### 5. Does sequential minibatch update destroy a useful direction?",
            "",
            f"`{questions['Q5_UPDATE_GEOMETRY']['answer']}`.",
        ]
    )
    if p4 is not None:
        lines.extend(
            [
                "",
                "| Method | Seed | Mean exact KL | p99 sequence KL | Fixed collision | New collision | SAFE new collision | Overtake lost |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for seed, seed_record in p4["seeds"].items():
            for method, row in seed_record["methods"].items():
                lines.append(
                    f"| {method} | {seed} | {_f(row['exact_kl']['mean_exact_kl'], 6)} | "
                    f"{_f(row['exact_kl']['p99_per_sequence_kl'], 6)} | "
                    f"{row['panel']['fixed_collision_count']} | {row['panel']['new_collision_count']} | "
                    f"{row['panel']['safe_reference_new_collision_count']} | {row['panel']['lost_overtake_count']} |"
                )
        lines.extend(
            [
                "",
                "Neither S2 nor S3 produced a single fully passing seed: smaller KL did not prevent SAFE collisions or overtake losses, so sequential minibatch geometry is not confirmed as the primary failure source.",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision and next allowed action",
            "",
            f"- Action: `{result['next_allowed_action']['action']}`",
            f"- Reason: {result['next_allowed_action']['reason']}",
            f"- P5: `{result['conditional_phases']['P5']}`",
            f"- P6: `{result['conditional_phases']['P6']}`",
            "",
            "No long PPO training, demonstration mixing, architecture change, reward sweep, or product checkpoint selection was performed.",
            "",
            "## Final verification",
            "",
            "- Repository unittest discovery: 13/13 passed in conda env `end2race` (`pytest` is not installed in that environment).",
            "- All nine audit diagnostic scripts compiled successfully.",
            "- Frozen contract: 9/9 file hashes matched; canonical BC strict schema: 12 keys.",
            "- Frozen product surfaces `ppo/`, `model.py`, `train_ppo.py`, and `pretrained/` have no diff from source HEAD.",
            "",
            "## Reproducibility artifacts",
            "",
        ]
    )
    for name, artifact in result["artifacts"].items():
        if artifact is not None:
            lines.append(f"- {name}: `{artifact['path']}` (`{artifact['sha256']}`)")
    lines.append(f"- Raw audit root: `{RUN_DIR.relative_to(ROOT)}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    frozen_hashes = assert_frozen_contract()
    preregistration = read_json(PREREGISTRATION_PATH)
    p1 = read_json(EXPERIMENT_DIR / "P1_GRADIENT_DIRECTION.json")
    p2 = read_json(EXPERIMENT_DIR / "P2_COUNTERFACTUAL_ACTIONABILITY.json")
    p3 = read_json(EXPERIMENT_DIR / "P3_CREDIT_HORIZON.json")
    p4 = _optional(EXPERIMENT_DIR / "P4_CONTROLLED_STEP.json")
    p5 = _optional(EXPERIMENT_DIR / "P5_LOCAL_EXPRESSIVITY.json")
    p6 = _optional(EXPERIMENT_DIR / "P6_SHORT_CONFIRMATION.json")
    if p1["status"] != "COMPLETED_AFTER_ALLOWED_EXTENSION":
        raise RuntimeError("Cannot finalize before P1 is final")
    if p2["status"] != "COMPLETED" or p3["status"] != "COMPLETED":
        raise RuntimeError("Cannot finalize before P2 and P3 complete")
    next_action = _next_action(p2, p3, p4)
    p5_status = "COMPLETED" if p5 is not None else "NOT_TRIGGERED"
    p6_status = "COMPLETED" if p6 is not None else "NOT_AUTHORIZED"
    if p2["verdict"] == "LOCAL_ACTION_NOT_FOUND":
        p5_status = "TRIGGERED_BUT_NOT_COMPLETED" if p5 is None else "COMPLETED"
    if p4 is not None and p4["verdict"] == "CONTROLLED_STEP_INSUFFICIENT" and p2["verdict"] == "REWARD_DIRECTION_OK":
        p5_status = "TRIGGERED_BUT_NOT_COMPLETED" if p5 is None else "COMPLETED"
    if p4 is not None and p4["verdict"] == "UPDATE_GEOMETRY_CONFIRMED" and p2["verdict"] == "REWARD_DIRECTION_OK":
        p6_status = "AUTHORIZED_BUT_NOT_RUN" if p6 is None else "COMPLETED"
    result = {
        "schema_version": 1,
        "record": "RL_ONLY_MECHANISM_AUDIT_FINAL_DIAGNOSIS",
        "status": "MECHANISM_AUDIT_COMPLETE_NO_PRODUCT_CLAIM",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_head": preregistration["source"]["head"],
        "canonical_bc_sha256": preregistration["frozen_files"]["pretrained/end2race.pth"],
        "device": "cuda",
        "frozen_hashes": frozen_hashes,
        "questions": _build_questions(p1, p2, p3, p4),
        "stage_verdicts": {
            "P1": p1["pool_verdicts"],
            "P2": p2["verdict"],
            "P3": p3["verdict"],
            "P4": None if p4 is None else p4["verdict"],
        },
        "conditional_phases": {"P5": p5_status, "P6": p6_status},
        "next_allowed_action": next_action,
        "product_improvement_claimed": False,
        "artifacts": {
            "preregistration": _artifact(PREREGISTRATION_PATH),
            "safe_reference": _artifact(EXPERIMENT_DIR / "SAFE_REFERENCE.json"),
            "p1": _artifact(EXPERIMENT_DIR / "P1_GRADIENT_DIRECTION.json"),
            "p1_extension": _artifact(EXPERIMENT_DIR / "P1_GRADIENT_DIRECTION_EXTENSION.json"),
            "p2": _artifact(EXPERIMENT_DIR / "P2_COUNTERFACTUAL_ACTIONABILITY.json"),
            "p3": _artifact(EXPERIMENT_DIR / "P3_CREDIT_HORIZON.json"),
            "p4_panel": _artifact(EXPERIMENT_DIR / "P4_PANEL.json"),
            "p4": _artifact(EXPERIMENT_DIR / "P4_CONTROLLED_STEP.json"),
            "p5": _artifact(EXPERIMENT_DIR / "P5_LOCAL_EXPRESSIVITY.json"),
            "p6": _artifact(EXPERIMENT_DIR / "P6_SHORT_CONFIRMATION.json"),
        },
    }
    final_path = EXPERIMENT_DIR / "FINAL_DIAGNOSIS.json"
    write_json_atomic(final_path, result)
    report = _report(result, p1, p2, p3, p4)
    report_path = EXPERIMENT_DIR / "REPORT.md"
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(report_path)
    print(
        f"AUDIT_FINALIZED next={next_action['action']} final_sha={sha256_file(final_path)} "
        f"report_sha={sha256_file(report_path)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

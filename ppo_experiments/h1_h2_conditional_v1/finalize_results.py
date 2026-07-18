#!/usr/bin/env python3
"""Finalize the preregistered H1/H2 conditional-exploration evidence."""

from __future__ import annotations

import hashlib
import json
import shlex
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RUNS = ROOT / "runs/ppo/h1_h2_conditional_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def rate_with_ci(evaluation: dict[str, Any], prefix: str) -> str:
    interval = evaluation[f"{prefix}_rate_wilson_95"]
    return (
        f"{pct(evaluation[f'{prefix}_rate'])} "
        f"[{pct(interval[0])}, {pct(interval[1])}]"
    )


def evaluation_row(arm: dict[str, Any]) -> str:
    evaluation = arm["evaluation"]
    return " | ".join(
        [
            str(arm["arm_id"]),
            str(arm["seed"]),
            f"U{arm['update']}",
            str(evaluation["collision"]),
            str(evaluation["follow"]),
            str(evaluation["overtake"]),
            str(evaluation["fixed_collision"]),
            str(evaluation["new_collision"]),
            str(evaluation["delta_collision"]),
            rate_with_ci(evaluation, "repair"),
            rate_with_ci(evaluation, "damage"),
            "PASS" if arm.get("screen_legal", True) else "FAIL",
        ]
    )


def classification_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows = document["rows"]
    assert len(rows) == 199
    assert len({row["base_id"] for row in rows}) == 199
    for row in rows:
        deterministic = row["deterministic_8s"]
        assert deterministic["status"] == "COMPLETE"
        assert deterministic["preflight_valid"] is True
        assert deterministic["action_finite"] is True
        assert deterministic["observation_finite"] is True
        assert row["trial_count"] in {0, 4, 8}
        assert row["trial_count"] == len(row["trials"])
        assert row["collision_count"] + row["safe_count"] == row["trial_count"]
        for trial in row["trials"]:
            assert trial["status"] == "COMPLETE"
            assert trial["action_finite"] is True
            assert trial["observation_finite"] is True
            assert trial["seed"] in {
                20260722,
                20260723,
                20260724,
                20260725,
                20260726,
                20260727,
                20260728,
                20260729,
            }
    return rows


def validate_full600(raw_path: Path) -> None:
    document = load_json(raw_path)
    rows = document["rows"]
    assert document["complete"] is True
    assert document["summary"]["total"] == 600
    assert document["summary"]["error"] == 0
    assert len(rows) == 600
    assert len({row["scenario_id"] for row in rows}) == 600
    assert all(row["action_finite"] for row in rows)
    assert all(row["observation_finite"] for row in rows)


def main() -> None:
    source_hashes = load_json(OUTPUT / "SOURCE_HASHES.json")
    support = load_json(OUTPUT / "SUPPORT_VALIDATION.json")
    posthoc = load_json(OUTPUT / "POSTHOC_A1_U2_FULL600.json")
    screen = load_json(OUTPUT / "H1_SCREEN_RESULTS.json")
    screen_selection = load_json(OUTPUT / "H1_SCREEN_SELECTION.json")
    retention = load_json(OUTPUT / "H1_RETENTION_RESULTS.json")
    selected = load_json(OUTPUT / "H1_SELECTED_CONFIG.json")
    repeatability = load_json(OUTPUT / "H1_REPEATABILITY_RESULTS.json")
    h2_base = load_json(OUTPUT / "H2_MATCHED_BASE.json")
    h2_i7 = load_json(OUTPUT / "H2_I7_CLASSIFICATION.json")
    h2_i8 = load_json(OUTPUT / "H2_I8_CLASSIFICATION.json")
    h2_contrast = load_json(OUTPUT / "H2_MATCHED_CONTRAST.json")
    h2_summary = load_json(OUTPUT / "H2_MATCHED_CONTRAST_SUMMARY.json")
    h2_gate = load_json(OUTPUT / "H2_CONDITIONAL_EXPLORATION_GATE.json")
    commands = load_json(OUTPUT / "COMMANDS.json")

    assert len(screen["arms"]) == 4
    assert {arm["update"] for arm in screen["arms"]} == {1}
    assert {arm["transitions"] for arm in screen["arms"]} == {25600}
    assert len(retention["arms"]) == 2
    assert {arm["update"] for arm in retention["arms"]} == {2}
    assert {arm["transitions"] for arm in retention["arms"]} == {51200}
    assert len(repeatability["seeds"]) == 3
    assert {arm["seed"] for arm in repeatability["seeds"]} == {
        20260719,
        20260720,
        20260721,
    }
    assert all(arm["update"] == 2 for arm in repeatability["seeds"])
    assert repeatability["H1_PRODUCT_DIRECTION_REPEATS"] is True
    assert repeatability["H1_SHORT_TRAINING_CONTINUES"] is False

    raw_evaluations = sorted(OUTPUT.glob("h1_u*_eval_*.json"))
    assert len(raw_evaluations) == 8
    for raw_path in raw_evaluations:
        validate_full600(raw_path)
    validate_full600(OUTPUT / "posthoc_a1_u2_full600_raw.json")

    i7_rows = classification_rows(h2_i7)
    i8_rows = classification_rows(h2_i8)
    assert h2_base["source_interval_8_base_count"] == 199
    assert h2_summary["primary_matched_count"] == 1
    assert h2_summary["fallback_matched_count"] == 2
    assert h2_summary["selected_matched_count"] == 0
    assert h2_contrast["selected_base_count"] == 0
    assert h2_gate["H2_CONDITIONAL_POOL_VALID"] is False
    assert h2_gate["status"] == "NOT_RUN_H2_MATCHED_POOL_TOO_SMALL"

    for base in h2_base["bases"]:
        source = base["source"]
        changed_i7 = {
            key
            for key in set(source) | set(base["I7"])
            if source.get(key) != base["I7"].get(key)
        }
        changed_i8 = {
            key
            for key in set(source) | set(base["I8"])
            if source.get(key) != base["I8"].get(key)
        }
        assert changed_i7 == {"scenario_id", "interval_idx", "opp_idx"}
        assert changed_i8 == {"scenario_id"}

    for relative, expected in support["post_implementation_source_hashes"].items():
        assert sha256(ROOT / relative) == expected, f"source hash drift: {relative}"

    not_run_status = "NOT_RUN_H2_MATCHED_POOL_TOO_SMALL"
    common_not_run = {
        "gate": "ppo_experiments/h1_h2_conditional_v1/H2_CONDITIONAL_EXPLORATION_GATE.json",
        "gate_status": not_run_status,
        "reason": "selected matched base count 0 is below the preregistered minimum 24",
        "schema_version": 1,
        "selected_matched_base_count": 0,
        "status": not_run_status,
    }
    write_json(
        OUTPUT / "H2_SCREEN_RESULTS.json",
        {**common_not_run, "arms": [], "stage": "H2_SCREEN"},
    )
    write_json(
        OUTPUT / "H2_SCREEN_SELECTION.json",
        {
            **common_not_run,
            "H2_CONTROL_SCREEN_WINNER": None,
            "H2_PAIRED_SCREEN_WINNER": None,
            "stage": "H2_SCREEN_SELECTION",
        },
    )
    write_json(
        OUTPUT / "H2_RETENTION_RESULTS.json",
        {
            **common_not_run,
            "H2_SHORT_RETENTION_SUPPORTED": False,
            "arms": [],
            "stage": "H2_RETENTION",
        },
    )
    write_json(
        OUTPUT / "H2_REPEATABILITY_RESULTS.json",
        {
            **common_not_run,
            "H2_PRODUCT_DIRECTION_REPEATS": False,
            "seeds": [],
            "stage": "H2_REPEATABILITY",
        },
    )

    evaluated_checkpoints = {
        arm["checkpoint"]
        for arm in screen["arms"] + retention["arms"] + repeatability["seeds"]
    }
    checkpoint_lines = [
        "stage\tconfig\tseed\tupdate\ttransitions\tcheckpoint\tsha256\tlocal_exists\tsha256_verified\tevaluated_full600"
    ]
    checkpoint_count = 0
    for manifest_path in sorted(RUNS.glob("*/checkpoint_manifest.json")):
        manifest = load_json(manifest_path)
        run_dir = manifest_path.parent
        for record in manifest["checkpoints"]:
            checkpoint = run_dir / record["path"]
            relative = checkpoint.relative_to(ROOT).as_posix()
            exists = checkpoint.is_file()
            verified = exists and sha256(checkpoint) == record["sha256"]
            checkpoint_lines.append(
                "\t".join(
                    [
                        "H1",
                        manifest["config"],
                        str(manifest["seed"]),
                        str(record["update"]),
                        str(record["update"] * 25600),
                        relative,
                        record["sha256"],
                        str(exists).lower(),
                        str(verified).lower(),
                        str(relative in evaluated_checkpoints).lower(),
                    ]
                )
            )
            assert verified
            checkpoint_count += 1
    assert checkpoint_count == 10
    (OUTPUT / "GLOBAL_CHECKPOINTS.tsv").write_text(
        "\n".join(checkpoint_lines) + "\n", encoding="utf-8"
    )

    verdict = {
        "forward_config": selected["H1_SELECTED_CONFIG"],
        "h1_early_preferred": False,
        "h1_full_preferred": selected["pool_preference"] == "H1_FULL_PREFERRED",
        "h1_product_direction_repeats": repeatability[
            "H1_PRODUCT_DIRECTION_REPEATS"
        ],
        "h1_ratio_25_supported": False,
        "h1_short_retention_supported": repeatability[
            "H1_SHORT_TRAINING_CONTINUES"
        ],
        "h2_4s_supported": False,
        "h2_conditional_pool_valid": h2_gate["H2_CONDITIONAL_POOL_VALID"],
        "h2_interval7_supported": False,
        "h2_paired_sampling_supported": False,
        "h2_product_direction_repeats": False,
        "h2_short_retention_supported": False,
        "overall_status": "FORWARD_SIGNAL",
        "schema_version": 1,
        "status_reasons": {
            "h1": "N1-H1F-p50 passed the registered product gate on 2 of 3 seeds; median collision=19 and median deltaC=-3",
            "h2": "primary matched count=1 and fallback matched count=2, both below 24; H2 training was not run",
        },
    }
    write_json(OUTPUT / "FINAL_VERDICT.json", verdict)

    metric_rows: list[dict[str, Any]] = []
    stability_lines = [
        "| Run | Updates | Max KL | Max clip fraction | Frozen actor max delta | log_std max delta | Optimizer steps |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for metrics_path in sorted(RUNS.glob("*/training_metrics.jsonl")):
        updates = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metric_rows.extend(updates)
        max_kl = max(update["approx_kl"] for update in updates)
        max_clip = max(update["clip_fraction"] for update in updates)
        max_frozen = max(
            update["actor_delta_from_bc"]["frozen_actor"]["max_abs_delta_from_bc"]
            for update in updates
        )
        max_log_std = max(
            update["actor_delta_from_bc"]["log_std_max_abs_delta_from_initial"]
            for update in updates
        )
        optimizer = ", ".join(
            f"U{update['update']} {update['actual_optimizer_steps']}/{update['planned_optimizer_steps']}"
            for update in updates
        )
        assert max_kl <= 0.05
        assert max_clip <= 0.50
        assert max_frozen == 0.0
        assert max_log_std == 0.0
        assert all(
            update["actual_optimizer_steps"] == update["planned_optimizer_steps"]
            for update in updates
        )
        stability_lines.append(
            f"| {metrics_path.parent.name} | {len(updates)} | {max_kl:.9f} | "
            f"{max_clip:.9f} | {max_frozen:.1f} | {max_log_std:.1f} | {optimizer} |"
        )

    selected_evaluations = repeatability["seeds"]
    assert sum(
        evaluation["evaluation"]["collision"] <= 21
        and evaluation["evaluation"]["fixed_collision"]
        > evaluation["evaluation"]["new_collision"]
        and evaluation["evaluation"]["overtake"] >= 328
        for evaluation in selected_evaluations
    ) == 2
    assert statistics.median(
        evaluation["evaluation"]["collision"] for evaluation in selected_evaluations
    ) == 19
    assert statistics.median(
        evaluation["evaluation"]["delta_collision"]
        for evaluation in selected_evaluations
    ) == -3

    post = posthoc["candidate_summary"]
    post_pair = posthoc["paired"]
    report: list[str] = [
        "# H1/H2 Conditional Exploration Final Report",
        "",
        "## Registered outcome",
        "",
        f"- Overall status: `{verdict['overall_status']}`.",
        f"- Forward config: `{verdict['forward_config']}`.",
        "- H1: full-pool 50% arm selected; product direction passed 2/3 seeds; short U1-to-U2 retention did not pass.",
        "- H2: matched pool stopped at the registered size gate (primary 1, fallback 2, required 24); no H2 arm, seed, checkpoint, or paired-training telemetry was created.",
        "- This result is an experiment forward signal, not a deployment or held-out-performance claim.",
        "",
        "## Preflight and baseline",
        "",
        f"- Starting repository reference: `{source_hashes['git_head']}`.",
        f"- Formal preflight commit: `{load_json(OUTPUT / 'FORMAL_PREFLIGHT.json')['git_head']}`.",
        f"- Support validation: `{support['status']}`; strict BC checkpoint load: `{support['canonical_checkpoint']['strict_load']}`.",
        "- Current CPU full-600 BC: collision 22, follow 233, overtake 345, error 0, 600 unique scenarios.",
        "- Formal evaluation contract: CPU, ego collision scope, 8 persistent workers, one Torch thread per worker.",
        "",
        "## Post-hoc A1 diagnostic",
        "",
        "The existing A1 update-2 checkpoint was evaluated once and was not eligible for formal selection.",
        "",
        "| Seed | Update | Collision | Follow | Overtake | Fixed | New | deltaC | Repair rate [Wilson 95%] | Damage rate [Wilson 95%] |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        f"| {posthoc['seed']} | {posthoc['update']} | {post['collision']} | {post['follow']} | {post['overtake']} | {post_pair['fixed_collision']} | {post_pair['new_collision']} | {post_pair['delta_collision']} | {rate_with_ci(post_pair, 'repair')} | {rate_with_ci(post_pair, 'damage')} |",
        "",
        "## H1 U1 screen",
        "",
        "| Arm | Seed | Update | Collision | Follow | Overtake | Fixed | New | deltaC | Repair rate [Wilson 95%] | Damage rate [Wilson 95%] | Legal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    report.extend(f"| {evaluation_row(arm)} |" for arm in screen["arms"])
    report.extend(
        [
            "",
            f"Full winner: `{screen_selection['H1_FULL_SCREEN_WINNER']}`. Early winner: `{screen_selection['H1_EARLY_SCREEN_WINNER']}`. The full-pool 25% ratio was not supported; the early-pool ratio comparison was inconclusive.",
            "",
            "## H1 U2 retention",
            "",
            "| Arm | Seed | Update | Collision | Follow | Overtake | Fixed | New | deltaC | Repair rate [Wilson 95%] | Damage rate [Wilson 95%] | Legal |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    report.extend(f"| {evaluation_row(arm)} |" for arm in retention["arms"])
    report.extend(
        [
            "",
            f"Pool preference: `{selected['pool_preference']}`. Selected config: `{selected['H1_SELECTED_CONFIG']}`. Selected-arm short retention: `{repeatability['H1_SHORT_TRAINING_CONTINUES']}`.",
            "",
            "## H1 repeatability",
            "",
            "| Arm | Seed | Update | Collision | Follow | Overtake | Fixed | New | deltaC | Repair rate [Wilson 95%] | Damage rate [Wilson 95%] | Legal | Product gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for arm in repeatability["seeds"]:
        product = repeatability["product_pass_by_seed"][str(arm["seed"])]
        report.append(f"| {evaluation_row(arm)} | {product} |")
    report.extend(
        [
            "",
            f"Registered repeat result: `{repeatability['H1_PRODUCT_DIRECTION_REPEATS']}` (2/3 product-pass seeds, median collision {repeatability['median_collision']}, median deltaC {repeatability['median_delta_collision']}).",
            "",
            "## H2 matched-pool reconstruction",
            "",
            f"- Source interval-8 bases: {h2_summary['source_base_count']}.",
            f"- Preflight valid/invalid: I7 {h2_summary['preflight']['I7']['valid']}/{h2_summary['preflight']['I7']['invalid']}; I8 {h2_summary['preflight']['I8']['valid']}/{h2_summary['preflight']['I8']['invalid']}.",
            f"- Deterministic 8-second safe: I7 {h2_summary['deterministic_safe_counts']['I7']}; I8 {h2_summary['deterministic_safe_counts']['I8']}.",
            f"- Stochastic trial-count distribution: I7 {dict(sorted(Counter(row['trial_count'] for row in i7_rows).items()))}; I8 {dict(sorted(Counter(row['trial_count'] for row in i8_rows).items()))}.",
            f"- Collision K=0..8 distribution: I7 {h2_summary['collision_count_distributions']['I7']}; I8 {h2_summary['collision_count_distributions']['I8']}.",
            f"- Collision-time distribution (seconds): I7 {h2_summary['collision_time_distributions_s']['I7']}; I8 {h2_summary['collision_time_distributions_s']['I8']}.",
            f"- Primary matched: {h2_summary['primary_matched_count']}; fallback matched: {h2_summary['fallback_matched_count']}; selected: {h2_summary['selected_matched_count']}.",
            f"- Selected tier/status: `{h2_summary['selected_tier']}` / `{h2_summary['status']}`.",
            f"- Matched manifest SHA-256: `{h2_summary['matched_manifest_hash']}`.",
            "",
            "## H2 conditional gate and training stages",
            "",
            "| Stage | Status | Arms | Checkpoints | Evaluations |",
            "|---|---|---:|---:|---:|",
            f"| Conditional exploration gate | {h2_gate['status']} | 0 | 0 | 0 |",
            f"| Screen | {not_run_status} | 0 | 0 | 0 |",
            f"| Selection | {not_run_status} | 0 | 0 | 0 |",
            f"| Retention | {not_run_status} | 0 | 0 | 0 |",
            f"| Repeatability | {not_run_status} | 0 | 0 | 0 |",
            "",
            "No discordant-pair mechanism table can be computed because the registered matched-pool size gate selected zero bases. The eight-trial classification rows were retained; unmatched I7/I8 pools were not used.",
            "",
            "## Training stability diagnostics",
            "",
        ]
    )
    report.extend(stability_lines)
    report.extend(
        [
            "",
            f"Across {len(metric_rows)} recorded H1 updates, max KL was {max(row['approx_kl'] for row in metric_rows):.9f}, max clip fraction was {max(row['clip_fraction'] for row in metric_rows):.9f}, frozen-actor delta was zero, log_std delta was zero, and every optimizer-step count matched the plan.",
            "",
            "## Source and manifest hashes",
            "",
            "### Starting reference hashes",
            "",
            "| Path | SHA-256 |",
            "|---|---|",
        ]
    )
    report.extend(
        f"| `{path}` | `{digest}` |"
        for path, digest in sorted(source_hashes["files"].items())
    )
    report.extend(
        [
            "",
            "### Post-implementation frozen hashes",
            "",
            "| Path | SHA-256 |",
            "|---|---|",
        ]
    )
    report.extend(
        f"| `{path}` | `{digest}` |"
        for path, digest in sorted(
            support["post_implementation_source_hashes"].items()
        )
    )
    report.extend(
        [
            "",
            f"H2 source manifest SHA-256: `{h2_base['source_manifest_hash']}`. H2 generated base manifest SHA-256: `{h2_base['manifest_hash']}`. H2 selected matched manifest SHA-256: `{h2_contrast['manifest_hash']}`.",
            "",
            "All 10 H1 checkpoint paths, hashes, local-presence checks, hash verification results, and evaluation status are in `GLOBAL_CHECKPOINTS.tsv`. Checkpoint binaries remain local.",
            "",
            "## Exact recorded commands",
            "",
        ]
    )
    for index, command in enumerate(commands["commands"], start=1):
        report.extend(
            [
                f"{index}. {command['purpose']}",
                "",
                "```sh",
                shlex.join(command["argv"]),
                "```",
                "",
            ]
        )
    report.extend(
        [
            "## Completion inventory",
            "",
            "All registered stage outputs exist. Every started H1 arm reached its registered terminal update; all eight formal H1 evaluations and the post-hoc diagnostic have 600 unique rows, zero errors, and finite observations/actions. H2 reached its registered pool-size kill gate without starting training. Final process and Git-worktree checks are performed after committing this report.",
            "",
        ]
    )
    (OUTPUT / "FINAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()

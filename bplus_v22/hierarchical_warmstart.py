"""Replacement Task-6 fit/calibration for the hierarchical residual policy.

This module is deliberately separate from :mod:`bplus_v22.warmstart`.  The
single-gate Task-6 releases are immutable historical evidence; none of their
schemas, manifests, checkpoints, or acceptance decisions are reused here.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from bplus_v22 import (
    ACTION_CORE_LR,
    ARMS,
    BC_CHECKPOINT_SHA256,
    BRAKE_BUDGET,
    INITIAL_BRAKE_LOGIT,
    OWNER_DECISION,
    SEED as POLICY_SEED,
    SIDECAR_FINETUNE_LR,
    STEER_BUDGET,
)
from bplus_v22.release import (
    file_sha256,
    validate_pinned_inputs,
    validate_source_preflight,
)
from bplus_v22.remediated_model import (
    ACTION_SCHEMA,
    CHECKPOINT_SCHEMA,
    INITIAL_INTERVENTION_LOGIT,
    RemediatedV22Policy,
    apply_intervention_logit_offset,
    initialize_hierarchical_priors,
)
from bplus_v22.hierarchical_identity import validate_hierarchical_identity
from bplus_v22.sidecar import (
    DATASET_RELPATH,
    REGISTRY_RELPATH,
    SPLIT_RELPATH,
    _episode_rows as d2_episode_rows,
    _tensor_digest,
    _validate_output_manifest,
    _write_json,
    _write_output_manifest,
    load_sidecar_bundle,
    validate_sidecar_release,
)
from bplus_v22.warmstart import (
    ACTOR_INPUT_FIELDS,
    D01_RELPATH,
    D01_OUTPUT_MANIFEST_SHA256,
    EVIDENCE_RELPATH,
    FORBIDDEN_ACTOR_FIELDS,
    SIDECAR_RELEASE_RELPATH,
    SIDECAR_OUTPUT_MANIFEST_SHA256,
    SIDECAR_STATE_DICT_SHA256,
    WarmstartBatchProvider,
    _policy_hashes as legacy_policy_hashes,
    _read_tsv,
    _set_deterministic_cuda,
    _torch_batch,
)
from d0.identity import (
    REGISTRY_FIELDS,
    append_opened_registry,
    registry_row_id,
    validate_registry_row,
)


MANIFEST_SCHEMA = "bplus-v2.2-hierarchical-warmstart-manifest-1"
MANIFEST_VALIDATION_SCHEMA = (
    "bplus-v2.2-hierarchical-warmstart-manifest-validation-1"
)
RELEASE_SCHEMA = "bplus-v2.2-hierarchical-warmstart-release-1"
REPORT_SCHEMA = "bplus-v2.2-hierarchical-warmstart-arm-report-1"
VALIDATION_SCHEMA = "bplus-v2.2-hierarchical-warmstart-validation-1"
SCORE_SCHEMA = "bplus-v2.2-hierarchical-calibration-scores-1"
RELEASE_LABEL = "HIERARCHICAL_ACTION_WARMSTART_REMEDIATION"

OLD_MANIFEST_RELPATH = (
    "Experiments/B1_route_r2_scaffold/artifacts/"
    "warmstart_manifest_20260712_091851"
)
OLD_EPISODES_SHA256 = (
    "baaa916db54364308458413d81c26e52b31585c07d1eecf1c8f5c1a8ca0bda20"
)
OLD_EXAMPLES_SHA256 = (
    "01043ad1b02a4948b51140944b7f8736b493e02bf689be9e99b90454f7983f93"
)
CANONICAL_EPISODES_RELPATH = f"{D01_RELPATH}/canonical_episodes.tsv"
CANONICAL_EPISODES_SHA256 = (
    "793193deefc942f556ec23ee4e34fea3597eac761eb0b1f676af2667ff6b62e2"
)
TASK8_RELPATH = (
    "Experiments/B1_route_r2_scaffold/artifacts/"
    "task8_manifests_20260712_113241/development_scenarios.tsv"
)
TASK8_SHA256 = (
    "8ff0d96b91aac134ab006e70900785c13c345dcb544867740aa8dd57072dfc46"
)
SPLIT_SHA256 = (
    "2f8146d7be0e36c3abcc084dcdbfa9e3df85983c37c6249294ab19b1431c49f3"
)
FAILED_TASK10_RELPATH = (
    "Experiments/B1_route_r2_scaffold/artifacts/"
    "task10_warmstart_20260712_105740"
)
FAILED_TASK10_OUTPUT_SHA256 = (
    "605d3413df35cef8ddd9cdd4769164f52016edeaa7c9e58e1c34ba234fb9ed46"
)

FIT_FOLDS = (0, 1, 2, 3)
CALIBRATION_FOLD = 4
EXPECTED_FIT_WITNESS_EPISODES = 58
EXPECTED_FIT_WITNESS_L4 = 47
EXPECTED_FIT_PRESERVATION_EPISODES = 484
EXPECTED_FIT_EPISODES = 542
EXPECTED_FIT_INTERVENTION = 252
EXPECTED_FIT_BRAKE = 175
EXPECTED_FIT_WITNESS_NOOP = 4446
EXPECTED_FIT_PRESERVATION_NOOP = 39204
EXPECTED_FIT_MACROS = 43902
EXPECTED_CAL_POSITIVE_EPISODES = 9
EXPECTED_CAL_POSITIVE_L4 = 7
EXPECTED_CAL_POSITIVE_INTERVENTION = 39
EXPECTED_CAL_STEER_ONLY_MACROS = 14
EXPECTED_CAL_BRAKE_MACROS = 25
EXPECTED_CAL_STEER_ONLY_EPISODES = 4
EXPECTED_CAL_BRAKE_EPISODES = 5
EXPECTED_CAL_NEGATIVE_EPISODES = 75
EXPECTED_CAL_NEGATIVE_L4 = 31
EXPECTED_CAL_NEGATIVE_MACROS = 6075

WARMSTART_SEED = 20260712
WARMSTART_UPDATES = 1024
WARMSTART_BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
FIT_POS_WEIGHT = (EXPECTED_FIT_MACROS - EXPECTED_FIT_INTERVENTION) / float(
    EXPECTED_FIT_INTERVENTION
)
MAX_FALSE_INTERVENTION_EPISODES = 7
REGISTRY_STAGE = "D3-R2-v2.2"
REGISTRY_OPENED_AT = "2026-07-12T15:27:13+08:00"
REGISTRY_USE_CLASS = "actor_pretrain"
REGISTRY_SPLIT_ID = "d3r2_v22_hierarchical_calibration_negative"
REGISTRY_DECISION_EFFECT = "action_choice"
REGISTRY_SOURCE_RUN_ID = "d01_full_reconcile_20260711_170200_a"

SCHEDULE_DOMAIN = b"end2race:bplus-v2.2:hierarchical-natural-cycle:v1\0"
CALIBRATION_TIE_DOMAIN = b"end2race:bplus-v2.2:hierarchical-calibration-tie:v1\0"
CALIBRATION_EXAMPLE_DOMAIN = b"end2race:bplus-v2.2:hierarchical-cal-example:v1\0"

EPISODE_FIELDS = (
    "episode_order",
    "partition",
    "role",
    "l2_id",
    "l3_id",
    "l4_id",
    "map_name",
    "skill",
    "opponent_raceline",
    "speedscale_hex",
    "resolved_ego_idx",
    "source_npz_relpath",
    "source_npz_sha256",
    "source_trajectory_sha256",
    "frame_start_global",
    "frame_count",
    "witness_branch_id",
    "intervention_start_step",
    "intervention_duration_steps",
    "target_brake_hex",
    "target_steer_hex",
    "confirmed_safe_pass",
    "action_clipped",
    "preservation_stratum",
)

EXAMPLE_FIELDS = (
    "example_index",
    "partition",
    "example_id",
    "role",
    "l2_id",
    "macro_index",
    "frame_index",
    "global_frame_index",
    "active_intervention",
    "target_brake_gate",
    "target_brake_hex",
    "target_steer_hex",
)

CALIBRATION_EPISODE_FIELDS = (
    "arm",
    "partition",
    "l2_id",
    "action_category",
    "macro_count",
    "raw_max_logit_hex",
    "applied_offset_hex",
    "adjusted_max_logit_hex",
    "intervention_decision",
    "tie_hash",
)

MANIFEST_CONFIG_KEYS = {
    "schema",
    "created_at",
    "owner_decision",
    "action_schema",
    "checkpoint_schema",
    "test_opened",
    "final_pool",
    "ppo_training_started",
    "arm_selection_performed",
    "fit_folds",
    "calibration_fold",
    "counts",
    "priors",
    "schedule",
    "loss_masks",
    "calibration",
    "source_hashes",
    "source_preflight_relpath",
    "source_preflight_output_sha256",
    "registry",
    "hierarchical_identity_relpath",
    "hierarchical_identity_output_sha256",
    "actor_input_fields",
    "forbidden_actor_fields",
}


def _write_tsv(
    path: Path, rows: Iterable[Mapping[str, str]], fields: tuple[str, ...]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            if set(row) != set(fields):
                raise ValueError(f"hierarchical warm-start TSV schema drift: {path.name}")
            writer.writerow({field: row[field] for field in fields})


def _prepare_output(output: Path) -> Path:
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("hierarchical warm-start output/partial already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    return partial


def _promote(partial: Path, output: Path) -> None:
    os.replace(partial, output)
    (output / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")


def _hash(domain: bytes, *parts: str) -> str:
    digest = hashlib.sha256(domain)
    for part in parts:
        digest.update(str(part).encode("utf-8") + b"\0")
    return digest.hexdigest()


def _validate_source_hashes(root: Path) -> dict[str, str]:
    paths = {
        "old_episodes": root / OLD_MANIFEST_RELPATH / "episodes.tsv",
        "old_macro_examples": root / OLD_MANIFEST_RELPATH / "macro_examples.tsv",
        "canonical_episodes": root / CANONICAL_EPISODES_RELPATH,
        "task8_development": root / TASK8_RELPATH,
        "scenario_split": root / SPLIT_RELPATH / "scenario_split.tsv",
        "failed_task10_output_manifest": root
        / FAILED_TASK10_RELPATH
        / "output_manifest.sha256",
        "bc_checkpoint": root / "pretrained/end2race.pth",
        "sidecar_output_manifest": root
        / SIDECAR_RELEASE_RELPATH
        / "output_manifest.sha256",
    }
    expected = {
        "old_episodes": OLD_EPISODES_SHA256,
        "old_macro_examples": OLD_EXAMPLES_SHA256,
        "canonical_episodes": CANONICAL_EPISODES_SHA256,
        "task8_development": TASK8_SHA256,
        "scenario_split": SPLIT_SHA256,
        "failed_task10_output_manifest": FAILED_TASK10_OUTPUT_SHA256,
        "bc_checkpoint": BC_CHECKPOINT_SHA256,
        "sidecar_output_manifest": SIDECAR_OUTPUT_MANIFEST_SHA256,
    }
    observed = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"hierarchical warm-start source missing: {path}")
        observed[name] = file_sha256(path)
        if observed[name] != expected[name]:
            raise ValueError(f"hierarchical warm-start source hash drift: {name}")
    _validate_output_manifest(root / OLD_MANIFEST_RELPATH)
    _validate_output_manifest(root / FAILED_TASK10_RELPATH)
    sidecar_validation = validate_sidecar_release(root / SIDECAR_RELEASE_RELPATH, root)
    if (
        not sidecar_validation["passed"]
        or sidecar_validation["state_dict_sha256"] != SIDECAR_STATE_DICT_SHA256
    ):
        raise ValueError("hierarchical warm-start sidecar release validation failed")
    sidecar_state, _, _, _ = load_sidecar_bundle(root / SIDECAR_RELEASE_RELPATH)
    observed["sidecar_state_dict"] = _tensor_digest(sidecar_state.items())
    if observed["sidecar_state_dict"] != SIDECAR_STATE_DICT_SHA256:
        raise ValueError("hierarchical warm-start sidecar state digest drift")
    return observed


def _strict_confirmed_pass(row: Mapping[str, str]) -> bool:
    return (
        row["model_id"] == "bc"
        and row["confirmed_safe_pass"] == "True"
        and row["four_state"] == "confirmed_pass"
        and row["corrected_outcome3"] == "overtake"
        and row["collision_any"] == "False"
        and row["censored"] == "False"
        and row["alignment_status"] == "ok"
        and row["physics_status"] == "ok"
        and row["frame_spacing_status"] == "ok"
    )


def _negative_episode_row(
    canonical: Mapping[str, str], metadata: Mapping[str, str]
) -> dict[str, str]:
    return {
        "episode_order": "-1",
        "partition": "calibration_negative",
        "role": "preservation",
        "l2_id": canonical["l2_id"],
        "l3_id": canonical["l3_id"],
        "l4_id": canonical["l4_id"],
        "map_name": canonical["map_name"],
        "skill": canonical["skill"],
        "opponent_raceline": canonical["opponent_raceline"],
        "speedscale_hex": canonical["speedscale_hex"],
        "resolved_ego_idx": metadata["resolved_ego_idx"],
        "source_npz_relpath": metadata["npz_relpath"],
        "source_npz_sha256": metadata["npz_sha256"],
        "source_trajectory_sha256": "NA",
        "frame_start_global": metadata["frame_start"],
        "frame_count": metadata["frame_count"],
        "witness_branch_id": "NA",
        "intervention_start_step": "NA",
        "intervention_duration_steps": "NA",
        "target_brake_hex": float(0.0).hex(),
        "target_steer_hex": float(0.0).hex(),
        "confirmed_safe_pass": "true",
        "action_clipped": "false",
        "preservation_stratum": "calibration_negative|"
        + "|".join(
            (
                canonical["map_name"],
                canonical["skill"],
                canonical["opponent_raceline"],
                canonical["l4_id"],
            )
        ),
    }


def build_hierarchical_episode_manifest(
    repo_root: str | Path,
) -> tuple[list[dict[str, str]], dict]:
    """Build the frozen fold-0--3 fit and fold-4 calibration populations."""

    root = Path(repo_root).resolve()
    _validate_source_hashes(root)
    split = {
        row["l2_id"]: row
        for row in _read_tsv(root / SPLIT_RELPATH / "scenario_split.tsv")
    }
    old = _read_tsv(root / OLD_MANIFEST_RELPATH / "episodes.tsv")
    old_ids = {row["l2_id"] for row in old}
    fit = []
    positive = []
    for source in old:
        row = dict(source)
        fold = int(split[row["l2_id"]]["outer_fold"])
        if fold in FIT_FOLDS:
            row["partition"] = "fit"
            fit.append(row)
        elif fold == CALIBRATION_FOLD and row["role"] == "witness":
            row["partition"] = "calibration_positive"
            positive.append(row)

    metadata = {row["l2_id"]: row for row in d2_episode_rows(root)}
    task8_ids = {
        row["l2_id"] for row in _read_tsv(root / TASK8_RELPATH)
    }
    negative_candidates = [
        row
        for row in _read_tsv(root / CANONICAL_EPISODES_RELPATH)
        if _strict_confirmed_pass(row)
        and row["l2_id"] in split
        and split[row["l2_id"]]["split"] == "non_test"
        and int(split[row["l2_id"]]["outer_fold"]) == CALIBRATION_FOLD
        and row["l2_id"] not in old_ids
        and row["l2_id"] not in task8_ids
    ]
    negative = [
        _negative_episode_row(row, metadata[row["l2_id"]])
        for row in sorted(negative_candidates, key=lambda value: value["l2_id"])
    ]
    episodes = fit + positive + negative
    for index, row in enumerate(episodes):
        row["episode_order"] = str(index)
        if set(row) != set(EPISODE_FIELDS):
            raise ValueError("hierarchical episode row schema mismatch")

    fit_witness = [row for row in fit if row["role"] == "witness"]
    fit_preservation = [row for row in fit if row["role"] == "preservation"]
    expected = (
        len(fit_witness),
        len({row["l4_id"] for row in fit_witness}),
        len(fit_preservation),
        len(fit),
        len(positive),
        len({row["l4_id"] for row in positive}),
        len(negative),
        len({row["l4_id"] for row in negative}),
    )
    if expected != (
        EXPECTED_FIT_WITNESS_EPISODES,
        EXPECTED_FIT_WITNESS_L4,
        EXPECTED_FIT_PRESERVATION_EPISODES,
        EXPECTED_FIT_EPISODES,
        EXPECTED_CAL_POSITIVE_EPISODES,
        EXPECTED_CAL_POSITIVE_L4,
        EXPECTED_CAL_NEGATIVE_EPISODES,
        EXPECTED_CAL_NEGATIVE_L4,
    ):
        raise ValueError(f"hierarchical episode population drift: {expected}")
    if {row["skill"] for row in negative} & {"skill_F"}:
        raise ValueError("fold-4 calibration negatives unexpectedly contain skill_F")
    if len({row["l2_id"] for row in episodes}) != len(episodes):
        raise ValueError("hierarchical episode partitions overlap")
    return episodes, {
        "fit_witness_episodes": len(fit_witness),
        "fit_witness_l4": len({row["l4_id"] for row in fit_witness}),
        "fit_preservation_episodes": len(fit_preservation),
        "fit_episodes": len(fit),
        "calibration_positive_episodes": len(positive),
        "calibration_positive_l4": len({row["l4_id"] for row in positive}),
        "calibration_negative_episodes": len(negative),
        "calibration_negative_l4": len({row["l4_id"] for row in negative}),
        "calibration_negative_skill_f": 0,
        "total_episodes": len(episodes),
    }


def make_calibration_registry_rows(
    episodes: Iterable[Mapping[str, str]]
) -> list[dict[str, str]]:
    rows = []
    for episode in episodes:
        if episode["partition"] != "calibration_negative":
            continue
        row = {
            "registry_schema": "bplus-opened-registry-1",
            "opened_at_utc": REGISTRY_OPENED_AT,
            "stage": REGISTRY_STAGE,
            "use_class": REGISTRY_USE_CLASS,
            "split_id": REGISTRY_SPLIT_ID,
            "l2_id": episode["l2_id"],
            "l3_id": episode["l3_id"],
            "l4_id": episode["l4_id"],
            "map_name": episode["map_name"],
            "source_manifest_sha256": D01_OUTPUT_MANIFEST_SHA256,
            "source_run_id": REGISTRY_SOURCE_RUN_ID,
            "decision_effect": REGISTRY_DECISION_EFFECT,
            "final_pool": "false",
            "evidence_relpath": EVIDENCE_RELPATH,
        }
        row["row_id"] = registry_row_id(row)
        rows.append(validate_registry_row(row))
    rows.sort(key=lambda row: row["row_id"])
    if (
        len(rows) != EXPECTED_CAL_NEGATIVE_EPISODES
        or len({row["row_id"] for row in rows}) != EXPECTED_CAL_NEGATIVE_EPISODES
    ):
        raise ValueError("hierarchical calibration registry row count drift")
    return rows


def _read_registry(path: Path) -> list[dict[str, str]]:
    rows = _read_tsv(path)
    return [validate_registry_row(row) for row in rows]


def _registry_snapshots(
    live_registry: Path, planned: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    live = _read_registry(live_registry)
    live_by_id = {row["row_id"]: row for row in live}
    present = [live_by_id.get(row["row_id"]) for row in planned]
    if all(value is None for value in present):
        before = live
        after = live + planned
        state = "ready"
    elif all(value == expected for value, expected in zip(present, planned)):
        if live[-len(planned) :] != planned:
            raise ValueError(
                "hierarchical calibration registry rows are not the latest append"
            )
        before = live[: -len(planned)]
        after = live
        state = "already_appended"
    else:
        raise ValueError(
            "hierarchical calibration registry has partial/conflicting planned rows"
        )
    if len({row["row_id"] for row in after}) != len(after):
        raise ValueError("hierarchical registry snapshot has duplicate row ID")
    return before, after, state


def _registry_live_state(
    live_registry: Path,
    planned: list[dict[str, str]],
    before_sha256: str,
    after_sha256: str,
) -> str:
    actual_sha = file_sha256(live_registry)
    current = {row["row_id"]: row for row in _read_registry(live_registry)}
    present = [current.get(row["row_id"]) for row in planned]
    if actual_sha == before_sha256 and all(value is None for value in present):
        return "ready"
    if actual_sha == after_sha256 and all(
        value == expected for value, expected in zip(present, planned)
    ):
        return "already_appended"
    raise ValueError(
        "hierarchical calibration registry is neither planned-before nor planned-after"
    )


def _old_examples_by_l2(root: Path) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {}
    for row in _read_tsv(root / OLD_MANIFEST_RELPATH / "macro_examples.tsv"):
        output.setdefault(row["l2_id"], []).append(row)
    return output


def _calibration_negative_examples(episode: Mapping[str, str]) -> list[dict[str, str]]:
    rows = []
    for macro_index, frame_index in enumerate(range(0, int(episode["frame_count"]), 10)):
        rows.append(
            {
                "example_index": "-1",
                "partition": "calibration_negative",
                "example_id": _hash(
                    CALIBRATION_EXAMPLE_DOMAIN, episode["l2_id"], str(frame_index)
                ),
                "role": "preservation_noop",
                "l2_id": episode["l2_id"],
                "macro_index": str(macro_index),
                "frame_index": str(frame_index),
                "global_frame_index": str(
                    int(episode["frame_start_global"]) + frame_index
                ),
                "active_intervention": "false",
                "target_brake_gate": "0",
                "target_brake_hex": float(0.0).hex(),
                "target_steer_hex": float(0.0).hex(),
            }
        )
    return rows


def build_hierarchical_macro_examples(
    repo_root: str | Path, episodes: Iterable[Mapping[str, str]]
) -> tuple[list[dict[str, str]], dict]:
    root = Path(repo_root).resolve()
    old = _old_examples_by_l2(root)
    rows = []
    for episode in episodes:
        if episode["partition"] == "calibration_negative":
            source_rows = _calibration_negative_examples(episode)
        else:
            source_rows = []
            for source in old[episode["l2_id"]]:
                row = dict(source)
                row["partition"] = episode["partition"]
                source_rows.append(row)
        rows.extend(source_rows)
    for index, row in enumerate(rows):
        row["example_index"] = str(index)
        if set(row) != set(EXAMPLE_FIELDS):
            raise ValueError("hierarchical macro example schema mismatch")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("hierarchical macro example IDs are not unique")

    fit = [row for row in rows if row["partition"] == "fit"]
    positive = [
        row
        for row in rows
        if row["partition"] == "calibration_positive"
        and row["active_intervention"] == "true"
    ]
    negative = [row for row in rows if row["partition"] == "calibration_negative"]
    fit_counts = {
        "intervention": sum(row["active_intervention"] == "true" for row in fit),
        "brake": sum(int(row["target_brake_gate"]) for row in fit),
        "witness_noop": sum(row["role"] == "witness_noop" for row in fit),
        "preservation_noop": sum(row["role"] == "preservation_noop" for row in fit),
        "total": len(fit),
    }
    if fit_counts != {
        "intervention": EXPECTED_FIT_INTERVENTION,
        "brake": EXPECTED_FIT_BRAKE,
        "witness_noop": EXPECTED_FIT_WITNESS_NOOP,
        "preservation_noop": EXPECTED_FIT_PRESERVATION_NOOP,
        "total": EXPECTED_FIT_MACROS,
    }:
        raise ValueError(f"hierarchical fit macro accounting drift: {fit_counts}")
    positive_counts = {
        "intervention": len(positive),
        "steer_only": sum(int(row["target_brake_gate"]) == 0 for row in positive),
        "brake": sum(int(row["target_brake_gate"]) == 1 for row in positive),
    }
    if positive_counts != {
        "intervention": EXPECTED_CAL_POSITIVE_INTERVENTION,
        "steer_only": EXPECTED_CAL_STEER_ONLY_MACROS,
        "brake": EXPECTED_CAL_BRAKE_MACROS,
    }:
        raise ValueError(
            f"hierarchical calibration-positive macro drift: {positive_counts}"
        )
    if len(negative) != EXPECTED_CAL_NEGATIVE_MACROS:
        raise ValueError("hierarchical calibration-negative macro count drift")
    return rows, {
        "fit": fit_counts,
        "calibration_positive": positive_counts,
        "calibration_negative": len(negative),
        "total": len(rows),
    }


def build_natural_cycle_schedule(
    examples: list[Mapping[str, str]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    fit = [
        int(row["example_index"])
        for row in examples
        if row["partition"] == "fit"
    ]
    if len(fit) != EXPECTED_FIT_MACROS:
        raise ValueError("hierarchical natural schedule fit population drift")
    ranked = np.asarray(
        sorted(
            fit,
            key=lambda index: (
                _hash(SCHEDULE_DOMAIN, examples[index]["example_id"]),
                examples[index]["example_id"],
            ),
        ),
        dtype=np.int32,
    )
    total = WARMSTART_UPDATES * WARMSTART_BATCH_SIZE
    schedule = np.resize(ranked, total).reshape(
        WARMSTART_UPDATES, WARMSTART_BATCH_SIZE
    )
    fit_indices = np.asarray(fit, dtype=np.int32)
    return schedule, fit_indices, {
        "schema": "bplus-v2.2-hierarchical-natural-cycle-schedule-1",
        "seed": WARMSTART_SEED,
        "ordering": "domain-separated hash(example_id),example_id",
        "outcome_aware_sampling": False,
        "unique_fit_macros": len(ranked),
        "updates": WARMSTART_UPDATES,
        "batch_size": WARMSTART_BATCH_SIZE,
        "scheduled_occurrences": total,
        "complete_cycles": total // len(ranked),
        "partial_cycle_occurrences": total % len(ranked),
    }


def build_hierarchical_priors(examples: list[Mapping[str, str]]) -> dict:
    fit = [row for row in examples if row["partition"] == "fit"]
    intervention = sum(row["active_intervention"] == "true" for row in fit)
    brake = sum(
        int(row["target_brake_gate"])
        for row in fit
        if row["active_intervention"] == "true"
    )
    if (len(fit), intervention, brake) != (
        EXPECTED_FIT_MACROS,
        EXPECTED_FIT_INTERVENTION,
        EXPECTED_FIT_BRAKE,
    ):
        raise ValueError("hierarchical unique-fit prior accounting drift")

    def prior(positive: int, total: int) -> dict:
        prevalence = positive / total
        logit64 = math.log(prevalence / (1.0 - prevalence))
        logit32 = float(np.float32(logit64))
        return {
            "positive": positive,
            "total": total,
            "prevalence": prevalence,
            "prevalence_hex": float(prevalence).hex(),
            "logit_float64": logit64,
            "logit_float64_hex": float(logit64).hex(),
            "applied_logit_float32": logit32,
            "applied_logit_float32_hex": float(logit32).hex(),
        }

    return {
        "schema": "bplus-v2.2-hierarchical-unique-fit-priors-1",
        "source": "unique fold-0--3 fit macros; schedule occurrences excluded",
        "intervention": prior(intervention, len(fit)),
        "conditional_brake": prior(brake, intervention),
        "intervention_bce_pos_weight": FIT_POS_WEIGHT,
        "intervention_bce_pos_weight_hex": float(FIT_POS_WEIGHT).hex(),
    }


def _calibration_indices(examples: list[Mapping[str, str]]) -> dict[str, np.ndarray]:
    values = {
        "positive_active": np.asarray(
            [
                int(row["example_index"])
                for row in examples
                if row["partition"] == "calibration_positive"
                and row["active_intervention"] == "true"
            ],
            dtype=np.int32,
        ),
        "negative": np.asarray(
            [
                int(row["example_index"])
                for row in examples
                if row["partition"] == "calibration_negative"
            ],
            dtype=np.int32,
        ),
    }
    if values["positive_active"].shape != (EXPECTED_CAL_POSITIVE_INTERVENTION,):
        raise ValueError("hierarchical positive calibration index drift")
    if values["negative"].shape != (EXPECTED_CAL_NEGATIVE_MACROS,):
        raise ValueError("hierarchical negative calibration index drift")
    return values


def _manifest_config(
    created_at: str,
    source_preflight_dir: str | Path,
    hierarchical_identity_release_dir: str | Path,
    episode_counts: Mapping,
    example_counts: Mapping,
    priors: Mapping,
    schedule_info: Mapping,
    source_hashes: Mapping,
    registry: Mapping,
) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at": str(created_at),
        "owner_decision": OWNER_DECISION,
        "action_schema": ACTION_SCHEMA,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "test_opened": False,
        "final_pool": False,
        "ppo_training_started": False,
        "arm_selection_performed": False,
        "fit_folds": list(FIT_FOLDS),
        "calibration_fold": CALIBRATION_FOLD,
        "counts": {"episodes": dict(episode_counts), "examples": dict(example_counts)},
        "priors": dict(priors),
        "schedule": dict(schedule_info),
        "loss_masks": {
            "intervention_bce": "all fit macros with frozen unique-fit positive weight",
            "steering_physical_mse": "I=1 only",
            "conditional_brake_bce": "I=1 only",
            "brake_physical_mse": "I=B=1 only",
        },
        "calibration": {
            "threshold_labels": "75 fold-4 negative episodes only",
            "threshold_rule": "nextafter(float32 eighth-largest episode max raw logit,+inf)",
            "positive_scores_select_threshold": False,
            "max_false_intervention_episodes": MAX_FALSE_INTERVENTION_EPISODES,
            "historically_fresh_or_generalization_evidence": False,
            "negative_skill_f_episodes": 0,
        },
        "source_hashes": dict(source_hashes),
        "source_preflight_relpath": str(Path(source_preflight_dir)),
        "source_preflight_output_sha256": file_sha256(
            Path(source_preflight_dir) / "output_manifest.sha256"
        ),
        "registry": dict(registry),
        "hierarchical_identity_relpath": str(
            Path(hierarchical_identity_release_dir)
        ),
        "hierarchical_identity_output_sha256": file_sha256(
            Path(hierarchical_identity_release_dir) / "output_manifest.sha256"
        ),
        "actor_input_fields": list(ACTOR_INPUT_FIELDS),
        "forbidden_actor_fields": list(FORBIDDEN_ACTOR_FIELDS),
    }


def create_hierarchical_warmstart_manifest(
    repo_root: str | Path,
    source_preflight_dir: str | Path,
    hierarchical_identity_release_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
) -> dict:
    root = Path(repo_root).resolve()
    output = Path(output_dir)
    partial = _prepare_output(output)
    try:
        source_preflight = validate_source_preflight(source_preflight_dir, root)
        if not source_preflight["passed"]:
            raise ValueError(
                f"hierarchical manifest source preflight failed: {source_preflight}"
            )
        identity_validation = validate_hierarchical_identity(
            hierarchical_identity_release_dir, repo_root=root
        )
        if not identity_validation["passed"]:
            raise ValueError(
                f"hierarchical manifest identity prerequisite failed: "
                f"{identity_validation}"
            )
        source_hashes = _validate_source_hashes(root)
        episodes, episode_counts = build_hierarchical_episode_manifest(root)
        registry_rows = make_calibration_registry_rows(episodes)
        registry_before, registry_after, registry_live_state = _registry_snapshots(
            root / REGISTRY_RELPATH, registry_rows
        )
        examples, example_counts = build_hierarchical_macro_examples(root, episodes)
        schedule, fit_indices, schedule_info = build_natural_cycle_schedule(examples)
        priors = build_hierarchical_priors(examples)
        calibration = _calibration_indices(examples)
        _write_tsv(partial / "episodes.tsv", episodes, EPISODE_FIELDS)
        _write_tsv(partial / "macro_examples.tsv", examples, EXAMPLE_FIELDS)
        _write_tsv(partial / "registry_rows.tsv", registry_rows, REGISTRY_FIELDS)
        _write_tsv(
            partial / "registry_before.snapshot.tsv",
            registry_before,
            REGISTRY_FIELDS,
        )
        _write_tsv(
            partial / "registry_after.expected.tsv",
            registry_after,
            REGISTRY_FIELDS,
        )
        registry_ledger = {
            "stage": REGISTRY_STAGE,
            "opened_at_utc": REGISTRY_OPENED_AT,
            "use_class": REGISTRY_USE_CLASS,
            "split_id": REGISTRY_SPLIT_ID,
            "decision_effect": REGISTRY_DECISION_EFFECT,
            "planned_rows": len(registry_rows),
            "before_rows": len(registry_before),
            "after_rows": len(registry_after),
            "before_sha256": file_sha256(
                partial / "registry_before.snapshot.tsv"
            ),
            "after_expected_sha256": file_sha256(
                partial / "registry_after.expected.tsv"
            ),
            "live_state_at_manifest_creation": registry_live_state,
        }
        np.save(partial / "training_schedule.npy", schedule, allow_pickle=False)
        np.save(partial / "fit_indices.npy", fit_indices, allow_pickle=False)
        np.savez(partial / "calibration_indices.npz", **calibration)
        _write_json(
            partial / "config.json",
            _manifest_config(
                created_at,
                source_preflight_dir,
                hierarchical_identity_release_dir,
                episode_counts,
                example_counts,
                priors,
                schedule_info,
                source_hashes,
                registry_ledger,
            ),
        )
        _write_json(
            partial / "validation.json",
            {
                "schema": MANIFEST_VALIDATION_SCHEMA,
                "passed": True,
                "mode": "artifact_only",
                "violations": [],
            },
        )
        _write_output_manifest(partial)
        check = validate_hierarchical_warmstart_manifest(
            partial, root, allow_partial=True
        )
        if not check["passed"]:
            raise AssertionError(f"hierarchical manifest self-validation failed: {check}")
        _write_json(partial / "validation.json", check)
        _write_output_manifest(partial)
        _promote(partial, output)
    except BaseException as error:
        if partial.exists():
            _write_json(
                partial / "FAILED.json",
                {"type": type(error).__name__, "message": str(error)},
            )
        raise
    validation = validate_hierarchical_warmstart_manifest(output, root)
    if not validation["passed"]:
        raise AssertionError(f"created invalid hierarchical manifest: {validation}")
    return {
        "passed": True,
        "episodes": validation["episodes"],
        "examples": validation["examples"],
        "output_manifest_sha256": file_sha256(output / "output_manifest.sha256"),
    }


def validate_hierarchical_warmstart_manifest(
    manifest_dir: str | Path,
    repo_root: str | Path = ".",
    *,
    allow_partial: bool = False,
) -> dict:
    manifest = Path(manifest_dir)
    root = Path(repo_root).resolve()
    violations = []
    details = {"episodes": 0, "examples": 0}
    try:
        if not allow_partial and not (manifest / "COMPLETE").is_file():
            raise ValueError("hierarchical manifest lacks COMPLETE")
        _validate_output_manifest(manifest)
        expected_files = {
            "calibration_indices.npz",
            "config.json",
            "episodes.tsv",
            "fit_indices.npy",
            "macro_examples.tsv",
            "registry_after.expected.tsv",
            "registry_before.snapshot.tsv",
            "registry_rows.tsv",
            "training_schedule.npy",
            "validation.json",
        }
        actual_files = {
            path.relative_to(manifest).as_posix()
            for path in manifest.rglob("*")
            if path.is_file() and path.name not in {"COMPLETE", "output_manifest.sha256"}
        }
        if actual_files != expected_files:
            raise ValueError("hierarchical manifest file inventory drift")
        config = json.loads((manifest / "config.json").read_text(encoding="utf-8"))
        if set(config) != MANIFEST_CONFIG_KEYS:
            raise ValueError("hierarchical manifest config schema drift")
        if (
            config["schema"] != MANIFEST_SCHEMA
            or config["owner_decision"] != OWNER_DECISION
            or config["action_schema"] != ACTION_SCHEMA
            or config["checkpoint_schema"] != CHECKPOINT_SCHEMA
            or config["test_opened"] is not False
            or config["final_pool"] is not False
            or config["ppo_training_started"] is not False
            or config["arm_selection_performed"] is not False
            or config["fit_folds"] != list(FIT_FOLDS)
            or config["calibration_fold"] != CALIBRATION_FOLD
            or config["actor_input_fields"] != list(ACTOR_INPUT_FIELDS)
            or config["forbidden_actor_fields"] != list(FORBIDDEN_ACTOR_FIELDS)
        ):
            raise ValueError("hierarchical manifest authority/scope drift")
        if config["source_hashes"] != _validate_source_hashes(root):
            raise ValueError("hierarchical manifest source provenance drift")
        source_preflight = root / config["source_preflight_relpath"]
        source_preflight_validation = validate_source_preflight(
            source_preflight, root
        )
        if not source_preflight_validation["passed"]:
            raise ValueError(
                "hierarchical manifest referenced source preflight failed"
            )
        if (
            file_sha256(source_preflight / "output_manifest.sha256")
            != config["source_preflight_output_sha256"]
        ):
            raise ValueError("hierarchical manifest source-preflight hash drift")
        identity_release = root / config["hierarchical_identity_relpath"]
        identity_validation = validate_hierarchical_identity(
            identity_release, repo_root=root
        )
        if not identity_validation["passed"]:
            raise ValueError(
                "hierarchical manifest referenced identity prerequisite failed"
            )
        if (
            file_sha256(identity_release / "output_manifest.sha256")
            != config["hierarchical_identity_output_sha256"]
        ):
            raise ValueError("hierarchical manifest identity output hash drift")
        episodes = _read_tsv(manifest / "episodes.tsv")
        examples = _read_tsv(manifest / "macro_examples.tsv")
        if any(tuple(row) != EPISODE_FIELDS for row in episodes):
            raise ValueError("hierarchical episode TSV column order drift")
        if any(tuple(row) != EXAMPLE_FIELDS for row in examples):
            raise ValueError("hierarchical example TSV column order drift")
        expected_episodes, episode_counts = build_hierarchical_episode_manifest(root)
        expected_examples, example_counts = build_hierarchical_macro_examples(
            root, expected_episodes
        )
        if episodes != expected_episodes or examples != expected_examples:
            raise ValueError("hierarchical manifest rows are not reproducible")
        registry_rows = _read_registry(manifest / "registry_rows.tsv")
        expected_registry_rows = make_calibration_registry_rows(episodes)
        if registry_rows != expected_registry_rows:
            raise ValueError("hierarchical calibration registry plan drift")
        registry_before = _read_registry(
            manifest / "registry_before.snapshot.tsv"
        )
        registry_after = _read_registry(
            manifest / "registry_after.expected.tsv"
        )
        if registry_after != registry_before + registry_rows:
            raise ValueError("hierarchical registry before/after append drift")
        registry_ledger = {
            "stage": REGISTRY_STAGE,
            "opened_at_utc": REGISTRY_OPENED_AT,
            "use_class": REGISTRY_USE_CLASS,
            "split_id": REGISTRY_SPLIT_ID,
            "decision_effect": REGISTRY_DECISION_EFFECT,
            "planned_rows": len(registry_rows),
            "before_rows": len(registry_before),
            "after_rows": len(registry_after),
            "before_sha256": file_sha256(
                manifest / "registry_before.snapshot.tsv"
            ),
            "after_expected_sha256": file_sha256(
                manifest / "registry_after.expected.tsv"
            ),
            "live_state_at_manifest_creation": config["registry"][
                "live_state_at_manifest_creation"
            ],
        }
        if registry_ledger["live_state_at_manifest_creation"] not in {
            "ready",
            "already_appended",
        }:
            raise ValueError("hierarchical recorded registry live state drift")
        if registry_ledger != config["registry"]:
            raise ValueError("hierarchical registry ledger drift")
        live_state = _registry_live_state(
            root / REGISTRY_RELPATH,
            registry_rows,
            registry_ledger["before_sha256"],
            registry_ledger["after_expected_sha256"],
        )
        if live_state not in {"ready", "already_appended"}:
            raise AssertionError("unreachable hierarchical registry live state")
        schedule, fit_indices, schedule_info = build_natural_cycle_schedule(examples)
        stored_schedule = np.load(manifest / "training_schedule.npy", allow_pickle=False)
        stored_fit = np.load(manifest / "fit_indices.npy", allow_pickle=False)
        if (
            stored_schedule.dtype != np.int32
            or stored_schedule.shape != (WARMSTART_UPDATES, WARMSTART_BATCH_SIZE)
            or not np.array_equal(stored_schedule, schedule)
            or stored_fit.dtype != np.int32
            or not np.array_equal(stored_fit, fit_indices)
        ):
            raise ValueError("hierarchical natural schedule drift")
        with np.load(manifest / "calibration_indices.npz", allow_pickle=False) as values:
            if set(values.files) != {"positive_active", "negative"}:
                raise ValueError("hierarchical calibration index inventory drift")
            expected_calibration = _calibration_indices(examples)
            if any(
                values[name].dtype != np.int32
                or not np.array_equal(values[name], expected_calibration[name])
                for name in values.files
            ):
                raise ValueError("hierarchical calibration indices drift")
        if config["counts"] != {
            "episodes": episode_counts,
            "examples": example_counts,
        }:
            raise ValueError("hierarchical manifest count ledger drift")
        if config["priors"] != build_hierarchical_priors(examples):
            raise ValueError("hierarchical manifest prior drift")
        if config["schedule"] != schedule_info:
            raise ValueError("hierarchical manifest schedule ledger drift")
        details = {"episodes": len(episodes), "examples": len(examples)}
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": MANIFEST_VALIDATION_SCHEMA,
        "passed": not violations,
        "mode": "artifact_only",
        **details,
        "violations": violations,
    }


def _hierarchical_loss(
    policy: RemediatedV22Policy,
    bc: torch.Tensor,
    lidar: torch.Tensor,
    scalar: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    distribution = policy.distribution(bc, lidar, scalar)
    intervention_target = targets["active_intervention"]
    brake_target = targets["target_brake_gate"]
    intervention_logits = distribution.intervention.logits.squeeze(-1)
    brake_logits = distribution.brake_gate_distribution.logits.squeeze(-1)
    pos_weight = torch.tensor(
        FIT_POS_WEIGHT, dtype=intervention_logits.dtype, device=intervention_logits.device
    )
    intervention_loss = F.binary_cross_entropy_with_logits(
        intervention_logits, intervention_target, pos_weight=pos_weight
    )
    active_count = torch.clamp(intervention_target.sum(), min=1.0)
    predicted_steer = torch.tanh(distribution.steer.mean.squeeze(-1)) * STEER_BUDGET
    steer_loss = torch.sum(
        intervention_target * (predicted_steer - targets["target_steer"]) ** 2
    ) / active_count
    conditional_brake_loss = torch.sum(
        intervention_target
        * F.binary_cross_entropy_with_logits(
            brake_logits, brake_target, reduction="none"
        )
    ) / active_count
    brake_active = intervention_target * brake_target
    brake_count = torch.clamp(brake_active.sum(), min=1.0)
    predicted_brake = torch.sigmoid(distribution.brake.mean.squeeze(-1)) * BRAKE_BUDGET
    brake_loss = torch.sum(
        brake_active * (predicted_brake - targets["target_brake"]) ** 2
    ) / brake_count
    total = intervention_loss + steer_loss + conditional_brake_loss + brake_loss
    if not torch.isfinite(total):
        raise FloatingPointError("hierarchical warm-start loss is nonfinite")
    return total, {
        "intervention": intervention_loss,
        "steer": steer_loss,
        "conditional_brake": conditional_brake_loss,
        "brake": brake_loss,
        "raw_intervention_logits": intervention_logits,
        "conditional_brake_logits": brake_logits,
        "predicted_steer": predicted_steer,
        "predicted_brake": predicted_brake,
    }


def _get_batch(
    provider: WarmstartBatchProvider,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    actor, labels = provider.get(indices)
    bc, lidar, scalar, targets = _torch_batch(actor, labels, device)
    active = np.asarray(
        [
            float(provider.examples[int(index)]["active_intervention"] == "true")
            for index in np.asarray(indices).tolist()
        ],
        dtype=np.float32,
    )
    targets["active_intervention"] = torch.as_tensor(active, device=device)
    return bc, lidar, scalar, targets


@torch.no_grad()
def _raw_scores(
    policy: RemediatedV22Policy,
    provider: WarmstartBatchProvider,
    indices: np.ndarray,
    device: torch.device,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    intervention = []
    brake = []
    for start in range(0, len(indices), chunk_size):
        batch = indices[start : start + chunk_size]
        bc, lidar, scalar, _ = _get_batch(provider, batch, device)
        feature = policy.action_core(policy.policy_feature(bc, lidar, scalar))
        intervention.append(
            policy.intervention_gate(feature).squeeze(-1).detach().cpu().numpy()
        )
        brake.append(policy.brake_gate(feature).squeeze(-1).detach().cpu().numpy())
    return (
        np.concatenate(intervention).astype(np.float32, copy=False),
        np.concatenate(brake).astype(np.float32, copy=False),
    )


def _confusion(choice: np.ndarray, target: np.ndarray) -> dict:
    choice = np.asarray(choice, dtype=np.bool_)
    target = np.asarray(target, dtype=np.bool_)
    tp = int(np.count_nonzero(choice & target))
    fp = int(np.count_nonzero(choice & ~target))
    tn = int(np.count_nonzero(~choice & ~target))
    fn = int(np.count_nonzero(~choice & target))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def _bce(logits: np.ndarray, target: np.ndarray) -> float:
    logits64 = np.asarray(logits, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    return float(np.mean(np.maximum(logits64, 0) - logits64 * target64 + np.log1p(np.exp(-np.abs(logits64)))))


def _distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(values.max()),
        "mean": float(values.mean(dtype=np.float64)),
    }


def derive_negative_only_calibration(
    examples: list[Mapping[str, str]],
    episodes: list[Mapping[str, str]],
    positive_raw: np.ndarray,
    negative_raw: np.ndarray,
    conditional_raw: np.ndarray,
    arm: str,
) -> tuple[dict, list[dict[str, str]]]:
    """Apply the pre-registered eighth-order-statistic threshold rule."""

    positive_rows = [
        row
        for row in examples
        if row["partition"] == "calibration_positive"
        and row["active_intervention"] == "true"
    ]
    negative_rows = [
        row for row in examples if row["partition"] == "calibration_negative"
    ]
    if positive_raw.shape != (len(positive_rows),) or negative_raw.shape != (
        len(negative_rows),
    ):
        raise ValueError("hierarchical calibration score shape mismatch")
    if conditional_raw.shape != positive_raw.shape:
        raise ValueError("hierarchical conditional calibration shape mismatch")
    if any(not np.all(np.isfinite(value)) for value in (positive_raw, negative_raw, conditional_raw)):
        raise ValueError("hierarchical calibration scores are nonfinite")

    negative_positions: dict[str, list[int]] = {}
    for position, row in enumerate(negative_rows):
        negative_positions.setdefault(row["l2_id"], []).append(position)
    maxima = []
    for l2_id, positions in negative_positions.items():
        maximum = np.float32(np.max(negative_raw[np.asarray(positions)]))
        maxima.append(
            (
                maximum,
                _hash(CALIBRATION_TIE_DOMAIN, l2_id),
                l2_id,
                positions,
            )
        )
    maxima.sort(key=lambda value: (-float(value[0]), value[1], value[2]))
    if len(maxima) != EXPECTED_CAL_NEGATIVE_EPISODES:
        raise ValueError("hierarchical calibration negative episode drift")
    eighth = maxima[MAX_FALSE_INTERVENTION_EPISODES][0]
    threshold = np.nextafter(eighth, np.float32(np.inf), dtype=np.float32)
    offset = np.float32(-threshold)
    negative_adjusted = negative_raw.astype(np.float32) + offset
    positive_adjusted = positive_raw.astype(np.float32) + offset
    negative_choice = negative_adjusted > 0.0
    positive_choice = positive_adjusted > 0.0

    episode_rows = []
    false_episodes = 0
    for maximum, tie_hash, l2_id, positions in maxima:
        adjusted = np.float32(maximum + offset)
        decision = bool(adjusted > 0.0)
        false_episodes += int(decision)
        episode_rows.append(
            {
                "arm": arm,
                "partition": "calibration_negative",
                "l2_id": l2_id,
                "action_category": "no_op",
                "macro_count": str(len(positions)),
                "raw_max_logit_hex": float(maximum).hex(),
                "applied_offset_hex": float(offset).hex(),
                "adjusted_max_logit_hex": float(adjusted).hex(),
                "intervention_decision": str(decision).lower(),
                "tie_hash": tie_hash,
            }
        )

    positive_positions: dict[str, list[int]] = {}
    for position, row in enumerate(positive_rows):
        positive_positions.setdefault(row["l2_id"], []).append(position)
    episode_by_l2 = {row["l2_id"]: row for row in episodes}
    positive_episode_choice = {}
    for l2_id in sorted(positive_positions):
        positions = positive_positions[l2_id]
        raw_max = np.float32(np.max(positive_raw[np.asarray(positions)]))
        adjusted = np.float32(raw_max + offset)
        decision = bool(adjusted > 0.0)
        source = episode_by_l2[l2_id]
        brake = float.fromhex(source["target_brake_hex"]) > 0.0
        category = "brake_containing" if brake else "steer_only"
        positive_episode_choice[l2_id] = (decision, category)
        episode_rows.append(
            {
                "arm": arm,
                "partition": "calibration_positive",
                "l2_id": l2_id,
                "action_category": category,
                "macro_count": str(len(positions)),
                "raw_max_logit_hex": float(raw_max).hex(),
                "applied_offset_hex": float(offset).hex(),
                "adjusted_max_logit_hex": float(adjusted).hex(),
                "intervention_decision": str(decision).lower(),
                "tie_hash": _hash(CALIBRATION_TIE_DOMAIN, l2_id),
            }
        )

    top_choice = np.concatenate([positive_choice, negative_choice])
    top_target = np.concatenate(
        [np.ones(len(positive_choice), dtype=np.bool_), np.zeros(len(negative_choice), dtype=np.bool_)]
    )
    top_confusion = _confusion(top_choice, top_target)
    brake_target = np.asarray(
        [int(row["target_brake_gate"]) for row in positive_rows], dtype=np.bool_
    )
    conditional_choice = conditional_raw > 0.0
    conditional_confusion = _confusion(conditional_choice, brake_target)
    positive_episode_recall = sum(value[0] for value in positive_episode_choice.values())
    steer_recall = sum(
        decision
        for decision, category in positive_episode_choice.values()
        if category == "steer_only"
    )
    brake_recall = sum(
        decision
        for decision, category in positive_episode_choice.values()
        if category == "brake_containing"
    )
    checks = {
        "false_intervention_episodes_le_7_of_75": false_episodes
        <= MAX_FALSE_INTERVENTION_EPISODES,
        "intervention_window_episode_recall_ge_6_of_9": positive_episode_recall >= 6,
        "intervention_macro_recall_ge_20_of_39": int(positive_choice.sum()) >= 20,
        "steer_only_episode_recall_ge_2_of_4": steer_recall >= 2,
        "brake_episode_recall_ge_3_of_5": brake_recall >= 3,
        "conditional_brake_recall_ge_half": conditional_confusion["recall"] >= 0.5,
        "conditional_brake_specificity_ge_half": conditional_confusion["specificity"] >= 0.5,
        "metrics_finite": all(
            math.isfinite(float(value))
            for value in (
                *top_confusion.values(),
                *conditional_confusion.values(),
                _bce(np.concatenate([positive_adjusted, negative_adjusted]), top_target),
                _bce(conditional_raw, brake_target),
            )
        ),
    }
    metrics = {
        "schema": SCORE_SCHEMA,
        "arm": arm,
        "threshold_selection_uses_positive_labels": False,
        "eighth_largest_negative_episode_max_raw_logit": float(eighth),
        "threshold_float32": float(threshold),
        "threshold_float32_hex": float(threshold).hex(),
        "applied_offset_float32": float(offset),
        "applied_offset_float32_hex": float(offset).hex(),
        "false_intervention_episodes": false_episodes,
        "negative_episodes": EXPECTED_CAL_NEGATIVE_EPISODES,
        "intervention_window_episode_true_positive": int(positive_episode_recall),
        "intervention_window_episodes": EXPECTED_CAL_POSITIVE_EPISODES,
        "intervention_macro_true_positive": int(positive_choice.sum()),
        "intervention_macros": EXPECTED_CAL_POSITIVE_INTERVENTION,
        "steer_only_episode_true_positive": int(steer_recall),
        "steer_only_episodes": EXPECTED_CAL_STEER_ONLY_EPISODES,
        "brake_episode_true_positive": int(brake_recall),
        "brake_episodes": EXPECTED_CAL_BRAKE_EPISODES,
        "top_gate_confusion": top_confusion,
        "conditional_brake_confusion": conditional_confusion,
        "top_gate_bce": _bce(
            np.concatenate([positive_adjusted, negative_adjusted]), top_target
        ),
        "conditional_brake_bce": _bce(conditional_raw, brake_target),
        "raw_positive_distribution": _distribution_summary(positive_raw),
        "raw_negative_distribution": _distribution_summary(negative_raw),
        "raw_conditional_brake_distribution": _distribution_summary(conditional_raw),
        "checks": checks,
        "passed": all(checks.values()),
    }
    return metrics, episode_rows


def _policy_hashes(policy: RemediatedV22Policy) -> dict[str, str]:
    values = legacy_policy_hashes(policy)
    values["hierarchical_action_state_sha256"] = _tensor_digest(
        (name, value)
        for name, value in policy.state_dict().items()
        if name.startswith(
            (
                "action_core.",
                "steer_mean.",
                "brake_gate.",
                "brake_mean.",
                "intervention_gate.",
                "intervention_logit_offset",
                "log_steer_std",
                "log_brake_std",
            )
        )
    )
    return values


def _per_tensor_digests(policy: RemediatedV22Policy) -> dict[str, str]:
    return {
        name: _tensor_digest(((name, value),))
        for name, value in policy.state_dict().items()
    }


def _checkpoint_payload(
    policy: RemediatedV22Policy,
    arm: str,
    manifest_sha256: str,
    calibration: Mapping,
    task6_passed: bool,
) -> dict:
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in policy.state_dict().items()
    }
    return {
        "schema": CHECKPOINT_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "release_label": RELEASE_LABEL,
        "arm": arm,
        "manifest_output_sha256": manifest_sha256,
        "task6_acceptance_passed": bool(task6_passed),
        "calibration_offset_float32": calibration["applied_offset_float32"],
        "state_dict": state,
        "state_dict_sha256": _tensor_digest(state.items()),
    }


def _finite_tree(value, path: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int, np.integer)):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"nonfinite hierarchical metric: {path}")
        return
    raise TypeError(f"unsupported hierarchical metric type: {path}/{type(value)}")


def _assert_training_invariants(reports: Mapping[str, Mapping]) -> None:
    if set(reports) != set(ARMS):
        raise ValueError("hierarchical report arm inventory drift")
    fresh = {arm: reports[arm]["fresh_hashes"] for arm in ARMS}
    initial = {arm: reports[arm]["initial_hashes"] for arm in ARMS}
    final = {arm: reports[arm]["final_hashes"] for arm in ARMS}
    if len({fresh[arm]["full_state_sha256"] for arm in ARMS}) != 1:
        raise ValueError("hierarchical arms do not share fresh full state")
    if len({initial[arm]["full_state_sha256"] for arm in ARMS}) != 1:
        raise ValueError("hierarchical arms do not share post-prior full state")
    if len({initial[arm]["bc_sha256"] for arm in ARMS}) != 1:
        raise ValueError("hierarchical arms do not share BC initialization")
    if len({initial[arm]["shadow_sidecar_sha256"] for arm in ARMS}) != 1:
        raise ValueError("hierarchical arms do not share diagnostic sidecar")
    for arm in ARMS:
        if reports[arm]["prior_changed_state_keys"] != [
            "brake_gate.bias",
            "intervention_gate.bias",
        ]:
            raise ValueError(f"hierarchical prior changed forbidden state: {arm}")
        for name in (
            "bc_sha256",
            "policy_sidecar_sha256",
            "policy_sidecar_encoder_sha256",
            "shadow_sidecar_sha256",
        ):
            if fresh[arm][name] != initial[arm][name]:
                raise ValueError(
                    f"hierarchical prior changed non-gate component: {arm}/{name}"
                )
        if final[arm]["bc_sha256"] != initial[arm]["bc_sha256"]:
            raise ValueError(f"hierarchical frozen BC mutated: {arm}")
        if final[arm]["shadow_sidecar_sha256"] != initial[arm]["shadow_sidecar_sha256"]:
            raise ValueError(f"hierarchical shadow sidecar mutated: {arm}")
        if final[arm]["hierarchical_action_state_sha256"] == initial[arm]["hierarchical_action_state_sha256"]:
            raise ValueError(f"hierarchical action head did not update: {arm}")
    for arm in ARMS[:2]:
        if final[arm]["policy_sidecar_sha256"] != initial[arm]["policy_sidecar_sha256"]:
            raise ValueError(f"hierarchical frozen policy sidecar mutated: {arm}")
    if final[ARMS[2]]["policy_sidecar_encoder_sha256"] == initial[ARMS[2]]["policy_sidecar_encoder_sha256"]:
        raise ValueError("hierarchical arm-C sidecar encoder did not update")


def run_hierarchical_warmstart(
    repo_root: str | Path,
    source_preflight_dir: str | Path,
    hierarchical_identity_release_dir: str | Path,
    manifest_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
    device_name: str = "cuda:0",
) -> dict:
    """Fit/calibrate all three arms.  This function never starts PPO."""

    root = Path(repo_root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("hierarchical warm-start must run from repo root")
    source = validate_source_preflight(source_preflight_dir, root)
    if not source["passed"]:
        raise ValueError(f"hierarchical source preflight failed: {source}")
    identity_validation = validate_hierarchical_identity(
        hierarchical_identity_release_dir, repo_root=root
    )
    if not identity_validation["passed"]:
        raise ValueError(
            f"hierarchical fit identity prerequisite failed: {identity_validation}"
        )
    manifest_validation = validate_hierarchical_warmstart_manifest(manifest_dir, root)
    if not manifest_validation["passed"]:
        raise ValueError(f"hierarchical manifest failed: {manifest_validation}")
    pinned = validate_pinned_inputs(root)
    if not pinned["passed"]:
        raise ValueError(f"hierarchical pinned inputs failed: {pinned}")
    if not os.environ.get("NUMBA_CACHE_DIR") or not Path(os.environ["NUMBA_CACHE_DIR"]).is_absolute():
        raise ValueError("hierarchical warm-start requires absolute NUMBA_CACHE_DIR")
    device = torch.device(device_name)
    _set_deterministic_cuda(device)

    output = Path(output_dir)
    partial = _prepare_output(output)
    provider = None
    try:
        manifest = Path(manifest_dir)
        manifest_config = json.loads((manifest / "config.json").read_text(encoding="utf-8"))
        if (
            (root / manifest_config["source_preflight_relpath"]).resolve()
            != (root / Path(source_preflight_dir)).resolve()
            or manifest_config["source_preflight_output_sha256"]
            != file_sha256(Path(source_preflight_dir) / "output_manifest.sha256")
        ):
            raise ValueError(
                "hierarchical fit source preflight differs from manifest preflight"
            )
        if (
            (root / manifest_config["hierarchical_identity_relpath"]).resolve()
            != (root / Path(hierarchical_identity_release_dir)).resolve()
            or manifest_config["hierarchical_identity_output_sha256"]
            != file_sha256(
                Path(hierarchical_identity_release_dir)
                / "output_manifest.sha256"
            )
        ):
            raise ValueError(
                "hierarchical fit identity release differs from manifest prerequisite"
            )
        registry_rows = _read_registry(manifest / "registry_rows.tsv")
        registry_ledger = manifest_config["registry"]
        live_registry = root / REGISTRY_RELPATH
        registry_state_before = _registry_live_state(
            live_registry,
            registry_rows,
            registry_ledger["before_sha256"],
            registry_ledger["after_expected_sha256"],
        )
        registry_before_observed_sha256 = file_sha256(live_registry)
        registry_append = append_opened_registry(live_registry, registry_rows)
        if file_sha256(live_registry) != registry_ledger["after_expected_sha256"]:
            raise AssertionError(
                "hierarchical live registry did not reach expected-after state"
            )
        expected_append = (
            (len(registry_rows), 0)
            if registry_state_before == "ready"
            else (0, len(registry_rows))
        )
        if (registry_append.appended, registry_append.skipped) != expected_append:
            raise AssertionError("hierarchical registry append accounting mismatch")
        if registry_append.total != registry_ledger["after_rows"]:
            raise AssertionError("hierarchical registry total-row accounting mismatch")
        shutil.copyfile(live_registry, partial / "opened_registry.snapshot.tsv")
        manifest_sha = file_sha256(manifest / "output_manifest.sha256")
        schedule = np.load(manifest / "training_schedule.npy", allow_pickle=False)
        with np.load(manifest / "calibration_indices.npz", allow_pickle=False) as values:
            positive_indices = values["positive_active"].copy()
            negative_indices = values["negative"].copy()
        episodes = _read_tsv(manifest / "episodes.tsv")
        examples = _read_tsv(manifest / "macro_examples.tsv")
        provider = WarmstartBatchProvider(root, manifest, device)
        sidecar_state, sidecar_mean, sidecar_std, _ = load_sidecar_bundle(
            root / SIDECAR_RELEASE_RELPATH
        )
        bc_state = {
            name: value.detach().cpu().contiguous()
            for name, value in provider.bc.state_dict().items()
        }
        for directory in ("curves", "checkpoints", "reports", "scores"):
            (partial / directory).mkdir()
        reports = {}
        priors = manifest_config["priors"]
        for arm in ARMS:
            gc.collect()
            torch.cuda.empty_cache()
            policy = RemediatedV22Policy(
                arm,
                bc_state_dict=bc_state,
                sidecar_state_dict=sidecar_state,
                sidecar_bc_mean=sidecar_mean,
                sidecar_bc_std=sidecar_std,
                initialization_seed=POLICY_SEED,
            ).to(device)
            policy.eval()
            fresh_hashes = _policy_hashes(policy)
            fresh_tensor_digests = _per_tensor_digests(policy)
            initialize_hierarchical_priors(
                policy,
                priors["intervention"]["applied_logit_float32"],
                priors["conditional_brake"]["applied_logit_float32"],
            )
            initial_hashes = _policy_hashes(policy)
            initial_tensor_digests = _per_tensor_digests(policy)
            prior_changed_state_keys = sorted(
                name
                for name in fresh_tensor_digests
                if fresh_tensor_digests[name] != initial_tensor_digests[name]
            )
            if prior_changed_state_keys != [
                "brake_gate.bias",
                "intervention_gate.bias",
            ]:
                raise ValueError(
                    f"hierarchical prior transition mutated forbidden state: "
                    f"{arm}/{prior_changed_state_keys}"
                )
            if not torch.equal(
                policy.intervention_logit_offset.detach(),
                torch.zeros_like(policy.intervention_logit_offset),
            ):
                raise ValueError("hierarchical fit started with nonzero calibration offset")
            frozen_snapshot = policy.frozen_snapshot()
            optimizer = torch.optim.AdamW(
                policy.optimizer_parameter_groups(), weight_decay=0.0
            )
            curve_names = (
                "total_loss",
                "intervention_loss",
                "steer_loss",
                "conditional_brake_loss",
                "brake_loss",
                "gradient_norm",
            )
            curves = {
                name: np.empty(WARMSTART_UPDATES, dtype=np.float64)
                for name in curve_names
            }
            policy.train()
            for update in range(WARMSTART_UPDATES):
                bc, lidar, scalar, targets = _get_batch(
                    provider, schedule[update], device
                )
                optimizer.zero_grad(set_to_none=True)
                loss, components = _hierarchical_loss(
                    policy, bc, lidar, scalar, targets
                )
                loss.backward()
                trainable = [
                    parameter
                    for parameter in policy.parameters()
                    if parameter.requires_grad
                ]
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, GRAD_CLIP_NORM, error_if_nonfinite=True
                )
                optimizer.step()
                curves["total_loss"][update] = float(loss.detach().item())
                curves["intervention_loss"][update] = float(
                    components["intervention"].detach().item()
                )
                curves["steer_loss"][update] = float(components["steer"].detach().item())
                curves["conditional_brake_loss"][update] = float(
                    components["conditional_brake"].detach().item()
                )
                curves["brake_loss"][update] = float(components["brake"].detach().item())
                curves["gradient_norm"][update] = float(gradient_norm.detach().item())
            if any(not np.all(np.isfinite(values)) for values in curves.values()):
                raise FloatingPointError(f"hierarchical training curve nonfinite: {arm}")
            policy.assert_frozen_unchanged(frozen_snapshot)
            policy.eval()
            positive_raw, conditional_raw = _raw_scores(
                policy, provider, positive_indices, device
            )
            negative_raw, _ = _raw_scores(
                policy, provider, negative_indices, device
            )
            calibration, episode_rows = derive_negative_only_calibration(
                examples,
                episodes,
                positive_raw,
                negative_raw,
                conditional_raw,
                arm,
            )
            apply_intervention_logit_offset(
                policy, calibration["applied_offset_float32"]
            )
            observed_offset = float(policy.intervention_logit_offset.detach().cpu().item())
            if observed_offset != calibration["applied_offset_float32"]:
                raise AssertionError("hierarchical checkpoint calibration offset mismatch")
            final_hashes = _policy_hashes(policy)
            report = {
                "schema": REPORT_SCHEMA,
                "release_label": RELEASE_LABEL,
                "arm": arm,
                "action_schema": ACTION_SCHEMA,
                "checkpoint_schema": CHECKPOINT_SCHEMA,
                "updates_completed": WARMSTART_UPDATES,
                "batch_size": WARMSTART_BATCH_SIZE,
                "early_stopping": False,
                "arm_selection_performed": False,
                "ppo_training_started": False,
                "losses_are_diagnostic_only": True,
                "fresh_initial_intervention_logit": INITIAL_INTERVENTION_LOGIT,
                "fresh_initial_brake_logit": INITIAL_BRAKE_LOGIT,
                "priors": priors,
                "manifest_output_sha256": manifest_sha,
                "schedule_sha256": file_sha256(manifest / "training_schedule.npy"),
                "fresh_hashes": fresh_hashes,
                "initial_hashes": initial_hashes,
                "prior_changed_state_keys": prior_changed_state_keys,
                "final_hashes": final_hashes,
                "calibration": calibration,
                "arm_acceptance_passed": calibration["passed"],
                # Rewritten to the all-arm decision before the release is sealed.
                "task6_acceptance_passed": False,
                "curve_final": {
                    name: float(values[-1]) for name, values in curves.items()
                },
                "curve_min_total_loss": float(curves["total_loss"].min()),
            }
            _finite_tree(report)
            np.savez(
                partial / "scores" / f"{arm}.npz",
                positive_raw_intervention=positive_raw,
                negative_raw_intervention=negative_raw,
                positive_raw_conditional_brake=conditional_raw,
            )
            np.savez(partial / "curves" / f"{arm}.npz", **curves)
            _write_tsv(
                partial / "reports" / f"{arm}_episodes.tsv",
                episode_rows,
                CALIBRATION_EPISODE_FIELDS,
            )
            _write_json(partial / "reports" / f"{arm}.json", report)
            torch.save(
                _checkpoint_payload(
                    policy,
                    arm,
                    manifest_sha,
                    calibration,
                    calibration["passed"],
                ),
                partial / "checkpoints" / f"{arm}.pt",
            )
            reports[arm] = report
            del policy, optimizer
            gc.collect()
            torch.cuda.empty_cache()

        _assert_training_invariants(reports)
        task6_passed = all(reports[arm]["arm_acceptance_passed"] for arm in ARMS)
        # §6.3 is an all-arm gate.  A locally passing checkpoint must not claim
        # PPO eligibility when any sibling arm failed.
        for arm in ARMS:
            reports[arm]["task6_acceptance_passed"] = task6_passed
            _write_json(partial / "reports" / f"{arm}.json", reports[arm])
            checkpoint_path = partial / "checkpoints" / f"{arm}.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            checkpoint["task6_acceptance_passed"] = task6_passed
            torch.save(checkpoint, checkpoint_path)
        config = {
            "schema": RELEASE_SCHEMA,
            "release_label": RELEASE_LABEL,
            "created_at": str(created_at),
            "owner_decision": OWNER_DECISION,
            "action_schema": ACTION_SCHEMA,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "manifest_relpath": str(Path(manifest_dir)),
            "manifest_output_sha256": manifest_sha,
            "source_preflight_relpath": str(Path(source_preflight_dir)),
            "source_preflight_output_sha256": file_sha256(
                Path(source_preflight_dir) / "output_manifest.sha256"
            ),
            "hierarchical_identity_relpath": str(
                Path(hierarchical_identity_release_dir)
            ),
            "hierarchical_identity_output_sha256": file_sha256(
                Path(hierarchical_identity_release_dir)
                / "output_manifest.sha256"
            ),
            "arms": list(ARMS),
            "updates_per_arm": WARMSTART_UPDATES,
            "batch_size": WARMSTART_BATCH_SIZE,
            "action_core_lr": ACTION_CORE_LR,
            "sidecar_finetune_lr": SIDECAR_FINETUNE_LR,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "task6_acceptance_passed": task6_passed,
            "ppo_checkpoint_eligible": task6_passed,
            "ppo_training_started": False,
            "closed_loop_evaluation_started": False,
            "arm_selection_performed": False,
            "candidate_promoted": False,
            "test_opened": False,
            "final_pool": False,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "numba_cache_dir": os.environ["NUMBA_CACHE_DIR"],
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms": True,
            "registry": {
                "state_before_append": registry_state_before,
                "before_observed_sha256": registry_before_observed_sha256,
                "after_sha256": file_sha256(live_registry),
                "rows_appended": registry_append.appended,
                "rows_skipped": registry_append.skipped,
                "total_rows": registry_append.total,
                "planned_rows": len(registry_rows),
            },
            "reports": {
                arm: {
                    "report_sha256": file_sha256(partial / "reports" / f"{arm}.json"),
                    "episode_decisions_sha256": file_sha256(
                        partial / "reports" / f"{arm}_episodes.tsv"
                    ),
                    "checkpoint_sha256": file_sha256(
                        partial / "checkpoints" / f"{arm}.pt"
                    ),
                    "curve_sha256": file_sha256(partial / "curves" / f"{arm}.npz"),
                    "scores_sha256": file_sha256(partial / "scores" / f"{arm}.npz"),
                }
                for arm in ARMS
            },
        }
        _write_json(partial / "config.json", config)
        _write_json(
            partial / "validation.json",
            {
                "schema": VALIDATION_SCHEMA,
                "passed": task6_passed,
                "integrity_passed": True,
                "task6_acceptance_passed": task6_passed,
                "mode": "pending_same_device_full",
                "arms": len(ARMS),
                "violations": [],
            },
        )
        _write_output_manifest(partial)
        check = validate_hierarchical_warmstart_release(
            partial,
            root,
            device_name=device_name,
            allow_partial=True,
        )
        if not check["integrity_passed"]:
            raise AssertionError(f"hierarchical release self-validation failed: {check}")
        _write_json(partial / "validation.json", check)
        _write_output_manifest(partial)
        artifact_check = validate_hierarchical_warmstart_release(
            partial, root, allow_partial=True
        )
        if not artifact_check["integrity_passed"]:
            raise AssertionError(
                f"hierarchical artifact-only validation failed: {artifact_check}"
            )
        _promote(partial, output)
    except BaseException as error:
        if partial.exists():
            _write_json(
                partial / "FAILED.json",
                {"type": type(error).__name__, "message": str(error)},
            )
        raise
    finally:
        if provider is not None:
            del provider
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    validation = validate_hierarchical_warmstart_release(output, root)
    if not validation["integrity_passed"]:
        raise AssertionError(f"created invalid hierarchical release: {validation}")
    return {
        "passed": validation["task6_acceptance_passed"],
        "integrity_passed": validation["integrity_passed"],
        "task6_acceptance_passed": validation["task6_acceptance_passed"],
        "ppo_training_started": False,
        "arm_selection_performed": False,
        "output_manifest_sha256": file_sha256(output / "output_manifest.sha256"),
    }


def _load_score_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        expected = {
            "positive_raw_intervention",
            "negative_raw_intervention",
            "positive_raw_conditional_brake",
        }
        if set(values.files) != expected:
            raise ValueError("hierarchical score array inventory drift")
        return (
            values["positive_raw_intervention"].copy(),
            values["negative_raw_intervention"].copy(),
            values["positive_raw_conditional_brake"].copy(),
        )


def validate_hierarchical_warmstart_release(
    release_dir: str | Path,
    repo_root: str | Path = ".",
    *,
    device_name: str | None = None,
    allow_partial: bool = False,
) -> dict:
    """Validate artifacts; with ``device_name``, recompute logits on CUDA."""

    release = Path(release_dir)
    root = Path(repo_root).resolve()
    violations = []
    details = {
        "mode": "artifact_only",
        "arms": 0,
        "task6_acceptance_passed": False,
    }
    provider = None
    try:
        if not allow_partial and not (release / "COMPLETE").is_file():
            raise ValueError("hierarchical release lacks COMPLETE")
        _validate_output_manifest(release)
        config = json.loads((release / "config.json").read_text(encoding="utf-8"))
        required_config = {
            "schema", "release_label", "created_at", "owner_decision",
            "action_schema", "checkpoint_schema", "manifest_relpath",
            "manifest_output_sha256", "source_preflight_relpath",
            "source_preflight_output_sha256", "hierarchical_identity_relpath",
            "hierarchical_identity_output_sha256", "arms", "updates_per_arm",
            "batch_size", "action_core_lr", "sidecar_finetune_lr",
            "gradient_clip_norm", "task6_acceptance_passed",
            "ppo_checkpoint_eligible", "ppo_training_started",
            "closed_loop_evaluation_started", "arm_selection_performed",
            "candidate_promoted", "test_opened", "final_pool", "device",
            "gpu_name", "python_executable", "python_version", "torch_version",
            "cuda_version", "numba_cache_dir", "cublas_workspace_config",
            "deterministic_algorithms", "registry", "reports",
        }
        if set(config) != required_config:
            raise ValueError("hierarchical release config schema drift")
        expected_files = {
            "config.json",
            "validation.json",
            "opened_registry.snapshot.tsv",
        }
        for arm in ARMS:
            expected_files.update(
                {
                    f"reports/{arm}.json",
                    f"reports/{arm}_episodes.tsv",
                    f"checkpoints/{arm}.pt",
                    f"curves/{arm}.npz",
                    f"scores/{arm}.npz",
                }
            )
        actual_files = {
            path.relative_to(release).as_posix()
            for path in release.rglob("*")
            if path.is_file()
            and path.name not in {"COMPLETE", "output_manifest.sha256"}
        }
        if actual_files != expected_files:
            raise ValueError("hierarchical release file inventory drift")
        stored_validation = json.loads(
            (release / "validation.json").read_text(encoding="utf-8")
        )
        if set(stored_validation) != {
            "schema",
            "passed",
            "integrity_passed",
            "task6_acceptance_passed",
            "mode",
            "arms",
            "violations",
        }:
            raise ValueError("hierarchical stored validation schema drift")
        if (
            stored_validation["schema"] != VALIDATION_SCHEMA
            or stored_validation["passed"] is not config["task6_acceptance_passed"]
            or stored_validation["integrity_passed"] is not True
            or stored_validation["task6_acceptance_passed"]
            is not config["task6_acceptance_passed"]
            or stored_validation["mode"]
            not in {"pending_same_device_full", "same_device_full", "artifact_only"}
            or stored_validation["arms"] != len(ARMS)
            or stored_validation["violations"] != []
        ):
            raise ValueError("hierarchical stored validation content drift")
        if not allow_partial and stored_validation["mode"] != "same_device_full":
            raise ValueError(
                "complete hierarchical release lacks same-device-full validation"
            )
        if (
            config["schema"] != RELEASE_SCHEMA
            or config["release_label"] != RELEASE_LABEL
            or config["owner_decision"] != OWNER_DECISION
            or config["action_schema"] != ACTION_SCHEMA
            or config["checkpoint_schema"] != CHECKPOINT_SCHEMA
            or config["arms"] != list(ARMS)
            or config["updates_per_arm"] != WARMSTART_UPDATES
            or config["batch_size"] != WARMSTART_BATCH_SIZE
            or config["action_core_lr"] != ACTION_CORE_LR
            or config["sidecar_finetune_lr"] != SIDECAR_FINETUNE_LR
            or config["gradient_clip_norm"] != GRAD_CLIP_NORM
            or config["ppo_checkpoint_eligible"] is not config["task6_acceptance_passed"]
            or config["ppo_training_started"] is not False
            or config["closed_loop_evaluation_started"] is not False
            or config["arm_selection_performed"] is not False
            or config["candidate_promoted"] is not False
            or config["test_opened"] is not False
            or config["final_pool"] is not False
            or config["deterministic_algorithms"] is not True
            or config["cublas_workspace_config"] not in {":4096:8", ":16:8"}
        ):
            raise ValueError("hierarchical release authority/scope drift")
        manifest = root / config["manifest_relpath"]
        manifest_check = validate_hierarchical_warmstart_manifest(manifest, root)
        if not manifest_check["passed"]:
            raise ValueError(f"hierarchical referenced manifest failed: {manifest_check}")
        if file_sha256(manifest / "output_manifest.sha256") != config["manifest_output_sha256"]:
            raise ValueError("hierarchical referenced manifest hash drift")
        manifest_config = json.loads(
            (manifest / "config.json").read_text(encoding="utf-8")
        )
        if (
            (root / manifest_config["source_preflight_relpath"]).resolve()
            != (root / config["source_preflight_relpath"]).resolve()
            or manifest_config["source_preflight_output_sha256"]
            != config["source_preflight_output_sha256"]
        ):
            raise ValueError("hierarchical release/manifest source-preflight mismatch")
        identity_release = root / config["hierarchical_identity_relpath"]
        identity_validation = validate_hierarchical_identity(
            identity_release, repo_root=root
        )
        if not identity_validation["passed"]:
            raise ValueError("hierarchical release identity prerequisite failed")
        if (
            config["hierarchical_identity_relpath"]
            != manifest_config["hierarchical_identity_relpath"]
            or config["hierarchical_identity_output_sha256"]
            != manifest_config["hierarchical_identity_output_sha256"]
            or file_sha256(identity_release / "output_manifest.sha256")
            != config["hierarchical_identity_output_sha256"]
        ):
            raise ValueError("hierarchical release/manifest identity mismatch")
        registry_rows = _read_registry(manifest / "registry_rows.tsv")
        registry_ledger = manifest_config["registry"]
        expected_registry_release = {
            "state_before_append": config["registry"]["state_before_append"],
            "before_observed_sha256": config["registry"][
                "before_observed_sha256"
            ],
            "after_sha256": registry_ledger["after_expected_sha256"],
            "rows_appended": config["registry"]["rows_appended"],
            "rows_skipped": config["registry"]["rows_skipped"],
            "total_rows": registry_ledger["after_rows"],
            "planned_rows": len(registry_rows),
        }
        if config["registry"] != expected_registry_release:
            raise ValueError("hierarchical release registry ledger schema drift")
        if config["registry"]["state_before_append"] == "ready":
            expected_counts = (len(registry_rows), 0)
            expected_before_sha = registry_ledger["before_sha256"]
        elif config["registry"]["state_before_append"] == "already_appended":
            expected_counts = (0, len(registry_rows))
            expected_before_sha = registry_ledger["after_expected_sha256"]
        else:
            raise ValueError("hierarchical release registry pre-state drift")
        if (
            (config["registry"]["rows_appended"], config["registry"]["rows_skipped"])
            != expected_counts
            or config["registry"]["before_observed_sha256"]
            != expected_before_sha
            or file_sha256(release / "opened_registry.snapshot.tsv")
            != registry_ledger["after_expected_sha256"]
            or _read_registry(release / "opened_registry.snapshot.tsv")
            != _read_registry(manifest / "registry_after.expected.tsv")
            or _registry_live_state(
                root / REGISTRY_RELPATH,
                registry_rows,
                registry_ledger["before_sha256"],
                registry_ledger["after_expected_sha256"],
            )
            != "already_appended"
        ):
            raise ValueError("hierarchical release registry provenance drift")
        source = root / config["source_preflight_relpath"]
        if file_sha256(source / "output_manifest.sha256") != config["source_preflight_output_sha256"]:
            raise ValueError("hierarchical source preflight hash drift")
        examples = _read_tsv(manifest / "macro_examples.tsv")
        episodes = _read_tsv(manifest / "episodes.tsv")
        with np.load(manifest / "calibration_indices.npz", allow_pickle=False) as values:
            positive_indices = values["positive_active"].copy()
            negative_indices = values["negative"].copy()
        if set(config["reports"]) != set(ARMS):
            raise ValueError("hierarchical release report inventory drift")
        reports = {}
        checkpoints = {}
        stored_scores = {}
        for arm in ARMS:
            paths = {
                "report": release / "reports" / f"{arm}.json",
                "episodes": release / "reports" / f"{arm}_episodes.tsv",
                "checkpoint": release / "checkpoints" / f"{arm}.pt",
                "curve": release / "curves" / f"{arm}.npz",
                "scores": release / "scores" / f"{arm}.npz",
            }
            record = config["reports"][arm]
            if set(record) != {
                "report_sha256", "episode_decisions_sha256", "checkpoint_sha256",
                "curve_sha256", "scores_sha256",
            }:
                raise ValueError(f"hierarchical report hash schema drift: {arm}")
            for key, config_key in (
                ("report", "report_sha256"),
                ("episodes", "episode_decisions_sha256"),
                ("checkpoint", "checkpoint_sha256"),
                ("curve", "curve_sha256"),
                ("scores", "scores_sha256"),
            ):
                if file_sha256(paths[key]) != record[config_key]:
                    raise ValueError(f"hierarchical arm artifact hash drift: {arm}/{key}")
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            _finite_tree(report)
            expected_report_keys = {
                "schema", "release_label", "arm", "action_schema",
                "checkpoint_schema", "updates_completed", "batch_size",
                "early_stopping", "arm_selection_performed", "ppo_training_started",
                "losses_are_diagnostic_only", "fresh_initial_intervention_logit",
                "fresh_initial_brake_logit", "priors", "manifest_output_sha256",
                "schedule_sha256", "fresh_hashes", "initial_hashes", "final_hashes",
                "prior_changed_state_keys",
                "calibration", "task6_acceptance_passed", "curve_final",
                "arm_acceptance_passed", "curve_min_total_loss",
            }
            if set(report) != expected_report_keys:
                raise ValueError(f"hierarchical arm report schema drift: {arm}")
            if (
                report["schema"] != REPORT_SCHEMA
                or report["release_label"] != RELEASE_LABEL
                or report["arm"] != arm
                or report["action_schema"] != ACTION_SCHEMA
                or report["checkpoint_schema"] != CHECKPOINT_SCHEMA
                or report["updates_completed"] != WARMSTART_UPDATES
                or report["batch_size"] != WARMSTART_BATCH_SIZE
                or report["early_stopping"] is not False
                or report["arm_selection_performed"] is not False
                or report["ppo_training_started"] is not False
                or report["losses_are_diagnostic_only"] is not True
                or report["fresh_initial_intervention_logit"] != INITIAL_INTERVENTION_LOGIT
                or report["fresh_initial_brake_logit"] != INITIAL_BRAKE_LOGIT
                or report["manifest_output_sha256"] != config["manifest_output_sha256"]
                or report["schedule_sha256"] != file_sha256(manifest / "training_schedule.npy")
                or report["priors"] != manifest_config["priors"]
                or report["arm_acceptance_passed"] is not report["calibration"]["passed"]
                or report["task6_acceptance_passed"] is not config["task6_acceptance_passed"]
            ):
                raise ValueError(f"hierarchical arm report scope drift: {arm}")
            positive_raw, negative_raw, conditional_raw = _load_score_arrays(paths["scores"])
            if (
                positive_raw.dtype != np.float32
                or negative_raw.dtype != np.float32
                or conditional_raw.dtype != np.float32
            ):
                raise ValueError(f"hierarchical score dtype drift: {arm}")
            recomputed, episode_rows = derive_negative_only_calibration(
                examples, episodes, positive_raw, negative_raw, conditional_raw, arm
            )
            if recomputed != report["calibration"]:
                raise ValueError(f"hierarchical recorded calibration drift: {arm}")
            if _read_tsv(paths["episodes"]) != episode_rows:
                raise ValueError(f"hierarchical episode-decision ledger drift: {arm}")
            with np.load(paths["curve"], allow_pickle=False) as curves:
                expected_curves = {
                    "total_loss", "intervention_loss", "steer_loss",
                    "conditional_brake_loss", "brake_loss", "gradient_norm",
                }
                if set(curves.files) != expected_curves:
                    raise ValueError(f"hierarchical curve inventory drift: {arm}")
                if any(
                    curves[name].shape != (WARMSTART_UPDATES,)
                    or curves[name].dtype != np.float64
                    or not np.all(np.isfinite(curves[name]))
                    for name in curves.files
                ):
                    raise ValueError(f"hierarchical curve shape/value drift: {arm}")
            checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
            if set(checkpoint) != {
                "schema", "action_schema", "release_label", "arm",
                "manifest_output_sha256", "task6_acceptance_passed",
                "calibration_offset_float32", "state_dict", "state_dict_sha256",
            }:
                raise ValueError(f"hierarchical checkpoint envelope schema drift: {arm}")
            state = checkpoint["state_dict"]
            if (
                checkpoint["schema"] != CHECKPOINT_SCHEMA
                or checkpoint["action_schema"] != ACTION_SCHEMA
                or checkpoint["release_label"] != RELEASE_LABEL
                or checkpoint["arm"] != arm
                or checkpoint["manifest_output_sha256"] != config["manifest_output_sha256"]
                or checkpoint["task6_acceptance_passed"] is not config["task6_acceptance_passed"]
                or checkpoint["calibration_offset_float32"] != recomputed["applied_offset_float32"]
                or checkpoint["state_dict_sha256"] != _tensor_digest(state.items())
                or "intervention_gate.weight" not in state
                or "intervention_logit_offset" not in state
                or float(state["intervention_logit_offset"].item())
                != recomputed["applied_offset_float32"]
            ):
                raise ValueError(f"hierarchical checkpoint content drift: {arm}")
            reports[arm] = report
            checkpoints[arm] = checkpoint
            stored_scores[arm] = (positive_raw, negative_raw, conditional_raw)

        _assert_training_invariants(reports)
        aggregate = all(reports[arm]["arm_acceptance_passed"] for arm in ARMS)
        if config["task6_acceptance_passed"] is not aggregate:
            raise ValueError("hierarchical aggregate Task-6 decision drift")

        if device_name is not None:
            device = torch.device(device_name)
            _set_deterministic_cuda(device)
            if (
                str(device) != config["device"]
                or torch.cuda.get_device_name(device) != config["gpu_name"]
                or torch.__version__ != config["torch_version"]
                or torch.version.cuda != config["cuda_version"]
            ):
                raise ValueError(
                    "hierarchical numerical validator is not on recorded same device"
                )
            provider = WarmstartBatchProvider(root, manifest, device)
            sidecar_state, sidecar_mean, sidecar_std, _ = load_sidecar_bundle(
                root / SIDECAR_RELEASE_RELPATH
            )
            bc_state = {
                name: value.detach().cpu().contiguous()
                for name, value in provider.bc.state_dict().items()
            }
            for arm in ARMS:
                policy = RemediatedV22Policy(
                    arm,
                    bc_state_dict=bc_state,
                    sidecar_state_dict=sidecar_state,
                    sidecar_bc_mean=sidecar_mean,
                    sidecar_bc_std=sidecar_std,
                    initialization_seed=POLICY_SEED,
                ).to(device)
                if _policy_hashes(policy) != reports[arm]["fresh_hashes"]:
                    raise ValueError(
                        f"hierarchical fresh-state recomputation mismatch: {arm}"
                    )
                initialize_hierarchical_priors(
                    policy,
                    manifest_config["priors"]["intervention"][
                        "applied_logit_float32"
                    ],
                    manifest_config["priors"]["conditional_brake"][
                        "applied_logit_float32"
                    ],
                )
                if _policy_hashes(policy) != reports[arm]["initial_hashes"]:
                    raise ValueError(
                        f"hierarchical post-prior recomputation mismatch: {arm}"
                    )
                policy.load_hierarchical_state_dict(checkpoints[arm]["state_dict"])
                policy.eval()
                if _policy_hashes(policy) != reports[arm]["final_hashes"]:
                    raise ValueError(
                        f"hierarchical loaded-checkpoint hash mismatch: {arm}"
                    )
                observed_positive, observed_conditional = _raw_scores(
                    policy, provider, positive_indices, device
                )
                observed_negative, _ = _raw_scores(
                    policy, provider, negative_indices, device
                )
                expected_positive, expected_negative, expected_conditional = stored_scores[arm]
                if not (
                    np.array_equal(observed_positive, expected_positive)
                    and np.array_equal(observed_negative, expected_negative)
                    and np.array_equal(observed_conditional, expected_conditional)
                ):
                    raise ValueError(f"hierarchical CUDA score recomputation mismatch: {arm}")
                del policy
                gc.collect()
                torch.cuda.empty_cache()
            details["mode"] = "same_device_full"
        details.update(
            {
                "arms": len(ARMS),
                "task6_acceptance_passed": aggregate,
            }
        )
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    finally:
        if provider is not None:
            del provider
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "schema": VALIDATION_SCHEMA,
        "passed": not violations and details["task6_acceptance_passed"],
        "integrity_passed": not violations,
        **details,
        "violations": violations,
    }


# Deliberately explicit aliases for callers that use the historical Task-6 naming.
create_hierarchical_manifest = create_hierarchical_warmstart_manifest
validate_hierarchical_manifest = validate_hierarchical_warmstart_manifest
run_hierarchical_warmstart_smoke = run_hierarchical_warmstart
validate_hierarchical_warmstart = validate_hierarchical_warmstart_release

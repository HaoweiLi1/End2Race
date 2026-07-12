"""Task-6 witness/preservation manifest and fixed-update warm-start smoke."""

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
    D2_DATASET_MANIFEST_SHA256,
    D2_TEST_SEAL_SHA256,
    HISTORY_OFFSETS,
    INITIAL_BRAKE_LOGIT,
    MACRO_STEPS,
    OWNER_DECISION,
    SEED as POLICY_SEED,
    SIDECAR_FINETUNE_LR,
    STEER_BUDGET,
)
from bplus_v22.model import V22Policy
from bplus_v22.release import file_sha256, validate_pinned_inputs, validate_source_preflight
from bplus_v22.sidecar import (
    DATASET_RELPATH,
    EVIDENCE_RELPATH,
    REGISTRY_RELPATH,
    SIGNALS_RELPATH,
    SPLIT_RELPATH,
    _episode_rows as d2_episode_rows,
    _read_registry,
    _tensor_digest,
    _validate_output_manifest,
    _validate_sealed_test_absence,
    _write_json,
    _write_output_manifest,
    load_sidecar_bundle,
    validate_sidecar_release,
)
from d0.identity import (
    REGISTRY_FIELDS,
    append_opened_registry,
    registry_row_id,
    validate_registry_row,
)
from d2.replay import build_speed_inputs
from d2r.data import D2RDataset
from ppo_utils import load_frozen_bc


D25_RELPATH = "Experiments/A4_d25_counterfactual/artifacts/full_oracle_20260711_185500"
D25_OUTPUT_MANIFEST_SHA256 = (
    "42a31686a1c654bfe702085d0a7ae4f587e02e4807ae9eba33fae7ad600dcca3"
)
D25_CASE_RESULTS_SHA256 = (
    "0ef0a09adba1d46d76151187a4d295ce149ad8409a458befc7950d7d3f7b7c1b"
)
D25_BRANCH_RESULTS_SHA256 = (
    "252af4959dfd9aeb91e4599e6fce47a68cfda54af759aa51a3c628389fdd0a2e"
)
D01_RELPATH = (
    "Experiments/A0_project_registry/artifacts/"
    "d01_full_reconcile_20260711_170200_a"
)
D01_OUTPUT_MANIFEST_SHA256 = (
    "425d62097b1463e72fca33f4e08690385bfbd21e6be3a91db900b92e4664bd89"
)
D01_CANONICAL_SHA256 = (
    "793193deefc942f556ec23ee4e34fea3597eac761eb0b1f676af2667ff6b62e2"
)
SIDECAR_RELEASE_RELPATH = (
    "Experiments/B1_route_r2_scaffold/artifacts/sidecar_init_20260712_080012"
)
SIDECAR_OUTPUT_MANIFEST_SHA256 = (
    "ac9e10661102efb1164aaa7b6d57fdbf0a63be9c1af454ddc9954d30031163a7"
)
SIDECAR_STATE_DICT_SHA256 = (
    "34158ecba356ec9d524529e0d928e452140f8da2f98c59d491f0a5cf26cd12e5"
)

EXPECTED_REGISTRY_BEFORE_SHA256 = (
    "753c478700a499fa24f1c216f77e810bd1f634ba9cc7d934a2ec707593b1439c"
)
EXPECTED_REGISTRY_AFTER_SHA256 = (
    "aff5f03db06836c6c51ff53944ed2ec2e521fbe777cc7d26228a15a9362d0b0d"
)
FAILED_WARMSTART_RELPATH = (
    "Experiments/B1_route_r2_scaffold/artifacts/"
    "warmstart_smoke_20260712_091950"
)
FAILED_WARMSTART_OUTPUT_MANIFEST_SHA256 = (
    "150b41fa68fbec40442741bdc6613355ab41b44cc0fdb4591fa9e455438dc8be"
)
PRIOR_WARMSTART_MANIFEST_RELPATH = (
    "Experiments/B1_route_r2_scaffold/artifacts/"
    "warmstart_manifest_20260712_091851"
)
PRIOR_WARMSTART_MANIFEST_OUTPUT_SHA256 = (
    "8b53294f7049d53a0a7261c9daa8acfe9df88857e8ba211aafe09bf05ad915a2"
)
REGISTRY_OPENED_AT = "2026-07-12T09:00:00+08:00"
REGISTRY_STAGE = "D3-R2-v2.2"
REGISTRY_USE_CLASS = "actor_pretrain"
REGISTRY_DECISION_EFFECT = "action_choice"

PRESERVATION_DOMAIN = b"end2race:bplus-v2.2:preservation:v1\0"
EXAMPLE_DOMAIN = b"end2race:bplus-v2.2:warmstart-example:v1\0"
DIAGNOSTIC_DOMAIN = b"end2race:bplus-v2.2:warmstart-diagnostic:v1\0"
WARMSTART_SEED = 20260712
WARMSTART_UPDATES = 1024
WARMSTART_BATCH_SIZE = 256
INTERVENTION_PER_BATCH = 128
WITNESS_NOOP_PER_BATCH = 64
PRESERVATION_PER_BATCH = 64
GRAD_CLIP_NORM = 1.0
GATE_BCE_ACCEPTANCE_MAX = 0.5382
GATE_SPECIFICITY_ACCEPTANCE_MIN = 0.05
EXPECTED_FIT_GATE_POSITIVES = 90089
EXPECTED_FIT_GATE_TOTAL = WARMSTART_UPDATES * WARMSTART_BATCH_SIZE
EXPECTED_DIAGNOSTIC_GATE_POSITIVES = 200
WARMSTART_RELEASE_LABEL = "ACTION_WARMSTART_REMEDIATION"
EXPECTED_WITNESSES = 67
EXPECTED_INTERVENTION_MACROS = 291
EXPECTED_PRESERVATION_CANDIDATES = 1061
EXPECTED_PRESERVATION_EPISODES = 602
EXPECTED_DIAGNOSTIC_PER_CLASS = EXPECTED_INTERVENTION_MACROS

ACTOR_INPUT_FIELDS = (
    "bc_feature",
    "lidar_history",
    "actual_speed_history",
    "previous_desired_steer_history",
    "previous_desired_speed_history",
)
PRIVILEGED_MANIFEST_FIELDS = (
    "role",
    "target_steer_hex",
    "target_brake_hex",
    "target_brake_gate",
    "active_intervention",
    "witness_branch_id",
    "confirmed_safe_pass",
)
FORBIDDEN_ACTOR_FIELDS = (
    "ego_pose",
    "opp_pose",
    "ego_progress",
    "opp_progress",
    "map_name",
    "l2_id",
    "l3_id",
    "l4_id",
    "collision",
    "confirmed_safe_pass",
    "witness_branch_id",
    "target_steer",
    "target_brake",
    "target_brake_gate",
)

EPISODE_FIELDS = (
    "episode_order",
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


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: Iterable[Mapping[str, str]], fields) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            if set(row) != set(fields):
                raise ValueError(f"warm-start TSV field mismatch: {path.name}")
            writer.writerow({field: row[field] for field in fields})


def _prepare_output(output: Path) -> Path:
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("warm-start output/partial already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    return partial


def _promote(partial: Path, output: Path) -> None:
    os.replace(partial, output)
    (output / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")


def _score(domain: bytes, *parts: str) -> str:
    digest = hashlib.sha256(domain)
    for part in parts:
        digest.update(str(part).encode("utf-8") + b"\0")
    return digest.hexdigest()


def _manifest_entries(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relpath = line.split("  ", 1)
        if relpath in entries:
            raise ValueError("warm-start source manifest has duplicate path")
        entries[relpath] = digest
    return entries


def _validate_frozen_sources(root: Path) -> None:
    checks = (
        (root / D25_RELPATH / "output_manifest.sha256", D25_OUTPUT_MANIFEST_SHA256),
        (root / D25_RELPATH / "case_results.tsv", D25_CASE_RESULTS_SHA256),
        (root / D25_RELPATH / "branch_results.tsv", D25_BRANCH_RESULTS_SHA256),
        (root / D01_RELPATH / "output_manifest.sha256", D01_OUTPUT_MANIFEST_SHA256),
        (root / D01_RELPATH / "canonical_episodes.tsv", D01_CANONICAL_SHA256),
        (root / SIDECAR_RELEASE_RELPATH / "output_manifest.sha256", SIDECAR_OUTPUT_MANIFEST_SHA256),
        (
            root / PRIOR_WARMSTART_MANIFEST_RELPATH / "output_manifest.sha256",
            PRIOR_WARMSTART_MANIFEST_OUTPUT_SHA256,
        ),
        (
            root / FAILED_WARMSTART_RELPATH / "output_manifest.sha256",
            FAILED_WARMSTART_OUTPUT_MANIFEST_SHA256,
        ),
        (root / "pretrained/end2race.pth", BC_CHECKPOINT_SHA256),
        (root / SPLIT_RELPATH / "test_seal.json", D2_TEST_SEAL_SHA256),
    )
    for path, expected in checks:
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"warm-start frozen source hash drift: {path}")
    sidecar = validate_sidecar_release(root / SIDECAR_RELEASE_RELPATH, root)
    if not sidecar["passed"] or sidecar["state_dict_sha256"] != SIDECAR_STATE_DICT_SHA256:
        raise ValueError("warm-start sidecar release validation failed")


def _validate_failed_predecessor(root: Path) -> dict:
    release = root / FAILED_WARMSTART_RELPATH
    _validate_output_manifest(release)
    config = json.loads((release / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("ppo_training_started") is not False
        or config.get("arm_selection_performed") is not False
        or config.get("closed_loop_evaluation_started") is not False
    ):
        raise ValueError("failed warm-start predecessor scope drift")
    metrics = {}
    for arm in ARMS:
        report = json.loads(
            (release / "reports" / f"{arm}.json").read_text(encoding="utf-8")
        )
        after = report["diagnostic_after"]
        if not (
            after["gate_recall"] == 0.0
            and after["gate_specificity"] == 1.0
            and after["gate_loss"] >= GATE_BCE_ACCEPTANCE_MAX
        ):
            raise ValueError("failed warm-start predecessor metric drift")
        metrics[arm] = {
            "gate_recall": after["gate_recall"],
            "gate_specificity": after["gate_specificity"],
            "gate_loss": after["gate_loss"],
        }
    return {
        "relpath": FAILED_WARMSTART_RELPATH,
        "output_manifest_sha256": FAILED_WARMSTART_OUTPUT_MANIFEST_SHA256,
        "integrity_validation_preserved": True,
        "task6_stage_decision": "FAILED",
        "reason": "all arms failed recall and marginal-BCE acceptance bars",
        "metrics": metrics,
    }


def _witness_episode_rows(root: Path) -> list[dict[str, str]]:
    d25 = root / D25_RELPATH
    cases = _read_tsv(d25 / "case_results.tsv")
    case_manifest = {row["l2_id"]: row for row in _read_tsv(d25 / "case_manifest.tsv")}
    branches = {
        (row["l2_id"], row["branch_id"]): row
        for row in _read_tsv(d25 / "branch_results.tsv")
    }
    output_entries = _manifest_entries(d25 / "output_manifest.sha256")
    selected = [row for row in cases if row["status"] == "recovered_confirmed_safe_pass"]
    if len(selected) != EXPECTED_WITNESSES:
        raise ValueError("warm-start witness count drift")
    rows = []
    for case in sorted(selected, key=lambda row: row["l2_id"]):
        l2_id = case["l2_id"]
        branch = branches[(l2_id, case["witness_branch_id"])]
        source = case_manifest[l2_id]
        if (
            branch["confirmed_safe_pass"] != "true"
            or branch["action_clipped"] != "false"
            or branch["rerun_match"] != "true"
            or branch["trajectory_sha256"] != branch["deterministic_rerun_sha256"]
            or branch["trajectory_sha256"] != case["witness_trajectory_sha256"]
            or int(branch["start_step"]) % MACRO_STEPS != 0
            or int(branch["duration_steps"]) % MACRO_STEPS != 0
        ):
            raise ValueError(f"warm-start invalid witness evidence: {l2_id}")
        candidates = sorted((d25 / "witnesses").glob(f"{l2_id[3:]}__{branch['branch_id']}.npz"))
        if len(candidates) != 1:
            raise ValueError(f"warm-start witness file mismatch: {l2_id}")
        witness_path = candidates[0]
        relpath = witness_path.relative_to(d25).as_posix()
        if output_entries.get(relpath) != file_sha256(witness_path):
            raise ValueError(f"warm-start witness output-manifest mismatch: {l2_id}")
        with np.load(witness_path, allow_pickle=False) as arrays:
            frame_count = len(arrays["time"])
            if (
                bool(np.asarray(arrays["collision"]).reshape(()))
                or bool(np.asarray(arrays["ego_collision"]).reshape(()))
                or bool(np.asarray(arrays["opp_collision"]).reshape(()))
                or str(np.asarray(arrays["state_label"]).reshape(())) != "overtaking"
            ):
                raise ValueError(f"warm-start witness terminal state drift: {l2_id}")
        if int(branch["start_step"]) + int(branch["duration_steps"]) > frame_count:
            raise ValueError(f"warm-start witness intervention exceeds trajectory: {l2_id}")
        rows.append(
            {
                "episode_order": "-1",
                "role": "witness",
                "l2_id": l2_id,
                "l3_id": source["l3_id"],
                "l4_id": source["l4_id"],
                "map_name": source["map_name"],
                "skill": source["skill"],
                "opponent_raceline": source["opponent_raceline"],
                "speedscale_hex": source["speedscale_hex"],
                "resolved_ego_idx": source["resolved_ego_idx"],
                "source_npz_relpath": f"{D25_RELPATH}/{relpath}",
                "source_npz_sha256": file_sha256(witness_path),
                "source_trajectory_sha256": branch["trajectory_sha256"],
                "frame_start_global": "NA",
                "frame_count": str(frame_count),
                "witness_branch_id": branch["branch_id"],
                "intervention_start_step": branch["start_step"],
                "intervention_duration_steps": branch["duration_steps"],
                "target_brake_hex": branch["brake_mps_hex"],
                "target_steer_hex": branch["steer_rad_hex"],
                "confirmed_safe_pass": "true",
                "action_clipped": "false",
                "preservation_stratum": "NA",
            }
        )
    return rows


def _preservation_episode_rows(
    root: Path, witness_ids: set[str]
) -> tuple[list[dict[str, str]], int]:
    metadata = {row["l2_id"]: row for row in d2_episode_rows(root)}
    canonical = _read_tsv(root / D01_RELPATH / "canonical_episodes.tsv")
    candidates = []
    for row in canonical:
        if row["model_id"] != "bc" or row["l2_id"] not in metadata:
            continue
        if (
            row["confirmed_safe_pass"] == "True"
            and row["four_state"] == "confirmed_pass"
            and row["corrected_outcome3"] == "overtake"
            and row["collision_any"] == "False"
            and row["censored"] == "False"
            and row["alignment_status"] == "ok"
            and row["physics_status"] == "ok"
            and row["frame_spacing_status"] == "ok"
        ):
            candidates.append(row)
    if len(candidates) != EXPECTED_PRESERVATION_CANDIDATES:
        raise ValueError("warm-start preservation candidate count drift")
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in candidates:
        key = (
            row["map_name"],
            row["skill"],
            row["opponent_raceline"],
            row["l4_id"],
        )
        groups.setdefault(key, []).append(row)
    selected = []
    for key in sorted(groups):
        chosen = min(
            groups[key],
            key=lambda row: (_score(PRESERVATION_DOMAIN, row["l2_id"]), row["l2_id"]),
        )
        selected.append((key, chosen))
    if len(selected) != EXPECTED_PRESERVATION_EPISODES:
        raise ValueError("warm-start preservation stratum count drift")
    if witness_ids & {row["l2_id"] for _, row in selected}:
        raise ValueError("warm-start preservation duplicates witness L2")
    rows = []
    for key, selected_row in sorted(selected, key=lambda item: item[1]["l2_id"]):
        source = metadata[selected_row["l2_id"]]
        rows.append(
            {
                "episode_order": "-1",
                "role": "preservation",
                "l2_id": selected_row["l2_id"],
                "l3_id": selected_row["l3_id"],
                "l4_id": selected_row["l4_id"],
                "map_name": selected_row["map_name"],
                "skill": selected_row["skill"],
                "opponent_raceline": selected_row["opponent_raceline"],
                "speedscale_hex": selected_row["speedscale_hex"],
                "resolved_ego_idx": source["resolved_ego_idx"],
                "source_npz_relpath": source["npz_relpath"],
                "source_npz_sha256": source["npz_sha256"],
                "source_trajectory_sha256": "NA",
                "frame_start_global": source["frame_start"],
                "frame_count": source["frame_count"],
                "witness_branch_id": "NA",
                "intervention_start_step": "NA",
                "intervention_duration_steps": "NA",
                "target_brake_hex": float(0.0).hex(),
                "target_steer_hex": float(0.0).hex(),
                "confirmed_safe_pass": "true",
                "action_clipped": "false",
                "preservation_stratum": "|".join(key),
            }
        )
    return rows, len(candidates)


def build_episode_manifest(root: str | Path) -> tuple[list[dict[str, str]], dict]:
    root = Path(root).resolve()
    witnesses = _witness_episode_rows(root)
    witness_ids = {row["l2_id"] for row in witnesses}
    preservation, candidate_count = _preservation_episode_rows(root, witness_ids)
    rows = witnesses + preservation
    for index, row in enumerate(rows):
        row["episode_order"] = str(index)
    if len(rows) != EXPECTED_WITNESSES + EXPECTED_PRESERVATION_EPISODES:
        raise AssertionError("warm-start episode manifest total drift")
    return rows, {
        "witnesses": len(witnesses),
        "preservation_candidates": candidate_count,
        "preservation_strata": len(preservation),
        "total_episodes": len(rows),
    }


def build_macro_examples(
    episodes: Iterable[Mapping[str, str]],
) -> tuple[list[dict[str, str]], dict]:
    rows = []
    counts = {"intervention": 0, "witness_noop": 0, "preservation_noop": 0}
    for episode in episodes:
        frame_count = int(episode["frame_count"])
        start = (
            int(episode["intervention_start_step"])
            if episode["role"] == "witness"
            else -1
        )
        duration = (
            int(episode["intervention_duration_steps"])
            if episode["role"] == "witness"
            else 0
        )
        episode_active = 0
        for macro_index, frame_index in enumerate(range(0, frame_count, MACRO_STEPS)):
            active = episode["role"] == "witness" and start <= frame_index < start + duration
            if active:
                brake_hex = episode["target_brake_hex"]
                steer_hex = episode["target_steer_hex"]
                category = "intervention"
            else:
                brake_hex = float(0.0).hex()
                steer_hex = float(0.0).hex()
                category = (
                    "witness_noop" if episode["role"] == "witness" else "preservation_noop"
                )
            brake = float.fromhex(brake_hex)
            steer = float.fromhex(steer_hex)
            if not 0.0 <= brake <= BRAKE_BUDGET or not -STEER_BUDGET <= steer <= STEER_BUDGET:
                raise ValueError("warm-start macro label outside action support")
            payload = (
                episode["role"],
                episode["l2_id"],
                str(frame_index),
                str(active).lower(),
                brake_hex,
                steer_hex,
            )
            example_id = _score(EXAMPLE_DOMAIN, *payload)
            global_frame = (
                str(int(episode["frame_start_global"]) + frame_index)
                if episode["role"] == "preservation"
                else "NA"
            )
            rows.append(
                {
                    "example_index": str(len(rows)),
                    "example_id": example_id,
                    "role": category,
                    "l2_id": episode["l2_id"],
                    "macro_index": str(macro_index),
                    "frame_index": str(frame_index),
                    "global_frame_index": global_frame,
                    "active_intervention": str(active).lower(),
                    "target_brake_gate": str(int(active and brake > 0.0)),
                    "target_brake_hex": brake_hex,
                    "target_steer_hex": steer_hex,
                }
            )
            counts[category] += 1
            episode_active += int(active)
        expected_active = duration // MACRO_STEPS
        if episode_active != expected_active:
            raise ValueError(
                f"warm-start intervention macro accounting mismatch: {episode['l2_id']}"
            )
    if counts["intervention"] != EXPECTED_INTERVENTION_MACROS:
        raise ValueError("warm-start intervention macro count drift")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("warm-start duplicate example ID")
    counts["total"] = len(rows)
    return rows, counts


def _cyclic_draw(pool: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(pool) == 0 or count <= 0:
        raise ValueError("warm-start cyclic draw requires nonempty pool/count")
    output = []
    remaining = count
    while remaining:
        permutation = pool[rng.permutation(len(pool))]
        take = min(remaining, len(permutation))
        output.append(permutation[:take])
        remaining -= take
    return np.concatenate(output).astype(np.int32, copy=False)


def build_training_schedule(
    examples: list[Mapping[str, str]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    pools = {
        role: np.asarray(
            [int(row["example_index"]) for row in examples if row["role"] == role],
            dtype=np.int32,
        )
        for role in ("intervention", "witness_noop", "preservation_noop")
    }
    rng = np.random.default_rng(WARMSTART_SEED)
    intervention = _cyclic_draw(
        pools["intervention"], WARMSTART_UPDATES * INTERVENTION_PER_BATCH, rng
    ).reshape(WARMSTART_UPDATES, INTERVENTION_PER_BATCH)
    witness = _cyclic_draw(
        pools["witness_noop"], WARMSTART_UPDATES * WITNESS_NOOP_PER_BATCH, rng
    ).reshape(WARMSTART_UPDATES, WITNESS_NOOP_PER_BATCH)
    preservation = _cyclic_draw(
        pools["preservation_noop"],
        WARMSTART_UPDATES * PRESERVATION_PER_BATCH,
        rng,
    ).reshape(WARMSTART_UPDATES, PRESERVATION_PER_BATCH)
    schedule = np.empty((WARMSTART_UPDATES, WARMSTART_BATCH_SIZE), dtype=np.int32)
    for update in range(WARMSTART_UPDATES):
        batch = np.concatenate([intervention[update], witness[update], preservation[update]])
        schedule[update] = batch[rng.permutation(len(batch))]

    def diagnostic_subset(role: str) -> np.ndarray:
        ranked = sorted(
            pools[role].tolist(),
            key=lambda index: (
                _score(DIAGNOSTIC_DOMAIN, examples[index]["example_id"]),
                examples[index]["example_id"],
            ),
        )
        return np.asarray(ranked[:EXPECTED_DIAGNOSTIC_PER_CLASS], dtype=np.int32)

    diagnostic = np.concatenate(
        [
            pools["intervention"],
            diagnostic_subset("witness_noop"),
            diagnostic_subset("preservation_noop"),
        ]
    ).astype(np.int32, copy=False)
    if diagnostic.shape != (EXPECTED_DIAGNOSTIC_PER_CLASS * 3,):
        raise AssertionError("warm-start diagnostic schedule shape drift")
    return schedule, diagnostic, {
        "updates": WARMSTART_UPDATES,
        "batch_size": WARMSTART_BATCH_SIZE,
        "intervention_per_batch": INTERVENTION_PER_BATCH,
        "witness_noop_per_batch": WITNESS_NOOP_PER_BATCH,
        "preservation_noop_per_batch": PRESERVATION_PER_BATCH,
        "diagnostic_examples": len(diagnostic),
    }


def build_gate_prior(
    examples: list[Mapping[str, str]],
    schedule: np.ndarray,
    diagnostic: np.ndarray,
) -> dict:
    labels = np.asarray(
        [int(row["target_brake_gate"]) for row in examples], dtype=np.int64
    )
    fit_labels = labels[np.asarray(schedule, dtype=np.int64).reshape(-1)]
    diagnostic_labels = labels[np.asarray(diagnostic, dtype=np.int64)]
    fit_positive = int(fit_labels.sum())
    fit_total = int(len(fit_labels))
    diagnostic_positive = int(diagnostic_labels.sum())
    diagnostic_total = int(len(diagnostic_labels))
    if (
        fit_positive != EXPECTED_FIT_GATE_POSITIVES
        or fit_total != EXPECTED_FIT_GATE_TOTAL
        or diagnostic_positive != EXPECTED_DIAGNOSTIC_GATE_POSITIVES
        or diagnostic_total != EXPECTED_DIAGNOSTIC_PER_CLASS * 3
    ):
        raise ValueError("warm-start gate-prior label accounting drift")
    prevalence = fit_positive / fit_total
    derived_bias = math.log(prevalence / (1.0 - prevalence))
    applied_bias = float(np.float32(derived_bias))
    diagnostic_prevalence = diagnostic_positive / diagnostic_total
    diagnostic_marginal_bce = -(
        diagnostic_prevalence * math.log(diagnostic_prevalence)
        + (1.0 - diagnostic_prevalence) * math.log1p(-diagnostic_prevalence)
    )
    if not diagnostic_marginal_bce < GATE_BCE_ACCEPTANCE_MAX:
        raise AssertionError("warm-start marginal BCE is not below frozen bound")
    return {
        "schema": "bplus-v2.2-warmstart-gate-prior-1",
        "source": "exact flattened fit schedule target_brake_gate occurrences",
        "uses_diagnostic_labels": False,
        "fit_positive_labels": fit_positive,
        "fit_total_labels": fit_total,
        "fit_prevalence": prevalence,
        "fit_prevalence_hex": float(prevalence).hex(),
        "derived_bias_float64": derived_bias,
        "derived_bias_float64_hex": float(derived_bias).hex(),
        "applied_bias_float32": applied_bias,
        "applied_bias_float32_hex": float(applied_bias).hex(),
        "diagnostic_positive_labels": diagnostic_positive,
        "diagnostic_total_labels": diagnostic_total,
        "diagnostic_marginal_bce": diagnostic_marginal_bce,
        "gate_bce_acceptance_max_strict": GATE_BCE_ACCEPTANCE_MAX,
        "gate_specificity_acceptance_min_strict": GATE_SPECIFICITY_ACCEPTANCE_MIN,
    }


def make_action_choice_registry_rows(
    episodes: Iterable[Mapping[str, str]],
) -> list[dict[str, str]]:
    rows = []
    for episode in episodes:
        witness = episode["role"] == "witness"
        row = {
            "registry_schema": "bplus-opened-registry-1",
            "opened_at_utc": REGISTRY_OPENED_AT,
            "stage": REGISTRY_STAGE,
            "use_class": REGISTRY_USE_CLASS,
            "split_id": (
                "d3r2_v22_action_choice_witness"
                if witness
                else "d3r2_v22_action_choice_preservation"
            ),
            "l2_id": episode["l2_id"],
            "l3_id": episode["l3_id"],
            "l4_id": episode["l4_id"],
            "map_name": episode["map_name"],
            "source_manifest_sha256": (
                D25_OUTPUT_MANIFEST_SHA256 if witness else D01_OUTPUT_MANIFEST_SHA256
            ),
            "source_run_id": (
                "full_oracle_20260711_185500"
                if witness
                else "d01_full_reconcile_20260711_170200_a"
            ),
            "decision_effect": REGISTRY_DECISION_EFFECT,
            "final_pool": "false",
            "evidence_relpath": EVIDENCE_RELPATH,
        }
        row["row_id"] = registry_row_id(row)
        rows.append(validate_registry_row(row))
    rows.sort(key=lambda row: row["row_id"])
    expected = EXPECTED_WITNESSES + EXPECTED_PRESERVATION_EPISODES
    if len(rows) != expected or len({row["row_id"] for row in rows}) != expected:
        raise ValueError("warm-start action-choice registry row count drift")
    return rows


def _registry_live_state(
    path: Path,
    planned: list[dict[str, str]],
    before_sha256: str,
    after_sha256: str,
) -> str:
    actual = file_sha256(path)
    current = {row["row_id"]: row for row in _read_registry(path)}
    present = [current.get(row["row_id"]) for row in planned]
    if actual == before_sha256 and all(value is None for value in present):
        return "ready"
    if actual == after_sha256 and all(
        value == expected for value, expected in zip(present, planned)
    ):
        return "already_appended"
    raise ValueError("warm-start registry is neither planned-before nor planned-after")


def create_warmstart_manifest(
    repo_root: str | Path,
    source_preflight_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
) -> dict:
    root = Path(repo_root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("warm-start manifest must run from repo root")
    source = validate_source_preflight(source_preflight_dir, root)
    if not source["passed"]:
        raise ValueError(f"warm-start source preflight failed: {source}")
    inputs = validate_pinned_inputs(root)
    if not inputs["passed"]:
        raise ValueError(f"warm-start pinned input failure: {inputs}")
    _validate_frozen_sources(root)
    episodes, episode_counts = build_episode_manifest(root)
    examples, example_counts = build_macro_examples(episodes)
    schedule, diagnostic, schedule_counts = build_training_schedule(examples)
    gate_prior = build_gate_prior(examples, schedule, diagnostic)
    registry_rows = make_action_choice_registry_rows(episodes)
    registry = root / REGISTRY_RELPATH
    prior_manifest = root / PRIOR_WARMSTART_MANIFEST_RELPATH
    _validate_output_manifest(prior_manifest)
    before_snapshot = prior_manifest / "registry_before.snapshot.tsv"
    if file_sha256(before_snapshot) != EXPECTED_REGISTRY_BEFORE_SHA256:
        raise ValueError("warm-start immutable registry-before snapshot drift")
    live_state = _registry_live_state(
        registry,
        registry_rows,
        EXPECTED_REGISTRY_BEFORE_SHA256,
        EXPECTED_REGISTRY_AFTER_SHA256,
    )
    if live_state != "already_appended":
        raise ValueError("warm-start remediation must reuse already-open action rows")
    existing = _read_registry(registry)
    seal_audit = _validate_sealed_test_absence(root, episodes, existing)
    failed_predecessor = _validate_failed_predecessor(root)

    output = Path(output_dir)
    partial = _prepare_output(output)
    try:
        _write_tsv(partial / "episodes.tsv", episodes, EPISODE_FIELDS)
        _write_tsv(partial / "macro_examples.tsv", examples, EXAMPLE_FIELDS)
        np.save(partial / "training_schedule.npy", schedule)
        np.save(partial / "diagnostic_indices.npy", diagnostic)
        _write_tsv(partial / "registry_rows.tsv", registry_rows, REGISTRY_FIELDS)
        shutil.copyfile(before_snapshot, partial / "registry_before.snapshot.tsv")
        shutil.copyfile(before_snapshot, partial / "registry_after.expected.tsv")
        append_result = append_opened_registry(
            partial / "registry_after.expected.tsv", registry_rows
        )
        if (
            append_result.appended != len(registry_rows)
            or append_result.skipped != 0
            or append_result.total != 12688
        ):
            raise AssertionError("warm-start prospective registry accounting failed")
        after_sha = file_sha256(partial / "registry_after.expected.tsv")
        if after_sha != EXPECTED_REGISTRY_AFTER_SHA256:
            raise AssertionError("warm-start expected registry-after hash drift")
        config = {
            "schema": "bplus-v2.2-warmstart-remediation-manifest-config-2",
            "release_label": "TASK6_REMEDIATION_MANIFEST",
            "created_at": str(created_at),
            "owner_decision": OWNER_DECISION,
            "stage": REGISTRY_STAGE,
            "decision_effect": REGISTRY_DECISION_EFFECT,
            "registry_use_class": REGISTRY_USE_CLASS,
            "registry_opened_at": REGISTRY_OPENED_AT,
            "registry_before_sha256": EXPECTED_REGISTRY_BEFORE_SHA256,
            "registry_after_expected_sha256": after_sha,
            "registry_rows": len(registry_rows),
            "registry_live_state_at_creation": live_state,
            "registry_rows_newly_opened_by_remediation": 0,
            "registry_rows_reused_by_remediation": len(registry_rows),
            "episode_counts": episode_counts,
            "example_counts": example_counts,
            "schedule": schedule_counts,
            "gate_prior": gate_prior,
            "fresh_initial_brake_logit": INITIAL_BRAKE_LOGIT,
            "fresh_initial_brake_logit_hex": float(INITIAL_BRAKE_LOGIT).hex(),
            "failed_predecessor": failed_predecessor,
            "preservation_rule": {
                "eligibility": "D0.1 corrected BC confirmed_safe_pass within D2 non_test",
                "strata": ["map_name", "skill", "opponent_raceline", "l4_id"],
                "selection": "minimum domain-separated SHA256(l2_id) per stratum",
                "domain_hex": PRESERVATION_DOMAIN.hex(),
            },
            "warmstart_seed": WARMSTART_SEED,
            "policy_initialization_seed": POLICY_SEED,
            "updates": WARMSTART_UPDATES,
            "batch_size": WARMSTART_BATCH_SIZE,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "action_core_lr": ACTION_CORE_LR,
            "sidecar_finetune_lr": SIDECAR_FINETUNE_LR,
            "loss": "MSE(tanh steer normalized)+BCE brake gate+conditional MSE(sigmoid brake magnitude)",
            "early_stopping": False,
            "arm_selection_performed": False,
            "arms_share_schedule": list(ARMS),
            "actor_input_fields": list(ACTOR_INPUT_FIELDS),
            "privileged_manifest_fields": list(PRIVILEGED_MANIFEST_FIELDS),
            "forbidden_actor_fields": list(FORBIDDEN_ACTOR_FIELDS),
            "privileged_fields_enter_actor_tensor": False,
            "d25_output_manifest_sha256": D25_OUTPUT_MANIFEST_SHA256,
            "d01_output_manifest_sha256": D01_OUTPUT_MANIFEST_SHA256,
            "d2_dataset_manifest_sha256": D2_DATASET_MANIFEST_SHA256,
            "sidecar_output_manifest_sha256": SIDECAR_OUTPUT_MANIFEST_SHA256,
            "sidecar_state_dict_sha256": SIDECAR_STATE_DICT_SHA256,
            "test_seal_sha256": D2_TEST_SEAL_SHA256,
            "sealed_test_audit": seal_audit,
            "source_preflight_relpath": str(Path(source_preflight_dir)),
            "source_preflight_output_manifest_sha256": file_sha256(
                Path(source_preflight_dir) / "output_manifest.sha256"
            ),
            "test_opened": False,
            "final_pool": False,
        }
        _write_json(partial / "config.json", config)
        _write_json(
            partial / "validation.json",
            {
                "schema": "bplus-v2.2-warmstart-remediation-manifest-validation-2",
                "passed": True,
                "live_state": live_state,
                "violations": [],
            },
        )
        _write_output_manifest(partial)
        _promote(partial, output)
    except BaseException as error:
        if partial.exists():
            _write_json(
                partial / "FAILED.json",
                {"type": type(error).__name__, "message": str(error)},
            )
        raise
    validation = validate_warmstart_manifest(output, root, check_live_registry=True)
    if not validation["passed"]:
        raise AssertionError(f"created invalid warm-start manifest: {validation}")
    return {
        "passed": True,
        "episodes": validation["episodes"],
        "examples": validation["examples"],
        "registry_rows": validation["registry_rows"],
        "registry_after_expected_sha256": after_sha,
        "output_manifest_sha256": file_sha256(output / "output_manifest.sha256"),
    }


def validate_warmstart_manifest(
    release_dir: str | Path,
    repo_root: str | Path = ".",
    *,
    check_live_registry: bool = False,
) -> dict:
    release = Path(release_dir)
    root = Path(repo_root).resolve()
    violations = []
    details = {
        "episodes": 0,
        "examples": 0,
        "registry_rows": 0,
        "live_state": "not_checked",
    }
    try:
        if not (release / "COMPLETE").is_file():
            raise ValueError("warm-start manifest lacks COMPLETE")
        _validate_output_manifest(release)
        config = json.loads((release / "config.json").read_text(encoding="utf-8"))
        if (
            config["schema"]
            != "bplus-v2.2-warmstart-remediation-manifest-config-2"
            or config["release_label"] != "TASK6_REMEDIATION_MANIFEST"
            or config["owner_decision"] != OWNER_DECISION
            or config["stage"] != REGISTRY_STAGE
            or config["decision_effect"] != REGISTRY_DECISION_EFFECT
            or config["registry_use_class"] != REGISTRY_USE_CLASS
            or config["registry_before_sha256"] != EXPECTED_REGISTRY_BEFORE_SHA256
            or config["registry_after_expected_sha256"]
            != EXPECTED_REGISTRY_AFTER_SHA256
            or config["registry_live_state_at_creation"] != "already_appended"
            or config["registry_rows_newly_opened_by_remediation"] != 0
            or config["registry_rows_reused_by_remediation"] != 669
            or config["registry_opened_at"] != REGISTRY_OPENED_AT
            or config["updates"] != WARMSTART_UPDATES
            or config["batch_size"] != WARMSTART_BATCH_SIZE
            or config["gradient_clip_norm"] != GRAD_CLIP_NORM
            or config["action_core_lr"] != ACTION_CORE_LR
            or config["sidecar_finetune_lr"] != SIDECAR_FINETUNE_LR
            or config["warmstart_seed"] != WARMSTART_SEED
            or config["policy_initialization_seed"] != POLICY_SEED
            or config["arms_share_schedule"] != list(ARMS)
            or config["early_stopping"] is not False
            or config["arm_selection_performed"] is not False
            or config["privileged_fields_enter_actor_tensor"] is not False
            or config["test_opened"] is not False
            or config["final_pool"] is not False
            or config["d25_output_manifest_sha256"]
            != D25_OUTPUT_MANIFEST_SHA256
            or config["d01_output_manifest_sha256"] != D01_OUTPUT_MANIFEST_SHA256
            or config["d2_dataset_manifest_sha256"]
            != D2_DATASET_MANIFEST_SHA256
            or config["sidecar_output_manifest_sha256"]
            != SIDECAR_OUTPUT_MANIFEST_SHA256
            or config["sidecar_state_dict_sha256"] != SIDECAR_STATE_DICT_SHA256
            or config["test_seal_sha256"] != D2_TEST_SEAL_SHA256
            or config["fresh_initial_brake_logit"] != INITIAL_BRAKE_LOGIT
            or config["fresh_initial_brake_logit_hex"]
            != float(INITIAL_BRAKE_LOGIT).hex()
        ):
            raise ValueError("warm-start manifest authority/scope mismatch")
        if config["actor_input_fields"] != list(ACTOR_INPUT_FIELDS):
            raise ValueError("warm-start actor input field drift")
        if (
            config["privileged_manifest_fields"]
            != list(PRIVILEGED_MANIFEST_FIELDS)
            or config["forbidden_actor_fields"] != list(FORBIDDEN_ACTOR_FIELDS)
        ):
            raise ValueError("warm-start privileged-field declaration drift")
        if set(config["actor_input_fields"]) & set(FORBIDDEN_ACTOR_FIELDS):
            raise ValueError("warm-start privileged field enters actor input")
        expected_rule = {
            "eligibility": "D0.1 corrected BC confirmed_safe_pass within D2 non_test",
            "strata": ["map_name", "skill", "opponent_raceline", "l4_id"],
            "selection": "minimum domain-separated SHA256(l2_id) per stratum",
            "domain_hex": PRESERVATION_DOMAIN.hex(),
        }
        if config["preservation_rule"] != expected_rule:
            raise ValueError("warm-start preservation rule drift")
        source = root / config["source_preflight_relpath"]
        source_validation = validate_source_preflight(source, root)
        if not source_validation["passed"]:
            raise ValueError(
                f"warm-start source preflight is stale: {source_validation}"
            )
        if file_sha256(source / "output_manifest.sha256") != config[
            "source_preflight_output_manifest_sha256"
        ]:
            raise ValueError("warm-start source-preflight manifest mismatch")
        _validate_frozen_sources(root)
        episodes = _read_tsv(release / "episodes.tsv")
        examples = _read_tsv(release / "macro_examples.tsv")
        registry_rows = [
            validate_registry_row(row) for row in _read_tsv(release / "registry_rows.tsv")
        ]
        expected_episodes, episode_counts = build_episode_manifest(root)
        expected_examples, example_counts = build_macro_examples(expected_episodes)
        expected_schedule, expected_diagnostic, schedule_counts = build_training_schedule(
            expected_examples
        )
        expected_gate_prior = build_gate_prior(
            expected_examples, expected_schedule, expected_diagnostic
        )
        expected_registry = make_action_choice_registry_rows(expected_episodes)
        if episodes != expected_episodes or config["episode_counts"] != episode_counts:
            raise ValueError("warm-start episode manifest recomputation mismatch")
        if examples != expected_examples or config["example_counts"] != example_counts:
            raise ValueError("warm-start macro manifest recomputation mismatch")
        if registry_rows != expected_registry:
            raise ValueError("warm-start registry rows recomputation mismatch")
        schedule = np.load(release / "training_schedule.npy", allow_pickle=False)
        diagnostic = np.load(release / "diagnostic_indices.npy", allow_pickle=False)
        if not np.array_equal(schedule, expected_schedule) or not np.array_equal(
            diagnostic, expected_diagnostic
        ):
            raise ValueError("warm-start fixed schedule recomputation mismatch")
        if config["schedule"] != schedule_counts:
            raise ValueError("warm-start schedule accounting mismatch")
        if config["gate_prior"] != expected_gate_prior:
            raise ValueError("warm-start gate-prior recomputation mismatch")
        if config["failed_predecessor"] != _validate_failed_predecessor(root):
            raise ValueError("warm-start failed-predecessor record mismatch")
        if file_sha256(release / "registry_before.snapshot.tsv") != config[
            "registry_before_sha256"
        ]:
            raise ValueError("warm-start registry before snapshot mismatch")
        if file_sha256(release / "registry_after.expected.tsv") != config[
            "registry_after_expected_sha256"
        ]:
            raise ValueError("warm-start registry expected-after snapshot mismatch")
        before_rows = _read_registry(release / "registry_before.snapshot.tsv")
        after_rows = _read_registry(release / "registry_after.expected.tsv")
        if len(before_rows) != 12019 or len(after_rows) != 12688:
            raise ValueError("warm-start registry total accounting mismatch")
        if after_rows[-len(registry_rows):] != registry_rows:
            raise ValueError("warm-start registry append order mismatch")
        seal_audit = _validate_sealed_test_absence(root, episodes, after_rows)
        if seal_audit != config["sealed_test_audit"]:
            raise ValueError("warm-start sealed-test audit mismatch")
        if check_live_registry:
            details["live_state"] = _registry_live_state(
                root / REGISTRY_RELPATH,
                registry_rows,
                config["registry_before_sha256"],
                config["registry_after_expected_sha256"],
            )
        details.update(
            {
                "episodes": len(episodes),
                "examples": len(examples),
                "registry_rows": len(registry_rows),
            }
        )
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": "bplus-v2.2-warmstart-remediation-manifest-validation-2",
        "passed": not violations,
        **details,
        "violations": violations,
    }


@torch.no_grad()
def _framewise_bc_features(
    bc,
    lidar: np.ndarray,
    actual_speed: np.ndarray,
    initial_speed_input: float,
    device: torch.device,
) -> np.ndarray:
    lidar = np.asarray(lidar, dtype=np.float32)
    actual_speed = np.asarray(actual_speed, dtype=np.float32)
    if lidar.ndim != 2 or lidar.shape[1] != 360 or actual_speed.shape != (len(lidar),):
        raise ValueError("warm-start witness BC replay input shape mismatch")
    speed_inputs = build_speed_inputs(actual_speed, initial_speed_input)
    hidden = torch.zeros((1, 1, bc.gru.hidden_size), device=device)
    output = np.empty((len(lidar), bc.gru.hidden_size), dtype=np.float32)
    for frame in range(len(lidar)):
        lidar_t = torch.as_tensor(lidar[frame], device=device).view(1, 1, 360)
        speed_t = torch.tensor([[[speed_inputs[frame]]]], dtype=torch.float32, device=device)
        feature, hidden = bc.forward_features(lidar_t, speed_t, hidden)
        output[frame] = feature[0, 0].detach().cpu().numpy()
    if not np.all(np.isfinite(output)):
        raise ValueError("warm-start witness BC features are nonfinite")
    return output


class WarmstartBatchProvider:
    """Materialize deployable actor inputs while keeping labels separate."""

    def __init__(
        self,
        root: Path,
        manifest_dir: Path,
        device: torch.device,
    ):
        self.root = root
        self.manifest_dir = manifest_dir
        self.device = device
        self.episodes = _read_tsv(manifest_dir / "episodes.tsv")
        self.examples = _read_tsv(manifest_dir / "macro_examples.tsv")
        if [int(row["example_index"]) for row in self.examples] != list(range(len(self.examples))):
            raise ValueError("warm-start example order is not contiguous")
        self.episode_by_l2 = {row["l2_id"]: row for row in self.episodes}
        self.dataset = D2RDataset(
            root / DATASET_RELPATH,
            root / SPLIT_RELPATH,
            root / SIGNALS_RELPATH,
        )
        self.bc = load_frozen_bc(root / "pretrained/end2race.pth", device, hidden_scale=4)
        self.witness_inputs: dict[tuple[str, int], tuple[np.ndarray, ...]] = {}
        self._materialize_witnesses()

    def _initial_speed(self, episode: Mapping[str, str]) -> float:
        path = self.root / "f1tenth_racetracks" / episode["map_name"] / "raceline1.csv"
        rows = np.loadtxt(path, delimiter=";", skiprows=1, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] < 6:
            raise ValueError("warm-start raceline speed asset shape drift")
        index = int(episode["resolved_ego_idx"]) % len(rows)
        return float(rows[index, 5] * 0.9)

    def _materialize_witnesses(self) -> None:
        for episode in (row for row in self.episodes if row["role"] == "witness"):
            path = self.root / episode["source_npz_relpath"]
            if file_sha256(path) != episode["source_npz_sha256"]:
                raise ValueError(f"warm-start witness source hash drift: {episode['l2_id']}")
            with np.load(path, allow_pickle=False) as data:
                lidar = np.asarray(data["ego_lidar"], dtype=np.float32)
                actual_speed = np.asarray(data["ego_actual_speed"], dtype=np.float32)
                desired_steer = np.asarray(data["ego_desired_steer"], dtype=np.float32)
                desired_speed = np.asarray(data["ego_desired_speed"], dtype=np.float32)
            count = len(lidar)
            if any(value.shape != (count,) for value in (actual_speed, desired_steer, desired_speed)):
                raise ValueError("warm-start witness signal shape drift")
            features = _framewise_bc_features(
                self.bc,
                lidar,
                actual_speed,
                self._initial_speed(episode),
                self.device,
            )
            previous_steer = np.zeros(count, dtype=np.float32)
            previous_speed = np.zeros(count, dtype=np.float32)
            previous_steer[1:] = desired_steer[:-1]
            previous_speed[1:] = desired_speed[:-1]
            macro_frames = np.arange(0, count, MACRO_STEPS, dtype=np.int64)
            history = np.column_stack(
                [np.maximum(0, macro_frames - offset) for offset in HISTORY_OFFSETS]
            )
            lidar_history = np.clip(lidar[history], 0.0, 30.0) / np.float32(30.0)
            speed_history = actual_speed[history] / np.float32(10.0)
            steer_history = previous_steer[history] / np.float32(0.52)
            command_speed_history = previous_speed[history] / np.float32(10.0)
            for position, frame in enumerate(macro_frames.tolist()):
                self.witness_inputs[(episode["l2_id"], frame)] = (
                    features[frame],
                    lidar_history[position],
                    speed_history[position],
                    steer_history[position],
                    command_speed_history[position],
                )
        expected = sum(row["role"] != "preservation_noop" for row in self.examples)
        if len(self.witness_inputs) != expected:
            raise ValueError("warm-start witness input/example accounting mismatch")

    def get(self, example_indices) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        indices = np.asarray(example_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("warm-start batch indices must be a nonempty vector")
        if indices.min() < 0 or indices.max() >= len(self.examples):
            raise ValueError("warm-start batch example index out of range")
        batch = len(indices)
        bc_feature = np.empty((batch, 1680), dtype=np.float32)
        lidar_history = np.empty((batch, 8, 360), dtype=np.float32)
        actual_speed_history = np.empty((batch, 8), dtype=np.float32)
        previous_steer_history = np.empty((batch, 8), dtype=np.float32)
        previous_speed_history = np.empty((batch, 8), dtype=np.float32)
        target_steer = np.empty(batch, dtype=np.float32)
        target_brake = np.empty(batch, dtype=np.float32)
        target_gate = np.empty(batch, dtype=np.float32)
        preservation_positions = []
        preservation_frames = []
        for position, example_index in enumerate(indices.tolist()):
            example = self.examples[example_index]
            target_steer[position] = np.float32(float.fromhex(example["target_steer_hex"]))
            target_brake[position] = np.float32(float.fromhex(example["target_brake_hex"]))
            target_gate[position] = np.float32(int(example["target_brake_gate"]))
            if example["role"] == "preservation_noop":
                preservation_positions.append(position)
                preservation_frames.append(int(example["global_frame_index"]))
            else:
                values = self.witness_inputs[(example["l2_id"], int(example["frame_index"]))]
                (
                    bc_feature[position],
                    lidar_history[position],
                    actual_speed_history[position],
                    previous_steer_history[position],
                    previous_speed_history[position],
                ) = values
        if preservation_positions:
            lidar, bc, scalar = self.dataset.input_batch(
                np.asarray(preservation_frames, dtype=np.int64)
            )
            positions = np.asarray(preservation_positions, dtype=np.int64)
            bc_feature[positions] = bc
            lidar_history[positions] = lidar
            actual_speed_history[positions] = scalar[:, :8]
            previous_steer_history[positions] = scalar[:, 8:16]
            previous_speed_history[positions] = scalar[:, 16:24]
        actor = {
            "bc_feature": bc_feature,
            "lidar_history": lidar_history,
            "actual_speed_history": actual_speed_history,
            "previous_desired_steer_history": previous_steer_history,
            "previous_desired_speed_history": previous_speed_history,
        }
        labels = {
            "target_steer": target_steer,
            "target_brake": target_brake,
            "target_brake_gate": target_gate,
        }
        if tuple(actor) != ACTOR_INPUT_FIELDS:
            raise AssertionError("warm-start actor field boundary drift")
        if set(actor) & set(FORBIDDEN_ACTOR_FIELDS):
            raise AssertionError("warm-start privileged field entered actor batch")
        if any(not np.all(np.isfinite(value)) for value in (*actor.values(), *labels.values())):
            raise ValueError("warm-start batch contains nonfinite value")
        return actor, labels


def _torch_batch(
    actor: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    bc = torch.as_tensor(actor["bc_feature"], device=device)
    lidar = torch.as_tensor(actor["lidar_history"], device=device)
    scalar = torch.as_tensor(
        np.concatenate(
            [
                actor["actual_speed_history"],
                actor["previous_desired_steer_history"],
                actor["previous_desired_speed_history"],
            ],
            axis=1,
        ),
        device=device,
    )
    targets = {name: torch.as_tensor(value, device=device) for name, value in labels.items()}
    return bc, lidar, scalar, targets


def _warmstart_loss(policy: V22Policy, bc, lidar, scalar, targets) -> tuple[torch.Tensor, dict]:
    distribution = policy.distribution(bc, lidar, scalar)
    predicted_steer = torch.tanh(distribution.steer.mean.squeeze(-1))
    target_steer = targets["target_steer"] / STEER_BUDGET
    steer_loss = F.mse_loss(predicted_steer, target_steer)
    gate_logits = distribution.gate.logits.squeeze(-1)
    gate_target = targets["target_brake_gate"]
    gate_loss = F.binary_cross_entropy_with_logits(gate_logits, gate_target)
    predicted_brake = torch.sigmoid(distribution.brake.mean.squeeze(-1)) * BRAKE_BUDGET
    gate_sum = torch.clamp(gate_target.sum(), min=1.0)
    brake_loss = torch.sum(
        gate_target * (predicted_brake - targets["target_brake"]) ** 2
    ) / gate_sum
    total = steer_loss + gate_loss + brake_loss
    if not torch.isfinite(total):
        raise FloatingPointError("warm-start loss is nonfinite")
    return total, {
        "steer": steer_loss,
        "gate": gate_loss,
        "brake": brake_loss,
        "predicted_steer": predicted_steer * STEER_BUDGET,
        "gate_logits": gate_logits,
        "predicted_brake": predicted_brake,
    }


@torch.no_grad()
def _diagnostics(
    policy: V22Policy,
    provider: WarmstartBatchProvider,
    indices: np.ndarray,
    device: torch.device,
) -> dict:
    actor, labels = provider.get(indices)
    bc, lidar, scalar, targets = _torch_batch(actor, labels, device)
    total, values = _warmstart_loss(policy, bc, lidar, scalar, targets)
    gate_target = targets["target_brake_gate"]
    gate_choice = (values["gate_logits"] > 0.0).to(gate_target.dtype)
    positive = gate_target == 1.0
    negative = ~positive
    true_positive = int(torch.sum((gate_choice == 1.0) & positive).item())
    false_negative = int(torch.sum((gate_choice == 0.0) & positive).item())
    true_negative = int(torch.sum((gate_choice == 0.0) & negative).item())
    false_positive = int(torch.sum((gate_choice == 1.0) & negative).item())
    predicted_positive = true_positive + false_positive
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    return {
        "examples": len(indices),
        "loss": float(total.item()),
        "steer_loss": float(values["steer"].item()),
        "gate_loss": float(values["gate"].item()),
        "brake_loss": float(values["brake"].item()),
        "gate_accuracy": float((gate_choice == gate_target).float().mean().item()),
        "gate_precision": float(precision),
        "gate_true_positive": true_positive,
        "gate_false_positive": false_positive,
        "gate_true_negative": true_negative,
        "gate_false_negative": false_negative,
        "gate_predicted_positive": predicted_positive,
        "gate_recall": (
            float((gate_choice[positive] == 1.0).float().mean().item())
            if torch.any(positive)
            else None
        ),
        "gate_specificity": (
            float((gate_choice[negative] == 0.0).float().mean().item())
            if torch.any(negative)
            else None
        ),
        "steer_mae_rad": float(
            torch.mean(torch.abs(values["predicted_steer"] - targets["target_steer"])).item()
        ),
        "brake_mae_mps_on_gate": (
            float(
                torch.mean(
                    torch.abs(
                        values["predicted_brake"][positive]
                        - targets["target_brake"][positive]
                    )
                ).item()
            )
            if torch.any(positive)
            else None
        ),
    }


def _gate_acceptance(metrics: Mapping) -> dict:
    recall = float(metrics["gate_recall"])
    precision = float(metrics["gate_precision"])
    specificity = float(metrics["gate_specificity"])
    gate_loss = float(metrics["gate_loss"])
    finite = all(math.isfinite(value) for value in (recall, precision, specificity, gate_loss))
    checks = {
        "finite_metrics": finite,
        "gate_recall_gt_zero": finite and recall > 0.0,
        "gate_loss_lt_marginal_bce_bound": finite
        and gate_loss < GATE_BCE_ACCEPTANCE_MAX,
        "gate_specificity_gt_near_zero_bound": finite
        and specificity > GATE_SPECIFICITY_ACCEPTANCE_MIN,
    }
    return {
        "schema": "bplus-v2.2-warmstart-gate-acceptance-1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "gate_recall_strict_min": 0.0,
            "gate_loss_strict_max": GATE_BCE_ACCEPTANCE_MAX,
            "gate_specificity_strict_min": GATE_SPECIFICITY_ACCEPTANCE_MIN,
        },
        "observed": {
            "gate_recall": recall,
            "gate_loss": gate_loss,
            "gate_precision": precision,
            "gate_specificity": specificity,
            "gate_true_positive": int(metrics["gate_true_positive"]),
            "gate_false_positive": int(metrics["gate_false_positive"]),
            "gate_true_negative": int(metrics["gate_true_negative"]),
            "gate_false_negative": int(metrics["gate_false_negative"]),
        },
    }


def _named_state_digest(policy: V22Policy, prefixes: tuple[str, ...]) -> str:
    return _tensor_digest(
        (name, value)
        for name, value in policy.state_dict().items()
        if name.startswith(prefixes)
    )


def _policy_hashes(policy: V22Policy) -> dict[str, str]:
    return {
        "full_state_sha256": _tensor_digest(policy.state_dict().items()),
        "bc_sha256": _named_state_digest(policy, ("bc.",)),
        "policy_sidecar_sha256": _tensor_digest(
            policy.policy_sidecar.state_dict().items()
        ),
        "policy_sidecar_encoder_sha256": policy.policy_sidecar_encoder_sha256(),
        "shadow_sidecar_sha256": policy.shadow_sha256(),
        "action_state_sha256": _named_state_digest(
            policy,
            (
                "bc_adapter.",
                "action_core.",
                "steer_mean.",
                "brake_gate.",
                "brake_mean.",
                "log_steer_std",
                "log_brake_std",
            ),
        ),
    }


def _apply_warmstart_gate_prior(policy: V22Policy, gate_prior: Mapping) -> None:
    expected_bias = torch.full_like(policy.brake_gate.bias, INITIAL_BRAKE_LOGIT)
    if not torch.equal(policy.brake_gate.bias.detach(), expected_bias):
        raise ValueError("warm-start fresh brake-gate bias is not -6.0")
    if not torch.equal(
        policy.brake_gate.weight.detach(),
        torch.zeros_like(policy.brake_gate.weight),
    ):
        raise ValueError("warm-start fresh brake-gate weight is not zero")
    with torch.no_grad():
        policy.brake_gate.bias.fill_(float(gate_prior["applied_bias_float32"]))
    observed = float(policy.brake_gate.bias.detach().cpu().item())
    if observed != float(gate_prior["applied_bias_float32"]):
        raise AssertionError("warm-start empirical gate bias application mismatch")


def _checkpoint_payload(policy: V22Policy, arm: str, report: Mapping) -> dict:
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in policy.state_dict().items()
    }
    return {
        "schema": "bplus-v2.2-warmstart-remediation-checkpoint-2",
        "release_label": WARMSTART_RELEASE_LABEL,
        "arm": arm,
        "state_dict": state,
        "state_dict_sha256": _tensor_digest(state.items()),
        "report": dict(report),
    }


def _state_hashes(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    """Recompute component digests directly from a serialized policy state."""

    def named(prefixes: tuple[str, ...]) -> str:
        return _tensor_digest(
            (name, value) for name, value in state.items() if name.startswith(prefixes)
        )

    def stripped(prefix: str, nested_prefixes: tuple[str, ...] | None = None) -> str:
        items = []
        for name, value in state.items():
            if not name.startswith(prefix):
                continue
            nested = name[len(prefix):]
            if nested_prefixes is None or nested.startswith(nested_prefixes):
                items.append((nested, value))
        return _tensor_digest(items)

    encoder_prefixes = (
        "beam_encoder.",
        "bc_projection.",
        "scalar_projection.",
        "fusion.",
    )
    return {
        "full_state_sha256": _tensor_digest(state.items()),
        "bc_sha256": named(("bc.",)),
        "policy_sidecar_sha256": stripped("policy_sidecar."),
        "policy_sidecar_encoder_sha256": stripped(
            "policy_sidecar.", encoder_prefixes
        ),
        "shadow_sidecar_sha256": stripped("shadow_sidecar."),
        "action_state_sha256": named(
            (
                "bc_adapter.",
                "action_core.",
                "steer_mean.",
                "brake_gate.",
                "brake_mean.",
                "log_steer_std",
                "log_brake_std",
            )
        ),
    }


def _finite_metric_tree(value, path: str = "metric") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_metric_tree(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_metric_tree(child, f"{path}[{index}]")
        return
    if isinstance(value, (int, np.integer)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"warm-start nonfinite {path}")
        return
    raise TypeError(f"warm-start unsupported metric value at {path}: {type(value)}")


def _assert_training_invariants(reports: Mapping[str, Mapping]) -> None:
    if set(reports) != set(ARMS):
        raise ValueError("warm-start report arm inventory mismatch")
    fresh = {arm: reports[arm]["fresh_initial_hashes"] for arm in ARMS}
    initial = {arm: reports[arm]["initial_hashes"] for arm in ARMS}
    final = {arm: reports[arm]["final_hashes"] for arm in ARMS}
    if len({initial[arm]["bc_sha256"] for arm in ARMS}) != 1:
        raise ValueError("warm-start arms do not share one BC initialization")
    if len({initial[arm]["shadow_sidecar_sha256"] for arm in ARMS}) != 1:
        raise ValueError("warm-start arms do not share one shadow sidecar")
    if len({initial[arm]["policy_sidecar_sha256"] for arm in ARMS}) != 1:
        raise ValueError("warm-start arms do not share one policy sidecar initialization")
    if any(
        initial[arm]["policy_sidecar_sha256"] != SIDECAR_STATE_DICT_SHA256
        or initial[arm]["shadow_sidecar_sha256"] != SIDECAR_STATE_DICT_SHA256
        for arm in ARMS
    ):
        raise ValueError("warm-start arm sidecar is not the pinned initialization")
    for arm in ARMS:
        for name in (
            "bc_sha256",
            "policy_sidecar_sha256",
            "policy_sidecar_encoder_sha256",
            "shadow_sidecar_sha256",
        ):
            if fresh[arm][name] != initial[arm][name]:
                raise ValueError(
                    f"warm-start empirical prior changed non-gate state: {arm}/{name}"
                )
        if fresh[arm]["action_state_sha256"] == initial[arm]["action_state_sha256"]:
            raise ValueError(f"warm-start empirical gate prior was not applied: {arm}")
        if final[arm]["bc_sha256"] != initial[arm]["bc_sha256"]:
            raise ValueError(f"warm-start frozen BC mutated: {arm}")
        if final[arm]["shadow_sidecar_sha256"] != initial[arm]["shadow_sidecar_sha256"]:
            raise ValueError(f"warm-start shadow diagnostic mutated: {arm}")
        if final[arm]["action_state_sha256"] == initial[arm]["action_state_sha256"]:
            raise ValueError(f"warm-start common action parameters did not update: {arm}")
    for arm in ARMS[:2]:
        if final[arm]["policy_sidecar_sha256"] != initial[arm]["policy_sidecar_sha256"]:
            raise ValueError(f"warm-start frozen policy sidecar mutated: {arm}")
    arm_c = ARMS[2]
    if (
        final[arm_c]["policy_sidecar_encoder_sha256"]
        == initial[arm_c]["policy_sidecar_encoder_sha256"]
    ):
        raise ValueError("warm-start C sidecar encoder did not update")
    if final[arm_c]["policy_sidecar_sha256"] == initial[arm_c]["policy_sidecar_sha256"]:
        raise ValueError("warm-start C policy sidecar did not update")


def _set_deterministic_cuda(device: torch.device) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError(
            "warm-start CUDA requires process-level "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("warm-start numerical work requires available CUDA")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(device)


def run_warmstart_smoke(
    repo_root: str | Path,
    source_preflight_dir: str | Path,
    manifest_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
    device_name: str = "cuda:0",
) -> dict:
    """Run the fixed supervised smoke for all arms; never select or start PPO."""

    root = Path(repo_root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("warm-start smoke must run from repo root")
    source = validate_source_preflight(source_preflight_dir, root)
    if not source["passed"]:
        raise ValueError(f"warm-start source preflight failed: {source}")
    manifest_validation = validate_warmstart_manifest(
        manifest_dir, root, check_live_registry=True
    )
    if not manifest_validation["passed"]:
        raise ValueError(f"warm-start manifest failed: {manifest_validation}")
    if manifest_validation["live_state"] not in {"ready", "already_appended"}:
        raise ValueError("warm-start manifest registry is not executable")
    inputs = validate_pinned_inputs(root)
    if not inputs["passed"]:
        raise ValueError(f"warm-start pinned inputs failed: {inputs}")
    _validate_frozen_sources(root)
    if not os.environ.get("NUMBA_CACHE_DIR") or not Path(
        os.environ["NUMBA_CACHE_DIR"]
    ).is_absolute():
        raise ValueError("warm-start smoke requires isolated absolute NUMBA_CACHE_DIR")
    device = torch.device(device_name)
    _set_deterministic_cuda(device)

    output = Path(output_dir)
    partial = _prepare_output(output)
    reports: dict[str, dict] = {}
    try:
        manifest = Path(manifest_dir)
        manifest_config = json.loads(
            (manifest / "config.json").read_text(encoding="utf-8")
        )
        gate_prior = manifest_config["gate_prior"]
        planned_rows = [
            validate_registry_row(row)
            for row in _read_tsv(manifest / "registry_rows.tsv")
        ]
        registry = root / REGISTRY_RELPATH
        registry_before_observed = file_sha256(registry)
        if registry_before_observed != EXPECTED_REGISTRY_AFTER_SHA256:
            raise ValueError("warm-start remediation requires existing action-choice rows")
        append_result = append_opened_registry(registry, planned_rows)
        registry_after = file_sha256(registry)
        if registry_after != manifest_config["registry_after_expected_sha256"]:
            raise AssertionError("warm-start live registry did not reach planned state")
        if (append_result.appended, append_result.skipped) != (
            0,
            len(planned_rows),
        ):
            raise AssertionError("warm-start registry append/skip accounting mismatch")
        if append_result.total != 12688:
            raise AssertionError("warm-start registry total-row accounting mismatch")
        shutil.copyfile(registry, partial / "opened_registry.snapshot.tsv")

        schedule = np.load(manifest / "training_schedule.npy", allow_pickle=False)
        diagnostic_indices = np.load(
            manifest / "diagnostic_indices.npy", allow_pickle=False
        )
        if schedule.shape != (WARMSTART_UPDATES, WARMSTART_BATCH_SIZE):
            raise ValueError("warm-start smoke schedule shape drift")
        provider = WarmstartBatchProvider(root, manifest, device)
        sidecar_state, sidecar_mean, sidecar_std, _ = load_sidecar_bundle(
            root / SIDECAR_RELEASE_RELPATH
        )
        bc_state = {
            name: value.detach().cpu().contiguous()
            for name, value in provider.bc.state_dict().items()
        }
        (partial / "curves").mkdir()
        (partial / "checkpoints").mkdir()
        (partial / "reports").mkdir()

        fresh_policy_hashes = {}
        initial_policy_hashes = {}
        for arm in ARMS:
            gc.collect()
            torch.cuda.empty_cache()
            policy = V22Policy(
                arm,
                bc_state_dict=bc_state,
                sidecar_state_dict=sidecar_state,
                sidecar_bc_mean=sidecar_mean,
                sidecar_bc_std=sidecar_std,
                initialization_seed=POLICY_SEED,
            ).to(device)
            policy.eval()
            fresh_hashes = _policy_hashes(policy)
            fresh_policy_hashes[arm] = fresh_hashes
            _apply_warmstart_gate_prior(policy, gate_prior)
            initial_hashes = _policy_hashes(policy)
            initial_policy_hashes[arm] = initial_hashes
            diagnostic_before = _diagnostics(
                policy, provider, diagnostic_indices, device
            )
            frozen_snapshot = policy.frozen_snapshot()
            optimizer = torch.optim.AdamW(
                policy.optimizer_parameter_groups(), weight_decay=0.0
            )
            curves = {
                "total_loss": np.empty(WARMSTART_UPDATES, dtype=np.float64),
                "steer_loss": np.empty(WARMSTART_UPDATES, dtype=np.float64),
                "gate_loss": np.empty(WARMSTART_UPDATES, dtype=np.float64),
                "brake_loss": np.empty(WARMSTART_UPDATES, dtype=np.float64),
                "gradient_norm": np.empty(WARMSTART_UPDATES, dtype=np.float64),
            }
            policy.train()
            for update in range(WARMSTART_UPDATES):
                actor, labels = provider.get(schedule[update])
                bc, lidar, scalar, targets = _torch_batch(
                    actor, labels, device
                )
                optimizer.zero_grad(set_to_none=True)
                loss, values = _warmstart_loss(
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
                curves["steer_loss"][update] = float(
                    values["steer"].detach().item()
                )
                curves["gate_loss"][update] = float(
                    values["gate"].detach().item()
                )
                curves["brake_loss"][update] = float(
                    values["brake"].detach().item()
                )
                curves["gradient_norm"][update] = float(
                    gradient_norm.detach().item()
                )
            if any(not np.all(np.isfinite(values)) for values in curves.values()):
                raise FloatingPointError(f"warm-start curve is nonfinite: {arm}")
            policy.assert_frozen_unchanged(frozen_snapshot)
            policy.eval()
            diagnostic_after = _diagnostics(
                policy, provider, diagnostic_indices, device
            )
            gate_acceptance = _gate_acceptance(diagnostic_after)
            final_hashes = _policy_hashes(policy)
            report = {
                "schema": "bplus-v2.2-warmstart-remediation-arm-report-2",
                "release_label": WARMSTART_RELEASE_LABEL,
                "arm": arm,
                "updates_completed": WARMSTART_UPDATES,
                "batch_size": WARMSTART_BATCH_SIZE,
                "early_stopping": False,
                "arm_selection_performed": False,
                "ppo_training_started": False,
                "fresh_initial_brake_logit": INITIAL_BRAKE_LOGIT,
                "gate_prior": gate_prior,
                "schedule_sha256": file_sha256(
                    manifest / "training_schedule.npy"
                ),
                "diagnostic_indices_sha256": file_sha256(
                    manifest / "diagnostic_indices.npy"
                ),
                "fresh_initial_hashes": fresh_hashes,
                "initial_hashes": initial_hashes,
                "final_hashes": final_hashes,
                "diagnostic_before": diagnostic_before,
                "diagnostic_after": diagnostic_after,
                "gate_acceptance": gate_acceptance,
                "curve_final": {
                    name: float(values[-1]) for name, values in curves.items()
                },
                "curve_min_total_loss": float(curves["total_loss"].min()),
            }
            _finite_metric_tree(report)
            reports[arm] = report
            np.savez(partial / "curves" / f"{arm}.npz", **curves)
            _write_json(partial / "reports" / f"{arm}.json", report)
            torch.save(
                _checkpoint_payload(policy, arm, report),
                partial / "checkpoints" / f"{arm}.pt",
            )
            del optimizer, policy
            gc.collect()
            torch.cuda.empty_cache()

        if (
            fresh_policy_hashes[ARMS[1]]["full_state_sha256"]
            != fresh_policy_hashes[ARMS[2]]["full_state_sha256"]
        ):
            raise AssertionError("warm-start fresh B/C policy states differ")
        if (
            initial_policy_hashes[ARMS[1]]["full_state_sha256"]
            != initial_policy_hashes[ARMS[2]]["full_state_sha256"]
        ):
            raise AssertionError("warm-start B/C initial policy states differ")
        _assert_training_invariants(reports)
        task6_acceptance = {
            arm: reports[arm]["gate_acceptance"] for arm in ARMS
        }
        task6_acceptance_passed = all(
            result["passed"] for result in task6_acceptance.values()
        )
        seal_audit = _validate_sealed_test_absence(
            root,
            _read_tsv(manifest / "episodes.tsv"),
            _read_registry(registry),
        )
        config = {
            "schema": "bplus-v2.2-warmstart-remediation-config-2",
            "release_label": WARMSTART_RELEASE_LABEL,
            "created_at": str(created_at),
            "owner_decision": OWNER_DECISION,
            "supervised_warmstart_started": True,
            "supervised_warmstart_completed": True,
            "ppo_training_started": False,
            "closed_loop_evaluation_started": False,
            "arm_selection_performed": False,
            "candidate_promoted": False,
            "losses_are_diagnostic_only": True,
            "task6_acceptance_passed": task6_acceptance_passed,
            "ppo_checkpoint_eligible": task6_acceptance_passed,
            "task6_acceptance": task6_acceptance,
            "gate_prior": gate_prior,
            "failed_predecessor": manifest_config["failed_predecessor"],
            "arms": list(ARMS),
            "updates_per_arm": WARMSTART_UPDATES,
            "batch_size": WARMSTART_BATCH_SIZE,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "action_core_lr": ACTION_CORE_LR,
            "sidecar_finetune_lr": SIDECAR_FINETUNE_LR,
            "manifest_relpath": str(Path(manifest_dir)),
            "manifest_output_sha256": file_sha256(
                manifest / "output_manifest.sha256"
            ),
            "source_preflight_relpath": str(Path(source_preflight_dir)),
            "source_preflight_output_manifest_sha256": file_sha256(
                Path(source_preflight_dir) / "output_manifest.sha256"
            ),
            "registry_before_observed_sha256": registry_before_observed,
            "registry_after_sha256": registry_after,
            "registry_rows_appended": append_result.appended,
            "registry_rows_already_present": append_result.skipped,
            "registry_total_rows": append_result.total,
            "sealed_test_audit": seal_audit,
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
            "reports": {
                arm: {
                    "checkpoint_sha256": file_sha256(
                        partial / "checkpoints" / f"{arm}.pt"
                    ),
                    "curve_sha256": file_sha256(
                        partial / "curves" / f"{arm}.npz"
                    ),
                    "report_sha256": file_sha256(
                        partial / "reports" / f"{arm}.json"
                    ),
                    "initial_state_sha256": reports[arm]["initial_hashes"][
                        "full_state_sha256"
                    ],
                    "fresh_initial_state_sha256": reports[arm][
                        "fresh_initial_hashes"
                    ]["full_state_sha256"],
                    "final_state_sha256": reports[arm]["final_hashes"][
                        "full_state_sha256"
                    ],
                }
                for arm in ARMS
            },
        }
        _write_json(partial / "config.json", config)
        _write_json(
            partial / "validation.json",
            {
                "schema": "bplus-v2.2-warmstart-remediation-validation-2",
                "passed": True,
                "integrity_passed": True,
                "task6_acceptance_passed": task6_acceptance_passed,
                "mode": "pending_same_device_full",
                "arms": len(ARMS),
                "updates_per_arm": WARMSTART_UPDATES,
                "violations": [],
            },
        )
        _write_output_manifest(partial)
        preliminary = validate_warmstart_release(
            partial,
            root,
            device_name=device_name,
            require_live_registry=True,
            allow_partial=True,
        )
        if not preliminary["integrity_passed"]:
            raise AssertionError(
                f"warm-start preliminary validation failed: {preliminary}"
            )
        _write_json(partial / "validation.json", preliminary)
        _write_output_manifest(partial)
        artifact_check = validate_warmstart_release(
            partial,
            root,
            require_live_registry=True,
            allow_partial=True,
        )
        if not artifact_check["integrity_passed"]:
            raise AssertionError(
                f"warm-start final artifact validation failed: {artifact_check}"
            )
        _promote(partial, output)
        del provider
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as error:
        if partial.exists():
            _write_json(
                partial / "FAILED.json",
                {"type": type(error).__name__, "message": str(error)},
            )
        raise
    validation = validate_warmstart_release(output, root)
    if not validation["integrity_passed"]:
        raise AssertionError(f"created invalid warm-start artifact: {validation}")
    return {
        "passed": validation["task6_acceptance_passed"],
        "integrity_passed": validation["integrity_passed"],
        "task6_acceptance_passed": validation["task6_acceptance_passed"],
        "release_label": WARMSTART_RELEASE_LABEL,
        "arms": validation["arms"],
        "updates_per_arm": validation["updates_per_arm"],
        "registry_after_sha256": registry_after,
        "output_manifest_sha256": file_sha256(output / "output_manifest.sha256"),
        "ppo_training_started": False,
        "arm_selection_performed": False,
    }


def validate_warmstart_release(
    release_dir: str | Path,
    repo_root: str | Path = ".",
    *,
    device_name: str | None = None,
    require_live_registry: bool = False,
    allow_partial: bool = False,
) -> dict:
    """Artifact-only by default; optional numerical recomputation requires CUDA."""

    release = Path(release_dir)
    root = Path(repo_root).resolve()
    violations = []
    details = {
        "mode": "artifact_only",
        "arms": 0,
        "updates_per_arm": 0,
        "registry_rows": 0,
        "task6_acceptance_passed": False,
    }
    try:
        if not allow_partial and not (release / "COMPLETE").is_file():
            raise ValueError("warm-start smoke release lacks COMPLETE")
        _validate_output_manifest(release)
        config = json.loads((release / "config.json").read_text(encoding="utf-8"))
        if (
            config["schema"] != "bplus-v2.2-warmstart-remediation-config-2"
            or config["release_label"] != WARMSTART_RELEASE_LABEL
            or config["owner_decision"] != OWNER_DECISION
            or config["supervised_warmstart_completed"] is not True
            or config["ppo_training_started"] is not False
            or config["closed_loop_evaluation_started"] is not False
            or config["arm_selection_performed"] is not False
            or config["candidate_promoted"] is not False
            or config["losses_are_diagnostic_only"] is not True
            or not isinstance(config["task6_acceptance_passed"], bool)
            or config["ppo_checkpoint_eligible"]
            is not config["task6_acceptance_passed"]
            or config["arms"] != list(ARMS)
            or config["updates_per_arm"] != WARMSTART_UPDATES
            or config["batch_size"] != WARMSTART_BATCH_SIZE
            or config["gradient_clip_norm"] != GRAD_CLIP_NORM
            or config["action_core_lr"] != ACTION_CORE_LR
            or config["sidecar_finetune_lr"] != SIDECAR_FINETUNE_LR
            or config["test_opened"] is not False
            or config["final_pool"] is not False
            or config["cublas_workspace_config"] not in {":4096:8", ":16:8"}
            or config["deterministic_algorithms"] is not True
        ):
            raise ValueError("warm-start smoke authority/scope mismatch")
        manifest = root / config["manifest_relpath"]
        manifest_validation = validate_warmstart_manifest(
            manifest, root, check_live_registry=require_live_registry
        )
        if not manifest_validation["passed"]:
            raise ValueError(
                f"warm-start referenced manifest invalid: {manifest_validation}"
            )
        if file_sha256(manifest / "output_manifest.sha256") != config[
            "manifest_output_sha256"
        ]:
            raise ValueError("warm-start manifest output hash mismatch")
        source = root / config["source_preflight_relpath"]
        if file_sha256(source / "output_manifest.sha256") != config[
            "source_preflight_output_manifest_sha256"
        ]:
            raise ValueError("warm-start source-preflight hash mismatch")
        manifest_config = json.loads(
            (manifest / "config.json").read_text(encoding="utf-8")
        )
        if (
            config["gate_prior"] != manifest_config["gate_prior"]
            or config["failed_predecessor"]
            != manifest_config["failed_predecessor"]
        ):
            raise ValueError("warm-start remediation provenance mismatch")
        expected_registry = manifest_config["registry_after_expected_sha256"]
        if (
            config["registry_after_sha256"] != expected_registry
            or file_sha256(release / "opened_registry.snapshot.tsv")
            != expected_registry
        ):
            raise ValueError("warm-start opened-registry snapshot mismatch")
        if config["registry_before_observed_sha256"] not in {
            manifest_config["registry_before_sha256"],
            expected_registry,
        }:
            raise ValueError("warm-start registry pre-append state mismatch")
        if (
            config["registry_rows_appended"],
            config["registry_rows_already_present"],
        ) != (0, 669):
            raise ValueError("warm-start registry append accounting mismatch")
        snapshot_rows = _read_registry(release / "opened_registry.snapshot.tsv")
        action_rows = [
            row
            for row in snapshot_rows
            if row["stage"] == REGISTRY_STAGE
            and row["decision_effect"] == REGISTRY_DECISION_EFFECT
        ]
        if len(snapshot_rows) != 12688 or len(action_rows) != 669:
            raise ValueError("warm-start registry row accounting mismatch")
        if any(
            row["use_class"] != REGISTRY_USE_CLASS or row["final_pool"] != "false"
            for row in action_rows
        ):
            raise ValueError("warm-start registry action-choice semantics mismatch")
        if require_live_registry and file_sha256(root / REGISTRY_RELPATH) != expected_registry:
            raise ValueError("warm-start live registry hash mismatch")

        if set(config["reports"]) != set(ARMS):
            raise ValueError("warm-start config report inventory mismatch")
        reports = {}
        for arm in ARMS:
            report_path = release / "reports" / f"{arm}.json"
            curve_path = release / "curves" / f"{arm}.npz"
            checkpoint_path = release / "checkpoints" / f"{arm}.pt"
            record = config["reports"][arm]
            if (
                file_sha256(report_path) != record["report_sha256"]
                or file_sha256(curve_path) != record["curve_sha256"]
                or file_sha256(checkpoint_path) != record["checkpoint_sha256"]
            ):
                raise ValueError(f"warm-start arm artifact hash mismatch: {arm}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            _finite_metric_tree(report)
            if (
                report["arm"] != arm
                or report["schema"]
                != "bplus-v2.2-warmstart-remediation-arm-report-2"
                or report["release_label"] != WARMSTART_RELEASE_LABEL
                or report["updates_completed"] != WARMSTART_UPDATES
                or report["batch_size"] != WARMSTART_BATCH_SIZE
                or report["early_stopping"] is not False
                or report["arm_selection_performed"] is not False
                or report["ppo_training_started"] is not False
                or report["fresh_initial_brake_logit"] != INITIAL_BRAKE_LOGIT
                or report["gate_prior"] != manifest_config["gate_prior"]
                or report["schedule_sha256"]
                != file_sha256(manifest / "training_schedule.npy")
                or report["diagnostic_indices_sha256"]
                != file_sha256(manifest / "diagnostic_indices.npy")
            ):
                raise ValueError(f"warm-start arm report scope mismatch: {arm}")
            with np.load(curve_path, allow_pickle=False) as curve_file:
                if set(curve_file.files) != {
                    "total_loss",
                    "steer_loss",
                    "gate_loss",
                    "brake_loss",
                    "gradient_norm",
                }:
                    raise ValueError(f"warm-start curve inventory mismatch: {arm}")
                for name in curve_file.files:
                    values = curve_file[name]
                    if values.shape != (WARMSTART_UPDATES,) or values.dtype != np.float64:
                        raise ValueError(f"warm-start curve shape/dtype mismatch: {arm}/{name}")
                    if not np.all(np.isfinite(values)):
                        raise ValueError(f"warm-start curve nonfinite: {arm}/{name}")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if (
                checkpoint.get("schema")
                != "bplus-v2.2-warmstart-remediation-checkpoint-2"
                or checkpoint.get("release_label") != WARMSTART_RELEASE_LABEL
                or checkpoint.get("arm") != arm
                or checkpoint.get("report") != report
            ):
                raise ValueError(f"warm-start checkpoint envelope mismatch: {arm}")
            state = checkpoint.get("state_dict", {})
            if not isinstance(state, Mapping) or not state:
                raise ValueError(f"warm-start checkpoint state missing: {arm}")
            reference = V22Policy(arm).state_dict()
            if set(state) != set(reference):
                raise ValueError(f"warm-start checkpoint state inventory mismatch: {arm}")
            for name, expected in reference.items():
                value = torch.as_tensor(state[name])
                if value.shape != expected.shape or value.dtype != expected.dtype:
                    raise ValueError(
                        f"warm-start checkpoint tensor shape/dtype mismatch: {arm}/{name}"
                    )
                if not torch.all(torch.isfinite(value)):
                    raise ValueError(
                        f"warm-start checkpoint tensor nonfinite: {arm}/{name}"
                    )
            observed_hashes = _state_hashes(state)
            if (
                observed_hashes != report["final_hashes"]
                or observed_hashes["full_state_sha256"]
                != checkpoint["state_dict_sha256"]
                or observed_hashes["full_state_sha256"]
                != record["final_state_sha256"]
                or report["initial_hashes"]["full_state_sha256"]
                != record["initial_state_sha256"]
                or report["fresh_initial_hashes"]["full_state_sha256"]
                != record["fresh_initial_state_sha256"]
            ):
                raise ValueError(f"warm-start checkpoint digest mismatch: {arm}")
            reports[arm] = report
        _assert_training_invariants(reports)
        observed_acceptance = {
            arm: _gate_acceptance(reports[arm]["diagnostic_after"])
            for arm in ARMS
        }
        if any(
            observed_acceptance[arm] != reports[arm]["gate_acceptance"]
            for arm in ARMS
        ):
            raise ValueError("warm-start arm acceptance recomputation mismatch")
        observed_task6_pass = all(
            result["passed"] for result in observed_acceptance.values()
        )
        if (
            config["task6_acceptance"] != observed_acceptance
            or config["task6_acceptance_passed"] is not observed_task6_pass
            or config["ppo_checkpoint_eligible"] is not observed_task6_pass
        ):
            raise ValueError("warm-start aggregate acceptance mismatch")
        if (
            reports[ARMS[1]]["initial_hashes"]["full_state_sha256"]
            != reports[ARMS[2]]["initial_hashes"]["full_state_sha256"]
        ):
            raise ValueError("warm-start serialized B/C initialization mismatch")
        _validate_frozen_sources(root)
        observed_seal = _validate_sealed_test_absence(
            root,
            _read_tsv(manifest / "episodes.tsv"),
            snapshot_rows,
        )
        if observed_seal != config["sealed_test_audit"]:
            raise ValueError("warm-start sealed-test audit mismatch")

        if device_name is not None:
            device = torch.device(device_name)
            _set_deterministic_cuda(device)
            provider = WarmstartBatchProvider(root, manifest, device)
            indices = np.load(
                manifest / "diagnostic_indices.npy", allow_pickle=False
            )
            sidecar_state, sidecar_mean, sidecar_std, _ = load_sidecar_bundle(
                root / SIDECAR_RELEASE_RELPATH
            )
            bc_state = {
                name: value.detach().cpu().contiguous()
                for name, value in provider.bc.state_dict().items()
            }
            for arm in ARMS:
                initial_policy = V22Policy(
                    arm,
                    bc_state_dict=bc_state,
                    sidecar_state_dict=sidecar_state,
                    sidecar_bc_mean=sidecar_mean,
                    sidecar_bc_std=sidecar_std,
                    initialization_seed=POLICY_SEED,
                ).to(device).eval()
                fresh_hashes = _policy_hashes(initial_policy)
                if fresh_hashes != reports[arm]["fresh_initial_hashes"]:
                    raise ValueError(
                        f"warm-start same-device fresh initialization mismatch: {arm}"
                    )
                _apply_warmstart_gate_prior(
                    initial_policy, manifest_config["gate_prior"]
                )
                if _policy_hashes(initial_policy) != reports[arm]["initial_hashes"]:
                    raise ValueError(
                        f"warm-start same-device prior initialization mismatch: {arm}"
                    )
                before = _diagnostics(initial_policy, provider, indices, device)
                if before != reports[arm]["diagnostic_before"]:
                    raise ValueError(
                        f"warm-start same-device initial diagnostic mismatch: {arm}"
                    )
                del initial_policy
                checkpoint = torch.load(
                    release / "checkpoints" / f"{arm}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                final_policy = V22Policy(
                    arm,
                    bc_state_dict=bc_state,
                    sidecar_state_dict=sidecar_state,
                    sidecar_bc_mean=sidecar_mean,
                    sidecar_bc_std=sidecar_std,
                    initialization_seed=POLICY_SEED,
                ).to(device)
                final_policy.load_state_dict(checkpoint["state_dict"])
                final_policy.eval()
                after = _diagnostics(final_policy, provider, indices, device)
                if after != reports[arm]["diagnostic_after"]:
                    raise ValueError(
                        f"warm-start same-device final diagnostic mismatch: {arm}"
                    )
                del final_policy
                gc.collect()
                torch.cuda.empty_cache()
            del provider
            gc.collect()
            torch.cuda.empty_cache()
            details["mode"] = "same_device_full"
        details.update(
            {
                "arms": len(ARMS),
                "updates_per_arm": WARMSTART_UPDATES,
                "registry_rows": len(action_rows),
                "task6_acceptance_passed": observed_task6_pass,
            }
        )
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    integrity_passed = not violations
    task6_passed = bool(details["task6_acceptance_passed"]) if integrity_passed else False
    return {
        "schema": "bplus-v2.2-warmstart-remediation-validation-2",
        "passed": integrity_passed and task6_passed,
        "integrity_passed": integrity_passed,
        "task6_acceptance_passed": task6_passed,
        **details,
        "violations": violations,
    }

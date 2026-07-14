#!/usr/bin/env python3
"""Prepare, execute, and summarize the B6 no-learning temporal-noise audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from bplus_v22.b4_direct import B4ScenarioSets
from bplus_v22.b4_env import run_b4_episode
from bplus_v22.b6_temporal import (
    B6_EXPECTED_EPISODES,
    B6_INNOVATION_SEEDS,
    B6_MATCHED_L4_COUNT,
    B6_MODES,
    B6_OUTCOMES,
    B6_PHASE0_SCHEMA,
    B6_RHO,
    B6Phase0Policy,
    ar1_conditional_log_prob,
    arm_order,
    exact_cluster_signflip_one_sided,
    factorized_log_prob,
    paired_cluster_bootstrap,
    select_matched_scenarios,
    selection_digest,
    trace_moments,
)
from bplus_v22.ppo_env import load_b2_scenario_sets
from d25.search import trajectory_digest


DEFAULT_TASK8 = Path(
    "Experiments/B1_route_r2_scaffold/artifacts/"
    "task8_manifests_20260712_113241"
)
DEFAULT_METADATA = Path(
    "Experiments/A3_d2_representation/artifacts/"
    "non_test_full_20260711_175713/episode_metadata.tsv"
)
DEFAULT_BC = Path("pretrained/end2race.pth")
PLAN_SCHEMA = "end2race-b6-temporal-phase0-run-plan-1"
RESULT_SCHEMA = "end2race-b6-temporal-phase0-episode-1"
SUMMARY_SCHEMA = "end2race-b6-temporal-phase0-summary-1"
BOOTSTRAP_SAMPLES = 200_000

SELECTION_FIELDS = (
    "matched_order",
    "archived_outcome",
    "training_order",
    "l2_id",
    "l4_id",
    "map_name",
    "skill",
    "opponent_raceline",
    "speedscale_hex",
    "resolved_ego_idx",
)

PAIR_FIELDS = (
    "matched_order",
    "archived_outcome",
    "innovation_seed",
    "l2_id",
    "l4_id",
    "map_name",
    "iid_outcome",
    "ar1_outcome",
    "iid_collision",
    "ar1_collision",
    "iid_repaired_collision",
    "ar1_repaired_collision",
    "repair_delta",
    "iid_safe_to_collision",
    "ar1_safe_to_collision",
    "safe_harm_delta",
    "iid_lost_overtake",
    "ar1_lost_overtake",
    "overtake_loss_delta",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "NA" if row.get(field, "") in (None, "") else row[field]
                    for field in fields
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def scenario_sets(task8: Path, metadata: Path) -> B4ScenarioSets:
    return B4ScenarioSets.from_b2(load_b2_scenario_sets(task8, metadata))


def selected_rows(task8: Path, metadata: Path):
    sets = scenario_sets(task8, metadata)
    return select_matched_scenarios(
        {
            "collision": sets.collision,
            "overtake": sets.overtake,
            "follow": sets.follow,
        }
    )


def selection_records(rows) -> list[dict[str, Any]]:
    return [
        {
            "matched_order": row.matched_order,
            "archived_outcome": row.archived_outcome,
            "training_order": row.scenario.training_order,
            "l2_id": row.scenario.l2_id,
            "l4_id": row.scenario.l4_id,
            "map_name": row.scenario.map_name,
            "skill": row.scenario.skill,
            "opponent_raceline": row.scenario.opponent_raceline,
            "speedscale_hex": float(row.scenario.speedscale).hex(),
            "resolved_ego_idx": row.scenario.resolved_ego_idx,
        }
        for row in rows
    ]


def prepare(args: argparse.Namespace) -> None:
    boundary = str(args.implementation_boundary)
    if len(boundary) != 40 or any(character not in "0123456789abcdef" for character in boundary):
        raise ValueError("B6 implementation boundary must be a full lowercase commit SHA")
    rows = selected_rows(args.task8, args.metadata)
    selection_path = args.plan_dir / "prospective_selection.tsv"
    plan_path = args.plan_dir / "run_plan.json"
    if selection_path.exists() or plan_path.exists():
        raise FileExistsError("B6 prospective plan already exists")
    write_tsv(selection_path, selection_records(rows), SELECTION_FIELDS)
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "FROZEN_BEFORE_OUTCOMES",
        "implementation_boundary": boundary,
        "scientific_change": "iid_100hz_noise_vs_ar1_rho_0.95_no_learning",
        "actor": "canonical_bc_plain_end2race",
        "actor_hz": 100,
        "simulator_hz": 100,
        "rho": B6_RHO,
        "marginal_std": {"steer": 0.03, "speed": 0.20},
        "innovation_seeds": list(B6_INNOVATION_SEEDS),
        "modes": list(B6_MODES),
        "outcomes": list(B6_OUTCOMES),
        "matched_l4_count": B6_MATCHED_L4_COUNT,
        "selected_scenario_count": len(rows),
        "expected_episode_count": B6_EXPECTED_EPISODES,
        "selection_rule": (
            "intersection of training L4 containing collision/overtake/follow; "
            "one L2 per L4/outcome by domain-separated SHA256"
        ),
        "selection_digest": selection_digest(rows),
        "selection_file_sha256": file_sha256(selection_path),
        "inputs": {
            "task8_complete_sha256": file_sha256(args.task8 / "COMPLETE"),
            "training_manifest_sha256": file_sha256(args.task8 / "training_scenarios.tsv"),
            "metadata_sha256": file_sha256(args.metadata),
            "bc_checkpoint_sha256": file_sha256(args.bc_checkpoint),
        },
        "primary_gate": {
            "collision_repair_net_pair_count_min": 12,
            "collision_repair_net_pair_count_equivalent_rate": 0.05,
            "l4_cluster_signflip_one_sided_max": 0.10,
        },
        "safe_collision_noninferiority_gate": {
            "point_difference_ar1_minus_iid_max": 0.0,
            "l4_cluster_bootstrap_upper_one_sided_90_max": 0.02,
        },
        "overtake_preservation_gate": {
            "lost_overtake_rate_ar1_minus_iid_max": 0.0,
            "l4_cluster_bootstrap_upper_one_sided_90_max": 0.05,
        },
        "integrity_gate": {
            "iid_abs_lag1_max": 0.02,
            "ar1_lag1_min": 0.93,
            "ar1_lag1_max": 0.97,
            "marginal_std_relative_error_max": 0.05,
            "pre_update_max_abs_ratio_minus_one": 1e-4,
        },
        "decision": (
            "learner GO only when all integrity, repair, safe-collision, and "
            "overtake-preservation gates pass conjunctively"
        ),
        "forbidden": [
            "learner",
            "Austin_600_mechanism_selection",
            "seed0",
            "sealed_pool",
            "hyperparameter_tuning",
        ],
    }
    atomic_write(plan_path, json_bytes(plan))
    print(json.dumps({"plan": str(plan_path), "selection_digest": plan["selection_digest"]}))


def validate_plan(args: argparse.Namespace) -> tuple[dict[str, Any], tuple[Any, ...]]:
    plan_path = args.plan_dir / "run_plan.json"
    selection_path = args.plan_dir / "prospective_selection.tsv"
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "FROZEN_BEFORE_OUTCOMES":
        raise ValueError("B6 RunPlan schema/status drift")
    if file_sha256(selection_path) != plan["selection_file_sha256"]:
        raise ValueError("B6 prospective selection file drift")
    expected_inputs = {
        "task8_complete_sha256": file_sha256(args.task8 / "COMPLETE"),
        "training_manifest_sha256": file_sha256(args.task8 / "training_scenarios.tsv"),
        "metadata_sha256": file_sha256(args.metadata),
        "bc_checkpoint_sha256": file_sha256(args.bc_checkpoint),
    }
    if expected_inputs != plan["inputs"]:
        raise ValueError("B6 RunPlan input hash drift")
    rows = selected_rows(args.task8, args.metadata)
    if selection_digest(rows) != plan["selection_digest"]:
        raise ValueError("B6 deterministic selection digest drift")
    if read_tsv(selection_path) != [
        {field: str(row[field]) for field in SELECTION_FIELDS}
        for row in selection_records(rows)
    ]:
        raise ValueError("B6 prospective selection rows drift")
    if int(plan["expected_episode_count"]) != B6_EXPECTED_EPISODES:
        raise ValueError("B6 expected episode count drift")
    return plan, rows


def task_id(row, innovation_seed: int, mode: str) -> str:
    return (
        f"m{row.matched_order:03d}_{row.archived_outcome}_"
        f"s{int(innovation_seed):02d}_{mode}"
    )


def _noise_digest(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def replay_log_prob(policy: B6Phase0Policy, result, device: torch.device) -> dict[str, float]:
    feature = torch.from_numpy(np.stack([row.feature for row in result.transitions])).to(device)
    raw = torch.from_numpy(np.stack([row.raw_action for row in result.transitions])).to(device)
    old = torch.tensor(
        [row.old_log_prob for row in result.transitions], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        mean = policy.mean_from_feature(feature)
        if policy.mode == "iid":
            replayed = factorized_log_prob(raw, mean, policy.action_std.to(mean))
        else:
            values = []
            for index in range(len(raw)):
                values.append(
                    ar1_conditional_log_prob(
                        raw[index : index + 1],
                        mean[index : index + 1],
                        previous_raw_action=None if index == 0 else raw[index - 1 : index],
                        previous_mean=None if index == 0 else mean[index - 1 : index],
                        std=policy.action_std.to(mean),
                        rho=policy.rho,
                    )
                )
            replayed = torch.cat(values)
    delta = replayed - old
    return {
        "max_abs_log_prob_delta": float(torch.max(torch.abs(delta)).item()),
        "max_abs_ratio_minus_one": float(
            torch.max(torch.abs(torch.exp(delta) - 1.0)).item()
        ),
    }


def run(args: argparse.Namespace) -> None:
    plan, rows = validate_plan(args)
    if len(str(args.execution_source_commit)) != 40:
        raise ValueError("B6 execution source commit must be full SHA")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("B6 requested unavailable CUDA")
    state = torch.load(args.bc_checkpoint, map_location="cpu", weights_only=True)
    torch.manual_seed(6106)
    policies = {
        mode: B6Phase0Policy(state, mode=mode, rho=B6_RHO).to(device)
        for mode in B6_MODES
    }
    plan_sha = file_sha256(args.plan_dir / "run_plan.json")
    episode_root = args.output_dir / "episodes"
    complete = args.output_dir / "COMPLETE"
    if complete.exists():
        raise FileExistsError("B6 output is already COMPLETE")
    tasks = []
    for row in rows:
        for innovation_seed in B6_INNOVATION_SEEDS:
            for mode in arm_order(row.scenario.l2_id, innovation_seed):
                tasks.append((row, innovation_seed, mode))
    if len(tasks) != B6_EXPECTED_EPISODES:
        raise AssertionError("B6 task count drift")

    for task_order, (row, innovation_seed, mode) in enumerate(tasks):
        identifier = task_id(row, innovation_seed, mode)
        path = episode_root / f"{identifier}.json"
        expected_identity = {
            "task_id": identifier,
            "task_order": task_order,
            "matched_order": row.matched_order,
            "archived_outcome": row.archived_outcome,
            "innovation_seed": innovation_seed,
            "mode": mode,
            "l2_id": row.scenario.l2_id,
            "l4_id": row.scenario.l4_id,
            "run_plan_sha256": plan_sha,
            "execution_source_commit": str(args.execution_source_commit),
        }
        if path.exists():
            observed = json.loads(path.read_text())
            if any(observed.get(key) != value for key, value in expected_identity.items()):
                raise ValueError(f"B6 existing episode identity drift: {identifier}")
            continue
        policy = policies[mode]
        policy.begin_episode(row.scenario.l2_id, innovation_seed)
        result = run_b4_episode(
            policy,
            device,
            row.scenario,
            episode_id=task_order,
            deterministic=False,
        )
        replay_error = replay_log_prob(policy, result, device)
        trace = policy.noise_trace
        innovations = policy.innovation_trace
        if len(trace) != result.step_count or len(innovations) != result.step_count:
            raise AssertionError("B6 sampler trace length drift")
        payload = {
            "schema": RESULT_SCHEMA,
            **expected_identity,
            "map_name": row.scenario.map_name,
            "training_order": row.scenario.training_order,
            "rho": B6_RHO if mode == "ar1" else 0.0,
            "step_count": result.step_count,
            "terminal_reason": result.terminal_reason,
            "corrected_outcome": result.outcome.corrected_outcome3,
            "collision_any": bool(result.outcome.collision_any),
            "terminal_reward": float(result.transitions[-1].reward),
            "projection_transition_count": result.projection_transition_count,
            "steer_projection_count": result.steer_projection_count,
            "speed_projection_count": result.speed_projection_count,
            "max_abs_steer_projection_delta": result.max_abs_steer_projection_delta,
            "max_abs_speed_projection_delta": result.max_abs_speed_projection_delta,
            "max_abs_conditional_log_prob_replay_error": replay_error[
                "max_abs_log_prob_delta"
            ],
            "max_abs_pre_update_ratio_minus_one": replay_error[
                "max_abs_ratio_minus_one"
            ],
            "trajectory_sha256": trajectory_digest(result.arrays),
            "noise_sha256": _noise_digest(trace),
            "innovation_sha256": _noise_digest(innovations),
            "trace_moments": trace_moments(trace),
        }
        atomic_write(path, json_bytes(payload))
        if (task_order + 1) % 10 == 0 or task_order + 1 == len(tasks):
            print(
                json.dumps(
                    {
                        "completed": task_order + 1,
                        "total": len(tasks),
                        "last": identifier,
                        "outcome": payload["corrected_outcome"],
                    }
                ),
                flush=True,
            )
    summarize(args, plan, rows)


def _binomial_two_sided(positive: int, negative: int) -> float:
    total = int(positive + negative)
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(0, min(positive, negative) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def _aggregate_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(row["trace_moments"]["count"] for row in rows)
    total = np.sum([row["trace_moments"]["sum"] for row in rows], axis=0)
    total_sq = np.sum([row["trace_moments"]["sum_sq"] for row in rows], axis=0)
    mean = total / count
    variance = total_sq / count - mean**2
    result: dict[str, Any] = {
        "transition_count": int(count),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(variance, 0.0)).tolist(),
    }
    lag_count = sum(row["trace_moments"]["lag_count"] for row in rows)
    previous_sum = np.sum([row["trace_moments"]["lag_prev_sum"] for row in rows], axis=0)
    next_sum = np.sum([row["trace_moments"]["lag_next_sum"] for row in rows], axis=0)
    previous_sq = np.sum([row["trace_moments"]["lag_prev_sq"] for row in rows], axis=0)
    next_sq = np.sum([row["trace_moments"]["lag_next_sq"] for row in rows], axis=0)
    cross = np.sum([row["trace_moments"]["lag_cross"] for row in rows], axis=0)
    covariance = cross - previous_sum * next_sum / lag_count
    previous_var = previous_sq - previous_sum**2 / lag_count
    next_var = next_sq - next_sum**2 / lag_count
    result["lag1_correlation"] = (
        covariance / np.sqrt(np.maximum(previous_var * next_var, 1e-300))
    ).tolist()
    result["window_mean_rms"] = {}
    for window in (10, 30, 50):
        window_rows = [row["trace_moments"]["windows"][str(window)] for row in rows]
        window_count = sum(row["count"] for row in window_rows)
        window_sq = np.sum([row["sum_sq"] for row in window_rows], axis=0)
        result["window_mean_rms"][str(window)] = np.sqrt(window_sq / window_count).tolist()
    return result


def _metric_summary(
    cluster_sums: Mapping[str, int],
    observations_per_cluster: int,
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    ordered = [int(cluster_sums[key]) for key in sorted(cluster_sums)]
    if len(ordered) != B6_MATCHED_L4_COUNT:
        raise AssertionError("B6 metric cluster count drift")
    effects = [value / observations_per_cluster for value in ordered]
    return {
        "net_pair_count": int(sum(ordered)),
        "rate_difference": float(sum(ordered) / (len(ordered) * observations_per_cluster)),
        "cluster_signflip_one_sided": exact_cluster_signflip_one_sided(ordered),
        "cluster_bootstrap": paired_cluster_bootstrap(
            effects, seed=bootstrap_seed, samples=BOOTSTRAP_SAMPLES
        ),
        "cluster_integer_effects": ordered,
    }


def summarize(args: argparse.Namespace, plan: dict[str, Any], rows) -> None:
    episode_root = args.output_dir / "episodes"
    episodes = [json.loads(path.read_text()) for path in sorted(episode_root.glob("*.json"))]
    if len(episodes) != B6_EXPECTED_EPISODES:
        raise ValueError(f"B6 expected {B6_EXPECTED_EPISODES} episodes, found {len(episodes)}")
    by_key: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    for row in episodes:
        if row.get("schema") != RESULT_SCHEMA:
            raise ValueError("B6 episode schema drift")
        key = (
            int(row["matched_order"]),
            row["archived_outcome"],
            int(row["innovation_seed"]),
            row["mode"],
        )
        if key in by_key:
            raise ValueError("B6 duplicate episode key")
        by_key[key] = row
    pairs: list[dict[str, Any]] = []
    selected_by_key = {(row.matched_order, row.archived_outcome): row for row in rows}
    for matched_order in range(B6_MATCHED_L4_COUNT):
        for outcome in B6_OUTCOMES:
            selected = selected_by_key[(matched_order, outcome)]
            for innovation_seed in B6_INNOVATION_SEEDS:
                iid = by_key[(matched_order, outcome, innovation_seed, "iid")]
                ar1 = by_key[(matched_order, outcome, innovation_seed, "ar1")]
                iid_collision = int(bool(iid["collision_any"]))
                ar1_collision = int(bool(ar1["collision_any"]))
                iid_repair = int(outcome == "collision" and not iid_collision)
                ar1_repair = int(outcome == "collision" and not ar1_collision)
                iid_harm = int(outcome != "collision" and iid_collision)
                ar1_harm = int(outcome != "collision" and ar1_collision)
                iid_lost = int(outcome == "overtake" and iid["corrected_outcome"] != "overtake")
                ar1_lost = int(outcome == "overtake" and ar1["corrected_outcome"] != "overtake")
                pairs.append(
                    {
                        "matched_order": matched_order,
                        "archived_outcome": outcome,
                        "innovation_seed": innovation_seed,
                        "l2_id": selected.scenario.l2_id,
                        "l4_id": selected.scenario.l4_id,
                        "map_name": selected.scenario.map_name,
                        "iid_outcome": iid["corrected_outcome"],
                        "ar1_outcome": ar1["corrected_outcome"],
                        "iid_collision": iid_collision,
                        "ar1_collision": ar1_collision,
                        "iid_repaired_collision": iid_repair,
                        "ar1_repaired_collision": ar1_repair,
                        "repair_delta": ar1_repair - iid_repair,
                        "iid_safe_to_collision": iid_harm,
                        "ar1_safe_to_collision": ar1_harm,
                        "safe_harm_delta": ar1_harm - iid_harm,
                        "iid_lost_overtake": iid_lost,
                        "ar1_lost_overtake": ar1_lost,
                        "overtake_loss_delta": ar1_lost - iid_lost,
                    }
                )
    write_tsv(args.output_dir / "paired_results.tsv", pairs, PAIR_FIELDS)

    cluster_sums: dict[str, dict[str, int]] = {
        "repair": defaultdict(int),
        "safe_harm": defaultdict(int),
        "overtake_loss": defaultdict(int),
    }
    for row in pairs:
        if row["archived_outcome"] == "collision":
            cluster_sums["repair"][row["l4_id"]] += int(row["repair_delta"])
        else:
            cluster_sums["safe_harm"][row["l4_id"]] += int(row["safe_harm_delta"])
        if row["archived_outcome"] == "overtake":
            cluster_sums["overtake_loss"][row["l4_id"]] += int(row["overtake_loss_delta"])
    repair = _metric_summary(
        cluster_sums["repair"], len(B6_INNOVATION_SEEDS), bootstrap_seed=610601
    )
    safe_harm = _metric_summary(
        cluster_sums["safe_harm"], 2 * len(B6_INNOVATION_SEEDS), bootstrap_seed=610602
    )
    overtake_loss = _metric_summary(
        cluster_sums["overtake_loss"], len(B6_INNOVATION_SEEDS), bootstrap_seed=610603
    )

    mode_outcomes: dict[str, dict[str, Counter[str]]] = {
        mode: {outcome: Counter() for outcome in B6_OUTCOMES} for mode in B6_MODES
    }
    projections: dict[str, Counter[str]] = {mode: Counter() for mode in B6_MODES}
    max_log_prob = 0.0
    max_ratio = 0.0
    for row in episodes:
        mode_outcomes[row["mode"]][row["archived_outcome"]][row["corrected_outcome"]] += 1
        projections[row["mode"]].update(
            {
                "transitions": int(row["step_count"]),
                "projection_transitions": int(row["projection_transition_count"]),
                "steer_projections": int(row["steer_projection_count"]),
                "speed_projections": int(row["speed_projection_count"]),
            }
        )
        max_log_prob = max(max_log_prob, float(row["max_abs_conditional_log_prob_replay_error"]))
        max_ratio = max(max_ratio, float(row["max_abs_pre_update_ratio_minus_one"]))
    noise = {
        mode: _aggregate_trace([row for row in episodes if row["mode"] == mode])
        for mode in B6_MODES
    }
    target_std = np.asarray([0.03, 0.20])
    relative_std_errors = {
        mode: np.abs(np.asarray(noise[mode]["std"]) / target_std - 1.0).tolist()
        for mode in B6_MODES
    }
    integrity = bool(
        max_ratio <= 1e-4
        and max(abs(value) for value in noise["iid"]["lag1_correlation"]) <= 0.02
        and all(0.93 <= value <= 0.97 for value in noise["ar1"]["lag1_correlation"])
        and max(value for mode in B6_MODES for value in relative_std_errors[mode]) <= 0.05
    )
    repair_gate = bool(
        repair["net_pair_count"] >= 12
        and repair["cluster_signflip_one_sided"] <= 0.10
    )
    safe_gate = bool(
        safe_harm["rate_difference"] <= 0.0
        and safe_harm["cluster_bootstrap"]["upper_one_sided_90"] <= 0.02
    )
    overtake_gate = bool(
        overtake_loss["rate_difference"] <= 0.0
        and overtake_loss["cluster_bootstrap"]["upper_one_sided_90"] <= 0.05
    )
    phase0_go = bool(integrity and repair_gate and safe_gate and overtake_gate)

    occurrence = {}
    for metric, field in (
        ("repair", "repair_delta"),
        ("safe_harm", "safe_harm_delta"),
        ("overtake_loss", "overtake_loss_delta"),
    ):
        relevant = [int(row[field]) for row in pairs if int(row[field]) != 0]
        positive = sum(value > 0 for value in relevant)
        negative = sum(value < 0 for value in relevant)
        occurrence[metric] = {
            "positive": positive,
            "negative": negative,
            "mcnemar_two_sided": _binomial_two_sided(positive, negative),
        }

    summary = {
        "schema": SUMMARY_SCHEMA,
        "phase0_decision": "GO_FOR_LEARNER_PROPOSAL" if phase0_go else "NO_GO",
        "learner_started": False,
        "run_plan_sha256": file_sha256(args.plan_dir / "run_plan.json"),
        "execution_source_commit": str(args.execution_source_commit),
        "episode_count": len(episodes),
        "pair_count": len(pairs),
        "matched_l4_count": B6_MATCHED_L4_COUNT,
        "innovation_seeds": list(B6_INNOVATION_SEEDS),
        "outcome_counts": {
            mode: {outcome: dict(mode_outcomes[mode][outcome]) for outcome in B6_OUTCOMES}
            for mode in B6_MODES
        },
        "repair_ar1_minus_iid": repair,
        "safe_collision_harm_ar1_minus_iid": safe_harm,
        "overtake_loss_ar1_minus_iid": overtake_loss,
        "occurrence_mcnemar": occurrence,
        "noise": noise,
        "relative_std_errors": relative_std_errors,
        "projection_counts": {mode: dict(projections[mode]) for mode in B6_MODES},
        "max_abs_conditional_log_prob_replay_error": max_log_prob,
        "max_abs_pre_update_ratio_minus_one": max_ratio,
        "gates": {
            "integrity": integrity,
            "collision_repair": repair_gate,
            "safe_collision_noninferiority": safe_gate,
            "overtake_preservation": overtake_gate,
            "all_conjunctive": phase0_go,
        },
        "scope": (
            "training-only matched-L4 common-random-number mechanism audit; "
            "not product evaluation and not evidence that PPO can learn the behavior"
        ),
    }
    atomic_write(args.output_dir / "summary.json", json_bytes(summary))
    report = f"""# B6 temporal-exploration phase-0 result

Decision: **{summary['phase0_decision']}**

This is a no-learning, training-only, matched-L4 mechanism audit. It does not
evaluate a candidate checkpoint and cannot establish PPO learnability.

| Gate | Pass |
|---|---:|
| integrity | `{integrity}` |
| collision repair | `{repair_gate}` |
| safe-to-collision non-inferiority | `{safe_gate}` |
| overtake preservation | `{overtake_gate}` |

| Direct paired effect (AR1 - iid) | Net | Rate | L4 sign-flip p | 90% upper cluster bound |
|---|---:|---:|---:|---:|
| collision repair | `{repair['net_pair_count']}` | `{repair['rate_difference']:.6f}` | `{repair['cluster_signflip_one_sided']:.6f}` | `{repair['cluster_bootstrap']['upper_one_sided_90']:.6f}` |
| safe-to-collision harm | `{safe_harm['net_pair_count']}` | `{safe_harm['rate_difference']:.6f}` | `{safe_harm['cluster_signflip_one_sided']:.6f}` | `{safe_harm['cluster_bootstrap']['upper_one_sided_90']:.6f}` |
| lost overtake | `{overtake_loss['net_pair_count']}` | `{overtake_loss['rate_difference']:.6f}` | `{overtake_loss['cluster_signflip_one_sided']:.6f}` | `{overtake_loss['cluster_bootstrap']['upper_one_sided_90']:.6f}` |

The learner remains unrun. A phase-0 GO would only authorize a separate
learner proposal; a NO-GO closes this AR(1) setting without changing the
canonical actor, evaluator, or sealed data.
"""
    atomic_write(args.output_dir / "report.md", report.encode("utf-8"))
    atomic_write(args.output_dir / "COMPLETE", (file_sha256(args.output_dir / "summary.json") + "\n").encode())
    print(json.dumps({"decision": summary["phase0_decision"], "summary": str(args.output_dir / "summary.json")}), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("command", choices=("prepare", "run", "summarize"))
    value.add_argument("--task8", type=Path, default=DEFAULT_TASK8)
    value.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    value.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC)
    value.add_argument("--plan-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--implementation-boundary", default="")
    value.add_argument("--execution-source-commit", default="")
    value.add_argument("--device", default="cuda:0")
    return value


def main() -> None:
    args = parser().parse_args()
    args.task8 = args.task8.resolve()
    args.metadata = args.metadata.resolve()
    args.bc_checkpoint = args.bc_checkpoint.resolve()
    args.plan_dir = args.plan_dir.resolve()
    if args.command == "prepare":
        prepare(args)
        return
    if args.output_dir is None:
        raise ValueError("B6 run/summarize requires --output-dir")
    args.output_dir = args.output_dir.resolve()
    plan, rows = validate_plan(args)
    if args.command == "run":
        run(args)
    else:
        summarize(args, plan, rows)


if __name__ == "__main__":
    main()

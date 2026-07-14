"""Versioned seed-1 B5-A learner and blocking safe-cap smoke contracts."""

from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from bplus_v22.b4_direct import (
    B4Batch,
    B4Curriculum,
    B4DirectHeadPolicy,
    B4ScenarioSets,
    B4Transition,
    FROZEN_B4_CONFIG,
    actor_snapshot_sha256,
    build_batch,
    build_optimizers,
    load_strict_plain_actor,
    project_raw_action,
    projection_metrics,
    replay_metrics,
    save_actor_snapshot,
    strict_plain_actor_from_state,
    validate_frozen_config,
)
from bplus_v22.b4_env import B4EpisodeResult, run_b4_episode
from bplus_v22.b4_runner import (
    B4_REPLAY_RATIO_ATOL,
    TRAINING_MANIFEST_NAME,
    _append_jsonl,
    _canonical_json,
    _curriculum_record,
    _file_sha256,
    _is_sha256,
    _outcome_record,
    _quarantine_uncommitted,
    _read_iteration_ledger,
    _repair_torn_ledger,
    _validate_resume_prefix,
    _write_json,
    _write_replay,
    run_b4_stochastic_plumbing_smoke,
)
from bplus_v22.b5_safe import (
    B5_PILOT_SCHEMA,
    B5_PLUMBING_SCHEMA,
    B5_POLICY_SCHEMA,
    B5_READY_SCHEMA,
    SAFE_CAP,
    SAFE_EPISODE_COUNT,
    SAFE_RETRY_MULTIPLIERS,
    SafeReference,
    file_sha256,
    load_b5_full_checkpoint,
    load_reference,
    safe_kl_metrics,
    save_b5_full_checkpoint,
    update_policy_with_safe_cap,
)
from bplus_v22.ppo_env import load_b2_scenario_sets


B5_RUN_PLAN_SCHEMA = "end2race-b2-run-plan-1"
B5_TRAIN_KIND = "b5_train"
B5_JOB_KIND = "b5_training"
B5_SEED_BASE = 507140
B5_REFERENCE_RELPATH = "inputs/b5/safe_reference.npz"
B4_PARENT_RUN_PLAN_SHA256 = "08f0fe4275ae60928a6d5a6ce9704679bc91a624258bf5aef7f7a268b2c5e381"
B4_SEED1_CURRICULUM_SHA256 = "40275f3d928b753fdc683ca20df83ad4097d9e8ac3c92f4a150fba3a50a5afa1"


def expected_b5_plan_config(
    task8_release: str | Path,
    d2_episode_metadata: str | Path,
    reference_path: str | Path,
) -> dict[str, Any]:
    task8 = Path(task8_release)
    metadata = Path(d2_episode_metadata)
    reference_path = Path(reference_path)
    reference = load_reference(reference_path)
    scenarios = B4ScenarioSets.from_b2(load_b2_scenario_sets(task8, metadata))
    curriculum_sha = B4Curriculum(scenarios, 1).digest(FROZEN_B4_CONFIG.iterations)
    if curriculum_sha != B4_SEED1_CURRICULUM_SHA256:
        raise ValueError("B5 exact B4 seed1 curriculum digest drift")
    return {
        "policy_contract": B5_POLICY_SCHEMA,
        "ppo": FROZEN_B4_CONFIG.as_dict(),
        "curriculum_schema": "end2race-b4-curriculum-1",
        "curriculum_sha256_by_seed": {"1": curriculum_sha},
        "b4_parent_run_plan_sha256": B4_PARENT_RUN_PLAN_SHA256,
        "training_manifest_sha256": _file_sha256(task8 / TRAINING_MANIFEST_NAME),
        "safe_reference": {
            "relpath": B5_REFERENCE_RELPATH,
            "sha256": file_sha256(reference_path),
            "episode_count": SAFE_EPISODE_COUNT,
            "frame_count": reference.frame_count,
            "cap": SAFE_CAP,
            "retry_multipliers": list(SAFE_RETRY_MULTIPLIERS),
            "metric": "mean_episode_mean_frame_equal_std_latent_mean_kl",
        },
        "bc_baseline_expected_collision": 24,
        "bc_baseline_expected_overtake": 138,
        "bc_baseline_topology": {
            "0": {"host_id": "local", "collision": 12, "terminal_overtake": 32},
            "1": {"host_id": "remote", "collision": 2, "terminal_overtake": 37},
            "2": {"host_id": "remote", "collision": 5, "terminal_overtake": 33},
            "3": {"host_id": "remote", "collision": 5, "terminal_overtake": 36},
        },
        "opened_development_panel": {
            "map": "Austin",
            "opponent_racelines": ["raceline0", "raceline1", "raceline2"],
            "opponent_speed_scales": [0.5, 0.6, 0.7, 0.8],
            "startpoint_count": 50,
            "bc_collision": 24,
            "bc_overtake": 342,
            "overtake_floor": 325,
            "collision_target": 16,
        },
        "inputs": {
            "bc_checkpoint": "repo/pretrained/end2race.pth",
            "task8_release": "inputs/task8",
            "training_manifest": "inputs/task8/training_scenarios.tsv",
            "development_manifest": "inputs/task8/development_scenarios.tsv",
            "d2_episode_metadata": "inputs/d2/episode_metadata.tsv",
            "safe_reference": B5_REFERENCE_RELPATH,
        },
        "forbidden_inputs": [
            "sidecar",
            "D2_test",
            "fresh_pool",
            "final_pool",
            "eval_results",
        ],
    }


def load_b5_run_plan(path: str | Path) -> tuple[Path, dict[str, Any]]:
    plan_path = Path(path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != B5_RUN_PLAN_SCHEMA or plan.get("kind") != B5_TRAIN_KIND:
        raise ValueError("B5 learner requires one b5_train RunPlan")
    observed = plan.get("plan_sha256")
    if not isinstance(observed, str) or not _is_sha256(observed):
        raise ValueError("B5 RunPlan digest is invalid")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    if observed != hashlib.sha256(_canonical_json(unsigned)).hexdigest():
        raise ValueError("B5 RunPlan digest mismatch")
    return plan_path, plan


def _staged_paths(plan_path: Path, plan: Mapping[str, Any]) -> dict[str, Path]:
    root = plan_path.parent.parent
    paths = {
        "root": root,
        "repo": root / "repo",
        "bc": root / "repo/pretrained/end2race.pth",
        "task8": root / "inputs/task8",
        "metadata": root / "inputs/d2/episode_metadata.tsv",
        "reference": root / B5_REFERENCE_RELPATH,
    }
    for name, path in paths.items():
        if name != "root" and not path.exists():
            raise ValueError(f"B5 staged input is missing: {name}={path}")
    if Path.cwd().resolve() != paths["repo"].resolve():
        raise ValueError("B5 learner must execute from the staged repository root")
    config = plan.get("config")
    if not isinstance(config, Mapping) or dict(config) != expected_b5_plan_config(
        paths["task8"], paths["metadata"], paths["reference"]
    ):
        raise ValueError("B5 staged config/reference/curriculum drift")
    validate_frozen_config(config["ppo"])
    return paths


def validate_b5_pilot_plan(
    plan_path: str | Path,
    job_id: str | None = None,
    *,
    allow_partial_resume: bool = False,
) -> dict[str, Any]:
    path, plan = load_b5_run_plan(plan_path)
    paths = _staged_paths(path, plan)
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1:
        raise ValueError("B5 RunPlan must contain exactly one seed-1 job")
    job = jobs[0]
    if (
        not isinstance(job, Mapping)
        or job.get("job_id") != "b5-seed1"
        or job.get("kind") != B5_JOB_KIND
        or job.get("seed") != 1
        or not isinstance(job.get("output_relpath"), str)
    ):
        raise ValueError("B5 RunPlan job contract mismatch")
    selected = dict(job) if job_id == "b5-seed1" else None
    if job_id is not None and selected is None:
        raise ValueError(f"B5 RunPlan does not contain job {job_id!r}")
    if selected is not None:
        output = paths["root"] / selected["output_relpath"]
        partial = output.with_name(output.name + ".partial")
        if allow_partial_resume:
            if output.exists() or not partial.is_dir():
                raise ValueError("B5 resume requires one incomplete partial release")
        elif output.exists() or partial.exists():
            raise ValueError("B5 fresh job output already exists")
    return {"plan": plan, "paths": paths, "job": selected}


def _validate_control_plane_ready(plan: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    control = paths["root"] / "control"
    baseline_path = control / "bc_baseline_preflight.json"
    plumbing_path = control / "plumbing_smoke.json"
    ready_path = control / "READY.json"
    for path in (baseline_path, plumbing_path, ready_path):
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError(f"B5 learner lacks safe control marker: {path.name}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    plumbing = json.loads(plumbing_path.read_text(encoding="utf-8"))
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if (
        baseline.get("schema") != "bplus-v2.2-b2-bc-baseline-preflight-2"
        or baseline.get("integrity_passed") is not True
        or baseline.get("passed") is not True
        or baseline.get("acceptance_passed") is not True
        or baseline.get("collision") != 24
        or baseline.get("terminal_overtake") != 138
        or baseline.get("candidate_evaluated") is not False
    ):
        raise ValueError("B5 learner baseline authorization mismatch")
    if (
        plumbing.get("schema") != B5_PLUMBING_SCHEMA
        or plumbing.get("passed") is not True
        or plumbing.get("run_plan_sha256") != plan.get("plan_sha256")
        or plumbing.get("reference_sha256")
        != plan["config"]["safe_reference"]["sha256"]
        or plumbing.get("product_outcomes_reported_or_compared") is not False
        or plumbing.get("candidate_selection_performed") is not False
        or plumbing.get("ppo_pilot_iteration_completed") is not False
    ):
        raise ValueError("B5 learner plumbing authorization mismatch")
    expected = {
        "schema": B5_READY_SCHEMA,
        "passed": True,
        "run_plan_sha256": plan.get("plan_sha256"),
        "source_commit": plan.get("source_commit"),
        "source_archive_sha256": plan.get("source_archive_sha256"),
        "inputs_archive_sha256": plan.get("inputs_archive_sha256"),
        "baseline_marker_sha256": _file_sha256(baseline_path),
        "plumbing_marker_sha256": _file_sha256(plumbing_path),
    }
    if ready != expected:
        raise ValueError("B5 learner READY authorization mismatch")
    return ready


def _synthetic_safe_reference(policy: B4DirectHeadPolicy, device: torch.device) -> SafeReference:
    generator = torch.Generator().manual_seed(507140)
    feature = torch.randn((SAFE_EPISODE_COUNT, 1680), generator=generator).to(device)
    with torch.no_grad():
        mean = policy.mean_from_feature(feature).detach().clone()
    outcomes = tuple(outcome for _map in range(4) for outcome in ("follow", "overtake") for _ in range(8))
    # Reorder maps to match the outcome construction: 16 rows per map.
    maps = tuple(map_name for map_name in ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring") for _outcome in ("follow", "overtake") for _ in range(8))
    return SafeReference(
        feature=feature,
        bc_mean=mean,
        episode_index=torch.arange(SAFE_EPISODE_COUNT, device=device),
        step_index=torch.zeros(SAFE_EPISODE_COUNT, dtype=torch.int64, device=device),
        lengths=(1,) * SAFE_EPISODE_COUNT,
        l2_ids=tuple(f"L2:{index:064x}" for index in range(SAFE_EPISODE_COUNT)),
        l4_ids=tuple(f"L4:{index:064x}" for index in range(SAFE_EPISODE_COUNT)),
        map_names=maps,
        outcomes=outcomes,
    )


def _synthetic_batch(policy: B4DirectHeadPolicy) -> B4Batch:
    generator = torch.Generator().manual_seed(507141)
    transitions = []
    for episode in range(16):
        for step in range(2):
            feature = torch.randn((1, 1680), generator=generator)
            privileged = torch.randn((1, 12), generator=generator)
            with torch.no_grad():
                mean = policy.mean_from_feature(feature)
                raw = mean + (0.25 if (episode + step) % 2 else -0.25) * policy.action_std
                old_log_prob = policy.log_prob(mean, raw)
                value = policy.value(privileged)
                executed, delta = project_raw_action(raw)
            terminal = step == 1
            transitions.append(
                B4Transition(
                    l2_id=f"L2:{episode:064x}",
                    episode_id=episode,
                    step_index=step,
                    feature=feature[0].numpy().astype(np.float32),
                    privileged_feature=privileged[0].numpy().astype(np.float32),
                    raw_action=raw[0].numpy().astype(np.float32),
                    executed_action=executed[0].numpy().astype(np.float32),
                    projection_delta=delta[0].numpy().astype(np.float32),
                    old_log_prob=float(old_log_prob.item()),
                    old_value=float(value.item()),
                    reward=float((-2, 0, 1)[episode % 3]) if terminal else 0.0,
                    terminated=terminal,
                )
            )
    return build_batch(transitions, policy.config)


def run_b5_plumbing_smoke(plan_path: str | Path, *, device_name: str = "cuda:0") -> dict[str, Any]:
    validated = validate_b5_pilot_plan(plan_path)
    from d25.oracle import simulate_episode

    plan = validated["plan"]
    paths = validated["paths"]
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("B5 plumbing smoke requested unavailable CUDA")
    bc_state = torch.load(paths["bc"], map_location="cpu", weights_only=True)
    policy = B4DirectHeadPolicy(bc_state).to(device)
    reference = load_reference(paths["reference"], device)
    initial_safe = safe_kl_metrics(policy, reference)
    if initial_safe["mean"] > 1e-10:
        raise AssertionError("B5 iteration-0 safe KL is not zero within numerical tolerance")
    scenarios = B4ScenarioSets.from_b2(load_b2_scenario_sets(paths["task8"], paths["metadata"]))
    selected_by_map = {}
    for scenario in sorted(
        (*scenarios.collision, *scenarios.overtake, *scenarios.follow),
        key=lambda row: row.training_order,
    ):
        selected_by_map.setdefault(scenario.map_name, scenario)
    expected_maps = ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")
    reports = []
    for episode_id, map_name in enumerate(expected_maps):
        scenario = selected_by_map[map_name]
        canonical = simulate_episode(policy.actor, device, scenario.simulator_case())
        result = run_b4_episode(policy, device, scenario, episode_id=episode_id, deterministic=True)
        mismatches = [
            name
            for name in sorted(set(canonical.arrays) | set(result.arrays))
            if name not in canonical.arrays
            or name not in result.arrays
            or not np.array_equal(np.asarray(canonical.arrays[name]), np.asarray(result.arrays[name]))
        ]
        if mismatches or canonical.outcome.corrected_outcome3 != result.outcome.corrected_outcome3:
            raise AssertionError(f"B5 deterministic identity failed on {map_name}: {mismatches}")
        reports.append({
            "map_name": map_name,
            "l2_id": scenario.l2_id,
            "step_count": result.step_count,
            "terminal_reason": result.terminal_reason,
            "trajectory_identity": True,
            "outcome_identity": True,
        })

    unchanged_b4 = run_b4_stochastic_plumbing_smoke(
        bc_state,
        device,
        scenarios,
        run_plan_sha256=plan["plan_sha256"],
        curriculum_sha256=plan["config"]["curriculum_sha256_by_seed"]["1"],
    )

    smoke_config = replace(FROZEN_B4_CONFIG, actor_lr=1e-2, actor_epochs=1, minibatch_size=8)
    solver_policy = B4DirectHeadPolicy(bc_state, smoke_config).to(device)
    actor_optimizer, critic_optimizer = build_optimizers(solver_policy)
    synthetic_reference = _synthetic_safe_reference(solver_policy, device)
    batch = _synthetic_batch(solver_policy)
    actor_before = copy.deepcopy(solver_policy.actor.output_layer.state_dict())
    optimizer_before = copy.deepcopy(actor_optimizer.state_dict())
    solver = update_policy_with_safe_cap(
        solver_policy,
        batch,
        synthetic_reference,
        actor_optimizer,
        critic_optimizer,
        seed=1,
        iteration=1,
        safe_cap=0.0,
    )
    if solver["actor_epochs_skipped"] != 1 or solver["critic_epochs_completed"] != 3:
        raise AssertionError("B5 safe solver smoke did not exercise reject/critic isolation")
    if any(
        not torch.equal(actor_before[name], value)
        for name, value in solver_policy.actor.output_layer.state_dict().items()
    ):
        raise AssertionError("B5 rejected actor epoch did not restore parameters exactly")
    optimizer_after = actor_optimizer.state_dict()
    if optimizer_before.keys() != optimizer_after.keys() or optimizer_before["param_groups"] != optimizer_after["param_groups"]:
        raise AssertionError("B5 rejected actor epoch did not restore Adam groups")
    if optimizer_before["state"].keys() != optimizer_after["state"].keys():
        raise AssertionError("B5 rejected actor epoch changed Adam inventory")
    for parameter_id, before in optimizer_before["state"].items():
        after = optimizer_after["state"][parameter_id]
        if before.keys() != after.keys():
            raise AssertionError("B5 rejected actor epoch changed Adam state fields")
        for name, value in before.items():
            if torch.is_tensor(value):
                equal = torch.equal(value, after[name])
            else:
                equal = value == after[name]
            if not equal:
                raise AssertionError("B5 rejected actor epoch did not restore Adam state")

    return {
        "schema": B5_PLUMBING_SCHEMA,
        "passed": True,
        "run_plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source_commit"],
        "reference_sha256": plan["config"]["safe_reference"]["sha256"],
        "reference_episode_count": len(reference.lengths),
        "reference_frame_count": reference.frame_count,
        "iteration0_safe": initial_safe,
        "map_reports": reports,
        "unchanged_b4_stochastic_plumbing": unchanged_b4,
        "safe_solver": solver,
        "actor_adam_restore_exact": True,
        "product_outcomes_reported_or_compared": False,
        "candidate_selection_performed": False,
        "ppo_pilot_iteration_completed": False,
    }


def run_b5_pilot_job(
    plan_path: str | Path,
    job_id: str,
    *,
    device_name: str = "cuda:0",
    resume: bool = False,
) -> dict[str, Any]:
    validated = validate_b5_pilot_plan(plan_path, job_id, allow_partial_resume=resume)
    plan = validated["plan"]
    paths = validated["paths"]
    job = validated["job"]
    _validate_control_plane_ready(plan, paths)
    seed = int(job["seed"])
    output = paths["root"] / job["output_relpath"]
    partial = output.with_name(output.name + ".partial")
    if not resume:
        partial.mkdir(parents=True)
    try:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("B5 learner requested unavailable CUDA")
        base_seed = B5_SEED_BASE + seed
        torch.manual_seed(base_seed)
        np.random.seed(base_seed)
        random.seed(base_seed)
        bc_state = torch.load(paths["bc"], map_location="cpu", weights_only=True)
        strict_plain_actor_from_state(bc_state)
        policy = B4DirectHeadPolicy(bc_state).to(device)
        actor_optimizer, critic_optimizer = build_optimizers(policy)
        reference = load_reference(paths["reference"], device)
        reference_sha = file_sha256(paths["reference"])
        scenarios = B4ScenarioSets.from_b2(load_b2_scenario_sets(paths["task8"], paths["metadata"]))
        curriculum_plan = B4Curriculum(scenarios, seed).plan(FROZEN_B4_CONFIG.iterations)
        curriculum_sha = B4Curriculum(scenarios, seed).digest(FROZEN_B4_CONFIG.iterations)
        if curriculum_sha != plan["config"]["curriculum_sha256_by_seed"][str(seed)]:
            raise ValueError("B5 learner curriculum differs from exact B4 seed1 order")
        if reference_sha != plan["config"]["safe_reference"]["sha256"]:
            raise ValueError("B5 learner safe reference digest differs from RunPlan")
        initial_safe = safe_kl_metrics(policy, reference)
        if initial_safe["mean"] > 1e-10:
            raise ValueError("B5 iteration-0 safe KL is nonzero")
        config_record = {
            "schema": B5_PILOT_SCHEMA,
            "seed": seed,
            "run_plan_sha256": plan["plan_sha256"],
            "source_commit": plan["source_commit"],
            "bc_checkpoint_sha256": _file_sha256(paths["bc"]),
            "bc_actor_tensor_sha256": actor_snapshot_sha256(policy.actor_state()),
            "training_manifest_sha256": _file_sha256(paths["task8"] / TRAINING_MANIFEST_NAME),
            "curriculum_sha256": curriculum_sha,
            "reference_sha256": reference_sha,
            "config": FROZEN_B4_CONFIG.as_dict(),
            "safe_cap": SAFE_CAP,
            "retry_multipliers": list(SAFE_RETRY_MULTIPLIERS),
        }
        curriculum_record = _curriculum_record(curriculum_plan)
        if resume:
            if json.loads((partial / "config.json").read_text(encoding="utf-8")) != config_record:
                raise ValueError("B5 resume config prefix drift")
            if json.loads((partial / "curriculum.json").read_text(encoding="utf-8")) != curriculum_record:
                raise ValueError("B5 resume curriculum prefix drift")
            _repair_torn_ledger(partial)
            ledger = _read_iteration_ledger(partial / "iterations.jsonl")
            committed = len(ledger)
            _quarantine_uncommitted(partial, committed)
            _validate_resume_prefix(
                partial,
                ledger,
                expected_bc_actor_tensor_sha256=config_record["bc_actor_tensor_sha256"],
            )
            loaded = load_b5_full_checkpoint(
                partial / f"checkpoints/iter_{committed:04d}.pt",
                policy,
                actor_optimizer,
                critic_optimizer,
                expected_seed=seed,
                expected_run_plan_sha256=plan["plan_sha256"],
                expected_curriculum_sha256=curriculum_sha,
                expected_reference_sha256=reference_sha,
            )
            if loaded != committed:
                raise ValueError("B5 resume checkpoint/ledger iteration mismatch")
            start_iteration = committed + 1
            resumed_from_iteration = committed
        else:
            _write_json(partial / "config.json", config_record)
            _write_json(partial / "curriculum.json", curriculum_record)
            actor0 = partial / "actors/iter_0000.pth"
            actor0_record = save_actor_snapshot(policy, actor0)
            if actor0_record["tensor_sha256"] != config_record["bc_actor_tensor_sha256"]:
                raise AssertionError("B5 iteration-0 actor differs from canonical BC")
            save_b5_full_checkpoint(
                policy,
                actor_optimizer,
                critic_optimizer,
                partial / "checkpoints/iter_0000.pt",
                completed_iteration=0,
                seed=seed,
                run_plan_sha256=plan["plan_sha256"],
                curriculum_sha256=curriculum_sha,
                reference_sha256=reference_sha,
            )
            start_iteration = 1
            resumed_from_iteration = None

        for iteration in range(start_iteration, FROZEN_B4_CONFIG.iterations + 1):
            episode_results: list[B4EpisodeResult] = []
            transitions = []
            for episode_index, scenario in enumerate(curriculum_plan[iteration - 1]):
                episode_id = (iteration - 1) * 16 + episode_index
                result = run_b4_episode(policy, device, scenario, episode_id=episode_id, deterministic=False)
                episode_results.append(result)
                transitions.extend(result.transitions)
            batch = build_batch(transitions, FROZEN_B4_CONFIG)
            preupdate = replay_metrics(policy, batch.to(device))
            if preupdate["max_abs_ratio_minus_one"] > B4_REPLAY_RATIO_ATOL:
                raise AssertionError("B5 pre-update raw-latent replay ratio is not one")
            projection = projection_metrics(batch)
            replay_path = partial / f"replay/iter_{iteration:04d}.npz"
            replay_sha = _write_replay(replay_path, batch)
            update = update_policy_with_safe_cap(
                policy,
                batch,
                reference,
                actor_optimizer,
                critic_optimizer,
                seed=seed,
                iteration=iteration,
            )
            actor_file_sha = actor_tensor_sha = None
            if iteration in FROZEN_B4_CONFIG.snapshots:
                actor_path = partial / f"actors/iter_{iteration:04d}.pth"
                actor_record = save_actor_snapshot(policy, actor_path)
                actor_file_sha = _file_sha256(actor_path)
                actor_tensor_sha = actor_record["tensor_sha256"]
            full_path = partial / f"checkpoints/iter_{iteration:04d}.pt"
            save_b5_full_checkpoint(
                policy,
                actor_optimizer,
                critic_optimizer,
                full_path,
                completed_iteration=iteration,
                seed=seed,
                run_plan_sha256=plan["plan_sha256"],
                curriculum_sha256=curriculum_sha,
                reference_sha256=reference_sha,
            )
            _append_jsonl(
                partial / "iterations.jsonl",
                {
                    "iteration": iteration,
                    "episode_count": len(episode_results),
                    "transition_count": batch.size,
                    "preupdate_replay": preupdate,
                    "projection": projection,
                    "update": update,
                    "replay_sha256": replay_sha,
                    "full_checkpoint_sha256": _file_sha256(full_path),
                    "actor_snapshot_file_sha256": actor_file_sha,
                    "actor_snapshot_tensor_sha256": actor_tensor_sha,
                    "outcomes": [_outcome_record(result) for result in episode_results],
                },
            )

        snapshot_files = {
            str(iteration): _file_sha256(partial / f"actors/iter_{iteration:04d}.pth")
            for iteration in FROZEN_B4_CONFIG.snapshots
        }
        snapshot_tensors = {
            str(iteration): actor_snapshot_sha256(
                torch.load(
                    partial / f"actors/iter_{iteration:04d}.pth",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            for iteration in FROZEN_B4_CONFIG.snapshots
        }
        summary = {
            "schema": B5_PILOT_SCHEMA,
            "passed": True,
            "integrity_passed": True,
            "seed": seed,
            "iterations": FROZEN_B4_CONFIG.iterations,
            "resumed_from_iteration": resumed_from_iteration,
            "run_plan_sha256": plan["plan_sha256"],
            "source_commit": plan["source_commit"],
            "bc_checkpoint_sha256": config_record["bc_checkpoint_sha256"],
            "bc_actor_tensor_sha256": config_record["bc_actor_tensor_sha256"],
            "training_manifest_sha256": config_record["training_manifest_sha256"],
            "curriculum_sha256": curriculum_sha,
            "reference_sha256": reference_sha,
            "safe_final": safe_kl_metrics(policy, reference),
            "actor_snapshot_file_sha256_by_iteration": snapshot_files,
            "actor_snapshot_tensor_sha256_by_iteration": snapshot_tensors,
            "final_full_checkpoint_sha256": _file_sha256(
                partial / f"checkpoints/iter_{FROZEN_B4_CONFIG.iterations:04d}.pt"
            ),
            "opened_development_kpi_evaluated": False,
            "fresh_pool_opened": False,
        }
        _write_json(partial / "summary.json", summary)
        (partial / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output)
        return summary
    except Exception:
        if partial.exists():
            (partial / "FAILED").write_text("FAILED\n", encoding="utf-8")
        raise

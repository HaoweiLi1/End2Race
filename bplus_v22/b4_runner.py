"""Resumable two-seed B4 direct-head learner and blocking smoke contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from bplus_v22.b4_direct import (
    B4_CURRICULUM_SCHEMA,
    B4_POLICY_SCHEMA,
    B4Batch,
    B4Curriculum,
    B4DirectHeadPolicy,
    B4ScenarioSets,
    FROZEN_B4_CONFIG,
    actor_snapshot_sha256,
    build_batch,
    build_optimizers,
    load_full_checkpoint,
    load_strict_plain_actor,
    projection_metrics,
    replay_metrics,
    save_actor_snapshot,
    save_full_checkpoint,
    strict_plain_actor_from_state,
    update_policy,
    validate_frozen_config,
)
from bplus_v22.b4_env import B4EpisodeResult, run_b4_episode
from bplus_v22.ppo_env import load_b2_scenario_sets


B4_RUN_PLAN_SCHEMA = "end2race-b2-run-plan-1"
B4_TRAIN_KIND = "b4_train"
B4_JOB_KIND = "b4_training"
B4_PILOT_SCHEMA = "end2race-b4-direct-head-pilot-1"
B4_PLUMBING_SCHEMA = "end2race-b4-plumbing-smoke-1"
B4_READY_SCHEMA = "end2race-b4-ready-1"
B4_SEED_BASE = 407130
B4_REPLAY_RATIO_ATOL = 1e-4
TRAINING_MANIFEST_NAME = "training_scenarios.tsv"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def expected_b4_plan_config(
    task8_release: str | Path,
    d2_episode_metadata: str | Path,
) -> dict[str, Any]:
    """Build the immutable numerical config and both frozen curriculum digests."""

    task8 = Path(task8_release)
    metadata = Path(d2_episode_metadata)
    scenarios = B4ScenarioSets.from_b2(load_b2_scenario_sets(task8, metadata))
    return {
        "policy_contract": B4_POLICY_SCHEMA,
        "ppo": FROZEN_B4_CONFIG.as_dict(),
        "curriculum_schema": B4_CURRICULUM_SCHEMA,
        "curriculum_sha256_by_seed": {
            str(seed): B4Curriculum(scenarios, seed).digest(FROZEN_B4_CONFIG.iterations)
            for seed in FROZEN_B4_CONFIG.seeds
        },
        "training_manifest_sha256": _file_sha256(task8 / TRAINING_MANIFEST_NAME),
        "bc_baseline_expected_collision": 24,
        "bc_baseline_expected_overtake": 138,
        "bc_baseline_topology": {
            "0": {"host_id": "local", "collision": 12, "terminal_overtake": 32},
            "1": {"host_id": "remote", "collision": 2, "terminal_overtake": 37},
            "2": {"host_id": "remote", "collision": 5, "terminal_overtake": 33},
            "3": {"host_id": "remote", "collision": 5, "terminal_overtake": 36},
        },
        "overtake_gate_per_seed": 132,
        "collision_feasibility_per_seed": 24,
        "collision_product_target_per_seed": 16,
        "collision_product_target_pooled": 33,
        "deterministic_speed_projection_required": 0,
        "inputs": {
            "bc_checkpoint": "repo/pretrained/end2race.pth",
            "task8_release": "inputs/task8",
            "training_manifest": "inputs/task8/training_scenarios.tsv",
            "development_manifest": "inputs/task8/development_scenarios.tsv",
            "d2_episode_metadata": "inputs/d2/episode_metadata.tsv",
        },
        "forbidden_inputs": ["sidecar", "D2_test", "fresh_pool", "final_pool", "eval_results"],
    }


def load_b4_run_plan(path: str | Path) -> tuple[Path, dict[str, Any]]:
    plan_path = Path(path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != B4_RUN_PLAN_SCHEMA or plan.get("kind") != B4_TRAIN_KIND:
        raise ValueError("B4 learner requires one b4_train RunPlan")
    observed = plan.get("plan_sha256")
    if not isinstance(observed, str) or not _is_sha256(observed):
        raise ValueError("B4 RunPlan digest is invalid")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    expected = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if observed != expected:
        raise ValueError("B4 RunPlan digest mismatch")
    return plan_path, plan


def _staged_paths(plan_path: Path, plan: Mapping[str, Any]) -> dict[str, Path]:
    root = plan_path.parent.parent
    paths = {
        "root": root,
        "repo": root / "repo",
        "bc": root / "repo/pretrained/end2race.pth",
        "task8": root / "inputs/task8",
        "metadata": root / "inputs/d2/episode_metadata.tsv",
    }
    for name, path in paths.items():
        if name != "root" and not path.exists():
            raise ValueError(f"B4 staged input is missing: {name}={path}")
    if Path.cwd().resolve() != paths["repo"].resolve():
        raise ValueError("B4 learner must execute from the staged repository root")
    config = plan.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("B4 RunPlan config is missing")
    observed = expected_b4_plan_config(paths["task8"], paths["metadata"])
    if dict(config) != observed:
        raise ValueError("B4 staged config/curriculum digest drift")
    validate_frozen_config(config["ppo"])
    return paths


def validate_b4_pilot_plan(
    plan_path: str | Path,
    job_id: str | None = None,
    *,
    allow_partial_resume: bool = False,
) -> dict[str, Any]:
    path, plan = load_b4_run_plan(plan_path)
    paths = _staged_paths(path, plan)
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 2:
        raise ValueError("B4 RunPlan must contain exactly two seed jobs")
    identities = set()
    selected = None
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ValueError("B4 RunPlan job is not an object")
        seed = job.get("seed")
        identity = (job.get("job_id"), seed)
        if (
            job.get("kind") != B4_JOB_KIND
            or type(seed) is not int
            or seed not in FROZEN_B4_CONFIG.seeds
            or not isinstance(job.get("output_relpath"), str)
        ):
            raise ValueError("B4 RunPlan job contract mismatch")
        identities.add(identity)
        if job_id is not None and job.get("job_id") == job_id:
            selected = dict(job)
    expected = {(f"b4-seed{seed}", seed) for seed in FROZEN_B4_CONFIG.seeds}
    if identities != expected:
        raise ValueError("B4 RunPlan seed job inventory drift")
    if job_id is not None and selected is None:
        raise ValueError(f"B4 RunPlan does not contain job {job_id!r}")
    if selected is not None:
        output = paths["root"] / selected["output_relpath"]
        partial = output.with_name(output.name + ".partial")
        if allow_partial_resume:
            if output.exists() or not partial.is_dir():
                raise ValueError("B4 resume requires one incomplete partial release")
        elif output.exists() or partial.exists():
            raise ValueError("B4 fresh job output already exists")
    return {"plan": plan, "paths": paths, "job": selected}


def _validate_control_plane_ready(
    plan: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    control = paths["root"] / "control"
    baseline_path = control / "bc_baseline_preflight.json"
    plumbing_path = control / "plumbing_smoke.json"
    ready_path = control / "READY.json"
    for path in (baseline_path, plumbing_path, ready_path):
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError(f"B4 learner lacks safe control marker: {path.name}")
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
        raise ValueError("B4 learner baseline authorization mismatch")
    if (
        plumbing.get("schema") != B4_PLUMBING_SCHEMA
        or plumbing.get("passed") is not True
        or plumbing.get("run_plan_sha256") != plan.get("plan_sha256")
        or plumbing.get("product_outcomes_reported_or_compared") is not False
        or plumbing.get("candidate_selection_performed") is not False
        or plumbing.get("ppo_pilot_iteration_completed") is not False
    ):
        raise ValueError("B4 learner plumbing authorization mismatch")
    expected_ready = {
        "schema": B4_READY_SCHEMA,
        "passed": True,
        "run_plan_sha256": plan.get("plan_sha256"),
        "source_commit": plan.get("source_commit"),
        "source_archive_sha256": plan.get("source_archive_sha256"),
        "inputs_archive_sha256": plan.get("inputs_archive_sha256"),
        "baseline_marker_sha256": _file_sha256(baseline_path),
        "plumbing_marker_sha256": _file_sha256(plumbing_path),
    }
    if ready != expected_ready:
        raise ValueError("B4 learner READY authorization mismatch")
    return ready


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_iteration_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("B4 iteration ledger has an incomplete final line")
    rows = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("iteration") != line_number:
            raise ValueError("B4 iteration ledger is non-contiguous")
        rows.append(value)
    if len(rows) > FROZEN_B4_CONFIG.iterations:
        raise ValueError("B4 iteration ledger exceeds the frozen schedule")
    return rows


def _repair_torn_ledger(partial: Path) -> None:
    path = partial / "iterations.jsonl"
    if not path.is_file():
        return
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return
    boundary = raw.rfind(b"\n")
    prefix = b"" if boundary < 0 else raw[: boundary + 1]
    attempts = partial / "attempt_failures"
    attempts.mkdir(exist_ok=True)
    target = attempts / f"torn_iterations_{len(list(attempts.glob('torn_iterations_*.jsonl'))) + 1:03d}.jsonl"
    os.replace(path, target)
    with path.open("xb") as handle:
        handle.write(prefix)
        handle.flush()
        os.fsync(handle.fileno())


def _quarantine_uncommitted(partial: Path, committed_iteration: int) -> None:
    extras: list[Path] = []
    for directory, pattern in (
        (partial / "checkpoints", "iter_*.pt"),
        (partial / "actors", "iter_*.pth"),
        (partial / "replay", "iter_*.npz"),
    ):
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            try:
                index = int(path.stem.split("_")[-1])
            except ValueError as error:
                raise ValueError(f"B4 malformed resume artifact: {path}") from error
            if index > committed_iteration:
                extras.append(path)
    failed = partial / "FAILED"
    if failed.exists():
        extras.append(failed)
    if not extras:
        return
    attempts = partial / "attempt_failures"
    attempts.mkdir(exist_ok=True)
    index = 1
    while (attempts / f"attempt_{index:03d}").exists():
        index += 1
    target = attempts / f"attempt_{index:03d}"
    target.mkdir()
    for source in extras:
        os.replace(source, target / source.relative_to(partial).as_posix().replace("/", "__"))


def _curriculum_record(curriculum_plan) -> dict[str, Any]:
    return {
        "schema": B4_CURRICULUM_SCHEMA,
        "rows": [
            {
                "iteration": iteration,
                "episode_index": episode_index,
                "global_episode_id": (iteration - 1) * 16 + episode_index,
                "l2_id": scenario.l2_id,
                "l4_id": scenario.l4_id,
                "map_name": scenario.map_name,
                "archived_bc_outcome": scenario.archived_bc_outcome,
            }
            for iteration, rows in enumerate(curriculum_plan, start=1)
            for episode_index, scenario in enumerate(rows)
        ],
    }


def _write_replay(path: Path, batch: B4Batch) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    payload = {
        name: getattr(batch, name).detach().cpu().numpy()
        for name in (
            "feature",
            "privileged_feature",
            "raw_action",
            "executed_action",
            "projection_delta",
            "old_log_prob",
            "old_value",
            "reward",
            "terminated",
            "episode_id",
            "step_index",
            "advantage",
            "normalized_advantage",
            "returns",
            "actor_weight",
        )
    }
    payload["l2_ids"] = np.asarray(batch.l2_ids, dtype="U67")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _file_sha256(path)


def _outcome_record(result: B4EpisodeResult) -> dict[str, Any]:
    candidate_collision = bool(result.outcome.collision_any)
    candidate_overtake = result.outcome.corrected_outcome3 == "overtake"
    bc_collision = result.scenario.archived_bc_outcome == "collision"
    bc_overtake = result.scenario.archived_bc_outcome == "overtake"
    return {
        "l2_id": result.scenario.l2_id,
        "archived_bc_outcome": result.scenario.archived_bc_outcome,
        "terminal_reason": result.terminal_reason,
        "step_count": result.step_count,
        "collision_any": candidate_collision,
        "terminal_overtake": candidate_overtake,
        "confirmed_safe_pass": result.outcome.confirmed_safe_pass is True,
        "interaction_attempt": result.outcome.interaction_attempt is True,
        "terminal_reward": result.transitions[-1].reward,
        "paired_delta_reward_diagnostic": float(
            2 * (int(bc_collision) - int(candidate_collision))
            + (int(candidate_overtake) - int(bc_overtake))
        ),
        "fixed_collision": bc_collision and not candidate_collision,
        "new_collision": not bc_collision and candidate_collision,
        "gained_overtake": not bc_overtake and candidate_overtake,
        "lost_overtake": bc_overtake and not candidate_overtake,
        "projection_transition_count": result.projection_transition_count,
        "steer_projection_count": result.steer_projection_count,
        "speed_projection_count": result.speed_projection_count,
        "max_abs_steer_projection_delta": result.max_abs_steer_projection_delta,
        "max_abs_speed_projection_delta": result.max_abs_speed_projection_delta,
    }


def run_b4_plumbing_smoke(
    plan_path: str | Path,
    *,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    """Run one deterministic identity episode on each product map."""

    validated = validate_b4_pilot_plan(plan_path)
    from d25.oracle import simulate_episode

    plan = validated["plan"]
    paths = validated["paths"]
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("B4 plumbing smoke requested unavailable CUDA")
    bc_state = torch.load(paths["bc"], map_location="cpu", weights_only=True)
    policy = B4DirectHeadPolicy(bc_state).to(device)
    scenarios = B4ScenarioSets.from_b2(
        load_b2_scenario_sets(paths["task8"], paths["metadata"])
    )
    selected_by_map: dict[str, Any] = {}
    for scenario in sorted(
        (*scenarios.collision, *scenarios.overtake, *scenarios.follow),
        key=lambda row: row.training_order,
    ):
        selected_by_map.setdefault(scenario.map_name, scenario)
    expected_maps = ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")
    if tuple(sorted(selected_by_map)) != tuple(sorted(expected_maps)):
        raise AssertionError("B4 plumbing smoke map inventory drift")
    reports = []
    for episode_id, map_name in enumerate(expected_maps):
        scenario = selected_by_map[map_name]
        reference = simulate_episode(policy.actor, device, scenario.simulator_case())
        result = run_b4_episode(
            policy,
            device,
            scenario,
            episode_id=episode_id,
            deterministic=True,
        )
        mismatches = []
        for name in sorted(set(reference.arrays) | set(result.arrays)):
            if name not in reference.arrays or name not in result.arrays:
                mismatches.append(f"missing:{name}")
            elif not np.array_equal(
                np.asarray(reference.arrays[name]), np.asarray(result.arrays[name])
            ):
                mismatches.append(name)
        features = torch.from_numpy(
            np.stack([row.feature for row in result.transitions])
        ).to(device)
        raw = torch.from_numpy(
            np.stack([row.raw_action for row in result.transitions])
        ).to(device)
        old = torch.tensor(
            [row.old_log_prob for row in result.transitions],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            replayed = policy.log_prob(policy.mean_from_feature(features), raw)
        max_log_prob_delta = float(torch.max(torch.abs(replayed - old)).item())
        outcome_identity = (
            reference.outcome.four_state == result.outcome.four_state
            and bool(reference.outcome.collision_any) == bool(result.outcome.collision_any)
            and reference.outcome.corrected_outcome3 == result.outcome.corrected_outcome3
        )
        if (
            mismatches
            or not outcome_identity
            or result.speed_projection_count != 0
            or max_log_prob_delta > B4_REPLAY_RATIO_ATOL
        ):
            raise AssertionError(f"B4 plumbing identity failed on {map_name}: {mismatches}")
        reports.append(
            {
                "map_name": map_name,
                "l2_id": scenario.l2_id,
                "step_count": result.step_count,
                "terminal_reason": result.terminal_reason,
                "trajectory_identity": True,
                "outcome_identity": True,
                "speed_projection_count": result.speed_projection_count,
                "steer_projection_count": result.steer_projection_count,
                "max_abs_replayed_log_prob_delta": max_log_prob_delta,
            }
        )
    policy.assert_frozen_exact()
    return {
        "schema": B4_PLUMBING_SCHEMA,
        "passed": True,
        "run_plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source_commit"],
        "training_manifest_sha256": _file_sha256(
            paths["task8"] / TRAINING_MANIFEST_NAME
        ),
        "bc_checkpoint_sha256": _file_sha256(paths["bc"]),
        "d2_episode_metadata_sha256": _file_sha256(paths["metadata"]),
        "scenario_selection": "first_physical_training_row_per_map_outcome_blind",
        "map_reports": reports,
        "plain_actor_key_count": len(policy.actor_state()),
        "trainable_actor_parameter_count": sum(
            parameter.numel() for parameter in policy.trainable_actor_parameters
        ),
        "product_outcomes_reported_or_compared": False,
        "candidate_selection_performed": False,
        "ppo_pilot_iteration_completed": False,
    }


def _validate_resume_prefix(
    partial: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_bc_actor_tensor_sha256: str,
) -> None:
    actor0 = partial / "actors/iter_0000.pth"
    checkpoint0 = partial / "checkpoints/iter_0000.pt"
    if not actor0.is_file() or actor0.is_symlink() or not checkpoint0.is_file():
        raise ValueError("B4 iteration-0 resume artifacts are incomplete")
    actor0_state = load_strict_plain_actor(actor0, "cpu").state_dict()
    if actor_snapshot_sha256(actor0_state) != expected_bc_actor_tensor_sha256:
        raise ValueError("B4 iteration-0 actor is not the canonical BC state")
    for iteration, row in enumerate(rows, start=1):
        replay = partial / f"replay/iter_{iteration:04d}.npz"
        checkpoint = partial / f"checkpoints/iter_{iteration:04d}.pt"
        if (
            not replay.is_file()
            or _file_sha256(replay) != row.get("replay_sha256")
            or not checkpoint.is_file()
            or _file_sha256(checkpoint) != row.get("full_checkpoint_sha256")
        ):
            raise ValueError("B4 committed resume prefix hash mismatch")
        if iteration in FROZEN_B4_CONFIG.snapshots:
            actor = partial / f"actors/iter_{iteration:04d}.pth"
            if not actor.is_file() or _file_sha256(actor) != row.get("actor_snapshot_file_sha256"):
                raise ValueError("B4 committed actor snapshot hash mismatch")


def run_b4_pilot_job(
    plan_path: str | Path,
    job_id: str,
    *,
    device_name: str = "cuda:0",
    resume: bool = False,
) -> dict[str, Any]:
    validated = validate_b4_pilot_plan(
        plan_path, job_id, allow_partial_resume=bool(resume)
    )
    plan = validated["plan"]
    paths = validated["paths"]
    job = validated["job"]
    _validate_control_plane_ready(plan, paths)
    seed = int(job["seed"])
    output = paths["root"] / str(job["output_relpath"])
    partial = output.with_name(output.name + ".partial")
    if not resume:
        partial.mkdir(parents=True)
    try:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("B4 learner requested unavailable CUDA")
        base_seed = B4_SEED_BASE + seed
        torch.manual_seed(base_seed)
        np.random.seed(base_seed)
        random.seed(base_seed)
        bc_state = torch.load(paths["bc"], map_location="cpu", weights_only=True)
        strict_plain_actor_from_state(bc_state)
        policy = B4DirectHeadPolicy(bc_state).to(device)
        actor_optimizer, critic_optimizer = build_optimizers(policy)
        scenarios = B4ScenarioSets.from_b2(
            load_b2_scenario_sets(paths["task8"], paths["metadata"])
        )
        curriculum = B4Curriculum(scenarios, seed)
        curriculum_plan = curriculum.plan(FROZEN_B4_CONFIG.iterations)
        curriculum_sha = curriculum.digest(FROZEN_B4_CONFIG.iterations)
        if curriculum_sha != plan["config"]["curriculum_sha256_by_seed"][str(seed)]:
            raise ValueError("B4 learner curriculum digest differs from RunPlan")
        training_manifest_sha = _file_sha256(
            paths["task8"] / TRAINING_MANIFEST_NAME
        )
        config_record = {
            "schema": B4_PILOT_SCHEMA,
            "seed": seed,
            "run_plan_sha256": plan["plan_sha256"],
            "source_commit": plan["source_commit"],
            "bc_checkpoint_sha256": _file_sha256(paths["bc"]),
            "bc_actor_tensor_sha256": actor_snapshot_sha256(policy.actor_state()),
            "training_manifest_sha256": training_manifest_sha,
            "curriculum_sha256": curriculum_sha,
            "config": FROZEN_B4_CONFIG.as_dict(),
        }
        curriculum_record = _curriculum_record(curriculum_plan)
        if resume:
            if json.loads((partial / "config.json").read_text(encoding="utf-8")) != config_record:
                raise ValueError("B4 resume config prefix drift")
            if json.loads((partial / "curriculum.json").read_text(encoding="utf-8")) != curriculum_record:
                raise ValueError("B4 resume curriculum prefix drift")
            _repair_torn_ledger(partial)
            ledger = _read_iteration_ledger(partial / "iterations.jsonl")
            committed = len(ledger)
            _quarantine_uncommitted(partial, committed)
            _validate_resume_prefix(
                partial,
                ledger,
                expected_bc_actor_tensor_sha256=config_record[
                    "bc_actor_tensor_sha256"
                ],
            )
            checkpoint = partial / f"checkpoints/iter_{committed:04d}.pt"
            loaded_iteration = load_full_checkpoint(
                checkpoint,
                policy,
                actor_optimizer,
                critic_optimizer,
                expected_seed=seed,
                expected_run_plan_sha256=plan["plan_sha256"],
                expected_curriculum_sha256=curriculum_sha,
            )
            if loaded_iteration != committed:
                raise ValueError("B4 resume checkpoint/ledger iteration mismatch")
            start_iteration = committed + 1
            resumed_from_iteration = committed
        else:
            _write_json(partial / "config.json", config_record)
            _write_json(partial / "curriculum.json", curriculum_record)
            actor0 = partial / "actors/iter_0000.pth"
            actor0_record = save_actor_snapshot(policy, actor0)
            if actor0_record["tensor_sha256"] != config_record["bc_actor_tensor_sha256"]:
                raise AssertionError("B4 iteration-0 actor differs from canonical BC")
            save_full_checkpoint(
                policy,
                actor_optimizer,
                critic_optimizer,
                partial / "checkpoints/iter_0000.pt",
                completed_iteration=0,
                seed=seed,
                run_plan_sha256=plan["plan_sha256"],
                curriculum_sha256=curriculum_sha,
            )
            start_iteration = 1
            resumed_from_iteration = None

        for iteration in range(start_iteration, FROZEN_B4_CONFIG.iterations + 1):
            episode_results: list[B4EpisodeResult] = []
            transitions = []
            for episode_index, scenario in enumerate(curriculum_plan[iteration - 1]):
                episode_id = (iteration - 1) * FROZEN_B4_CONFIG.episodes_per_iteration + episode_index
                result = run_b4_episode(
                    policy,
                    device,
                    scenario,
                    episode_id=episode_id,
                    deterministic=False,
                )
                episode_results.append(result)
                transitions.extend(result.transitions)
            batch = build_batch(transitions, FROZEN_B4_CONFIG)
            preupdate = replay_metrics(policy, batch.to(device))
            if preupdate["max_abs_ratio_minus_one"] > B4_REPLAY_RATIO_ATOL:
                raise AssertionError("B4 pre-update raw-latent replay ratio is not one")
            projection = projection_metrics(batch)
            replay_path = partial / f"replay/iter_{iteration:04d}.npz"
            replay_sha = _write_replay(replay_path, batch)
            update = update_policy(
                policy,
                batch,
                actor_optimizer,
                critic_optimizer,
                seed=seed,
                iteration=iteration,
            )
            actor_file_sha = None
            actor_tensor_sha = None
            if iteration in FROZEN_B4_CONFIG.snapshots:
                actor_path = partial / f"actors/iter_{iteration:04d}.pth"
                actor_record = save_actor_snapshot(policy, actor_path)
                actor_file_sha = _file_sha256(actor_path)
                actor_tensor_sha = actor_record["tensor_sha256"]
            full_path = partial / f"checkpoints/iter_{iteration:04d}.pt"
            save_full_checkpoint(
                policy,
                actor_optimizer,
                critic_optimizer,
                full_path,
                completed_iteration=iteration,
                seed=seed,
                run_plan_sha256=plan["plan_sha256"],
                curriculum_sha256=curriculum_sha,
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
            "schema": B4_PILOT_SCHEMA,
            "passed": True,
            "integrity_passed": True,
            "seed": seed,
            "iterations": FROZEN_B4_CONFIG.iterations,
            "resumed_from_iteration": resumed_from_iteration,
            "run_plan_sha256": plan["plan_sha256"],
            "source_commit": plan["source_commit"],
            "bc_checkpoint_sha256": config_record["bc_checkpoint_sha256"],
            "bc_actor_tensor_sha256": config_record["bc_actor_tensor_sha256"],
            "training_manifest_sha256": training_manifest_sha,
            "curriculum_sha256": curriculum_sha,
            "actor_snapshot_file_sha256_by_iteration": snapshot_files,
            "actor_snapshot_tensor_sha256_by_iteration": snapshot_tensors,
            "final_full_checkpoint_sha256": _file_sha256(
                partial / f"checkpoints/iter_{FROZEN_B4_CONFIG.iterations:04d}.pt"
            ),
            "product_kpi_evaluated": False,
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

#!/usr/bin/env python3
"""Plan, smoke, and execute the B7 plain recurrent PPO engineering run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch

from bplus_v22.b4_direct import load_strict_plain_actor
from bplus_v22.b4_direct import actor_snapshot_sha256
from bplus_v22.b7_env import run_b7_episode
from bplus_v22.b7_recurrent import (
    B7Episode,
    B7RecurrentPolicy,
    B7ScenarioSampler,
    FROZEN_B7_CONFIG,
    actor_tensor_digest,
    build_batch,
    build_optimizers,
    preupdate_replay_metrics,
    update_policy,
)
from bplus_v22.ppo_env import load_b2_scenario_sets


PLAN_SCHEMA = "end2race-b7-run-plan-1"
RESULT_SCHEMA = "end2race-b7-plain-recurrent-result-1"
CHECKPOINT_SCHEMA = "end2race-b7-full-checkpoint-1"
SEED_BASE = 5071700


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(path if path.exists() else temporary)
    with temporary.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_actor(path: Path, policy: B7RecurrentPolicy) -> dict[str, Any]:
    state = policy.actor_state()
    _atomic_torch(path, state)
    loaded = load_strict_plain_actor(path)
    if any(not torch.equal(loaded.state_dict()[name], state[name]) for name in state):
        raise AssertionError("B7 actor-only strict-load roundtrip failed")
    return {
        "file_sha256": sha256(path),
        "tensor_sha256": actor_tensor_digest(policy),
        "key_count": len(state),
    }


def make_plan(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    inputs = {
        "bc": args.bc,
        "task8": args.task8,
        "metadata": args.metadata,
    }
    resolved = {name: (repo / value).resolve() for name, value in inputs.items()}
    if not resolved["bc"].is_file() or not resolved["metadata"].is_file():
        raise ValueError("B7 plan input file is missing")
    if not (resolved["task8"] / "COMPLETE").is_file():
        raise ValueError("B7 Task-8 input release is incomplete")
    source_commit = str(args.source_commit)
    source_archive_sha256 = str(args.source_archive_sha256)
    if len(source_commit) != 40 or len(source_archive_sha256) != 64:
        raise ValueError("B7 source identity is malformed")
    value: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "kind": "b7_plain_recurrent_train",
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "created_at": args.created_at,
        "primary_host": "remote",
        "primary_seed": 1,
        "conditional_replication_seed": 0,
        "config": FROZEN_B7_CONFIG.as_dict(),
        "inputs": {
            "bc_relpath": inputs["bc"],
            "bc_sha256": sha256(resolved["bc"]),
            "task8_relpath": inputs["task8"],
            "training_manifest_sha256": sha256(
                resolved["task8"] / "training_scenarios.tsv"
            ),
            "development_manifest_sha256": sha256(
                resolved["task8"] / "development_scenarios.tsv"
            ),
            "metadata_relpath": inputs["metadata"],
            "metadata_sha256": sha256(resolved["metadata"]),
        },
        "execution": {
            "remote_display": ":1",
            "device": "cuda:0",
            "threads_per_process": 1,
            "learner_count": 1,
            "seed1_only_until_288_gate": True,
        },
        "evaluation": {
            "panel": "opened-development 288",
            "bc_collision": 24,
            "bc_overtake": 138,
            "candidate_iteration": 10,
            "candidate_collision_max": 18,
            "candidate_overtake_min": 132,
            "fixed_minus_new_min": 6,
            "l4_cluster_signflip_one_sided_max": 0.10,
            "deterministic_speed_projection_count": 0,
            "opened_development_target_collision_max": 16,
        },
        "automatic_followup": {
            "seed0_only_if_seed1_passes": True,
            "austin600_only_if_both_seeds_pass": True,
            "sealed_pool_opened": False,
        },
    }
    value["plan_sha256"] = hashlib.sha256(_json_bytes(value)).hexdigest()
    _atomic_json(output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def load_plan(path: str | Path, repo: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    plan = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    observed = plan.get("plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    expected = hashlib.sha256(_json_bytes(unsigned)).hexdigest()
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("kind") != "b7_plain_recurrent_train"
        or observed != expected
        or plan.get("config") != FROZEN_B7_CONFIG.as_dict()
    ):
        raise ValueError("B7 RunPlan identity/config drift")
    root = Path(repo).resolve()
    inputs = plan["inputs"]
    paths = {
        "bc": (root / inputs["bc_relpath"]).resolve(),
        "task8": (root / inputs["task8_relpath"]).resolve(),
        "metadata": (root / inputs["metadata_relpath"]).resolve(),
    }
    checks = {
        "bc_sha256": sha256(paths["bc"]),
        "training_manifest_sha256": sha256(paths["task8"] / "training_scenarios.tsv"),
        "development_manifest_sha256": sha256(paths["task8"] / "development_scenarios.tsv"),
        "metadata_sha256": sha256(paths["metadata"]),
    }
    if any(inputs[name] != value for name, value in checks.items()):
        raise ValueError("B7 RunPlan input digest drift")
    return plan, paths


def _rng_state() -> dict[str, Any]:
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    return value


def _restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if "torch_cuda" in value:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def _save_checkpoint(
    path: Path,
    policy: B7RecurrentPolicy,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    iteration: int,
    seed: int,
    plan_sha256: str,
    consecutive_rejections: int,
    previous_outcomes: Mapping[str, str],
) -> None:
    _atomic_torch(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "iteration": int(iteration),
            "seed": int(seed),
            "plan_sha256": plan_sha256,
            "config": FROZEN_B7_CONFIG.as_dict(),
            "actor": policy.actor_state(),
            "critic": {
                name: value.detach().cpu().clone()
                for name, value in policy.critic.state_dict().items()
            },
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "consecutive_rejections": int(consecutive_rejections),
            "previous_outcomes": dict(previous_outcomes),
            "rng": _rng_state(),
        },
    )


def _load_checkpoint(
    path: Path,
    policy: B7RecurrentPolicy,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    plan_sha256: str,
) -> tuple[int, int, dict[str, str]]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        value.get("schema") != CHECKPOINT_SCHEMA
        or value.get("seed") != seed
        or value.get("plan_sha256") != plan_sha256
        or value.get("config") != FROZEN_B7_CONFIG.as_dict()
    ):
        raise ValueError("B7 full checkpoint identity drift")
    policy.actor.load_state_dict(value["actor"], strict=True)
    policy.critic.load_state_dict(value["critic"], strict=True)
    actor_optimizer.load_state_dict(value["actor_optimizer"])
    critic_optimizer.load_state_dict(value["critic_optimizer"])
    _restore_rng(value["rng"])
    policy.assert_frozen_exact()
    return (
        int(value["iteration"]),
        int(value["consecutive_rejections"]),
        dict(value["previous_outcomes"]),
    )


def _replay_payload(batch: B7Batch) -> dict[str, np.ndarray]:
    offsets = [0]
    for episode in batch.episodes:
        offsets.append(offsets[-1] + episode.length)
    payload: dict[str, np.ndarray] = {
        "offsets": np.asarray(offsets, dtype=np.int64),
        "l2_ids": np.asarray(
            [episode.episode.scenario.l2_id for episode in batch.episodes], dtype="U67"
        ),
        "archived_bc_outcomes": np.asarray(
            [episode.episode.scenario.archived_bc_outcome for episode in batch.episodes],
            dtype="U9",
        ),
        "candidate_outcomes": np.asarray(
            [episode.episode.corrected_outcome3 for episode in batch.episodes], dtype="U9"
        ),
        "sampler_roles": np.asarray(
            [episode.episode.sampler_role for episode in batch.episodes], dtype="U20"
        ),
        "hard_priorities": np.asarray(
            [episode.episode.hard_priority or 0 for episode in batch.episodes], dtype=np.int8
        ),
    }
    for name in (
        "lidar",
        "previous_speed",
        "privileged_feature",
        "old_mean",
        "bc_mean",
        "raw_action",
        "old_log_prob",
        "old_value",
        "reward",
        "advantage",
        "normalized_advantage",
        "returns",
    ):
        payload[name] = torch.cat([getattr(episode, name) for episode in batch.episodes]).numpy()
    return payload


def _save_replay(path: Path, batch: B7Batch) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(path if path.exists() else temporary)
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **_replay_payload(batch))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256(path)


def _outcome_record(episode: B7Episode) -> dict[str, Any]:
    archived = episode.scenario.archived_bc_outcome
    candidate = episode.corrected_outcome3
    return {
        "l2_id": episode.scenario.l2_id,
        "l4_id": episode.scenario.l4_id,
        "map_name": episode.scenario.map_name,
        "sampler_role": episode.sampler_role,
        "hard_priority": episode.hard_priority,
        "archived_bc_outcome": archived,
        "candidate_outcome": candidate,
        "step_count": len(episode.transitions),
        "terminal_reason": episode.terminal_reason,
        "fixed_collision": archived == "collision" and candidate != "collision",
        "new_collision": archived != "collision" and candidate == "collision",
        "gained_overtake": archived != "overtake" and candidate == "overtake",
        "lost_overtake": archived == "overtake" and candidate != "overtake",
        "steer_projection_count": sum(
            bool(row.projection_delta[0] != 0.0) for row in episode.transitions
        ),
        "speed_projection_count": sum(
            bool(row.projection_delta[1] != 0.0) for row in episode.transitions
        ),
    }


def run_smoke(args: argparse.Namespace) -> int:
    plan, paths = load_plan(args.plan, args.repo)
    device = torch.device(args.device)
    bc_state = torch.load(paths["bc"], map_location="cpu", weights_only=True)
    policy = B7RecurrentPolicy(bc_state).to(device)
    scenario_sets = load_b2_scenario_sets(paths["task8"], paths["metadata"])
    sampler = B7ScenarioSampler(scenario_sets.training, 1)
    selections = sampler.select(1, None)
    chosen = []
    seen_maps = set()
    for selection in selections:
        if selection.scenario.map_name not in seen_maps:
            chosen.append(selection)
            seen_maps.add(selection.scenario.map_name)
    reports = []
    for episode_id, selection in enumerate(chosen):
        episode, result = run_b7_episode(
            policy, device, selection, episode_id=episode_id, deterministic=True
        )
        # Framewise replay must exactly reproduce the deterministic rollout
        # means; no actor update or KPI comparison is performed in this smoke.
        with torch.no_grad():
            mean = policy.sequence_means(
                torch.from_numpy(np.stack([row.lidar for row in episode.transitions])).to(device),
                torch.tensor(
                    [row.previous_speed for row in episode.transitions],
                    dtype=torch.float32,
                    device=device,
                ),
            )
        old = torch.from_numpy(np.stack([row.old_mean for row in episode.transitions])).to(device)
        bc = torch.from_numpy(np.stack([row.bc_mean for row in episode.transitions])).to(device)
        max_old = float(torch.max(torch.abs(mean - old)).item())
        max_bc = float(torch.max(torch.abs(old - bc)).item())
        if max_old > 1e-5 or max_bc > 1e-5:
            raise AssertionError("B7 deterministic recurrent replay/BC identity drift")
        reports.append(
            {
                "map_name": selection.scenario.map_name,
                "l2_id": selection.scenario.l2_id,
                "steps": len(episode.transitions),
                "outcome": episode.corrected_outcome3,
                "max_abs_old_replay_mean_delta": max_old,
                "max_abs_bc_mean_delta": max_bc,
                "trajectory_array_count": len(result.arrays),
            }
        )
    output = {
        "schema": "end2race-b7-production-smoke-1",
        "passed": True,
        "plan_sha256": plan["plan_sha256"],
        "map_reports": reports,
        "candidate_kpi_compared": False,
        "actor_update_performed": False,
    }
    if args.output:
        _atomic_json(Path(args.output).resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def run_train(args: argparse.Namespace) -> int:
    plan, paths = load_plan(args.plan, args.repo)
    seed = int(args.seed)
    if seed not in {0, 1}:
        raise ValueError("B7 permits primary seed1 or conditionally authorized seed0")
    if seed == 0:
        if not args.seed0_authorization:
            raise ValueError("B7 seed0 requires a passed seed1 288 authorization artifact")
        authorization = json.loads(
            Path(args.seed0_authorization).resolve().read_text(encoding="utf-8")
        )
        if (
            authorization.get("schema") != "end2race-b7-eval-merge-1"
            or authorization.get("seed1_minimum_continue_gate_pass") is not True
            or authorization.get("training_run_plan_sha256") != plan["plan_sha256"]
        ):
            raise ValueError("B7 seed0 authorization artifact is invalid")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("B7 requested unavailable CUDA")
    base_seed = SEED_BASE + seed
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    random.seed(base_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(base_seed)
    bc_state = torch.load(paths["bc"], map_location="cpu", weights_only=True)
    policy = B7RecurrentPolicy(bc_state).to(device)
    actor_optimizer, critic_optimizer = build_optimizers(policy)
    scenario_sets = load_b2_scenario_sets(paths["task8"], paths["metadata"])
    sampler = B7ScenarioSampler(scenario_sets.training, seed)

    output = Path(args.output).resolve()
    partial = output.with_name(output.name + ".partial")
    if args.resume:
        if output.exists() or not partial.is_dir():
            raise ValueError("B7 resume requires one incomplete partial release")
        ledger = [
            json.loads(line)
            for line in (partial / "iterations.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        completed, consecutive_rejections, previous_outcomes = _load_checkpoint(
            partial / f"checkpoints/iter_{len(ledger):04d}.pt",
            policy,
            actor_optimizer,
            critic_optimizer,
            seed=seed,
            plan_sha256=plan["plan_sha256"],
        )
        if completed != len(ledger):
            raise ValueError("B7 resume checkpoint/ledger mismatch")
        start = completed + 1
    else:
        if output.exists() or partial.exists():
            raise FileExistsError(output if output.exists() else partial)
        partial.mkdir(parents=True)
        _atomic_json(
            partial / "config.json",
            {
                "schema": RESULT_SCHEMA,
                "plan_sha256": plan["plan_sha256"],
                "source_commit": plan["source_commit"],
                "seed": seed,
                "config": FROZEN_B7_CONFIG.as_dict(),
                "initial_actor_tensor_sha256": actor_tensor_digest(policy),
            },
        )
        actor0 = _atomic_actor(partial / "actors/iter_0000.pth", policy)
        if actor0["tensor_sha256"] != actor_snapshot_sha256(bc_state):
            raise AssertionError("B7 iteration-0 actor tensor differs from canonical BC")
        previous_outcomes: dict[str, str] = {}
        consecutive_rejections = 0
        _save_checkpoint(
            partial / "checkpoints/iter_0000.pt",
            policy,
            actor_optimizer,
            critic_optimizer,
            iteration=0,
            seed=seed,
            plan_sha256=plan["plan_sha256"],
            consecutive_rejections=0,
            previous_outcomes=previous_outcomes,
        )
        start = 1

    stopped_early = False
    for iteration in range(start, FROZEN_B7_CONFIG.iterations + 1):
        selections = sampler.select(iteration, previous_outcomes or None)
        episodes: list[B7Episode] = []
        trajectory_reports = []
        for index, selection in enumerate(selections):
            episode_id = (iteration - 1) * FROZEN_B7_CONFIG.episodes_per_iteration + index
            episode, result = run_b7_episode(
                policy,
                device,
                selection,
                episode_id=episode_id,
                deterministic=False,
            )
            episodes.append(episode)
            trajectory_reports.append(
                {
                    **_outcome_record(episode),
                    "trajectory_keys": sorted(result.arrays),
                }
            )
        batch = build_batch(episodes)
        replay_path = partial / f"replay/iter_{iteration:04d}.npz"
        replay_sha = _save_replay(replay_path, batch)
        update = update_policy(
            policy,
            batch,
            actor_optimizer,
            critic_optimizer,
            seed=seed,
            iteration=iteration,
            consecutive_rejections=consecutive_rejections,
        )
        consecutive_rejections = int(update["consecutive_rejections_after"])
        previous_outcomes = {
            episode.scenario.l2_id: episode.corrected_outcome3 for episode in episodes
        }
        actor_record = None
        if iteration == FROZEN_B7_CONFIG.candidate_iteration:
            actor_record = _atomic_actor(
                partial / f"actors/iter_{iteration:04d}.pth", policy
            )
        checkpoint = partial / f"checkpoints/iter_{iteration:04d}.pt"
        _save_checkpoint(
            checkpoint,
            policy,
            actor_optimizer,
            critic_optimizer,
            iteration=iteration,
            seed=seed,
            plan_sha256=plan["plan_sha256"],
            consecutive_rejections=consecutive_rejections,
            previous_outcomes=previous_outcomes,
        )
        row = {
            "iteration": iteration,
            "episode_count": batch.episode_count,
            "transition_count": batch.total_steps,
            "selection": [
                {
                    "episode_index": index,
                    "l2_id": selection.scenario.l2_id,
                    "l4_id": selection.scenario.l4_id,
                    "map_name": selection.scenario.map_name,
                    "archived_bc_outcome": selection.scenario.archived_bc_outcome,
                    "role": selection.role,
                    "hard_priority": selection.hard_priority,
                }
                for index, selection in enumerate(selections)
            ],
            "outcomes": trajectory_reports,
            "replay_sha256": replay_sha,
            "update": update,
            "actor_snapshot": actor_record,
            "full_checkpoint_sha256": sha256(checkpoint),
        }
        _append_jsonl(partial / "iterations.jsonl", row)
        print(
            "B7_ITERATION "
            + json.dumps(
                {
                    "iteration": iteration,
                    "episodes": batch.episode_count,
                    "transitions": batch.total_steps,
                    "accepted": update["actor_update_accepted"],
                    "safe_kl": update["current_rollout_bc_safe_mean_kl"]["mean"],
                    "rollout_kl": update["old_policy_rollout_mean_kl"]["mean"],
                    "consecutive_rejections": consecutive_rejections,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if update["early_stop_required"]:
            stopped_early = True
            break

    completed_iterations = len(
        [line for line in (partial / "iterations.jsonl").read_text().splitlines() if line]
    )
    candidate_exists = (partial / "actors/iter_0010.pth").is_file()
    summary = {
        "schema": RESULT_SCHEMA,
        "integrity_passed": True,
        "seed": seed,
        "plan_sha256": plan["plan_sha256"],
        "completed_iterations": completed_iterations,
        "stopped_after_three_consecutive_actor_rejections": stopped_early,
        "candidate_iteration": 10 if candidate_exists else None,
        "candidate_actor_file_sha256": (
            sha256(partial / "actors/iter_0010.pth") if candidate_exists else None
        ),
        "candidate_actor_tensor_sha256": (
            actor_tensor_digest(policy) if candidate_exists else None
        ),
        "status": "COMPLETE_CANDIDATE_READY_FOR_288" if candidate_exists else "EARLY_STOP_NO_CANDIDATE",
        "evaluation_performed": False,
        "seed0_started": seed == 0,
        "sealed_pool_opened": False,
    }
    _atomic_json(partial / "summary.json", summary)
    (partial / "COMPLETE").write_text(sha256(partial / "summary.json") + "\n", encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--repo", default=".")
    plan.add_argument("--output", required=True)
    plan.add_argument("--source-commit", required=True)
    plan.add_argument("--source-archive-sha256", required=True)
    plan.add_argument("--created-at", required=True)
    plan.add_argument("--bc", default="pretrained/end2race.pth")
    plan.add_argument(
        "--task8",
        default="Experiments/B1_route_r2_scaffold/artifacts/task8_manifests_20260712_113241",
    )
    plan.add_argument(
        "--metadata",
        default="Experiments/A3_d2_representation/artifacts/non_test_full_20260711_175713/episode_metadata.tsv",
    )
    plan.set_defaults(func=make_plan)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--repo", default=".")
    smoke.add_argument("--plan", required=True)
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--output")
    smoke.set_defaults(func=run_smoke)

    train = sub.add_parser("train")
    train.add_argument("--repo", default=".")
    train.add_argument("--plan", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--seed0-authorization")
    train.set_defaults(func=run_train)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

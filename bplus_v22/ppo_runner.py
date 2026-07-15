"""B2 BC-direct PPO learner and strict checkpoint/resume contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from bplus_v22 import ARMS, SEED
from bplus_v22.exploration import (
    BehaviorExplorationBatch,
    BehaviorExplorationConfig,
    KEYED_ACTION_NOISE_SCHEMA,
)
from bplus_v22.macro import MacroSignals
from bplus_v22.objective import OvertakeDual
from bplus_v22.ppo import (
    B2Critics,
    RunningCollisionScale,
    b2_constrained_advantage,
    build_b2_optimizers,
    compute_b2_losses,
)
from bplus_v22.ppo_buffer import (
    EpisodeCompleteMacroBuffer,
    MacroReplayRecord,
)
from bplus_v22.ppo_env import (
    B2Curriculum,
    B2EpisodeResult,
    load_b2_scenario_sets,
    run_b2_episode,
)
from bplus_v22.remediated_model import (
    B3_CONDITIONAL_BRAKE_PRIOR_PROBABILITY,
    B3_INTERVENTION_PRIOR_PROBABILITY,
    HierarchicalResidualAction,
    HierarchicalResidualDistribution,
    RemediatedV22Policy,
    UnifiedV22Policy,
)
from bplus_v22.sidecar import load_sidecar_bundle
from d25.oracle import load_bc_model


RUN_PLAN_SCHEMA = "end2race-b2-run-plan-1"
PILOT_SCHEMA = "bplus-v2.2-b2-ppo-pilot-1"
CHECKPOINT_SCHEMA = "bplus-v2.2-b2-ppo-checkpoint-1"
B3_PILOT_SCHEMA = "bplus-v2.2-b3-ppo-pilot-1"
B3_CHECKPOINT_SCHEMA = "bplus-v2.2-b3-ppo-checkpoint-1"
B2_POLICY_CONTRACT = "centered_fresh_prior"
B3_POLICY_CONTRACT = "unified_standard_mode_v1"
ACTION_NOISE_SCHEMA = KEYED_ACTION_NOISE_SCHEMA
MINIBATCH_ORDER_SCHEMA = "end2race:bplus-v2.2:b2-minibatch:v1"
TRAINING_MANIFEST_NAME = "training_scenarios.tsv"
COLLISION_SCALE_DECAY = 0.99
FULL_TOP_OFFSET = 3.8027754227
FULL_BRAKE_OFFSET = 6.0
EXPLORATION_MULTIPLIERS = (1.0,) * 5 + (0.8, 0.6, 0.4, 0.2) + (0.0,) * 11
ITERATIONS = 20
B3_ITERATIONS = 40
EPISODES_PER_ITERATION = 16
PPO_EPOCHS = 3
MINIBATCH_SIZE = 128
CLIP_EPSILON = 0.05
ENTROPY_COEFFICIENT = 0.001
TARGET_KL = 0.03
MAX_GRAD_NORM = 0.5
CRITIC_LR = 5e-5
REPLAY_FLOAT32_ATOL = 1e-4


def _policy_contract(config: Mapping[str, Any]) -> str:
    value = config.get("policy_contract", B2_POLICY_CONTRACT)
    if value not in {B2_POLICY_CONTRACT, B3_POLICY_CONTRACT}:
        raise ValueError(f"unknown PPO policy contract: {value!r}")
    return str(value)


def _iterations(config: Mapping[str, Any]) -> int:
    return B3_ITERATIONS if _policy_contract(config) == B3_POLICY_CONTRACT else ITERATIONS


def _checkpoint_schema(config: Mapping[str, Any]) -> str:
    return (
        B3_CHECKPOINT_SCHEMA
        if _policy_contract(config) == B3_POLICY_CONTRACT
        else CHECKPOINT_SCHEMA
    )


def _pilot_schema(config: Mapping[str, Any]) -> str:
    return B3_PILOT_SCHEMA if _policy_contract(config) == B3_POLICY_CONTRACT else PILOT_SCHEMA


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be lowercase SHA256")
    return text


def load_run_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("schema") != RUN_PLAN_SCHEMA:
        raise ValueError("B2 run-plan schema mismatch")
    observed = _validate_digest(payload.get("plan_sha256", ""), "run-plan digest")
    unsigned = dict(payload)
    unsigned.pop("plan_sha256", None)
    expected = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if observed != expected:
        raise ValueError("B2 run-plan digest mismatch")
    if payload.get("kind") not in {"b2_train", "b3_train"}:
        raise ValueError("ppo-pilot requires a supported B2/B3 training plan")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("B2 run-plan config is missing")
    return payload


def _staged_paths(plan_path: Path, plan: Mapping[str, Any]) -> dict[str, Path]:
    root = plan_path.resolve().parent.parent
    paths = {
        "root": root,
        "repo": root / "repo",
        "bc": root / "repo/pretrained/end2race.pth",
        "sidecar": root / "inputs/sidecar",
        "task8": root / "inputs/task8",
        "metadata": root / "inputs/d2/episode_metadata.tsv",
    }
    for name, path in paths.items():
        if name == "root":
            continue
        if not path.exists():
            raise ValueError(f"B2 staged input is missing: {name}={path}")
    if Path.cwd().resolve() != paths["repo"].resolve():
        raise ValueError("B2 pilot must execute from staged repository root")
    return paths


def _validate_control_plane_ready(
    plan: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    """Make direct learner CLI invocation fail closed without runner authorization."""

    control = paths["root"] / "control"
    ready_path = control / "READY.json"
    baseline_path = control / "bc_baseline_preflight.json"
    plumbing_path = control / "plumbing_smoke.json"
    for path in (ready_path, baseline_path, plumbing_path):
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError(f"B2 learner lacks one safe control-plane marker: {path.name}")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    plumbing = json.loads(plumbing_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema",
        "passed",
        "run_plan_sha256",
        "source_commit",
        "source_archive_sha256",
        "inputs_archive_sha256",
        "baseline_marker_sha256",
        "plumbing_marker_sha256",
    }
    if (
        not isinstance(ready, dict)
        or set(ready) != expected_keys
        or ready.get("schema") != "end2race-b2-ready-1"
        or ready.get("passed") is not True
        or ready.get("run_plan_sha256") != plan.get("plan_sha256")
        or ready.get("source_commit") != plan.get("source_commit")
        or ready.get("source_archive_sha256") != plan.get("source_archive_sha256")
        or ready.get("inputs_archive_sha256") != plan.get("inputs_archive_sha256")
        or ready.get("baseline_marker_sha256") != _file_sha256(baseline_path)
        or ready.get("plumbing_marker_sha256") != _file_sha256(plumbing_path)
    ):
        raise ValueError("B2 learner READY authorization mismatch")
    if (
        not isinstance(baseline, dict)
        or baseline.get("schema") != "bplus-v2.2-b2-bc-baseline-preflight-2"
        or baseline.get("integrity_passed") is not True
        or baseline.get("passed") is not True
        or baseline.get("acceptance_passed") is not True
        or baseline.get("candidate_evaluated") is not False
        or type(baseline.get("scenario_count")) is not int
        or baseline.get("scenario_count") != 288
        or type(baseline.get("shard_count")) is not int
        or baseline.get("shard_count") != 4
        or type(baseline.get("collision")) is not int
        or baseline.get("collision") != 24
        or type(baseline.get("terminal_overtake")) is not int
        or baseline.get("terminal_overtake") != 138
        or baseline.get("collision_by_shard") != [12, 2, 5, 5]
        or baseline.get("terminal_overtake_by_shard") != [32, 37, 33, 36]
        or not isinstance(baseline.get("count_checks"), dict)
        or set(baseline["count_checks"])
        != {
            "collision_by_shard",
            "terminal_overtake_by_shard",
            "collision_total",
            "terminal_overtake_total",
        }
        or any(value is not True for value in baseline["count_checks"].values())
    ):
        raise ValueError("B2 learner baseline authorization mismatch")
    if (
        not isinstance(plumbing, dict)
        or plumbing.get("schema") != "bplus-v2.2-b2-plumbing-smoke-1"
        or plumbing.get("passed") is not True
        or plumbing.get("product_outcomes_reported_or_compared") is not False
        or plumbing.get("arm_selection_performed") is not False
        or plumbing.get("ppo_pilot_iteration_completed") is not False
    ):
        raise ValueError("B2 learner plumbing authorization mismatch")
    return ready


def _validate_config(config: Mapping[str, Any]) -> None:
    contract = _policy_contract(config)
    if contract == B3_POLICY_CONTRACT:
        expected = {
            "policy_contract": B3_POLICY_CONTRACT,
            "iterations": B3_ITERATIONS,
            "episodes_per_iteration": EPISODES_PER_ITERATION,
            "collision_episodes_per_iteration": 8,
            "remaining_episodes_per_iteration": 8,
            "ppo_epochs": PPO_EPOCHS,
            "minibatch_size": MINIBATCH_SIZE,
            "clip_eps": CLIP_EPSILON,
            "action_core_lr": 3e-5,
            "head_lr": 3e-4,
            "sidecar_lr": 3e-6,
            "critic_lr": CRITIC_LR,
            "entropy_coef": ENTROPY_COEFFICIENT,
            "max_grad_norm": MAX_GRAD_NORM,
            "target_kl": TARGET_KL,
            "replay_float32_atol": REPLAY_FLOAT32_ATOL,
            "collision_scale_decay": COLLISION_SCALE_DECAY,
            "deterministic_contract": "standard_mode_of_training_distribution",
            "dual_freeze_through_iteration": 0,
            "bc_baseline_expected_collision": 24,
            "bc_baseline_expected_overtake": 138,
        }
        for name, value in expected.items():
            if config.get(name) != value:
                raise ValueError(f"B3 locked config drift: {name}={config.get(name)!r}")
        exploration = config.get("exploration")
        if not isinstance(exploration, Mapping) or exploration != {
            "intervention_prior_probability": B3_INTERVENTION_PRIOR_PROBABILITY,
            "conditional_brake_prior_probability": B3_CONDITIONAL_BRAKE_PRIOR_PROBABILITY,
            "external_gate_offsets_forbidden": True,
            "steer_std_scale": 0.1,
            "brake_std_scale": 1.0,
        }:
            raise ValueError("B3 unified exploration contract drift")
        return
    expected = {
        "iterations": ITERATIONS,
        "episodes_per_iteration": EPISODES_PER_ITERATION,
        "collision_episodes_per_iteration": 8,
        "remaining_episodes_per_iteration": 8,
        "ppo_epochs": PPO_EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "clip_eps": CLIP_EPSILON,
        "action_core_lr": 3e-5,
        "head_lr": 3e-4,
        "sidecar_lr": 3e-6,
        "critic_lr": CRITIC_LR,
        "entropy_coef": ENTROPY_COEFFICIENT,
        "max_grad_norm": MAX_GRAD_NORM,
        "target_kl": TARGET_KL,
        "replay_float32_atol": REPLAY_FLOAT32_ATOL,
        "collision_scale_decay": COLLISION_SCALE_DECAY,
        "deterministic_contract": "centered_fresh_prior",
        "dual_freeze_through_iteration": 9,
        "bc_baseline_expected_collision": 24,
        "bc_baseline_expected_overtake": 138,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"B2 locked config drift: {name}={config.get(name)!r}")
    exploration = config.get("exploration")
    if not isinstance(exploration, Mapping):
        raise ValueError("B2 exploration config is missing")
    if (
        exploration.get("intervention_full_offset") != FULL_TOP_OFFSET
        or exploration.get("conditional_brake_full_offset") != FULL_BRAKE_OFFSET
        or tuple(exploration.get("multipliers", ())) != EXPLORATION_MULTIPLIERS
        or exploration.get("steer_std_scale") != 0.1
        or exploration.get("brake_std_scale") != 1.0
    ):
        raise ValueError("B2 exploration schedule drift")


def validate_pilot_plan(
    plan_path: str | Path,
    job_id: str | None = None,
    *,
    allow_partial_resume: bool = False,
) -> dict[str, Any]:
    path = Path(plan_path).resolve()
    plan = load_run_plan(path)
    _validate_config(plan["config"])
    paths = _staged_paths(path, plan)
    jobs = {str(item["job_id"]): item for item in plan.get("jobs", [])}
    if len(jobs) != 6:
        raise ValueError("B2 run plan must contain six learner jobs")
    identities = {(item.get("arm"), item.get("seed")) for item in jobs.values()}
    if identities != {(arm, seed) for arm in ARMS for seed in (0, 1)}:
        raise ValueError("B2 learner arm/seed inventory drift")
    if any(item.get("kind") != "learner" or item.get("shardable") for item in jobs.values()):
        raise ValueError("B2 learner job topology drift")
    selected = None
    if job_id is not None:
        if job_id not in jobs:
            raise ValueError(f"unknown B2 learner job: {job_id}")
        selected = jobs[job_id]
        output = paths["root"] / str(selected["output_relpath"])
        partial = output.with_name(output.name + ".partial")
        if output.exists():
            raise FileExistsError(f"B2 learner output exists: {output}")
        if allow_partial_resume:
            if not partial.is_dir():
                raise FileNotFoundError(f"B2 learner partial is absent: {partial}")
        elif partial.exists():
            raise FileExistsError(f"B2 learner partial exists: {partial}")
    return {"plan": plan, "paths": paths, "job": selected}


def exploration_for_iteration(iteration: int) -> BehaviorExplorationConfig:
    index = int(iteration)
    if index != iteration or not 1 <= index <= ITERATIONS:
        raise ValueError("B2 iteration is outside 1..20")
    multiplier = EXPLORATION_MULTIPLIERS[index - 1]
    return BehaviorExplorationConfig(
        intervention_logit_offset=FULL_TOP_OFFSET * multiplier,
        conditional_brake_logit_offset=FULL_BRAKE_OFFSET * multiplier,
        steer_std_scale=0.1,
        brake_std_scale=1.0,
        schedule_id=f"b2-fixed-exploration-iter{index:02d}-m{multiplier:.1f}",
    )


def behavior_for_config(
    iteration: int, config: Mapping[str, Any]
) -> BehaviorExplorationConfig:
    if _policy_contract(config) == B2_POLICY_CONTRACT:
        return exploration_for_iteration(iteration)
    index = int(iteration)
    if index != iteration or not 1 <= index <= B3_ITERATIONS:
        raise ValueError("B3 iteration is outside 1..40")
    return BehaviorExplorationConfig(
        intervention_logit_offset=0.0,
        conditional_brake_logit_offset=0.0,
        steer_std_scale=0.1,
        brake_std_scale=1.0,
        schedule_id=f"b3-unified-policy-iter{index:02d}",
    )


def load_fresh_policy(
    arm: str,
    seed: int,
    bc_path: str | Path,
    sidecar_release: str | Path,
    device: torch.device,
    policy_contract: str = B2_POLICY_CONTRACT,
) -> RemediatedV22Policy:
    if arm not in ARMS or int(seed) not in (0, 1):
        raise ValueError("invalid B2 policy identity")
    bc = load_bc_model(str(bc_path), device)
    sidecar_state, sidecar_mean, sidecar_std, _ = load_sidecar_bundle(sidecar_release)
    policy_class = (
        UnifiedV22Policy
        if policy_contract == B3_POLICY_CONTRACT
        else RemediatedV22Policy
    )
    if policy_contract not in {B2_POLICY_CONTRACT, B3_POLICY_CONTRACT}:
        raise ValueError("invalid PPO policy contract")
    policy = policy_class(
        arm,
        bc_state_dict=bc.state_dict(),
        sidecar_state_dict=sidecar_state,
        sidecar_bc_mean=sidecar_mean,
        sidecar_bc_std=sidecar_std,
        initialization_seed=SEED + int(seed),
    ).to(device)
    if not torch.equal(
        policy.intervention_logit_offset,
        torch.zeros_like(policy.intervention_logit_offset),
    ):
        raise AssertionError("fresh B2 policy has persistent exploration state")
    return policy


def _macro_record(
    result: B2EpisodeResult,
    row,
    *,
    arm: str,
    seed: int,
    iteration: int,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
) -> MacroReplayRecord:
    action_tensor = torch.as_tensor(row.action, dtype=torch.float32).reshape(1, 4)
    action = HierarchicalResidualAction.from_tensor(action_tensor)
    requested = (
        HierarchicalResidualDistribution.requested_residual(action)[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    signals = MacroSignals(
        reward=0.0,
        collision_cost=float(row.collision_cost),
        performance_reward=float(row.performance_reward),
        length=int(row.length),
        discount=float(row.discount),
        terminated=bool(row.terminated),
        truncated=False,
    )
    behavior = row.behavior
    episode_id = (
        f"{result.scenario.l2_id}:repeat{row.episode_repeat}:iter{iteration}:seed{seed}"
    )
    return MacroReplayRecord(
        scenario_id=result.scenario.l2_id,
        l2_id=result.scenario.l2_id,
        episode_id=episode_id,
        macro_index=int(row.macro_index),
        arm=arm,
        training_seed=int(seed),
        policy_iteration=int(iteration),
        checkpoint_schema=checkpoint_schema,
        bc_feature=row.bc_feature,
        lidar_history=row.lidar_history,
        scalar_history=row.scalar_history,
        privileged_critic_feature=row.privileged_feature,
        latent=row.action,
        old_log_prob=float(row.old_log_prob),
        old_entropy=float(row.old_entropy),
        entropy_intervention=float(row.entropy_intervention),
        entropy_steer_given_intervention=float(
            row.entropy_steer_given_intervention
        ),
        entropy_brake_gate_given_intervention=float(
            row.entropy_brake_gate_given_intervention
        ),
        entropy_brake_magnitude_given_brake=float(
            row.entropy_brake_magnitude_given_brake
        ),
        intervention_offset=float(behavior["intervention_logit_offset"]),
        conditional_brake_offset=float(behavior["conditional_brake_logit_offset"]),
        steer_std_scale=float(behavior["steer_std_scale"]),
        brake_std_scale=float(behavior["brake_std_scale"]),
        schedule_id=str(behavior["schedule_id"]),
        requested_residual=requested,
        applied_composition_digest=row.composition_sha256,
        signals=signals,
        collision_value=float(row.collision_value),
        performance_value=float(row.performance_value),
        episode_start=row.macro_index == 0,
        bc_hidden_reset=row.macro_index == 0,
    )


def _replay_hook(policy: RemediatedV22Policy):
    def replay(**batch):
        count = len(batch["bc_feature"])
        context = BehaviorExplorationBatch(
            intervention_logit_offset=batch["intervention_offset"].reshape(-1, 1).detach(),
            conditional_brake_logit_offset=batch[
                "conditional_brake_offset"
            ].reshape(-1, 1).detach(),
            steer_std_scale=batch["steer_std_scale"].reshape(-1, 1).detach(),
            brake_std_scale=batch["brake_std_scale"].reshape(-1, 1).detach(),
            schedule_ids=("serialized-replay",) * count,
        )
        distribution = policy.behavior_distribution(
            batch["bc_feature"],
            batch["lidar_history"],
            batch["scalar_history"],
            context,
        )
        action = HierarchicalResidualAction.from_tensor(batch["latent"])
        return {
            "log_prob": distribution.log_prob(action),
            "entropy": distribution.entropy(),
        }

    return replay


def _keyed_permutation(count: int, seed: int, iteration: int, epoch: int) -> np.ndarray:
    digest = hashlib.sha256(MINIBATCH_ORDER_SCHEMA.encode("ascii") + b"\0")
    digest.update(f"{seed}:{iteration}:{epoch}".encode("ascii"))
    generator = np.random.default_rng(int.from_bytes(digest.digest()[:8], "big"))
    return generator.permutation(int(count))


def _slice_batch(batch: Mapping[str, torch.Tensor], indices: np.ndarray):
    tensor_index = torch.as_tensor(indices, dtype=torch.long, device=batch["old_log_prob"].device)
    return {name: value.index_select(0, tensor_index) for name, value in batch.items()}


def _clip(parameters: Sequence[torch.nn.Parameter]) -> float:
    selected = [value for value in parameters if value.grad is not None]
    if not selected:
        raise RuntimeError("B2 optimizer group received no gradient")
    return float(
        torch.nn.utils.clip_grad_norm_(
            selected, MAX_GRAD_NORM, error_if_nonfinite=True
        ).item()
    )


def update_policy(
    policy: RemediatedV22Policy,
    critics: B2Critics,
    optimizers,
    collision_scale: RunningCollisionScale,
    tensor_batch: Mapping[str, torch.Tensor],
    *,
    dual_value: float,
    seed: int,
    iteration: int,
) -> dict[str, Any]:
    """Run the frozen three-epoch, 128-minibatch B2 PPO update."""

    count = len(tensor_batch["old_log_prob"])
    replay = _replay_hook(policy)
    policy.train()
    with torch.no_grad():
        original = replay(
            bc_feature=tensor_batch["bc_feature"],
            lidar_history=tensor_batch["lidar_history"],
            scalar_history=tensor_batch["scalar_history"],
            latent=tensor_batch["latent"],
            intervention_offset=tensor_batch["intervention_offset"],
            conditional_brake_offset=tensor_batch["conditional_brake_offset"],
            steer_std_scale=tensor_batch["steer_std_scale"],
            brake_std_scale=tensor_batch["brake_std_scale"],
        )
    replay_log_prob_delta = torch.abs(
        original["log_prob"] - tensor_batch["old_log_prob"]
    )
    replay_entropy_delta = torch.abs(
        original["entropy"] - tensor_batch["old_entropy"]
    )
    max_log_prob_delta = float(torch.max(replay_log_prob_delta).item())
    max_entropy_delta = float(torch.max(replay_entropy_delta).item())
    if max_log_prob_delta > REPLAY_FLOAT32_ATOL:
        raise AssertionError(
            f"B2 serialized replay log-prob mismatch: {max_log_prob_delta}"
        )
    if max_entropy_delta > REPLAY_FLOAT32_ATOL:
        raise AssertionError(
            f"B2 serialized replay entropy mismatch: {max_entropy_delta}"
        )
    preupdate_ratio = torch.exp(
        original["log_prob"] - tensor_batch["old_log_prob"]
    )
    max_ratio_delta = float(torch.max(torch.abs(preupdate_ratio - 1.0)).item())

    actor_advantage, scale_record = b2_constrained_advantage(
        tensor_batch["collision_advantage"],
        tensor_batch["performance_advantage"],
        dual_value,
        collision_scale,
        update_collision_scale=True,
        collision_event_present=bool(
            torch.any(tensor_batch["collision_cost"] > 0).item()
        ),
    )
    actor_groups = [
        [value for value in group["params"]] for group in optimizers.actor.param_groups
    ]
    metrics: list[dict[str, float | int]] = []
    early_stopped = False
    for epoch in range(PPO_EPOCHS):
        order = _keyed_permutation(count, seed, iteration, epoch)
        for batch_start in range(0, count, MINIBATCH_SIZE):
            indices = order[batch_start : batch_start + MINIBATCH_SIZE]
            mini = _slice_batch(tensor_batch, indices)
            losses = compute_b2_losses(
                critics,
                mini,
                replay,
                collision_scale,
                dual_value=dual_value,
                clip_epsilon=CLIP_EPSILON,
                entropy_coefficient=ENTROPY_COEFFICIENT,
                update_collision_scale=False,
                precomputed_actor_advantage=actor_advantage.index_select(
                    0,
                    torch.as_tensor(
                        indices,
                        dtype=torch.long,
                        device=actor_advantage.device,
                    ),
                ),
                precomputed_collision_scale=scale_record,
            )
            if float(losses.approx_kl.item()) > TARGET_KL * 1.5:
                early_stopped = True
                break
            optimizers.actor.zero_grad(set_to_none=True)
            optimizers.collision_critic.zero_grad(set_to_none=True)
            optimizers.performance_critic.zero_grad(set_to_none=True)
            losses.actor_loss.backward()
            losses.critic_losses["collision"].backward()
            losses.critic_losses["performance"].backward()
            actor_norms = [_clip(group) for group in actor_groups]
            collision_norm = _clip(list(critics.collision.parameters()))
            performance_norm = _clip(list(critics.performance.parameters()))
            optimizers.actor.step()
            optimizers.collision_critic.step()
            optimizers.performance_critic.step()
            metrics.append(
                {
                    "epoch": epoch,
                    "batch_start": batch_start,
                    "batch_size": len(indices),
                    "policy_loss": float(losses.policy_loss.item()),
                    "actor_loss": float(losses.actor_loss.item()),
                    "collision_critic_loss": float(
                        losses.critic_losses["collision"].item()
                    ),
                    "performance_critic_loss": float(
                        losses.critic_losses["performance"].item()
                    ),
                    "entropy": float(losses.entropy.item()),
                    "approx_kl": float(losses.approx_kl.item()),
                    "clip_fraction": float(losses.clip_fraction.item()),
                    "ratio_min": float(losses.ratio_min.item()),
                    "ratio_mean": float(losses.ratio_mean.item()),
                    "ratio_max": float(losses.ratio_max.item()),
                    "actor_grad_norm_max": max(actor_norms),
                    "collision_grad_norm": collision_norm,
                    "performance_grad_norm": performance_norm,
                }
            )
        if early_stopped:
            break
    policy.assert_frozen_unchanged(policy._b2_frozen_snapshot)
    return {
        "updates": len(metrics),
        "early_stopped": early_stopped,
        "preupdate_replay_tolerance": REPLAY_FLOAT32_ATOL,
        "preupdate_replay_max_abs_log_prob_delta": max_log_prob_delta,
        "preupdate_replay_max_abs_entropy_delta": max_entropy_delta,
        "preupdate_replay_max_abs_ratio_minus_one": max_ratio_delta,
        "rollout_actor_advantage_mean": float(actor_advantage.mean().item()),
        "rollout_actor_advantage_std": float(
            actor_advantage.std(unbiased=False).item()
        ),
        "collision_scale_record": {
            "informative": scale_record.informative,
            "batch_variance": scale_record.batch_variance,
            "variance_before": scale_record.variance_before,
            "variance_after": scale_record.variance_after,
            "scale": scale_record.scale,
        },
        "minibatches": metrics,
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    *,
    policy: RemediatedV22Policy,
    critics: B2Critics,
    optimizers,
    dual: OvertakeDual,
    collision_scale: RunningCollisionScale,
    arm: str,
    seed: int,
    iteration: int,
    curriculum_digest: str,
    training_manifest_sha256: str,
    plan_sha256: str,
    config: Mapping[str, Any],
    source_commit: str,
    occurrence_count: Mapping[str, int],
    optimizer_update_count: int,
) -> str:
    if not torch.equal(
        policy.intervention_logit_offset,
        torch.zeros_like(policy.intervention_logit_offset),
    ):
        raise ValueError("B2 checkpoint cannot absorb behavior exploration offset")
    _validate_config(config)
    total_iterations = _iterations(config)
    checkpoint_schema = _checkpoint_schema(config)
    if not isinstance(source_commit, str) or len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("B2 checkpoint source commit must be a resolved SHA-1")
    cursor = dict(sorted(occurrence_count.items()))
    if any(
        not str(l2_id).startswith("L2:")
        or int(count) != count
        or int(count) < 0
        for l2_id, count in cursor.items()
    ):
        raise ValueError("B2 checkpoint scenario cursor is invalid")
    checkpoint_iteration = int(iteration)
    if (
        checkpoint_iteration != iteration
        or not 0 <= checkpoint_iteration <= total_iterations
    ):
        raise ValueError("B2 checkpoint iteration is invalid")
    expected_episode_count = checkpoint_iteration * EPISODES_PER_ITERATION
    if sum(cursor.values()) != expected_episode_count:
        raise ValueError("B2 checkpoint scenario cursor count/iteration mismatch")
    update_count = int(optimizer_update_count)
    if update_count != optimizer_update_count or update_count < 0:
        raise ValueError("B2 checkpoint optimizer update count is invalid")
    if checkpoint_iteration == 0 and update_count != 0:
        raise ValueError("B2 iteration-0 checkpoint cannot contain optimizer updates")
    payload = {
        "schema": checkpoint_schema,
        "checkpoint_id": f"{arm}-seed{seed}-iter{checkpoint_iteration:04d}",
        "arm": arm,
        "seed": int(seed),
        "iteration": checkpoint_iteration,
        "curriculum_sha256": _validate_digest(curriculum_digest, "curriculum digest"),
        "training_manifest_sha256": _validate_digest(
            training_manifest_sha256, "training manifest digest"
        ),
        "run_plan_sha256": _validate_digest(plan_sha256, "run-plan digest"),
        "source_commit": source_commit,
        "source_worktree_clean": True,
        "config": dict(config),
        "policy_state": policy.state_dict(),
        "critics_state": critics.state_dict(),
        "optimizers_state": optimizers.state_dict(),
        "dual_state": dual.state_dict(),
        "collision_scale_state": collision_scale.state_dict(),
        "scenario_occurrence_count": cursor,
        "rollout_episodes_completed": expected_episode_count,
        "optimizer_minibatches_completed": update_count,
        "action_noise_schema": ACTION_NOISE_SCHEMA,
        "minibatch_order_schema": MINIBATCH_ORDER_SCHEMA,
        "rng_state": _rng_state(),
        "next_schedule": (
            None
            if checkpoint_iteration >= total_iterations
            else behavior_for_config(checkpoint_iteration + 1, config).as_dict()
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    torch.save(payload, partial)
    os.replace(partial, path)
    return _file_sha256(path)


def load_checkpoint(
    path: str | Path,
    *,
    policy: RemediatedV22Policy,
    critics: B2Critics,
    optimizers,
    expected_arm: str,
    expected_seed: int,
    expected_curriculum_digest: str,
    expected_training_manifest_sha256: str,
    expected_plan_sha256: str,
    expected_source_commit: str,
    expected_config: Mapping[str, Any],
    expected_iteration: int,
    expected_occurrence_count: Mapping[str, int],
    expected_checkpoint_sha256: str,
) -> tuple[int, OvertakeDual, RunningCollisionScale, dict[str, Any]]:
    source = Path(path)
    if _file_sha256(source) != _validate_digest(
        expected_checkpoint_sha256, "checkpoint digest"
    ):
        raise ValueError("B2 checkpoint file digest mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    expected_keys = {
        "schema", "checkpoint_id", "arm", "seed", "iteration",
        "curriculum_sha256", "training_manifest_sha256", "run_plan_sha256",
        "source_commit", "source_worktree_clean", "config", "policy_state", "critics_state",
        "optimizers_state", "dual_state", "collision_scale_state",
        "scenario_occurrence_count", "rollout_episodes_completed",
        "optimizer_minibatches_completed", "action_noise_schema",
        "minibatch_order_schema", "rng_state", "next_schedule",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("B2 checkpoint field inventory mismatch")
    iteration = int(expected_iteration)
    total_iterations = _iterations(expected_config)
    checkpoint_schema = _checkpoint_schema(expected_config)
    if iteration != expected_iteration or not 0 <= iteration <= total_iterations:
        raise ValueError("B2 expected checkpoint iteration is invalid")
    expected_cursor = dict(sorted(expected_occurrence_count.items()))
    _validate_config(expected_config)
    if (
        payload.get("schema") != checkpoint_schema
        or payload.get("checkpoint_id")
        != f"{expected_arm}-seed{int(expected_seed)}-iter{iteration:04d}"
        or payload.get("arm") != expected_arm
        or payload.get("seed") != int(expected_seed)
        or payload.get("iteration") != iteration
        or payload.get("curriculum_sha256") != expected_curriculum_digest
        or payload.get("training_manifest_sha256") != expected_training_manifest_sha256
        or payload.get("run_plan_sha256") != expected_plan_sha256
        or payload.get("source_commit") != expected_source_commit
        or payload.get("source_worktree_clean") is not True
        or payload.get("config") != dict(expected_config)
        or payload.get("scenario_occurrence_count") != expected_cursor
        or payload.get("rollout_episodes_completed")
        != iteration * EPISODES_PER_ITERATION
        or payload.get("action_noise_schema") != ACTION_NOISE_SCHEMA
        or payload.get("minibatch_order_schema") != MINIBATCH_ORDER_SCHEMA
    ):
        raise ValueError("B2 checkpoint envelope mismatch")
    update_count = payload.get("optimizer_minibatches_completed")
    if int(update_count) != update_count or int(update_count) < 0:
        raise ValueError("B2 checkpoint optimizer-update cursor is invalid")
    expected_next = (
        None
        if iteration >= total_iterations
        else behavior_for_config(iteration + 1, expected_config).as_dict()
    )
    if payload.get("next_schedule") != expected_next:
        raise ValueError("B2 checkpoint next-schedule cursor mismatch")
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict) or set(rng_state) != {
        "python", "numpy", "torch_cpu", "torch_cuda"
    }:
        raise ValueError("B2 checkpoint RNG-state schema mismatch")
    if torch.cuda.is_available() and len(rng_state["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("B2 checkpoint CUDA RNG device inventory mismatch")
    frozen_before = policy.frozen_snapshot()
    if _policy_contract(expected_config) == B3_POLICY_CONTRACT:
        if not isinstance(policy, UnifiedV22Policy):
            raise ValueError("B3 checkpoint requires a unified policy instance")
        policy.load_unified_state_dict(payload["policy_state"])
    else:
        if isinstance(policy, UnifiedV22Policy):
            raise ValueError("B2 checkpoint cannot load into a unified policy")
        policy.load_hierarchical_state_dict(payload["policy_state"])
    policy.assert_frozen_unchanged(frozen_before)
    if not torch.equal(
        policy.intervention_logit_offset,
        torch.zeros_like(policy.intervention_logit_offset),
    ):
        raise ValueError("B2 checkpoint contains persistent behavior exploration")
    critics.load_state_dict(payload["critics_state"], strict=True)
    optimizers.load_state_dict(payload["optimizers_state"])
    dual = OvertakeDual.from_state_dict(payload["dual_state"])
    collision_scale = RunningCollisionScale.from_state_dict(
        payload["collision_scale_state"]
    )
    _restore_rng_state(rng_state)
    return iteration, dual, collision_scale, payload


def load_policy_only_checkpoint(
    path: str | Path,
    *,
    expected_arm: str,
    expected_seed: int,
    expected_iteration: int,
    expected_training_manifest_sha256: str,
    expected_plan_sha256: str,
    expected_checkpoint_sha256: str,
    device: torch.device,
) -> tuple[RemediatedV22Policy, dict[str, Any]]:
    source = Path(path)
    if _file_sha256(source) != expected_checkpoint_sha256:
        raise ValueError("B2 policy-only checkpoint file digest mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload_config = payload.get("config")
    if not isinstance(payload_config, Mapping):
        raise ValueError("PPO policy-only checkpoint lacks locked config")
    _validate_config(payload_config)
    contract = _policy_contract(payload_config)
    if (
        payload.get("schema") != _checkpoint_schema(payload_config)
        or payload.get("arm") != expected_arm
        or payload.get("seed") != int(expected_seed)
        or payload.get("iteration") != int(expected_iteration)
        or payload.get("training_manifest_sha256")
        != expected_training_manifest_sha256
        or payload.get("run_plan_sha256") != expected_plan_sha256
    ):
        raise ValueError("B2 policy-only checkpoint envelope mismatch")
    policy_class = UnifiedV22Policy if contract == B3_POLICY_CONTRACT else RemediatedV22Policy
    policy = policy_class(
        expected_arm, initialization_seed=SEED + int(expected_seed)
    ).to(device)
    if contract == B3_POLICY_CONTRACT:
        policy.load_unified_state_dict(payload["policy_state"])
    else:
        policy.load_hierarchical_state_dict(payload["policy_state"])
    if not torch.equal(
        policy.intervention_logit_offset,
        torch.zeros_like(policy.intervention_logit_offset),
    ):
        raise ValueError("B2 candidate checkpoint contains persistent exploration")
    policy.eval()
    return policy, payload


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_replay_ledger(
    root: Path, iteration: int, batch
) -> tuple[str, dict[str, Any]]:
    """Persist the full macro replay contract before any policy update."""

    replay_dir = root / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    path = replay_dir / f"iter_{int(iteration):04d}.npz"
    if path.exists():
        raise FileExistsError(path)
    payload: dict[str, np.ndarray] = {}
    for name in batch.__dataclass_fields__:
        value = getattr(batch, name)
        array = np.asarray(value)
        if array.dtype == object:
            raise TypeError(f"B2 replay ledger field is not portable: {name}")
        payload[name] = array
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("xb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    latent = np.asarray(batch.latent)
    intervention = latent[:, 0] > 0.5
    brake = latent[:, 2] > 0.5
    if np.any(brake & ~intervention):
        raise AssertionError("B2 replay ledger contains brake outside intervention")
    diagnostics = {
        "macro_count": int(len(latent)),
        "intervention_macro_count": int(np.count_nonzero(intervention)),
        "intervention_macro_rate": float(np.mean(intervention)),
        "brake_macro_count": int(np.count_nonzero(brake)),
        "joint_brake_macro_rate": float(np.mean(brake)),
        "conditional_brake_rate": (
            None
            if not np.any(intervention)
            else float(np.count_nonzero(brake) / np.count_nonzero(intervention))
        ),
        "steer_only_macro_count": int(np.count_nonzero(intervention & ~brake)),
        "entropy_total_mean": float(np.mean(batch.old_entropy)),
        "entropy_intervention_mean": float(np.mean(batch.entropy_intervention)),
        "entropy_steer_given_intervention_mean": float(
            np.mean(batch.entropy_steer_given_intervention)
        ),
        "entropy_brake_gate_given_intervention_mean": float(
            np.mean(batch.entropy_brake_gate_given_intervention)
        ),
        "entropy_brake_magnitude_given_brake_mean": float(
            np.mean(batch.entropy_brake_magnitude_given_brake)
        ),
        "mean_abs_requested_steer": float(
            np.mean(np.abs(batch.requested_residual[:, 0]))
        ),
        "mean_requested_brake": float(
            np.mean(-batch.requested_residual[:, 1])
        ),
    }
    return _file_sha256(path), diagnostics


def _latent_branch_presence(latent_values) -> dict[str, bool]:
    """Return outcome-blind branch coverage for one production-shaped batch."""

    latent = np.asarray(latent_values)
    if latent.ndim != 2 or latent.shape[1] != 4 or len(latent) == 0:
        raise ValueError("B2 branch-presence latent must have shape [N,4]")
    intervention = latent[:, 0] > 0.5
    brake = latent[:, 2] > 0.5
    if np.any(brake & ~intervention):
        raise AssertionError("B2 branch presence found brake outside intervention")
    return {
        "intervention_branch_present": bool(np.any(intervention)),
        "joint_brake_branch_present": bool(np.any(intervention & brake)),
        "steer_only_branch_present": bool(np.any(intervention & ~brake)),
    }


def _curriculum_record(curriculum_plan) -> dict[str, Any]:
    return {
        "rows": [
            {
                "iteration": iteration,
                "episode_index": index,
                "l2_id": row.l2_id,
                "l4_id": row.l4_id,
                "bc_collision_any": row.bc_collision_any,
                "archived_bc_outcome": row.archived_bc_outcome,
            }
            for iteration, rows in enumerate(curriculum_plan, start=1)
            for index, row in enumerate(rows)
        ]
    }


def _expected_occurrence_cursor(curriculum_plan, through_iteration: int) -> dict[str, int]:
    limit = int(through_iteration)
    if limit != through_iteration or not 0 <= limit <= len(curriculum_plan):
        raise ValueError("B2 curriculum resume iteration is invalid")
    counts: dict[str, int] = {}
    for rows in curriculum_plan[:limit]:
        for scenario in rows:
            counts[scenario.l2_id] = counts.get(scenario.l2_id, 0) + 1
    return dict(sorted(counts.items()))


def _read_iteration_ledger(
    path: Path, max_iterations: int = ITERATIONS
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("B2 iteration ledger has an incomplete final line")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"B2 iteration ledger JSON is invalid at line {line_number}"
            ) from error
        if not isinstance(row, dict) or row.get("iteration") != line_number:
            raise ValueError("B2 iteration ledger sequence is not contiguous")
        rows.append(row)
    if len(rows) > int(max_iterations):
        raise ValueError("B2 iteration ledger exceeds the frozen schedule")
    return rows


def _repair_torn_iteration_ledger(partial: Path) -> None:
    """Preserve a torn append and restore its complete newline-delimited prefix."""

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
    index = 1
    while (attempts / f"torn_iterations_{index:03d}.jsonl").exists():
        index += 1
    os.replace(path, attempts / f"torn_iterations_{index:03d}.jsonl")
    with path.open("xb") as handle:
        handle.write(prefix)
        handle.flush()
        os.fsync(handle.fileno())


def _quarantine_uncommitted_resume_files(
    partial: Path, committed_iteration: int
) -> None:
    extras: list[Path] = []
    for directory, pattern in (
        (partial / "checkpoints", "iter_*.pt"),
        (partial / "replay", "iter_*.npz"),
    ):
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            try:
                index = int(path.stem.split("_")[-1])
            except ValueError as error:
                raise ValueError(f"B2 resume found malformed iteration file: {path}") from error
            if index > committed_iteration:
                extras.append(path)
    failed = partial / "FAILED"
    if failed.exists():
        extras.append(failed)
    if not extras:
        return
    attempts = partial / "attempt_failures"
    attempts.mkdir(exist_ok=True)
    attempt = 1
    while (attempts / f"attempt_{attempt:03d}").exists():
        attempt += 1
    target = attempts / f"attempt_{attempt:03d}"
    target.mkdir()
    for source in extras:
        destination = target / source.relative_to(partial).as_posix().replace("/", "__")
        os.replace(source, destination)


def _validate_resume_prefix(
    partial: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, str, int]:
    checkpoint0 = partial / "checkpoints/iter_0000.pt"
    if not checkpoint0.is_file():
        raise ValueError("B2 resume lacks iteration-0 checkpoint")
    committed = len(rows)
    optimizer_updates = 0
    for iteration, row in enumerate(rows, start=1):
        checkpoint = partial / f"checkpoints/iter_{iteration:04d}.pt"
        replay = partial / f"replay/iter_{iteration:04d}.npz"
        if (
            not checkpoint.is_file()
            or _file_sha256(checkpoint) != row.get("checkpoint_sha256")
            or not replay.is_file()
            or _file_sha256(replay) != row.get("replay_ledger_sha256")
        ):
            raise ValueError("B2 resume committed iteration file/hash mismatch")
        update = row.get("update")
        if not isinstance(update, Mapping) or int(update.get("updates", -1)) < 0:
            raise ValueError("B2 resume iteration update ledger is invalid")
        optimizer_updates += int(update["updates"])
    checkpoint = partial / f"checkpoints/iter_{committed:04d}.pt"
    return committed, _file_sha256(checkpoint), optimizer_updates


def run_plumbing_smoke(
    plan_path: str | Path, *, device_name: str = "cuda:0"
) -> dict[str, Any]:
    """Run the frozen four-map/all-arm single-update integrity smoke."""

    validated = validate_pilot_plan(plan_path)
    plan = validated["plan"]
    paths = validated["paths"]
    config = plan["config"]
    policy_contract = _policy_contract(config)
    checkpoint_schema = _checkpoint_schema(config)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("B2 plumbing smoke requested unavailable CUDA")
    scenarios = load_b2_scenario_sets(paths["task8"], paths["metadata"])
    selected_by_map: dict[str, Any] = {}
    for scenario in scenarios.training:
        selected_by_map.setdefault(scenario.map_name, scenario)
    expected_maps = ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")
    if tuple(sorted(selected_by_map)) != tuple(sorted(expected_maps)):
        raise ValueError("B2 plumbing smoke map inventory drift")
    selected = tuple(selected_by_map[name] for name in expected_maps)
    training_manifest = paths["task8"] / TRAINING_MANIFEST_NAME
    bc_checkpoint = paths["bc"]
    sidecar_bundle = paths["sidecar"] / "sidecar_bundle.pt"
    d2_metadata = paths["metadata"]
    behavior = behavior_for_config(1, config)
    arm_reports: dict[str, Any] = {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        policy = load_fresh_policy(
            arm,
            0,
            paths["bc"],
            paths["sidecar"],
            device,
            policy_contract,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 1000)
            critics = B2Critics().to(device)
        optimizers = build_b2_optimizers(
            policy, critics, critic_learning_rate=CRITIC_LR
        )
        policy._b2_frozen_snapshot = policy.frozen_snapshot()
        buffer = EpisodeCompleteMacroBuffer(target_episodes=len(selected))
        for scenario in selected:
            result = run_b2_episode(
                policy,
                critics,
                device,
                scenario,
                behavior,
                0,
                0,
            )
            buffer.add_episode(
                [
                    _macro_record(
                        result,
                        row,
                        arm=arm,
                        seed=0,
                        iteration=1,
                        checkpoint_schema=checkpoint_schema,
                    )
                    for row in result.transitions
                ]
            )
        batch = buffer.collate()
        branch_presence = _latent_branch_presence(batch.latent)
        if not all(branch_presence.values()):
            raise AssertionError("B2 plumbing smoke did not cover all action branches")
        update = update_policy(
            policy,
            critics,
            optimizers,
            RunningCollisionScale(decay=COLLISION_SCALE_DECAY),
            batch.tensors(device),
            dual_value=1.0,
            seed=0,
            iteration=1,
        )
        arm_reports[arm] = {
            "episode_count": buffer.episode_count,
            **branch_presence,
            "optimizer_update_executed": int(update["updates"]) > 0,
            "preupdate_replay_tolerance": update["preupdate_replay_tolerance"],
            "preupdate_replay_max_abs_log_prob_delta": update[
                "preupdate_replay_max_abs_log_prob_delta"
            ],
            "preupdate_replay_max_abs_entropy_delta": update[
                "preupdate_replay_max_abs_entropy_delta"
            ],
            "preupdate_replay_max_abs_ratio_minus_one": update[
                "preupdate_replay_max_abs_ratio_minus_one"
            ],
            "finite_update_metrics": all(
                np.isfinite(float(value))
                for row in update["minibatches"]
                for key, value in row.items()
                if key not in {"epoch", "batch_start", "batch_size"}
            ),
        }
    if any(
        report["episode_count"] != 4
        or not report["optimizer_update_executed"]
        or not report["intervention_branch_present"]
        or not report["joint_brake_branch_present"]
        or not report["steer_only_branch_present"]
        or not report["finite_update_metrics"]
        for report in arm_reports.values()
    ):
        raise AssertionError("B2 plumbing smoke integrity failed")
    return {
        "schema": "bplus-v2.2-b2-plumbing-smoke-1",
        "passed": True,
        "run_plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source_commit"],
        "training_manifest_sha256": _file_sha256(training_manifest),
        "bc_checkpoint_sha256": _file_sha256(bc_checkpoint),
        "sidecar_bundle_sha256": _file_sha256(sidecar_bundle),
        "d2_episode_metadata_sha256": _file_sha256(d2_metadata),
        "scenario_selection": "first_physical_training_row_per_map_outcome_blind",
        "selected_scenarios": [
            {
                "training_order": int(scenario.training_order),
                "map_name": scenario.map_name,
                "l2_id": scenario.l2_id,
                "l4_id": scenario.l4_id,
                "skill": scenario.skill,
                "opponent_raceline": scenario.opponent_raceline,
                "speedscale_hex": float(scenario.speedscale).hex(),
                "resolved_ego_idx": int(scenario.resolved_ego_idx),
            }
            for scenario in selected
        ],
        "arms": arm_reports,
        "product_outcomes_reported_or_compared": False,
        "arm_selection_performed": False,
        "ppo_pilot_iteration_completed": False,
    }


def run_pilot_job(
    plan_path: str | Path,
    job_id: str,
    *,
    device_name: str = "cuda:0",
    resume: bool = False,
) -> dict[str, Any]:
    validated = validate_pilot_plan(
        plan_path, job_id, allow_partial_resume=bool(resume)
    )
    plan = validated["plan"]
    paths = validated["paths"]
    job = validated["job"]
    config = plan["config"]
    policy_contract = _policy_contract(config)
    total_iterations = _iterations(config)
    checkpoint_schema = _checkpoint_schema(config)
    pilot_schema = _pilot_schema(config)
    _validate_control_plane_ready(plan, paths)
    arm = str(job["arm"])
    seed = int(job["seed"])
    output = paths["root"] / str(job["output_relpath"])
    partial = output.with_name(output.name + ".partial")
    if not resume:
        partial.mkdir(parents=True)
    try:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("B2 CUDA learner requested but CUDA is unavailable")
        torch.manual_seed(SEED + seed)
        np.random.seed(SEED + seed)
        random.seed(SEED + seed)
        policy = load_fresh_policy(
            arm,
            seed,
            paths["bc"],
            paths["sidecar"],
            device,
            policy_contract,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + seed + 1000)
            critics = B2Critics().to(device)
        optimizers = build_b2_optimizers(
            policy, critics, critic_learning_rate=CRITIC_LR
        )
        policy._b2_frozen_snapshot = policy.frozen_snapshot()
        collision_scale = RunningCollisionScale(decay=COLLISION_SCALE_DECAY)
        scenarios = load_b2_scenario_sets(paths["task8"], paths["metadata"])
        curriculum = B2Curriculum(scenarios, seed)
        curriculum_plan = curriculum.plan(total_iterations)
        curriculum_digest = curriculum.digest(total_iterations)
        training_manifest = paths["task8"] / TRAINING_MANIFEST_NAME
        training_manifest_sha = _file_sha256(training_manifest)
        archived_rate = float(
            np.mean(
                [
                    row.archived_bc_outcome == "overtake"
                    for iteration_rows in curriculum_plan
                    for row in iteration_rows
                ]
            )
        )
        dual_floor = max(0.0, archived_rate - 0.01)
        dual = OvertakeDual(floor=dual_floor)
        config_record = {
            "schema": pilot_schema,
            "arm": arm,
            "seed": seed,
            "run_plan_sha256": plan["plan_sha256"],
            "source_commit": plan["source_commit"],
            "curriculum_sha256": curriculum_digest,
            "training_manifest_sha256": training_manifest_sha,
            "archived_bc_overtake_rate": archived_rate,
            "dual_floor": dual_floor,
            "config": config,
        }
        curriculum_record = _curriculum_record(curriculum_plan)
        checkpoint0 = partial / "checkpoints/iter_0000.pt"
        if resume:
            observed_config = json.loads(
                (partial / "config.json").read_text(encoding="utf-8")
            )
            observed_curriculum = json.loads(
                (partial / "curriculum.json").read_text(encoding="utf-8")
            )
            if observed_config != config_record or observed_curriculum != curriculum_record:
                raise ValueError("B2 resume config/curriculum prefix drift")
            _repair_torn_iteration_ledger(partial)
            iteration_rows_ledger = _read_iteration_ledger(
                partial / "iterations.jsonl", total_iterations
            )
            committed_iteration = len(iteration_rows_ledger)
            _quarantine_uncommitted_resume_files(partial, committed_iteration)
            committed_iteration, checkpoint_sha, recorded_updates = (
                _validate_resume_prefix(partial, iteration_rows_ledger)
            )
            expected_cursor = _expected_occurrence_cursor(
                curriculum_plan, committed_iteration
            )
            _, dual, collision_scale, payload = load_checkpoint(
                partial / f"checkpoints/iter_{committed_iteration:04d}.pt",
                policy=policy,
                critics=critics,
                optimizers=optimizers,
                expected_arm=arm,
                expected_seed=seed,
                expected_curriculum_digest=curriculum_digest,
                expected_training_manifest_sha256=training_manifest_sha,
                expected_plan_sha256=plan["plan_sha256"],
                expected_source_commit=plan["source_commit"],
                expected_config=config,
                expected_iteration=committed_iteration,
                expected_occurrence_count=expected_cursor,
                expected_checkpoint_sha256=checkpoint_sha,
            )
            if int(payload["optimizer_minibatches_completed"]) != recorded_updates:
                raise ValueError("B2 resume optimizer-update ledger/checkpoint drift")
            occurrence_count = dict(payload["scenario_occurrence_count"])
            optimizer_update_count = recorded_updates
            start_iteration = committed_iteration + 1
            resumed_from_iteration = committed_iteration
            sha0 = _file_sha256(checkpoint0)
        else:
            _write_json(partial / "config.json", config_record)
            _write_json(partial / "curriculum.json", curriculum_record)
            sha0 = save_checkpoint(
                checkpoint0,
                policy=policy,
                critics=critics,
                optimizers=optimizers,
                dual=dual,
                collision_scale=collision_scale,
                arm=arm,
                seed=seed,
                iteration=0,
                curriculum_digest=curriculum_digest,
                training_manifest_sha256=training_manifest_sha,
                plan_sha256=plan["plan_sha256"],
                config=config,
                source_commit=plan["source_commit"],
                occurrence_count={},
                optimizer_update_count=0,
            )
            occurrence_count = {}
            optimizer_update_count = 0
            start_iteration = 1
            resumed_from_iteration = None
        for iteration in range(start_iteration, total_iterations + 1):
            iteration_rows = curriculum_plan[iteration - 1]
            behavior = behavior_for_config(iteration, config)
            buffer = EpisodeCompleteMacroBuffer(target_episodes=EPISODES_PER_ITERATION)
            outcome_rows = []
            for scenario in iteration_rows:
                repeat = occurrence_count.get(scenario.l2_id, 0)
                occurrence_count[scenario.l2_id] = repeat + 1
                result = run_b2_episode(
                    policy,
                    critics,
                    device,
                    scenario,
                    behavior,
                    seed,
                    repeat,
                )
                records = [
                    _macro_record(
                        result,
                        row,
                        arm=arm,
                        seed=seed,
                        iteration=iteration,
                        checkpoint_schema=checkpoint_schema,
                    )
                    for row in result.transitions
                ]
                buffer.add_episode(records)
                outcome_rows.append(
                    {
                        "l2_id": scenario.l2_id,
                        "collision_any": bool(result.outcome.collision_any),
                        "terminal_overtake": result.outcome.corrected_outcome3 == "overtake",
                        "confirmed_safe_pass": result.outcome.confirmed_safe_pass is True,
                        "interaction_attempt": result.outcome.interaction_attempt is True,
                        "macro_count": len(records),
                        "micro_count": result.micro_steps,
                    }
                )
            if not buffer.ready or buffer.episode_count != EPISODES_PER_ITERATION:
                raise AssertionError("B2 fixed episode rollout did not complete")
            batch = buffer.collate()
            replay_sha, action_diagnostics = _write_replay_ledger(
                partial, iteration, batch
            )
            tensors = batch.tensors(device)
            update = update_policy(
                policy,
                critics,
                optimizers,
                collision_scale,
                tensors,
                dual_value=dual.value,
                seed=seed,
                iteration=iteration,
            )
            optimizer_update_count += int(update["updates"])
            dual_record = None
            if policy_contract == B3_POLICY_CONTRACT:
                overtake_rate = float(
                    np.mean([row["terminal_overtake"] for row in outcome_rows])
                )
                dual_record = dual.update_with_record(
                    overtake_rate, completed_episodes=len(outcome_rows)
                ).ordered_log()
            elif EXPLORATION_MULTIPLIERS[iteration - 1] == 0.0:
                overtake_rate = float(np.mean([row["terminal_overtake"] for row in outcome_rows]))
                dual_record = dual.update_with_record(
                    overtake_rate, completed_episodes=len(outcome_rows)
                ).ordered_log()
            elif dual.value != 1.0 or dual.completed_episodes != 0:
                raise AssertionError("B2 dual changed during exploration phase")
            checkpoint = partial / f"checkpoints/iter_{iteration:04d}.pt"
            checkpoint_sha = save_checkpoint(
                checkpoint,
                policy=policy,
                critics=critics,
                optimizers=optimizers,
                dual=dual,
                collision_scale=collision_scale,
                arm=arm,
                seed=seed,
                iteration=iteration,
                curriculum_digest=curriculum_digest,
                training_manifest_sha256=training_manifest_sha,
                plan_sha256=plan["plan_sha256"],
                config=config,
                source_commit=plan["source_commit"],
                occurrence_count=occurrence_count,
                optimizer_update_count=optimizer_update_count,
            )
            _append_jsonl(
                partial / "iterations.jsonl",
                {
                    "iteration": iteration,
                    "schedule": behavior.as_dict(),
                    "episode_count": len(outcome_rows),
                    "macro_count": len(batch.old_log_prob),
                    "replay_ledger_sha256": replay_sha,
                    "action_diagnostics": action_diagnostics,
                    "collision_count": sum(row["collision_any"] for row in outcome_rows),
                    "overtake_count": sum(row["terminal_overtake"] for row in outcome_rows),
                    "confirmed_safe_pass_count": sum(
                        row["confirmed_safe_pass"] for row in outcome_rows
                    ),
                    "dual_before_update": (
                        dual_record["value_before"] if dual_record is not None else dual.value
                    ),
                    "dual_after_update": dual.value,
                    "dual_update": dual_record,
                    "collision_scale": collision_scale.state_dict(),
                    "update": update,
                    "checkpoint_sha256": checkpoint_sha,
                    "outcomes": outcome_rows,
                },
            )
        summary = {
            "schema": pilot_schema,
            "passed": True,
            "integrity_passed": True,
            "arm": arm,
            "seed": seed,
            "iterations": total_iterations,
            "resumed_from_iteration": resumed_from_iteration,
            "optimizer_minibatches_completed": optimizer_update_count,
            "iteration0_checkpoint_sha256": sha0,
            "final_checkpoint_sha256": _file_sha256(
                partial / f"checkpoints/iter_{total_iterations:04d}.pt"
            ),
            "training_manifest_sha256": training_manifest_sha,
            "curriculum_sha256": curriculum_digest,
            "run_plan_sha256": plan["plan_sha256"],
            "product_kpi_evaluated": False,
        }
        if policy_contract == B2_POLICY_CONTRACT:
            summary["iteration20_checkpoint_sha256"] = summary[
                "final_checkpoint_sha256"
            ]
        _write_json(partial / "summary.json", summary)
        (partial / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output)
        return summary
    except Exception:
        # Keep the failed partial for diagnosis; runner status records nonzero exit.
        if partial.exists():
            (partial / "FAILED").write_text("FAILED\n", encoding="utf-8")
        raise

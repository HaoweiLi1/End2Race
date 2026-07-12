#!/usr/bin/env python3
"""B2 schedule, checkpoint, resume, and replay-runner contracts."""

from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from bplus_v22 import ARM_BC_FROZEN
from bplus_v22.exploration import ActionNoiseKey
from bplus_v22.objective import OvertakeDual
from bplus_v22.ppo import B2Critics, RunningCollisionScale, build_b2_optimizers
from bplus_v22.ppo_runner import (
    COLLISION_SCALE_DECAY,
    ITERATIONS,
    exploration_for_iteration,
    _read_iteration_ledger,
    _repair_torn_iteration_ledger,
    load_checkpoint,
    save_checkpoint,
    update_policy,
)
from bplus_v22.remediated_model import RemediatedV22Policy


def main() -> None:
    first = exploration_for_iteration(1)
    assert first.intervention_logit_offset == 3.8027754227
    assert first.conditional_brake_logit_offset == 6.0
    assert first.steer_std_scale == 0.1
    assert exploration_for_iteration(6).intervention_logit_offset == 3.8027754227 * 0.8
    assert exploration_for_iteration(10).intervention_logit_offset == 0.0
    assert exploration_for_iteration(ITERATIONS).conditional_brake_logit_offset == 0.0

    policy = RemediatedV22Policy(ARM_BC_FROZEN, initialization_seed=123)
    critics = B2Critics(hidden_dim=8)
    optimizers = build_b2_optimizers(policy, critics, critic_learning_rate=5e-5)
    dual = OvertakeDual(floor=0.3)
    scale = RunningCollisionScale(decay=COLLISION_SCALE_DECAY)
    scale.normalize(
        torch.tensor([-1.0, 0.0, 1.0, 0.0]),
        update=True,
        event_present=True,
    )
    curriculum = "1" * 64
    manifest = "2" * 64
    plan = "3" * 64
    config = {
        "iterations": 20,
        "episodes_per_iteration": 16,
        "collision_episodes_per_iteration": 8,
        "remaining_episodes_per_iteration": 8,
        "ppo_epochs": 3,
        "minibatch_size": 128,
        "clip_eps": 0.05,
        "action_core_lr": 3e-5,
        "head_lr": 3e-4,
        "sidecar_lr": 3e-6,
        "critic_lr": 5e-5,
        "entropy_coef": 0.001,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
        "replay_float32_atol": 1e-4,
        "collision_scale_decay": 0.99,
        "deterministic_contract": "centered_fresh_prior",
        "dual_freeze_through_iteration": 9,
        "bc_baseline_expected_collision": 24,
        "bc_baseline_expected_overtake": 138,
        "exploration": {
            "intervention_full_offset": 3.8027754227,
            "conditional_brake_full_offset": 6.0,
            "multipliers": [1.0] * 5 + [0.8, 0.6, 0.4, 0.2] + [0.0] * 11,
            "steer_std_scale": 0.1,
            "brake_std_scale": 1.0,
        },
    }
    source_commit = "4" * 40
    occurrence_count = {"L2:test": 48}
    original = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "iter_0003.pt"
        digest = save_checkpoint(
            checkpoint,
            policy=policy,
            critics=critics,
            optimizers=optimizers,
            dual=dual,
            collision_scale=scale,
            arm=ARM_BC_FROZEN,
            seed=0,
            iteration=3,
            curriculum_digest=curriculum,
            training_manifest_sha256=manifest,
            plan_sha256=plan,
            config=config,
            source_commit=source_commit,
            occurrence_count=occurrence_count,
            optimizer_update_count=7,
        )
        assert len(digest) == 64 and checkpoint.is_file()
        expected_random = (random.random(), float(np.random.rand()), float(torch.rand(())))
        behavior = exploration_for_iteration(4).as_batch(torch.zeros(1, 1))
        bc_feature = torch.linspace(-1.0, 1.0, 1680).reshape(1, -1)
        lidar_history = torch.zeros(1, 8, 360)
        scalar_history = torch.zeros(1, 24)
        expected_distribution = policy.behavior_distribution(
            bc_feature, lidar_history, scalar_history, behavior
        )
        key = ActionNoiseKey(0, "L2:test", 48, 0)
        expected_action = expected_distribution.sample_keyed([key])
        expected_log_prob = expected_distribution.log_prob(expected_action)
        expected_entropy = expected_distribution.entropy()
        with torch.no_grad():
            policy.intervention_gate.bias.add_(10.0)
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        iteration, restored_dual, restored_scale, payload = load_checkpoint(
            checkpoint,
            policy=policy,
            critics=critics,
            optimizers=optimizers,
            expected_arm=ARM_BC_FROZEN,
            expected_seed=0,
            expected_curriculum_digest=curriculum,
            expected_training_manifest_sha256=manifest,
            expected_plan_sha256=plan,
            expected_source_commit=source_commit,
            expected_config=config,
            expected_iteration=3,
            expected_occurrence_count=occurrence_count,
            expected_checkpoint_sha256=digest,
        )
        assert iteration == 3 and restored_dual.state_dict() == dual.state_dict()
        assert restored_scale.state_dict() == scale.state_dict()
        assert payload["next_schedule"] == exploration_for_iteration(4).as_dict()
        assert payload["scenario_occurrence_count"] == occurrence_count
        assert payload["optimizer_minibatches_completed"] == 7
        assert all(torch.equal(policy.state_dict()[name], value) for name, value in original.items())
        observed_random = (random.random(), float(np.random.rand()), float(torch.rand(())))
        assert observed_random == expected_random
        observed_distribution = policy.behavior_distribution(
            bc_feature, lidar_history, scalar_history, behavior
        )
        observed_action = observed_distribution.sample_keyed([key])
        assert torch.equal(observed_action.as_tensor(), expected_action.as_tensor())
        assert torch.equal(
            observed_distribution.log_prob(observed_action), expected_log_prob
        )
        assert torch.equal(observed_distribution.entropy(), expected_entropy)

    # A rollout whose macro count leaves a singleton final minibatch must update
    # without re-normalizing that singleton or crashing.
    torch.manual_seed(91)
    count = 129
    update_policy_model = RemediatedV22Policy(
        ARM_BC_FROZEN, initialization_seed=91
    )
    update_critics = B2Critics(hidden_dim=8)
    update_optimizers = build_b2_optimizers(
        update_policy_model, update_critics, critic_learning_rate=5e-5
    )
    with torch.no_grad():
        for parameter in update_policy_model.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.01)
    update_policy_model._b2_frozen_snapshot = update_policy_model.frozen_snapshot()
    bc_feature = torch.randn(count, 1680) * 0.01
    lidar_history = torch.zeros(count, 8, 360)
    scalar_history = torch.zeros(count, 24)
    behavior = exploration_for_iteration(1).as_batch(torch.zeros(count, 1))
    action_rows = []
    old_log_prob_rows = []
    old_entropy_rows = []
    for index in range(count):
        row_distribution = update_policy_model.behavior_distribution(
            bc_feature[index : index + 1],
            lidar_history[index : index + 1],
            scalar_history[index : index + 1],
            exploration_for_iteration(1).as_batch(torch.zeros(1, 1)),
        )
        row_action = row_distribution.sample_keyed(
            [ActionNoiseKey(0, f"L2:{index}", 0, 0)]
        )
        action_rows.append(row_action.as_tensor())
        old_log_prob_rows.append(row_distribution.log_prob(row_action))
        old_entropy_rows.append(row_distribution.entropy())
    synthetic = {
        "bc_feature": bc_feature,
        "lidar_history": lidar_history,
        "scalar_history": scalar_history,
        "privileged_critic_feature": torch.randn(count, 12),
        "latent": torch.cat(action_rows).detach(),
        "old_log_prob": torch.cat(old_log_prob_rows).detach(),
        "old_entropy": torch.cat(old_entropy_rows).detach(),
        "intervention_offset": behavior.intervention_logit_offset[:, 0],
        "conditional_brake_offset": behavior.conditional_brake_logit_offset[:, 0],
        "steer_std_scale": behavior.steer_std_scale[:, 0],
        "brake_std_scale": behavior.brake_std_scale[:, 0],
        "collision_cost": torch.cat([torch.ones(1), torch.zeros(count - 1)]),
        "collision_advantage": torch.linspace(-1.0, 1.0, count),
        "collision_return": torch.linspace(0.0, 1.0, count),
        "performance_advantage": torch.linspace(1.0, -1.0, count),
        "performance_return": torch.linspace(1.0, 0.0, count),
    }
    wrong_context = dict(synthetic)
    wrong_context["intervention_offset"] = synthetic["intervention_offset"] + 0.5
    try:
        update_policy(
            update_policy_model,
            update_critics,
            update_optimizers,
            RunningCollisionScale(decay=COLLISION_SCALE_DECAY),
            wrong_context,
            dual_value=1.0,
            seed=0,
            iteration=1,
        )
        raise AssertionError("wrong serialized behavior context was accepted")
    except AssertionError as error:
        assert "serialized replay log-prob mismatch" in str(error)
    update_scale = RunningCollisionScale(decay=COLLISION_SCALE_DECAY)
    update_result = update_policy(
        update_policy_model,
        update_critics,
        update_optimizers,
        update_scale,
        synthetic,
        dual_value=1.0,
        seed=0,
        iteration=1,
    )
    assert any(row["batch_size"] == 1 for row in update_result["minibatches"])
    assert np.isfinite(update_result["rollout_actor_advantage_std"])
    assert 0.0 < update_result["preupdate_replay_max_abs_log_prob_delta"] <= 1e-4

    # Interrupted and uninterrupted continuation from one exact boundary must
    # produce identical next PPO state and diagnostics.
    resume_dual = OvertakeDual(floor=0.3)
    resume_cursor = {"L2:resume": 16}
    with tempfile.TemporaryDirectory() as directory:
        resume_checkpoint = Path(directory) / "iter_0001.pt"
        resume_digest = save_checkpoint(
            resume_checkpoint,
            policy=update_policy_model,
            critics=update_critics,
            optimizers=update_optimizers,
            dual=resume_dual,
            collision_scale=update_scale,
            arm=ARM_BC_FROZEN,
            seed=0,
            iteration=1,
            curriculum_digest=curriculum,
            training_manifest_sha256=manifest,
            plan_sha256=plan,
            config=config,
            source_commit=source_commit,
            occurrence_count=resume_cursor,
            optimizer_update_count=update_result["updates"],
        )
        behavior2 = exploration_for_iteration(2).as_batch(torch.zeros(count, 1))
        action_rows2 = []
        old_log_prob_rows2 = []
        old_entropy_rows2 = []
        for index in range(count):
            row_distribution = update_policy_model.behavior_distribution(
                bc_feature[index : index + 1],
                lidar_history[index : index + 1],
                scalar_history[index : index + 1],
                exploration_for_iteration(2).as_batch(torch.zeros(1, 1)),
            )
            row_action = row_distribution.sample_keyed(
                [ActionNoiseKey(0, f"L2:{index}", 1, 0)]
            )
            action_rows2.append(row_action.as_tensor())
            old_log_prob_rows2.append(row_distribution.log_prob(row_action))
            old_entropy_rows2.append(row_distribution.entropy())
        synthetic2 = {
            **synthetic,
            "latent": torch.cat(action_rows2).detach(),
            "old_log_prob": torch.cat(old_log_prob_rows2).detach(),
            "old_entropy": torch.cat(old_entropy_rows2).detach(),
            "intervention_offset": behavior2.intervention_logit_offset[:, 0],
            "conditional_brake_offset": behavior2.conditional_brake_logit_offset[:, 0],
            "steer_std_scale": behavior2.steer_std_scale[:, 0],
            "brake_std_scale": behavior2.brake_std_scale[:, 0],
        }
        uninterrupted = update_policy(
            update_policy_model,
            update_critics,
            update_optimizers,
            update_scale,
            synthetic2,
            dual_value=1.0,
            seed=0,
            iteration=2,
        )

        torch.manual_seed(91)
        restored_policy = RemediatedV22Policy(
            ARM_BC_FROZEN, initialization_seed=91
        )
        restored_critics = B2Critics(hidden_dim=8)
        restored_optimizers = build_b2_optimizers(
            restored_policy, restored_critics, critic_learning_rate=5e-5
        )
        restored_policy._b2_frozen_snapshot = restored_policy.frozen_snapshot()
        _, _, restored_update_scale, _ = load_checkpoint(
            resume_checkpoint,
            policy=restored_policy,
            critics=restored_critics,
            optimizers=restored_optimizers,
            expected_arm=ARM_BC_FROZEN,
            expected_seed=0,
            expected_curriculum_digest=curriculum,
            expected_training_manifest_sha256=manifest,
            expected_plan_sha256=plan,
            expected_source_commit=source_commit,
            expected_config=config,
            expected_iteration=1,
            expected_occurrence_count=resume_cursor,
            expected_checkpoint_sha256=resume_digest,
        )
        resumed = update_policy(
            restored_policy,
            restored_critics,
            restored_optimizers,
            restored_update_scale,
            synthetic2,
            dual_value=1.0,
            seed=0,
            iteration=2,
        )
        assert uninterrupted == resumed
        assert update_scale.state_dict() == restored_update_scale.state_dict()
        assert all(
            torch.equal(value, restored_policy.state_dict()[name])
            for name, value in update_policy_model.state_dict().items()
        )
        assert all(
            torch.equal(value, restored_critics.state_dict()[name])
            for name, value in update_critics.state_dict().items()
        )
    with tempfile.TemporaryDirectory() as directory:
        partial = Path(directory)
        ledger = partial / "iterations.jsonl"
        ledger.write_bytes(b'{"iteration":1}\n{"iteration":2')
        _repair_torn_iteration_ledger(partial)
        assert _read_iteration_ledger(ledger) == [{"iteration": 1}]
        assert len(list((partial / "attempt_failures").glob("torn_*.jsonl"))) == 1
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

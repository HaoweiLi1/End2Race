#!/usr/bin/env python3
"""Blocking CPU contracts for B4 plain-End2Race direct-head PPO."""

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import torch

from bplus_v22.b4_direct import (
    B4Config,
    B4Curriculum,
    B4DirectHeadPolicy,
    B4ScenarioSets,
    B4Transition,
    CANONICAL_ACTOR_KEYS,
    FROZEN_B4_CONFIG,
    actor_snapshot_sha256,
    build_batch,
    build_optimizers,
    load_full_checkpoint,
    load_strict_plain_actor,
    project_raw_action,
    replay_metrics,
    save_actor_snapshot,
    save_full_checkpoint,
    strict_plain_actor_from_state,
    update_policy,
    validate_frozen_config,
    weighted_mean_variance,
)
from bplus_v22.b4_eval import B4DeterministicActor, b4_variant, summarize
from bplus_v22.b4_runner import _validate_resume_prefix, expected_b4_plan_config
from bplus_v22.ppo_env import load_b2_scenario_sets
from model import End2Race


REPO = Path(__file__).resolve().parent.parent
BC = REPO / "pretrained/end2race.pth"
TASK8 = REPO / "Experiments/B1_route_r2_scaffold/artifacts/task8_manifests_20260712_113241"
METADATA = (
    REPO
    / "Experiments/A3_d2_representation/artifacts/non_test_full_20260711_175713/episode_metadata.tsv"
)


def synthetic_rollout(
    policy: B4DirectHeadPolicy,
    *,
    action_offset: float = 0.35,
) -> list[B4Transition]:
    generator = torch.Generator().manual_seed(20260713)
    rows: list[B4Transition] = []
    for episode_id in range(16):
        length = 1 + (episode_id % 3)
        for step in range(length):
            feature = torch.randn((1, 1680), generator=generator)
            privileged = torch.randn((1, 12), generator=generator)
            with torch.no_grad():
                mean = policy.mean_from_feature(feature)
                sign = -1.0 if (episode_id + step) % 2 else 1.0
                raw = mean + sign * action_offset * policy.action_std
                old_log_prob = policy.log_prob(mean, raw)
                value = policy.value(privileged)
                executed, delta = project_raw_action(raw)
            terminal = step == length - 1
            reward = 0.0
            if terminal:
                reward = (-2.0, 0.0, 1.0)[episode_id % 3]
            rows.append(
                B4Transition(
                    l2_id=f"L2:{episode_id:064x}",
                    episode_id=episode_id,
                    step_index=step,
                    feature=feature[0].numpy().astype(np.float32),
                    privileged_feature=privileged[0].numpy().astype(np.float32),
                    raw_action=raw[0].numpy().astype(np.float32),
                    executed_action=executed[0].numpy().astype(np.float32),
                    projection_delta=delta[0].numpy().astype(np.float32),
                    old_log_prob=float(old_log_prob.item()),
                    old_value=float(value.item()),
                    reward=reward,
                    terminated=terminal,
                )
            )
    return rows


def synthetic_eval_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_collision = set(range(24))
    baseline_overtake = set(range(24, 162))
    for index in range(288):
        collision = index in baseline_collision
        overtake = index in baseline_overtake
        rows.append(
            {
                "variant": "BC",
                "collision_any": collision,
                "terminal_overtake": overtake,
                "fixed_collision": False,
                "new_collision": False,
                "gained_overtake": False,
                "lost_overtake": False,
                "collision_to_confirmed_pass": False,
                "deterministic_speed_projection_count": 0,
                "deterministic_steer_projection_count": 0,
            }
        )

    # iter10 is feasible but weaker; iter20 hits the product target; iter30 has
    # fewer collisions but seed1 deliberately violates the overtake gate.
    contracts = {
        (1, 10): (21, 135, 5, 2),
        (1, 20): (16, 133, 9, 1),
        (1, 30): (15, 131, 10, 1),
    }
    for (seed, iteration), (collision_count, overtake_count, fixed, new) in contracts.items():
        # Fix the first `fixed` BC collisions and add `new` collisions after
        # the baseline collision range. Fill remaining collision inventory
        # from baseline collisions so the exact total is controlled.
        fixed_indices = set(range(fixed))
        new_indices = set(range(200, 200 + new))
        retained_needed = collision_count - new
        collision_indices = set(range(fixed, fixed + retained_needed)) | new_indices
        assert len(collision_indices) == collision_count
        overtake_indices = set(range(24, 24 + overtake_count))
        for index in range(288):
            bc_collision = index in baseline_collision
            bc_overtake = index in baseline_overtake
            collision = index in collision_indices
            overtake = index in overtake_indices
            rows.append(
                {
                    "variant": b4_variant(seed, iteration),
                    "collision_any": collision,
                    "terminal_overtake": overtake,
                    "fixed_collision": bc_collision and not collision,
                    "new_collision": not bc_collision and collision,
                    "gained_overtake": not bc_overtake and overtake,
                    "lost_overtake": bc_overtake and not overtake,
                    "collision_to_confirmed_pass": False,
                    "deterministic_speed_projection_count": 0,
                    "deterministic_steer_projection_count": 0,
                }
            )
    return rows


def main() -> None:
    torch.manual_seed(7)
    bc_state = torch.load(BC, map_location="cpu", weights_only=True)
    assert tuple(bc_state) == CANONICAL_ACTOR_KEYS
    strict = strict_plain_actor_from_state(bc_state)
    policy = B4DirectHeadPolicy(bc_state)
    assert tuple(policy.actor.state_dict()) == CANONICAL_ACTOR_KEYS
    assert sum(parameter.numel() for parameter in policy.actor.parameters()) == 11_301_482
    assert sum(parameter.numel() for parameter in policy.trainable_actor_parameters) == 706_862
    assert all(
        torch.equal(policy.actor.state_dict()[name], bc_state[name])
        for name in CANONICAL_ACTOR_KEYS
    )
    lidar = torch.randn(2, 3, 360)
    speed = torch.randn(2, 3, 1)
    hidden = torch.randn(1, 2, 1680)
    strict.eval()
    policy.actor.eval()
    with torch.no_grad():
        expected_action, expected_hidden = strict(lidar, speed, hidden)
        actual_action, actual_hidden = policy.actor(lidar, speed, hidden)
    # Separate CPU GRU invocations can differ by a few ulps under oneDNN even
    # with identical tensors; the production same-host trajectory identity is
    # enforced by the staged four-map smoke under the pinned runtime.
    assert torch.allclose(expected_action, actual_action, atol=1e-5, rtol=0.0)
    assert torch.allclose(expected_hidden, actual_hidden, atol=1e-5, rtol=0.0)

    mean = torch.tensor([[0.51, 0.05], [-0.7, 21.0]], dtype=torch.float32)
    raw = torch.tensor([[0.60, -0.1], [-0.7, 21.0]], dtype=torch.float32)
    executed, delta = project_raw_action(raw)
    assert torch.equal(executed, torch.tensor([[0.52, 0.0], [-0.52, 20.0]]))
    assert torch.equal(delta, executed - raw)
    expected_log_prob = torch.distributions.Normal(mean, policy.action_std).log_prob(raw).sum(-1)
    assert torch.allclose(policy.log_prob(mean, raw), expected_log_prob, atol=1e-6, rtol=0.0)

    transitions = synthetic_rollout(policy)
    batch = build_batch(transitions)
    assert batch.size == sum(1 + episode % 3 for episode in range(16))
    assert replay_metrics(policy, batch)["max_abs_ratio_minus_one"] <= 1e-4
    assert torch.isclose(batch.actor_weight.sum(), torch.tensor(float(batch.size)), atol=1e-4)
    for episode_id in range(16):
        selected = batch.episode_id == episode_id
        assert torch.isclose(
            batch.actor_weight[selected].sum(),
            torch.tensor(batch.size / 16),
            atol=1e-5,
        )
    weighted_mean, weighted_variance = weighted_mean_variance(
        batch.normalized_advantage, batch.actor_weight
    )
    assert abs(float(weighted_mean)) < 2e-6
    assert abs(float(weighted_variance) - 1.0) < 2e-5
    # Direct manual check for the first episode's terminal-zero-bootstrap GAE.
    first = transitions[0]
    expected_advantage = first.reward - first.old_value
    assert abs(float(batch.advantage[0]) - expected_advantage) < 1e-6
    episode1_rows = [row for row in transitions if row.episode_id == 1]
    assert len(episode1_rows) == 2
    last_advantage = episode1_rows[1].reward - episode1_rows[1].old_value
    first_delta = (
        episode1_rows[0].reward
        + FROZEN_B4_CONFIG.gamma * episode1_rows[1].old_value
        - episode1_rows[0].old_value
    )
    first_advantage = first_delta + (
        FROZEN_B4_CONFIG.gamma
        * FROZEN_B4_CONFIG.gae_lambda
        * last_advantage
    )
    assert abs(float(batch.advantage[1]) - first_advantage) < 2e-6
    assert 0.66 < (FROZEN_B4_CONFIG.gamma * FROZEN_B4_CONFIG.gae_lambda) ** 100 < 0.68

    # A reward change in episode 0 must not leak into episode 1.
    changed = synthetic_rollout(policy)
    changed[0].reward = 1.0
    changed_batch = build_batch(changed)
    episode1 = batch.episode_id == 1
    assert torch.equal(batch.advantage[episode1], changed_batch.advantage[episode1])

    custom = replace(
        FROZEN_B4_CONFIG,
        actor_lr=1e-2,
        target_weighted_kl=1e-12,
        minibatch_size=8,
    )
    update_policy_model = B4DirectHeadPolicy(bc_state, custom)
    actor_optimizer, critic_optimizer = build_optimizers(update_policy_model)
    update_batch = build_batch(synthetic_rollout(update_policy_model), custom)
    frozen_before = update_policy_model.frozen_state()
    critic_before = {
        name: value.detach().clone()
        for name, value in update_policy_model.critic.state_dict().items()
    }
    output_before = {
        name: value.detach().clone()
        for name, value in update_policy_model.actor.output_layer.state_dict().items()
    }
    update = update_policy(
        update_policy_model,
        update_batch,
        actor_optimizer,
        critic_optimizer,
        seed=1,
        iteration=1,
    )
    assert update["critic_epochs_completed"] == 3
    assert update["critic_optimizer_steps"] == 3 * int(np.ceil(update_batch.size / 8))
    assert update["actor_epochs_completed"] >= 1
    assert len(update["actor_epoch_metrics"]) == update["actor_epochs_completed"]
    assert all(
        {
            "weighted_kl",
            "unweighted_kl",
            "weighted_clip_fraction",
            "unweighted_clip_fraction",
        }.issubset(metrics)
        for metrics in update["actor_epoch_metrics"]
    )
    assert any(
        not torch.equal(output_before[name], value)
        for name, value in update_policy_model.actor.output_layer.state_dict().items()
    )
    assert any(
        not torch.equal(critic_before[name], value)
        for name, value in update_policy_model.critic.state_dict().items()
    )
    assert all(
        torch.equal(frozen_before[name], value)
        for name, value in update_policy_model.frozen_state().items()
    )
    # A tiny KL target plus the aggressive test LR should stop actor epochs,
    # while the assertion above proves all critic epochs still ran.
    assert update["actor_stopped_early"] is True

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        actor_path = root / "actor.pth"
        record = save_actor_snapshot(update_policy_model, actor_path)
        loaded = load_strict_plain_actor(actor_path)
        assert tuple(loaded.state_dict()) == CANONICAL_ACTOR_KEYS
        assert record["tensor_sha256"] == actor_snapshot_sha256(loaded.state_dict())
        full = root / "full.pt"
        plan_sha = "a" * 64
        curriculum_sha = "b" * 64
        save_full_checkpoint(
            update_policy_model,
            actor_optimizer,
            critic_optimizer,
            full,
            completed_iteration=1,
            seed=1,
            run_plan_sha256=plan_sha,
            curriculum_sha256=curriculum_sha,
        )
        restored = B4DirectHeadPolicy(bc_state, custom)
        restored_actor_opt, restored_critic_opt = build_optimizers(restored)
        completed = load_full_checkpoint(
            full,
            restored,
            restored_actor_opt,
            restored_critic_opt,
            expected_seed=1,
            expected_run_plan_sha256=plan_sha,
            expected_curriculum_sha256=curriculum_sha,
            restore_rng=False,
        )
        assert completed == 1
        assert all(
            torch.equal(restored.actor.state_dict()[name], value)
            for name, value in update_policy_model.actor.state_dict().items()
        )
        try:
            load_strict_plain_actor(full)
            raise RuntimeError("B4 strict deployment loader accepted a full checkpoint")
        except Exception:
            pass

        # Resume validation binds iteration 0 to the canonical BC tensor hash,
        # independently of later committed iteration hashes.
        resume_root = root / "resume"
        actor0 = resume_root / "actors/iter_0000.pth"
        save_actor_snapshot(policy, actor0)
        checkpoint0 = resume_root / "checkpoints/iter_0000.pt"
        checkpoint0.parent.mkdir(parents=True)
        checkpoint0.write_bytes(b"full-checkpoint-placeholder")
        canonical_tensor_sha = actor_snapshot_sha256(policy.actor_state())
        _validate_resume_prefix(
            resume_root,
            (),
            expected_bc_actor_tensor_sha256=canonical_tensor_sha,
        )
        drifted_actor0 = torch.load(actor0, map_location="cpu", weights_only=True)
        drifted_actor0["output_layer.2.bias"] = (
            drifted_actor0["output_layer.2.bias"] + 1.0
        )
        torch.save(drifted_actor0, actor0)
        try:
            _validate_resume_prefix(
                resume_root,
                (),
                expected_bc_actor_tensor_sha256=canonical_tensor_sha,
            )
            raise RuntimeError("B4 resume accepted a non-BC iteration-0 actor")
        except ValueError as error:
            assert "iteration-0 actor" in str(error)

    scenario_sets = B4ScenarioSets.from_b2(load_b2_scenario_sets(TASK8, METADATA))
    assert (len(scenario_sets.collision), len(scenario_sets.overtake), len(scenario_sets.follow)) == (
        81,
        1001,
        558,
    )
    digests = []
    for seed in (1,):
        curriculum = B4Curriculum(scenario_sets, seed)
        plan = curriculum.plan()
        assert len(plan) == 30 and all(len(rows) == 16 for rows in plan)
        for rows in plan:
            outcomes = [row.archived_bc_outcome for row in rows]
            assert outcomes.count("collision") == 6
            assert outcomes.count("overtake") == 6
            assert outcomes.count("follow") == 4
        digests.append(curriculum.digest())
    assert digests == [
        "40275f3d928b753fdc683ca20df83ad4097d9e8ac3c92f4a150fba3a50a5afa1",
    ]
    plan_config = expected_b4_plan_config(TASK8, METADATA)
    validate_frozen_config(plan_config["ppo"])
    assert plan_config["curriculum_sha256_by_seed"] == {"1": digests[0]}
    assert "sidecar" not in plan_config["inputs"]
    assert "sidecar" in plan_config["forbidden_inputs"]

    summary = summarize(synthetic_eval_rows())
    assert summary["selected_iteration"] == 20
    assert summary["selected_pair_verdict"] == "OPENED_DEVELOPMENT_PRODUCT_TARGET_HIT"
    assert summary["same_iteration_snapshots"]["iter10"]["feasible"] is True
    assert summary["same_iteration_snapshots"]["iter20"]["terminal_overtake"] == 133
    assert summary["same_iteration_snapshots"]["iter20"]["collision"] == 16
    assert summary["same_iteration_snapshots"]["iter30"]["feasible"] is False
    assert summary["automatic_b3_fallback_authorized"] is False
    assert summary["snapshot_selection_performed"] is True
    assert summary["architecture_arm_selection_performed"] is False

    class ConstantActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = End2Race().gru

        def forward(self, lidar, speed, hidden):
            action = torch.tensor([[[0.7, 21.0]]], dtype=lidar.dtype)
            return action, hidden

    adapter = B4DeterministicActor(ConstantActor())
    action, _ = adapter(torch.zeros(1, 1, 360), torch.zeros(1, 1, 1), torch.zeros(1, 1, 1680))
    adapter.observe_applied_command(0.52, float(action[0, 0, 1]))
    accounting = adapter.accounting()
    assert accounting["deterministic_speed_projection_count"] == 1
    assert accounting["deterministic_steer_projection_count"] == 1

    print("B4 direct-head PPO blocking contracts passed")


if __name__ == "__main__":
    main()

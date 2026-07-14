"""B7 recurrent PPO unit and mechanism regressions."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import torch

from bplus_v22.b7_eval import exact_cluster_signflip_one_sided
from bplus_v22.b7_recurrent import (
    B7Episode,
    B7RecurrentPolicy,
    B7ScenarioSampler,
    B7Transition,
    FROZEN_B7_CONFIG,
    build_batch,
    build_optimizers,
    collision_reward_schedule,
    task_reward_schedule,
    update_policy,
)
from bplus_v22.ppo_env import B2Scenario
from model import End2Race


def _scenario(index: int, outcome: str, map_name: str | None = None) -> B2Scenario:
    return B2Scenario(
        training_order=index,
        l2_id=f"L2:{index:064x}",
        l4_id=f"L4:{index:064x}",
        map_name=map_name or ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")[index % 4],
        skill="skill_S",
        opponent_raceline="raceline2",
        speedscale=0.8,
        resolved_ego_idx=index,
        bc_collision_any=outcome == "collision",
        archived_bc_outcome=outcome,
    )


def _population() -> tuple[B2Scenario, ...]:
    rows = []
    for index in range(1640):
        outcome = "collision" if index < 160 else ("overtake" if index < 1000 else "follow")
        rows.append(_scenario(index, outcome))
    return tuple(rows)


def test_reward_redistribution_discount_equivalence():
    config = FROZEN_B7_CONFIG
    rewards = collision_reward_schedule(801, config)
    assert np.count_nonzero(rewards) == 100
    distributed = sum(
        config.gamma**offset * float(rewards[801 - 100 + offset]) for offset in range(100)
    )
    terminal_only = -2.0 * config.gamma**99
    assert abs(distributed - terminal_only) < 1e-6
    safe = task_reward_schedule(801, collision_any=False, terminal_overtake=True)
    assert np.count_nonzero(safe) == 1 and safe[-1] == 1.0
    collision = task_reward_schedule(801, collision_any=True, terminal_overtake=True)
    assert np.array_equal(collision, rewards)


def test_sampler_counts_uniqueness_and_hard_priority():
    sampler = B7ScenarioSampler(_population(), 1)
    first = sampler.select(1, None)
    assert len(first) == len({row.scenario.l2_id for row in first}) == 32
    assert sum(row.role == "representative" for row in first) == 16
    assert sum(row.role == "archived_collision" for row in first) == 8
    assert sum(row.role == "current_hard" for row in first) == 8
    assert all(row.hard_priority == 0 for row in first if row.role == "current_hard")
    representative_maps = [row.scenario.map_name for row in first if row.role == "representative"]
    assert all(representative_maps.count(map_name) == 4 for map_name in set(representative_maps))

    previous = {}
    for row in first:
        archived = row.scenario.archived_bc_outcome
        previous[row.scenario.l2_id] = "collision" if archived != "collision" else "follow"
    second = sampler.select(2, previous)
    assert len(second) == len({row.scenario.l2_id for row in second}) == 32
    priorities = [row.hard_priority for row in second if row.role == "current_hard"]
    assert priorities == sorted(priorities)
    assert any(value in {1, 4} for value in priorities)
    assert sampler.select(1, None) == first


def _synthetic_batch(policy: B7RecurrentPolicy, config=FROZEN_B7_CONFIG):
    generator = torch.Generator().manual_seed(717001)
    episodes = []
    for episode_index in range(config.episodes_per_iteration):
        length = 2
        lidar = torch.rand((length, 360), generator=generator) * 20.0
        speed = torch.rand((length,), generator=generator) * 6.0
        with torch.no_grad():
            mean = policy.sequence_means(lidar, speed)
        raw = mean + torch.tensor([[0.005, -0.03], [-0.004, 0.02]])
        log_prob = policy.log_prob(mean, raw)
        privileged = torch.randn((length, 13), generator=generator)
        with torch.no_grad():
            value = policy.value(privileged)
        candidate = ("collision", "overtake", "follow")[episode_index % 3]
        archived = "collision" if episode_index % 4 == 0 else (
            "overtake" if episode_index % 2 else "follow"
        )
        scenario = _scenario(10_000 + episode_index, archived)
        rewards = task_reward_schedule(
            length,
            collision_any=candidate == "collision",
            terminal_overtake=candidate == "overtake",
        )
        transitions = []
        for step in range(length):
            executed = raw[step].clone()
            executed[0] = torch.clamp(executed[0], -0.52, 0.52)
            executed[1] = torch.clamp(executed[1], 0.0, 20.0)
            transitions.append(
                B7Transition(
                    step_index=step,
                    lidar=lidar[step].numpy().astype(np.float32),
                    previous_speed=float(speed[step]),
                    privileged_feature=privileged[step].numpy().astype(np.float32),
                    old_mean=mean[step].numpy().astype(np.float32),
                    bc_mean=mean[step].numpy().astype(np.float32),
                    raw_action=raw[step].numpy().astype(np.float32),
                    executed_action=executed.numpy().astype(np.float32),
                    projection_delta=(executed - raw[step]).numpy().astype(np.float32),
                    old_log_prob=float(log_prob[step]),
                    old_value=float(value[step]),
                    reward=float(rewards[step]),
                    terminated=step == length - 1,
                )
            )
        episodes.append(
            B7Episode(
                scenario=scenario,
                episode_id=episode_index,
                sampler_role=("representative", "archived_collision", "current_hard")[
                    episode_index % 3
                ],
                hard_priority=0 if episode_index % 3 == 2 else None,
                transitions=tuple(transitions),
                collision_any=candidate == "collision",
                terminal_overtake=candidate == "overtake",
                corrected_outcome3=candidate,
                terminal_reason=(
                    "any_agent_collision" if candidate == "collision" else "product_horizon"
                ),
            )
        )
    return build_batch(episodes, config)


def _policy(config=FROZEN_B7_CONFIG):
    torch.manual_seed(717000)
    actor = End2Race(mask_prob=0.0, hidden_scale=4)
    return B7RecurrentPolicy(actor.state_dict(), config)


def test_one_recurrent_actor_step_and_isolated_weighted_critic():
    config = replace(FROZEN_B7_CONFIG, head_lr=1e-6, gru_lr=1e-7)
    policy = _policy(config)
    batch = _synthetic_batch(policy, config)
    actor_optimizer, critic_optimizer = build_optimizers(policy)
    actor_before = {name: value.clone() for name, value in policy.actor.state_dict().items()}
    critic_before = {name: value.clone() for name, value in policy.critic.state_dict().items()}
    result = update_policy(
        policy,
        batch,
        actor_optimizer,
        critic_optimizer,
        seed=1,
        iteration=1,
        consecutive_rejections=0,
    )
    assert result["actor_update_accepted"] is True
    assert result["actor_optimizer_steps_attempted"] == 1
    assert result["actor_optimizer_steps_committed"] == 1
    assert result["critic_epochs_completed"] == 3
    assert any(
        not torch.equal(actor_before[name], value)
        for name, value in policy.actor.state_dict().items()
        if name.startswith("gru.")
    )
    assert any(
        not torch.equal(actor_before[name], value)
        for name, value in policy.actor.state_dict().items()
        if name.startswith("output_layer.")
    )
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in policy.actor.state_dict().items()
        if not name.startswith("gru.") and not name.startswith("output_layer.")
    )
    assert any(
        not torch.equal(critic_before[name], value)
        for name, value in policy.critic.state_dict().items()
    )


def test_consecutive_iterations_clear_stale_critic_gradients():
    config = replace(FROZEN_B7_CONFIG, head_lr=1e-6, gru_lr=1e-7)
    policy = _policy(config)
    actor_optimizer, critic_optimizer = build_optimizers(policy)
    first = update_policy(
        policy,
        _synthetic_batch(policy, config),
        actor_optimizer,
        critic_optimizer,
        seed=1,
        iteration=1,
        consecutive_rejections=0,
    )
    assert first["critic_epochs_completed"] == 3
    assert any(parameter.grad is not None for parameter in policy.critic.parameters())
    second = update_policy(
        policy,
        _synthetic_batch(policy, config),
        actor_optimizer,
        critic_optimizer,
        seed=1,
        iteration=2,
        consecutive_rejections=0,
    )
    assert second["critic_epochs_completed"] == 3
    assert second["actor_optimizer_steps_attempted"] == 1


def test_rejected_step_restores_actor_and_adam_then_halves_lrs():
    config = replace(
        FROZEN_B7_CONFIG,
        head_lr=1e-2,
        gru_lr=1e-3,
        safe_kl_cap=1e-12,
        rollout_kl_cap=1e-12,
    )
    policy = _policy(config)
    actor_optimizer, critic_optimizer = build_optimizers(policy)
    # Prime nonempty Adam moments so rollback covers exp_avg, exp_avg_sq and
    # step counters, not only a pristine optimizer.
    for parameter in policy.trainable_actor_parameters:
        parameter.grad = torch.full_like(parameter, 1e-8)
    actor_optimizer.step()
    actor_optimizer.zero_grad(set_to_none=True)
    batch = _synthetic_batch(policy, config)
    actor_before = {name: value.clone() for name, value in policy.actor.state_dict().items()}
    optimizer_before = copy.deepcopy(actor_optimizer.state_dict())
    result = update_policy(
        policy,
        batch,
        actor_optimizer,
        critic_optimizer,
        seed=1,
        iteration=1,
        consecutive_rejections=0,
    )
    assert result["actor_update_accepted"] is False
    assert result["actor_optimizer_steps_committed"] == 0
    assert result["critic_epochs_completed"] == 3
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in policy.actor.state_dict().items()
    )
    optimizer_after = actor_optimizer.state_dict()
    assert optimizer_after["state"].keys() == optimizer_before["state"].keys()
    for parameter_id, before in optimizer_before["state"].items():
        after = optimizer_after["state"][parameter_id]
        assert after.keys() == before.keys()
        for name, value in before.items():
            assert torch.equal(after[name], value) if torch.is_tensor(value) else after[name] == value
    lrs = {group["role"]: group["lr"] for group in actor_optimizer.param_groups}
    assert lrs == {"gru": config.gru_lr / 2, "head": config.head_lr / 2}


def test_exact_l4_cluster_signflip():
    assert exact_cluster_signflip_one_sided([1, 1]) == 0.25
    assert exact_cluster_signflip_one_sided([0, 0]) == 1.0
    assert exact_cluster_signflip_one_sided([2, -1]) == 0.5


if __name__ == "__main__":
    test_reward_redistribution_discount_equivalence()
    test_sampler_counts_uniqueness_and_hard_priority()
    test_one_recurrent_actor_step_and_isolated_weighted_critic()
    test_consecutive_iterations_clear_stale_critic_gradients()
    test_rejected_step_restores_actor_and_adam_then_halves_lrs()
    test_exact_l4_cluster_signflip()
    print("B7 recurrent PPO contracts passed")

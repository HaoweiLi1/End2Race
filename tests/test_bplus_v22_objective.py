#!/usr/bin/env python3
"""Dual update and direct-outcome lexicographic selection regression."""

import torch

from bplus_v22 import ARM_BC_FROZEN, ARM_SIDECAR_FINETUNE, ARM_SIDECAR_FROZEN
from bplus_v22.objective import (
    DirectOutcomeCandidate,
    OvertakeDual,
    SeedDirectOutcome,
    allowed_overtake_loss,
    clip_gradient_group,
    constrained_policy_advantage,
    detached_actor_objective,
    separate_critic_losses,
    select_candidate,
)
from bplus_v22.buffer import EpisodeOutcomeSignals, EpisodeOutcomeStore


def candidate(arm, snapshot, fixed, new, gained, lost, confirmed=1, n=288, **kwargs):
    n0 = n // 2
    n1 = n - n0
    fixed0, new0 = fixed // 2, new // 2
    gained0, lost0 = gained // 2, lost // 2
    seeds = (
        SeedDirectOutcome(0, n0, fixed0, new0, gained0, lost0),
        SeedDirectOutcome(
            1,
            n1,
            fixed - fixed0,
            new - new0,
            gained - gained0,
            lost - lost0,
        ),
    )
    return DirectOutcomeCandidate(
        arm=arm,
        snapshot_iteration=snapshot,
        paired_episode_count=n,
        fixed_collision=fixed,
        new_collision=new,
        gained_overtake=gained,
        lost_overtake=lost,
        collision_to_confirmed_pass=confirmed,
        seed_outcomes=seeds,
        **kwargs,
    )


def main() -> None:
    assert allowed_overtake_loss(96) == 0
    assert allowed_overtake_loss(100) == 1
    assert allowed_overtake_loss(288) == 2
    dual = OvertakeDual(floor=0.50)
    assert dual.value == 1.0
    first = dual.update_with_record(0.40, 16)
    assert first.value_after == 1.0 and first.ema_after is None and not first.updated
    second = dual.update_with_record(0.40, 16)
    assert abs(second.value_after - 1.05) < 1e-12 and second.updated
    assert tuple(second.ordered_log()) == (
        "completed_episodes_before",
        "completed_episodes_after",
        "observed_overtake_rate",
        "ema_before",
        "ema_after",
        "value_before",
        "value_after",
        "updated",
    )
    restored = OvertakeDual.from_state_dict(dual.state_dict())
    assert restored.state_dict() == dual.state_dict()
    for _ in range(100):
        dual.update(0.0, 32)
    assert dual.value == 3.0
    for _ in range(200):
        dual.update(1.0, 32)
    assert dual.value == 0.0

    collision = torch.tensor([2.0, 0.0, 1.0, 3.0])
    performance = torch.tensor([0.0, 2.0, 3.0, 1.0])
    combined = constrained_policy_advantage(collision, performance, dual_value=1.0)
    assert combined.shape == collision.shape and torch.all(torch.isfinite(combined))
    assert abs(float(combined.mean())) < 1e-6

    # Actor optimization differentiates only through actor log-probabilities.
    actor_logits = torch.tensor([0.2, -0.1, 0.4, -0.3], requires_grad=True)
    privileged = torch.arange(8.0).reshape(4, 2).detach().requires_grad_()
    collision_from_critic = privileged[:, 0] * 2.0
    performance_from_critic = privileged[:, 1] * 3.0
    alarm = torch.ones(4, requires_grad=True)
    actor_loss = detached_actor_objective(
        actor_logits, collision_from_critic, performance_from_critic, 1.0
    )
    actor_loss.backward()
    assert actor_logits.grad is not None
    assert privileged.grad is None and alarm.grad is None

    predictions = {
        "reward": torch.tensor([1.0, 2.0], requires_grad=True),
        "collision": torch.tensor([3.0, 4.0], requires_grad=True),
        "performance": torch.tensor([5.0, 6.0], requires_grad=True),
    }
    targets = {
        "reward": torch.tensor([0.0, 0.0]),
        "collision": torch.tensor([3.0, 3.0]),
        "performance": torch.tensor([7.0, 7.0]),
    }
    losses = separate_critic_losses(predictions, targets)
    assert set(losses) == set(predictions)
    assert float(losses["reward"]) == 2.5
    assert float(losses["collision"]) == 0.5
    assert float(losses["performance"]) == 2.5

    actor_parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    collision_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    actor_parameter.grad = torch.tensor([3.0, 4.0])
    collision_parameter.grad = torch.tensor([2.0])
    actor_norm = clip_gradient_group("actor", [actor_parameter], 1.0)
    critic_norm = clip_gradient_group("collision_critic", [collision_parameter], 0.5)
    assert actor_norm.pre_clip == 5.0 and actor_norm.post_clip <= 1.000001
    assert critic_norm.pre_clip == 2.0 and critic_norm.post_clip <= 0.500001

    store = EpisodeOutcomeStore()
    store.add(EpisodeOutcomeSignals(True, True, False, False, 1.25))
    store.add(EpisodeOutcomeSignals(False, False, True, True, 2.5))
    columns = store.as_columns()
    assert tuple(columns) == (
        "any_agent_collision",
        "ego_involved_collision",
        "terminal_overtake",
        "confirmed_safe_pass",
        "progress",
    )
    assert columns["any_agent_collision"].tolist() == [True, False]
    assert columns["confirmed_safe_pass"].tolist() == [False, True]

    # N=288 permits two net losses; three fails the declared 1pp tolerance.
    passing_tolerance = candidate(ARM_BC_FROZEN, 20, 8, 1, 0, 2)
    failed_overtake = candidate(ARM_BC_FROZEN, 20, 8, 1, 0, 3)
    failed_collision = candidate(ARM_SIDECAR_FROZEN, 20, 2, 2, 1, 0)
    passing_early = candidate(ARM_SIDECAR_FROZEN, 20, 6, 2, 2, 1)
    passing_late = candidate(ARM_SIDECAR_FINETUNE, 40, 10, 1, 2, 1)
    assert passing_tolerance.passes
    assert not failed_overtake.passes and not failed_collision.passes
    assert passing_early.passes and passing_late.passes
    # Equal net overtake; larger net collision improvement wins despite later snapshot.
    assert select_candidate([failed_overtake, passing_early, passing_late]) == passing_late
    more_overtake_less_safety = candidate(ARM_BC_FROZEN, 20, 5, 2, 5, 0)
    # Once both pass the overtake floor, the larger collision improvement wins.
    assert select_candidate([more_overtake_less_safety, passing_late]) == passing_late
    # Equal direct outcomes prefer the earlier snapshot.
    tied_late = candidate(ARM_SIDECAR_FINETUNE, 40, 6, 2, 2, 1)
    assert select_candidate([passing_early, tied_late]) == passing_early
    collapsed = candidate(
        ARM_SIDECAR_FINETUNE, 20, 10, 0, 3, 0, map_or_skill_collapse=True
    )
    assert not collapsed.passes
    assert select_candidate([collapsed, failed_overtake]) is None
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""B2 two-head clipped PPO and optimizer-isolation regressions."""

import math

import torch
import torch.nn as nn

from bplus_v22.ppo import (
    B2Critics,
    PolicyReplayTerms,
    RunningCollisionScale,
    b2_constrained_advantage,
    build_b2_optimizers,
    clipped_policy_objective,
    compute_b2_losses,
    two_head_critic_losses,
)


def assert_raises(kind, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
        raise AssertionError(f"expected {kind.__name__}")
    except kind:
        pass


def tensor_batch(count=4):
    return {
        "bc_feature": torch.zeros(count, 1680),
        "lidar_history": torch.zeros(count, 8, 360),
        "scalar_history": torch.zeros(count, 24),
        "privileged_critic_feature": torch.randn(count, 12, requires_grad=True),
        "latent": torch.zeros(count, 4),
        "old_log_prob": torch.zeros(count),
        "old_entropy": torch.ones(count),
        "intervention_offset": torch.full((count,), 1.25),
        "conditional_brake_offset": torch.full((count,), 2.5),
        "steer_std_scale": torch.full((count,), 0.1),
        "brake_std_scale": torch.ones(count),
        "collision_cost": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "collision_advantage": torch.tensor([0.0, 1.0, 0.0, -1.0], requires_grad=True),
        "collision_return": torch.tensor([0.0, 1.0, 0.0, 0.0], requires_grad=True),
        "performance_advantage": torch.tensor([-1.0, 0.0, 1.0, 0.0], requires_grad=True),
        "performance_return": torch.tensor([0.0, 0.0, 1.0, 1.0], requires_grad=True),
    }


class DummyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Parameter(torch.tensor([0.0]))
        self.sidecar = nn.Parameter(torch.tensor([0.0]))
        self.frozen = nn.Parameter(torch.tensor([0.0]), requires_grad=False)

    def optimizer_parameter_groups(self):
        return [
            {"name": "action_core", "params": [self.action], "lr": 3e-5},
            {"name": "sidecar", "params": [self.sidecar], "lr": 3e-6},
        ]


def main() -> None:
    # Hand-computed clipping covers positive and negative advantages on both sides.
    ratio = torch.tensor([1.2, 0.8, 1.2, 0.8])
    result = clipped_policy_objective(
        ratio.log(),
        torch.zeros(4),
        torch.tensor([1.0, 1.0, -1.0, -1.0]),
        0.1,
    )
    # min surrogates: [1.1, 0.8, -1.2, -0.9], mean=-0.05, loss=+0.05.
    assert torch.allclose(result.loss, torch.tensor(0.05), atol=1e-6)
    assert torch.allclose(result.ratio, ratio)
    assert torch.allclose(result.clip_fraction, torch.tensor(1.0))
    assert float(result.approx_kl) > 0.0

    scale = RunningCollisionScale(decay=0.9)
    empty, empty_record = scale.normalize(
        torch.zeros(4), update=True, event_present=False
    )
    assert torch.equal(empty, torch.zeros(4))
    assert not empty_record.informative and scale.variance is None
    informative, info_record = scale.normalize(
        torch.tensor([-1.0, 0.0, 1.0, 0.0]),
        update=True,
        event_present=True,
    )
    assert info_record.informative and scale.variance is not None
    before = scale.variance
    scale.normalize(torch.zeros(4), update=True, event_present=False)
    assert scale.variance == before  # collision-empty batch cannot collapse the scale
    restored = RunningCollisionScale.from_state_dict(scale.state_dict())
    assert restored.state_dict() == scale.state_dict()
    combined, _ = b2_constrained_advantage(
        torch.zeros(4),
        torch.tensor([-1.0, 0.0, 1.0, 0.0]),
        1.0,
        restored,
        update_collision_scale=False,
        collision_event_present=False,
    )
    assert torch.all(torch.isfinite(combined)) and not combined.requires_grad

    # Exactly two critics exist, and each loss updates only its own head.
    critics = B2Critics(hidden_dim=8)
    assert not hasattr(critics, "reward")
    privileged = torch.randn(5, 12)
    predictions = critics(privileged)
    targets = {
        "collision": torch.ones(5, requires_grad=True),
        "performance": torch.zeros(5, requires_grad=True),
    }
    critic_losses = two_head_critic_losses(predictions, targets)
    assert tuple(critic_losses) == ("collision", "performance")
    critic_losses["collision"].backward()
    assert any(parameter.grad is not None for parameter in critics.collision.parameters())
    assert all(parameter.grad is None for parameter in critics.performance.parameters())
    assert targets["collision"].grad is None
    assert_raises(
        ValueError,
        two_head_critic_losses,
        {**predictions, "reward": torch.zeros(5)},
        targets,
    )

    # Replay hook is forced to consume both offsets saved in the rollout.
    batch = tensor_batch()
    actor_parameter = nn.Parameter(torch.tensor(0.0))
    hook_seen = {}

    def replay_hook(**kwargs):
        hook_seen.update(kwargs)
        assert set(kwargs) == {
            "bc_feature",
            "lidar_history",
            "scalar_history",
            "latent",
            "intervention_offset",
            "conditional_brake_offset",
            "steer_std_scale",
            "brake_std_scale",
        }
        log_prob = batch["old_log_prob"] + actor_parameter
        entropy = torch.full_like(log_prob, 0.25) + actor_parameter * 0.0
        return PolicyReplayTerms(log_prob, entropy)

    loss_critics = B2Critics(hidden_dim=8)
    loss_scale = RunningCollisionScale(decay=0.9)
    losses = compute_b2_losses(
        loss_critics,
        batch,
        replay_hook,
        loss_scale,
        dual_value=1.0,
        clip_epsilon=0.05,
        entropy_coefficient=0.001,
        update_collision_scale=True,
    )
    assert torch.equal(hook_seen["intervention_offset"], torch.full((4,), 1.25))
    assert torch.equal(hook_seen["conditional_brake_offset"], torch.full((4,), 2.5))
    assert torch.equal(losses.ratio, torch.ones(4))
    losses.actor_loss.backward()
    assert actor_parameter.grad is not None
    assert batch["collision_advantage"].grad is None
    assert batch["performance_advantage"].grad is None
    assert batch["privileged_critic_feature"].grad is None
    assert all(parameter.grad is None for parameter in loss_critics.parameters())
    assert set(losses.critic_losses) == {"collision", "performance"}

    # Critic targets stay detached and each B2 critic remains isolated here too.
    losses.critic_losses["performance"].backward()
    assert all(parameter.grad is None for parameter in loss_critics.collision.parameters())
    assert any(parameter.grad is not None for parameter in loss_critics.performance.parameters())
    assert batch["performance_return"].grad is None

    # Exact optimizer inventory: actor groups and the two heads never overlap.
    policy = DummyPolicy()
    optimizer_critics = B2Critics(hidden_dim=8)
    optimizers = build_b2_optimizers(
        policy, optimizer_critics, critic_learning_rate=5e-5
    )
    assert optimizers.actor_group_names == ("action_core", "sidecar")
    assert tuple(optimizers.state_dict()) == (
        "actor",
        "collision_critic",
        "performance_critic",
    )
    optimizer_ids = [
        {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        for optimizer in (
            optimizers.actor,
            optimizers.collision_critic,
            optimizers.performance_critic,
        )
    ]
    assert not (optimizer_ids[0] & optimizer_ids[1])
    assert not (optimizer_ids[0] & optimizer_ids[2])
    assert not (optimizer_ids[1] & optimizer_ids[2])

    # Shape and nonfinite failures are fail-closed rather than silently broadcast.
    assert_raises(
        ValueError,
        clipped_policy_objective,
        torch.zeros(3),
        torch.zeros(4),
        torch.zeros(4),
        0.05,
    )
    bad_batch = tensor_batch()
    bad_batch["old_log_prob"][0] = math.nan
    assert_raises(
        ValueError,
        compute_b2_losses,
        B2Critics(hidden_dim=8),
        bad_batch,
        replay_hook,
        RunningCollisionScale(),
        dual_value=1.0,
        clip_epsilon=0.05,
        entropy_coefficient=0.001,
        update_collision_scale=True,
    )
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

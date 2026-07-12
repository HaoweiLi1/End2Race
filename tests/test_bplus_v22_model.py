#!/usr/bin/env python3
"""A/B/C initialization, hurdle math, and parameter-isolation regression."""

import copy

import torch
import torch.nn.functional as F

from bplus_v22 import (
    ACTION_CORE_LR,
    ARM_BC_FROZEN,
    ARM_SIDECAR_FINETUNE,
    ARM_SIDECAR_FROZEN,
    SIDECAR_FINETUNE_LR,
)
from bplus_v22.model import MacroResidualAction, V22Policy
from d2r.model import D2RGeometryNet
from model import End2Race


def equal_state(left, right):
    return set(left) == set(right) and all(torch.equal(left[k], right[k]) for k in left)


def train_two_steps(policy, bc_feature, lidar_history, scalar_history):
    snapshot = policy.frozen_snapshot()
    shadow_hash = policy.shadow_sha256()
    sidecar_hash = policy.policy_sidecar_encoder_sha256()
    groups = policy.optimizer_parameter_groups()
    optimizer = torch.optim.Adam(groups)
    target = MacroResidualAction(
        torch.full((len(bc_feature), 1), 0.5),
        torch.ones(len(bc_feature), 1),
        torch.full((len(bc_feature), 1), 0.2),
    )
    policy.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        distribution = policy.distribution(bc_feature, lidar_history, scalar_history)
        loss = -distribution.log_prob(target).mean()
        loss.backward()
        optimizer.step()
    policy.assert_frozen_unchanged(snapshot)
    assert policy.shadow_sha256() == shadow_hash
    return sidecar_hash, policy.policy_sidecar_encoder_sha256(), groups


def main() -> None:
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        bc_state = copy.deepcopy(End2Race(mask_prob=0.0, hidden_scale=4).state_dict())
        sidecar_state = copy.deepcopy(D2RGeometryNet().state_dict())
    policies = {
        arm: V22Policy(
            arm,
            bc_state_dict=bc_state,
            sidecar_state_dict=sidecar_state,
            initialization_seed=20260711,
        )
        for arm in (ARM_BC_FROZEN, ARM_SIDECAR_FROZEN, ARM_SIDECAR_FINETUNE)
    }
    a, b, c = (policies[key] for key in policies)
    assert equal_state(a.action_core.state_dict(), b.action_core.state_dict())
    assert equal_state(b.action_core.state_dict(), c.action_core.state_dict())
    assert equal_state(b.steer_mean.state_dict(), c.steer_mean.state_dict())
    assert b.policy_sidecar_encoder_sha256() == c.policy_sidecar_encoder_sha256()
    assert len({policy.shadow_sha256() for policy in policies.values()}) == 1

    batch = 2
    bc_feature = torch.randn(batch, 1680)
    lidar_history = torch.rand(batch, 8, 360)
    scalar_history = torch.randn(batch, 24)
    base = torch.randn(batch, 2)
    diagnostics = []
    for policy in policies.values():
        distribution = policy.distribution(bc_feature, lidar_history, scalar_history)
        action = distribution.deterministic()
        assert torch.equal(action.brake_gate, torch.zeros_like(action.brake_gate))
        delta = distribution.physical_delta(action)
        assert torch.equal(delta, torch.zeros_like(delta))
        assert torch.equal(policy.compose(base, action), base)
        assert torch.all(distribution.brake_probability < 0.01)
        diagnostics.append(policy.diagnostic(bc_feature, lidar_history, scalar_history))

    # Fixed-window pooling is exactly the locked 360 -> 18 adaptive pool,
    # while providing a deterministic CUDA backward for arm C.
    encoded = b.policy_sidecar.encode_beams(lidar_history, pool=False)
    assert torch.equal(
        b.policy_sidecar.beam_pool(encoded),
        F.avg_pool1d(encoded, kernel_size=20, stride=20),
    )
    for name in diagnostics[0]:
        assert torch.equal(diagnostics[0][name], diagnostics[1][name])
        assert torch.equal(diagnostics[1][name], diagnostics[2][name])

    distribution = a.distribution(bc_feature, lidar_history, scalar_history)
    sampled = distribution.sample()
    stored = sampled.as_tensor()
    restored = MacroResidualAction.from_tensor(stored)
    assert torch.equal(restored.steer_latent, sampled.steer_latent)
    manual = distribution.steer.log_prob(sampled.steer_latent)
    manual += distribution.gate.log_prob(sampled.brake_gate)
    manual += sampled.brake_gate * distribution.brake.log_prob(sampled.brake_latent)
    assert torch.allclose(distribution.log_prob(sampled), manual.squeeze(-1))
    entropy = distribution.steer.entropy() + distribution.gate.entropy()
    entropy += distribution.brake_probability * distribution.brake.entropy()
    assert torch.allclose(distribution.entropy(), entropy.squeeze(-1))
    forced_brake = MacroResidualAction(
        torch.full((batch, 1), 10.0), torch.ones(batch, 1), torch.zeros(batch, 1)
    )
    physical = distribution.physical_delta(forced_brake)
    assert torch.all(physical[:, 0] <= 0.2) and torch.all(physical[:, 0] >= -0.2)
    assert torch.all(physical[:, 1] < 0.0) and torch.all(physical[:, 1] >= -1.0)

    # Frozen BC recurrent action path is identical in all arms.
    lidar_step = torch.randn(batch, 1, 360)
    speed_step = torch.randn(batch, 1, 1)
    bc_outputs = []
    for policy in policies.values():
        hidden = policy.zero_hidden(batch, "cpu")
        bc_outputs.append(policy.bc_step(lidar_step, speed_step, hidden))
    for index in range(3):
        assert torch.equal(bc_outputs[0][index], bc_outputs[1][index])
        assert torch.equal(bc_outputs[1][index], bc_outputs[2][index])

    assert any(p.requires_grad for p in a.bc_adapter.parameters())
    assert not any(p.requires_grad for p in b.bc_adapter.parameters())
    assert not any(p.requires_grad for p in c.bc_adapter.parameters())
    assert not any(p.requires_grad for p in b.policy_sidecar.parameters())
    assert any(p.requires_grad for p in c.policy_sidecar.beam_encoder.parameters())
    assert not any(p.requires_grad for p in c.policy_sidecar.collision_head.parameters())
    for policy in policies.values():
        assert not any(p.requires_grad for p in policy.bc.parameters())
        assert not any(p.requires_grad for p in policy.shadow_sidecar.parameters())

    before_b, after_b, groups_b = train_two_steps(
        b, bc_feature, lidar_history, scalar_history
    )
    before_c, after_c, groups_c = train_two_steps(
        c, bc_feature, lidar_history, scalar_history
    )
    assert before_b == after_b
    assert before_c != after_c
    assert [(group["name"], group["lr"]) for group in groups_b] == [
        ("action_core", ACTION_CORE_LR)
    ]
    assert [(group["name"], group["lr"]) for group in groups_c] == [
        ("action_core", ACTION_CORE_LR),
        ("sidecar", SIDECAR_FINETUNE_LR),
    ]
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

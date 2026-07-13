#!/usr/bin/env python3
"""Hierarchical NO_OP/intervention math and bounded composition."""

import torch

from bplus_v22 import ARMS, ARM_BC_FROZEN
from bplus_v22.exploration import (
    ActionNoiseKey,
    BehaviorExplorationConfig,
    DETERMINISTIC_CENTERED,
    DETERMINISTIC_STANDARD,
)
from bplus_v22.remediated_model import (
    B3_CONDITIONAL_BRAKE_PRIOR_PROBABILITY,
    B3_INTERVENTION_PRIOR_PROBABILITY,
    HierarchicalResidualAction,
    HierarchicalResidualDistribution,
    RemediatedV22Policy,
    UnifiedV22Policy,
)


def main() -> None:
    torch.set_num_threads(1)
    policy = RemediatedV22Policy(ARM_BC_FROZEN)
    bc = torch.randn(4, 1680)
    lidar = torch.rand(4, 8, 360)
    scalar = torch.randn(4, 24)
    distribution = policy.distribution(bc, lidar, scalar)
    fresh = distribution.deterministic()
    assert torch.equal(fresh.intervention_gate, torch.zeros_like(fresh.intervention_gate))
    bases = torch.tensor([[0.51, 0.2], [-0.51, 0.1], [0.8, -1.0], [-0.8, 2.0]])
    ledger = distribution.compose(bases, fresh)
    deployed = torch.tensor([[0.51, 0.2], [-0.51, 0.1], [0.52, 0.0], [-0.52, 2.0]])
    assert torch.equal(ledger.applied_residual, torch.zeros_like(ledger.applied_residual))
    assert torch.equal(ledger.command, deployed)
    assert not torch.any(ledger.external_clip_would_change)

    sampled = distribution.sample()
    restored = HierarchicalResidualAction.from_tensor(sampled.as_tensor())
    assert torch.equal(restored.intervention_gate, sampled.intervention_gate)
    zero = torch.zeros_like(sampled.intervention_gate)
    manual = distribution.intervention.log_prob(sampled.intervention_gate)
    conditional = distribution.steer.log_prob(sampled.steer_latent)
    conditional += distribution.brake_gate_distribution.log_prob(sampled.brake_gate)
    conditional += torch.where(
        sampled.brake_gate.bool(), distribution.brake.log_prob(sampled.brake_latent), zero
    )
    manual += torch.where(sampled.intervention_gate.bool(), conditional, zero)
    assert torch.allclose(distribution.log_prob(sampled), manual.squeeze(-1))
    conditional_entropy = distribution.steer.entropy()
    conditional_entropy += distribution.brake_gate_distribution.entropy()
    conditional_entropy += distribution.brake_probability * distribution.brake.entropy()
    manual_entropy = distribution.intervention.entropy()
    manual_entropy += distribution.intervention_probability * conditional_entropy
    assert torch.allclose(distribution.entropy(), manual_entropy.squeeze(-1))

    no_op = HierarchicalResidualAction(
        torch.zeros(4, 1), torch.zeros(4, 1), torch.zeros(4, 1), torch.zeros(4, 1)
    )
    no_op_ledger = distribution.compose(bases, no_op)
    assert torch.equal(no_op_ledger.applied_residual, torch.zeros_like(no_op_ledger.applied_residual))

    # Conditional distributions cannot contaminate NO_OP, including when their
    # finite parameters make their own density underflow to -inf.
    extreme = HierarchicalResidualDistribution(
        torch.full((4, 1), -9.0),
        torch.full((4, 1), 1e30),
        torch.ones(4, 1),
        torch.full((4, 1), 1e30),
        torch.full((4, 1), 1e30),
        torch.ones(4, 1),
    )
    expected_noop = extreme.intervention.log_prob(no_op.intervention_gate).squeeze(-1)
    assert torch.equal(extreme.log_prob(no_op), expected_noop)
    assert torch.all(torch.isfinite(extreme.log_prob(no_op)))

    no_brake = HierarchicalResidualAction(
        torch.ones(4, 1), torch.zeros(4, 1), torch.zeros(4, 1), torch.zeros(4, 1)
    )
    no_brake_logp = extreme.log_prob(no_brake)
    expected_no_brake = extreme.intervention.log_prob(no_brake.intervention_gate)
    expected_no_brake += extreme.steer.log_prob(no_brake.steer_latent)
    expected_no_brake += extreme.brake_gate_distribution.log_prob(no_brake.brake_gate)
    assert torch.equal(no_brake_logp, expected_no_brake.squeeze(-1))
    steer_only = HierarchicalResidualAction(
        torch.ones(4, 1), torch.ones(4, 1), torch.zeros(4, 1), torch.zeros(4, 1)
    )
    steer_ledger = distribution.compose(bases, steer_only)
    assert torch.equal(steer_ledger.applied_residual[:, 1], torch.zeros(4))
    assert torch.all(steer_ledger.command[:, 0].abs() <= 0.52)
    both = HierarchicalResidualAction(
        torch.ones(4, 1), torch.full((4, 1), -2.0), torch.ones(4, 1), torch.full((4, 1), 10.0)
    )
    both_ledger = distribution.compose(bases, both)
    assert torch.all(both_ledger.command[:, 0].abs() <= 0.52)
    assert torch.all(both_ledger.command[:, 1] >= 0)

    # A held macro latent is re-projected for each changing 100 Hz BC base.
    for micro_base in (
        torch.tensor([[0.00, 2.0]] * 4),
        torch.tensor([[0.50, 0.05]] * 4),
        torch.tensor([[-0.51, -0.2]] * 4),
    ):
        micro = distribution.compose(micro_base, both)
        assert not torch.any(micro.external_clip_would_change)

    # Projection preserves a witness request that already fits the headroom.
    feasible_steer = torch.atanh(torch.tensor([[0.25]]))  # requests +0.05 rad
    feasible_brake = torch.logit(torch.tensor([[0.10]]))  # requests 0.10 m/s
    feasible = HierarchicalResidualAction(
        torch.ones(1, 1), feasible_steer, torch.ones(1, 1), feasible_brake
    )
    feasible_ledger = distribution.compose(torch.tensor([[0.40, 0.20]]), feasible)
    assert torch.allclose(feasible_ledger.requested_residual, torch.tensor([[0.05, -0.10]]))
    assert torch.allclose(feasible_ledger.applied_residual, torch.tensor([[0.05, -0.10]]))

    # Replaying the exact stored latent under an unchanged policy gives ratio 1.
    replay_ratio = torch.exp(distribution.log_prob(sampled) - distribution.log_prob(restored))
    assert torch.equal(replay_ratio, torch.ones_like(replay_ratio))

    # Old three-coordinate actions and states fail closed.
    try:
        HierarchicalResidualAction.from_tensor(torch.zeros(4, 3))
        raise AssertionError("old action schema was accepted")
    except ValueError as error:
        assert "[...,4]" in str(error)
    old_state = policy.state_dict()
    old_state.pop("intervention_gate.weight")
    try:
        policy.load_hierarchical_state_dict(old_state)
        raise AssertionError("old checkpoint schema was accepted")
    except ValueError as error:
        assert "schema mismatch" in str(error)

    # B3 keeps raw -6 initialization but uses one effective distribution for
    # sampling, replay and standard deterministic deployment.
    unified = UnifiedV22Policy(ARM_BC_FROZEN).eval()
    assert torch.equal(
        unified.intervention_gate.bias,
        torch.full_like(unified.intervention_gate.bias, -6.0),
    )
    assert torch.equal(
        unified.brake_gate.bias,
        torch.full_like(unified.brake_gate.bias, -6.0),
    )
    unified_distribution = unified.distribution(bc, lidar, scalar)
    assert torch.allclose(
        unified_distribution.intervention_probability,
        torch.full((4, 1), B3_INTERVENTION_PRIOR_PROBABILITY),
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        unified_distribution.conditional_brake_probability,
        torch.full((4, 1), B3_CONDITIONAL_BRAKE_PRIOR_PROBABILITY),
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        unified_distribution.unconditional_brake_probability,
        torch.full((4, 1), 0.05),
        atol=1e-7,
        rtol=0.0,
    )
    unified_fresh = unified.deterministic_action(
        bc, lidar, scalar, DETERMINISTIC_STANDARD
    )
    assert torch.equal(
        unified_fresh.intervention_gate,
        torch.zeros_like(unified_fresh.intervention_gate),
    )
    assert unified.primary_deterministic_mode == DETERMINISTIC_STANDARD
    for arm in ARMS:
        fresh_arm = UnifiedV22Policy(arm).eval()
        fresh_distribution = fresh_arm.distribution(bc, lidar, scalar)
        fresh_action = fresh_arm.deterministic_action(
            bc, lidar, scalar, DETERMINISTIC_STANDARD
        )
        assert torch.equal(
            fresh_action.intervention_gate,
            torch.zeros_like(fresh_action.intervention_gate),
        )
        assert torch.allclose(
            fresh_distribution.intervention_probability,
            torch.full((4, 1), 0.10),
            atol=1e-7,
            rtol=0.0,
        )
    try:
        unified.deterministic_action(bc, lidar, scalar, DETERMINISTIC_CENTERED)
        raise AssertionError("B3 accepted the historical centered rule")
    except ValueError as error:
        assert "standard mode" in str(error)

    zero_behavior = BehaviorExplorationConfig(
        0.0, 0.0, 0.1, 1.0, "b3-unified"
    ).as_batch(torch.empty(4, 1))
    unified_behavior = unified.behavior_distribution(
        bc, lidar, scalar, zero_behavior
    )
    assert torch.equal(
        unified_behavior.intervention_logits,
        unified_distribution.intervention_logits,
    )
    nonzero_behavior = BehaviorExplorationConfig(
        0.1, 0.0, 0.1, 1.0, "forbidden"
    ).as_batch(torch.empty(4, 1))
    try:
        unified.behavior_distribution(bc, lidar, scalar, nonzero_behavior)
        raise AssertionError("B3 accepted an external gate offset")
    except ValueError as error:
        assert "forbids behavior gate offsets" in str(error)

    # Deterministic keyed sampling sees the analytic B3 prior without touching
    # global RNG, and unchanged replay remains ratio one.
    count = 4096
    udist = HierarchicalResidualDistribution(
        unified_distribution.intervention_logits[0:1].expand(count, -1),
        unified_distribution.steer.mean[0:1].expand(count, -1),
        unified_distribution.steer.stddev[0:1].expand(count, -1),
        unified_distribution.conditional_brake_logits[0:1].expand(count, -1),
        unified_distribution.brake.mean[0:1].expand(count, -1),
        unified_distribution.brake.stddev[0:1].expand(count, -1),
    )
    keys = tuple(ActionNoiseKey(31, f"L2:b3:{i}", 0, i) for i in range(count))
    uaction = udist.sample_keyed(keys)
    assert abs(float(uaction.intervention_gate.mean()) - 0.10) < 0.02
    assert abs(float(uaction.brake_gate.mean()) - 0.05) < 0.015
    assert torch.equal(
        torch.exp(udist.log_prob(uaction) - udist.log_prob(uaction)),
        torch.ones(count),
    )

    b2_state = RemediatedV22Policy(ARM_BC_FROZEN).state_dict()
    try:
        unified.load_unified_state_dict(b2_state)
        raise AssertionError("B3 accepted a B2 checkpoint")
    except ValueError as error:
        assert "unified checkpoint" in str(error)
    b3_state = unified.state_dict()
    try:
        RemediatedV22Policy(ARM_BC_FROZEN).load_hierarchical_state_dict(b3_state)
        raise AssertionError("B2 accepted a B3 checkpoint")
    except RuntimeError:
        pass
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""B2 dual-offset behavior, keyed sampling, and deterministic-mode contract."""

from dataclasses import FrozenInstanceError
import math

import torch

from bplus_v22 import ARM_BC_FROZEN
from bplus_v22.exploration import (
    ActionNoiseKey,
    BehaviorExplorationBatch,
    BehaviorExplorationConfig,
    DETERMINISTIC_CENTERED,
    DETERMINISTIC_STANDARD,
    KEYED_ACTION_COMPONENTS,
    keyed_component_draws,
)
from bplus_v22.remediated_model import (
    HierarchicalResidualAction,
    HierarchicalResidualDistribution,
    RemediatedV22Policy,
)


def expect(error_type, function) -> None:
    try:
        function()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def inputs(batch: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9182)
        return (
            torch.randn(batch, 1680),
            torch.rand(batch, 8, 360),
            torch.randn(batch, 24),
        )


def test_config_and_analytic_probabilities() -> None:
    full_top_offset = math.log(0.1 / 0.9) + 6.0
    config = BehaviorExplorationConfig(
        intervention_logit_offset=full_top_offset,
        conditional_brake_logit_offset=6.0,
        steer_std_scale=0.1,
        brake_std_scale=1.0,
        schedule_id="b2-full-v1",
    )
    assert BehaviorExplorationConfig.from_dict(config.as_dict()) == config
    expect(FrozenInstanceError, lambda: setattr(config, "schedule_id", "changed"))
    expect(
        ValueError,
        lambda: BehaviorExplorationConfig(0.0, 0.0, 0.0, 1.0, "bad"),
    )

    policy = RemediatedV22Policy(ARM_BC_FROZEN).eval()
    bc, lidar, scalar = inputs(3)
    reference = torch.empty(3, 1)
    behavior = config.as_batch(reference)
    distribution = policy.behavior_distribution(bc, lidar, scalar, behavior)
    assert torch.allclose(
        distribution.intervention_probability,
        torch.full((3, 1), 0.1),
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        distribution.conditional_brake_probability,
        torch.full((3, 1), 0.5),
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        distribution.unconditional_brake_probability,
        torch.full((3, 1), 0.05),
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.equal(
        distribution.brake_probability,
        distribution.conditional_brake_probability,
    )
    assert torch.allclose(
        distribution.steer.stddev,
        torch.full((3, 1), 0.015),
        atol=1e-8,
        rtol=0.0,
    )

    # Replay batches may contain distinct schedule entries; offsets are applied
    # row-by-row, without ambiguous scalar broadcasting.
    mixed = BehaviorExplorationBatch(
        intervention_logit_offset=torch.tensor([[full_top_offset], [6.0], [0.0]]),
        conditional_brake_logit_offset=torch.tensor([[6.0], [0.0], [-1.0]]),
        steer_std_scale=torch.tensor([[0.1], [1.0], [0.5]]),
        brake_std_scale=torch.ones(3, 1),
        schedule_ids=("full", "top-half", "zero"),
    )
    mixed_distribution = policy.behavior_distribution(bc, lidar, scalar, mixed)
    assert torch.allclose(
        mixed_distribution.intervention_probability[:, 0],
        torch.sigmoid(torch.tensor([-6.0 + full_top_offset, 0.0, -6.0])),
    )
    assert torch.allclose(
        mixed_distribution.conditional_brake_probability[:, 0],
        torch.sigmoid(torch.tensor([0.0, -6.0, -7.0])),
    )


def test_joint_log_prob_and_entropy() -> None:
    batch = 3
    top_logits = torch.tensor([[-2.0], [0.25], [1.5]])
    steer_mean = torch.tensor([[0.1], [-0.2], [0.3]])
    steer_std = torch.tensor([[0.2], [0.4], [0.6]])
    brake_logits = torch.tensor([[1.0], [-1.0], [0.5]])
    brake_mean = torch.tensor([[0.3], [0.1], [-0.4]])
    brake_std = torch.tensor([[0.5], [0.7], [0.9]])
    distribution = HierarchicalResidualDistribution(
        top_logits, steer_mean, steer_std, brake_logits, brake_mean, brake_std
    )
    action = HierarchicalResidualAction(
        intervention_gate=torch.tensor([[0.0], [1.0], [1.0]]),
        steer_latent=torch.tensor([[0.0], [0.25], [-0.5]]),
        brake_gate=torch.tensor([[0.0], [0.0], [1.0]]),
        brake_latent=torch.tensor([[0.0], [0.0], [0.75]]),
    )
    zero = torch.zeros(batch, 1)
    manual_conditional = distribution.steer.log_prob(action.steer_latent)
    manual_conditional += distribution.brake_gate_distribution.log_prob(
        action.brake_gate
    )
    manual_conditional += torch.where(
        action.brake_gate.bool(),
        distribution.brake.log_prob(action.brake_latent),
        zero,
    )
    manual_log_prob = distribution.intervention.log_prob(action.intervention_gate)
    manual_log_prob += torch.where(
        action.intervention_gate.bool(), manual_conditional, zero
    )
    assert torch.allclose(distribution.log_prob(action), manual_log_prob[:, 0])

    manual_conditional_entropy = distribution.steer.entropy()
    manual_conditional_entropy += distribution.brake_gate_distribution.entropy()
    manual_conditional_entropy += (
        distribution.conditional_brake_probability * distribution.brake.entropy()
    )
    manual_entropy = distribution.intervention.entropy()
    manual_entropy += distribution.intervention_probability * manual_conditional_entropy
    assert torch.allclose(distribution.entropy(), manual_entropy[:, 0])


def test_keyed_component_sampling() -> None:
    count = 4096
    reference = torch.empty(count, 1)
    keys = tuple(ActionNoiseKey(7, f"L2-{index:05d}", 0, index) for index in range(count))
    global_before = torch.get_rng_state().clone()
    first = keyed_component_draws(keys, reference)
    global_after = torch.get_rng_state().clone()
    second = keyed_component_draws(keys, reference)
    assert torch.equal(global_before, global_after)
    for name in (
        "intervention_uniform",
        "steer_standard_normal",
        "brake_uniform",
        "brake_standard_normal",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name))
        assert torch.all(torch.isfinite(getattr(first, name)))
    assert KEYED_ACTION_COMPONENTS == (
        "intervention_gate",
        "steer_latent",
        "brake_gate",
        "brake_latent",
    )

    changed = keyed_component_draws(
        (ActionNoiseKey(7, "L2-00000", 0, 1),), torch.empty(1, 1)
    )
    for name in (
        "intervention_uniform",
        "steer_standard_normal",
        "brake_uniform",
        "brake_standard_normal",
    ):
        assert not torch.equal(getattr(first, name)[0:1], getattr(changed, name))

    distribution = HierarchicalResidualDistribution(
        torch.full((count, 1), math.log(0.1 / 0.9)),
        torch.zeros(count, 1),
        torch.full((count, 1), 0.015),
        torch.zeros(count, 1),
        torch.zeros(count, 1),
        torch.full((count, 1), 0.25),
    )
    from_draws = distribution.sample_from_draws(first)
    from_keys = distribution.sample_keyed(keys)
    assert torch.equal(from_draws.as_tensor(), from_keys.as_tensor())
    top_rate = float(from_keys.intervention_gate.mean())
    joint_rate = float(from_keys.brake_gate.mean())
    conditional_rate = float(
        from_keys.brake_gate.sum() / from_keys.intervention_gate.sum()
    )
    assert abs(top_rate - 0.1) < 0.02
    assert abs(conditional_rate - 0.5) < 0.07
    assert abs(joint_rate - 0.05) < 0.015

    inactive = from_keys.intervention_gate == 0
    assert torch.all(from_keys.steer_latent[inactive] == 0)
    assert torch.all(from_keys.brake_latent[from_keys.brake_gate == 0] == 0)
    # Inactive canonical actions did not skip their independently keyed draws.
    assert torch.any(first.steer_standard_normal[inactive] != 0)
    assert torch.any(first.brake_standard_normal[inactive] != 0)


def test_nonpersistent_context_and_deterministic_modes() -> None:
    policy = RemediatedV22Policy(ARM_BC_FROZEN).eval()
    bc, lidar, scalar = inputs(2)
    state_before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    parameter_names = tuple(name for name, _ in policy.named_parameters())
    config = BehaviorExplorationConfig(8.0, 9.0, 0.1, 1.0, "temporary")
    behavior = config.as_batch(torch.empty(2, 1))
    distribution = policy.behavior_distribution(bc, lidar, scalar, behavior)
    distribution.sample_keyed(
        (ActionNoiseKey(3, "L2-a", 0, 0), ActionNoiseKey(3, "L2-b", 0, 0))
    )
    assert tuple(name for name, _ in policy.named_parameters()) == parameter_names
    assert tuple(policy.state_dict()) == tuple(state_before)
    for name, value in policy.state_dict().items():
        assert torch.equal(value, state_before[name])
    assert not any("behavior" in name or "schedule" in name for name in policy.state_dict())

    # A temporary behavior context cannot leak into primary deterministic eval.
    fresh = policy.deterministic_action(bc, lidar, scalar, DETERMINISTIC_CENTERED)
    assert torch.equal(fresh.as_tensor(), torch.zeros_like(fresh.as_tensor()))

    epsilon = 1e-4
    with torch.no_grad():
        policy.intervention_gate.bias.fill_(-6.0 + epsilon)
        policy.brake_gate.bias.fill_(-6.0 + epsilon)
    centered = policy.deterministic_action(bc, lidar, scalar, DETERMINISTIC_CENTERED)
    standard = policy.deterministic_action(bc, lidar, scalar, DETERMINISTIC_STANDARD)
    assert torch.all(centered.intervention_gate == 1)
    assert torch.all(centered.brake_gate == 1)
    assert torch.all(standard.intervention_gate == 0)
    assert torch.all(standard.brake_gate == 0)

    # Primary centered comparison is strict: equality with the frozen fresh
    # prior remains NO_OP, including the conditional brake threshold.
    with torch.no_grad():
        policy.intervention_gate.bias.fill_(-6.0)
        policy.brake_gate.bias.fill_(-6.0 + epsilon)
    equality = policy.deterministic_action(bc, lidar, scalar)
    assert torch.all(equality.intervention_gate == 0)
    with torch.no_grad():
        policy.intervention_gate.bias.fill_(-6.0 + epsilon)
        policy.brake_gate.bias.fill_(-6.0)
    conditional_equality = policy.deterministic_action(bc, lidar, scalar)
    assert torch.all(conditional_equality.intervention_gate == 1)
    assert torch.all(conditional_equality.brake_gate == 0)

    # Standard Bernoulli diagnostic retains its conventional strict > 0 rule.
    with torch.no_grad():
        policy.intervention_gate.bias.zero_()
        policy.brake_gate.bias.fill_(epsilon)
    standard_equality = policy.deterministic_action(
        bc, lidar, scalar, DETERMINISTIC_STANDARD
    )
    assert torch.all(standard_equality.intervention_gate == 0)
    with torch.no_grad():
        policy.intervention_gate.bias.fill_(epsilon)
    standard_positive = policy.deterministic_action(
        bc, lidar, scalar, DETERMINISTIC_STANDARD
    )
    assert torch.all(standard_positive.intervention_gate == 1)
    assert torch.all(standard_positive.brake_gate == 1)

    # Historical calibration remains supported by the historical API but is
    # explicitly rejected by B2 behavior and deterministic paths.
    with torch.no_grad():
        policy.intervention_logit_offset.fill_(0.25)
    expect(
        ValueError,
        lambda: policy.behavior_distribution(bc, lidar, scalar, behavior),
    )
    expect(
        ValueError,
        lambda: policy.deterministic_action(bc, lidar, scalar),
    )


def main() -> None:
    torch.set_num_threads(1)
    test_config_and_analytic_probabilities()
    test_joint_log_prob_and_entropy()
    test_keyed_component_sampling()
    test_nonpersistent_context_and_deterministic_modes()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

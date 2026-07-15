#!/usr/bin/env python3
"""Fresh hierarchical NO_OP composes deployed BC identity at every micro-step."""

import copy
import gc

import torch

from bplus_v22 import ARMS
from bplus_v22.hierarchical_identity import (
    HierarchicalIdentityActor,
    load_hierarchical_checkpoint,
)
from bplus_v22.remediated_model import RemediatedV22Policy
from d2r.model import D2RGeometryNet
from model import End2Race


def _deployed(action: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.clamp(action[..., 0:1], -0.52, 0.52),
            torch.clamp_min(action[..., 1:2], 0.0),
        ],
        dim=-1,
    )


def main() -> None:
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(442)
        bc = End2Race(mask_prob=0.0, hidden_scale=4).eval()
        bc_state = copy.deepcopy(bc.state_dict())
        sidecar_state = copy.deepcopy(D2RGeometryNet().state_dict())
        lidar = torch.rand(23, 1, 1, 360) * 30.0
        speed = torch.rand(23, 1, 1, 1) * 8.0
    hidden = torch.zeros(1, 1, bc.gru.hidden_size)
    reference = []
    with torch.no_grad():
        for step in range(23):
            action, hidden = bc(lidar[step], speed[step], hidden)
            reference.append(_deployed(action.detach().clone()))
    del bc
    gc.collect()

    action_digests = []
    outputs_by_arm = []
    for arm in ARMS:
        policy = RemediatedV22Policy(
            arm,
            bc_state_dict=bc_state,
            sidecar_state_dict=sidecar_state,
            sidecar_bc_mean=torch.zeros(1680),
            sidecar_bc_std=torch.ones(1680),
        ).eval()
        actor = HierarchicalIdentityActor(policy).eval()
        hidden = torch.zeros(1, 1, actor.gru.hidden_size)
        outputs = []
        with torch.no_grad():
            for step in range(23):
                actor.observe_actual_speed(float(speed[step].item()) + 0.125)
                action, hidden = actor(lidar[step], speed[step], hidden)
                actor.observe_applied_command(
                    float(action[0, -1, 0]), float(action[0, -1, 1])
                )
                outputs.append(action.detach().clone())
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(outputs, reference)
        )
        accounting = actor.accounting()
        assert accounting["micro_steps"] == 23
        assert accounting["macro_decisions"] == 3
        assert accounting["macro_lengths"] == [10, 10, 3]
        assert accounting["natural_intervention_decisions"] == 0
        assert accounting["natural_brake_decisions"] == 0
        assert accounting["max_abs_requested_residual"] == 0.0
        assert accounting["max_abs_applied_residual"] == 0.0
        action_digests.append(accounting["natural_action_sequence_sha256"])
        outputs_by_arm.append(outputs)

        # Historical single-gate checkpoints must not load into the new policy.
        try:
            load_hierarchical_checkpoint(
                policy,
                {
                    "schema": "bplus-v2.2-warmstart-remediation-checkpoint-2",
                    "arm": arm,
                    "state_dict": policy.state_dict(),
                },
                expected_arm=arm,
            )
        except ValueError as error:
            assert "legacy checkpoint rejected" in str(error)
        else:
            raise AssertionError("legacy checkpoint unexpectedly accepted")

        # A calibrated/warm-started head is outside this fresh identity gate.
        with torch.no_grad():
            policy.intervention_logit_offset.fill_(7.0)
        try:
            HierarchicalIdentityActor(policy)
        except ValueError as error:
            assert "calibrated/warm-started" in str(error)
        else:
            raise AssertionError("non-fresh policy unexpectedly accepted")
        del actor, policy
        gc.collect()

    assert len(set(action_digests)) == 1
    for step in range(23):
        assert torch.equal(outputs_by_arm[0][step], outputs_by_arm[1][step])
        assert torch.equal(outputs_by_arm[1][step], outputs_by_arm[2][step])
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

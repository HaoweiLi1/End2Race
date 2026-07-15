#!/usr/bin/env python3
"""Task-10 natural residual composition and deployment-clipping ledger."""

import copy

import torch
import torch.nn as nn

from bplus_v22 import ARM_BC_FROZEN
from bplus_v22.closed_loop import NaturalResidualActor
from bplus_v22.model import V22Policy
from model import End2Race


class StubAlarm(nn.Module):
    def evaluate(self, fold, lidar_history, bc_feature, scalar_history):
        assert fold == 2
        return {"raw": 0.2, "calibrated": 0.3, "threshold": 0.25, "alarm": True}


def main() -> None:
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        bc = End2Race(mask_prob=0.0, hidden_scale=4).eval()
        state = copy.deepcopy(bc.state_dict())
        lidar = torch.rand(23, 1, 1, 360) * 30.0
        speed = torch.rand(23, 1, 1, 1) * 8.0 + 2.0
    hidden = torch.zeros(1, 1, bc.gru.hidden_size)
    reference = []
    with torch.no_grad():
        for step in range(23):
            action, hidden = bc(lidar[step], speed[step], hidden)
            reference.append(action.detach().clone())

    policy = V22Policy(ARM_BC_FROZEN, bc_state_dict=state).eval()
    with torch.no_grad():
        policy.brake_gate.bias.fill_(2.0)
    actor = NaturalResidualActor(policy, StubAlarm()).eval()
    actor.reset_runtime(2)
    hidden = torch.zeros(1, 1, actor.gru.hidden_size)
    outputs = []
    with torch.no_grad():
        for step in range(23):
            actor.observe_actual_speed(float(speed[step].item()))
            action, hidden = actor(lidar[step], speed[step], hidden)
            actor.observe_applied_command(float(action[0, -1, 0]), float(action[0, -1, 1]))
            outputs.append(action.detach().clone())
    for actual, base in zip(outputs, reference):
        assert torch.equal(actual[..., 0], base[..., 0])
        assert torch.equal(actual[..., 1], base[..., 1] - 0.5)
    accounting = actor.accounting()
    assert accounting["macro_lengths"] == [10, 10, 3]
    assert actor.clip_micro_steps == 0
    assert len(actor.decision_records) == 3
    assert all(row["brake_gate"] == 1 and row["oof_alarm"] for row in actor.decision_records)

    # The adapter independently catches evaluator clipping of a requested command.
    actor.reset_runtime(2)
    hidden = torch.zeros(1, 1, actor.gru.hidden_size)
    with torch.no_grad():
        actor.observe_actual_speed(float(speed[0].item()))
        actor(lidar[0], speed[0], hidden)
        actor.observe_applied_command(0.52, 0.0)
    assert actor.clip_micro_steps == 1 and actor.decision_records[0]["clip_micro_steps"] == 1
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

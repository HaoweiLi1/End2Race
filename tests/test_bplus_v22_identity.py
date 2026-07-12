#!/usr/bin/env python3
"""No-learning adapter preserves BC actions and exact macro accounting."""

import copy
import gc

import torch

from bplus_v22 import ARMS
from bplus_v22.identity import ZeroResidualActor
from bplus_v22.model import V22Policy
from model import End2Race


def main() -> None:
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44)
        bc = End2Race(mask_prob=0.0, hidden_scale=4).eval()
        state = copy.deepcopy(bc.state_dict())
        lidar = torch.rand(23, 1, 1, 360) * 30.0
        speed = torch.rand(23, 1, 1, 1) * 8.0
    hidden = torch.zeros(1, 1, bc.gru.hidden_size)
    reference = []
    with torch.no_grad():
        for step in range(23):
            action, hidden = bc(lidar[step], speed[step], hidden)
            reference.append(action.detach().clone())
    del bc
    gc.collect()
    diagnostics = {}
    for arm in ARMS:
        policy = V22Policy(arm, bc_state_dict=state).eval()
        adapter = ZeroResidualActor(policy).eval()
        outputs = []
        hidden = torch.zeros(1, 1, adapter.gru.hidden_size)
        with torch.no_grad():
            for step in range(23):
                adapter.observe_actual_speed(float(speed[step].item()) + 0.25)
                action, hidden = adapter(lidar[step], speed[step], hidden)
                if step == 0:
                    _, scalar = adapter._history()
                    assert torch.equal(scalar[:, 8:], torch.zeros_like(scalar[:, 8:]))
                adapter.observe_applied_command(
                    float(min(0.52, max(-0.52, action[0, -1, 0].item()))),
                    float(max(0.0, action[0, -1, 1].item())),
                )
                outputs.append(action.detach().clone())
        for actual, expected in zip(outputs, reference):
            assert torch.equal(actual, expected)
        accounting = adapter.accounting()
        assert accounting["micro_steps"] == 23
        assert accounting["macro_decisions"] == 3
        assert accounting["macro_lengths"] == [10, 10, 3]
        assert accounting["max_abs_residual"] == 0.0
        diagnostics[arm] = accounting["diagnostic_sha256"]
        adapter.reset_runtime()
        assert adapter.micro_steps == 0 and adapter.macro_decisions == 0
        del adapter, policy
        gc.collect()
    assert len(set(diagnostics.values())) == 1

    # A warm-started policy may naturally brake; Task 9 still composes exact
    # zero after recording that natural checkpoint decision.
    policy = V22Policy(ARMS[0], bc_state_dict=state).eval()
    with torch.no_grad():
        policy.brake_gate.bias.fill_(2.0)
    adapter = ZeroResidualActor(policy, require_natural_zero=False).eval()
    hidden = torch.zeros(1, 1, adapter.gru.hidden_size)
    forced_outputs = []
    with torch.no_grad():
        for step in range(23):
            adapter.observe_actual_speed(float(speed[step].item()) + 0.25)
            action, hidden = adapter(lidar[step], speed[step], hidden)
            adapter.observe_applied_command(float(action[0, -1, 0]), float(action[0, -1, 1]))
            forced_outputs.append(action.detach().clone())
    assert all(torch.equal(actual, expected) for actual, expected in zip(forced_outputs, reference))
    forced_accounting = adapter.accounting()
    assert forced_accounting["natural_brake_decisions"] == 3
    assert forced_accounting["max_abs_natural_residual"] > 0.0
    assert forced_accounting["max_abs_residual"] == 0.0
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

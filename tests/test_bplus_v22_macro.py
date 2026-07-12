#!/usr/bin/env python3
"""Macro transition and variable-discount GAE regression."""

import numpy as np

from bplus_v22 import MICRO_GAMMA
from bplus_v22.buffer import variable_discount_gae
from bplus_v22.macro import MacroClock, aggregate_micro_signals, discounted_sum


def legacy_gae(reward, value, terminated, truncated, trunc_next, last_value, gamma, lam):
    output = np.zeros(len(reward), dtype=np.float64)
    gae = 0.0
    for t in reversed(range(len(reward))):
        if terminated[t]:
            next_value, boundary = 0.0, True
        elif truncated[t]:
            next_value, boundary = trunc_next[t], True
        elif t == len(reward) - 1:
            next_value, boundary = last_value, False
        else:
            next_value, boundary = value[t + 1], False
        delta = reward[t] + gamma * next_value - value[t]
        gae = delta if boundary else delta + gamma * lam * gae
        output[t] = gae
    return output


def main() -> None:
    values = np.arange(1.0, 11.0)
    expected = sum((MICRO_GAMMA**i) * value for i, value in enumerate(values))
    assert np.isclose(discounted_sum(values), expected)
    full = aggregate_micro_signals(
        values,
        np.zeros(10),
        np.ones(10),
        np.zeros(10, dtype=bool),
        np.zeros(10, dtype=bool),
    )
    assert full.length == 10 and not full.terminated and not full.truncated
    assert np.isclose(full.reward, expected)
    assert np.isclose(full.performance_reward, sum(MICRO_GAMMA**i for i in range(10)))

    short = aggregate_micro_signals(
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [False, False, True],
        [False, False, False],
    )
    assert short.length == 3 and short.terminated
    assert np.isclose(short.collision_cost, MICRO_GAMMA**2)
    try:
        aggregate_micro_signals([1.0], [0.0], [0.0], [False], [False])
        raise AssertionError("short non-boundary macro accepted")
    except ValueError:
        pass

    clock = MacroClock()
    assert clock.needs_action
    clock.begin()
    for _ in range(9):
        assert not clock.step()
    assert clock.step() and clock.needs_action
    clock.begin()
    assert clock.step(truncated=True) and clock.elapsed == 1

    reward = np.array([1.0, -0.5, 2.0, 0.25])
    value = np.array([0.2, 0.1, -0.3, 0.4])
    terminated = np.array([False, True, False, False])
    truncated = np.array([False, False, False, True])
    trunc_next = np.array([0.0, 0.0, 0.0, 0.7])
    expected_adv = legacy_gae(
        reward, value, terminated, truncated, trunc_next, 0.9, MICRO_GAMMA, 0.99
    )
    actual, returns = variable_discount_gae(
        reward,
        value,
        np.full(len(reward), MICRO_GAMMA),
        terminated,
        truncated,
        trunc_next,
        last_value=0.9,
    )
    assert np.allclose(actual, expected_adv)
    assert np.allclose(returns, expected_adv + value)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

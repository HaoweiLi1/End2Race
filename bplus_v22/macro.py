"""Exact 100 Hz micro-step to 10 Hz macro-transition accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from bplus_v22 import MACRO_STEPS, MICRO_GAMMA


@dataclass(frozen=True)
class MacroSignals:
    """Discounted signals emitted by one possibly-short macro transition."""

    reward: float
    collision_cost: float
    performance_reward: float
    length: int
    discount: float
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if not 1 <= int(self.length) <= MACRO_STEPS:
            raise ValueError("macro length must be in 1..10")
        if bool(self.terminated) and bool(self.truncated):
            raise ValueError("macro transition cannot terminate and truncate")
        values = (self.reward, self.collision_cost, self.performance_reward, self.discount)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("macro transition contains nonfinite signal")
        expected = MICRO_GAMMA ** int(self.length)
        if not np.isclose(self.discount, expected, rtol=0.0, atol=1e-15):
            raise ValueError("macro transition discount drift")


def discounted_sum(values: Iterable[float], gamma: float = MICRO_GAMMA) -> float:
    """Return sum_i gamma**i * values[i], rejecting empty/nonfinite input."""

    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("discounted sum requires a non-empty vector")
    if not np.all(np.isfinite(array)) or not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("discounted sum input is invalid")
    weights = np.power(float(gamma), np.arange(len(array), dtype=np.float64))
    return float(np.dot(weights, array))


def aggregate_micro_signals(
    rewards,
    collision_costs,
    performance_rewards,
    terminated,
    truncated,
) -> MacroSignals:
    """Aggregate one action-held micro-step segment into exactly one transition."""

    reward = np.asarray(rewards, dtype=np.float64)
    collision = np.asarray(collision_costs, dtype=np.float64)
    performance = np.asarray(performance_rewards, dtype=np.float64)
    term = np.asarray(terminated, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    arrays = (reward, collision, performance, term, trunc)
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("macro micro-step inputs must be vectors")
    if len({len(value) for value in arrays}) != 1 or not 1 <= len(reward) <= MACRO_STEPS:
        raise ValueError("macro micro-step inputs have invalid lengths")
    if not np.all(np.isfinite(reward)) or not np.all(np.isfinite(collision)):
        raise ValueError("macro reward/cost contains nonfinite value")
    if not np.all(np.isfinite(performance)) or np.any(collision < 0.0):
        raise ValueError("macro performance/collision signal is invalid")
    boundary = term | trunc
    if np.count_nonzero(boundary) > 1:
        raise ValueError("macro segment contains multiple episode boundaries")
    if np.any(boundary[:-1]):
        raise ValueError("macro segment continues after an episode boundary")
    if term[-1] and trunc[-1]:
        raise ValueError("final micro-step cannot terminate and truncate")
    if len(reward) < MACRO_STEPS and not boundary[-1]:
        raise ValueError("short macro segment requires a terminal boundary")
    return MacroSignals(
        reward=discounted_sum(reward),
        collision_cost=discounted_sum(collision),
        performance_reward=discounted_sum(performance),
        length=len(reward),
        discount=MICRO_GAMMA ** len(reward),
        terminated=bool(term[-1]),
        truncated=bool(trunc[-1]),
    )


class MacroClock:
    """Tracks when one held residual action must be replaced or emitted."""

    def __init__(self, macro_steps: int = MACRO_STEPS):
        if int(macro_steps) != MACRO_STEPS:
            raise ValueError("B+ v2.2 macro length is locked to 10")
        self.macro_steps = MACRO_STEPS
        self.reset()

    def reset(self) -> None:
        self._elapsed = 0
        self._active = False

    @property
    def needs_action(self) -> bool:
        return not self._active

    @property
    def elapsed(self) -> int:
        return self._elapsed

    def begin(self) -> None:
        if self._active:
            raise RuntimeError("cannot begin a new action inside an active macro")
        self._active = True
        self._elapsed = 0

    def step(self, *, terminated: bool = False, truncated: bool = False) -> bool:
        """Advance one micro-step and return True exactly when the macro emits."""

        if not self._active:
            raise RuntimeError("macro clock step requires an active action")
        if terminated and truncated:
            raise ValueError("micro-step cannot terminate and truncate")
        self._elapsed += 1
        boundary = bool(terminated or truncated)
        emit = boundary or self._elapsed == self.macro_steps
        if emit:
            self._active = False
        if self._elapsed > self.macro_steps:
            raise AssertionError("macro clock exceeded locked action duration")
        return emit

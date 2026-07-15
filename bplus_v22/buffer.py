"""Multi-objective variable-length macro GAE and rollout storage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bplus_v22 import MACRO_LAMBDA
from bplus_v22.macro import MacroSignals


CHANNELS = ("reward", "collision", "performance")


@dataclass(frozen=True)
class EpisodeOutcomeSignals:
    """Direct closed-loop signals retained without scalarization."""

    any_agent_collision: bool
    ego_involved_collision: bool
    terminal_overtake: bool
    confirmed_safe_pass: bool
    progress: float

    def __post_init__(self) -> None:
        if self.ego_involved_collision and not self.any_agent_collision:
            raise ValueError("ego collision must also be an any-agent collision")
        if self.confirmed_safe_pass and not self.terminal_overtake:
            raise ValueError("confirmed safe pass must be a terminal overtake")
        if not np.isfinite(self.progress):
            raise ValueError("episode progress is nonfinite")


class EpisodeOutcomeStore:
    """Strict append-only episode store with one independent column per signal."""

    def __init__(self) -> None:
        self.records: list[EpisodeOutcomeSignals] = []

    def add(self, record: EpisodeOutcomeSignals) -> None:
        if not isinstance(record, EpisodeOutcomeSignals):
            raise TypeError("episode outcome store requires EpisodeOutcomeSignals")
        self.records.append(record)

    def as_columns(self) -> dict[str, np.ndarray]:
        if not self.records:
            raise RuntimeError("episode outcome store is empty")
        return {
            "any_agent_collision": np.asarray(
                [row.any_agent_collision for row in self.records], dtype=bool
            ),
            "ego_involved_collision": np.asarray(
                [row.ego_involved_collision for row in self.records], dtype=bool
            ),
            "terminal_overtake": np.asarray(
                [row.terminal_overtake for row in self.records], dtype=bool
            ),
            "confirmed_safe_pass": np.asarray(
                [row.confirmed_safe_pass for row in self.records], dtype=bool
            ),
            "progress": np.asarray([row.progress for row in self.records], dtype=np.float64),
        }


def variable_discount_gae(
    rewards,
    values,
    discounts,
    terminated,
    truncated,
    trunc_next_values,
    last_value: float,
    gae_lambda: float = MACRO_LAMBDA,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE where each transition may span a different micro-step count."""

    reward = np.asarray(rewards, dtype=np.float64)
    value = np.asarray(values, dtype=np.float64)
    discount = np.asarray(discounts, dtype=np.float64)
    term = np.asarray(terminated, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    trunc_next = np.asarray(trunc_next_values, dtype=np.float64)
    arrays = (reward, value, discount, term, trunc, trunc_next)
    if any(item.ndim != 1 for item in arrays) or len({len(item) for item in arrays}) != 1:
        raise ValueError("GAE inputs must be equal-length vectors")
    if len(reward) == 0:
        raise ValueError("GAE requires at least one transition")
    if not all(np.all(np.isfinite(item)) for item in (reward, value, discount, trunc_next)):
        raise ValueError("GAE contains nonfinite input")
    if np.any(discount <= 0.0) or np.any(discount > 1.0):
        raise ValueError("GAE discounts must be in (0,1]")
    if np.any(term & trunc):
        raise ValueError("GAE transition cannot terminate and truncate")
    if not np.isfinite(last_value) or not np.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("GAE bootstrap/lambda is invalid")

    advantage = np.zeros_like(reward)
    gae = 0.0
    for index in reversed(range(len(reward))):
        if term[index]:
            next_value = 0.0
            boundary = True
        elif trunc[index]:
            next_value = float(trunc_next[index])
            boundary = True
        elif index == len(reward) - 1:
            next_value = float(last_value)
            boundary = False
        else:
            next_value = float(value[index + 1])
            boundary = False
        delta = reward[index] + discount[index] * next_value - value[index]
        gae = delta if boundary else delta + discount[index] * gae_lambda * gae
        advantage[index] = gae
    returns = advantage + value
    return advantage.astype(np.float32), returns.astype(np.float32)


@dataclass(frozen=True)
class MacroRecord:
    signals: MacroSignals
    log_prob: float
    reward_value: float
    collision_value: float
    performance_value: float
    reward_trunc_next_value: float = 0.0
    collision_trunc_next_value: float = 0.0
    performance_trunc_next_value: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.log_prob,
            self.reward_value,
            self.collision_value,
            self.performance_value,
            self.reward_trunc_next_value,
            self.collision_trunc_next_value,
            self.performance_trunc_next_value,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("macro record contains nonfinite value")


class MultiObjectiveMacroBuffer:
    """Small strict store used by v2.2 rollout/replay integration."""

    def __init__(self, capacity: int):
        if int(capacity) <= 0:
            raise ValueError("macro buffer capacity must be positive")
        self.capacity = int(capacity)
        self.records: list[MacroRecord] = []

    def reset(self) -> None:
        self.records.clear()

    def add(self, record: MacroRecord) -> None:
        if len(self.records) >= self.capacity:
            raise RuntimeError("macro buffer overflow")
        self.records.append(record)

    def compute_advantages(self, last_values: dict[str, float]) -> dict[str, np.ndarray]:
        if len(self.records) != self.capacity:
            raise RuntimeError(
                f"macro buffer has {len(self.records)} transitions, expected {self.capacity}"
            )
        if set(last_values) != set(CHANNELS):
            raise ValueError("macro buffer requires reward/collision/performance bootstraps")
        discounts = np.asarray([row.signals.discount for row in self.records])
        terminated = np.asarray([row.signals.terminated for row in self.records])
        truncated = np.asarray([row.signals.truncated for row in self.records])
        output: dict[str, np.ndarray] = {}
        specs = {
            "reward": ("reward", "reward_value", "reward_trunc_next_value"),
            "collision": ("collision_cost", "collision_value", "collision_trunc_next_value"),
            "performance": (
                "performance_reward",
                "performance_value",
                "performance_trunc_next_value",
            ),
        }
        for channel, (signal_name, value_name, trunc_name) in specs.items():
            rewards = np.asarray([getattr(row.signals, signal_name) for row in self.records])
            values = np.asarray([getattr(row, value_name) for row in self.records])
            trunc_next = np.asarray([getattr(row, trunc_name) for row in self.records])
            advantage, returns = variable_discount_gae(
                rewards,
                values,
                discounts,
                terminated,
                truncated,
                trunc_next,
                last_value=float(last_values[channel]),
            )
            output[f"{channel}_advantage"] = advantage
            output[f"{channel}_return"] = returns
        output["log_prob"] = np.asarray([row.log_prob for row in self.records], dtype=np.float32)
        output["macro_length"] = np.asarray(
            [row.signals.length for row in self.records], dtype=np.int16
        )
        return output

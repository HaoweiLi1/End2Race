"""B6 phase-0 contracts for temporally coherent action exploration.

This module contains no learner.  It compares the canonical BC actor under
two keyed, common-random-number exploration processes while the actor and
simulator continue to run at 100 Hz:

* iid: ``epsilon_t = sigma * xi_t``;
* AR(1): ``epsilon_0 = sigma * xi_0`` and
  ``epsilon_t = rho * epsilon_{t-1} + sqrt(1-rho**2) * sigma * xi_t``.

The selection and innovation streams are domain separated and independent of
global NumPy/PyTorch RNG state so the audit is resumable and replayable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from bplus_v22.b4_direct import ACTION_DIM, B4DirectHeadPolicy
from bplus_v22.ppo_env import B2Scenario


B6_PHASE0_SCHEMA = "end2race-b6-temporal-exploration-phase0-1"
B6_SELECTION_SCHEMA = "end2race-b6-temporal-selection-1"
B6_NOISE_SCHEMA = "end2race-b6-keyed-common-innovation-1"
B6_MODES = ("iid", "ar1")
B6_OUTCOMES = ("collision", "overtake", "follow")
B6_RHO = 0.95
B6_INNOVATION_SEEDS = (0, 1, 2, 3)
B6_MATCHED_L4_COUNT = 60
B6_EXPECTED_EPISODES = (
    B6_MATCHED_L4_COUNT
    * len(B6_OUTCOMES)
    * len(B6_INNOVATION_SEEDS)
    * len(B6_MODES)
)

_SELECTION_DOMAIN = b"end2race:b6:temporal-phase0:selection:v1\0"
_INNOVATION_DOMAIN = b"end2race:b6:temporal-phase0:innovation:v1\0"
_ORDER_DOMAIN = b"end2race:b6:temporal-phase0:arm-order:v1\0"


def _hash(*values: object, domain: bytes) -> bytes:
    digest = hashlib.sha256(domain)
    for value in values:
        digest.update(str(value).encode("utf-8") + b"\0")
    return digest.digest()


@dataclass(frozen=True)
class B6SelectedScenario:
    matched_order: int
    archived_outcome: str
    scenario: B2Scenario

    def __post_init__(self) -> None:
        if int(self.matched_order) != self.matched_order or self.matched_order < 0:
            raise ValueError("B6 matched order is invalid")
        if self.archived_outcome not in B6_OUTCOMES:
            raise ValueError("B6 archived outcome is invalid")
        if self.scenario.archived_bc_outcome != self.archived_outcome:
            raise ValueError("B6 selected scenario outcome drift")


def select_matched_scenarios(
    groups: Mapping[str, Sequence[B2Scenario]],
) -> tuple[B6SelectedScenario, ...]:
    """Select one L2 per outcome for every L4 shared by all three outcomes."""

    if tuple(groups) != B6_OUTCOMES:
        raise ValueError("B6 selection groups/order drift")
    by_outcome: dict[str, dict[str, list[B2Scenario]]] = {}
    for outcome in B6_OUTCOMES:
        rows = tuple(groups[outcome])
        if any(row.archived_bc_outcome != outcome for row in rows):
            raise ValueError(f"B6 {outcome} source contains another outcome")
        by_l4: dict[str, list[B2Scenario]] = defaultdict(list)
        for row in rows:
            by_l4[row.l4_id].append(row)
        by_outcome[outcome] = dict(by_l4)
    shared = set.intersection(
        *(set(by_outcome[outcome]) for outcome in B6_OUTCOMES)
    )
    if len(shared) != B6_MATCHED_L4_COUNT:
        raise ValueError(
            f"B6 expected {B6_MATCHED_L4_COUNT} fully matched L4, found {len(shared)}"
        )
    ordered_l4 = sorted(
        shared,
        key=lambda value: (
            _hash("l4", value, domain=_SELECTION_DOMAIN),
            value,
        ),
    )
    selected: list[B6SelectedScenario] = []
    for matched_order, l4_id in enumerate(ordered_l4):
        triplet = []
        for outcome in B6_OUTCOMES:
            row = min(
                by_outcome[outcome][l4_id],
                key=lambda item: (
                    _hash(outcome, l4_id, item.l2_id, domain=_SELECTION_DOMAIN),
                    item.l2_id,
                ),
            )
            triplet.append(row)
            selected.append(B6SelectedScenario(matched_order, outcome, row))
        if len({row.l4_id for row in triplet}) != 1 or len(
            {row.map_name for row in triplet}
        ) != 1:
            raise ValueError("B6 matched triplet L4/map drift")
    if len(selected) != B6_MATCHED_L4_COUNT * len(B6_OUTCOMES):
        raise AssertionError("B6 selected scenario count drift")
    if len({row.scenario.l2_id for row in selected}) != len(selected):
        raise AssertionError("B6 selected L2 values are not unique")
    return tuple(selected)


def selection_digest(rows: Sequence[B6SelectedScenario]) -> str:
    digest = hashlib.sha256(_SELECTION_DOMAIN)
    digest.update(B6_SELECTION_SCHEMA.encode("ascii") + b"\0")
    for row in rows:
        digest.update(
            f"{row.matched_order}:{row.archived_outcome}:".encode("ascii")
        )
        digest.update(row.scenario.l4_id.encode("ascii") + b":")
        digest.update(row.scenario.l2_id.encode("ascii") + b"\n")
    return digest.hexdigest()


def keyed_standard_normal(
    l2_id: str,
    innovation_seed: int,
    step_index: int,
) -> np.ndarray:
    """Return two deterministic N(0,1) draws using a SHA256 Box-Muller map."""

    if not l2_id.startswith("L2:"):
        raise ValueError("B6 innovation requires an L2 identity")
    if int(innovation_seed) != innovation_seed or innovation_seed < 0:
        raise ValueError("B6 innovation seed is invalid")
    if int(step_index) != step_index or step_index < 0:
        raise ValueError("B6 innovation step is invalid")
    digest = _hash(
        B6_NOISE_SCHEMA,
        l2_id,
        int(innovation_seed),
        int(step_index),
        domain=_INNOVATION_DOMAIN,
    )
    first = int.from_bytes(digest[:8], "big")
    second = int.from_bytes(digest[8:16], "big")
    scale = float(1 << 64)
    u1 = (first + 0.5) / scale
    u2 = (second + 0.5) / scale
    radius = math.sqrt(-2.0 * math.log(u1))
    angle = 2.0 * math.pi * u2
    return np.asarray(
        [radius * math.cos(angle), radius * math.sin(angle)],
        dtype=np.float64,
    )


def arm_order(l2_id: str, innovation_seed: int) -> tuple[str, str]:
    """Balance potential process-order effects without changing either arm."""

    value = _hash(l2_id, innovation_seed, domain=_ORDER_DOMAIN)[0]
    return B6_MODES if value % 2 == 0 else tuple(reversed(B6_MODES))


def factorized_log_prob(
    raw_action: torch.Tensor,
    conditional_mean: torch.Tensor,
    conditional_std: torch.Tensor,
) -> torch.Tensor:
    if (
        raw_action.shape != conditional_mean.shape
        or raw_action.shape[-1] != ACTION_DIM
        or conditional_std.shape != (ACTION_DIM,)
    ):
        raise ValueError("B6 conditional Normal shape drift")
    std = conditional_std.to(raw_action)
    value = -0.5 * (
        ((raw_action - conditional_mean) / std) ** 2
        + math.log(2.0 * math.pi)
    ) - torch.log(std)
    return value.sum(dim=-1)


def ar1_conditional_log_prob(
    raw_action: torch.Tensor,
    current_mean: torch.Tensor,
    *,
    previous_raw_action: torch.Tensor | None,
    previous_mean: torch.Tensor | None,
    std: torch.Tensor,
    rho: float = B6_RHO,
) -> torch.Tensor:
    """AR(1) log-probability under candidate current and previous means."""

    if not 0.0 < float(rho) < 1.0:
        raise ValueError("B6 AR(1) rho is invalid")
    first = previous_raw_action is None and previous_mean is None
    if (previous_raw_action is None) != (previous_mean is None):
        raise ValueError("B6 previous raw/mean must both be present or absent")
    if first:
        conditional_mean = current_mean
        conditional_std = std
    else:
        if previous_raw_action.shape != current_mean.shape or previous_mean.shape != current_mean.shape:
            raise ValueError("B6 previous action/mean shape drift")
        previous_noise = previous_raw_action - previous_mean
        conditional_mean = current_mean + float(rho) * previous_noise
        conditional_std = std * math.sqrt(1.0 - float(rho) ** 2)
    return factorized_log_prob(raw_action, conditional_mean, conditional_std)


class B6Phase0Policy(B4DirectHeadPolicy):
    """Canonical B4 wrapper with one keyed iid or AR(1) behavior process."""

    def __init__(
        self,
        bc_state: Mapping[str, torch.Tensor],
        *,
        mode: str,
        rho: float = B6_RHO,
    ):
        if mode not in B6_MODES:
            raise ValueError("B6 phase-0 mode is invalid")
        if not 0.0 < float(rho) < 1.0:
            raise ValueError("B6 phase-0 rho is invalid")
        super().__init__(bc_state)
        self.mode = mode
        self.rho = float(rho)
        self._l2_id: str | None = None
        self._innovation_seed: int | None = None
        self._step_index = 0
        self._previous_noise: torch.Tensor | None = None
        self._noise_trace: list[np.ndarray] = []
        self._innovation_trace: list[np.ndarray] = []

    def begin_episode(self, l2_id: str, innovation_seed: int) -> None:
        if not l2_id.startswith("L2:") or int(innovation_seed) != innovation_seed:
            raise ValueError("B6 episode key is invalid")
        self._l2_id = l2_id
        self._innovation_seed = int(innovation_seed)
        self._step_index = 0
        self._previous_noise = None
        self._noise_trace = []
        self._innovation_trace = []

    @property
    def noise_trace(self) -> np.ndarray:
        if not self._noise_trace:
            return np.empty((0, ACTION_DIM), dtype=np.float64)
        return np.stack(self._noise_trace)

    @property
    def innovation_trace(self) -> np.ndarray:
        if not self._innovation_trace:
            return np.empty((0, ACTION_DIM), dtype=np.float64)
        return np.stack(self._innovation_trace)

    def sample_raw(self, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._l2_id is None or self._innovation_seed is None:
            raise RuntimeError("B6 begin_episode must precede sampling")
        if mean.shape != (1, ACTION_DIM):
            raise ValueError("B6 phase-0 sampler requires batch size one")
        innovation_np = keyed_standard_normal(
            self._l2_id,
            self._innovation_seed,
            self._step_index,
        )
        innovation = torch.as_tensor(innovation_np, dtype=mean.dtype, device=mean.device).reshape(1, -1)
        std = self.action_std.to(mean)
        if self._step_index == 0 or self.mode == "iid":
            noise = std * innovation
        else:
            if self._previous_noise is None:
                raise AssertionError("B6 AR(1) noise state is missing")
            noise = self.rho * self._previous_noise + math.sqrt(
                1.0 - self.rho**2
            ) * std * innovation
        raw = mean + noise
        if self.mode == "iid" or self._step_index == 0:
            log_prob = factorized_log_prob(raw, mean, std)
        else:
            conditional_mean = mean + self.rho * self._previous_noise
            conditional_std = std * math.sqrt(1.0 - self.rho**2)
            log_prob = factorized_log_prob(raw, conditional_mean, conditional_std)
        self._previous_noise = noise.detach()
        self._noise_trace.append(noise[0].detach().cpu().double().numpy())
        self._innovation_trace.append(innovation_np.copy())
        self._step_index += 1
        return raw, log_prob


def exact_cluster_signflip_one_sided(effects: Sequence[int]) -> float:
    """Exact conditional sign-flip tail probability for integer cluster effects."""

    values = tuple(int(value) for value in effects)
    if not values:
        raise ValueError("B6 sign-flip requires at least one cluster")
    observed = sum(values)
    distribution: Counter[int] = Counter({0: 1})
    for value in values:
        updated: Counter[int] = Counter()
        for total, count in distribution.items():
            updated[total + value] += count
            updated[total - value] += count
        distribution = updated
    numerator = sum(count for total, count in distribution.items() if total >= observed)
    denominator = 1 << len(values)
    if sum(distribution.values()) != denominator:
        raise AssertionError("B6 sign-flip state count drift")
    return float(numerator / denominator)


def paired_cluster_bootstrap(
    effects: Sequence[float],
    *,
    seed: int,
    samples: int = 200_000,
) -> dict[str, float]:
    values = np.asarray(tuple(effects), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("B6 bootstrap effects are invalid")
    generator = np.random.default_rng(int(seed))
    draws = np.empty(int(samples), dtype=np.float64)
    batch = 10_000
    for start in range(0, int(samples), batch):
        stop = min(start + batch, int(samples))
        indices = generator.integers(0, len(values), size=(stop - start, len(values)))
        draws[start:stop] = values[indices].mean(axis=1)
    return {
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "upper_one_sided_90": float(np.quantile(draws, 0.90)),
    }


def trace_moments(noise: np.ndarray) -> dict[str, Any]:
    value = np.asarray(noise, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM or len(value) == 0:
        raise ValueError("B6 noise trace is invalid")
    result: dict[str, Any] = {
        "count": int(len(value)),
        "sum": value.sum(axis=0).tolist(),
        "sum_sq": np.square(value).sum(axis=0).tolist(),
        "lag_count": max(0, int(len(value) - 1)),
        "lag_prev_sum": value[:-1].sum(axis=0).tolist(),
        "lag_next_sum": value[1:].sum(axis=0).tolist(),
        "lag_prev_sq": np.square(value[:-1]).sum(axis=0).tolist(),
        "lag_next_sq": np.square(value[1:]).sum(axis=0).tolist(),
        "lag_cross": (value[:-1] * value[1:]).sum(axis=0).tolist(),
        "windows": {},
    }
    for window in (10, 30, 50):
        if len(value) < window:
            means = np.empty((0, ACTION_DIM), dtype=np.float64)
        else:
            cumulative = np.vstack(
                (np.zeros((1, ACTION_DIM)), np.cumsum(value, axis=0))
            )
            means = (cumulative[window:] - cumulative[:-window]) / window
        result["windows"][str(window)] = {
            "count": int(len(means)),
            "sum_sq": np.square(means).sum(axis=0).tolist(),
        }
    return result

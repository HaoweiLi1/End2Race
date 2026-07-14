#!/usr/bin/env python3
"""Contract and production-shaped smoke checks for B6 temporal phase-0."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from bplus_v22.b4_direct import B4ScenarioSets
from bplus_v22.b4_env import run_b4_episode
from bplus_v22.b6_temporal import (
    B6_INNOVATION_SEEDS,
    B6_MATCHED_L4_COUNT,
    B6_OUTCOMES,
    B6Phase0Policy,
    ar1_conditional_log_prob,
    exact_cluster_signflip_one_sided,
    keyed_standard_normal,
    select_matched_scenarios,
    selection_digest,
)
from bplus_v22.ppo_env import load_b2_scenario_sets


REPO = Path(__file__).resolve().parent.parent
TASK8 = REPO / "Experiments/B1_route_r2_scaffold/artifacts/task8_manifests_20260712_113241"
METADATA = REPO / "Experiments/A3_d2_representation/artifacts/non_test_full_20260711_175713/episode_metadata.tsv"
BC = REPO / "pretrained/end2race.pth"


def _selection():
    sets = B4ScenarioSets.from_b2(load_b2_scenario_sets(TASK8, METADATA))
    return select_matched_scenarios(
        {"collision": sets.collision, "overtake": sets.overtake, "follow": sets.follow}
    )


def main() -> None:
    selected = _selection()
    assert len(selected) == B6_MATCHED_L4_COUNT * len(B6_OUTCOMES)
    assert len(selection_digest(selected)) == 64
    counts = Counter((row.matched_order, row.archived_outcome) for row in selected)
    assert set(counts.values()) == {1}
    for matched_order in range(B6_MATCHED_L4_COUNT):
        triplet = [row for row in selected if row.matched_order == matched_order]
        assert {row.archived_outcome for row in triplet} == set(B6_OUTCOMES)
        assert len({row.scenario.l4_id for row in triplet}) == 1
        assert len({row.scenario.map_name for row in triplet}) == 1

    l2_id = selected[0].scenario.l2_id
    first = keyed_standard_normal(l2_id, B6_INNOVATION_SEEDS[0], 0)
    np.random.seed(999)
    torch.manual_seed(999)
    assert np.array_equal(first, keyed_standard_normal(l2_id, B6_INNOVATION_SEEDS[0], 0))
    assert not np.array_equal(first, keyed_standard_normal(l2_id, B6_INNOVATION_SEEDS[0], 1))

    raw = torch.tensor([[0.3, 4.7]], dtype=torch.float64)
    current = torch.tensor([[0.2, 4.5]], dtype=torch.float64)
    previous_raw = torch.tensor([[0.15, 4.8]], dtype=torch.float64)
    previous_mean = torch.tensor([[0.10, 4.4]], dtype=torch.float64)
    std = torch.tensor([0.03, 0.20], dtype=torch.float64)
    rho = 0.95
    expected_mean = current + rho * (previous_raw - previous_mean)
    expected_std = std * np.sqrt(1.0 - rho**2)
    expected = (
        -0.5 * (((raw - expected_mean) / expected_std) ** 2 + np.log(2.0 * np.pi))
        - torch.log(expected_std)
    ).sum(dim=-1)
    observed = ar1_conditional_log_prob(
        raw,
        current,
        previous_raw_action=previous_raw,
        previous_mean=previous_mean,
        std=std,
        rho=rho,
    )
    assert torch.allclose(observed, expected, atol=0.0, rtol=0.0)
    changed_previous_mean = ar1_conditional_log_prob(
        raw,
        current,
        previous_raw_action=previous_raw,
        previous_mean=previous_mean + 0.1,
        std=std,
        rho=rho,
    )
    assert not torch.equal(observed, changed_previous_mean)

    # For cluster effects [1, 1], only one of four sign assignments reaches
    # the observed sum of two.
    assert exact_cluster_signflip_one_sided([1, 1]) == 0.25
    assert exact_cluster_signflip_one_sided([0, 0, 0]) == 1.0

    device = torch.device("cpu")
    state = torch.load(BC, map_location="cpu", weights_only=True)
    policies = {
        mode: B6Phase0Policy(state, mode=mode).to(device) for mode in ("iid", "ar1")
    }
    scenario = next(row.scenario for row in selected if row.archived_outcome == "collision")
    results = {}
    for mode, policy in policies.items():
        policy.begin_episode(scenario.l2_id, 0)
        results[mode] = run_b4_episode(
            policy,
            device,
            scenario,
            episode_id=0 if mode == "iid" else 1,
            deterministic=False,
            sim_duration=0.05,
        )
        assert results[mode].step_count == len(policy.noise_trace)
        assert results[mode].step_count == len(policy.innovation_trace)
        assert results[mode].speed_projection_count == 0
    common = min(results["iid"].step_count, results["ar1"].step_count)
    assert np.array_equal(
        policies["iid"].innovation_trace[:common],
        policies["ar1"].innovation_trace[:common],
    )
    assert np.array_equal(
        policies["iid"].noise_trace[0], policies["ar1"].noise_trace[0]
    )
    assert not np.array_equal(
        policies["iid"].noise_trace[1], policies["ar1"].noise_trace[1]
    )
    print("B6 temporal phase-0 contracts and production-shaped smoke passed")


if __name__ == "__main__":
    main()

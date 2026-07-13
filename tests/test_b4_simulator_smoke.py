#!/usr/bin/env python3
"""Production-shaped deterministic identity and stochastic PPO smoke for B4."""

from pathlib import Path

import numpy as np
import torch

from bplus_v22.b4_direct import B4DirectHeadPolicy, B4ScenarioSets
from bplus_v22.b4_env import run_b4_episode
from bplus_v22.b4_runner import (
    B4_REPLAY_RATIO_ATOL,
    run_b4_stochastic_plumbing_smoke,
)
from bplus_v22.ppo_env import load_b2_scenario_sets
from d25.oracle import simulate_episode


REPO = Path(__file__).resolve().parent.parent
BC = REPO / "pretrained/end2race.pth"
TASK8 = REPO / "Experiments/B1_route_r2_scaffold/artifacts/task8_manifests_20260712_113241"
METADATA = (
    REPO
    / "Experiments/A3_d2_representation/artifacts/non_test_full_20260711_175713/episode_metadata.tsv"
)
MAPS = ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")


def main() -> None:
    device = torch.device("cpu")
    bc_state = torch.load(BC, map_location="cpu", weights_only=True)
    policy = B4DirectHeadPolicy(bc_state).to(device)
    scenarios = B4ScenarioSets.from_b2(load_b2_scenario_sets(TASK8, METADATA))
    selected = {}
    for scenario in sorted(
        (*scenarios.collision, *scenarios.overtake, *scenarios.follow),
        key=lambda row: row.training_order,
    ):
        selected.setdefault(scenario.map_name, scenario)
    assert tuple(sorted(selected)) == tuple(sorted(MAPS))

    reports = []
    for episode_id, map_name in enumerate(MAPS):
        scenario = selected[map_name]
        reference = simulate_episode(policy.actor, device, scenario.simulator_case())
        result = run_b4_episode(
            policy,
            device,
            scenario,
            episode_id=episode_id,
            deterministic=True,
        )
        names = sorted(set(reference.arrays) | set(result.arrays))
        mismatches = [
            name
            for name in names
            if name not in reference.arrays
            or name not in result.arrays
            or not np.array_equal(
                np.asarray(reference.arrays[name]), np.asarray(result.arrays[name])
            )
        ]
        outcome_identity = (
            reference.outcome.four_state == result.outcome.four_state
            and bool(reference.outcome.collision_any)
            == bool(result.outcome.collision_any)
            and reference.outcome.corrected_outcome3
            == result.outcome.corrected_outcome3
        )
        features = torch.from_numpy(
            np.stack([row.feature for row in result.transitions])
        ).to(device)
        raw = torch.from_numpy(
            np.stack([row.raw_action for row in result.transitions])
        ).to(device)
        old = torch.tensor(
            [row.old_log_prob for row in result.transitions], dtype=torch.float32
        )
        with torch.no_grad():
            replayed = policy.log_prob(policy.mean_from_feature(features), raw)
        max_log_prob_delta = float(torch.max(torch.abs(replayed - old)).item())
        assert not mismatches
        assert outcome_identity
        assert result.speed_projection_count == 0
        assert max_log_prob_delta <= B4_REPLAY_RATIO_ATOL
        reports.append(
            (
                map_name,
                result.step_count,
                result.terminal_reason,
                result.speed_projection_count,
                result.steer_projection_count,
                max_log_prob_delta,
            )
        )
    policy.assert_frozen_exact()
    assert reports == [
        (map_name, 801, "product_horizon", 0, 0, 0.0) for map_name in MAPS
    ]
    stochastic = run_b4_stochastic_plumbing_smoke(bc_state, device, scenarios)
    assert [row["terminal_reason"] for row in stochastic["episode_reports"]] == [
        "any_agent_collision",
        "product_horizon",
        "product_horizon",
    ]
    assert [row["terminal_reward"] for row in stochastic["episode_reports"]] == [
        -2.0,
        0.0,
        1.0,
    ]
    assert stochastic["raw_stored_latent_exact"] is True
    assert stochastic["dense_reward_excluded_from_reward_advantage_return"] is True
    assert stochastic["actor_early_stop_exercised"] is True
    assert stochastic["critic_epochs_completed"] == 3
    assert stochastic["plain_actor_strict_load"] is True
    assert stochastic["full_checkpoint_recovery"] is True
    print("B4 production-shaped deterministic + stochastic plumbing smoke passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""B2 scenario/curriculum/history/critic-state contracts."""

from pathlib import Path

import numpy as np
import torch

from bplus_v22.ppo_env import (
    ActorHistory,
    B2Curriculum,
    build_privileged_features,
    load_b2_scenario_sets,
)
from ppo_utils import RewardState
from utils import load_reference_line


ROOT = Path(__file__).resolve().parents[1]
TASK8 = ROOT / "Experiments/B1_route_r2_scaffold/artifacts/task8_manifests_20260712_113241"
METADATA = ROOT / (
    "Experiments/A3_d2_representation/artifacts/"
    "non_test_full_20260711_175713/episode_metadata.tsv"
)


def test_scenario_partition_and_frozen_curriculum():
    scenarios = load_b2_scenario_sets(TASK8, METADATA)
    assert len(scenarios.collision) == 81
    assert len(scenarios.remaining) == 1559
    assert len(scenarios.development_rows) == 288
    assert not ({row.l2_id for row in scenarios.collision} & {row.l2_id for row in scenarios.remaining})
    seed0 = B2Curriculum(scenarios, 0)
    plan = seed0.plan()
    assert len(plan) == 20
    assert all(len(rows) == 16 for rows in plan)
    assert all(sum(row.bc_collision_any for row in rows) == 8 for rows in plan)
    assert plan == B2Curriculum(scenarios, 0).plan()
    assert plan != B2Curriculum(scenarios, 1).plan()
    assert seed0.digest() == B2Curriculum(scenarios, 0).digest()
    assert len({row.l2_id for rows in plan for row in rows if row.bc_collision_any}) == 81


def test_actor_history_clamps_episode_start_and_uses_command_history():
    history = ActorHistory()
    lidar0 = torch.arange(360, dtype=torch.float32).reshape(1, 360) / 10.0
    history.append(lidar0, torch.tensor([[5.0]]), torch.zeros(1, 2))
    lidar, scalar = history.tensors()
    assert lidar.shape == (1, 8, 360)
    assert scalar.shape == (1, 24)
    assert torch.equal(lidar[:, 0], lidar[:, -1])
    assert torch.all(scalar[:, 8:]) == 0

    history.append(
        torch.ones(1, 360) * 15.0,
        torch.tensor([[6.0]]),
        torch.tensor([[0.26, 8.0]]),
    )
    _, scalar = history.tensors()
    assert float(scalar[0, 0]) == np.float32(0.6)
    assert float(scalar[0, 8]) == np.float32(0.5)
    assert float(scalar[0, 16]) == np.float32(0.8)
    assert float(scalar[0, 7]) == np.float32(0.5)
    assert float(scalar[0, 15]) == 0.0
    assert float(scalar[0, 23]) == 0.0


def test_privileged_feature_exact_schema_is_finite():
    reference = load_reference_line("Austin", "raceline1")
    # Use two distinct reference points so projection and track phase are valid.
    ego = reference.xy[10]
    opp = reference.xy[20]
    ego_theta = float(np.arctan2(*(reference.xy[11] - reference.xy[10])[::-1]))
    opp_theta = float(np.arctan2(*(reference.xy[21] - reference.xy[20])[::-1]))
    obs = {
        "poses_x": np.asarray([ego[0], opp[0]], dtype=np.float64),
        "poses_y": np.asarray([ego[1], opp[1]], dtype=np.float64),
        "poses_theta": np.asarray([ego_theta, opp_theta], dtype=np.float64),
        "linear_vels_x": np.asarray([5.0, 4.0], dtype=np.float64),
    }
    state = RewardState(0.0, 0.0, True)
    state.safe_overtake_hold_time = 0.35
    state.overtake_started = True
    state.safe_overtake_held = False
    value = build_privileged_features(obs, reference, state, 0.7)
    assert value.shape == (12,)
    assert np.all(np.isfinite(value))
    assert abs(float(value[2])) <= 0.5
    assert abs(float(value[3])) <= 0.4
    assert value[6] == np.float32(0.5)
    assert value[7] == np.float32(1.0)
    assert value[8] == np.float32(0.0)
    assert value[9] == np.float32(0.7)


if __name__ == "__main__":
    test_scenario_partition_and_frozen_curriculum()
    test_actor_history_clamps_episode_start_and_uses_command_history()
    test_privileged_feature_exact_schema_is_finite()
    print("ALL TESTS PASSED")

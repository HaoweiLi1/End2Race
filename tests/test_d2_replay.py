#!/usr/bin/env python3
"""Tests for exact framewise BC feature replay."""

import numpy as np
import torch

from d2.replay import ReplayMismatch, build_speed_inputs, replay_bc_features
from model import End2Race


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def expect_raises(name, exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"FAIL {name}: expected {exc_type.__name__}")


@torch.no_grad()
def archive_actions(model, lidar, actual_speed, initial_speed):
    speed_inputs = build_speed_inputs(actual_speed, initial_speed)
    hidden = torch.zeros(1, 1, model.gru.hidden_size)
    actions = []
    for t in range(len(lidar)):
        x = torch.as_tensor(lidar[t], dtype=torch.float32).view(1, 1, -1)
        v = torch.tensor([[[speed_inputs[t]]]], dtype=torch.float32)
        feature, hidden = model.forward_features(x, v, hidden)
        action = model.output_layer(feature)[0, 0]
        actions.append([float(np.clip(action[0].item(), -0.52, 0.52)), action[1].item()])
    return np.asarray(actions, dtype=np.float32)


def main():
    torch.manual_seed(7)
    model = End2Race(mask_prob=0.0, hidden_scale=1).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    rng = np.random.default_rng(11)
    lidar = rng.uniform(0.1, 10.0, size=(7, 360)).astype(np.float32)
    actual = np.linspace(0.0, 3.0, len(lidar), dtype=np.float32)
    initial = 4.5
    archived = archive_actions(model, lidar, actual, initial)

    expected_inputs = np.concatenate([[initial], actual[:-1]]).astype(np.float32)
    check("one-frame-lag-input", np.array_equal(build_speed_inputs(actual, initial), expected_inputs))

    result = replay_bc_features(
        model,
        lidar,
        actual,
        initial,
        archived[:, 0],
        archived[:, 1],
        torch.device("cpu"),
    )
    check("feature-shape", result.features.shape == (7, model.gru.hidden_size))
    check("feature-float32", result.features.dtype == np.float32)
    check("exact-actions", result.mismatched_frames == 0 and result.max_abs_error == (0.0, 0.0))
    check("generic-device-alias", result.features.shape[0] == len(lidar))

    repeated = replay_bc_features(
        model,
        lidar,
        actual,
        initial,
        archived[:, 0],
        archived[:, 1],
        torch.device("cpu"),
    )
    check("hidden-reset-determinism", np.array_equal(result.features, repeated.features))

    wrong_steer = archived[:, 0].copy()
    wrong_steer[3] = np.nextafter(wrong_steer[3], np.float32(np.inf))
    expect_raises(
        "one-bit-action-mismatch",
        ReplayMismatch,
        lambda: replay_bc_features(
            model, lidar, actual, initial, wrong_steer, archived[:, 1], torch.device("cpu")
        ),
    )

    current_speed_archive = archive_actions(model, lidar, np.concatenate([actual[1:], actual[-1:]]), actual[0])
    expect_raises(
        "current-speed-contract-rejected",
        ReplayMismatch,
        lambda: replay_bc_features(
            model,
            lidar,
            actual,
            initial,
            current_speed_archive[:, 0],
            current_speed_archive[:, 1],
            torch.device("cpu"),
        ),
    )

    expect_raises(
        "length-mismatch",
        ValueError,
        lambda: replay_bc_features(
            model,
            lidar[:-1],
            actual,
            initial,
            archived[:, 0],
            archived[:, 1],
            torch.device("cpu"),
        ),
    )

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

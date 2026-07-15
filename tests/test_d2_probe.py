#!/usr/bin/env python3
"""Unit tests for D2 probe sampling, normalization, and model heads."""

import numpy as np
import torch

from d2.models import (
    ProbeNet,
    DeployableTemporalFeatureView,
    TemporalFeatureView,
    compute_normalization,
    decode_predictions,
    deterministic_fit_indices,
    apply_platt_calibrator,
    fit_platt_calibrator,
    inverse_sampling_weights,
)


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def main():
    features = np.arange(120 * 4, dtype=np.float32).reshape(120, 4)
    episode_index = np.repeat(np.arange(12), 10)
    train_episode = np.zeros(12, dtype=bool)
    train_episode[:8] = True
    any_target_200 = np.zeros(120, dtype=np.uint8)
    any_target_200[[3, 77, 95]] = 1
    ttc = np.full(120, 5.0, dtype=np.float32)
    ttc[[4, 78, 96]] = 1.0
    indices = deterministic_fit_indices(
        episode_index,
        train_episode,
        any_target_200,
        ttc,
        background_stride=4,
    )
    check("train-episodes-only", np.all(episode_index[indices] < 8))
    check("positive-retained", 3 in indices and 77 in indices)
    check("heldout-positive-excluded", 95 not in indices)
    check("critical-ttc-retained", 4 in indices and 78 in indices)
    check("heldout-ttc-excluded", 96 not in indices)
    weights = inverse_sampling_weights(indices, any_target_200, ttc, background_stride=4)
    by_index = dict(zip(indices.tolist(), weights.tolist()))
    check("positive-unit-weight", by_index[3] == 1.0 and by_index[77] == 1.0)
    check("critical-ttc-unit-weight", by_index[4] == 1.0 and by_index[78] == 1.0)
    check("background-inverse-weight", by_index[0] == 4.0 and by_index[8] == 4.0)

    mean1, std1 = compute_normalization(features, indices, chunk_size=7)
    poisoned = features.copy()
    poisoned[episode_index >= 8] = 1e9
    mean2, std2 = compute_normalization(poisoned, indices, chunk_size=9)
    check("normalization-heldout-blind", np.array_equal(mean1, mean2) and np.array_equal(std1, std2))
    check("normalization-finite", np.all(np.isfinite(mean1)) and np.all(std1 > 0))

    torch.manual_seed(1)
    linear = ProbeNet("linear", input_dim=4)
    mlp = ProbeNet("mlp", input_dim=4)
    temporal = ProbeNet("temporal", input_dim=8)
    x = torch.zeros(5, 4)
    check("linear-head-shape", linear(x).shape == (5, 8))
    check("mlp-head-shape", mlp(x).shape == (5, 8))
    check("temporal-head-shape", temporal(torch.zeros(5, 8)).shape == (5, 8))
    decoded = decode_predictions(mlp(x)).detach().numpy()
    check("decoded-shape", decoded.shape == (5, 8))
    check("probability-bounds", np.all((decoded[:, :6] >= 0) & (decoded[:, :6] <= 1)))
    check("ttc-bounds", np.all((decoded[:, 7] >= 0) & (decoded[:, 7] <= 5)))

    calibration_target = np.array([0] * 90 + [1] * 10, dtype=bool)
    uncalibrated = np.where(calibration_target, 0.9, 0.2).astype(np.float64)
    calibrator = fit_platt_calibrator(
        uncalibrated, calibration_target, np.ones(100, dtype=bool)
    )
    calibrated = apply_platt_calibrator(uncalibrated, calibrator)
    before = np.mean((uncalibrated - calibration_target) ** 2)
    after = np.mean((calibrated - calibration_target) ** 2)
    check("platt-improves-brier", after < before)
    check("platt-bounds", np.all((calibrated >= 0) & (calibrated <= 1)))

    base = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
    temporal_view = TemporalFeatureView(
        base,
        episode_index=np.repeat([0, 1], 6),
        episode_starts=np.array([0, 6]),
        offsets=(0, 2, 5),
    )
    tapped = temporal_view[np.array([0, 1, 5, 6, 7, 11])]
    check("temporal-shape", tapped.shape == (6, 6))
    check("temporal-start-padding-ep0", np.array_equal(tapped[1], np.concatenate([base[1], base[0], base[0]])))
    check("temporal-start-padding-ep1", np.array_equal(tapped[4], np.concatenate([base[7], base[6], base[6]])))
    check("temporal-no-cross-episode", np.array_equal(tapped[3], np.concatenate([base[6], base[6], base[6]])))

    frame_count = 120
    base_deploy = np.arange(frame_count * 2, dtype=np.float32).reshape(frame_count, 2)
    lidar = np.repeat(np.arange(frame_count, dtype=np.float32)[:, None], 360, axis=1)
    scalar_signal = np.arange(frame_count, dtype=np.float32)
    deploy_view = DeployableTemporalFeatureView(
        base_deploy,
        episode_index=np.repeat([0, 1], 60),
        episode_starts=np.array([0, 60]),
        ego_lidar=lidar,
        ego_actual_speed=scalar_signal,
        previous_desired_steer=scalar_signal + 1,
        previous_desired_speed=scalar_signal + 2,
    )
    deploy = deploy_view[np.array([5, 55, 60, 65, 119])]
    check("deployable-temporal-dim", deploy.shape == (5, 1094))
    check("deployable-no-cross", np.all(deploy[2, 2:1082] == 0.0))
    check("deployable-early-clamp", np.all(deploy[0, 2:362] == 5.0))

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

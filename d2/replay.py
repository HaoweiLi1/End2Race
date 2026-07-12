"""Exact batch-1, framewise replay of the frozen BC recurrent actor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


class ReplayMismatch(RuntimeError):
    """Archived actions were not reproduced bit-for-bit."""


@dataclass(frozen=True)
class ReplayResult:
    features: np.ndarray
    predicted_actions: np.ndarray
    mismatched_frames: int
    max_abs_error: tuple[float, float]


def build_speed_inputs(actual_speed: np.ndarray, initial_speed_input: float) -> np.ndarray:
    actual = np.asarray(actual_speed, dtype=np.float32)
    if actual.ndim != 1 or len(actual) == 0 or not np.all(np.isfinite(actual)):
        raise ValueError("actual_speed must be a nonempty finite vector")
    initial = float(initial_speed_input)
    if not math.isfinite(initial):
        raise ValueError("initial speed input must be finite")
    output = np.empty_like(actual, dtype=np.float32)
    output[0] = np.float32(initial)
    output[1:] = actual[:-1]
    return output


def _vector(value, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} length mismatch")
    return array


@torch.no_grad()
def replay_bc_features(
    model,
    lidar: np.ndarray,
    actual_speed: np.ndarray,
    initial_speed_input: float,
    archived_desired_steer: np.ndarray,
    archived_desired_speed: np.ndarray,
    device: torch.device,
) -> ReplayResult:
    """Replay one episode and require exact archived action identity.

    The evaluator calls the GRU once per frame with batch size one. This
    function intentionally mirrors that execution order; whole-sequence
    forwarding is not used as the equality oracle.
    """
    lidar = np.asarray(lidar, dtype=np.float32)
    if lidar.ndim != 2 or lidar.shape[1] != 360 or len(lidar) == 0:
        raise ValueError("lidar must have shape [T, 360]")
    if not np.all(np.isfinite(lidar)):
        raise ValueError("lidar contains non-finite values")
    n = len(lidar)
    actual = _vector(actual_speed, "actual_speed", n)
    archived_steer = _vector(archived_desired_steer, "archived_desired_steer", n)
    archived_speed = _vector(archived_desired_speed, "archived_desired_speed", n)
    speed_inputs = build_speed_inputs(actual, initial_speed_input)

    parameter = next(model.parameters(), None)
    requested = torch.device(device)
    device_matches = parameter is not None and parameter.device.type == requested.type
    if device_matches and requested.index is not None:
        device_matches = parameter.device.index == requested.index
    if not device_matches:
        raise ValueError("BC model is not resident on the requested replay device")
    if model.training:
        raise ValueError("BC model must be in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("BC model parameters must be frozen")

    hidden = torch.zeros((1, 1, model.gru.hidden_size), dtype=torch.float32, device=device)
    features = np.empty((n, model.gru.hidden_size), dtype=np.float32)
    predicted = np.empty((n, 2), dtype=np.float32)
    for frame in range(n):
        lidar_t = torch.as_tensor(lidar[frame], dtype=torch.float32, device=device).view(1, 1, 360)
        speed_t = torch.tensor(
            [[[speed_inputs[frame]]]], dtype=torch.float32, device=device
        )
        feature_t, hidden = model.forward_features(lidar_t, speed_t, hidden)
        action_t = model.output_layer(feature_t)[0, 0]
        features[frame] = feature_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        predicted[frame, 0] = np.float32(np.clip(action_t[0].item(), -0.52, 0.52))
        predicted[frame, 1] = np.float32(action_t[1].item())

    archived = np.column_stack([archived_steer, archived_speed]).astype(np.float32, copy=False)
    mismatch_mask = np.any(predicted.view(np.uint32) != archived.view(np.uint32), axis=1)
    abs_error = np.abs(predicted.astype(np.float64) - archived.astype(np.float64))
    max_error = (float(np.max(abs_error[:, 0])), float(np.max(abs_error[:, 1])))
    mismatch_count = int(np.count_nonzero(mismatch_mask))
    if mismatch_count:
        first = int(np.flatnonzero(mismatch_mask)[0])
        raise ReplayMismatch(
            f"BC replay mismatch in {mismatch_count}/{n} frames; first={first}; "
            f"pred={predicted[first].tolist()} archived={archived[first].tolist()} "
            f"max_abs_error={max_error}"
        )
    return ReplayResult(
        features=features,
        predicted_actions=predicted,
        mismatched_frames=0,
        max_abs_error=max_error,
    )

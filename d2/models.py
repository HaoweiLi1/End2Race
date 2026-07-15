"""Locked linear and MLP multi-task heads for D2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import expit


CLASSIFICATION_ARRAYS = (
    "ego_target_050",
    "ego_target_100",
    "ego_target_200",
    "any_target_050",
    "any_target_100",
    "any_target_200",
)
VALID_ARRAYS = tuple(name.replace("target", "valid") for name in CLASSIFICATION_ARRAYS)
PREDICTION_NAMES = (
    "ego_probability_050",
    "ego_probability_100",
    "ego_probability_200",
    "any_probability_050",
    "any_probability_100",
    "any_probability_200",
    "closing_rate",
    "corridor_ttc",
)


@dataclass(frozen=True)
class TrainConfig:
    family: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 1e-4
    background_stride: int = 10
    closing_scale: float = 5.0
    ttc_scale: float = 5.0
    closing_loss_weight: float = 0.25
    ttc_loss_weight: float = 0.50
    # TTC<2 is 4.5% of non-test frames; 25x gives that gate region roughly
    # equal effective mass to the capped background without discarding it.
    ttc_critical_weight: float = 25.0

    def __post_init__(self):
        if self.family not in {"linear", "mlp", "temporal", "temporal_deployable"}:
            raise ValueError("unknown locked probe family")


LOCKED_CONFIGS = {
    "linear": TrainConfig("linear", epochs=8, batch_size=2048, learning_rate=3e-3),
    "mlp": TrainConfig("mlp", epochs=8, batch_size=1024, learning_rate=1e-3),
    "temporal": TrainConfig("temporal", epochs=8, batch_size=512, learning_rate=1e-3),
    "temporal_deployable": TrainConfig(
        "temporal_deployable", epochs=8, batch_size=512, learning_rate=1e-3
    ),
}


class TemporalFeatureView:
    """Causal capacity-matched taps at 0, 0.10, 0.25, and 0.50 seconds."""

    def __init__(self, base_features, episode_index, episode_starts, offsets=(0, 10, 25, 50)):
        self.base = base_features
        self.episode_index = np.asarray(episode_index, dtype=np.int64)
        self.episode_starts = np.asarray(episode_starts, dtype=np.int64)
        self.offsets = tuple(int(value) for value in offsets)
        if self.episode_index.shape != (len(base_features),):
            raise ValueError("temporal episode-index shape mismatch")
        if not self.offsets or self.offsets[0] != 0 or any(value < 0 for value in self.offsets):
            raise ValueError("temporal offsets must start at zero and be nonnegative")
        self.shape = (len(base_features), int(base_features.shape[1]) * len(self.offsets))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        indices = np.asarray(item, dtype=np.int64)
        scalar = indices.ndim == 0
        indices = indices.reshape(-1)
        starts = self.episode_starts[self.episode_index[indices]]
        taps = [
            np.asarray(self.base[np.maximum(starts, indices - offset)], dtype=np.float32)
            for offset in self.offsets
        ]
        output = np.concatenate(taps, axis=1)
        return output[0] if scalar else output


class DeployableTemporalFeatureView:
    """Current frozen feature plus causal LiDAR deltas/speed/command history."""

    def __init__(
        self,
        base_features,
        episode_index,
        episode_starts,
        ego_lidar,
        ego_actual_speed,
        previous_desired_steer,
        previous_desired_speed,
        offsets=(0, 10, 25, 50),
    ):
        self.base = base_features
        self.episode_index = np.asarray(episode_index, dtype=np.int64)
        self.episode_starts = np.asarray(episode_starts, dtype=np.int64)
        self.lidar = ego_lidar
        self.actual_speed = ego_actual_speed
        self.previous_steer = previous_desired_steer
        self.previous_speed = previous_desired_speed
        self.offsets = tuple(int(value) for value in offsets)
        frame_count = len(base_features)
        if self.episode_index.shape != (frame_count,) or any(
            len(value) != frame_count
            for value in (ego_lidar, ego_actual_speed, previous_desired_steer, previous_desired_speed)
        ):
            raise ValueError("deployable temporal signal length mismatch")
        if self.offsets != (0, 10, 25, 50):
            raise ValueError("deployable temporal offsets are locked to 0/10/25/50")
        # 1680 current feature + 3*360 LiDAR deltas + 4 actual speeds
        # + 4 previous steer + 4 previous speed = 2772.
        self.shape = (frame_count, int(base_features.shape[1]) + 3 * 360 + 12)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        indices = np.asarray(item, dtype=np.int64)
        scalar = indices.ndim == 0
        indices = indices.reshape(-1)
        starts = self.episode_starts[self.episode_index[indices]]
        history_indices = [np.maximum(starts, indices - offset) for offset in self.offsets]
        current_lidar = np.asarray(self.lidar[indices], dtype=np.float32)
        lidar_deltas = [
            current_lidar - np.asarray(self.lidar[history_indices[position]], dtype=np.float32)
            for position in range(1, len(self.offsets))
        ]
        scalar_history = []
        for signal in (self.actual_speed, self.previous_steer, self.previous_speed):
            scalar_history.append(
                np.column_stack(
                    [np.asarray(signal[history], dtype=np.float32) for history in history_indices]
                )
            )
        output = np.concatenate(
            [np.asarray(self.base[indices], dtype=np.float32), *lidar_deltas, *scalar_history],
            axis=1,
        )
        return output[0] if scalar else output


class ProbeNet(nn.Module):
    def __init__(self, family: str, input_dim: int = 1680):
        super().__init__()
        if family == "linear":
            self.net = nn.Linear(input_dim, 8)
        elif family == "mlp":
            self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 8))
        elif family == "temporal":
            self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 8))
        elif family == "temporal_deployable":
            self.net = nn.Sequential(nn.Linear(input_dim, 77), nn.ReLU(), nn.Linear(77, 8))
        else:
            raise ValueError("unknown D2 probe family")
        self.family = family
        self.input_dim = int(input_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def decode_predictions(raw: torch.Tensor, closing_scale: float = 5.0, ttc_scale: float = 5.0) -> torch.Tensor:
    probabilities = torch.sigmoid(raw[:, :6])
    closing = raw[:, 6:7] * float(closing_scale)
    ttc = torch.sigmoid(raw[:, 7:8]) * float(ttc_scale)
    return torch.cat([probabilities, closing, ttc], dim=1)


def fit_platt_calibrator(probability, target, valid) -> dict:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if probability.shape != target.shape or target.shape != valid.shape or probability.ndim != 1:
        raise ValueError("Platt calibration arrays have inconsistent shapes")
    p = np.clip(probability[valid], 1e-6, 1.0 - 1e-6)
    y = target[valid].astype(np.float64)
    if len(p) == 0 or np.all(y == y[0]):
        raise ValueError("Platt calibration requires nonempty data with both classes")
    x = np.log(p) - np.log1p(-p)

    def objective(parameters):
        slope, intercept = parameters
        z = slope * x + intercept
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        residual = expit(z) - y
        gradient = np.array(
            [np.mean(residual * x), np.mean(residual)], dtype=np.float64
        )
        return loss, gradient

    result = minimize(
        objective,
        x0=np.array([1.0, 0.0], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=((0.0, 20.0), (-30.0, 30.0)),
        options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Platt calibration failed: {result.message}")
    return {
        "schema": "d2-platt-calibrator-1",
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "fit_count": len(p),
        "positive_count": int(np.count_nonzero(y)),
        "negative_count": int(len(y) - np.count_nonzero(y)),
        "fit_loss": float(result.fun),
    }


def apply_platt_calibrator(probability, calibrator: Mapping) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("probability outside [0,1] for Platt calibration")
    p = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(p) - np.log1p(-p)
    calibrated = expit(float(calibrator["slope"]) * logit + float(calibrator["intercept"]))
    return calibrated.astype(np.float32)


def deterministic_fit_indices(
    episode_index,
    train_episode_mask,
    any_target_200,
    corridor_ttc,
    background_stride: int = 10,
) -> np.ndarray:
    episode_index = np.asarray(episode_index, dtype=np.int64)
    train_episode_mask = np.asarray(train_episode_mask, dtype=bool)
    any_target = np.asarray(any_target_200, dtype=bool)
    ttc = np.asarray(corridor_ttc, dtype=np.float32)
    if episode_index.ndim != 1 or any_target.shape != episode_index.shape or ttc.shape != episode_index.shape:
        raise ValueError("D2 sampler frame arrays have inconsistent shapes")
    if len(episode_index) and (episode_index.min() < 0 or episode_index.max() >= len(train_episode_mask)):
        raise ValueError("D2 sampler episode index out of range")
    if background_stride <= 0:
        raise ValueError("background stride must be positive")
    train_frame = train_episode_mask[episode_index]
    frame_ordinal = np.arange(len(episode_index), dtype=np.int64)
    retained = train_frame & (
        (frame_ordinal % int(background_stride) == 0)
        | any_target
        | (ttc < 2.0)
    )
    indices = np.flatnonzero(retained).astype(np.int64)
    if len(indices) == 0:
        raise ValueError("D2 sampler selected no training frames")
    return indices


def inverse_sampling_weights(
    indices,
    any_target_200,
    corridor_ttc,
    background_stride: int,
) -> np.ndarray:
    """Horvitz-Thompson weights for deterministic background thinning."""
    indices = np.asarray(indices, dtype=np.int64)
    any_target = np.asarray(any_target_200, dtype=bool)
    ttc = np.asarray(corridor_ttc, dtype=np.float32)
    if any_target.shape != ttc.shape or any_target.ndim != 1:
        raise ValueError("sampling-weight arrays have inconsistent shapes")
    forced = any_target[indices] | (ttc[indices] < 2.0)
    return np.where(forced, 1.0, float(background_stride)).astype(np.float32)


def compute_normalization(features, indices, chunk_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("normalization indices must be nonempty")
    if chunk_size <= 0:
        raise ValueError("normalization chunk size must be positive")
    feature_dim = int(features.shape[1])
    total = np.zeros(feature_dim, dtype=np.float64)
    total_sq = np.zeros(feature_dim, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), chunk_size):
        batch = np.asarray(features[indices[start:start + chunk_size]], dtype=np.float64)
        if not np.all(np.isfinite(batch)):
            raise ValueError("non-finite frozen feature")
        total += np.sum(batch, axis=0)
        total_sq += np.sum(batch * batch, axis=0)
        count += len(batch)
    mean = total / count
    variance = np.maximum(0.0, total_sq / count - mean * mean)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _class_counts(arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> dict:
    counts = {}
    for target_name, valid_name in zip(CLASSIFICATION_ARRAYS, VALID_ARRAYS):
        target = np.asarray(arrays[target_name][indices], dtype=bool)
        valid = np.asarray(arrays[valid_name][indices], dtype=bool)
        positive = int(np.count_nonzero(target & valid))
        negative = int(np.count_nonzero((~target) & valid))
        if positive == 0 or negative == 0:
            raise ValueError(f"probe fit lacks both classes for {target_name}")
        counts[target_name] = {"positive": positive, "negative": negative}
    return counts


def train_probe(
    features,
    arrays: Mapping[str, np.ndarray],
    train_episode_mask: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    seed: int,
) -> tuple[ProbeNet, np.ndarray, np.ndarray, dict]:
    indices = deterministic_fit_indices(
        arrays["episode_index"],
        train_episode_mask,
        arrays["any_target_200"],
        arrays["corridor_ttc"],
        background_stride=config.background_stride,
    )
    mean, std = compute_normalization(features, indices)
    sample_weight = inverse_sampling_weights(
        indices,
        arrays["any_target_200"],
        arrays["corridor_ttc"],
        config.background_stride,
    )
    class_counts = _class_counts(arrays, indices)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = ProbeNet(config.family, input_dim=features.shape[1]).to(device)
    full_train_frame = np.asarray(train_episode_mask, dtype=bool)[
        np.asarray(arrays["episode_index"], dtype=np.int64)
    ]
    initial_prevalence = []
    for target_name, valid_name in zip(CLASSIFICATION_ARRAYS, VALID_ARRAYS):
        valid_full = full_train_frame & np.asarray(arrays[valid_name], dtype=bool)
        prevalence = float(np.mean(np.asarray(arrays[target_name], dtype=bool)[valid_full]))
        initial_prevalence.append(float(np.clip(prevalence, 1e-6, 1.0 - 1e-6)))
    initial_closing = float(np.mean(np.asarray(arrays["closing_rate"], dtype=np.float32)[full_train_frame]))
    initial_ttc = float(np.mean(np.asarray(arrays["corridor_ttc"], dtype=np.float32)[full_train_frame]))
    output_layer = model.net if config.family == "linear" else model.net[2]
    with torch.no_grad():
        bias = output_layer.bias
        prevalence_tensor = torch.as_tensor(initial_prevalence, dtype=bias.dtype, device=device)
        bias[:6].copy_(torch.log(prevalence_tensor) - torch.log1p(-prevalence_tensor))
        bias[6] = initial_closing / config.closing_scale
        ttc_fraction = torch.tensor(
            float(np.clip(initial_ttc / config.ttc_scale, 1e-6, 1.0 - 1e-6)),
            dtype=bias.dtype,
            device=device,
        )
        bias[7] = torch.log(ttc_fraction) - torch.log1p(-ttc_fraction)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    mean_t = torch.as_tensor(mean, device=device).view(1, -1)
    std_t = torch.as_tensor(std, device=device).view(1, -1)
    rng = np.random.default_rng(int(seed))
    history = []
    model.train()
    for epoch in range(config.epochs):
        shuffled = indices[rng.permutation(len(indices))]
        sums = np.zeros(4, dtype=np.float64)
        batches = 0
        for start in range(0, len(shuffled), config.batch_size):
            batch_indices = shuffled[start:start + config.batch_size]
            batch_positions = np.searchsorted(indices, batch_indices)
            # `indices` is sorted and every shuffled element came from it.
            if not np.array_equal(indices[batch_positions], batch_indices):
                raise AssertionError("D2 sampled-index lookup failed")
            sampling_weight = torch.as_tensor(sample_weight[batch_positions], device=device)
            x = torch.as_tensor(
                np.asarray(features[batch_indices], dtype=np.float32), device=device
            )
            x = (x - mean_t) / std_t
            raw = model(x)
            target = torch.as_tensor(
                np.column_stack([arrays[name][batch_indices] for name in CLASSIFICATION_ARRAYS]).astype(np.float32),
                device=device,
            )
            valid = torch.as_tensor(
                np.column_stack([arrays[name][batch_indices] for name in VALID_ARRAYS]).astype(np.float32),
                device=device,
            )
            cls_element = F.binary_cross_entropy_with_logits(raw[:, :6], target, reduction="none")
            weighted_valid = valid * sampling_weight[:, None]
            cls_loss = torch.sum(cls_element * weighted_valid) / torch.clamp(torch.sum(weighted_valid), min=1.0)
            closing_target = torch.as_tensor(
                np.asarray(arrays["closing_rate"][batch_indices], dtype=np.float32), device=device
            ) / config.closing_scale
            closing_element = F.smooth_l1_loss(raw[:, 6], closing_target, reduction="none")
            closing_loss = torch.sum(closing_element * sampling_weight) / torch.sum(sampling_weight)
            ttc_target = torch.as_tensor(
                np.asarray(arrays["corridor_ttc"][batch_indices], dtype=np.float32), device=device
            ) / config.ttc_scale
            ttc_weight = torch.where(
                ttc_target * config.ttc_scale < 2.0,
                torch.tensor(config.ttc_critical_weight, device=device),
                torch.tensor(1.0, device=device),
            )
            ttc_element = F.smooth_l1_loss(torch.sigmoid(raw[:, 7]), ttc_target, reduction="none")
            combined_ttc_weight = ttc_weight * sampling_weight
            ttc_loss = torch.sum(ttc_element * combined_ttc_weight) / torch.sum(combined_ttc_weight)
            loss = cls_loss + config.closing_loss_weight * closing_loss + config.ttc_loss_weight * ttc_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            sums += [float(loss.item()), float(cls_loss.item()), float(closing_loss.item()), float(ttc_loss.item())]
            batches += 1
        history.append(
            {
                "epoch": epoch,
                "loss": float(sums[0] / batches),
                "classification_loss": float(sums[1] / batches),
                "closing_loss": float(sums[2] / batches),
                "ttc_loss": float(sums[3] / batches),
            }
        )
    model.eval()
    report = {
        "config": asdict(config),
        "seed": int(seed),
        "sampled_frame_count": len(indices),
        "sampling_weight_sum": float(np.sum(sample_weight)),
        "class_counts": class_counts,
        "initial_prevalence": initial_prevalence,
        "initial_closing_rate": initial_closing,
        "initial_corridor_ttc": initial_ttc,
        "history": history,
    }
    return model, mean, std, report


@torch.no_grad()
def predict_probe(
    model: ProbeNet,
    features,
    frame_indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    output = np.empty((len(frame_indices), 8), dtype=np.float32)
    mean_t = torch.as_tensor(mean, device=device).view(1, -1)
    std_t = torch.as_tensor(std, device=device).view(1, -1)
    model.eval()
    for start in range(0, len(frame_indices), batch_size):
        selected = frame_indices[start:start + batch_size]
        x = torch.as_tensor(np.asarray(features[selected], dtype=np.float32), device=device)
        raw = model((x - mean_t) / std_t)
        output[start:start + len(selected)] = decode_predictions(raw).cpu().numpy().astype(np.float32)
    return output

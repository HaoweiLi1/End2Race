"""Locked configuration for the D2R-G representation redesign."""

from __future__ import annotations

from dataclasses import dataclass


FAMILY = "spatiotemporal_geometry"
HISTORY_OFFSETS = (0, 5, 10, 20, 35, 50, 75, 100)
LIDAR_BEAMS = 360
LIDAR_MAX_M = 30.0
TTC_BIN_WIDTH_S = 0.1
TTC_BIN_COUNT = 50
BACKGROUND_STRIDE = 20
SEED = 20260711
REGISTRY_OPENED_AT = "2026-07-11T21:00:00+08:00"
EVIDENCE_RELPATH = "Experiments/A5_d2r_geometry"


@dataclass(frozen=True)
class TrainConfig:
    family: str = FAMILY
    epochs: int = 6
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    background_stride: int = BACKGROUND_STRIDE
    ttc_critical_weight: float = 25.0
    collision_loss_weight: float = 1.0
    ttc_loss_weight: float = 1.0
    rel_loss_weight: float = 0.25
    lateral_loss_weight: float = 0.25
    closing_loss_weight: float = 0.50

    def __post_init__(self):
        if self.family != FAMILY:
            raise ValueError("D2R-G has exactly one locked family")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("D2R-G epochs and batch size must be positive")
        if self.background_stride != BACKGROUND_STRIDE:
            raise ValueError("D2R-G background stride is locked")


LOCKED_CONFIG = TrainConfig()

"""Causal deployable data view and labels for D2R-G."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from d2.probe import ProbeDataset
from d2.release import file_sha256
from d0.identity import registry_row_id, validate_registry_row
from d25.search import DATASET_MANIFEST_SHA256, EPISODE_METADATA_SHA256
from d2r import (
    BACKGROUND_STRIDE,
    HISTORY_OFFSETS,
    LIDAR_BEAMS,
    TTC_BIN_COUNT,
    TTC_BIN_WIDTH_S,
    EVIDENCE_RELPATH,
    REGISTRY_OPENED_AT,
)


SIGNALS_MANIFEST_SHA256 = "d653d77dcf270ce9b9e714d23a9b5600b15dfb90996e627610153be1763b513a"


def make_registry_rows(
    episodes: Iterable[Mapping],
    opened_at_utc: str = REGISTRY_OPENED_AT,
    evidence_relpath: str = EVIDENCE_RELPATH,
) -> list[dict[str, str]]:
    if opened_at_utc != REGISTRY_OPENED_AT:
        raise ValueError("D2R-G registry opening time is locked")
    if evidence_relpath != EVIDENCE_RELPATH:
        raise ValueError("D2R-G evidence root is locked")
    rows = []
    for episode in episodes:
        row = {
            "registry_schema": "bplus-opened-registry-1",
            "opened_at_utc": opened_at_utc,
            "stage": "D2R-G",
            "use_class": "probe_fit",
            "split_id": f"d2r_g_non_test_{DATASET_MANIFEST_SHA256[:16]}",
            "l2_id": str(episode["l2_id"]),
            "l3_id": str(episode["l3_id"]),
            "l4_id": str(episode["l4_id"]),
            "map_name": str(episode["map_name"]),
            "source_manifest_sha256": DATASET_MANIFEST_SHA256,
            "source_run_id": "non_test_full_20260711_175713",
            "decision_effect": "representation_choice",
            "final_pool": "false",
            "evidence_relpath": evidence_relpath,
        }
        row["row_id"] = registry_row_id(row)
        rows.append(validate_registry_row(row))
    rows.sort(key=lambda row: row["row_id"])
    if len(rows) != 1928 or len({row["l2_id"] for row in rows}) != 1928:
        raise ValueError("D2R-G registry population must be 1,928 distinct L2 rows")
    return rows


def causal_history_indices(
    frame_indices,
    episode_index,
    episode_starts,
    offsets=HISTORY_OFFSETS,
) -> np.ndarray:
    indices = np.asarray(frame_indices, dtype=np.int64)
    frame_episode = np.asarray(episode_index, dtype=np.int64)
    starts = np.asarray(episode_starts, dtype=np.int64)
    offsets = tuple(int(value) for value in offsets)
    if indices.ndim != 1:
        raise ValueError("D2R frame indices must be one-dimensional")
    if frame_episode.ndim != 1 or starts.ndim != 1:
        raise ValueError("D2R episode accounting must be vectors")
    if offsets != HISTORY_OFFSETS:
        raise ValueError("D2R history offsets are locked")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(frame_episode)):
        raise ValueError("D2R frame index out of range")
    episodes = frame_episode[indices]
    if len(episodes) and (episodes.min() < 0 or episodes.max() >= len(starts)):
        raise ValueError("D2R episode index out of range")
    history = np.column_stack(
        [np.maximum(starts[episodes], indices - offset) for offset in offsets]
    ).astype(np.int64, copy=False)
    if np.any(history > indices[:, None]):
        raise AssertionError("D2R history contains a future frame")
    if len(indices) and np.any(frame_episode[history] != episodes[:, None]):
        raise AssertionError("D2R history crosses an episode boundary")
    return history


def ttc_bin_indices(ttc) -> np.ndarray:
    values = np.asarray(ttc, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 5.0 + 1e-6):
        raise ValueError("D2R TTC target lies outside [0,5]")
    bins = np.floor(values / TTC_BIN_WIDTH_S).astype(np.int64)
    return np.clip(bins, 0, TTC_BIN_COUNT - 1)


def ttc_bin_centers() -> np.ndarray:
    return ((np.arange(TTC_BIN_COUNT, dtype=np.float32) + 0.5) * TTC_BIN_WIDTH_S).astype(
        np.float32
    )


def deterministic_fit_indices(
    episode_index,
    train_episode_mask,
    any_target_200,
    corridor_ttc,
    background_stride: int = BACKGROUND_STRIDE,
) -> np.ndarray:
    frame_episode = np.asarray(episode_index, dtype=np.int64)
    train_mask = np.asarray(train_episode_mask, dtype=bool)
    target = np.asarray(any_target_200, dtype=bool)
    ttc = np.asarray(corridor_ttc, dtype=np.float32)
    if frame_episode.ndim != 1 or target.shape != frame_episode.shape or ttc.shape != frame_episode.shape:
        raise ValueError("D2R sampling arrays have inconsistent shapes")
    if len(frame_episode) and (frame_episode.min() < 0 or frame_episode.max() >= len(train_mask)):
        raise ValueError("D2R sampling episode index out of range")
    if int(background_stride) != BACKGROUND_STRIDE:
        raise ValueError("D2R background stride is locked to 20")
    ordinal = np.arange(len(frame_episode), dtype=np.int64)
    train_frame = train_mask[frame_episode]
    retained = train_frame & (
        target | (ttc < 2.0) | (ordinal % BACKGROUND_STRIDE == 0)
    )
    output = np.flatnonzero(retained).astype(np.int64)
    if len(output) == 0:
        raise ValueError("D2R sampler selected no frame")
    return output


def inverse_sampling_weights(indices, any_target_200, corridor_ttc) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    target = np.asarray(any_target_200, dtype=bool)
    ttc = np.asarray(corridor_ttc, dtype=np.float32)
    if target.shape != ttc.shape or target.ndim != 1:
        raise ValueError("D2R sampling-weight arrays have inconsistent shapes")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(target)):
        raise ValueError("D2R sampling-weight index out of range")
    forced = target[indices] | (ttc[indices] < 2.0)
    return np.where(forced, 1.0, float(BACKGROUND_STRIDE)).astype(np.float32)


class D2RDataset:
    """Memmapped D2 labels plus deployable causal signal history."""

    def __init__(self, dataset_dir: str | Path, split_dir: str | Path, signals_dir: str | Path):
        dataset_dir = Path(dataset_dir)
        signals_dir = Path(signals_dir)
        if file_sha256(dataset_dir / "dataset_manifest.json") != DATASET_MANIFEST_SHA256:
            raise ValueError("D2R dataset manifest SHA drift")
        if file_sha256(dataset_dir / "episode_metadata.tsv") != EPISODE_METADATA_SHA256:
            raise ValueError("D2R episode metadata SHA drift")
        self.base = ProbeDataset(dataset_dir, split_dir)
        if not (signals_dir / "COMPLETE").is_file():
            raise ValueError("D2R deployable signal release lacks COMPLETE")
        if file_sha256(signals_dir / "signals_manifest.json") != SIGNALS_MANIFEST_SHA256:
            raise ValueError("D2R deployable signals manifest SHA drift")
        manifest = json.loads((signals_dir / "signals_manifest.json").read_text(encoding="utf-8"))
        self.signals = {
            name: np.load(signals_dir / entry["relpath"], mmap_mode="r", allow_pickle=False)
            for name, entry in manifest["signals"].items()
        }
        if set(self.signals) != {
            "ego_lidar",
            "ego_actual_speed",
            "previous_desired_steer",
            "previous_desired_speed",
        }:
            raise ValueError("D2R deployable signal set mismatch")
        if self.signals["ego_lidar"].shape != (self.base.frame_count, LIDAR_BEAMS):
            raise ValueError("D2R LiDAR shape mismatch")
        if any(len(value) != self.base.frame_count for value in self.signals.values()):
            raise ValueError("D2R deployable signal length mismatch")
        self.episode_starts = np.asarray(
            [int(row["frame_start"]) for row in self.base.episodes], dtype=np.int64
        )

    @property
    def frame_count(self) -> int:
        return self.base.frame_count

    @property
    def episode_count(self) -> int:
        return self.base.episode_count

    def history(self, frame_indices) -> np.ndarray:
        return causal_history_indices(
            frame_indices,
            self.base.episode_index,
            self.episode_starts,
        )

    def input_batch(self, frame_indices) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.asarray(frame_indices, dtype=np.int64)
        history = self.history(indices)
        lidar = np.asarray(self.signals["ego_lidar"][history], dtype=np.float32)
        if lidar.shape != (len(indices), len(HISTORY_OFFSETS), LIDAR_BEAMS):
            raise AssertionError("D2R history LiDAR shape drift")
        lidar = np.clip(lidar, 0.0, 30.0) / np.float32(30.0)
        scalar = np.concatenate(
            [
                np.asarray(self.signals["ego_actual_speed"][history], dtype=np.float32) / 10.0,
                np.asarray(self.signals["previous_desired_steer"][history], dtype=np.float32) / 0.52,
                np.asarray(self.signals["previous_desired_speed"][history], dtype=np.float32) / 10.0,
            ],
            axis=1,
        )
        bc = np.asarray(self.base.features[indices], dtype=np.float32)
        if scalar.shape != (len(indices), 24) or bc.shape != (len(indices), 1680):
            raise AssertionError("D2R deployable input shape drift")
        if not np.all(np.isfinite(lidar)) or not np.all(np.isfinite(scalar)) or not np.all(np.isfinite(bc)):
            raise ValueError("D2R deployable input contains nonfinite values")
        return lidar, bc, scalar

    def target_batch(self, frame_indices) -> dict[str, np.ndarray]:
        indices = np.asarray(frame_indices, dtype=np.int64)
        arrays = self.base.arrays
        return {
            "classification": np.column_stack(
                [arrays[name][indices] for name in (
                    "ego_target_050", "ego_target_100", "ego_target_200",
                    "any_target_050", "any_target_100", "any_target_200",
                )]
            ).astype(np.float32),
            "valid": np.column_stack(
                [arrays[name][indices] for name in (
                    "ego_valid_050", "ego_valid_100", "ego_valid_200",
                    "any_valid_050", "any_valid_100", "any_valid_200",
                )]
            ).astype(np.float32),
            "rel_s": np.clip(np.asarray(arrays["rel_s"][indices], dtype=np.float32), -10.0, 10.0),
            "lateral_gap": np.clip(
                np.asarray(arrays["lateral_gap"][indices], dtype=np.float32), 0.0, 2.0
            ),
            "closing_rate": np.clip(
                np.asarray(arrays["closing_rate"][indices], dtype=np.float32), -5.0, 5.0
            ),
            "ttc": np.asarray(arrays["corridor_ttc"][indices], dtype=np.float32),
            "ttc_bin": ttc_bin_indices(arrays["corridor_ttc"][indices]),
        }

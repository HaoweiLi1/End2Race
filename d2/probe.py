"""Five-outer/three-inner grouped D2 probe evaluation."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from d2.dataset import _prepare_output, _promote, _write_json, _write_output_manifest
from d2.metrics import binary_metrics, evaluate_alarm_threshold, select_alarm_threshold, ttc_mae
from d2.models import (
    CLASSIFICATION_ARRAYS,
    DeployableTemporalFeatureView,
    LOCKED_CONFIGS,
    PREDICTION_NAMES,
    TemporalFeatureView,
    VALID_ARRAYS,
    apply_platt_calibrator,
    fit_platt_calibrator,
    predict_probe,
    train_probe,
)
from d2.release import file_sha256
from d2.split import validate_split


def _read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class ProbeDataset:
    def __init__(self, dataset_dir: str | Path, split_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.split_dir = Path(split_dir)
        if not (self.dataset_dir / "COMPLETE").is_file():
            raise ValueError("D2 non-test dataset lacks COMPLETE")
        if not (self.split_dir / "COMPLETE").is_file():
            raise ValueError("D2 split lock lacks COMPLETE")
        manifest = json.loads((self.dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
        self.features = np.load(
            self.dataset_dir / manifest["arrays"]["features"]["relpath"],
            mmap_mode="r",
            allow_pickle=False,
        )
        self.arrays = {
            name: np.load(self.dataset_dir / entry["relpath"], mmap_mode="r", allow_pickle=False)
            for name, entry in manifest["arrays"].items()
            if name != "features"
        }
        self.episodes = _read_tsv(self.dataset_dir / "episode_metadata.tsv")
        if len(self.episodes) != int(manifest["episode_count"]):
            raise ValueError("D2 episode metadata count mismatch")
        split_rows = _read_tsv(self.split_dir / "scenario_split.tsv")
        validate_split(split_rows, expected_l2=3036)
        split_by_l2 = {row["l2_id"]: row for row in split_rows}
        self.split_rows = []
        for episode in self.episodes:
            row = split_by_l2.get(episode["l2_id"])
            if row is None or row["split"] != "non_test":
                raise ValueError("dataset episode missing from non-test split")
            if row["outer_fold"] != episode["outer_fold"]:
                raise ValueError("dataset/split outer fold mismatch")
            self.split_rows.append(row)
        self.episode_count = len(self.episodes)
        self.frame_count = len(self.features)
        self.episode_index = np.asarray(self.arrays["episode_index"], dtype=np.int64)
        if self.episode_index.shape != (self.frame_count,):
            raise ValueError("D2 episode-index array shape mismatch")
        self.outer_fold = np.asarray([int(row["outer_fold"]) for row in self.split_rows], dtype=np.int64)
        self.inner_fold = {
            outer: np.asarray(
                [int(row[f"inner_fold_outer{outer}"]) if row[f"inner_fold_outer{outer}"] else -1 for row in self.split_rows],
                dtype=np.int64,
            )
            for outer in range(5)
        }
        self.ego_collision = np.asarray([row["ego_collision"] == "True" for row in self.episodes])
        self.any_collision = np.asarray([row["collision_any"] == "True" for row in self.episodes])
        self.final_time = np.asarray([float.fromhex(row["final_time_hex"]) for row in self.episodes])

    def frame_indices(self, episode_mask: np.ndarray) -> np.ndarray:
        episode_mask = np.asarray(episode_mask, dtype=bool)
        if episode_mask.shape != (self.episode_count,):
            raise ValueError("episode mask shape mismatch")
        return np.flatnonzero(episode_mask[self.episode_index]).astype(np.int64)

    def subset_alarm_args(self, frame_indices: np.ndarray, episode_mask: np.ndarray, target_name: str):
        episode_ids = np.flatnonzero(episode_mask)
        remap = np.full(self.episode_count, -1, dtype=np.int64)
        remap[episode_ids] = np.arange(len(episode_ids))
        local_episode = remap[self.episode_index[frame_indices]]
        if np.any(local_episode < 0):
            raise ValueError("alarm frames escape episode subset")
        prefix = target_name.split("_", 1)[0]
        event_flags = self.ego_collision if prefix == "ego" else self.any_collision
        return {
            "valid": np.asarray(self.arrays[target_name.replace("target", "valid")][frame_indices], dtype=bool),
            "positive_window": np.asarray(self.arrays[target_name][frame_indices], dtype=bool),
            "episode_index": local_episode,
            "episode_any_collision": self.any_collision[episode_ids],
            "episode_ego_collision": event_flags[episode_ids],
            "time": np.asarray(self.arrays["time"][frame_indices], dtype=np.float64),
            "final_time": self.final_time[episode_ids],
        }


def _save_bundle(path: Path, model, mean, std, config, seed, train_report) -> str:
    torch.save(
        {
            "schema": "d2-probe-bundle-1",
            "family": config.family,
            "input_dim": model.input_dim,
            "state_dict": model.state_dict(),
            "normalization_mean": mean,
            "normalization_std": std,
            "config": asdict(config),
            "seed": int(seed),
            "train_report": train_report,
        },
        path,
    )
    return file_sha256(path)


def _training_prevalence(dataset: ProbeDataset, episode_mask: np.ndarray, target_name: str) -> float:
    indices = dataset.frame_indices(episode_mask)
    valid = np.asarray(dataset.arrays[target_name.replace("target", "valid")][indices], dtype=bool)
    target = np.asarray(dataset.arrays[target_name][indices], dtype=bool)
    if not np.any(valid):
        raise ValueError("training fold has no valid classification frames")
    return float(np.mean(target[valid]))


def _select_thresholds(
    dataset: ProbeDataset,
    predictions: np.ndarray,
    frame_indices: np.ndarray,
    episode_mask: np.ndarray,
) -> dict:
    thresholds = {}
    for head_index, target_name in enumerate(CLASSIFICATION_ARRAYS):
        args = dataset.subset_alarm_args(frame_indices, episode_mask, target_name)
        selected = select_alarm_threshold(
            predictions[:, head_index],
            false_alarm_limit=0.10,
            **args,
        )
        thresholds[PREDICTION_NAMES[head_index]] = selected
    return thresholds


def _fit_calibrators(
    dataset: ProbeDataset,
    predictions: np.ndarray,
    frame_indices: np.ndarray,
) -> dict:
    calibrators = {}
    for head_index, target_name in enumerate(CLASSIFICATION_ARRAYS):
        calibrators[PREDICTION_NAMES[head_index]] = fit_platt_calibrator(
            predictions[:, head_index],
            np.asarray(dataset.arrays[target_name][frame_indices], dtype=bool),
            np.asarray(
                dataset.arrays[target_name.replace("target", "valid")][frame_indices],
                dtype=bool,
            ),
        )
    return calibrators


def _apply_calibrators(predictions: np.ndarray, calibrators: Mapping[str, Mapping]) -> np.ndarray:
    output = np.asarray(predictions, dtype=np.float32).copy()
    for head_index in range(6):
        name = PREDICTION_NAMES[head_index]
        output[:, head_index] = apply_platt_calibrator(output[:, head_index], calibrators[name])
    return output


def _evaluate_outer(
    dataset: ProbeDataset,
    predictions: np.ndarray,
    frame_indices: np.ndarray,
    outer_mask: np.ndarray,
    train_mask: np.ndarray,
    thresholds: Mapping[str, Mapping],
) -> dict:
    classification = {}
    alarms = {}
    for head_index, target_name in enumerate(CLASSIFICATION_ARRAYS):
        target = np.asarray(dataset.arrays[target_name][frame_indices], dtype=bool)
        valid = np.asarray(dataset.arrays[target_name.replace("target", "valid")][frame_indices], dtype=bool)
        prevalence = _training_prevalence(dataset, train_mask, target_name)
        classification[PREDICTION_NAMES[head_index]] = binary_metrics(
            target[valid], predictions[valid, head_index], prevalence_reference=prevalence
        )
        args = dataset.subset_alarm_args(frame_indices, outer_mask, target_name)
        alarms[PREDICTION_NAMES[head_index]] = evaluate_alarm_threshold(
            predictions[:, head_index],
            threshold=float(thresholds[PREDICTION_NAMES[head_index]]["threshold"]),
            **args,
        )
    ttc = ttc_mae(
        np.asarray(dataset.arrays["corridor_ttc"][frame_indices], dtype=np.float64),
        predictions[:, 7].astype(np.float64),
    )
    closing_target = np.asarray(dataset.arrays["closing_rate"][frame_indices], dtype=np.float64)
    closing_mae = float(np.mean(np.abs(predictions[:, 6] - closing_target)))
    return {
        "frame_count": len(frame_indices),
        "episode_count": int(np.count_nonzero(outer_mask)),
        "ego_collision_episode_count": int(np.count_nonzero(dataset.ego_collision & outer_mask)),
        "any_collision_episode_count": int(np.count_nonzero(dataset.any_collision & outer_mask)),
        "classification": classification,
        "alarms": alarms,
        "ttc_lt2": ttc,
        "closing_rate_mae": closing_mae,
    }


def _aggregate_oof(dataset: ProbeDataset, predictions: np.ndarray, fold_reports: list[dict]) -> dict:
    classification = {}
    for head_index, target_name in enumerate(CLASSIFICATION_ARRAYS):
        valid = np.asarray(dataset.arrays[target_name.replace("target", "valid")], dtype=bool)
        target = np.asarray(dataset.arrays[target_name], dtype=bool)
        global_prevalence = float(np.mean(target[valid]))
        metric = binary_metrics(target[valid], predictions[valid, head_index], global_prevalence)
        model_sse = float(np.sum((predictions[valid, head_index].astype(np.float64) - target[valid]) ** 2))
        reference_sse = 0.0
        valid_count = 0
        for outer, fold in enumerate(fold_reports):
            outer_mask = dataset.outer_fold == outer
            indices = dataset.frame_indices(outer_mask)
            fold_valid = np.asarray(dataset.arrays[target_name.replace("target", "valid")][indices], dtype=bool)
            fold_target = np.asarray(dataset.arrays[target_name][indices], dtype=np.float64)
            prevalence = fold["training_prevalence"][PREDICTION_NAMES[head_index]]
            reference_sse += float(np.sum((prevalence - fold_target[fold_valid]) ** 2))
            valid_count += int(np.count_nonzero(fold_valid))
        nested_reference_brier = reference_sse / valid_count
        nested_brier = model_sse / valid_count
        metric["nested_reference_brier"] = nested_reference_brier
        metric["nested_brier_skill"] = 1.0 - nested_brier / nested_reference_brier
        classification[PREDICTION_NAMES[head_index]] = metric

    alarms = {}
    for head_index in range(6):
        name = PREDICTION_NAMES[head_index]
        safe_count = sum(fold["evaluation"]["alarms"][name]["safe_episode_count"] for fold in fold_reports)
        safe_alarm = sum(fold["evaluation"]["alarms"][name]["safe_episode_alarm_count"] for fold in fold_reports)
        event_count = sum(fold["evaluation"]["alarms"][name]["event_episode_count"] for fold in fold_reports)
        event_alarm = sum(fold["evaluation"]["alarms"][name]["event_episode_alarm_count"] for fold in fold_reports)
        alarms[name] = {
            "safe_episode_count": safe_count,
            "safe_episode_alarm_count": safe_alarm,
            "safe_episode_false_alarm_rate": safe_alarm / safe_count,
            "event_episode_count": event_count,
            "event_episode_alarm_count": event_alarm,
            "event_recall": event_alarm / event_count,
            "threshold_mode": "outer-specific-inner-OOF",
        }
    ttc = ttc_mae(
        np.asarray(dataset.arrays["corridor_ttc"], dtype=np.float64),
        predictions[:, 7].astype(np.float64),
    )
    closing_mae = float(
        np.mean(
            np.abs(
                predictions[:, 6].astype(np.float64)
                - np.asarray(dataset.arrays["closing_rate"], dtype=np.float64)
            )
        )
    )
    gate = {
        "ego_1s_recall": alarms["ego_probability_100"]["event_recall"],
        "ego_1s_safe_fa": alarms["ego_probability_100"]["safe_episode_false_alarm_rate"],
        "ego_2s_recall": alarms["ego_probability_200"]["event_recall"],
        "ego_2s_safe_fa": alarms["ego_probability_200"]["safe_episode_false_alarm_rate"],
        "ego_1s_brier_skill": classification["ego_probability_100"]["nested_brier_skill"],
        "ttc_lt2_mae": ttc["mae"],
        "heldout_ego_collision_episodes": int(np.count_nonzero(dataset.ego_collision)),
    }
    conditions = {
        "ego_1s_recall_ge_0p60": gate["ego_1s_recall"] >= 0.60,
        "ego_1s_safe_fa_le_0p10": gate["ego_1s_safe_fa"] <= 0.10 + 1e-12,
        "ego_2s_recall_ge_0p40": gate["ego_2s_recall"] >= 0.40,
        "ego_2s_safe_fa_le_0p10": gate["ego_2s_safe_fa"] <= 0.10 + 1e-12,
        "ego_1s_brier_skill_ge_0p10": gate["ego_1s_brier_skill"] >= 0.10,
        "ttc_lt2_mae_le_0p30": gate["ttc_lt2_mae"] is not None and gate["ttc_lt2_mae"] <= 0.30,
        "heldout_ego_collision_episodes_ge_30": gate["heldout_ego_collision_episodes"] >= 30,
    }
    return {
        "classification": classification,
        "alarms": alarms,
        "ttc_lt2": ttc,
        "closing_rate_mae": closing_mae,
        "gate_values": gate,
        "gate_conditions": conditions,
        "gate_passed": all(conditions.values()),
    }


def run_family_oof(
    dataset_dir: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    family: str,
    created_at: str,
    device_name: str = "cuda",
    seed: int = 20260711,
    signals_dir: str | Path | None = None,
) -> dict:
    if family not in LOCKED_CONFIGS:
        raise ValueError("family must be linear, mlp, or temporal")
    dataset = ProbeDataset(dataset_dir, split_dir)
    config = LOCKED_CONFIGS[family]
    signal_manifest_sha256 = None
    if family == "temporal":
        episode_starts = np.asarray(
            [int(row["frame_start"]) for row in dataset.episodes], dtype=np.int64
        )
        probe_features = TemporalFeatureView(
            dataset.features,
            dataset.episode_index,
            episode_starts,
            offsets=(0, 10, 25, 50),
        )
    elif family == "temporal_deployable":
        if signals_dir is None:
            raise ValueError("temporal_deployable requires --signals-dir")
        signals_dir = Path(signals_dir)
        if not (signals_dir / "COMPLETE").is_file():
            raise ValueError("deployable temporal signals lack COMPLETE")
        signal_manifest = json.loads(
            (signals_dir / "signals_manifest.json").read_text(encoding="utf-8")
        )
        signal_arrays = {
            name: np.load(signals_dir / entry["relpath"], mmap_mode="r", allow_pickle=False)
            for name, entry in signal_manifest["signals"].items()
        }
        episode_starts = np.asarray(
            [int(row["frame_start"]) for row in dataset.episodes], dtype=np.int64
        )
        probe_features = DeployableTemporalFeatureView(
            dataset.features,
            dataset.episode_index,
            episode_starts,
            signal_arrays["ego_lidar"],
            signal_arrays["ego_actual_speed"],
            signal_arrays["previous_desired_steer"],
            signal_arrays["previous_desired_speed"],
            offsets=(0, 10, 25, 50),
        )
        signal_manifest_sha256 = file_sha256(signals_dir / "signals_manifest.json")
    else:
        probe_features = dataset.features
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    output_dir = Path(output_dir)
    partial = _prepare_output(output_dir)
    (partial / "models").mkdir()
    _write_json(
        partial / "config.json",
        {
            "schema": "d2-oof-config-1",
            "family": family,
            "created_at": str(created_at),
            "seed": int(seed),
            "device": str(device_name),
            "train_config": asdict(config),
            "dataset_manifest_sha256": file_sha256(Path(dataset_dir) / "dataset_manifest.json"),
            "split_manifest_sha256": file_sha256(Path(split_dir) / "scenario_split.tsv"),
            "outer_folds": 5,
            "inner_folds": 3,
            "temporal_offsets_frames": [0, 10, 25, 50] if family == "temporal" else None,
            "deployable_signals_manifest_sha256": signal_manifest_sha256,
        },
    )
    predictions = np.lib.format.open_memmap(
        partial / "oof_predictions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(dataset.frame_count, 8),
    )
    predictions[:] = np.nan
    raw_predictions = np.lib.format.open_memmap(
        partial / "raw_oof_predictions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(dataset.frame_count, 8),
    )
    raw_predictions[:] = np.nan
    fold_reports = []
    for outer in range(5):
        outer_mask = dataset.outer_fold == outer
        outer_train = ~outer_mask
        calibration_indices = []
        calibration_predictions = []
        inner_reports = []
        for inner in range(3):
            calibration_mask = outer_train & (dataset.inner_fold[outer] == inner)
            fit_mask = outer_train & ~calibration_mask
            if np.any(fit_mask & calibration_mask) or np.any(fit_mask & outer_mask):
                raise AssertionError("D2 nested fold leakage")
            fit_l4 = {dataset.split_rows[index]["l4_id"] for index in np.flatnonzero(fit_mask)}
            held_l4 = {dataset.split_rows[index]["l4_id"] for index in np.flatnonzero(calibration_mask | outer_mask)}
            if fit_l4 & held_l4:
                raise AssertionError("D2 L4 block leakage")
            model_seed = int(seed + outer * 100 + inner)
            model, mean, std, train_report = train_probe(
                probe_features, dataset.arrays, fit_mask, config, device, model_seed
            )
            frame_indices = dataset.frame_indices(calibration_mask)
            predicted = predict_probe(model, probe_features, frame_indices, mean, std, device)
            bundle_path = partial / "models" / f"outer{outer}_inner{inner}.pt"
            bundle_sha = _save_bundle(bundle_path, model, mean, std, config, model_seed, train_report)
            calibration_indices.append(frame_indices)
            calibration_predictions.append(predicted)
            inner_reports.append(
                {
                    "inner": inner,
                    "fit_episode_count": int(np.count_nonzero(fit_mask)),
                    "calibration_episode_count": int(np.count_nonzero(calibration_mask)),
                    "calibration_frame_count": len(frame_indices),
                    "bundle_relpath": bundle_path.relative_to(partial).as_posix(),
                    "bundle_sha256": bundle_sha,
                    "train": train_report,
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        calibration_indices = np.concatenate(calibration_indices)
        calibration_predictions = np.concatenate(calibration_predictions, axis=0)
        order = np.argsort(calibration_indices, kind="mergesort")
        calibration_indices = calibration_indices[order]
        calibration_predictions = calibration_predictions[order]
        expected_calibration = dataset.frame_indices(outer_train)
        if not np.array_equal(calibration_indices, expected_calibration):
            raise AssertionError("inner OOF calibration does not cover outer training exactly")
        calibrators = _fit_calibrators(dataset, calibration_predictions, calibration_indices)
        calibrated_inner = _apply_calibrators(calibration_predictions, calibrators)
        thresholds = _select_thresholds(dataset, calibrated_inner, calibration_indices, outer_train)

        outer_seed = int(seed + outer * 100 + 99)
        model, mean, std, train_report = train_probe(
            probe_features, dataset.arrays, outer_train, config, device, outer_seed
        )
        outer_indices = dataset.frame_indices(outer_mask)
        outer_raw_predictions = predict_probe(model, probe_features, outer_indices, mean, std, device)
        outer_predictions = _apply_calibrators(outer_raw_predictions, calibrators)
        raw_predictions[outer_indices] = outer_raw_predictions
        predictions[outer_indices] = outer_predictions
        bundle_path = partial / "models" / f"outer{outer}_refit.pt"
        bundle_sha = _save_bundle(bundle_path, model, mean, std, config, outer_seed, train_report)
        training_prevalence = {
            PREDICTION_NAMES[index]: _training_prevalence(dataset, outer_train, target_name)
            for index, target_name in enumerate(CLASSIFICATION_ARRAYS)
        }
        evaluation = _evaluate_outer(
            dataset,
            outer_predictions,
            outer_indices,
            outer_mask,
            outer_train,
            thresholds,
        )
        fold_report = {
            "outer": outer,
            "train_episode_count": int(np.count_nonzero(outer_train)),
            "heldout_episode_count": int(np.count_nonzero(outer_mask)),
            "heldout_frame_count": len(outer_indices),
            "inner": inner_reports,
            "calibrators": calibrators,
            "thresholds": thresholds,
            "training_prevalence": training_prevalence,
            "refit_bundle_relpath": bundle_path.relative_to(partial).as_posix(),
            "refit_bundle_sha256": bundle_sha,
            "refit_train": train_report,
            "evaluation": evaluation,
        }
        fold_reports.append(fold_report)
        _write_json(partial / f"outer{outer}_report.json", fold_report)
        print(
            f"D2_OOF family={family} outer={outer}/4 "
            f"ego1_recall={evaluation['alarms']['ego_probability_100']['event_recall']:.4f} "
            f"ttc_mae={evaluation['ttc_lt2']['mae']}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    predictions.flush()
    raw_predictions.flush()
    if np.any(~np.isfinite(predictions)) or np.any(~np.isfinite(raw_predictions)):
        raise ValueError("D2 OOF predictions are incomplete or non-finite")
    report = _aggregate_oof(dataset, predictions, fold_reports)
    all_episode_mask = np.ones(dataset.episode_count, dtype=bool)
    all_indices = np.arange(dataset.frame_count, dtype=np.int64)
    final_calibrators = _fit_calibrators(dataset, np.asarray(raw_predictions), all_indices)
    final_calibrated_predictions = _apply_calibrators(np.asarray(raw_predictions), final_calibrators)
    report["final_non_test_thresholds"] = _select_thresholds(
        dataset, final_calibrated_predictions, all_indices, all_episode_mask
    )
    report["final_calibrators"] = final_calibrators
    report.update(
        {
            "schema": "d2-oof-report-1",
            "family": family,
            "episode_count": dataset.episode_count,
            "frame_count": dataset.frame_count,
            "validation_passed": True,
        }
    )
    _write_json(partial / "oof_report.json", report)
    del predictions
    del raw_predictions
    _write_output_manifest(partial)
    independent = validate_probe_release(partial, allow_partial=True)
    _promote(partial, output_dir)
    return independent


def validate_probe_release(release_dir: str | Path, allow_partial: bool = False) -> dict:
    release_dir = Path(release_dir)
    if not allow_partial and not (release_dir / "COMPLETE").is_file():
        raise ValueError("D2 probe release lacks COMPLETE")
    manifest_path = release_dir / "output_manifest.sha256"
    expected = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line:
            digest, relpath = line.split("  ", 1)
            expected[relpath] = digest
    actual = {
        path.relative_to(release_dir).as_posix()
        for path in release_dir.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    }
    if set(expected) != actual:
        raise ValueError("D2 probe output inventory mismatch")
    for relpath, digest in expected.items():
        if file_sha256(release_dir / relpath) != digest:
            raise ValueError(f"D2 probe output hash mismatch: {relpath}")
    report = json.loads((release_dir / "oof_report.json").read_text(encoding="utf-8"))
    predictions = np.load(release_dir / "oof_predictions.npy", mmap_mode="r", allow_pickle=False)
    raw_predictions = np.load(release_dir / "raw_oof_predictions.npy", mmap_mode="r", allow_pickle=False)
    if predictions.shape != (int(report["frame_count"]), 8) or not np.all(np.isfinite(predictions)):
        raise ValueError("D2 probe prediction shape/content mismatch")
    if raw_predictions.shape != predictions.shape or not np.all(np.isfinite(raw_predictions)):
        raise ValueError("D2 raw probe prediction shape/content mismatch")
    if np.any(predictions[:, :6] < 0.0) or np.any(predictions[:, :6] > 1.0):
        raise ValueError("D2 probe probability outside [0,1]")
    if np.any(predictions[:, 7] < 0.0) or np.any(predictions[:, 7] > 5.0):
        raise ValueError("D2 probe TTC prediction outside [0,5]")
    if not report.get("validation_passed"):
        raise ValueError("D2 probe report validation flag false")
    return {
        "passed": True,
        "family": report["family"],
        "gate_passed": report["gate_passed"],
        "gate_values": report["gate_values"],
        "episode_count": report["episode_count"],
        "frame_count": report["frame_count"],
        "output_manifest_sha256": file_sha256(manifest_path),
        "oof_report_sha256": file_sha256(release_dir / "oof_report.json"),
    }

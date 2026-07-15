"""Nested L4-grouped fitting and evaluation for the single D2R-G family."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from d2.dataset import _prepare_output, _promote, _write_json, _write_output_manifest
from d0.identity import append_opened_registry
from d2.metrics import binary_metrics, evaluate_alarm_threshold, select_alarm_threshold, ttc_mae
from d2.models import (
    CLASSIFICATION_ARRAYS,
    PREDICTION_NAMES as D2_PREDICTION_NAMES,
    VALID_ARRAYS,
    apply_platt_calibrator,
    compute_normalization,
    fit_platt_calibrator,
)
from d2.release import file_sha256
from d2r import EVIDENCE_RELPATH, LOCKED_CONFIG, REGISTRY_OPENED_AT, SEED, TrainConfig
from d2r.data import (
    D2RDataset,
    deterministic_fit_indices,
    inverse_sampling_weights,
    make_registry_rows,
)
from d2r.model import D2RGeometryNet, decode_ttc_logits, initialize_classification_bias


PREDICTION_NAMES = (*D2_PREDICTION_NAMES[:6], "rel_s", "lateral_gap", "closing_rate", "corridor_ttc")


def _class_counts(dataset: D2RDataset, indices: np.ndarray) -> dict:
    arrays = dataset.base.arrays
    result = {}
    for target_name, valid_name in zip(CLASSIFICATION_ARRAYS, VALID_ARRAYS):
        target = np.asarray(arrays[target_name][indices], dtype=bool)
        valid = np.asarray(arrays[valid_name][indices], dtype=bool)
        positive = int(np.count_nonzero(target & valid))
        negative = int(np.count_nonzero((~target) & valid))
        if positive == 0 or negative == 0:
            raise ValueError(f"D2R fit lacks both classes for {target_name}")
        result[target_name] = {"positive": positive, "negative": negative}
    return result


def _training_prevalence(dataset: D2RDataset, episode_mask: np.ndarray) -> list[float]:
    frame_mask = np.asarray(episode_mask, dtype=bool)[dataset.base.episode_index]
    values = []
    for target_name, valid_name in zip(CLASSIFICATION_ARRAYS, VALID_ARRAYS):
        valid = frame_mask & np.asarray(dataset.base.arrays[valid_name], dtype=bool)
        if not np.any(valid):
            raise ValueError(f"D2R fit has no valid frame for {target_name}")
        prevalence = float(np.mean(np.asarray(dataset.base.arrays[target_name], dtype=bool)[valid]))
        values.append(float(np.clip(prevalence, 1e-6, 1.0 - 1e-6)))
    return values


def _initialize_ttc_bias(
    model: D2RGeometryNet,
    dataset: D2RDataset,
    indices: np.ndarray,
    sample_weight: np.ndarray,
) -> None:
    bins = dataset.target_batch(indices)["ttc_bin"]
    counts = np.bincount(bins, weights=sample_weight.astype(np.float64), minlength=50)
    probability = (counts + 1.0) / (float(np.sum(counts)) + len(counts))
    with torch.no_grad():
        model.ttc_head.bias.copy_(
            torch.as_tensor(np.log(probability), dtype=model.ttc_head.bias.dtype, device=model.ttc_head.bias.device)
        )


def train_model(
    dataset: D2RDataset,
    train_episode_mask: np.ndarray,
    device: torch.device,
    seed: int,
    config: TrainConfig = LOCKED_CONFIG,
    *,
    max_batches_per_epoch: int | None = None,
) -> tuple[D2RGeometryNet, np.ndarray, np.ndarray, dict]:
    train_episode_mask = np.asarray(train_episode_mask, dtype=bool)
    if train_episode_mask.shape != (dataset.episode_count,):
        raise ValueError("D2R train episode mask shape mismatch")
    indices = deterministic_fit_indices(
        dataset.base.arrays["episode_index"],
        train_episode_mask,
        dataset.base.arrays["any_target_200"],
        dataset.base.arrays["corridor_ttc"],
        background_stride=config.background_stride,
    )
    mean, std = compute_normalization(dataset.base.features, indices)
    sample_weight = inverse_sampling_weights(
        indices,
        dataset.base.arrays["any_target_200"],
        dataset.base.arrays["corridor_ttc"],
    )
    prevalence = _training_prevalence(dataset, train_episode_mask)
    class_counts = _class_counts(dataset, indices)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = D2RGeometryNet().to(device)
    initialize_classification_bias(model, prevalence)
    _initialize_ttc_bias(model, dataset, indices, sample_weight)
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
        sums = np.zeros(6, dtype=np.float64)
        batches = 0
        for start in range(0, len(shuffled), config.batch_size):
            if max_batches_per_epoch is not None and batches >= int(max_batches_per_epoch):
                break
            batch_indices = shuffled[start:start + config.batch_size]
            positions = np.searchsorted(indices, batch_indices)
            if not np.array_equal(indices[positions], batch_indices):
                raise AssertionError("D2R sampled-index lookup failed")
            weight = torch.as_tensor(sample_weight[positions], device=device)
            lidar_np, bc_np, scalar_np = dataset.input_batch(batch_indices)
            targets_np = dataset.target_batch(batch_indices)
            lidar = torch.as_tensor(lidar_np, device=device)
            bc = (torch.as_tensor(bc_np, device=device) - mean_t) / std_t
            scalar = torch.as_tensor(scalar_np, device=device)
            output = model(lidar, bc, scalar)

            target = torch.as_tensor(targets_np["classification"], device=device)
            valid = torch.as_tensor(targets_np["valid"], device=device)
            cls_element = F.binary_cross_entropy_with_logits(
                output["collision_logits"], target, reduction="none"
            )
            weighted_valid = valid * weight[:, None]
            cls_loss = torch.sum(cls_element * weighted_valid) / torch.clamp(
                torch.sum(weighted_valid), min=1.0
            )

            ttc_target = torch.as_tensor(targets_np["ttc"], device=device)
            ttc_bin = torch.as_tensor(targets_np["ttc_bin"], dtype=torch.long, device=device)
            ttc_element = F.cross_entropy(output["ttc_logits"], ttc_bin, reduction="none")
            critical = torch.where(
                ttc_target < 2.0,
                torch.tensor(config.ttc_critical_weight, device=device),
                torch.tensor(1.0, device=device),
            )
            ttc_weight = weight * critical
            ttc_loss = torch.sum(ttc_element * ttc_weight) / torch.sum(ttc_weight)

            def geometry_loss(name: str) -> torch.Tensor:
                target_value = torch.as_tensor(targets_np[name], device=device)
                element = F.smooth_l1_loss(output[name], target_value, reduction="none")
                return torch.sum(element * weight) / torch.sum(weight)

            rel_loss = geometry_loss("rel_s")
            lateral_loss = geometry_loss("lateral_gap")
            closing_loss = geometry_loss("closing_rate")
            loss = (
                config.collision_loss_weight * cls_loss
                + config.ttc_loss_weight * ttc_loss
                + config.rel_loss_weight * rel_loss
                + config.lateral_loss_weight * lateral_loss
                + config.closing_loss_weight * closing_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("D2R nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            sums += [
                float(loss.item()),
                float(cls_loss.item()),
                float(ttc_loss.item()),
                float(rel_loss.item()),
                float(lateral_loss.item()),
                float(closing_loss.item()),
            ]
            batches += 1
        if batches == 0:
            raise ValueError("D2R epoch executed no batch")
        history.append(
            {
                "epoch": epoch,
                "batches": batches,
                "loss": float(sums[0] / batches),
                "classification_loss": float(sums[1] / batches),
                "ttc_loss": float(sums[2] / batches),
                "rel_loss": float(sums[3] / batches),
                "lateral_loss": float(sums[4] / batches),
                "closing_loss": float(sums[5] / batches),
            }
        )
    model.eval()
    return model, mean, std, {
        "schema": "d2r-train-report-1",
        "config": asdict(config),
        "seed": int(seed),
        "sampled_frame_count": len(indices),
        "sampling_weight_sum": float(np.sum(sample_weight)),
        "class_counts": class_counts,
        "initial_prevalence": prevalence,
        "micro_max_batches_per_epoch": max_batches_per_epoch,
        "history": history,
    }


@torch.no_grad()
def predict_model(
    model: D2RGeometryNet,
    dataset: D2RDataset,
    frame_indices,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    indices = np.asarray(frame_indices, dtype=np.int64)
    output = np.empty((len(indices), len(PREDICTION_NAMES)), dtype=np.float32)
    mean_t = torch.as_tensor(mean, device=device).view(1, -1)
    std_t = torch.as_tensor(std, device=device).view(1, -1)
    model.eval()
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        lidar_np, bc_np, scalar_np = dataset.input_batch(batch)
        result = model(
            torch.as_tensor(lidar_np, device=device),
            (torch.as_tensor(bc_np, device=device) - mean_t) / std_t,
            torch.as_tensor(scalar_np, device=device),
        )
        output[start:start + len(batch), :6] = torch.sigmoid(
            result["collision_logits"]
        ).cpu().numpy()
        output[start:start + len(batch), 6] = result["rel_s"].cpu().numpy()
        output[start:start + len(batch), 7] = result["lateral_gap"].cpu().numpy()
        output[start:start + len(batch), 8] = result["closing_rate"].cpu().numpy()
        output[start:start + len(batch), 9] = decode_ttc_logits(result["ttc_logits"]).cpu().numpy()
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("D2R prediction contains nonfinite values")
    return output


def _fit_calibrators(dataset: D2RDataset, predictions: np.ndarray, frame_indices: np.ndarray) -> dict:
    result = {}
    arrays = dataset.base.arrays
    for head, (target_name, valid_name) in enumerate(zip(CLASSIFICATION_ARRAYS, VALID_ARRAYS)):
        result[D2_PREDICTION_NAMES[head]] = fit_platt_calibrator(
            predictions[:, head],
            np.asarray(arrays[target_name][frame_indices], dtype=bool),
            np.asarray(arrays[valid_name][frame_indices], dtype=bool),
        )
    return result


def _apply_calibrators(predictions: np.ndarray, calibrators: Mapping) -> np.ndarray:
    output = np.asarray(predictions, dtype=np.float32).copy()
    for head in range(6):
        name = D2_PREDICTION_NAMES[head]
        output[:, head] = apply_platt_calibrator(output[:, head], calibrators[name])
    return output


def _select_thresholds(
    dataset: D2RDataset,
    predictions: np.ndarray,
    frame_indices: np.ndarray,
    episode_mask: np.ndarray,
) -> dict:
    thresholds = {}
    for head, target_name in enumerate(CLASSIFICATION_ARRAYS):
        args = dataset.base.subset_alarm_args(frame_indices, episode_mask, target_name)
        thresholds[D2_PREDICTION_NAMES[head]] = select_alarm_threshold(
            predictions[:, head], false_alarm_limit=0.10, **args
        )
    return thresholds


def _fold_evaluation(
    dataset: D2RDataset,
    predictions: np.ndarray,
    frame_indices: np.ndarray,
    outer_mask: np.ndarray,
    train_mask: np.ndarray,
    thresholds: Mapping,
) -> dict:
    arrays = dataset.base.arrays
    classification = {}
    alarms = {}
    training_prevalence = _training_prevalence(dataset, train_mask)
    for head, (target_name, valid_name) in enumerate(zip(CLASSIFICATION_ARRAYS, VALID_ARRAYS)):
        target = np.asarray(arrays[target_name][frame_indices], dtype=bool)
        valid = np.asarray(arrays[valid_name][frame_indices], dtype=bool)
        name = D2_PREDICTION_NAMES[head]
        classification[name] = binary_metrics(
            target[valid], predictions[valid, head], training_prevalence[head]
        )
        args = dataset.base.subset_alarm_args(frame_indices, outer_mask, target_name)
        alarms[name] = evaluate_alarm_threshold(
            predictions[:, head], threshold=float(thresholds[name]["threshold"]), **args
        )
    targets = {
        "rel_s": np.clip(np.asarray(arrays["rel_s"][frame_indices], dtype=np.float64), -10.0, 10.0),
        "lateral_gap": np.clip(
            np.asarray(arrays["lateral_gap"][frame_indices], dtype=np.float64), 0.0, 2.0
        ),
        "closing_rate": np.clip(
            np.asarray(arrays["closing_rate"][frame_indices], dtype=np.float64), -5.0, 5.0
        ),
    }
    geometry = {
        name: {"mae": float(np.mean(np.abs(predictions[:, column] - target)))}
        for name, column, target in (
            ("rel_s", 6, targets["rel_s"]),
            ("lateral_gap", 7, targets["lateral_gap"]),
            ("closing_rate", 8, targets["closing_rate"]),
        )
    }
    return {
        "frame_count": len(frame_indices),
        "episode_count": int(np.count_nonzero(outer_mask)),
        "ego_collision_episode_count": int(np.count_nonzero(dataset.base.ego_collision & outer_mask)),
        "any_collision_episode_count": int(np.count_nonzero(dataset.base.any_collision & outer_mask)),
        "classification": classification,
        "alarms": alarms,
        "geometry": geometry,
        "ttc_lt2": ttc_mae(
            np.asarray(arrays["corridor_ttc"][frame_indices], dtype=np.float64),
            predictions[:, 9].astype(np.float64),
        ),
    }


def _save_bundle(
    path: Path,
    model: D2RGeometryNet,
    mean: np.ndarray,
    std: np.ndarray,
    config: TrainConfig,
    seed: int,
    train_report: Mapping,
) -> str:
    torch.save(
        {
            "schema": "d2r-model-bundle-1",
            "family": config.family,
            "state_dict": model.state_dict(),
            "normalization_mean": mean,
            "normalization_std": std,
            "config": asdict(config),
            "seed": int(seed),
            "train_report": dict(train_report),
        },
        path,
    )
    return file_sha256(path)


def _aggregate_oof(dataset: D2RDataset, predictions: np.ndarray, fold_reports: list[dict]) -> dict:
    arrays = dataset.base.arrays
    completed = {int(report["outer_fold"]) for report in fold_reports}
    if completed != set(range(5)):
        raise ValueError("D2R complete OOF aggregation requires all five folds")
    classification = {}
    for head, (target_name, valid_name) in enumerate(zip(CLASSIFICATION_ARRAYS, VALID_ARRAYS)):
        valid = np.asarray(arrays[valid_name], dtype=bool)
        target = np.asarray(arrays[target_name], dtype=bool)
        metric = binary_metrics(target[valid], predictions[valid, head], float(np.mean(target[valid])))
        model_sse = float(np.sum((predictions[valid, head].astype(np.float64) - target[valid]) ** 2))
        reference_sse = 0.0
        valid_count = 0
        for fold in fold_reports:
            outer = int(fold["outer_fold"])
            mask = dataset.base.outer_fold == outer
            indices = dataset.base.frame_indices(mask)
            fold_valid = np.asarray(arrays[valid_name][indices], dtype=bool)
            fold_target = np.asarray(arrays[target_name][indices], dtype=np.float64)
            prevalence = float(fold["training_prevalence"][head])
            reference_sse += float(np.sum((prevalence - fold_target[fold_valid]) ** 2))
            valid_count += int(np.count_nonzero(fold_valid))
        nested_reference = reference_sse / valid_count
        metric["nested_reference_brier"] = nested_reference
        metric["nested_brier_skill"] = 1.0 - (model_sse / valid_count) / nested_reference
        classification[D2_PREDICTION_NAMES[head]] = metric
    alarms = {}
    for head in range(6):
        name = D2_PREDICTION_NAMES[head]
        safe = sum(report["evaluation"]["alarms"][name]["safe_episode_count"] for report in fold_reports)
        safe_alarm = sum(
            report["evaluation"]["alarms"][name]["safe_episode_alarm_count"] for report in fold_reports
        )
        event = sum(report["evaluation"]["alarms"][name]["event_episode_count"] for report in fold_reports)
        event_alarm = sum(
            report["evaluation"]["alarms"][name]["event_episode_alarm_count"] for report in fold_reports
        )
        alarms[name] = {
            "safe_episode_count": safe,
            "safe_episode_alarm_count": safe_alarm,
            "safe_episode_false_alarm_rate": safe_alarm / safe,
            "event_episode_count": event,
            "event_episode_alarm_count": event_alarm,
            "event_recall": event_alarm / event,
            "threshold_mode": "outer-specific-inner-OOF",
        }
    ttc = ttc_mae(
        np.asarray(arrays["corridor_ttc"], dtype=np.float64), predictions[:, 9].astype(np.float64)
    )
    geometry = {}
    for name, column, low, high in (
        ("rel_s", 6, -10.0, 10.0),
        ("lateral_gap", 7, 0.0, 2.0),
        ("closing_rate", 8, -5.0, 5.0),
    ):
        target = np.clip(np.asarray(arrays[name], dtype=np.float64), low, high)
        geometry[name] = {"mae": float(np.mean(np.abs(predictions[:, column] - target)))}
    gate_values = {
        "ego_1s_recall": alarms["ego_probability_100"]["event_recall"],
        "ego_1s_safe_fa": alarms["ego_probability_100"]["safe_episode_false_alarm_rate"],
        "ego_2s_recall": alarms["ego_probability_200"]["event_recall"],
        "ego_2s_safe_fa": alarms["ego_probability_200"]["safe_episode_false_alarm_rate"],
        "ego_1s_brier_skill": classification["ego_probability_100"]["nested_brier_skill"],
        "ttc_lt2_mae": ttc["mae"],
        "heldout_ego_collision_episodes": int(np.count_nonzero(dataset.base.ego_collision)),
    }
    conditions = {
        "ego_1s_recall_ge_0p60": gate_values["ego_1s_recall"] >= 0.60,
        "ego_1s_safe_fa_le_0p10": gate_values["ego_1s_safe_fa"] <= 0.10 + 1e-12,
        "ego_2s_recall_ge_0p40": gate_values["ego_2s_recall"] >= 0.40,
        "ego_2s_safe_fa_le_0p10": gate_values["ego_2s_safe_fa"] <= 0.10 + 1e-12,
        "ego_1s_brier_skill_ge_0p10": gate_values["ego_1s_brier_skill"] >= 0.10,
        "ttc_lt2_mae_le_0p30": gate_values["ttc_lt2_mae"] is not None
        and gate_values["ttc_lt2_mae"] <= 0.30,
        "heldout_ego_collision_episodes_ge_30": gate_values["heldout_ego_collision_episodes"] >= 30,
    }
    return {
        "schema": "d2r-oof-report-1",
        "classification": classification,
        "alarms": alarms,
        "geometry": geometry,
        "ttc_lt2": ttc,
        "gate_values": gate_values,
        "gate_conditions": conditions,
        "gate_passed": all(conditions.values()),
    }


def run_oof(
    dataset_dir: str | Path,
    split_dir: str | Path,
    signals_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
    device_name: str = "cuda:0",
    seed: int = SEED,
    outer_folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    registry_path: str | Path | None = None,
    registry_opened_at: str = REGISTRY_OPENED_AT,
    evidence_relpath: str = EVIDENCE_RELPATH,
) -> dict:
    outer_folds = tuple(int(value) for value in outer_folds)
    if not outer_folds or len(set(outer_folds)) != len(outer_folds) or any(value not in range(5) for value in outer_folds):
        raise ValueError("D2R outer folds must be unique values in 0..4")
    dataset = D2RDataset(dataset_dir, split_dir, signals_dir)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("D2R CUDA requested but unavailable")
    output_dir = Path(output_dir)
    partial = _prepare_output(output_dir)
    (partial / "models").mkdir()
    registry_record = None
    if registry_path is not None:
        registry_path = Path(registry_path)
        before_sha = file_sha256(registry_path)
        registry_rows = make_registry_rows(
            dataset.base.episodes,
            opened_at_utc=registry_opened_at,
            evidence_relpath=evidence_relpath,
        )
        append_result = append_opened_registry(registry_path, registry_rows)
        snapshot = partial / "opened_registry.snapshot.tsv"
        shutil.copyfile(registry_path, snapshot)
        registry_record = {
            "before_sha256": before_sha,
            "after_sha256": file_sha256(snapshot),
            "required_rows": len(registry_rows),
            "appended": append_result.appended,
            "already_present": append_result.skipped,
            "live_total": append_result.total,
            "opened_at": registry_opened_at,
            "evidence_relpath": evidence_relpath,
        }
    _write_json(
        partial / "config.json",
        {
            "schema": "d2r-oof-config-1",
            "family": LOCKED_CONFIG.family,
            "created_at": str(created_at),
            "seed": int(seed),
            "device": str(device_name),
            "train_config": asdict(LOCKED_CONFIG),
            "outer_folds": list(outer_folds),
            "inner_folds": 3,
            "registry": registry_record,
            "dataset_manifest_sha256": file_sha256(Path(dataset_dir) / "dataset_manifest.json"),
            "signals_manifest_sha256": file_sha256(Path(signals_dir) / "signals_manifest.json"),
            "split_manifest_sha256": file_sha256(Path(split_dir) / "scenario_split.tsv"),
            "source_sha256": {
                "d2r_init": file_sha256(Path(__file__).with_name("__init__.py")),
                "d2r_data": file_sha256(Path(__file__).with_name("data.py")),
                "d2r_model": file_sha256(Path(__file__).with_name("model.py")),
                "d2r_train": file_sha256(Path(__file__)),
                "d2r_release": file_sha256(Path(__file__).with_name("release.py")),
                "d2r_cli": file_sha256(Path(__file__).parents[1] / "d2r_cli.py"),
            },
        },
    )
    predictions = np.lib.format.open_memmap(
        partial / "oof_predictions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(dataset.frame_count, len(PREDICTION_NAMES)),
    )
    predictions[:] = np.nan
    fold_reports = []
    for outer in outer_folds:
        outer_mask = dataset.base.outer_fold == outer
        outer_train = ~outer_mask
        calibration_indices = []
        calibration_predictions = []
        inner_reports = []
        for inner in range(3):
            calibration_mask = outer_train & (dataset.base.inner_fold[outer] == inner)
            fit_mask = outer_train & ~calibration_mask
            fit_l4 = {
                dataset.base.split_rows[index]["l4_id"] for index in np.flatnonzero(fit_mask)
            }
            held_l4 = {
                dataset.base.split_rows[index]["l4_id"]
                for index in np.flatnonzero(calibration_mask | outer_mask)
            }
            if fit_l4 & held_l4:
                raise AssertionError("D2R L4 leakage")
            model_seed = int(seed + outer * 100 + inner)
            model, mean, std, train_report = train_model(
                dataset, fit_mask, device, model_seed
            )
            frame_indices = dataset.base.frame_indices(calibration_mask)
            predicted = predict_model(model, dataset, frame_indices, mean, std, device)
            bundle = partial / "models" / f"outer{outer}_inner{inner}.pt"
            bundle_sha = _save_bundle(
                bundle, model, mean, std, LOCKED_CONFIG, model_seed, train_report
            )
            calibration_indices.append(frame_indices)
            calibration_predictions.append(predicted)
            inner_reports.append(
                {
                    "inner_fold": inner,
                    "fit_episodes": int(np.count_nonzero(fit_mask)),
                    "calibration_episodes": int(np.count_nonzero(calibration_mask)),
                    "fit_l4_count": len(fit_l4),
                    "calibration_l4_count": len(
                        {
                            dataset.base.split_rows[index]["l4_id"]
                            for index in np.flatnonzero(calibration_mask)
                        }
                    ),
                    "bundle_sha256": bundle_sha,
                    "train_report": train_report,
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        calibration_indices = np.concatenate(calibration_indices)
        calibration_predictions = np.concatenate(calibration_predictions)
        order = np.argsort(calibration_indices, kind="mergesort")
        calibration_indices = calibration_indices[order]
        calibration_predictions = calibration_predictions[order]
        expected_calibration = dataset.base.frame_indices(outer_train)
        if not np.array_equal(calibration_indices, expected_calibration):
            raise AssertionError("D2R inner OOF calibration coverage mismatch")
        calibrators = _fit_calibrators(dataset, calibration_predictions, calibration_indices)
        calibrated = _apply_calibrators(calibration_predictions, calibrators)
        thresholds = _select_thresholds(dataset, calibrated, calibration_indices, outer_train)

        refit_seed = int(seed + outer * 100 + 99)
        model, mean, std, train_report = train_model(dataset, outer_train, device, refit_seed)
        outer_indices = dataset.base.frame_indices(outer_mask)
        outer_predictions = predict_model(model, dataset, outer_indices, mean, std, device)
        outer_predictions = _apply_calibrators(outer_predictions, calibrators)
        predictions[outer_indices] = outer_predictions
        bundle = partial / "models" / f"outer{outer}_refit.pt"
        bundle_sha = _save_bundle(bundle, model, mean, std, LOCKED_CONFIG, refit_seed, train_report)
        training_prevalence = _training_prevalence(dataset, outer_train)
        report = {
            "schema": "d2r-outer-report-1",
            "outer_fold": outer,
            "outer_episodes": int(np.count_nonzero(outer_mask)),
            "train_episodes": int(np.count_nonzero(outer_train)),
            "training_prevalence": training_prevalence,
            "inner_reports": inner_reports,
            "calibrators": calibrators,
            "thresholds": thresholds,
            "refit_bundle_sha256": bundle_sha,
            "refit_train_report": train_report,
            "evaluation": _fold_evaluation(
                dataset,
                outer_predictions,
                outer_indices,
                outer_mask,
                outer_train,
                thresholds,
            ),
        }
        _write_json(partial / f"outer{outer}_report.json", report)
        fold_reports.append(report)
        predictions.flush()
        print(
            f"D2R_OUTER outer={outer} episodes={int(np.count_nonzero(outer_mask))} "
            f"ttc_mae={report['evaluation']['ttc_lt2']['mae']}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    predictions.flush()
    if outer_folds == (0, 1, 2, 3, 4):
        if np.any(~np.isfinite(predictions)):
            raise AssertionError("D2R complete OOF contains missing predictions")
        oof_report = _aggregate_oof(dataset, predictions, fold_reports)
    else:
        completed_frames = np.isfinite(np.asarray(predictions[:, 0]))
        expected_frames = np.isin(dataset.base.outer_fold[dataset.base.episode_index], outer_folds)
        if not np.array_equal(completed_frames, expected_frames):
            raise AssertionError("D2R engineering OOF frame coverage mismatch")
        oof_report = {
            "schema": "d2r-engineering-smoke-report-1",
            "outer_folds": list(outer_folds),
            "complete_oof": False,
            "gate_passed": False,
            "note": "engineering smoke cannot select a family or open test",
            "fold_evaluations": [report["evaluation"] for report in fold_reports],
        }
    _write_json(partial / "oof_report.json", oof_report)
    _write_output_manifest(partial)
    from d2r.release import validate_release

    preliminary = validate_release(
        partial,
        dataset_dir,
        split_dir,
        signals_dir,
        allow_partial=True,
    )
    if not preliminary["passed"]:
        raise AssertionError(f"D2R preliminary independent validation failed: {preliminary}")
    _write_json(partial / "validation.json", preliminary)
    _write_output_manifest(partial)
    final_validation = validate_release(
        partial,
        dataset_dir,
        split_dir,
        signals_dir,
        allow_partial=True,
    )
    if not final_validation["passed"]:
        raise AssertionError(f"D2R final independent validation failed: {final_validation}")
    _promote(partial, output_dir)
    return {
        "passed": True,
        "complete_oof": outer_folds == (0, 1, 2, 3, 4),
        "gate_passed": bool(oof_report["gate_passed"]),
        "oof_report_sha256": file_sha256(output_dir / "oof_report.json"),
        "output_manifest_sha256": file_sha256(output_dir / "output_manifest.sha256"),
        "validation": final_validation,
    }

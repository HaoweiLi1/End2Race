"""Slice synthesis for the completed, unopened-test D2R-G result."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from d2.dataset import _prepare_output, _promote, _write_json, _write_output_manifest
from d2.metrics import binary_metrics, ttc_mae
from d2.release import file_sha256
from d2r.data import D2RDataset


def _fold_thresholds(probe_dir: Path, probability_name: str) -> np.ndarray:
    values = []
    for outer in range(5):
        report = json.loads(
            (probe_dir / f"outer{outer}_report.json").read_text(encoding="utf-8")
        )
        values.append(float(report["thresholds"][probability_name]["threshold"]))
    return np.asarray(values, dtype=np.float64)


def _alarm_slice(
    dataset: D2RDataset,
    probability: np.ndarray,
    target_name: str,
    thresholds: np.ndarray,
    episode_mask: np.ndarray,
) -> dict:
    base = dataset.base
    valid = np.asarray(base.arrays[target_name.replace("target", "valid")], dtype=bool)
    target = np.asarray(base.arrays[target_name], dtype=bool)
    frame_episode = base.episode_index
    frame_threshold = thresholds[base.outer_fold[frame_episode]]
    alarm = episode_mask[frame_episode] & valid & (probability >= frame_threshold)
    any_alarm = np.zeros(base.episode_count, dtype=bool)
    event_alarm = np.zeros(base.episode_count, dtype=bool)
    selected = np.flatnonzero(alarm)
    if len(selected):
        np.logical_or.at(any_alarm, frame_episode[selected], True)
    event_selected = np.flatnonzero(alarm & target)
    if len(event_selected):
        np.logical_or.at(event_alarm, frame_episode[event_selected], True)
    safe = episode_mask & ~base.any_collision
    event = episode_mask & base.ego_collision
    return {
        "safe_episode_count": int(np.count_nonzero(safe)),
        "safe_episode_false_alarm_rate": float(np.mean(any_alarm[safe])) if np.any(safe) else None,
        "ego_collision_episode_count": int(np.count_nonzero(event)),
        "ego_event_recall": float(np.mean(event_alarm[event])) if np.any(event) else None,
    }


def _ttc_bins(target: np.ndarray, predicted: np.ndarray) -> dict:
    output = {}
    for low, high, token in (
        (0.0, 0.25, "000_025"),
        (0.25, 0.5, "025_050"),
        (0.5, 1.0, "050_100"),
        (1.0, 1.5, "100_150"),
        (1.5, 2.0, "150_200"),
    ):
        mask = (target >= low) & (target < high)
        output[token] = {
            "low_s": low,
            "high_s": high,
            "count": int(np.count_nonzero(mask)),
            "target_mean": float(np.mean(target[mask])) if np.any(mask) else None,
            "prediction_mean": float(np.mean(predicted[mask])) if np.any(mask) else None,
            "mae": float(np.mean(np.abs(predicted[mask] - target[mask]))) if np.any(mask) else None,
        }
    return output


def _slice_report(
    dataset: D2RDataset,
    predictions: np.ndarray,
    probe_dir: Path,
    episode_mask: np.ndarray,
) -> dict:
    base = dataset.base
    frames = base.frame_indices(episode_mask)
    valid = np.asarray(base.arrays["ego_valid_100"][frames], dtype=bool)
    target = np.asarray(base.arrays["ego_target_100"][frames], dtype=bool)
    probability = predictions[frames, 1].astype(np.float64)
    classification = None
    if np.any(valid) and np.any(target[valid]) and np.any(~target[valid]):
        prevalence = float(np.mean(target[valid]))
        classification = binary_metrics(target[valid], probability[valid], prevalence)
    target_ttc = np.asarray(base.arrays["corridor_ttc"][frames], dtype=np.float64)
    predicted_ttc = predictions[frames, 9].astype(np.float64)
    geometry = {}
    for name, column, low, high in (
        ("rel_s", 6, -10.0, 10.0),
        ("lateral_gap", 7, 0.0, 2.0),
        ("closing_rate", 8, -5.0, 5.0),
    ):
        truth = np.clip(np.asarray(base.arrays[name][frames], dtype=np.float64), low, high)
        geometry[name] = {"mae": float(np.mean(np.abs(predictions[frames, column] - truth)))}
    return {
        "episode_count": int(np.count_nonzero(episode_mask)),
        "frame_count": len(frames),
        "any_collision_episode_count": int(np.count_nonzero(episode_mask & base.any_collision)),
        "ego_collision_episode_count": int(np.count_nonzero(episode_mask & base.ego_collision)),
        "ego_1s_classification": classification,
        "ego_1s_alarm": _alarm_slice(
            dataset,
            predictions[:, 1],
            "ego_target_100",
            _fold_thresholds(probe_dir, "ego_probability_100"),
            episode_mask,
        ),
        "ego_2s_alarm": _alarm_slice(
            dataset,
            predictions[:, 2],
            "ego_target_200",
            _fold_thresholds(probe_dir, "ego_probability_200"),
            episode_mask,
        ),
        "geometry": geometry,
        "ttc_lt2": ttc_mae(target_ttc, predicted_ttc),
        "ttc_bins_lt2": _ttc_bins(target_ttc, predicted_ttc),
    }


def create_summary(
    dataset_dir: str | Path,
    split_dir: str | Path,
    signals_dir: str | Path,
    probe_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
) -> dict:
    dataset = D2RDataset(dataset_dir, split_dir, signals_dir)
    probe_dir = Path(probe_dir)
    if not (probe_dir / "COMPLETE").is_file():
        raise ValueError("D2R probe release lacks COMPLETE")
    report = json.loads((probe_dir / "oof_report.json").read_text(encoding="utf-8"))
    if report["gate_passed"]:
        raise ValueError("D2R failure summary cannot summarize a passing family")
    predictions = np.load(probe_dir / "oof_predictions.npy", mmap_mode="r", allow_pickle=False)
    if predictions.shape != (dataset.frame_count, 10) or np.any(~np.isfinite(predictions)):
        raise ValueError("D2R complete predictions invalid")
    fields = {
        "map": np.asarray([row["map_name"] for row in dataset.base.episodes]),
        "skill": np.asarray([row["skill"] for row in dataset.base.episodes]),
        "opponent_raceline": np.asarray(
            [row["opponent_raceline"] for row in dataset.base.episodes]
        ),
        "speedscale": np.asarray([row["speedscale_hex"] for row in dataset.base.episodes]),
    }
    slices = {
        "all": _slice_report(
            dataset, predictions, probe_dir, np.ones(dataset.episode_count, dtype=bool)
        )
    }
    for field, values in fields.items():
        for value in sorted(set(values.tolist())):
            slices[f"{field}:{value}"] = _slice_report(
                dataset, predictions, probe_dir, values == value
            )
    summary = {
        "schema": "d2r-evidence-summary-1",
        "created_at": str(created_at),
        "decision": "STOP_D3_TEST_UNOPENED_D2R_G_FAILED_TTC_AND_2S_FA",
        "test_opened": False,
        "selected_family": None,
        "population": {
            "episodes": dataset.episode_count,
            "frames": dataset.frame_count,
            "ego_collision_episodes": int(np.count_nonzero(dataset.base.ego_collision)),
        },
        "gate_values": report["gate_values"],
        "gate_conditions": report["gate_conditions"],
        "geometry": report["geometry"],
        "source": {
            "probe_oof_report_sha256": file_sha256(probe_dir / "oof_report.json"),
            "probe_output_manifest_sha256": file_sha256(probe_dir / "output_manifest.sha256"),
            "probe_oof_predictions_sha256": file_sha256(probe_dir / "oof_predictions.npy"),
        },
        "slices": slices,
    }
    output_dir = Path(output_dir)
    partial = _prepare_output(output_dir)
    _write_json(partial / "d2r_summary.json", summary)
    lines = [
        "# D2R-G Spatiotemporal Geometry — Evidence Summary",
        "",
        "Decision: `STOP_D3_TEST_UNOPENED_D2R_G_FAILED_TTC_AND_2S_FA`.",
        "",
        "| metric | result | required | gate |",
        "|---|---:|---:|:---:|",
        f"| 1s ego recall | {report['gate_values']['ego_1s_recall']:.3f} | >=0.600 | {'pass' if report['gate_conditions']['ego_1s_recall_ge_0p60'] else 'fail'} |",
        f"| 1s safe FA | {report['gate_values']['ego_1s_safe_fa']:.3f} | <=0.100 | {'pass' if report['gate_conditions']['ego_1s_safe_fa_le_0p10'] else 'fail'} |",
        f"| 2s ego recall | {report['gate_values']['ego_2s_recall']:.3f} | >=0.400 | {'pass' if report['gate_conditions']['ego_2s_recall_ge_0p40'] else 'fail'} |",
        f"| 2s safe FA | {report['gate_values']['ego_2s_safe_fa']:.3f} | <=0.100 | {'pass' if report['gate_conditions']['ego_2s_safe_fa_le_0p10'] else 'fail'} |",
        f"| 1s Brier skill | {report['gate_values']['ego_1s_brier_skill']:.3f} | >=0.100 | {'pass' if report['gate_conditions']['ego_1s_brier_skill_ge_0p10'] else 'fail'} |",
        f"| TTC<2 MAE | {report['gate_values']['ttc_lt2_mae']:.3f}s | <=0.300s | {'pass' if report['gate_conditions']['ttc_lt2_mae_le_0p30'] else 'fail'} |",
        "",
        "No test was opened and no family was selected. Detailed map, skill, raceline, speedscale, geometry, and TTC-bin slices are in `d2r_summary.json`.",
    ]
    (partial / "d2r_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_output_manifest(partial)
    _promote(partial, output_dir)
    return {
        "passed": True,
        "decision": summary["decision"],
        "summary_json_sha256": file_sha256(output_dir / "d2r_summary.json"),
        "summary_md_sha256": file_sha256(output_dir / "d2r_summary.md"),
        "output_manifest_sha256": file_sha256(output_dir / "output_manifest.sha256"),
    }

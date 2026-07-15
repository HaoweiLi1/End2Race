"""Evidence synthesis and slice audit for the unopened-test D2 result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from d2.dataset import _prepare_output, _promote, _write_json, _write_output_manifest
from d2.metrics import binary_metrics, ttc_mae
from d2.probe import ProbeDataset
from d2.release import file_sha256


def _fold_thresholds(probe_dir: Path, probability_name: str) -> np.ndarray:
    values = []
    for outer in range(5):
        report = json.loads((probe_dir / f"outer{outer}_report.json").read_text(encoding="utf-8"))
        values.append(float(report["thresholds"][probability_name]["threshold"]))
    return np.asarray(values, dtype=np.float64)


def _alarm_slice(
    dataset: ProbeDataset,
    probability: np.ndarray,
    target_name: str,
    thresholds: np.ndarray,
    episode_mask: np.ndarray,
) -> dict:
    valid = np.asarray(dataset.arrays[target_name.replace("target", "valid")], dtype=bool)
    target = np.asarray(dataset.arrays[target_name], dtype=bool)
    frame_episode = dataset.episode_index
    frame_in_slice = episode_mask[frame_episode]
    frame_threshold = thresholds[dataset.outer_fold[frame_episode]]
    alarm = frame_in_slice & valid & (probability >= frame_threshold)
    any_alarm = np.zeros(dataset.episode_count, dtype=bool)
    event_alarm = np.zeros(dataset.episode_count, dtype=bool)
    selected = np.flatnonzero(alarm)
    if len(selected):
        np.logical_or.at(any_alarm, frame_episode[selected], True)
    event_selected = np.flatnonzero(alarm & target)
    if len(event_selected):
        np.logical_or.at(event_alarm, frame_episode[event_selected], True)
    safe = episode_mask & ~dataset.any_collision
    event = episode_mask & dataset.ego_collision
    return {
        "safe_episode_count": int(np.count_nonzero(safe)),
        "safe_episode_false_alarm_rate": (
            float(np.mean(any_alarm[safe])) if np.any(safe) else None
        ),
        "ego_collision_episode_count": int(np.count_nonzero(event)),
        "ego_event_recall": float(np.mean(event_alarm[event])) if np.any(event) else None,
    }


def _slice_report(
    dataset: ProbeDataset,
    predictions: np.ndarray,
    probe_dir: Path,
    episode_mask: np.ndarray,
) -> dict:
    frame_indices = dataset.frame_indices(episode_mask)
    target_name = "ego_target_100"
    valid = np.asarray(dataset.arrays["ego_valid_100"][frame_indices], dtype=bool)
    target = np.asarray(dataset.arrays[target_name][frame_indices], dtype=bool)
    probability = predictions[frame_indices, 1].astype(np.float64)
    prevalence = float(np.mean(target[valid])) if np.any(valid) else 0.0
    classification = (
        binary_metrics(target[valid], probability[valid], prevalence)
        if np.any(valid) and np.any(target[valid]) and np.any(~target[valid])
        else None
    )
    ttc = ttc_mae(
        np.asarray(dataset.arrays["corridor_ttc"][frame_indices], dtype=np.float64),
        predictions[frame_indices, 7].astype(np.float64),
    )
    return {
        "episode_count": int(np.count_nonzero(episode_mask)),
        "frame_count": len(frame_indices),
        "any_collision_episode_count": int(np.count_nonzero(episode_mask & dataset.any_collision)),
        "ego_collision_episode_count": int(np.count_nonzero(episode_mask & dataset.ego_collision)),
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
        "ttc_lt2": ttc,
    }


def _prevalence_baseline(dataset: ProbeDataset) -> dict:
    ttc_target = np.asarray(dataset.arrays["corridor_ttc"], dtype=np.float64)
    prediction = np.empty(dataset.frame_count, dtype=np.float64)
    for outer in range(5):
        train_episode = dataset.outer_fold != outer
        held_episode = ~train_episode
        train_frames = train_episode[dataset.episode_index]
        held_frames = held_episode[dataset.episode_index]
        prediction[held_frames] = float(np.mean(ttc_target[train_frames]))
    return {
        "ego_1s_brier_skill": 0.0,
        "ego_1s_event_recall_at_safe_fa_le_0p10": 0.0,
        "ego_2s_event_recall_at_safe_fa_le_0p10": 0.0,
        "safe_episode_false_alarm_rate": 0.0,
        "ttc_lt2": ttc_mae(ttc_target, prediction),
        "note": "constant alarm must stay off to meet episode-level safe FA <= 10%",
    }


def create_d2_summary(
    dataset_dir: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    family_dirs: Mapping[str, str | Path],
    t2_family: str = "temporal_deployable",
) -> dict:
    dataset = ProbeDataset(dataset_dir, split_dir)
    family_dirs = {name: Path(path) for name, path in family_dirs.items()}
    reports = {
        name: json.loads((path / "oof_report.json").read_text(encoding="utf-8"))
        for name, path in family_dirs.items()
    }
    if t2_family not in family_dirs:
        raise ValueError("T2 family missing from D2 summary inputs")
    t2_dir = family_dirs[t2_family]
    predictions = np.load(t2_dir / "oof_predictions.npy", mmap_mode="r", allow_pickle=False)
    if predictions.shape != (dataset.frame_count, 8):
        raise ValueError("T2 OOF prediction shape mismatch")

    episode_fields = {
        "map": np.asarray([row["map_name"] for row in dataset.episodes]),
        "skill": np.asarray([row["skill"] for row in dataset.episodes]),
        "opponent_raceline": np.asarray([row["opponent_raceline"] for row in dataset.episodes]),
        "speedscale": np.asarray([row["speedscale_hex"] for row in dataset.episodes]),
    }
    slices = {"all": _slice_report(dataset, predictions, t2_dir, np.ones(dataset.episode_count, dtype=bool))}
    for field, values in episode_fields.items():
        for value in sorted(set(values.tolist())):
            slices[f"{field}:{value}"] = _slice_report(
                dataset, predictions, t2_dir, values == value
            )

    family_table = {}
    for name, report in reports.items():
        family_table[name] = {
            "gate_passed": bool(report["gate_passed"]),
            **report["gate_values"],
            "oof_report_sha256": file_sha256(family_dirs[name] / "oof_report.json"),
            "output_manifest_sha256": file_sha256(family_dirs[name] / "output_manifest.sha256"),
        }
    summary = {
        "schema": "d2-evidence-summary-1",
        "decision": "STOP_D3_TEST_UNOPENED_CONTINUE_D2P5_DIAGNOSTIC",
        "test_opened": False,
        "selected_family": None,
        "population": {
            "episodes": dataset.episode_count,
            "frames": dataset.frame_count,
            "ego_collision_episodes": int(np.count_nonzero(dataset.ego_collision)),
            "any_collision_episodes": int(np.count_nonzero(dataset.any_collision)),
        },
        "prevalence_baseline": _prevalence_baseline(dataset),
        "families": family_table,
        "t2_slices": slices,
        "gate_failure": {
            "blocking_metric": "ttc_lt2_mae",
            "required": 0.30,
            "observed": float(reports[t2_family]["gate_values"]["ttc_lt2_mae"]),
            "classification_subgate_passed": all(
                (
                    reports[t2_family]["gate_values"]["ego_1s_recall"] >= 0.60,
                    reports[t2_family]["gate_values"]["ego_1s_safe_fa"] <= 0.10,
                    reports[t2_family]["gate_values"]["ego_2s_recall"] >= 0.40,
                    reports[t2_family]["gate_values"]["ego_2s_safe_fa"] <= 0.10,
                    reports[t2_family]["gate_values"]["ego_1s_brier_skill"] >= 0.10,
                )
            ),
        },
    }
    output_dir = Path(output_dir)
    partial = _prepare_output(output_dir)
    _write_json(partial / "d2_summary.json", summary)
    lines = [
        "# D2 Episode-Held-Out Representation Probe — Evidence Report",
        "",
        f"Population: {dataset.episode_count} non-test episodes / {dataset.frame_count} frames; "
        f"{int(np.count_nonzero(dataset.ego_collision))} ego-collision episodes.",
        "",
        "| family | 1s recall | 1s safe FA | 2s recall | 2s safe FA | 1s BSS | TTC<2 MAE | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, values in family_table.items():
        lines.append(
            f"| {name} | {values['ego_1s_recall']:.3f} | {values['ego_1s_safe_fa']:.3f} | "
            f"{values['ego_2s_recall']:.3f} | {values['ego_2s_safe_fa']:.3f} | "
            f"{values['ego_1s_brier_skill']:.3f} | {values['ttc_lt2_mae']:.3f} | "
            f"{'PASS' if values['gate_passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "The final T2 temporal family passes every collision-classification sub-gate but "
            f"fails TTC MAE ({summary['gate_failure']['observed']:.3f}s versus <=0.300s).",
            "The grouped probe test remains sealed and no family is selected. D3/PPO is blocked; "
            "D2.5 may continue only as an action-space diagnostic.",
            "",
            "Detailed map/skill/raceline/speedscale slices are in `d2_summary.json`.",
        ]
    )
    (partial / "d2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_output_manifest(partial)
    _promote(partial, output_dir)
    return {
        "passed": True,
        "decision": summary["decision"],
        "test_opened": False,
        "summary_json_sha256": file_sha256(output_dir / "d2_summary.json"),
        "summary_md_sha256": file_sha256(output_dir / "d2_summary.md"),
        "output_manifest_sha256": file_sha256(output_dir / "output_manifest.sha256"),
    }


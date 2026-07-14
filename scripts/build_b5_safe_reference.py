#!/usr/bin/env python3
"""Build the fixed 64-episode B5-A canonical-BC safe reference artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from bplus_v22.b4_direct import load_strict_plain_actor
from bplus_v22.b5_safe import (
    B5_REFERENCE_SCHEMA,
    SafeReference,
    file_sha256,
    save_reference,
    select_reference_rows,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1640 or [int(row["training_order"]) for row in rows] != list(
        range(1640)
    ):
        raise ValueError("B5 reference requires the canonical 1,640-row training manifest")
    return rows


def _raceline_initial_speed(repo: Path, row: dict[str, str]) -> float:
    path = repo / f"f1tenth_racetracks/{row['map_name']}/raceline1.csv"
    values = np.loadtxt(path, delimiter=";", skiprows=1)
    return float(values[int(row["resolved_ego_idx"]) % len(values), 5])


def _npz_path(npz_root: Path, row: dict[str, str]) -> Path:
    path = npz_root / row["npz_relpath"]
    if not path.is_file() or file_sha256(path) != row["npz_sha256"]:
        raise ValueError(f"B5 source NPZ is missing or hash-mismatched: {row['l2_id']}")
    return path


def build(
    *,
    repo: Path,
    npz_root: Path,
    training_manifest: Path,
    bc_checkpoint: Path,
    output: Path,
    device_name: str,
) -> dict[str, object]:
    selected = select_reference_rows(_rows(training_manifest))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("B5 reference builder requested unavailable CUDA")
    actor = load_strict_plain_actor(bc_checkpoint, device)
    actor.eval()
    features = []
    means = []
    episode_indices = []
    step_indices = []
    lengths = []
    l2_ids = []
    l4_ids = []
    map_names = []
    outcomes = []
    max_abs_stored_steer_error = 0.0
    max_abs_stored_speed_error = 0.0
    source_npz_sha256 = []

    with torch.inference_mode():
        for episode_index, row in enumerate(selected):
            path = _npz_path(npz_root, row)
            source_npz_sha256.append(row["npz_sha256"])
            with np.load(path, allow_pickle=False) as value:
                lidar = np.asarray(value["ego_lidar"], dtype=np.float32)
                actual_speed = np.asarray(value["ego_actual_speed"], dtype=np.float32)
                stored_steer = np.asarray(value["ego_desired_steer"], dtype=np.float32)
                stored_speed = np.asarray(value["ego_desired_speed"], dtype=np.float32)
            if lidar.ndim != 2 or lidar.shape[1] != 360 or len(actual_speed) != len(lidar):
                raise ValueError(f"B5 source trajectory shape drift: {row['l2_id']}")
            previous_speed = np.empty((len(lidar), 1), dtype=np.float32)
            previous_speed[0, 0] = _raceline_initial_speed(repo, row) * 0.9
            previous_speed[1:, 0] = actual_speed[:-1]
            hidden = None
            episode_features = []
            episode_means = []
            for step in range(len(lidar)):
                lidar_tensor = torch.from_numpy(lidar[step : step + 1]).to(device).reshape(1, 1, 360)
                speed_tensor = (
                    torch.from_numpy(previous_speed[step : step + 1])
                    .to(device)
                    .reshape(1, 1, 1)
                )
                feature, hidden = actor.forward_features(lidar_tensor, speed_tensor, hidden)
                mean = actor.output_layer(feature[:, -1, :])
                episode_features.append(feature[:, -1, :].cpu())
                episode_means.append(mean.cpu())
            feature = torch.cat(episode_features, dim=0).to(torch.float32)
            mean = torch.cat(episode_means, dim=0).to(torch.float32)
            if len(stored_steer) != len(mean) or len(stored_speed) != len(mean):
                raise ValueError(f"B5 stored action length drift: {row['l2_id']}")
            steer_error = np.max(
                np.abs(np.clip(mean[:, 0].numpy(), -0.52, 0.52) - stored_steer)
            )
            speed_error = np.max(np.abs(mean[:, 1].numpy() - stored_speed))
            max_abs_stored_steer_error = max(max_abs_stored_steer_error, float(steer_error))
            max_abs_stored_speed_error = max(max_abs_stored_speed_error, float(speed_error))
            if steer_error > 2e-3 or speed_error > 1e-2:
                raise ValueError(
                    f"B5 canonical BC replay mismatch for {row['l2_id']}: "
                    f"steer={steer_error}, speed={speed_error}"
                )
            outcome = Path(row["npz_relpath"]).parent.name
            features.append(feature)
            means.append(mean)
            episode_indices.append(torch.full((len(mean),), episode_index, dtype=torch.int64))
            step_indices.append(torch.arange(len(mean), dtype=torch.int64))
            lengths.append(len(mean))
            l2_ids.append(row["l2_id"])
            l4_ids.append(row["l4_id"])
            map_names.append(row["map_name"])
            outcomes.append(outcome)

    reference = SafeReference(
        feature=torch.cat(features),
        bc_mean=torch.cat(means),
        episode_index=torch.cat(episode_indices),
        step_index=torch.cat(step_indices),
        lengths=tuple(lengths),
        l2_ids=tuple(l2_ids),
        l4_ids=tuple(l4_ids),
        map_names=tuple(map_names),
        outcomes=tuple(outcomes),
    )
    digest = save_reference(reference, output)
    return {
        "schema": B5_REFERENCE_SCHEMA,
        "passed": True,
        "path": str(output.resolve()),
        "sha256": digest,
        "episode_count": len(reference.lengths),
        "frame_count": reference.frame_count,
        "selection": [
            {
                "episode_index": index,
                "l2_id": l2_id,
                "l4_id": l4_id,
                "map_name": map_name,
                "outcome": outcome,
                "length": length,
                "source_npz_sha256": npz_sha,
            }
            for index, (l2_id, l4_id, map_name, outcome, length, npz_sha) in enumerate(
                zip(
                    reference.l2_ids,
                    reference.l4_ids,
                    reference.map_names,
                    reference.outcomes,
                    reference.lengths,
                    source_npz_sha256,
                )
            )
        ],
        "max_abs_stored_bc_steer_error": max_abs_stored_steer_error,
        "max_abs_stored_bc_speed_error": max_abs_stored_speed_error,
        "bc_checkpoint_sha256": file_sha256(bc_checkpoint),
        "training_manifest_sha256": file_sha256(training_manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--npz-root", type=Path, default=Path.cwd())
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(args.report.suffix + ".partial").exists():
        raise FileExistsError(args.report)
    result = build(
        repo=args.repo.resolve(),
        npz_root=args.npz_root.resolve(),
        training_manifest=args.training_manifest.resolve(),
        bc_checkpoint=args.bc_checkpoint.resolve(),
        output=args.output.resolve(),
        device_name=args.device,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    partial = args.report.with_suffix(args.report.suffix + ".partial")
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, args.report)
    print(json.dumps({key: result[key] for key in ("sha256", "episode_count", "frame_count")}))


if __name__ == "__main__":
    main()

"""Registry-gated extraction of frozen BC features and D2 labels."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from d0.identity import (
    REGISTRY_FIELDS,
    append_opened_registry,
    registry_row_id,
    validate_registry_row,
)
from d0.outcomes import centerline_length
from d2.labels import LabelConfig, ReferenceProjector, build_episode_labels
from d2.release import file_sha256
from d2.replay import replay_bc_features
from d2.split import SPLIT_FIELDS, split_digest, validate_split
from ppo_utils import load_frozen_bc


BC_SHA256 = "b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4"
HORIZON_TOKENS = ("050", "100", "200")
ARRAY_DTYPES = {
    "features": np.float32,
    "time": np.float32,
    "episode_index": np.int32,
    "closing_rate": np.float32,
    "corridor_ttc": np.float32,
    "rel_s": np.float32,
    "lateral_gap": np.float32,
    "ego_v_s": np.float32,
    "opp_v_s": np.float32,
    **{f"{prefix}_{kind}_{token}": np.uint8
       for prefix in ("ego", "any")
       for kind in ("target", "valid")
       for token in HORIZON_TOKENS},
}
REQUIRED_NPZ_KEYS = {
    "time",
    "ego_lidar",
    "ego_desired_steer",
    "ego_desired_speed",
    "ego_actual_speed",
    "ego_pose",
    "ego_progress",
    "opp_actual_speed",
    "opp_pose",
    "opp_progress",
    "ego_collision",
    "opp_collision",
    "final_time",
    "final_ego_progress",
    "final_opp_progress",
}
SOURCE_INVENTORY_FIELDS = (
    "l2_id",
    "npz_relpath",
    "npz_sha256",
    "frame_start",
    "frame_count",
)
EPISODE_FIELDS = (
    "episode_index",
    "l2_id",
    "l3_id",
    "l4_id",
    "map_name",
    "skill",
    "opponent_raceline",
    "speedscale_hex",
    "outer_fold",
    "representative_l1_id",
    "resolved_ego_idx",
    "npz_relpath",
    "npz_sha256",
    "frame_start",
    "frame_count",
    "final_time_hex",
    "ego_collision",
    "opp_collision",
    "collision_any",
)


def _read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: Iterable[Mapping], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if tuple(row) != fields:
                raise ValueError(f"field order mismatch while writing {path.name}")
            writer.writerow(row)


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_output(output_dir: Path) -> Path:
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"D2 dataset output exists or is nonempty: {output_dir}")
        output_dir.rmdir()
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"D2 dataset partial exists: {partial}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    return partial


def _promote(partial: Path, output_dir: Path) -> None:
    for path in partial.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted((path for path in partial.rglob("*") if path.is_dir()), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(partial)
    os.replace(partial, output_dir)
    _fsync_directory(output_dir.parent)
    complete = output_dir / "COMPLETE"
    with complete.open("w", encoding="utf-8") as handle:
        handle.write("COMPLETE\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(output_dir)


def make_registry_rows(
    selected_rows: Iterable[Mapping],
    source_manifest_sha256: str,
    opened_at_utc: str,
    evidence_relpath: str,
) -> list[dict]:
    rows = []
    for selected in selected_rows:
        if selected["split"] != "non_test":
            raise ValueError("test row cannot be registered as non-test probe fit")
        row = {
            "registry_schema": "bplus-opened-registry-1",
            "opened_at_utc": str(opened_at_utc),
            "stage": "D2",
            "use_class": "probe_fit",
            "split_id": f"d2_non_test_{str(source_manifest_sha256)[:16]}",
            "l2_id": str(selected["l2_id"]),
            "l3_id": str(selected["l3_id"]),
            "l4_id": str(selected["l4_id"]),
            "map_name": str(selected["map_name"]),
            "source_manifest_sha256": str(source_manifest_sha256),
            "source_run_id": "20260710_121955",
            "decision_effect": "representation_choice",
            "final_pool": "false",
            "evidence_relpath": str(evidence_relpath),
        }
        row["row_id"] = registry_row_id(row)
        rows.append(validate_registry_row(row))
    rows.sort(key=lambda row: row["row_id"])
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate D2 registry row ID")
    return rows


def _validate_registry_snapshot(path: Path, required_rows: Iterable[Mapping]) -> None:
    existing = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != REGISTRY_FIELDS:
            raise ValueError("opened registry header mismatch")
        for raw in reader:
            row = validate_registry_row(raw)
            prior = existing.get(row["row_id"])
            if prior is not None and prior != row:
                raise ValueError("conflicting registry duplicate")
            existing[row["row_id"]] = row
    for required in required_rows:
        if existing.get(required["row_id"]) != dict(required):
            raise ValueError("required D2 registry row missing or corrupt")


def _select_non_test(
    split_rows: list[dict],
    source_rows: list[dict],
    max_episodes_per_map: int | None,
) -> tuple[list[dict], dict[str, dict]]:
    non_test = [row for row in split_rows if row["split"] == "non_test"]
    if max_episodes_per_map is not None:
        if max_episodes_per_map <= 0:
            raise ValueError("max_episodes_per_map must be positive")
        chosen = []
        for map_name in sorted({row["map_name"] for row in non_test}):
            map_rows = sorted(
                (row for row in non_test if row["map_name"] == map_name),
                key=lambda row: row["l2_id"],
            )
            chosen.extend(map_rows[:max_episodes_per_map])
        non_test = sorted(chosen, key=lambda row: row["l2_id"])
    else:
        non_test.sort(key=lambda row: row["l2_id"])
    sources = {row["l2_id"]: row for row in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("duplicate L2 source locator")
    if set(row["l2_id"] for row in non_test) - set(sources):
        raise ValueError("selected non-test row lacks source locator")
    return non_test, sources


def _scan_sources(repo_root: Path, selected: list[dict], sources: Mapping[str, Mapping]) -> tuple[list[dict], int]:
    inventory = []
    frame_start = 0
    for row in selected:
        source = sources[row["l2_id"]]
        path = repo_root / source["npz_relpath"]
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        if file_sha256(path) != source["npz_sha256"]:
            raise ValueError(f"NPZ SHA256 mismatch: {source['npz_relpath']}")
        with np.load(path, allow_pickle=False) as data:
            if not REQUIRED_NPZ_KEYS.issubset(data.files):
                raise ValueError(f"NPZ missing D2 fields: {source['npz_relpath']}")
            time = np.asarray(data["time"])
            if time.ndim != 1 or len(time) == 0:
                raise ValueError(f"NPZ has invalid time: {source['npz_relpath']}")
            frame_count = len(time)
        inventory.append(
            {
                "l2_id": row["l2_id"],
                "npz_relpath": source["npz_relpath"],
                "npz_sha256": source["npz_sha256"],
                "frame_start": str(frame_start),
                "frame_count": str(frame_count),
            }
        )
        frame_start += frame_count
    return inventory, frame_start


def _initial_speed_cache(repo_root: Path, maps: Iterable[str]) -> dict[str, np.ndarray]:
    cache = {}
    for map_name in sorted(set(maps)):
        path = repo_root / "f1tenth_racetracks" / map_name / "raceline1.csv"
        rows = np.loadtxt(path, delimiter=";", skiprows=1, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] < 6:
            raise ValueError(f"invalid raceline asset: {path}")
        cache[map_name] = rows[:, 5]
    return cache


def _projection_error(projector: ReferenceProjector, poses: np.ndarray) -> float:
    count = min(16, len(poses))
    indices = np.unique(np.linspace(0, len(poses) - 1, count, dtype=int))
    points = np.asarray(poses, dtype=np.float64)[indices, :2]
    fast = projector.project_many(points)
    exact = projector.project_many_exhaustive(points)
    errors = [np.max(np.abs(left - right)) if len(left) else 0.0 for left, right in zip(fast[:2], exact[:2])]
    angle = np.arctan2(np.sin(fast[2] - exact[2]), np.cos(fast[2] - exact[2]))
    errors.append(float(np.max(np.abs(angle))) if len(angle) else 0.0)
    maximum = float(max(errors))
    if maximum > 1e-9:
        raise ValueError(f"fast/exhaustive reference projection mismatch: {maximum}")
    return maximum


def _array_files(directory: Path, total_frames: int, feature_dim: int) -> dict[str, np.memmap]:
    arrays_dir = directory / "arrays"
    arrays_dir.mkdir()
    arrays = {}
    for name, dtype in ARRAY_DTYPES.items():
        shape = (total_frames, feature_dim) if name == "features" else (total_frames,)
        arrays[name] = np.lib.format.open_memmap(
            arrays_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
    return arrays


def _write_output_manifest(directory: Path) -> None:
    files = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    )
    lines = [f"{file_sha256(directory / relpath)}  {relpath}" for relpath in files]
    (directory / "output_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_non_test_dataset(
    repo_root: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    registry_path: str | Path,
    created_at: str,
    registry_opened_at: str,
    evidence_relpath: str,
    model_relpath: str = "pretrained/end2race.pth",
    device_name: str = "cuda",
    max_episodes_per_map: int | None = None,
) -> dict:
    repo_root = Path(repo_root).resolve()
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    registry_path = Path(registry_path)
    if not (split_dir / "COMPLETE").is_file():
        raise ValueError("D2 split lock lacks COMPLETE")
    split_rows = _read_tsv(split_dir / "scenario_split.tsv")
    validate_split(split_rows, expected_l2=3036)
    source_rows = _read_tsv(split_dir / "non_test_sources.tsv")
    selected, sources = _select_non_test(split_rows, source_rows, max_episodes_per_map)
    split_file_sha = file_sha256(split_dir / "scenario_split.tsv")
    source_file_sha = file_sha256(split_dir / "non_test_sources.tsv")
    model_path = repo_root / model_relpath
    if file_sha256(model_path) != BC_SHA256:
        raise ValueError("frozen BC checkpoint SHA256 mismatch")

    partial = _prepare_output(output_dir)
    config = {
        "schema": "d2-dataset-config-1",
        "analysis_version": "d2",
        "partition": "non_test",
        "created_at": str(created_at),
        "registry_opened_at": str(registry_opened_at),
        "repo_root": str(repo_root),
        "split_relpath": str(evidence_relpath),
        "split_manifest_file_sha256": split_file_sha,
        "split_manifest_domain_sha256": split_digest(split_rows),
        "non_test_sources_sha256": source_file_sha,
        "model_relpath": str(model_relpath),
        "model_sha256": BC_SHA256,
        "device": str(device_name),
        "max_episodes_per_map": max_episodes_per_map,
        "selected_episode_count": len(selected),
        "feature_dim": 1680,
        "feature_dtype": "float32",
        "replay": "framewise-batch1-hidden-reset-lagged-actual-speed-v1",
        "labels": "d2-labels-1",
        "source_sha256": {
            "dataset.py": file_sha256(Path(__file__)),
            "labels.py": file_sha256(Path(__file__).with_name("labels.py")),
            "replay.py": file_sha256(Path(__file__).with_name("replay.py")),
            "model.py": file_sha256(repo_root / "model.py"),
        },
    }
    _write_json(partial / "config.json", config)

    registry_rows = make_registry_rows(
        selected,
        source_manifest_sha256=split_file_sha,
        opened_at_utc=registry_opened_at,
        evidence_relpath=evidence_relpath,
    )
    append_result = append_opened_registry(registry_path, registry_rows)
    _validate_registry_snapshot(registry_path, registry_rows)
    snapshot = partial / "opened_registry.after_partition.tsv"
    shutil.copyfile(registry_path, snapshot)
    _validate_registry_snapshot(snapshot, registry_rows)

    inventory, total_frames = _scan_sources(repo_root, selected, sources)
    _write_tsv(partial / "source_inventory.tsv", inventory, SOURCE_INVENTORY_FIELDS)
    arrays = _array_files(partial, total_frames, feature_dim=1680)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    bc = load_frozen_bc(str(model_path), device, hidden_scale=4)
    if bc.gru.hidden_size != 1680:
        raise ValueError("unexpected frozen BC feature dimension")
    label_config = LabelConfig()
    projectors = {
        map_name: ReferenceProjector.from_asset(repo_root / "f1tenth_racetracks", map_name)
        for map_name in sorted({row["map_name"] for row in selected})
    }
    track_lengths = {
        map_name: centerline_length(repo_root / "f1tenth_racetracks", map_name)
        for map_name in projectors
    }
    speed_cache = _initial_speed_cache(repo_root, projectors)

    metadata = []
    projection_max_error = 0.0
    target_counts = Counter()
    valid_counts = Counter()
    for episode_index, (row, source_info) in enumerate(zip(selected, inventory)):
        source = sources[row["l2_id"]]
        path = repo_root / source["npz_relpath"]
        start = int(source_info["frame_start"])
        count = int(source_info["frame_count"])
        stop = start + count
        with np.load(path, allow_pickle=False) as data:
            resolved_ego_idx = int(source["resolved_ego_idx"])
            speeds = speed_cache[row["map_name"]]
            initial_speed_input = float(speeds[resolved_ego_idx % len(speeds)] * 0.9)
            replay = replay_bc_features(
                bc,
                data["ego_lidar"],
                data["ego_actual_speed"],
                initial_speed_input,
                data["ego_desired_steer"],
                data["ego_desired_speed"],
                device,
            )
            labels = build_episode_labels(
                data,
                projectors[row["map_name"]],
                track_lengths[row["map_name"]],
                label_config,
            )
            projection_max_error = max(
                projection_max_error,
                _projection_error(projectors[row["map_name"]], data["ego_pose"]),
                _projection_error(projectors[row["map_name"]], data["opp_pose"]),
            )
            arrays["features"][start:stop] = replay.features
            arrays["time"][start:stop] = np.asarray(data["time"], dtype=np.float32)
            arrays["episode_index"][start:stop] = episode_index
            for name in ("closing_rate", "corridor_ttc", "rel_s", "lateral_gap", "ego_v_s", "opp_v_s"):
                arrays[name][start:stop] = labels[name]
            for prefix in ("ego", "any"):
                for kind in ("target", "valid"):
                    for token in HORIZON_TOKENS:
                        name = f"{prefix}_{kind}_{token}"
                        values = np.asarray(labels[name], dtype=np.uint8)
                        arrays[name][start:stop] = values
                        if kind == "target":
                            target_counts[name] += int(np.sum(values))
                        else:
                            valid_counts[name] += int(np.sum(values))
            ego_collision = bool(np.asarray(data["ego_collision"]).reshape(()))
            opp_collision = bool(np.asarray(data["opp_collision"]).reshape(()))
            metadata.append(
                {
                    "episode_index": str(episode_index),
                    "l2_id": row["l2_id"],
                    "l3_id": row["l3_id"],
                    "l4_id": row["l4_id"],
                    "map_name": row["map_name"],
                    "skill": row["skill"],
                    "opponent_raceline": row["opponent_raceline"],
                    "speedscale_hex": row["speedscale_hex"],
                    "outer_fold": row["outer_fold"],
                    "representative_l1_id": row["representative_l1_id"],
                    "resolved_ego_idx": str(resolved_ego_idx),
                    "npz_relpath": source["npz_relpath"],
                    "npz_sha256": source["npz_sha256"],
                    "frame_start": str(start),
                    "frame_count": str(count),
                    "final_time_hex": float(np.asarray(data["final_time"]).reshape(())).hex(),
                    "ego_collision": str(ego_collision),
                    "opp_collision": str(opp_collision),
                    "collision_any": str(ego_collision or opp_collision),
                }
            )
        if (episode_index + 1) % 25 == 0 or episode_index + 1 == len(selected):
            print(
                f"D2_EXTRACT episodes={episode_index + 1}/{len(selected)} "
                f"frames={stop}/{total_frames}",
                flush=True,
            )

    for array in arrays.values():
        array.flush()
    del arrays
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _write_tsv(partial / "episode_metadata.tsv", metadata, EPISODE_FIELDS)
    replay_validation = {
        "schema": "d2-replay-validation-1",
        "passed": True,
        "episode_count": len(selected),
        "frame_count": total_frames,
        "action_mismatched_frames": 0,
        "max_abs_steer_error": 0.0,
        "max_abs_speed_error": 0.0,
        "fast_exhaustive_projection_max_error": projection_max_error,
        "target_frame_counts": dict(sorted(target_counts.items())),
        "valid_frame_counts": dict(sorted(valid_counts.items())),
        "ego_collision_episode_count": sum(row["ego_collision"] == "True" for row in metadata),
        "any_collision_episode_count": sum(row["collision_any"] == "True" for row in metadata),
        "registry": {
            "required": len(registry_rows),
            "appended": append_result.appended,
            "already_present": append_result.skipped,
            "live_total": append_result.total,
            "snapshot_sha256": file_sha256(snapshot),
        },
    }
    _write_json(partial / "replay_validation.json", replay_validation)

    array_manifest = {}
    for name in ARRAY_DTYPES:
        path = partial / "arrays" / f"{name}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = [total_frames, 1680] if name == "features" else [total_frames]
        if list(array.shape) != expected_shape or array.dtype != np.dtype(ARRAY_DTYPES[name]):
            raise ValueError(f"emitted D2 array shape/dtype mismatch: {name}")
        array_manifest[name] = {
            "relpath": f"arrays/{name}.npy",
            "shape": expected_shape,
            "dtype": array.dtype.str,
            "sha256": file_sha256(path),
        }
    dataset_manifest = {
        "schema": "d2-dataset-manifest-1",
        "partition": "non_test",
        "episode_count": len(selected),
        "frame_count": total_frames,
        "feature_dim": 1680,
        "arrays": array_manifest,
        "episode_metadata_sha256": file_sha256(partial / "episode_metadata.tsv"),
        "source_inventory_sha256": file_sha256(partial / "source_inventory.tsv"),
        "registry_snapshot_sha256": file_sha256(snapshot),
    }
    _write_json(partial / "dataset_manifest.json", dataset_manifest)
    _write_output_manifest(partial)
    independent = validate_dataset_release(partial, allow_partial=True)
    if not independent["passed"]:
        raise AssertionError("independent D2 dataset validation failed")
    _promote(partial, output_dir)
    return independent


def validate_dataset_release(release_dir: str | Path, allow_partial: bool = False) -> dict:
    release_dir = Path(release_dir)
    if not allow_partial and not (release_dir / "COMPLETE").is_file():
        raise ValueError("D2 dataset release lacks COMPLETE")
    manifest_path = release_dir / "output_manifest.sha256"
    if not manifest_path.is_file():
        raise ValueError("D2 dataset output manifest missing")
    expected = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, relpath = line.split("  ", 1)
        if relpath in expected:
            raise ValueError("duplicate D2 output manifest path")
        expected[relpath] = digest
    actual = {
        path.relative_to(release_dir).as_posix()
        for path in release_dir.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    }
    if set(expected) != actual:
        raise ValueError("D2 dataset output inventory mismatch")
    for relpath, digest in expected.items():
        if file_sha256(release_dir / relpath) != digest:
            raise ValueError(f"D2 dataset output hash mismatch: {relpath}")
    dataset = json.loads((release_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    frames = int(dataset["frame_count"])
    for name, entry in dataset["arrays"].items():
        path = release_dir / entry["relpath"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"D2 array manifest hash mismatch: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != entry["shape"] or array.dtype.str != entry["dtype"]:
            raise ValueError(f"D2 array manifest shape/dtype mismatch: {name}")
        if array.shape[0] != frames:
            raise ValueError(f"D2 array frame count mismatch: {name}")
    episodes = _read_tsv(release_dir / "episode_metadata.tsv")
    if len(episodes) != int(dataset["episode_count"]):
        raise ValueError("D2 episode metadata count mismatch")
    if sum(int(row["frame_count"]) for row in episodes) != frames:
        raise ValueError("D2 episode frame accounting mismatch")
    validation = json.loads((release_dir / "replay_validation.json").read_text(encoding="utf-8"))
    if not validation.get("passed") or validation.get("action_mismatched_frames") != 0:
        raise ValueError("D2 replay validation did not pass")
    return {
        "passed": True,
        "episode_count": len(episodes),
        "frame_count": frames,
        "ego_collision_episode_count": validation["ego_collision_episode_count"],
        "any_collision_episode_count": validation["any_collision_episode_count"],
        "output_manifest_sha256": file_sha256(manifest_path),
        "dataset_manifest_sha256": file_sha256(release_dir / "dataset_manifest.json"),
    }


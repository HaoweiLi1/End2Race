"""Deployable non-test temporal signals aligned to the frozen-feature dataset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from d2.dataset import (
    _prepare_output,
    _promote,
    _read_tsv,
    _write_json,
    _write_output_manifest,
)
from d2.release import file_sha256


SIGNAL_DTYPES = {
    "ego_lidar": np.float32,
    "ego_actual_speed": np.float32,
    "previous_desired_steer": np.float32,
    "previous_desired_speed": np.float32,
}


def extract_deployable_signals(
    repo_root: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    registry_path: str | Path,
    created_at: str,
) -> dict:
    repo_root = Path(repo_root).resolve()
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    registry_path = Path(registry_path)
    if not (dataset_dir / "COMPLETE").is_file():
        raise ValueError("D2 non-test dataset lacks COMPLETE")
    dataset_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    episodes = _read_tsv(dataset_dir / "episode_metadata.tsv")
    total_frames = int(dataset_manifest["frame_count"])
    if len(episodes) != int(dataset_manifest["episode_count"]):
        raise ValueError("D2 signal episode count mismatch")

    partial = _prepare_output(output_dir)
    _write_json(
        partial / "config.json",
        {
            "schema": "d2-deployable-signals-config-1",
            "created_at": str(created_at),
            "partition": "non_test",
            "source_dataset_manifest_sha256": file_sha256(dataset_dir / "dataset_manifest.json"),
            "source_episode_metadata_sha256": file_sha256(dataset_dir / "episode_metadata.tsv"),
            "signals": sorted(SIGNAL_DTYPES),
            "previous_command_t0": "zero",
            "test_opened": False,
        },
    )
    snapshot = partial / "opened_registry.snapshot.tsv"
    shutil.copyfile(registry_path, snapshot)

    arrays_dir = partial / "arrays"
    arrays_dir.mkdir()
    arrays = {}
    for name, dtype in SIGNAL_DTYPES.items():
        shape = (total_frames, 360) if name == "ego_lidar" else (total_frames,)
        arrays[name] = np.lib.format.open_memmap(
            arrays_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )

    expected_start = 0
    for episode_index, episode in enumerate(episodes):
        start = int(episode["frame_start"])
        count = int(episode["frame_count"])
        stop = start + count
        if start != expected_start:
            raise ValueError("D2 signal frame accounting is not contiguous")
        path = repo_root / episode["npz_relpath"]
        if file_sha256(path) != episode["npz_sha256"]:
            raise ValueError(f"D2 signal source SHA mismatch: {episode['npz_relpath']}")
        with np.load(path, allow_pickle=False) as data:
            lidar = np.asarray(data["ego_lidar"], dtype=np.float32)
            actual_speed = np.asarray(data["ego_actual_speed"], dtype=np.float32)
            desired_steer = np.asarray(data["ego_desired_steer"], dtype=np.float32)
            desired_speed = np.asarray(data["ego_desired_speed"], dtype=np.float32)
            if lidar.shape != (count, 360) or any(
                value.shape != (count,) for value in (actual_speed, desired_steer, desired_speed)
            ):
                raise ValueError("D2 deployable signal source shape mismatch")
            arrays["ego_lidar"][start:stop] = lidar
            arrays["ego_actual_speed"][start:stop] = actual_speed
            previous_steer = np.empty(count, dtype=np.float32)
            previous_speed = np.empty(count, dtype=np.float32)
            previous_steer[0] = 0.0
            previous_speed[0] = 0.0
            previous_steer[1:] = desired_steer[:-1]
            previous_speed[1:] = desired_speed[:-1]
            arrays["previous_desired_steer"][start:stop] = previous_steer
            arrays["previous_desired_speed"][start:stop] = previous_speed
        expected_start = stop
        if (episode_index + 1) % 100 == 0 or episode_index + 1 == len(episodes):
            print(
                f"D2_SIGNALS episodes={episode_index + 1}/{len(episodes)} "
                f"frames={stop}/{total_frames}",
                flush=True,
            )
    if expected_start != total_frames:
        raise ValueError("D2 deployable signal total frame mismatch")
    for array in arrays.values():
        array.flush()
    del arrays

    signal_manifest = {}
    for name, dtype in SIGNAL_DTYPES.items():
        path = arrays_dir / f"{name}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = [total_frames, 360] if name == "ego_lidar" else [total_frames]
        if list(array.shape) != expected_shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"D2 deployable signal array mismatch: {name}")
        signal_manifest[name] = {
            "relpath": f"arrays/{name}.npy",
            "shape": expected_shape,
            "dtype": array.dtype.str,
            "sha256": file_sha256(path),
        }
    manifest = {
        "schema": "d2-deployable-signals-manifest-1",
        "episode_count": len(episodes),
        "frame_count": total_frames,
        "signals": signal_manifest,
        "registry_snapshot_sha256": file_sha256(snapshot),
    }
    _write_json(partial / "signals_manifest.json", manifest)
    _write_output_manifest(partial)
    independent = validate_signals_release(partial, allow_partial=True)
    _promote(partial, output_dir)
    return independent


def validate_signals_release(release_dir: str | Path, allow_partial: bool = False) -> dict:
    release_dir = Path(release_dir)
    if not allow_partial and not (release_dir / "COMPLETE").is_file():
        raise ValueError("D2 deployable signals release lacks COMPLETE")
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
        raise ValueError("D2 deployable signals inventory mismatch")
    for relpath, digest in expected.items():
        if file_sha256(release_dir / relpath) != digest:
            raise ValueError(f"D2 deployable signals hash mismatch: {relpath}")
    manifest = json.loads((release_dir / "signals_manifest.json").read_text(encoding="utf-8"))
    frames = int(manifest["frame_count"])
    for name, entry in manifest["signals"].items():
        path = release_dir / entry["relpath"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"D2 deployable signal manifest hash mismatch: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != entry["shape"] or array.dtype.str != entry["dtype"]:
            raise ValueError(f"D2 deployable signal shape/dtype mismatch: {name}")
        if len(array) != frames:
            raise ValueError(f"D2 deployable signal frame mismatch: {name}")
    return {
        "passed": True,
        "episode_count": int(manifest["episode_count"]),
        "frame_count": frames,
        "signals_manifest_sha256": file_sha256(release_dir / "signals_manifest.json"),
        "output_manifest_sha256": file_sha256(manifest_path),
    }


"""Run-scoped, atomic evaluation artifact handling."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evaluation.schema import EVALUATION_SCHEMA_VERSION, json_safe


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_directory(
    output_root: str | Path,
    suite_name: str,
    checkpoint_path: str | Path,
    checkpoint_sha: str,
    run_id: str,
) -> Path:
    model_stem = Path(checkpoint_path).stem
    return Path(output_root) / suite_name / f"{model_stem}__{checkpoint_sha[:12]}" / run_id


def _atomic_replace(path: Path, writer: Any, suffix: str = ".tmp") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)

    def write(temporary_path: Path) -> None:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    _atomic_replace(destination, write, suffix=".json.tmp")


def atomic_write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json_safe(value) for key, value in row.items()})
    encoded = buffer.getvalue()

    def write(temporary_path: Path) -> None:
        with temporary_path.open("w", encoding="utf-8", newline="") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    _atomic_replace(Path(path), write, suffix=".csv.tmp")


def validate_trace_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    if not arrays:
        raise ValueError("Trace must contain at least one array")
    lengths: set[int] = set()
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.ndim < 1:
            raise ValueError(f"Trace array {name!r} must have a leading time dimension")
        if array.dtype == object or not (
            np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
        ):
            raise TypeError(f"Trace array {name!r} must have a numeric or boolean dtype, got {array.dtype}")
        lengths.add(int(array.shape[0]))
    if len(lengths) != 1:
        raise ValueError(f"Trace arrays do not share one leading dimension: {sorted(lengths)}")
    return lengths.pop()


def atomic_write_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    validate_trace_arrays(arrays)
    numeric_arrays = {name: np.asarray(value) for name, value in arrays.items()}

    def write(temporary_path: Path) -> None:
        np.savez_compressed(temporary_path, **numeric_arrays)

    _atomic_replace(Path(path), write, suffix=".npz")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def initialize_run(
    run_dir: str | Path,
    manifest: Mapping[str, Any],
    scenario_manifest: Mapping[str, Any],
    *,
    resume: bool,
) -> Path:
    """Create one run or validate the immutable identity of a resumed run."""

    destination = Path(run_dir)
    manifest_path = destination / "manifest.json"
    scenarios_path = destination / "scenario_manifest.json"
    if destination.exists():
        if not resume:
            raise FileExistsError(f"Run directory already exists; pass --resume to reuse it: {destination}")
        if not manifest_path.is_file() or not scenarios_path.is_file():
            raise ValueError("Cannot resume a run without both manifest files")
        existing_manifest = load_json(manifest_path)
        existing_scenarios = load_json(scenarios_path)
        if existing_manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION:
            raise ValueError("Resume manifest schema version does not match")
        if existing_manifest.get("config") != json_safe(manifest.get("config")):
            raise ValueError("Resume configuration does not match the existing run")
        if existing_manifest.get("checkpoint") != json_safe(manifest.get("checkpoint")):
            raise ValueError("Resume checkpoint identity does not match the existing run")
        if existing_scenarios != json_safe(scenario_manifest):
            raise ValueError("Resume scenario manifest does not match the existing run")
        return destination

    destination.mkdir(parents=True, exist_ok=False)
    for child in ("episodes", "traces", "videos", "errors"):
        (destination / child).mkdir()
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(scenarios_path, scenario_manifest)
    return destination


_REQUIRED_EPISODE_FIELDS = {
    "schema_version",
    "scenario_id",
    "outcome",
    "ego_collision",
    "opponent_collision",
    "opponent_only_collision",
    "collision_step",
    "steps",
    "elapsed_time_s",
    "final_ego_progress_m",
    "final_opp_progress_m",
    "final_relative_progress_m",
    "ego_distance_m",
    "ego_mean_measured_speed_mps",
    "ego_speed_variance",
    "ego_min_measured_speed_mps",
    "ego_mean_desired_speed_mps",
    "ego_max_abs_steer_rad",
    "ego_max_steer_delta_rad",
    "ego_min_lidar_m",
    "trace_path",
    "video_path",
}


def _existing_run_artifact(episode_path: Path, relative_path: Any) -> bool:
    if not isinstance(relative_path, str) or not relative_path:
        return False
    run_dir = episode_path.parent.parent.resolve()
    artifact = (run_dir / relative_path).resolve()
    return artifact.is_relative_to(run_dir) and artifact.is_file()


def valid_episode_file(
    path: str | Path,
    scenario_id: str,
    *,
    trace_mode: str | None = None,
    require_video: bool = False,
) -> bool:
    episode_path = Path(path)
    try:
        episode = load_json(episode_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    basic_valid = bool(
        isinstance(episode, dict)
        and _REQUIRED_EPISODE_FIELDS.issubset(episode)
        and episode.get("schema_version") == EVALUATION_SCHEMA_VERSION
        and episode.get("scenario_id") == scenario_id
        and episode.get("outcome") in {"collision", "overtake", "follow"}
    )
    if not basic_valid:
        return False
    if trace_mode not in {None, "none", "collision", "all"}:
        raise ValueError(f"Unsupported trace mode for validation: {trace_mode}")
    trace_required = trace_mode == "all" or (
        trace_mode == "collision" and bool(episode.get("ego_collision"))
    )
    if trace_required and not _existing_run_artifact(episode_path, episode.get("trace_path")):
        return False
    if require_video and not _existing_run_artifact(episode_path, episode.get("video_path")):
        return False
    return True

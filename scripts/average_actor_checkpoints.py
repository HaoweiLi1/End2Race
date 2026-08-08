import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import End2Race


def parse_arguments():
    parser = argparse.ArgumentParser(description="Average compatible actor checkpoints")
    parser.add_argument("--source-paths", type=Path, nargs=4, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--hidden-scale", type=int, default=4)
    parser.add_argument("--evaluation-alias", type=str, default="CTv2_U42_U45_EQUAL_AVG")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state_dict(path):
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Actor checkpoint must contain a state dict: {path}")
    if len(state_dict) != 12:
        raise RuntimeError(f"Actor checkpoint must contain 12 keys, got {len(state_dict)}: {path}")
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Actor state value must be a tensor: {name}")
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"Actor state tensor must be finite: {path}:{name}")
    return state_dict


def average_state_dicts(state_dicts):
    if len(state_dicts) != 4:
        raise RuntimeError(f"Exactly four actor state dicts are required, got {len(state_dicts)}")
    names = list(state_dicts[0])
    for index, state_dict in enumerate(state_dicts[1:], start=1):
        if list(state_dict) != names:
            raise RuntimeError(f"Actor state keys or order differ at source index {index}")

    averaged = {}
    for name in names:
        values = [state_dict[name] for state_dict in state_dicts]
        reference = values[0]
        for index, value in enumerate(values[1:], start=1):
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise RuntimeError(f"Actor tensor contract differs at source index {index}: {name}")
        if torch.is_floating_point(reference):
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for value in values:
                accumulator.add_(value.to(dtype=torch.float64))
            result = (accumulator / 4.0).to(dtype=reference.dtype)
            if not bool(torch.isfinite(result).all().item()):
                raise RuntimeError(f"Averaged actor tensor is not finite: {name}")
            averaged[name] = result
        else:
            if any(not torch.equal(reference, value) for value in values[1:]):
                raise RuntimeError(f"Non-floating actor tensor differs across sources: {name}")
            averaged[name] = reference.clone()
    return averaged


def relative_l2_distance(left, right):
    difference_squared = 0.0
    reference_squared = 0.0
    for name in left:
        if not torch.is_floating_point(left[name]):
            continue
        difference = left[name].to(dtype=torch.float64) - right[name].to(dtype=torch.float64)
        difference_squared += float(torch.sum(difference * difference).item())
        reference = right[name].to(dtype=torch.float64)
        reference_squared += float(torch.sum(reference * reference).item())
    if reference_squared <= 0.0:
        raise RuntimeError("Reference actor has zero floating-point norm")
    return math.sqrt(difference_squared / reference_squared)


def atomic_torch_save(path, value):
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path, value):
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    args = parse_arguments()
    source_paths = [path.expanduser().resolve() for path in args.source_paths]
    output_path = args.output_path.expanduser().resolve()
    alias_path = output_path.parent / f"{args.evaluation_alias}.pth"
    manifest_path = output_path.parent / "model_manifest.json"

    if len(set(source_paths)) != 4:
        raise RuntimeError("The four source actor paths must be unique")
    if any(not path.is_file() for path in source_paths):
        missing = [str(path) for path in source_paths if not path.is_file()]
        raise RuntimeError(f"Source actor checkpoint is missing: {missing}")
    if output_path.exists() or alias_path.exists() or manifest_path.exists() or output_path.parent.exists():
        raise RuntimeError(f"Output directory must not already exist: {output_path.parent}")

    output_path.parent.mkdir(parents=True)
    try:
        source_state_dicts = [load_state_dict(path) for path in source_paths]
        averaged_state_dict = average_state_dicts(source_state_dicts)
        atomic_torch_save(output_path, averaged_state_dict)
        os.link(output_path, alias_path)

        loaded_average = load_state_dict(output_path)
        model = End2Race(hidden_scale=args.hidden_scale)
        model.load_state_dict(loaded_average, strict=True)
        source_sha256 = [sha256_file(path) for path in source_paths]
        relative_distances = [relative_l2_distance(loaded_average, state_dict) for state_dict in source_state_dicts]
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
        manifest = {
            "schema_version": 1,
            "method": "equal_weight_checkpoint_average",
            "source_count": 4,
            "sources": [
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": digest,
                    "weight": 0.25,
                }
                for path, digest in zip(source_paths, source_sha256)
            ],
            "algorithm": "Each floating tensor is accumulated in source order as float64, divided by 4, and cast to its original dtype. Every non-floating tensor must be exactly equal across sources.",
            "output_path": str(output_path.relative_to(PROJECT_ROOT)),
            "output_sha256": sha256_file(output_path),
            "evaluation_alias_path": str(alias_path.relative_to(PROJECT_ROOT)),
            "evaluation_alias_sha256": sha256_file(alias_path),
            "evaluation_alias_hardlink": output_path.stat().st_ino == alias_path.stat().st_ino,
            "state_key_count": len(loaded_average),
            "strict_load": True,
            "finite": True,
            "hidden_scale": args.hidden_scale,
            "relative_l2_to_sources": relative_distances,
            "git_commit": commit,
            "worktree_status": status,
        }
        atomic_write_json(manifest_path, manifest)
    except Exception:
        alias_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        try:
            output_path.parent.rmdir()
        except OSError:
            pass
        raise

    print(json.dumps(manifest, indent=2, allow_nan=False))

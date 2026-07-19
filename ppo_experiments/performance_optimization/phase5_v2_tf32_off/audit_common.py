"""Shared, experiment-only helpers for the Phase 5-v2 audit."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import subprocess
import sys
from typing import Any, Iterator

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent
CURRENT_HEAD = "e1c0d2b61e4ebc5c619f4c013dad330acf1fdfa0"
CONFIG_NAME = "N1-H1F-p50"
SEED = 20260917
WORKER_COUNT = 6
BC_SHA256 = "b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4"
MANIFEST_SHA256 = "ad35a6d56dddfe7c5e0460877f3aeb41ecd428d83c61ca1d1dc82c2b8709b0b8"
SOURCE_HASHES = {
    "model.py": "0c34a5b4f26d3dfa158a9c187b911e209504cbb4cfea5ed41b0f283e09134eb3",
    "ppo/policy.py": "a78e3a551e4e77af5d73382c5c05aeb327410628cc478419c26da09224b8ad72",
    "ppo/buffer.py": "6fce9e10cff431f9cfdda318526d05785266c353b6637b3a78290806b185c0da",
    "ppo/environment.py": "c55fa36f20a234f01f037f29176e0d888e9c423fe0883b28dfabcee7c2dff7ec",
    "ppo/vec_env.py": "b91366af88de80e17def840356098c4ae7d318409dc43135547d6e0b4b4e8855",
    "ppo/config.py": "7125b62c99ec9fd7b03092e73edfac55f3a88a92fb84e17ea78139f0ab8d8d68",
    "train_ppo.py": "676afc8cdec903ae1d51b22556b4d3c192e77499d13a710be1127721f5db7637",
    "pretrained/end2race.pth": BC_SHA256,
    "ppo/hard_pools/h1_expanded_det.json": MANIFEST_SHA256,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def tensor_hash(value: torch.Tensor | np.ndarray) -> str:
    array = value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def state_dict_hash(state: dict[str, Any]) -> str:
    records = []
    for name in sorted(state):
        value = state[name]
        if torch.is_tensor(value):
            records.append((name, tensor_hash(value)))
        else:
            records.append((name, normalize(value)))
    return canonical_hash(records)


def normalize(value: Any) -> Any:
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return {"tensor_sha256": tensor_hash(value)}
    if isinstance(value, dict):
        return [[repr(key), normalize(item)] for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))]
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def object_hash(value: Any) -> str:
    return canonical_hash(normalize(value))


def source_hashes() -> dict[str, str]:
    return {relative: sha256_file(PROJECT_ROOT / relative) for relative in SOURCE_HASHES}


def assert_locked_sources() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    if head != CURRENT_HEAD:
        raise RuntimeError(f"HEAD changed: expected {CURRENT_HEAD}, got {head}")
    actual = source_hashes()
    differences = {name: {"expected": SOURCE_HASHES[name], "actual": actual[name]} for name in actual if actual[name] != SOURCE_HASHES[name]}
    if differences:
        raise RuntimeError(f"locked source hash changed: {differences}")


def runtime_record() -> dict[str, Any]:
    import stable_baselines3
    import sb3_contrib

    return {
        "python_executable": sys.executable,
        "python": sys.version,
        "numpy": np.__version__,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "sb3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }


def flags_record() -> dict[str, Any]:
    return {
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


@contextmanager
def backend_flags(tf32_off: bool) -> Iterator[dict[str, Any]]:
    original = flags_record()
    try:
        if tf32_off:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.benchmark = False
        active = flags_record()
        yield active
    finally:
        torch.backends.cudnn.allow_tf32 = original["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = original["cuda_matmul_allow_tf32"]
        torch.set_float32_matmul_precision(original["float32_matmul_precision"])
        torch.backends.cudnn.benchmark = original["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = original["cudnn_deterministic"]
        if flags_record() != original:
            raise RuntimeError("backend flags were not restored exactly")


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_hashes(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": sha256_bytes(pickle.dumps(state["python"], protocol=5)),
        "numpy": sha256_bytes(pickle.dumps(state["numpy"], protocol=5)),
        "torch_cpu": tensor_hash(state["torch_cpu"]),
        "torch_cuda": [tensor_hash(item) for item in state["torch_cuda"]],
    }


def diff_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().double().cpu().reshape(-1)
    cand = candidate.detach().double().cpu().reshape(-1)
    difference = (cand - ref).abs()
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(cand).all())
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    diff_norm = float(torch.linalg.vector_norm(cand - ref).item())
    denominator = max(ref_norm, torch.finfo(torch.float64).tiny)
    if ref.numel() == 0:
        cosine = 1.0
    elif ref_norm == 0.0 and float(torch.linalg.vector_norm(cand).item()) == 0.0:
        cosine = 1.0
    elif ref_norm == 0.0 or float(torch.linalg.vector_norm(cand).item()) == 0.0:
        cosine = 0.0
    else:
        cosine = float(torch.nn.functional.cosine_similarity(ref, cand, dim=0).item())
    return {
        "count": int(ref.numel()),
        "finite": finite,
        "mean_abs": float(difference.mean().item()) if difference.numel() else 0.0,
        "p50_abs": float(torch.quantile(difference, 0.50).item()) if difference.numel() else 0.0,
        "p95_abs": float(torch.quantile(difference, 0.95).item()) if difference.numel() else 0.0,
        "p99_abs": float(torch.quantile(difference, 0.99).item()) if difference.numel() else 0.0,
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "relative_l2": diff_norm / denominator,
        "cosine": cosine,
    }


def flatten_mapping(mapping: dict[str, torch.Tensor], prefixes: tuple[str, ...] | None = None) -> torch.Tensor:
    names = [name for name in sorted(mapping) if prefixes is None or name.startswith(prefixes)]
    return torch.cat([mapping[name].detach().reshape(-1) for name in names])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def provenance(backend: str, batch: Any, flags: dict[str, Any], rollout_hash: str | None = None) -> dict[str, Any]:
    return {
        "git_head": CURRENT_HEAD,
        "worktree": "production tracked files clean; experiment artifacts and pre-existing patch untracked",
        "runtime": runtime_record(),
        "flags": flags,
        "source_hashes": source_hashes(),
        "bc_sha256": BC_SHA256,
        "hard_pool_manifest_sha256": MANIFEST_SHA256,
        "config": CONFIG_NAME,
        "seed": SEED,
        "backend": backend,
        "batch_or_microbatch": batch,
        "rollout_hash": rollout_hash,
    }


def finite_number(value: float) -> bool:
    return math.isfinite(float(value))

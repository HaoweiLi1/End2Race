"""Canonical hashing and immutable experiment constants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_COMMIT = "4fac86858802353e5b0892ff9d3c874bc15d781b"
BC_RELATIVE_PATH = "pretrained/end2race.pth"
BC_SHA256 = "b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4"
GUIDE_RELATIVE_PATH = ".agents/PPO_V1_2_EXPERIMENT_GUIDE.md"
EVALUATION_BASELINE = {"ego_collision": 21, "follow": 233, "overtake": 346, "error": 0, "total": 600}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def austin_asset_hashes() -> dict[str, str]:
    directory = PROJECT_ROOT / "f1tenth_racetracks" / "Austin"
    names = ("Austin_map.png", "Austin_map.yaml", "raceline0.csv", "raceline1.csv", "raceline2.csv")
    return {str((directory / name).relative_to(PROJECT_ROOT)): file_sha256(directory / name) for name in names}

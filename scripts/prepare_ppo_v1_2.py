#!/usr/bin/env python3
"""Generate and verify the frozen 125-arm PPO V1.2 sweep manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ppo_v1_2.experiment_spec import BC_SHA256, PROJECT_ROOT, austin_asset_hashes, canonical_hash, file_sha256
from experiments.ppo_v1_2.registry import build_manifest, validate_manifest
from utils import atomic_write_json


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("configs/ppo_v1_2/sweep_manifest.json"))
    parser.add_argument("--hard-pool-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_pool_hashes(root: Path | None) -> dict[str, str | None]:
    from experiments.ppo_v1_2.config_schema import HARD_POOL_IDS

    hashes: dict[str, str | None] = {pool_id: None for pool_id in HARD_POOL_IDS}
    if root is None:
        return hashes
    for pool_id in HARD_POOL_IDS:
        path = root / "pools" / f"{pool_id}.json"
        if path.is_file():
            document = json.loads(path.read_text(encoding="utf-8"))
            hashes[pool_id] = str(document["manifest_hash"])
    return hashes


def main() -> int:
    args = parse_arguments()
    bc_actual = file_sha256(PROJECT_ROOT / "pretrained" / "end2race.pth")
    if bc_actual != BC_SHA256:
        raise RuntimeError(f"Canonical BC hash drift: {bc_actual}")
    pool_hashes = load_pool_hashes(args.hard_pool_root)
    manifest = build_manifest(experiment_head=git_head(), hard_pool_hashes=pool_hashes)
    manifest["austin_asset_hashes"] = austin_asset_hashes()
    manifest["manifest_hash"] = canonical_hash({key: value for key, value in manifest.items() if key not in {"generated_at", "manifest_hash"}})
    validate_manifest(manifest)
    output = (PROJECT_ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    atomic_write_json(output, manifest)
    print(json.dumps({"output": str(output), "training_arm_count": 125, "manifest_hash": manifest["manifest_hash"], "dry_run": args.dry_run}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

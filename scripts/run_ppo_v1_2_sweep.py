#!/usr/bin/env python3
"""CLI for the PPO V1.2 unattended stage-barrier runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ppo_v1_2.runner import SweepRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("runs/ppo_v1_2/sweep_manifest.runtime.json"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/ppo_v1_2"))
    parser.add_argument("--hard-pool-root", type=Path, default=Path("runs/ppo_v1_2/hard_pools"))
    parser.add_argument("--bc-outcomes", type=Path, default=Path("runs/ppo_v1/v1_1_pilot_20_updates/train_bc_outcomes.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    runner = SweepRunner(args.manifest, args.run_root, args.hard_pool_root, args.bc_outcomes, dry_run=args.dry_run)
    result = runner.run()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

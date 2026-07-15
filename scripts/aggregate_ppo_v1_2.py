#!/usr/bin/env python3
"""Rebuild PPO V1.2 global tables from lower-level run results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ppo_v1_2.aggregate import global_aggregate
from experiments.ppo_v1_2.config_schema import STAGES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/ppo_v1_2"))
    args = parser.parse_args()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.run_root.glob("*/**/attempt_*/run_result.json"))]
    latest = {row["arm_id"]: row for row in results}
    selections = {
        stage: json.loads((args.run_root / stage / "stage_selection.json").read_text(encoding="utf-8"))
        for stage in STAGES
        if (args.run_root / stage / "stage_selection.json").is_file()
    }
    completion = global_aggregate(args.run_root, list(latest.values()), selections)
    print(json.dumps(completion, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

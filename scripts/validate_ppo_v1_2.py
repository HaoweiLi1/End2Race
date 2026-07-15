#!/usr/bin/env python3
"""Validate a PPO V1.2 manifest and any completed run results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ppo_v1_2.registry import validate_manifest
from experiments.ppo_v1_2.result_schema import validate_run_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    by_id = {arm["arm_id"]: arm for arm in manifest["arms"]}
    count = 0
    if args.run_root is not None:
        for path in sorted(args.run_root.glob("*/**/attempt_*/run_result.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            validate_run_result(result, by_id[result["arm_id"]])
            count += 1
    print(json.dumps({"manifest": "PASS", "training_arm_count": 125, "validated_run_results": count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

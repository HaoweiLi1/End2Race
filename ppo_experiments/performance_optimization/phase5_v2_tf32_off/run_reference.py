#!/usr/bin/env python3
"""Regenerate current-HEAD contract/profile references without production edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for path in (PROJECT_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_common import (
    CONFIG_NAME,
    CURRENT_HEAD,
    SEED,
    WORKER_COUNT,
    assert_locked_sources,
    backend_flags,
    provenance,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("contract", "profile"))
    parser.add_argument("--tf32-off", action="store_true")
    parser.add_argument("--zero-lr", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-output", type=Path)
    args = parser.parse_args()
    assert_locked_sources()
    with backend_flags(args.tf32_off) as flags:
        if args.kind == "contract":
            from ppo_experiments.performance_optimization.validate_pipeline import capture

            result = capture(
                CONFIG_NAME,
                SEED,
                "central_subproc",
                WORKER_COUNT,
                args.zero_lr,
                args.label,
                args.action_output,
            )
        else:
            if args.zero_lr or args.action_output is not None:
                raise ValueError("profile does not accept zero-LR or action output")
            from ppo_experiments.performance_optimization.profile_pipeline import profile_one

            result = profile_one(CONFIG_NAME, SEED, args.label, "central_subproc", WORKER_COUNT)
        result["phase5_v2_provenance"] = provenance(
            args.label,
            1,
            flags,
            result.get("frozen_rollout_sha256", {}).get("observations") if args.kind == "contract" else None,
        )
        result["phase5_v2_provenance"]["model_initial_hash"] = None
        result["phase5_v2_provenance"]["optimizer_initial_hash"] = result.get("optimizer_state", {}).get("initial_sha256")
        result["phase5_v2_provenance"]["rng_initial_hash"] = None
        result["phase5_v2_provenance"]["checkpoint_hash"] = result.get("checkpoint", {}).get("sha256")
        result["phase5_v2_provenance"]["numerical_metrics"] = result.get("replay_identity", {})
        result["phase5_v2_provenance"]["timing_metrics"] = {
            key: result[key]
            for key in ("rollout_s", "ppo_train_s", "total_update_s")
            if key in result
        }
    assert_locked_sources()
    if result.get("head", CURRENT_HEAD) != CURRENT_HEAD:
        raise RuntimeError("regenerated artifact did not record the current HEAD")
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "head": CURRENT_HEAD, "label": args.label}, sort_keys=True))


if __name__ == "__main__":
    main()

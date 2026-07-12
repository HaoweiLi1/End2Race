#!/usr/bin/env python3
"""Command-line entry point for the locked D2.5 oracle."""

from __future__ import annotations

import argparse
import json

from d25.search import EVIDENCE_RELPATH, REGISTRY_OPENED_AT, run_oracle
from d25.validate import validate_release


def parse_args():
    parser = argparse.ArgumentParser(description="D2.5 counterfactual recoverability oracle")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--mode", choices=("baseline_smoke", "branch_smoke", "full"), required=True)
    run.add_argument("--repo-root", default=".")
    run.add_argument("--dataset-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--registry", required=True)
    run.add_argument("--created-at", required=True)
    run.add_argument("--registry-opened-at", default=REGISTRY_OPENED_AT)
    run.add_argument("--evidence-relpath", default=EVIDENCE_RELPATH)
    run.add_argument("--device", default="cuda:0")

    validate = subparsers.add_parser("validate")
    validate.add_argument("release_dir")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "run":
        result = run_oracle(
            repo_root=args.repo_root,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            registry_path=args.registry,
            mode=args.mode,
            created_at=args.created_at,
            registry_opened_at=args.registry_opened_at,
            evidence_relpath=args.evidence_relpath,
            device_name=args.device,
        )
    else:
        result = validate_release(args.release_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

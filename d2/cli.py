#!/usr/bin/env python3
"""Stage-explicit command line interface for the D2 probe."""

from __future__ import annotations

import argparse
import json

from d2.release import create_split_release, validate_split_release
from d2.dataset import extract_non_test_dataset, validate_dataset_release
from d2.probe import run_family_oof, validate_probe_release
from d2.signals import extract_deployable_signals, validate_signals_release
from d2.summary import create_d2_summary


def parse_args():
    parser = argparse.ArgumentParser(description="End2Race D2 representation probe")
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("split", help="Create the outcome-blind D2 split lock")
    split.add_argument("--d0-dir", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--source-relpath", required=True)
    split.add_argument("--created-at", required=True)

    validate = sub.add_parser("validate-split", help="Independently validate a D2 split lock")
    validate.add_argument("--d0-dir", required=True)
    validate.add_argument("--release-dir", required=True)

    extract = sub.add_parser("extract-non-test", help="Registry-gated frozen BC feature extraction")
    extract.add_argument("--repo-root", required=True)
    extract.add_argument("--split-dir", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--registry", required=True)
    extract.add_argument("--created-at", required=True)
    extract.add_argument("--registry-opened-at", required=True)
    extract.add_argument("--evidence-relpath", required=True)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--max-episodes-per-map", type=int)

    validate_dataset = sub.add_parser("validate-dataset", help="Independently validate a D2 dataset")
    validate_dataset.add_argument("--release-dir", required=True)

    probe = sub.add_parser("probe-oof", help="Run locked grouped nested-CV for one probe family")
    probe.add_argument("--dataset-dir", required=True)
    probe.add_argument("--split-dir", required=True)
    probe.add_argument("--output-dir", required=True)
    probe.add_argument(
        "--family",
        choices=("linear", "mlp", "temporal", "temporal_deployable"),
        required=True,
    )
    probe.add_argument("--created-at", required=True)
    probe.add_argument("--device", default="cuda")
    probe.add_argument("--seed", type=int, default=20260711)
    probe.add_argument("--signals-dir")

    validate_probe = sub.add_parser("validate-probe", help="Independently validate a D2 OOF probe")
    validate_probe.add_argument("--release-dir", required=True)

    signals = sub.add_parser("extract-signals", help="Extract deployable non-test temporal signals")
    signals.add_argument("--repo-root", required=True)
    signals.add_argument("--dataset-dir", required=True)
    signals.add_argument("--output-dir", required=True)
    signals.add_argument("--registry", required=True)
    signals.add_argument("--created-at", required=True)

    validate_signals = sub.add_parser("validate-signals", help="Validate deployable temporal signals")
    validate_signals.add_argument("--release-dir", required=True)

    summary = sub.add_parser("summarize", help="Create the unopened-test D2 evidence synthesis")
    summary.add_argument("--dataset-dir", required=True)
    summary.add_argument("--split-dir", required=True)
    summary.add_argument("--output-dir", required=True)
    summary.add_argument("--linear-dir", required=True)
    summary.add_argument("--mlp-dir", required=True)
    summary.add_argument("--temporal-dir", required=True)
    summary.add_argument("--temporal-deployable-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "split":
        result = create_split_release(
            args.d0_dir,
            args.output_dir,
            source_relpath=args.source_relpath,
            created_at=args.created_at,
        )
    elif args.command == "validate-split":
        result = validate_split_release(args.d0_dir, args.release_dir)
    elif args.command == "extract-non-test":
        result = extract_non_test_dataset(
            repo_root=args.repo_root,
            split_dir=args.split_dir,
            output_dir=args.output_dir,
            registry_path=args.registry,
            created_at=args.created_at,
            registry_opened_at=args.registry_opened_at,
            evidence_relpath=args.evidence_relpath,
            device_name=args.device,
            max_episodes_per_map=args.max_episodes_per_map,
        )
    elif args.command == "validate-dataset":
        result = validate_dataset_release(args.release_dir)
    elif args.command == "probe-oof":
        result = run_family_oof(
            dataset_dir=args.dataset_dir,
            split_dir=args.split_dir,
            output_dir=args.output_dir,
            family=args.family,
            created_at=args.created_at,
            device_name=args.device,
            seed=args.seed,
            signals_dir=args.signals_dir,
        )
    elif args.command == "validate-probe":
        result = validate_probe_release(args.release_dir)
    elif args.command == "extract-signals":
        result = extract_deployable_signals(
            repo_root=args.repo_root,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            registry_path=args.registry,
            created_at=args.created_at,
        )
    elif args.command == "validate-signals":
        result = validate_signals_release(args.release_dir)
    else:
        result = create_d2_summary(
            dataset_dir=args.dataset_dir,
            split_dir=args.split_dir,
            output_dir=args.output_dir,
            family_dirs={
                "linear": args.linear_dir,
                "mlp": args.mlp_dir,
                "temporal": args.temporal_dir,
                "temporal_deployable": args.temporal_deployable_dir,
            },
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

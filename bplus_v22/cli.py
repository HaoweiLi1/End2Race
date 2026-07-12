#!/usr/bin/env python3
"""Audited B+ v2.2 preflight, initialization, and validation CLI."""

from __future__ import annotations

import argparse
import json

from bplus_v22.release import create_source_preflight, validate_source_preflight
from bplus_v22.identity import run_zero_identity, validate_zero_identity
from bplus_v22.checkpoint_preflight import (
    run_checkpoint_preflight,
    validate_checkpoint_preflight,
)
from bplus_v22.manifests import create_manifest_release, validate_manifest_release
from bplus_v22.closed_loop import run_closed_loop_warmstart, validate_closed_loop_release
from bplus_v22.sidecar import (
    create_registry_plan,
    run_sidecar_initialization,
    validate_registry_plan,
    validate_sidecar_release,
)
from bplus_v22.warmstart import (
    create_warmstart_manifest,
    run_warmstart_smoke,
    validate_warmstart_manifest,
    validate_warmstart_release,
)
from bplus_v22.hierarchical_identity import (
    run_hierarchical_identity,
    validate_hierarchical_identity,
)
from bplus_v22.hierarchical_warmstart import (
    create_hierarchical_warmstart_manifest,
    run_hierarchical_warmstart,
    validate_hierarchical_warmstart_manifest,
    validate_hierarchical_warmstart_release,
)
from bplus_v22.hierarchical_checkpoint_preflight import (
    run_hierarchical_checkpoint_preflight,
    validate_hierarchical_checkpoint_preflight,
)
from bplus_v22.hierarchical_closed_loop import (
    run_hierarchical_closed_loop,
    validate_hierarchical_closed_loop,
)


def parse_args():
    parser = argparse.ArgumentParser(description="B+ v2.2 structural preflight")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("source-preflight")
    create.add_argument("--output-dir", required=True)
    create.add_argument("--created-at", required=True)
    create.add_argument("--repo-root", default=".")
    validate = sub.add_parser("validate-source-preflight")
    validate.add_argument("release_dir")
    validate.add_argument("--repo-root", default=".")
    identity = sub.add_parser("zero-identity")
    identity.add_argument("--repo-root", default=".")
    identity.add_argument("--source-preflight-dir", required=True)
    identity.add_argument("--output-dir", required=True)
    identity.add_argument("--created-at", required=True)
    identity.add_argument("--device", default="cuda:0")
    identity.add_argument("--sidecar-release-dir")
    validate_identity = sub.add_parser("validate-zero-identity")
    validate_identity.add_argument("release_dir")
    registry_plan = sub.add_parser("sidecar-registry-plan")
    registry_plan.add_argument("--repo-root", default=".")
    registry_plan.add_argument("--source-preflight-dir", required=True)
    registry_plan.add_argument("--output-dir", required=True)
    registry_plan.add_argument("--created-at", required=True)
    validate_registry = sub.add_parser("validate-sidecar-registry-plan")
    validate_registry.add_argument("release_dir")
    validate_registry.add_argument("--repo-root", default=".")
    validate_registry.add_argument("--check-live", action="store_true")
    sidecar = sub.add_parser("sidecar-fit")
    sidecar.add_argument("--repo-root", default=".")
    sidecar.add_argument("--source-preflight-dir", required=True)
    sidecar.add_argument("--registry-plan-dir", required=True)
    sidecar.add_argument("--output-dir", required=True)
    sidecar.add_argument("--created-at", required=True)
    sidecar.add_argument("--device", default="cuda:0")
    validate_sidecar = sub.add_parser("validate-sidecar")
    validate_sidecar.add_argument("release_dir")
    validate_sidecar.add_argument("--repo-root", default=".")
    validate_sidecar.add_argument("--dataset-dir")
    validate_sidecar.add_argument("--split-dir")
    validate_sidecar.add_argument("--signals-dir")
    validate_sidecar.add_argument("--device")
    validate_sidecar.add_argument("--require-live-registry", action="store_true")
    warmstart_manifest = sub.add_parser("warmstart-manifest")
    warmstart_manifest.add_argument("--repo-root", default=".")
    warmstart_manifest.add_argument("--source-preflight-dir", required=True)
    warmstart_manifest.add_argument("--output-dir", required=True)
    warmstart_manifest.add_argument("--created-at", required=True)
    validate_manifest = sub.add_parser("validate-warmstart-manifest")
    validate_manifest.add_argument("release_dir")
    validate_manifest.add_argument("--repo-root", default=".")
    validate_manifest.add_argument("--check-live-registry", action="store_true")
    warmstart = sub.add_parser("warmstart-smoke")
    warmstart.add_argument("--repo-root", default=".")
    warmstart.add_argument("--source-preflight-dir", required=True)
    warmstart.add_argument("--manifest-dir", required=True)
    warmstart.add_argument("--output-dir", required=True)
    warmstart.add_argument("--created-at", required=True)
    warmstart.add_argument("--device", default="cuda:0")
    validate_warmstart = sub.add_parser("validate-warmstart")
    validate_warmstart.add_argument("release_dir")
    validate_warmstart.add_argument("--repo-root", default=".")
    validate_warmstart.add_argument("--device")
    validate_warmstart.add_argument("--require-live-registry", action="store_true")
    manifests = sub.add_parser("task8-manifests")
    manifests.add_argument("--repo-root", default=".")
    manifests.add_argument("--source-preflight-dir", required=True)
    manifests.add_argument("--output-dir", required=True)
    manifests.add_argument("--created-at", required=True)
    validate_manifests = sub.add_parser("validate-task8-manifests")
    validate_manifests.add_argument("release_dir")
    validate_manifests.add_argument("--repo-root", default=".")
    task9 = sub.add_parser("task9-checkpoint-preflight")
    task9.add_argument("--repo-root", default=".")
    task9.add_argument("--source-preflight-dir", required=True)
    task9.add_argument("--manifest-release-dir", required=True)
    task9.add_argument("--output-dir", required=True)
    task9.add_argument("--created-at", required=True)
    task9.add_argument("--device", default="cuda:0")
    validate_task9 = sub.add_parser("validate-task9-checkpoint-preflight")
    validate_task9.add_argument("release_dir")
    task10 = sub.add_parser("task10-warmstart")
    task10.add_argument("--repo-root", default=".")
    task10.add_argument("--source-preflight-dir", required=True)
    task10.add_argument("--manifest-release-dir", required=True)
    task10.add_argument("--output-dir", required=True)
    task10.add_argument("--created-at", required=True)
    task10.add_argument("--device", default="cuda:0")
    validate_task10 = sub.add_parser("validate-task10-warmstart")
    validate_task10.add_argument("release_dir")

    hierarchical_identity = sub.add_parser("hierarchical-zero-identity")
    hierarchical_identity.add_argument("--repo-root", default=".")
    hierarchical_identity.add_argument("--source-preflight-dir", required=True)
    hierarchical_identity.add_argument("--sidecar-release-dir", required=True)
    hierarchical_identity.add_argument("--output-dir", required=True)
    hierarchical_identity.add_argument("--created-at", required=True)
    hierarchical_identity.add_argument("--device", default="cuda:0")
    validate_hierarchical_identity_parser = sub.add_parser(
        "validate-hierarchical-zero-identity"
    )
    validate_hierarchical_identity_parser.add_argument("release_dir")
    validate_hierarchical_identity_parser.add_argument("--repo-root", default=".")

    hierarchical_manifest = sub.add_parser("hierarchical-warmstart-manifest")
    hierarchical_manifest.add_argument("--repo-root", default=".")
    hierarchical_manifest.add_argument("--source-preflight-dir", required=True)
    hierarchical_manifest.add_argument(
        "--hierarchical-identity-release-dir", required=True
    )
    hierarchical_manifest.add_argument("--output-dir", required=True)
    hierarchical_manifest.add_argument("--created-at", required=True)
    validate_hierarchical_manifest = sub.add_parser(
        "validate-hierarchical-warmstart-manifest"
    )
    validate_hierarchical_manifest.add_argument("release_dir")
    validate_hierarchical_manifest.add_argument("--repo-root", default=".")

    hierarchical_warmstart = sub.add_parser("hierarchical-warmstart")
    hierarchical_warmstart.add_argument("--repo-root", default=".")
    hierarchical_warmstart.add_argument("--source-preflight-dir", required=True)
    hierarchical_warmstart.add_argument(
        "--hierarchical-identity-release-dir", required=True
    )
    hierarchical_warmstart.add_argument("--manifest-dir", required=True)
    hierarchical_warmstart.add_argument("--output-dir", required=True)
    hierarchical_warmstart.add_argument("--created-at", required=True)
    hierarchical_warmstart.add_argument("--device", default="cuda:0")
    validate_hierarchical_warmstart = sub.add_parser(
        "validate-hierarchical-warmstart"
    )
    validate_hierarchical_warmstart.add_argument("release_dir")
    validate_hierarchical_warmstart.add_argument("--repo-root", default=".")
    validate_hierarchical_warmstart.add_argument("--device")

    hierarchical_task9 = sub.add_parser("hierarchical-task9")
    hierarchical_task9.add_argument("--repo-root", default=".")
    hierarchical_task9.add_argument("--source-preflight-dir", required=True)
    hierarchical_task9.add_argument("--manifest-release-dir", required=True)
    hierarchical_task9.add_argument("--warmstart-release-dir", required=True)
    hierarchical_task9.add_argument(
        "--warmstart-output-manifest-sha256", required=True
    )
    hierarchical_task9.add_argument("--output-dir", required=True)
    hierarchical_task9.add_argument("--created-at", required=True)
    hierarchical_task9.add_argument("--device", default="cuda:0")
    validate_hierarchical_task9 = sub.add_parser("validate-hierarchical-task9")
    validate_hierarchical_task9.add_argument("release_dir")

    hierarchical_task10 = sub.add_parser("hierarchical-task10")
    hierarchical_task10.add_argument("--repo-root", default=".")
    hierarchical_task10.add_argument("--source-preflight-dir", required=True)
    hierarchical_task10.add_argument("--manifest-release-dir", required=True)
    hierarchical_task10.add_argument("--warmstart-release-dir", required=True)
    hierarchical_task10.add_argument(
        "--warmstart-output-manifest-sha256", required=True
    )
    hierarchical_task10.add_argument("--output-dir", required=True)
    hierarchical_task10.add_argument("--created-at", required=True)
    hierarchical_task10.add_argument("--device", default="cuda:0")
    validate_hierarchical_task10 = sub.add_parser("validate-hierarchical-task10")
    validate_hierarchical_task10.add_argument("release_dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "source-preflight":
        result = create_source_preflight(args.output_dir, args.created_at, args.repo_root)
    elif args.command == "validate-source-preflight":
        result = validate_source_preflight(args.release_dir, args.repo_root)
    elif args.command == "zero-identity":
        result = run_zero_identity(
            args.repo_root,
            args.source_preflight_dir,
            args.output_dir,
            args.created_at,
            args.device,
            args.sidecar_release_dir,
        )
    elif args.command == "sidecar-registry-plan":
        result = create_registry_plan(
            args.repo_root,
            args.source_preflight_dir,
            args.output_dir,
            args.created_at,
        )
    elif args.command == "validate-sidecar-registry-plan":
        result = validate_registry_plan(
            args.release_dir, args.repo_root, check_live=args.check_live
        )
    elif args.command == "sidecar-fit":
        result = run_sidecar_initialization(
            args.repo_root,
            args.source_preflight_dir,
            args.registry_plan_dir,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-sidecar":
        result = validate_sidecar_release(
            args.release_dir,
            args.repo_root,
            dataset_dir=args.dataset_dir,
            split_dir=args.split_dir,
            signals_dir=args.signals_dir,
            device_name=args.device,
            require_live_registry=args.require_live_registry,
        )
    elif args.command == "warmstart-manifest":
        result = create_warmstart_manifest(
            args.repo_root,
            args.source_preflight_dir,
            args.output_dir,
            args.created_at,
        )
    elif args.command == "validate-warmstart-manifest":
        result = validate_warmstart_manifest(
            args.release_dir,
            args.repo_root,
            check_live_registry=args.check_live_registry,
        )
    elif args.command == "warmstart-smoke":
        result = run_warmstart_smoke(
            args.repo_root,
            args.source_preflight_dir,
            args.manifest_dir,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-warmstart":
        result = validate_warmstart_release(
            args.release_dir,
            args.repo_root,
            device_name=args.device,
            require_live_registry=args.require_live_registry,
        )
    elif args.command == "task8-manifests":
        result = create_manifest_release(
            args.repo_root,
            args.source_preflight_dir,
            args.output_dir,
            args.created_at,
        )
    elif args.command == "validate-task8-manifests":
        result = validate_manifest_release(args.release_dir, args.repo_root)
    elif args.command == "task9-checkpoint-preflight":
        result = run_checkpoint_preflight(
            args.repo_root,
            args.source_preflight_dir,
            args.manifest_release_dir,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-task9-checkpoint-preflight":
        result = validate_checkpoint_preflight(args.release_dir)
    elif args.command == "task10-warmstart":
        result = run_closed_loop_warmstart(
            args.repo_root,
            args.source_preflight_dir,
            args.manifest_release_dir,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-task10-warmstart":
        result = validate_closed_loop_release(args.release_dir)
    elif args.command == "hierarchical-zero-identity":
        result = run_hierarchical_identity(
            args.repo_root,
            args.source_preflight_dir,
            args.sidecar_release_dir,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-hierarchical-zero-identity":
        result = validate_hierarchical_identity(
            args.release_dir, repo_root=args.repo_root
        )
    elif args.command == "hierarchical-warmstart-manifest":
        result = create_hierarchical_warmstart_manifest(
            args.repo_root,
            args.source_preflight_dir,
            args.hierarchical_identity_release_dir,
            args.output_dir,
            args.created_at,
        )
    elif args.command == "validate-hierarchical-warmstart-manifest":
        result = validate_hierarchical_warmstart_manifest(
            args.release_dir, args.repo_root
        )
    elif args.command == "hierarchical-warmstart":
        result = run_hierarchical_warmstart(
            args.repo_root,
            args.source_preflight_dir,
            args.hierarchical_identity_release_dir,
            args.manifest_dir,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-hierarchical-warmstart":
        result = validate_hierarchical_warmstart_release(
            args.release_dir, args.repo_root, device_name=args.device
        )
    elif args.command == "hierarchical-task9":
        result = run_hierarchical_checkpoint_preflight(
            args.repo_root,
            args.source_preflight_dir,
            args.manifest_release_dir,
            args.warmstart_release_dir,
            args.warmstart_output_manifest_sha256,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-hierarchical-task9":
        result = validate_hierarchical_checkpoint_preflight(args.release_dir)
    elif args.command == "hierarchical-task10":
        result = run_hierarchical_closed_loop(
            args.repo_root,
            args.source_preflight_dir,
            args.manifest_release_dir,
            args.warmstart_release_dir,
            args.warmstart_output_manifest_sha256,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-hierarchical-task10":
        result = validate_hierarchical_closed_loop(args.release_dir)
    elif args.command == "validate-zero-identity":
        result = validate_zero_identity(args.release_dir)
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

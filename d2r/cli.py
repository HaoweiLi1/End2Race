#!/usr/bin/env python3
"""CLI for the locked D2R-G representation redesign."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch

from d2r import EVIDENCE_RELPATH, REGISTRY_OPENED_AT, SEED
from d2r.data import D2RDataset
from d2r.release import validate_release
from d2r.summary import create_summary
from d2r.train import predict_model, run_oof, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="D2R-G spatiotemporal geometry probe")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("micro", "run", "validate", "summarize"):
        command = sub.add_parser(name)
        command.add_argument("--dataset-dir", required=True)
        command.add_argument("--split-dir", required=True)
        command.add_argument("--signals-dir", required=True)
        if name not in {"validate", "summarize"}:
            command.add_argument("--device", default="cuda:0")
        if name == "run":
            command.add_argument("--output-dir", required=True)
            command.add_argument("--created-at", required=True)
            command.add_argument("--outer-folds", default="0,1,2,3,4")
            command.add_argument("--registry", required=True)
            command.add_argument("--registry-opened-at", default=REGISTRY_OPENED_AT)
            command.add_argument("--evidence-relpath", default=EVIDENCE_RELPATH)
        elif name == "validate":
            command.add_argument("release_dir")
        elif name == "summarize":
            command.add_argument("--probe-dir", required=True)
            command.add_argument("--output-dir", required=True)
            command.add_argument("--created-at", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "micro":
        dataset = D2RDataset(args.dataset_dir, args.split_dir, args.signals_dir)
        device = torch.device(args.device)
        mask = dataset.base.outer_fold != 0
        model, mean, std, report = train_model(
            dataset, mask, device, SEED, max_batches_per_epoch=2
        )
        indices = dataset.base.frame_indices(dataset.base.outer_fold == 0)[:128]
        predictions = predict_model(model, dataset, indices, mean, std, device)
        second, mean2, std2, report2 = train_model(
            dataset, mask, device, SEED, max_batches_per_epoch=2
        )
        predictions2 = predict_model(second, dataset, indices, mean2, std2, device)
        state_equal = all(
            torch.equal(model.state_dict()[name], second.state_dict()[name])
            for name in model.state_dict()
        )
        deterministic = bool(
            state_equal
            and np.array_equal(mean, mean2)
            and np.array_equal(std, std2)
            and report["history"] == report2["history"]
            and np.array_equal(predictions, predictions2)
        )
        result = {
            "passed": bool(np.all(np.isfinite(predictions)) and deterministic),
            "deterministic": deterministic,
            "sampled_frame_count": report["sampled_frame_count"],
            "epochs": len(report["history"]),
            "batches_per_epoch": [row["batches"] for row in report["history"]],
            "prediction_shape": list(predictions.shape),
            "prediction_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
            "final_loss": report["history"][-1]["loss"],
        }
    elif args.command == "run":
        folds = tuple(int(value) for value in args.outer_folds.split(","))
        result = run_oof(
            dataset_dir=args.dataset_dir,
            split_dir=args.split_dir,
            signals_dir=args.signals_dir,
            output_dir=args.output_dir,
            created_at=args.created_at,
            device_name=args.device,
            outer_folds=folds,
            registry_path=args.registry,
            registry_opened_at=args.registry_opened_at,
            evidence_relpath=args.evidence_relpath,
        )
    elif args.command == "validate":
        result = validate_release(
            args.release_dir,
            args.dataset_dir,
            args.split_dir,
            args.signals_dir,
        )
    else:
        result = create_summary(
            args.dataset_dir,
            args.split_dir,
            args.signals_dir,
            args.probe_dir,
            args.output_dir,
            args.created_at,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

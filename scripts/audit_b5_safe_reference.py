#!/usr/bin/env python3
"""Read-only pre-RunPlan audit of the B5 safe cap against BC and B4 snapshots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from bplus_v22.b4_direct import B4DirectHeadPolicy, strict_plain_actor_from_state
from bplus_v22.b5_safe import SAFE_CAP, file_sha256, load_reference, safe_kl_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actor", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".partial").exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("B5 reference audit requested unavailable CUDA")
    reference = load_reference(args.reference, device)
    actors: dict[str, Path] = {}
    for value in args.actor:
        name, separator, path_text = value.partition("=")
        path = Path(path_text).resolve()
        if not separator or not name or name in actors or not path.is_file():
            raise ValueError(f"invalid B5 actor specification: {value!r}")
        actors[name] = path
    if set(actors) != {"BC", "b4_iter10", "b4_iter20", "b4_iter30"}:
        raise ValueError("B5 reference audit requires BC and all three B4 snapshots")
    metrics = {}
    for name, path in actors.items():
        state = torch.load(path, map_location="cpu", weights_only=True)
        strict_plain_actor_from_state(state)
        policy = B4DirectHeadPolicy(state).to(device)
        metrics[name] = {
            "checkpoint_sha256": file_sha256(path),
            "safe": safe_kl_metrics(policy, reference),
        }
    bc_mean = float(metrics["BC"]["safe"]["mean"])
    iter10_mean = float(metrics["b4_iter10"]["safe"]["mean"])
    passed = bc_mean <= 1e-10 and iter10_mean > SAFE_CAP
    result = {
        "schema": "end2race-b5-safe-reference-audit-1",
        "passed": passed,
        "safe_cap": SAFE_CAP,
        "reference_sha256": file_sha256(args.reference),
        "reference_episode_count": len(reference.lengths),
        "reference_frame_count": reference.frame_count,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

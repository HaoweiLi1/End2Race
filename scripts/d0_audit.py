#!/usr/bin/env python3
"""CLI entry point for the D0.1 canonical audit."""

from __future__ import annotations

import argparse
import sys

from d0 import default_runconfig
from d0.scan import run_scan


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="D0.1 canonical P1 audit")
    parser.add_argument("--mode", required=True, choices=("geometry", "smoke", "full"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-root")
    parser.add_argument("--assets-root")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        return 4
    config = default_runconfig()
    if args.eval_root:
        config["eval_root"] = args.eval_root
    if args.assets_root:
        config["assets_root"] = args.assets_root
    return run_scan(args.mode, args.output_dir, config, workers=args.workers)


if __name__ == "__main__":
    sys.exit(main())

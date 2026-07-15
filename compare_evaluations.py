#!/usr/bin/env python3
"""Compare two End2Race evaluation runs by stable scenario ID."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from evaluation.compare import compare_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run")
    parser.add_argument("candidate_run")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = Path(args.baseline_run).resolve()
    candidate = Path(args.candidate_run).resolve()
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else candidate / "comparisons" / f"from-{baseline.name}"
    )
    result = compare_runs(baseline, candidate, output)
    print(f"comparison_dir={output}")
    print(
        f"paired={result['paired_scenarios']} fixed_collisions={result['fixed_ego_collisions']} "
        f"new_collisions={result['new_ego_collisions']} gained_overtakes={result['gained_overtakes']} "
        f"lost_overtakes={result['lost_overtakes']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)

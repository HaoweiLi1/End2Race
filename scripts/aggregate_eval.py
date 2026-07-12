#!/usr/bin/env python3
"""Validated aggregation for parallel multi-agent evaluation runs.

Replaces the exit-code counting in evaluate.sh / sweep wrappers. Outcomes are
read exclusively from per-episode metrics JSON files; worker exit codes are
used only as a success/failure signal (0 = success). Aggregation fails loudly
unless every requested episode produced exactly one valid metrics JSON (and,
with --require_npz, one non-empty NPZ), so a crashed worker can never be
silently counted as a valid episode again (the full_disc_r8192 seed1 iter300
488/600 case).

Exit codes: 0 = aggregated and validated; 2 = validation failed.

Usage:
  python aggregate_eval.py --tmp_dir T --expected_total N \
      --model_path M --map_name MAP [--noise 0.0] [--result_tag TAG] \
      [--require_npz] [--offset 0] [--tsv_out PATH]
"""

import argparse
import json
import os
import sys

VALID_OUTCOMES = {"following": "follow", "overtaking": "overtake", "collision": "collision"}


def fail(errors):
    shown = errors[:30]
    for e in shown:
        print(f"VALIDATION ERROR: {e}", file=sys.stderr)
    if len(errors) > len(shown):
        print(f"VALIDATION ERROR: ... and {len(errors) - len(shown)} more", file=sys.stderr)
    print(f"AGGREGATION REJECTED: {len(errors)} validation error(s)", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp_dir", required=True)
    ap.add_argument("--expected_total", type=int, required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--map_name", required=True)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--result_tag", default=None)
    ap.add_argument("--require_npz", action="store_true")
    ap.add_argument("--offset", default="0", help="start-offset label for the TSV line only")
    ap.add_argument("--tsv_out", default=None)
    args = ap.parse_args()

    n = args.expected_total
    errors = []
    episodes = {}

    # Every job id in 0..N-1 must have a zero exit code and a valid JSON.
    for i in range(n):
        exit_path = os.path.join(args.tmp_dir, f"{i}.exit")
        json_path = os.path.join(args.tmp_dir, f"{i}.json")
        try:
            with open(exit_path) as f:
                code = int(f.read().strip())
        except (OSError, ValueError) as e:
            errors.append(f"job {i}: unreadable exit file ({e})")
            code = None
        if code is not None and code != 0:
            errors.append(f"job {i}: worker exit code {code}")
        if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
            errors.append(f"job {i}: missing or empty metrics JSON")
            continue
        try:
            with open(json_path) as f:
                metric = json.load(f)
        except (OSError, ValueError) as e:
            errors.append(f"job {i}: unparseable metrics JSON ({e})")
            continue
        outcome = metric.get("outcome") or metric.get("state_label")
        key = metric.get("episode_key")
        if outcome not in VALID_OUTCOMES:
            errors.append(f"job {i}: invalid outcome {outcome!r}")
            continue
        if not key:
            errors.append(f"job {i}: missing episode_key")
            continue
        if key in episodes:
            errors.append(f"job {i}: duplicate episode_key {key}")
            continue
        if args.require_npz:
            npz = metric.get("npz_path")
            if not npz or not os.path.exists(npz) or os.path.getsize(npz) == 0:
                errors.append(f"job {i}: missing or empty NPZ {npz!r}")
                continue
        episodes[key] = (i, outcome, metric)

    # Reject stale/extra artifacts beyond the requested job ids.
    for name in sorted(os.listdir(args.tmp_dir)):
        stem, ext = os.path.splitext(name)
        if ext in (".json", ".exit") and stem.isdigit() and int(stem) >= n:
            errors.append(f"unexpected extra artifact {name} (expected job ids 0..{n - 1})")

    if len(episodes) != n:
        errors.append(f"episode count {len(episodes)} != expected {n}")
    if errors:
        fail(errors)

    counts = {"follow": 0, "overtake": 0, "collision": 0}
    ego_coll = opp_coll = 0
    for _, outcome, metric in episodes.values():
        counts[VALID_OUTCOMES[outcome]] += 1
        ego_coll += bool(metric.get("ego_collision"))
        opp_coll += bool(metric.get("opp_collision"))

    from utils import write_multiagent_results, load_json_file

    result_path = write_multiagent_results(
        args.model_path, args.map_name, args.noise, args.tmp_dir, n,
        counts["follow"], counts["overtake"], counts["collision"], 0,
        result_tag=args.result_tag,
        extra_final={
            "ego_collision_count": ego_coll,
            "opp_collision_count": opp_coll,
            "validated": True,
        },
    )

    # A stale results.json in the result dir would merge old episodes in;
    # reject instead of reporting a silently inflated set.
    written = load_json_file(result_path)
    if len(written.get("episodes", {})) != n:
        fail([
            f"results.json holds {len(written.get('episodes', {}))} episodes, expected {n}: "
            f"stale pre-existing results in {result_path}; delete the result dir and re-aggregate"
        ])

    if args.tsv_out:
        tag = args.result_tag or os.path.splitext(os.path.basename(args.model_path))[0]
        with open(args.tsv_out, "w") as f:
            f.write(f"{tag}\t{args.map_name}\t{args.offset}\t{counts['collision']}\t"
                    f"{counts['overtake']}\t{counts['follow']}\t0\n")

    print(f"RESULT total={n} collision={counts['collision']} overtake={counts['overtake']} "
          f"follow={counts['follow']} error=0 ego_collision={ego_coll} opp_collision={opp_coll} "
          f"results_json={result_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline collision classification for multiagent eval results.

Reads the collision episodes saved by eval_multiagent.py and buckets each one
by cause and phase using the final recorded frame:
- cause: ego-opponent distance <= --car_dist_thresh means car-vs-car,
  otherwise the ego hit the wall on its own.
- phase: |ego-opp progress| < --alongside_thresh means the cars overlapped
  longitudinally at impact (side contact during a pass attempt); otherwise
  the sign splits pre-overtake from post-overtake.

The resulting 2x2 table drives the next PPO tuning direction: a high wall
share suggests a wall-distance reward term, a high post-overtake car-vs-car
share suggests rear-clearance tuning.

Usage:
    python analyze_collisions.py eval_results/end2race_Austin_version2
"""

import argparse
import glob
import os
import numpy as np

CAUSES = ('car', 'wall')
PHASES = ('pre', 'alongside', 'post')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Classify multiagent eval collision episodes.')
    parser.add_argument('results_dir', type=str,
                        help='eval results directory containing a collision/ subfolder')
    parser.add_argument('--car_dist_thresh', type=float, default=1.0,
                        help='max ego-opponent distance (m) at the final frame to call it car-vs-car')
    parser.add_argument('--alongside_thresh', type=float, default=0.6,
                        help='|ego-opp progress| (m) at the final frame below which the impact counts as alongside')
    return parser.parse_args()

def classify_episode(path, car_dist_thresh, alongside_thresh):
    """Return (cause, phase, final ego-opponent distance) for one collision npz."""
    data = np.load(path, allow_pickle=True)
    ego_pose = np.asarray(data['ego_pose'], dtype=float)
    opp_pose = np.asarray(data['opp_pose'], dtype=float)
    final_dist = float(np.linalg.norm(ego_pose[-1, :2] - opp_pose[-1, :2]))
    cause = 'car' if final_dist <= car_dist_thresh else 'wall'

    ego_progress = float(np.asarray(data['ego_progress'], dtype=float)[-1])
    opp_progress = float(np.asarray(data['opp_progress'], dtype=float)[-1])
    rel_s = ego_progress - opp_progress
    if abs(rel_s) < alongside_thresh:
        phase = 'alongside'
    else:
        phase = 'post' if rel_s > 0 else 'pre'
    return cause, phase, final_dist

def episode_key_from_path(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name[2:] if name.startswith('c_') else name

def main():
    args = parse_arguments()
    episode_paths = sorted(glob.glob(os.path.join(args.results_dir, 'collision', '*.npz')))
    if not episode_paths:
        print(f"No collision npz files found under {args.results_dir}/collision/")
        return

    buckets = {(cause, phase): [] for cause in CAUSES for phase in PHASES}
    for path in episode_paths:
        cause, phase, final_dist = classify_episode(path, args.car_dist_thresh, args.alongside_thresh)
        buckets[(cause, phase)].append((episode_key_from_path(path), final_dist))

    total = len(episode_paths)
    print(f"Collision episodes: {total} ({args.results_dir})")
    print(f"car-vs-car threshold: final ego-opp distance <= {args.car_dist_thresh} m")
    print(f"alongside threshold: final |rel_s| < {args.alongside_thresh} m\n")

    print(f"{'':>12} {'pre-overtake':>14} {'alongside':>14} {'post-overtake':>14}")
    for cause in CAUSES:
        row = "".join(f"{len(buckets[(cause, phase)]):>14}" for phase in PHASES)
        print(f"{cause:>12}{row}")
    print()

    for cause in CAUSES:
        for phase in PHASES:
            episodes = buckets[(cause, phase)]
            if not episodes:
                continue
            print(f"[{cause}/{phase}] {len(episodes)} episodes:")
            for key, final_dist in episodes:
                print(f"  {key} (final ego-opp dist {final_dist:.2f} m)")

if __name__ == '__main__':
    main()

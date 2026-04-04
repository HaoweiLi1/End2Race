"""
Episode quality validator for collected demonstration data.

Usage:
    python episode_validator.py <csv_file>
    python episode_validator.py --scan_dir Dataset_Austin_0404/success/

CSV format: [time, steer, desired_speed, lidar_0, ..., lidar_359]
Sampling interval: 0.1s
LiDAR: 360 beams, 1 degree resolution, index 0 = front, clockwise
"""

import argparse
import csv
import os
import sys
import numpy as np


# ── Thresholds ──────────────────────────────────────────────────────────────

PROXIMITY_THRESHOLD = 0.25      # meters, front min lidar (P5=0.24 of real data)
PROXIMITY_SIDE_THRESHOLD = 0.25 # meters, side/rear proximity (P5=0.24 of real data)

REVERSAL_WINDOW = 10            # frames = 1.0s at 0.1s interval
REVERSAL_MAX_PER_WINDOW = 8     # max sign changes in steering rate per window (P75=8)

STEER_JERK_THRESHOLD = 65.0     # rad/s^2, max steering acceleration (P90=65.7)
STEER_JUMP_THRESHOLD = 0.45     # rad, max single-step steer change (P90=0.43)

DT = 0.1                        # sampling interval (seconds)

# LiDAR sector definitions (index 0 = front, clockwise)
# 360 beams, 1 degree per beam
SECTOR_FRONT = list(range(0, 30)) + list(range(330, 360))       # front ±30°
SECTOR_RIGHT = list(range(60, 120))                              # right side
SECTOR_REAR  = list(range(150, 210))                             # rear ±30°
SECTOR_LEFT  = list(range(240, 300))                             # left side


# ── Core detection functions ────────────────────────────────────────────────

def check_proximity(lidar, threshold=PROXIMITY_THRESHOLD, side_threshold=PROXIMITY_SIDE_THRESHOLD):
    """
    Detect frames where ego is dangerously close to wall or opponent.

    Args:
        lidar: np.array shape (n_frames, 360)

    Returns:
        list of dicts, each describing one proximity event
    """
    issues = []
    n_frames = lidar.shape[0]

    for i in range(n_frames):
        frame = lidar[i]
        global_min = np.min(frame)

        # Check each sector
        front_min = np.min(frame[SECTOR_FRONT])
        right_min = np.min(frame[SECTOR_RIGHT])
        rear_min  = np.min(frame[SECTOR_REAR])
        left_min  = np.min(frame[SECTOR_LEFT])

        # Side/rear proximity is more indicative of bad overtaking behavior
        if right_min < side_threshold or left_min < side_threshold:
            sector = "right" if right_min < left_min else "left"
            min_val = min(right_min, left_min)
            issues.append({
                'type': 'side_proximity',
                'frame': i,
                'time': round(i * DT, 2),
                'sector': sector,
                'min_range': round(float(min_val), 3),
            })
        elif rear_min < side_threshold:
            issues.append({
                'type': 'rear_proximity',
                'frame': i,
                'time': round(i * DT, 2),
                'sector': 'rear',
                'min_range': round(float(rear_min), 3),
            })
        elif front_min < threshold:
            issues.append({
                'type': 'front_proximity',
                'frame': i,
                'time': round(i * DT, 2),
                'sector': 'front',
                'min_range': round(float(front_min), 3),
            })

    return issues


def check_steering(steer):
    """
    Detect frequent steering reversals and jerky steering.

    Args:
        steer: np.array shape (n_frames,)

    Returns:
        list of dicts, each describing one steering issue
    """
    issues = []

    if len(steer) < 3:
        return issues

    steer_rate = np.diff(steer) / DT                   # rad/s
    steer_accel = np.diff(steer_rate) / DT              # rad/s^2 (jerk)
    steer_diff = np.diff(steer)                         # raw single-step change
    sign_changes = np.abs(np.diff(np.sign(steer_rate)))  # 2 where sign flips

    # ── 1. Sliding window reversal detection ──
    flagged_windows = set()
    for i in range(len(sign_changes) - REVERSAL_WINDOW + 1):
        window = sign_changes[i:i + REVERSAL_WINDOW]
        n_reversals = np.sum(window > 0)
        if n_reversals > REVERSAL_MAX_PER_WINDOW:
            # Record the center of the window, avoid duplicate reports
            center = i + REVERSAL_WINDOW // 2
            bucket = center // REVERSAL_WINDOW
            if bucket not in flagged_windows:
                flagged_windows.add(bucket)
                issues.append({
                    'type': 'steering_oscillation',
                    'frame_start': i,
                    'frame_end': i + REVERSAL_WINDOW,
                    'time_start': round(i * DT, 2),
                    'time_end': round((i + REVERSAL_WINDOW) * DT, 2),
                    'reversals': int(n_reversals),
                })

    # ── 2. Steering jerk (sudden acceleration of steering) ──
    jerk_peaks = np.where(np.abs(steer_accel) > STEER_JERK_THRESHOLD)[0]
    for idx in jerk_peaks:
        issues.append({
            'type': 'steering_jerk',
            'frame': int(idx + 1),
            'time': round((idx + 1) * DT, 2),
            'jerk': round(float(steer_accel[idx]), 2),
        })

    # ── 3. Large single-step steering jump ──
    jump_indices = np.where(np.abs(steer_diff) > STEER_JUMP_THRESHOLD)[0]
    for idx in jump_indices:
        issues.append({
            'type': 'steering_jump',
            'frame': int(idx),
            'time': round(idx * DT, 2),
            'delta': round(float(steer_diff[idx]), 3),
        })

    return issues


# ── Main validation entry point ────────────────────────────────────────────

def validate_episode(csv_path):
    """
    Validate a single episode CSV.

    Returns:
        dict with keys:
            'file': str
            'is_valid': bool
            'issues': list of issue dicts
            'summary': dict of issue counts by type
    """
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    steer = data[:, 1]
    lidar = data[:, 3:]  # columns 3..362

    all_issues = []
    all_issues.extend(check_proximity(lidar))
    all_issues.extend(check_steering(steer))

    # Sort by time
    all_issues.sort(key=lambda x: x.get('time', x.get('time_start', 0)))

    # Summary counts
    summary = {}
    for issue in all_issues:
        t = issue['type']
        summary[t] = summary.get(t, 0) + 1

    return {
        'file': os.path.basename(csv_path),
        'is_valid': len(all_issues) == 0,
        'issues': all_issues,
        'summary': summary,
    }


def print_report(result):
    """Pretty-print validation result."""
    fname = result['file']
    if result['is_valid']:
        print(f"[PASS] {fname}")
        return

    print(f"[FAIL] {fname}")
    for itype, count in result['summary'].items():
        print(f"  {itype}: {count} events")

    # Print first few issues as examples
    shown = 0
    for issue in result['issues']:
        if shown >= 10:
            remaining = len(result['issues']) - shown
            print(f"  ... and {remaining} more issues")
            break
        if 'time_start' in issue:
            print(f"    t={issue['time_start']}-{issue['time_end']}s  {issue['type']}  reversals={issue.get('reversals', '')}")
        else:
            detail = issue.get('min_range') or issue.get('jerk') or issue.get('delta') or ''
            print(f"    t={issue['time']}s  {issue['type']}  {detail}")
        shown += 1


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Episode quality validator')
    parser.add_argument('csv_file', nargs='?', help='Single CSV file to validate')
    parser.add_argument('--scan_dir', type=str, help='Directory of CSVs to batch validate')
    args = parser.parse_args()

    if args.scan_dir:
        csv_files = sorted([
            os.path.join(args.scan_dir, f)
            for f in os.listdir(args.scan_dir) if f.endswith('.csv')
        ])
        if not csv_files:
            print(f"No CSV files found in {args.scan_dir}")
            sys.exit(1)

        pass_count = 0
        fail_count = 0
        for csv_path in csv_files:
            result = validate_episode(csv_path)
            print_report(result)
            if result['is_valid']:
                pass_count += 1
            else:
                fail_count += 1

        print(f"\n{'='*40}")
        print(f"Total: {pass_count + fail_count}  Pass: {pass_count}  Fail: {fail_count}")

    elif args.csv_file:
        result = validate_episode(args.csv_file)
        print_report(result)
        sys.exit(0 if result['is_valid'] else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

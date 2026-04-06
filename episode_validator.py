"""
Episode quality validator for collected demonstration data.

Usage:
    python episode_validator.py <csv_file>                          # validate single file
    python episode_validator.py --scan_dir Dataset/success/         # validate all, print report
    python episode_validator.py --analyze Dataset/success/          # output per-case JSON + metrics
    python episode_validator.py --calibrate Dataset/success/        # print percentile distributions

CSV format: [time, steer, desired_speed, lidar_0, ..., lidar_359]
Sampling interval: 0.1s
LiDAR: 360 beams, 1 degree resolution, index 0 = front, clockwise
"""

import argparse
import json
import os
import sys
import numpy as np


# ── Vehicle geometry ───────────────────────────────────────────────────────
# F1/10 car body dimensions (same as lattice_planner.py)
CAR_WIDTH = 0.31       # meters
CAR_LENGTH = 0.58      # meters
HALF_W = CAR_WIDTH / 2   # 0.155
HALF_L = CAR_LENGTH / 2  # 0.29

def _precompute_d_edge(n_beams=360):
    """Precompute sensor-to-ego-surface distance for each lidar beam.

    Beam i is at angle i degrees clockwise from front.
    d_edge(θ) = min(|half_L / cos(θ)|, |half_W / sin(θ)|)
    """
    angles = np.arange(n_beams) * (np.pi / 180)
    c = np.abs(np.cos(angles))
    s = np.abs(np.sin(angles))
    # Avoid division by zero: when cos≈0 → front/back unreachable, only side matters
    d_l = np.where(c > 1e-9, HALF_L / c, np.inf)
    d_w = np.where(s > 1e-9, HALF_W / s, np.inf)
    return np.minimum(d_l, d_w)

D_EDGE = _precompute_d_edge()  # (360,) constant lookup table

# ── Thresholds ──────────────────────────────────────────────────────────────

PROXIMITY_THRESHOLD = 0.07      # meters, ego surface to obstacle (uniform all directions)
PROXIMITY_SIDE_THRESHOLD = 0.07 # meters, ego surface to obstacle

REVERSAL_WINDOW = 10            # frames = 1.0s at 0.1s interval
REVERSAL_MAX_PER_WINDOW = 8     # max sign changes in steering rate per window

SPEED_VARIANCE_THRESHOLD = 2.0      # (m/s)^2, ~P98 of real data
STEER_AUTOCORR_THRESHOLD = -0.1     # lag-1, ~P2 of real data (negative = actively oscillating)

DT = 0.1                        # sampling interval (seconds)

# LiDAR sector definitions (index 0 = front, clockwise)
# 360 beams, 1 degree per beam, full coverage including diagonals
SECTOR_FRONT       = list(range(0, 30)) + list(range(330, 360))  # front ±30°
SECTOR_FRONT_RIGHT = list(range(30, 60))                          # right-front diagonal
SECTOR_RIGHT       = list(range(60, 120))                         # right side
SECTOR_REAR_RIGHT  = list(range(120, 150))                        # right-rear diagonal
SECTOR_REAR        = list(range(150, 210))                        # rear ±30°
SECTOR_REAR_LEFT   = list(range(210, 240))                        # left-rear diagonal
SECTOR_LEFT        = list(range(240, 300))                        # left side
SECTOR_FRONT_LEFT  = list(range(300, 330))                        # left-front diagonal

SECTOR_SIDE_ALL = (SECTOR_FRONT_RIGHT + SECTOR_RIGHT + SECTOR_REAR_RIGHT
                   + SECTOR_REAR_LEFT + SECTOR_LEFT + SECTOR_FRONT_LEFT)


# ── Core detection functions ────────────────────────────────────────────────

def check_proximity(lidar, threshold=PROXIMITY_THRESHOLD, side_threshold=PROXIMITY_SIDE_THRESHOLD):
    """Detect frames where ego surface is dangerously close to obstacle.

    Converts raw lidar (sensor-to-obstacle) to surface distance by
    subtracting the precomputed sensor-to-ego-edge offset per beam.
    """
    # Convert all frames at once: surface_dist = lidar - D_EDGE
    surface_dist = lidar - D_EDGE  # (n_frames, 360)
    np.clip(surface_dist, 0.0, None, out=surface_dist)

    issues = []
    sectors = [
        ('front',       SECTOR_FRONT,       threshold),
        ('front_right', SECTOR_FRONT_RIGHT, side_threshold),
        ('right',       SECTOR_RIGHT,       side_threshold),
        ('rear_right',  SECTOR_REAR_RIGHT,  side_threshold),
        ('rear',        SECTOR_REAR,        side_threshold),
        ('rear_left',   SECTOR_REAR_LEFT,   side_threshold),
        ('left',        SECTOR_LEFT,        side_threshold),
        ('front_left',  SECTOR_FRONT_LEFT,  side_threshold),
    ]

    for i in range(surface_dist.shape[0]):
        frame = surface_dist[i]
        worst_sector = None
        worst_val = float('inf')
        for sector_name, sector_idx, thresh in sectors:
            sector_min = np.min(frame[sector_idx])
            if sector_min < thresh and sector_min < worst_val:
                worst_val = sector_min
                worst_sector = sector_name

        if worst_sector is not None:
            ptype = 'front_proximity' if worst_sector == 'front' else 'side_proximity'
            issues.append({
                'type': ptype,
                'frame': i,
                'time': round(i * DT, 2),
                'sector': worst_sector,
                'min_range': round(float(worst_val), 3),
            })
    return issues


def check_steering(steer):
    """Detect frequent steering reversals via sliding window."""
    issues = []
    if len(steer) < 3:
        return issues

    steer_rate = np.diff(steer) / DT
    sign_changes = np.abs(np.diff(np.sign(steer_rate)))

    flagged_windows = set()
    for i in range(len(sign_changes) - REVERSAL_WINDOW + 1):
        window = sign_changes[i:i + REVERSAL_WINDOW]
        n_reversals = np.sum(window > 0)
        if n_reversals > REVERSAL_MAX_PER_WINDOW:
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
    return issues


def check_speed_variance(speed):
    """Detect episodes with abnormally high speed variance."""
    if len(speed) < 2:
        return []
    variance = float(np.var(speed))
    if variance > SPEED_VARIANCE_THRESHOLD:
        return [{'type': 'speed_variance', 'variance': round(variance, 4)}]
    return []


def check_steering_autocorrelation(steer):
    """Detect episodes with low steering autocorrelation (lag-1)."""
    if len(steer) < 3:
        return []
    steer_centered = steer - np.mean(steer)
    norm = np.sum(steer_centered ** 2)
    if norm < 1e-10:
        return []
    autocorr = float(np.sum(steer_centered[:-1] * steer_centered[1:]) / norm)
    if autocorr < STEER_AUTOCORR_THRESHOLD:
        return [{'type': 'low_steer_autocorrelation', 'autocorr_lag1': round(autocorr, 4)}]
    return []


# ── Metrics computation ────────────────────────────────────────────────────

def compute_metrics(csv_path):
    """Compute all raw metrics for one episode (used by --analyze and --calibrate)."""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    steer = data[:, 1]
    speed = data[:, 2]
    lidar = data[:, 3:]

    surface_dist = np.clip(lidar - D_EDGE, 0.0, None)

    metrics = {
        'global_min_surface_dist': round(float(np.min(surface_dist)), 4),
        'side_min_surface_dist': round(float(np.min(surface_dist[:, SECTOR_SIDE_ALL])), 4),
        'front_min_surface_dist': round(float(np.min(surface_dist[:, SECTOR_FRONT])), 4),
        'speed_mean': round(float(np.mean(speed)), 4),
        'speed_variance': round(float(np.var(speed)), 4),
        'n_frames': len(steer),
        'duration': round(len(steer) * DT, 1),
    }

    if len(steer) >= 3:
        steer_rate = np.diff(steer) / DT
        sign_changes = np.abs(np.diff(np.sign(steer_rate)))

        # Max reversals in any 1s window
        window = 10
        max_win_rev = 0
        for i in range(len(sign_changes) - window + 1):
            max_win_rev = max(max_win_rev, int(np.sum(sign_changes[i:i+window] > 0)))
        metrics['max_window_reversals'] = max_win_rev

        # Autocorrelation lag-1
        sc = steer - np.mean(steer)
        norm = np.sum(sc ** 2)
        metrics['steer_autocorr_lag1'] = round(float(np.sum(sc[:-1] * sc[1:]) / norm), 4) if norm > 1e-10 else 1.0
    else:
        metrics.update({
            'max_window_reversals': 0,
            'steer_autocorr_lag1': 1.0,
        })
    return metrics


# ── Validation entry point ─────────────────────────────────────────────────

def validate_episode(csv_path):
    """Validate a single episode CSV. Returns dict with is_valid, issues, summary."""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    steer = data[:, 1]
    speed = data[:, 2]
    lidar = data[:, 3:]

    all_issues = []
    all_issues.extend(check_proximity(lidar))
    all_issues.extend(check_steering(steer))
    all_issues.extend(check_speed_variance(speed))
    all_issues.extend(check_steering_autocorrelation(steer))

    all_issues.sort(key=lambda x: x.get('time', x.get('time_start', 0)))

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


# ── Output helpers ─────────────────────────────────────────────────────────

def print_report(result):
    """Pretty-print validation result for one episode."""
    fname = result['file']
    if result['is_valid']:
        print(f"[PASS] {fname}")
        return

    print(f"[FAIL] {fname}")
    for itype, count in result['summary'].items():
        print(f"  {itype}: {count} events")

    shown = 0
    for issue in result['issues']:
        if shown >= 10:
            print(f"  ... and {len(result['issues']) - shown} more issues")
            break
        if 'time_start' in issue:
            print(f"    t={issue['time_start']}-{issue['time_end']}s  {issue['type']}  reversals={issue.get('reversals', '')}")
        else:
            detail = issue.get('min_range') or issue.get('variance') or issue.get('autocorr_lag1') or ''
            print(f"    t={issue['time']}s  {issue['type']}  {detail}")
        shown += 1


def print_percentiles(name, arr):
    arr = np.array(arr)
    print(f"{name} (n={len(arr)}):")
    for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        print(f"  P{p:3d}: {np.percentile(arr, p):.4f}")
    print()


# ── CLI modes ──────────────────────────────────────────────────────────────

def get_csv_files(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')])


def mode_validate_single(csv_file):
    """Validate one CSV, exit 0=pass 1=fail."""
    result = validate_episode(csv_file)
    print_report(result)
    sys.exit(0 if result['is_valid'] else 1)


def mode_scan(scan_dir):
    """Validate all CSVs in directory, print per-file report + summary."""
    csv_files = get_csv_files(scan_dir)
    if not csv_files:
        print(f"No CSV files found in {scan_dir}")
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


def mode_analyze(scan_dir):
    """Compute per-case metrics + validation, save JSON."""
    csv_files = get_csv_files(scan_dir)
    if not csv_files:
        print(f"No CSV files found in {scan_dir}")
        sys.exit(1)

    print(f"Analyzing {len(csv_files)} episodes...")

    results = []
    for csv_path in csv_files:
        metrics = compute_metrics(csv_path)
        validation = validate_episode(csv_path)
        results.append({
            'file': os.path.basename(csv_path),
            'metrics': metrics,
            'is_valid': validation['is_valid'],
            'issue_summary': validation['summary'],
            'issues': validation['issues'],
        })

    results.sort(key=lambda r: (r['is_valid'], -len(r['issues'])))

    out_dir = os.path.dirname(scan_dir.rstrip('/'))
    out_path = os.path.join(out_dir, 'episode_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    fail_count = sum(1 for r in results if not r['is_valid'])
    pass_count = len(results) - fail_count
    print(f"\nTotal: {len(results)}  Pass: {pass_count}  Fail: {fail_count}")

    print(f"\nTop 10 worst cases:")
    for r in results[:10]:
        m = r['metrics']
        tag = "[FAIL]" if not r['is_valid'] else "[PASS]"
        issues_str = ", ".join(f"{k}:{v}" for k, v in r['issue_summary'].items()) if r['issue_summary'] else "none"
        print(f"  {tag} {r['file']}")
        print(f"    speed_var={m['speed_variance']}  autocorr={m['steer_autocorr_lag1']}  side_min={m['side_min_surface_dist']}  max_rev={m['max_window_reversals']}")
        print(f"    issues: {issues_str}")


def mode_calibrate(scan_dir):
    """Print percentile distributions for all metrics to calibrate thresholds."""
    csv_files = get_csv_files(scan_dir)
    if not csv_files:
        print(f"No CSV files found in {scan_dir}")
        sys.exit(1)

    print(f"Calibrating on {len(csv_files)} episodes...\n")

    all_metrics = {
        'global_min_surface_dist': [], 'side_min_surface_dist': [], 'front_min_surface_dist': [],
        'speed_variance': [],
        'max_window_reversals': [],
        'steer_autocorr_lag1': [],
    }

    for csv_path in csv_files:
        m = compute_metrics(csv_path)
        for key in all_metrics:
            all_metrics[key].append(m[key])

    print("=" * 50)
    print("SURFACE DISTANCE (ego edge to obstacle)")
    print("=" * 50)
    print_percentiles("Global min surface dist (m)", all_metrics['global_min_surface_dist'])
    print_percentiles("Side min surface dist (m)", all_metrics['side_min_surface_dist'])
    print_percentiles("Front min surface dist (m)", all_metrics['front_min_surface_dist'])

    print("=" * 50)
    print("STEERING")
    print("=" * 50)
    print_percentiles("Max window reversals (per 1s)", all_metrics['max_window_reversals'])

    print("=" * 50)
    print("SPEED")
    print("=" * 50)
    print_percentiles("Speed variance (m/s)^2", all_metrics['speed_variance'])

    print("=" * 50)
    print("STEERING AUTOCORRELATION")
    print("=" * 50)
    print_percentiles("Steer autocorr lag-1", all_metrics['steer_autocorr_lag1'])


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Episode quality validator')
    parser.add_argument('csv_file', nargs='?', help='Single CSV file to validate')
    parser.add_argument('--scan_dir', type=str, help='Validate all CSVs, print report')
    parser.add_argument('--analyze', type=str, metavar='DIR', help='Compute metrics + validation, save JSON')
    parser.add_argument('--calibrate', type=str, metavar='DIR', help='Print percentile distributions')
    args = parser.parse_args()

    if args.calibrate:
        mode_calibrate(args.calibrate)
    elif args.analyze:
        mode_analyze(args.analyze)
    elif args.scan_dir:
        mode_scan(args.scan_dir)
    elif args.csv_file:
        mode_validate_single(args.csv_file)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

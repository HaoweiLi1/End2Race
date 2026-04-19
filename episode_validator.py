import argparse
import json
import os
import sys
import numpy as np

def _precompute_d_edge(n_beams=360):
    """Sensor-to-ego-surface distance per beam: d_edge(theta) = min(|HALF_L/cos|, |HALF_W/sin|).

    f1tenth_gym beam i corresponds to vehicle-frame angle -π + i*(fov/(N-1))
    with fov=2π, so after downsampling 1440→360 beam i is at angle -π + i°.
    The LiDAR originates at the ego's geometric center (verified empirically).
    """
    half_w = 0.31 / 2   # 0.155, half car width
    half_l = 0.58 / 2   # 0.290, half car length (matches get_vertices in f1tenth_gym)
    angles = -np.pi + np.arange(n_beams) * (2 * np.pi / n_beams)
    c = np.abs(np.cos(angles)) + 1e-12
    s = np.abs(np.sin(angles)) + 1e-12
    return np.minimum(half_l / c, half_w / s)

# ── Detection functions ───────────────────────────────────────────────────

def check_proximity(lidar):
    """Detect frames where ego surface is too close to obstacles."""
    threshold = 0.1  # meters, uniform all directions

    # f1tenth_gym beam mapping (360-beam downsampled):
    #   beam   0 → -180° (rear),   beam  90 → -90° (right)
    #   beam 180 →    0° (front),  beam 270 → +90° (left)
    sectors = [
        ('rear',        list(range(0, 30)) + list(range(330, 360))),
        ('rear_right',  list(range(30, 60))),
        ('right',       list(range(60, 120))),
        ('front_right', list(range(120, 150))),
        ('front',       list(range(150, 210))),
        ('front_left',  list(range(210, 240))),
        ('left',        list(range(240, 300))),
        ('rear_left',   list(range(300, 330))),
    ]

    surface_dist = np.clip(lidar - _precompute_d_edge(), 0.0, None)
    issues = []
    for i in range(len(surface_dist)):
        frame = surface_dist[i]
        worst_sector, worst_val = None, float('inf')
        for name, idx in sectors:
            val = np.min(frame[idx])
            if val < threshold and val < worst_val:
                worst_val, worst_sector = val, name
        if worst_sector is not None:
            issues.append({
                'type': 'proximity',
                'frame': i, 'time': round(i * 0.1, 2),
                'sector': worst_sector, 'min_range': round(float(worst_val), 3),
            })
    return issues


def _check_oscillation(signal, window_size, max_reversals, min_amp, issue_type):
    """Sliding-window reversal detection for any 1D signal."""
    if len(signal) < 3:
        return []

    diff = np.abs(np.diff(signal))
    rate = np.diff(signal) / 0.1
    rate_filtered = np.where(diff >= min_amp, rate, 0.0)
    sign_changes = np.abs(np.diff(np.sign(rate_filtered)))

    issues = []
    flagged = set()
    for i in range(len(sign_changes) - window_size + 1):
        n_rev = np.sum(sign_changes[i:i + window_size] > 0)
        if n_rev > max_reversals:
            bucket = (i + window_size // 2) // window_size
            if bucket not in flagged:
                flagged.add(bucket)
                issues.append({
                    'type': issue_type,
                    'frame_start': i, 'frame_end': i + window_size,
                    'time_start': round(i * 0.1, 2), 'time_end': round((i + window_size) * 0.1, 2),
                    'reversals': int(n_rev),
                })
    return issues


def check_steering(steer):
    """Detect steering issues: oscillation, large jumps, and low autocorrelation."""
    window = 10         # frames = 1.0s
    max_reversals = 6   # max large reversals per window
    min_amp = 0.3       # rad, min amplitude to count as reversal
    max_jump = 0.6      # rad, max single-step change
    autocorr_min = -0.2 # lag-1 autocorrelation floor

    if len(steer) < 3:
        return []

    issues = _check_oscillation(steer, window, max_reversals, min_amp, 'steering_oscillation')

    # Large single-step jump
    steer_diff = np.abs(np.diff(steer))
    for idx in np.where(steer_diff > max_jump)[0]:
        issues.append({
            'type': 'steering_jump',
            'frame': int(idx), 'time': round(idx * 0.1, 2),
            'delta': round(float(np.diff(steer)[idx]), 3),
        })

    # Low autocorrelation (global oscillation pattern)
    if np.var(steer) >= min_amp ** 2:
        sc = steer - np.mean(steer)
        norm = np.sum(sc ** 2)
        if norm > 1e-10:
            autocorr = float(np.sum(sc[:-1] * sc[1:]) / norm)
            if autocorr < autocorr_min:
                issues.append({'type': 'low_steer_autocorrelation', 'autocorr_lag1': round(autocorr, 4)})

    return issues


# ── Metrics & Validation ─────────────────────────────────────────────────

def _load_episode(csv_path):
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 1], data[:, 2], data[:, 3:]  # steer, speed, lidar


def _max_window_reversals(signal, window_size, min_amp):
    """Max reversals in any window."""
    if len(signal) < 3:
        return 0
    diff = np.abs(np.diff(signal))
    rate = np.diff(signal) / 0.1
    rate_filtered = np.where(diff >= min_amp, rate, 0.0)
    sign_changes = np.abs(np.diff(np.sign(rate_filtered)))
    max_rev = 0
    for i in range(len(sign_changes) - window_size + 1):
        max_rev = max(max_rev, int(np.sum(sign_changes[i:i + window_size] > 0)))
    return max_rev


def compute_metrics(csv_path):
    """Compute raw metrics for one episode."""
    steer, speed, lidar = _load_episode(csv_path)
    surface_dist = np.clip(lidar - _precompute_d_edge(), 0.0, None)

    # beam 180 is front (±30° = beams 150-210), sides are the remaining beams
    sector_front = list(range(150, 210))
    sector_side = list(range(60, 120)) + list(range(240, 300))

    metrics = {
        'global_min_surface_dist': round(float(np.min(surface_dist)), 4),
        'side_min_surface_dist': round(float(np.min(surface_dist[:, sector_side])), 4),
        'front_min_surface_dist': round(float(np.min(surface_dist[:, sector_front])), 4),
        'speed_mean': round(float(np.mean(speed)), 4),
        'n_frames': len(steer),
        'duration': round(len(steer) * 0.1, 1),
        'max_steer_reversals': _max_window_reversals(steer, 10, 0.3),
        'max_speed_reversals': _max_window_reversals(speed, 10, 2.0),
    }

    if len(steer) >= 3:
        sc = steer - np.mean(steer)
        norm = np.sum(sc ** 2)
        metrics['steer_autocorr_lag1'] = round(float(np.sum(sc[:-1] * sc[1:]) / norm), 4) if norm > 1e-10 else 1.0
    else:
        metrics['steer_autocorr_lag1'] = 1.0

    return metrics


def validate_episode(csv_path):
    """Validate a single episode CSV."""
    steer, _, lidar = _load_episode(csv_path)

    all_issues = check_proximity(lidar) + check_steering(steer)
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


# ── Output ────────────────────────────────────────────────────────────────

def print_report(result):
    fname = result['file']
    if result['is_valid']:
        print(f"[PASS] {fname}")
        return

    print(f"[FAIL] {fname}")
    for itype, count in result['summary'].items():
        print(f"  {itype}: {count} events")

    for issue in result['issues'][:10]:
        if 'time_start' in issue:
            print(f"    t={issue['time_start']}-{issue['time_end']}s  {issue['type']}  reversals={issue.get('reversals', '')}")
        elif 'time' in issue:
            detail = issue.get('min_range', issue.get('delta', ''))
            print(f"    t={issue['time']}s  {issue['type']}  {detail}")
        else:
            detail = issue.get('autocorr_lag1', '')
            print(f"    {issue['type']}  {detail}")
    if len(result['issues']) > 10:
        print(f"  ... and {len(result['issues']) - 10} more issues")


def get_csv_files(directory):
    return sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv'))


def compute_percentiles(values):
    """Compute percentile distribution for a list of values."""
    arr = np.array(values)
    return {f'P{p}': round(float(np.percentile(arr, p)), 4) for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]}


def run_directory(scan_dir):
    """Validate all CSVs in directory, save analysis + percentiles to JSON."""
    csv_files = get_csv_files(scan_dir)
    if not csv_files:
        print(f"No CSV files found in {scan_dir}")
        sys.exit(1)

    print(f"Analyzing {len(csv_files)} episodes...")
    results = []
    for csv_path in csv_files:
        metrics = compute_metrics(csv_path)
        validation = validate_episode(csv_path)
        print_report(validation)
        results.append({
            'file': os.path.basename(csv_path),
            'metrics': metrics,
            'is_valid': validation['is_valid'],
            'issue_summary': validation['summary'],
            'issues': validation['issues'],
        })

    # Compute percentile distributions
    metric_keys = ['global_min_surface_dist', 'side_min_surface_dist', 'front_min_surface_dist', 'max_steer_reversals', 'max_speed_reversals', 'steer_autocorr_lag1']
    percentiles = {}
    for key in metric_keys:
        values = [r['metrics'][key] for r in results]
        percentiles[key] = compute_percentiles(values)

    # Sort by severity
    results.sort(key=lambda r: (r['is_valid'], -len(r['issues'])))

    fail_count = sum(1 for r in results if not r['is_valid'])
    pass_count = len(results) - fail_count

    output = {
        'summary': {'total': len(results), 'pass': pass_count, 'fail': fail_count},
        'percentiles': percentiles,
        'episodes': results,
    }

    out_path = os.path.join(os.path.dirname(scan_dir.rstrip('/')), 'episode_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Total: {len(results)}  Pass: {pass_count}  Fail: {fail_count}")
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Episode quality validator')
    parser.add_argument('--input_csv', type=str, required=True, help='CSV file or directory to validate')
    args = parser.parse_args()

    path = args.input_csv
    if os.path.isfile(path):
        result = validate_episode(path)
        print_report(result)
        sys.exit(0 if result['is_valid'] else 1)
    elif os.path.isdir(path):
        run_directory(path)
    else:
        print(f"Path not found: {path}")
        sys.exit(1)

if __name__ == '__main__':
    main()

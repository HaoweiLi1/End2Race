import argparse
import contextlib
import csv
import glob
import io
import json
import multiprocessing
import os
import re
import shutil
import sys
from collections import Counter

# Cap BLAS threads to 1 so --multidataset_dir workers don't oversubscribe CPU.
# Must be set before numpy imports its BLAS backend.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────

PROXIMITY_THRESHOLD = 0.15  # m, uniform over all sectors

# f1tenth_gym 360-beam mapping: angle_i = -π + i·(2π/N)
# beam 0 = rear (-180°), beam 180 = front (0°)
SECTORS = [
    ('rear',        list(range(0, 30)) + list(range(330, 360))),
    ('rear_right',  list(range(30, 60))),
    ('right',       list(range(60, 120))),
    ('front_right', list(range(120, 150))),
    ('front',       list(range(150, 210))),
    ('front_left',  list(range(210, 240))),
    ('left',        list(range(240, 300))),
    ('rear_left',   list(range(300, 330))),
]

# Steering: the 3 legacy rules are merged into one 'steering' fail category.
STEER_WINDOW        = 10    # frames = 1.0 s (oscillation window)
STEER_MAX_REVERSALS = 6     # fail if any window has more than this many reversals
STEER_MIN_AMP       = 0.3   # rad, reversal amplitude threshold
STEER_MAX_JUMP      = 0.6   # rad, single-step jump threshold
STEER_AUTOCORR_MIN  = -0.5  # lag-1 autocorrelation floor


# ── Geometry ─────────────────────────────────────────────────────────────

def _precompute_d_edge(n_beams=360):
    """LiDAR-to-car-surface distance per beam (subtracted from raw scan)."""
    # Car is a 0.58×0.31 rectangle; LiDAR at its geometric center.
    half_l = 0.58 / 2   # 0.290  (matches get_vertices in f1tenth_gym)
    half_w = 0.31 / 2   # 0.155
    angles = -np.pi + np.arange(n_beams) * (2 * np.pi / n_beams)
    c = np.abs(np.cos(angles)) + 1e-12
    s = np.abs(np.sin(angles)) + 1e-12
    return np.minimum(half_l / c, half_w / s)


def _load_episode(csv_path):
    """Return (steer[T], lidar[T, 360]) from a demonstration CSV."""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    return data[:, 1], data[:, 3:]


# ── Steering signal helpers ──────────────────────────────────────────────

def _max_window_reversals(signal, window_size, min_amp):
    """Max count of sign-flipping large changes over any window."""
    diff = np.abs(np.diff(signal))
    rate = np.diff(signal) / 0.1
    rate_filtered = np.where(diff >= min_amp, rate, 0.0)
    sign_changes = np.abs(np.diff(np.sign(rate_filtered)))
    max_rev = 0
    for i in range(len(sign_changes) - window_size + 1):
        max_rev = max(max_rev, int(np.sum(sign_changes[i:i + window_size] > 0)))
    return max_rev


def _steer_autocorr_lag1(steer):
    """lag-1 autocorrelation of the centered steering signal, in [-1, 1]."""
    sc = steer - np.mean(steer)
    norm = float(np.sum(sc ** 2))
    return float(np.sum(sc[:-1] * sc[1:]) / norm)


# ── Fail detectors ───────────────────────────────────────────────────────

def _proximity_fail(surface_dist):
    """True iff any beam in any frame is closer than threshold."""
    return float(np.min(surface_dist)) < PROXIMITY_THRESHOLD


def _steering_fail(steer):
    """True iff any of the 3 steering sub-rules trigger (OR-merged)."""
    # 1. Local oscillation: too many large reversals inside a 1-s window
    if _max_window_reversals(steer, STEER_WINDOW, STEER_MIN_AMP) > STEER_MAX_REVERSALS:
        return True
    # 2. Single-step jump
    if float(np.max(np.abs(np.diff(steer)))) > STEER_MAX_JUMP:
        return True
    # 3. Global sawtooth (low lag-1 autocorrelation), only meaningful when steer varies
    if float(np.var(steer)) >= STEER_MIN_AMP ** 2:
        if _steer_autocorr_lag1(steer) < STEER_AUTOCORR_MIN:
            return True
    return False


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(surface_dist, steer):
    """Four quality metrics for a single episode."""
    # Per-sector minima; include only those below threshold (the dangerous ones).
    danger = {}
    for name, idx in SECTORS:
        m = float(np.min(surface_dist[:, idx]))
        if m < PROXIMITY_THRESHOLD:
            danger[name] = round(m, 4)
    return {
        'global_min_surface_dist': round(float(np.min(surface_dist)), 4),
        'danger_sectors':          danger,
        'max_steer_reversals':     _max_window_reversals(steer, STEER_WINDOW, STEER_MIN_AMP),
        'steer_autocorr_lag1':     round(_steer_autocorr_lag1(steer), 4),
    }


# ── Validate one CSV ─────────────────────────────────────────────────────

def validate_episode(csv_path):
    steer, lidar = _load_episode(csv_path)
    surface_dist = np.clip(lidar - _precompute_d_edge(), 0.0, None)

    prox = _proximity_fail(surface_dist)
    steer_fail = _steering_fail(steer)

    if prox and steer_fail:
        status = 'FAIL (proximity + steering)'
    elif prox:
        status = 'FAIL (proximity)'
    elif steer_fail:
        status = 'FAIL (steering)'
    else:
        status = 'PASS'

    return {
        'file':    os.path.basename(csv_path),
        'status':  status,
        'metrics': compute_metrics(surface_dist, steer),
    }


def print_report(result):
    print(f"[{result['status']}] {result['file']}")
    for k, v in result['metrics'].items():
        print(f"  {k:26s} = {v}")


# ── Validate a directory ─────────────────────────────────────────────────

METRIC_FIELDS = ('global_min_surface_dist', 'danger_sectors', 'max_steer_reversals', 'steer_autocorr_lag1')
PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 100]
NUMERIC_METRICS = ('global_min_surface_dist', 'max_steer_reversals', 'steer_autocorr_lag1')


def _gather_csvs(*dirs):
    """Return sorted list of all .csv paths across the given directories."""
    out = []
    for d in dirs:
        if os.path.isdir(d):
            out += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.csv')]
    return sorted(out)


def _move_pair(src_dir, dst_dir, csv_name):
    """Move a CSV and its matching MP4 (if any) from src_dir to dst_dir.
    No-op if already in place."""
    if src_dir == dst_dir:
        return
    base, _ = os.path.splitext(csv_name)
    for fname in (csv_name, base + '.mp4'):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(src):
            shutil.move(src, dst)


def validate_directory(input_dir):
    """Re-validate and re-classify every CSV in <dir>/success and <dir>/low_quality.

    Idempotent: rescanning an already-classified dir yields the same layout.
    Writes <dir>/fails.csv summarising every failure with its 4 metrics.
    """
    success_dir       = os.path.join(input_dir, 'success')
    low_quality_dir   = os.path.join(input_dir, 'low_quality')
    fails_csv_path    = os.path.join(input_dir, 'fails.csv')
    success_csv_path  = os.path.join(input_dir, 'success.csv')

    # 1. Collect inputs from BOTH subdirs so we ignore prior classification.
    all_csvs = _gather_csvs(success_dir, low_quality_dir)

    # 2. Validate every file.
    results = [(path, validate_episode(path)) for path in all_csvs]

    # 3. Re-classify (create target dirs on demand).
    os.makedirs(success_dir, exist_ok=True)
    for path, r in results:
        target = success_dir if r['status'] == 'PASS' else low_quality_dir
        if target == low_quality_dir:
            os.makedirs(low_quality_dir, exist_ok=True)
        _move_pair(os.path.dirname(path), target, os.path.basename(path))

    # 4. Write fails.csv (fails only, with status) and success.csv (passes, no status column).
    fails   = [r for _, r in results if r['status'] != 'PASS']
    passes  = [r for _, r in results if r['status'] == 'PASS']

    def _metric_row(r):
        m = r['metrics']
        return [m['global_min_surface_dist'],
                json.dumps(m['danger_sectors']),
                m['max_steer_reversals'],
                m['steer_autocorr_lag1']]

    with open(fails_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'status', *METRIC_FIELDS])
        for r in fails:
            writer.writerow([r['file'], r['status'], *_metric_row(r)])

    with open(success_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', *METRIC_FIELDS])
        for r in passes:
            writer.writerow([r['file'], *_metric_row(r)])

    # 5. Summary.
    total = len(results)
    n_pass = sum(1 for _, r in results if r['status'] == 'PASS')
    print(f"Total: {total}   Pass: {n_pass}   Fail: {len(fails)}")
    for reason, count in Counter(r['status'] for r in fails).most_common():
        print(f"  {reason}: {count}")

    # 6. Percentile distribution of numeric metrics (across all episodes).
    print("\nPercentiles (across all episodes):")
    header = "  {:26s}".format("") + "".join(f"{'P'+str(p):>8s}" for p in PERCENTILES)
    print(header)
    for key in NUMERIC_METRICS:
        values = np.array([r['metrics'][key] for _, r in results], dtype=float)
        pct = np.percentile(values, PERCENTILES)
        row = "  {:26s}".format(key) + "".join(f"{v:8.3f}" for v in pct)
        print(row)

    print(f"\nFail summary    → {fails_csv_path}")
    print(f"Success summary → {success_csv_path}")


# ── Multi-dataset (parallel over cw* subdirs) ────────────────────────────

def _validate_one_dir_worker(input_dir):
    """Worker: run validate_directory capturing stdout/stderr. Returns (input_dir, output)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        validate_directory(input_dir)
    return input_dir, buf.getvalue()


def validate_multidataset(root_dir, workers):
    """Iterate all cw* subdirs under root_dir in parallel; merge output into <root_dir>/validate.log."""
    dirs = sorted(
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if d.startswith('cw') and os.path.isdir(os.path.join(root_dir, d))
    )
    total = len(dirs)
    log_path = os.path.join(root_dir, 'validate.log')
    print(f"Batch Validation (workers={workers}, total={total})  →  {log_path}")
    print("=" * 60)

    with open(log_path, 'w') as log, multiprocessing.Pool(workers) as pool:
        for i, (d, out) in enumerate(pool.imap_unordered(_validate_one_dir_worker, dirs), 1):
            name = os.path.basename(d)
            log.write(f"\n==== [{i}/{total}] {name} ====\n")
            log.write(out)
            log.flush()
            print(f"[{i}/{total}] {name} done")

    print("=" * 60)
    print(f"All {total} directories done.  Log: {log_path}")


# ── Dataset manifest selection (5 modes) ────────────────────────────────

# Filename: '{f|o}_ol{raceline}_e{ego_idx}_o{opp_idx}_s{speed}.csv'
_FILENAME_RE = re.compile(r'^([of])_ol(\d+)_e(\d+)_o(\d+)_s([\d.]+)\.csv$')

# Reversal penalty in best_per_segment score (autocorr − β · reversals/MAX).
SCORE_BETA = 0.5


def _parse_episode_filename(fname):
    m = _FILENAME_RE.match(fname)
    return m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(5))


def _cw_success_counts(root):
    cw_dirs = [d for d in os.listdir(root) if d.startswith('cw') and os.path.isdir(os.path.join(root, d))]
    return {d: len(glob.glob(os.path.join(root, d, 'success', '*.csv'))) for d in cw_dirs}


def _select_best_group(root):
    counts = _cw_success_counts(root)
    best = max(counts, key=counts.get)
    paths = sorted(glob.glob(os.path.join(root, best, 'success', '*.csv')))
    print(f"[best_group] {best}: {len(paths)} episodes")
    return paths


def _select_best_200(root, top_n=200):
    """Top-N cw*/ groups by PASS count. ~107k episodes ≈ 13 GB on GPU."""
    counts = _cw_success_counts(root)
    top_dirs = sorted(counts, key=counts.get, reverse=True)[:top_n]
    paths = sorted(p for d in top_dirs for p in glob.glob(os.path.join(root, d, 'success', '*.csv')))
    print(f"[best_200] top-{top_n}/{len(counts)}: {len(paths)} episodes")
    return paths


def _select_best_per_segment(root, stratify):
    """Per segment, pick max-score episode. stratify ∈ {'follow','overtake','both'}."""
    frames = []
    for sc in sorted(glob.glob(os.path.join(root, 'cw*', 'success.csv'))):
        df = pd.read_csv(sc)
        df['cw_dir'] = os.path.basename(os.path.dirname(sc))
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)

    big[['outcome', 'opp_raceline', 'ego_idx', 'opp_speed']] = big['file'].apply(
        lambda f: pd.Series(_parse_episode_filename(f)))
    big['score'] = big['steer_autocorr_lag1'] - SCORE_BETA * (big['max_steer_reversals'] / STEER_MAX_REVERSALS)

    so_keys = ['opp_raceline', 'ego_idx', 'opp_speed', 'outcome']
    per_so  = big.loc[big.groupby(so_keys)['score'].idxmax()].reset_index(drop=True)

    if stratify == 'both':
        winners = per_so
    else:
        # 'follow'/'overtake': prefer that outcome per segment; fill missing with the other.
        preferred = 'f' if stratify == 'follow' else 'o'
        fallback  = 'o' if stratify == 'follow' else 'f'
        seg_keys  = ['opp_raceline', 'ego_idx', 'opp_speed']
        pref = per_so[per_so['outcome'] == preferred]
        covered = set(map(tuple, pref[seg_keys].values))
        fb = per_so[per_so['outcome'] == fallback]
        fb = fb[~fb[seg_keys].apply(tuple, axis=1).isin(covered)]
        winners = pd.concat([pref, fb], ignore_index=True)

    label = {'follow': 'follow_first', 'overtake': 'overtake_first', 'both': 'merge'}[stratify]
    n_f = int((winners['outcome'] == 'f').sum())
    n_o = int((winners['outcome'] == 'o').sum())
    print(f"[{label}] {len(winners)} episodes (f={n_f}, o={n_o})")
    return [os.path.join(root, r['cw_dir'], 'success', r['file']) for _, r in winners.iterrows()]


def _write_manifest(root, mode, paths):
    manifest_path = os.path.join(root, f'manifest_{mode}.csv')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path'])
        for p in paths:
            writer.writerow([os.path.relpath(p, root)])
    print(f"  manifest → {manifest_path} ({len(paths)} rows)")


def write_manifests(root):
    print("\nWriting dataset manifests")
    _write_manifest(root, 'best_group',     _select_best_group(root))
    _write_manifest(root, 'best_200',       _select_best_200(root))
    _write_manifest(root, 'follow_first',   _select_best_per_segment(root, 'follow'))
    _write_manifest(root, 'overtake_first', _select_best_per_segment(root, 'overtake'))
    _write_manifest(root, 'merge',          _select_best_per_segment(root, 'both'))


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Episode quality validator')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_csv', type=str)
    group.add_argument('--input_dir', type=str)
    group.add_argument('--multidataset_dir', type=str, help='Parent dir with cw*/ groups; validates each and writes 5 manifests at the root')
    args = parser.parse_args()

    if args.input_csv:
        result = validate_episode(args.input_csv)
        print_report(result)
        sys.exit(0 if result['status'] == 'PASS' else 1)

    if args.input_dir:
        validate_directory(args.input_dir)
        return

    validate_multidataset(args.multidataset_dir, 8)
    write_manifests(args.multidataset_dir)


if __name__ == '__main__':
    main()

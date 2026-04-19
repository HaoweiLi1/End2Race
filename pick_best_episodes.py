"""
Pick the best lattice_params for each segment and copy to output dir.

Selection criteria:
1. Status must be 'o' (overtake) or 'f' (follow) — i.e., success
2. No status preference — overtake and follow are treated equally
3. Among all candidates, rank by combined score:
   score = w_safety * min_surface_dist
         + w_speed  * speed_mean
         - w_steer_rev  * max_steer_reversals
         - w_speed_rev  * max_speed_reversals
         + w_autocorr   * steer_autocorr_lag1
   Higher score = better (safer + smoother + faster).
"""

import os
import csv
import json
import shutil
import argparse
import glob


def load_status_matrix(matrix_path):
    """Load status_matrix.csv -> {segment: {lattice_params: status}}"""
    with open(matrix_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        segments = header[1:]

        status = {seg: {} for seg in segments}
        for row in reader:
            lp = row[0]
            for i, seg in enumerate(segments):
                if i + 1 < len(row) and row[i + 1].strip():
                    status[seg][lp] = row[i + 1].strip()
    return segments, status


def load_episode_metrics(base_dir):
    """Load all episode_analysis.json -> {lattice_params: {filename: metrics}}"""
    all_metrics = {}
    for cw_dir in sorted(glob.glob(os.path.join(base_dir, "cw*"))):
        lp = os.path.basename(cw_dir)
        analysis_path = os.path.join(cw_dir, "episode_analysis.json")
        if not os.path.exists(analysis_path):
            continue
        with open(analysis_path) as f:
            data = json.load(f)
        metrics_map = {}
        for ep in data["episodes"]:
            metrics_map[ep["file"]] = ep["metrics"]
        all_metrics[lp] = metrics_map
    return all_metrics


def compute_score(metrics, weights):
    """Compute a quality score from metrics. Higher = better."""
    if not metrics:
        return float("-inf")
    return (
        weights["safety"]    * metrics["global_min_surface_dist"]
        + weights["speed"]     * metrics["speed_mean"]
        - weights["steer_rev"] * metrics["max_steer_reversals"]
        - weights["speed_rev"] * metrics["max_speed_reversals"]
        + weights["autocorr"]  * metrics["steer_autocorr_lag1"]
    )


def pick_best(segments, status, all_metrics, base_dir, weights):
    """For each segment, pick the highest-scoring lattice_params."""
    results = {}
    stats = {"o_picked": 0, "f_picked": 0, "no_candidate": 0}

    for seg in segments:
        candidates = []
        for lp, st in status[seg].items():
            if st not in ("o", "f"):
                continue

            filename = f"{st}_{seg}.csv"
            src_path = os.path.join(base_dir, lp, "success", filename)
            if not os.path.exists(src_path):
                continue

            metrics = None
            if lp in all_metrics and filename in all_metrics[lp]:
                metrics = all_metrics[lp][filename]

            candidates.append({
                "lattice_params": lp,
                "status": st,
                "filename": filename,
                "metrics": metrics,
                "score": compute_score(metrics, weights),
            })

        if not candidates:
            stats["no_candidate"] += 1
            continue

        # Pick highest score; no preference between 'o' and 'f'
        candidates.sort(key=lambda c: -c["score"])
        best = candidates[0]
        results[seg] = best

        if best["status"] == "o":
            stats["o_picked"] += 1
        else:
            stats["f_picked"] += 1

    return results, stats


def copy_picked(results, base_dir, output_dir):
    """Copy selected CSV files to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    copied = 0
    missing = 0
    for seg, info in sorted(results.items()):
        lp = info["lattice_params"]
        filename = info["filename"]
        src = os.path.join(base_dir, lp, "success", filename)

        if not os.path.exists(src):
            print(f"  [MISSING] {src}")
            missing += 1
            continue

        dst = os.path.join(output_dir, filename)
        shutil.copy2(src, dst)
        copied += 1

    return copied, missing


def main():
    parser = argparse.ArgumentParser(description="Pick best (most stable) episode per segment")
    parser.add_argument("--base_dir", type=str, default="Dataset_Austin")
    parser.add_argument("--output_dir", type=str, default="Dataset_Austin1")
    # Stability-focused defaults: smoothness > safety > speed
    parser.add_argument("--w_safety",    type=float, default=1.0)
    parser.add_argument("--w_speed",     type=float, default=0.05)
    parser.add_argument("--w_steer_rev", type=float, default=0.3)
    parser.add_argument("--w_speed_rev", type=float, default=0.2)
    parser.add_argument("--w_autocorr",  type=float, default=1.0)
    args = parser.parse_args()

    weights = {
        "safety":    args.w_safety,
        "speed":     args.w_speed,
        "steer_rev": args.w_steer_rev,
        "speed_rev": args.w_speed_rev,
        "autocorr":  args.w_autocorr,
    }

    matrix_path = os.path.join(args.base_dir, "status_matrix.csv")
    print(f"Loading status matrix from {matrix_path}")
    segments, status = load_status_matrix(matrix_path)
    print(f"  {len(segments)} segments")

    print(f"Loading episode metrics...")
    all_metrics = load_episode_metrics(args.base_dir)
    print(f"  {len(all_metrics)} lattice_params with metrics")

    print(f"Picking best episodes with weights: {weights}")
    results, stats = pick_best(segments, status, all_metrics, args.base_dir, weights)

    print(f"\n{'='*50}")
    print(f"Total segments:    {len(segments)}")
    print(f"Overtake picked:   {stats['o_picked']}")
    print(f"Follow picked:     {stats['f_picked']}")
    print(f"No candidate:      {stats['no_candidate']}")
    print(f"Total picked:      {stats['o_picked'] + stats['f_picked']}")
    print(f"{'='*50}")

    print(f"\nCopying to {args.output_dir}/ ...")
    copied, missing = copy_picked(results, args.base_dir, args.output_dir)
    print(f"  Copied: {copied}, Missing: {missing}")

    log_path = os.path.join(args.output_dir, "selection_log.json")
    log_data = {
        "config": weights,
        "stats": stats,
        "selections": {
            seg: {
                "lattice_params": info["lattice_params"],
                "status": info["status"],
                "filename": info["filename"],
                "score": round(info["score"], 4),
                "metrics": info["metrics"],
            }
            for seg, info in sorted(results.items())
        },
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"  Selection log saved to {log_path}")


if __name__ == "__main__":
    main()

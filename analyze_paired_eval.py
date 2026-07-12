#!/usr/bin/env python3
"""Paired episode analysis: BC canonical vs a candidate policy on matched OL1 keys.

Implements the analysis required by logs/ppo_external_review_synthesis_20260707.md:
  - outcome transition matrix (BC -> candidate)
  - changed-episode sets (fixed_collision / new_collision / lost_overtake /
    gained_overtake) with speedscale breakdown
  - McNemar discordant-pair counts for collision and overtake
  - trajectory metrics for changed episodes, recomputing approximate
    front_risk / side_risk / side_risk_gate from saved poses/progress with the
    D4-A training reward parameters.

Usage:
  python analyze_paired_eval.py <bc_eval_dir> <candidate_eval_dir> [--out report.md]

Both dirs are evaluate_ol1.sh outputs (collision/follow/overtake subdirs with
<prefix>_<key>.npz). Pure offline analysis: no training, no new rollouts.
"""

import argparse
import glob
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

# D4-A training reward geometry defaults (ppo_utils.RewardWeights + pipeline overrides).
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31
LATERAL_MARGIN = 0.20
FRONT_BASE_MARGIN = 0.9
TIME_GAP = 0.8
SIDE_LONG_MARGIN = 0.75
SIDE_GATE_EDGE_MARGIN = 0.2
DT = 0.01

STATE_BY_DIR = {"collision": "collision", "follow": "follow", "overtake": "overtake"}


def index_episodes(eval_dir):
    """Map episode key -> (outcome, npz_path)."""
    out = {}
    for sub, state in STATE_BY_DIR.items():
        for p in glob.glob(os.path.join(eval_dir, sub, "*.npz")):
            key = os.path.basename(p)[2:-4]  # strip 'c_'/'f_'/'o_' prefix and '.npz'
            out[key] = (state, p)
    return out


def speedscale_of(key):
    try:
        return key.rsplit("_s", 1)[1]
    except IndexError:
        return "?"


def traj_metrics(path, cfg):
    z = np.load(path, allow_pickle=True)
    rel = (z["opp_progress"] - z["ego_progress"]).astype(np.float64)  # >0: ego behind
    ego = z["ego_pose"][:, :2].astype(np.float64)
    opp = z["opp_pose"][:, :2].astype(np.float64)
    dist = np.linalg.norm(ego - opp, axis=1)
    lat = np.sqrt(np.maximum(dist ** 2 - np.minimum(np.abs(rel), dist) ** 2, 0.0))

    # Track-frame speeds from progress derivative (smoothed lightly).
    def dspeed(x):
        d = np.gradient(x) / DT
        if len(d) >= 9:
            k = np.ones(9) / 9.0
            d = np.convolve(d, k, mode="same")
        return d

    ego_v = dspeed(z["ego_progress"].astype(np.float64))
    opp_v = dspeed(z["opp_progress"].astype(np.float64))

    # Recomputed risks with TRAINING parameters (rel>0 == ego behind here).
    car_length = cfg["car_length"]
    car_width = cfg["car_width"]
    lateral_margin = cfg["lateral_margin"]
    front_base_margin = cfg["front_base_margin"]
    time_gap = cfg["time_gap"]
    side_longitudinal_margin = cfg["side_longitudinal_margin"]
    side_gate_edge_margin = cfg["side_gate_edge_margin"]

    lat_safe = car_width + lateral_margin
    lo_base = np.clip((lat_safe - lat) / lat_safe, 0.0, 1.0)
    edge = np.maximum(0.0, lat - car_width)
    lo_edge = np.clip((side_gate_edge_margin - edge) / side_gate_edge_margin, 0.0, 1.0)
    closing = np.maximum(0.0, ego_v - opp_v)
    req_front = car_length + front_base_margin + time_gap * closing
    front_gap = np.maximum(0.0, rel)
    front_risk = np.where(rel > 0.0, lo_base * np.clip((req_front - front_gap) / req_front, 0.0, 1.0), 0.0)
    side_gap = car_length + side_longitudinal_margin
    long_ov = np.clip((side_gap - np.abs(rel)) / side_gap, 0.0, 1.0)
    side_risk = long_ov * lo_base
    side_risk_gate = long_ov * lo_edge

    alongside = np.abs(rel) < 0.6
    i_min = int(np.argmin(dist))
    final_rel = float(rel[-1])
    phase = "alongside" if abs(final_rel) < 0.6 else ("pre" if final_rel > 0 else "post")

    return {
        "len": len(rel),
        "final_rel_s": final_rel,
        "phase": phase,
        "min_dist": float(dist.min()),
        "min_lat_alongside": float(lat[alongside].min()) if alongside.any() else float("nan"),
        "rel_at_min_dist": float(rel[i_min]),
        "mean_front_risk": float(front_risk.mean()),
        "max_front_risk": float(front_risk.max()),
        "mean_side_risk": float(side_risk.mean()),
        "mean_side_gate": float(side_risk_gate.mean()),
        "max_side_gate": float(side_risk_gate.max()),
        "frac_gate_active": float((side_risk_gate > 0.5).mean()),
        "desired_speed": z["ego_desired_speed"].astype(np.float64),
        "desired_steer": z["ego_desired_steer"].astype(np.float64),
    }


def pair_diffs(cand_path, bc_path, cfg):
    c = traj_metrics(cand_path, cfg)
    b = traj_metrics(bc_path, cfg)
    n = min(c["len"], b["len"])
    dv = float(np.mean(c["desired_speed"][:n] - b["desired_speed"][:n]))
    dst = float(np.mean(np.abs(c["desired_steer"][:n] - b["desired_steer"][:n])))
    cand_steer = c["desired_steer"][:n]
    bc_steer = b["desired_steer"][:n]
    cand_delta = np.diff(cand_steer)
    max_steer_delta = float(np.max(np.abs(cand_delta))) if cand_delta.size else 0.0
    reversals = int(np.sum((cand_delta[:-1] * cand_delta[1:]) < 0.0)) if cand_delta.size >= 2 else 0
    max_pair_steer_delta = float(np.max(np.abs(cand_steer - bc_steer))) if n else 0.0
    return c, b, dv, dst, max_steer_delta, reversals, max_pair_steer_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bc_dir")
    ap.add_argument("cand_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--car_length", type=float, default=CAR_LENGTH)
    ap.add_argument("--car_width", type=float, default=CAR_WIDTH)
    ap.add_argument("--lateral_margin", type=float, default=LATERAL_MARGIN)
    ap.add_argument("--front_base_margin", type=float, default=FRONT_BASE_MARGIN)
    ap.add_argument("--time_gap", type=float, default=TIME_GAP)
    ap.add_argument("--side_longitudinal_margin", type=float, default=SIDE_LONG_MARGIN)
    ap.add_argument("--side_gate_edge_margin", type=float, default=SIDE_GATE_EDGE_MARGIN)
    args = ap.parse_args()
    cfg = {
        "car_length": float(args.car_length),
        "car_width": float(args.car_width),
        "lateral_margin": float(args.lateral_margin),
        "front_base_margin": float(args.front_base_margin),
        "time_gap": float(args.time_gap),
        "side_longitudinal_margin": float(args.side_longitudinal_margin),
        "side_gate_edge_margin": float(args.side_gate_edge_margin),
    }

    bc = index_episodes(args.bc_dir)
    cand = index_episodes(args.cand_dir)
    keys = sorted(set(bc) & set(cand))
    lines = []
    w = lines.append
    w(f"# Paired episode analysis")
    w(f"- BC:        `{args.bc_dir}`  ({len(bc)} eps)")
    w(f"- candidate: `{args.cand_dir}`  ({len(cand)} eps)")
    w(f"- matched keys: {len(keys)}")
    if len(keys) < len(bc) or len(keys) < len(cand):
        w(f"- WARNING: unmatched keys bc-only={len(set(bc)-set(cand))} cand-only={len(set(cand)-set(bc))}")
    w("")
    w("## Risk Config")
    for key in (
        "car_length",
        "car_width",
        "lateral_margin",
        "front_base_margin",
        "time_gap",
        "side_longitudinal_margin",
        "side_gate_edge_margin",
    ):
        w(f"- {key}: {cfg[key]}")
    w("")

    # Transition matrix.
    states = ["collision", "follow", "overtake"]
    tm = Counter((bc[k][0], cand[k][0]) for k in keys)
    w("## Transition matrix (rows: BC, cols: candidate)")
    w("| BC \\ cand | collision | follow | overtake |")
    w("|---|---:|---:|---:|")
    for r in states:
        w(f"| {r} | " + " | ".join(str(tm.get((r, c), 0)) for c in states) + " |")
    w("")

    sets = {
        "fixed_collision": [k for k in keys if bc[k][0] == "collision" and cand[k][0] != "collision"],
        "new_collision": [k for k in keys if bc[k][0] != "collision" and cand[k][0] == "collision"],
        "lost_overtake": [k for k in keys if bc[k][0] == "overtake" and cand[k][0] != "overtake"],
        "gained_overtake": [k for k in keys if bc[k][0] != "overtake" and cand[k][0] == "overtake"],
    }

    # McNemar discordant pairs (binomial sign test, two-sided).
    def mcnemar(a, b):
        n = a + b
        if n == 0:
            return 1.0
        from math import comb
        p = sum(comb(n, i) for i in range(0, min(a, b) + 1)) / 2 ** n * 2
        return min(1.0, p)

    w("## Discordant pairs (McNemar sign test)")
    w(f"- collision: fixed={len(sets['fixed_collision'])} new={len(sets['new_collision'])} "
      f"p={mcnemar(len(sets['fixed_collision']), len(sets['new_collision'])):.3f}")
    w(f"- overtake:  gained={len(sets['gained_overtake'])} lost={len(sets['lost_overtake'])} "
      f"p={mcnemar(len(sets['gained_overtake']), len(sets['lost_overtake'])):.3f}")
    fixed_collision = len(sets["fixed_collision"])
    new_collision = len(sets["new_collision"])
    gained_overtake = len(sets["gained_overtake"])
    lost_overtake = len(sets["lost_overtake"])
    if gained_overtake == 0:
        collision_per_gain = float("inf") if new_collision > 0 else 0.0
    else:
        collision_per_gain = new_collision / gained_overtake
    w("")
    w("## Net Changes")
    w(f"- net_collision_improvement: {fixed_collision - new_collision} "
      f"(fixed {fixed_collision} - new {new_collision})")
    w(f"- net_overtake_improvement: {gained_overtake - lost_overtake} "
      f"(gained {gained_overtake} - lost {lost_overtake})")
    w(f"- new_collision_per_gained_overtake: {collision_per_gain:.3g}")
    w("")

    # Speedscale breakdown.
    w("## Changed-episode sets (speedscale breakdown)")
    for name, ks in sets.items():
        by_ss = Counter(speedscale_of(k) for k in ks)
        w(f"### {name} ({len(ks)})")
        w(f"- by speedscale: {dict(sorted(by_ss.items()))}")
        for k in ks:
            w(f"- `{k}`: BC={bc[k][0]} -> cand={cand[k][0]}")
        w("")

    # Trajectory metrics for changed episodes.
    w("## Trajectory metrics for changed episodes (candidate vs BC, training-param risks)")
    w("| set | key | cand phase | cand min_lat@along | cand max_side_gate | cand max_front_risk | dv_desired (m/s) | |dsteer| mean | max cand Δsteer | reversals | max cand-BC steer | BC min_lat@along |")
    w("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, ks in sets.items():
        for k in ks:
            try:
                c, b, dv, dst, max_ds, reversals, max_pair_ds = pair_diffs(cand[k][1], bc[k][1], cfg)
                w(f"| {name} | `{k}` | {c['phase']} | {c['min_lat_alongside']:.3f} | {c['max_side_gate']:.2f} | "
                  f"{c['max_front_risk']:.2f} | {dv:+.3f} | {dst:.4f} | {max_ds:.4f} | {reversals} | "
                  f"{max_pair_ds:.4f} | {b['min_lat_alongside']:.3f} |")
            except Exception as e:  # noqa: BLE001
                w(f"| {name} | `{k}` | ERROR {e} | | | | | | | | | | |")
    w("")

    # Aggregate exposure comparison over ALL matched keys (context for gate strength).
    agg = defaultdict(list)
    for k in keys:
        try:
            c = traj_metrics(cand[k][1], cfg); b = traj_metrics(bc[k][1], cfg)
            agg["cand_gate"].append(c["mean_side_gate"]); agg["bc_gate"].append(b["mean_side_gate"])
            agg["cand_fr"].append(c["mean_front_risk"]); agg["bc_fr"].append(b["mean_front_risk"])
        except Exception:  # noqa: BLE001
            pass
    if agg["cand_gate"]:
        w("## All-episode exposure (means over 200 eps, training-param risks)")
        w(f"- mean side_risk_gate: cand {np.mean(agg['cand_gate']):.4f} vs BC {np.mean(agg['bc_gate']):.4f}")
        w(f"- mean front_risk:     cand {np.mean(agg['cand_fr']):.4f} vs BC {np.mean(agg['bc_fr']):.4f}")

    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()

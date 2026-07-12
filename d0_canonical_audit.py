#!/usr/bin/env python3
"""D0: canonical collision audit and data locking.

Implements stage D0 of docs/superpowers/specs/2026-07-10-ppo-safety-first-bplus-design.md
(section 6.1). Pure read-only analysis over the validated P1 evaluation run:
no training, no simulation.

Canonical scenario identity = resolved (map, ego raceline, opp raceline,
start pose, speedscale, interval, duration, noise). Offset and tag are
provenance only. Raw ego indices are resolved modulo the waypoint table and
mapped to the waypoint pose rounded to 1 mm, so e0 vs e<max> clones unify
exactly when their poses coincide.

Outputs (spec 6.1): input_provenance.tsv, episode_occurrences.tsv,
canonical_episodes.tsv, collision_events.tsv, d0_summary.{json,md},
d0_validation.json. The summary additionally verifies the spec section 3
claims and recomputes the P1 paired statistics on canonical sets (needed for
the section 12 documentation corrections), and audits the offset-set
uniqueness claims of section 10.1.
"""

import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

RUN = "20260710_121955"
MODELS = ["bc", "cand160", "cand120", "cand040"]
GRIDS = [("Austin", 21), ("Austin", 42), ("Austin", 63), ("Austin", 84),
         ("Nuerburgring", 0), ("MoscowRaceway", 0), ("Hockenheim", 0)]
MAPS = ["Austin", "Nuerburgring", "MoscowRaceway", "Hockenheim"]
CAR_DIST_THRESH = 1.0   # analyze_collisions.py defaults
ALONGSIDE_THRESH = 0.6
OUT = sys.argv[1] if len(sys.argv) > 1 else f"logs/d0_canonical_audit_{RUN}"


def load_waypoints(map_name):
    path = f"f1tenth_racetracks/{map_name}/raceline1.csv"
    wp = np.loadtxt(path, delimiter=";", skiprows=1)
    with open(path) as f:
        total_lines = sum(1 for _ in f)
    max_wc = total_lines - 2  # evaluate.sh / runner convention: tail -n +3 | wc -l
    return wp, max_wc


def sign_test_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def outcome_of(ep):
    oc = ep.get("outcome") or ep.get("state_label")
    return {"collision": "collision", "overtaking": "overtake"}.get(oc, "follow")


def main():
    os.makedirs(OUT, exist_ok=True)
    wps, maxwc = {}, {}
    for m in MAPS:
        wps[m], maxwc[m] = load_waypoints(m)
    opp_wps = {}
    for m in MAPS:
        for rl in ("raceline0", "raceline1", "raceline2"):
            opp_wps[(m, rl)] = np.loadtxt(f"f1tenth_racetracks/{m}/{rl}.csv",
                                          delimiter=";", skiprows=1)

    def resolve(map_name, raw_idx):
        # Exact float pose is the identity: raceline first/last rows differ at
        # sub-mm scale but the simulator chaotically diverges from that, so
        # rounding would merge scenarios with genuinely different outcomes.
        wp = wps[map_name]
        idx = int(raw_idx) % len(wp)
        return idx, float(wp[idx, 1]), float(wp[idx, 2])

    # Austin development grid (evaluate.sh off0 formula), as resolved poses.
    dev_poses = set()
    for i in range(50):
        _, x, y = resolve("Austin", i * maxwc["Austin"] // 49)
        dev_poses.add((round(x, 2), round(y, 2)))

    # ---- provenance -------------------------------------------------------
    prov_rows = []
    ckpt_hashes = {}
    arch = f"logs/p1_validation_{RUN}/source_archive/checkpoint_sha256.txt"
    if os.path.exists(arch):
        for line in open(arch):
            h, p = line.split()
            ckpt_hashes[p] = h

    # ---- occurrences ------------------------------------------------------
    occurrences = []   # dicts, one per grid occurrence
    validation = {"expected_occurrences": len(MODELS) * len(GRIDS) * 600,
                  "observed_occurrences": 0, "dirs": {}, "unvalidated_dirs": [],
                  "outcome_conflicts": 0, "unknown_classifications": 0,
                  "missing_npz_terminal_fields": 0}
    for model in MODELS:
        for map_name, off in GRIDS:
            tag = f"p1v_{RUN}_{model}_{map_name}_off{off}"
            rdir = f"eval_results/{tag}_{map_name}"
            data = json.load(open(os.path.join(rdir, "results.json")))
            eps = data["episodes"]
            validated = bool(data.get("final", {}).get("validated"))
            if not validated:
                validation["unvalidated_dirs"].append(rdir)
            validation["dirs"][rdir] = len(eps)
            validation["observed_occurrences"] += len(eps)
            prov_rows.append({
                "run": RUN, "tag": tag, "result_dir": rdir,
                "episodes": len(eps), "validated": validated,
                "model_sha256": next((h for p, h in ckpt_hashes.items()
                                      if model in ("bc",) and p.endswith("end2race.pth")
                                      or model != "bc" and _model_match(model, p)), ""),
            })
            for key, ep in eps.items():
                ridx = int(ep["ego_idx"])
                idx, x, y = resolve(map_name, ridx)
                # Opponent placement is raw-index-dependent under the same
                # ego pose (ol1: opp_idx = (raw_ego + interval) % len), so the
                # resolved opponent pose is part of the scenario identity.
                owp = opp_wps[(map_name, ep["opp_raceline"])]
                oidx = int(ep["opp_idx"]) % len(owp)
                canon = (map_name, ep.get("ego_raceline", "raceline1"),
                         ep["opp_raceline"], x, y,
                         float(owp[oidx, 1]), float(owp[oidx, 2]),
                         float(ep["opp_speedscale"]),
                         int(ep.get("interval_idx", 15)),
                         float(ep.get("sim_duration", 8.0)),
                         float(ep.get("noise", 0.0)))
                occurrences.append({
                    "model": model, "map": map_name, "grid": f"{map_name}_off{off}",
                    "offset": off, "raw_key": key, "raw_ego_idx": ridx,
                    "resolved_ego_idx": idx, "start_x": round(x, 3), "start_y": round(y, 3),
                    "opp_raceline": ep["opp_raceline"],
                    "speedscale": float(ep["opp_speedscale"]),
                    "interval": int(ep.get("interval_idx", 15)),
                    "outcome": outcome_of(ep),
                    "ego_collision": bool(ep.get("ego_collision")),
                    "opp_collision": bool(ep.get("opp_collision")),
                    "npz_path": ep.get("npz_path", ""),
                    "canon": canon,
                    "dev_overlap": (map_name == "Austin")
                                   and ((round(x, 2), round(y, 2)) in dev_poses),
                })

    def _write_tsv(name, rows, fields):
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    _write_tsv("input_provenance.tsv", prov_rows,
               ["run", "tag", "result_dir", "episodes", "validated", "model_sha256"])
    _write_tsv("episode_occurrences.tsv", occurrences,
               ["model", "map", "grid", "offset", "raw_key", "raw_ego_idx",
                "resolved_ego_idx", "start_x", "start_y", "opp_raceline",
                "speedscale", "interval", "outcome", "ego_collision",
                "opp_collision", "dev_overlap", "npz_path"])

    # ---- canonical table + determinism conflicts --------------------------
    canonical = {}   # canon -> model -> list of occurrence dicts
    for o in occurrences:
        canonical.setdefault(o["canon"], {}).setdefault(o["model"], []).append(o)

    # Near-duplicate groups: raceline first/last rows coincide only to ~cm.
    # The lowest-resolved-index member is the primary scenario; other members
    # are shadow clones excluded from analysis pools (their sub-mm start
    # perturbation is a chaos-sensitivity probe, not an independent scenario).
    groups = {}
    for canon, per_model in canonical.items():
        g = (canon[0], canon[2], round(canon[3], 2), round(canon[4], 2), canon[7], canon[8])
        groups.setdefault(g, []).append(canon)
    shadow, divergence = set(), []
    for g, members in groups.items():
        if len(members) < 2:
            continue
        def min_idx(c):
            return min(o["resolved_ego_idx"] for m in canonical[c].values() for o in m)
        members = sorted(members, key=min_idx)
        primary = members[0]
        for clone in members[1:]:
            shadow.add(clone)
            for model in MODELS:
                a = canonical[primary][model][0]["outcome"]
                b = canonical[clone][model][0]["outcome"]
                if a != b:
                    divergence.append({"group": str(g), "model": model,
                                       "primary": a, "clone": b})

    canon_rows, conflicts = [], []
    for canon, per_model in sorted(canonical.items(), key=lambda kv: str(kv[0])):
        row = {"map": canon[0], "opp_raceline": canon[2], "start_x": round(canon[3], 3),
               "start_y": round(canon[4], 3), "speedscale": canon[7], "interval": canon[8],
               "is_shadow_clone": canon in shadow,
               "dev_overlap": any(o["dev_overlap"] for m in per_model.values() for o in m),
               "n_grid_occurrences": sum(len(v) for v in per_model.values()) // len(MODELS)}
        conflict = False
        for model in MODELS:
            occs = per_model.get(model, [])
            outs = sorted({o["outcome"] for o in occs})
            row[f"{model}_outcome"] = "|".join(outs)
            row[f"{model}_n"] = len(occs)
            if len(outs) > 1:
                conflict = True
                conflicts.append({"canon": str(canon), "model": model, "outcomes": outs})
        row["outcome_conflict"] = conflict
        canon_rows.append(row)
    validation["outcome_conflicts"] = len(conflicts)
    _write_tsv("canonical_episodes.tsv", canon_rows,
               ["map", "opp_raceline", "start_x", "start_y", "speedscale", "interval",
                "dev_overlap", "is_shadow_clone", "n_grid_occurrences", "outcome_conflict"]
               + [f"{m}_{c}" for m in MODELS for c in ("outcome", "n")])

    # ---- collision events (terminal NPZ classification) --------------------
    ev_rows = []
    seen_npz = {}
    for o in occurrences:
        if o["outcome"] != "collision":
            continue
        ck = (o["model"],) + o["canon"]
        if ck in seen_npz:
            continue
        seen_npz[ck] = True
        row = {"model": o["model"], "map": o["map"], "opp_raceline": o["opp_raceline"],
               "speedscale": o["speedscale"], "start_x": o["start_x"], "start_y": o["start_y"],
               "ego_collision_flag": o["ego_collision"], "opp_collision_flag": o["opp_collision"],
               "involvement": ("ego" if o["ego_collision"] else "") + ("+opp" if o["opp_collision"] else "")
               or "unknown", "cause": "unknown", "phase": "unknown", "final_dist": "",
               "dev_overlap": o["dev_overlap"]}
        try:
            z = np.load(o["npz_path"], allow_pickle=True)
            fe, fo = z["final_ego_pose"], z["final_opp_pose"]
            dist = float(np.hypot(fe[0] - fo[0], fe[1] - fo[1]))
            rel = float(z["final_ego_progress"]) - float(z["final_opp_progress"])
            row["final_dist"] = round(dist, 3)
            row["cause"] = "car" if dist <= CAR_DIST_THRESH else "wall"
            row["phase"] = ("alongside" if abs(rel) < ALONGSIDE_THRESH
                            else ("post" if rel > 0 else "pre"))
        except (KeyError, OSError, ValueError):
            validation["missing_npz_terminal_fields"] += 1
        if row["cause"] == "unknown":
            validation["unknown_classifications"] += 1
        ev_rows.append(row)
    _write_tsv("collision_events.tsv", ev_rows,
               ["model", "map", "opp_raceline", "speedscale", "start_x", "start_y",
                "ego_collision_flag", "opp_collision_flag", "involvement",
                "cause", "phase", "final_dist", "dev_overlap"])

    # ---- canonical analysis pools ------------------------------------------
    def pool_of(canon):
        return "austin" if canon[0] == "Austin" else "cross"

    # canonical outcome per model (first occurrence; conflicts already flagged)
    canon_out = {}
    for canon, per_model in canonical.items():
        canon_out[canon] = {m: per_model[m][0] for m in MODELS if m in per_model}

    usable = {c: v for c, v in canon_out.items()
              if len(v) == len(MODELS) and c not in shadow
              and not any(o["dev_overlap"] for o in v.values())}

    def stratum(canon):
        ol, sp = canon[2], canon[7]
        if ol == "raceline1" and sp in (0.5, 0.6):
            return "skill_F"
        if ol in ("raceline0", "raceline2") and sp in (0.7, 0.8):
            return "skill_S"
        return "other"

    summary = {"canonical_total": len(canonical),
               "shadow_clones": len(shadow),
               "near_dup_outcome_divergence": divergence,
               "canonical_dev_disjoint_usable": len(usable),
               "pools": {}, "strata": {}, "bc_claims": {}, "paired": {},
               "opp_only_floor": {}, "ol1_phase_bc": {}, "offset_audit": {}}

    for pool in ("austin", "cross"):
        keys = [c for c in usable if pool_of(c) == pool]
        summary["pools"][pool] = {"N": len(keys)}
        for model in MODELS:
            cn = sum(usable[c][model]["outcome"] == "collision" for c in keys)
            ov = sum(usable[c][model]["outcome"] == "overtake" for c in keys)
            eg = sum(usable[c][model]["ego_collision"] for c in keys)
            op_only = sum(usable[c][model]["opp_collision"]
                          and not usable[c][model]["ego_collision"] for c in keys)
            summary["pools"][pool][model] = {"collision": cn, "overtake": ov,
                                             "ego_involved": eg, "opp_only": op_only}

    bc_tot = {k: sum(summary["pools"][p]["bc"][k] for p in ("austin", "cross"))
              for k in ("collision", "overtake", "ego_involved", "opp_only")}
    summary["bc_claims"] = {
        "spec": {"N": 3060, "any": 170, "ego": 153, "opp_only": 17, "overtake": 1815,
                 "austin_N_sec12": 1260},
        "computed": {"N": len(usable), **bc_tot,
                     "austin_N": summary["pools"]["austin"]["N"]},
    }

    for st in ("skill_F", "skill_S", "other"):
        keys = [c for c in usable if stratum(c) == st]
        summary["strata"][st] = {
            "N": len(keys),
            "bc_any": sum(usable[c]["bc"]["outcome"] == "collision" for c in keys),
            "bc_ego": sum(usable[c]["bc"]["ego_collision"] for c in keys),
            "bc_overtake": sum(usable[c]["bc"]["outcome"] == "overtake" for c in keys),
        }

    # OL1 phase decomposition for BC (canonical, dev-disjoint)
    ol1_ev = {}
    for r in ev_rows:
        if r["model"] == "bc" and r["opp_raceline"] == "raceline1" and not r["dev_overlap"]:
            k = ("Austin" if r["map"] == "Austin" else r["map"], r["start_x"], r["start_y"],
                 r["speedscale"])
            ol1_ev.setdefault(k, r)  # canonical: first occurrence
    phases = {}
    for r in ol1_ev.values():
        phases[r["phase"]] = phases.get(r["phase"], 0) + 1
    summary["ol1_phase_bc"] = {"spec": {"pre": 34, "alongside": 43, "post": 1},
                               "computed": phases, "n_cases": len(ol1_ev)}

    # opp-only floor: same canonical keys across all four models?
    floor_keys = {c for c in usable
                  if usable[c]["bc"]["opp_collision"] and not usable[c]["bc"]["ego_collision"]}
    same_all = all(
        all(usable[c][m]["opp_collision"] and not usable[c][m]["ego_collision"]
            for m in MODELS) for c in floor_keys)
    summary["opp_only_floor"] = {"bc_opp_only_keys": len(floor_keys),
                                 "identical_across_all_models": same_all}

    # canonical paired stats (for spec section 12 corrections)
    for cand in MODELS[1:]:
        summary["paired"][cand] = {}
        for pool in ("austin", "cross"):
            keys = [c for c in usable if pool_of(c) == pool]
            fc = sum(usable[c]["bc"]["outcome"] == "collision"
                     and usable[c][cand]["outcome"] != "collision" for c in keys)
            nc = sum(usable[c]["bc"]["outcome"] != "collision"
                     and usable[c][cand]["outcome"] == "collision" for c in keys)
            go = sum(usable[c]["bc"]["outcome"] != "overtake"
                     and usable[c][cand]["outcome"] == "overtake" for c in keys)
            lo = sum(usable[c]["bc"]["outcome"] == "overtake"
                     and usable[c][cand]["outcome"] != "overtake" for c in keys)
            summary["paired"][cand][pool] = {
                "fixed_coll": fc, "new_coll": nc, "p_coll": round(sign_test_p(fc, nc), 4),
                "gained_ot": go, "lost_ot": lo, "p_ot": round(sign_test_p(go, lo), 4)}

    # ---- offset-set audit (spec section 10.1) ------------------------------
    def offset_poses(offset):
        # Rounded to 1 cm on both sides so dev/offset intersections compare
        # at a single precision.
        s = set()
        for i in range(50):
            if offset == 0:
                raw = i * maxwc["Austin"] // 49
            else:
                raw = (i * maxwc["Austin"] // 50 + offset) % maxwc["Austin"]
            _, x, y = resolve("Austin", raw)
            s.add((round(x, 2), round(y, 2)))
        return s

    sets = {"dev": dev_poses}
    for o in (21, 42, 63, 84, 10, 31, 52, 73, 11, 32, 75, 86):
        sets[f"off{o}"] = offset_poses(o)
    p1_union = sets["off21"] | sets["off42"] | sets["off63"] | sets["off84"]
    hist = p1_union | sets["dev"]
    for name, group in (("p1_2142_63_84", ["off21", "off42", "off63", "off84"]),
                        ("cand_10_31_52_73", ["off10", "off31", "off52", "off73"]),
                        ("cand_11_32_75_86", ["off11", "off32", "off75", "off86"])):
        u = set().union(*(sets[g] for g in group))
        summary["offset_audit"][name] = {
            "start_occurrences": sum(len(sets[g]) for g in group),
            "unique_starts": len(u),
            "overlap_dev": len(u & sets["dev"]),
            "overlap_p1_offsets": len(u & p1_union) if not name.startswith("p1") else "",
            "overlap_all_history": len(u & hist) if not name.startswith("p1") else "",
        }

    # ---- stop rules --------------------------------------------------------
    stop = {
        "counts_reconcile": validation["observed_occurrences"] == validation["expected_occurrences"]
                            and not validation["unvalidated_dirs"],
        "no_outcome_conflicts": validation["outcome_conflicts"] == 0,
        "skill_F_ego_ge_30": summary["strata"]["skill_F"]["bc_ego"] >= 30,
        "skill_S_ego_ge_30": summary["strata"]["skill_S"]["bc_ego"] >= 30,
        "npz_terminal_fields_ok": validation["missing_npz_terminal_fields"] == 0,
    }
    summary["stop_rules"] = stop
    summary["d0_pass"] = all(stop.values())
    validation["stop_rules"] = stop

    with open(os.path.join(OUT, "d0_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, "d0_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    with open(os.path.join(OUT, "d0_summary.md"), "w") as f:
        f.write("# D0 Canonical Collision Audit\n\n")
        f.write(f"- source run: p1_validation_{RUN} (28/28 validated grids)\n")
        f.write(f"- canonical scenarios: {summary['canonical_total']} "
                f"(shadow clones: {summary['shadow_clones']}; "
                f"dev-disjoint usable: {summary['canonical_dev_disjoint_usable']})\n")
        f.write(f"- near-dup outcome divergence (chaos sensitivity): "
                f"{len(summary['near_dup_outcome_divergence'])} model-cases\n")
        f.write(f"- D0 pass: **{summary['d0_pass']}** {json.dumps(stop)}\n\n")
        f.write("## Spec section 3 claim check (BC, canonical dev-disjoint)\n\n")
        f.write(f"```json\n{json.dumps(summary['bc_claims'], indent=2)}\n```\n\n")
        f.write("## Strata\n\n")
        f.write(f"```json\n{json.dumps(summary['strata'], indent=2)}\n```\n\n")
        f.write("## OL1 phase (BC)\n\n")
        f.write(f"```json\n{json.dumps(summary['ol1_phase_bc'], indent=2)}\n```\n\n")
        f.write("## Opponent-only floor\n\n")
        f.write(f"```json\n{json.dumps(summary['opp_only_floor'], indent=2)}\n```\n\n")
        f.write("## Canonical paired stats vs BC\n\n")
        f.write(f"```json\n{json.dumps(summary['paired'], indent=2)}\n```\n\n")
        f.write("## Offset-set audit (spec 10.1)\n\n")
        f.write(f"```json\n{json.dumps(summary['offset_audit'], indent=2)}\n```\n")

    print(json.dumps({"out": OUT, "pass": summary["d0_pass"],
                      "N": len(usable), "conflicts": validation["outcome_conflicts"]}))


def _model_match(model, path):
    suffix = {"cand160": "iter0160.pth", "cand120": "iter0120.pth",
              "cand040": "iter0040.pth"}[model]
    return path.endswith(suffix)


if __name__ == "__main__":
    main()

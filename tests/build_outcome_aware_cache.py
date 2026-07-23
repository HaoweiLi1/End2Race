#!/usr/bin/env python3
"""Standalone driver for the isolated outcome-aware hard-neighbor pool.

This never enters the PPO scheduler and never mutates the baseline or
``boundary-aware-v1`` caches (it only reads them). Two modes:

1. Inspection (safe to run while training is in flight)::

       python tests/build_outcome_aware_cache.py --dry-run --limit-pairs 12 \
           --env-workers 2 --filter-mode safe_overtake

   Replays only the boundary candidates selected by the first N boundary pairs
   plus their "other" endpoints, then prints the richer labels (outcome, mode,
   fishtail, clearance) and the filter decision per candidate. Writes no cache.

2. Full build (run when the training pipeline is free)::

       python tests/build_outcome_aware_cache.py \
           --filter-mode safe_overtake \
           --output-cache-dir post-trained/collision-cache/outcome-aware-v1-safe

   Replays every boundary candidate + every pair "other" endpoint + every base
   collision, applies the filter, writes a self-verifying cache, then loads it
   back to confirm the pool is a pure function of (labels x filter spec).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppo.hard_neighbors import (
    discover_boundary_candidates,
    materialize_boundary_candidates,
)
from ppo.scenarios import expanded_scenarios
import ppo.outcome_aware_hard as oah
from ppo.outcome_aware_hard import (
    FilterSpec,
    apply_outcome_aware_filter,
    classify_labeled_scenarios,
    resolve_outcome_aware_collision_scenarios,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CACHE = PROJECT_ROOT / "post-trained/collision-cache/default"
DEFAULT_MODEL = PROJECT_ROOT / "pretrained/end2race.pth"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/inspect the outcome-aware hard pool")
    parser.add_argument("--map-name", default="Austin")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hidden-scale", type=int, default=4)
    parser.add_argument("--base-cache-dir", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--output-cache-dir", type=Path, default=None)
    parser.add_argument("--filter-mode", default="safe_overtake", choices=oah.VALID_FILTER_MODES)
    parser.add_argument("--safe-clearance-m", type=float, default=0.10)
    parser.add_argument("--require-any-pair-safe", action="store_true",
                        help="Keep a boundary if ANY source pair is safe (default: require ALL)")
    parser.add_argument("--no-rear-end-quota", action="store_true")
    parser.add_argument("--env-workers", type=int, default=2)
    parser.add_argument("--start-method", default="forkserver")
    parser.add_argument("--limit-pairs", type=int, default=None,
                        help="Inspection only: replay just the first N boundary pairs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Replay a subset and print labels/decisions; write no cache")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def _base_candidates_and_outcomes(map_name: str, base_cache_dir: Path):
    candidates = expanded_scenarios(map_name)
    outcomes = _read_jsonl(base_cache_dir / "candidate_outcomes.jsonl")
    if len(outcomes) != len(candidates):
        raise SystemExit("base cache candidate_outcomes.jsonl does not match expanded_scenarios")
    return candidates, outcomes


def _filter_spec(args: argparse.Namespace) -> FilterSpec:
    return FilterSpec(
        mode=args.filter_mode,
        safe_clearance_m=args.safe_clearance_m,
        require_all_source_pairs_safe=not args.require_any_pair_safe,
        keep_rear_end_quota=not args.no_rear_end_quota,
    ).validate()


def run_inspection(args: argparse.Namespace) -> None:
    candidates, base_outcomes = _base_candidates_and_outcomes(args.map_name, args.base_cache_dir)
    discovery = discover_boundary_candidates(candidates, base_outcomes)
    boundary_candidates = materialize_boundary_candidates(discovery.candidates, candidates)
    candidate_by_id = {c.scenario_id: c for c in boundary_candidates}
    base_by_id = {c.scenario_id: c for c in candidates}

    pairs = list(discovery.pair_records)[: args.limit_pairs]
    subset_ids: list[str] = []
    for pair in pairs:
        for sid in pair.get("selected_scenario_ids", []):
            if sid not in subset_ids:
                subset_ids.append(sid)
    subset_candidates = tuple(candidate_by_id[sid] for sid in subset_ids)

    other_ids: list[str] = []
    for pair in pairs:
        oid = oah._other_endpoint_id(pair)
        if oid not in other_ids:
            other_ids.append(oid)
    other_candidates = tuple(base_by_id[oid] for oid in other_ids)

    print(f"Inspecting {len(pairs)} pairs -> {len(subset_candidates)} boundary candidates "
          f"+ {len(other_candidates)} other endpoints (workers={args.env_workers})", flush=True)

    boundary_labels, _ = classify_labeled_scenarios(
        args.model, args.hidden_scale, args.map_name, args.env_workers,
        _reindex(subset_candidates), args.start_method)
    other_labels, _ = classify_labeled_scenarios(
        args.model, args.hidden_scale, args.map_name, args.env_workers,
        _reindex(other_candidates), args.start_method)

    spec = _filter_spec(args)
    final, audit = apply_outcome_aware_filter(
        spec,
        base_collisions=(),
        boundary_candidates=_reindex(subset_candidates),
        boundary_labels=boundary_labels,
        pair_records=pairs,
        pair_other_labels=other_labels,
    )

    print(f"\n=== richer boundary-candidate labels (mode={spec.mode}) ===")
    print(f"{'scenario_id':<40}{'outcome':<14}{'mode':<20}{'fish':<6}{'slipMax':>8}{'dHead':>8}{'kept':>6}")
    decision_by_id = {d["scenario_id"]: d for d in audit["per_candidate_decisions"]}
    for label in boundary_labels:
        d = decision_by_id.get(label.scenario_id, {})
        fish = "" if label.collision_is_fishtail is None else str(label.collision_is_fishtail)
        slip = "" if label.collision_ego_slip_max_deg is None else f"{label.collision_ego_slip_max_deg:.1f}"
        dh = "" if label.collision_delta_heading_deg is None else f"{label.collision_delta_heading_deg:.1f}"
        print(f"{label.scenario_id:<40}{label.outcome:<14}{str(label.collision_mode):<20}"
              f"{fish:<6}{slip:>8}{dh:>8}{str(d.get('kept','')):>6}")

    print(f"\n=== other-endpoint outcomes ===")
    from collections import Counter
    print("outcome:", dict(Counter(l.outcome for l in other_labels)),
          " min_clearance overtake>=thr:",
          sum(1 for l in other_labels if l.outcome == "overtake" and l.min_obb_clearance_m >= spec.safe_clearance_m))
    print("\n=== filter audit (subset) ===")
    print(json.dumps({k: v for k, v in audit.items() if k != "per_candidate_decisions"}, indent=2))


def _reindex(candidates):
    from ppo.scenarios import ScenarioSpec  # local import to avoid confusion
    return tuple(candidates)


def run_full_build(args: argparse.Namespace) -> None:
    if args.output_cache_dir is None:
        raise SystemExit("full build requires --output-cache-dir")
    ns = SimpleNamespace(
        pretrained_model_path=str(args.model),
        hidden_scale=args.hidden_scale,
        map_name=args.map_name,
        env_workers=args.env_workers,
        collision_cache_dir=str(args.base_cache_dir),
        reclassify_collisions=False,
        outcome_aware_cache_dir=str(args.output_cache_dir),
        outcome_aware_filter_mode=args.filter_mode,
        outcome_aware_safe_clearance_m=args.safe_clearance_m,
        outcome_aware_require_all_pairs_safe=not args.require_any_pair_safe,
        outcome_aware_keep_rear_end=not args.no_rear_end_quota,
        outcome_aware_drop_unsafe_base=False,
    )
    final, info = resolve_outcome_aware_collision_scenarios(ns, expanded_scenarios(args.map_name), args.start_method)
    print("\n=== build info ===")
    print(json.dumps(info, indent=2))
    print(f"\nfinal collision pool: {len(final)} scenarios written to {args.output_cache_dir}")


def main() -> None:
    args = parse_arguments()
    if args.env_workers <= 0:
        raise SystemExit("--env-workers must be positive")
    if args.dry_run or args.limit_pairs is not None:
        if args.limit_pairs is None:
            raise SystemExit("--dry-run requires --limit-pairs N (subset inspection)")
        run_inspection(args)
    else:
        run_full_build(args)


if __name__ == "__main__":
    main()

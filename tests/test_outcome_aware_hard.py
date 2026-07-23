"""Comprehensive tests for the isolated outcome-aware hard-neighbor pool.

Run the fast, simulator-free suite (default)::

    python -m pytest tests/test_outcome_aware_hard.py -q
    # or
    python tests/test_outcome_aware_hard.py

The default suite touches no simulator and no running PPO. The single real-BC
smoke test is skipped unless ``END2RACE_RUN_SIM=1`` and is deliberately tiny
(2 scenarios, 1 worker) so it never contends with an in-flight training run.

Anchor test
-----------
``test_all_mode_reconstructs_boundary_aware_v1`` re-derives the final pool of
the already-validated shipped ``boundary-aware-v1`` cache from its own recorded
boundary outcomes and asserts byte-equality against its
``collision_scenarios.json``. This proves the filter/pool-assembly logic without
re-running any rollout.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from ppo.hard_neighbors import (
    discover_boundary_candidates,
    materialize_boundary_candidates,
)
from ppo.scenarios import ScenarioSpec, expanded_scenarios
import ppo.outcome_aware_hard as oah
from ppo.outcome_aware_hard import (
    CandidateLabel,
    FilterSpec,
    apply_outcome_aware_filter,
    build_outcome_aware_pool,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CACHE_DIR = PROJECT_ROOT / "post-trained/collision-cache/default"
HARD_CACHE_DIR = PROJECT_ROOT / "post-trained/collision-cache/boundary-aware-v1"
MAP_NAME = "Austin"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def _base_candidates_and_outcomes():
    candidates = expanded_scenarios(MAP_NAME)
    outcomes = _read_jsonl(BASE_CACHE_DIR / "candidate_outcomes.jsonl")
    if len(outcomes) != len(candidates):
        raise RuntimeError("base cache outcomes/candidates mismatch")
    return candidates, outcomes


def _dummy_collision_label(index: int, scenario_id: str, *, mode="post_overtake_rear", fishtail=True) -> CandidateLabel:
    return CandidateLabel(
        candidate_index=index,
        scenario_id=scenario_id,
        outcome="ego_collision",
        terminal_relative_position_m=0.3,
        min_obb_clearance_m=0.0,
        steps=300,
        elapsed_time_s=3.0,
        collision_time_s=3.0,
        collision_center_distance_m=0.5,
        collision_bearing_deg=110.0,
        collision_rel_track_m=0.3,
        collision_delta_heading_deg=40.0,
        collision_ego_slip_max_deg=15.0,
        collision_true_ego_slip_deg=12.0,
        collision_partner_is_opponent=True,
        collision_mode=mode,
        collision_is_fishtail=fishtail,
    )


def _noncollision_label(index: int, scenario_id: str, outcome: str, clearance: float) -> CandidateLabel:
    return CandidateLabel(
        candidate_index=index,
        scenario_id=scenario_id,
        outcome=outcome,
        terminal_relative_position_m=0.5 if outcome == "overtake" else -0.5,
        min_obb_clearance_m=clearance,
        steps=400,
        elapsed_time_s=4.0,
    )


class RecordRoundTripTests(unittest.TestCase):
    def test_label_record_round_trip(self) -> None:
        label = _dummy_collision_label(0, "hard-x")
        restored = CandidateLabel.from_record(label.to_record())
        self.assertEqual(restored, label)

    def test_label_record_rejects_wrong_schema(self) -> None:
        record = _dummy_collision_label(0, "hard-x").to_record()
        record["label_schema"] = 999
        with self.assertRaises(RuntimeError):
            CandidateLabel.from_record(record)

    def test_label_record_rejects_extra_field(self) -> None:
        record = _dummy_collision_label(0, "hard-x").to_record()
        record["surprise"] = 1
        with self.assertRaises(RuntimeError):
            CandidateLabel.from_record(record)


class FilterLogicTests(unittest.TestCase):
    """Synthetic pairs/candidates exercise each mode's selection rule."""

    def _fixture(self):
        # Two boundary candidates: one from a safe-overtake pair (fishtail),
        # one from a follow pair; plus a rear-end collision candidate.
        base_collisions = (
            ScenarioSpec("collision-a", "collision", 0, 10, 25, "raceline0", 0.7, 15, MAP_NAME),
        )
        boundary_candidates = (
            ScenarioSpec("hard-safe-fish", "hard_neighbor", 0, 10, 25, "raceline0", 0.75, 15, MAP_NAME),
            ScenarioSpec("hard-follow-fish", "hard_neighbor", 1, 20, 35, "raceline0", 0.55, 15, MAP_NAME),
            ScenarioSpec("hard-safe-rearend", "hard_neighbor", 2, 30, 45, "raceline0", 0.65, 12, MAP_NAME),
        )
        boundary_labels = (
            _dummy_collision_label(0, "hard-safe-fish", mode="post_overtake_rear", fishtail=True),
            _dummy_collision_label(1, "hard-follow-fish", mode="post_overtake_rear", fishtail=True),
            _dummy_collision_label(2, "hard-safe-rearend", mode="rear_end_opp", fishtail=False),
        )
        pair_records = [
            {
                "pair_id": "boundary-0", "low_outcome": "other", "high_outcome": "ego_collision",
                "low_scenario_id": "other-safe", "high_scenario_id": "collision-a",
                "selected_scenario_ids": ["hard-safe-fish"],
            },
            {
                "pair_id": "boundary-1", "low_outcome": "ego_collision", "high_outcome": "other",
                "low_scenario_id": "collision-b", "high_scenario_id": "other-follow",
                "selected_scenario_ids": ["hard-follow-fish"],
            },
            {
                "pair_id": "boundary-2", "low_outcome": "other", "high_outcome": "ego_collision",
                "low_scenario_id": "other-safe2", "high_scenario_id": "collision-c",
                "selected_scenario_ids": ["hard-safe-rearend"],
            },
        ]
        pair_other_labels = (
            _noncollision_label(0, "other-safe", "overtake", 0.3),     # safe
            _noncollision_label(1, "other-follow", "follow", 0.0),     # not safe
            _noncollision_label(2, "other-safe2", "overtake", 0.25),   # safe
        )
        return base_collisions, boundary_candidates, boundary_labels, pair_records, pair_other_labels

    def _run(self, spec):
        base, cand, labels, pairs, others = self._fixture()
        return apply_outcome_aware_filter(spec, base, cand, labels, pairs, others)

    def test_all_mode_keeps_every_collision(self):
        final, audit = self._run(FilterSpec(mode="all"))
        ids = [s.scenario_id for s in final]
        self.assertEqual(ids, ["collision-a", "hard-safe-fish", "hard-follow-fish", "hard-safe-rearend"])
        self.assertEqual(audit["boundary_collision_kept"], 3)

    def test_safe_overtake_drops_follow_side(self):
        final, audit = self._run(FilterSpec(mode="safe_overtake"))
        ids = [s.scenario_id for s in final]
        self.assertIn("hard-safe-fish", ids)
        self.assertIn("hard-safe-rearend", ids)
        self.assertNotIn("hard-follow-fish", ids)  # follow-side pair dropped
        self.assertEqual(audit["boundary_collision_kept"], 2)

    def test_fishtail_mode_keeps_only_safe_post_overtake_fishtail(self):
        final, _ = self._run(FilterSpec(mode="fishtail"))
        ids = [s.scenario_id for s in final]
        self.assertEqual(ids, ["collision-a", "hard-safe-fish"])

    def test_fishtail_rearend_adds_rear_end_quota(self):
        final, _ = self._run(FilterSpec(mode="fishtail_rearend"))
        ids = [s.scenario_id for s in final]
        self.assertIn("hard-safe-fish", ids)      # safe fishtail
        self.assertIn("hard-safe-rearend", ids)   # rear-end quota
        self.assertNotIn("hard-follow-fish", ids)

    def test_safe_clearance_threshold_is_enforced(self):
        base, cand, labels, pairs, others = self._fixture()
        # Lower the safe overtake's clearance below threshold -> pair now unsafe.
        others = (replace(others[0], min_obb_clearance_m=0.05), others[1], others[2])
        final, _ = apply_outcome_aware_filter(FilterSpec(mode="safe_overtake"), base, cand, labels, pairs, others)
        self.assertNotIn("hard-safe-fish", [s.scenario_id for s in final])

    def test_require_all_pairs_safe_vs_any(self):
        # A candidate produced by two pairs, one safe one follow.
        base = ()
        cand = (ScenarioSpec("hard-multi", "hard_neighbor", 0, 10, 25, "raceline0", 0.7, 15, MAP_NAME),)
        labels = (_dummy_collision_label(0, "hard-multi"),)
        pairs = [
            {"pair_id": "p0", "low_outcome": "other", "high_outcome": "ego_collision",
             "low_scenario_id": "o-safe", "high_scenario_id": "c0", "selected_scenario_ids": ["hard-multi"]},
            {"pair_id": "p1", "low_outcome": "other", "high_outcome": "ego_collision",
             "low_scenario_id": "o-follow", "high_scenario_id": "c1", "selected_scenario_ids": ["hard-multi"]},
        ]
        others = (
            _noncollision_label(0, "o-safe", "overtake", 0.3),
            _noncollision_label(1, "o-follow", "follow", 0.0),
        )
        final_all, _ = apply_outcome_aware_filter(
            FilterSpec(mode="safe_overtake", require_all_source_pairs_safe=True), base, cand, labels, pairs, others)
        final_any, _ = apply_outcome_aware_filter(
            FilterSpec(mode="safe_overtake", require_all_source_pairs_safe=False), base, cand, labels, pairs, others)
        self.assertEqual([s.scenario_id for s in final_all], [])       # one pair unsafe -> drop
        self.assertEqual([s.scenario_id for s in final_any], ["hard-multi"])  # any safe -> keep

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            FilterSpec(mode="nonsense").validate()


@unittest.skipUnless(BASE_CACHE_DIR.exists() and HARD_CACHE_DIR.exists(), "requires base + hard caches")
class BoundaryAwareReconstructionTests(unittest.TestCase):
    """Anchor: `all` mode must reproduce the shipped hard cache pool exactly."""

    def test_all_mode_reconstructs_boundary_aware_v1(self):
        candidates, base_outcomes = _base_candidates_and_outcomes()
        base_collisions = tuple(
            candidate for candidate, outcome in zip(candidates, base_outcomes)
            if outcome["outcome"] == "ego_collision"
        )
        discovery = discover_boundary_candidates(candidates, base_outcomes)
        boundary_candidates = materialize_boundary_candidates(discovery.candidates, candidates)

        # Recorded binary outcomes of the boundary candidates from the shipped cache.
        boundary_outcomes = _read_jsonl(HARD_CACHE_DIR / "boundary_candidate_outcomes.jsonl")
        self.assertEqual(len(boundary_outcomes), len(boundary_candidates))

        # Synthesize labels: for `all` mode only `outcome` matters.
        boundary_labels = []
        for index, (candidate, outcome) in enumerate(zip(boundary_candidates, boundary_outcomes)):
            self.assertEqual(outcome["scenario_id"], candidate.scenario_id)
            if outcome["outcome"] == "ego_collision":
                boundary_labels.append(_dummy_collision_label(index, candidate.scenario_id))
            else:
                boundary_labels.append(_noncollision_label(index, candidate.scenario_id, "overtake", 0.3))

        final, audit = build_outcome_aware_pool(
            base_collisions=base_collisions,
            base_outcomes=base_outcomes,
            pair_records=discovery.pair_records,
            boundary_candidates=boundary_candidates,
            boundary_labels=boundary_labels,
            pair_other_labels=[],  # unused for `all`
            base_collision_labels=[],
            spec=FilterSpec(mode="all"),
        )

        shipped = json.loads((HARD_CACHE_DIR / "collision_scenarios.json").read_text())
        self.assertEqual([asdict(s) for s in final], shipped)
        self.assertEqual(audit["final_collision_count"], len(shipped))


class CacheRoundTripTests(unittest.TestCase):
    """Build -> load self-verification and tamper rejection, no simulator."""

    def _tiny_build_inputs(self):
        base_collisions = (
            ScenarioSpec("collision-a", "collision", 0, 10, 25, "raceline0", 0.7, 15, MAP_NAME),
        )
        boundary_candidates = (
            ScenarioSpec("hard-safe-fish", "hard_neighbor", 0, 10, 25, "raceline0", 0.75, 15, MAP_NAME),
            ScenarioSpec("hard-follow-fish", "hard_neighbor", 1, 20, 35, "raceline0", 0.55, 15, MAP_NAME),
        )
        boundary_labels = [
            _dummy_collision_label(0, "hard-safe-fish"),
            _dummy_collision_label(1, "hard-follow-fish"),
        ]
        pair_records = [
            {"pair_id": "boundary-0", "low_outcome": "other", "high_outcome": "ego_collision",
             "low_scenario_id": "other-safe", "high_scenario_id": "collision-a",
             "selected_scenario_ids": ["hard-safe-fish"]},
            {"pair_id": "boundary-1", "low_outcome": "ego_collision", "high_outcome": "other",
             "low_scenario_id": "collision-b", "high_scenario_id": "other-follow",
             "selected_scenario_ids": ["hard-follow-fish"]},
        ]
        pair_other_labels = [
            _noncollision_label(0, "other-safe", "overtake", 0.3),
            _noncollision_label(1, "other-follow", "follow", 0.0),
        ]
        base_outcomes = [{"candidate_index": 0, "scenario_id": "collision-a", "outcome": "ego_collision"}]
        base_collision_labels = [_dummy_collision_label(0, "collision-a")]
        return (base_collisions, boundary_candidates, boundary_labels, pair_records,
                pair_other_labels, base_outcomes, base_collision_labels)

    def _build(self, cache_dir: Path, spec: FilterSpec):
        (base_collisions, boundary_candidates, boundary_labels, pair_records,
         pair_other_labels, base_outcomes, base_collision_labels) = self._tiny_build_inputs()
        final, audit = build_outcome_aware_pool(
            base_collisions=base_collisions, base_outcomes=base_outcomes,
            pair_records=pair_records, boundary_candidates=boundary_candidates,
            boundary_labels=boundary_labels, pair_other_labels=pair_other_labels,
            base_collision_labels=base_collision_labels, spec=spec,
        )
        summary = oah._summary(base_outcomes, pair_records, boundary_labels, pair_other_labels, final, audit)
        config = {"outcome_aware_schema": oah.OUTCOME_AWARE_CACHE_SCHEMA, "filter_spec": spec.to_config(), "tag": "test"}
        oah.publish_outcome_aware_cache(
            cache_dir, config, base_outcomes, pair_records, boundary_labels,
            pair_other_labels, base_collision_labels, final, audit, summary,
            {"candidate_count": 2, "env_workers": 1, "wall_seconds": 0.0, "scenarios_per_second": 0.0},
        )
        return config, base_collisions, base_outcomes, pair_records, boundary_candidates

    def test_build_then_load_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "oa-cache"
            spec = FilterSpec(mode="safe_overtake")
            config, base_collisions, base_outcomes, pair_records, boundary_candidates = self._build(cache_dir, spec)
            self.assertTrue(oah.cache_exists(cache_dir))
            final, summary = oah.load_outcome_aware_cache(
                cache_dir, config, base_collisions, base_outcomes, pair_records, boundary_candidates, spec)
            ids = [s.scenario_id for s in final]
            self.assertEqual(ids, ["collision-a", "hard-safe-fish"])  # follow-side dropped
            self.assertIn("filter_audit", summary)

    def test_refuses_to_overwrite_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "oa-cache"
            self._build(cache_dir, FilterSpec(mode="all"))
            with self.assertRaises(RuntimeError):
                self._build(cache_dir, FilterSpec(mode="all"))

    def test_tampered_pool_is_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "oa-cache"
            spec = FilterSpec(mode="all")
            config, base_collisions, base_outcomes, pair_records, boundary_candidates = self._build(cache_dir, spec)
            # Corrupt the final pool but leave manifest as-is -> manifest hash mismatch.
            (cache_dir / "collision_scenarios.json").write_text("[]\n")
            with self.assertRaises(RuntimeError):
                oah.load_outcome_aware_cache(
                    cache_dir, config, base_collisions, base_outcomes, pair_records, boundary_candidates, spec)

    def test_config_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "oa-cache"
            spec = FilterSpec(mode="all")
            config, base_collisions, base_outcomes, pair_records, boundary_candidates = self._build(cache_dir, spec)
            wrong = dict(config, tag="mutated")
            with self.assertRaises(RuntimeError):
                oah.load_outcome_aware_cache(
                    cache_dir, wrong, base_collisions, base_outcomes, pair_records, boundary_candidates, spec)


@unittest.skipUnless(os.environ.get("END2RACE_RUN_SIM") == "1", "set END2RACE_RUN_SIM=1 to run the tiny real-BC smoke")
class RealReplaySmokeTests(unittest.TestCase):
    """Tiny (2 scenario, 1 worker) real BC replay; proves the classifier contract."""

    def test_boundary_labels_match_shipped_binary_outcomes(self):
        candidates, base_outcomes = _base_candidates_and_outcomes()
        discovery = discover_boundary_candidates(candidates, base_outcomes)
        boundary_candidates = materialize_boundary_candidates(discovery.candidates, candidates)
        shipped = _read_jsonl(HARD_CACHE_DIR / "boundary_candidate_outcomes.jsonl")
        # Pick the first confirmed collision and the first "other" for coverage.
        col_idx = next(i for i, o in enumerate(shipped) if o["outcome"] == "ego_collision")
        oth_idx = next(i for i, o in enumerate(shipped) if o["outcome"] == "other")
        subset = [boundary_candidates[col_idx], boundary_candidates[oth_idx]]

        labels, meta = oah.classify_labeled_scenarios(
            str(PROJECT_ROOT / "pretrained/end2race.pth"), 4, MAP_NAME, 1, subset, "forkserver")
        by_id = {label.scenario_id: label for label in labels}
        self.assertEqual(
            by_id[subset[0].scenario_id].outcome, "ego_collision",
            "confirmed collision must reproduce under the frozen BC contract",
        )
        self.assertIn(by_id[subset[1].scenario_id].outcome, {"overtake", "follow"})
        col_label = by_id[subset[0].scenario_id]
        self.assertIsNotNone(col_label.collision_mode)
        self.assertIn(col_label.collision_mode, oah.COLLISION_MODES)
        self.assertIsNotNone(col_label.collision_bearing_deg)


if __name__ == "__main__":
    unittest.main(verbosity=2)

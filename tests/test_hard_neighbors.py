from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from ppo.hard_neighbors import (
    discover_boundary_candidates,
    materialize_boundary_candidates,
)
from ppo.scenarios import ScenarioScheduler, ScenarioSpec
from train_ppo import parse_arguments


def make_lattice() -> tuple[ScenarioSpec, ...]:
    scenarios = []
    index = 0
    for interval_idx in (8, 10):
        for speed_scale in (0.5, 0.6):
            scenarios.append(
                ScenarioSpec(
                    scenario_id=f"toy-{index}",
                    pool="collision",
                    startpoint_ordinal=0,
                    ego_idx=10,
                    opp_idx=10 + interval_idx,
                    opp_raceline="raceline1",
                    opp_speedscale=speed_scale,
                    interval_idx=interval_idx,
                    map_name="Austin",
                )
            )
            index += 1
    return tuple(scenarios)


def make_outcomes(
    scenarios: tuple[ScenarioSpec, ...],
    values: tuple[str, ...],
) -> tuple[dict, ...]:
    return tuple(
        {
            "candidate_index": index,
            "scenario_id": scenario.scenario_id,
            "outcome": outcome,
        }
        for index, (scenario, outcome) in enumerate(zip(scenarios, values))
    )


def make_scheduler_pool(pool: str, count: int) -> tuple[ScenarioSpec, ...]:
    return tuple(
        ScenarioSpec(
            scenario_id=f"{pool}-{index:03d}",
            pool=pool,
            startpoint_ordinal=index,
            ego_idx=10 + index,
            opp_idx=18 + index,
            opp_raceline="raceline1",
            opp_speedscale=0.5,
            interval_idx=8,
            map_name="Austin",
        )
        for index in range(count)
    )


class BoundaryDiscoveryTests(unittest.TestCase):

    def test_discovers_only_one_axis_outcome_flips(self) -> None:
        scenarios = make_lattice()
        outcomes = make_outcomes(
            scenarios,
            ("ego_collision", "other", "other", "other"),
        )
        discovery = discover_boundary_candidates(
            scenarios,
            outcomes,
            interval_indices=(8, 10),
            speed_scales=(0.5, 0.6),
            max_candidates_per_family=8,
        )

        self.assertEqual(len(discovery.pair_records), 2)
        self.assertEqual(discovery.generated_candidate_count, 2)
        refined = {
            (plan.interval_idx, plan.speed_milli)
            for plan in discovery.candidates
        }
        self.assertEqual(refined, {(8, 550), (9, 500)})
        self.assertNotIn((9, 550), refined)
        self.assertEqual(
            {record["axis"] for record in discovery.pair_records},
            {"speed", "interval"},
        )

    def test_invalid_endpoint_is_not_a_boundary(self) -> None:
        scenarios = make_lattice()
        outcomes = make_outcomes(
            scenarios,
            ("ego_collision", "invalid", "other", "other"),
        )
        discovery = discover_boundary_candidates(
            scenarios,
            outcomes,
            interval_indices=(8, 10),
            speed_scales=(0.5, 0.6),
            max_candidates_per_family=8,
        )

        self.assertEqual(len(discovery.pair_records), 1)
        self.assertEqual(
            {(plan.interval_idx, plan.speed_milli) for plan in discovery.candidates},
            {(9, 500)},
        )

    def test_family_cap_is_deterministic(self) -> None:
        scenarios = make_lattice()
        outcomes = make_outcomes(
            scenarios,
            ("ego_collision", "other", "other", "other"),
        )
        first = discover_boundary_candidates(
            scenarios,
            outcomes,
            interval_indices=(8, 10),
            speed_scales=(0.5, 0.6),
            max_candidates_per_family=1,
        )
        second = discover_boundary_candidates(
            scenarios,
            outcomes,
            interval_indices=(8, 10),
            speed_scales=(0.5, 0.6),
            max_candidates_per_family=1,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.generated_candidate_count, 2)
        self.assertEqual(len(first.candidates), 1)
        self.assertEqual(
            sum(len(record["selected_scenario_ids"]) for record in first.pair_records),
            1,
        )

    def test_materialization_preserves_physics_and_finite_reset(self) -> None:
        scenarios = make_lattice()
        outcomes = make_outcomes(
            scenarios,
            ("ego_collision", "other", "other", "other"),
        )
        discovery = discover_boundary_candidates(
            scenarios,
            outcomes,
            interval_indices=(8, 10),
            speed_scales=(0.5, 0.6),
            max_candidates_per_family=8,
        )
        materialized = materialize_boundary_candidates(discovery.candidates, scenarios)

        self.assertEqual(len(materialized), 2)
        self.assertEqual(len({scenario.scenario_id for scenario in materialized}), 2)
        for scenario in materialized:
            self.assertEqual(scenario.pool, "hard_neighbor")
            self.assertEqual(scenario.opp_idx, (scenario.ego_idx + scenario.interval_idx) % 2096)
            reset_spec = scenario.to_reset_spec("collision")
            self.assertTrue(np.isfinite(reset_spec.poses).all())
            self.assertTrue(np.isfinite(reset_spec.initial_speed_feature))


class TrainingSwitchTests(unittest.TestCase):

    def test_hard_neighbors_are_off_by_default_and_explicitly_enabled(self) -> None:
        with patch("sys.argv", ["train_ppo.py"]):
            baseline = parse_arguments()
        with patch("sys.argv", ["train_ppo.py", "--hard_neighbors"]):
            legacy_treatment = parse_arguments()
        with patch(
            "sys.argv",
            [
                "train_ppo.py",
                "--hard_neighbors",
                "--hard_neighbor_fraction",
                "0.20",
            ],
        ):
            treatment = parse_arguments()

        self.assertFalse(baseline.hard_neighbors)
        self.assertIsNone(baseline.hard_neighbor_fraction)
        self.assertTrue(legacy_treatment.hard_neighbors)
        self.assertIsNone(legacy_treatment.hard_neighbor_fraction)
        self.assertTrue(treatment.hard_neighbors)
        self.assertEqual(treatment.hard_neighbor_fraction, 0.20)
        self.assertEqual(
            treatment.hard_neighbor_cache_dir,
            "post-trained/collision-cache/boundary-aware-v1",
        )


class HardNeighborSchedulingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.base = make_scheduler_pool("collision", 17)
        self.hard = make_scheduler_pool("hard_neighbor", 11)
        self.ordinary = make_scheduler_pool("ordinary", 13)

    def test_fraction_is_exact_over_each_quota_cycle(self) -> None:
        for fraction, expected_hard, expected_positions in (
            (0.20, 20, {2, 7}),
            (0.10, 10, {5}),
        ):
            with self.subTest(fraction=fraction):
                scheduler = ScenarioScheduler(
                    42,
                    self.base + self.hard,
                    self.ordinary,
                    hard_neighbor_fraction=fraction,
                )
                pools = [
                    scheduler._next_collision_scenario().pool
                    for _ in range(100)
                ]
                self.assertEqual(pools.count("hard_neighbor"), expected_hard)
                self.assertEqual(
                    {
                        index
                        for index, pool in enumerate(pools[:10])
                        if pool == "hard_neighbor"
                    },
                    expected_positions,
                )

    def test_base_and_ordinary_relative_orders_match_legacy_scheduler(self) -> None:
        legacy = ScenarioScheduler(42, self.base, self.ordinary)
        stratified = ScenarioScheduler(
            42,
            self.base + self.hard,
            self.ordinary,
            hard_neighbor_fraction=0.20,
        )

        legacy_base_ids = [
            legacy._next_collision_scenario().scenario_id
            for _ in range(50)
        ]
        stratified_base_ids = []
        while len(stratified_base_ids) < len(legacy_base_ids):
            scenario = stratified._next_collision_scenario()
            if scenario.pool == "collision":
                stratified_base_ids.append(scenario.scenario_id)
        self.assertEqual(stratified_base_ids, legacy_base_ids)

        legacy_ordinary_ids = [
            legacy.ordinary.next().scenario_id
            for _ in range(50)
        ]
        stratified_ordinary_ids = [
            stratified.ordinary.next().scenario_id
            for _ in range(50)
        ]
        self.assertEqual(stratified_ordinary_ids, legacy_ordinary_ids)

    def test_scheduler_state_round_trip_is_exact(self) -> None:
        scheduler = ScenarioScheduler(
            42,
            self.base + self.hard,
            self.ordinary,
            hard_neighbor_fraction=0.10,
        )
        for _ in range(37):
            scheduler._next_collision_scenario()
            scheduler.ordinary.next()
        state = scheduler.state_dict()

        restored = ScenarioScheduler(
            42,
            self.base + self.hard,
            self.ordinary,
            hard_neighbor_fraction=0.10,
        )
        restored.load_state_dict(state)
        expected = [
            (
                scheduler._next_collision_scenario().scenario_id,
                scheduler.ordinary.next().scenario_id,
            )
            for _ in range(100)
        ]
        actual = [
            (
                restored._next_collision_scenario().scenario_id,
                restored.ordinary.next().scenario_id,
            )
            for _ in range(100)
        ]
        self.assertEqual(actual, expected)

    def test_stratified_mode_requires_both_collision_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty base and hard-neighbor"):
            ScenarioScheduler(
                42,
                self.base,
                self.ordinary,
                hard_neighbor_fraction=0.20,
            )


if __name__ == "__main__":
    unittest.main()

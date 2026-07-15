from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from evaluation.artifacts import (
    atomic_write_json,
    atomic_write_npz,
    checkpoint_sha256,
    initialize_run,
    run_directory,
    valid_episode_file,
)
from evaluation.compare import compare_runs
from evaluation.metrics import ClosedTrack, aggregate_episodes
from evaluation.multiagent import MultiAgentEvaluator, TRACE_ARRAY_NAMES, collision_flags
from evaluation.schema import EVALUATION_SCHEMA_VERSION, Scenario


def make_scenario(index: int = 0, duration: float = 0.03) -> Scenario:
    return Scenario(
        map_name="Austin",
        ego_raceline="raceline1",
        opponent_raceline="raceline0",
        ego_start_index=index,
        opponent_start_index=index + 15,
        interval_index=15,
        opponent_speed_scale=0.5,
        simulation_duration_s=duration,
    )


class FakeActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = SimpleNamespace(hidden_size=2)
        self.speed_features: list[float] = []

    def forward(self, lidar: torch.Tensor, speed: torch.Tensor, hidden: torch.Tensor):
        self.speed_features.append(float(speed.item()))
        action = torch.tensor([[[0.8, 3.0]]], dtype=torch.float32, device=lidar.device)
        return action, hidden + 1.0


class FakeTracker:
    def plan(self, *args):
        return 0.1, 2.0


class FakePlanner:
    def __init__(self) -> None:
        self.tracker = FakeTracker()
        self.conf = SimpleNamespace(tracker_steps=2)
        self.plan_calls = 0

    def plan(self, *args):
        self.plan_calls += 1
        return np.zeros((2, 5), dtype=np.float64)


class FakeEnvironment:
    def __init__(self, collision_schedule: list[list[int]]) -> None:
        self.collision_schedule = collision_schedule
        self.step_index = 0
        self.actions: list[np.ndarray] = []
        self.closed = False
        self.unwrapped = SimpleNamespace(timestep=0.01)

    def _observation(self, step: int, collisions: list[int]) -> dict:
        return {
            "scans": np.stack(
                (
                    np.full(720, 10.0 + step, dtype=np.float32),
                    np.full(720, 20.0 + step, dtype=np.float32),
                )
            ),
            "poses_x": np.asarray((0.1 + 0.1 * step, 0.6 + 0.05 * step)),
            "poses_y": np.asarray((0.0, 0.0)),
            "poses_theta": np.asarray((0.0, 0.0)),
            "linear_vels_x": np.asarray((1.0 + step, 2.0 + step)),
            "collisions": np.asarray(collisions),
        }

    def reset(self, poses: np.ndarray):
        self.step_index = 0
        return self._observation(0, [0, 0]), 0.01, False, {}

    def step(self, action: np.ndarray):
        self.actions.append(np.asarray(action).copy())
        collisions = self.collision_schedule[self.step_index]
        self.step_index += 1
        # Deliberately adversarial: opponent-only collision also raises base done.
        return self._observation(self.step_index, collisions), 0.01, bool(any(collisions)), {}

    def close(self):
        self.closed = True


def fake_track_setup(scenario: Scenario):
    track = ClosedTrack.from_points(
        np.asarray(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
    )
    poses = np.asarray(((0.1, 0.0, 0.0), (0.6, 0.0, 0.0)))
    return poses, np.asarray((4.0, 3.0)), track


def make_run_directories(root: Path) -> None:
    for name in ("episodes", "traces", "videos", "errors"):
        (root / name).mkdir(parents=True, exist_ok=True)


class ScenarioAndArtifactTests(unittest.TestCase):
    def test_scenario_id_is_stable_and_encodes_contract_fields(self):
        first = make_scenario()
        second = Scenario.from_dict(json.loads(json.dumps(first.to_dict())))
        self.assertEqual(first.scenario_id, second.scenario_id)
        for token in ("map-Austin", "er-raceline1", "or-raceline0", "e-0000", "o-0015", "d-15", "s-0p5", "T-0p03"):
            self.assertIn(token, first.scenario_id)

    def test_checkpoint_sha_drives_model_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "actor.pth"
            checkpoint.write_bytes(b"checkpoint contents")
            expected = hashlib.sha256(b"checkpoint contents").hexdigest()
            actual = checkpoint_sha256(checkpoint)
            self.assertEqual(actual, expected)
            path = run_directory(temporary, "suite", checkpoint, actual, "run")
            self.assertEqual(path.parts[-2], f"actor__{expected[:12]}")

    def test_atomic_json_and_npz_leave_complete_loadable_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_write_json(root / "value.json", {"number": np.int64(3)})
            self.assertEqual(json.loads((root / "value.json").read_text()), {"number": 3})
            arrays = {"x": np.arange(6, dtype=np.float32).reshape(3, 2), "flags": np.zeros((3, 2), dtype=bool)}
            atomic_write_npz(root / "trace.npz", arrays)
            with np.load(root / "trace.npz", allow_pickle=False) as trace:
                np.testing.assert_array_equal(trace["x"], arrays["x"])
                self.assertEqual(trace["flags"].dtype, np.bool_)
            self.assertEqual(list(root.glob(".*.tmp*")), [])

    def test_resume_rejects_configuration_checkpoint_and_scenario_mismatch(self):
        scenario_manifest = {"schema_version": EVALUATION_SCHEMA_VERSION, "scenarios": [make_scenario().to_dict()]}
        manifest = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "checkpoint": {"sha256": "a" * 64},
            "config": {"trace_mode": "none"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            initialize_run(run, manifest, scenario_manifest, resume=False)
            initialize_run(run, manifest, scenario_manifest, resume=True)
            changed_config = {**manifest, "config": {"trace_mode": "all"}}
            with self.assertRaisesRegex(ValueError, "configuration"):
                initialize_run(run, changed_config, scenario_manifest, resume=True)
            changed_checkpoint = {**manifest, "checkpoint": {"sha256": "b" * 64}}
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                initialize_run(run, changed_checkpoint, scenario_manifest, resume=True)
            changed_scenarios = {"schema_version": EVALUATION_SCHEMA_VERSION, "scenarios": [make_scenario(1).to_dict()]}
            with self.assertRaisesRegex(ValueError, "scenario"):
                initialize_run(run, manifest, changed_scenarios, resume=True)

    def test_resume_episode_validation_requires_requested_trace(self):
        scenario = make_scenario()
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            episode_path = run / "episodes" / f"{scenario.scenario_id}.json"
            episode = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "scenario_id": scenario.scenario_id,
                "outcome": "follow",
                "ego_collision": False,
                "opponent_collision": False,
                "opponent_only_collision": False,
                "collision_step": None,
                "steps": 1,
                "elapsed_time_s": 0.01,
                "final_ego_progress_m": 1.0,
                "final_opp_progress_m": 2.0,
                "final_relative_progress_m": -1.0,
                "ego_distance_m": 0.01,
                "ego_mean_measured_speed_mps": 1.0,
                "ego_speed_variance": 0.0,
                "ego_min_measured_speed_mps": 1.0,
                "ego_mean_desired_speed_mps": 1.0,
                "ego_max_abs_steer_rad": 0.0,
                "ego_max_steer_delta_rad": 0.0,
                "ego_min_lidar_m": 1.0,
                "trace_path": "traces/missing.npz",
                "video_path": None,
            }
            atomic_write_json(episode_path, episode)
            self.assertTrue(valid_episode_file(episode_path, scenario.scenario_id, trace_mode="none"))
            self.assertFalse(valid_episode_file(episode_path, scenario.scenario_id, trace_mode="all"))


class CollisionAndTraceTests(unittest.TestCase):
    def _evaluate(self, collision_schedule: list[list[int]]):
        actor = FakeActor()
        environment = FakeEnvironment(collision_schedule)
        evaluator = MultiAgentEvaluator(
            actor,
            "cpu",
            environment_factory=lambda scenario: environment,
            planner_factory=lambda scenario: FakePlanner(),
            track_setup=fake_track_setup,
        )
        temporary = tempfile.TemporaryDirectory()
        run = Path(temporary.name)
        make_run_directories(run)
        episode = evaluator.evaluate_scenario(make_scenario(), run, trace_mode="all", record_video=False)
        return temporary, run, actor, environment, episode

    def test_collision_flags_are_ego_indexed(self):
        self.assertEqual(collision_flags([1, 0]), (True, False))
        self.assertEqual(collision_flags([0, 1]), (False, True))
        self.assertEqual(collision_flags([1, 1]), (True, True))

    def test_ego_collision_ends_episode_and_preserves_real_metrics(self):
        temporary, _run, _actor, environment, episode = self._evaluate([[1, 0], [0, 0], [0, 0]])
        self.addCleanup(temporary.cleanup)
        self.assertEqual(episode["outcome"], "collision")
        self.assertTrue(episode["ego_collision"])
        self.assertFalse(episode["opponent_collision"])
        self.assertEqual(episode["steps"], 1)
        self.assertGreater(episode["ego_mean_measured_speed_mps"], 0.0)
        self.assertGreater(episode["ego_distance_m"], 0.0)
        self.assertEqual(len(environment.actions), 1)

    def test_opponent_only_collision_is_recorded_and_does_not_end_episode(self):
        temporary, _run, _actor, environment, episode = self._evaluate([[0, 1], [1, 0], [0, 0]])
        self.addCleanup(temporary.cleanup)
        self.assertEqual(episode["outcome"], "collision")
        self.assertTrue(episode["opponent_collision"])
        self.assertTrue(episode["opponent_only_collision"])
        self.assertEqual(episode["collision_step"], 1)
        self.assertEqual(episode["steps"], 2)
        self.assertEqual(len(environment.actions), 2)

    def test_trace_shapes_dtypes_action_clipping_timing_and_safe_npz(self):
        temporary, run, actor, _environment, episode = self._evaluate([[0, 1], [1, 0], [0, 0]])
        self.addCleanup(temporary.cleanup)
        trace_path = run / episode["trace_path"]
        with np.load(trace_path, allow_pickle=False) as trace:
            self.assertEqual(set(trace.files), set(TRACE_ARRAY_NAMES))
            lengths = {trace[name].shape[0] for name in trace.files}
            self.assertEqual(lengths, {2})
            for name in trace.files:
                self.assertNotEqual(trace[name].dtype, object)
            self.assertEqual(trace["ego_lidar_360"].shape, (2, 360))
            self.assertEqual(trace["decision_poses"].shape, (2, 2, 3))
            self.assertEqual(trace["post_step_collisions"].shape, (2, 2))
            self.assertAlmostEqual(float(trace["ego_raw_action"][0, 0]), 0.8)
            self.assertAlmostEqual(float(trace["ego_executed_action"][0, 0]), 0.52)
            np.testing.assert_allclose(trace["decision_poses"][1], trace["post_step_poses"][0])
            self.assertEqual(float(trace["ego_lidar_360"][0, 0]), 10.0)
            self.assertEqual(float(trace["ego_lidar_360"][1, 0]), 11.0)
            np.testing.assert_allclose(trace["time_s"], [0.0, 0.01])
        np.testing.assert_allclose(actor.speed_features, [3.6, 1.0], rtol=0, atol=1e-6)


class MetricsAndComparisonTests(unittest.TestCase):
    def test_closed_track_progress_unwraps_across_seam(self):
        track = ClosedTrack.from_points(
            np.asarray(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
        )
        previous_wrapped = track.project(np.asarray((0.0, 1.0)))
        self.assertAlmostEqual(previous_wrapped, 39.0)
        wrapped, unwrapped = track.unwrap(previous_wrapped, previous_wrapped, np.asarray((1.0, 0.0)))
        self.assertAlmostEqual(wrapped, 1.0)
        self.assertAlmostEqual(unwrapped, 41.0)
        self.assertGreater(unwrapped - 39.5, 0.0)

    def test_aggregate_counts_rates_and_means(self):
        episodes = [
            {
                "outcome": "collision",
                "ego_collision": True,
                "opponent_only_collision": False,
                "ego_mean_measured_speed_mps": 2.0,
                "ego_distance_m": 3.0,
                "final_relative_progress_m": -1.0,
            },
            {
                "outcome": "overtake",
                "ego_collision": False,
                "opponent_only_collision": True,
                "ego_mean_measured_speed_mps": 4.0,
                "ego_distance_m": 5.0,
                "final_relative_progress_m": 2.0,
            },
            {
                "outcome": "follow",
                "ego_collision": False,
                "opponent_only_collision": False,
                "ego_mean_measured_speed_mps": 3.0,
                "ego_distance_m": 4.0,
                "final_relative_progress_m": -2.0,
            },
        ]
        summary = aggregate_episodes(episodes, total_scenarios=4)
        self.assertEqual(summary["completed_scenarios"], 3)
        self.assertEqual(summary["error_scenarios"], 1)
        self.assertEqual(summary["ego_collision_count"], 1)
        self.assertEqual(summary["opponent_only_collision_count"], 1)
        self.assertAlmostEqual(summary["ego_collision_rate"], 1 / 3)
        self.assertEqual(summary["mean_ego_speed_mps"], 3.0)

    def test_paired_comparison_counts_all_transitions(self):
        scenarios = [make_scenario(index) for index in range(4)]
        baseline_outcomes = ("collision", "follow", "follow", "overtake")
        candidate_outcomes = ("follow", "collision", "overtake", "follow")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            output = root / "comparison"
            for run in (baseline, candidate):
                (run / "episodes").mkdir(parents=True)
                atomic_write_json(
                    run / "scenario_manifest.json",
                    {
                        "schema_version": EVALUATION_SCHEMA_VERSION,
                        "scenarios": [scenario.to_dict() for scenario in scenarios],
                    },
                )
            for scenario, baseline_outcome, candidate_outcome in zip(
                scenarios, baseline_outcomes, candidate_outcomes
            ):
                for run, outcome in ((baseline, baseline_outcome), (candidate, candidate_outcome)):
                    atomic_write_json(
                        run / "episodes" / f"{scenario.scenario_id}.json",
                        {
                            "schema_version": EVALUATION_SCHEMA_VERSION,
                            "scenario_id": scenario.scenario_id,
                            "outcome": outcome,
                            "ego_collision": outcome == "collision",
                            "final_relative_progress_m": 1.0 if outcome == "overtake" else -1.0,
                        },
                    )
            result = compare_runs(baseline, candidate, output)
            self.assertEqual(result["fixed_ego_collisions"], 1)
            self.assertEqual(result["new_ego_collisions"], 1)
            self.assertEqual(result["gained_overtakes"], 1)
            self.assertEqual(result["lost_overtakes"], 1)
            self.assertEqual(result["outcome_transition_counts"]["follow->overtake"], 1)
            self.assertTrue((output / "comparison.json").is_file())
            self.assertEqual(len((output / "comparison.csv").read_text().splitlines()), 5)


if __name__ == "__main__":
    unittest.main()

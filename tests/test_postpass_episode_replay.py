from __future__ import annotations

from pathlib import Path
import unittest

from scripts.validate_postpass_reward import (
    TrackProjector,
    replay_episode_geometry,
    setting_episode_result,
    validate_trace,
)


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = (
    ROOT / "eval_results" / "end2race_Austin" / "multiagents" / "traces"
)


@unittest.skipUnless(TRACE_ROOT.is_dir(), "Austin600 terminal traces are unavailable")
class PostPassRealEpisodeReplayTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.projector = TrackProjector(
            ROOT / "f1tenth_racetracks" / "Austin" / "raceline1.csv"
        )

    def replay(self, episode_key: str, collision: bool):
        trace = validate_trace(
            TRACE_ROOT / f"{episode_key}.npz",
            expected_collision=collision,
        )
        geometry = replay_episode_geometry(trace, self.projector)
        return setting_episode_result(
            geometry,
            trace["time_s"],
            pass_margin_m=0.05,
            safe_rear_gap_m=0.60,
            closing_deadband_mps=0.10,
            clear_mode="latched",
        )

    def test_primary_tail_collision_has_preterminal_signal(self):
        result = self.replay("ol0_e1283_o1279_s0.7", collision=True)

        self.assertTrue(result["pass_detected"])
        self.assertTrue(result["preterminal_triggered"])
        self.assertGreater(result["preterminal_trigger_steps"], 1)
        self.assertGreater(result["first_trigger_lead_to_terminal_s"], 0.10)

    def test_successful_overtake_has_small_but_nonzero_wait_signal(self):
        result = self.replay("ol0_e0_o15_s0.5", collision=False)

        self.assertTrue(result["pass_detected"])
        self.assertTrue(result["preterminal_triggered"])
        self.assertGreater(result["basis_sum_m"], 0.0)
        self.assertLess(result["basis_sum_m"], 0.02)
        self.assertGreater(result["proposed_reward_sum"], -0.02)
        self.assertLess(result["proposed_reward_sum"], 0.0)

    def test_ordinary_follow_has_no_postpass_phase_or_penalty(self):
        result = self.replay("ol0_e0_o15_s0.8", collision=False)

        self.assertFalse(result["pass_detected"])
        self.assertFalse(result["triggered"])
        self.assertEqual(result["basis_sum_m"], 0.0)
        self.assertEqual(result["proposed_reward_sum"], 0.0)

    def test_prepass_collision_is_not_misclassified_as_postpass(self):
        result = self.replay("ol0_e727_o739_s0.5", collision=True)

        self.assertFalse(result["pass_detected"])
        self.assertFalse(result["triggered"])
        self.assertEqual(result["basis_sum_m"], 0.0)

    def test_opponent_collision_before_pass_disables_treatment(self):
        result = self.replay("ol0_e1368_o1370_s0.8", collision=False)

        self.assertTrue(result["pass_detected"])
        self.assertEqual(result["active_steps"], 0)
        self.assertFalse(result["triggered"])
        self.assertEqual(result["basis_sum_m"], 0.0)


if __name__ == "__main__":
    unittest.main()

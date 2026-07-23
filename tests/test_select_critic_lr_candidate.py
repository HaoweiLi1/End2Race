import json
from pathlib import Path
import tempfile
import unittest

from scripts.select_critic_lr_candidate import CANDIDATES, LATE_UPDATES, select


class CriticLearningRateCandidateSelectionTests(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _args_for(candidate):
        return {
            "critic": "privilege_gru",
            "env_workers": 12,
            "batch_size": 12800,
            "num_updates": 45,
            "actor_epochs": 2,
            "critic_epochs": 5,
            "gru_learning_rate": 3.0e-6,
            "head_learning_rate": 3.0e-5,
            "critic_learning_rate": 3.0e-4,
            "steering_latent_std": 0.03,
            "speed_physical_std": 0.15,
            "clip_range": 0.20,
            "target_kl": None,
            "hard_neighbors": candidate.hard_neighbors,
            "hard_neighbor_fraction": candidate.hard_neighbor_fraction,
        }

    def _write_candidate(
        self,
        candidate,
        collisions,
        overtakes,
        *,
        error_update=None,
    ):
        run_dir = self.root / "post-trained" / candidate.run_name
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (run_dir / "run_config.json").write_text(
            json.dumps({"args": self._args_for(candidate)}),
            encoding="utf-8",
        )
        for update, collision_count, overtake_count in zip(
            LATE_UPDATES,
            collisions,
            overtakes,
        ):
            (checkpoint_dir / f"actor_u{update:04d}.pth").write_bytes(b"actor")
            following_count = 600 - collision_count - overtake_count
            error_count = 1 if update == error_update else 0
            result_dir = (
                self.root
                / "eval_results"
                / f"{candidate.run_name}_u{update:04d}_Austin"
                / "multiagents"
            )
            result_dir.mkdir(parents=True)
            episodes = {
                f"episode-{index:03d}": {"episode_key": f"episode-{index:03d}"}
                for index in range(600)
            }
            (result_dir / "results_multi.json").write_text(
                json.dumps(
                    {
                        "final": {
                            "total_episodes": 600,
                            "following_count": following_count,
                            "overtaking_count": overtake_count,
                            "collision_count": collision_count,
                            "error_count": error_count,
                            "avg_speed_mean": 5.0,
                        },
                        "episodes": episodes,
                    }
                ),
                encoding="utf-8",
            )

    def test_selects_aggregate_late_collision_winner(self):
        self._write_candidate(CANDIDATES[0], [12, 11, 12], [350, 355, 357])
        self._write_candidate(CANDIDATES[1], [20, 22, 17], [366, 365, 365])
        self._write_candidate(CANDIDATES[2], [10, 13, 11], [360, 358, 359])
        self._write_candidate(CANDIDATES[3], [11, 12, 12], [362, 360, 361])

        result = select(self.root)

        self.assertEqual(result["selected_candidate"]["key"], "hard020")
        self.assertEqual(
            result["selected_candidate"]["best_late_checkpoint"]["update"],
            35,
        )
        self.assertEqual(
            result["critic_lr_experiment"]["hard_neighbor_fraction"],
            0.20,
        )
        self.assertEqual(
            result["critic_lr_experiment"]["starts_from"],
            "pretrained/end2race.pth",
        )

    def test_rejects_incomplete_evaluation(self):
        self._write_candidate(CANDIDATES[0], [12, 11, 12], [350, 355, 357])
        self._write_candidate(
            CANDIDATES[2],
            [10, 13, 11],
            [360, 358, 359],
            error_update=40,
        )
        self._write_candidate(CANDIDATES[1], [20, 22, 17], [366, 365, 365])
        self._write_candidate(CANDIDATES[3], [11, 12, 12], [362, 360, 361])

        with self.assertRaisesRegex(ValueError, "contains 1 errors"):
            select(self.root)


if __name__ == "__main__":
    unittest.main()

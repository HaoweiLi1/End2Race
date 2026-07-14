#!/usr/bin/env python3
"""B7 288-row merge, paired gate, and L4 statistic regression."""

import csv
import hashlib
from pathlib import Path
import tempfile

from bplus_v22.b7_eval import (
    B7EvaluationShard,
    exact_cluster_signflip_one_sided,
    file_sha256,
    merge_candidate_shards,
)
from bplus_v22.ppo_eval import read_task8_development


REPO = Path(__file__).resolve().parent.parent
DEVELOPMENT = REPO / "Experiments/B1_route_r2_scaffold/artifacts/task8_manifests_20260712_113241/development_scenarios.tsv"
BC = REPO / "pretrained/end2race.pth"


def main() -> None:
    cases = read_task8_development(DEVELOPMENT, hashlib.sha256(DEVELOPMENT.read_bytes()).hexdigest())
    with tempfile.TemporaryDirectory(prefix="b7-eval-test-") as directory:
        root = Path(directory)
        baseline_path = root / "baseline.tsv"
        fields = (
            "variant",
            "task8_row_index",
            "manifest_order",
            "l2_id",
            "l4_id",
            "collision_any",
            "terminal_overtake",
            "four_state",
            "checkpoint_sha256",
            "scenario_manifest_sha256",
            "trajectory_sha256",
        )
        with baseline_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for index, case in enumerate(cases):
                collision = index < 24
                overtake = 24 <= index < 162
                writer.writerow(
                    {
                        "variant": "BC",
                        "task8_row_index": index,
                        "manifest_order": case["manifest_order"],
                        "l2_id": case["l2_id"],
                        # Give every changed occurrence its own cluster so the
                        # exact directional tail is transparent.
                        "l4_id": f"L4:{index:064x}",
                        "collision_any": collision,
                        "terminal_overtake": overtake,
                        "four_state": "collision" if collision else ("confirmed_pass" if overtake else "safe_follow"),
                        "checkpoint_sha256": "b" * 64,
                        "scenario_manifest_sha256": "c" * 64,
                        "trajectory_sha256": hashlib.sha256(f"BC:{index}".encode()).hexdigest(),
                    }
                )
        rows_by_shard = [[] for _ in range(4)]
        # Fix eight BC collisions and introduce two new collisions: 24 -> 18,
        # net paired effect 6. Preserve all overtakes.
        collision_indices = set(range(8, 24)) | {200, 201}
        for index, case in enumerate(cases):
            before_collision = index < 24
            before_overtake = 24 <= index < 162
            collision = index in collision_indices
            row = {
                "task8_row_index": index,
                "l4_id": f"L4:{index:064x}",
                "collision_any": collision,
                "terminal_overtake": before_overtake,
                "fixed_collision": before_collision and not collision,
                "new_collision": not before_collision and collision,
                "gained_overtake": False,
                "lost_overtake": False,
                "deterministic_speed_projection_count": 0,
            }
            rows_by_shard[index % 4].append(row)
        plan_sha = "d" * 64
        baseline_sha = file_sha256(baseline_path)
        candidate_sha = file_sha256(BC)
        shards = [
            B7EvaluationShard(
                shard_index=index,
                shard_count=4,
                candidate_checkpoint_sha256=candidate_sha,
                training_run_plan_sha256=plan_sha,
                baseline_rows_sha256=baseline_sha,
                rows=tuple(rows),
            )
            for index, rows in enumerate(rows_by_shard)
        ]
        merged, summary = merge_candidate_shards(
            shards=shards,
            task8_rows=cases,
            baseline_rows_path=baseline_path,
            candidate_path=BC,
            training_run_plan_sha256=plan_sha,
        )
        assert len(merged) == 288
        assert summary["candidate"]["collision"] == 18
        assert summary["candidate"]["fixed_minus_new"] == 6
        assert summary["candidate"]["l4_cluster_signflip_one_sided_p"] <= 0.10
        assert summary["seed1_minimum_continue_gate_pass"] is True
        assert summary["opened_development_target_collision_le_16"] is False
        assert exact_cluster_signflip_one_sided([1, 1]) == 0.25
    print("B7 288-row evaluation contracts passed")


if __name__ == "__main__":
    main()

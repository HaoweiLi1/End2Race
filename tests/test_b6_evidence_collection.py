#!/usr/bin/env python3
"""Compact B6 episode-ledger collection contract."""

import json
from pathlib import Path
import tempfile

from bplus_v22.b6_temporal import B6_EXPECTED_EPISODES
from scripts.collect_b6_temporal_phase0_evidence import collect


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        episodes = root / "episodes"
        episodes.mkdir()
        for index in range(B6_EXPECTED_EPISODES):
            collision = index % 3 == 0
            overtake = index % 3 == 1
            row = {
                "schema": "end2race-b6-temporal-phase0-episode-1",
                "task_id": f"task-{index:04d}",
                "task_order": index,
                "run_plan_sha256": "a" * 64,
                "execution_source_commit": "b" * 40,
                "collision_any": collision,
                "corrected_outcome": (
                    "collision" if collision else ("overtake" if overtake else "follow")
                ),
                "terminal_reward": -2.0 if collision else (1.0 if overtake else 0.0),
            }
            (episodes / f"{index:04d}.json").write_text(json.dumps(row) + "\n")
        output = root / "episode_results.jsonl"
        report = collect(episodes, output)
        assert report["episode_count"] == B6_EXPECTED_EPISODES
        assert len(report["output_sha256"]) == 64
        packed = [json.loads(line) for line in output.read_text().splitlines()]
        assert [row["task_order"] for row in packed] == list(range(B6_EXPECTED_EPISODES))
        assert not list(root.glob("*.partial"))
    print("B6 evidence collection contract passed")


if __name__ == "__main__":
    main()

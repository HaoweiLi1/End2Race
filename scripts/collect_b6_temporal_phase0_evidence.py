#!/usr/bin/env python3
"""Pack atomic B6 episode JSON files into one reviewable JSONL ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from bplus_v22.b6_temporal import B6_EXPECTED_EPISODES


RESULT_SCHEMA = "end2race-b6-temporal-phase0-episode-1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(episode_root: Path, output: Path) -> dict[str, object]:
    rows = [json.loads(path.read_text()) for path in episode_root.glob("*.json")]
    if len(rows) != B6_EXPECTED_EPISODES:
        raise ValueError(
            f"B6 collection expected {B6_EXPECTED_EPISODES} rows, found {len(rows)}"
        )
    rows.sort(key=lambda row: int(row["task_order"]))
    if [int(row["task_order"]) for row in rows] != list(range(B6_EXPECTED_EPISODES)):
        raise ValueError("B6 task order is not complete and contiguous")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("B6 task identities are not unique")
    if {row.get("schema") for row in rows} != {RESULT_SCHEMA}:
        raise ValueError("B6 episode schema drift")
    if len({row["run_plan_sha256"] for row in rows}) != 1:
        raise ValueError("B6 episode RunPlan identity drift")
    if len({row["execution_source_commit"] for row in rows}) != 1:
        raise ValueError("B6 episode source identity drift")
    if any(
        float(row["terminal_reward"])
        != (-2.0 if row["collision_any"] else (1.0 if row["corrected_outcome"] == "overtake" else 0.0))
        for row in rows
    ):
        raise ValueError("B6 episode terminal reward ledger drift")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError(output if output.exists() else temporary)
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return {
        "episode_count": len(rows),
        "run_plan_sha256": rows[0]["run_plan_sha256"],
        "execution_source_commit": rows[0]["execution_source_commit"],
        "output_sha256": sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(args.episode_root.resolve(), args.output.resolve()),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

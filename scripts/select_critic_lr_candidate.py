#!/usr/bin/env python3
"""Select the late-checkpoint winner for the critic-LR 5e-4 experiment.

The four candidate configurations are fixed deliberately:

* no hard-neighbor sampling;
* the original uniform merged hard-neighbor pool (40.5% realized share);
* 20% hard-neighbor sampling within collision resets;
* 10% hard-neighbor sampling within collision resets.

Configuration selection uses the aggregate U35/U40/U45 evaluation trajectory,
not the single best checkpoint.  The selected checkpoint is recorded for audit,
but the follow-up critic-LR experiment must start from the canonical BC actor so
that critic learning rate remains the only optimizer change.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any


LATE_UPDATES = (35, 40, 45)
EXPECTED_EPISODES = 600


@dataclass(frozen=True)
class Candidate:
    key: str
    run_name: str
    hard_neighbors: bool
    hard_neighbor_fraction: float | None


CANDIDATES = (
    Candidate(
        key="nohard",
        run_name="ppo_privilege_gru_0722_long45_clip020",
        hard_neighbors=False,
        hard_neighbor_fraction=None,
    ),
    Candidate(
        key="hardfull",
        run_name="ppo_privilege_gru_0722_long45_clip020_hard",
        hard_neighbors=True,
        hard_neighbor_fraction=None,
    ),
    Candidate(
        key="hard020",
        run_name="ppo_privilege_gru_0723_long45_clip020_hard020",
        hard_neighbors=True,
        hard_neighbor_fraction=0.20,
    ),
    Candidate(
        key="hard010",
        run_name="ppo_privilege_gru_0723_long45_clip020_hard010",
        hard_neighbors=True,
        hard_neighbor_fraction=0.10,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select no-hard/original-hard/20%-hard/10%-hard using aggregate "
            "U35/U40/U45 evaluation results and emit the winning candidate key."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="End2Race repository root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results/critic_lr5e4_candidate_selection.json"),
        help="Selection audit JSON, relative to --root unless absolute",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Required artifact is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require_equal(name: str, actual: Any, expected: Any, path: Path) -> None:
    if isinstance(expected, float):
        matches = isinstance(actual, (int, float)) and math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(
            f"{path}: expected {name}={expected!r}, got {actual!r}"
        )


def validate_run_config(root: Path, candidate: Candidate) -> None:
    path = root / "post-trained" / candidate.run_name / "run_config.json"
    document = load_json(path)
    args = document.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"{path}: missing args object")

    expected = {
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
    for name, expected_value in expected.items():
        require_equal(name, args.get(name), expected_value, path)


def load_checkpoint_evaluation(
    root: Path,
    candidate: Candidate,
    update: int,
) -> dict[str, Any]:
    checkpoint = (
        root
        / "post-trained"
        / candidate.run_name
        / "checkpoints"
        / f"actor_u{update:04d}.pth"
    )
    if not checkpoint.is_file():
        raise ValueError(f"Required checkpoint is missing: {checkpoint}")

    path = (
        root
        / "eval_results"
        / f"{candidate.run_name}_u{update:04d}_Austin"
        / "multiagents"
        / "results_multi.json"
    )
    document = load_json(path)
    final = document.get("final")
    episodes = document.get("episodes")
    if not isinstance(final, dict) or not isinstance(episodes, dict):
        raise ValueError(f"{path}: expected final and episodes objects")

    total = int(final.get("total_episodes", -1))
    following = int(final.get("following_count", -1))
    overtaking = int(final.get("overtaking_count", -1))
    collisions = int(final.get("collision_count", -1))
    errors = int(final.get("error_count", -1))
    if total != EXPECTED_EPISODES:
        raise ValueError(
            f"{path}: expected {EXPECTED_EPISODES} total episodes, got {total}"
        )
    if len(episodes) != EXPECTED_EPISODES:
        raise ValueError(
            f"{path}: expected {EXPECTED_EPISODES} episode rows, got {len(episodes)}"
        )
    if errors != 0:
        raise ValueError(f"{path}: evaluation contains {errors} errors")
    if following + overtaking + collisions != EXPECTED_EPISODES:
        raise ValueError(
            f"{path}: following + overtaking + collision does not equal "
            f"{EXPECTED_EPISODES}"
        )

    return {
        "update": update,
        "collision_count": collisions,
        "overtaking_count": overtaking,
        "following_count": following,
        "avg_speed_mean": float(final["avg_speed_mean"]),
        "results_path": str(path.relative_to(root)),
        "checkpoint_path": str(checkpoint.relative_to(root)),
    }


def summarize_candidate(root: Path, candidate: Candidate) -> dict[str, Any]:
    validate_run_config(root, candidate)
    checkpoints = [
        load_checkpoint_evaluation(root, candidate, update)
        for update in LATE_UPDATES
    ]
    collision_sum = sum(row["collision_count"] for row in checkpoints)
    overtake_sum = sum(row["overtaking_count"] for row in checkpoints)
    collision_max = max(row["collision_count"] for row in checkpoints)
    final_collisions = checkpoints[-1]["collision_count"]
    best_checkpoint = min(
        checkpoints,
        key=lambda row: (
            row["collision_count"],
            -row["overtaking_count"],
            -row["update"],
        ),
    )
    return {
        "key": candidate.key,
        "run_name": candidate.run_name,
        "hard_neighbors": candidate.hard_neighbors,
        "hard_neighbor_fraction": candidate.hard_neighbor_fraction,
        "late_updates": list(LATE_UPDATES),
        "late_collision_sum": collision_sum,
        "late_collision_mean": collision_sum / len(checkpoints),
        "late_collision_max": collision_max,
        "late_overtake_sum": overtake_sum,
        "late_overtake_mean": overtake_sum / len(checkpoints),
        "u45_collision_count": final_collisions,
        "checkpoints": checkpoints,
        "best_late_checkpoint": best_checkpoint,
    }


def candidate_score(summary: dict[str, Any]) -> tuple[Any, ...]:
    """Safety-first score over the late trajectory, then performance."""

    return (
        summary["late_collision_sum"],
        summary["late_collision_max"],
        -summary["late_overtake_sum"],
        summary["u45_collision_count"],
        summary["key"],
    )


def select(root: Path) -> dict[str, Any]:
    summaries = [summarize_candidate(root, candidate) for candidate in CANDIDATES]
    winner = min(summaries, key=candidate_score)
    output_name = (
        "ppo_privilege_gru_0723_long45_clip020_criticlr5e4_"
        f"{winner['key']}"
    )
    return {
        "selection_rule": [
            "minimum aggregate collision count over U35/U40/U45",
            "minimum worst-checkpoint collision count over U35/U40/U45",
            "maximum aggregate overtake count over U35/U40/U45",
            "minimum U45 collision count",
            "candidate key for deterministic final tie-break",
        ],
        "selection_scope": {
            "updates": list(LATE_UPDATES),
            "episodes_per_checkpoint": EXPECTED_EPISODES,
            "map": "Austin",
        },
        "candidates": summaries,
        "selected_candidate": winner,
        "critic_lr_experiment": {
            "starts_from": "pretrained/end2race.pth",
            "critic_learning_rate": 5.0e-4,
            "hard_neighbors": winner["hard_neighbors"],
            "hard_neighbor_fraction": winner["hard_neighbor_fraction"],
            "output_dir": f"post-trained/{output_name}",
            "note": (
                "The selected PPO checkpoint is recorded for audit only. "
                "The critic-LR experiment restarts from BC to preserve a "
                "single-variable comparison."
            ),
        },
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = root / output
    try:
        result = select(root)
        atomic_write_json(output, result)
    except ValueError as error:
        print(f"candidate selection failed: {error}", file=sys.stderr)
        return 2

    winner = result["selected_candidate"]
    best = winner["best_late_checkpoint"]
    print(
        "Selected critic-LR candidate "
        f"{winner['key']} ({winner['run_name']}): "
        f"late collisions={winner['late_collision_sum']}, "
        f"late overtakes={winner['late_overtake_sum']}, "
        f"best checkpoint=U{best['update']} "
        f"({best['collision_count']} collisions, "
        f"{best['overtaking_count']} overtakes). "
        f"Audit: {output}",
        file=sys.stderr,
    )
    print(winner["key"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

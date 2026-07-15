"""Scenario-paired comparison of two completed evaluation runs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.artifacts import atomic_write_csv, atomic_write_json, load_json
from evaluation.schema import EVALUATION_SCHEMA_VERSION


COMPARISON_FIELDS = (
    "scenario_id",
    "baseline_outcome",
    "candidate_outcome",
    "baseline_ego_collision",
    "candidate_ego_collision",
    "fixed_ego_collision",
    "new_ego_collision",
    "gained_overtake",
    "lost_overtake",
    "baseline_final_relative_progress_m",
    "candidate_final_relative_progress_m",
)


def _scenario_ids(run_dir: Path) -> list[str]:
    manifest = load_json(run_dir / "scenario_manifest.json")
    if manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported scenario manifest schema in {run_dir}")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError(f"Scenario manifest does not contain a scenario list: {run_dir}")
    identifiers = [str(scenario["scenario_id"]) for scenario in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Scenario manifest contains duplicate IDs: {run_dir}")
    return identifiers


def _episodes(run_dir: Path, scenario_ids: list[str]) -> dict[str, dict[str, Any]]:
    episodes: dict[str, dict[str, Any]] = {}
    for scenario_id in scenario_ids:
        path = run_dir / "episodes" / f"{scenario_id}.json"
        if not path.is_file():
            raise ValueError(f"Run is incomplete; missing episode: {path}")
        episode = load_json(path)
        if episode.get("scenario_id") != scenario_id:
            raise ValueError(f"Episode ID does not match its filename: {path}")
        if episode.get("outcome") not in {"collision", "overtake", "follow"}:
            raise ValueError(f"Episode has an invalid outcome: {path}")
        episodes[scenario_id] = episode
    return episodes


def compare_runs(
    baseline_run: str | Path,
    candidate_run: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    baseline_path = Path(baseline_run).resolve()
    candidate_path = Path(candidate_run).resolve()
    destination = Path(output_dir)
    baseline_ids = _scenario_ids(baseline_path)
    candidate_ids = _scenario_ids(candidate_path)
    if set(baseline_ids) != set(candidate_ids):
        missing_candidate = sorted(set(baseline_ids) - set(candidate_ids))
        missing_baseline = sorted(set(candidate_ids) - set(baseline_ids))
        raise ValueError(
            "Scenario sets are incompatible: "
            f"missing from candidate={missing_candidate}, missing from baseline={missing_baseline}"
        )
    baseline_episodes = _episodes(baseline_path, baseline_ids)
    candidate_episodes = _episodes(candidate_path, baseline_ids)

    rows: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    for scenario_id in sorted(baseline_ids):
        baseline = baseline_episodes[scenario_id]
        candidate = candidate_episodes[scenario_id]
        baseline_outcome = str(baseline["outcome"])
        candidate_outcome = str(candidate["outcome"])
        baseline_collision = bool(baseline["ego_collision"])
        candidate_collision = bool(candidate["ego_collision"])
        transitions[f"{baseline_outcome}->{candidate_outcome}"] += 1
        rows.append(
            {
                "scenario_id": scenario_id,
                "baseline_outcome": baseline_outcome,
                "candidate_outcome": candidate_outcome,
                "baseline_ego_collision": baseline_collision,
                "candidate_ego_collision": candidate_collision,
                "fixed_ego_collision": baseline_collision and not candidate_collision,
                "new_ego_collision": not baseline_collision and candidate_collision,
                "gained_overtake": baseline_outcome != "overtake" and candidate_outcome == "overtake",
                "lost_overtake": baseline_outcome == "overtake" and candidate_outcome != "overtake",
                "baseline_final_relative_progress_m": baseline["final_relative_progress_m"],
                "candidate_final_relative_progress_m": candidate["final_relative_progress_m"],
            }
        )

    all_outcomes = ("collision", "overtake", "follow")
    transition_counts = {
        f"{source}->{target}": int(transitions[f"{source}->{target}"])
        for source in all_outcomes
        for target in all_outcomes
    }
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "baseline_run": str(baseline_path),
        "candidate_run": str(candidate_path),
        "paired_scenarios": len(rows),
        "fixed_ego_collisions": sum(bool(row["fixed_ego_collision"]) for row in rows),
        "new_ego_collisions": sum(bool(row["new_ego_collision"]) for row in rows),
        "gained_overtakes": sum(bool(row["gained_overtake"]) for row in rows),
        "lost_overtakes": sum(bool(row["lost_overtake"]) for row in rows),
        "outcome_transition_counts": transition_counts,
    }
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "comparison.json", result)
    atomic_write_csv(destination / "comparison.csv", rows, COMPARISON_FIELDS)
    return result

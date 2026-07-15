#!/usr/bin/env python3
"""End-to-end synthetic tree tests for D0.1 scan and atomic outputs."""

import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from d0.identity import geometry_manifest
from d0.scan import run_scan, validate_emitted_output


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_assets(root):
    directory = root / "assets" / "Austin"
    directory.mkdir(parents=True)
    xs = [0.0, 0.5, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 0.0]
    lines = ["s;x;y;heading;kappa;speed"]
    for index, x in enumerate(xs):
        speed = 5.0 if index in (0, len(xs) - 1) else 5.0 + index * 0.01
        lines.append(f"{index};{x};0.0;0.0;0.0;{speed}")
    (directory / "raceline1.csv").write_text("\n".join(lines) + "\n")


def make_npz(path, rel, raw_label, collision=False, ego_collision=False, opp_collision=False):
    rel = np.asarray(rel, dtype=np.float64)
    n = len(rel) - 1
    time = np.arange(n, dtype=np.float64) * 0.01
    zeros = np.zeros(n, dtype=np.float64)
    poses = np.zeros((n, 3), dtype=np.float64)
    lidars = np.ones((n, 360), dtype=np.float64)
    final_dist = 0.5 if collision else 2.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        time=time,
        ego_lidar=lidars,
        opp_lidar=lidars,
        ego_desired_steer=zeros,
        ego_desired_speed=np.full(n, 5.0),
        ego_actual_speed=np.full(n, 4.9),
        ego_pose=poses,
        ego_progress=rel[:-1],
        opp_desired_steer=zeros,
        opp_desired_speed=np.full(n, 4.0),
        opp_actual_speed=np.full(n, 3.9),
        opp_pose=poses,
        opp_progress=zeros,
        collision=np.array(collision, dtype=bool),
        ego_collision=np.array(ego_collision, dtype=bool),
        opp_collision=np.array(opp_collision, dtype=bool),
        final_time=np.float64(time[-1] + 0.01),
        final_ego_pose=np.array([0.0, 0.0, 0.0]),
        final_opp_pose=np.array([final_dist, 0.0, 0.0]),
        final_ego_progress=np.float64(rel[-1]),
        final_opp_progress=np.float64(0.0),
        state_label=np.array(raw_label),
    )


def build_tree(root):
    write_assets(root)
    (root / "pretrained").mkdir()
    model_paths = {}
    for model in ("bc", "cand160"):
        path = root / "pretrained" / f"{model}.pth"
        path.write_bytes(f"checkpoint-{model}".encode())
        model_paths[model] = path

    config = {
        "schema": "d0.1-runconfig-1",
        "analysis_version": "d0.1",
        "classifier_version": "d0.1-traj-1",
        "source_run_id": "synthetic",
        "repository_root": str(root),
        "eval_root": "eval_results",
        "assets_root": "assets",
        "goal_root": "logs/goal",
        "opened_registry": "logs/goal/opened_registry.tsv",
        "opened_at_utc": "2026-07-10T23:02:12+08:00",
        "tag_template": "p1v_{run}_{model}_{map}_off{offset}",
        "result_dir_template": "eval_results/{tag}_{map}",
        "models": {
            model: {"path": str(path.relative_to(root)), "sha256": sha(path)}
            for model, path in model_paths.items()
        },
        "grids": [["Austin", 1]],
        "offset_start_formula": "test",
        "zero_start_formula": "test",
        "dev_start_formula": "test",
        "max_wc_convention": "line count minus 2",
        "ego_raceline": "raceline1",
        "opponent_racelines": ["raceline1"],
        "opponent_speedscales": [0.5, 0.6],
        "interval_idx": 1,
        "sim_dt": 0.01,
        "duration_s": 0.81,
        "duration_ticks": 81,
        "noise": 0.0,
        "noise_seed": 42,
        "expected_occurrences": {"smoke": 16, "full": 16},
        "bootstrap": {"B": 100, "seed": 20260710},
        "classifier": {
            "attempt_m": 0.6,
            "confirmed_lead_m": 2.0,
            "confirmed_hold_s": 0.7,
            "car_distance_m": 1.0,
            "alongside_strict_m": 0.6,
        },
        "reconciliation_targets": {
            "synthetic_fixture": {
                "source": "tests/test_d0_scan.py",
                "modes": ["full"],
                "values": {
                    "geometry_reconciliation.exact_N": 8,
                },
            }
        },
        "_test_start_count": 4,
        "_strict_canonical_contract": False,
    }
    s0 = geometry_manifest(config)
    check("fixture-exact-eight", len(s0.sets["exact"]) == 8)
    check("fixture-primary-six", len(s0.sets["primary"]) == 6)
    check("fixture-sensb-four", len(s0.sets["sensB"]) == 4)

    by_grid = {}
    for occurrence in s0.occurrences:
        by_grid.setdefault(occurrence["grid_id"], []).append(occurrence)
    track_length = 32.0
    for model in config["models"]:
        for map_name, offset in config["grids"]:
            tag = config["tag_template"].format(run="synthetic", model=model, map=map_name, offset=offset)
            result_dir = root / config["result_dir_template"].format(tag=tag, map=map_name)
            episodes = {}
            outcome_counts = {"following": 0, "overtaking": 0, "collision": 0}
            for index, occurrence in enumerate(sorted(by_grid[f"{map_name}_off{offset}"], key=lambda x: x["episode_key"])):
                key = occurrence["episode_key"]
                if index == 0:
                    raw = "overtaking"
                    rel = np.linspace(track_length - 5.0, track_length - 2.0, 82)
                    collision = ego_collision = opp_collision = False
                    state_dir = "overtake"
                elif index == 1:
                    raw = "collision"
                    rel = np.linspace(-2.0, -0.2, 82)
                    collision = ego_collision = True
                    opp_collision = False
                    state_dir = "collision"
                elif index == 2:
                    raw = "overtaking"
                    rel = np.concatenate([np.linspace(-2.0, 1.9, 10), np.full(71, 2.1), [2.2]])
                    collision = ego_collision = opp_collision = False
                    state_dir = "overtake"
                else:
                    raw = "following"
                    rel = np.linspace(-2.0, -1.0, 82)
                    collision = ego_collision = opp_collision = False
                    state_dir = "follow"
                prefix = {"following": "f", "overtaking": "o", "collision": "c"}[raw]
                npz_rel = Path(config["eval_root"]) / f"{tag}_{map_name}" / state_dir / f"{prefix}_{key}.npz"
                make_npz(root / npz_rel, rel, raw, collision, ego_collision, opp_collision)
                episodes[key] = {
                    "state": {"following": 1, "overtaking": 2, "collision": 3}[raw],
                    "state_label": raw,
                    "outcome": raw,
                    "ego_collision": ego_collision,
                    "opp_collision": opp_collision,
                    "map_name": map_name,
                    "ego_raceline": "raceline1",
                    "opp_raceline": occurrence["opponent_raceline"],
                    "ego_idx": occurrence["raw_ego_idx"],
                    "opp_idx": occurrence["resolved_opp_idx"],
                    "interval_idx": 1,
                    "opp_speedscale": float.fromhex(occurrence["speedscale_hex"]),
                    "sim_duration": 0.81,
                    "noise": 0.0,
                    "npz_path": npz_rel.as_posix(),
                    "collision_occurred": collision,
                }
                outcome_counts[raw] += 1
            result_dir.mkdir(parents=True, exist_ok=True)
            results = {
                "episodes": episodes,
                "final": {
                    "total_episodes": len(episodes),
                    "following_count": outcome_counts["following"],
                    "overtaking_count": outcome_counts["overtaking"],
                    "collision_count": outcome_counts["collision"],
                    "error_count": 0,
                    "ego_collision_count": outcome_counts["collision"],
                    "opp_collision_count": 0,
                    "validated": True,
                },
            }
            (result_dir / "results.json").write_text(json.dumps(results, sort_keys=True))
    return config


def read_tsv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def output_hashes(path):
    result = {}
    for item in sorted(Path(path).iterdir()):
        if item.is_file() and item.name != "COMPLETE":
            result[item.name] = sha(item)
    return result


def refresh_manifest_entry(directory, name):
    manifest = Path(directory) / "output_manifest.sha256"
    digest = sha(Path(directory) / name)
    rows = []
    for line in manifest.read_text().splitlines():
        prior, prior_name = line.split("  ", 1)
        rows.append(f"{digest if prior_name == name else prior}  {prior_name}")
    manifest.write_text("\n".join(rows) + "\n")


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "repo"
        root.mkdir()
        config = build_tree(root)

        output = root / "logs" / "goal" / "full_a"
        rc = run_scan("full", output, config, workers=1)
        check("scan-success", rc == 0)
        check("complete-last", (output / "COMPLETE").is_file() and not Path(str(output) + ".partial").exists())
        validation = json.loads((output / "d0_validation.json").read_text())
        check("release-pass", validation["release"]["passed"] is True)
        s0_manifest = json.loads((output / "s0_manifest.json").read_text())
        check("three-estimand-sizes", [s0_manifest["counts"][key] for key in ("exact", "primary", "sensB")] == [8, 6, 4])

        canonical = read_tsv(output / "canonical_episodes.tsv")
        check("canonical-model-rows", len(canonical) == 16)
        corrections = read_tsv(output / "outcome_corrections.tsv")
        check("correction-complete", len(corrections) == 2)
        collisions = read_tsv(output / "collision_events.tsv")
        check("collision-direct-inferred", len(collisions) == 2 and {row["involvement"] for row in collisions} == {"ego_only"} and {row["cause"] for row in collisions} == {"car"})

        matrices = read_tsv(output / "transition_matrix_primary.tsv")
        grouped = {}
        for row in matrices:
            grouped.setdefault(row["matrix_id"], []).append(row)
        check("matrix-sixteen-cells", all(len(rows) == 16 for rows in grouped.values()))
        check("matrix-sums", all(sum(int(row["count"]) for row in rows) == int(rows[0]["expected_n"]) for rows in grouped.values()))
        check("registry-appended", len(read_tsv(root / config["opened_registry"])) == 8)
        check("independent-validator", validate_emitted_output(output).passed)

        summary = json.loads((output / "d0_summary.json").read_text())
        check(
            "summary-required-sections",
            all(
                key in summary
                for key in (
                    "bc_breakdown",
                    "strata",
                    "collision_phases",
                    "opponent_only_floor",
                    "correction_counts",
                    "reconciliation",
                )
            ),
        )
        fixture_reference = summary["reconciliation"]["references"]["synthetic_fixture"]
        check(
            "summary-target-reconciliation",
            fixture_reference["status"] == "match"
            and fixture_reference["checks"][0]["observed"] == 8,
        )

        corrupt_summary = root / "logs" / "goal" / "corrupt_summary"
        shutil.copytree(output, corrupt_summary)
        corrupt = json.loads((corrupt_summary / "d0_summary.json").read_text())
        corrupt["estimands"]["primary"]["bc"]["collision"] += 1
        (corrupt_summary / "d0_summary.json").write_text(
            json.dumps(corrupt, sort_keys=True, indent=2) + "\n"
        )
        refresh_manifest_entry(corrupt_summary, "d0_summary.json")
        check(
            "independent-validator-recomputes-full-summary",
            not validate_emitted_output(corrupt_summary).passed,
        )

        corrupt_registry = root / "logs" / "goal" / "corrupt_registry"
        shutil.copytree(output, corrupt_registry)
        registry_rows = read_tsv(corrupt_registry / "opened_registry.snapshot.tsv")
        registry_rows[0]["map_name"] = "CorruptMap"
        with (corrupt_registry / "opened_registry.snapshot.tsv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=registry_rows[0].keys(),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(registry_rows)
        refresh_manifest_entry(corrupt_registry, "opened_registry.snapshot.tsv")
        check(
            "independent-validator-recomputes-registry",
            not validate_emitted_output(corrupt_registry).passed,
        )

        second = root / "logs" / "goal" / "full_b"
        check("rerun-success", run_scan("full", second, config, workers=1) == 0)
        check("rerun-byte-equality", output_hashes(output) == output_hashes(second))
        check("registry-idempotent", len(read_tsv(root / config["opened_registry"])) == 8)

        occupied = root / "logs" / "goal" / "occupied"
        occupied.mkdir()
        marker = occupied / "user.txt"
        marker.write_text("keep")
        check("nonempty-exit3", run_scan("full", occupied, config, workers=1) == 3 and marker.read_text() == "keep")

        failed = root / "logs" / "goal" / "failed"
        check("fault-exit2", run_scan("full", failed, config, workers=1, _fault_after="inventory") == 2)
        partial = Path(str(failed) + ".partial")
        check("fault-partial", partial.is_dir() and (partial / "FAILED").is_file())
        check("fault-no-complete", not failed.exists() and not (partial / "COMPLETE").exists())

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

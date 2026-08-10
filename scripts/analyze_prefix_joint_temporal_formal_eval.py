import argparse
import hashlib
import json
import math
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import episode_key


MAPS = ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")
UPDATES = (27, 28, 29, 30)
IDENTITY_FIELDS = ("scenario_id", "ego_idx", "opp_idx", "opp_raceline", "opp_speedscale", "interval_idx", "map_name")
REQUIRED_TRACE_FIELDS = {
    "time_s", "ego_lidar_360", "opp_lidar_360", "ego_raw_action", "ego_executed_action", "opp_executed_action",
    "ego_measured_speed_mps", "opp_measured_speed_mps", "ego_pose", "opp_pose", "collisions", "ego_opp_collision",
    "ego_wall_collision", "opp_wall_collision", "action_applied", "terminal_post_step",
}
BASELINE_ROOTS = {
    "bc": Path("eval_results/pretrained_end2race"),
    "u44": Path("eval_results/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44"),
    "z6": Path("eval_results/ppo_prefix_reset_consensus1of3"),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate and analyze the fixed prefix joint-temporal formal evaluation")
    parser.add_argument("--run-dir", type=Path, default=Path("post-trained/ppo_prefix_reset_joint_temporal_rho0p90_postfailure_exact_actor_exploratory"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("eval_results"))
    parser.add_argument("--panel-root", type=Path, default=Path("post-trained/panels/standard_multiagent_600_v1"))
    parser.add_argument("--alias-prefix", type=str, default="PJTE_u")
    parser.add_argument("--report", type=Path, default=Path("post-trained/ppo_prefix_reset_joint_temporal_rho0p90_postfailure_exact_actor_exploratory/formal_eval_report.json"))
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assert_finite(value, label):
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite value at {label}: {value}")


def validate_trace(task):
    label, trace_path, outcome, steps, simulation_time_s = task
    with np.load(trace_path, allow_pickle=False) as payload:
        missing = REQUIRED_TRACE_FIELDS - set(payload.files)
        if missing:
            raise RuntimeError(f"{label}/{trace_path.name}: missing fields {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    lengths = {key: value.shape[0] for key, value in arrays.items() if value.ndim >= 1}
    if len(lengths) != len(arrays) or len(set(lengths.values())) != 1:
        raise RuntimeError(f"{label}/{trace_path.name}: arrays are not aligned")
    length = next(iter(lengths.values()))
    if length != int(steps) + 1:
        raise RuntimeError(f"{label}/{trace_path.name}: trace length does not equal steps+1")
    for key, value in arrays.items():
        if value.dtype.kind not in "buif" or not bool(np.isfinite(value).all()):
            raise RuntimeError(f"{label}/{trace_path.name}: invalid numeric array {key}")
    shapes = {
        "time_s": (length,), "ego_lidar_360": (length, 360), "opp_lidar_360": (length, 360),
        "ego_raw_action": (length, 2), "ego_executed_action": (length, 2), "opp_executed_action": (length, 2),
        "ego_pose": (length, 3), "opp_pose": (length, 3), "collisions": (length, 2),
    }
    for key, expected in shapes.items():
        if arrays[key].shape != expected:
            raise RuntimeError(f"{label}/{trace_path.name}: {key} shape changed")
    if not bool(np.all(np.diff(arrays["time_s"]) > 0.0)) or not math.isclose(float(arrays["time_s"][-1]), float(simulation_time_s), rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"{label}/{trace_path.name}: time contract failed")
    terminal = arrays["terminal_post_step"].astype(bool)
    action_applied = arrays["action_applied"].astype(bool)
    if int(terminal.sum()) != 1 or not bool(terminal[-1]) or int((~action_applied).sum()) != 1 or bool(action_applied[-1]) or not bool(action_applied[:-1].all()):
        raise RuntimeError(f"{label}/{trace_path.name}: terminal/action contract failed")
    collisions = arrays["collisions"].astype(bool)
    ego_opp = arrays["ego_opp_collision"].astype(bool)
    ego_wall = arrays["ego_wall_collision"].astype(bool)
    opp_wall = arrays["opp_wall_collision"].astype(bool)
    if bool(np.any(ego_opp & ego_wall)) or not np.array_equal(collisions[:, 0], ego_opp | ego_wall) or not np.array_equal(collisions[:, 1], ego_opp | opp_wall):
        raise RuntimeError(f"{label}/{trace_path.name}: typed collision contract failed")
    observed = "ego-opp" if bool(ego_opp.any()) else "ego-wall" if bool(ego_wall.any()) else None
    expected = outcome if outcome in ("ego-opp", "ego-wall") else None
    if observed != expected:
        raise RuntimeError(f"{label}/{trace_path.name}: collision marker and result differ")
    return label


def panel_by_key(path):
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for scenario in scenarios:
        key = episode_key(scenario["opp_raceline"], scenario["ego_idx"], scenario["opp_idx"], scenario["opp_speedscale"])
        if key in rows:
            raise RuntimeError(f"duplicate panel key {key}")
        rows[key] = scenario
    if len(rows) != 600:
        raise RuntimeError(f"{path}: expected 600 unique scenarios")
    return rows


def load_result(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    assert_finite(result, str(path))
    return result


def outcome_counts(episodes):
    counts = Counter(record["outcome"] for record in episodes.values())
    return {
        "collision": counts["ego-opp"] + counts["ego-wall"],
        "overtake": counts["overtake"],
        "follow": counts["follow"],
        "ego_opp": counts["ego-opp"],
        "ego_wall": counts["ego-wall"],
    }


def validate_result_summary(result, label):
    counts = outcome_counts(result["episodes"])
    final = result["final"]
    expected = {
        "total_episodes": 600,
        "error_count": 0,
        "collision_count": counts["collision"],
        "overtaking_count": counts["overtake"],
        "following_count": counts["follow"],
        "ego_opp_collision_count": counts["ego_opp"],
        "ego_wall_collision_count": counts["ego_wall"],
    }
    for key, value in expected.items():
        if final.get(key) != value:
            raise RuntimeError(f"{label}: aggregate {key} does not reconcile")
    return counts


def exact_mcnemar(first, second):
    total = int(first) + int(second)
    if total == 0:
        return 1.0
    lower = min(int(first), int(second))
    probability = sum(math.comb(total, index) for index in range(lower + 1)) / (2.0 ** total)
    return min(1.0, 2.0 * probability)


def paired_changes(treatment, baseline):
    if set(treatment) != set(baseline):
        raise RuntimeError("paired episode keys differ")
    removed_collision = created_collision = lost_overtake = gained_overtake = 0
    for key in treatment:
        treatment_outcome = treatment[key]["outcome"]
        baseline_outcome = baseline[key]["outcome"]
        treatment_collision = treatment_outcome in ("ego-opp", "ego-wall")
        baseline_collision = baseline_outcome in ("ego-opp", "ego-wall")
        removed_collision += baseline_collision and not treatment_collision
        created_collision += treatment_collision and not baseline_collision
        lost_overtake += baseline_outcome == "overtake" and treatment_outcome != "overtake"
        gained_overtake += treatment_outcome == "overtake" and baseline_outcome != "overtake"
    return {
        "collision_removed": int(removed_collision),
        "collision_created": int(created_collision),
        "collision_exact_mcnemar_p": exact_mcnemar(removed_collision, created_collision),
        "overtake_lost": int(lost_overtake),
        "overtake_gained": int(gained_overtake),
        "overtake_exact_mcnemar_p": exact_mcnemar(lost_overtake, gained_overtake),
    }


def baseline_directory(name, update, map_name):
    if name == "z6":
        return BASELINE_ROOTS[name] / f"update{update}" / map_name / "multiagents"
    return BASELINE_ROOTS[name] / map_name / "multiagents"


def validate_training(run_dir):
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert_finite(rows, "training_metrics")
    formal = [row for row in rows if row.get("phase") == "formal"]
    if len(rows) != 31 or [row.get("update") for row in formal] != list(range(1, 31)):
        raise RuntimeError("training metrics are incomplete")
    maximum_exact = max(max(row["preupdate_exact_max_abs_log_ratio"], row["preupdate_exact_max_abs_ratio_minus_one"]) for row in formal)
    maximum_batched = max(max(row["preupdate_max_abs_log_ratio"], row["preupdate_max_abs_ratio_minus_one"]) for row in formal)
    if maximum_exact > 5.0e-5 or any(row["actor_replay_mode"] != "collection_equivalent" or row["actor_optimizer_steps_completed"] != 16 or row["joint_temporal_treatment_leak_count"] != 0 or row["joint_temporal_action_identity_checked_count"] != 102400 for row in formal):
        raise RuntimeError("formal training contract failed")
    return {
        "metric_rows": len(rows),
        "formal_updates": len(formal),
        "maximum_exact_ratio_error": maximum_exact,
        "maximum_batched_ratio_error": maximum_batched,
        "minimum_active_fraction": min(row["joint_temporal_active_fraction"] for row in formal),
        "maximum_active_fraction": max(row["joint_temporal_active_fraction"] for row in formal),
        "all_actor_replay_collection_equivalent": True,
        "all_actor_steps_16_of_16": True,
        "all_treatment_leak_counts_zero": True,
        "all_action_identity_counts_102400": True,
    }


if __name__ == "__main__":
    args = parse_arguments()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite {args.report}")
    training = validate_training(args.run_dir)
    panels = {map_name: panel_by_key(args.panel_root / f"{map_name}_600_scenarios.json") for map_name in MAPS}
    treatment_by_update = {}
    baseline_by_update = {}
    update_reports = {}
    trace_tasks = []
    checkpoint_hashes = {}
    for update in UPDATES:
        actor_path = (
            args.run_dir
            / "checkpoints"
            / f"actor_u{update:04d}.pth"
        )
        state = torch.load(actor_path, map_location="cpu", weights_only=True)
        if len(state) != 12:
            raise RuntimeError(f"U{update} actor is not strict 12-key")
        checkpoint_hashes[f"U{update}"] = sha256_file(actor_path)
        treatment_by_update[update] = {}
        baseline_by_update[update] = {name: {} for name in BASELINE_ROOTS}
        map_reports = {}
        aggregate = Counter()
        for map_name in MAPS:
            directory = args.evaluation_root / f"{args.alias_prefix}{update}" / map_name / "multiagents"
            result_path = directory / "results_multi.json"
            manifest_path = directory / "eval_manifest.json"
            trace_directory = directory / "traces"
            if (directory / "episodes.partial.jsonl").exists():
                raise RuntimeError(f"U{update}/{map_name}: partial evaluation remains")
            result = load_result(result_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            panel = panels[map_name]
            if set(result["episodes"]) != set(panel) or {path.stem for path in trace_directory.glob("*.npz")} != set(panel):
                raise RuntimeError(f"U{update}/{map_name}: panel/result/trace keys differ")
            manifest_expected = {
                "actor_path": str(actor_path), "actor_sha256": checkpoint_hashes[f"U{update}"], "panel_file": str(args.panel_root / f"{map_name}_600_scenarios.json"),
                "panel_sha256": sha256_file(args.panel_root / f"{map_name}_600_scenarios.json"), "panel_id": "standard_multiagent_600_v1", "map_name": map_name,
                "scenario_count": 600, "result_episode_count": 600, "trace_count": 600, "collision_scope": "ego", "device": "cuda", "sim_duration_s": 8.0,
                "hidden_scale": 4, "save_traces": True, "complete": True, "comparison_ready": True, "trace_result_key_sets_equal": True, "unique_episode_keys": True, "error_count": 0,
            }
            for key, expected in manifest_expected.items():
                if manifest.get(key) != expected:
                    raise RuntimeError(f"U{update}/{map_name}: manifest {key} changed")
            for key, record in result["episodes"].items():
                for field in IDENTITY_FIELDS:
                    if record.get(field) != panel[key].get(field):
                        raise RuntimeError(f"U{update}/{map_name}/{key}: identity {field} differs")
                trace_tasks.append((f"U{update}/{map_name}", trace_directory / f"{key}.npz", record["outcome"], record["steps"], record["simulation_time_s"]))
                treatment_by_update[update][f"{map_name}:{key}"] = record
            counts = validate_result_summary(result, f"U{update}/{map_name}")
            aggregate.update(counts)
            map_reports[map_name] = counts
            for baseline_name in BASELINE_ROOTS:
                baseline_result = load_result(baseline_directory(baseline_name, update, map_name) / "results_multi.json")
                if set(baseline_result["episodes"]) != set(panel):
                    raise RuntimeError(f"{baseline_name}/U{update}/{map_name}: episode keys differ")
                validate_result_summary(baseline_result, f"{baseline_name}/U{update}/{map_name}")
                for key, record in baseline_result["episodes"].items():
                    baseline_by_update[update][baseline_name][f"{map_name}:{key}"] = record
        per_map_bc_gate = {map_name: map_reports[map_name]["collision"] <= outcome_counts({key: value for key, value in baseline_by_update[update]["bc"].items() if key.startswith(f"{map_name}:")})["collision"] and map_reports[map_name]["overtake"] >= outcome_counts({key: value for key, value in baseline_by_update[update]["bc"].items() if key.startswith(f"{map_name}:")})["overtake"] for map_name in MAPS}
        comparisons = {name: paired_changes(treatment_by_update[update], baseline_by_update[update][name]) for name in BASELINE_ROOTS}
        update_reports[f"U{update}"] = {
            "maps": map_reports,
            "aggregate": dict(aggregate),
            "per_map_bc_gate": per_map_bc_gate,
            "all_per_map_bc_gates_pass": all(per_map_bc_gate.values()),
            "formal_600_task_achieved": aggregate["collision"] < 40 and aggregate["overtake"] > 1500 and all(per_map_bc_gate.values()),
            "paired_comparisons": comparisons,
        }
    context = mp.get_context("forkserver")
    with context.Pool(processes=max(1, args.workers)) as pool:
        validated = list(pool.imap_unordered(validate_trace, trace_tasks, chunksize=8))
    if Counter(validated) != Counter({f"U{update}/{map_name}": 600 for update in UPDATES for map_name in MAPS}):
        raise RuntimeError("trace validation counts differ")
    u29 = update_reports["U29"]
    u30 = update_reports["U30"]
    z6_u30 = outcome_counts(baseline_by_update[30]["z6"])
    z6_u29 = outcome_counts(baseline_by_update[29]["z6"])
    u30_z6_pair = u30["paired_comparisons"]["z6"]
    l2_mechanism = {
        "collision_at_most_91_and_p_below_0p05": u30["aggregate"]["collision"] <= 91 and u30_z6_pair["collision_exact_mcnemar_p"] < 0.05,
        "overtake_at_least_1507_and_pair_loss_minus_gain_at_most_15": u30["aggregate"]["overtake"] >= 1507 and u30_z6_pair["overtake_lost"] - u30_z6_pair["overtake_gained"] <= 15,
        "adjacent_collision_direction": u29["aggregate"]["collision"] < z6_u29["collision"] and u30["aggregate"]["collision"] < z6_u30["collision"],
    }
    report = {
        "schema_version": 1,
        "experiment_id": "ppo_prefix_reset_joint_temporal_rho0p90_postfailure_exact_actor_exploratory",
        "evidence_status": "post_failure_exact_actor_exploratory_not_original_confirmatory",
        "training": training,
        "quality": {
            "formal_package_count": 16, "episode_count": 9600, "trace_count": 9600, "all_panels_600_unique": True,
            "all_errors_zero": True, "all_results_finite": True, "all_result_trace_panel_keys_equal": True,
            "all_trace_arrays_aligned_and_finite": True, "all_collision_markers_match_results": True, "all_terminal_contracts_pass": True,
        },
        "checkpoint_sha256": checkpoint_hashes,
        "updates": update_reports,
        "u30_l2_mechanism_criteria": l2_mechanism,
        "u30_l2_mechanism_pass": all(l2_mechanism.values()),
        "u30_l2_product_pass": u30["all_per_map_bc_gates_pass"],
        "formal_600_task_achieved_updates": [name for name, value in update_reports.items() if value["formal_600_task_achieved"]],
        "formal_600_task_achieved": any(value["formal_600_task_achieved"] for value in update_reports.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps({"report": str(args.report), "formal_600_task_achieved": report["formal_600_task_achieved"], "updates": {key: value["aggregate"] for key, value in update_reports.items()}}, indent=2))

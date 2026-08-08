import argparse
import hashlib
import json
import math
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np


ACTORS = {
    "bc": Path("pretrained/end2race.pth"),
    "u42": Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update42/actor.pth"),
    "u43": Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update43/actor.pth"),
    "u44": Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/actor.pth"),
    "u45": Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update45/actor.pth"),
}
IDENTITY_FIELDS = (
    "scenario_id",
    "ego_idx",
    "opp_idx",
    "opp_raceline",
    "opp_speedscale",
    "interval_idx",
    "map_name",
)
REQUIRED_TRACE_FIELDS = {
    "time_s",
    "ego_lidar_360",
    "opp_lidar_360",
    "ego_raw_action",
    "ego_executed_action",
    "opp_executed_action",
    "ego_measured_speed_mps",
    "opp_measured_speed_mps",
    "ego_pose",
    "opp_pose",
    "collisions",
    "ego_opp_collision",
    "ego_wall_collision",
    "opp_wall_collision",
    "action_applied",
    "terminal_post_step",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def episode_key(scenario):
    raceline = int(scenario["opp_raceline"].removeprefix("raceline"))
    speed = format(float(scenario["opp_speedscale"]), "g")
    return f"ol{raceline}_e{scenario['ego_idx']}_o{scenario['opp_idx']}_s{speed}"


def assert_finite(value, label):
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite result value at {label}: {value}")


def validate_trace(task):
    actor, trace_path, outcome, steps, simulation_time_s = task
    with np.load(trace_path, allow_pickle=False) as payload:
        missing = REQUIRED_TRACE_FIELDS - set(payload.files)
        if missing:
            raise RuntimeError(f"{actor}/{trace_path.name}: missing fields {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]) for key in payload.files}

    lengths = {key: value.shape[0] for key, value in arrays.items() if value.ndim >= 1}
    if len(lengths) != len(arrays) or len(set(lengths.values())) != 1:
        raise RuntimeError(f"{actor}/{trace_path.name}: unaligned arrays {lengths}")
    length = next(iter(lengths.values()))
    if length != int(steps) + 1:
        raise RuntimeError(f"{actor}/{trace_path.name}: trace length {length} != steps+1 {int(steps) + 1}")

    for key, value in arrays.items():
        if value.dtype.kind not in "buif":
            raise RuntimeError(f"{actor}/{trace_path.name}: unsupported dtype {key}={value.dtype}")
        if not bool(np.isfinite(value).all()):
            raise RuntimeError(f"{actor}/{trace_path.name}: non-finite values in {key}")

    shape_contract = {
        "time_s": (length,),
        "ego_lidar_360": (length, 360),
        "opp_lidar_360": (length, 360),
        "ego_raw_action": (length, 2),
        "ego_executed_action": (length, 2),
        "opp_executed_action": (length, 2),
        "ego_pose": (length, 3),
        "opp_pose": (length, 3),
        "collisions": (length, 2),
    }
    for key, expected_shape in shape_contract.items():
        if arrays[key].shape != expected_shape:
            raise RuntimeError(f"{actor}/{trace_path.name}: {key} shape {arrays[key].shape} != {expected_shape}")

    time_s = arrays["time_s"]
    if not bool(np.all(np.diff(time_s) > 0.0)):
        raise RuntimeError(f"{actor}/{trace_path.name}: time_s is not strictly increasing")
    if not math.isclose(float(time_s[-1]), float(simulation_time_s), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"{actor}/{trace_path.name}: final time does not match result")

    terminal = arrays["terminal_post_step"].astype(bool)
    action_applied = arrays["action_applied"].astype(bool)
    if int(terminal.sum()) != 1 or not bool(terminal[-1]):
        raise RuntimeError(f"{actor}/{trace_path.name}: terminal_post_step contract failed")
    if int((~action_applied).sum()) != 1 or bool(action_applied[-1]) or not bool(action_applied[:-1].all()):
        raise RuntimeError(f"{actor}/{trace_path.name}: action_applied contract failed")

    collisions = arrays["collisions"].astype(bool)
    ego_opp = arrays["ego_opp_collision"].astype(bool)
    ego_wall = arrays["ego_wall_collision"].astype(bool)
    opp_wall = arrays["opp_wall_collision"].astype(bool)
    if bool(np.any(ego_opp & ego_wall)):
        raise RuntimeError(f"{actor}/{trace_path.name}: ego collision markers overlap")
    if not np.array_equal(collisions[:, 0], ego_opp | ego_wall):
        raise RuntimeError(f"{actor}/{trace_path.name}: ego collision marker mismatch")
    if not np.array_equal(collisions[:, 1], ego_opp | opp_wall):
        raise RuntimeError(f"{actor}/{trace_path.name}: opponent collision marker mismatch")
    observed = "ego-opp" if bool(ego_opp.any()) else "ego-wall" if bool(ego_wall.any()) else None
    expected = outcome if outcome in ("ego-opp", "ego-wall") else None
    if observed != expected:
        raise RuntimeError(f"{actor}/{trace_path.name}: trace outcome {observed} != result {expected}")
    return actor, trace_path.stem, length


def atomic_write(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cohort-panel", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--split-name", choices=("development", "validation"), default="development")
    parser.add_argument("--collision-only-validation", action="store_true")
    args = parser.parse_args()

    if args.collision_only_validation and args.split_name != "validation":
        raise ValueError("--collision-only-validation requires --split-name validation")

    for output_path in (args.report, args.cohort_panel):
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}")

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    panel_by_key = {episode_key(scenario): scenario for scenario in panel}
    if len(panel_by_key) != len(panel):
        raise RuntimeError(f"{args.split_name} panel episode keys are not unique")
    panel_keys = set(panel_by_key)
    if set(split_manifest["scenario_keys_by_split"][args.split_name]) != panel_keys:
        raise RuntimeError(f"{args.split_name} panel does not match frozen split manifest")
    development_starts = set(split_manifest["development_ego_idx"])
    validation_starts = set(split_manifest["validation_ego_idx"])
    if development_starts & validation_starts:
        raise RuntimeError("development and validation startpoints overlap")
    panel_starts = development_starts if args.split_name == "development" else validation_starts
    if {scenario["ego_idx"] for scenario in panel} != panel_starts:
        raise RuntimeError(f"{args.split_name} panel startpoints do not match split manifest")

    actor_reports = {}
    episodes = {}
    trace_tasks = []
    panel_sha256 = sha256_file(args.panel)
    for actor, actor_path in ACTORS.items():
        actor_directory = args.evaluation_root / actor
        result_path = actor_directory / "results_multi.json"
        manifest_path = actor_directory / "eval_manifest.json"
        trace_directory = actor_directory / "traces"
        if (actor_directory / "episodes.partial.jsonl").exists():
            raise RuntimeError(f"{actor}: partial result still exists")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actor_episodes = result["episodes"]
        result_keys = set(actor_episodes)
        trace_keys = {path.stem for path in trace_directory.glob("*.npz")}
        if result_keys != panel_keys or trace_keys != panel_keys:
            raise RuntimeError(f"{actor}: panel/result/trace key sets differ")
        expected_actor_sha256 = sha256_file(actor_path)
        manifest_contract = {
            "actor_path": str(actor_path),
            "actor_sha256": expected_actor_sha256,
            "panel_file": str(args.panel),
            "panel_sha256": panel_sha256,
            "panel_id": "bc_collision_only_anchor_validation_actor_screen" if args.collision_only_validation else "bc_anchor_gate_a_development",
            "map_name": "Austin",
            "scenario_count": len(panel),
            "result_episode_count": len(panel),
            "trace_count": len(panel),
            "collision_scope": "ego",
            "device": "cuda",
            "sim_duration_s": 8.0,
            "hidden_scale": 4,
            "save_traces": True,
            "complete": True,
            "comparison_ready": True,
            "trace_result_key_sets_equal": True,
            "unique_episode_keys": True,
            "error_count": 0,
        }
        for key, expected_value in manifest_contract.items():
            if manifest.get(key) != expected_value:
                raise RuntimeError(f"{actor}: manifest {key}={manifest.get(key)!r} != {expected_value!r}")
        for key, record in actor_episodes.items():
            assert_finite(record, f"{actor}.{key}")
            scenario = panel_by_key[key]
            for field in IDENTITY_FIELDS:
                if record.get(field) != scenario.get(field):
                    raise RuntimeError(f"{actor}/{key}: identity mismatch for {field}")
            trace_tasks.append((actor, trace_directory / f"{key}.npz", record["outcome"], record["steps"], record["simulation_time_s"]))
        outcome_counts = Counter(record["outcome"] for record in actor_episodes.values())
        final = result["final"]
        if final["total_episodes"] != len(panel) or final["error_count"] != 0:
            raise RuntimeError(f"{actor}: invalid final counts")
        if final["overtaking_count"] != outcome_counts["overtake"] or final["following_count"] != outcome_counts["follow"]:
            raise RuntimeError(f"{actor}: success summary does not reconcile")
        if final["ego_opp_collision_count"] != outcome_counts["ego-opp"] or final["ego_wall_collision_count"] != outcome_counts["ego-wall"]:
            raise RuntimeError(f"{actor}: collision summary does not reconcile")
        actor_reports[actor] = {
            "actor_path": str(actor_path),
            "actor_sha256": expected_actor_sha256,
            "manifest_sha256": sha256_file(manifest_path),
            "result_sha256": sha256_file(result_path),
            "final": final,
        }
        episodes[actor] = actor_episodes

    context = mp.get_context("forkserver")
    with context.Pool(processes=max(1, args.workers)) as pool:
        validated = list(pool.imap_unordered(validate_trace, trace_tasks, chunksize=8))
    validated_by_actor = Counter(actor for actor, _, _ in validated)
    if validated_by_actor != Counter({actor: len(panel) for actor in ACTORS}):
        raise RuntimeError(f"trace validation count mismatch: {validated_by_actor}")

    bc_safe = sorted(
        key
        for key, record in episodes["bc"].items()
        if record["outcome"] == "overtake" and record["opp_raceline"] in ("raceline0", "raceline2")
    )
    regression_count = {
        key: sum(episodes[actor][key]["outcome"] != "overtake" for actor in ("u42", "u43", "u44", "u45"))
        for key in bc_safe
    }
    u44_single = sorted(key for key in bc_safe if episodes["u44"][key]["outcome"] != "overtake")
    consensus = sorted(key for key in u44_single if regression_count[key] >= 3)
    collision_keys = sorted(key for key in consensus if episodes["u44"][key]["outcome"] in ("ego-opp", "ego-wall"))
    lost_overtake_keys = sorted(key for key in consensus if episodes["u44"][key]["outcome"] == "follow")
    selected_cohort = collision_keys if args.collision_only_validation else consensus
    selected_starts = {episodes["bc"][key]["ego_idx"] for key in selected_cohort}
    selected_raceline_counts = Counter(episodes["bc"][key]["opp_raceline"] for key in selected_cohort)
    selected_speed_counts = Counter(format(float(episodes["bc"][key]["opp_speedscale"]), "g") for key in selected_cohort)
    speed_levels = sorted({format(float(scenario["opp_speedscale"]), "g") for scenario in panel}, key=float)
    reported_speed_counts = {speed: selected_speed_counts[speed] for speed in speed_levels}
    if args.collision_only_validation:
        criteria = {
            "stable_collision_cohort_at_least_4": len(collision_keys) >= 4,
            "stable_collision_unique_ego_startpoints_at_least_3": len(selected_starts) >= 3,
            "stable_collision_has_raceline0_and_raceline2": selected_raceline_counts["raceline0"] >= 1 and selected_raceline_counts["raceline2"] >= 1,
            "all_five_evaluations_complete_finite_and_identity_aligned": len(validated) == len(panel) * len(ACTORS),
        }
    else:
        criteria = {
            "consensus_cohort_at_least_20": len(consensus) >= 20,
            "unique_ego_startpoints_at_least_10": len(selected_starts) >= 10,
            "raceline0_and_raceline2_each_at_least_3": selected_raceline_counts["raceline0"] >= 3 and selected_raceline_counts["raceline2"] >= 3,
            "u44_ego_collision_cases_at_least_8": len(collision_keys) >= 8,
            "all_five_evaluations_complete_finite_and_identity_aligned": len(validated) == len(panel) * len(ACTORS),
            "stability_control_reported_and_consensus_sufficient": not (len(u44_single) >= 20 and len(consensus) < 20),
        }
    gate_a_pass = all(criteria.values())

    cohort_panel = [scenario for scenario in panel if episode_key(scenario) in set(selected_cohort)]
    cohort_content = (json.dumps(cohort_panel, indent=2, sort_keys=True) + "\n").encode("utf-8")
    cohort_sha256 = hashlib.sha256(cohort_content).hexdigest()
    report = {
        "schema_version": 1,
        "experiment_id": "bc_collision_only_anchor_validation" if args.collision_only_validation else "front_corridor_temporal_bc_safe_anchor",
        "gate": "V0" if args.collision_only_validation else "A",
        "verdict": "pass" if gate_a_pass else "fail",
        "panel": {
            "path": str(args.panel),
            "sha256": panel_sha256,
            "scenario_count": len(panel),
            "unique_episode_key_count": len(panel_keys),
            "split_name": args.split_name,
            "unique_ego_startpoint_count": len(panel_starts),
            "map_name": "Austin",
            "split_manifest_path": str(args.split_manifest),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "development_validation_startpoint_overlap": 0,
            "validation_actor_evaluation_present": args.split_name == "validation" or (args.evaluation_root.parent / "validation").exists(),
        },
        "evaluation_contract": {
            "fresh_deterministic_cuda": True,
            "collision_scope": "ego",
            "sim_duration_s": 8.0,
            "all_speed_scales_retained": True,
            "actors": actor_reports,
        },
        "quality_validation": {
            "actor_episode_count": len(validated),
            "trace_count_by_actor": dict(sorted(validated_by_actor.items())),
            "panel_result_trace_key_sets_equal": True,
            "episode_identity_equal_across_actors": True,
            "all_trace_arrays_aligned_and_finite": True,
            "collision_markers_match_outcomes": True,
            "terminal_contract_complete": True,
        },
        "cohort_definition": {
            "bc_safe_overtake_count": len(bc_safe),
            "u44_single_checkpoint_regression_count": len(u44_single),
            "consensus_regression_count": len(selected_cohort),
            "consensus_unique_ego_startpoint_count": len(selected_starts),
            "consensus_by_raceline": dict(sorted(selected_raceline_counts.items())),
            "consensus_by_speed_scale": reported_speed_counts,
            "consensus_by_u44_stratum": {"collision": len(collision_keys), "lost_overtake": 0 if args.collision_only_validation else len(lost_overtake_keys)},
            "bc_safe_regression_count_distribution": dict(sorted(Counter(regression_count.values()).items())),
            "u44_single_scenario_keys": u44_single,
            "consensus_scenario_keys": selected_cohort,
            "collision_scenario_keys": collision_keys,
            "lost_overtake_scenario_keys": [] if args.collision_only_validation else lost_overtake_keys,
            "cohort_panel_path": str(args.cohort_panel),
            "cohort_panel_sha256": cohort_sha256,
        },
        "admission_criteria": criteria,
        "next_action": "Run collision-only branch0 and full-BC validation" if gate_a_pass and args.collision_only_validation else "Gate B may be implemented on the frozen development cohort; validation remains unopened" if gate_a_pass else "Record inconclusive validation sample and stop before intervention branches" if args.collision_only_validation else "Stop this direction before Gate B",
    }
    report_content = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.cohort_panel.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.cohort_panel, cohort_content)
    atomic_write(args.report, report_content)
    print(json.dumps({
        "verdict": report["verdict"],
        "bc_safe_overtake_count": len(bc_safe),
        "u44_single_checkpoint_regression_count": len(u44_single),
        "consensus_regression_count": len(selected_cohort),
        "consensus_unique_ego_startpoint_count": len(selected_starts),
        "consensus_by_raceline": dict(sorted(selected_raceline_counts.items())),
        "consensus_by_u44_stratum": report["cohort_definition"]["consensus_by_u44_stratum"],
        "validated_actor_episodes": len(validated),
        "criteria": criteria,
    }, indent=2, sort_keys=True))

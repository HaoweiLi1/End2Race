import argparse
from collections import Counter
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_bc_anchor_gate_a import assert_finite, validate_trace
from scripts.run_bc_anchor_gate_b import circular_distance, intervention_window
from utils import atomic_write_json, episode_key, load_raceline_waypoints

ACTORS = {
    "bc": Path("pretrained/end2race.pth"),
    "u42": Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update42/actor.pth"),
    "u43": Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update43/actor.pth"),
    "u44": Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/actor.pth"),
    "u45": Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update45/actor.pth"),
}
IDENTITY_FIELDS = ("scenario_id", "ego_idx", "opp_idx", "opp_raceline", "opp_speedscale", "interval_idx", "map_name")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("candidates", "final"), required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--full-evaluation-root", type=Path, required=True)
    parser.add_argument("--candidate-panel", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-evaluation-root", type=Path, required=True)
    parser.add_argument("--cohort-panel", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, payload):
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return hashlib.sha256(content).hexdigest()


def scenario_key(scenario):
    return episode_key(scenario["opp_raceline"], scenario["ego_idx"], scenario["opp_idx"], scenario["opp_speedscale"])


def load_panel(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a bare ScenarioSpec list: {path}")
    scenarios = {scenario_key(scenario): scenario for scenario in payload}
    if len(scenarios) != len(payload):
        raise RuntimeError(f"panel episode keys are not unique: {path}")
    return payload, scenarios


def validate_actor_evaluation(actor, actor_path, evaluation_root, panel_path, panel_by_key, panel_id, workers, validate_all_traces):
    result_path = evaluation_root / actor / "results_multi.json"
    manifest_path = evaluation_root / actor / "eval_manifest.json"
    trace_root = evaluation_root / actor / "traces"
    if (evaluation_root / actor / "episodes.partial.jsonl").exists():
        raise RuntimeError(f"{actor}: partial evaluation is still present")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = result["episodes"]
    panel_keys = set(panel_by_key)
    if set(episodes) != panel_keys:
        raise RuntimeError(f"{actor}: result keys do not match panel")
    trace_keys = {path.stem for path in trace_root.glob("*.npz")}
    if trace_keys != panel_keys:
        raise RuntimeError(f"{actor}: trace keys do not match panel")
    expected_manifest = {
        "actor_path": str(actor_path),
        "actor_sha256": sha256_file(actor_path),
        "panel_file": str(panel_path),
        "panel_sha256": sha256_file(panel_path),
        "panel_id": panel_id,
        "map_name": "Austin",
        "scenario_count": len(panel_by_key),
        "result_episode_count": len(panel_by_key),
        "trace_count": len(panel_by_key),
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
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"{actor}: manifest {key}={manifest.get(key)!r} != {expected!r}")
    trace_tasks = []
    for key, record in episodes.items():
        assert_finite(record, f"{actor}.{key}")
        scenario = panel_by_key[key]
        for field in IDENTITY_FIELDS:
            if record.get(field) != scenario.get(field):
                raise RuntimeError(f"{actor}/{key}: result identity mismatch for {field}")
        if validate_all_traces:
            trace_tasks.append((actor, trace_root / f"{key}.npz", record["outcome"], record["steps"], record["simulation_time_s"]))
    if validate_all_traces:
        context = mp.get_context("forkserver")
        with context.Pool(processes=max(1, workers)) as pool:
            validated = list(pool.imap_unordered(validate_trace, trace_tasks, chunksize=8))
        if len(validated) != len(panel_by_key):
            raise RuntimeError(f"{actor}: trace validation count mismatch")
    counts = Counter(record["outcome"] for record in episodes.values())
    final = result["final"]
    if final["total_episodes"] != len(panel_by_key) or final["error_count"] != 0:
        raise RuntimeError(f"{actor}: invalid aggregate count")
    if final["overtaking_count"] != counts["overtake"] or final["following_count"] != counts["follow"]:
        raise RuntimeError(f"{actor}: success aggregate does not reconcile")
    if final["ego_opp_collision_count"] != counts["ego-opp"] or final["ego_wall_collision_count"] != counts["ego-wall"]:
        raise RuntimeError(f"{actor}: collision aggregate does not reconcile")
    return episodes, {
        "actor_path": str(actor_path),
        "actor_sha256": sha256_file(actor_path),
        "manifest_sha256": sha256_file(manifest_path),
        "result_sha256": sha256_file(result_path),
        "trace_count": len(panel_by_key),
        "trace_payload_validated": bool(validate_all_traces),
        "outcomes": dict(sorted(counts.items())),
    }


def hash_rank(key):
    return hashlib.sha256(f"collision-only-anchor-overlap-v2|source|{key}".encode("utf-8")).hexdigest()


def window_task(task):
    key, trace_path, stratum = task
    return key, intervention_window(trace_path, stratum)


def select_controls(source_tasks, safe_keys, scenarios, control_windows):
    waypoint_count = len(load_raceline_waypoints("Austin", "raceline1.csv")) - 1
    available = set(safe_keys)
    selected = []
    for source in source_tasks:
        scenario = source["scenario"]
        candidates = [key for key in available if scenarios[key]["opp_raceline"] == scenario["opp_raceline"] and float(scenarios[key]["opp_speedscale"]) == float(scenario["opp_speedscale"])]
        candidates.sort(key=lambda key: (circular_distance(scenario["ego_idx"], scenarios[key]["ego_idx"], waypoint_count), hashlib.sha256(key.encode("utf-8")).hexdigest()))
        if not candidates:
            raise RuntimeError(f"no exact unused control for {source['episode_key']}")
        control_key = candidates[0]
        available.remove(control_key)
        selected.append({
            "episode_key": control_key,
            "role": "control",
            "stratum": "control",
            "matched_source_key": source["episode_key"],
            "circular_ego_index_distance": circular_distance(scenario["ego_idx"], scenarios[control_key]["ego_idx"], waypoint_count),
            "scenario": scenarios[control_key],
            "window": control_windows[control_key],
        })
    return selected


if __name__ == "__main__":
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    panel, panel_by_key = load_panel(args.panel)
    panel_manifest = json.loads(args.panel_manifest.read_text(encoding="utf-8"))
    if panel_manifest.get("status") != "frozen_before_any_actor_outcome" or panel_manifest.get("panel_sha256") != sha256_file(args.panel):
        raise RuntimeError("panel manifest does not identify the frozen input")
    if len(panel) != 2880 or len({scenario["ego_idx"] for scenario in panel}) != 40:
        raise RuntimeError("full panel is not the preregistered 40 x 2 x 4 x 9 Cartesian product")

    if args.stage == "candidates":
        if args.candidate_panel.exists() or args.candidate_manifest.exists():
            raise FileExistsError("refusing to overwrite the frozen candidate screen")
        bc_episodes, bc_report = validate_actor_evaluation("bc", ACTORS["bc"], args.full_evaluation_root, args.panel, panel_by_key, "collision_only_anchor_overlap_v2_full", args.workers, True)
        u44_episodes, u44_report = validate_actor_evaluation("u44", ACTORS["u44"], args.full_evaluation_root, args.panel, panel_by_key, "collision_only_anchor_overlap_v2_full", args.workers, True)
        candidate_keys = sorted(key for key in panel_by_key if bc_episodes[key]["outcome"] == "overtake" and u44_episodes[key]["outcome"] in ("ego-opp", "ego-wall"))
        candidate_panel = [panel_by_key[key] for key in candidate_keys]
        args.candidate_panel.parent.mkdir(parents=True, exist_ok=True)
        candidate_sha256 = atomic_write(args.candidate_panel, candidate_panel)
        candidate_manifest = {
            "schema_version": 1,
            "experiment_id": "bc_collision_only_anchor_overlap_v2",
            "status": "candidate_set_frozen_before_u42_u43_u45_outcomes",
            "selection_rule": "BC outcome is overtake and U44 outcome is ego-opp or ego-wall",
            "full_panel_path": str(args.panel),
            "full_panel_sha256": sha256_file(args.panel),
            "candidate_panel_path": str(args.candidate_panel),
            "candidate_panel_sha256": candidate_sha256,
            "candidate_count": len(candidate_panel),
            "candidate_scenario_keys": candidate_keys,
            "full_evaluation_quality": {"bc": bc_report, "u44": u44_report},
        }
        atomic_write(args.candidate_manifest, candidate_manifest)
        print(json.dumps({"candidate_count": len(candidate_panel), "candidate_panel_sha256": candidate_sha256, "bc_outcomes": bc_report["outcomes"], "u44_outcomes": u44_report["outcomes"]}, indent=2, sort_keys=True))
        sys.exit(0)

    if not args.candidate_panel.exists() or not args.candidate_manifest.exists():
        raise FileNotFoundError("candidate stage must finish before final analysis")
    if args.cohort_panel.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite the completed overlap Gate")
    candidate_panel, candidate_by_key = load_panel(args.candidate_panel)
    candidate_manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    if candidate_manifest.get("status") != "candidate_set_frozen_before_u42_u43_u45_outcomes" or candidate_manifest.get("candidate_panel_sha256") != sha256_file(args.candidate_panel):
        raise RuntimeError("candidate manifest does not identify the frozen candidate panel")

    bc_episodes, bc_report = validate_actor_evaluation("bc", ACTORS["bc"], args.full_evaluation_root, args.panel, panel_by_key, "collision_only_anchor_overlap_v2_full", args.workers, False)
    u44_episodes, u44_report = validate_actor_evaluation("u44", ACTORS["u44"], args.full_evaluation_root, args.panel, panel_by_key, "collision_only_anchor_overlap_v2_full", args.workers, False)
    for actor in ("bc", "u44"):
        frozen = candidate_manifest["full_evaluation_quality"][actor]
        current = bc_report if actor == "bc" else u44_report
        for field in ("actor_sha256", "manifest_sha256", "result_sha256", "trace_count"):
            if frozen[field] != current[field]:
                raise RuntimeError(f"{actor}: full evaluation changed after candidate freeze")
    expected_candidate_keys = sorted(key for key in panel_by_key if bc_episodes[key]["outcome"] == "overtake" and u44_episodes[key]["outcome"] in ("ego-opp", "ego-wall"))
    if set(candidate_by_key) != set(expected_candidate_keys):
        raise RuntimeError("candidate panel no longer equals the frozen selection rule")

    actor_episodes = {"bc": bc_episodes, "u44": u44_episodes}
    actor_reports = {"bc": bc_report, "u44": u44_report}
    for actor in ("u42", "u43", "u45"):
        actor_episodes[actor], actor_reports[actor] = validate_actor_evaluation(actor, ACTORS[actor], args.candidate_evaluation_root, args.candidate_panel, candidate_by_key, "collision_only_anchor_overlap_v2_candidate", args.workers, True)

    stable_keys = sorted(key for key in candidate_by_key if sum(actor_episodes[actor][key]["outcome"] != "overtake" for actor in ("u42", "u43", "u44", "u45")) >= 3)
    u44_trace_root = args.full_evaluation_root / "u44" / "traces"
    safe_keys = sorted(key for key in panel_by_key if bc_episodes[key]["outcome"] == "overtake" and u44_episodes[key]["outcome"] == "overtake")
    window_tasks = [(key, u44_trace_root / f"{key}.npz", "collision") for key in stable_keys]
    window_tasks.extend((key, u44_trace_root / f"{key}.npz", "control") for key in safe_keys)
    context = mp.get_context("forkserver")
    with context.Pool(processes=max(1, args.workers)) as pool:
        windows = dict(pool.imap_unordered(window_task, window_tasks, chunksize=4))
    eligible_stable = [key for key in stable_keys if windows[key]["eligible"]]
    eligible_safe = [key for key in safe_keys if windows[key]["eligible"]]

    stable_by_stratum = {}
    safe_by_stratum = {}
    for key in eligible_stable:
        stratum = (panel_by_key[key]["opp_raceline"], format(float(panel_by_key[key]["opp_speedscale"]), "g"))
        stable_by_stratum.setdefault(stratum, []).append(key)
    for key in eligible_safe:
        stratum = (panel_by_key[key]["opp_raceline"], format(float(panel_by_key[key]["opp_speedscale"]), "g"))
        safe_by_stratum.setdefault(stratum, []).append(key)
    selected_source_keys = []
    support_table = {}
    for stratum in sorted(set(stable_by_stratum) | set(safe_by_stratum)):
        sources = sorted(stable_by_stratum.get(stratum, []), key=hash_rank)
        controls = sorted(safe_by_stratum.get(stratum, []))
        supported_count = min(len(sources), len(controls))
        selected_source_keys.extend(sources[:supported_count])
        support_table[f"{stratum[0]}/s{stratum[1]}"] = {"eligible_stable_source_count": len(sources), "eligible_safe_control_count": len(controls), "selected_source_count": supported_count, "unsupported_source_count": len(sources) - supported_count}
    selected_source_keys.sort(key=hash_rank)
    source_tasks = [{"episode_key": key, "role": "cohort", "stratum": "collision", "matched_source_key": None, "circular_ego_index_distance": None, "scenario": panel_by_key[key], "window": windows[key]} for key in selected_source_keys]
    control_tasks = select_controls(source_tasks, eligible_safe, panel_by_key, windows)
    if len(control_tasks) != len(source_tasks):
        raise RuntimeError("exact matched control construction did not close")
    if any(task["scenario"]["opp_raceline"] != panel_by_key[task["matched_source_key"]]["opp_raceline"] or float(task["scenario"]["opp_speedscale"]) != float(panel_by_key[task["matched_source_key"]]["opp_speedscale"]) for task in control_tasks):
        raise RuntimeError("matched control stratum mismatch")

    source_starts = {panel_by_key[key]["ego_idx"] for key in selected_source_keys}
    source_racelines = Counter(panel_by_key[key]["opp_raceline"] for key in selected_source_keys)
    criteria = {
        "overlap_supported_collision_sources_at_least_12": len(selected_source_keys) >= 12,
        "overlap_supported_unique_ego_startpoints_at_least_8": len(source_starts) >= 8,
        "overlap_supported_sources_each_raceline_at_least_2": source_racelines["raceline0"] >= 2 and source_racelines["raceline2"] >= 2,
        "exact_same_raceline_speed_controls_without_replacement": len(control_tasks) == len(source_tasks),
        "all_full_and_candidate_evaluations_complete_finite_identity_aligned": True,
    }
    verdict = "pass" if all(criteria.values()) else "inconclusive"
    cohort = [panel_by_key[key] for key in selected_source_keys]
    args.cohort_panel.parent.mkdir(parents=True, exist_ok=True)
    cohort_sha256 = atomic_write(args.cohort_panel, cohort)
    source_to_control = {task["matched_source_key"]: task["episode_key"] for task in control_tasks}
    report = {
        "schema_version": 1,
        "experiment_id": "bc_collision_only_anchor_overlap_v2",
        "gate": "V0",
        "verdict": verdict,
        "panel": {"path": str(args.panel), "sha256": sha256_file(args.panel), "scenario_count": len(panel), "unique_ego_startpoint_count": len({scenario["ego_idx"] for scenario in panel}), "exact_startpoint_overlap_with_prior_candidate_and_Austin600": panel_manifest["exact_ego_startpoint_overlap_with_excluded_panels"]},
        "evaluation_contract": {"fresh_deterministic_cuda": True, "collision_scope": "ego", "sim_duration_s": 8.0, "actors": actor_reports, "staged_screen_equivalence": "U42/U43/U45 are required only for BC-overtake/U44-collision candidates"},
        "quality_validation": {"full_bc_u44_episode_count": 2 * len(panel), "candidate_u42_u43_u45_episode_count": 3 * len(candidate_panel), "trace_count": 2 * len(panel) + 3 * len(candidate_panel), "panel_result_trace_key_sets_equal": True, "episode_identity_equal_across_required_actors": True, "all_trace_arrays_aligned_and_finite": True, "collision_markers_match_outcomes": True, "terminal_contract_complete": True},
        "cohort_definition": {
            "potential_bc_safe_u44_collision_count": len(candidate_panel),
            "stable_collision_count_before_window_filter": len(stable_keys),
            "eligible_stable_collision_count": len(eligible_stable),
            "eligible_safe_control_pool_count": len(eligible_safe),
            "consensus_regression_count": len(selected_source_keys),
            "consensus_unique_ego_startpoint_count": len(source_starts),
            "consensus_by_raceline": dict(sorted(source_racelines.items())),
            "consensus_by_speed_scale": dict(sorted(Counter(format(float(panel_by_key[key]["opp_speedscale"]), "g") for key in selected_source_keys).items(), key=lambda item: float(item[0]))),
            "consensus_by_u44_stratum": {"collision": len(selected_source_keys), "lost_overtake": 0},
            "consensus_scenario_keys": selected_source_keys,
            "collision_scenario_keys": selected_source_keys,
            "lost_overtake_scenario_keys": [],
            "cohort_panel_path": str(args.cohort_panel),
            "cohort_panel_sha256": cohort_sha256,
        },
        "control_support": {
            "estimand": "hash-selected overlap-supported stable collision sources with exact same-raceline/speed controls",
            "support_by_stratum": support_table,
            "source_count": len(source_tasks),
            "control_count": len(control_tasks),
            "expected_control_scenario_keys": [task["episode_key"] for task in control_tasks],
            "expected_source_to_control": source_to_control,
            "without_replacement": True,
        },
        "admission_criteria": criteria,
        "next_action": "Run branch0 and full-BC on the frozen overlap-supported plan" if verdict == "pass" else "Stop without branches; record independent-panel sample insufficiency as inconclusive",
    }
    atomic_write_json(args.report, report)
    print(json.dumps({"verdict": verdict, "potential_candidate_count": len(candidate_panel), "stable_count": len(stable_keys), "eligible_stable_count": len(eligible_stable), "selected_source_count": len(selected_source_keys), "selected_unique_startpoints": len(source_starts), "selected_by_raceline": dict(sorted(source_racelines.items())), "control_count": len(control_tasks), "criteria": criteria}, indent=2, sort_keys=True))

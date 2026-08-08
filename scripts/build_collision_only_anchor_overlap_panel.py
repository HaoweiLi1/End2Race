import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import get_circular_startpoints, get_opponent_startpoint, load_raceline_waypoints


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startpoint-count", type=int, default=40)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_scenarios(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if "scenarios" in payload:
        return payload["scenarios"]
    if "entries" in payload:
        return [entry.get("scenario", entry) for entry in payload["entries"]]
    raise ValueError(f"unsupported panel schema: {path}")


def atomic_write(path, payload):
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return hashlib.sha256(content).hexdigest()


if __name__ == "__main__":
    args = parse_arguments()
    if args.startpoint_count != 40:
        raise ValueError("the preregistered panel requires exactly 40 startpoints")

    panel_path = args.output_dir / "full_scenarios.json"
    manifest_path = args.output_dir / "panel_manifest.json"
    if panel_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite the frozen panel in {args.output_dir}")

    exclusion_paths = (
        Path("post-trained/panels/heldout_hard_v1/candidate_scenarios.json"),
        Path("post-trained/panels/standard_multiagent_600_v1/Austin_600_scenarios.json"),
    )
    excluded_startpoints = set()
    exclusion_inputs = []
    for path in exclusion_paths:
        scenarios = load_scenarios(path)
        excluded_startpoints.update(int(scenario["ego_idx"]) for scenario in scenarios)
        exclusion_inputs.append({"path": str(path), "sha256": sha256_file(path), "scenario_count": len(scenarios), "unique_ego_startpoint_count": len({int(scenario["ego_idx"]) for scenario in scenarios})})

    waypoint_count = len(load_raceline_waypoints("Austin", "raceline1.csv")) - 1
    offsets = sorted(range(waypoint_count), key=lambda offset: hashlib.sha256(f"collision-only-anchor-overlap-v2|Austin|{offset}".encode("utf-8")).hexdigest())
    selected_offset = None
    selected_startpoints = None
    for offset in offsets:
        startpoints = [int(value) for value in get_circular_startpoints("Austin", "raceline1.csv", args.startpoint_count, offset)]
        if len(set(startpoints)) == args.startpoint_count and not (set(startpoints) & excluded_startpoints):
            selected_offset = offset
            selected_startpoints = startpoints
            break
    if selected_startpoints is None:
        raise RuntimeError("no exact-disjoint circular startpoint set exists")

    intervals = (8, 10, 12, 15)
    speed_scales = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
    opponent_racelines = ("raceline0", "raceline2")
    scenarios = []
    for startpoint_ordinal, ego_idx in enumerate(selected_startpoints):
        for opponent_raceline in opponent_racelines:
            for interval_idx in intervals:
                opp_idx = int(get_opponent_startpoint("Austin", "raceline1", opponent_raceline, ego_idx, interval_idx))
                for speed_scale in speed_scales:
                    scenarios.append({
                        "scenario_id": f"bc-anchor-overlap-v2-sp{startpoint_ordinal:02d}-ego{ego_idx:04d}-{opponent_raceline}-i{interval_idx:02d}-v{int(round(speed_scale * 1000)):04d}",
                        "pool": "collision_only_anchor_overlap_v2",
                        "startpoint_ordinal": startpoint_ordinal,
                        "ego_idx": ego_idx,
                        "opp_idx": opp_idx,
                        "opp_raceline": opponent_raceline,
                        "opp_speedscale": speed_scale,
                        "interval_idx": interval_idx,
                        "map_name": "Austin",
                        "ego_raceline": "raceline1",
                        "sim_duration": 8.0,
                        "timestep": 0.01,
                        "integrator": "RK4",
                    })

    if len(scenarios) != args.startpoint_count * len(opponent_racelines) * len(intervals) * len(speed_scales):
        raise RuntimeError("panel Cartesian product is incomplete")
    panel_keys = {(scenario["opp_raceline"], scenario["ego_idx"], scenario["opp_idx"], scenario["opp_speedscale"]) for scenario in scenarios}
    if len(panel_keys) != len(scenarios):
        raise RuntimeError("panel episode keys are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_sha256 = atomic_write(panel_path, scenarios)
    manifest = {
        "schema_version": 1,
        "panel_id": "collision_only_anchor_overlap_v2",
        "status": "frozen_before_any_actor_outcome",
        "map_name": "Austin",
        "panel_path": str(panel_path),
        "panel_sha256": panel_sha256,
        "scenario_count": len(scenarios),
        "unique_episode_key_count": len(panel_keys),
        "unique_ego_startpoint_count": len(selected_startpoints),
        "ego_startpoints": selected_startpoints,
        "startpoint_algorithm": "SHA256-ranked offset search over 40 circular progress starts; first exact-disjoint set",
        "selected_offset": selected_offset,
        "raceline1_unique_waypoint_count": waypoint_count,
        "opponent_racelines": list(opponent_racelines),
        "interval_indices": list(intervals),
        "opponent_speed_scales": list(speed_scales),
        "excluded_ego_startpoint_count": len(excluded_startpoints),
        "excluded_panel_inputs": exclusion_inputs,
        "exact_ego_startpoint_overlap_with_excluded_panels": len(set(selected_startpoints) & excluded_startpoints),
    }
    atomic_write(manifest_path, manifest)
    print(json.dumps({"panel_sha256": panel_sha256, "scenario_count": len(scenarios), "ego_startpoints": selected_startpoints, "selected_offset": selected_offset}, indent=2, sort_keys=True))

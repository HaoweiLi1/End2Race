import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latticeplanner.utils import get_vertices
from ppo.env import make_environment
from ppo.reward import rectangle_clearance, rectangle_clearance_components
from utils import atomic_write_json


MAP_NAMES = ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")
OFFSETS = (0, 50, 100, 150)
GAMMA = 0.999
VEHICLE_LENGTH_M = 0.58
VEHICLE_WIDTH_M = 0.31
LONGITUDINAL_SAFE_M = 0.6
LATERAL_SAFE_M = 0.2
WALL_SAFE_M = 0.2
POTENTIAL_MAXIMUM = 0.05
DIRECTION_WINDOW_STEPS = 10
TIMESTEP_SECONDS = 0.01
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20260810


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-root", type=Path, default=Path("eval_results/pretrained_end2race"))
    parser.add_argument("--u44-root", type=Path, default=Path("eval_results/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44"))
    parser.add_argument("--output", type=Path, default=Path("eval_results/signed_interaction_phase_potential_gate_v2/gate_report.json"))
    return parser.parse_args()


def load_results(root, map_name):
    path = root / map_name / "multiagents" / "results_multi.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["final"]["total_episodes"] != 600 or payload["final"]["error_count"] != 0 or len(payload["episodes"]) != 600:
        raise RuntimeError(f"Invalid formal result package: {path}")
    return payload["episodes"]


def load_trace(root, map_name, episode_key):
    path = root / map_name / "multiagents" / "traces" / f"{episode_key}.npz"
    with np.load(path, allow_pickle=False) as payload:
        trace = {name: np.asarray(payload[name]) for name in ("ego_pose", "opp_pose", "ego_opp_collision", "ego_wall_collision", "action_applied", "terminal_post_step")}
    length = len(trace["ego_pose"])
    if trace["ego_pose"].shape != (length, 3) or trace["opp_pose"].shape != (length, 3):
        raise RuntimeError(f"Invalid pose arrays: {path}")
    if any(len(value) != length for value in trace.values()) or not np.isfinite(trace["ego_pose"]).all() or not np.isfinite(trace["opp_pose"]).all():
        raise RuntimeError(f"Invalid trace arrays: {path}")
    if not bool(trace["terminal_post_step"][-1]) or bool(trace["action_applied"][-1]) or not bool(trace["action_applied"][:-1].all()):
        raise RuntimeError(f"Invalid terminal contract: {path}")
    return trace


def front_geometry(trace):
    ego_pose = trace["ego_pose"]
    opp_pose = trace["opp_pose"]
    relative = opp_pose[:, :2] - ego_pose[:, :2]
    heading = ego_pose[:, 2]
    forward = np.stack((np.cos(heading), np.sin(heading)), axis=1)
    lateral = np.stack((-np.sin(heading), np.cos(heading)), axis=1)
    longitudinal_center = np.sum(relative * forward, axis=1)
    lateral_center = np.sum(relative * lateral, axis=1)
    relative_yaw = opp_pose[:, 2] - heading
    opponent_lateral_extent = 0.5 * (VEHICLE_LENGTH_M * np.abs(np.sin(relative_yaw)) + VEHICLE_WIDTH_M * np.abs(np.cos(relative_yaw)))
    lateral_overlap = 0.5 * VEHICLE_WIDTH_M + opponent_lateral_extent - np.abs(lateral_center)
    return (longitudinal_center > 0.0) & (lateral_overlap > 0.0), longitudinal_center, lateral_overlap


def episode_potentials(trace, map_clearance):
    front, _, _ = front_geometry(trace)
    current = np.empty(len(front), dtype=np.float64)
    signed = np.empty(len(front), dtype=np.float64)
    vehicle_distances = np.empty(len(front), dtype=np.float64)
    vehicle_shortfalls = np.empty(len(front), dtype=np.float64)
    for index in range(len(front)):
        ego_vertices = get_vertices(trace["ego_pose"][index], VEHICLE_LENGTH_M, VEHICLE_WIDTH_M)
        opp_vertices = get_vertices(trace["opp_pose"][index], VEHICLE_LENGTH_M, VEHICLE_WIDTH_M)
        _, longitudinal_clearance, lateral_clearance = rectangle_clearance_components(ego_vertices, opp_vertices, trace["ego_pose"][index, 2])
        wall_clearance = map_clearance.rectangle_clearance(ego_vertices)
        vehicle_distance = float(np.hypot(longitudinal_clearance / LONGITUDINAL_SAFE_M, lateral_clearance / LATERAL_SAFE_M))
        vehicle_shortfall = max(0.0, 1.0 - vehicle_distance)
        wall_shortfall = max(0.0, 1.0 - wall_clearance / WALL_SAFE_M)
        vehicle_distances[index] = vehicle_distance
        vehicle_shortfalls[index] = vehicle_shortfall
        current[index] = -POTENTIAL_MAXIMUM * max(vehicle_shortfall * vehicle_shortfall, wall_shortfall * wall_shortfall)
        signed[index] = -POTENTIAL_MAXIMUM * max(float(front[index]) * vehicle_shortfall * vehicle_shortfall, wall_shortfall * wall_shortfall)
    if not np.isfinite(current).all() or not np.isfinite(signed).all() or np.any(current > 0.0) or np.any(signed > 0.0) or np.any(current < -POTENTIAL_MAXIMUM) or np.any(signed < -POTENTIAL_MAXIMUM):
        raise RuntimeError("Potential contract failed")
    return front, current, signed, vehicle_distances, vehicle_shortfalls


def shaping_rewards(trace, potential):
    reward = np.zeros(len(potential), dtype=np.float64)
    terminated = trace["terminal_post_step"] & (trace["ego_opp_collision"] | trace["ego_wall_collision"])
    next_potential = potential.copy()
    next_potential[terminated] = 0.0
    reward[1:] = GAMMA * next_potential[1:] - potential[:-1]
    return reward


def minimum_clearance_index(trace):
    action_rows = int(trace["action_applied"].sum()) + 1
    clearances = np.empty(action_rows, dtype=np.float64)
    for index in range(action_rows):
        ego_vertices = get_vertices(trace["ego_pose"][index], VEHICLE_LENGTH_M, VEHICLE_WIDTH_M)
        opp_vertices = get_vertices(trace["opp_pose"][index], VEHICLE_LENGTH_M, VEHICLE_WIDTH_M)
        clearances[index] = rectangle_clearance(ego_vertices, opp_vertices)
    return int(np.argmin(clearances))


def summarize_episode(trace, event_index, map_clearance):
    if event_index < max(OFFSETS) + DIRECTION_WINDOW_STEPS or event_index >= len(trace["ego_pose"]):
        raise RuntimeError(f"Event index cannot support the frozen offsets: {event_index}")
    front, current_potential, signed_potential, vehicle_distances, vehicle_shortfalls = episode_potentials(trace, map_clearance)
    current_reward = shaping_rewards(trace, current_potential)
    signed_reward = shaping_rewards(trace, signed_potential)
    start = max(1, event_index - 149)
    indices = np.arange(start, event_index + 1)
    current_negative = float(np.maximum(-current_reward[indices], 0.0).sum())
    signed_negative = float(np.maximum(-signed_reward[indices], 0.0).sum())
    delta_reward = signed_reward[indices] - current_reward[indices]
    offset_indices = [event_index - offset for offset in OFFSETS]
    distance_closing_rates = [(vehicle_distances[index - DIRECTION_WINDOW_STEPS] - vehicle_distances[index]) / (DIRECTION_WINDOW_STEPS * TIMESTEP_SECONDS) for index in offset_indices]
    causal_rates = [(vehicle_shortfalls[index] - vehicle_shortfalls[index - DIRECTION_WINDOW_STEPS]) / (DIRECTION_WINDOW_STEPS * TIMESTEP_SECONDS) for index in offset_indices]
    return {
        "event_index": int(event_index),
        "front_at_offsets": [bool(front[index]) for index in offset_indices],
        "normalized_obb_distance_at_offsets": [float(vehicle_distances[index]) for index in offset_indices],
        "normalized_obb_distance_causal_closing_rate_at_offsets_per_second": [float(value) for value in distance_closing_rates],
        "distance_closing_zero_clearing_at_offsets": [1 if value > 1e-12 else -1 if value < -1e-12 else 0 for value in distance_closing_rates],
        "vehicle_shortfall_at_offsets": [float(vehicle_shortfalls[index]) for index in offset_indices],
        "vehicle_shortfall_causal_rate_at_offsets_per_second": [float(value) for value in causal_rates],
        "closing_zero_clearing_at_offsets": [1 if value > 1e-12 else -1 if value < -1e-12 else 0 for value in causal_rates],
        "window_transition_count": int(len(indices)),
        "front_window_fraction": float(front[max(0, event_index - 150):event_index + 1].mean()),
        "potential_changed_window_fraction": float((np.abs(signed_potential[max(0, event_index - 150):event_index + 1] - current_potential[max(0, event_index - 150):event_index + 1]) > 1e-12).mean()),
        "current_negative_shaping_mass": current_negative,
        "signed_negative_shaping_mass": signed_negative,
        "negative_shaping_released": float(current_negative - signed_negative),
        "negative_shaping_released_fraction": float((current_negative - signed_negative) / current_negative) if current_negative > 0.0 else None,
        "delta_reward_sum": float(delta_reward.sum()),
        "delta_reward_positive_mass": float(np.maximum(delta_reward, 0.0).sum()),
        "delta_reward_negative_mass": float(np.maximum(-delta_reward, 0.0).sum()),
        "current_reward_std": float(current_reward[indices].std()),
        "signed_reward_std": float(signed_reward[indices].std()),
        "delta_reward_abs_values": np.abs(delta_reward).tolist(),
    }


def summarize_cohort(rows):
    released = np.asarray([row["negative_shaping_released"] for row in rows], dtype=np.float64)
    current_negative = np.asarray([row["current_negative_shaping_mass"] for row in rows], dtype=np.float64)
    signed_negative = np.asarray([row["signed_negative_shaping_mass"] for row in rows], dtype=np.float64)
    current_std = np.asarray([row["current_reward_std"] for row in rows], dtype=np.float64)
    signed_std = np.asarray([row["signed_reward_std"] for row in rows], dtype=np.float64)
    delta_values = np.concatenate([np.asarray(row["delta_reward_abs_values"], dtype=np.float64) for row in rows])
    release_fractions = np.asarray([row["negative_shaping_released_fraction"] for row in rows if row["negative_shaping_released_fraction"] is not None], dtype=np.float64)
    return {
        "episode_count": int(len(rows)),
        "front_at_offsets_count": [int(sum(row["front_at_offsets"][index] for row in rows)) for index in range(len(OFFSETS))],
        "front_at_offsets_rate": [float(np.mean([row["front_at_offsets"][index] for row in rows])) for index in range(len(OFFSETS))],
        "front_window_fraction_mean": float(np.mean([row["front_window_fraction"] for row in rows])),
        "potential_changed_window_fraction_mean": float(np.mean([row["potential_changed_window_fraction"] for row in rows])),
        "current_negative_shaping_mass_total": float(current_negative.sum()),
        "signed_negative_shaping_mass_total": float(signed_negative.sum()),
        "negative_shaping_released_total": float(released.sum()),
        "negative_shaping_released_median": float(np.median(released)),
        "negative_shaping_released_fraction": float(released.sum() / current_negative.sum()) if current_negative.sum() > 0.0 else 0.0,
        "episode_release_fraction_defined_count": int(len(release_fractions)),
        "episode_release_fraction_median": float(np.median(release_fractions)) if len(release_fractions) else None,
        "closing_zero_clearing_at_offsets_count": [[int(sum(row["closing_zero_clearing_at_offsets"][index] == value for row in rows)) for value in (1, 0, -1)] for index in range(len(OFFSETS))],
        "distance_closing_zero_clearing_at_offsets_count": [[int(sum(row["distance_closing_zero_clearing_at_offsets"][index] == value for row in rows)) for value in (1, 0, -1)] for index in range(len(OFFSETS))],
        "vehicle_shortfall_positive_zero_at_offsets_count": [[int(sum(row["vehicle_shortfall_at_offsets"][index] > 0.0 for row in rows)), int(sum(row["vehicle_shortfall_at_offsets"][index] == 0.0 for row in rows))] for index in range(len(OFFSETS))],
        "normalized_obb_distance_causal_closing_rate_at_offsets_median": [float(np.median([row["normalized_obb_distance_causal_closing_rate_at_offsets_per_second"][index] for row in rows])) for index in range(len(OFFSETS))],
        "vehicle_shortfall_causal_rate_at_offsets_median": [float(np.median([row["vehicle_shortfall_causal_rate_at_offsets_per_second"][index] for row in rows])) for index in range(len(OFFSETS))],
        "current_reward_std_mean": float(current_std.mean()),
        "signed_reward_std_mean": float(signed_std.mean()),
        "signed_over_current_reward_std_mean_ratio": float(signed_std.mean() / current_std.mean()) if current_std.mean() > 0.0 else 0.0,
        "absolute_delta_reward_p50_p95_p99_max": [float(value) for value in np.quantile(delta_values, (0.5, 0.95, 0.99, 1.0))],
    }


def binary_auc(positive_scores, negative_scores):
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0.0).sum() + 0.5 * (comparisons == 0.0).sum()) / comparisons.size)


def startpoint_cluster_bootstrap_auc(positive_scores, negative_scores, groups, seed):
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    if len(positive) != len(negative):
        raise RuntimeError("Paired bootstrap inputs differ in length")
    group_values = np.asarray(groups)
    unique_groups = np.asarray(sorted(set(group_values.tolist())))
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        sampled_groups = unique_groups[rng.integers(0, len(unique_groups), size=len(unique_groups))]
        sample = np.concatenate([np.flatnonzero(group_values == group) for group in sampled_groups])
        values[index] = binary_auc(positive[sample], negative[sample])
    return [float(value) for value in np.quantile(values, (0.025, 0.5, 0.975))]


def startpoint_group(map_name, episode_key):
    ego_component = next(component for component in episode_key.split("_") if component.startswith("e"))
    return f"{map_name}:{ego_component}"


def initial_position(trace):
    return np.asarray(trace["ego_pose"][0, :2], dtype=np.float64)


if __name__ == "__main__":
    args = parse_arguments()
    bc_root = args.bc_root.resolve()
    u44_root = args.u44_root.resolve()
    created = []
    safe_candidates = {}
    result_tables = {}
    for map_name in MAP_NAMES:
        bc_results = load_results(bc_root, map_name)
        u44_results = load_results(u44_root, map_name)
        if set(bc_results) != set(u44_results):
            raise RuntimeError(f"BC/U44 episode identities differ for {map_name}")
        result_tables[map_name] = (bc_results, u44_results)
        for episode_key in sorted(bc_results):
            bc_episode = bc_results[episode_key]
            u44_episode = u44_results[episode_key]
            if not bc_episode["ego_collision_occurred"] and u44_episode["ego_opp_collision_occurred"]:
                created.append((map_name, episode_key, u44_episode))
            if u44_episode["outcome"] == "overtake" and not u44_episode["ego_collision_occurred"]:
                identity = (map_name, str(u44_episode["opp_raceline"]), float(u44_episode["opp_speedscale"]))
                safe_candidates.setdefault(identity, []).append((episode_key, u44_episode))
    if len(created) != 23:
        raise RuntimeError(f"Expected 23 U44-created ego-opponent collisions, got {len(created)}")

    used_controls = set()
    pairs = []
    for map_name, source_key, source_episode in created:
        source_trace = load_trace(u44_root, map_name, source_key)
        source_position = initial_position(source_trace)
        identity = (map_name, str(source_episode["opp_raceline"]), float(source_episode["opp_speedscale"]))
        candidates = []
        for control_key, control_episode in safe_candidates[identity]:
            if (map_name, control_key) in used_controls:
                continue
            control_trace = load_trace(u44_root, map_name, control_key)
            distance = float(np.linalg.norm(initial_position(control_trace) - source_position))
            candidates.append((distance, control_key, control_episode, control_trace))
        if not candidates:
            raise RuntimeError(f"No unused matched safe control for {map_name}/{source_key}")
        distance, control_key, control_episode, control_trace = min(candidates, key=lambda row: (row[0], row[1]))
        used_controls.add((map_name, control_key))
        pairs.append((map_name, source_key, control_key, distance, source_trace, control_trace))

    source_rows = []
    control_rows = []
    pair_rows = []
    global_flip_counts = []
    flip_signed_potential_jumps = []
    flip_delta_reward = []
    for map_name in MAP_NAMES:
        environment = make_environment(0, map_name)()
        map_clearance = environment.transition_reward.map_clearance
        map_pairs = [row for row in pairs if row[0] == map_name]
        for _, source_key, control_key, distance, source_trace, control_trace in map_pairs:
            source_event = int(np.flatnonzero(source_trace["ego_opp_collision"])[0])
            control_event = minimum_clearance_index(control_trace)
            source_summary = summarize_episode(source_trace, source_event, map_clearance)
            control_summary = summarize_episode(control_trace, control_event, map_clearance)
            bc_control = result_tables[map_name][0][control_key]
            source_rows.append(source_summary)
            control_rows.append(control_summary)
            pair_rows.append({
                "map_name": map_name,
                "source_episode_key": source_key,
                "source_startpoint_group": startpoint_group(map_name, source_key),
                "control_episode_key": control_key,
                "initial_position_distance_m": float(distance),
                "source_event_index": int(source_event),
                "control_event_index": int(control_event),
                "control_is_also_bc_safe_overtake": bool(bc_control["outcome"] == "overtake" and not bc_control["ego_collision_occurred"]),
                "control_minus_source_negative_shaping_released": float(control_summary["negative_shaping_released"] - source_summary["negative_shaping_released"]),
            })
        _, u44_results = result_tables[map_name]
        for episode_key in sorted(u44_results):
            trace = load_trace(u44_root, map_name, episode_key)
            front, _, _ = front_geometry(trace)
            action_rows = int(trace["action_applied"].sum()) + 1
            flip_indices = np.flatnonzero(front[1:action_rows] != front[:action_rows - 1]) + 1
            global_flip_counts.append(int(len(flip_indices)))
            for index in flip_indices:
                local = {name: value[index - 1:index + 1] for name, value in trace.items()}
                _, current_potential, signed_potential, _, _ = episode_potentials(local, map_clearance)
                delta_phi = signed_potential - current_potential
                flip_signed_potential_jumps.append(float(abs(signed_potential[1] - signed_potential[0])))
                flip_delta_reward.append(float(abs(GAMMA * delta_phi[1] - delta_phi[0])))
        environment.close()

    source_cohort = summarize_cohort(source_rows)
    control_cohort = summarize_cohort(control_rows)
    for pair_row, source_row, control_row in zip(pair_rows, source_rows, control_rows):
        pair_row["source_normalized_obb_distance_closing_rates"] = source_row["normalized_obb_distance_causal_closing_rate_at_offsets_per_second"]
        pair_row["control_normalized_obb_distance_closing_rates"] = control_row["normalized_obb_distance_causal_closing_rate_at_offsets_per_second"]
        pair_row["source_active_risk_rates"] = source_row["vehicle_shortfall_causal_rate_at_offsets_per_second"]
        pair_row["control_active_risk_rates"] = control_row["vehicle_shortfall_causal_rate_at_offsets_per_second"]
    pair_difference = np.asarray([row["control_minus_source_negative_shaping_released"] for row in pair_rows], dtype=np.float64)
    fraction_pair_difference = np.asarray([control["negative_shaping_released_fraction"] - source["negative_shaping_released_fraction"] for source, control in zip(source_rows, control_rows) if source["negative_shaping_released_fraction"] is not None and control["negative_shaping_released_fraction"] is not None], dtype=np.float64)
    startpoint_groups = [row["source_startpoint_group"] for row in pair_rows]
    directional_diagnostics = []
    for index, offset in enumerate(OFFSETS):
        source_distance_rates = np.asarray([row["normalized_obb_distance_causal_closing_rate_at_offsets_per_second"][index] for row in source_rows], dtype=np.float64)
        control_distance_rates = np.asarray([row["normalized_obb_distance_causal_closing_rate_at_offsets_per_second"][index] for row in control_rows], dtype=np.float64)
        source_distance_closing = (source_distance_rates > 1e-12).astype(np.float64)
        control_distance_closing = (control_distance_rates > 1e-12).astype(np.float64)
        source_rates = np.asarray([row["vehicle_shortfall_causal_rate_at_offsets_per_second"][index] for row in source_rows], dtype=np.float64)
        control_rates = np.asarray([row["vehicle_shortfall_causal_rate_at_offsets_per_second"][index] for row in control_rows], dtype=np.float64)
        source_closing = (source_rates > 1e-12).astype(np.float64)
        control_closing = (control_rates > 1e-12).astype(np.float64)
        source_risk = np.asarray([row["vehicle_shortfall_at_offsets"][index] for row in source_rows], dtype=np.float64)
        control_risk = np.asarray([row["vehicle_shortfall_at_offsets"][index] for row in control_rows], dtype=np.float64)
        directional_diagnostics.append({
            "offset_steps_before_event": int(offset),
            "offset_seconds_before_event": float(offset * TIMESTEP_SECONDS),
            "source_distance_closing_zero_clearing_count": [int((source_distance_rates > 1e-12).sum()), int((np.abs(source_distance_rates) <= 1e-12).sum()), int((source_distance_rates < -1e-12).sum())],
            "control_distance_closing_zero_clearing_count": [int((control_distance_rates > 1e-12).sum()), int((np.abs(control_distance_rates) <= 1e-12).sum()), int((control_distance_rates < -1e-12).sum())],
            "normalized_obb_distance_closing_rate_source_outcome_auc": binary_auc(source_distance_rates, control_distance_rates),
            "normalized_obb_distance_closing_rate_startpoint_cluster_bootstrap_auc_p2p5_p50_p97p5": startpoint_cluster_bootstrap_auc(source_distance_rates, control_distance_rates, startpoint_groups, BOOTSTRAP_SEED + 200 + index),
            "normalized_obb_distance_closing_indicator_source_outcome_auc": binary_auc(source_distance_closing, control_distance_closing),
            "normalized_obb_distance_closing_indicator_startpoint_cluster_bootstrap_auc_p2p5_p50_p97p5": startpoint_cluster_bootstrap_auc(source_distance_closing, control_distance_closing, startpoint_groups, BOOTSTRAP_SEED + 300 + index),
            "source_closing_zero_clearing_count": [int((source_rates > 1e-12).sum()), int((np.abs(source_rates) <= 1e-12).sum()), int((source_rates < -1e-12).sum())],
            "control_closing_zero_clearing_count": [int((control_rates > 1e-12).sum()), int((np.abs(control_rates) <= 1e-12).sum()), int((control_rates < -1e-12).sum())],
            "source_active_risk_positive_zero_count": [int((source_risk > 0.0).sum()), int((source_risk == 0.0).sum())],
            "control_active_risk_positive_zero_count": [int((control_risk > 0.0).sum()), int((control_risk == 0.0).sum())],
            "causal_risk_rate_source_outcome_auc": binary_auc(source_rates, control_rates),
            "causal_risk_rate_startpoint_cluster_bootstrap_auc_p2p5_p50_p97p5": startpoint_cluster_bootstrap_auc(source_rates, control_rates, startpoint_groups, BOOTSTRAP_SEED + index),
            "closing_indicator_source_outcome_auc": binary_auc(source_closing, control_closing),
            "closing_indicator_startpoint_cluster_bootstrap_auc_p2p5_p50_p97p5": startpoint_cluster_bootstrap_auc(source_closing, control_closing, startpoint_groups, BOOTSTRAP_SEED + 100 + index),
            "current_risk_level_source_outcome_auc": binary_auc(source_risk, control_risk),
        })
    flip_counts = np.asarray(global_flip_counts, dtype=np.float64)
    potential_jumps = np.asarray(flip_signed_potential_jumps, dtype=np.float64)
    delta_rewards = np.asarray(flip_delta_reward, dtype=np.float64)
    matching_sensitivity = {}
    for maximum_distance in (10.0, 20.0):
        values = np.asarray([row["control_minus_source_negative_shaping_released"] for row in pair_rows if row["initial_position_distance_m"] <= maximum_distance], dtype=np.float64)
        matching_sensitivity[f"distance_at_most_{maximum_distance:g}m"] = {
            "pair_count": int(len(values)),
            "positive_zero_negative": [int((values > 0.0).sum()), int((values == 0.0).sum()), int((values < 0.0).sum())],
            "mean": float(values.mean()),
            "median": float(np.median(values)),
        }
    identity_pass = source_cohort["front_at_offsets_count"] == [4, 3, 3, 1]
    total_selectivity_pass = control_cohort["negative_shaping_released_total"] > source_cohort["negative_shaping_released_total"]
    paired_selectivity_pass = float(np.median(pair_difference)) > 0.0
    legacy_procedural_verdict = "pass_to_formal_training" if identity_pass and total_selectivity_pass and paired_selectivity_pass else "stop_current_binary_front_potential"
    verdict = "diagnostic_complete_scientific_effect_inconclusive"
    report = {
        "schema_version": 2,
        "verdict": verdict,
        "legacy_procedural_verdict": legacy_procedural_verdict,
        "training_decision": "do_not_start_current_binary_front_arm_under_the_existing_budget_priority",
        "scientific_falsification": False,
        "interaction_phase_method_class_falsified": False,
        "method": "offline signed interaction-phase potential discriminability audit",
        "new_simulation": False,
        "actor_update": False,
        "maps": list(MAP_NAMES),
        "formal_trace_episode_count": 2400,
        "source_map_counts": {map_name: int(sum(row[0] == map_name for row in created)) for map_name in MAP_NAMES},
        "source_definition": "BC has no ego collision and U44 has ego-opponent collision",
        "control_definition": "U44 safe overtake; same map, opponent raceline and speed; nearest initial ego XY; one-to-one without replacement",
        "event_definition": {"source": "first ego-opponent collision", "control": "global minimum OBB surface clearance"},
        "potential_definition": {
            "current": "-0.05*max(vehicle_shortfall^2, wall_shortfall^2)",
            "signed": "-0.05*max(front*vehicle_shortfall^2, wall_shortfall^2)",
            "front": "opponent center ahead in ego body longitudinal axis and positive OBB lateral projection overlap",
            "gamma": GAMMA,
        },
        "criteria": {
            "identity_reproduces_4_3_3_1": bool(identity_pass),
            "legacy_safe_control_total_release_strictly_exceeds_collision_release": bool(total_selectivity_pass),
            "legacy_paired_median_control_minus_collision_release_positive": bool(paired_selectivity_pass),
            "legacy_absolute_release_criteria_scientifically_valid": False,
            "jitter_is_diagnostic_not_a_hard_gate": True,
            "directional_outcome_auc_is_diagnostic_not_a_method_class_gate": True,
        },
        "absolute_release_scale_confound": {
            "source_over_control_current_negative_shaping_mass": float(source_cohort["current_negative_shaping_mass_total"] / control_cohort["current_negative_shaping_mass_total"]),
            "source_over_control_negative_shaping_released": float(source_cohort["negative_shaping_released_total"] / control_cohort["negative_shaping_released_total"]),
            "source_release_fraction": float(source_cohort["negative_shaping_released_fraction"]),
            "control_release_fraction": float(control_cohort["negative_shaping_released_fraction"]),
            "paired_control_minus_source_release_fraction_defined_count": int(len(fraction_pair_difference)),
            "paired_control_minus_source_release_fraction_positive_zero_negative": [int((fraction_pair_difference > 0.0).sum()), int((fraction_pair_difference == 0.0).sum()), int((fraction_pair_difference < 0.0).sum())],
            "paired_control_minus_source_release_fraction_median": float(np.median(fraction_pair_difference)) if len(fraction_pair_difference) else None,
        },
        "causal_clearing_closing_diagnostic": {
            "untruncated_distance_quantity": "normalized_OBB_distance=hypot(longitudinal_clearance/0.6,lateral_clearance/0.2)",
            "untruncated_distance_closing_rate": "(distance_t_minus_10_steps-distance_t)/0.1s; positive is closing and negative is clearing",
            "active_risk_quantity": "vehicle_shortfall=max(0,1-normalized_OBB_distance)",
            "active_risk_causal_rate": "(risk_t-risk_t_minus_10_steps)/0.1s; positive is closing and negative is clearing",
            "bootstrap_unit": "source map plus ego startpoint cluster; matched control stays attached",
            "bootstrap_group_count": int(len(set(startpoint_groups))),
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "outcome_auc_scope": "diagnostic only; it does not test local action usefulness or close the method class",
            "offsets": directional_diagnostics,
        },
        "source_created_collision": source_cohort,
        "matched_safe_overtake_control": control_cohort,
        "paired_selectivity": {
            "pair_count": int(len(pair_rows)),
            "control_minus_collision_release_positive_count": int((pair_difference > 0.0).sum()),
            "zero_count": int((pair_difference == 0.0).sum()),
            "negative_count": int((pair_difference < 0.0).sum()),
            "mean": float(pair_difference.mean()),
            "median": float(np.median(pair_difference)),
            "minimum": float(pair_difference.min()),
            "maximum": float(pair_difference.max()),
        },
        "matching": {
            "initial_position_distance_m_p50_p95_max": [float(value) for value in np.quantile([row[3] for row in pairs], (0.5, 0.95, 1.0))],
            "controls_also_bc_safe_overtake_count": int(sum(row["control_is_also_bc_safe_overtake"] for row in pair_rows)),
            "distance_sensitivity": matching_sensitivity,
            "pairs": pair_rows,
        },
        "global_front_flip_diagnostic": {
            "episode_count": int(len(flip_counts)),
            "episodes_with_at_least_one_flip": int((flip_counts > 0).sum()),
            "flips_per_episode_p50_p95_p99_max": [float(value) for value in np.quantile(flip_counts, (0.5, 0.95, 0.99, 1.0))],
            "flip_transition_count": int(len(potential_jumps)),
            "absolute_signed_potential_jump_p50_p95_p99_max": [float(value) for value in np.quantile(potential_jumps, (0.5, 0.95, 0.99, 1.0))],
            "absolute_added_shaping_delta_p50_p95_p99_max": [float(value) for value in np.quantile(delta_rewards, (0.5, 0.95, 0.99, 1.0))],
            "nonzero_signed_potential_jump_count": int((potential_jumps > 1e-12).sum()),
            "nonzero_added_shaping_delta_count": int((delta_rewards > 1e-12).sum()),
            "added_shaping_delta_above_0p02_count": int((delta_rewards > 0.02).sum()),
        },
        "known_risks": [
            "The binary front gate has no clearing-versus-closing state and is false in most source and control decision windows.",
            "The untruncated closing rate has a local 0.5-second association, but the current vehicle-shortfall support is almost absent from 0.5 to 1.5 seconds.",
            "The absolute release comparisons are confounded by baseline risk mass and cannot establish a harmful direction.",
        ],
        "evidence_boundary": "The outcome AUROC is diagnostic on current U44 trajectories; it does not measure local action value, actor training effect, or an information-theoretic method-class limit.",
    }
    atomic_write_json(args.output, report)
    print(f"VERDICT={verdict}")
    print(f"LEGACY_PROCEDURAL_VERDICT={legacy_procedural_verdict}")
    print(f"SOURCE_FRONT_COUNTS={source_cohort['front_at_offsets_count']}")
    print(f"CONTROL_FRONT_COUNTS={control_cohort['front_at_offsets_count']}")
    print(f"SOURCE_RELEASE={source_cohort['negative_shaping_released_total']:.9f}")
    print(f"CONTROL_RELEASE={control_cohort['negative_shaping_released_total']:.9f}")
    print(f"PAIRED_POSITIVE_ZERO_NEGATIVE={(pair_difference > 0.0).sum()}/{(pair_difference == 0.0).sum()}/{(pair_difference < 0.0).sum()}")
    print(f"RELEASE_FRACTIONS={source_cohort['negative_shaping_released_fraction']:.9f}/{control_cohort['negative_shaping_released_fraction']:.9f}")
    print("DISTANCE_DIRECTION_AUROC=" + "/".join(f"{row['normalized_obb_distance_closing_rate_source_outcome_auc']:.6f}" for row in directional_diagnostics))
    print("ACTIVE_RISK_DIRECTION_AUROC=" + "/".join(f"{row['causal_risk_rate_source_outcome_auc']:.6f}" for row in directional_diagnostics))
    print(f"FLIP_COUNT={len(potential_jumps)}")

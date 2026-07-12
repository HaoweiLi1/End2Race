"""Exact evaluator replay with bounded D2.5 counterfactual interventions."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import gym
import numpy as np
import torch
from f110_gym.envs.base_classes import Integrator

from d0.outcomes import OutcomeRecord, centerline_length, classify_outcome
from d25 import BranchSpec
from demonstration import setup_opp_planner
from eval_multiagent import load_eval_model
from latticeplanner.utils import obsDict2oppoArray, project_point_to_centerline
from utils import (
    find_corresponding_waypoint,
    load_positions_and_speeds_from_params,
    multi_episode_key,
)


ARRAY_KEYS = (
    "time",
    "ego_lidar",
    "opp_lidar",
    "ego_desired_steer",
    "ego_desired_speed",
    "ego_actual_speed",
    "ego_pose",
    "ego_progress",
    "opp_desired_steer",
    "opp_desired_speed",
    "opp_actual_speed",
    "opp_pose",
    "opp_progress",
    "collision",
    "ego_collision",
    "opp_collision",
    "final_time",
    "final_ego_pose",
    "final_opp_pose",
    "final_ego_progress",
    "final_opp_progress",
    "state_label",
)


@dataclass(frozen=True)
class SimulationResult:
    arrays: dict
    outcome: OutcomeRecord
    action_clipped: bool
    episode_key: str


def compose_branch_action(base_steer: float, base_speed: float, branch: BranchSpec | None, step: int):
    steer = float(base_steer)
    speed = float(base_speed)
    active = False
    if branch is not None and branch.start_step <= step < branch.start_step + branch.duration_steps:
        active = True
        steer += branch.intervention.steer_rad
        speed -= branch.intervention.brake_mps
    clipped_steer = float(np.clip(steer, -0.52, 0.52))
    clipped_speed = float(max(0.0, speed))
    clipped = active and (clipped_steer != steer or clipped_speed != speed)
    return clipped_steer, clipped_speed, clipped, active


def _resolve_case(case: Mapping) -> dict:
    required = {
        "map_name",
        "resolved_ego_idx",
        "opponent_raceline",
        "speedscale_hex",
    }
    missing = required - set(case)
    if missing:
        raise ValueError(f"D2.5 case missing fields: {sorted(missing)}")
    map_name = str(case["map_name"])
    ego_idx = int(case["resolved_ego_idx"])
    opp_raceline = str(case["opponent_raceline"])
    speedscale = float.fromhex(str(case["speedscale_hex"]))
    base_path = Path("f1tenth_racetracks") / map_name
    ego_waypoints = np.loadtxt(base_path / "raceline1.csv", delimiter=";", skiprows=1)
    if opp_raceline == "raceline1":
        opp_waypoints = ego_waypoints
        opp_idx = (ego_idx + 15) % len(opp_waypoints)
    else:
        opp_waypoints = np.loadtxt(base_path / f"{opp_raceline}.csv", delimiter=";", skiprows=1)
        ego_row = np.array(
            [ego_waypoints[ego_idx % len(ego_waypoints), 1], ego_waypoints[ego_idx % len(ego_waypoints), 2]]
        )
        opp_xy_speed = np.column_stack([opp_waypoints[:, 1], opp_waypoints[:, 2]])
        nearest = int(np.argmin(np.linalg.norm(opp_xy_speed - ego_row, axis=1)))
        opp_idx = (nearest + 15) % len(opp_waypoints)
    return {
        "map_name": map_name,
        "ego_raceline": "raceline1",
        "opp_raceline": opp_raceline,
        "ego_idx": ego_idx,
        "opp_idx": int(opp_idx),
        "interval_idx": 15,
        "opp_speedscale": speedscale,
    }


def _trajectory_arrays(records: dict, terminal: Mapping, state_label: str) -> dict:
    return {
        "time": np.asarray(records["time"], dtype=np.float32),
        "ego_lidar": np.asarray(records["ego_lidar"], dtype=np.float32),
        "opp_lidar": np.asarray(records["opp_lidar"], dtype=np.float32),
        "ego_desired_steer": np.asarray(records["ego_desired_steer"], dtype=np.float32),
        "ego_desired_speed": np.asarray(records["ego_desired_speed"], dtype=np.float32),
        "ego_actual_speed": np.asarray(records["ego_actual_speed"], dtype=np.float32),
        "ego_pose": np.asarray(records["ego_pose"], dtype=np.float32),
        "ego_progress": np.asarray(records["ego_progress"], dtype=np.float32),
        "opp_desired_steer": np.asarray(records["opp_desired_steer"], dtype=np.float32),
        "opp_desired_speed": np.asarray(records["opp_desired_speed"], dtype=np.float32),
        "opp_actual_speed": np.asarray(records["opp_actual_speed"], dtype=np.float32),
        "opp_pose": np.asarray(records["opp_pose"], dtype=np.float32),
        "opp_progress": np.asarray(records["opp_progress"], dtype=np.float32),
        "collision": np.asarray(terminal["collision"], dtype=bool),
        "ego_collision": np.asarray(terminal["ego_collision"], dtype=bool),
        "opp_collision": np.asarray(terminal["opp_collision"], dtype=bool),
        "final_time": np.float32(terminal["final_time"]),
        "final_ego_pose": np.asarray(terminal["final_ego_pose"], dtype=np.float32),
        "final_opp_pose": np.asarray(terminal["final_opp_pose"], dtype=np.float32),
        "final_ego_progress": np.float32(terminal["final_ego_progress"]),
        "final_opp_progress": np.float32(terminal["final_opp_progress"]),
        "state_label": np.asarray(state_label),
    }


def classify_trajectory(arrays: Mapping, map_name: str) -> OutcomeRecord:
    raw = str(np.asarray(arrays["state_label"]).reshape(()))
    json_episode = {
        "outcome": raw,
        "state_label": raw,
        "ego_collision": bool(np.asarray(arrays["ego_collision"]).reshape(())),
        "opp_collision": bool(np.asarray(arrays["opp_collision"]).reshape(())),
    }
    return classify_outcome(
        arrays,
        json_episode,
        centerline_length("f1tenth_racetracks", map_name),
    )


def simulate_episode(
    model,
    device: torch.device,
    case: Mapping,
    branch: BranchSpec | None = None,
    sim_duration: float = 8.0,
) -> SimulationResult:
    np.random.seed(42)
    scenario = _resolve_case(case)
    map_name = scenario["map_name"]
    env = gym.make(
        "f110-v0",
        map=f"f1tenth_racetracks/{map_name}/{map_name}_map",
        map_ext=".png",
        num_agents=2,
        timestep=0.01,
        integrator=Integrator.RK4,
    )
    try:
        positions, initial_speeds = load_positions_and_speeds_from_params(scenario, map_name)
        opponent = setup_opp_planner(map_name, scenario["opp_raceline"])
        hidden = torch.zeros((1, 1, model.gru.hidden_size), device=device)
        prev_speed = float(initial_speeds[0]) * 0.9
        centerline_wp = np.loadtxt(
            f"f1tenth_racetracks/{map_name}/raceline1.csv", delimiter=";", skiprows=1
        )
        centerline = centerline_wp[:, 1:3]
        # Preserve the evaluator's scalar accumulation order.  This matters at
        # the seam because even a one-ulp length drift changes archived
        # progress values after the wrap correction.
        track_length = float(
            sum(
                np.linalg.norm(centerline[index + 1] - centerline[index])
                for index in range(len(centerline) - 1)
            )
        )
        obs, _, done, _ = env.reset(poses=positions)
        initial_ego_progress, _ = project_point_to_centerline(
            np.array([obs["poses_x"][0], obs["poses_y"][0]]), centerline
        )
        initial_opp_progress, _ = project_point_to_centerline(
            np.array([obs["poses_x"][1], obs["poses_y"][1]]), centerline
        )
        final_state = "overtaking" if initial_ego_progress > initial_opp_progress else "following"
        final_ego_progress = float(initial_ego_progress)
        final_opp_progress = float(initial_opp_progress)
        lap_time = 0.0
        collision = ego_collision = opp_collision = False
        tracker_count = 0
        tracker_steps = 10
        opp_traj = None
        action_clipped = False
        records = {name: [] for name in (
            "time", "ego_lidar", "opp_lidar", "ego_desired_steer", "ego_desired_speed",
            "ego_actual_speed", "ego_pose", "ego_progress", "opp_desired_steer",
            "opp_desired_speed", "opp_actual_speed", "opp_pose", "opp_progress",
        )}
        step = 0
        while not done and lap_time < sim_duration:
            ego_lidar = np.asarray(obs["scans"][0]).reshape(-1)
            if len(ego_lidar) != 360:
                ego_lidar = ego_lidar[np.linspace(0, len(ego_lidar) - 1, 360, dtype=int)]
            opp_lidar = np.asarray(obs["scans"][1]).reshape(-1)
            if len(opp_lidar) != 360:
                opp_lidar = opp_lidar[np.linspace(0, len(opp_lidar) - 1, 360, dtype=int)]
            actual_speed_observer = getattr(model, "observe_actual_speed", None)
            if actual_speed_observer is not None:
                # Narrow compatibility hook for deployable sidecars.  The BC
                # call below intentionally retains the evaluator's historical
                # one-step-lagged speed input, while a sidecar can observe the
                # current physical speed available in the same observation.
                actual_speed_observer(float(obs["linear_vels_x"][0]))
            with torch.no_grad():
                lidar_t = torch.tensor(ego_lidar, dtype=torch.float32, device=device).view(1, 1, 360)
                speed_t = torch.tensor([[[prev_speed]]], dtype=torch.float32, device=device)
                action_sequence, hidden = model(lidar_t, speed_t, hidden)
                base_steer = float(action_sequence[0, -1, 0].item())
                base_speed = float(action_sequence[0, -1, 1].item())
            base_steer = float(np.clip(base_steer, -0.52, 0.52))
            ego_steer, ego_speed, clipped, _ = compose_branch_action(
                base_steer, base_speed, branch, step
            )
            applied_command_observer = getattr(model, "observe_applied_command", None)
            if applied_command_observer is not None:
                applied_command_observer(ego_steer, ego_speed)
            action_clipped = action_clipped or clipped
            prev_speed = obs["linear_vels_x"][0]
            if tracker_count == 0:
                opp_traj = opponent.plan(
                    obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1],
                    obsDict2oppoArray(obs, 1), obs["linear_vels_x"][1]
                )
            opp_steer, opp_speed = opponent.tracker.plan(
                obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1],
                obs["linear_vels_x"][1], opp_traj
            )
            opp_steer = float(np.clip(opp_steer, -0.52, 0.52))
            opp_speed = float(opp_speed) * scenario["opp_speedscale"]
            ego_progress, _ = project_point_to_centerline(
                np.array([obs["poses_x"][0], obs["poses_y"][0]]), centerline
            )
            opp_progress, _ = project_point_to_centerline(
                np.array([obs["poses_x"][1], obs["poses_y"][1]]), centerline
            )
            if ego_progress < initial_ego_progress - track_length / 2:
                ego_progress += track_length
            if opp_progress < initial_opp_progress - track_length / 2:
                opp_progress += track_length
            values = {
                "time": float(lap_time),
                "ego_lidar": ego_lidar.copy(),
                "opp_lidar": opp_lidar.copy(),
                "ego_desired_steer": ego_steer,
                "ego_desired_speed": ego_speed,
                "ego_actual_speed": float(obs["linear_vels_x"][0]),
                "ego_pose": [obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]],
                "ego_progress": float(ego_progress),
                "opp_desired_steer": opp_steer,
                "opp_desired_speed": opp_speed,
                "opp_actual_speed": float(obs["linear_vels_x"][1]),
                "opp_pose": [obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1]],
                "opp_progress": float(opp_progress),
            }
            for name, value in values.items():
                records[name].append(value)
            actions = np.array([[ego_steer, ego_speed], [opp_steer, opp_speed]])
            obs, timestep, done, _ = env.step(actions)
            lap_time += timestep
            ego_progress, _ = project_point_to_centerline(
                np.array([obs["poses_x"][0], obs["poses_y"][0]]), centerline
            )
            opp_progress, _ = project_point_to_centerline(
                np.array([obs["poses_x"][1], obs["poses_y"][1]]), centerline
            )
            if ego_progress < initial_ego_progress - track_length / 2:
                ego_progress += track_length
            if opp_progress < initial_opp_progress - track_length / 2:
                opp_progress += track_length
            final_state = "overtaking" if ego_progress > opp_progress else "following"
            final_ego_progress = float(ego_progress)
            final_opp_progress = float(opp_progress)
            if np.any(obs["collisions"]):
                collision = True
                ego_collision = bool(obs["collisions"][0])
                opp_collision = bool(np.any(obs["collisions"][1:]))
                done = True
            tracker_count = (tracker_count + 1) % tracker_steps
            step += 1
        state_label = "collision" if collision else final_state
        terminal = {
            "collision": collision,
            "ego_collision": ego_collision,
            "opp_collision": opp_collision,
            "final_time": float(lap_time),
            "final_ego_pose": [obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]],
            "final_opp_pose": [obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1]],
            "final_ego_progress": final_ego_progress,
            "final_opp_progress": final_opp_progress,
        }
        arrays = _trajectory_arrays(records, terminal, state_label)
        outcome = classify_trajectory(arrays, map_name)
        episode_key = multi_episode_key(
            scenario["opp_raceline"], scenario["ego_idx"], scenario["opp_idx"],
            scenario["opp_speedscale"]
        )
        return SimulationResult(arrays, outcome, action_clipped, episode_key)
    finally:
        env.close()
        gc.collect()


def compare_archived(arrays: Mapping, archived: Mapping) -> dict:
    mismatches = []
    max_abs_error = {}
    for key in ARRAY_KEYS:
        if key not in arrays or key not in archived:
            mismatches.append(f"missing:{key}")
            continue
        actual = np.asarray(arrays[key])
        expected = np.asarray(archived[key])
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            mismatches.append(f"shape_or_dtype:{key}")
            continue
        if not np.array_equal(actual, expected):
            mismatches.append(key)
            if actual.dtype.kind in "fiu" and actual.shape == expected.shape:
                max_abs_error[key] = float(
                    np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
                )
    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "max_abs_error": max_abs_error,
    }


def load_bc_model(model_path: str, device: torch.device):
    return load_eval_model(model_path, hidden_scale=4, device=device)

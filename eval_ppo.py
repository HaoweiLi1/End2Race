#!/usr/bin/env python3
"""Deterministic multi-agent PPO gate evaluation for End2Race checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import torch

from utils_ppo import (
    End2RaceActorCritic,
    End2RaceHazardActorCritic,
    End2RacePPOEnv,
    centerline_arc_length,
    compute_opp_idx_like_eval_multiagent,
    load_actor_critic_checkpoint,
    wrap_rel_s,
)

try:  # available in the full End2Race repository
    from latticeplanner.utils import project_point_to_centerline
except Exception:  # allows --help/import smoke tests outside the repo
    project_point_to_centerline = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic PPO gate evaluation")
    p.add_argument("--mode", choices=("compatibility", "safety_augmented"), default="compatibility")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--baseline_json", type=str, default="")
    p.add_argument("--report_json", type=str, default="eval_results/ppo_gate.json")
    p.add_argument("--scenario_grid", choices=("single", "default"), default="single")
    p.add_argument("--map_name", type=str, default="Austin")
    p.add_argument("--ego_idx", type=int, default=0)
    p.add_argument("--interval_idx", type=int, default=15)
    p.add_argument("--ego_raceline", type=str, default="raceline1")
    p.add_argument("--opp_raceline", type=str, default="raceline1")
    p.add_argument("--opp_speedscale", type=float, default=0.5)
    p.add_argument("--sim_duration", type=float, default=8.0)
    p.add_argument("--hidden_scale", type=int, default=4)
    p.add_argument("--max_speed", type=float, default=20.0)
    p.add_argument("--num_episodes", type=int, default=1)
    p.add_argument("--noise", type=float, default=0.0, help="Fraction of policy LiDAR beams to mask to zero")
    p.add_argument("--singleagent_baseline_json", type=str, default="")
    p.add_argument("--singleagent_map", type=str, default="")
    p.add_argument("--singleagent_lap_num", type=int, default=1)
    p.add_argument("--singleagent_noise", type=float, default=0.0)
    p.add_argument("--save_actor_if_gate_passed", action="store_true")
    p.add_argument("--save_actor_path", type=str, default="pretrained/end2race_ppo.pth")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_policy(args: argparse.Namespace, device: torch.device) -> Tuple[torch.nn.Module, bool]:
    use_hazard = args.mode == "safety_augmented"
    if use_hazard:
        inner = End2RaceActorCritic(hidden_scale=args.hidden_scale).to(device)
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("mode") == "safety_augmented" and "actor_critic" in ckpt:
            ac = End2RaceHazardActorCritic(inner).to(device)
            ac.load_state_dict(ckpt["actor_critic"])
        else:
            load_actor_critic_checkpoint(inner, args.model_path, device)
            ac = End2RaceHazardActorCritic(inner).to(device)
    else:
        ac = End2RaceActorCritic(hidden_scale=args.hidden_scale).to(device)
        load_actor_critic_checkpoint(ac, args.model_path, device)
    ac.eval()
    return ac, use_hazard


def make_scenario(args: argparse.Namespace, ego_idx: int, interval_idx: int) -> Dict[str, object]:
    opp_idx = compute_opp_idx_like_eval_multiagent(
        args.map_name,
        args.ego_raceline,
        args.opp_raceline,
        ego_idx,
        interval_idx,
    )
    return {
        "map_name": args.map_name,
        "ego_raceline": args.ego_raceline,
        "opp_raceline": args.opp_raceline,
        "ego_idx": int(ego_idx),
        "interval_idx": int(interval_idx),
        "opp_idx": int(opp_idx),
        "opp_speedscale": float(args.opp_speedscale),
    }


def build_scenarios(args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.scenario_grid == "single":
        return [make_scenario(args, args.ego_idx, args.interval_idx) for _ in range(args.num_episodes)]
    offsets = [0, 50, 100, 150]
    intervals = [args.interval_idx, -5, 5, 15, 30]
    scenarios: List[Dict[str, object]] = []
    while len(scenarios) < args.num_episodes:
        for offset in offsets:
            for interval in intervals:
                scenarios.append(make_scenario(args, args.ego_idx + offset, interval))
                if len(scenarios) >= args.num_episodes:
                    break
            if len(scenarios) >= args.num_episodes:
                break
    return scenarios


def compare_with_baseline(report: Dict[str, object], baseline: Dict[str, object]) -> Tuple[bool, Dict[str, bool]]:
    eps = 1e-9
    checks = {
        "collision_rate_ok": float(report["collision_rate"]) <= float(baseline.get("collision_rate", 1.0)) + eps,
        "post_overtake_collision_rate_ok": float(report["post_overtake_collision_rate"])
        <= float(baseline.get("post_overtake_collision_rate", 1.0)) + eps,
        "overtake_rate_improved": float(report["overtake_rate"]) > float(baseline.get("overtake_rate", -1.0)) + eps,
        "safe_overtake_held_rate_improved": float(report["safe_overtake_held_rate"])
        > float(baseline.get("safe_overtake_held_rate", -1.0)) + eps,
        "mean_centerline_progress_improved": float(report["mean_centerline_progress"])
        > float(baseline.get("mean_centerline_progress", -1.0)) + eps,
    }
    checks["performance_improved"] = bool(
        checks["overtake_rate_improved"]
        or checks["safe_overtake_held_rate_improved"]
        or checks["mean_centerline_progress_improved"]
    )
    checks["single_agent_regression_ok"] = bool(report.get("single_agent_regression_passed") is True)
    passed = bool(
        checks["collision_rate_ok"]
        and checks["post_overtake_collision_rate_ok"]
        and checks["single_agent_regression_ok"]
        and checks["performance_improved"]
    )
    return passed, checks


def parse_singleagent_stdout(stdout: str) -> Dict[str, object]:
    status_match = re.search(r"Status:\s*(.+)", stdout)
    progress_match = re.search(r"Lap Progress:\s*([0-9.]+)\s*laps", stdout)
    speed_match = re.search(r"Average Speed:\s*([0-9.]+)\s*m/s", stdout)
    variance_match = re.search(r"Speed Variance:\s*([0-9.]+)", stdout)
    return {
        "status": status_match.group(1).strip() if status_match else "unknown",
        "lap_progress": float(progress_match.group(1)) if progress_match else 0.0,
        "avg_speed": float(speed_match.group(1)) if speed_match else 0.0,
        "speed_variance": float(variance_match.group(1)) if variance_match else 0.0,
    }


def run_singleagent_regression(args: argparse.Namespace, ac: torch.nn.Module, use_hazard: bool) -> Tuple[bool | None, Dict[str, object] | None]:
    if use_hazard or not args.singleagent_baseline_json:
        return None, None
    with open(args.singleagent_baseline_json, "r") as f:
        baseline = json.load(f)
    eval_map = args.singleagent_map or args.map_name
    with tempfile.NamedTemporaryFile(suffix="_end2race_actor.pth", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(ac.actor.state_dict(), tmp_path)
        cmd = [
            sys.executable,
            "eval_singleagent.py",
            "--model_path",
            tmp_path,
            "--map_name",
            eval_map,
            "--lap_num",
            str(args.singleagent_lap_num),
            "--noise",
            str(args.singleagent_noise),
        ]
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        current = parse_singleagent_stdout(proc.stdout)
        current["returncode"] = proc.returncode
        current["stderr_tail"] = proc.stderr[-1000:]
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    bc_status = str(baseline.get("status", ""))
    current_status = str(current.get("status", ""))
    collision_regressed = "Collision occurred" in current_status and "Collision occurred" not in bc_status
    incomplete_regressed = "Incomplete" in current_status and "Incomplete" not in bc_status and "Collision occurred" not in bc_status
    progress_ok = float(current["lap_progress"]) >= float(baseline.get("lap_progress", 0.0)) - float(baseline.get("lap_progress_tolerance", 0.05))
    speed_ok = float(current["avg_speed"]) >= float(baseline.get("avg_speed", 0.0)) - float(baseline.get("avg_speed_tolerance", 0.0))
    passed = bool(proc.returncode == 0 and not collision_regressed and not incomplete_regressed and progress_ok and speed_ok)
    return passed, {
        "map": eval_map,
        "current": current,
        "baseline": baseline,
        "collision_regressed": collision_regressed,
        "incomplete_regressed": incomplete_regressed,
        "progress_ok": progress_ok,
        "speed_ok": speed_ok,
    }


def maybe_mask_lidar(obs: Dict[str, np.ndarray], noise: float, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    if noise <= 0.0:
        return obs
    obs = dict(obs)
    lidar = np.array(obs["lidar"], dtype=np.float32, copy=True)
    n = int(len(lidar) * noise)
    if n > 0:
        idx = rng.choice(len(lidar), min(n, len(lidar)), replace=False)
        lidar[idx] = 0.0
    obs["lidar"] = lidar
    return obs


def require_project_point() -> object:
    if project_point_to_centerline is None:
        raise ImportError("latticeplanner.utils.project_point_to_centerline is required for eval_ppo.py")
    return project_point_to_centerline


def run_episode(
    ac: torch.nn.Module,
    env: End2RacePPOEnv,
    device: torch.device,
    use_hazard: bool,
    scenario: Dict[str, object],
    noise: float = 0.0,
) -> Dict[str, float]:
    project = require_project_point()
    rng = np.random.default_rng(12345)
    hidden_size = int(ac.actor.gru.hidden_size)
    hidden = torch.zeros(1, 1, hidden_size, device=device)
    obs = env.reset(scenario=scenario)
    obs = maybe_mask_lidar(obs, noise, rng)
    centerline = env.centerline
    tl = centerline_arc_length(centerline)
    assert env._prev_obs is not None
    ego_xy0 = np.array([env._prev_obs["poses_x"][0], env._prev_obs["poses_y"][0]], dtype=np.float64)
    opp_xy0 = np.array([env._prev_obs["poses_x"][1], env._prev_obs["poses_y"][1]], dtype=np.float64)
    p0, _ = project(ego_xy0, centerline)
    opp_p0, _ = project(opp_xy0, centerline)
    rel_s0 = wrap_rel_s(float(p0 - opp_p0), tl)

    collision = ego_collision = opp_collision = safe_held = False
    speeds: List[float] = []
    traj_len = 0.0
    clipped_flags: List[float] = []
    last_xy = ego_xy0.copy()
    done = False
    while not done:
        lidar = torch.tensor(obs["lidar"], dtype=torch.float32, device=device).view(1, 1, -1)
        spd = torch.tensor(obs["prev_speed"], dtype=torch.float32, device=device).view(1, 1, -1)
        with torch.no_grad():
            if use_hazard:
                haz = torch.tensor(obs["hazard"], dtype=torch.float32, device=device).view(1, 1, -1)
                dist, _, hidden = ac.forward(lidar, spd, haz, hidden)
            else:
                dist, _, hidden = ac.forward(lidar, spd, hidden)
            raw = dist.mean.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        obs, _, done, info = env.step(raw)
        obs = maybe_mask_lidar(obs, noise, rng)
        assert env._prev_obs is not None
        xy = np.array([env._prev_obs["poses_x"][0], env._prev_obs["poses_y"][0]], dtype=np.float64)
        traj_len += float(np.linalg.norm(xy - last_xy))
        last_xy = xy.copy()
        speeds.append(float(env._prev_obs["linear_vels_x"][0]))
        collision = collision or bool(info.get("any_collision", False))
        ego_collision = ego_collision or bool(info.get("ego_collision", False))
        opp_collision = opp_collision or bool(info.get("opp_collision", False))
        safe_held = safe_held or bool(info.get("safe_overtake_held", False))
        clipped_flags.append(float(info.get("action_was_clipped", False)))

    post_overtake_collision = bool(env._reward_state.post_overtake_collision) if env._reward_state else False
    side_near_miss = bool(env._reward_state and env._reward_state.min_side_gap < env.reward_weights.side_dist_thresh)
    rear_near_miss = bool(env._reward_state and env._reward_state.min_rear_gap < env.reward_weights.rear_dist_thresh)
    ego_xy_f = np.array([env._prev_obs["poses_x"][0], env._prev_obs["poses_y"][0]], dtype=np.float64)
    opp_xy_f = np.array([env._prev_obs["poses_x"][1], env._prev_obs["poses_y"][1]], dtype=np.float64)
    pf, _ = project(ego_xy_f, centerline)
    opp_pf, _ = project(opp_xy_f, centerline)
    progress_delta = float(pf - p0)
    if tl > 0:
        if progress_delta < -tl / 2:
            progress_delta += tl
        elif progress_delta > tl / 2:
            progress_delta -= tl
    centerline_progress_ratio = progress_delta / tl if tl > 0 else 0.0
    rel_sf = wrap_rel_s(float(pf - opp_pf), tl)
    rel_progress = wrap_rel_s(rel_sf - rel_s0, tl) / tl if tl > 0 else 0.0
    overtake = bool(rel_sf > env.reward_weights.overtake_margin_s)
    following = bool(not overtake and not collision)
    return {
        "collision": float(collision),
        "ego_collision": float(ego_collision),
        "opp_collision": float(opp_collision),
        "overtake": float(overtake),
        "following": float(following),
        "safe_overtake_held": float(safe_held),
        "post_overtake_collision": float(post_overtake_collision),
        "side_near_miss": float(side_near_miss),
        "rear_near_miss": float(rear_near_miss),
        "mean_centerline_progress": float(centerline_progress_ratio),
        "mean_relative_progress": float(rel_progress),
        "mean_total_distance": float(traj_len),
        "mean_avg_speed_noncollision": float(np.mean(speeds)) if speeds and not collision else 0.0,
        "mean_speed_variance_noncollision": float(np.var(speeds)) if speeds and not collision else 0.0,
        "clipped_action_fraction": float(np.mean(clipped_flags)) if clipped_flags else 0.0,
    }


def aggregate_episode_metrics(episodes: List[Dict[str, float]]) -> Dict[str, object]:
    keys = episodes[0].keys() if episodes else []
    report: Dict[str, object] = {"episodes": len(episodes)}
    mapping = {
        "collision": "collision_rate",
        "ego_collision": "ego_collision_rate",
        "opp_collision": "opponent_collision_rate",
        "overtake": "overtake_rate",
        "following": "following_rate",
        "safe_overtake_held": "safe_overtake_held_rate",
        "post_overtake_collision": "post_overtake_collision_rate",
        "side_near_miss": "side_near_miss_rate",
        "rear_near_miss": "rear_near_miss_rate",
    }
    for src, dst in mapping.items():
        report[dst] = float(np.mean([e[src] for e in episodes])) if episodes else 0.0
    for key in keys:
        if key.startswith("mean_") or key == "clipped_action_fraction":
            report[key] = float(np.mean([e[key] for e in episodes])) if episodes else 0.0
    return report


def main() -> None:
    args = parse_args()
    if args.baseline_json and args.mode == "compatibility" and not args.singleagent_baseline_json:
        raise ValueError("--singleagent_baseline_json is required with --baseline_json for compatibility gate evaluation")
    device = torch.device(args.device)
    ac, use_hazard = build_policy(args, device)
    scenarios = build_scenarios(args)
    env = End2RacePPOEnv(
        map_name=args.map_name,
        mode="safety_augmented" if use_hazard else "compatibility",
        max_speed=args.max_speed,
        sim_duration=args.sim_duration,
        terminate_on_success=False,
        terminate_on_severe_unsafe=False,
        seed=0,
    )
    episode_metrics: List[Dict[str, float]] = []
    try:
        for scenario in scenarios:
            episode_metrics.append(run_episode(ac, env, device, use_hazard, scenario, noise=args.noise))
    finally:
        env.close()
    report = aggregate_episode_metrics(episode_metrics)
    report["single_agent_regression_passed"] = None
    single_passed, single_details = run_singleagent_regression(args, ac, use_hazard)
    report["single_agent_regression_passed"] = single_passed
    if single_details is not None:
        report["single_agent_regression"] = single_details
    if args.baseline_json:
        with open(args.baseline_json, "r") as f:
            baseline = json.load(f)
        gate_passed, gate_checks = compare_with_baseline(report, baseline)
        report["gate_passed"] = gate_passed
        report["gate_checks"] = gate_checks
        if gate_passed and args.save_actor_if_gate_passed and not use_hazard:
            os.makedirs(os.path.dirname(args.save_actor_path) or ".", exist_ok=True)
            torch.save(ac.actor.state_dict(), args.save_actor_path)
            report["saved_actor_path"] = args.save_actor_path
    else:
        report["gate_passed"] = None
    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

_LOCAL_F110_GYM = os.path.join(os.path.dirname(__file__), "f1tenth_gym", "gym")
if os.path.isdir(_LOCAL_F110_GYM) and _LOCAL_F110_GYM not in sys.path:
    sys.path.insert(0, _LOCAL_F110_GYM)

import gym
import numpy as np
import f110_gym  # noqa: F401 - registers f110-v0 with gym.
from f110_gym.envs.base_classes import Integrator

from latticeplanner.utils import (
    find_corresponding_waypoint,
    load_centerline_from_map,
    obsDict2oppoArray,
    project_point_to_centerline,
)

from demonstration import setup_opp_planner
from utils import (
    load_raceline_xytheta_speed,
    load_two_agent_positions_and_speeds,
    resolve_two_agent_indices,
)


def centerline_arc_length(centerline: np.ndarray) -> float:
    """Return total arc length of a polyline centerline."""
    centerline = np.asarray(centerline, dtype=np.float64)
    if len(centerline) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())


def wrap_rel_s(delta_s: float, track_length: float) -> float:
    """Wrap relative progress into [-track_length / 2, track_length / 2]."""
    if track_length <= 0.0:
        return float(delta_s)
    return float((delta_s + 0.5 * track_length) % track_length - 0.5 * track_length)


def advance_progress(
    p_new_raw: float,
    p_last: float,
    initial_p: float,
    track_length: float,
) -> Tuple[float, float]:
    """Unwrap raw centerline progress so deltas remain smooth near lap wrap."""
    del initial_p
    if track_length <= 0.0:
        p_new = float(p_new_raw)
        return p_new, p_new - float(p_last)

    p_new_raw = float(p_new_raw)
    p_last = float(p_last)
    base_laps = int(np.floor((p_last - p_new_raw) / track_length))
    candidates = [p_new_raw + (base_laps + offset) * track_length for offset in (-1, 0, 1, 2)]
    p_new = min(candidates, key=lambda p: abs(p - p_last))
    return float(p_new), float(p_new - p_last)


@dataclass
class RewardWeights:
    w_progress: float = 1.0
    w_rel_progress: float = 0.2
    w_overtake_progress: float = 0.4
    safe_factor_power: float = 1.0
    w_opponent_risk: float = 1.0
    w_unsafe_merge_back: float = 2.0
    w_collision: float = 120.0
    w_post_overtake_collision: float = 60.0
    w_overtake_success: float = 25.0
    w_smooth: float = 0.01
    w_steer_mag: float = 0.003
    w_speed: float = 0.0
    w_timeout: float = 0.0

    side_s_thresh: float = 0.5
    side_dist_thresh: float = 1.5
    rear_s_thresh: float = 2.0
    rear_dist_thresh: float = 2.5
    front_s_thresh: float = 1.5
    front_too_close_s: float = 0.8
    overtake_margin_s: float = 0.6
    severe_ttc: float = 0.15
    safe_overtake_hold_duration: float = 0.7
    rear_clearance_safe_s: float = 2.0
    rear_clearance_safe_dist: float = 2.5
    lateral_clearance_safe: float = 0.6
    progress_clip_back: float = 0.05
    progress_clip_forward: float = 0.12
    rel_progress_clip: float = 0.08
    smooth_speed_scale: float = 0.05


@dataclass
class RewardState:
    track_length: float
    initial_ego_s: float
    initial_opp_s: float
    last_ego_s: float
    last_opp_s: float
    was_ahead: bool
    overtake_started: bool = False
    safe_overtake_hold_time: float = 0.0
    safe_overtake_held: bool = False
    severe_unsafe: bool = False
    post_overtake_collision: bool = False
    had_safe_overtake_bonus: bool = False

    @classmethod
    def from_obs(cls, obs: Dict[str, Any], centerline: np.ndarray) -> "RewardState":
        track_length = centerline_arc_length(centerline)
        ego_s, _ = project_point_to_centerline(
            np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64),
            centerline,
        )
        opp_s, _ = project_point_to_centerline(
            np.array([obs["poses_x"][1], obs["poses_y"][1]], dtype=np.float64),
            centerline,
        )
        rel_s = wrap_rel_s(float(ego_s) - float(opp_s), track_length)
        return cls(
            track_length=track_length,
            initial_ego_s=float(ego_s),
            initial_opp_s=float(opp_s),
            last_ego_s=float(ego_s),
            last_opp_s=float(opp_s),
            was_ahead=rel_s > 0.0,
        )


def _load_centerline(map_name: str) -> np.ndarray:
    map_directory = os.path.join("f1tenth_racetracks", map_name)
    return np.asarray(load_centerline_from_map(map_directory), dtype=np.float64)


def _relative_metrics(obs: Dict[str, Any], centerline: np.ndarray, rw: RewardWeights) -> Dict[str, float]:
    del rw
    ego_pos = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
    opp_pos = np.array([obs["poses_x"][1], obs["poses_y"][1]], dtype=np.float64)
    ego_s, _ = project_point_to_centerline(ego_pos, centerline)
    opp_s, _ = project_point_to_centerline(opp_pos, centerline)
    track_length = centerline_arc_length(centerline)
    rel_s = wrap_rel_s(float(ego_s) - float(opp_s), track_length)

    dx, dy = opp_pos - ego_pos
    ego_theta = float(obs["poses_theta"][0])
    rel_y_ego = -np.sin(ego_theta) * dx + np.cos(ego_theta) * dy
    rel_dist = float(np.linalg.norm(opp_pos - ego_pos))
    ego_v = float(obs["linear_vels_x"][0])
    opp_v = float(obs["linear_vels_x"][1])
    rel_v = ego_v - opp_v
    return {
        "ego_s_raw": float(ego_s),
        "opp_s_raw": float(opp_s),
        "rel_s": rel_s,
        "rel_y_ego": float(rel_y_ego),
        "rel_dist": rel_dist,
        "ego_v": ego_v,
        "opp_v": opp_v,
        "rel_v": rel_v,
    }


def _risk_terms(rel_s: float, rel_dist: float, rel_v: float, rw: RewardWeights) -> Dict[str, float]:
    side_risk = 0.0
    if abs(rel_s) <= rw.side_s_thresh and rel_dist <= rw.side_dist_thresh:
        side_risk = (1.0 - abs(rel_s) / max(rw.side_s_thresh, 1e-6)) * (
            1.0 - rel_dist / max(rw.side_dist_thresh, 1e-6)
        )

    rear_risk = 0.0
    if 0.0 < rel_s <= rw.rear_s_thresh and rel_dist <= rw.rear_dist_thresh:
        rear_risk = (1.0 - rel_s / max(rw.rear_s_thresh, 1e-6)) * (
            1.0 - rel_dist / max(rw.rear_dist_thresh, 1e-6)
        )

    front_risk = 0.0
    if -rw.front_s_thresh <= rel_s < 0.0:
        front_gap = abs(rel_s)
        front_risk = 1.0 - front_gap / max(rw.front_s_thresh, 1e-6)
        if front_gap <= rw.front_too_close_s:
            front_risk = max(front_risk, 1.0 - front_gap / max(rw.front_too_close_s, 1e-6))

    ttc = np.inf
    if rel_s < 0.0 and rel_v > 1e-6:
        ttc = abs(rel_s) / rel_v
    elif rel_s > 0.0 and rel_v < -1e-6:
        ttc = rel_s / (-rel_v)

    opponent_risk = float(np.clip(max(side_risk, rear_risk, front_risk), 0.0, 1.0))
    severe_unsafe = bool((ttc < rw.severe_ttc) or (side_risk > 0.95))
    return {
        "side_risk": float(np.clip(side_risk, 0.0, 1.0)),
        "rear_risk": float(np.clip(rear_risk, 0.0, 1.0)),
        "front_risk": float(np.clip(front_risk, 0.0, 1.0)),
        "opponent_risk": opponent_risk,
        "ttc": float(ttc) if np.isfinite(ttc) else float("inf"),
        "severe_unsafe": severe_unsafe,
    }


def compute_shaped_reward(
    obs: Dict[str, Any],
    reward_state: RewardState,
    centerline: np.ndarray,
    rw: RewardWeights,
    prev_exec_action: np.ndarray,
    executed_ego_action: np.ndarray,
    dt: float,
    timeout: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Compute PPO shaped reward and update RewardState in place."""
    metrics = _relative_metrics(obs, centerline, rw)
    risk = _risk_terms(metrics["rel_s"], metrics["rel_dist"], metrics["rel_v"], rw)

    ego_s, delta_ego_s = advance_progress(
        metrics["ego_s_raw"],
        reward_state.last_ego_s,
        reward_state.initial_ego_s,
        reward_state.track_length,
    )
    opp_s, delta_opp_s = advance_progress(
        metrics["opp_s_raw"],
        reward_state.last_opp_s,
        reward_state.initial_opp_s,
        reward_state.track_length,
    )
    prev_rel_s = wrap_rel_s(
        reward_state.last_ego_s - reward_state.last_opp_s,
        reward_state.track_length,
    )
    rel_s = wrap_rel_s(ego_s - opp_s, reward_state.track_length)
    delta_rel_s = wrap_rel_s(rel_s - prev_rel_s, reward_state.track_length)

    progress_raw = float(np.clip(delta_ego_s, -rw.progress_clip_back, rw.progress_clip_forward))
    rel_progress_raw = float(np.clip(delta_rel_s, -rw.rel_progress_clip, rw.rel_progress_clip))
    safe_factor = max(0.0, 1.0 - risk["opponent_risk"]) ** rw.safe_factor_power
    overtake_progress_raw = max(rel_progress_raw, 0.0)

    prev_exec_action = np.asarray(prev_exec_action, dtype=np.float64)
    executed_ego_action = np.asarray(executed_ego_action, dtype=np.float64)
    action_delta = executed_ego_action - prev_exec_action
    smooth_raw = float(action_delta[0] ** 2 + (rw.smooth_speed_scale * action_delta[1]) ** 2)
    steer_mag_raw = float(executed_ego_action[0] ** 2)
    collision = bool(np.any(obs["collisions"]))

    if not reward_state.was_ahead and rel_s > rw.overtake_margin_s:
        reward_state.overtake_started = True

    safe_window = (
        rel_s > rw.overtake_margin_s
        and not collision
        and risk["opponent_risk"] < 0.25
        and reward_state.overtake_started
    )
    if safe_window:
        reward_state.safe_overtake_hold_time += float(dt)
    elif rel_s <= 0.0 or collision:
        reward_state.safe_overtake_hold_time = 0.0

    if reward_state.safe_overtake_hold_time >= rw.safe_overtake_hold_duration:
        reward_state.safe_overtake_held = True

    if collision and (reward_state.overtake_started or reward_state.safe_overtake_held):
        reward_state.post_overtake_collision = True

    reward_state.severe_unsafe = bool(risk["severe_unsafe"])

    rear_clearance_deficit = 0.0
    dist_deficit = 0.0
    lateral_deficit = 0.0
    unsafe_merge_back = 0.0
    if reward_state.overtake_started and rel_s > 0.0:
        rear_clearance_deficit = max(0.0, rw.rear_clearance_safe_s - rel_s) / max(
            rw.rear_clearance_safe_s, 1e-6
        )
        dist_deficit = max(0.0, rw.rear_clearance_safe_dist - metrics["rel_dist"]) / max(
            rw.rear_clearance_safe_dist, 1e-6
        )
        lateral_deficit = max(0.0, rw.lateral_clearance_safe - abs(metrics["rel_y_ego"])) / max(
            rw.lateral_clearance_safe, 1e-6
        )
        steer_aggression = abs(executed_ego_action[0]) / 0.52
        unsafe_merge_back = max(
            risk["rear_risk"],
            rear_clearance_deficit * max(dist_deficit, lateral_deficit),
        )
        unsafe_merge_back *= 1.0 + steer_aggression

    success_bonus = 0.0
    if reward_state.safe_overtake_held and not reward_state.had_safe_overtake_bonus:
        success_bonus = 1.0
        reward_state.had_safe_overtake_bonus = True

    reward_progress = rw.w_progress * progress_raw
    reward_rel_progress = rw.w_rel_progress * safe_factor * rel_progress_raw
    reward_overtake_progress = rw.w_overtake_progress * safe_factor * overtake_progress_raw
    reward_opponent_risk = -rw.w_opponent_risk * risk["opponent_risk"]
    reward_unsafe_merge_back = -rw.w_unsafe_merge_back * unsafe_merge_back
    reward_smooth = -rw.w_smooth * smooth_raw
    reward_steer_mag = -rw.w_steer_mag * steer_mag_raw
    reward_collision = -rw.w_collision if collision else 0.0
    reward_post_overtake_collision = (
        -rw.w_post_overtake_collision if reward_state.post_overtake_collision and collision else 0.0
    )
    reward_overtake_success = rw.w_overtake_success * success_bonus
    reward_speed = rw.w_speed * metrics["ego_v"]
    reward_timeout = -rw.w_timeout if timeout else 0.0

    total = (
        reward_progress
        + reward_rel_progress
        + reward_overtake_progress
        + reward_opponent_risk
        + reward_unsafe_merge_back
        + reward_smooth
        + reward_steer_mag
        + reward_collision
        + reward_post_overtake_collision
        + reward_overtake_success
        + reward_speed
        + reward_timeout
    )

    reward_state.last_ego_s = ego_s
    reward_state.last_opp_s = opp_s
    reward_state.was_ahead = rel_s > 0.0

    terms = {
        "reward_progress": float(reward_progress),
        "reward_rel_progress": float(reward_rel_progress),
        "reward_overtake_progress": float(reward_overtake_progress),
        "reward_opponent_risk": float(reward_opponent_risk),
        "reward_unsafe_merge_back": float(reward_unsafe_merge_back),
        "reward_smooth": float(reward_smooth),
        "reward_steer_mag": float(reward_steer_mag),
        "reward_collision": float(reward_collision),
        "reward_post_overtake_collision": float(reward_post_overtake_collision),
        "reward_overtake_success": float(reward_overtake_success),
        "reward_speed": float(reward_speed),
        "reward_timeout": float(reward_timeout),
        "delta_ego_s": float(delta_ego_s),
        "delta_opp_s": float(delta_opp_s),
        "delta_rel_s": float(delta_rel_s),
        "rel_s": float(rel_s),
        "rel_y_ego": float(metrics["rel_y_ego"]),
        "rel_dist": float(metrics["rel_dist"]),
        "rel_v": float(metrics["rel_v"]),
        "opponent_risk": float(risk["opponent_risk"]),
        "safe_factor": float(safe_factor),
        "unsafe_merge_back": float(unsafe_merge_back),
        "lateral_deficit": float(lateral_deficit),
        "rear_clearance_deficit": float(rear_clearance_deficit),
        "distance_clearance_deficit": float(dist_deficit),
        "side_risk": float(risk["side_risk"]),
        "rear_risk": float(risk["rear_risk"]),
        "front_risk": float(risk["front_risk"]),
        "ttc": float(risk["ttc"]),
        "safe_overtake_hold_time": float(reward_state.safe_overtake_hold_time),
        "safe_overtake_held": float(reward_state.safe_overtake_held),
        "severe_unsafe": float(reward_state.severe_unsafe),
        "post_overtake_collision": float(reward_state.post_overtake_collision),
    }
    return float(total), terms


def build_hazard(obs: Dict[str, Any], centerline: np.ndarray, rw: RewardWeights) -> np.ndarray:
    """Build a 7D hazard vector for reward/metric logging only."""
    metrics = _relative_metrics(obs, centerline, rw)
    risk = _risk_terms(metrics["rel_s"], metrics["rel_dist"], metrics["rel_v"], rw)
    hazard = np.array(
        [
            metrics["rel_s"] / 5.0,
            metrics["rel_y_ego"] / 2.0,
            metrics["rel_dist"] / 5.0,
            metrics["rel_v"] / 5.0,
            1.0 if risk["side_risk"] > 0.0 else 0.0,
            1.0 if risk["rear_risk"] > 0.0 else 0.0,
            1.0 if risk["front_risk"] > 0.0 else 0.0,
        ],
        dtype=np.float32,
    )
    return np.clip(hazard, -5.0, 5.0).astype(np.float32)


def sample_opp_speedscale(stage: int, rng: np.random.Generator) -> float:
    if stage <= 1:
        return float(rng.uniform(0.45, 0.75))
    if stage == 2:
        return float(rng.uniform(0.5, 0.9))
    return float(rng.uniform(0.4, 1.0))


def sample_scenario(
    stage: int,
    rng: np.random.Generator,
    map_name: str,
    ego_raceline_choices: Sequence[str],
    opp_raceline_choices: Sequence[str],
) -> Dict[str, Any]:
    ego_raceline = str(rng.choice(list(ego_raceline_choices)))
    opp_raceline = str(rng.choice(list(opp_raceline_choices)))
    ego_waypoints = load_raceline_xytheta_speed(map_name, ego_raceline)
    opp_waypoints = load_raceline_xytheta_speed(map_name, opp_raceline)

    ego_idx = int(rng.integers(0, len(ego_waypoints)))
    if stage <= 1:
        interval_idx = int(rng.integers(8, 22))
    elif stage == 2:
        interval_idx = int(rng.integers(5, 32))
    else:
        interval_idx = int(rng.integers(3, 45))

    if opp_raceline == ego_raceline:
        opp_idx = (ego_idx + interval_idx) % len(opp_waypoints)
    else:
        ego_map_idx = int(find_corresponding_waypoint(ego_waypoints[ego_idx], opp_waypoints))
        opp_idx = (ego_map_idx + interval_idx) % len(opp_waypoints)

    return {
        "map_name": map_name,
        "ego_raceline": ego_raceline,
        "opp_raceline": opp_raceline,
        "ego_idx": ego_idx,
        "interval_idx": interval_idx,
        "opp_idx": int(opp_idx),
        "opp_speedscale": sample_opp_speedscale(stage, rng),
    }


def downsample_for_eval_compat(lidar: np.ndarray, target_points: int = 360) -> np.ndarray:
    lidar = np.asarray(lidar, dtype=np.float32).reshape(-1)
    if len(lidar) != target_points:
        indices = np.linspace(0, len(lidar) - 1, target_points, dtype=int)
        lidar = lidar[indices]
    lidar = np.nan_to_num(lidar, nan=0.0, posinf=30.0, neginf=0.0)
    return np.clip(lidar, 0.0, 30.0).astype(np.float32)


class End2RacePPOEnv:
    """Two-agent End2Race PPO environment with LiDAR+speed policy observations."""

    def __init__(
        self,
        map_name: str,
        max_speed: float = 20.0,
        sim_duration: float = 8.0,
        terminate_on_success: bool = True,
        terminate_on_severe_unsafe: bool = False,
        seed: int = 0,
        reward_weights: Optional[RewardWeights] = None,
        ego_raceline_choices: Optional[Sequence[str]] = None,
        opp_raceline_choices: Optional[Sequence[str]] = None,
    ):
        self.map_name = map_name
        self.max_speed = float(max_speed)
        self.sim_duration = float(sim_duration)
        self.terminate_on_success = bool(terminate_on_success)
        self.terminate_on_severe_unsafe = bool(terminate_on_severe_unsafe)
        self.reward_weights = reward_weights or RewardWeights()
        self.rng = np.random.default_rng(seed)
        self.stage = 1
        self.ego_raceline_choices = tuple(ego_raceline_choices or ("raceline1",))
        self.opp_raceline_choices = tuple(opp_raceline_choices or ("raceline1",))

        self.env = gym.make(
            "f110-v0",
            map=f"f1tenth_racetracks/{map_name}/{map_name}_map",
            map_ext=".png",
            num_agents=2,
            timestep=0.01,
            integrator=Integrator.RK4,
        )
        self.timestep = float(self.env.timestep)
        self.centerline = _load_centerline(map_name)

        self._raw_obs: Optional[Dict[str, Any]] = None
        self._reward_state: Optional[RewardState] = None
        self._opponent = None
        self._opp_traj = None
        self._tracker_count = 0
        self._tracker_steps = 10
        self._prev_speed = 0.0
        self._prev_exec_action = np.zeros(2, dtype=np.float32)
        self._t = 0.0
        self._scenario: Optional[Dict[str, Any]] = None

    def close(self) -> None:
        self.env.close()

    def reset(self, scenario: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        if scenario is None:
            scenario = sample_scenario(
                self.stage,
                self.rng,
                self.map_name,
                self.ego_raceline_choices,
                self.opp_raceline_choices,
            )
        self._scenario = self._complete_scenario(dict(scenario))

        self._opponent = setup_opp_planner(self.map_name, self._scenario["opp_raceline"])
        self._opp_traj = None
        self._tracker_count = 0
        self._tracker_steps = int(getattr(self._opponent.conf, "tracker_steps", 10))
        if hasattr(self._opponent.tracker, "prev_error"):
            self._opponent.tracker.prev_error = 0.0
        self._opponent.prev_opp_pose = np.array([0.0, 0.0])
        self._opponent.prev_traj_local = np.zeros_like(self._opponent.prev_traj_local)
        self._opponent.best_traj = None
        self._opponent.goal_grid = None

        positions, initial_speeds = self._scenario_positions(self._scenario)
        obs, _, _, _ = self.env.reset(poses=positions)
        self._raw_obs = obs
        self._prev_speed = float(initial_speeds[0]) * 0.9
        self._t = 0.0
        self._prev_exec_action = np.array([0.0, self._prev_speed], dtype=np.float32)
        self._reward_state = RewardState.from_obs(obs, self.centerline)
        return self._build_obs(obs)

    def step(self, raw_ego_action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self._raw_obs is None or self._reward_state is None:
            raise RuntimeError("End2RacePPOEnv.step() called before reset().")

        executed_ego_action = np.asarray(raw_ego_action, dtype=np.float32).reshape(2).copy()
        executed_ego_action[0] = np.clip(executed_ego_action[0], -0.52, 0.52)
        executed_ego_action[1] = np.clip(executed_ego_action[1], 0.0, self.max_speed)

        opp_action = self._opp_planner_step(self._raw_obs)
        actions = np.vstack((executed_ego_action, opp_action)).astype(np.float32)
        next_obs, _, env_done, env_info = self.env.step(actions)
        self._t += self.timestep

        collision_any = bool(np.any(next_obs["collisions"]))
        timeout = bool(self._t >= self.sim_duration)

        reward, reward_terms = compute_shaped_reward(
            next_obs,
            self._reward_state,
            self.centerline,
            self.reward_weights,
            self._prev_exec_action,
            executed_ego_action,
            self.timestep,
            timeout=timeout,
        )

        success = bool(self.terminate_on_success and self._reward_state.safe_overtake_held)
        severe = bool(self.terminate_on_severe_unsafe and self._reward_state.severe_unsafe)
        terminated = bool(collision_any or env_done or severe)
        truncated = bool((not terminated) and (timeout or success))
        action_was_clipped = bool(
            np.any(np.abs(np.asarray(raw_ego_action, dtype=np.float32).reshape(2) - executed_ego_action) > 1e-6)
        )

        self._prev_exec_action = executed_ego_action.copy()
        self._prev_speed = float(next_obs["linear_vels_x"][0])
        self._raw_obs = next_obs

        info: Dict[str, Any] = {
            **reward_terms,
            "terminated": terminated,
            "truncated": truncated,
            "collision": collision_any,
            "env_done": bool(env_done),
            "timeout": timeout,
            "success": success,
            "severe": severe,
            "time": float(self._t),
            "executed_ego_action": executed_ego_action.copy(),
            "raw_ego_action": np.asarray(raw_ego_action, dtype=np.float32).reshape(2).copy(),
            "action_was_clipped": action_was_clipped,
            "opp_action": opp_action.copy(),
            "hazard": build_hazard(next_obs, self.centerline, self.reward_weights),
            "env_info": env_info,
        }
        return self._build_obs(next_obs), float(reward), terminated, truncated, info

    def _build_obs(self, raw_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return {
            "lidar": downsample_for_eval_compat(raw_obs["scans"][0]),
            "prev_speed": np.array([self._prev_speed], dtype=np.float32),
        }

    def _opp_planner_step(self, prev_obs: Dict[str, Any]) -> np.ndarray:
        if self._opponent is None:
            raise RuntimeError("Opponent planner is not initialized.")

        if self._tracker_count == 0 or self._opp_traj is None:
            opp_poses = obsDict2oppoArray(prev_obs, 1)
            self._opp_traj = self._opponent.plan(
                prev_obs["poses_x"][1],
                prev_obs["poses_y"][1],
                prev_obs["poses_theta"][1],
                opp_poses,
                prev_obs["linear_vels_x"][1],
            )

        opp_steer, opp_speed = self._opponent.tracker.plan(
            prev_obs["poses_x"][1],
            prev_obs["poses_y"][1],
            prev_obs["poses_theta"][1],
            prev_obs["linear_vels_x"][1],
            self._opp_traj,
        )
        opp_steer = float(np.clip(opp_steer, -0.52, 0.52))
        opp_speed = float(opp_speed) * float(self._scenario["opp_speedscale"])
        self._tracker_count = (self._tracker_count + 1) % max(self._tracker_steps, 1)
        return np.array([opp_steer, opp_speed], dtype=np.float32)

    def _complete_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        scenario.setdefault("map_name", self.map_name)
        scenario.setdefault("ego_raceline", self.ego_raceline_choices[0])
        scenario.setdefault("opp_raceline", self.opp_raceline_choices[0])
        scenario.setdefault("ego_idx", 0)
        scenario.setdefault("interval_idx", 15)
        scenario.setdefault("opp_speedscale", sample_opp_speedscale(self.stage, self.rng))

        ego_idx, opp_idx = resolve_two_agent_indices(
            self.map_name,
            scenario["ego_raceline"],
            scenario["opp_raceline"],
            scenario["ego_idx"],
            scenario["interval_idx"],
            scenario.get("opp_idx"),
        )
        scenario["ego_idx"] = ego_idx
        scenario["opp_idx"] = opp_idx
        return scenario

    def _scenario_positions(self, scenario: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        positions, initial_speeds = load_two_agent_positions_and_speeds(
            self.map_name,
            scenario["ego_raceline"],
            scenario["opp_raceline"],
            scenario["ego_idx"],
            scenario["opp_idx"],
        )
        return positions.astype(np.float64), initial_speeds.astype(np.float64)

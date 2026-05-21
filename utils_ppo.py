"""Shared PPO utilities for End2Race.

This file merges the functionality that was previously split across
``actor_critic.py``, ``ppo_env.py``, ``ppo_rewards.py`` and
``rollout_buffer.py``.  The public entry points are used by ``train_ppo.py``
and ``eval_ppo.py``.

The implementation follows the two-stage FINAL_SPEC design:

* v1 / compatibility: 360-d LiDAR + previous speed, actor-only checkpoint can
  be loaded by the existing End2Race evaluators.
* v2 / safety_augmented: same backbone plus a 7-d hazard vector that enters
  through residual heads after the GRU.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from model import End2Race

# Optional simulator imports. Keep f1tenth_gym lazy because importing it can
# initialize pyglet/OpenGL on some machines. Unit tests can still import the
# actor-critic and buffer without the simulator installed.
try:  # pragma: no cover - gym is available in the real End2Race environment
    import gym
except Exception:  # pragma: no cover
    gym = None  # type: ignore[assignment]
Integrator = None  # type: ignore[assignment]

try:  # pragma: no cover - available in the repository
    from latticeplanner.utils import (
        load_centerline_from_map,
        obsDict2oppoArray,
        project_point_to_centerline,
    )
except Exception:  # pragma: no cover
    load_centerline_from_map = None  # type: ignore[assignment]
    obsDict2oppoArray = None  # type: ignore[assignment]
    project_point_to_centerline = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Actor-critic models
# -----------------------------------------------------------------------------


class End2RaceActorCritic(nn.Module):
    """v1 compatibility PPO actor-critic around the existing End2Race model.

    ``actor`` is a plain End2Race instance.  Its output is interpreted as the
    Gaussian policy mean, while a PPO-only value head consumes the GRU features.
    Saving ``actor.state_dict()`` therefore produces a checkpoint that remains
    loadable by ``eval_singleagent.py`` and ``eval_multiagent.py``.
    """

    def __init__(
        self,
        hidden_scale: int = 4,
        steer_std: float = 0.03,
        speed_std: float = 0.25,
    ) -> None:
        super().__init__()
        self.actor = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)
        h = int(self.actor.gru.hidden_size)
        mid = max(h // 4, 1)
        self.value_head = nn.Sequential(nn.Linear(h, mid), nn.ReLU(), nn.Linear(mid, 1))
        self.log_std = nn.Parameter(
            torch.tensor([math.log(steer_std), math.log(speed_std)], dtype=torch.float32)
        )

    def forward_features(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return GRU output and hidden state using End2Race's exact feature path."""
        processed_lidar = (-1.0 / (1.0 + torch.exp(-self.actor.k * lidar)) + 1.0) * 2.0
        speed_embedding = self.actor.speed_mlp(speed_input)
        features = torch.cat([processed_lidar, speed_embedding], dim=2)
        gru_out, last_hidden = self.actor.gru(features, hidden)
        return gru_out, last_hidden

    def forward(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        gru_out, last_hidden = self.forward_features(lidar, speed_input, hidden)
        mean = self.actor.output_layer(gru_out)
        std = self.log_std.exp().view(1, 1, -1).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        value = self.value_head(gru_out).squeeze(-1)
        return dist, value, last_hidden

    def evaluate_actions(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hidden: Optional[torch.Tensor],
        raw_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value, _ = self.forward(lidar, speed_input, hidden)
        logp = dist.log_prob(raw_actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value

    def act(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hidden: Optional[torch.Tensor],
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value, new_hidden = self.forward(lidar, speed_input, hidden)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value, new_hidden

    def save_actor_backbone(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def save_full_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Any,
        ppo_config: Dict[str, Any],
        reward_config: Dict[str, Any],
        stage: int,
        iteration: int,
        metrics: Dict[str, Any],
    ) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "version": "ppo_v1_compat",
                "mode": "compatibility",
                "actor_critic": self.state_dict(),
                "actor": self.actor.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "hidden_scale": self.actor.hidden_scale,
                "log_std": self.log_std.detach().cpu(),
                "ppo_config": ppo_config,
                "reward_config": reward_config,
                "stage": stage,
                "iteration": iteration,
                "metrics": metrics,
            },
            path,
        )


class End2RaceHazardActorCritic(nn.Module):
    """v2 PPO model with residual hazard-conditioned actor/value deltas."""

    def __init__(
        self,
        v1_actor_critic: End2RaceActorCritic,
        hazard_dim: int = 7,
        hazard_emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.actor = copy.deepcopy(v1_actor_critic.actor)
        self.log_std = nn.Parameter(v1_actor_critic.log_std.detach().clone())
        h = int(self.actor.gru.hidden_size)
        mid = max(h // 4, 1)
        self.base_value_head = copy.deepcopy(v1_actor_critic.value_head)
        self.hazard_mlp = nn.Sequential(
            nn.Linear(hazard_dim, hazard_emb_dim),
            nn.ReLU(),
            nn.Linear(hazard_emb_dim, hazard_emb_dim),
            nn.ReLU(),
        )
        self.delta_actor = nn.Sequential(nn.Linear(h + hazard_emb_dim, mid), nn.ReLU(), nn.Linear(mid, 2))
        self.delta_value = nn.Sequential(nn.Linear(h + hazard_emb_dim, mid), nn.ReLU(), nn.Linear(mid, 1))
        nn.init.zeros_(self.delta_actor[-1].weight)
        nn.init.zeros_(self.delta_actor[-1].bias)
        nn.init.zeros_(self.delta_value[-1].weight)
        nn.init.zeros_(self.delta_value[-1].bias)

    def forward_features(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        processed_lidar = (-1.0 / (1.0 + torch.exp(-self.actor.k * lidar)) + 1.0) * 2.0
        speed_embedding = self.actor.speed_mlp(speed_input)
        features = torch.cat([processed_lidar, speed_embedding], dim=2)
        gru_out, last_hidden = self.actor.gru(features, hidden)
        return gru_out, last_hidden

    def forward(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hazard: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        gru_out, last_hidden = self.forward_features(lidar, speed_input, hidden)
        base_mean = self.actor.output_layer(gru_out)
        base_value = self.base_value_head(gru_out).squeeze(-1)
        hazard_emb = self.hazard_mlp(hazard)
        joint = torch.cat([gru_out, hazard_emb], dim=-1)
        mean = base_mean + self.delta_actor(joint)
        value = base_value + self.delta_value(joint).squeeze(-1)
        std = self.log_std.exp().view(1, 1, -1).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist, value, last_hidden

    def evaluate_actions(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hazard: torch.Tensor,
        hidden: Optional[torch.Tensor],
        raw_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value, _ = self.forward(lidar, speed_input, hazard, hidden)
        logp = dist.log_prob(raw_actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value

    def act(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hazard: torch.Tensor,
        hidden: Optional[torch.Tensor],
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value, new_hidden = self.forward(lidar, speed_input, hazard, hidden)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value, new_hidden

    def save_actor_backbone(self, path: str) -> None:
        # Actor-only v2 checkpoint is not deployment-compatible, but the method is
        # useful for single-agent smoke checks of the shared backbone.
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def save_full_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Any,
        ppo_config: Dict[str, Any],
        reward_config: Dict[str, Any],
        stage: int,
        iteration: int,
        metrics: Dict[str, Any],
    ) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "version": "ppo_v2_hazard",
                "mode": "safety_augmented",
                "actor_critic": self.state_dict(),
                "actor": self.actor.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "hidden_scale": self.actor.hidden_scale,
                "log_std": self.log_std.detach().cpu(),
                "ppo_config": ppo_config,
                "reward_config": reward_config,
                "stage": stage,
                "iteration": iteration,
                "metrics": metrics,
            },
            path,
        )


def load_actor_critic_checkpoint(ac_module: nn.Module, path: str, device: torch.device) -> Dict[str, Any]:
    """Load a plain actor state dict, actor dict, or full PPO checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta: Dict[str, Any] = {}
    if isinstance(ckpt, dict) and "actor_critic" in ckpt:
        ac_module.load_state_dict(ckpt["actor_critic"])
        meta = {k: v for k, v in ckpt.items() if k != "actor_critic"}
    elif isinstance(ckpt, dict) and "actor" in ckpt and hasattr(ac_module, "actor"):
        ac_module.actor.load_state_dict(ckpt["actor"])
        meta = ckpt
    elif hasattr(ac_module, "actor"):
        ac_module.actor.load_state_dict(ckpt)
    else:
        ac_module.load_state_dict(ckpt)
    return meta


def load_end2race_actor(model: End2Race, path: str, device: torch.device) -> Dict[str, Any]:
    """Load a plain End2Race actor from a plain state dict or PPO checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "actor" in ckpt:
        model.load_state_dict(ckpt["actor"])
        return ckpt
    if isinstance(ckpt, dict) and "actor_critic" in ckpt:
        actor_state = {
            key[len("actor.") :]: value
            for key, value in ckpt["actor_critic"].items()
            if key.startswith("actor.")
        }
        if not actor_state:
            raise KeyError(f"No actor weights found in PPO checkpoint: {path}")
        model.load_state_dict(actor_state)
        return ckpt
    model.load_state_dict(ckpt)
    return {}


def build_policy_from_checkpoint(
    mode: str,
    model_path: str,
    device: torch.device,
    hidden_scale: int = 4,
) -> Tuple[nn.Module, bool]:
    """Build a v1 or v2 PPO policy from any supported checkpoint format.

    Supported inputs:
      - plain End2Race actor state_dict, e.g. pretrained/end2race.pth
      - v1 actor-only state_dict, e.g. pretrained/end2race_ppo.pth
      - v1 full PPO checkpoint, e.g. pretrained/end2race_ppo_full.pt
      - v2 full PPO checkpoint, e.g. pretrained/end2race_ppo_aug.pt
    """
    use_hazard = mode == "safety_augmented"
    if use_hazard:
        inner = End2RaceActorCritic(hidden_scale=hidden_scale).to(device)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("mode") == "safety_augmented" and "actor_critic" in ckpt:
            ac = End2RaceHazardActorCritic(inner).to(device)
            ac.load_state_dict(ckpt["actor_critic"])
        else:
            load_actor_critic_checkpoint(inner, model_path, device)
            ac = End2RaceHazardActorCritic(inner).to(device)
    else:
        ac = End2RaceActorCritic(hidden_scale=hidden_scale).to(device)
        load_actor_critic_checkpoint(ac, model_path, device)
    ac.eval()
    return ac, use_hazard


def parse_csv_strings(value: str) -> List[str]:
    if value is None or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(item) for item in parse_csv_strings(value)]


def parse_csv_floats(value: str) -> List[float]:
    return [float(item) for item in parse_csv_strings(value)]



# -----------------------------------------------------------------------------
# Reward and hazard logic
# -----------------------------------------------------------------------------


def _require_project_point() -> Any:
    if project_point_to_centerline is None:
        raise ImportError("latticeplanner.utils.project_point_to_centerline is required for PPO rewards")
    return project_point_to_centerline


def centerline_arc_length(centerline: np.ndarray) -> float:
    if len(centerline) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1)))


def wrap_rel_s(delta_s: float, track_length: float) -> float:
    if track_length <= 0:
        return float(delta_s)
    return float((delta_s + track_length / 2.0) % track_length - track_length / 2.0)


def advance_progress(
    p_new_raw: float,
    p_last: float,
    initial_p: float,
    track_length: float,
) -> Tuple[float, float]:
    """Unwrap progress along a lap so deltas remain smooth near wrap-around."""
    p = float(p_new_raw)
    if track_length > 0 and p < initial_p - track_length / 2.0:
        p += track_length
    delta = p - p_last
    if track_length > 0:
        if delta < -track_length / 2.0:
            p += track_length
            delta = p - p_last
        elif delta > track_length / 2.0:
            p -= track_length
            delta = p - p_last
    return p, float(delta)


@dataclass
class RewardWeights:
    w_progress: float = 1.0
    w_rel_progress: float = 0.4
    w_follow_stable: float = 0.01
    w_tailgate: float = 2.0
    w_side: float = 3.0
    w_rear: float = 3.0
    w_merge_back: float = 2.0
    w_obstacle_clearance: float = 2.0
    w_smooth: float = 0.02
    w_steer_mag: float = 0.005
    w_aggressive_steer: float = 1.0
    w_ttc: float = 1.0
    w_collision: float = 100.0
    w_overtake_success: float = 20.0
    w_timeout: float = 5.0
    side_s_thresh: float = 0.5
    side_dist_thresh: float = 1.5
    rear_s_thresh: float = 1.5
    rear_dist_thresh: float = 2.0
    front_s_thresh: float = 1.5
    front_too_close_s: float = 0.8
    overtake_margin_s: float = 0.5
    ttc_thresh: float = 0.5
    obstacle_clearance_thresh: float = 0.3
    steer_safe_when_beside: float = 0.15
    severe_ttc: float = 0.15
    safe_overtake_hold_duration: float = 0.75


@dataclass
class RewardState:
    track_length: float
    initial_ego_s: float
    initial_opp_s: float
    last_ego_s: float
    last_opp_s: float
    was_ahead: bool = False
    overtake_started: bool = False
    safe_overtake_hold_time: float = 0.0
    safe_overtake_held: bool = False
    severe_unsafe: bool = False
    min_side_gap: float = field(default_factory=lambda: float("inf"))
    min_rear_gap: float = field(default_factory=lambda: float("inf"))
    post_overtake_collision: bool = False
    had_safe_overtake_bonus: bool = False

    @classmethod
    def from_obs(cls, obs: Dict[str, Any], centerline: np.ndarray, scenario: Dict[str, Any]) -> "RewardState":
        del scenario
        project = _require_project_point()
        ego_xy = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
        opp_xy = np.array([obs["poses_x"][1], obs["poses_y"][1]], dtype=np.float64)
        s_e0, _ = project(ego_xy, centerline)
        s_o0, _ = project(opp_xy, centerline)
        se = float(s_e0)
        so = float(s_o0)
        return cls(
            track_length=centerline_arc_length(centerline),
            initial_ego_s=se,
            initial_opp_s=so,
            last_ego_s=se,
            last_opp_s=so,
        )


def build_hazard(obs: Dict[str, Any], centerline: np.ndarray, rw: RewardWeights) -> np.ndarray:
    project = _require_project_point()
    ego_xy = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
    opp_xy = np.array([obs["poses_x"][1], obs["poses_y"][1]], dtype=np.float64)
    s_e, _ = project(ego_xy, centerline)
    s_o, _ = project(opp_xy, centerline)
    tl = centerline_arc_length(centerline)
    rel_s = wrap_rel_s(float(s_e - s_o), tl)

    th_ego = float(obs["poses_theta"][0])
    x_ego, y_ego = float(obs["poses_x"][0]), float(obs["poses_y"][0])
    x_opp, y_opp = float(obs["poses_x"][1]), float(obs["poses_y"][1])
    dx, dy = x_opp - x_ego, y_opp - y_ego
    c, s = np.cos(th_ego), np.sin(th_ego)
    rel_y_ego = float(-s * dx + c * dy)

    v_ego = float(obs["linear_vels_x"][0])
    v_opp = float(obs["linear_vels_x"][1])
    rel_v = v_opp - v_ego
    rel_dist = float(np.hypot(dx, dy))

    side_flag = float(abs(rel_s) < rw.side_s_thresh and rel_dist < rw.side_dist_thresh)
    rear_close_flag = float(0.0 < rel_s < rw.rear_s_thresh and rel_dist < rw.rear_dist_thresh)
    front_close_flag = float(rel_s < 0.0 and abs(rel_s) < rw.front_s_thresh)
    return np.array(
        [
            np.clip(rel_s / 5.0, -5.0, 5.0),
            np.clip(rel_y_ego / 2.0, -5.0, 5.0),
            np.clip(rel_dist / 5.0, 0.0, 5.0),
            np.clip(rel_v / 5.0, -5.0, 5.0),
            side_flag,
            rear_close_flag,
            front_close_flag,
        ],
        dtype=np.float32,
    )


def compute_shaped_reward(
    prev_obs: Dict[str, Any],
    next_obs: Dict[str, Any],
    exec_action: np.ndarray,
    raw_action: np.ndarray,
    prev_exec_action: np.ndarray,
    state: RewardState,
    centerline: np.ndarray,
    weights: RewardWeights,
    terminate_on_success: bool,
    timeout: bool,
    collision_any: bool,
    sim_timestep: float,
) -> Tuple[float, Dict[str, float]]:
    del prev_obs, raw_action
    project = _require_project_point()
    rw = weights

    ego_xy_n = np.array([next_obs["poses_x"][0], next_obs["poses_y"][0]], dtype=np.float64)
    opp_xy_n = np.array([next_obs["poses_x"][1], next_obs["poses_y"][1]], dtype=np.float64)
    s_e_raw, _ = project(ego_xy_n, centerline)
    s_o_raw, _ = project(opp_xy_n, centerline)

    tl = state.track_length if state.track_length > 0 else centerline_arc_length(centerline)
    se_old = state.last_ego_s
    so_old = state.last_opp_s
    rel_s_old = wrap_rel_s(se_old - so_old, tl)

    s_e, d_se = advance_progress(s_e_raw, se_old, state.initial_ego_s, tl)
    s_o, _d_so = advance_progress(s_o_raw, so_old, state.initial_opp_s, tl)
    state.last_ego_s = s_e
    state.last_opp_s = s_o

    rel_s = wrap_rel_s(s_e - s_o, tl)
    delta_rel_s = wrap_rel_s(rel_s - rel_s_old, tl)

    dx = float(next_obs["poses_x"][1] - next_obs["poses_x"][0])
    dy = float(next_obs["poses_y"][1] - next_obs["poses_y"][0])
    rel_dist = float(np.hypot(dx, dy))
    v_e = float(next_obs["linear_vels_x"][0])
    v_o = float(next_obs["linear_vels_x"][1])

    side_hazard = abs(rel_s) < rw.side_s_thresh and rel_dist < rw.side_dist_thresh
    rear_hazard = 0.0 < rel_s < rw.rear_s_thresh and rel_dist < rw.rear_dist_thresh
    side_risk = max(0.0, rw.side_dist_thresh - rel_dist) / max(rw.side_dist_thresh, 1e-6)
    rear_risk = max(0.0, rw.rear_s_thresh - rel_s) / max(rw.rear_s_thresh, 1e-6)

    scan = np.asarray(next_obs["scans"][0], dtype=np.float32).flatten()
    min_lidar = float(np.min(scan)) if scan.size else 30.0
    steer = float(exec_action[0])
    merge_back_indicator = float((rear_hazard or side_hazard) and abs(steer) > 0.05)

    r = 0.0
    terms: Dict[str, float] = {}

    prog = rw.w_progress * float(np.clip(d_se, -0.05, 0.08))
    r += prog
    terms["progress"] = prog

    if not side_hazard and not rear_hazard:
        part = rw.w_rel_progress * float(np.clip(delta_rel_s, -0.05, 0.08))
        r += part
        terms["rel_progress"] = part
    else:
        terms["rel_progress"] = 0.0

    if rel_s < 0.0:
        if abs(rel_s) < rw.front_too_close_s:
            part = -rw.w_tailgate * (rw.front_too_close_s - abs(rel_s))
            r += part
            terms["tailgate"] = float(part)
            terms["follow_stable"] = 0.0
        else:
            part = rw.w_follow_stable
            r += part
            terms["follow_stable"] = float(part)
            terms["tailgate"] = 0.0
    else:
        terms["tailgate"] = 0.0
        terms["follow_stable"] = 0.0

    if side_hazard:
        part = -rw.w_side * side_risk - rw.w_aggressive_steer * max(0.0, abs(steer) - rw.steer_safe_when_beside)
        r += part
        terms["side"] = float(part)
        state.min_side_gap = min(state.min_side_gap, rel_dist)
    else:
        terms["side"] = 0.0

    if rear_hazard:
        part = -rw.w_rear * rear_risk - rw.w_merge_back * merge_back_indicator
        r += part
        terms["rear"] = float(part)
        state.min_rear_gap = min(state.min_rear_gap, rel_dist)
    else:
        terms["rear"] = 0.0

    if min_lidar < rw.obstacle_clearance_thresh:
        part = -rw.w_obstacle_clearance * (rw.obstacle_clearance_thresh - min_lidar)
        r += part
        terms["obstacle_clearance"] = float(part)
    else:
        terms["obstacle_clearance"] = 0.0

    smooth_pen = rw.w_smooth * (
        float(np.square(steer - float(prev_exec_action[0])))
        + float(np.square(float(exec_action[1]) - float(prev_exec_action[1])))
    )
    steer_pen = rw.w_steer_mag * steer * steer
    r -= smooth_pen + steer_pen
    terms["smooth"] = float(-smooth_pen)
    terms["steer_mag"] = float(-steer_pen)

    rear_ttc = None
    front_ttc = None
    if rel_s > 0.0:
        rear_ttc = rel_s / max(max(0.0, v_o - v_e), 1e-3)
    if rel_s < 0.0:
        front_ttc = abs(rel_s) / max(max(0.0, v_e - v_o), 1e-3)

    ttc_pen = 0.0
    if rear_ttc is not None and rear_ttc < rw.ttc_thresh:
        ttc_pen += rw.w_ttc * (rw.ttc_thresh - rear_ttc) / max(rw.ttc_thresh, 1e-6)
    if front_ttc is not None and front_ttc < rw.ttc_thresh:
        ttc_pen += rw.w_ttc * (rw.ttc_thresh - front_ttc) / max(rw.ttc_thresh, 1e-6)
    r -= ttc_pen
    terms["ttc"] = float(-ttc_pen)

    state.severe_unsafe = False
    if rear_ttc is not None and rear_ttc < rw.severe_ttc:
        state.severe_unsafe = True
    if front_ttc is not None and front_ttc < rw.severe_ttc:
        state.severe_unsafe = True

    ahead_now = rel_s > rw.overtake_margin_s  # wrapped progress, safe near lap boundary
    if ahead_now and not state.was_ahead:
        state.overtake_started = True
    state.was_ahead = ahead_now

    safe_sep = not side_hazard and not rear_hazard
    if state.overtake_started and ahead_now and safe_sep:
        state.safe_overtake_hold_time += sim_timestep
        if state.safe_overtake_hold_time >= rw.safe_overtake_hold_duration:
            state.safe_overtake_held = True
    else:
        state.safe_overtake_hold_time = 0.0

    if collision_any:
        if state.overtake_started or ahead_now:
            state.post_overtake_collision = True
        r -= rw.w_collision
        terms["collision"] = float(-rw.w_collision)
    else:
        terms["collision"] = 0.0

    if terminate_on_success and state.safe_overtake_held and not state.had_safe_overtake_bonus:
        r += rw.w_overtake_success
        terms["overtake_success"] = float(rw.w_overtake_success)
        state.had_safe_overtake_bonus = True
    else:
        terms["overtake_success"] = 0.0

    if timeout:
        r -= rw.w_timeout
        terms["timeout"] = float(-rw.w_timeout)
    else:
        terms["timeout"] = 0.0

    terms["total"] = float(r)
    terms["rel_s"] = float(rel_s)
    terms["min_lidar"] = float(min_lidar)
    return float(r), terms


# -----------------------------------------------------------------------------
# Scenario helpers and PPO environment wrapper
# -----------------------------------------------------------------------------


def _require_gym() -> Any:
    global Integrator
    if gym is None:
        raise ImportError("gym is required for End2RacePPOEnv")
    if Integrator is None:
        import f110_gym  # noqa: F401 - registers f110-v0 lazily
        from f110_gym.envs.base_classes import Integrator as _Integrator

        Integrator = _Integrator
    return gym


def _require_planner_imports() -> Tuple[Any, Any, Any, Any]:
    if load_centerline_from_map is None or obsDict2oppoArray is None:
        raise ImportError("latticeplanner.utils is required for End2RacePPOEnv")
    from demonstration import setup_opp_planner
    from utils import find_corresponding_waypoint, load_positions_and_speeds_from_params

    return setup_opp_planner, find_corresponding_waypoint, load_positions_and_speeds_from_params, obsDict2oppoArray


def load_centerline_for_map(map_name: str) -> np.ndarray:
    if load_centerline_from_map is None:
        raise ImportError("latticeplanner.utils.load_centerline_from_map is required")
    map_directory = os.path.join("f1tenth_racetracks", map_name)
    return load_centerline_from_map(map_directory)


def downsample_for_eval_compat(lidar: np.ndarray, target_points: int = 360) -> np.ndarray:
    lidar = np.asarray(lidar, dtype=np.float32).flatten()
    if len(lidar) > target_points:
        indices = np.linspace(0, len(lidar) - 1, target_points, dtype=int)
        lidar = lidar[indices]
    return lidar.astype(np.float32)


def _load_waypoints_csv(csv_path: str) -> np.ndarray:
    with open(csv_path, "r") as f:
        lines = f.readlines()[1:]
    wps = []
    for line in lines:
        parts = line.strip().split(";")
        if len(parts) >= 6:
            wps.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])])
    if not wps:
        raise ValueError(f"No waypoints parsed from {csv_path}")
    return np.asarray(wps, dtype=np.float64)


def compute_opp_idx_like_eval_multiagent(
    map_name: str,
    ego_raceline: str,
    opp_raceline: str,
    ego_idx: int,
    interval_idx: int,
) -> int:
    _, find_corresponding_waypoint, _, _ = _require_planner_imports()
    base_path = os.path.join("f1tenth_racetracks", map_name)
    ego_wps = _load_waypoints_csv(os.path.join(base_path, f"{ego_raceline}.csv"))
    if opp_raceline != ego_raceline:
        opp_wps = _load_waypoints_csv(os.path.join(base_path, f"{opp_raceline}.csv"))
        ego_wp = ego_wps[ego_idx % len(ego_wps)]
        ego_map_idx = find_corresponding_waypoint(ego_wp, opp_wps)
        return int((ego_map_idx + interval_idx) % len(opp_wps))
    return int((ego_idx + interval_idx) % len(ego_wps))


def sample_ego_idx(rng: np.random.Generator, map_name: str, ego_raceline: str) -> int:
    wps = _load_waypoints_csv(os.path.join("f1tenth_racetracks", map_name, f"{ego_raceline}.csv"))
    return int(rng.integers(0, len(wps)))


def sample_opp_speedscale(stage: int, rng: np.random.Generator) -> float:
    if stage <= 1:
        return float(rng.uniform(0.4, 0.6))
    if stage == 2:
        return float(rng.uniform(0.4, 0.7))
    if stage == 3:
        return float(rng.uniform(0.3, 0.6))
    return float(rng.uniform(0.3, 0.8))


def sample_scenario(
    stage: int,
    rng: np.random.Generator,
    map_name: str,
    ego_raceline_choices: Sequence[str],
    opp_raceline_choices: Sequence[str],
) -> Dict[str, Any]:
    ego_raceline = str(rng.choice(list(ego_raceline_choices)))
    opp_raceline = str(rng.choice(list(opp_raceline_choices)))
    ego_idx = sample_ego_idx(rng, map_name, ego_raceline)
    if stage == 1:
        interval_idx = int(rng.integers(low=10, high=40))
    elif stage == 2:
        interval_idx = int(rng.choice([-5, -3, -1, 0, 1, 3, 5]))
    elif stage == 3:
        interval_idx = int(rng.integers(low=8, high=30))
    else:
        interval_idx = int(rng.integers(low=-10, high=50))
    opp_idx = compute_opp_idx_like_eval_multiagent(map_name, ego_raceline, opp_raceline, ego_idx, interval_idx)
    return {
        "map_name": map_name,
        "ego_raceline": ego_raceline,
        "opp_raceline": opp_raceline,
        "ego_idx": int(ego_idx),
        "interval_idx": int(interval_idx),
        "opp_idx": int(opp_idx),
        "opp_speedscale": sample_opp_speedscale(stage, rng),
    }


_BaseEnv = object if gym is None else gym.Env


class End2RacePPOEnv(_BaseEnv):
    """PPO-friendly two-agent F110 wrapper.

    ``step`` accepts raw Gaussian actions.  The wrapper clips them internally
    before sending actions to the simulator, while logging the clipped fraction.
    """

    metadata = {"render.modes": ["human", "human_fast", "rgb_array"]}

    def __init__(
        self,
        map_name: str,
        mode: str = "compatibility",
        max_speed: float = 20.0,
        sim_duration: float = 8.0,
        render: bool = False,
        terminate_on_success: bool = True,
        terminate_on_severe_unsafe: bool = False,
        integrator: Any = None,
        seed: int = 0,
        reward_weights: Optional[RewardWeights] = None,
        ego_raceline_choices: Optional[Sequence[str]] = None,
        opp_raceline_choices: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        gym_mod = _require_gym()
        if integrator is None:
            integrator = Integrator.RK4
        if mode not in ("compatibility", "safety_augmented"):
            raise ValueError(f"Unknown PPO mode: {mode}")
        self.map_name = map_name
        self.mode = mode
        self.max_speed = float(max_speed)
        self.sim_duration = float(sim_duration)
        self._render_flag = bool(render)
        self.terminate_on_success = bool(terminate_on_success)
        self.terminate_on_severe_unsafe = bool(terminate_on_severe_unsafe)
        self.reward_weights = reward_weights or RewardWeights()

        self.env = gym_mod.make(
            "f110-v0",
            map=f"f1tenth_racetracks/{map_name}/{map_name}_map",
            map_ext=".png",
            num_agents=2,
            timestep=0.01,
            integrator=integrator,
        )

        spaces = {
            "lidar": gym_mod.spaces.Box(low=0.0, high=30.0, shape=(360,), dtype=np.float32),
            "prev_speed": gym_mod.spaces.Box(low=-5.0, high=20.0, shape=(1,), dtype=np.float32),
        }
        if mode == "safety_augmented":
            spaces["hazard"] = gym_mod.spaces.Box(low=-5.0, high=5.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym_mod.spaces.Dict(spaces)
        self.raw_action_space = gym_mod.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.exec_action_space = gym_mod.spaces.Box(
            low=np.array([-0.52, 0.0], dtype=np.float32),
            high=np.array([0.52, self.max_speed], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = self.raw_action_space

        self.rng = np.random.default_rng(seed)
        self.stage = 0
        self.ego_raceline_choices = list(ego_raceline_choices or ["raceline0", "raceline1", "raceline2"])
        self.opp_raceline_choices = list(opp_raceline_choices or ["raceline0", "raceline1", "raceline2"])
        self.opp_speedscale = 0.5
        self.centerline = load_centerline_for_map(map_name)

        self.opponent: Any = None
        self.opp_raceline: Optional[str] = None
        self.tracker_steps = 10
        self._prev_speed = 0.0
        self._prev_obs: Optional[Dict[str, Any]] = None
        self._prev_exec_action = np.zeros(2, dtype=np.float32)
        self._reward_state: Optional[RewardState] = None
        self._t = 0.0
        self._tracker_count = 0
        self._opp_traj: Any = None
        self.params: Dict[str, Any] = {}

    def _configure_opp_raceline(self, opp_raceline: str) -> None:
        setup_opp_planner, _, _, _ = _require_planner_imports()
        if self.opponent is None or self.opp_raceline != opp_raceline:
            self.opponent = setup_opp_planner(self.map_name, opp_raceline)
            self.opp_raceline = opp_raceline
            self.tracker_steps = int(getattr(self.opponent.conf, "tracker_steps", 10))

    def _build_obs(self, raw_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        lidar = downsample_for_eval_compat(raw_obs["scans"][0], target_points=360)
        lidar = np.clip(lidar, 0.0, 30.0).astype(np.float32)
        out: Dict[str, np.ndarray] = {
            "lidar": lidar,
            "prev_speed": np.array([self._prev_speed], dtype=np.float32),
        }
        if self.mode == "safety_augmented":
            out["hazard"] = build_hazard(raw_obs, self.centerline, self.reward_weights)
        return out

    def reset(self, scenario: Optional[Dict[str, Any]] = None):
        if scenario is None:
            scenario = sample_scenario(
                self.stage,
                self.rng,
                self.map_name,
                self.ego_raceline_choices,
                self.opp_raceline_choices,
            )
        self.params = scenario
        self.opp_speedscale = float(scenario["opp_speedscale"])
        self._configure_opp_raceline(str(scenario["opp_raceline"]))
        _, _, load_positions_and_speeds_from_params, _ = _require_planner_imports()
        params_lc = {
            "ego_raceline": scenario["ego_raceline"],
            "opp_raceline": scenario["opp_raceline"],
            "ego_idx": int(scenario["ego_idx"]),
            "opp_idx": int(scenario["opp_idx"]),
        }
        positions, initial_speeds = load_positions_and_speeds_from_params(params_lc, self.map_name)
        obs, _, _, _ = self.env.reset(poses=positions)
        self._prev_speed = float(initial_speeds[0] * 0.9)
        self._prev_obs = obs
        self._prev_exec_action = np.zeros(2, dtype=np.float32)
        self._reward_state = RewardState.from_obs(obs, self.centerline, scenario)
        self._t = 0.0
        self._tracker_count = 0
        self._opp_traj = None
        return self._build_obs(obs)

    def _opp_planner_step(self, prev_obs: Dict[str, Any]) -> np.ndarray:
        assert self.opponent is not None
        if obsDict2oppoArray is None:
            raise ImportError("latticeplanner.utils.obsDict2oppoArray is required")
        if self._tracker_count == 0:
            opp_poses = obsDict2oppoArray(prev_obs, 1)
            self._opp_traj = self.opponent.plan(
                prev_obs["poses_x"][1],
                prev_obs["poses_y"][1],
                prev_obs["poses_theta"][1],
                opp_poses,
                prev_obs["linear_vels_x"][1],
            )
        opp_steer, opp_speed = self.opponent.tracker.plan(
            prev_obs["poses_x"][1],
            prev_obs["poses_y"][1],
            prev_obs["poses_theta"][1],
            prev_obs["linear_vels_x"][1],
            self._opp_traj,
        )
        opp_steer = float(np.clip(opp_steer, -0.52, 0.52))
        opp_speed = float(opp_speed * self.opp_speedscale)
        self._tracker_count = (self._tracker_count + 1) % max(self.tracker_steps, 1)
        return np.array([opp_steer, opp_speed], dtype=np.float32)

    def _is_done(self, next_obs: Dict[str, Any], env_done: bool) -> Tuple[bool, bool]:
        assert self._reward_state is not None
        terminated = bool(
            env_done
            or np.any(next_obs["collisions"])
            or (self.terminate_on_success and self._reward_state.safe_overtake_held)
            or (self.terminate_on_severe_unsafe and self._reward_state.severe_unsafe)
        )
        truncated = bool(self._t >= self.sim_duration) and not terminated
        return terminated, truncated

    def step(self, raw_ego_action: np.ndarray):
        assert self._prev_obs is not None and self._reward_state is not None
        raw_ego_action = np.asarray(raw_ego_action, dtype=np.float32).reshape(-1)
        if raw_ego_action.shape[0] != 2:
            raise ValueError(f"Expected raw ego action shape (2,), got {raw_ego_action.shape}")
        exec_ego_action = np.array(
            [
                np.clip(raw_ego_action[0], -0.52, 0.52),
                np.clip(raw_ego_action[1], 0.0, self.max_speed),
            ],
            dtype=np.float32,
        )
        clipped = bool(np.any(np.abs(raw_ego_action - exec_ego_action) > 1e-7))
        opp_action = self._opp_planner_step(self._prev_obs)
        action = np.array([exec_ego_action, opp_action], dtype=np.float32)
        next_obs, _, env_done, env_info = self.env.step(action)
        self._t += float(self.env.timestep)
        collisions = np.asarray(next_obs.get("collisions", np.zeros(2)), dtype=np.float32)
        collision_any = bool(np.any(collisions))
        timeout = bool(self._t >= self.sim_duration)
        reward, reward_terms = compute_shaped_reward(
            prev_obs=self._prev_obs,
            next_obs=next_obs,
            exec_action=exec_ego_action,
            raw_action=raw_ego_action,
            prev_exec_action=self._prev_exec_action,
            state=self._reward_state,
            centerline=self.centerline,
            weights=self.reward_weights,
            terminate_on_success=self.terminate_on_success,
            timeout=timeout,
            collision_any=collision_any,
            sim_timestep=float(self.env.timestep),
        )
        terminated, truncated = self._is_done(next_obs, env_done)
        done = terminated or truncated
        info = {
            **env_info,
            **reward_terms,
            "raw_action": raw_ego_action.copy(),
            "exec_action": exec_ego_action.copy(),
            "action_was_clipped": clipped,
            "ego_collision": bool(collisions[0]) if len(collisions) > 0 else False,
            "opp_collision": bool(collisions[1]) if len(collisions) > 1 else False,
            "any_collision": collision_any,
            "safe_overtake_held": bool(self._reward_state.safe_overtake_held),
            "severe_unsafe": bool(self._reward_state.severe_unsafe),
            "terminated": terminated,
            "truncated": truncated,
        }
        self._prev_speed = float(next_obs["linear_vels_x"][0])
        self._prev_obs = next_obs
        self._prev_exec_action = exec_ego_action.copy()
        return self._build_obs(next_obs), float(reward), done, info

    def render(self, mode: str = "human"):
        return self.env.render(mode=mode)

    def close(self) -> None:
        self.env.close()


# -----------------------------------------------------------------------------
# Recurrent rollout buffer and sequence helpers
# -----------------------------------------------------------------------------


class RolloutBuffer:
    """Rollout storage and GAE for one serial worker.

    This buffer intentionally supports the simple serial PPO path first.  The
    data are replayed as one recurrent sequence with hidden resets at episode
    starts.  ``recurrent_minibatches`` additionally supports chunked updates for
    longer rollouts.
    """

    def __init__(self, rollout_steps: int, gamma: float = 0.997, gae_lambda: float = 0.95, hazard_dim: int = 0) -> None:
        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.hazard_dim = int(hazard_dim)
        self.lidar = np.zeros((self.rollout_steps, 360), dtype=np.float32)
        self.prev_speed = np.zeros((self.rollout_steps, 1), dtype=np.float32)
        self.hazard = np.zeros((self.rollout_steps, self.hazard_dim), dtype=np.float32) if self.hazard_dim > 0 else None
        self.raw_actions = np.zeros((self.rollout_steps, 2), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.values = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.log_probs = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.terminateds = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.truncateds = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.next_values_at_trunc = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.episode_starts = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.advantages = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.returns = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.ptr = 0

    def reset_ptr(self) -> None:
        self.ptr = 0

    def add(
        self,
        lidar: np.ndarray,
        prev_speed: float,
        hazard: Optional[np.ndarray],
        raw_action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        terminated: bool,
        truncated: bool,
        next_value_at_trunc: float,
        episode_start: bool,
    ) -> None:
        i = self.ptr
        if i >= self.rollout_steps:
            raise IndexError("RolloutBuffer overflow")
        self.lidar[i] = np.asarray(lidar, dtype=np.float32)
        self.prev_speed[i, 0] = float(prev_speed)
        if self.hazard is not None:
            if hazard is None:
                raise ValueError("hazard_dim > 0 but hazard was None")
            self.hazard[i] = np.asarray(hazard, dtype=np.float32)
        self.raw_actions[i] = np.asarray(raw_action, dtype=np.float32)
        self.rewards[i] = float(reward)
        self.values[i] = float(value)
        self.log_probs[i] = float(log_prob)
        self.terminateds[i] = 1.0 if terminated else 0.0
        self.truncateds[i] = 1.0 if truncated else 0.0
        self.next_values_at_trunc[i] = float(next_value_at_trunc)
        self.episode_starts[i] = 1.0 if episode_start else 0.0
        self.ptr += 1

    def compute_returns_and_advantage(
        self,
        last_value: float,
        last_terminated: bool,
        last_truncated: bool,
    ) -> None:
        """GAE that distinguishes terminations (cut bootstrap) from truncations
        (bootstrap V(s') because the episode was time-cut, not task-ended).

        In both cases the GAE chain is reset across the episode boundary
        because the next sample comes from a different trajectory.
        """
        if self.ptr != self.rollout_steps:
            raise RuntimeError(f"RolloutBuffer has {self.ptr}/{self.rollout_steps} steps")
        gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_value = 0.0 if last_terminated else float(last_value)
                episode_boundary = bool(last_terminated or last_truncated)
            elif self.truncateds[t] > 0.5:
                next_value = float(self.next_values_at_trunc[t])
                episode_boundary = True
            elif self.terminateds[t] > 0.5:
                next_value = 0.0
                episode_boundary = True
            else:
                next_value = float(self.values[t + 1])
                episode_boundary = False

            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            if episode_boundary:
                gae = delta
            else:
                gae = delta + self.gamma * self.gae_lambda * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def _slice_tensors(self, device: torch.device, start: int, end: int) -> Tuple[torch.Tensor, ...]:
        lidar = torch.from_numpy(self.lidar[start:end]).unsqueeze(0).to(device)
        spd = torch.from_numpy(self.prev_speed[start:end]).unsqueeze(0).to(device)
        act = torch.from_numpy(self.raw_actions[start:end]).unsqueeze(0).to(device)
        old_logp = torch.from_numpy(self.log_probs[start:end]).unsqueeze(0).to(device)
        adv = torch.from_numpy(self.advantages[start:end]).unsqueeze(0).to(device)
        ret = torch.from_numpy(self.returns[start:end]).unsqueeze(0).to(device)
        old_values = torch.from_numpy(self.values[start:end]).unsqueeze(0).to(device)
        starts = torch.from_numpy(self.episode_starts[start:end]).unsqueeze(0).to(device)
        if start == 0:
            starts[:, 0] = 1.0
        if self.hazard is not None:
            haz = torch.from_numpy(self.hazard[start:end]).unsqueeze(0).to(device)
        else:
            haz = None
        return lidar, spd, haz, act, old_logp, adv, ret, old_values, starts

    def full_batch_tensors(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        return self._slice_tensors(device, 0, self.rollout_steps)

    def recurrent_minibatches(self, device: torch.device, chunk_len: int = 0, shuffle: bool = True) -> Iterable[Tuple[torch.Tensor, ...]]:
        if chunk_len <= 0 or chunk_len >= self.rollout_steps:
            yield self.full_batch_tensors(device)
            return
        starts = list(range(0, self.rollout_steps, chunk_len))
        if shuffle:
            np.random.shuffle(starts)
        for s in starts:
            e = min(s + chunk_len, self.rollout_steps)
            yield self._slice_tensors(device, s, e)


@torch.no_grad()
def forward_frozen_bc_sequence(
    frozen_bc: End2Race,
    lidar_b: torch.Tensor,
    spd_b: torch.Tensor,
    episode_starts_b: torch.Tensor,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Replay frozen BC with an independent recurrent state and episode resets."""
    hidden = torch.zeros(1, lidar_b.shape[0], hidden_size, device=device)
    actions: List[torch.Tensor] = []
    for t in range(lidar_b.shape[1]):
        if episode_starts_b[0, t].item() > 0.5:
            hidden = torch.zeros_like(hidden)
        action_t, hidden = frozen_bc(lidar_b[:, t : t + 1], spd_b[:, t : t + 1], hidden)
        actions.append(action_t)
    return torch.cat(actions, dim=1)


def forward_policy_sequence(
    ac: nn.Module,
    lidar_b: torch.Tensor,
    spd_b: torch.Tensor,
    haz_b: Optional[torch.Tensor],
    episode_starts_b: torch.Tensor,
    hidden_size: int,
    use_hazard: bool,
    device: torch.device,
) -> Tuple[torch.distributions.Normal, torch.Tensor]:
    """Replay a rollout while resetting recurrent state at episode boundaries."""
    hidden = torch.zeros(1, lidar_b.shape[0], hidden_size, device=device)
    means: List[torch.Tensor] = []
    stds: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    for t in range(lidar_b.shape[1]):
        if episode_starts_b[0, t].item() > 0.5:
            hidden = torch.zeros_like(hidden)
        if use_hazard:
            if haz_b is None:
                raise ValueError("hazard tensor is required in safety_augmented mode")
            dist_t, value_t, hidden = ac.forward(lidar_b[:, t : t + 1], spd_b[:, t : t + 1], haz_b[:, t : t + 1], hidden)
        else:
            dist_t, value_t, hidden = ac.forward(lidar_b[:, t : t + 1], spd_b[:, t : t + 1], hidden)
        means.append(dist_t.mean)
        stds.append(dist_t.stddev)
        values.append(value_t)
    dist = torch.distributions.Normal(torch.cat(means, dim=1), torch.cat(stds, dim=1))
    value = torch.cat(values, dim=1)
    return dist, value


__all__ = [
    "End2RaceActorCritic",
    "End2RaceHazardActorCritic",
    "End2RacePPOEnv",
    "build_policy_from_checkpoint",
    "RewardWeights",
    "RewardState",
    "RolloutBuffer",
    "advance_progress",
    "build_hazard",
    "centerline_arc_length",
    "compute_opp_idx_like_eval_multiagent",
    "compute_shaped_reward",
    "downsample_for_eval_compat",
    "forward_frozen_bc_sequence",
    "forward_policy_sequence",
    "load_actor_critic_checkpoint",
    "load_end2race_actor",
    "load_centerline_for_map",
    "sample_scenario",
    "parse_csv_strings",
    "parse_csv_ints",
    "parse_csv_floats",
    "project_point_to_centerline",
    "wrap_rel_s",
]

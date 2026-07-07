"""PPO-specific helpers for End2Race fine-tuning.

This file intentionally contains PPO reward, curriculum, checkpoint, and
recurrent replay helpers. Generic raceline, LiDAR, and track geometry helpers
belong in utils.py. The main PPO environment, rollout buffer, collection loop,
and PPO update live in train_ppo.py.
"""

import os
import torch
import numpy as np
from model import End2Race, End2Race_PPO
from utils import (load_raceline_with_speed, project_to_reference, resolve_two_agent_indices,
                   unwrap_progress, wrap_rel_s)

# ---------------------------------------------------------------------------
# PPO logging keys
# ---------------------------------------------------------------------------
BOOL_INFO_KEYS = (
    "terminated",
    "truncated",
    "collision",
    "opp_collision",
    "timeout",
    "success",
    "action_was_clipped",
)

MEAN_INFO_KEYS = (
    "reward_progress",
    "reward_rel_progress",
    "reward_clearance",
    "reward_collision",
    "reward_overtake_success",
    "reward_closing_potential",
    "delta_ego_s",
    "delta_opp_s",
    "rel_s",
    "rel_progress_potential",
    "delta_rel_potential",
    "delta_rel_potential_gated",
    "ego_d",
    "opp_d",
    "lat_gap",
    "rel_dist",
    "ego_v_s",
    "opp_v_s",
    "clearance_risk",
    "front_risk",
    "rear_risk",
    "side_risk",
    "side_risk_gate",
    "lateral_overlap",
    "required_front_gap",
    "required_rear_gap",
    "overtake_started",
    "started_behind",
    "safe_overtake_hold_time",
    "safe_overtake_held",
    "ego_lat_offset",
)

REQUIRED_V1_REWARD_FIELDS = (
    "w_progress",
    "w_rel_progress",
    "w_clearance_risk",
    "w_collision",
    "w_overtake_success",
)

# ---------------------------------------------------------------------------
# PPO reward configuration and state
# ---------------------------------------------------------------------------
class RewardWeights:
    """Compact v1 reward weights."""

    def __init__(self):
        # The dense terms are frequency-aware: progress terms are measured in
        # meters, while clearance risk is integrated with dt as a per-second penalty.
        self.w_progress = 0.25
        self.w_rel_progress = 3.0
        self.w_clearance_risk = 5.0
        self.w_collision = 120.0
        self.w_overtake_success = 25.0

        self.progress_clip_back = 0.05
        self.progress_clip_forward = 0.12
        self.rel_progress_clip = 0.08
        self.rel_behind_cap = 6.0
        self.rel_ahead_cap = 2.0

        self.overtake_start_margin_s = 0.2
        self.success_margin_s = 2.0
        self.success_clearance_threshold = 0.25
        self.safe_overtake_hold_duration = 0.7

        self.car_length = 0.58
        self.car_width = 0.31
        self.lateral_margin = 0.20
        self.front_base_margin = 0.45
        self.rear_base_margin = 0.60
        self.time_gap = 0.40
        self.side_longitudinal_margin = 0.75

        # D3: potential-based closing-speed shaping, Phi = -k * closing * front_risk.
        # 0 disables (bit-exact legacy behavior). closing_potential_gamma must equal
        # the RL discount for policy invariance; train_ppo syncs it to args.gamma
        # unless explicitly overridden.
        self.w_closing_potential = 0.0
        self.closing_potential_gamma = 0.997

        # D4-A: sharper side-risk scale used ONLY in the positive relative-progress
        # gate (the dense clearance penalty keeps the original side_risk). The base
        # side_risk under-weights near-edge contact: at lat_gap 0.35 (edges ~4 cm
        # apart) it reads only ~0.31, so the gate still leaves ~69% of the squeeze
        # reward. This rescales the gate's lateral overlap over EDGE clearance in
        # [0, side_gate_edge_margin], so a squeeze inside the margin loses its
        # progress reward while a clean pass (edge clearance >= margin) keeps it.
        # 0 disables (bit-exact: gate falls back to the original side_risk).
        self.side_gate_edge_margin = 0.0

class RewardState:

    def __init__(self, last_ego_s, last_opp_s, started_behind, last_closing_phi=0.0):
        self.last_ego_s = last_ego_s
        self.last_opp_s = last_opp_s
        self.started_behind = started_behind
        self.overtake_started = False
        self.safe_overtake_hold_time = 0.0
        self.safe_overtake_held = False
        self.had_safe_overtake_bonus = False
        self.last_closing_phi = float(last_closing_phi)

    @classmethod
    def from_obs(cls, obs, ref, rw=None):
        geom = relative_geometry(obs, ref)
        ego_s, opp_s = geom['ego_s_raw'], geom['opp_s_raw']
        rel_s = wrap_rel_s(ego_s - opp_s, ref.track_length)
        last_closing_phi = 0.0
        if rw is not None and rw.w_closing_potential != 0.0:
            risk = clearance_risk(rel_s, geom['lat_gap'], geom['ego_v_s'], geom['opp_v_s'], rw)
            closing = max(0.0, geom['ego_v_s'] - geom['opp_v_s'])
            last_closing_phi = -rw.w_closing_potential * closing * risk['front_risk']
        return cls(
            last_ego_s=ego_s,
            last_opp_s=opp_s,
            started_behind=rel_s < 0.0,
            last_closing_phi=last_closing_phi,
        )

# ---------------------------------------------------------------------------
# PPO reward helpers
# ---------------------------------------------------------------------------
def rel_progress_potential(rel_s, behind_cap, ahead_cap):
    """Saturated relative-progress potential (rewards catching/passing, caps the lead)."""
    return float(np.clip(rel_s, -behind_cap, ahead_cap))

def relative_geometry(obs, ref):
    """Compute opponent-relative geometry in the reference-line frame."""
    ego_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]], dtype=np.float64)
    opp_pos = np.array([obs['poses_x'][1], obs['poses_y'][1]], dtype=np.float64)

    ego_s, ego_d, ego_ref_theta = project_to_reference(ego_pos, ref)
    opp_s, opp_d, opp_ref_theta = project_to_reference(opp_pos, ref)

    ego_v_s = float(obs['linear_vels_x'][0]) * np.cos(float(obs['poses_theta'][0]) - ego_ref_theta)
    opp_v_s = float(obs['linear_vels_x'][1]) * np.cos(float(obs['poses_theta'][1]) - opp_ref_theta)

    return {
        'ego_s_raw': float(ego_s),
        'opp_s_raw': float(opp_s),
        'ego_d': float(ego_d),
        'opp_d': float(opp_d),
        'lat_gap': float(abs(ego_d - opp_d)),
        'rel_dist': float(np.linalg.norm(opp_pos - ego_pos)),
        'ego_v_s': float(ego_v_s),
        'opp_v_s': float(opp_v_s),
    }

def clearance_risk(rel_s, lat_gap, ego_v_s, opp_v_s, rw):
    """Dynamic opponent clearance risk in Frenet-like coordinates."""
    # front_risk: ego is behind and closing too fast.
    # rear_risk: ego is ahead but leaves insufficient rear gap.
    # side_risk: vehicles are side-by-side and laterally too close.
    lat_safe = rw.car_width + rw.lateral_margin
    lateral_overlap = float(np.clip((lat_safe - lat_gap) / lat_safe, 0.0, 1.0))

    front_risk = 0.0
    front_gap = max(0.0, -rel_s)
    front_closing = max(0.0, ego_v_s - opp_v_s)
    required_front_gap = rw.car_length + rw.front_base_margin + rw.time_gap * front_closing
    if rel_s < 0.0:
        front_risk = lateral_overlap * np.clip(
            (required_front_gap - front_gap) / required_front_gap,
            0.0,
            1.0,
        )

    rear_risk = 0.0
    rear_gap = max(0.0, rel_s)
    rear_closing = max(0.0, opp_v_s - ego_v_s)
    required_rear_gap = rw.car_length + rw.rear_base_margin + rw.time_gap * rear_closing
    if rel_s > 0.0:
        rear_risk = lateral_overlap * np.clip(
            (required_rear_gap - rear_gap) / required_rear_gap,
            0.0,
            1.0,
        )

    side_gap = rw.car_length + rw.side_longitudinal_margin
    longitudinal_overlap = float(np.clip((side_gap - abs(rel_s)) / side_gap, 0.0, 1.0))
    side_risk = longitudinal_overlap * lateral_overlap

    # D4-A gate-only sharper side risk: ramp lateral overlap over edge clearance
    # (lat_gap - car_width) in [0, margin]. Defaults to side_risk when disabled.
    side_risk_gate = side_risk
    if rw.side_gate_edge_margin > 0.0:
        edge_gap = max(0.0, lat_gap - rw.car_width)
        lateral_overlap_edge = float(
            np.clip((rw.side_gate_edge_margin - edge_gap) / rw.side_gate_edge_margin, 0.0, 1.0)
        )
        side_risk_gate = longitudinal_overlap * lateral_overlap_edge

    return {
        'clearance_risk': float(np.clip(max(front_risk, rear_risk, side_risk), 0.0, 1.0)),
        'front_risk': float(front_risk),
        'rear_risk': float(rear_risk),
        'side_risk': float(side_risk),
        'side_risk_gate': float(side_risk_gate),
        'lateral_overlap': float(lateral_overlap),
        'required_front_gap': float(required_front_gap),
        'required_rear_gap': float(required_rear_gap),
    }

def compute_shaped_reward(obs, reward_state, ref, rw, dt):
    """Compute compact PPO reward and update RewardState in place."""
    geom = relative_geometry(obs, ref)
    ego_s, delta_ego_s = unwrap_progress(geom['ego_s_raw'], reward_state.last_ego_s, ref.track_length)
    opp_s, delta_opp_s = unwrap_progress(geom['opp_s_raw'], reward_state.last_opp_s, ref.track_length)

    prev_rel_s = wrap_rel_s(reward_state.last_ego_s - reward_state.last_opp_s, ref.track_length)
    rel_s = wrap_rel_s(ego_s - opp_s, ref.track_length)

    prev_phi = rel_progress_potential(prev_rel_s, rw.rel_behind_cap, rw.rel_ahead_cap)
    phi = rel_progress_potential(rel_s, rw.rel_behind_cap, rw.rel_ahead_cap)
    delta_rel_phi = float(np.clip(phi - prev_phi, -rw.rel_progress_clip, rw.rel_progress_clip))

    risk = clearance_risk(rel_s, geom['lat_gap'], geom['ego_v_s'], geom['opp_v_s'], rw)

    # Risk gate on the positive part only: closing on the opponent inside the
    # front-risk corridor earns nothing, while losing ground always costs
    # full price. An overtake through the lateral corridor (front_risk = 0)
    # keeps the full closing reward, and the asymmetry blocks the
    # "close offset, retreat aligned" reward-pumping loop. side_risk joins the
    # gate so that squeezing past alongside with insufficient lateral gap
    # cannot net positive relative progress (front_risk vanishes near
    # rel_s = 0, which D1-b showed makes narrow-gap squeezes profitable).
    delta_rel_gated = float(
        max(delta_rel_phi, 0.0) * (1.0 - max(risk['front_risk'], risk['side_risk_gate']))
        + min(delta_rel_phi, 0.0)
    )

    # Only an ego collision is penalized; an opponent-only crash (e.g. solo
    # wall hit) must not charge ego the collision penalty.
    collision = bool(obs['collisions'][0])

    # D3: potential-based closing-speed shaping (optimal-policy invariant).
    # Phi <= 0 is a "closing-speed debt" inside the front corridor; reducing
    # closing speed there refunds potential immediately, giving braking its
    # first dense positive credit. True termination uses Phi = 0, consistent
    # with the V(terminal) = 0 bootstrap; truncations keep the actual Phi.
    reward_closing_potential = 0.0
    if rw.w_closing_potential != 0.0:
        closing = max(0.0, geom['ego_v_s'] - geom['opp_v_s'])
        closing_phi = -rw.w_closing_potential * closing * risk['front_risk']
        phi_next = 0.0 if collision else closing_phi
        reward_closing_potential = (
            rw.closing_potential_gamma * phi_next - reward_state.last_closing_phi
        )
        reward_state.last_closing_phi = closing_phi

    # Training segments start with opponent ahead. Once ego establishes a
    # positive lead beyond the start margin, the overtake phase has begun.
    # Use a positive margin rather than a bare sign crossing: smooth crossings
    # can pass through a small positive rel_s before reaching the margin.
    if (
        reward_state.started_behind
        and not reward_state.overtake_started
        and rel_s > rw.overtake_start_margin_s
    ):
        reward_state.overtake_started = True

    safe_window = (
        reward_state.overtake_started
        and rel_s >= rw.success_margin_s
        and risk['clearance_risk'] < rw.success_clearance_threshold
        and not collision
    )
    reward_state.safe_overtake_hold_time = (
        reward_state.safe_overtake_hold_time + float(dt) if safe_window else 0.0
    )
    if reward_state.safe_overtake_hold_time >= rw.safe_overtake_hold_duration:
        reward_state.safe_overtake_held = True

    success_bonus = 0.0
    if reward_state.safe_overtake_held and not reward_state.had_safe_overtake_bonus:
        success_bonus = 1.0
        reward_state.had_safe_overtake_bonus = True

    progress_raw = float(np.clip(delta_ego_s, -rw.progress_clip_back, rw.progress_clip_forward))
    reward_progress = rw.w_progress * progress_raw
    reward_rel_progress = rw.w_rel_progress * delta_rel_gated
    reward_clearance = -rw.w_clearance_risk * risk['clearance_risk'] * float(dt)
    reward_collision = -rw.w_collision if collision else 0.0
    reward_overtake_success = rw.w_overtake_success * success_bonus

    total = reward_progress + reward_rel_progress + reward_clearance + reward_collision + reward_overtake_success
    if rw.w_closing_potential != 0.0:
        total += reward_closing_potential

    reward_state.last_ego_s = ego_s
    reward_state.last_opp_s = opp_s

    terms = {
        'reward_progress': float(reward_progress),
        'reward_rel_progress': float(reward_rel_progress),
        'reward_clearance': float(reward_clearance),
        'reward_collision': float(reward_collision),
        'reward_overtake_success': float(reward_overtake_success),
        'reward_closing_potential': float(reward_closing_potential),
        'delta_ego_s': float(delta_ego_s),
        'delta_opp_s': float(delta_opp_s),
        'rel_s': float(rel_s),
        'rel_progress_potential': float(phi),
        'delta_rel_potential': float(delta_rel_phi),
        'delta_rel_potential_gated': float(delta_rel_gated),
        'ego_d': float(geom['ego_d']),
        'opp_d': float(geom['opp_d']),
        'lat_gap': float(geom['lat_gap']),
        'rel_dist': float(geom['rel_dist']),
        'ego_v_s': float(geom['ego_v_s']),
        'opp_v_s': float(geom['opp_v_s']),
        'clearance_risk': float(risk['clearance_risk']),
        'front_risk': float(risk['front_risk']),
        'rear_risk': float(risk['rear_risk']),
        'side_risk': float(risk['side_risk']),
        'side_risk_gate': float(risk['side_risk_gate']),
        'lateral_overlap': float(risk['lateral_overlap']),
        'required_front_gap': float(risk['required_front_gap']),
        'required_rear_gap': float(risk['required_rear_gap']),
        'overtake_started': float(reward_state.overtake_started),
        'started_behind': float(reward_state.started_behind),
        'safe_overtake_hold_time': float(reward_state.safe_overtake_hold_time),
        'safe_overtake_held': float(reward_state.safe_overtake_held),
    }
    return float(total), terms

# ---------------------------------------------------------------------------
# PPO scenario curriculum
# ---------------------------------------------------------------------------
def sample_opp_speedscale(stage, rng, speedscale_range=None):
    """Sample the opponent speed scale for the given curriculum stage.

    speedscale_range=(lo, hi) overrides the stage schedule (D4-A eval-aligned
    sampling); None keeps the original stage behavior bit-for-bit.
    """
    if speedscale_range is not None:
        return float(rng.uniform(speedscale_range[0], speedscale_range[1]))
    if stage <= 1:
        return float(rng.uniform(0.45, 0.75))
    if stage == 2:
        return float(rng.uniform(0.50, 0.90))
    return float(rng.uniform(0.40, 1.00))

def sample_scenario(stage, rng, map_name, ego_raceline_choices, opp_raceline_choices,
                    interval_range=None, speedscale_range=None):
    """Sample a training segment with opponent initialized ahead of ego.

    interval_range/speedscale_range override the stage schedule (D4-A); None
    keeps the original stage behavior bit-for-bit.
    """
    ego_raceline = str(rng.choice(tuple(ego_raceline_choices)))
    opp_raceline = str(rng.choice(tuple(opp_raceline_choices)))
    _, _, ego_wp = load_raceline_with_speed(map_name, f"{ego_raceline}.csv", 0)

    ego_idx = int(rng.integers(0, len(ego_wp)))
    if interval_range is not None:
        interval_idx = int(rng.integers(interval_range[0], interval_range[1]))
    elif stage <= 1:
        interval_idx = int(rng.integers(8, 22))
    elif stage == 2:
        interval_idx = int(rng.integers(5, 32))
    else:
        interval_idx = int(rng.integers(3, 45))

    ego_idx, opp_idx = resolve_two_agent_indices(
        map_name, ego_raceline, opp_raceline, ego_idx, interval_idx
    )
    return {
        'map_name': map_name,
        'ego_raceline': ego_raceline,
        'opp_raceline': opp_raceline,
        'ego_idx': ego_idx,
        'interval_idx': interval_idx,
        'opp_idx': opp_idx,
        'opp_speedscale': sample_opp_speedscale(stage, rng, speedscale_range),
    }

# ---------------------------------------------------------------------------
# PPO reward argument helpers
# ---------------------------------------------------------------------------
def reward_weight_names():
    """Return numeric RewardWeights fields and fail if RewardWeights is not compact v1."""
    rw = RewardWeights()
    names = tuple(
        key for key, value in vars(rw).items()
        if isinstance(value, (int, float, np.integer, np.floating))
    )
    missing = [name for name in REQUIRED_V1_REWARD_FIELDS if name not in names]
    if missing:
        raise RuntimeError(
            "train_ppo.py requires the compact v1 PPO RewardWeights definition. "
            f"Missing RewardWeights fields: {missing}. "
            "Restore the validated fixed v1 RewardWeights first."
        )
    return names

def apply_reward_overrides(reward_weights, args):
    """Override RewardWeights fields from matching command line arguments."""
    for name in reward_weight_names():
        value = getattr(args, name)
        if value is not None:
            setattr(reward_weights, name, float(value))

def make_fixed_scenario(args):
    """Build the fixed training scenario from arguments, or None for sampling."""
    if not args.fixed_scenario:
        return None
    return {
        "ego_raceline": args.ego_raceline,
        "opp_raceline": args.opp_raceline,
        "ego_idx": args.ego_idx,
        "interval_idx": args.interval_idx,
        "opp_speedscale": args.opp_speedscale,
    }

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _load_actor_state(actor, state):
    """Load an actor state_dict, tolerating a fresh residual head in residual mode.

    A plain BC checkpoint has no res_head/residual_budgets entries; those stay
    at their init values (zero residual output). Any other mismatch is fatal.
    """
    residual_only_prefixes = ("res_head.", "residual_budgets")
    has_residual_keys = any(key.startswith(residual_only_prefixes) for key in state.keys())
    if hasattr(actor, "res_head") and not has_residual_keys:
        missing, unexpected = actor.load_state_dict(state, strict=False)
        bad_missing = [key for key in missing if not key.startswith(residual_only_prefixes)]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"Residual actor load mismatch: missing={bad_missing}, unexpected={list(unexpected)}."
            )
        return
    actor.load_state_dict(state)

def load_actor_critic(ac, path, device):
    """Load a full PPO checkpoint, an actor-only checkpoint, or a plain BC checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "actor_critic" in ckpt:
        ac.load_state_dict(ckpt["actor_critic"])
        return ckpt

    if isinstance(ckpt, dict) and "actor" in ckpt:
        _load_actor_state(ac.actor, ckpt["actor"])
        return ckpt

    _load_actor_state(ac.actor, ckpt)
    return {}

def load_frozen_bc(path, device, hidden_scale):
    """Load the original BC actor as a frozen anchor policy."""
    bc = End2Race(mask_prob=0.0, hidden_scale=hidden_scale).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "actor" in ckpt:
        bc.load_state_dict(ckpt["actor"])
    else:
        bc.load_state_dict(ckpt)
    bc.eval()
    for param in bc.parameters():
        param.requires_grad = False
    return bc

def save_actor_backbone(ac, path):
    """Save only the End2Race actor so original evaluators can load it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ac.actor.state_dict(), path)

def save_full_checkpoint(ac, path, optimizer, iteration, config, adv_norm_state=None):
    """Save a full PPO checkpoint for resume."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "actor_critic": ac.state_dict(),
        "actor": ac.actor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
        "config": dict(config),
        "hidden_scale": ac.actor.hidden_scale,
        "log_std": ac.log_std.detach().cpu().clone(),
    }
    if adv_norm_state is not None:
        payload["adv_norm_state"] = dict(adv_norm_state)
    torch.save(payload, path)

# ---------------------------------------------------------------------------
# Torch tensor and recurrent replay helpers
# ---------------------------------------------------------------------------
def obs_to_tensors(obs, device):
    """Convert a policy observation dict to [1, 1, dim] tensors."""
    lidar = torch.as_tensor(obs["lidar"], dtype=torch.float32, device=device).view(1, 1, -1)
    speed = torch.as_tensor(obs["prev_speed"], dtype=torch.float32, device=device).view(1, 1, -1)
    return lidar, speed

def zero_hidden(hidden_size, device):
    """Create a fresh GRU hidden state."""
    return torch.zeros(1, 1, hidden_size, device=device)

def forward_policy_sequence(ac, lidar_b, speed_b, starts_b, device):
    """Replay a recurrent rollout and reset hidden at stored episode starts."""
    hidden = zero_hidden(ac.actor.gru.hidden_size, device)
    means, stds = [], []

    for t in range(lidar_b.shape[1]):
        if starts_b[0, t].item() > 0.5:
            hidden = zero_hidden(ac.actor.gru.hidden_size, device)
        dist_t, hidden = ac(lidar_b[:, t:t + 1], speed_b[:, t:t + 1], hidden)
        means.append(dist_t.mean)
        stds.append(dist_t.stddev)

    return torch.distributions.Normal(torch.cat(means, dim=1), torch.cat(stds, dim=1))

def forward_frozen_bc_sequence(bc, lidar_b, speed_b, starts_b, device):
    """Replay the frozen BC actor on the same recurrent sequence."""
    hidden = zero_hidden(bc.gru.hidden_size, device)
    means = []
    for t in range(lidar_b.shape[1]):
        if starts_b[0, t].item() > 0.5:
            hidden = zero_hidden(bc.gru.hidden_size, device)
        mean_t, hidden = bc(lidar_b[:, t:t + 1], speed_b[:, t:t + 1], hidden)
        means.append(mean_t)
    return torch.cat(means, dim=1)

@torch.no_grad()
def validate_replay_identity(ac, buffer, device, atol, steer_only=False):
    """Check that recurrent replay reproduces rollout log probabilities before PPO update."""
    lidar_b, speed_b, _, act_b, old_logp_b, _, _, starts_b = buffer.tensors(device)
    dist = forward_policy_sequence(ac, lidar_b, speed_b, starts_b, device)
    replay_logp = dist.log_prob(act_b)[..., 0] if steer_only else dist.log_prob(act_b).sum(-1)
    abs_err = (replay_logp - old_logp_b).abs()
    max_err = float(abs_err.max().item())
    mean_err = float(abs_err.mean().item())
    if max_err > atol:
        raise RuntimeError(
            f"Replay identity failed: max |new_logp-old_logp|={max_err:.6g}, "
            f"mean={mean_err:.6g}, atol={atol}."
        )
    return {"replay_logp_max_error": max_err, "replay_logp_mean_error": mean_err}

@torch.no_grad()
def value_of_obs(ac, obs, device):
    """Evaluate the privileged critic on a single observation."""
    priv = torch.as_tensor(obs["priv"], dtype=torch.float32, device=device)
    return float(ac.critic(priv).view(-1)[0].item())

# ---------------------------------------------------------------------------
# PPO logging helpers
# ---------------------------------------------------------------------------
def summarize_iteration(iteration, rollout, update):
    """Format one training iteration into a compact log line."""
    keys = [
        ("ret", rollout.get("mean_completed_return")),
        ("coll", rollout.get("collision_rate")),
        ("succ", rollout.get("success_rate")),
        ("clip", rollout.get("action_was_clipped_rate")),
        ("clear", rollout.get("mean_clearance_risk")),
        ("rel_s", rollout.get("mean_rel_s")),
        ("lat", rollout.get("mean_lat_gap")),
        ("ot", rollout.get("mean_overtake_started")),
        ("fr", rollout.get("mean_front_risk")),
        ("sr", rollout.get("mean_side_risk")),
        ("srg", rollout.get("mean_side_risk_gate")),
        ("dsteer", rollout.get("steer_dev")),
        ("dspeed", rollout.get("speed_dev")),
        ("dspd_c", rollout.get("speed_dev_corridor")),
        ("clos_c", rollout.get("closing_corridor")),
        ("rsat", rollout.get("residual_sat_frac")),
        ("loff", rollout.get("mean_ego_lat_offset")),
        ("along", rollout.get("alongside_frac")),
        ("alat", rollout.get("alongside_lat_gap")),
        ("pol", update.get("policy_loss")),
        ("vf", update.get("value_loss")),
        ("ev", rollout.get("value_ev")),
        ("ascl", update.get("adv_scale")),
        ("kl", update.get("post_step_approx_kl")),
        ("bc", update.get("bc_anchor")),
        ("sanc", update.get("steer_anchor")),
        ("vanc", update.get("speed_anchor")),
        ("bcpre", update.get("bc_anchor_pre")),
        ("bcpost", update.get("bc_anchor_post")),
        ("bcw", update.get("bc_weight_mean")),
        ("ascale", update.get("anchor_speed_scale")),
        ("std_s", update.get("std_steer")),
        ("std_v", update.get("std_speed")),
    ]
    body = " ".join(f"{name}={value:.4g}" for name, value in keys if value is not None and np.isfinite(value))
    return f"iter={iteration:05d} {body}"

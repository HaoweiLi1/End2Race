#!/usr/bin/env python3
"""Compact v1 PPO fine-tuning script for End2Race.

Design assumptions:
- Actor observation stays deployable: LiDAR 360 + previous ego speed + GRU hidden.
- The model class is End2Race_PPO from model.py.
- Reward uses simulator geometry internally but does not expose privileged state to actor.
- The critic is privileged and asymmetric: it consumes simulator-state features
  ('priv' in the observation dict) and is discarded at deployment.
- Ego collision is true termination. Time limit and opponent-only collisions are
  truncations and bootstrap V(s_next).
"""

import os
import math
import argparse
import gym
import f110_gym
import numpy as np
import torch
import torch.optim as optim
from f110_gym.envs.base_classes import Integrator
from latticeplanner.utils import obsDict2oppoArray
from demonstration import setup_opp_planner
from model import End2Race, End2RaceResidual, End2Race_PPO, PRIV_DIM
from utils import (STEER_LIMIT, LIDAR_DIM, ACTION_DIM, downsample_lidar_for_model,
                   load_positions_and_speeds_from_params, load_reference_line,
                   resolve_two_agent_indices, wrap_rel_s)
from ppo_utils import (BOOL_INFO_KEYS, MEAN_INFO_KEYS, RewardWeights, RewardState,
                       apply_reward_overrides, compute_shaped_reward,
                       forward_frozen_bc_sequence, forward_policy_sequence,
                       load_actor_critic, load_frozen_bc, make_fixed_scenario,
                       obs_to_tensors, relative_geometry, reward_weight_names,
                       sample_opp_speedscale, sample_scenario, save_actor_backbone,
                       save_full_checkpoint, summarize_iteration,
                       validate_replay_identity, value_of_obs, zero_hidden)

# Speed normalization for the privileged critic input: raceline vx max (m/s).
PRIV_SPEED_NORM = 7.5

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Compact v1 PPO fine-tuning for End2Race.')

    # Data and model paths
    parser.add_argument("--model_path", type=str, default="pretrained/end2race.pth")
    parser.add_argument("--bc_model_path", type=str, default="pretrained/end2race.pth")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_actor_path", type=str, default="pretrained/end2race_ppo_v1.pth")
    parser.add_argument("--save_full_path", type=str, default="pretrained/end2race_ppo_v1_full.pt")

    # Environment and scenario configuration
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--max_speed", type=float, default=20.0)
    parser.add_argument("--sim_duration", type=float, default=8.0)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--fixed_scenario", action="store_true")
    parser.add_argument("--ego_idx", type=int, default=0)
    parser.add_argument("--interval_idx", type=int, default=15)
    parser.add_argument("--ego_raceline", type=str, default="raceline1")
    parser.add_argument("--opp_raceline", type=str, default="raceline1")
    parser.add_argument("--opp_speedscale", type=float, default=0.5)

    # PPO configuration
    parser.add_argument("--rollout_steps", type=int, default=1024)
    parser.add_argument("--ppo_epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.05)
    parser.add_argument("--actor_lr", type=float, default=1e-5)
    parser.add_argument("--critic_lr", type=float, default=5e-5)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.001)
    parser.add_argument("--beta_bc", type=float, default=2.0)
    parser.add_argument("--anchor_speed_scale", type=float, default=None)
    parser.add_argument("--pre_overtake_bc_multiplier", type=float, default=1.0)
    parser.add_argument("--freeze_speed", action="store_true",
                        help="Composite policy: speed comes from the frozen BC net at rollout and eval; "
                             "PPO trains steering only (1-D Gaussian).")
    parser.add_argument("--residual", action="store_true",
                        help="D2 bounded asymmetric residual: frozen BC backbone, trainable residual "
                             "head on GRU features, PPO distribution in pre-tanh residual space. "
                             "steer = BC + tanh(r)*steer_budget; speed = BC + tanh(r)*(down|up budget).")
    parser.add_argument("--residual_steer_budget", type=float, default=0.2)
    parser.add_argument("--residual_speed_up_budget", type=float, default=0.2)
    parser.add_argument("--residual_speed_down_budget", type=float, default=1.0)
    parser.add_argument("--residual_steer_std", type=float, default=0.15,
                        help="Pre-tanh residual std for steer; effective action noise near zero "
                             "residual is steer_budget * std (0.2*0.15 = 0.03 rad, matching BC-mode).")
    parser.add_argument("--residual_speed_std", type=float, default=0.25,
                        help="Pre-tanh residual std for speed; brake-side effective noise near zero "
                             "residual is speed_down_budget * std (1.0*0.25 = 0.25 m/s).")
    parser.add_argument("--residual_lr", type=float, default=3e-5,
                        help="Learning rate for the fresh residual head (Adam is scale-free per "
                             "coordinate, so the tanh*budget squash slows action-space movement "
                             "~5x for steer at equal lr; 3e-5 vs the 1e-5 fine-tune default).")
    parser.add_argument("--residual_presat_limit", type=float, default=2.0,
                        help="Bound loss activates on |r_mean| beyond this pre-tanh magnitude to "
                             "prevent tanh saturation lock-in (tanh(2.0) = 0.96).")
    parser.add_argument("--lateral_offset_prob", type=float, default=0.0,
                        help="D1-b curriculum: fraction of episodes where ego spawns laterally "
                             "offset from the raceline (0 disables, preserving D1-a behavior).")
    parser.add_argument("--lateral_offset_min", type=float, default=0.3)
    parser.add_argument("--lateral_offset_max", type=float, default=0.8)
    parser.add_argument("--opp_speedscale_min", type=float, default=None,
                        help="D4-A eval-aligned sampling: override opponent speedscale range "
                             "(with --opp_speedscale_max). None keeps the stage schedule.")
    parser.add_argument("--opp_speedscale_max", type=float, default=None)
    parser.add_argument("--interval_min", type=int, default=None,
                        help="D4-A: override the ego-opponent gap (interval_idx) sampling range "
                             "(with --interval_max). None keeps the stage schedule.")
    parser.add_argument("--interval_max", type=int, default=None)
    parser.add_argument("--bound_coef", type=float, default=0.01)
    parser.add_argument("--target_kl", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--adv_norm", type=str, default="batch", choices=("batch", "running"),
                        help="Advantage normalization: 'batch' divides by the per-batch std "
                             "(compresses rare -120 spikes batch-relative); 'running' centers "
                             "per batch but scales by an EMA of batch variance so rare-event "
                             "advantages keep their magnitude across batches.")
    parser.add_argument("--adv_norm_decay", type=float, default=0.99,
                        help="EMA decay for the running advantage variance (adv_norm=running).")
    parser.add_argument("--snapshot_every", type=int, default=0,
                        help="Additionally save non-overwriting actor snapshots every N iterations "
                             "(<save_actor_path>_iterNNNN.pth); 0 disables. Enables the "
                             "pre-registered 'last good checkpoint' rule for stop-loss runs.")

    # Model configuration
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--steer_std", type=float, default=0.03)
    parser.add_argument("--speed_std", type=float, default=0.25)

    # Training configuration
    parser.add_argument("--total_iterations", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    # Replay identity validation
    replay_group = parser.add_mutually_exclusive_group()
    replay_group.add_argument("--validate_replay_identity", dest="validate_replay_identity", action="store_true", default=True)
    replay_group.add_argument("--no_validate_replay_identity", "--no-validate_replay_identity", dest="validate_replay_identity", action="store_false")
    parser.add_argument("--replay_identity_atol", type=float, default=1e-5)

    # Reward weight overrides
    for name in reward_weight_names():
        parser.add_argument(f"--{name}", type=float, default=None)

    return parser.parse_args()

class End2RacePPOEnv:
    """Two-agent F1Tenth PPO environment for End2Race v1 training.

    Ego action comes from the PPO actor. Opponent action comes from the same
    lattice planner used by the original multi-agent evaluation scripts.
    """

    def __init__(self, map_name, max_speed=20.0, sim_duration=8.0, seed=0,
                 reward_weights=None, ego_raceline_choices=None, opp_raceline_choices=None,
                 lateral_offset_prob=0.0, lateral_offset_min=0.3, lateral_offset_max=0.8,
                 speedscale_range=None, interval_range=None):
        self.map_name = map_name
        self.max_speed = float(max_speed)
        self.sim_duration = float(sim_duration)
        self.reward_weights = reward_weights or RewardWeights()
        self.rng = np.random.default_rng(seed)
        self.stage = 1
        self.lateral_offset_prob = float(lateral_offset_prob)
        self.lateral_offset_min = float(lateral_offset_min)
        self.lateral_offset_max = float(lateral_offset_max)
        # D4-A eval-aligned sampling overrides (None keeps the stage schedule).
        self.speedscale_range = speedscale_range
        self.interval_range = interval_range
        self._ego_lat_offset = 0.0
        self.ego_raceline_choices = tuple(ego_raceline_choices or ('raceline1',))
        self.opp_raceline_choices = tuple(opp_raceline_choices or ('raceline1',))

        # Setup environment
        self.env = gym.make(
            "f110-v0",
            map=f"f1tenth_racetracks/{map_name}/{map_name}_map",
            map_ext=".png",
            num_agents=2,
            timestep=0.01,
            integrator=Integrator.RK4,
        )
        self.timestep = float(self.env.timestep)
        self.ref = load_reference_line(map_name, 'raceline1')

        # Episode state
        self._raw_obs = None
        self._reward_state = None
        self._opponent = None
        self._opp_traj = None
        self._tracker_count = 0
        self._tracker_steps = 10
        self._prev_speed = 0.0
        self._t = 0.0
        self._step_count = 0
        self._max_episode_steps = int(round(self.sim_duration / self.timestep))
        self._scenario = None

    def close(self):
        self.env.close()

    def reset(self, scenario=None):
        self._scenario = self._complete_scenario(
            sample_scenario(
                self.stage,
                self.rng,
                self.map_name,
                self.ego_raceline_choices,
                self.opp_raceline_choices,
                interval_range=self.interval_range,
                speedscale_range=self.speedscale_range,
            )
            if scenario is None
            else dict(scenario)
        )

        self._reset_opponent()
        positions, initial_speeds = load_positions_and_speeds_from_params(self._scenario, self.map_name)
        obs = self._reset_sim_with_offset(positions.astype(np.float64))

        self._raw_obs = obs
        self._prev_speed = float(initial_speeds[0]) * 0.9
        self._reward_state = RewardState.from_obs(obs, self.ref, self.reward_weights)
        self._step_count = 0
        self._t = 0.0
        return self._policy_obs(obs)

    def step(self, raw_ego_action):
        raw_ego_action = np.asarray(raw_ego_action, dtype=np.float32).reshape(2)
        ego_action = self._clip_ego_action(raw_ego_action)
        opp_action = self._opponent_action(self._raw_obs)

        # Step environment
        actions = np.vstack((ego_action, opp_action)).astype(np.float32)
        obs, _, env_done, env_info = self.env.step(actions)
        self._step_count += 1
        self._t = self._step_count * self.timestep

        reward, reward_terms = compute_shaped_reward(
            obs,
            self._reward_state,
            self.ref,
            self.reward_weights,
            self.timestep,
        )

        # Termination bookkeeping. Only an ego collision is a true failure
        # terminal; an opponent-only collision (e.g. solo wall crash) ends the
        # episode as a truncation so ego is not charged for it.
        ego_collision = bool(obs['collisions'][0])
        opp_collision = bool(np.any(obs['collisions'][1:]))
        timeout = bool(self._step_count >= self._max_episode_steps)
        terminated = bool(ego_collision)
        truncated = bool((not terminated) and (timeout or opp_collision or env_done))
        success = bool(self._reward_state.safe_overtake_held)

        self._raw_obs = obs
        self._prev_speed = float(obs['linear_vels_x'][0])

        info = {
            **reward_terms,
            'terminated': terminated,
            'truncated': truncated,
            'collision': ego_collision,
            'opp_collision': opp_collision,
            'env_done': bool(env_done),
            'timeout': timeout,
            'success': success,
            'time': float(self._t),
            'ego_lat_offset': float(abs(self._ego_lat_offset)),
            'raw_ego_action': raw_ego_action.copy(),
            'executed_ego_action': ego_action.copy(),
            'action_was_clipped': bool(np.any(np.abs(raw_ego_action - ego_action) > 1e-6)),
            'opp_action': opp_action.copy(),
            'env_info': env_info,
        }
        return self._policy_obs(obs), float(reward), terminated, truncated, info

    def _reset_sim_with_offset(self, poses):
        """Reset the simulator, optionally spawning ego laterally offset (D1-b curriculum).

        Offset spawns are rejected and resampled when the ego starts too close
        to a wall; after repeated failures the episode falls back to the
        on-raceline spawn.
        """
        self._ego_lat_offset = 0.0
        if self.lateral_offset_prob > 0.0 and self.rng.random() < self.lateral_offset_prob:
            for _ in range(10):
                offset = float(self.rng.uniform(self.lateral_offset_min, self.lateral_offset_max))
                if self.rng.random() < 0.5:
                    offset = -offset
                candidate = poses.copy()
                theta = candidate[0, 2]
                candidate[0, 0] += -math.sin(theta) * offset
                candidate[0, 1] += math.cos(theta) * offset
                obs, _, _, _ = self.env.reset(poses=candidate)
                if float(np.min(obs['scans'][0])) > 0.45:
                    self._ego_lat_offset = offset
                    return obs
        obs, _, _, _ = self.env.reset(poses=poses)
        return obs

    def _complete_scenario(self, scenario):
        scenario.setdefault('ego_raceline', self.ego_raceline_choices[0])
        scenario.setdefault('opp_raceline', self.opp_raceline_choices[0])
        scenario.setdefault('ego_idx', 0)
        scenario.setdefault('interval_idx', 15)
        scenario.setdefault('opp_speedscale', sample_opp_speedscale(self.stage, self.rng, self.speedscale_range))
        ego_idx, opp_idx = resolve_two_agent_indices(
            self.map_name,
            scenario['ego_raceline'],
            scenario['opp_raceline'],
            scenario['ego_idx'],
            scenario['interval_idx'],
            scenario.get('opp_idx'),
        )
        scenario['ego_idx'] = ego_idx
        scenario['opp_idx'] = opp_idx
        return scenario

    def _reset_opponent(self):
        self._opponent = setup_opp_planner(self.map_name, self._scenario['opp_raceline'])
        self._opp_traj = None
        self._tracker_count = 0
        self._tracker_steps = int(self._opponent.conf.tracker_steps)
        self._opponent.tracker.prev_error = 0.0
        self._opponent.prev_opp_pose = np.array([0.0, 0.0])
        self._opponent.prev_traj_local = np.zeros_like(self._opponent.prev_traj_local)
        self._opponent.best_traj = None
        self._opponent.goal_grid = None

    def _policy_obs(self, obs):
        return {
            'lidar': downsample_lidar_for_model(obs['scans'][0]),
            'prev_speed': np.array([self._prev_speed], dtype=np.float32),
            'priv': self._priv_state(obs),
        }

    def _priv_state(self, obs):
        """Privileged simulator-state features for the critic only.

        Must be built after the RewardState update for the same obs so that
        hold-time and overtake flags stay in sync with the returned observation.
        """
        geom = relative_geometry(obs, self.ref)
        rw = self.reward_weights
        rs = self._reward_state
        rel_s = wrap_rel_s(geom['ego_s_raw'] - geom['opp_s_raw'], self.ref.track_length)
        track_phase = 2.0 * math.pi * geom['ego_s_raw'] / self.ref.track_length
        return np.array([
            rel_s / rw.rel_behind_cap,
            geom['lat_gap'],
            geom['ego_v_s'] / PRIV_SPEED_NORM,
            geom['opp_v_s'] / PRIV_SPEED_NORM,
            geom['ego_d'],
            geom['opp_d'],
            rs.safe_overtake_hold_time / rw.safe_overtake_hold_duration,
            float(rs.overtake_started),
            float(rs.had_safe_overtake_bonus),
            float(self._scenario['opp_speedscale']),
            math.sin(track_phase),
            math.cos(track_phase),
        ], dtype=np.float32)

    def _clip_ego_action(self, raw_action):
        action = np.asarray(raw_action, dtype=np.float32).reshape(2).copy()
        action[0] = np.clip(action[0], -STEER_LIMIT, STEER_LIMIT)
        action[1] = np.clip(action[1], 0.0, self.max_speed)
        return action

    def _opponent_action(self, obs):
        # Replan only every tracker_steps; reuse the cached trajectory otherwise.
        if self._tracker_count == 0 or self._opp_traj is None:
            self._opp_traj = self._opponent.plan(
                obs['poses_x'][1],
                obs['poses_y'][1],
                obs['poses_theta'][1],
                obsDict2oppoArray(obs, 1),
                obs['linear_vels_x'][1],
            )

        opp_steer, opp_speed = self._opponent.tracker.plan(
            obs['poses_x'][1],
            obs['poses_y'][1],
            obs['poses_theta'][1],
            obs['linear_vels_x'][1],
            self._opp_traj,
        )
        self._tracker_count = (self._tracker_count + 1) % self._tracker_steps

        return np.array(
            [
                float(np.clip(opp_steer, -STEER_LIMIT, STEER_LIMIT)),
                float(opp_speed) * float(self._scenario['opp_speedscale']),
            ],
            dtype=np.float32,
        )

class RolloutBuffer:
    """Serial recurrent PPO buffer with true termination and truncation separated."""

    def __init__(self, rollout_steps, gamma, gae_lambda):
        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.lidar = np.zeros((self.rollout_steps, LIDAR_DIM), dtype=np.float32)
        self.prev_speed = np.zeros((self.rollout_steps, 1), dtype=np.float32)
        self.priv = np.zeros((self.rollout_steps, PRIV_DIM), dtype=np.float32)
        self.raw_actions = np.zeros((self.rollout_steps, ACTION_DIM), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.values = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.log_probs = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.terminateds = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.truncateds = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.trunc_next_values = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.episode_starts = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.advantages = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.returns = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(self, obs, raw_action, reward, value, log_prob,
            terminated, truncated, trunc_next_value, episode_start):
        if self.ptr >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer overflow.")
        self.lidar[self.ptr] = np.asarray(obs["lidar"], dtype=np.float32).reshape(LIDAR_DIM)
        self.prev_speed[self.ptr] = np.asarray(obs["prev_speed"], dtype=np.float32).reshape(1)
        self.priv[self.ptr] = np.asarray(obs["priv"], dtype=np.float32).reshape(PRIV_DIM)
        self.raw_actions[self.ptr] = np.asarray(raw_action, dtype=np.float32).reshape(ACTION_DIM)
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.terminateds[self.ptr] = float(terminated)
        self.truncateds[self.ptr] = float(truncated)
        self.trunc_next_values[self.ptr] = float(trunc_next_value)
        self.episode_starts[self.ptr] = float(episode_start)
        self.ptr += 1

    def compute_returns_and_advantage(self, candidate_last_value):
        """Compute GAE.

        Collisions/true terminations use zero bootstrap. Time-limit truncations use
        the stored V(s_next), because the task could have continued beyond the
        artificial cutoff.
        """
        if self.ptr != self.rollout_steps:
            raise RuntimeError(f"Buffer has {self.ptr} steps, expected {self.rollout_steps}.")

        gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            if self.terminateds[t] > 0.5:
                next_value = 0.0
                boundary = True
            elif self.truncateds[t] > 0.5:
                next_value = float(self.trunc_next_values[t])
                boundary = True
            elif t == self.rollout_steps - 1:
                next_value = float(candidate_last_value)
                boundary = False
            else:
                next_value = float(self.values[t + 1])
                boundary = False

            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            gae = delta if boundary else delta + self.gamma * self.gae_lambda * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def tensors(self, device):
        return (
            torch.as_tensor(self.lidar, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(self.prev_speed, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(self.priv, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(self.raw_actions, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(self.log_probs, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(self.advantages, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(self.returns, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(self.episode_starts, dtype=torch.float32, device=device).unsqueeze(0),
        )

def collect_rollout(env, ac, buffer, device, scenario, frozen_bc=None, freeze_speed=False,
                    residual=False):
    """Collect one on-policy rollout with the current actor-critic.

    With freeze_speed, the executed speed command comes from the frozen BC net
    (its own recurrent state) and the stored action/log-prob cover steering only.

    With residual, the policy distribution lives in pre-tanh residual space:
    the buffer stores the sampled residuals (whose log-probs the PPO ratio is
    built on) while the environment executes the composed BC+residual action.
    """
    buffer.reset()
    obs = env.reset(scenario=scenario)
    hidden = zero_hidden(ac.actor.gru.hidden_size, device)
    bc_hidden = zero_hidden(frozen_bc.gru.hidden_size, device) if freeze_speed else None
    episode_start = True

    episode_return = 0.0
    completed_returns = []
    steer_devs = []
    speed_devs = []
    residual_sat = []
    info_values = {key: [] for key in BOOL_INFO_KEYS + MEAN_INFO_KEYS}

    for _ in range(buffer.rollout_steps):
        lidar_t, speed_t = obs_to_tensors(obs, device)
        if freeze_speed:
            with torch.no_grad():
                dist, next_hidden = ac(lidar_t, speed_t, hidden)
                bc_out, next_bc_hidden = frozen_bc(lidar_t, speed_t, bc_hidden)
                sampled = dist.sample()
                logp_t = dist.log_prob(sampled)[..., 0]
            raw_steer = float(sampled.view(-1)[0].item())
            bc_speed = float(bc_out.view(-1)[1].item())
            raw_action = np.array([raw_steer, bc_speed], dtype=np.float32)
            policy_action = raw_action
            steer_devs.append(abs(float(dist.mean.view(-1)[0].item()) - float(bc_out.view(-1)[0].item())))
        elif residual:
            with torch.no_grad():
                dist, base, next_hidden = ac.forward_residual_rollout(lidar_t, speed_t, hidden)
                sampled_r = dist.sample()
                logp_t = dist.log_prob(sampled_r).sum(-1)
                env_action = ac.actor.compose(base, sampled_r)
                mean_delta = ac.actor.residual_delta(dist.mean)
            raw_action = env_action.view(-1).detach().cpu().numpy().astype(np.float32)
            policy_action = sampled_r.view(-1).detach().cpu().numpy().astype(np.float32)
            steer_devs.append(abs(float(mean_delta.view(-1)[0].item())))
            speed_devs.append(float(mean_delta.view(-1)[1].item()))
            residual_sat.append(float((dist.mean.abs() > 2.0).float().mean().item()))
        else:
            with torch.no_grad():
                action_t, logp_t, next_hidden = ac.act(
                    lidar_t, speed_t, hidden, deterministic=False
                )
            raw_action = action_t.view(-1).detach().cpu().numpy().astype(np.float32)
            policy_action = raw_action
        value = value_of_obs(ac, obs, device)

        next_obs, reward, terminated, truncated, info = env.step(raw_action)

        trunc_next_value = 0.0
        if truncated:
            trunc_next_value = value_of_obs(ac, next_obs, device)

        buffer.add(
            obs=obs,
            raw_action=policy_action,
            reward=reward,
            value=value,
            log_prob=float(logp_t.view(-1)[0].item()),
            terminated=terminated,
            truncated=truncated,
            trunc_next_value=trunc_next_value,
            episode_start=episode_start,
        )

        episode_return += float(reward)
        for key, values in info_values.items():
            if key in info:
                values.append(float(info[key]))

        if terminated or truncated:
            completed_returns.append(episode_return)
            episode_return = 0.0
            obs = env.reset(scenario=scenario)
            hidden = zero_hidden(ac.actor.gru.hidden_size, device)
            if freeze_speed:
                bc_hidden = zero_hidden(frozen_bc.gru.hidden_size, device)
            episode_start = True
        else:
            obs = next_obs
            hidden = next_hidden.detach()
            if freeze_speed:
                bc_hidden = next_bc_hidden.detach()
            episode_start = False

    candidate_last_value = value_of_obs(ac, obs, device)
    buffer.compute_returns_and_advantage(candidate_last_value=candidate_last_value)

    return_var = float(np.var(buffer.returns))
    value_ev = float(1.0 - np.var(buffer.returns - buffer.values) / max(return_var, 1e-8))

    # Alongside diagnostics: how much pass-attempt exposure the rollout had and
    # how much lateral clearance the policy kept while alongside.
    rel_arr = np.asarray(info_values["rel_s"], dtype=np.float64)
    lat_arr = np.asarray(info_values["lat_gap"], dtype=np.float64)
    alongside = np.abs(rel_arr) < 0.6
    alongside_frac = float(alongside.mean()) if len(alongside) else float("nan")
    alongside_lat_gap = float(lat_arr[alongside].mean()) if alongside.any() else float("nan")

    # D3 corridor diagnostics: does the (residual) policy mean brake inside the
    # front corridor? speed_devs is per-step in residual mode, aligned with info.
    speed_dev_corridor = float("nan")
    closing_corridor = float("nan")
    fr_arr = np.asarray(info_values["front_risk"], dtype=np.float64)
    corridor = fr_arr > 0.1
    if corridor.any():
        ego_v = np.asarray(info_values["ego_v_s"], dtype=np.float64)
        opp_v = np.asarray(info_values["opp_v_s"], dtype=np.float64)
        closing_corridor = float(np.maximum(ego_v - opp_v, 0.0)[corridor].mean())
        if len(speed_devs) == len(fr_arr):
            speed_dev_corridor = float(np.asarray(speed_devs, dtype=np.float64)[corridor].mean())

    metrics = {
        "alongside_frac": alongside_frac,
        "alongside_lat_gap": alongside_lat_gap,
        "value_ev": value_ev,
        "steer_dev": float(np.mean(steer_devs)) if steer_devs else float("nan"),
        "speed_dev": float(np.mean(speed_devs)) if speed_devs else float("nan"),
        "speed_dev_corridor": speed_dev_corridor,
        "closing_corridor": closing_corridor,
        "residual_sat_frac": float(np.mean(residual_sat)) if residual_sat else float("nan"),
        "rollout_return": float(np.sum(buffer.rewards)),
        "completed_episodes": float(len(completed_returns)),
        "mean_completed_return": float(np.mean(completed_returns)) if completed_returns else float("nan"),
        "partial_episode_return": float(episode_return),
        "bootstrap_value": float(candidate_last_value),
        "adv_mean": float(np.mean(buffer.advantages)),
        "adv_std": float(np.std(buffer.advantages)),
        "return_mean": float(np.mean(buffer.returns)),
        "return_std": float(np.std(buffer.returns)),
    }

    for key in BOOL_INFO_KEYS:
        values = info_values[key]
        metrics[f"{key}_rate"] = float(np.mean(values)) if values else float("nan")
    for key in MEAN_INFO_KEYS:
        values = info_values[key]
        metrics[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")

    return metrics

def ppo_update(ac, frozen_bc, buffer, optimizer, device, args, adv_norm_state=None):
    """Run clipped PPO epochs with BC anchor, bound loss, and KL early stop."""
    lidar_b, speed_b, priv_b, act_b, old_logp_b, adv_b, ret_b, starts_b = buffer.tensors(device)
    adv_scale = None
    if args.adv_norm == "running":
        # Center per batch, but scale by an EMA of batch variance: a batch
        # containing a rare -120 collision advantage no longer has that spike
        # squashed to a fixed z-score by its own inflated std.
        batch_var = float(adv_b.var(unbiased=False).item())
        if adv_norm_state.get("var") is None:
            adv_norm_state["var"] = batch_var
        else:
            decay = float(args.adv_norm_decay)
            adv_norm_state["var"] = decay * adv_norm_state["var"] + (1.0 - decay) * batch_var
        adv_scale = math.sqrt(max(adv_norm_state["var"], 0.0))
        adv_b = (adv_b - adv_b.mean()) / (adv_scale + 1e-8)
    else:
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

    # The frozen BC replay depends only on rollout inputs, so compute it once
    # for all epochs. In residual mode the composed mean minus BC base IS the
    # applied residual, so no BC replay is needed for the anchor terms.
    bc_mean = None
    if not args.residual:
        with torch.no_grad():
            bc_mean = forward_frozen_bc_sequence(frozen_bc, lidar_b, speed_b, starts_b, device)

    metrics = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "bc_anchor": [],
        "bc_anchor_unweighted": [],
        "bc_anchor_pre": [],
        "bc_anchor_post": [],
        "bc_pre_fraction": [],
        "bc_weight_mean": [],
        "anchor_speed_scale": [],
        "steer_anchor": [],
        "speed_anchor": [],
        "bound_loss": [],
        "approx_kl": [],
        "post_step_approx_kl": [],
        "clip_fraction": [],
        "ratio_mean": [],
        "ratio_min": [],
        "ratio_max": [],
        "grad_norm": [],
    }
    updates = 0
    early_stopped = False

    anchor_speed_scale = args.max_speed if args.anchor_speed_scale is None else float(args.anchor_speed_scale)
    if anchor_speed_scale <= 0.0:
        raise ValueError("--anchor_speed_scale must be positive when set.")
    action_scale = torch.tensor([STEER_LIMIT, anchor_speed_scale], dtype=torch.float32, device=device).view(1, 1, 2)

    for _ in range(args.ppo_epochs):
        dist = forward_policy_sequence(ac, lidar_b, speed_b, starts_b, device)
        values = ac.critic(priv_b).squeeze(-1)
        if args.freeze_speed:
            new_logp = dist.log_prob(act_b)[..., 0]
        else:
            new_logp = dist.log_prob(act_b).sum(-1)
        log_ratio = new_logp - old_logp_b
        ratio = torch.exp(log_ratio)

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > args.clip_eps).float().mean()
            ratio_mean = ratio.mean()
            ratio_min = ratio.min()
            ratio_max = ratio.max()

        surr1 = ratio * adv_b
        surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_b
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (values - ret_b).pow(2).mean()
        entropy = dist.entropy()[..., 0].mean() if args.freeze_speed else dist.entropy().sum(-1).mean()

        if args.residual:
            # Deviation from BC equals the applied (bounded) residual at the
            # policy mean; with beta_bc = 0 this is diagnostics only.
            anchor_per_dim = (ac.actor.residual_delta(dist.mean) / action_scale).pow(2)
        else:
            anchor_per_dim = ((dist.mean - bc_mean) / action_scale).pow(2)
        # In freeze_speed mode the speed head is unused, so it gets no anchor pressure.
        anchor_per_step = anchor_per_dim[..., 0] if args.freeze_speed else anchor_per_dim.sum(dim=-1)

        pre_overtake_mask = ((priv_b[..., 0] < 0.0) & (priv_b[..., 7] < 0.5)).float()
        pre_multiplier = max(float(args.pre_overtake_bc_multiplier), 0.0)
        bc_weights = 1.0 + (pre_multiplier - 1.0) * pre_overtake_mask

        steer_anchor = (anchor_per_dim[..., 0] * bc_weights).mean()
        speed_anchor = (anchor_per_dim[..., 1] * bc_weights).mean()
        bc_anchor = (anchor_per_step * bc_weights).mean()
        bc_anchor_unweighted = anchor_per_step.mean()
        pre_count = pre_overtake_mask.sum().clamp_min(1.0)
        post_mask = 1.0 - pre_overtake_mask
        post_count = post_mask.sum().clamp_min(1.0)
        bc_anchor_pre = (anchor_per_step * pre_overtake_mask).sum() / pre_count
        bc_anchor_post = (anchor_per_step * post_mask).sum() / post_count

        if args.residual:
            # Actions are bounded by construction; instead keep the pre-tanh
            # residual mean out of the tanh saturation zone (dead gradients).
            bound_loss = (
                torch.relu(dist.mean.abs() - args.residual_presat_limit).pow(2).sum(dim=-1).mean()
            )
        else:
            steer_bound = torch.relu(dist.mean[..., 0].abs() - STEER_LIMIT).pow(2).mean()
            speed_bound = (
                torch.relu(-dist.mean[..., 1]).pow(2).mean()
                + torch.relu(dist.mean[..., 1] - args.max_speed).pow(2).mean()
            )
            bound_loss = steer_bound if args.freeze_speed else steer_bound + speed_bound

        loss = (
            policy_loss
            + args.vf_coef * value_loss
            - args.ent_coef * entropy
            + args.beta_bc * bc_anchor
            + args.bound_coef * bound_loss
        )

        metrics["policy_loss"].append(float(policy_loss.item()))
        metrics["value_loss"].append(float(value_loss.item()))
        metrics["entropy"].append(float(entropy.item()))
        metrics["bc_anchor"].append(float(bc_anchor.item()))
        metrics["bc_anchor_unweighted"].append(float(bc_anchor_unweighted.item()))
        metrics["bc_anchor_pre"].append(float(bc_anchor_pre.item()))
        metrics["bc_anchor_post"].append(float(bc_anchor_post.item()))
        metrics["bc_pre_fraction"].append(float(pre_overtake_mask.mean().item()))
        metrics["bc_weight_mean"].append(float(bc_weights.mean().item()))
        metrics["anchor_speed_scale"].append(float(anchor_speed_scale))
        metrics["steer_anchor"].append(float(steer_anchor.item()))
        metrics["speed_anchor"].append(float(speed_anchor.item()))
        metrics["bound_loss"].append(float(bound_loss.item()))
        metrics["approx_kl"].append(float(approx_kl.item()))
        metrics["clip_fraction"].append(float(clip_fraction.item()))
        metrics["ratio_mean"].append(float(ratio_mean.item()))
        metrics["ratio_min"].append(float(ratio_min.item()))
        metrics["ratio_max"].append(float(ratio_max.item()))

        if approx_kl.item() > args.target_kl * 1.5:
            early_stopped = True
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(ac.parameters(), args.max_grad_norm)
        optimizer.step()
        updates += 1

        with torch.no_grad():
            post_dist = forward_policy_sequence(ac, lidar_b, speed_b, starts_b, device)
            if args.freeze_speed:
                post_logp = post_dist.log_prob(act_b)[..., 0]
            else:
                post_logp = post_dist.log_prob(act_b).sum(-1)
            post_log_ratio = post_logp - old_logp_b
            post_kl = ((post_log_ratio.exp() - 1.0) - post_log_ratio).mean()
        metrics["grad_norm"].append(float(grad_norm.item()))
        metrics["post_step_approx_kl"].append(float(post_kl.item()))

    out = {key: float(np.mean(value)) if value else float("nan") for key, value in metrics.items()}
    out["num_updates"] = float(updates)
    out["early_stopped"] = float(early_stopped)
    out["std_steer"] = float(ac.log_std.detach().exp()[0].item())
    out["std_speed"] = float(ac.log_std.detach().exp()[1].item())
    if adv_scale is not None:
        out["adv_scale"] = float(adv_scale)
    return out

def main():
    args = parse_arguments()

    if args.residual and args.freeze_speed:
        raise ValueError("--residual and --freeze_speed are mutually exclusive modes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)

    # D4-A eval-aligned sampling overrides (both bounds required, or neither).
    speedscale_range = None
    if (args.opp_speedscale_min is None) != (args.opp_speedscale_max is None):
        raise ValueError("--opp_speedscale_min and --opp_speedscale_max must be set together.")
    if args.opp_speedscale_min is not None:
        speedscale_range = (float(args.opp_speedscale_min), float(args.opp_speedscale_max))
    interval_range = None
    if (args.interval_min is None) != (args.interval_max is None):
        raise ValueError("--interval_min and --interval_max must be set together.")
    if args.interval_min is not None:
        interval_range = (int(args.interval_min), int(args.interval_max))

    # Setup environment
    env = End2RacePPOEnv(
        map_name=args.map_name,
        max_speed=args.max_speed,
        sim_duration=args.sim_duration,
        seed=args.train_seed,
        lateral_offset_prob=args.lateral_offset_prob,
        lateral_offset_min=args.lateral_offset_min,
        lateral_offset_max=args.lateral_offset_max,
        speedscale_range=speedscale_range,
        interval_range=interval_range,
    )
    env.stage = args.stage
    apply_reward_overrides(env.reward_weights, args)
    # PBRS invariance requires the shaping gamma to equal the RL discount;
    # sync unless the user explicitly overrode --closing_potential_gamma.
    if args.closing_potential_gamma is None:
        env.reward_weights.closing_potential_gamma = args.gamma

    min_rollout_steps = int(math.ceil(args.sim_duration / env.timestep))
    if args.rollout_steps < min_rollout_steps:
        raise ValueError(
            f"--rollout_steps must be at least one full episode: {min_rollout_steps}; "
            f"got {args.rollout_steps}."
        )

    # Create actor-critic and load pretrained weights. In residual mode the
    # Gaussian stds are pre-tanh residual stds.
    ac = End2Race_PPO(
        hidden_scale=args.hidden_scale,
        steer_std=args.residual_steer_std if args.residual else args.steer_std,
        speed_std=args.residual_speed_std if args.residual else args.speed_std,
        residual_mode=args.residual,
        residual_steer_budget=args.residual_steer_budget,
        residual_speed_up_budget=args.residual_speed_up_budget,
        residual_speed_down_budget=args.residual_speed_down_budget,
    ).to(device)

    source_path = args.resume if args.resume else args.model_path
    if not os.path.exists(source_path):
        raise FileNotFoundError(source_path)
    loaded_ckpt = load_actor_critic(ac, source_path, device)
    ac.train()

    # D2 hard isolation: freeze the entire BC backbone; only the residual
    # head (plus log_std and the critic, which live outside ac.actor) train.
    frozen_param_snapshot = {}
    if args.residual:
        for param in ac.actor.parameters():
            param.requires_grad = False
        for param in ac.actor.res_head.parameters():
            param.requires_grad = True
        frozen_param_snapshot = {
            name: param.detach().clone()
            for name, param in ac.actor.named_parameters()
            if not param.requires_grad
        }

    def assert_residual_isolation(stage):
        for name, param in ac.actor.named_parameters():
            if name not in frozen_param_snapshot:
                continue
            if param.grad is not None and float(param.grad.abs().max().item()) > 0.0:
                raise RuntimeError(f"[{stage}] Frozen actor param received gradient: {name}")
            if not torch.equal(param.detach(), frozen_param_snapshot[name]):
                raise RuntimeError(f"[{stage}] Frozen actor param changed: {name}")

    frozen_bc = load_frozen_bc(args.bc_model_path, device, args.hidden_scale)
    buffer = RolloutBuffer(args.rollout_steps, args.gamma, args.gae_lambda)

    # Setup optimizer with separate actor/critic learning rates. Frozen
    # backbone params are excluded (no-op for the legacy modes, where every
    # actor param is trainable).
    critic_params = list(ac.critic.parameters())
    critic_param_ids = {id(param) for param in critic_params}
    actor_params = [
        param for param in ac.parameters()
        if id(param) not in critic_param_ids and param.requires_grad
    ]
    actor_lr = args.residual_lr if args.residual else args.actor_lr
    optimizer = optim.Adam(
        [
            {"params": critic_params, "lr": args.critic_lr},
            {"params": actor_params, "lr": actor_lr},
        ]
    )

    start_iter = 0
    adv_norm_state = {"var": None}
    if args.resume:
        if "optimizer" not in loaded_ckpt or "iteration" not in loaded_ckpt:
            raise RuntimeError("--resume checkpoint must contain optimizer and iteration.")
        optimizer.load_state_dict(loaded_ckpt["optimizer"])
        start_iter = int(loaded_ckpt["iteration"])
        if "adv_norm_state" in loaded_ckpt:
            adv_norm_state = dict(loaded_ckpt["adv_norm_state"])

    scenario = make_fixed_scenario(args)

    # Train model
    try:
        for iteration in range(start_iter, args.total_iterations):
            rollout_metrics = collect_rollout(
                env, ac, buffer, device, scenario,
                frozen_bc=frozen_bc, freeze_speed=args.freeze_speed,
                residual=args.residual,
            )
            replay_metrics = {}
            if args.validate_replay_identity:
                replay_metrics = validate_replay_identity(
                    ac, buffer, device, args.replay_identity_atol, steer_only=args.freeze_speed
                )
            update_metrics = ppo_update(ac, frozen_bc, buffer, optimizer, device, args,
                                        adv_norm_state=adv_norm_state)
            if args.residual and iteration == start_iter:
                assert_residual_isolation("after first update")

            if args.log_every > 0 and ((iteration + 1) % args.log_every == 0 or iteration == start_iter):
                print(summarize_iteration(iteration + 1, rollout_metrics, {**update_metrics, **replay_metrics}), flush=True)

            if args.save_every > 0 and (iteration + 1) % args.save_every == 0:
                save_full_checkpoint(ac, args.save_full_path, optimizer, iteration + 1, vars(args),
                                     adv_norm_state=adv_norm_state)
                save_actor_backbone(ac, args.save_actor_path)
            if args.snapshot_every > 0 and (iteration + 1) % args.snapshot_every == 0:
                snap_base, snap_ext = os.path.splitext(args.save_actor_path)
                save_actor_backbone(ac, f"{snap_base}_iter{iteration + 1:04d}{snap_ext}")

        save_full_checkpoint(ac, args.save_full_path, optimizer, args.total_iterations, vars(args),
                             adv_norm_state=adv_norm_state)
        save_actor_backbone(ac, args.save_actor_path)

        if args.residual:
            assert_residual_isolation("after training")

        # Fail fast if the actor-only checkpoint is not loadable by the evaluator.
        if args.residual:
            test_actor = End2RaceResidual(hidden_scale=args.hidden_scale).to(device)
        else:
            test_actor = End2Race(mask_prob=0.0, hidden_scale=args.hidden_scale).to(device)
        test_actor.load_state_dict(torch.load(args.save_actor_path, map_location=device, weights_only=False))
        print(f"Saved actor-only checkpoint: {args.save_actor_path}")
        print(f"Saved full PPO checkpoint:   {args.save_full_path}")
    finally:
        env.close()

if __name__ == "__main__":
    main()

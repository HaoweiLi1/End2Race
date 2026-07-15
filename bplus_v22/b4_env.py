"""Canonical 100 Hz complete-episode collection for B4 direct-head PPO."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from typing import Mapping, MutableSequence

import numpy as np
import torch

from bplus_v22.b4_direct import (
    ACTION_DIM,
    B4DirectHeadPolicy,
    B4Transition,
    project_raw_action,
)
from bplus_v22.ppo_env import B2Scenario, build_privileged_features
from ppo_utils import RewardState


class B4InvalidEpisode(RuntimeError):
    """Infrastructure/simulator termination that must not enter an update."""


def terminal_task_reward(collision_any: bool, terminal_overtake: bool) -> float:
    """Return the only reward that may enter a B4 rollout."""

    return float(-2 * int(bool(collision_any)) + int(bool(terminal_overtake)))


def assert_terminal_reward_ledger(
    transitions: tuple[B4Transition, ...] | list[B4Transition],
    *,
    collision_any: bool,
    terminal_overtake: bool,
) -> None:
    """Fail closed if dense diagnostics leak into the actor/critic replay."""

    if not transitions:
        raise AssertionError("B4 terminal reward ledger is empty")
    if any(float(row.reward) != 0.0 for row in transitions[:-1]):
        raise AssertionError("B4 nonterminal replay reward is not exactly zero")
    expected = terminal_task_reward(collision_any, terminal_overtake)
    if float(transitions[-1].reward) != expected:
        raise AssertionError("B4 final replay reward is not -2*C+O")


@dataclass(frozen=True)
class B4EpisodeResult:
    scenario: B2Scenario
    episode_id: int
    transitions: tuple[B4Transition, ...]
    arrays: Mapping[str, np.ndarray]
    outcome: object
    step_count: int
    terminal_reason: str
    projection_transition_count: int
    steer_projection_count: int
    speed_projection_count: int
    max_abs_steer_projection_delta: float
    max_abs_speed_projection_delta: float

    def __post_init__(self) -> None:
        if not self.transitions or self.step_count != len(self.transitions):
            raise ValueError("B4 episode transition count drift")
        if self.terminal_reason not in {"any_agent_collision", "product_horizon"}:
            raise ValueError("B4 episode terminal reason is invalid")
        if not self.transitions[-1].terminated or any(
            row.terminated for row in self.transitions[:-1]
        ):
            raise ValueError("B4 episode terminal boundary drift")
        assert_terminal_reward_ledger(
            self.transitions,
            collision_any=bool(self.outcome.collision_any),
            terminal_overtake=self.outcome.corrected_outcome3 == "overtake",
        )
        for row in self.transitions:
            row.validate()


def _record_projection(
    delta: torch.Tensor,
    counts: dict[str, float | int],
) -> None:
    absolute = torch.abs(delta.detach()).reshape(-1, ACTION_DIM)
    changed = absolute > 0.0
    counts["projection_transition_count"] += int(torch.any(changed, dim=1).sum().item())
    counts["steer_projection_count"] += int(changed[:, 0].sum().item())
    counts["speed_projection_count"] += int(changed[:, 1].sum().item())
    counts["max_abs_steer_projection_delta"] = max(
        float(counts["max_abs_steer_projection_delta"]),
        float(absolute[:, 0].max().item()),
    )
    counts["max_abs_speed_projection_delta"] = max(
        float(counts["max_abs_speed_projection_delta"]),
        float(absolute[:, 1].max().item()),
    )


def run_b4_episode(
    policy: B4DirectHeadPolicy,
    device: torch.device,
    scenario: B2Scenario,
    *,
    episode_id: int,
    deterministic: bool = False,
    sim_duration: float = 8.0,
    sample_ledger: MutableSequence[tuple[np.ndarray, float]] | None = None,
) -> B4EpisodeResult:
    """Collect exactly one valid complete episode or raise ``B4InvalidEpisode``.

    The simulator executes ``projection(raw_latent)``.  Old policy probability
    is always recorded on the unprojected latent.  ``deterministic=True`` is a
    preflight-only mode where the raw latent is the actor mean.
    """

    # Heavy simulator imports stay local so pure B4 contracts run on CPU-only
    # hosts without requiring a live Gym/F110 environment.
    import gym
    from f110_gym.envs.base_classes import Integrator
    from d25.oracle import _resolve_case, _trajectory_arrays, classify_trajectory
    from demonstration import setup_opp_planner
    from latticeplanner.utils import obsDict2oppoArray, project_point_to_centerline
    from ppo_utils import RewardWeights, compute_shaped_reward
    from utils import load_positions_and_speeds_from_params, load_reference_line

    if int(episode_id) != episode_id or episode_id < 0:
        raise ValueError("B4 episode id must be nonnegative")
    if not math.isfinite(float(sim_duration)) or sim_duration <= 0.0:
        raise ValueError("B4 simulator duration is invalid")
    np.random.seed(42)
    resolved = _resolve_case(scenario.simulator_case())
    map_name = resolved["map_name"]
    environment = gym.make(
        "f110-v0",
        map=f"f1tenth_racetracks/{map_name}/{map_name}_map",
        map_ext=".png",
        num_agents=2,
        timestep=0.01,
        integrator=Integrator.RK4,
    )
    try:
        positions, initial_speeds = load_positions_and_speeds_from_params(resolved, map_name)
        opponent = setup_opp_planner(map_name, resolved["opp_raceline"])
        # Match the already-audited B2 collector's explicit fresh planner state.
        opponent.tracker.prev_error = 0.0
        opponent.prev_opp_pose = np.asarray([0.0, 0.0])
        opponent.prev_traj_local = np.zeros_like(opponent.prev_traj_local)
        opponent.best_traj = None
        opponent.goal_grid = None
        tracker_count = 0
        tracker_steps = 10
        opp_traj = None

        hidden = policy.zero_hidden(1, device)
        previous_speed = float(initial_speeds[0]) * 0.9
        reference = load_reference_line(map_name, "raceline1")
        centerline_wp = np.loadtxt(
            f"f1tenth_racetracks/{map_name}/raceline1.csv", delimiter=";", skiprows=1
        )
        centerline = centerline_wp[:, 1:3]
        track_length = float(
            sum(
                np.linalg.norm(centerline[index + 1] - centerline[index])
                for index in range(len(centerline) - 1)
            )
        )
        obs, _, initial_done, _ = environment.reset(poses=positions)
        if initial_done:
            raise B4InvalidEpisode("simulator returned done at reset")
        reward_weights = RewardWeights()
        reward_state = RewardState.from_obs(obs, reference, reward_weights)
        initial_ego_progress, _ = project_point_to_centerline(
            np.asarray([obs["poses_x"][0], obs["poses_y"][0]]), centerline
        )
        initial_opp_progress, _ = project_point_to_centerline(
            np.asarray([obs["poses_x"][1], obs["poses_y"][1]]), centerline
        )
        final_state = "overtaking" if initial_ego_progress > initial_opp_progress else "following"
        final_ego_progress = float(initial_ego_progress)
        final_opp_progress = float(initial_opp_progress)
        collision = ego_collision = opp_collision = False
        records = {
            name: []
            for name in (
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
            )
        }
        transitions: list[B4Transition] = []
        projection = {
            "projection_transition_count": 0,
            "steer_projection_count": 0,
            "speed_projection_count": 0,
            "max_abs_steer_projection_delta": 0.0,
            "max_abs_speed_projection_delta": 0.0,
        }
        lap_time = 0.0
        step_index = 0
        terminal_reason: str | None = None
        policy.actor.eval()
        policy.critic.eval()

        while terminal_reason is None:
            ego_lidar = np.asarray(obs["scans"][0]).reshape(-1)
            if len(ego_lidar) != 360:
                ego_lidar = ego_lidar[
                    np.linspace(0, len(ego_lidar) - 1, 360, dtype=int)
                ]
            opp_lidar = np.asarray(obs["scans"][1]).reshape(-1)
            if len(opp_lidar) != 360:
                opp_lidar = opp_lidar[
                    np.linspace(0, len(opp_lidar) - 1, 360, dtype=int)
                ]
            lidar_tensor = torch.as_tensor(
                ego_lidar, dtype=torch.float32, device=device
            ).reshape(1, 1, 360)
            speed_tensor = torch.tensor(
                [[[previous_speed]]], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                feature, next_hidden = policy.feature_step(
                    lidar_tensor, speed_tensor, hidden
                )
                mean = policy.mean_from_feature(feature)
                if deterministic:
                    raw_action = mean
                    old_log_prob = policy.log_prob(mean, raw_action)
                else:
                    raw_action, old_log_prob = policy.sample_raw(mean)
                executed_action, projection_delta = project_raw_action(raw_action)
                privileged = build_privileged_features(
                    obs, reference, reward_state, resolved["opp_speedscale"]
                )
                privileged_tensor = torch.as_tensor(
                    privileged, dtype=torch.float32, device=device
                ).reshape(1, -1)
                old_value = policy.value(privileged_tensor)
            _record_projection(projection_delta, projection)
            command = executed_action[0]
            ego_steer = float(command[0].item())
            ego_speed = float(command[1].item())
            transition = B4Transition(
                l2_id=scenario.l2_id,
                episode_id=int(episode_id),
                step_index=step_index,
                feature=feature[0].cpu().numpy().astype(np.float32),
                privileged_feature=privileged.astype(np.float32),
                raw_action=raw_action[0].cpu().numpy().astype(np.float32),
                executed_action=executed_action[0].cpu().numpy().astype(np.float32),
                projection_delta=projection_delta[0].cpu().numpy().astype(np.float32),
                old_log_prob=float(old_log_prob.item()),
                old_value=float(old_value.item()),
            )
            # This assertion sits directly on the production collector seam:
            # the replay action must be the sampler output, never the projected
            # physical command.  The optional smoke ledger keeps an independent
            # copy for end-to-end verification.
            sampled_raw = raw_action[0].detach().cpu().numpy().astype(np.float32)
            if not np.array_equal(transition.raw_action, sampled_raw):
                raise B4InvalidEpisode("collector did not store the sampled raw latent")
            if float(transition.old_log_prob) != float(old_log_prob.item()):
                raise B4InvalidEpisode("collector old log-probability ledger drift")
            if sample_ledger is not None:
                sample_ledger.append((sampled_raw.copy(), float(old_log_prob.item())))
            transitions.append(transition)

            if tracker_count == 0:
                opp_traj = opponent.plan(
                    obs["poses_x"][1],
                    obs["poses_y"][1],
                    obs["poses_theta"][1],
                    obsDict2oppoArray(obs, 1),
                    obs["linear_vels_x"][1],
                )
            opp_steer, opp_speed = opponent.tracker.plan(
                obs["poses_x"][1],
                obs["poses_y"][1],
                obs["poses_theta"][1],
                obs["linear_vels_x"][1],
                opp_traj,
            )
            opp_steer = float(np.clip(opp_steer, -0.52, 0.52))
            opp_speed = float(opp_speed) * resolved["opp_speedscale"]
            ego_progress, _ = project_point_to_centerline(
                np.asarray([obs["poses_x"][0], obs["poses_y"][0]]), centerline
            )
            opp_progress, _ = project_point_to_centerline(
                np.asarray([obs["poses_x"][1], obs["poses_y"][1]]), centerline
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
            previous_speed = float(obs["linear_vels_x"][0])
            obs, timestep, simulator_done, _ = environment.step(
                np.asarray([[ego_steer, ego_speed], [opp_steer, opp_speed]])
            )
            timestep = float(timestep)
            if not math.isfinite(timestep) or timestep <= 0.0:
                raise B4InvalidEpisode("simulator returned an invalid timestep")
            lap_time += timestep
            # This legacy shaped reward is retained only because it advances
            # RewardState for the privileged critic features.  Its returned
            # scalar/terms are deliberately discarded and never touch replay.
            _diagnostic_shaped_reward = compute_shaped_reward(
                obs, reward_state, reference, reward_weights, timestep
            )
            ego_progress, _ = project_point_to_centerline(
                np.asarray([obs["poses_x"][0], obs["poses_y"][0]]), centerline
            )
            opp_progress, _ = project_point_to_centerline(
                np.asarray([obs["poses_x"][1], obs["poses_y"][1]]), centerline
            )
            if ego_progress < initial_ego_progress - track_length / 2:
                ego_progress += track_length
            if opp_progress < initial_opp_progress - track_length / 2:
                opp_progress += track_length
            final_state = "overtaking" if ego_progress > opp_progress else "following"
            final_ego_progress = float(ego_progress)
            final_opp_progress = float(opp_progress)
            collision_now = bool(np.any(obs["collisions"]))
            # Match the product evaluator's literal ``while lap_time < 8``
            # boundary.  A tolerance here changes some 0.01-s trajectories by
            # one complete simulator step because of binary accumulation.
            horizon_now = lap_time >= float(sim_duration)
            if collision_now:
                collision = True
                ego_collision = bool(obs["collisions"][0])
                opp_collision = bool(np.any(obs["collisions"][1:]))
                terminal_reason = "any_agent_collision"
            elif horizon_now:
                terminal_reason = "product_horizon"
            elif simulator_done:
                raise B4InvalidEpisode(
                    "simulator ended before collision or product horizon"
                )
            tracker_count = (tracker_count + 1) % tracker_steps
            hidden = next_hidden
            step_index += 1

        if not transitions or terminal_reason is None:
            raise B4InvalidEpisode("B4 episode produced no complete trajectory")
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
        if bool(outcome.collision_any) != collision:
            raise B4InvalidEpisode("simulator/classifier collision disagreement")
        transitions[-1].terminated = True
        transitions[-1].reward = terminal_task_reward(
            bool(outcome.collision_any),
            outcome.corrected_outcome3 == "overtake",
        )
        assert_terminal_reward_ledger(
            transitions,
            collision_any=bool(outcome.collision_any),
            terminal_overtake=outcome.corrected_outcome3 == "overtake",
        )
        for row in transitions:
            row.validate()
        return B4EpisodeResult(
            scenario=scenario,
            episode_id=int(episode_id),
            transitions=tuple(transitions),
            arrays=arrays,
            outcome=outcome,
            step_count=len(transitions),
            terminal_reason=terminal_reason,
            projection_transition_count=int(projection["projection_transition_count"]),
            steer_projection_count=int(projection["steer_projection_count"]),
            speed_projection_count=int(projection["speed_projection_count"]),
            max_abs_steer_projection_delta=float(
                projection["max_abs_steer_projection_delta"]
            ),
            max_abs_speed_projection_delta=float(
                projection["max_abs_speed_projection_delta"]
            ),
        )
    finally:
        environment.close()
        gc.collect()

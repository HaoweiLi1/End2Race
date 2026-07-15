"""Logging-only callback for PPO V1 rollouts and completed episodes."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PPOV1MetricsCallback(BaseCallback):
    """Record reward components, outcomes, sampler branches, and actions."""

    def __init__(self, run_dir: str | Path, n_envs: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.run_dir = Path(run_dir)
        self.n_envs = int(n_envs)
        self.episode_log_path = self.run_dir / "episodes.jsonl"
        self.update_log_path = self.run_dir / "updates.jsonl"
        self._episodes = [self._new_episode() for _ in range(self.n_envs)]
        self._rollout_records: list[dict[str, Any]] = []
        self._component_sums: Counter[str] = Counter()
        self._branch_transition_counts: Counter[str] = Counter()
        self._reset_branch_counts: Counter[str] = Counter()
        self._unique_scenario_ids: set[str] = set()
        self._pending_initial_reset_infos: list[dict[str, Any]] = []
        self._initial_resets_recorded = False
        self._partial_episodes_carried_in = 0
        self._actions: list[np.ndarray] = []
        self._transition_count = 0
        self.update_index = 0
        self.latest_update_summary: dict[str, Any] = {}

    @staticmethod
    def _new_episode() -> dict[str, Any]:
        return {
            "steps": 0,
            "reward_total": 0.0,
            "reward_progress": 0.0,
            "reward_relative": 0.0,
            "reward_collision": 0.0,
        }

    @staticmethod
    def _append_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")

    @staticmethod
    def _sampler_branch(info: dict[str, Any]) -> str:
        return str(info.get("sampler_branch") or "unknown")

    def _record_reset(self, info: dict[str, Any]) -> None:
        self._reset_branch_counts[self._sampler_branch(info)] += 1

    def _on_training_start(self) -> None:
        # The initial vector reset happens in BaseAlgorithm._setup_learn before
        # on_training_start().  Record it exactly once; later model.learn()
        # calls continue the same environments and must not recount it.
        if not self._initial_resets_recorded:
            reset_infos = list(getattr(self.training_env, "reset_infos", []))
            if len(reset_infos) != self.n_envs:
                raise RuntimeError(
                    f"Expected {self.n_envs} initial reset infos, got {len(reset_infos)}"
                )
            self._pending_initial_reset_infos = [dict(info) for info in reset_infos]
            self._initial_resets_recorded = True

    def _on_rollout_start(self) -> None:
        self._rollout_records = []
        self._component_sums = Counter()
        self._branch_transition_counts = Counter()
        self._reset_branch_counts = Counter()
        self._unique_scenario_ids = set()
        self._partial_episodes_carried_in = sum(episode["steps"] > 0 for episode in self._episodes)
        self._actions = []
        self._transition_count = 0
        for info in self._pending_initial_reset_infos:
            self._record_reset(info)
        self._pending_initial_reset_infos = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = np.asarray(self.locals.get("dones", np.zeros(self.n_envs, dtype=bool)), dtype=bool)
        actions = np.asarray(self.locals.get("actions", []), dtype=np.float64)
        if actions.size:
            self._actions.append(actions.reshape(self.n_envs, -1).copy())

        for env_index, info in enumerate(infos):
            episode = self._episodes[env_index]
            episode["steps"] += 1
            for key in ("reward_total", "reward_progress", "reward_relative", "reward_collision"):
                value = float(info.get(key, 0.0))
                episode[key] += value
                self._component_sums[key] += value
            self._transition_count += 1
            branch = self._sampler_branch(info)
            self._branch_transition_counts[branch] += 1
            scenario_id = info.get("scenario_id")
            if scenario_id is not None:
                self._unique_scenario_ids.add(str(scenario_id))

            if env_index < dones.size and dones[env_index]:
                if bool(info.get("ego_collision", False)):
                    outcome = "ego_collision"
                else:
                    outcome = "overtake" if float(info.get("relative_position_m", 0.0)) > 0.0 else "follow"
                record = {
                    **episode,
                    "outcome": outcome,
                    "scenario_id": info.get("scenario_id"),
                    "sampler_branch": info.get("sampler_branch"),
                    "opponent_collision_latched": bool(info.get("opponent_collision_latched", False)),
                    "final_relative_position_m": float(info.get("relative_position_m", 0.0)),
                    "termination_reason": info.get("termination_reason"),
                    "elapsed_time": float(info.get("elapsed_time", 0.0)),
                }
                self._append_json(self.episode_log_path, record)
                self._rollout_records.append(record)
                self._episodes[env_index] = self._new_episode()
                reset_infos = list(getattr(self.training_env, "reset_infos", []))
                if len(reset_infos) != self.n_envs:
                    raise RuntimeError(
                        f"Expected {self.n_envs} auto-reset infos, got {len(reset_infos)}"
                    )
                self._record_reset(dict(reset_infos[env_index]))
        return True

    def _on_rollout_end(self) -> None:
        self.update_index += 1
        if self._actions:
            action_array = np.concatenate(self._actions, axis=0)
            action_statistics = {
                "count": int(action_array.shape[0]),
                "steering_mean": float(np.mean(action_array[:, 0])),
                "steering_std": float(np.std(action_array[:, 0])),
                "steering_min": float(np.min(action_array[:, 0])),
                "steering_max": float(np.max(action_array[:, 0])),
                "speed_mean": float(np.mean(action_array[:, 1])),
                "speed_std": float(np.std(action_array[:, 1])),
                "speed_min": float(np.min(action_array[:, 1])),
                "speed_max": float(np.max(action_array[:, 1])),
            }
        else:
            action_statistics = {"count": 0}
        outcome_counts = Counter(record["outcome"] for record in self._rollout_records)
        completed_by_branch = Counter(
            str(record.get("sampler_branch") or "unknown")
            for record in self._rollout_records
        )
        outcome_by_branch = {
            outcome: Counter(
                str(record.get("sampler_branch") or "unknown")
                for record in self._rollout_records
                if record["outcome"] == outcome
            )
            for outcome in ("ego_collision", "follow", "overtake")
        }
        partial_episodes_carried_out = sum(episode["steps"] > 0 for episode in self._episodes)
        denominator = max(self._transition_count, 1)
        summary = {
            "update": self.update_index,
            "transitions": self._transition_count,
            "completed_episodes": len(self._rollout_records),
            "completed_episodes_by_sampler_branch": dict(sorted(completed_by_branch.items())),
            "ego_collision_episodes_by_sampler_branch": dict(sorted(outcome_by_branch["ego_collision"].items())),
            "follow_episodes_by_sampler_branch": dict(sorted(outcome_by_branch["follow"].items())),
            "overtake_episodes_by_sampler_branch": dict(sorted(outcome_by_branch["overtake"].items())),
            "outcomes": dict(sorted(outcome_counts.items())),
            "unique_scenario_id_count": len(self._unique_scenario_ids),
            "unique_scenario_ids": sorted(self._unique_scenario_ids),
            "reset_count_by_sampler_branch": dict(sorted(self._reset_branch_counts.items())),
            "partial_episodes_carried_across_rollout_boundary": partial_episodes_carried_out,
            "partial_episodes_carried_in": self._partial_episodes_carried_in,
            "partial_episodes_carried_out": partial_episodes_carried_out,
            "reward_component_means": {
                key: float(self._component_sums[key] / denominator)
                for key in ("reward_progress", "reward_relative", "reward_collision", "reward_total")
            },
            "sampler_branch_transitions": dict(sorted(self._branch_transition_counts.items())),
            "action_statistics": action_statistics,
        }
        self.latest_update_summary = summary
        self._append_json(self.update_log_path, summary)
        for name, value in summary["reward_component_means"].items():
            self.logger.record(f"ppo_v1/{name}", value)
        for name, value in outcome_counts.items():
            self.logger.record(f"ppo_v1/episodes_{name}", int(value))
        for branch, value in completed_by_branch.items():
            self.logger.record(f"ppo_v1/completed_episodes_{branch}", int(value))
        for branch, value in self._reset_branch_counts.items():
            self.logger.record(f"ppo_v1/resets_{branch}", int(value))
        self.logger.record("ppo_v1/unique_scenario_ids", len(self._unique_scenario_ids))
        self.logger.record("ppo_v1/partial_episodes_carried_in", self._partial_episodes_carried_in)
        self.logger.record("ppo_v1/partial_episodes_carried_out", partial_episodes_carried_out)

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
        self._branch_counts: Counter[str] = Counter()
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

    def _on_rollout_start(self) -> None:
        self._rollout_records = []
        self._component_sums = Counter()
        self._branch_counts = Counter()
        self._actions = []
        self._transition_count = 0

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
            branch = str(info.get("sampler_branch") or "unknown")
            self._branch_counts[branch] += 1

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
        denominator = max(self._transition_count, 1)
        summary = {
            "update": self.update_index,
            "transitions": self._transition_count,
            "completed_episodes": len(self._rollout_records),
            "outcomes": dict(sorted(outcome_counts.items())),
            "reward_component_means": {
                key: float(self._component_sums[key] / denominator)
                for key in ("reward_progress", "reward_relative", "reward_collision", "reward_total")
            },
            "sampler_branch_transitions": dict(sorted(self._branch_counts.items())),
            "action_statistics": action_statistics,
        }
        self.latest_update_summary = summary
        self._append_json(self.update_log_path, summary)
        for name, value in summary["reward_component_means"].items():
            self.logger.record(f"ppo_v1/{name}", value)
        for name, value in outcome_counts.items():
            self.logger.record(f"ppo_v1/episodes_{name}", int(value))

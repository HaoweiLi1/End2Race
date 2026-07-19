"""Run artifacts, JSONL records, and actor/critic checkpoints for PPO training."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import numbers
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import sb3_contrib
import stable_baselines3
import torch

from model import End2Race
from ppo.scenarios import ScenarioSpec


def require_finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise RuntimeError(f"{name} must be finite, got {value!r}")
    return float(value)


def require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} must be finite")


class TrainingRecorder:

    def __init__(self, output_dir: str | Path, hidden_scale: int) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.output_dir.exists():
            if not self.output_dir.is_dir():
                raise RuntimeError(f"PPO output path is not a directory: {self.output_dir}")
            if any(self.output_dir.iterdir()):
                raise RuntimeError(f"PPO output directory must be empty: {self.output_dir}")
        else:
            self.output_dir.mkdir(parents=True)
        self.hidden_scale = int(hidden_scale)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir()
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self.metrics_path.touch()
        self.episodes_path.touch()

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, allow_nan=False) + "\n")

    @staticmethod
    def _cpu_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu() for name, tensor in state_dict.items()}

    def write_run_config(self, args, ppo_config: dict[str, Any], device: torch.device, training_constants: dict[str, Any]) -> None:
        payload = {
            "args": dict(vars(args)),
            "ppo_config": ppo_config,
            "device": str(device),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "stable_baselines3_version": stable_baselines3.__version__,
            "sb3_contrib_version": sb3_contrib.__version__,
            "started_at": self.started_at,
            **training_constants,
        }
        self._write_json(self.output_dir / "run_config.json", payload)

    def write_scenario_pools(
        self,
        collision_scenarios: Sequence[ScenarioSpec],
        ordinary_scenarios: Sequence[ScenarioSpec],
        cache_info: dict[str, Any],
    ) -> None:
        self._write_json(self.output_dir / "collision_scenarios.json", [asdict(scenario) for scenario in collision_scenarios])
        self._write_json(self.output_dir / "ordinary_scenarios.json", [asdict(scenario) for scenario in ordinary_scenarios])
        self._write_json(self.output_dir / "collision_cache_info.json", cache_info)

    def record_episode(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.episodes_path, record)

    def record_metrics(self, record: dict[str, Any]) -> None:
        for name, value in record.items():
            if isinstance(value, numbers.Real) and not isinstance(value, bool):
                require_finite_number(name, value)
        self._append_jsonl(self.metrics_path, record)

    def _save_actor(self, path: Path, state_dict: Mapping[str, torch.Tensor]) -> Path:
        checkpoint = self._cpu_state_dict(state_dict)
        if len(checkpoint) != 12:
            raise RuntimeError(f"Expected a 12-key actor checkpoint, got {len(checkpoint)} keys")
        torch.save(checkpoint, path)
        with torch.random.fork_rng(devices=[]):
            actor = End2Race(mask_prob=0.0, hidden_scale=self.hidden_scale)
            actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        return path

    def save_warmup_critic(self, state_dict: Mapping[str, torch.Tensor]) -> Path:
        path = self.checkpoints_dir / "critic_warmup.pt"
        torch.save(self._cpu_state_dict(state_dict), path)
        return path

    def save_formal_checkpoints(
        self,
        update: int,
        actor_state_dict: Mapping[str, torch.Tensor],
        critic_state_dict: Mapping[str, torch.Tensor],
    ) -> tuple[Path, Path]:
        actor_path = self._save_actor(self.checkpoints_dir / f"actor_u{update:04d}.pth", actor_state_dict)
        critic_path = self.checkpoints_dir / f"critic_u{update:04d}.pt"
        torch.save(self._cpu_state_dict(critic_state_dict), critic_path)
        return actor_path, critic_path

    def save_final_actor(self, state_dict: Mapping[str, torch.Tensor]) -> Path:
        return self._save_actor(self.output_dir / "actor_final.pth", state_dict)

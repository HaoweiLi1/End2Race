#!/usr/bin/env python3
"""Regression checks for the strict single-variable B5-A control plane."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from unittest import mock

import torch

from Experiments import runner
from bplus_v22.b5_runner import (
    B4_SEED1_CURRICULUM_SHA256,
    expected_b5_plan_config,
)
from bplus_v22.b5_safe import SAFE_EPISODE_COUNT, SafeReference, save_reference


REPO = Path(__file__).resolve().parents[1]


def _reference(path: Path) -> None:
    feature = torch.zeros((SAFE_EPISODE_COUNT, 1680), dtype=torch.float32)
    mean = torch.zeros((SAFE_EPISODE_COUNT, 2), dtype=torch.float32)
    maps = tuple(
        map_name
        for map_name in ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")
        for _outcome in ("follow", "overtake")
        for _ in range(8)
    )
    outcomes = tuple(
        outcome
        for _map in range(4)
        for outcome in ("follow", "overtake")
        for _ in range(8)
    )
    save_reference(
        SafeReference(
            feature=feature,
            bc_mean=mean,
            episode_index=torch.arange(SAFE_EPISODE_COUNT),
            step_index=torch.zeros(SAFE_EPISODE_COUNT, dtype=torch.int64),
            lengths=(1,) * SAFE_EPISODE_COUNT,
            l2_ids=tuple(f"L2:{index:064x}" for index in range(SAFE_EPISODE_COUNT)),
            l4_ids=tuple(f"L4:{index:064x}" for index in range(SAFE_EPISODE_COUNT)),
            map_names=maps,
            outcomes=outcomes,
        ),
        path,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        reference_path = Path(directory) / "safe_reference.npz"
        _reference(reference_path)
        config = expected_b5_plan_config(
            REPO / runner.TASK8_RELEASE,
            REPO / runner.D2_METADATA,
            reference_path,
        )
        assert config["ppo"]["actor_lr"] == 3e-5
        assert config["ppo"]["clip_eps"] == 0.10
        assert config["ppo"]["actor_epochs"] == 3
        assert config["ppo"]["target_weighted_kl"] == 0.015
        assert config["ppo"]["iterations"] == 30
        assert config["ppo"]["snapshots"] == [0, 10, 20, 30]
        assert config["curriculum_sha256_by_seed"] == {
            "1": B4_SEED1_CURRICULUM_SHA256
        }
        assert config["safe_reference"]["cap"] == 0.01
        assert config["safe_reference"]["retry_multipliers"] == [
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
        ]

        run_id = "b5_control_plane_test"
        jobs, queues = runner._b5_training_jobs()
        plan = runner.RunPlan(
            schema=runner.PLAN_SCHEMA,
            run_id=run_id,
            kind="b5_train",
            created_at="2026-07-14T00:00:00+00:00",
            source_commit="0" * 40,
            source_tree="1" * 40,
            source_archive_path=str((Path(directory) / "source.tar").resolve()),
            source_archive_sha256="2" * 64,
            source_archive_size=1,
            inputs_archive_path=str((Path(directory) / "inputs.tar").resolve()),
            inputs_archive_sha256="3" * 64,
            inputs_archive_size=1,
            source_inputs=(),
            inputs=(),
            hosts=runner._default_hosts(
                run_id,
                "GPU-00000000-0000-0000-0000-000000000000",
                "GPU-11111111-1111-1111-1111-111111111111",
                {"python": "3.10", "torch": "test"},
            ),
            jobs=jobs,
            queues=queues,
            required_cli=runner.B5_REQUIRED_TRAIN_CLI,
            module_path_contract=runner.MODULE_PATH_CONTRACT,
            config=config,
            collection_root=str((Path(directory) / "collection").resolve()),
        )
        plan = runner._seal_plan(plan)
        runner._verify_plan(plan)
        assert len(plan.jobs) == 1
        assert plan.jobs[0].host_id == "remote"
        assert plan.jobs[0].seed == 1
        assert plan.jobs[0].kind == "b5_training"
        assert plan.queues == {"b5-seed1-remote": ("b5-seed1",)}

        payload_commands = "\n".join(
            " ".join(command)
            for command in runner._collect_payload_commands(
                plan, Path(directory) / "collection.partial"
            )
        )
        assert "inputs/b5/safe_reference.npz" in payload_commands
        assert "control/input_contract/safe_reference.npz" in payload_commands

        collected_root = Path(directory) / "collected"
        collected_control = collected_root / "control"
        collected_control.mkdir(parents=True)
        baseline_marker = collected_control / "bc_baseline_preflight.json"
        plumbing_marker = collected_control / "plumbing_smoke.json"
        ready_marker = collected_control / "READY.json"
        collected_reference = collected_control / "input_contract/safe_reference.npz"
        collected_reference.parent.mkdir()
        baseline_marker.write_text("{}\n", encoding="utf-8")
        plumbing_marker.write_text("{}\n", encoding="utf-8")
        collected_reference.write_bytes(reference_path.read_bytes())
        ready_marker.write_text(
            json.dumps(
                {
                    "schema": "end2race-b5-safe-ready-1",
                    "passed": True,
                    "run_plan_sha256": plan.plan_sha256,
                    "source_commit": plan.source_commit,
                    "source_archive_sha256": plan.source_archive_sha256,
                    "inputs_archive_sha256": plan.inputs_archive_sha256,
                    "baseline_marker_sha256": runner._sha256_file(baseline_marker),
                    "plumbing_marker_sha256": runner._sha256_file(plumbing_marker),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with mock.patch.object(runner, "_validate_baseline_marker"), mock.patch.object(
            runner, "_validate_plumbing_marker"
        ) as validate_plumbing:
            runner._validate_ready_marker(
                plan,
                collected_root,
                b5_reference_path=collected_reference,
            )
        validate_plumbing.assert_called_once_with(
            plan,
            collected_root,
            plumbing_marker,
            b5_reference_path=collected_reference,
        )

        changed = replace(
            plan,
            config={
                **plan.config,
                "ppo": {**plan.config["ppo"], "actor_lr": 1e-5},
            },
        )
        changed = runner._seal_plan(changed)
        try:
            runner._verify_plan(changed)
        except runner.RunnerError:
            pass
        else:
            raise AssertionError("B5 control plane accepted a non-B4 actor LR")

    print("B5 strict single-variable control-plane contracts passed")


if __name__ == "__main__":
    main()

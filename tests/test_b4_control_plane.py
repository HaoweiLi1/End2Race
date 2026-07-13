#!/usr/bin/env python3
"""B4 immutable plan topology and marker contracts without creating a RunPlan."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from bplus_v22.b4_cli import B4_EVAL_CONFIG
from bplus_v22.b4_runner import (
    _validate_control_plane_ready,
    expected_b4_plan_config,
)
from Experiments.runner import (
    B4_REQUIRED_EVAL_CLI,
    B4_REQUIRED_TRAIN_CLI,
    CANONICAL_BC_SHA256,
    HostSpec,
    InputEntry,
    MODULE_PATH_CONTRACT,
    PLAN_SCHEMA,
    RunPlan,
    RunnerError,
    _b4_eval_jobs,
    _b4_training_jobs,
    _collect_payload_commands,
    _collect_status_commands,
    _seal_plan,
    _verify_plan,
)


REPO = Path(__file__).resolve().parent.parent
TASK8 = REPO / "Experiments/B1_route_r2_scaffold/artifacts/task8_manifests_20260712_113241"
METADATA = (
    REPO
    / "Experiments/A3_d2_representation/artifacts/non_test_full_20260711_175713/episode_metadata.tsv"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hosts(run_id: str) -> tuple[HostSpec, HostSpec]:
    root = f"/home/haowei/end2race_runs/{run_id}"
    environment = {"python": "3.10", "torch": "test"}
    return (
        HostSpec(
            "local",
            "local",
            root,
            "/home/haowei/miniconda3/envs/end2race/bin/python",
            ":0",
            "gpu0",
            "local-gpu",
            None,
            environment,
        ),
        HostSpec(
            "remote",
            "remote",
            root,
            "/home/haowei/miniconda3/envs/end2race/bin/python",
            ":1",
            "gpu1",
            "remote-gpu",
            "haowei@192.168.2.127",
            environment,
        ),
    )


def training_plan() -> RunPlan:
    run_id = "b4_contract_test"
    jobs, queues = _b4_training_jobs()
    config = expected_b4_plan_config(TASK8, METADATA)
    value = RunPlan(
        schema=PLAN_SCHEMA,
        run_id=run_id,
        kind="b4_train",
        created_at="2026-07-13T00:00:00+08:00",
        source_commit="1" * 40,
        source_tree="2" * 40,
        source_archive_path="/tmp/b4.source.tar",
        source_archive_sha256="3" * 64,
        source_archive_size=1,
        inputs_archive_path="/tmp/b4.inputs.tar",
        inputs_archive_sha256="4" * 64,
        inputs_archive_size=1,
        source_inputs=(
            InputEntry("source", "pretrained/end2race.pth", CANONICAL_BC_SHA256, 1),
        ),
        inputs=(
            InputEntry(
                "task8_release",
                "task8/training_scenarios.tsv",
                sha(TASK8 / "training_scenarios.tsv"),
                (TASK8 / "training_scenarios.tsv").stat().st_size,
            ),
            InputEntry(
                "task8_release",
                "task8/development_scenarios.tsv",
                sha(TASK8 / "development_scenarios.tsv"),
                (TASK8 / "development_scenarios.tsv").stat().st_size,
            ),
            InputEntry(
                "d2_opened_episode_metadata",
                "d2/episode_metadata.tsv",
                sha(METADATA),
                METADATA.stat().st_size,
            ),
        ),
        hosts=hosts(run_id),
        jobs=jobs,
        queues=queues,
        required_cli=B4_REQUIRED_TRAIN_CLI,
        module_path_contract=MODULE_PATH_CONTRACT,
        config=config,
        collection_root="/tmp/b4_contract_collection",
    )
    return _seal_plan(value)


def evaluation_plan(parent: RunPlan) -> RunPlan:
    run_id = "b4_eval_contract_test"
    jobs, queues = _b4_eval_jobs()
    checkpoint_set = [
        {
            "seed": seed,
            "iteration": iteration,
            "relpath": f"inputs/checkpoints/seed{seed}_iter{iteration}.pth",
            "sha256": f"{seed + iteration // 10:x}" * 64,
            "size": 1,
        }
        for seed in (1,)
        for iteration in (10, 20, 30)
    ]
    scenarios = [
        {"row_index": index, "l2_id": f"L2:{index:064x}", "shard_index": index % 4}
        for index in range(288)
    ]
    variants = ["BC"] + [
        f"seed{seed}_iter{iteration}"
        for seed in (1,)
        for iteration in (10, 20, 30)
    ]
    value = RunPlan(
        schema=PLAN_SCHEMA,
        run_id=run_id,
        kind="b4_eval",
        created_at="2026-07-13T00:00:00+08:00",
        source_commit="1" * 40,
        source_tree="2" * 40,
        source_archive_path="/tmp/b4eval.source.tar",
        source_archive_sha256="3" * 64,
        source_archive_size=1,
        inputs_archive_path="/tmp/b4eval.inputs.tar",
        inputs_archive_sha256="4" * 64,
        inputs_archive_size=1,
        source_inputs=parent.source_inputs,
        inputs=parent.inputs,
        hosts=hosts(run_id),
        jobs=jobs,
        queues=queues,
        required_cli=B4_REQUIRED_EVAL_CLI,
        module_path_contract=MODULE_PATH_CONTRACT,
        config=dict(B4_EVAL_CONFIG),
        collection_root="/tmp/b4_eval_contract_collection",
        parent_plan_sha256=parent.plan_sha256,
        evaluation_contract={
            "manifest_relpath": "inputs/task8/development_scenarios.tsv",
            "manifest_sha256": sha(TASK8 / "development_scenarios.tsv"),
            "checkpoint_set": checkpoint_set,
            "checkpoint_set_sha256": hashlib.sha256(
                (json.dumps(checkpoint_set, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest(),
            "training_manifest_sha256": sha(TASK8 / "training_scenarios.tsv"),
            "shard_count": 4,
            "assignment": "physical_row_index_mod_shard_count",
            "scenarios": scenarios,
            "variants": variants,
            "expected_scenario_count": 288,
            "expected_variant_count": 4,
            "expected_episode_rows": 1152,
        },
    )
    return _seal_plan(value)


def main() -> None:
    train = training_plan()
    _verify_plan(train)
    assert [(job.job_id, job.seed, job.host_id) for job in train.jobs] == [
        ("b4-seed1", 1, "remote"),
    ]
    assert all(job.kind == "b4_training" for job in train.jobs)
    assert all("b4-pilot" in job.argv for job in train.jobs)
    assert "sidecar" not in train.config["inputs"]

    status_commands = "\n".join(
        " ".join(command)
        for command in _collect_status_commands(train, Path("/tmp/b4-collection"))
    )
    payload_commands = "\n".join(
        " ".join(command)
        for command in _collect_payload_commands(train, Path("/tmp/b4-collection"))
    )
    assert "/hosts/local/status.json" not in status_commands
    assert "/hosts/remote/status.json" in status_commands
    assert "/hosts/local/preflight.json" in status_commands
    assert "/hosts/remote/preflight.json" in status_commands
    assert "/hosts/local/outputs" not in payload_commands
    assert "/hosts/remote/outputs" in payload_commands

    drifted = replace(
        train,
        config={
            **train.config,
            "ppo": {**train.config["ppo"], "gae_lambda": 0.995},
        },
    )
    drifted = _seal_plan(replace(drifted, plan_sha256=""))
    try:
        _verify_plan(drifted)
        raise RuntimeError("B4 plan accepted a post-approval lambda change")
    except RunnerError as error:
        assert "frozen" in str(error)

    evaluation = evaluation_plan(train)
    _verify_plan(evaluation)
    assert len(evaluation.jobs) == 4
    assert all(job.kind == "b4_evaluation_shard" for job in evaluation.jobs)
    assert evaluation.evaluation_contract["expected_episode_rows"] == 1152
    assert evaluation.evaluation_contract["variants"] == [
        "BC",
        "seed1_iter10",
        "seed1_iter20",
        "seed1_iter30",
    ]

    # Learner-side authorization uses the B4 marker schema and hash-binds both
    # the reused canonical BC baseline and the B4-specific four-map smoke.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        control = root / "control"
        control.mkdir()
        baseline = {
            "schema": "bplus-v2.2-b2-bc-baseline-preflight-2",
            "integrity_passed": True,
            "passed": True,
            "acceptance_passed": True,
            "candidate_evaluated": False,
            "collision": 24,
            "terminal_overtake": 138,
        }
        plumbing = {
            "schema": "end2race-b4-plumbing-smoke-2",
            "passed": True,
            "run_plan_sha256": train.plan_sha256,
            "product_outcomes_reported_or_compared": False,
            "candidate_selection_performed": False,
            "ppo_pilot_iteration_completed": False,
        }
        baseline_path = control / "bc_baseline_preflight.json"
        plumbing_path = control / "plumbing_smoke.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        plumbing_path.write_text(json.dumps(plumbing), encoding="utf-8")
        ready = {
            "schema": "end2race-b4-ready-1",
            "passed": True,
            "run_plan_sha256": train.plan_sha256,
            "source_commit": train.source_commit,
            "source_archive_sha256": train.source_archive_sha256,
            "inputs_archive_sha256": train.inputs_archive_sha256,
            "baseline_marker_sha256": sha(baseline_path),
            "plumbing_marker_sha256": sha(plumbing_path),
        }
        (control / "READY.json").write_text(json.dumps(ready), encoding="utf-8")
        assert _validate_control_plane_ready(train.__dict__, {"root": root}) == ready
        ready["schema"] = "end2race-b2-ready-1"
        (control / "READY.json").write_text(json.dumps(ready), encoding="utf-8")
        try:
            _validate_control_plane_ready(train.__dict__, {"root": root})
            raise RuntimeError("B4 learner accepted a B2 READY marker")
        except ValueError as error:
            assert "READY" in str(error)

    print("B4 immutable control-plane contracts passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dry-run and integrity regression for the immutable experiment control plane."""

from __future__ import annotations

from contextlib import redirect_stdout
import csv
from dataclasses import replace
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "end2race_experiment_runner", ROOT / "Experiments/runner.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_runner_binds_lazy_imports_to_its_own_repo() -> None:
    assert sys.path[0] == str(ROOT)
    assert runner.REPO_ROOT == ROOT

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        staged = base / "isolated/repo"
        (staged / "Experiments").mkdir(parents=True)
        copied = staged / "Experiments/runner.py"
        shutil.copy2(ROOT / "Experiments/runner.py", copied)
        unrelated = base / "unrelated"
        unrelated.mkdir()
        code = (
            "import importlib.util,sys;"
            f"p={str(copied)!r};"
            "s=importlib.util.spec_from_file_location('staged_runner',p);"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;"
            "s.loader.exec_module(m);print(sys.path[0]);print(m.REPO_ROOT)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=unrelated,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.splitlines()
        assert lines == [str(staged), str(staged)]


def host_specs(run_id: str) -> tuple[runner.HostSpec, runner.HostSpec]:
    root = f"/home/haowei/end2race_runs/{run_id}"
    environment = {
        "python": "3.10",
        "torch": "2.7.0",
        "numpy": "2.0.0",
        "numba": "0.61.0",
        "gym": "0.19.0",
        "scipy": "1.15.0",
    }
    return (
        runner.HostSpec(
            "local",
            "local",
            root,
            runner.PINNED_PYTHON,
            ":0",
            "GPU-local-test",
            runner.LOCAL_GPU_NAME,
            None,
            environment,
        ),
        runner.HostSpec(
            "remote",
            "remote",
            root,
            runner.PINNED_PYTHON,
            ":1",
            "GPU-remote-test",
            runner.REMOTE_GPU_NAME,
            runner.REMOTE_HOST,
            environment,
        ),
    )


def training_plan(
    run_id: str, collection_root: Path, kind: str = "b2_train"
) -> runner.RunPlan:
    jobs, queues = runner._training_jobs()
    return runner._seal_plan(
        runner.RunPlan(
            schema=runner.PLAN_SCHEMA,
            run_id=run_id,
            kind=kind,
            created_at="2026-07-12T12:00:00+08:00",
            source_commit="1" * 40,
            source_tree="2" * 40,
            source_archive_path=str(collection_root.parent / "source.tar"),
            source_archive_sha256="3" * 64,
            source_archive_size=123,
            inputs_archive_path=str(collection_root.parent / "inputs.tar"),
            inputs_archive_sha256="4" * 64,
            inputs_archive_size=456,
            source_inputs=(),
            inputs=(),
            hosts=host_specs(run_id),
            jobs=jobs,
            queues=queues,
            required_cli=runner.REQUIRED_TRAIN_CLI,
            module_path_contract=runner.MODULE_PATH_CONTRACT,
            config=runner._shared_training_config(kind),
            collection_root=str(collection_root),
        )
    )


def evaluation_plan(run_id: str, collection_root: Path) -> runner.RunPlan:
    jobs, queues = runner._eval_jobs()
    scenarios = [
        {"row_index": index, "l2_id": f"L2:{index}", "shard_index": index}
        for index in range(4)
    ]
    variants = ["BC", "BC_FROZEN::seed0"]
    contract = {
        "manifest_relpath": "inputs/task8/development_scenarios.tsv",
        "manifest_sha256": "5" * 64,
        "checkpoint_set": [
            {
                "arm": "BC_FROZEN",
                "seed": 0,
                "relpath": "inputs/checkpoints/BC_FROZEN_seed0_iter20.pt",
                "sha256": "8" * 64,
                "size": 123,
            }
        ],
        "checkpoint_set_sha256": "6" * 64,
        "training_manifest_sha256": "9" * 64,
        "shard_count": 4,
        "assignment": "physical_row_index_mod_shard_count",
        "scenarios": scenarios,
        "variants": variants,
        "expected_scenario_count": 4,
        "expected_episode_rows": 8,
    }
    return runner._seal_plan(
        runner.RunPlan(
            schema=runner.PLAN_SCHEMA,
            run_id=run_id,
            kind="b2_eval",
            created_at="2026-07-12T12:30:00+08:00",
            source_commit="1" * 40,
            source_tree="2" * 40,
            source_archive_path=str(collection_root.parent / "source.tar"),
            source_archive_sha256="3" * 64,
            source_archive_size=123,
            inputs_archive_path=str(collection_root.parent / "inputs.tar"),
            inputs_archive_sha256="4" * 64,
            inputs_archive_size=456,
            source_inputs=(),
            inputs=(),
            hosts=host_specs(run_id),
            jobs=jobs,
            queues=queues,
            required_cli=("ppo-evaluate", "ppo-merge-eval"),
            module_path_contract=runner.MODULE_PATH_CONTRACT,
            config={"evaluation_offsets": [0.0, 0.0]},
            collection_root=str(collection_root),
            parent_plan_sha256="7" * 64,
            evaluation_contract=contract,
        )
    )


def test_plan_digest_and_tamper() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "plan.json"
        plan = training_plan("b2_test_digest", Path(temporary) / "collected")
        runner.write_plan(path, plan)
        assert runner.load_plan(path) == plan
        value = json.loads(path.read_text(encoding="utf-8"))
        value["config"]["iterations"] = 21
        path.write_text(json.dumps(value), encoding="utf-8")
        try:
            runner.load_plan(path)
            raise AssertionError("tampered plan was accepted")
        except runner.RunnerError as error:
            assert "digest mismatch" in str(error)


def test_learner_queues_are_complete_and_nonshardable() -> None:
    jobs, queues = runner._training_jobs()
    assert len(jobs) == 6
    assert set(queues) == {"learner-seed0-remote", "learner-seed1-local"}
    by_id = {job.job_id: job for job in jobs}
    for seed, host in ((0, "remote"), (1, "local")):
        ids = queues[f"learner-seed{seed}-{host}"]
        assert [by_id[item].arm for item in ids] == list(runner.ARMS)
        assert all(by_id[item].seed == seed for item in ids)
        assert all(by_id[item].host_id == host for item in ids)
        assert all(by_id[item].gpu_exclusive for item in ids)
        assert all(not by_id[item].shardable for item in ids)


def test_dry_run_never_accesses_remote_or_old_worktree() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        plan_path = temp / "plan.json"
        plan = training_plan("b2_test_dryrun", temp / "collection")
        runner.write_plan(plan_path, plan)
        stdout = io.StringIO()
        with mock.patch.object(
            subprocess, "run", side_effect=AssertionError("subprocess.run called")
        ), mock.patch.object(
            subprocess, "Popen", side_effect=AssertionError("subprocess.Popen called")
        ), redirect_stdout(stdout):
            assert runner.stage(plan_path, plan, ("local", "remote"), True) == 0
            assert runner.baseline_preflight(plan, True) == 0
            assert runner.preflight(plan, ("local", "remote"), True) == 0
            assert runner.plumbing_smoke(plan, True) == 0
            assert runner.execute(plan, ("local", "remote"), True) == 0
            assert runner.resume(plan, ("local", "remote"), True) == 0
            assert runner.status(plan, ("local", "remote"), True) == 0
            assert runner.collect(plan, True) == 0
        text = stdout.getvalue()
        assert "/home/haowei/end2race_runs/b2_test_dryrun" in text
        assert runner.REMOTE_HOST in text
        assert "flock -n" in text
        assert runner.PINNED_PYTHON in text
        assert "~/Documents/End2Race" not in text
        assert "/home/haowei/Documents/End2Race" not in text
        assert "b2-exploration-sweep" not in text


def test_remote_stage_is_explicit_allowlist() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        plan_path = temp / "plan.json"
        plan = training_plan("b2_test_stage", temp / "collection")
        runner.write_plan(plan_path, plan)
        commands = runner._remote_stage_commands(
            plan_path, plan, runner._host(plan, "remote")
        )
        rendered = "\n".join(runner._display_command(command) for command in commands)
        assert rendered.count("rsync") == 3
        assert "source.tar" in rendered and "inputs.tar" in rendered
        assert "run_plan.json" in rendered
        assert "--delete" not in rendered
        assert "git pull" not in rendered and "git checkout" not in rendered
        assert "~/Documents/End2Race" not in rendered
        assert "set -eu; umask 077; test ! -e" in rendered
        assert "PYTHONDONTWRITEBYTECODE=1" in rendered


def test_remote_commands_have_ssh_keepalive() -> None:
    plan = training_plan("b2_keepalive", Path("/tmp/b2_keepalive_collection"))
    command = runner._execute_command(plan, runner._host(plan, "remote"))
    assert command[:7] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=6",
    ]


def test_eval_source_delta_is_control_only() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repo = Path(temporary) / "repo"
        (repo / "Experiments").mkdir(parents=True)
        (repo / "tests").mkdir()
        (repo / "bplus_v22").mkdir()
        (repo / "Experiments/runner.py").write_text("parent\n", encoding="utf-8")
        (repo / "tests/test_experiment_runner.py").write_text("parent\n", encoding="utf-8")
        (repo / "bplus_v22/cli.py").write_text("numeric\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Runner Test"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "runner@example.invalid"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "parent"], cwd=repo, check=True)
        parent = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        (repo / "Experiments/runner.py").write_text("control fix\n", encoding="utf-8")
        (repo / "tests/test_experiment_runner.py").write_text("control test\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "control"], cwd=repo, check=True)
        control = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        assert runner._validate_eval_control_only_source_delta(repo, parent, control) == (
            "Experiments/runner.py",
            "tests/test_experiment_runner.py",
        )
        (repo / "bplus_v22/cli.py").write_text("changed numeric\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "numeric"], cwd=repo, check=True)
        numeric = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        try:
            runner._validate_eval_control_only_source_delta(repo, parent, numeric)
            raise AssertionError("numerical eval-source drift was accepted")
        except runner.RunnerError as error:
            assert "numerical/non-control" in str(error)


def test_deterministic_input_archive_and_safe_extract() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        first = temp / "first.txt"
        second = temp / "second.txt"
        first.write_text("alpha\n", encoding="utf-8")
        second.write_text("beta\n", encoding="utf-8")
        archive_a = temp / "a.tar"
        archive_b = temp / "b.tar"
        files = [
            ("test", "group/second.txt", second),
            ("test", "group/first.txt", first),
        ]
        entries_a = runner._deterministic_input_archive(archive_a, files)
        entries_b = runner._deterministic_input_archive(archive_b, files)
        assert entries_a == entries_b
        assert runner._sha256_file(archive_a) == runner._sha256_file(archive_b)
        extracted = temp / "extracted"
        runner._safe_extract(archive_a, extracted)
        assert (extracted / "group/first.txt").read_text() == "alpha\n"

        malicious = temp / "malicious.tar"
        with tarfile.open(malicious, "w") as archive:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        try:
            runner._safe_extract(malicious, temp / "bad")
            raise AssertionError("path-traversal tar was accepted")
        except runner.RunnerError as error:
            assert "unsafe tar member" in str(error)


def make_release(path: Path, files: dict[str, bytes]) -> None:
    path.mkdir(parents=True)
    lines = []
    for relpath, value in sorted(files.items()):
        target = path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
        lines.append(f"{runner._sha256_file(target)}  {relpath}")
    (path / "output_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path / "COMPLETE").write_text("complete\n", encoding="utf-8")


def test_plan_uses_clean_commit_and_explicit_input_bundles() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        repo = temp / "repo"
        repo.mkdir()
        (repo / "bplus_v22").mkdir()
        (repo / "pretrained").mkdir()
        cli = (
            "capabilities ppo-baseline-preflight ppo-pilot "
            "ppo-evaluate ppo-merge-eval ppo-plumbing-smoke\n"
        )
        (repo / "bplus_v22/cli.py").write_text(cli, encoding="utf-8")
        bc = repo / "pretrained/end2race.pth"
        bc.write_bytes(b"tiny canonical bc")
        (repo / ".gitignore").write_text("runtime_inputs/\n", encoding="utf-8")
        sidecar_rel = Path("runtime_inputs/sidecar")
        task8_rel = Path("runtime_inputs/task8")
        metadata_rel = Path("runtime_inputs/d2/episode_metadata.tsv")
        sidecar_bundle = b"tiny sidecar"
        make_release(repo / sidecar_rel, {"sidecar_bundle.pt": sidecar_bundle})
        make_release(
            repo / task8_rel,
            {
                "config.json": b"{}\n",
                "training_scenarios.tsv": b"training_order\tl2_id\n0\tL2:a\n",
                "development_scenarios.tsv": b"manifest_order\tl2_id\n0\tL2:b\n",
            },
        )
        metadata = repo / metadata_rel
        metadata.parent.mkdir(parents=True)
        metadata.write_bytes(b"l2_id\tcollision_any\nL2:a\tFalse\n")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Runner Test"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "runner@example.invalid"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
        output = temp / "control/b2_fixture_plan.json"
        environment = {"python": "fixture", "torch": "fixture", "numpy": "fixture"}
        with mock.patch.object(runner, "SIDECAR_RELEASE", sidecar_rel), mock.patch.object(
            runner, "TASK8_RELEASE", task8_rel
        ), mock.patch.object(
            runner, "CANONICAL_BC_SHA256", runner._sha256_file(bc)
        ), mock.patch.object(
            runner, "CANONICAL_SIDECAR_SHA256", runner._sha256_bytes(sidecar_bundle)
        ), mock.patch.object(
            runner, "D2_METADATA", metadata_rel
        ), mock.patch.object(
            runner, "D2_METADATA_SHA256", runner._sha256_file(metadata)
        ):
            plan = runner.build_training_plan(
                repo=repo,
                run_id="b2_fixture_plan",
                commit="HEAD",
                output=output,
                local_gpu_uuid="GPU-local-fixture",
                remote_gpu_uuid="GPU-remote-fixture",
                environment=environment,
            )
        assert output.is_file()
        assert Path(plan.source_archive_path).is_file()
        assert Path(plan.inputs_archive_path).is_file()
        assert runner.load_plan(output) == plan
        assert {entry.role for entry in plan.inputs} == {
            "sidecar_release",
            "task8_release",
            "d2_opened_episode_metadata",
        }
        assert plan.source_inputs[0].relpath == "pretrained/end2race.pth"
        with tarfile.open(plan.source_archive_path, "r") as archive:
            assert "bplus_v22/cli.py" in archive.getnames()
            assert not any(name.startswith("runtime_inputs/") for name in archive.getnames())


def write_eval_shards(plan: runner.RunPlan, collection: Path) -> None:
    assert plan.evaluation_contract is not None
    contract = plan.evaluation_contract
    fields = [
        "row_index",
        "l2_id",
        "variant_id",
        "shard_index",
        "manifest_sha256",
        "checkpoint_set_sha256",
    ]
    for shard in range(4):
        host = "local" if shard == 0 else "remote"
        directory = collection / f"hosts/{host}/outputs/eval/shard{shard}"
        directory.mkdir(parents=True)
        (directory / "COMPLETE").write_text("complete\n", encoding="utf-8")
        with (directory / "episodes.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            scenario = contract["scenarios"][shard]
            for variant in contract["variants"]:
                writer.writerow(
                    {
                        "row_index": scenario["row_index"],
                        "l2_id": scenario["l2_id"],
                        "variant_id": variant,
                        "shard_index": shard,
                        "manifest_sha256": contract["manifest_sha256"],
                        "checkpoint_set_sha256": contract["checkpoint_set_sha256"],
                    }
                )


def write_atomic_eval_release(plan: runner.RunPlan, root: Path, shard: int) -> Path:
    assert plan.evaluation_contract is not None
    contract = plan.evaluation_contract
    output = root / f"outputs/eval/shard{shard}"
    output.mkdir(parents=True)
    checkpoint_by_variant = {"BC": runner.CANONICAL_BC_SHA256}
    checkpoint_by_variant.update(
        {
            f"{item['arm']}::seed{item['seed']}": item["sha256"]
            for item in contract["checkpoint_set"]
        }
    )
    rows = []
    scenario = next(item for item in contract["scenarios"] if item["shard_index"] == shard)
    for variant in contract["variants"]:
        rows.append(
            {
                "schema": "bplus-v2.2-ppo-eval-row-1",
                "task8_row_index": scenario["row_index"],
                "l2_id": scenario["l2_id"],
                "variant": variant,
                "scenario_manifest_sha256": contract["manifest_sha256"],
                "checkpoint_manifest_sha256": contract["training_manifest_sha256"],
                "checkpoint_sha256": checkpoint_by_variant[variant],
                "external_clip_micro_steps": 0,
                "trajectory_sha256": f"{shard + 1:x}" * 64,
                "collision_any": False,
                "terminal_overtake": True,
            }
        )
    payload = {
        "schema": "bplus-v2.2-ppo-eval-shard-1",
        "shard_index": shard,
        "shard_count": 4,
        "scenario_manifest_sha256": contract["manifest_sha256"],
        "checkpoint_manifest_sha256": contract["training_manifest_sha256"],
        "bc_checkpoint_sha256": runner.CANONICAL_BC_SHA256,
        "checkpoint_sha256_by_variant": checkpoint_by_variant,
        "rows": rows,
    }
    (output / "shard.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    control_rows = [
        {
            **row,
            "row_index": row["task8_row_index"],
            "variant_id": row["variant"],
            "shard_index": shard,
            "manifest_sha256": contract["manifest_sha256"],
            "checkpoint_set_sha256": contract["checkpoint_set_sha256"],
        }
        for row in rows
    ]
    fields = sorted({name for row in control_rows for name in row})
    with (output / "episodes.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(control_rows)
    (output / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
    return output


def test_eval_resume_recovers_atomic_release_and_continues_fresh() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = evaluation_plan("b2_eval_resume", root / "collection")
        local = replace(runner._host(original, "local"), stage_root=str(root))
        remote = replace(runner._host(original, "remote"), stage_root=str(root))
        plan = replace(original, hosts=(local, remote))
        control = root / "control"
        control.mkdir()
        plan_path = control / "run_plan.json"
        plan_path.write_text("{}\n", encoding="utf-8")
        status = {
            "schema": "end2race-host-status-1",
            "plan_sha256": plan.plan_sha256,
            "host": "remote",
            "state": "FAILED",
            "jobs": {
                "eval-shard1": {"state": "COMPLETE", "exit_code": 0},
                "eval-shard2": {"state": "FAILED", "exit_code": 120},
            },
            "resume_attempts": 0,
        }
        (control / "status.json").write_text(json.dumps(status), encoding="utf-8")
        (control / "status.jsonl").write_text("{}\n", encoding="utf-8")
        write_atomic_eval_release(plan, root, 1)
        write_atomic_eval_release(plan, root, 2)

        calls = []

        def run_shard3(argv, **_kwargs):
            calls.append(argv)
            write_atomic_eval_release(plan, root, 3)
            return mock.Mock(returncode=0)

        with mock.patch.object(runner, "load_plan", return_value=plan), mock.patch.object(
            runner, "check_preflight_host"
        ), mock.patch.object(runner, "_assert_live_environment"), mock.patch.object(
            runner, "_probe_gpu"
        ), mock.patch.object(runner.subprocess, "run", side_effect=run_shard3):
            assert runner.execute_host(plan_path, "remote", resume=True) == 0
        assert len(calls) == 1 and "--resume" not in calls[0]
        final = json.loads((control / "status.json").read_text(encoding="utf-8"))
        assert final["state"] == "COMPLETE"
        assert final["jobs"]["eval-shard2"]["status_recovered_from_complete_release"] is True
        assert final["jobs"]["eval-shard2"]["prior_exit_code"] == 120
        assert final["jobs"]["eval-shard3"]["state"] == "COMPLETE"


def test_eval_resume_rejects_incomplete_shard_without_cli_resume() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = evaluation_plan("b2_eval_incomplete", root / "collection")
        local = replace(runner._host(original, "local"), stage_root=str(root))
        remote = replace(runner._host(original, "remote"), stage_root=str(root))
        plan = replace(original, hosts=(local, remote))
        control = root / "control"
        control.mkdir()
        plan_path = control / "run_plan.json"
        plan_path.write_text("{}\n", encoding="utf-8")
        status = {
            "schema": "end2race-host-status-1",
            "plan_sha256": plan.plan_sha256,
            "host": "remote",
            "state": "FAILED",
            "jobs": {
                "eval-shard1": {"state": "COMPLETE", "exit_code": 0},
                "eval-shard2": {"state": "FAILED", "exit_code": 120},
            },
            "resume_attempts": 0,
        }
        (control / "status.json").write_text(json.dumps(status), encoding="utf-8")
        (control / "status.jsonl").write_text("{}\n", encoding="utf-8")
        write_atomic_eval_release(plan, root, 1)
        partial = root / "outputs/eval/shard2.partial"
        partial.mkdir(parents=True)
        with mock.patch.object(runner, "load_plan", return_value=plan), mock.patch.object(
            runner, "check_preflight_host"
        ), mock.patch.object(runner, "_assert_live_environment"), mock.patch.object(
            runner, "_probe_gpu"
        ), mock.patch.object(
            runner.subprocess,
            "run",
            side_effect=AssertionError("incomplete eval invoked CLI resume"),
        ):
            try:
                runner.execute_host(plan_path, "remote", resume=True)
                raise AssertionError("incomplete eval shard was resumed")
            except runner.RunnerError as error:
                assert "only recovers" in str(error)


def test_eval_cartesian_merge_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        collection = Path(temporary) / "collection"
        plan = evaluation_plan("b2_test_eval", collection)
        runner._verify_plan(plan)
        jobs = {job.job_id: job for job in plan.jobs}
        assert plan.queues["eval-local"] == ("eval-shard0",)
        assert plan.queues["eval-remote-sequential"] == (
            "eval-shard1",
            "eval-shard2",
            "eval-shard3",
        )
        assert jobs["eval-shard0"].host_id == "local"
        assert all(jobs[f"eval-shard{index}"].host_id == "remote" for index in (1, 2, 3))
        write_eval_shards(plan, collection)
        result = runner.validate_eval_collection(plan, collection)
        assert result == {
            "passed": True,
            "scenario_count": 4,
            "variant_count": 2,
            "episode_rows": 8,
        }
        shard = collection / "hosts/remote/outputs/eval/shard1/episodes.tsv"
        lines = shard.read_text(encoding="utf-8").splitlines()
        shard.write_text("\n".join(lines + [lines[-1]]) + "\n", encoding="utf-8")
        try:
            runner.validate_eval_collection(plan, collection)
            raise AssertionError("duplicate eval row was accepted")
        except runner.RunnerError as error:
            assert "duplicate eval Cartesian row" in str(error)


def test_fail_closed_names_and_pinned_wrapper() -> None:
    for value in ("x", "../escape", "bad id", "a/../../b"):
        try:
            runner._validate_run_id(value)
            raise AssertionError(f"unsafe run id accepted: {value}")
        except runner.RunnerError:
            pass
    assert "b2-exploration-sweep" not in runner.LEGACY_SHOW_ONLY
    assert "b2-ppo-pilot-seed0" not in runner.LEGACY_SHOW_ONLY
    wrapper = (ROOT / "run.sh").read_text(encoding="utf-8")
    assert runner.PINNED_PYTHON in wrapper
    assert "exec python3" not in wrapper
    assert "run  <job>" not in wrapper


def test_complete_requires_atomic_release_envelope() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = training_plan("b2_test_complete", root / "collection")
        job = next(item for item in plan.jobs if item.host_id == "local")
        output = root / job.output_relpath
        output.mkdir(parents=True)
        (output / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "schema": "bplus-v2.2-b2-ppo-pilot-1",
                    "integrity_passed": True,
                    "passed": True,
                    "arm": job.arm,
                    "seed": job.seed,
                    "iterations": 20,
                    "run_plan_sha256": plan.plan_sha256,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runner._validate_job_output(plan, root, job)
        (output / "COMPLETE").unlink()
        try:
            runner._validate_job_output(plan, root, job)
            raise AssertionError("missing COMPLETE marker was accepted")
        except runner.RunnerError:
            pass


def test_b3_plan_and_final_checkpoint_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = training_plan("b3_test_contract", root / "collection", "b3_train")
        runner._verify_plan(plan)
        assert plan.kind == "b3_train"
        assert plan.config["policy_contract"] == "unified_standard_mode_v1"
        assert plan.config["iterations"] == 40
        assert plan.config["deterministic_contract"] == (
            "standard_mode_of_training_distribution"
        )
        assert plan.config["dual_freeze_through_iteration"] == 0
        assert plan.config["exploration"] == {
            "intervention_prior_probability": 0.10,
            "conditional_brake_prior_probability": 0.50,
            "external_gate_offsets_forbidden": True,
            "steer_std_scale": 0.1,
            "brake_std_scale": 1.0,
        }

        job = next(item for item in plan.jobs if item.host_id == "local")
        output = root / job.output_relpath
        output.mkdir(parents=True)
        (output / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "schema": "bplus-v2.2-b3-ppo-pilot-1",
                    "integrity_passed": True,
                    "passed": True,
                    "arm": job.arm,
                    "seed": job.seed,
                    "iterations": 40,
                    "run_plan_sha256": plan.plan_sha256,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runner._validate_job_output(plan, root, job)

        release = root / "learner"
        checkpoint = release / "checkpoints/iter_0040.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"b3 checkpoint")
        (release / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        (release / "summary.json").write_text(
            json.dumps(
                {
                    "schema": "bplus-v2.2-b3-ppo-pilot-1",
                    "integrity_passed": True,
                    "passed": True,
                    "arm": job.arm,
                    "seed": job.seed,
                    "iterations": 40,
                    "run_plan_sha256": plan.plan_sha256,
                    "final_checkpoint_sha256": runner._sha256_file(checkpoint),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runner._validate_training_checkpoint_source(
            plan, str(job.arm), int(job.seed), checkpoint
        )


def test_eval_checkpoint_requires_complete_parent_learner() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = training_plan("b2_test_checkpoint_parent", root / "collection")
        arm, seed = runner.ARMS[0], 0
        release = root / "learner"
        checkpoint = release / "checkpoints/iter_0020.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (release / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        (release / "summary.json").write_text(
            json.dumps(
                {
                    "schema": "bplus-v2.2-b2-ppo-pilot-1",
                    "integrity_passed": True,
                    "passed": True,
                    "arm": arm,
                    "seed": seed,
                    "iterations": 20,
                    "run_plan_sha256": parent.plan_sha256,
                    "iteration20_checkpoint_sha256": runner._sha256_file(checkpoint),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runner._validate_training_checkpoint_source(parent, arm, seed, checkpoint)
        (release / "COMPLETE").unlink()
        try:
            runner._validate_training_checkpoint_source(parent, arm, seed, checkpoint)
            raise AssertionError("checkpoint from incomplete learner was accepted")
        except runner.RunnerError:
            pass


def marker_fixture(root: Path) -> tuple[runner.RunPlan, dict, dict]:
    task8 = root / "inputs/task8"
    task8.mkdir(parents=True)
    maps = ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")
    development = task8 / "development_scenarios.tsv"
    with development.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "manifest_order",
                "panel",
                "l2_id",
                "l4_id",
                "map_name",
                "skill",
                "opponent_raceline",
                "speedscale_hex",
                "resolved_ego_idx",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        for index in range(288):
            writer.writerow(
                {
                    "manifest_order": index,
                    "panel": "development",
                    "l2_id": f"L2:dev:{index:03d}",
                    "l4_id": f"L4:dev:{index:03d}",
                    "map_name": maps[index % 4],
                    "skill": "skill_F",
                    "opponent_raceline": "raceline0",
                    "speedscale_hex": "0x1.0000000000000p-1",
                    "resolved_ego_idx": 100 + index,
                }
            )
    training = task8 / "training_scenarios.tsv"
    training_fields = (
        "training_order",
        "l2_id",
        "l4_id",
        "map_name",
        "skill",
        "opponent_raceline",
        "speedscale_hex",
        "resolved_ego_idx",
    )
    with training.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=training_fields, delimiter="\t")
        writer.writeheader()
        for index, map_name in enumerate(maps):
            writer.writerow(
                {
                    "training_order": index,
                    "l2_id": f"L2:train:{index}",
                    "l4_id": f"L4:train:{index}",
                    "map_name": map_name,
                    "skill": "skill_F",
                    "opponent_raceline": "raceline0",
                    "speedscale_hex": "0x1.0000000000000p-1",
                    "resolved_ego_idx": 100 + index,
                }
            )
    base = training_plan("b2_marker_fixture", root / "collection")
    plan = runner._seal_plan(
        replace(
            base,
            source_inputs=(
                runner.InputEntry(
                    "source",
                    "pretrained/end2race.pth",
                    runner.CANONICAL_BC_SHA256,
                    1,
                ),
            ),
            inputs=(
                runner.InputEntry(
                    "task8_release",
                    "task8/development_scenarios.tsv",
                    runner._sha256_file(development),
                    development.stat().st_size,
                ),
                runner.InputEntry(
                    "task8_release",
                    "task8/training_scenarios.tsv",
                    runner._sha256_file(training),
                    training.stat().st_size,
                ),
                runner.InputEntry(
                    "sidecar_release",
                    "sidecar/sidecar_bundle.pt",
                    runner.CANONICAL_SIDECAR_SHA256,
                    1,
                ),
                runner.InputEntry(
                    "d2_opened_episode_metadata",
                    "d2/episode_metadata.tsv",
                    runner.D2_METADATA_SHA256,
                    1,
                ),
            ),
            plan_sha256="",
        )
    )
    collision_by_shard = (12, 2, 5, 5)
    confirmed_by_shard = (29, 31, 32, 34)
    terminal_only_by_shard = (3, 6, 1, 2)
    baseline_rows = []
    for index in range(288):
        shard_index = index % 4
        ordinal = index // 4
        if ordinal < collision_by_shard[shard_index]:
            four_state = "collision"
        elif ordinal < (
            collision_by_shard[shard_index] + confirmed_by_shard[shard_index]
        ):
            four_state = "confirmed_pass"
        elif ordinal < (
            collision_by_shard[shard_index]
            + confirmed_by_shard[shard_index]
            + terminal_only_by_shard[shard_index]
        ):
            four_state = "terminal_overtake_only"
        else:
            four_state = "safe_follow"
        baseline_rows.append(
            {
                "task8_row_index": index,
                "l2_id": f"L2:dev:{index:03d}",
                "l4_id": f"L4:dev:{index:03d}",
                "map_name": maps[index % 4],
                "trajectory_sha256": f"{index:064x}",
                "four_state": four_state,
                "collision_any": four_state == "collision",
                "ego_collision": four_state == "collision",
                "opp_collision": False,
                "terminal_overtake": four_state
                in {"confirmed_pass", "terminal_overtake_only"},
                "confirmed_safe_pass": four_state == "confirmed_pass",
                "interaction_attempt": True,
            }
        )
    from bplus_v22.ppo_eval import BCBaselineShard, merge_bc_baseline_shards

    shards = []
    producers = runner._baseline_expected_producers(plan)
    for shard_index in range(4):
        shard_rows = tuple(
            row for row in baseline_rows if row["task8_row_index"] % 4 == shard_index
        )
        shards.append(
            BCBaselineShard(
                shard_index=shard_index,
                shard_count=4,
                run_plan_sha256=plan.plan_sha256,
                source_commit=plan.source_commit,
                source_archive_sha256=plan.source_archive_sha256,
                inputs_archive_sha256=plan.inputs_archive_sha256,
                scenario_manifest_sha256=runner._sha256_file(development),
                bc_checkpoint_sha256=runner.CANONICAL_BC_SHA256,
                producer_host_id=producers[shard_index][0],
                producer_gpu_uuid=producers[shard_index][1],
                rows=shard_rows,
                collision=sum(row["collision_any"] for row in shard_rows),
                terminal_overtake=sum(
                    row["terminal_overtake"] for row in shard_rows
                ),
            )
        )
    baseline = merge_bc_baseline_shards(
        shards=shards,
        task8_rows=runner._read_tsv_rows(development),
        run_plan_sha256=plan.plan_sha256,
        source_commit=plan.source_commit,
        source_archive_sha256=plan.source_archive_sha256,
        inputs_archive_sha256=plan.inputs_archive_sha256,
        scenario_manifest_sha256=runner._sha256_file(development),
        bc_checkpoint_sha256=runner.CANONICAL_BC_SHA256,
        expected_producers=producers,
    )
    selected = []
    for index, map_name in enumerate(maps):
        selected.append(
            {
                "training_order": index,
                "map_name": map_name,
                "l2_id": f"L2:train:{index}",
                "l4_id": f"L4:train:{index}",
                "skill": "skill_F",
                "opponent_raceline": "raceline0",
                "speedscale_hex": "0x1.0000000000000p-1",
                "resolved_ego_idx": 100 + index,
            }
        )
    arm = {
        "episode_count": 4,
        "intervention_branch_present": True,
        "joint_brake_branch_present": True,
        "steer_only_branch_present": True,
        "optimizer_update_executed": True,
        "finite_update_metrics": True,
        "preupdate_replay_tolerance": 1e-4,
        "preupdate_replay_max_abs_log_prob_delta": 1e-5,
        "preupdate_replay_max_abs_entropy_delta": 1e-6,
        "preupdate_replay_max_abs_ratio_minus_one": 1e-5,
    }
    plumbing = {
        "schema": "bplus-v2.2-b2-plumbing-smoke-1",
        "passed": True,
        "run_plan_sha256": plan.plan_sha256,
        "source_commit": plan.source_commit,
        "training_manifest_sha256": runner._sha256_file(training),
        "bc_checkpoint_sha256": runner.CANONICAL_BC_SHA256,
        "sidecar_bundle_sha256": runner.CANONICAL_SIDECAR_SHA256,
        "d2_episode_metadata_sha256": runner.D2_METADATA_SHA256,
        "scenario_selection": "first_physical_training_row_per_map_outcome_blind",
        "selected_scenarios": selected,
        "arms": {name: dict(arm) for name in runner.ARMS},
        "product_outcomes_reported_or_compared": False,
        "arm_selection_performed": False,
        "ppo_pilot_iteration_completed": False,
    }
    return plan, baseline, plumbing


def test_gate_marker_semantics_and_tamper_rejection() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, baseline, plumbing = marker_fixture(root)
        control = root / "control"
        control.mkdir()
        baseline_path = control / "bc_baseline_preflight.json"
        plumbing_path = control / "plumbing_smoke.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        plumbing_path.write_text(json.dumps(plumbing), encoding="utf-8")
        runner._validate_baseline_marker(plan, root)
        runner._validate_plumbing_marker(plan, root)
        ready_path = control / "READY.json"
        ready_path.write_text(
            json.dumps(
                {
                    "schema": "end2race-b2-ready-1",
                    "passed": True,
                    "run_plan_sha256": plan.plan_sha256,
                    "source_commit": plan.source_commit,
                    "source_archive_sha256": plan.source_archive_sha256,
                    "inputs_archive_sha256": plan.inputs_archive_sha256,
                    "baseline_marker_sha256": runner._sha256_file(baseline_path),
                    "plumbing_marker_sha256": runner._sha256_file(plumbing_path),
                }
            ),
            encoding="utf-8",
        )
        runner._validate_ready_marker(plan, root)

        baseline["rows"][1]["trajectory_sha256"] = int("1" * 64)
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        try:
            runner._validate_baseline_marker(plan, root)
            raise AssertionError("numeric trajectory digest was accepted")
        except runner.RunnerError:
            pass
        baseline["rows"][1]["trajectory_sha256"] = f"{1:064x}"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

        plumbing["selected_scenarios"][0]["training_order"] = False
        plumbing_path.write_text(json.dumps(plumbing), encoding="utf-8")
        try:
            runner._validate_plumbing_marker(plan, root)
            raise AssertionError("boolean plumbing order was accepted")
        except runner.RunnerError:
            pass
        plumbing["selected_scenarios"][0]["training_order"] = 0
        plumbing["arms"][runner.ARMS[0]]["episode_count"] = 4.0
        plumbing_path.write_text(json.dumps(plumbing), encoding="utf-8")
        try:
            runner._validate_plumbing_marker(plan, root)
            raise AssertionError("float plumbing episode count was accepted")
        except runner.RunnerError:
            pass
        plumbing["arms"][runner.ARMS[0]]["episode_count"] = 4
        plumbing_path.write_text(json.dumps(plumbing), encoding="utf-8")

        plumbing["collision_count"] = 0
        plumbing_path.write_text(json.dumps(plumbing), encoding="utf-8")
        try:
            runner._validate_plumbing_marker(plan, root)
            raise AssertionError("outcome-bearing plumbing field was accepted")
        except runner.RunnerError:
            pass
        plumbing.pop("collision_count")
        plumbing_path.write_text(json.dumps(plumbing), encoding="utf-8")

        baseline["scenario_count"] = True
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        try:
            runner._validate_baseline_marker(plan, root)
            raise AssertionError("boolean baseline count was accepted")
        except runner.RunnerError:
            pass
        baseline["scenario_count"] = 288
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

        baseline["rows"][17]["l2_id"] = "L2:tampered"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        try:
            runner._validate_baseline_marker(plan, root)
            raise AssertionError("tampered baseline L2 was accepted")
        except runner.RunnerError:
            pass
        baseline["rows"][17]["l2_id"] = "L2:dev:017"
        baseline["rows"][17]["collision_any"] = "false"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        try:
            runner._validate_baseline_marker(plan, root)
            raise AssertionError("string baseline boolean was accepted")
        except runner.RunnerError:
            pass

        plumbing["selected_scenarios"][0]["l2_id"] = "L2:wrong"
        plumbing_path.write_text(json.dumps(plumbing), encoding="utf-8")
        try:
            runner._validate_plumbing_marker(plan, root)
            raise AssertionError("wrong plumbing selection was accepted")
        except runner.RunnerError:
            pass
        plumbing["selected_scenarios"][0]["l2_id"] = "L2:train:0"
        plumbing["arms"][runner.ARMS[0]]["joint_brake_branch_present"] = False
        plumbing_path.write_text(json.dumps(plumbing), encoding="utf-8")
        try:
            runner._validate_plumbing_marker(plan, root)
            raise AssertionError("missing plumbing branch was accepted")
        except runner.RunnerError:
            pass
        plumbing["arms"][runner.ARMS[0]]["joint_brake_branch_present"] = True
        plumbing["arms"][runner.ARMS[0]][
            "preupdate_replay_max_abs_log_prob_delta"
        ] = float("nan")
        plumbing_path.write_text(json.dumps(plumbing), encoding="utf-8")
        try:
            runner._validate_plumbing_marker(plan, root)
            raise AssertionError("NaN plumbing replay delta was accepted")
        except runner.RunnerError:
            pass


def test_show_phase_order_and_partial_quarantine() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan_path = root / "plan.json"
        plan = training_plan("b2_show_order", root / "collection")
        runner.write_plan(plan_path, plan)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            runner._show(plan_path, plan)
        value = stdout.getvalue()
        phases = [
            " stage ",
            " baseline-preflight ",
            " preflight ",
            " plumbing-smoke ",
            " execute ",
            " resume ",
            " status ",
            " collect ",
        ]
        positions = [value.index(item) for item in phases]
        assert positions == sorted(positions)

        eval_plan = evaluation_plan("b2_eval_show_order", root / "eval_collection")
        eval_stdout = io.StringIO()
        with redirect_stdout(eval_stdout):
            runner._show(root / "eval_plan.json", eval_plan)
        eval_value = eval_stdout.getvalue()
        assert " resume " in eval_value
        assert eval_value.index(" execute ") < eval_value.index(" resume ")
        assert eval_value.index(" resume ") < eval_value.index(" collect ")
        assert eval_value.index(" collect ") < eval_value.index(" merge-eval ")

        collection = root / "collected"
        partial = root / "collected.partial"
        partial.mkdir()
        evidence = partial / "kept.txt"
        evidence.write_text("keep\n", encoding="utf-8")
        target = runner._quarantine_collection_partial(collection, partial)
        assert (target / "collected.partial/kept.txt").read_text() == "keep\n"
        assert not partial.exists()


def test_marker_reuse_and_atomic_install() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, baseline, _ = marker_fixture(root)
        control = root / "control"
        control.mkdir()
        incoming = control / ".baseline.incoming.json"
        incoming.write_text(json.dumps(baseline), encoding="utf-8")
        remote = replace(
            runner._host(plan, "remote"),
            stage_root=str(root),
        )
        installed_plan = replace(
            plan,
            hosts=(runner._host(plan, "local"), remote),
        )
        with mock.patch.object(runner, "check_stage_host"):
            # Call the exact implementation with a mocked plan loader so this
            # unit test exercises atomic install without constructing a staged archive.
            with mock.patch.object(runner, "load_plan", return_value=installed_plan):
                external = root / "external_marker.json"
                external.write_text(json.dumps(baseline), encoding="utf-8")
                hardlink = control / ".baseline.hardlink.json"
                os.link(external, hardlink)
                try:
                    runner.install_marker_host(
                        root / "control/run_plan.json",
                        "remote",
                        "baseline",
                        hardlink,
                    )
                    raise AssertionError("hardlinked incoming marker was accepted")
                except runner.RunnerError:
                    pass
                dangling = control / "bc_baseline_preflight.json"
                dangling.symlink_to(root / "missing_marker.json")
                try:
                    runner.install_marker_host(
                        root / "control/run_plan.json",
                        "remote",
                        "baseline",
                        incoming,
                    )
                    raise AssertionError("dangling immutable marker was overwritten")
                except runner.RunnerError:
                    pass
                dangling.unlink()
                assert runner.install_marker_host(
                    root / "control/run_plan.json", "remote", "baseline", incoming
                ) == 0
                destination = control / "bc_baseline_preflight.json"
                original_sha = runner._sha256_file(destination)
                retry = control / ".baseline.retry.json"
                retry.write_bytes(destination.read_bytes())
                assert runner.install_marker_host(
                    root / "control/run_plan.json", "remote", "baseline", retry
                ) == 0
                assert not retry.exists()
                tampered = control / ".baseline.tampered.json"
                changed = json.loads(destination.read_text(encoding="utf-8"))
                changed["rows"][0]["l2_id"] = "L2:wrong"
                tampered.write_text(json.dumps(changed), encoding="utf-8")
                try:
                    runner.install_marker_host(
                        root / "control/run_plan.json",
                        "remote",
                        "baseline",
                        tampered,
                    )
                    raise AssertionError("tampered incoming marker was installed")
                except runner.RunnerError:
                    pass
                assert runner._sha256_file(destination) == original_sha

        local = replace(runner._host(plan, "local"), stage_root=str(root))
        reuse_plan = replace(plan, hosts=(local, remote))
        commands = []
        with mock.patch.object(
            runner, "_run_command", side_effect=lambda argv, dry_run: commands.append(argv) or 0
        ):
            assert runner.baseline_preflight(reuse_plan, False) == 0
        rendered = "\n".join(runner._display_command(command) for command in commands)
        assert "_baseline-host" not in rendered
        assert "_check-stage-host" in rendered
        assert "_install-marker" in rendered


def test_baseline_topology_commands_and_terminal_failure() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = training_plan("b2_baseline_topology", root / "collection")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            assert runner.baseline_preflight(plan, True) == 0
        rendered = stdout.getvalue()
        assert rendered.count("_baseline-host") == 2
        assert "--host local" in rendered and "--host remote" in rendered
        assert rendered.count("_install-baseline-shard") == 3
        assert rendered.count("_merge-baseline-host") == 1
        assert "shard_1.json" in rendered
        assert "shard_2.json" in rendered
        assert "shard_3.json" in rendered
        assert rendered.count("flock -n") == 2

        local = replace(runner._host(plan, "local"), stage_root=str(root))
        remote = replace(runner._host(plan, "remote"), stage_root=str(root))
        terminal_plan = replace(plan, hosts=(local, remote))
        control = root / "control"
        control.mkdir()
        failed = control / "bc_baseline_preflight.failed.json"
        failed.write_text("{}\n", encoding="utf-8")
        with mock.patch.object(runner, "_run_command", return_value=0), mock.patch.object(
            runner, "_validate_baseline_marker"
        ), mock.patch.object(
            runner.subprocess,
            "Popen",
            side_effect=AssertionError("terminal failure reran simulator"),
        ):
            assert runner.baseline_preflight(terminal_plan, False) == 2


def test_remote_baseline_shard_atomic_install() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, baseline, _ = marker_fixture(root)
        local = replace(runner._host(plan, "local"), stage_root=str(root))
        installed_plan = replace(
            plan, hosts=(local, runner._host(plan, "remote"))
        )
        summary = next(item for item in baseline["shards"] if item["shard_index"] == 1)
        rows = []
        for row in baseline["rows"]:
            if row["baseline_shard_index"] != 1:
                continue
            raw = dict(row)
            raw.pop("baseline_shard_index")
            raw.pop("producer_host_id")
            raw.pop("producer_gpu_uuid")
            rows.append(raw)
        shard = {
            "schema": "bplus-v2.2-b2-bc-baseline-shard-1",
            "shard_index": 1,
            "shard_count": 4,
            "run_plan_sha256": installed_plan.plan_sha256,
            "source_commit": installed_plan.source_commit,
            "source_archive_sha256": installed_plan.source_archive_sha256,
            "inputs_archive_sha256": installed_plan.inputs_archive_sha256,
            "scenario_manifest_sha256": baseline["scenario_manifest_sha256"],
            "bc_checkpoint_sha256": runner.CANONICAL_BC_SHA256,
            "producer_host_id": summary["producer_host_id"],
            "producer_gpu_uuid": summary["producer_gpu_uuid"],
            "opened_development_only": True,
            "candidate_evaluated": False,
            "scenario_count": 72,
            "collision": summary["collision"],
            "terminal_overtake": summary["terminal_overtake"],
            "rows": rows,
        }
        control = root / "control"
        control.mkdir(exist_ok=True)
        incoming = control / ".shard_1.incoming.json"
        incoming.write_text(json.dumps(shard), encoding="utf-8")
        with mock.patch.object(runner, "load_plan", return_value=installed_plan), mock.patch.object(
            runner, "check_stage_host"
        ):
            assert runner.install_baseline_shard(
                root / "control/run_plan.json", "local", 1, incoming
            ) == 0
            destination = runner._baseline_shard_path(root, 1)
            assert destination.is_file() and destination.stat().st_nlink == 1
            original_sha = runner._sha256_file(destination)
            retry = control / ".shard_1.retry.json"
            retry.write_bytes(destination.read_bytes())
            assert runner.install_baseline_shard(
                root / "control/run_plan.json", "local", 1, retry
            ) == 0
            assert not retry.exists()
            changed = dict(shard)
            changed["producer_gpu_uuid"] = "GPU-wrong"
            tampered = control / ".shard_1.tampered.json"
            tampered.write_text(json.dumps(changed), encoding="utf-8")
            try:
                runner.install_baseline_shard(
                    root / "control/run_plan.json", "local", 1, tampered
                )
                raise AssertionError("tampered baseline shard was installed")
            except runner.RunnerError:
                pass
            assert runner._sha256_file(destination) == original_sha


def test_collect_status_first_failure_then_clean_retry() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        collection = root / "collection"
        plan = training_plan("b2_collect_retry", collection)

        def incomplete_status(_command, check=False):
            partial = root / "collection.partial"
            for host_id in ("local", "remote"):
                directory = partial / f"hosts/{host_id}"
                jobs = {
                    job.job_id: {"state": "COMPLETE"}
                    for job in runner._host_jobs(plan, host_id)
                }
                state = "RUNNING" if host_id == "remote" else "COMPLETE"
                (directory / "status.json").write_text(
                    json.dumps(
                        {
                            "plan_sha256": plan.plan_sha256,
                            "host": host_id,
                            "state": state,
                            "jobs": jobs,
                        }
                    ),
                    encoding="utf-8",
                )
                (directory / "status.jsonl").write_text(
                    json.dumps({"event": "host_complete"}) + "\n",
                    encoding="utf-8",
                )
                (directory / "STAGED").write_text(
                    plan.plan_sha256 + "\n", encoding="utf-8"
                )
                (directory / "preflight.json").write_text(
                    json.dumps(
                        {
                            "schema": "end2race-host-preflight-1",
                            "plan_sha256": plan.plan_sha256,
                            "host": host_id,
                        }
                    ),
                    encoding="utf-8",
                )
            return mock.Mock(returncode=0)

        with mock.patch.object(
            runner, "_collect_status_commands", return_value=[["status"]]
        ), mock.patch.object(
            runner, "_collect_payload_commands", return_value=[["payload"]]
        ), mock.patch.object(
            runner.subprocess, "run", side_effect=incomplete_status
        ) as run_mock:
            with redirect_stdout(io.StringIO()):
                try:
                    runner.collect(plan, False)
                    raise AssertionError("incomplete remote status was collected")
                except runner.RunnerError:
                    pass
        assert run_mock.call_count == 1
        failure = json.loads(
            (root / "collection.partial/failure.json").read_text(encoding="utf-8")
        )
        assert failure["phase"] == "status_validation"
        assert failure["plan_sha256"] == plan.plan_sha256

        with mock.patch.object(
            runner, "_collect_status_commands", return_value=[["status"]]
        ), mock.patch.object(
            runner, "_collect_payload_commands", return_value=[["payload"]]
        ), mock.patch.object(
            runner.subprocess, "run", return_value=mock.Mock(returncode=0)
        ), mock.patch.object(
            runner, "_validate_collected_statuses"
        ), mock.patch.object(
            runner, "_validate_job_output"
        ), mock.patch.object(
            runner, "_validate_baseline_marker"
        ), mock.patch.object(
            runner, "_validate_collected_baseline_shards"
        ), mock.patch.object(
            runner, "_validate_plumbing_marker"
        ), mock.patch.object(
            runner, "_validate_ready_marker"
        ), mock.patch.object(
            runner, "_sha256_file", return_value="a" * 64
        ):
            with redirect_stdout(io.StringIO()):
                assert runner.collect(plan, False) == 0
        assert collection.is_dir()
        attempts = root / "collection.attempt_failures/attempt_001"
        assert (attempts / "collection.partial/failure.json").is_file()


def test_baseline_failure_collection_is_control_only() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = training_plan("b2_failed_collection", root / "collection")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            assert runner.collect_baseline_failure(plan, True) == 0
        rendered = stdout.getvalue()
        assert "bc_baseline_preflight.failed.json" in rendered
        assert "baseline_shards" in rendered
        assert "development_scenarios.tsv" in rendered
        assert rendered.count("STAGED") == 4
        assert "status.json" not in rendered
        assert "READY.json" not in rendered
        assert "/outputs" not in rendered

        local = replace(runner._host(plan, "local"), stage_root=str(root))
        terminal_plan = replace(
            plan, hosts=(local, runner._host(plan, "remote"))
        )
        (root / "control").mkdir()
        (root / "control/bc_baseline_preflight.failed.json").write_text(
            "{}\n", encoding="utf-8"
        )
        with mock.patch.object(
            runner, "collect_baseline_failure", return_value=7
        ) as failure_collect:
            assert runner.collect(terminal_plan, False) == 7
        failure_collect.assert_called_once_with(terminal_plan, False)


def test_extracted_source_drift_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "control").mkdir()
        (root / "repo").mkdir()
        source = root / "repo/module.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        with tarfile.open(root / "control/source.tar", "w") as archive:
            archive.add(source, arcname="module.py", recursive=False)
        runner._verify_extracted_source_tree(root)
        source.write_text("VALUE = 2\n", encoding="utf-8")
        try:
            runner._verify_extracted_source_tree(root)
            raise AssertionError("post-preflight source drift was accepted")
        except runner.RunnerError as error:
            assert "source digest drift" in str(error)


def test_stage_publication_is_required() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = training_plan("b2_stage_gate", root / "collection")
        local = replace(runner._host(plan, "local"), stage_root=str(root))
        plan = replace(plan, hosts=(local, runner._host(plan, "remote")))
        with mock.patch.object(runner, "load_plan", return_value=plan), mock.patch.object(
            runner, "_verify_staged_files"
        ), mock.patch.object(runner, "_verify_extracted_source_tree"):
            try:
                runner.check_stage_host(root / "control/run_plan.json", "local")
                raise AssertionError("missing STAGED publication was accepted")
            except runner.RunnerError as error:
                assert "not atomically staged" in str(error)
            (root / "control").mkdir()
            (root / "control/STAGED").write_text(
                plan.plan_sha256 + "\n", encoding="utf-8"
            )
            assert runner.check_stage_host(
                root / "control/run_plan.json", "local"
            ) == 0
        with mock.patch.object(
            runner,
            "_critical_environment",
            return_value={**local.expected_environment, "torch": "drifted"},
        ):
            try:
                runner._assert_live_environment(local)
                raise AssertionError("live environment drift was accepted")
            except runner.RunnerError as error:
                assert "environment drift" in str(error)


def test_critical_environment_normalizes_python_patch() -> None:
    environment = runner._critical_environment(runner.PINNED_PYTHON)
    assert environment["python"] == f"{sys.version_info.major}.{sys.version_info.minor}"


def test_runner_and_evaluator_baseline_contract_cannot_drift() -> None:
    import bplus_v22.ppo_eval as ppo_eval

    with tempfile.TemporaryDirectory() as temporary:
        plan = training_plan("b2_contract_consistency", Path(temporary) / "collection")
        assert len(runner._baseline_expected_producers(plan)) == runner.SHARD_COUNT
        with mock.patch.object(
            ppo_eval,
            "EXPECTED_BC_COLLISIONS_BY_SHARD",
            (13, 2, 5, 4),
        ):
            try:
                runner._baseline_expected_producers(plan)
                raise AssertionError("runner/evaluator baseline drift was accepted")
            except runner.RunnerError as error:
                assert "acceptance contract drift" in str(error)


def main() -> None:
    test_runner_binds_lazy_imports_to_its_own_repo()
    test_plan_digest_and_tamper()
    test_learner_queues_are_complete_and_nonshardable()
    test_dry_run_never_accesses_remote_or_old_worktree()
    test_remote_stage_is_explicit_allowlist()
    test_remote_commands_have_ssh_keepalive()
    test_eval_source_delta_is_control_only()
    test_deterministic_input_archive_and_safe_extract()
    test_plan_uses_clean_commit_and_explicit_input_bundles()
    test_eval_cartesian_merge_contract()
    test_eval_resume_recovers_atomic_release_and_continues_fresh()
    test_eval_resume_rejects_incomplete_shard_without_cli_resume()
    test_fail_closed_names_and_pinned_wrapper()
    test_complete_requires_atomic_release_envelope()
    test_b3_plan_and_final_checkpoint_contract()
    test_eval_checkpoint_requires_complete_parent_learner()
    test_gate_marker_semantics_and_tamper_rejection()
    test_show_phase_order_and_partial_quarantine()
    test_marker_reuse_and_atomic_install()
    test_baseline_topology_commands_and_terminal_failure()
    test_remote_baseline_shard_atomic_install()
    test_collect_status_first_failure_then_clean_retry()
    test_baseline_failure_collection_is_control_only()
    test_extracted_source_drift_is_rejected()
    test_stage_publication_is_required()
    test_critical_environment_normalizes_python_patch()
    test_runner_and_evaluator_baseline_contract_cannot_drift()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dry-run and integrity regression for the immutable experiment control plane."""

from __future__ import annotations

from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
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


def host_specs(run_id: str) -> tuple[runner.HostSpec, runner.HostSpec]:
    root = f"/home/haowei/end2race_runs/{run_id}"
    environment = {
        "python": "3.10.19",
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


def training_plan(run_id: str, collection_root: Path) -> runner.RunPlan:
    jobs, queues = runner._training_jobs()
    return runner._seal_plan(
        runner.RunPlan(
            schema=runner.PLAN_SCHEMA,
            run_id=run_id,
            kind="b2_train",
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
            config=runner._shared_training_config(),
            collection_root=str(collection_root),
        )
    )


def evaluation_plan(run_id: str, collection_root: Path) -> runner.RunPlan:
    jobs, queues = runner._eval_jobs()
    scenarios = [
        {"row_index": index, "l2_id": f"L2:{index}", "shard_index": index}
        for index in range(4)
    ]
    variants = ["BC", "BC_FROZEN_seed0"]
    contract = {
        "manifest_relpath": "inputs/task8/development_scenarios.tsv",
        "manifest_sha256": "5" * 64,
        "checkpoint_set": [],
        "checkpoint_set_sha256": "6" * 64,
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
            "ppo-evaluate ppo-merge-eval\n"
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


def main() -> None:
    test_plan_digest_and_tamper()
    test_learner_queues_are_complete_and_nonshardable()
    test_dry_run_never_accesses_remote_or_old_worktree()
    test_remote_stage_is_explicit_allowlist()
    test_deterministic_input_archive_and_safe_extract()
    test_plan_uses_clean_commit_and_explicit_input_bundles()
    test_eval_cartesian_merge_contract()
    test_fail_closed_names_and_pinned_wrapper()
    test_complete_requires_atomic_release_envelope()
    test_eval_checkpoint_requires_complete_parent_learner()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

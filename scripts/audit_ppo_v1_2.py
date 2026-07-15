#!/usr/bin/env python3
"""Independent read-only integrity audit for completed PPO V1.2 results."""

from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from experiments.ppo_v1_2.aggregate import TOP_COUNTS, global_aggregate
from experiments.ppo_v1_2.config_schema import LEGAL_STATUSES, STAGES
from experiments.ppo_v1_2.experiment_spec import BC_SHA256, PROJECT_ROOT, austin_asset_hashes, canonical_hash, file_sha256
from experiments.ppo_v1_2.registry import validate_manifest
from experiments.ppo_v1_2.result_schema import validate_evaluation
from experiments.ppo_v1_2.runner import materialize_arm
from experiments.ppo_v1_2.selectors import rank_arms, select_checkpoint, select_top
from model import End2Race
from rl.sb3_end2race_policy import CRITIC_PROFILES, END2RACE_OBSERVATION_SIZE, End2RaceGRUPolicy, NOOP_SPEED_BOUND
from utils import atomic_write_json


class Auditor:
    def __init__(self, root: Path, manifest_path: Path, output: Path) -> None:
        self.root = root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.output = output.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.checks: list[dict[str, Any]] = []
        self.mismatches: list[dict[str, Any]] = []
        self.results: dict[str, dict[str, Any]] = {}
        for path in sorted(self.root.glob("*/**/attempt_*/run_result.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            previous = self.results.get(row["arm_id"])
            if previous is None or int(row["attempt"]) > int(previous["attempt"]):
                self.results[row["arm_id"]] = row
        self.selections = {
            stage: json.loads((self.root / stage / "stage_selection.json").read_text(encoding="utf-8"))
            for stage in STAGES
            if (self.root / stage / "stage_selection.json").is_file()
        }

    def record(self, name: str, passed: bool, path: str, expected: Any, actual: Any) -> None:
        row = {"check": name, "status": "PASS" if passed else "FAIL", "path": path, "expected": expected, "actual": actual}
        self.checks.append(row)
        if not passed:
            self.mismatches.append(row)

    def guarded(self, name: str, path: str, operation: Callable[[], None]) -> None:
        before = len(self.mismatches)
        try:
            operation()
        except Exception as error:
            self.record(name, False, path, "no exception", f"{type(error).__name__}: {error}")
            return
        if len(self.mismatches) == before:
            self.record(name, True, path, "PASS", "PASS")

    def audit_identity(self) -> None:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        self.record("experiment_head", head == self.manifest["experiment_head"], str(self.manifest_path), self.manifest["experiment_head"], head)
        bc = file_sha256(PROJECT_ROOT / "pretrained/end2race.pth")
        self.record("canonical_bc_hash", bc == BC_SHA256 == self.manifest["canonical_bc_sha256"], "pretrained/end2race.pth", BC_SHA256, bc)
        assets = austin_asset_hashes()
        self.record("austin_asset_hashes", assets == self.manifest.get("austin_asset_hashes"), str(self.manifest_path), self.manifest.get("austin_asset_hashes"), assets)

    def audit_manifest_and_terminals(self) -> None:
        validate_manifest(self.manifest)
        self.record("manifest_arm_count", len(self.manifest["arms"]) == 125, str(self.manifest_path), 125, len(self.manifest["arms"]))
        for arm in self.manifest["arms"]:
            status = arm["status"]
            legal_terminal = status in set(LEGAL_STATUSES) - {"PENDING", "RUNNING"}
            self.record("arm_terminal_status", legal_terminal, arm["arm_id"], "legal terminal status", status)
            self.record("attempt_count", 0 <= int(arm["attempt_count"]) <= 2, arm["arm_id"], "0..2", arm["attempt_count"])
            result = self.results.get(arm["arm_id"])
            if status in {"COMPLETED", "FAILED"}:
                self.record("terminal_result_present", result is not None, arm["arm_id"], True, result is not None)
            if result and status == "COMPLETED":
                self.record("resolved_config_hash", canonical_hash(result["resolved_config"]) == arm["config_hash"] == result["config_hash"], arm["arm_id"], arm["config_hash"], canonical_hash(result["resolved_config"]))

    @staticmethod
    def _optimizer_max_step(model_zip: Path) -> int:
        with zipfile.ZipFile(model_zip) as archive:
            state = torch.load(BytesIO(archive.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
        steps = []
        for parameter_state in state["state"].values():
            if "step" in parameter_state:
                value = parameter_state["step"]
                steps.append(int(value.item()) if isinstance(value, torch.Tensor) else int(value))
        if not steps or len(set(steps)) != 1:
            raise ValueError(f"Optimizer state does not contain one common Adam step: {model_zip}")
        return steps[0]

    def audit_checkpoints(self) -> None:
        for result in self.results.values():
            if result["status"] != "COMPLETED":
                continue
            attempt = self.root / result["stage"] / result["arm_id"] / f"attempt_{result['attempt']}"
            for checkpoint in result["checkpoints"]:
                actor_path = attempt / checkpoint["checkpoint"]
                state = torch.load(actor_path, map_location="cpu", weights_only=True)
                strict = len(state) == 12
                if strict:
                    loaded = End2Race(mask_prob=0.0, hidden_scale=4).load_state_dict(state, strict=True)
                    strict = not loaded.missing_keys and not loaded.unexpected_keys
                self.record("actor_12_key_strict_load", strict, str(actor_path), "12-key strict load", len(state))
                try:
                    validate_evaluation(checkpoint["metrics"])
                    valid = True
                except Exception as error:
                    valid = False
                    detail = str(error)
                self.record("checkpoint_600_case_evaluation", valid, str(actor_path), "600/0 error/exclusive", "PASS" if valid else detail)
            recomputed = select_checkpoint(result["checkpoints"])
            expected_update = result["selected_checkpoint"]["update"] if result.get("selected_checkpoint") else None
            actual_update = recomputed["update"] if recomputed else None
            self.record("checkpoint_selection_tuple", expected_update == actual_update, result["arm_id"], expected_update, actual_update)
            metrics_path = attempt / "training_metrics.jsonl"
            training = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
            frozen = all(
                update["parameter_deltas_from_fresh_start"][name]["max_abs_delta"] == 0.0
                for update in training
                for name in ("perception", "speed_mlp", "dummy_embedding", "log_std")
            )
            self.record("frozen_parameter_delta", frozen, str(metrics_path), 0.0, "0.0" if frozen else "nonzero")
            if result["resolved_config"].get("target_kl") is not None:
                last = max(result["checkpoints"], key=lambda row: row["update"])
                model_zip = attempt / "checkpoints" / f"update_{last['update']:04d}" / "model.zip"
                adam_step = self._optimizer_max_step(model_zip)
                recorded = sum(int(row["actual_optimizer_steps"]) for row in training[: int(last["update"])])
                self.record("target_kl_adam_steps", adam_step == recorded, str(model_zip), recorded, adam_step)
            summary = json.loads((attempt / "run_summary.json").read_text(encoding="utf-8"))
            visit_counts = summary.get("hard_pool_visit_counts", {})
            sampling_ok = summary.get("hard_pool_id") == result["hard_pool_id"] and summary.get("hard_sampling_mode") == result["hard_sampling_mode"] and isinstance(visit_counts, dict)
            self.record("hard_pool_sampling_evidence", sampling_ok, str(attempt / "run_summary.json"), [result["hard_pool_id"], result["hard_sampling_mode"], "visit_counts"], [summary.get("hard_pool_id"), summary.get("hard_sampling_mode"), type(visit_counts).__name__])
            pool_path = self.root / "hard_pools" / "pools" / f"{result['hard_pool_id']}.json"
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            pool_hash = canonical_hash({key: value for key, value in pool.items() if key != "manifest_hash"})
            self.record("hard_pool_id_hash", pool_hash == pool["manifest_hash"] == result["hard_pool_hash"], str(pool_path), result["hard_pool_hash"], pool_hash)

    def audit_actor_contract(self) -> None:
        actor_space = spaces.Box(-np.inf, np.inf, shape=(END2RACE_OBSERVATION_SIZE,), dtype=np.float32)
        action_space = spaces.Box(np.asarray([-0.52, -NOOP_SPEED_BOUND], dtype=np.float32), np.asarray([0.52, NOOP_SPEED_BOUND], dtype=np.float32), dtype=np.float32)
        actor_obs = torch.linspace(0.1, 4.0, steps=2 * END2RACE_OBSERVATION_SIZE).reshape(2, END2RACE_OBSERVATION_SIZE)
        pair = (torch.zeros(1, 2, 1680), torch.zeros(1, 2, 1680))
        rnn_states = RNNStates(pair, pair)
        starts = torch.zeros(2)
        reference = None
        for profile in CRITIC_PROFILES:
            observation_space = spaces.Dict({"actor": actor_space, "critic": spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)}) if profile == "C3_PRIVILEGED_PHYSICAL" else actor_space
            instance = End2RaceGRUPolicy(observation_space, action_space, lambda _: 1.0, optimizer_profile="ppo_v1", critic_profile=profile)
            observation = {"actor": actor_obs, "critic": torch.zeros(2, 12)} if profile == "C3_PRIVILEGED_PHYSICAL" else actor_obs
            action, _value, _log_prob, next_states = instance.forward(observation, rnn_states, starts, deterministic=True)
            current = (action.detach(), next_states.pi[0].detach())
            if reference is None:
                reference = current
            passed = torch.equal(reference[0], current[0]) and torch.equal(reference[1], current[1])
            self.record("critic_actor_action_hidden_contract", passed, profile, "bitwise C0 identity", "PASS" if passed else "different")

    def audit_stages(self) -> None:
        for stage in STAGES:
            stage_results = json.loads((self.root / stage / "stage_results.json").read_text(encoding="utf-8"))
            ranked = rank_arms(stage_results)
            stored_rank = json.loads((self.root / stage / "stage_rank.json").read_text(encoding="utf-8"))
            actual_ids = [row["arm_id"] for row in ranked]
            expected_ids = [row["arm_id"] for row in stored_rank]
            self.record("stage_rank_tuple", actual_ids == expected_ids, str(self.root / stage / "stage_rank.json"), expected_ids, actual_ids)
            selected = [row["arm_id"] for row in select_top(stage_results, TOP_COUNTS[stage])]
            stored = self.selections[stage]["selected_arm_ids"]
            self.record("stage_top_k_selection", selected == stored, str(self.root / stage / "stage_selection.json"), stored, selected)

        by_id = {arm["arm_id"]: arm for arm in self.manifest["arms"]}
        for arm_id, result in self.results.items():
            expected = materialize_arm(by_id[arm_id], self.selections, self.manifest["hard_pool_hashes"])
            self.record("stage_dependency_chain", expected["config_hash"] == result["config_hash"], arm_id, expected["config_hash"], result["config_hash"])

    def audit_global_bitwise(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ppo-v1-2-audit-") as directory:
            temporary = Path(directory)
            results = []
            for arm in self.manifest["arms"]:
                result = self.results.get(arm["arm_id"])
                if result is None:
                    result = {"arm_id": arm["arm_id"], "stage": arm["stage"], "status": arm["status"], "attempt": arm["attempt_count"], "resolved_config": arm["resolved_config"], "config_hash": arm["config_hash"], "metadata": arm.get("metadata", {}), "checkpoints": [], "selected_checkpoint": None}
                results.append(result)
            global_aggregate(temporary, results, self.selections)
            names = ("GLOBAL_RUNS.tsv", "GLOBAL_CHECKPOINTS.tsv", "GLOBAL_FAILURES.tsv", "GLOBAL_SELECTIONS.json", "FINAL_REPEATABILITY.tsv", "EXPERIMENT_COMPLETION.json")
            for name in names:
                expected = (self.root / name).read_bytes()
                actual = (temporary / name).read_bytes()
                self.record("global_bitwise_regeneration", expected == actual, str(self.root / name), file_sha256(self.root / name), file_sha256(temporary / name))

    def audit_report_trace(self) -> None:
        report = (self.root / "REPORT.md").read_text(encoding="utf-8")
        completion = json.loads((self.root / "EXPERIMENT_COMPLETION.json").read_text(encoding="utf-8"))
        expected_lines = [f"| {status} | {count} |" for status, count in sorted(completion["status_counts"].items())]
        for stage in STAGES:
            ranks = json.loads((self.root / stage / "stage_rank.json").read_text(encoding="utf-8"))
            expected_lines.extend(f"| {row['rank']} | {row['arm_id']} | {row.get('selected_ego_collision')} | {row.get('selected_overtake')} |" for row in ranks)
        missing = [line for line in expected_lines if line not in report]
        self.record("report_number_traceability", not missing, str(self.root / "REPORT.md"), "all status/rank rows", missing[:10])

    def run(self) -> dict[str, Any]:
        self.guarded("identity_group", str(self.manifest_path), self.audit_identity)
        self.guarded("manifest_terminal_group", str(self.manifest_path), self.audit_manifest_and_terminals)
        self.guarded("checkpoint_group", str(self.root), self.audit_checkpoints)
        self.guarded("actor_contract_group", "rl/sb3_end2race_policy.py", self.audit_actor_contract)
        self.guarded("stage_group", str(self.root), self.audit_stages)
        self.guarded("global_group", str(self.root), self.audit_global_bitwise)
        self.guarded("report_group", str(self.root / "REPORT.md"), self.audit_report_trace)
        result = {"status": "PASS" if not self.mismatches else "FAIL", "check_count": len(self.checks), "mismatch_count": len(self.mismatches), "checks": self.checks}
        self.output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.output / "AUDIT_RESULTS.json", result)
        columns = ("check", "status", "path", "expected", "actual")
        with (self.output / "MISMATCHES.tsv").open("w", encoding="utf-8") as handle:
            handle.write("\t".join(columns) + "\n")
            for row in self.mismatches:
                handle.write("\t".join(json.dumps(row[column], sort_keys=True, separators=(",", ":")) for column in columns) + "\n")
        lines = ["# PPO V1.2 AUDIT REPORT", "", "| check | status | path | expected | actual |", "|---|---|---|---|---|"]
        for row in self.checks:
            lines.append(f"| {row['check']} | {row['status']} | `{row['path']}` | `{json.dumps(row['expected'], sort_keys=True)}` | `{json.dumps(row['actual'], sort_keys=True)}` |")
        (self.output / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/ppo_v1_2"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/ppo_v1_2/sweep_manifest.runtime.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_v1_2/audit"))
    args = parser.parse_args()
    result = Auditor(args.run_root, args.manifest, args.output).run()
    print(json.dumps({key: result[key] for key in ("status", "check_count", "mismatch_count")}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

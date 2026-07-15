#!/usr/bin/env python3
"""Structural checks for replacement hierarchical Task 9 and Task 10."""

import copy
from pathlib import Path
import tempfile

import torch
import torch.nn as nn

from bplus_v22 import ARM_BC_FROZEN
from bplus_v22.hierarchical_checkpoint_preflight import (
    HierarchicalForcedZeroActor,
    _write_output_manifest,
    load_hierarchical_warmstart_release,
)
from bplus_v22.hierarchical_closed_loop import (
    HierarchicalClosedLoopActor,
    MODE_BRAKE_OFF,
    MODE_FULL,
    MODE_STEER_OFF,
)
from bplus_v22.release import file_sha256
from bplus_v22.remediated_model import (
    ACTION_SCHEMA,
    CHECKPOINT_SCHEMA,
    RemediatedV22Policy,
)
from bplus_v22.sidecar import _tensor_digest
from model import End2Race


class StubAlarm(nn.Module):
    def evaluate(self, fold, lidar_history, bc_feature, scalar_history):
        assert fold == 2
        assert lidar_history.shape == (1, 8, 360)
        assert bc_feature.shape == (1, 1680)
        assert scalar_history.shape == (1, 24)
        return {"raw": 0.2, "calibrated": 0.3, "threshold": 0.25, "alarm": True}


def _run_actor(actor, lidar, speed):
    hidden = torch.zeros(1, 1, actor.gru.hidden_size)
    outputs = []
    with torch.no_grad():
        for step in range(len(lidar)):
            actor.observe_actual_speed(float(speed[step].item()) + 0.125)
            action, hidden = actor(lidar[step], speed[step], hidden)
            actor.observe_applied_command(
                float(action[0, -1, 0].item()), float(action[0, -1, 1].item())
            )
            outputs.append(action.detach().clone())
    return outputs


def _make_release(path: Path, policy: RemediatedV22Policy, schema: str) -> str:
    (path / "checkpoints").mkdir(parents=True)
    state = copy.deepcopy(policy.state_dict())
    state_sha = _tensor_digest(state.items())
    checkpoint = path / "checkpoints" / f"{ARM_BC_FROZEN}.pt"
    torch.save(
        {
            "schema": schema,
            "action_schema": ACTION_SCHEMA,
            "release_label": "synthetic-test-only",
            "arm": ARM_BC_FROZEN,
            "manifest_output_sha256": "1" * 64,
            "task6_acceptance_passed": True,
            "calibration_offset_float32": float(state["intervention_logit_offset"].item()),
            "state_dict": state,
            "state_dict_sha256": state_sha,
        },
        checkpoint,
    )
    # The loader requires all real arms.  Reuse this structurally valid state
    # while changing the arm envelope, exactly as a tiny synthetic fixture.
    reports = {}
    from bplus_v22 import ARMS

    for arm in ARMS:
        arm_path = path / "checkpoints" / f"{arm}.pt"
        if arm != ARM_BC_FROZEN:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["arm"] = arm
            torch.save(payload, arm_path)
        digest = file_sha256(arm_path)
        reports[arm] = {"checkpoint_sha256": digest}
    (path / "config.json").write_text(
        __import__("json").dumps(
            {
                "task6_acceptance_passed": True,
                "ppo_checkpoint_eligible": True,
                "action_schema": ACTION_SCHEMA,
                "checkpoint_schema": CHECKPOINT_SCHEMA,
                "manifest_output_sha256": "1" * 64,
                "ppo_training_started": False,
                "closed_loop_evaluation_started": False,
                "arm_selection_performed": False,
                "test_opened": False,
                "final_pool": False,
                "reports": reports,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "validation.json").write_text("{}\n", encoding="utf-8")
    _write_output_manifest(path)
    (path / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
    return file_sha256(path / "output_manifest.sha256")


def main() -> None:
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(809)
        bc = End2Race(mask_prob=0.0, hidden_scale=4).eval()
        state = copy.deepcopy(bc.state_dict())
        lidar = torch.rand(23, 1, 1, 360) * 30.0
        speed = torch.rand(23, 1, 1, 1) * 5.0 + 2.0
    hidden = torch.zeros(1, 1, bc.gru.hidden_size)
    reference = []
    with torch.no_grad():
        for step in range(23):
            action, hidden = bc(lidar[step], speed[step], hidden)
            reference.append(action.detach().clone())

    policy = RemediatedV22Policy(ARM_BC_FROZEN, bc_state_dict=state).eval()
    with torch.no_grad():
        policy.intervention_gate.bias.fill_(8.0)
        policy.brake_gate.bias.fill_(8.0)
        policy.steer_mean.bias.fill_(2.0)
        policy.brake_mean.bias.fill_(2.0)

    # Replacement Task 9 records a naturally active four-coordinate action,
    # yet always returns the exact raw BC tensor under forced physical zero.
    forced = HierarchicalForcedZeroActor(policy).eval()
    outputs = _run_actor(forced, lidar, speed)
    assert all(torch.equal(actual, expected) for actual, expected in zip(outputs, reference))
    accounting = forced.accounting()
    assert accounting["macro_lengths"] == [10, 10, 3]
    assert accounting["natural_intervention_decisions"] == 3
    assert accounting["natural_brake_decisions"] == 3
    assert accounting["max_abs_requested_residual"] > 0.0
    assert accounting["max_abs_forced_residual"] == 0.0
    assert len(forced.records) == 3
    assert all(row["intervention_gate"] == 1 and row["brake_gate"] == 1 for row in forced.records)

    # Replacement Task 10 preserves one latent per macro and reprojects it for
    # every micro-step.  Both ablations are exact, not small-number heuristics.
    mode_records = {}
    for mode in (MODE_FULL, MODE_STEER_OFF, MODE_BRAKE_OFF):
        actor = HierarchicalClosedLoopActor(policy, StubAlarm(), mode).eval()
        actor.reset_runtime(2)
        actions = _run_actor(actor, lidar, speed)
        accounting = actor.accounting()
        assert accounting["macro_lengths"] == [10, 10, 3]
        assert accounting["external_clip_micro_steps"] == 0
        assert all(torch.all(action[..., 0].abs() <= 0.52) for action in actions)
        assert all(torch.all(action[..., 1] >= 0.0) for action in actions)
        assert [row["micro_count"] for row in actor.records] == [10, 10, 3]
        assert all(row["_composition_digest"].hexdigest() != "" for row in actor.records)
        mode_records[mode] = actor.records
    assert all(
        row["effective"].steer_latent.item() == 0.0
        and row["max_abs_applied_steer_delta"] == 0.0
        for row in mode_records[MODE_STEER_OFF]
    )
    assert all(
        row["effective"].brake_gate.item() == 0.0
        and row["max_brake_delta"] == 0.0
        for row in mode_records[MODE_BRAKE_OFF]
    )
    assert any(row["max_abs_applied_steer_delta"] > 0.0 for row in mode_records[MODE_FULL])
    assert any(row["max_brake_delta"] > 0.0 for row in mode_records[MODE_FULL])

    # New release path and manifest hash are arguments; an old schema cannot
    # enter either replacement evaluator even if its inventory is self-consistent.
    with tempfile.TemporaryDirectory() as temporary:
        import bplus_v22.hierarchical_checkpoint_preflight as preflight_module

        original_validator = preflight_module.validate_hierarchical_warmstart_release
        preflight_module.validate_hierarchical_warmstart_release = lambda release, root: {
            "passed": True,
            "integrity_passed": True,
            "task6_acceptance_passed": True,
            "violations": [],
        }
        release = Path(temporary) / "accepted"
        digest = _make_release(release, policy, CHECKPOINT_SCHEMA)
        payloads, hashes, config = load_hierarchical_warmstart_release(release, digest, ".")
        from bplus_v22 import ARMS

        assert set(payloads) == set(hashes) == set(config["reports"]) == set(ARMS)
        old = Path(temporary) / "old"
        old_digest = _make_release(old, policy, "bplus-v2.2-warmstart-remediation-checkpoint-2")
        try:
            load_hierarchical_warmstart_release(old, old_digest, ".")
            raise AssertionError("old single-gate checkpoint schema was accepted")
        except ValueError as error:
            assert "schema mismatch" in str(error)
        preflight_module.validate_hierarchical_warmstart_release = original_validator
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

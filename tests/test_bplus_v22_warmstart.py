#!/usr/bin/env python3
"""Task-6 witness/preservation lock and supervised-loss regressions."""

from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from bplus_v22 import (
    ARM_BC_FROZEN,
    BRAKE_BUDGET,
    INITIAL_BRAKE_LOGIT,
    STEER_BUDGET,
)
from bplus_v22.model import V22Policy
from bplus_v22.sidecar import file_sha256
from bplus_v22.warmstart import (
    ACTOR_INPUT_FIELDS,
    EXPECTED_INTERVENTION_MACROS,
    EXPECTED_FIT_GATE_POSITIVES,
    EXPECTED_FIT_GATE_TOTAL,
    EXPECTED_PRESERVATION_CANDIDATES,
    EXPECTED_PRESERVATION_EPISODES,
    EXPECTED_REGISTRY_BEFORE_SHA256,
    EXPECTED_WITNESSES,
    FORBIDDEN_ACTOR_FIELDS,
    INTERVENTION_PER_BATCH,
    PRESERVATION_PER_BATCH,
    REGISTRY_DECISION_EFFECT,
    REGISTRY_STAGE,
    REGISTRY_USE_CLASS,
    WARMSTART_BATCH_SIZE,
    WARMSTART_UPDATES,
    WITNESS_NOOP_PER_BATCH,
    _policy_hashes,
    _apply_warmstart_gate_prior,
    _gate_acceptance,
    _registry_live_state,
    _state_hashes,
    _warmstart_loss,
    build_episode_manifest,
    build_gate_prior,
    build_macro_examples,
    build_training_schedule,
    make_action_choice_registry_rows,
)
from d0.identity import append_opened_registry


def main() -> None:
    root = Path(".").resolve()
    episodes, episode_counts = build_episode_manifest(root)
    assert episode_counts == {
        "witnesses": EXPECTED_WITNESSES,
        "preservation_candidates": EXPECTED_PRESERVATION_CANDIDATES,
        "preservation_strata": EXPECTED_PRESERVATION_EPISODES,
        "total_episodes": EXPECTED_WITNESSES + EXPECTED_PRESERVATION_EPISODES,
    }
    witnesses = [row for row in episodes if row["role"] == "witness"]
    preservation = [row for row in episodes if row["role"] == "preservation"]
    assert len(witnesses) == 67 and len(preservation) == 602
    assert {row["confirmed_safe_pass"] for row in witnesses} == {"true"}
    assert {row["action_clipped"] for row in witnesses} == {"false"}
    assert not ({row["l2_id"] for row in witnesses} & {row["l2_id"] for row in preservation})
    assert len({row["preservation_stratum"] for row in preservation}) == 602

    examples, counts = build_macro_examples(episodes)
    assert counts["intervention"] == EXPECTED_INTERVENTION_MACROS
    assert [int(row["example_index"]) for row in examples] == list(range(len(examples)))
    episode_by_l2 = {row["l2_id"]: row for row in episodes}
    for example in examples:
        episode = episode_by_l2[example["l2_id"]]
        if example["role"] == "intervention":
            assert example["active_intervention"] == "true"
            assert example["target_brake_hex"] == episode["target_brake_hex"]
            assert example["target_steer_hex"] == episode["target_steer_hex"]
        else:
            assert example["active_intervention"] == "false"
            assert float.fromhex(example["target_brake_hex"]) == 0.0
            assert float.fromhex(example["target_steer_hex"]) == 0.0

    schedule, diagnostic, schedule_counts = build_training_schedule(examples)
    repeated_schedule, repeated_diagnostic, repeated_counts = build_training_schedule(examples)
    assert np.array_equal(schedule, repeated_schedule)
    assert np.array_equal(diagnostic, repeated_diagnostic)
    assert schedule_counts == repeated_counts
    assert schedule.shape == (WARMSTART_UPDATES, WARMSTART_BATCH_SIZE)
    roles = np.asarray([row["role"] for row in examples])
    for batch in schedule:
        batch_roles = roles[batch]
        assert np.count_nonzero(batch_roles == "intervention") == INTERVENTION_PER_BATCH
        assert np.count_nonzero(batch_roles == "witness_noop") == WITNESS_NOOP_PER_BATCH
        assert np.count_nonzero(batch_roles == "preservation_noop") == PRESERVATION_PER_BATCH
    diagnostic_roles = roles[diagnostic]
    for role in ("intervention", "witness_noop", "preservation_noop"):
        assert np.count_nonzero(diagnostic_roles == role) == EXPECTED_INTERVENTION_MACROS
    gate_prior = build_gate_prior(examples, schedule, diagnostic)
    assert gate_prior["fit_positive_labels"] == EXPECTED_FIT_GATE_POSITIVES
    assert gate_prior["fit_total_labels"] == EXPECTED_FIT_GATE_TOTAL
    assert gate_prior["fit_prevalence"] == 90089 / 262144
    assert abs(gate_prior["derived_bias_float64"] - (-0.6470161225499584)) < 1e-15
    assert gate_prior["diagnostic_positive_labels"] == 200
    assert abs(gate_prior["diagnostic_marginal_bce"] - 0.538180595747381) < 1e-15
    assert gate_prior["diagnostic_marginal_bce"] < 0.5382

    rows = make_action_choice_registry_rows(episodes)
    assert len(rows) == 669
    assert {row["stage"] for row in rows} == {REGISTRY_STAGE}
    assert {row["use_class"] for row in rows} == {REGISTRY_USE_CLASS}
    assert {row["decision_effect"] for row in rows} == {REGISTRY_DECISION_EFFECT}
    assert {row["final_pool"] for row in rows} == {"false"}
    with tempfile.TemporaryDirectory() as temporary:
        registry = Path(temporary) / "registry.tsv"
        immutable_before = (
            root
            / "Experiments/B1_route_r2_scaffold/artifacts/"
            "sidecar_init_20260712_080012/opened_registry.snapshot.tsv"
        )
        shutil.copyfile(immutable_before, registry)
        assert file_sha256(registry) == EXPECTED_REGISTRY_BEFORE_SHA256
        assert _registry_live_state(
            registry, rows, EXPECTED_REGISTRY_BEFORE_SHA256, "0" * 64
        ) == "ready"
        before_d25 = [
            line for line in registry.read_text(encoding="utf-8").splitlines()
            if "\tD2.5\t" in line
        ]
        result = append_opened_registry(registry, rows)
        assert (result.appended, result.skipped, result.total) == (669, 0, 12688)
        after_sha = file_sha256(registry)
        assert _registry_live_state(
            registry, rows, EXPECTED_REGISTRY_BEFORE_SHA256, after_sha
        ) == "already_appended"
        repeated = append_opened_registry(registry, rows)
        assert (repeated.appended, repeated.skipped, repeated.total) == (0, 669, 12688)
        after_d25 = [
            line for line in registry.read_text(encoding="utf-8").splitlines()
            if "\tD2.5\t" in line
        ]
        assert before_d25 == after_d25

    assert tuple(ACTOR_INPUT_FIELDS)
    assert not (set(ACTOR_INPUT_FIELDS) & set(FORBIDDEN_ACTOR_FIELDS))
    torch.set_num_threads(1)
    policy = V22Policy(ARM_BC_FROZEN)
    assert INITIAL_BRAKE_LOGIT == -6.0
    assert torch.equal(
        policy.brake_gate.bias,
        torch.full_like(policy.brake_gate.bias, -6.0),
    )
    assert torch.equal(
        policy.brake_gate.weight,
        torch.zeros_like(policy.brake_gate.weight),
    )
    fresh_state = {
        name: value.detach().clone() for name, value in policy.state_dict().items()
    }
    _apply_warmstart_gate_prior(policy, gate_prior)
    changed = [
        name
        for name, value in policy.state_dict().items()
        if not torch.equal(value, fresh_state[name])
    ]
    assert changed == ["brake_gate.bias"]
    assert float(policy.brake_gate.bias.item()) == gate_prior["applied_bias_float32"]
    batch = 4
    bc = torch.randn(batch, 1680)
    lidar = torch.rand(batch, 8, 360)
    scalar = torch.randn(batch, 24)
    targets = {
        "target_steer": torch.tensor(
            [-STEER_BUDGET, STEER_BUDGET, 0.0, STEER_BUDGET]
        ),
        "target_brake": torch.tensor([BRAKE_BUDGET, BRAKE_BUDGET, 0.0, 0.5]),
        "target_brake_gate": torch.tensor([1.0, 1.0, 0.0, 1.0]),
    }
    initial_hashes = _policy_hashes(policy)
    assert _state_hashes(policy.state_dict()) == initial_hashes
    loss, components = _warmstart_loss(policy, bc, lidar, scalar, targets)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(components[name]) for name in ("steer", "gate", "brake"))
    loss.backward()
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in policy.parameters()
        if parameter.requires_grad
    )
    passing = _gate_acceptance(
        {
            "gate_recall": 0.5,
            "gate_loss": 0.5,
            "gate_precision": 0.4,
            "gate_specificity": 0.8,
            "gate_true_positive": 10,
            "gate_false_positive": 15,
            "gate_true_negative": 60,
            "gate_false_negative": 10,
        }
    )
    assert passing["passed"]
    all_negative = _gate_acceptance(
        {
            "gate_recall": 0.0,
            "gate_loss": 0.4,
            "gate_precision": 0.0,
            "gate_specificity": 1.0,
            "gate_true_positive": 0,
            "gate_false_positive": 0,
            "gate_true_negative": 75,
            "gate_false_negative": 20,
        }
    )
    all_positive = _gate_acceptance(
        {
            "gate_recall": 1.0,
            "gate_loss": 0.4,
            "gate_precision": 0.2,
            "gate_specificity": 0.0,
            "gate_true_positive": 20,
            "gate_false_positive": 75,
            "gate_true_negative": 0,
            "gate_false_negative": 0,
        }
    )
    marginal_only = _gate_acceptance(
        {
            "gate_recall": 0.5,
            "gate_loss": 0.5382,
            "gate_precision": 0.3,
            "gate_specificity": 0.7,
            "gate_true_positive": 10,
            "gate_false_positive": 20,
            "gate_true_negative": 55,
            "gate_false_negative": 10,
        }
    )
    assert not all_negative["passed"]
    assert not all_positive["passed"]
    assert not marginal_only["passed"]
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

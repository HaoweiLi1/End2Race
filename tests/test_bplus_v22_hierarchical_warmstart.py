#!/usr/bin/env python3
"""Structural regressions for the replacement hierarchical Task 6."""

from pathlib import Path
import tempfile

import numpy as np
import torch

import bplus_v22.hierarchical_warmstart as hierarchical_module
from bplus_v22 import ARM_BC_FROZEN, INITIAL_BRAKE_LOGIT
from bplus_v22.hierarchical_warmstart import (
    EXPECTED_CAL_BRAKE_MACROS,
    EXPECTED_CAL_NEGATIVE_EPISODES,
    EXPECTED_CAL_NEGATIVE_MACROS,
    EXPECTED_CAL_POSITIVE_INTERVENTION,
    EXPECTED_CAL_STEER_ONLY_MACROS,
    EXPECTED_FIT_BRAKE,
    EXPECTED_FIT_EPISODES,
    EXPECTED_FIT_INTERVENTION,
    EXPECTED_FIT_MACROS,
    EXPECTED_FIT_PRESERVATION_EPISODES,
    EXPECTED_FIT_WITNESS_EPISODES,
    EXPECTED_FIT_WITNESS_NOOP,
    FIT_POS_WEIGHT,
    REGISTRY_DECISION_EFFECT,
    REGISTRY_OPENED_AT,
    REGISTRY_SPLIT_ID,
    WARMSTART_BATCH_SIZE,
    WARMSTART_UPDATES,
    _hierarchical_loss,
    build_hierarchical_episode_manifest,
    build_hierarchical_macro_examples,
    build_hierarchical_priors,
    build_natural_cycle_schedule,
    create_hierarchical_warmstart_manifest,
    derive_negative_only_calibration,
    make_calibration_registry_rows,
    validate_hierarchical_warmstart_manifest,
)
from bplus_v22.remediated_model import (
    INITIAL_INTERVENTION_LOGIT,
    RemediatedV22Policy,
    apply_intervention_logit_offset,
    initialize_hierarchical_priors,
)
from bplus_v22.release import create_source_preflight


def main() -> None:
    root = Path(".").resolve()
    episodes, episode_counts = build_hierarchical_episode_manifest(root)
    assert episode_counts["fit_witness_episodes"] == EXPECTED_FIT_WITNESS_EPISODES
    assert (
        episode_counts["fit_preservation_episodes"]
        == EXPECTED_FIT_PRESERVATION_EPISODES
    )
    assert episode_counts["fit_episodes"] == EXPECTED_FIT_EPISODES
    assert episode_counts["calibration_positive_episodes"] == 9
    assert episode_counts["calibration_negative_episodes"] == 75
    assert episode_counts["calibration_negative_skill_f"] == 0
    partitions = {row["partition"] for row in episodes}
    assert partitions == {"fit", "calibration_positive", "calibration_negative"}
    assert len({row["l2_id"] for row in episodes}) == len(episodes)
    registry_rows = make_calibration_registry_rows(episodes)
    assert len(registry_rows) == 75
    assert {row["opened_at_utc"] for row in registry_rows} == {
        REGISTRY_OPENED_AT
    }
    assert {row["split_id"] for row in registry_rows} == {REGISTRY_SPLIT_ID}
    assert {row["decision_effect"] for row in registry_rows} == {
        REGISTRY_DECISION_EFFECT
    }

    examples, counts = build_hierarchical_macro_examples(root, episodes)
    assert counts["fit"] == {
        "intervention": EXPECTED_FIT_INTERVENTION,
        "brake": EXPECTED_FIT_BRAKE,
        "witness_noop": EXPECTED_FIT_WITNESS_NOOP,
        "preservation_noop": 39204,
        "total": EXPECTED_FIT_MACROS,
    }
    assert counts["calibration_positive"] == {
        "intervention": EXPECTED_CAL_POSITIVE_INTERVENTION,
        "steer_only": EXPECTED_CAL_STEER_ONLY_MACROS,
        "brake": EXPECTED_CAL_BRAKE_MACROS,
    }
    assert counts["calibration_negative"] == EXPECTED_CAL_NEGATIVE_MACROS
    assert [int(row["example_index"]) for row in examples] == list(
        range(len(examples))
    )

    schedule, fit_indices, schedule_info = build_natural_cycle_schedule(examples)
    schedule2, fit_indices2, schedule_info2 = build_natural_cycle_schedule(examples)
    assert schedule.shape == (WARMSTART_UPDATES, WARMSTART_BATCH_SIZE)
    assert schedule.dtype == np.int32
    assert np.array_equal(schedule, schedule2)
    assert np.array_equal(fit_indices, fit_indices2)
    assert schedule_info == schedule_info2
    assert len(np.unique(schedule[: EXPECTED_FIT_MACROS // WARMSTART_BATCH_SIZE])) > 43000
    assert {
        examples[int(index)]["partition"] for index in schedule.reshape(-1)
    } == {"fit"}
    # Every complete cycle visits every fit macro exactly once: no class mixture.
    first_cycle = schedule.reshape(-1)[:EXPECTED_FIT_MACROS]
    assert len(np.unique(first_cycle)) == EXPECTED_FIT_MACROS
    assert set(first_cycle.tolist()) == set(fit_indices.tolist())

    priors = build_hierarchical_priors(examples)
    assert priors["intervention"]["positive"] == 252
    assert priors["intervention"]["total"] == 43902
    assert priors["intervention"]["prevalence"] == 252 / 43902
    assert priors["conditional_brake"]["positive"] == 175
    assert priors["conditional_brake"]["total"] == 252
    assert priors["conditional_brake"]["prevalence"] == 175 / 252
    assert priors["intervention_bce_pos_weight"] == FIT_POS_WEIGHT

    with tempfile.TemporaryDirectory(dir=root) as temporary:
        source_preflight = Path(temporary) / "source_preflight"
        created_source = create_source_preflight(
            source_preflight, "2026-07-12T11:59:00+08:00", root
        )
        assert created_source["passed"]
        identity = Path(temporary) / "hierarchical_identity"
        identity.mkdir()
        (identity / "output_manifest.sha256").write_text(
            "unit-test identity prerequisite\n", encoding="utf-8"
        )
        manifest = Path(temporary) / "manifest"
        original_identity_validator = hierarchical_module.validate_hierarchical_identity
        hierarchical_module.validate_hierarchical_identity = lambda *args, **kwargs: {
            "passed": True,
            "violations": [],
        }
        try:
            created = create_hierarchical_warmstart_manifest(
                root,
                source_preflight,
                identity,
                manifest,
                "2026-07-12T12:00:00+08:00",
            )
            assert created["passed"]
            assert (manifest / "COMPLETE").is_file()
            valid = validate_hierarchical_warmstart_manifest(manifest, root)
            assert valid["passed"], valid
            stored = np.load(
                manifest / "training_schedule.npy", allow_pickle=False
            )
            stored[0, 0], stored[0, 1] = stored[0, 1], stored[0, 0]
            np.save(manifest / "training_schedule.npy", stored, allow_pickle=False)
            invalid = validate_hierarchical_warmstart_manifest(manifest, root)
            assert not invalid["passed"]
        finally:
            hierarchical_module.validate_hierarchical_identity = (
                original_identity_validator
            )

    torch.set_num_threads(1)
    policy = RemediatedV22Policy(ARM_BC_FROZEN)
    assert INITIAL_INTERVENTION_LOGIT == -6.0
    assert INITIAL_BRAKE_LOGIT == -6.0
    assert torch.equal(
        policy.intervention_gate.bias,
        torch.full_like(policy.intervention_gate.bias, -6.0),
    )
    assert torch.equal(
        policy.brake_gate.bias, torch.full_like(policy.brake_gate.bias, -6.0)
    )
    initialize_hierarchical_priors(
        policy,
        priors["intervention"]["applied_logit_float32"],
        priors["conditional_brake"]["applied_logit_float32"],
    )
    assert float(policy.intervention_gate.bias.item()) == priors["intervention"][
        "applied_logit_float32"
    ]
    assert float(policy.brake_gate.bias.item()) == priors["conditional_brake"][
        "applied_logit_float32"
    ]

    batch = 5
    bc = torch.randn(batch, 1680)
    lidar = torch.rand(batch, 8, 360)
    scalar = torch.randn(batch, 24)
    targets = {
        "active_intervention": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
        "target_steer": torch.tensor([0.1, -0.1, 0.2, -0.2, 0.15]),
        "target_brake_gate": torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]),
        "target_brake": torch.tensor([1.0, 0.0, 1.0, 0.5, 0.75]),
    }
    loss, components = _hierarchical_loss(policy, bc, lidar, scalar, targets)
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(components[name])
        for name in ("intervention", "steer", "conditional_brake", "brake")
    )
    changed_inactive = {name: value.clone() for name, value in targets.items()}
    changed_inactive["target_steer"][2:] *= -1
    changed_inactive["target_brake"][2:] = torch.tensor([0.0, 1.0, 0.0])
    changed_inactive["target_brake_gate"][2:] = 1.0 - changed_inactive[
        "target_brake_gate"
    ][2:]
    changed_loss, changed_components = _hierarchical_loss(
        policy, bc, lidar, scalar, changed_inactive
    )
    assert torch.equal(loss, changed_loss)
    for name in ("steer", "conditional_brake", "brake"):
        assert torch.equal(components[name], changed_components[name])
    loss.backward()
    assert policy.intervention_gate.weight.grad is not None
    assert torch.any(policy.intervention_gate.weight.grad != 0)
    policy.eval()
    with torch.no_grad():
        raw_distribution = policy.distribution(bc, lidar, scalar)
        raw_logits = raw_distribution.intervention.logits.clone()
        apply_intervention_logit_offset(policy, 1.25)
        calibrated_distribution = policy.distribution(bc, lidar, scalar)
        assert torch.equal(
            calibrated_distribution.intervention.logits,
            raw_logits + torch.full_like(raw_logits, 1.25),
        )
        calibrated_action = calibrated_distribution.deterministic()
        assert torch.equal(
            calibrated_action.intervention_gate,
            (calibrated_distribution.intervention.logits > 0).to(
                calibrated_action.intervention_gate.dtype
            ),
        )
        assert float(policy.state_dict()["intervention_logit_offset"].item()) == 1.25

    positive_rows = [
        row
        for row in examples
        if row["partition"] == "calibration_positive"
        and row["active_intervention"] == "true"
    ]
    negative_rows = [
        row for row in examples if row["partition"] == "calibration_negative"
    ]
    positive_raw = np.full(len(positive_rows), 5.0, dtype=np.float32)
    conditional_raw = np.asarray(
        [
            5.0 if int(row["target_brake_gate"]) else -5.0
            for row in positive_rows
        ],
        dtype=np.float32,
    )
    # Give each negative episode one ordered maximum and lower remaining macros.
    negative_raw = np.full(len(negative_rows), -5.0, dtype=np.float32)
    first_by_l2 = {}
    for index, row in enumerate(negative_rows):
        first_by_l2.setdefault(row["l2_id"], index)
    assert len(first_by_l2) == EXPECTED_CAL_NEGATIVE_EPISODES
    for rank, index in enumerate(first_by_l2.values()):
        negative_raw[index] = np.float32(rank / 100.0)
    metrics, episode_decisions = derive_negative_only_calibration(
        examples,
        episodes,
        positive_raw,
        negative_raw,
        conditional_raw,
        ARM_BC_FROZEN,
    )
    assert metrics["passed"], metrics
    assert metrics["false_intervention_episodes"] <= 7
    assert metrics["intervention_window_episode_true_positive"] == 9
    assert metrics["intervention_macro_true_positive"] == 39
    assert metrics["steer_only_episode_true_positive"] == 4
    assert metrics["brake_episode_true_positive"] == 5
    assert metrics["conditional_brake_confusion"]["recall"] == 1.0
    assert metrics["conditional_brake_confusion"]["specificity"] == 1.0
    assert len(episode_decisions) == 84
    # Positive scores are pass/fail only and cannot alter the negative threshold.
    low_positive = np.full_like(positive_raw, -10.0)
    low_metrics, _ = derive_negative_only_calibration(
        examples,
        episodes,
        low_positive,
        negative_raw,
        conditional_raw,
        ARM_BC_FROZEN,
    )
    assert (
        low_metrics["applied_offset_float32"]
        == metrics["applied_offset_float32"]
    )
    assert not low_metrics["passed"]

    print("bplus v2.2 hierarchical warm-start tests passed")


if __name__ == "__main__":
    main()

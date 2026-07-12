#!/usr/bin/env python3
"""Pure B2 deterministic evaluator, physical sharding, and strict merge tests."""

from dataclasses import dataclass, replace
import hashlib
from types import SimpleNamespace

import torch
import torch.nn as nn

from bplus_v22 import ARMS, ARM_BC_FROZEN, PILOT_SEEDS
from bplus_v22.ppo_eval import (
    BC_VARIANT,
    B2DeterministicActor,
    CandidateCheckpoint,
    EvaluationShard,
    LoadedCandidatePolicy,
    candidate_variant,
    evaluate_shard,
    evaluate_bc_baseline_preflight,
    load_candidate_policies,
    merge_evaluation_shards,
    physical_shard_rows,
    validate_task8_rows,
)
from bplus_v22.remediated_model import RemediatedV22Policy


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def expect(error_type, function) -> None:
    try:
        function()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def task8_rows() -> list[dict[str, str]]:
    rows = []
    for index in range(288):
        rows.append(
            {
                # Deliberately reverse this provenance field.  Sharding and
                # pairing must use the physical list/TSV position instead.
                "manifest_order": str(287 - index),
                "panel": ("representative", "skill_F", "skill_S")[index % 3],
                "l2_id": f"L2:{index:064x}",
                "l4_id": f"L4:{index:064x}",
                "map_name": ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")[
                    index % 4
                ],
                "skill": ("other", "skill_F", "skill_S")[index % 3],
                "opponent_raceline": f"raceline{index % 3}",
                "speedscale_hex": float(0.5 + 0.1 * (index % 3)).hex(),
                "resolved_ego_idx": str(100 + index),
                "physical_test_index": str(index),
            }
        )
    return rows


def checkpoint_specs(training_manifest: str) -> tuple[CandidateCheckpoint, ...]:
    return tuple(
        CandidateCheckpoint(
            arm=arm,
            seed=seed,
            checkpoint_id=f"{arm}-seed{seed}-iter20",
            checkpoint_sha256=digest(f"checkpoint:{arm}:{seed}"),
            training_manifest_sha256=training_manifest,
        )
        for arm in ARMS
        for seed in PILOT_SEEDS
    )


class FakePolicy(nn.Module):
    def __init__(self, arm: str, seed: int):
        super().__init__()
        self.arm = arm
        self.seed = seed
        self.register_buffer("intervention_logit_offset", torch.zeros(1))


class FakeActor(nn.Module):
    def __init__(self, policy: FakePolicy):
        super().__init__()
        self.policy = policy

    def accounting(self):
        primary = 1 + self.policy.seed
        standard = 2 + ARMS.index(self.policy.arm)
        return {
            "micro_steps": 11,
            "macro_decisions": 2,
            "macro_lengths": [10, 1],
            "short_terminal_macro": True,
            "primary_intervention_decisions": primary,
            "primary_brake_decisions": min(primary, 1),
            "standard_intervention_decisions": standard,
            "standard_brake_decisions": min(standard, 1),
            "mean_abs_applied_steer_delta": 0.01,
            "max_abs_applied_steer_delta": 0.02,
            "mean_brake_delta": 0.03,
            "max_brake_delta": 0.04,
            "external_clip_micro_steps": 0,
        }


class FakeBC(nn.Module):
    pass


@dataclass(frozen=True)
class FakeOutcome:
    four_state: str
    collision_any: bool
    ego_collision: bool
    corrected_outcome3: str
    confirmed_safe_pass: bool
    interaction_attempt: bool


def outcome(collision: bool, overtake: bool, confirmed: bool = False) -> FakeOutcome:
    if collision:
        state = "collision"
        corrected = "collision"
    elif overtake:
        state = "confirmed_pass" if confirmed else "terminal_overtake_only"
        corrected = "overtake"
    else:
        state = "safe_follow"
        corrected = "follow"
    return FakeOutcome(state, collision, collision, corrected, confirmed, True)


def fake_simulator(model, device, case):
    index = int(case["physical_test_index"])
    if isinstance(model, FakeBC):
        collision = index < 24
        overtake = 24 <= index < 162
        token = f"BC:{index}"
    else:
        arm_index = ARMS.index(model.policy.arm)
        seed = model.policy.seed
        collision = (index + arm_index + seed) % (11 + arm_index) == 0
        overtake = not collision and (index + seed) % 3 == 0
        token = f"{model.policy.arm}:{seed}:{index}"
    result_outcome = outcome(
        collision,
        overtake,
        confirmed=(not collision and index % 10 == 0),
    )
    return SimpleNamespace(
        arrays={"token": token},
        outcome=result_outcome,
        action_clipped=False,
        episode_key=token,
    )


def fake_trajectory_digest(arrays) -> str:
    return digest(str(arrays["token"]))


def loader(expected: CandidateCheckpoint, device: torch.device):
    return LoadedCandidatePolicy(
        policy=FakePolicy(expected.arm, expected.seed),
        checkpoint_id=expected.checkpoint_id,
        checkpoint_sha256=expected.checkpoint_sha256,
        training_manifest_sha256=expected.training_manifest_sha256,
    )


def test_short_centered_actor_mechanics() -> None:
    torch.set_num_threads(1)
    policy = RemediatedV22Policy(ARM_BC_FROZEN).eval()
    with torch.no_grad():
        # Centered primary activates because -5.9 > -6; standard diagnostic
        # remains inactive because -5.9 is not > 0.
        policy.intervention_gate.bias.fill_(-5.9)
        policy.brake_gate.bias.fill_(-5.9)
    actor = B2DeterministicActor(policy).eval()
    hidden = torch.zeros(1, 1, actor.gru.hidden_size)
    commands = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(190)
        lidars = torch.rand(23, 1, 1, 360) * 30.0
        speeds = torch.rand(23, 1, 1, 1) * 5.0
    with torch.no_grad():
        for step in range(23):
            actor.observe_actual_speed(float(speeds[step].item()) + 0.25)
            command, hidden = actor(lidars[step], speeds[step], hidden)
            actor.observe_applied_command(
                float(command[0, -1, 0]), float(command[0, -1, 1])
            )
            commands.append(command)
    accounting = actor.accounting()
    assert accounting["macro_lengths"] == [10, 10, 3]
    assert accounting["primary_intervention_decisions"] == 3
    assert accounting["primary_brake_decisions"] == 3
    assert accounting["standard_intervention_decisions"] == 0
    assert accounting["standard_brake_decisions"] == 0
    assert accounting["external_clip_micro_steps"] == 0
    assert all(torch.all(command[..., 0].abs() <= 0.52) for command in commands)
    assert all(torch.all(command[..., 1] >= 0.0) for command in commands)


def test_physical_sharding() -> None:
    rows = task8_rows()
    validate_task8_rows(rows)
    shards = [physical_shard_rows(rows, index, 4) for index in range(4)]
    assert [len(shard) for shard in shards] == [72, 72, 72, 72]
    for shard_index, shard in enumerate(shards):
        assert all(physical % 4 == shard_index for physical, _ in shard)
        assert [physical for physical, _ in shard[:3]] == [shard_index + 4 * n for n in range(3)]
    # Provenance values are reversed, proving assignment did not parse them.
    assert shards[0][0][1]["manifest_order"] == "287"


def build_shards():
    rows = task8_rows()
    scenario_manifest = digest("task8-development")
    training_manifest = digest("b2-training-manifest")
    bc_sha = digest("canonical-bc")
    specs = checkpoint_specs(training_manifest)
    shards = tuple(
        evaluate_shard(
            task8_rows=rows,
            scenario_manifest_sha256=scenario_manifest,
            checkpoint_manifest_sha256=training_manifest,
            bc_model=FakeBC(),
            bc_checkpoint_sha256=bc_sha,
            checkpoints=specs,
            policy_loader=loader,
            device=torch.device("cpu"),
            shard_index=shard_index,
            shard_count=4,
            actor_factory=FakeActor,
            simulator=fake_simulator,
            trajectory_digest_fn=fake_trajectory_digest,
        )
        for shard_index in range(4)
    )
    return rows, scenario_manifest, training_manifest, bc_sha, specs, shards


def test_injected_eval_and_strict_merge() -> None:
    rows, scenario_manifest, training_manifest, bc_sha, specs, shards = build_shards()
    assert all(len(shard.rows) == 72 * 7 for shard in shards)
    merged, summary = merge_evaluation_shards(
        shards=shards,
        task8_rows=rows,
        scenario_manifest_sha256=scenario_manifest,
        checkpoint_manifest_sha256=training_manifest,
        bc_checkpoint_sha256=bc_sha,
        checkpoints=specs,
    )
    assert len(merged) == 2016
    assert summary["scenario_count"] == 288
    assert summary["variant_count"] == 7
    assert summary["result_count"] == 2016
    assert summary["bc_baseline"] == {
        "episodes": 288,
        "collision": 24,
        "terminal_overtake": 138,
    }
    assert set(summary["arms_pooled"]) == set(ARMS)
    assert summary["opened_development_only"] is True
    assert summary["arm_selection_performed"] is False
    assert set(summary["variants"]) == {BC_VARIANT} | {
        candidate_variant(arm, seed) for arm in ARMS for seed in PILOT_SEEDS
    }
    assert all(value["episodes"] == 288 for value in summary["variants"].values())
    candidates = [row for row in merged if row["variant"] != BC_VARIANT]
    assert all(row["deterministic_mode"] == "centered_fresh_prior" for row in candidates)
    assert all(row["standard_mode_is_diagnostic_only"] is True for row in candidates)
    assert any(row["fixed_collision"] for row in candidates)
    assert any(row["new_collision"] for row in candidates)
    assert all(
        "direction_verdict" in summary["variants"][candidate_variant(arm, seed)]
        for arm in ARMS
        for seed in PILOT_SEEDS
    )

    baseline_preflight = evaluate_bc_baseline_preflight(
        task8_rows=rows,
        scenario_manifest_sha256=scenario_manifest,
        bc_model=FakeBC(),
        bc_checkpoint_sha256=bc_sha,
        device=torch.device("cpu"),
        simulator=fake_simulator,
        trajectory_digest_fn=fake_trajectory_digest,
    )
    assert baseline_preflight["passed"] is True
    assert baseline_preflight["collision"] == 24
    assert baseline_preflight["terminal_overtake"] == 138

    # Duplicate and missing Cartesian cells fail independently.
    duplicate = replace(shards[0], rows=shards[0].rows + (shards[0].rows[0],))
    expect(
        ValueError,
        lambda: merge_evaluation_shards(
            shards=(duplicate, *shards[1:]),
            task8_rows=rows,
            scenario_manifest_sha256=scenario_manifest,
            checkpoint_manifest_sha256=training_manifest,
            bc_checkpoint_sha256=bc_sha,
            checkpoints=specs,
        ),
    )
    missing = replace(shards[0], rows=shards[0].rows[:-1])
    expect(
        ValueError,
        lambda: merge_evaluation_shards(
            shards=(missing, *shards[1:]),
            task8_rows=rows,
            scenario_manifest_sha256=scenario_manifest,
            checkpoint_manifest_sha256=training_manifest,
            bc_checkpoint_sha256=bc_sha,
            checkpoints=specs,
        ),
    )

    bad_inventory = dict(shards[0].checkpoint_sha256_by_variant)
    bad_inventory[candidate_variant(ARMS[0], 0)] = digest("wrong-checkpoint")
    mismatch = replace(shards[0], checkpoint_sha256_by_variant=bad_inventory)
    expect(
        ValueError,
        lambda: merge_evaluation_shards(
            shards=(mismatch, *shards[1:]),
            task8_rows=rows,
            scenario_manifest_sha256=scenario_manifest,
            checkpoint_manifest_sha256=training_manifest,
            bc_checkpoint_sha256=bc_sha,
            checkpoints=specs,
        ),
    )
    manifest_mismatch = replace(shards[0], scenario_manifest_sha256=digest("wrong-manifest"))
    expect(
        ValueError,
        lambda: merge_evaluation_shards(
            shards=(manifest_mismatch, *shards[1:]),
            task8_rows=rows,
            scenario_manifest_sha256=scenario_manifest,
            checkpoint_manifest_sha256=training_manifest,
            bc_checkpoint_sha256=bc_sha,
            checkpoints=specs,
        ),
    )

    bad_row = dict(shards[0].rows[1])
    bad_row["checkpoint_sha256"] = digest("wrong-row-checkpoint")
    row_mismatch = replace(shards[0], rows=(shards[0].rows[0], bad_row, *shards[0].rows[2:]))
    expect(
        ValueError,
        lambda: merge_evaluation_shards(
            shards=(row_mismatch, *shards[1:]),
            task8_rows=rows,
            scenario_manifest_sha256=scenario_manifest,
            checkpoint_manifest_sha256=training_manifest,
            bc_checkpoint_sha256=bc_sha,
            checkpoints=specs,
        ),
    )


def test_loader_envelope_mismatch() -> None:
    training_manifest = digest("training")
    specs = checkpoint_specs(training_manifest)

    def bad_loader(expected, device):
        loaded = loader(expected, device)
        if expected == specs[0]:
            return replace(loaded, checkpoint_sha256=digest("different"))
        return loaded

    expect(
        ValueError,
        lambda: load_candidate_policies(
            specs, training_manifest, bad_loader, torch.device("cpu")
        ),
    )
    wrong_manifest_specs = (replace(specs[0], training_manifest_sha256=digest("wrong")), *specs[1:])
    expect(
        ValueError,
        lambda: load_candidate_policies(
            wrong_manifest_specs, training_manifest, loader, torch.device("cpu")
        ),
    )


def main() -> None:
    test_short_centered_actor_mechanics()
    test_physical_sharding()
    test_injected_eval_and_strict_merge()
    test_loader_envelope_mismatch()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pure B2 deterministic evaluator, physical sharding, and strict merge tests."""

from dataclasses import dataclass, replace
import hashlib
from types import SimpleNamespace

import torch
import torch.nn as nn

from bplus_v22 import ARMS, ARM_BC_FROZEN, PILOT_SEEDS
from bplus_v22.ppo_eval import (
    BASELINE_SHARD_COUNT,
    BC_VARIANT,
    BCBaselineShard,
    B2DeterministicActor,
    CandidateCheckpoint,
    EvaluationShard,
    LoadedCandidatePolicy,
    candidate_variant,
    evaluate_bc_baseline_shard,
    evaluate_shard,
    load_candidate_policies,
    merge_bc_baseline_shards,
    merge_evaluation_shards,
    physical_shard_rows,
    validate_bc_baseline_shard,
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


BASELINE_COLLISION_INDICES: set[int] = set()
BASELINE_OVERTAKE_INDICES: set[int] = set()
for _shard, (_collisions, _overtakes) in enumerate(
    zip((12, 2, 5, 5), (32, 37, 33, 36))
):
    _physical = list(range(_shard, 288, BASELINE_SHARD_COUNT))
    BASELINE_COLLISION_INDICES.update(_physical[:_collisions])
    BASELINE_OVERTAKE_INDICES.update(
        _physical[_collisions : _collisions + _overtakes]
    )


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
        collision = index in BASELINE_COLLISION_INDICES
        overtake = index in BASELINE_OVERTAKE_INDICES
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
        confirmed=(overtake and index % 10 == 0),
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


def baseline_bindings():
    return {
        "run_plan_sha256": digest("baseline-run-plan"),
        "source_commit": "1" * 40,
        "source_archive_sha256": digest("baseline-source"),
        "inputs_archive_sha256": digest("baseline-inputs"),
        "scenario_manifest_sha256": digest("baseline-task8"),
        "bc_checkpoint_sha256": digest("baseline-bc"),
    }


def baseline_producers():
    return {
        0: ("local", "GPU-local"),
        1: ("remote", "GPU-remote"),
        2: ("remote", "GPU-remote"),
        3: ("remote", "GPU-remote"),
    }


def build_baseline_shards(simulator=fake_simulator):
    rows = task8_rows()
    bindings = baseline_bindings()
    producers = baseline_producers()
    shards = tuple(
        evaluate_bc_baseline_shard(
            task8_rows=rows,
            bc_model=FakeBC(),
            device=torch.device("cpu"),
            shard_index=shard_index,
            shard_count=BASELINE_SHARD_COUNT,
            producer_host_id=producers[shard_index][0],
            producer_gpu_uuid=producers[shard_index][1],
            simulator=simulator,
            trajectory_digest_fn=fake_trajectory_digest,
            **bindings,
        )
        for shard_index in range(BASELINE_SHARD_COUNT)
    )
    return rows, bindings, producers, shards


def test_baseline_only_shards_and_topology_merge() -> None:
    rows, bindings, producers, shards = build_baseline_shards()
    assert all(isinstance(shard, BCBaselineShard) for shard in shards)
    assert validate_bc_baseline_shard(shards[0].to_dict()) == shards[0]
    assert [len(shard.rows) for shard in shards] == [72, 72, 72, 72]
    assert [shard.collision for shard in shards] == [12, 2, 5, 5]
    assert [shard.terminal_overtake for shard in shards] == [32, 37, 33, 36]
    assert all(
        all(row["task8_row_index"] % 4 == shard.shard_index for row in shard.rows)
        for shard in shards
    )
    # Serialized JSON mappings are the production merge input.  Shard arrival
    # order must not affect physical-row ordering in the canonical envelope.
    merged = merge_bc_baseline_shards(
        shards=tuple(shard.to_dict() for shard in reversed(shards)),
        task8_rows=rows,
        expected_producers=producers,
        **bindings,
    )
    assert merged["integrity_passed"] is True
    assert merged["passed"] is True
    assert merged["acceptance_passed"] is True
    assert merged["collision"] == 24
    assert merged["terminal_overtake"] == 138
    assert merged["collision_by_shard"] == [12, 2, 5, 5]
    assert merged["terminal_overtake_by_shard"] == [32, 37, 33, 36]
    assert [row["task8_row_index"] for row in merged["rows"]] == list(range(288))
    assert merged["rows"][0]["producer_host_id"] == "local"
    assert all(row["producer_host_id"] == "remote" for row in merged["rows"][1::4])
    assert all(len(item["file_sha256"]) == 64 for item in merged["shards"])


def test_baseline_count_drift_is_complete_acceptance_failure() -> None:
    drift_index = next(
        index
        for index in range(3, 288, 4)
        if index not in BASELINE_COLLISION_INDICES
        and index not in BASELINE_OVERTAKE_INDICES
    )

    def drift_simulator(model, device, case):
        result = fake_simulator(model, device, case)
        if int(case["physical_test_index"]) != drift_index:
            return result
        return SimpleNamespace(
            arrays=result.arrays,
            outcome=outcome(False, True),
            action_clipped=False,
            episode_key=result.episode_key,
        )

    rows, bindings, producers, shards = build_baseline_shards(drift_simulator)
    merged = merge_bc_baseline_shards(
        shards=shards,
        task8_rows=rows,
        expected_producers=producers,
        **bindings,
    )
    assert merged["integrity_passed"] is True
    assert merged["passed"] is False
    assert merged["acceptance_passed"] is False
    assert merged["collision"] == 24
    assert merged["terminal_overtake"] == 139
    assert merged["terminal_overtake_by_shard"] == [32, 37, 33, 37]
    assert merged["count_checks"]["terminal_overtake_by_shard"] is False
    assert merged["count_checks"]["terminal_overtake_total"] is False
    assert len(merged["rows"]) == 288
    diagnostic = merged["rows"][drift_index]
    assert diagnostic["l2_id"] == rows[drift_index]["l2_id"]
    assert diagnostic["terminal_overtake"] is True
    assert diagnostic["producer_host_id"] == "remote"


def test_baseline_merge_rejects_binding_and_inventory_drift() -> None:
    rows, bindings, producers, shards = build_baseline_shards()
    expect(
        ValueError,
        lambda: merge_bc_baseline_shards(
            shards=shards[:-1],
            task8_rows=rows,
            expected_producers=producers,
            **bindings,
        ),
    )
    duplicate = (shards[0], shards[0], shards[2], shards[3])
    expect(
        ValueError,
        lambda: merge_bc_baseline_shards(
            shards=duplicate,
            task8_rows=rows,
            expected_producers=producers,
            **bindings,
        ),
    )
    bad_host = replace(shards[1], producer_host_id="local")
    expect(
        ValueError,
        lambda: merge_bc_baseline_shards(
            shards=(shards[0], bad_host, shards[2], shards[3]),
            task8_rows=rows,
            expected_producers=producers,
            **bindings,
        ),
    )
    bad_gpu = replace(shards[1], producer_gpu_uuid="GPU-wrong")
    expect(
        ValueError,
        lambda: merge_bc_baseline_shards(
            shards=(shards[0], bad_gpu, shards[2], shards[3]),
            task8_rows=rows,
            expected_producers=producers,
            **bindings,
        ),
    )
    for field, wrong in (
        ("run_plan_sha256", digest("wrong-plan")),
        ("source_commit", "2" * 40),
        ("source_archive_sha256", digest("wrong-source")),
        ("inputs_archive_sha256", digest("wrong-inputs")),
        ("scenario_manifest_sha256", digest("wrong-task8")),
        ("bc_checkpoint_sha256", digest("wrong-bc")),
    ):
        bad_binding = replace(shards[2], **{field: wrong})
        expect(
            ValueError,
            lambda bad_binding=bad_binding: merge_bc_baseline_shards(
                shards=(shards[0], shards[1], bad_binding, shards[3]),
                task8_rows=rows,
                expected_producers=producers,
                **bindings,
            ),
        )
    bad_rows = list(shards[3].rows)
    bad_rows[0] = {**bad_rows[0], "l2_id": "L2:wrong"}
    bad_identity = replace(shards[3], rows=tuple(bad_rows))
    expect(
        ValueError,
        lambda: merge_bc_baseline_shards(
            shards=(shards[0], shards[1], shards[2], bad_identity),
            task8_rows=rows,
            expected_producers=producers,
            **bindings,
        ),
    )


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
    test_baseline_only_shards_and_topology_merge()
    test_baseline_count_drift_is_complete_acceptance_failure()
    test_baseline_merge_rejects_binding_and_inventory_drift()
    test_injected_eval_and_strict_merge()
    test_loader_envelope_mismatch()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

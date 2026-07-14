#!/usr/bin/env python3
"""Blocking CPU contracts for the B5-A canonical-BC safe hard cap."""

from dataclasses import replace
import copy
from pathlib import Path
import tempfile

import numpy as np
import torch

from bplus_v22.b4_direct import (
    B4DirectHeadPolicy,
    B4Transition,
    FROZEN_B4_CONFIG,
    build_batch,
    build_optimizers,
    project_raw_action,
)
from bplus_v22.b5_safe import (
    SAFE_EPISODE_COUNT,
    SAFE_MAPS,
    SAFE_OUTCOMES,
    SafeReference,
    load_b5_full_checkpoint,
    load_reference,
    safe_kl_metrics,
    save_b5_full_checkpoint,
    save_reference,
    select_reference_rows,
    update_policy_with_safe_cap,
)


REPO = Path(__file__).resolve().parent.parent
BC = REPO / "pretrained/end2race.pth"


def reference(policy: B4DirectHeadPolicy) -> SafeReference:
    generator = torch.Generator().manual_seed(51)
    feature = torch.randn((SAFE_EPISODE_COUNT, 1680), generator=generator)
    with torch.no_grad():
        mean = policy.mean_from_feature(feature).detach().clone()
    maps = tuple(
        map_name
        for map_name in SAFE_MAPS
        for _outcome in SAFE_OUTCOMES
        for _ in range(8)
    )
    outcomes = tuple(
        outcome for _map in SAFE_MAPS for outcome in SAFE_OUTCOMES for _ in range(8)
    )
    return SafeReference(
        feature=feature,
        bc_mean=mean,
        episode_index=torch.arange(SAFE_EPISODE_COUNT),
        step_index=torch.zeros(SAFE_EPISODE_COUNT, dtype=torch.int64),
        lengths=(1,) * SAFE_EPISODE_COUNT,
        l2_ids=tuple(f"L2:{index:064x}" for index in range(SAFE_EPISODE_COUNT)),
        l4_ids=tuple(f"L4:{index:064x}" for index in range(SAFE_EPISODE_COUNT)),
        map_names=maps,
        outcomes=outcomes,
    )


def rollout(policy: B4DirectHeadPolicy):
    generator = torch.Generator().manual_seed(52)
    rows = []
    for episode in range(16):
        for step in range(2):
            feature = torch.randn((1, 1680), generator=generator)
            privileged = torch.randn((1, 12), generator=generator)
            with torch.no_grad():
                mean = policy.mean_from_feature(feature)
                raw = mean + (0.2 if (episode + step) % 2 else -0.2) * policy.action_std
                log_prob = policy.log_prob(mean, raw)
                value = policy.value(privileged)
                executed, delta = project_raw_action(raw)
            terminal = step == 1
            rows.append(
                B4Transition(
                    l2_id=f"L2:{episode:064x}",
                    episode_id=episode,
                    step_index=step,
                    feature=feature[0].numpy().astype(np.float32),
                    privileged_feature=privileged[0].numpy().astype(np.float32),
                    raw_action=raw[0].numpy().astype(np.float32),
                    executed_action=executed[0].numpy().astype(np.float32),
                    projection_delta=delta[0].numpy().astype(np.float32),
                    old_log_prob=float(log_prob.item()),
                    old_value=float(value.item()),
                    reward=float((-2, 0, 1)[episode % 3]) if terminal else 0.0,
                    terminated=terminal,
                )
            )
    return rows


def optimizer_equal(left, right) -> bool:
    if left.keys() != right.keys() or left["param_groups"] != right["param_groups"]:
        return False
    if left["state"].keys() != right["state"].keys():
        return False
    for key, lstate in left["state"].items():
        rstate = right["state"][key]
        if lstate.keys() != rstate.keys():
            return False
        for name, value in lstate.items():
            if torch.is_tensor(value):
                if not torch.equal(value, rstate[name]):
                    return False
            elif value != rstate[name]:
                return False
    return True


def policy_for(config):
    state = torch.load(BC, map_location="cpu", weights_only=True)
    return B4DirectHeadPolicy(state, config)


def safe_after_multiplier(config, multiplier):
    policy = policy_for(config)
    ref = reference(policy)
    batch = build_batch(rollout(policy), config)
    actor, critic = build_optimizers(policy)
    update = update_policy_with_safe_cap(
        policy,
        batch,
        ref,
        actor,
        critic,
        seed=1,
        iteration=1,
        safe_cap=1e9,
        retry_multipliers=(multiplier,),
    )
    return float(update["safe_after"]["mean"])


def main() -> None:
    rows = []
    order = 0
    for map_name in SAFE_MAPS:
        for outcome in SAFE_OUTCOMES:
            for l4 in range(10):
                for duplicate in range(2):
                    rows.append(
                        {
                            "training_order": str(order),
                            "l2_id": f"L2:{order:064x}",
                            "l4_id": f"L4:{SAFE_MAPS.index(map_name):02x}{SAFE_OUTCOMES.index(outcome):02x}{l4:060x}",
                            "map_name": map_name,
                            "npz_relpath": f"eval_results/source/{outcome}/row_{order}.npz",
                        }
                    )
                    order += 1
    selected = select_reference_rows(rows)
    assert len(selected) == 64
    assert len({row["l2_id"] for row in selected}) == 64
    assert len({row["l4_id"] for row in selected}) == 64
    assert selected == select_reference_rows(list(reversed(rows)))

    config = replace(
        FROZEN_B4_CONFIG,
        actor_lr=1e-2,
        actor_epochs=1,
        critic_epochs=3,
        minibatch_size=64,
    )
    policy = policy_for(config)
    ref = reference(policy)
    assert safe_kl_metrics(policy, ref)["mean"] == 0.0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reference.npz"
        digest = save_reference(ref, path)
        loaded = load_reference(path)
        assert len(digest) == 64
        assert torch.equal(loaded.feature, ref.feature)
        assert torch.equal(loaded.bc_mean, ref.bc_mean)
        assert loaded.l2_ids == ref.l2_ids

    # Every nonzero candidate is rejected at cap zero. Actor and complete Adam
    # state return exactly, while the critic still completes all three epochs.
    batch = build_batch(rollout(policy), config)
    actor, critic = build_optimizers(policy)
    actor_state = copy.deepcopy(policy.actor.output_layer.state_dict())
    optimizer_state = copy.deepcopy(actor.state_dict())
    rejected = update_policy_with_safe_cap(
        policy,
        batch,
        ref,
        actor,
        critic,
        seed=1,
        iteration=1,
        safe_cap=0.0,
    )
    assert rejected["actor_epochs_skipped"] == 1
    assert rejected["actor_epochs_accepted"] == 0
    assert rejected["critic_epochs_completed"] == 3
    assert all(
        torch.equal(actor_state[name], value)
        for name, value in policy.actor.output_layer.state_dict().items()
    )
    assert optimizer_equal(optimizer_state, actor.state_dict())

    full = safe_after_multiplier(config, 1.0)
    half = safe_after_multiplier(config, 0.5)
    assert 0.0 < half < full
    cap = (full + half) / 2.0
    policy = policy_for(config)
    ref = reference(policy)
    batch = build_batch(rollout(policy), config)
    actor, critic = build_optimizers(policy)
    accepted = update_policy_with_safe_cap(
        policy,
        batch,
        ref,
        actor,
        critic,
        seed=1,
        iteration=1,
        safe_cap=cap,
        retry_multipliers=(1.0, 0.5),
    )
    assert accepted["actor_epochs_accepted"] == 1
    assert accepted["actor_epoch_records"][0]["accepted_multiplier"] == 0.5
    assert accepted["safe_after"]["mean"] <= cap
    assert all(abs(group["lr"] - config.actor_lr) < 1e-15 for group in actor.param_groups)

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "full.pt"
        save_b5_full_checkpoint(
            policy,
            actor,
            critic,
            checkpoint,
            completed_iteration=1,
            seed=1,
            run_plan_sha256="a" * 64,
            curriculum_sha256="b" * 64,
            reference_sha256="c" * 64,
        )
        restored = policy_for(config)
        restored_actor, restored_critic = build_optimizers(restored)
        completed = load_b5_full_checkpoint(
            checkpoint,
            restored,
            restored_actor,
            restored_critic,
            expected_seed=1,
            expected_run_plan_sha256="a" * 64,
            expected_curriculum_sha256="b" * 64,
            expected_reference_sha256="c" * 64,
            restore_rng=False,
        )
        assert completed == 1
        assert all(
            torch.equal(value, restored.actor.state_dict()[name])
            for name, value in policy.actor.state_dict().items()
        )
    print("B5 safe-reference and actor+Adam hard-cap contracts passed")


if __name__ == "__main__":
    main()

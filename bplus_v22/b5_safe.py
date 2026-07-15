"""B5-A canonical-BC safe-reference hard-cap contracts.

B5-A deliberately reuses the complete B4 actor, rollout, reward, curriculum,
and PPO configuration.  Its only scientific change is an empirical cumulative
mean-KL cap on one fixed set of canonical-BC safe trajectories.  Rejected
actor epochs restore both the output head and the complete Adam state before
the same minibatch order is retried at a lower temporary learning rate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from bplus_v22.b4_direct import (
    ACTION_DIM,
    FEATURE_DIM,
    B4Batch,
    B4DirectHeadPolicy,
    _atomic_torch_save,
    _clone_cpu_state,
    _minibatches,
    _permutation,
    _restore_rng_state,
    _rng_state,
    mean_bound_penalty,
    replay_metrics,
    strict_plain_actor_from_state,
)


B5_POLICY_SCHEMA = "end2race-b5-safe-direct-head-policy-1"
B5_REFERENCE_SCHEMA = "end2race-b5-safe-reference-1"
B5_SELECTION_VERSION = "end2race-b5-safe-selection-v1"
B5_SELECTION_DOMAIN = b"end2race:b5:safe-reference:selection:v1\0"
B5_FULL_CHECKPOINT_SCHEMA = "end2race-b5-safe-full-checkpoint-1"
B5_PILOT_SCHEMA = "end2race-b5-safe-pilot-1"
B5_PLUMBING_SCHEMA = "end2race-b5-safe-plumbing-smoke-1"
B5_READY_SCHEMA = "end2race-b5-safe-ready-1"

SAFE_MAPS = ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")
SAFE_OUTCOMES = ("follow", "overtake")
SAFE_EPISODES_PER_STRATUM = 8
SAFE_EPISODE_COUNT = len(SAFE_MAPS) * len(SAFE_OUTCOMES) * SAFE_EPISODES_PER_STRATUM
SAFE_CAP = 0.01
SAFE_RETRY_MULTIPLIERS = (1.0, 0.5, 0.25, 0.125, 0.0625)
SAFE_METRIC_CHUNK = 8192


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_hash(
    purpose: str,
    map_name: str,
    outcome: str,
    l4_id: str,
    l2_id: str,
) -> str:
    """Return the frozen domain-separated B5 reference selection key."""

    if purpose not in {"within_l4", "rank_l4"}:
        raise ValueError("B5 reference selection purpose is invalid")
    digest = hashlib.sha256(B5_SELECTION_DOMAIN)
    for value in (purpose, map_name, outcome, l4_id, l2_id):
        digest.update(value.encode("utf-8") + b"\0")
    return digest.hexdigest()


def select_reference_rows(rows: Sequence[Mapping[str, str]]) -> tuple[dict[str, str], ...]:
    """Select exactly eight unique-L4 rows from each map/safe-outcome stratum."""

    selected: list[dict[str, str]] = []
    l2_seen = set()
    for row in rows:
        l2_id = str(row.get("l2_id", ""))
        if l2_id in l2_seen:
            raise ValueError("B5 training reference input has duplicate L2")
        l2_seen.add(l2_id)
    for map_name in SAFE_MAPS:
        for outcome in SAFE_OUTCOMES:
            candidates = [
                dict(row)
                for row in rows
                if row.get("map_name") == map_name
                and f"/{outcome}/" in str(row.get("npz_relpath", ""))
            ]
            by_l4: dict[str, list[dict[str, str]]] = {}
            for row in candidates:
                by_l4.setdefault(str(row["l4_id"]), []).append(row)
            representatives = []
            for l4_id, members in by_l4.items():
                representatives.append(
                    min(
                        members,
                        key=lambda row: (
                            selection_hash(
                                "within_l4",
                                map_name,
                                outcome,
                                l4_id,
                                str(row["l2_id"]),
                            ),
                            str(row["l2_id"]),
                        ),
                    )
                )
            ordered = sorted(
                representatives,
                key=lambda row: (
                    selection_hash(
                        "rank_l4",
                        map_name,
                        outcome,
                        str(row["l4_id"]),
                        str(row["l2_id"]),
                    ),
                    str(row["l4_id"]),
                    str(row["l2_id"]),
                ),
            )
            if len(ordered) < SAFE_EPISODES_PER_STRATUM:
                raise ValueError(
                    f"B5 reference stratum lacks eight unique L4 rows: {map_name}/{outcome}"
                )
            selected.extend(ordered[:SAFE_EPISODES_PER_STRATUM])
    if len(selected) != SAFE_EPISODE_COUNT:
        raise AssertionError("B5 reference selection count drift")
    if len({row["l2_id"] for row in selected}) != SAFE_EPISODE_COUNT:
        raise AssertionError("B5 reference L2 selection is not unique")
    for map_name in SAFE_MAPS:
        for outcome in SAFE_OUTCOMES:
            group = [
                row
                for row in selected
                if row["map_name"] == map_name and f"/{outcome}/" in row["npz_relpath"]
            ]
            if len(group) != SAFE_EPISODES_PER_STRATUM or len(
                {row["l4_id"] for row in group}
            ) != SAFE_EPISODES_PER_STRATUM:
                raise AssertionError("B5 reference unique-L4 stratum contract drift")
    return tuple(selected)


@dataclass(frozen=True)
class SafeReference:
    feature: torch.Tensor
    bc_mean: torch.Tensor
    episode_index: torch.Tensor
    step_index: torch.Tensor
    lengths: tuple[int, ...]
    l2_ids: tuple[str, ...]
    l4_ids: tuple[str, ...]
    map_names: tuple[str, ...]
    outcomes: tuple[str, ...]
    selection_version: str = B5_SELECTION_VERSION

    def __post_init__(self) -> None:
        count = int(self.feature.shape[0])
        if self.feature.ndim != 2 or self.feature.shape[1] != FEATURE_DIM:
            raise ValueError("B5 reference feature shape is invalid")
        if self.bc_mean.shape != (count, ACTION_DIM):
            raise ValueError("B5 reference BC mean shape is invalid")
        if self.episode_index.shape != (count,) or self.step_index.shape != (count,):
            raise ValueError("B5 reference frame index shape is invalid")
        if len(self.lengths) != SAFE_EPISODE_COUNT or sum(self.lengths) != count:
            raise ValueError("B5 reference episode lengths are invalid")
        metadata = (self.l2_ids, self.l4_ids, self.map_names, self.outcomes)
        if any(len(values) != SAFE_EPISODE_COUNT for values in metadata):
            raise ValueError("B5 reference episode metadata count is invalid")
        if len(set(self.l2_ids)) != SAFE_EPISODE_COUNT:
            raise ValueError("B5 reference episode L2 inventory is not unique")
        if self.selection_version != B5_SELECTION_VERSION:
            raise ValueError("B5 reference selection version drift")
        if self.feature.dtype != torch.float32 or self.bc_mean.dtype != torch.float32:
            raise ValueError("B5 reference feature/mean must be float32")
        if not torch.all(torch.isfinite(self.feature)) or not torch.all(
            torch.isfinite(self.bc_mean)
        ):
            raise ValueError("B5 reference contains non-finite values")
        expected_episode = torch.repeat_interleave(
            torch.arange(SAFE_EPISODE_COUNT, dtype=torch.int64),
            torch.tensor(self.lengths, dtype=torch.int64),
        )
        if not torch.equal(self.episode_index.cpu(), expected_episode):
            raise ValueError("B5 reference episode index is not contiguous")
        expected_steps = torch.cat(
            [torch.arange(length, dtype=torch.int64) for length in self.lengths]
        )
        if not torch.equal(self.step_index.cpu(), expected_steps):
            raise ValueError("B5 reference step index is not contiguous")
        strata = {
            (map_name, outcome): 0 for map_name in SAFE_MAPS for outcome in SAFE_OUTCOMES
        }
        l4_by_stratum: dict[tuple[str, str], set[str]] = {
            key: set() for key in strata
        }
        for l4_id, map_name, outcome in zip(self.l4_ids, self.map_names, self.outcomes):
            key = (map_name, outcome)
            if key not in strata:
                raise ValueError("B5 reference contains an unapproved stratum")
            strata[key] += 1
            l4_by_stratum[key].add(l4_id)
        if any(value != SAFE_EPISODES_PER_STRATUM for value in strata.values()) or any(
            len(values) != SAFE_EPISODES_PER_STRATUM for values in l4_by_stratum.values()
        ):
            raise ValueError("B5 reference stratum/L4 balance drift")

    @property
    def frame_count(self) -> int:
        return int(self.feature.shape[0])

    def to(self, device: torch.device | str) -> "SafeReference":
        return SafeReference(
            feature=self.feature.to(device),
            bc_mean=self.bc_mean.to(device),
            episode_index=self.episode_index.to(device),
            step_index=self.step_index.to(device),
            lengths=self.lengths,
            l2_ids=self.l2_ids,
            l4_ids=self.l4_ids,
            map_names=self.map_names,
            outcomes=self.outcomes,
            selection_version=self.selection_version,
        )


def save_reference(reference: SafeReference, path: str | Path) -> str:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() or temporary.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": B5_REFERENCE_SCHEMA,
        "selection_version": reference.selection_version,
        "safe_maps": list(SAFE_MAPS),
        "safe_outcomes": list(SAFE_OUTCOMES),
        "episodes_per_stratum": SAFE_EPISODES_PER_STRATUM,
        "episode_count": SAFE_EPISODE_COUNT,
        "frame_count": reference.frame_count,
    }
    with temporary.open("xb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            feature=reference.feature.cpu().numpy(),
            bc_mean=reference.bc_mean.cpu().numpy(),
            episode_index=reference.episode_index.cpu().numpy(),
            step_index=reference.step_index.cpu().numpy(),
            lengths=np.asarray(reference.lengths, dtype=np.int64),
            l2_ids=np.asarray(reference.l2_ids, dtype="U67"),
            l4_ids=np.asarray(reference.l4_ids, dtype="U67"),
            map_names=np.asarray(reference.map_names),
            outcomes=np.asarray(reference.outcomes),
        )
        handle.flush()
    temporary.replace(destination)
    load_reference(destination)
    return file_sha256(destination)


def load_reference(path: str | Path, device: torch.device | str = "cpu") -> SafeReference:
    with np.load(Path(path), allow_pickle=False) as value:
        metadata = json.loads(str(value["metadata_json"].item()))
        if (
            metadata.get("schema") != B5_REFERENCE_SCHEMA
            or metadata.get("selection_version") != B5_SELECTION_VERSION
            or metadata.get("safe_maps") != list(SAFE_MAPS)
            or metadata.get("safe_outcomes") != list(SAFE_OUTCOMES)
            or metadata.get("episodes_per_stratum") != SAFE_EPISODES_PER_STRATUM
            or metadata.get("episode_count") != SAFE_EPISODE_COUNT
        ):
            raise ValueError("B5 reference metadata drift")
        reference = SafeReference(
            feature=torch.from_numpy(np.asarray(value["feature"], dtype=np.float32).copy()),
            bc_mean=torch.from_numpy(np.asarray(value["bc_mean"], dtype=np.float32).copy()),
            episode_index=torch.from_numpy(
                np.asarray(value["episode_index"], dtype=np.int64).copy()
            ),
            step_index=torch.from_numpy(np.asarray(value["step_index"], dtype=np.int64).copy()),
            lengths=tuple(int(item) for item in value["lengths"]),
            l2_ids=tuple(str(item) for item in value["l2_ids"]),
            l4_ids=tuple(str(item) for item in value["l4_ids"]),
            map_names=tuple(str(item) for item in value["map_names"]),
            outcomes=tuple(str(item) for item in value["outcomes"]),
            selection_version=str(metadata["selection_version"]),
        )
    if metadata.get("frame_count") != reference.frame_count:
        raise ValueError("B5 reference frame count drift")
    return reference.to(device)


def safe_kl_metrics(
    policy: B4DirectHeadPolicy,
    reference: SafeReference,
    *,
    chunk_size: int = SAFE_METRIC_CHUNK,
) -> dict[str, Any]:
    """Evaluate the approved episode-equivalent latent mean KL in float64."""

    if chunk_size <= 0:
        raise ValueError("B5 safe metric chunk size must be positive")
    device = next(policy.parameters()).device
    if reference.feature.device != device:
        raise ValueError("B5 safe reference and actor must share one device")
    std = policy.action_std.to(device=device, dtype=torch.float64)
    frame_parts = []
    policy.actor.output_layer.eval()
    with torch.no_grad():
        for start in range(0, reference.frame_count, int(chunk_size)):
            stop = min(reference.frame_count, start + int(chunk_size))
            mean = policy.mean_from_feature(reference.feature[start:stop]).to(torch.float64)
            bc_mean = reference.bc_mean[start:stop].to(torch.float64)
            frame_parts.append(0.5 * torch.sum(((mean - bc_mean) / std) ** 2, dim=1))
    frame_kl = torch.cat(frame_parts)
    episode_means = torch.empty(SAFE_EPISODE_COUNT, dtype=torch.float64, device=device)
    cursor = 0
    for episode, length in enumerate(reference.lengths):
        episode_means[episode] = frame_kl[cursor : cursor + length].mean()
        cursor += length
    if cursor != reference.frame_count:
        raise AssertionError("B5 safe metric episode boundary drift")
    stratum: dict[str, float] = {}
    for map_name in SAFE_MAPS:
        for outcome in SAFE_OUTCOMES:
            indices = [
                index
                for index, pair in enumerate(zip(reference.map_names, reference.outcomes))
                if pair == (map_name, outcome)
            ]
            stratum[f"{map_name}/{outcome}"] = float(
                episode_means[torch.tensor(indices, device=device)].mean().item()
            )
    return {
        "mean": float(episode_means.mean().item()),
        "frame_p50": float(torch.quantile(frame_kl, 0.50).item()),
        "frame_p95": float(torch.quantile(frame_kl, 0.95).item()),
        "frame_max": float(frame_kl.max().item()),
        "episode_p50": float(torch.quantile(episode_means, 0.50).item()),
        "episode_p95": float(torch.quantile(episode_means, 0.95).item()),
        "episode_max": float(episode_means.max().item()),
        "stratum_mean": stratum,
        "episode_count": SAFE_EPISODE_COUNT,
        "frame_count": reference.frame_count,
    }


def _clone_module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _restore_actor_epoch(
    policy: B4DirectHeadPolicy,
    actor_optimizer: torch.optim.Optimizer,
    actor_state: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, Any],
) -> None:
    policy.actor.output_layer.load_state_dict(actor_state, strict=True)
    actor_optimizer.load_state_dict(copy.deepcopy(optimizer_state))


def _set_actor_lr(actor_optimizer: torch.optim.Optimizer, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError("B5 actor retry LR is invalid")
    for group in actor_optimizer.param_groups:
        group["lr"] = float(value)


def update_policy_with_safe_cap(
    policy: B4DirectHeadPolicy,
    batch: B4Batch,
    reference: SafeReference,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    iteration: int,
    safe_cap: float = SAFE_CAP,
    retry_multipliers: Sequence[float] = SAFE_RETRY_MULTIPLIERS,
) -> dict[str, Any]:
    """Run the B4 PPO update with only the approved cumulative safe-KL cap."""

    if not math.isfinite(float(safe_cap)) or float(safe_cap) < 0.0:
        raise ValueError("B5 safe cap is invalid")
    multipliers = tuple(float(item) for item in retry_multipliers)
    if multipliers != tuple(sorted(set(multipliers), reverse=True)) or any(
        not math.isfinite(item) or item <= 0.0 or item > 1.0 for item in multipliers
    ):
        raise ValueError("B5 retry multiplier ladder is invalid")
    config = policy.config
    device = next(policy.parameters()).device
    batch = batch.to(device)
    if reference.feature.device != device:
        reference = reference.to(device)
    actor_parameter_ids = {
        id(parameter) for group in actor_optimizer.param_groups for parameter in group["params"]
    }
    critic_parameter_ids = {
        id(parameter) for group in critic_optimizer.param_groups for parameter in group["params"]
    }
    if actor_parameter_ids != set(map(id, policy.trainable_actor_parameters)):
        raise ValueError("B5 actor optimizer inventory drift")
    if critic_parameter_ids != set(map(id, policy.critic.parameters())):
        raise ValueError("B5 critic optimizer inventory drift")
    if actor_parameter_ids & critic_parameter_ids:
        raise ValueError("B5 actor/critic optimizer overlap")

    initial_safe = safe_kl_metrics(policy, reference)
    if initial_safe["mean"] > float(safe_cap) + 1e-12:
        raise ValueError("B5 pre-epoch actor already violates the cumulative safe cap")

    actor_epochs_considered = 0
    actor_epochs_accepted = 0
    actor_epochs_skipped = 0
    actor_stopped_early = False
    candidate_steps = 0
    accepted_steps = 0
    actor_losses: list[float] = []
    actor_epoch_records: list[dict[str, Any]] = []
    actor_batch_denominator = float(min(config.minibatch_size, batch.size))
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    policy.actor.output_layer.train()

    for epoch in range(config.actor_epochs):
        actor_epochs_considered += 1
        _set_actor_lr(actor_optimizer, config.actor_lr)
        pre_actor = _clone_module_state(policy.actor.output_layer)
        pre_optimizer = copy.deepcopy(actor_optimizer.state_dict())
        order = _permutation(batch.size, seed, iteration, epoch, "actor")
        attempts: list[dict[str, Any]] = []
        accepted = False
        accepted_losses: list[float] = []
        steps_per_attempt = math.ceil(batch.size / config.minibatch_size)
        for multiplier in multipliers:
            _restore_actor_epoch(policy, actor_optimizer, pre_actor, pre_optimizer)
            _set_actor_lr(actor_optimizer, config.actor_lr * multiplier)
            losses = []
            for cpu_indices in _minibatches(order, config.minibatch_size):
                indices = cpu_indices.to(device)
                mean = policy.mean_from_feature(batch.feature[indices])
                new_log_prob = policy.log_prob(mean, batch.raw_action[indices])
                ratio = torch.exp(new_log_prob - batch.old_log_prob[indices])
                advantage = batch.normalized_advantage[indices]
                surrogate = torch.minimum(
                    ratio * advantage,
                    torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
                    * advantage,
                )
                weighted_surrogate = (
                    batch.actor_weight[indices] * surrogate
                ).sum() / actor_batch_denominator
                bound = (
                    batch.actor_weight[indices] * mean_bound_penalty(mean)
                ).sum() / actor_batch_denominator
                loss = -weighted_surrogate + config.mean_bound_coef * bound
                actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(parameter.grad is not None for parameter in policy.frozen_actor_parameters):
                    raise AssertionError("B5 frozen actor received an actor gradient")
                if any(parameter.grad is not None for parameter in policy.critic.parameters()):
                    raise AssertionError("B5 critic received an actor gradient")
                nn.utils.clip_grad_norm_(
                    policy.trainable_actor_parameters, config.max_grad_norm
                )
                actor_optimizer.step()
                candidate_steps += 1
                losses.append(float(loss.detach().item()))
            safe = safe_kl_metrics(policy, reference)
            attempts.append(
                {
                    "multiplier": multiplier,
                    "temporary_lr": config.actor_lr * multiplier,
                    "candidate_optimizer_steps": steps_per_attempt,
                    "safe": safe,
                    "accepted": safe["mean"] <= float(safe_cap),
                }
            )
            if safe["mean"] <= float(safe_cap):
                accepted = True
                accepted_losses = losses
                break

        if not accepted:
            _restore_actor_epoch(policy, actor_optimizer, pre_actor, pre_optimizer)
            actor_epochs_skipped += 1
            _set_actor_lr(actor_optimizer, config.actor_lr)
            actor_epoch_records.append(
                {
                    "epoch": epoch + 1,
                    "accepted": False,
                    "skipped": True,
                    "attempts": attempts,
                    "post_epoch_safe": safe_kl_metrics(policy, reference),
                }
            )
            continue

        actor_epochs_accepted += 1
        accepted_steps += steps_per_attempt
        actor_losses.extend(accepted_losses)
        # The lower LR is a temporary hard-cap solver setting, never persistent
        # optimizer configuration.  Accepted Adam moments remain intact.
        _set_actor_lr(actor_optimizer, config.actor_lr)
        rollout = replay_metrics(policy, batch)
        safe = safe_kl_metrics(policy, reference)
        if safe["mean"] > float(safe_cap):
            raise AssertionError("B5 accepted actor epoch violates the safe cap")
        actor_epoch_records.append(
            {
                "epoch": epoch + 1,
                "accepted": True,
                "skipped": False,
                "accepted_multiplier": attempts[-1]["multiplier"],
                "attempts": attempts,
                "post_epoch_safe": safe,
                "rollout_replay": rollout,
            }
        )
        # Preserve literal B4 semantics: rollout KL never rejects an accepted
        # epoch; it only prevents later actor epochs.
        if rollout["weighted_kl"] > config.target_weighted_kl:
            actor_stopped_early = epoch + 1 < config.actor_epochs
            break

    critic_losses: list[float] = []
    critic_steps = 0
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    policy.critic.train()
    for epoch in range(config.critic_epochs):
        order = _permutation(batch.size, seed, iteration, epoch, "critic")
        for cpu_indices in _minibatches(order, config.minibatch_size):
            indices = cpu_indices.to(device)
            predicted = policy.value(batch.privileged_feature[indices])
            loss = torch.mean((predicted - batch.returns[indices]) ** 2)
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in policy.actor.parameters()):
                raise AssertionError("B5 actor received a critic gradient")
            nn.utils.clip_grad_norm_(tuple(policy.critic.parameters()), config.max_grad_norm)
            critic_optimizer.step()
            critic_steps += 1
            critic_losses.append(float(loss.detach().item()))

    _set_actor_lr(actor_optimizer, config.actor_lr)
    policy.assert_frozen_exact()
    final_safe = safe_kl_metrics(policy, reference)
    if final_safe["mean"] > float(safe_cap):
        raise AssertionError("B5 final actor violates the cumulative safe cap")
    expected_critic_steps = config.critic_epochs * math.ceil(
        batch.size / config.minibatch_size
    )
    if critic_steps != expected_critic_steps:
        raise AssertionError("B5 critic did not complete every frozen epoch")
    final_rollout = replay_metrics(policy, batch)
    return {
        "actor_epochs_considered": actor_epochs_considered,
        "actor_epochs_accepted": actor_epochs_accepted,
        "actor_epochs_skipped": actor_epochs_skipped,
        "actor_stopped_early": actor_stopped_early,
        "actor_candidate_optimizer_steps": candidate_steps,
        "actor_accepted_optimizer_steps": accepted_steps,
        "actor_epoch_records": actor_epoch_records,
        "critic_epochs_completed": config.critic_epochs,
        "critic_optimizer_steps": critic_steps,
        "actor_loss_mean": None if not actor_losses else float(np.mean(actor_losses)),
        "critic_loss_mean": float(np.mean(critic_losses)),
        "safe_cap": float(safe_cap),
        "safe_before": initial_safe,
        "safe_after": final_safe,
        **final_rollout,
    }


def save_b5_full_checkpoint(
    policy: B4DirectHeadPolicy,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    path: str | Path,
    *,
    completed_iteration: int,
    seed: int,
    run_plan_sha256: str,
    curriculum_sha256: str,
    reference_sha256: str,
) -> dict[str, Any]:
    policy.assert_frozen_exact()
    if completed_iteration < 0 or completed_iteration > policy.config.iterations:
        raise ValueError("B5 checkpoint iteration is invalid")
    if seed not in policy.config.seeds:
        raise ValueError("B5 checkpoint seed is invalid")
    for name, digest in (
        ("run_plan_sha256", run_plan_sha256),
        ("curriculum_sha256", curriculum_sha256),
        ("reference_sha256", reference_sha256),
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"B5 {name} is invalid")
    if any(abs(float(group["lr"]) - policy.config.actor_lr) > 1e-15 for group in actor_optimizer.param_groups):
        raise ValueError("B5 checkpoint actor optimizer retained a temporary retry LR")
    payload = {
        "schema": B5_FULL_CHECKPOINT_SCHEMA,
        "policy_schema": B5_POLICY_SCHEMA,
        "completed_iteration": int(completed_iteration),
        "seed": int(seed),
        "run_plan_sha256": run_plan_sha256,
        "curriculum_sha256": curriculum_sha256,
        "reference_sha256": reference_sha256,
        "config": policy.config.as_dict(),
        "safe_cap": SAFE_CAP,
        "retry_multipliers": list(SAFE_RETRY_MULTIPLIERS),
        "actor_state_dict": policy.actor_state(),
        "critic_state_dict": _clone_cpu_state(policy.critic),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "initial_frozen_state": {
            name: value.clone() for name, value in policy._initial_frozen.items()
        },
        "rng_state": _rng_state(),
    }
    _atomic_torch_save(payload, Path(path))
    return payload


def load_b5_full_checkpoint(
    path: str | Path,
    policy: B4DirectHeadPolicy,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    expected_seed: int,
    expected_run_plan_sha256: str,
    expected_curriculum_sha256: str,
    expected_reference_sha256: str,
    restore_rng: bool = True,
) -> int:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != B5_FULL_CHECKPOINT_SCHEMA:
        raise ValueError("B5 full checkpoint schema mismatch")
    checks = {
        "policy_schema": B5_POLICY_SCHEMA,
        "seed": int(expected_seed),
        "run_plan_sha256": expected_run_plan_sha256,
        "curriculum_sha256": expected_curriculum_sha256,
        "reference_sha256": expected_reference_sha256,
        "config": policy.config.as_dict(),
        "safe_cap": SAFE_CAP,
        "retry_multipliers": list(SAFE_RETRY_MULTIPLIERS),
    }
    if any(payload.get(name) != value for name, value in checks.items()):
        raise ValueError("B5 full checkpoint identity/config mismatch")
    actor_state = payload.get("actor_state_dict")
    strict_plain_actor_from_state(actor_state)
    policy.actor.load_state_dict(actor_state, strict=True)
    policy.critic.load_state_dict(payload["critic_state_dict"], strict=True)
    actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    initial_frozen = payload.get("initial_frozen_state")
    if not isinstance(initial_frozen, Mapping) or initial_frozen.keys() != policy._initial_frozen.keys():
        raise ValueError("B5 full checkpoint frozen inventory mismatch")
    if any(
        not torch.equal(initial_frozen[name].cpu(), policy._initial_frozen[name])
        for name in initial_frozen
    ):
        raise ValueError("B5 checkpoint was initialized from different frozen tensors")
    if any(abs(float(group["lr"]) - policy.config.actor_lr) > 1e-15 for group in actor_optimizer.param_groups):
        raise ValueError("B5 checkpoint actor optimizer contains a temporary retry LR")
    policy.assert_frozen_exact()
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    completed = payload.get("completed_iteration")
    if type(completed) is not int or not 0 <= completed <= policy.config.iterations:
        raise ValueError("B5 full checkpoint completed iteration is invalid")
    return completed

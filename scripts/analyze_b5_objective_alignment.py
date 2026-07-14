#!/usr/bin/env python3
"""Function-space audit of B5-A and the proposed opened-panel weighting.

The script is read-only.  It replays canonical BC histories to form a fixed
function probe, decomposes the first-epoch PPO gradient by archived BC outcome,
and compares the historical B5-A objective with the proposed Austin
opened-development prevalence weighting.  It does not train or select a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.func import functional_call, jvp

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bplus_v22.b4_direct import mean_bound_penalty
from bplus_v22.b5_safe import load_reference
from model import End2Race


SCHEMA = "end2race-b5-objective-alignment-audit-1"
STD = np.asarray([0.03, 0.20], dtype=np.float64)
OUTCOMES = ("collision", "overtake", "follow")
PROBE_GROUPS = ("bc_collision", "bc_overtake", "bc_follow", "safe_reference")
PRODUCT_COUNTS = {"collision": 24, "overtake": 342, "follow": 234}
TRAIN_COUNTS = {"collision": 6, "overtake": 6, "follow": 4}
PREVALENCE = {
    outcome: (PRODUCT_COUNTS[outcome] / 600.0) / (TRAIN_COUNTS[outcome] / 16.0)
    for outcome in OUTCOMES
}
PRODUCT_PROBE_FRAMES = 32
GRADIENT_ITERATIONS = 10
CHUNK = 4096


@dataclass(frozen=True)
class Probe:
    feature: torch.Tensor
    weight: torch.Tensor

    def __post_init__(self) -> None:
        if self.feature.ndim != 2 or self.feature.shape[1] != 1680:
            raise ValueError("function probe feature shape drift")
        if self.weight.shape != (len(self.feature),):
            raise ValueError("function probe weight shape drift")
        if not torch.isclose(self.weight.double().sum(), torch.tensor(1.0, dtype=torch.float64)):
            raise ValueError("function probe weights do not sum to one")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b4-evaluation-root",
        type=Path,
        default=Path(
            "Experiments/B4_direct_head_ppo/product_evaluations/"
            "b4_product_seed1_20260714_003027"
        ),
    )
    parser.add_argument(
        "--b5-training-root",
        type=Path,
        default=Path(
            "Experiments/B5_safe_trust_region/runs/b5_seed1_20260714_021544/"
            "hosts/remote/outputs/train/seed1"
        ),
    )
    parser.add_argument(
        "--safe-reference",
        type=Path,
        default=Path(
            "Experiments/B5_safe_trust_region/artifacts/reference_20260714_015710/"
            "safe_reference.npz"
        ),
    )
    parser.add_argument("--bc-checkpoint", type=Path, default=Path("pretrained/end2race.pth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gradient-iterations", type=int, default=GRADIENT_ITERATIONS)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty objective-alignment table")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("objective-alignment field order drift")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def actor_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        value = torch.load(path, map_location="cpu", weights_only=False)
    if "actor_state_dict" in value:
        value = value["actor_state_dict"]
    return value


def load_head(path: Path, device: torch.device) -> torch.nn.Module:
    state = actor_state(path)
    head = End2Race(mask_prob=0.0, hidden_scale=4).output_layer
    head.load_state_dict(
        {
            key.removeprefix("output_layer."): tensor
            for key, tensor in state.items()
            if key.startswith("output_layer.")
        },
        strict=True,
    )
    return head.eval().to(device)


def build_metric_index(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "BC").glob("shard[0-4]/metrics/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["npz_path"] = path.parent.parent / "npz" / f"{row['case_id']}.npz"
        rows.append(row)
    if len(rows) != 600 or len({row["case_id"] for row in rows}) != 600:
        raise ValueError("BC product-history inventory drift")
    return sorted(rows, key=lambda row: row["case_id"])


def raceline_initial_speed(metric: Mapping[str, Any]) -> float:
    path = REPO / "f1tenth_racetracks" / metric["map_name"] / f"{metric['ego_raceline']}.csv"
    rows = np.loadtxt(path, delimiter=";", skiprows=1)
    return float(rows[int(metric["ego_idx"]) % len(rows), 5])


def previous_speed(data: np.lib.npyio.NpzFile, metric: Mapping[str, Any]) -> np.ndarray:
    actual = np.asarray(data["ego_actual_speed"], dtype=np.float32)
    value = np.empty((len(actual), 1), dtype=np.float32)
    value[0, 0] = raceline_initial_speed(metric) * 0.9
    value[1:, 0] = actual[:-1]
    return value


def product_probe(
    rows: Sequence[Mapping[str, Any]],
    backbone: End2Race,
    device: torch.device,
) -> tuple[dict[str, Probe], dict[str, float]]:
    features: dict[str, list[torch.Tensor]] = {group: [] for group in PROBE_GROUPS[:3]}
    max_steer_error = 0.0
    max_speed_error = 0.0
    with torch.inference_mode():
        for episode_index, metric in enumerate(rows, start=1):
            with np.load(metric["npz_path"], allow_pickle=False) as data:
                lidar = torch.from_numpy(np.asarray(data["ego_lidar"], dtype=np.float32)).to(device)
                speed = torch.from_numpy(previous_speed(data, metric)).to(device)
                frame_indices = np.rint(
                    np.linspace(0, len(lidar) - 1, PRODUCT_PROBE_FRAMES)
                ).astype(np.int64)
                if len(np.unique(frame_indices)) != PRODUCT_PROBE_FRAMES:
                    raise ValueError("product history is too short for fixed function probe")
                wanted = set(int(value) for value in frame_indices)
                selected = []
                hidden = None
                for step in range(len(lidar)):
                    feature, hidden = backbone.forward_features(
                        lidar[step : step + 1, None, :],
                        speed[step : step + 1, None, :],
                        hidden,
                    )
                    if step in wanted:
                        selected.append((step, feature[:, -1, :].cpu()))
                selected.sort()
                episode_feature = torch.cat([value for _, value in selected], dim=0)
                predicted = backbone.output_layer(episode_feature.to(device)).cpu().numpy()
                selected_index = np.asarray([step for step, _ in selected], dtype=np.int64)
                stored_steer = np.asarray(data["ego_desired_steer"], dtype=np.float32)[selected_index]
                stored_speed = np.asarray(data["ego_desired_speed"], dtype=np.float32)[selected_index]
                max_steer_error = max(
                    max_steer_error,
                    float(np.max(np.abs(np.clip(predicted[:, 0], -0.52, 0.52) - stored_steer))),
                )
                max_speed_error = max(
                    max_speed_error,
                    float(np.max(np.abs(predicted[:, 1] - stored_speed))),
                )
                outcome = str(metric["outcome"])
                if outcome == "overtaking":
                    outcome = "overtake"
                elif outcome == "following":
                    outcome = "follow"
                if outcome not in OUTCOMES:
                    raise ValueError(f"unexpected BC product outcome: {outcome}")
                features[f"bc_{outcome}"].append(episode_feature)
            if episode_index % 50 == 0:
                print(f"product_probe_episodes={episode_index}/600", flush=True)
    if max_steer_error > 2e-5 or max_speed_error > 2e-5:
        raise AssertionError(
            f"stepwise BC probe mismatch: steer={max_steer_error}, speed={max_speed_error}"
        )
    probes = {}
    for group, episodes in features.items():
        feature = torch.cat(episodes, dim=0).contiguous()
        episode_count = len(episodes)
        weight = torch.full(
            (len(feature),), 1.0 / (episode_count * PRODUCT_PROBE_FRAMES), dtype=torch.float64
        )
        probes[group] = Probe(feature, weight)
    return probes, {
        "max_abs_bc_steer_error": max_steer_error,
        "max_abs_bc_speed_error": max_speed_error,
    }


def safe_probe(path: Path) -> Probe:
    reference = load_reference(path, "cpu")
    weight = torch.empty(reference.frame_count, dtype=torch.float64)
    cursor = 0
    for length in reference.lengths:
        weight[cursor : cursor + length] = 1.0 / (len(reference.lengths) * length)
        cursor += length
    return Probe(reference.feature.cpu(), weight)


def means(head: torch.nn.Module, feature: torch.Tensor, device: torch.device) -> np.ndarray:
    values = []
    with torch.inference_mode():
        for start in range(0, len(feature), CHUNK):
            values.append(head(feature[start : start + CHUNK].to(device)).cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def weighted_direction_metrics(
    direction: np.ndarray,
    reference: np.ndarray,
    weight: torch.Tensor,
) -> dict[str, float | str]:
    direction = np.asarray(direction, dtype=np.float64) / STD
    reference = np.asarray(reference, dtype=np.float64) / STD
    weights = weight.numpy().astype(np.float64)
    direction_sq = float(np.sum(weights * np.sum(direction * direction, axis=1)))
    reference_sq = float(np.sum(weights * np.sum(reference * reference, axis=1)))
    inner = float(np.sum(weights * np.sum(direction * reference, axis=1)))
    direction_norm = math.sqrt(max(0.0, direction_sq))
    reference_norm = math.sqrt(max(0.0, reference_sq))
    cosine: float | str = ""
    if direction_norm > 1e-14 and reference_norm > 1e-14:
        cosine = inner / (direction_norm * reference_norm)
    return {
        "direction_norm": direction_norm,
        "reference_norm": reference_norm,
        "inner_product": inner,
        "cosine": cosine,
    }


def weighted_normalize(advantage: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denominator = weight.sum()
    mean = (weight * advantage).sum() / denominator
    variance = (weight * (advantage - mean) ** 2).sum() / denominator
    return (advantage - mean) / torch.sqrt(variance + 1e-8)


def normal_log_prob(mean: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    std = torch.tensor([0.03, 0.20], dtype=mean.dtype, device=mean.device)
    return torch.sum(
        -0.5 * ((action - mean) / std) ** 2
        - torch.log(std)
        - 0.5 * math.log(2.0 * math.pi),
        dim=1,
    )


def replay_inventory(
    replay_path: Path,
    curriculum: Mapping[str, str],
) -> dict[str, Any]:
    with np.load(replay_path, allow_pickle=False) as value:
        result = {name: np.asarray(value[name]).copy() for name in value.files}
    labels = np.asarray([curriculum[str(l2_id)] for l2_id in result["l2_ids"]])
    if set(labels) != set(OUTCOMES):
        raise ValueError("replay archived outcome inventory drift")
    result["outcome"] = labels
    return result


def objective_gradients(
    head: torch.nn.Module,
    replay: Mapping[str, Any],
    device: torch.device,
    objective: str,
) -> tuple[dict[str, tuple[torch.Tensor, ...]], dict[str, float]]:
    feature = torch.from_numpy(replay["feature"]).to(device)
    action = torch.from_numpy(replay["raw_action"]).to(device)
    old_log_prob = torch.from_numpy(replay["old_log_prob"]).to(device)
    advantage = torch.from_numpy(replay["advantage"]).to(device)
    trust_weight = torch.from_numpy(replay["actor_weight"]).to(device)
    outcome = np.asarray(replay["outcome"])
    if objective == "b5_original":
        objective_weight = trust_weight.clone()
    elif objective == "opened_austin_prevalence":
        multipliers = torch.tensor(
            [PREVALENCE[str(label)] for label in outcome],
            dtype=trust_weight.dtype,
            device=device,
        )
        objective_weight = trust_weight * multipliers
    else:
        raise ValueError(f"unknown gradient objective: {objective}")
    if abs(float(objective_weight.mean()) - 1.0) > 2e-5:
        raise AssertionError("objective weights do not retain global mean one")
    normalized = weighted_normalize(advantage, objective_weight)
    if objective == "b5_original":
        stored = torch.from_numpy(replay["normalized_advantage"]).to(device)
        max_error = float(torch.max(torch.abs(normalized - stored)).item())
        if max_error > 2e-5:
            raise AssertionError(f"stored B5 normalized advantage mismatch: {max_error}")
    parameters = tuple(head.parameters())
    gradients: dict[str, tuple[torch.Tensor, ...]] = {}
    for component in OUTCOMES:
        head.zero_grad(set_to_none=True)
        for start in range(0, len(feature), CHUNK):
            stop = min(start + CHUNK, len(feature))
            mask = torch.from_numpy((outcome[start:stop] == component).astype(np.float32)).to(device)
            mean = head(feature[start:stop])
            log_prob = normal_log_prob(mean, action[start:stop])
            ratio = torch.exp(log_prob - old_log_prob[start:stop])
            current_advantage = normalized[start:stop]
            surrogate = torch.minimum(
                ratio * current_advantage,
                torch.clamp(ratio, 0.9, 1.1) * current_advantage,
            )
            current_weight = objective_weight[start:stop] * mask
            loss = (
                -(current_weight * surrogate).sum()
                + 0.01 * (current_weight * mean_bound_penalty(mean)).sum()
            ) / len(feature)
            loss.backward()
        gradients[component] = tuple(
            parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter)
            for parameter in parameters
        )
    gradients["all"] = tuple(
        sum((gradients[outcome_name][index] for outcome_name in OUTCOMES), torch.zeros_like(parameter))
        for index, parameter in enumerate(parameters)
    )
    with torch.no_grad():
        pre_log_prob = []
        for start in range(0, len(feature), CHUNK):
            stop = min(start + CHUNK, len(feature))
            pre_log_prob.append(normal_log_prob(head(feature[start:stop]), action[start:stop]))
        pre_log_prob = torch.cat(pre_log_prob)
        max_ratio_error = float(
            torch.max(torch.abs(torch.exp(pre_log_prob - old_log_prob) - 1.0)).item()
        )
    return gradients, {
        "objective_weight_mean": float(objective_weight.mean().item()),
        "normalized_advantage_weighted_mean": float(
            (objective_weight * normalized).sum().item() / objective_weight.sum().item()
        ),
        "max_abs_preupdate_ratio_minus_one": max_ratio_error,
    }


def jvp_outputs(
    head: torch.nn.Module,
    tangent: Sequence[torch.Tensor],
    feature: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    names = tuple(name for name, _ in head.named_parameters())
    primals = tuple(parameter.detach() for parameter in head.parameters())
    tangent = tuple(value.to(device) for value in tangent)
    values = []
    for start in range(0, len(feature), CHUNK):
        current = feature[start : start + CHUNK].to(device)

        def function(*parameters: torch.Tensor) -> torch.Tensor:
            return functional_call(head, dict(zip(names, parameters)), (current,))

        _, directional = jvp(function, primals, tangent)
        values.append(directional.detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def scalar_mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> None:
    args = parse_args()
    if not 1 <= args.gradient_iterations <= 30:
        raise ValueError("gradient iteration range must be within B5-A")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    bc_state = actor_state(args.bc_checkpoint)
    backbone = End2Race(mask_prob=0.0, hidden_scale=4)
    backbone.load_state_dict(bc_state, strict=True)
    backbone.eval().to(device)
    probes, replay_validation = product_probe(
        build_metric_index(args.b4_evaluation_root), backbone, device
    )
    probes["safe_reference"] = safe_probe(args.safe_reference)

    b4_head = load_head(args.b4_evaluation_root / "models/seed1_iter30.pth", device)
    bc_head = load_head(args.bc_checkpoint, device)
    bc_means = {group: means(bc_head, probe.feature, device) for group, probe in probes.items()}
    b4_direction = {
        group: means(b4_head, probe.feature, device) - bc_means[group]
        for group, probe in probes.items()
    }
    reference = load_reference(args.safe_reference, "cpu")
    safe_bc_error = float(
        np.max(np.abs(bc_means["safe_reference"] - reference.bc_mean.numpy()))
    )
    if safe_bc_error > 2e-5:
        raise AssertionError(f"safe-reference BC mean mismatch: {safe_bc_error}")

    checkpoint_paths = {
        iteration: args.b5_training_root / f"checkpoints/iter_{iteration:04d}.pt"
        for iteration in range(31)
    }
    checkpoint_means: dict[int, dict[str, np.ndarray]] = {}
    for iteration, path in checkpoint_paths.items():
        head = load_head(path, device)
        checkpoint_means[iteration] = {
            group: means(head, probe.feature, device) for group, probe in probes.items()
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"checkpoint_function_probe={iteration}/30", flush=True)

    function_rows: list[dict[str, Any]] = []
    for iteration in range(1, 31):
        for group, probe in probes.items():
            delta = checkpoint_means[iteration][group] - checkpoint_means[iteration - 1][group]
            b4 = weighted_direction_metrics(delta, b4_direction[group], probe.weight)
            safe_displacement = checkpoint_means[iteration - 1][group] - bc_means[group]
            cap = weighted_direction_metrics(delta, safe_displacement, probe.weight)
            function_rows.append(
                {
                    "iteration": iteration,
                    "probe_group": group,
                    "standardized_update_norm": b4["direction_norm"],
                    "b4_global_direction_norm": b4["reference_norm"],
                    "cosine_with_b4_global_direction": b4["cosine"],
                    "inner_with_b4_global_direction": b4["inner_product"],
                    "cosine_with_preupdate_bc_displacement": cap["cosine"],
                    "first_order_bc_displacement_increase": cap["inner_product"],
                    "mean_signed_steer_update": float(np.sum(probe.weight.numpy() * delta[:, 0])),
                    "mean_signed_speed_update": float(np.sum(probe.weight.numpy() * delta[:, 1])),
                }
            )

    curriculum_rows = json.loads(
        (args.b5_training_root / "curriculum.json").read_text(encoding="utf-8")
    )["rows"]
    curriculum = {row["l2_id"]: row["archived_bc_outcome"] for row in curriculum_rows}
    gradient_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    for iteration in range(1, args.gradient_iterations + 1):
        replay = replay_inventory(
            args.b5_training_root / f"replay/iter_{iteration:04d}.npz", curriculum
        )
        head = load_head(checkpoint_paths[iteration - 1], device)
        iteration_objective: dict[str, dict[str, Any]] = {}
        for objective in ("b5_original", "opened_austin_prevalence"):
            gradients, checks = objective_gradients(head, replay, device, objective)
            objective_records = []
            for component in (*OUTCOMES, "all"):
                tangent = tuple(-value for value in gradients[component])
                parameter_norm = math.sqrt(
                    sum(float(torch.sum(value.double() ** 2).item()) for value in tangent)
                )
                component_records = []
                for group, probe in probes.items():
                    functional_direction = jvp_outputs(head, tangent, probe.feature, device)
                    b4 = weighted_direction_metrics(
                        functional_direction, b4_direction[group], probe.weight
                    )
                    actual = weighted_direction_metrics(
                        functional_direction,
                        checkpoint_means[iteration][group]
                        - checkpoint_means[iteration - 1][group],
                        probe.weight,
                    )
                    cap = weighted_direction_metrics(
                        functional_direction,
                        checkpoint_means[iteration - 1][group] - bc_means[group],
                        probe.weight,
                    )
                    row = {
                        "iteration": iteration,
                        "objective": objective,
                        "gradient_component": component,
                        "probe_group": group,
                        "parameter_gradient_l2": parameter_norm,
                        "functional_direction_norm": b4["direction_norm"],
                        "cosine_with_b4_global_direction": b4["cosine"],
                        "inner_with_b4_global_direction": b4["inner_product"],
                        "cosine_with_actual_b5_update": actual["cosine"],
                        "cosine_with_preupdate_bc_displacement": cap["cosine"],
                        "first_order_bc_displacement_increase": cap["inner_product"],
                    }
                    gradient_rows.append(row)
                    component_records.append(row)
                objective_records.extend(component_records)
            all_records = [row for row in objective_records if row["gradient_component"] == "all"]
            collision_records = [
                row for row in objective_records if row["gradient_component"] == "collision"
            ]
            safe_all = next(row for row in all_records if row["probe_group"] == "safe_reference")
            iteration_objective[objective] = {
                "checks": checks,
                "mean_b4_global_cosine": scalar_mean(
                    [float(row["cosine_with_b4_global_direction"]) for row in all_records]
                ),
                "safe_cap_alignment_cosine": safe_all[
                    "cosine_with_preupdate_bc_displacement"
                ],
                "safe_cap_first_order_increase": safe_all[
                    "first_order_bc_displacement_increase"
                ],
                "mean_collision_component_functional_norm": scalar_mean(
                    [float(row["functional_direction_norm"]) for row in collision_records]
                ),
            }
        raw = iteration_objective["b5_original"]
        corrected = iteration_objective["opened_austin_prevalence"]
        collision_norm_ratio = (
            corrected["mean_collision_component_functional_norm"]
            / raw["mean_collision_component_functional_norm"]
            if raw["mean_collision_component_functional_norm"] > 0.0
            else ""
        )
        iteration_rows.append(
            {
                "iteration": iteration,
                "raw_mean_b4_global_cosine": raw["mean_b4_global_cosine"],
                "corrected_mean_b4_global_cosine": corrected["mean_b4_global_cosine"],
                "corrected_minus_raw_b4_cosine": corrected["mean_b4_global_cosine"]
                - raw["mean_b4_global_cosine"],
                "raw_safe_cap_alignment_cosine": raw["safe_cap_alignment_cosine"],
                "corrected_safe_cap_alignment_cosine": corrected[
                    "safe_cap_alignment_cosine"
                ],
                "corrected_minus_raw_cap_cosine": (
                    float(corrected["safe_cap_alignment_cosine"])
                    - float(raw["safe_cap_alignment_cosine"])
                    if corrected["safe_cap_alignment_cosine"] != ""
                    and raw["safe_cap_alignment_cosine"] != ""
                    else ""
                ),
                "raw_safe_cap_first_order_increase": raw["safe_cap_first_order_increase"],
                "corrected_safe_cap_first_order_increase": corrected[
                    "safe_cap_first_order_increase"
                ],
                "collision_component_functional_norm_ratio": collision_norm_ratio,
                "raw_max_abs_preupdate_ratio_minus_one": raw["checks"][
                    "max_abs_preupdate_ratio_minus_one"
                ],
                "corrected_max_abs_preupdate_ratio_minus_one": corrected["checks"][
                    "max_abs_preupdate_ratio_minus_one"
                ],
            }
        )
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"gradient_decomposition={iteration}/{args.gradient_iterations}", flush=True)

    finite_cap = [
        row for row in iteration_rows if row["corrected_minus_raw_cap_cosine"] != ""
    ]
    summary = {
        "schema": SCHEMA,
        "analysis_status": "read-only post-hoc opened-development mechanism audit",
        "device": str(device),
        "input": {
            "b4_evaluation_root": str(args.b4_evaluation_root),
            "b5_training_root": str(args.b5_training_root),
            "safe_reference": str(args.safe_reference),
            "safe_reference_sha256": sha256_file(args.safe_reference),
            "b5_curriculum_sha256": sha256_file(args.b5_training_root / "curriculum.json"),
            "gradient_iterations": args.gradient_iterations,
        },
        "probe": {
            "groups": {
                group: {"frames": len(probe.feature), "weight_sum": float(probe.weight.sum())}
                for group, probe in probes.items()
            },
            "product_frames_per_episode": PRODUCT_PROBE_FRAMES,
            "product_replay_validation": replay_validation,
            "safe_reference_bc_mean_max_abs_error": safe_bc_error,
        },
        "opened_austin_prevalence_weights": PREVALENCE,
        "aggregate_comparison": {
            "iterations_with_lower_b4_global_cosine": sum(
                row["corrected_minus_raw_b4_cosine"] < 0.0 for row in iteration_rows
            ),
            "iterations_with_lower_safe_cap_cosine": sum(
                row["corrected_minus_raw_cap_cosine"] < 0.0 for row in finite_cap
            ),
            "finite_safe_cap_comparison_iterations": len(finite_cap),
            "median_corrected_minus_raw_b4_cosine": float(
                np.median([row["corrected_minus_raw_b4_cosine"] for row in iteration_rows])
            ),
            "median_corrected_minus_raw_cap_cosine": float(
                np.median([row["corrected_minus_raw_cap_cosine"] for row in finite_cap])
            )
            if finite_cap
            else "",
            "median_collision_component_functional_norm_ratio": float(
                np.median(
                    [
                        row["collision_component_functional_norm_ratio"]
                        for row in iteration_rows
                        if row["collision_component_functional_norm_ratio"] != ""
                    ]
                )
            ),
        },
        "interpretation_contract": {
            "gradient": (
                "full-rollout first-order loss gradient at the pre-update checkpoint, decomposed "
                "with one shared globally normalized advantage; it is not the executed Adam epoch"
            ),
            "function_space": (
                "J times negative loss-gradient on fixed canonical-BC probes; product probes use "
                "32 deterministic frames per episode and safe-reference uses every frame"
            ),
            "weight_status": (
                "weights are tuned from the already-open Austin development panel, not a fresh or "
                "universal product prevalence estimate"
            ),
            "causal_boundary": (
                "alignment can support or weaken the objective-mismatch hypothesis but cannot by "
                "itself prove that a weighted PPO run will improve collision"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "function_updates.tsv", function_rows)
    write_tsv(args.output_dir / "gradient_alignment.tsv", gradient_rows)
    write_tsv(args.output_dir / "gradient_iteration_summary.tsv", iteration_rows)
    json_dump(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

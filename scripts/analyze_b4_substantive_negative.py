#!/usr/bin/env python3
"""Reproduce the post-hoc diagnosis of the closed B4 PPO experiment.

This script is deliberately read-only with respect to experiment artifacts.  It
derives compact, reviewable tables from the frozen product evaluation, training
replays, curriculum, and actor snapshots.  It does not select a candidate or
authorize another experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import End2Race


VARIANTS = ("BC", "seed1_iter10", "seed1_iter20", "seed1_iter30")
CANDIDATES = VARIANTS[1:]
STD = np.asarray([0.03, 0.20], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path(
            "Experiments/B4_direct_head_ppo/product_evaluations/"
            "b4_product_seed1_20260714_003027"
        ),
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path(
            "Experiments/B4_direct_head_ppo/runs/b4_seed1_20260714_003027/"
            "hosts/remote/outputs/train/seed1"
        ),
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path(
            "Experiments/B1_route_r2_scaffold/artifacts/"
            "task8_manifests_20260712_113241/training_scenarios.tsv"
        ),
    )
    parser.add_argument("--bc-checkpoint", type=Path, default=Path("pretrained/end2race.pth"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/ppo/evidence/b4_substantive_negative")
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Use 1 to match the production evaluator's recurrent batch shape.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            # Explicit NA values keep heterogeneous diagnostic rows readable
            # and avoid ambiguous/trailing empty TSV fields.
            writer.writerow(
                {
                    field: "NA" if row.get(field, "") in ("", None) else row[field]
                    for field in fields
                }
            )


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def factorized_normal_log_prob(
    raw_action: np.ndarray, mean: np.ndarray, std: np.ndarray = STD
) -> np.ndarray:
    noise = np.asarray(raw_action, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return np.sum(
        -0.5 * (noise / std) ** 2 - np.log(std) - 0.5 * math.log(2.0 * math.pi),
        axis=-1,
    )


def load_actor(path: Path, device: torch.device) -> End2Race:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "actor_state_dict" in state:
        state = state["actor_state_dict"]
    actor = End2Race(mask_prob=0.0, hidden_scale=4)
    actor.load_state_dict(state, strict=True)
    actor.eval().to(device)
    return actor


def actor_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "actor_state_dict" in state:
        state = state["actor_state_dict"]
    return state


def variant_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "BC": args.bc_checkpoint,
        "seed1_iter10": args.evaluation_root / "models/seed1_iter10.pth",
        "seed1_iter20": args.evaluation_root / "models/seed1_iter20.pth",
        "seed1_iter30": args.evaluation_root / "models/seed1_iter30.pth",
    }


def build_metric_index(root: Path, variant: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    # Exclude preserved failed-attempt directories such as
    # ``shard0.failed_wrong_source_sha_*``.
    for path in sorted((root / variant).glob("shard[0-4]/metrics/*.json")):
        row = json.loads(path.read_text())
        row["local_metric_path"] = str(path)
        row["local_npz_path"] = str(path.parent.parent / "npz" / f"{row['case_id']}.npz")
        if row["case_id"] in index:
            raise AssertionError(f"duplicate metric row: {variant}/{row['case_id']}")
        index[row["case_id"]] = row
    if len(index) != 600:
        raise AssertionError(f"expected 600 {variant} metrics, found {len(index)}")
    return index


def outcome_flags(bc: str, candidate: str) -> dict[str, bool]:
    return {
        "fixed_collision": bc == "collision" and candidate != "collision",
        "new_collision": bc != "collision" and candidate == "collision",
        "persistent_collision": bc == "collision" and candidate == "collision",
        "gained_overtake": bc != "overtaking" and candidate == "overtaking",
        "lost_overtake": bc == "overtaking" and candidate != "overtaking",
    }


def summarize_product_rows(rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_variant: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_variant[row["variant"]][row["case_id"]] = row
    if set(by_variant) != set(VARIANTS):
        raise AssertionError(f"unexpected variants: {set(by_variant)}")
    cases = sorted(by_variant["BC"])
    if any(set(by_variant[v]) != set(cases) for v in VARIANTS):
        raise AssertionError("variant case sets differ")

    summary: dict[str, Any] = {}
    changed: list[dict[str, Any]] = []
    for variant in VARIANTS:
        counts = Counter(row["outcome"] for row in by_variant[variant].values())
        item: dict[str, Any] = {
            "collision": counts["collision"],
            "overtaking": counts["overtaking"],
            "following": counts["following"],
        }
        if variant != "BC":
            transitions = Counter()
            for case in cases:
                bc = by_variant["BC"][case]
                cand = by_variant[variant][case]
                flags = outcome_flags(bc["outcome"], cand["outcome"])
                transitions.update({key: int(value) for key, value in flags.items()})
                if any(flags[key] for key in ("fixed_collision", "new_collision", "gained_overtake", "lost_overtake")):
                    changed.append(
                        {
                            "variant": variant,
                            "case_id": case,
                            "startpoint_ordinal": int(bc["startpoint_ordinal"]),
                            "ego_idx": int(bc["ego_idx"]),
                            "opp_raceline": bc["opp_raceline"],
                            "opp_speedscale": float(bc["opp_speedscale"]),
                            "bc_outcome": bc["outcome"],
                            "candidate_outcome": cand["outcome"],
                            **{key: int(value) for key, value in flags.items()},
                        }
                    )
            item.update(transitions)
        summary[variant] = item

    expected = {
        "BC": (24, 342, 234),
        "seed1_iter10": (24, 332, 244),
        "seed1_iter20": (36, 294, 270),
        "seed1_iter30": (39, 296, 265),
    }
    for variant, values in expected.items():
        got = summary[variant]
        if (got["collision"], got["overtaking"], got["following"]) != values:
            raise AssertionError(f"product count mismatch for {variant}: {got}")
    return summary, changed


def terminal_geometry(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        ego = np.asarray(data["final_ego_pose"], dtype=np.float64)
        opp = np.asarray(data["final_opp_pose"], dtype=np.float64)
        rel_progress = float(data["final_opp_progress"] - data["final_ego_progress"])
        distance = float(np.linalg.norm(ego[:2] - opp[:2]))
        if abs(rel_progress) <= 0.6:
            phase = "alongside"
        elif rel_progress > 0.6:
            phase = "ego_behind"
        else:
            phase = "ego_ahead"
        # This is explicitly an inference, not the simulator's collision-object label.
        cause = "opponent_proximity_inferred" if distance <= 1.0 else "wall_inferred"
        return {
            "collision_terminal_distance_m": distance,
            "collision_relative_progress_m": rel_progress,
            "collision_phase_inferred": phase,
            "collision_cause_inferred": cause,
            "collision_final_time_s": float(data["final_time"]),
        }


def raceline_initial_speed(metric: dict[str, Any]) -> float:
    path = Path("f1tenth_racetracks") / metric["map_name"] / f"{metric['ego_raceline']}.csv"
    rows = np.loadtxt(path, delimiter=";", skiprows=1)
    return float(rows[int(metric["ego_idx"]) % len(rows), 5])


def speed_inputs(data: np.lib.npyio.NpzFile, metric: dict[str, Any]) -> np.ndarray:
    actual = np.asarray(data["ego_actual_speed"], dtype=np.float32)
    result = np.empty((len(actual), 1), dtype=np.float32)
    result[0, 0] = raceline_initial_speed(metric) * 0.9
    result[1:, 0] = actual[:-1]
    return result


def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
        f"{prefix}_max": float(np.max(values)),
    }


def process_histories(
    records: list[dict[str, Any]],
    backbone: End2Race,
    heads: dict[str, torch.nn.Module],
    device: torch.device,
    batch_size: int,
    validate_bc_actions: bool = False,
    retain_features_for: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return per-case frozen features and head actions for supplied histories."""
    retain_features_for = retain_features_for or set()
    result: dict[str, dict[str, Any]] = {}
    max_bc_steer_error = 0.0
    max_bc_speed_error = 0.0
    max_bc_steer_case = ""
    max_bc_speed_case = ""
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        loaded = []
        max_len = 0
        for record in batch:
            data = np.load(record["local_npz_path"], allow_pickle=False)
            lidar = np.asarray(data["ego_lidar"], dtype=np.float32)
            speed = speed_inputs(data, record)
            loaded.append((record, data, lidar, speed))
            max_len = max(max_len, len(lidar))

        lidar_batch = np.zeros((len(batch), max_len, 360), dtype=np.float32)
        speed_batch = np.zeros((len(batch), max_len, 1), dtype=np.float32)
        for index, (_, _, lidar, speed) in enumerate(loaded):
            lidar_batch[index, : len(lidar)] = lidar
            speed_batch[index, : len(speed)] = speed

        with torch.inference_mode():
            lidar_tensor = torch.from_numpy(lidar_batch).to(device)
            speed_tensor = torch.from_numpy(speed_batch).to(device)
            if batch_size == 1:
                # Match eval_multiagent.py exactly: batch one, one recurrent
                # step per call.  A fused full-sequence GRU can accumulate
                # materially different floating-point trajectories in this
                # recurrent network even though its equations are identical.
                hidden = None
                feature_steps = []
                for step in range(max_len):
                    feature, hidden = backbone.forward_features(
                        lidar_tensor[:, step : step + 1],
                        speed_tensor[:, step : step + 1],
                        hidden,
                    )
                    feature_steps.append(feature)
                features = torch.cat(feature_steps, dim=1)
            else:
                features, _ = backbone.forward_features(lidar_tensor, speed_tensor)
            actions = {name: head(features).cpu().numpy() for name, head in heads.items()}
            features = features.cpu().numpy()

        for index, (record, data, lidar, _) in enumerate(loaded):
            length = len(lidar)
            case = record["case_id"]
            case_actions = {name: value[index, :length] for name, value in actions.items()}
            result[case] = {
                "actions": case_actions,
                "length": length,
            }
            if case in retain_features_for:
                result[case]["feature"] = features[index, :length]
            if validate_bc_actions:
                stored_steer = np.asarray(data["ego_desired_steer"], dtype=np.float32)
                stored_speed = np.asarray(data["ego_desired_speed"], dtype=np.float32)
                predicted = case_actions["BC"]
                steer_error = float(
                    np.max(np.abs(np.clip(predicted[:, 0], -0.52, 0.52) - stored_steer))
                )
                speed_error = float(np.max(np.abs(predicted[:, 1] - stored_speed)))
                if steer_error > max_bc_steer_error:
                    max_bc_steer_error = steer_error
                    max_bc_steer_case = case
                if speed_error > max_bc_speed_error:
                    max_bc_speed_error = speed_error
                    max_bc_speed_case = case
            data.close()

    if validate_bc_actions:
        # The production evaluator advances a batch-one GRU one step at a time;
        # this analysis uses a fused full-sequence kernel.  Its accumulated
        # numerical discrepancy must remain small relative to the exploration
        # scales (0.03 rad / 0.20 m/s) and is reported rather than hidden.
        if max_bc_steer_error > 2e-3 or max_bc_speed_error > 1e-2:
            raise AssertionError(
                "BC teacher-forcing mismatch: "
                f"steer={max_bc_steer_error} ({max_bc_steer_case}), "
                f"speed={max_bc_speed_error} ({max_bc_speed_case})"
            )
        result["__validation__"] = {
            "max_abs_stored_bc_steer_error": max_bc_steer_error,
            "max_abs_stored_bc_steer_error_case": max_bc_steer_case,
            "max_abs_stored_bc_speed_error": max_bc_speed_error,
            "max_abs_stored_bc_speed_error_case": max_bc_speed_case,
        }
    return result


def drift_rows(
    histories: dict[str, dict[str, Any]],
    product_by_variant: dict[str, dict[str, dict[str, str]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_cases = sorted(case for case in histories if not case.startswith("__"))
    for case in all_cases:
        item = histories[case]
        bc_raw = item["actions"]["BC"].astype(np.float64)
        bc_exec = bc_raw.copy()
        bc_exec[:, 0] = np.clip(bc_exec[:, 0], -0.52, 0.52)
        bc_outcome = product_by_variant["BC"][case]["outcome"]
        for variant in CANDIDATES:
            raw = item["actions"][variant].astype(np.float64)
            executed = raw.copy()
            executed[:, 0] = np.clip(executed[:, 0], -0.52, 0.52)
            raw_delta = raw - bc_raw
            exec_delta = executed - bc_exec
            kl = 0.5 * np.sum((raw_delta / STD) ** 2, axis=1)
            groups = ["all", f"bc_{bc_outcome}"]
            flags = outcome_flags(
                bc_outcome, product_by_variant[variant][case]["outcome"]
            )
            groups.extend(key for key, value in flags.items() if value)
            for group in groups:
                bucket = buckets[(variant, group)]
                bucket["steer_abs"].append(np.abs(exec_delta[:, 0]))
                bucket["speed_abs"].append(np.abs(exec_delta[:, 1]))
                bucket["speed_signed"].append(exec_delta[:, 1])
                bucket["kl"].append(kl)
                bucket["episode_steer_abs"].append(np.asarray([np.mean(np.abs(exec_delta[:, 0]))]))
                bucket["episode_speed_abs"].append(np.asarray([np.mean(np.abs(exec_delta[:, 1]))]))
                bucket["episode_kl"].append(np.asarray([np.mean(kl)]))

    output = []
    for (variant, group), values in sorted(buckets.items()):
        row: dict[str, Any] = {"variant": variant, "group": group}
        for key in ("steer_abs", "speed_abs", "kl"):
            row.update(quantiles(np.concatenate(values[key]), f"transition_{key}"))
        signed = np.concatenate(values["speed_signed"])
        row["transition_speed_signed_mean"] = float(np.mean(signed))
        row["transition_count"] = int(len(signed))
        row["episode_count"] = len(values["episode_kl"])
        for key in ("episode_steer_abs", "episode_speed_abs", "episode_kl"):
            row.update(quantiles(np.concatenate(values[key]), key))
        output.append(row)

    validation = histories.get("__validation__", {})
    return output, validation


def parameter_drift_rows(paths: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    states = {name: actor_state(path) for name, path in paths.items()}
    output_keys = [key for key in states["BC"] if key.startswith("output_layer.")]
    frozen_keys = [key for key in states["BC"] if key not in output_keys]
    rows = []
    previous = "BC"
    frozen_exact = {}
    for variant in CANDIDATES:
        references = ("BC",) if previous == "BC" else ("BC", previous)
        for reference in references:
            delta_parts = []
            reference_parts = []
            max_abs = 0.0
            for key in output_keys:
                delta = (states[variant][key] - states[reference][key]).double().reshape(-1)
                delta_parts.append(delta)
                reference_parts.append(states[reference][key].double().reshape(-1))
                max_abs = max(max_abs, float(delta.abs().max()))
            delta = torch.cat(delta_parts)
            ref = torch.cat(reference_parts)
            rows.append(
                {
                    "variant": variant,
                    "reference": reference,
                    "output_parameter_count": int(delta.numel()),
                    "delta_l2": float(torch.linalg.vector_norm(delta)),
                    "reference_l2": float(torch.linalg.vector_norm(ref)),
                    "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ref)),
                    "delta_max_abs": max_abs,
                }
            )
        frozen_exact[variant] = all(torch.equal(states[variant][key], states["BC"][key]) for key in frozen_keys)
        previous = variant
    if not all(frozen_exact.values()):
        raise AssertionError(f"frozen actor drift detected: {frozen_exact}")
    return rows, {"frozen_key_count": len(frozen_keys), "frozen_exact_by_variant": frozen_exact}


def add_collision_geometry(
    changed: list[dict[str, Any]], metric_indices: dict[str, dict[str, dict[str, Any]]]
) -> None:
    for row in changed:
        collision_variant = None
        if row["fixed_collision"]:
            collision_variant = "BC"
        elif row["new_collision"] or row["persistent_collision"]:
            collision_variant = row["variant"]
        if collision_variant:
            metric = metric_indices[collision_variant][row["case_id"]]
            row["collision_source_variant"] = collision_variant
            row.update(terminal_geometry(Path(metric["local_npz_path"])))
        else:
            row.update(
                {
                    "collision_source_variant": "",
                    "collision_terminal_distance_m": "",
                    "collision_relative_progress_m": "",
                    "collision_phase_inferred": "",
                    "collision_cause_inferred": "",
                    "collision_final_time_s": "",
                }
            )


def precursor_and_feature_overlap(
    changed: list[dict[str, Any]],
    bc_histories: dict[str, dict[str, Any]],
    iter10_histories: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    iter10_changed = [row for row in changed if row["variant"] == "seed1_iter10"]
    precursor_rows = []
    vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in iter10_changed:
        if not (row["fixed_collision"] or row["new_collision"]):
            continue
        label = "fixed_collision" if row["fixed_collision"] else "new_collision"
        source = bc_histories if row["fixed_collision"] else iter10_histories
        item = source[row["case_id"]]
        bc = item["actions"]["BC"].astype(np.float64)
        candidate = item["actions"]["seed1_iter10"].astype(np.float64)
        delta = candidate - bc
        vector = np.mean(item["feature"][-50:], axis=0).astype(np.float64)
        vectors[label].append(vector)
        out: dict[str, Any] = {
            "case_id": row["case_id"],
            "label": label,
            "history_policy": "BC" if row["fixed_collision"] else "seed1_iter10",
            "history_steps": item["length"],
        }
        for window in (50, 100):
            window_delta = delta[-min(window, len(delta)) :]
            prefix = f"last_{window // 100 if window >= 100 else 0.5:g}s"
            out[f"{prefix}_steer_signed_mean"] = float(np.mean(window_delta[:, 0]))
            out[f"{prefix}_steer_abs_mean"] = float(np.mean(np.abs(window_delta[:, 0])))
            out[f"{prefix}_steer_abs_max"] = float(np.max(np.abs(window_delta[:, 0])))
            out[f"{prefix}_speed_signed_mean"] = float(np.mean(window_delta[:, 1]))
            out[f"{prefix}_speed_abs_mean"] = float(np.mean(np.abs(window_delta[:, 1])))
            out[f"{prefix}_speed_abs_max"] = float(np.max(np.abs(window_delta[:, 1])))
        precursor_rows.append(out)

    fixed = np.stack(vectors["fixed_collision"])
    new = np.stack(vectors["new_collision"])
    combined = np.concatenate([fixed, new], axis=0)
    labels = np.asarray([0] * len(fixed) + [1] * len(new))
    normalized = combined / np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12)
    distances = 1.0 - normalized @ normalized.T
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)

    def pair_values(left: np.ndarray, right: np.ndarray, same: bool) -> np.ndarray:
        values = []
        for i in left:
            for j in right:
                if not same or i < j:
                    values.append(distances[i, j])
        return np.asarray(values)

    fixed_idx = np.arange(len(fixed))
    new_idx = np.arange(len(fixed), len(combined))
    within_fixed = pair_values(fixed_idx, fixed_idx, True)
    within_new = pair_values(new_idx, new_idx, True)
    cross = pair_values(fixed_idx, new_idx, False)
    fixed_centroid = np.mean(normalized[: len(fixed)], axis=0)
    new_centroid = np.mean(normalized[len(fixed) :], axis=0)
    centroid_cosine = 1.0 - float(
        np.dot(fixed_centroid, new_centroid)
        / (np.linalg.norm(fixed_centroid) * np.linalg.norm(new_centroid))
    )
    overlap = {
        "method": "cosine distance of mean frozen-GRU feature over final 0.5 s of collision trajectory",
        "fixed_count": len(fixed),
        "new_count": len(new),
        "within_fixed_distance_median": float(np.median(within_fixed)),
        "within_new_distance_median": float(np.median(within_new)),
        "cross_distance_median": float(np.median(cross)),
        "centroid_cosine_distance": centroid_cosine,
        "nearest_neighbor_opposite_label_fraction": float(np.mean(labels[nearest] != labels)),
        "limitation": "descriptive n=11+11 comparison on policy-induced trajectories; not a representation sufficiency test",
    }
    return overlap, precursor_rows


def replay_noise_rows(training_root: Path, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    pooled: dict[str, list[np.ndarray]] = defaultdict(list)
    max_log_prob_error = 0.0
    for iteration in range(1, 31):
        replay_path = training_root / f"replay/iter_{iteration:04d}.npz"
        checkpoint_path = training_root / f"checkpoints/iter_{iteration - 1:04d}.pt"
        with np.load(replay_path, allow_pickle=False) as replay:
            feature = np.asarray(replay["feature"], dtype=np.float32)
            raw = np.asarray(replay["raw_action"], dtype=np.float64)
            old_log_prob = np.asarray(replay["old_log_prob"], dtype=np.float64).reshape(-1)
            episode_id = np.asarray(replay["episode_id"], dtype=np.int64)
            projection_delta = np.asarray(replay["projection_delta"], dtype=np.float64)
        # These are the experiment's own hash-bound full checkpoints.  They
        # include NumPy RNG state, which PyTorch's restricted weights-only
        # loader intentionally rejects.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        head = End2Race(mask_prob=0.0, hidden_scale=4).output_layer
        state = checkpoint["actor_state_dict"]
        head.load_state_dict(
            {key.removeprefix("output_layer."): value for key, value in state.items() if key.startswith("output_layer.")},
            strict=True,
        )
        head.eval().to(device)
        means = []
        with torch.inference_mode():
            for start in range(0, len(feature), 8192):
                means.append(head(torch.from_numpy(feature[start : start + 8192]).to(device)).cpu().numpy())
        mean = np.concatenate(means).astype(np.float64)
        noise = raw - mean
        reconstructed = factorized_normal_log_prob(raw, mean)
        log_prob_error = np.abs(reconstructed - old_log_prob)
        max_log_prob_error = max(max_log_prob_error, float(np.max(log_prob_error)))

        pair_left: list[np.ndarray] = []
        pair_right: list[np.ndarray] = []
        block_means: list[np.ndarray] = []
        zero_cross = np.zeros(2, dtype=np.int64)
        pair_count = np.zeros(2, dtype=np.int64)
        for episode in np.unique(episode_id):
            values = noise[episode_id == episode]
            if len(values) > 1:
                pair_left.append(values[:-1])
                pair_right.append(values[1:])
                zero_cross += np.sum(values[:-1] * values[1:] < 0, axis=0)
                pair_count += len(values) - 1
            for start in range(0, len(values) - 49, 50):
                block_means.append(np.mean(values[start : start + 50], axis=0))
        left = np.concatenate(pair_left)
        right = np.concatenate(pair_right)
        blocks = np.stack(block_means)
        lag1 = [float(np.corrcoef(left[:, d], right[:, d])[0, 1]) for d in range(2)]
        row = {
            "iteration": iteration,
            "transition_count": len(noise),
            "steer_noise_mean": float(np.mean(noise[:, 0])),
            "steer_noise_std": float(np.std(noise[:, 0])),
            "speed_noise_mean": float(np.mean(noise[:, 1])),
            "speed_noise_std": float(np.std(noise[:, 1])),
            "steer_lag1_autocorrelation": lag1[0],
            "speed_lag1_autocorrelation": lag1[1],
            "steer_zero_cross_fraction": float(zero_cross[0] / pair_count[0]),
            "speed_zero_cross_fraction": float(zero_cross[1] / pair_count[1]),
            "steer_50step_mean_std_ratio": float(np.std(blocks[:, 0]) / np.std(noise[:, 0])),
            "speed_50step_mean_std_ratio": float(np.std(blocks[:, 1]) / np.std(noise[:, 1])),
            "max_abs_reconstructed_log_prob_error": float(np.max(log_prob_error)),
            "steer_projection_count": int(np.sum(np.abs(projection_delta[:, 0]) > 0)),
            "speed_projection_count": int(np.sum(np.abs(projection_delta[:, 1]) > 0)),
        }
        rows.append(row)
        pooled["noise"].append(noise)
        pooled["left"].append(left)
        pooled["right"].append(right)
        pooled["blocks"].append(blocks)

    noise = np.concatenate(pooled["noise"])
    left = np.concatenate(pooled["left"])
    right = np.concatenate(pooled["right"])
    blocks = np.concatenate(pooled["blocks"])
    aggregate = {
        "transition_count": len(noise),
        "steer_noise_mean": float(np.mean(noise[:, 0])),
        "steer_noise_std": float(np.std(noise[:, 0])),
        "speed_noise_mean": float(np.mean(noise[:, 1])),
        "speed_noise_std": float(np.std(noise[:, 1])),
        "steer_lag1_autocorrelation": float(np.corrcoef(left[:, 0], right[:, 0])[0, 1]),
        "speed_lag1_autocorrelation": float(np.corrcoef(left[:, 1], right[:, 1])[0, 1]),
        "steer_50step_mean_std_ratio": float(np.std(blocks[:, 0]) / np.std(noise[:, 0])),
        "speed_50step_mean_std_ratio": float(np.std(blocks[:, 1]) / np.std(noise[:, 1])),
        "theoretical_iid_50step_ratio": 1.0 / math.sqrt(50.0),
        "max_abs_reconstructed_log_prob_error": max_log_prob_error,
        "steer_projection_count": sum(row["steer_projection_count"] for row in rows),
        "speed_projection_count": sum(row["speed_projection_count"] for row in rows),
    }
    if max_log_prob_error > 2e-3:
        raise AssertionError(f"replay log-prob reconstruction mismatch: {max_log_prob_error}")
    return rows, aggregate


def distribution_rows(
    training_root: Path,
    training_manifest: Path,
    product_by_variant: dict[str, dict[str, dict[str, str]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    curriculum = json.loads((training_root / "curriculum.json").read_text())["rows"]
    manifest = {row["l2_id"]: row for row in read_tsv(training_manifest)}
    training_counts = Counter((row["map_name"], row["archived_bc_outcome"]) for row in curriculum)
    unique_l2 = len({row["l2_id"] for row in curriculum})
    product_counts = Counter(row["outcome"] for row in product_by_variant["BC"].values())

    condition: dict[tuple[str, float], Counter[str]] = defaultdict(Counter)
    for row in curriculum:
        if row["map_name"] != "Austin":
            continue
        source = manifest[row["l2_id"]]
        key = (source["opponent_raceline"], round(float.fromhex(source["speedscale_hex"]), 1))
        condition[key][f"train_{row['archived_bc_outcome']}"] += 1
    for case, bc in product_by_variant["BC"].items():
        key = (bc["opp_raceline"], float(bc["opp_speedscale"]))
        condition[key][f"product_bc_{bc['outcome']}"] += 1
        for variant in CANDIDATES:
            flags = outcome_flags(bc["outcome"], product_by_variant[variant][case]["outcome"])
            for flag, value in flags.items():
                condition[key][f"{variant}_{flag}"] += int(value)

    fields = [
        "opp_raceline",
        "opp_speedscale",
        "train_collision",
        "train_overtake",
        "train_follow",
        "product_bc_collision",
        "product_bc_overtaking",
        "product_bc_following",
    ]
    for variant in CANDIDATES:
        fields.extend(
            f"{variant}_{flag}"
            for flag in ("fixed_collision", "new_collision", "gained_overtake", "lost_overtake")
        )
    rows = []
    for key in sorted(condition):
        counter = condition[key]
        row = {"opp_raceline": key[0], "opp_speedscale": key[1]}
        row.update({field: counter[field] for field in fields[2:]})
        rows.append(row)

    by_map = {
        map_name: {
            outcome: training_counts[(map_name, outcome)]
            for outcome in ("collision", "overtake", "follow")
        }
        for map_name in sorted({row["map_name"] for row in curriculum})
    }
    summary = {
        "training_episode_count": len(curriculum),
        "training_unique_l2_count": unique_l2,
        "training_outcomes": dict(Counter(row["archived_bc_outcome"] for row in curriculum)),
        "training_by_map": by_map,
        "product_bc_episode_count": 600,
        "product_bc_outcomes": dict(product_counts),
        "training_collision_fraction": Counter(row["archived_bc_outcome"] for row in curriculum)["collision"]
        / len(curriculum),
        "product_collision_fraction": product_counts["collision"] / 600.0,
        "collision_fraction_amplification": (
            Counter(row["archived_bc_outcome"] for row in curriculum)["collision"] / len(curriculum)
        )
        / (product_counts["collision"] / 600.0),
    }
    return rows, summary


def training_update_summary(training_root: Path) -> dict[str, Any]:
    iterations = [json.loads(line) for line in (training_root / "iterations.jsonl").read_text().splitlines()]
    final_kl = [float(row["update"]["weighted_kl"]) for row in iterations]
    all_epoch_kl = [
        float(epoch["weighted_kl"])
        for row in iterations
        for epoch in row["update"]["actor_epoch_metrics"]
    ]
    epochs = Counter(int(row["update"]["actor_epochs_completed"]) for row in iterations)
    return {
        "iterations": len(iterations),
        "actor_epochs_completed_histogram": {str(key): value for key, value in sorted(epochs.items())},
        "final_epoch_weighted_kl_mean": float(np.mean(final_kl)),
        "final_epoch_weighted_kl_max": float(np.max(final_kl)),
        "all_actor_epoch_weighted_kl_max": float(np.max(all_epoch_kl)),
        "target_weighted_kl": 0.015,
        "max_target_multiple": float(np.max(all_epoch_kl) / 0.015),
        "iterations_final_weighted_kl_above_target": int(np.sum(np.asarray(final_kl) > 0.015)),
        "critic_epoch_failures": sum(row["update"]["critic_epochs_completed"] != 3 for row in iterations),
        "terminal_label_count": sum(int(row["episode_count"]) for row in iterations),
        "transition_count": sum(int(row["transition_count"]) for row in iterations),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.batch_size != 1:
        raise ValueError(
            "exact product-history replay requires --batch-size 1; batched/fused GRU "
            "execution is not numerically interchangeable with eval_multiagent.py"
        )

    paired_rows = read_tsv(args.evaluation_root / "final/paired_rows.tsv")
    product_summary, changed = summarize_product_rows(paired_rows)
    product_by_variant: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in paired_rows:
        product_by_variant[row["variant"]][row["case_id"]] = row
    metric_indices = {variant: build_metric_index(args.evaluation_root, variant) for variant in VARIANTS}
    add_collision_geometry(changed, metric_indices)

    paths = variant_paths(args)
    actors = {name: load_actor(path, device) for name, path in paths.items()}
    backbone = actors["BC"]
    heads = {name: actor.output_layer for name, actor in actors.items()}
    bc_records = [metric_indices["BC"][case] for case in sorted(metric_indices["BC"])]
    iter10_fixed_cases = {
        row["case_id"]
        for row in changed
        if row["variant"] == "seed1_iter10" and row["fixed_collision"]
    }
    bc_histories = process_histories(
        bc_records,
        backbone,
        heads,
        device,
        args.batch_size,
        validate_bc_actions=True,
        retain_features_for=iter10_fixed_cases,
    )
    drift, teacher_forcing_validation = drift_rows(bc_histories, product_by_variant)

    iter10_collision_cases = sorted(
        case
        for case, row in product_by_variant["seed1_iter10"].items()
        if row["outcome"] == "collision"
    )
    iter10_histories = process_histories(
        [metric_indices["seed1_iter10"][case] for case in iter10_collision_cases],
        backbone,
        heads,
        device,
        args.batch_size,
        retain_features_for=set(iter10_collision_cases),
    )
    feature_overlap, precursor = precursor_and_feature_overlap(changed, bc_histories, iter10_histories)
    precursor_index = {row["case_id"]: row for row in precursor}
    for row in changed:
        if row["variant"] == "seed1_iter10" and row["case_id"] in precursor_index:
            row.update(precursor_index[row["case_id"]])

    parameter_drift, frozen_validation = parameter_drift_rows(paths)
    noise_rows, noise_summary = replay_noise_rows(args.training_root, device)
    condition_rows, distribution = distribution_rows(
        args.training_root, args.training_manifest, product_by_variant
    )
    update_summary = training_update_summary(args.training_root)

    write_tsv(args.output_dir / "changed_cases.tsv", changed)
    write_tsv(args.output_dir / "action_drift.tsv", drift)
    write_tsv(args.output_dir / "parameter_drift.tsv", parameter_drift)
    write_tsv(args.output_dir / "exploration_noise.tsv", noise_rows)
    write_tsv(args.output_dir / "condition_coverage.tsv", condition_rows)
    write_tsv(args.output_dir / "iter10_collision_precursors.tsv", precursor)
    summary = {
        "schema": "end2race-b4-substantive-negative-analysis-1",
        "closed_experiment": "B4",
        "device": str(device),
        "input_provenance": {
            "run_id": "b4_seed1_20260714_003027",
            "training_source_commit": json.loads((args.training_root / "config.json").read_text())[
                "source_commit"
            ],
            "run_plan_sha256": json.loads((args.training_root / "config.json").read_text())[
                "run_plan_sha256"
            ],
            "final_paired_rows_sha256": sha256_file(args.evaluation_root / "final/paired_rows.tsv"),
            "final_product_summary_sha256": sha256_file(args.evaluation_root / "final/summary.json"),
            "training_summary_sha256": sha256_file(args.training_root / "summary.json"),
            "training_iterations_sha256": sha256_file(args.training_root / "iterations.jsonl"),
            "training_curriculum_sha256": sha256_file(args.training_root / "curriculum.json"),
        },
        "product": product_summary,
        "teacher_forcing_validation": teacher_forcing_validation,
        "frozen_actor_validation": frozen_validation,
        "feature_overlap": feature_overlap,
        "exploration_noise": noise_summary,
        "distribution": distribution,
        "training_update": update_summary,
        "claim_boundary": {
            "established": [
                "B4 is an integrity-valid substantive negative on the frozen 600-case product grid",
                "the output head changed behavior materially at iter10 and later snapshots",
                "all frozen actor tensors remained exact",
            ],
            "supported_not_proven": [
                "cumulative BC-relative drift and absent BC-preserving constraints contributed to case swapping",
                "100 Hz iid exploration and collision-heavy curriculum increased optimization variance/nonselectivity",
            ],
            "unresolved": [
                "frozen GRU representation sufficiency",
                "causal benefit of a BC anchor, temporally coherent exploration, or sampler change",
                "seed variance because only seed1 was authorized",
            ],
        },
    }
    json_dump(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Layerwise Float64/FP32 semantic and TF32 causal oracle for A/B/C."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for path in (PROJECT_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_common import assert_locked_sources, backend_flags, diff_metrics, provenance, write_json
from model import End2Race


STEER_BOUND = 0.52
LATENT_STD = 0.03
SPEED_STD = 0.15
STAGES = (
    "raw_lidar",
    "processed_lidar",
    "previous_speed",
    "speed_embedding",
    "gru_feature",
    "gru_output",
    "next_hidden",
    "output_linear1",
    "output_relu",
    "raw_physical_mean",
    "clipped_physical_steering_mean",
    "transformed_latent_steering_mean",
    "sampled_latent_action",
    "physical_sampled_action",
    "log_probability",
)


def postprocess(model: End2Race, gru_output: torch.Tensor, epsilon: torch.Tensor) -> dict[str, torch.Tensor]:
    linear1 = model.output_layer[0](gru_output)
    relu = model.output_layer[1](linear1)
    raw_mean = model.output_layer[2](relu)
    normalized = (raw_mean[..., 0] / STEER_BOUND).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    latent_mean = 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))
    latent_steer = latent_mean + LATENT_STD * epsilon[..., 0]
    sampled_speed = raw_mean[..., 1] + SPEED_STD * epsilon[..., 1]
    sampled_latent = torch.stack((latent_steer, sampled_speed), dim=-1)
    physical = torch.stack((STEER_BOUND * torch.tanh(latent_steer), sampled_speed), dim=-1)
    action_epsilon = torch.finfo(physical.dtype).eps
    action_normalized = (physical[..., 0] / STEER_BOUND).clamp(-1.0 + action_epsilon, 1.0 - action_epsilon)
    replay_latent = 0.5 * (torch.log1p(action_normalized) - torch.log1p(-action_normalized))
    steer_distribution = torch.distributions.Normal(latent_mean, torch.as_tensor(LATENT_STD, device=raw_mean.device, dtype=raw_mean.dtype))
    speed_distribution = torch.distributions.Normal(raw_mean[..., 1], torch.as_tensor(SPEED_STD, device=raw_mean.device, dtype=raw_mean.dtype))
    jacobian = torch.log(torch.as_tensor(STEER_BOUND, device=raw_mean.device, dtype=raw_mean.dtype)) + torch.log1p(-action_normalized.square())
    logp = steer_distribution.log_prob(replay_latent) - jacobian + speed_distribution.log_prob(physical[..., 1])
    return {
        "output_linear1": linear1,
        "output_relu": relu,
        "raw_physical_mean": raw_mean,
        "clipped_physical_steering_mean": raw_mean[..., 0].clamp(-STEER_BOUND, STEER_BOUND).unsqueeze(-1),
        "transformed_latent_steering_mean": latent_mean.unsqueeze(-1),
        "sampled_latent_action": sampled_latent,
        "physical_sampled_action": physical,
        "log_probability": logp.unsqueeze(-1),
    }


def preprocess(model: End2Race, lidar: torch.Tensor, speed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    processed = (-1.0 / (1.0 + torch.exp(-model.k * lidar)) + 1.0) * 2.0
    embedding = model.speed_mlp(speed)
    return processed, embedding, torch.cat((processed, embedding), dim=-1)


def empty_outputs(shape: tuple[int, int], model: End2Race, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    batch, steps = shape
    widths = {
        "raw_lidar": 360,
        "processed_lidar": 360,
        "previous_speed": 1,
        "speed_embedding": 60,
        "gru_feature": 420,
        "gru_output": 1680,
        "next_hidden": 1680,
        "output_linear1": 420,
        "output_relu": 420,
        "raw_physical_mean": 2,
        "clipped_physical_steering_mean": 1,
        "transformed_latent_steering_mean": 1,
        "sampled_latent_action": 2,
        "physical_sampled_action": 2,
        "log_probability": 1,
    }
    return {name: torch.zeros((batch, steps, width), device=device, dtype=dtype) for name, width in widths.items()}


def scalar_reference(
    model: End2Race,
    lidar: torch.Tensor,
    speed: torch.Tensor,
    initial_hidden: torch.Tensor,
    episode_starts: torch.Tensor,
    lengths: list[int],
    epsilon: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch, steps, _ = lidar.shape
    output = empty_outputs((batch, steps), model, lidar.device, lidar.dtype)
    final_hidden = torch.zeros_like(initial_hidden)
    for sequence in range(batch):
        hidden = initial_hidden[:, sequence : sequence + 1]
        for step in range(lengths[sequence]):
            hidden = hidden * (1.0 - episode_starts[sequence, step]).reshape(1, 1, 1)
            step_lidar = lidar[sequence : sequence + 1, step : step + 1]
            step_speed = speed[sequence : sequence + 1, step : step + 1]
            processed, embedding, feature = preprocess(model, step_lidar, step_speed)
            gru_output, hidden = model.gru(feature, hidden)
            values = {
                "raw_lidar": step_lidar,
                "processed_lidar": processed,
                "previous_speed": step_speed,
                "speed_embedding": embedding,
                "gru_feature": feature,
                "gru_output": gru_output,
                "next_hidden": hidden.transpose(0, 1),
                **postprocess(model, gru_output, epsilon[sequence : sequence + 1, step : step + 1]),
            }
            for name in STAGES:
                output[name][sequence, step] = values[name][0, 0]
        final_hidden[:, sequence : sequence + 1] = hidden
    return output, final_hidden


def timestep_batched(
    model: End2Race,
    lidar: torch.Tensor,
    speed: torch.Tensor,
    initial_hidden: torch.Tensor,
    episode_starts: torch.Tensor,
    lengths: list[int],
    epsilon: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch, steps, _ = lidar.shape
    output = empty_outputs((batch, steps), model, lidar.device, lidar.dtype)
    hidden = initial_hidden
    for step in range(steps):
        active = [index for index, length in enumerate(lengths) if step < length]
        if not active:
            break
        indices = torch.as_tensor(active, device=lidar.device)
        reset = (1.0 - episode_starts[indices, step]).reshape(1, -1, 1)
        active_hidden = hidden[:, indices] * reset
        step_lidar = lidar[indices, step : step + 1]
        step_speed = speed[indices, step : step + 1]
        processed, embedding, feature = preprocess(model, step_lidar, step_speed)
        gru_output, next_hidden = model.gru(feature, active_hidden)
        by_sequence = []
        position = {sequence: offset for offset, sequence in enumerate(active)}
        for sequence in range(batch):
            offset = position.get(sequence)
            by_sequence.append(hidden[:, sequence : sequence + 1] if offset is None else next_hidden[:, offset : offset + 1])
        hidden = torch.cat(by_sequence, dim=1)
        values = {
            "raw_lidar": step_lidar,
            "processed_lidar": processed,
            "previous_speed": step_speed,
            "speed_embedding": embedding,
            "gru_feature": feature,
            "gru_output": gru_output,
            "next_hidden": next_hidden.transpose(0, 1),
            **postprocess(model, gru_output, epsilon[indices, step : step + 1]),
        }
        for name in STAGES:
            output[name][indices, step] = values[name][:, 0]
    return output, hidden


def packed_candidate(
    model: End2Race,
    lidar: torch.Tensor,
    speed: torch.Tensor,
    initial_hidden: torch.Tensor,
    episode_starts: torch.Tensor,
    lengths: list[int],
    epsilon: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if torch.count_nonzero(episode_starts[:, 1:]).item() != 0:
        raise RuntimeError("packed oracle sequences must already be split at episode starts")
    processed, embedding, feature = preprocess(model, lidar, speed)
    hidden = initial_hidden * (1.0 - episode_starts[:, 0]).reshape(1, -1, 1)
    packed = pack_padded_sequence(feature, torch.as_tensor(lengths).cpu(), batch_first=True, enforce_sorted=False)
    packed_output, final_hidden = model.gru(packed, hidden)
    gru_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=lidar.shape[1])
    output = empty_outputs(lidar.shape[:2], model, lidar.device, lidar.dtype)
    valid = torch.arange(lidar.shape[1], device=lidar.device).unsqueeze(0) < torch.as_tensor(lengths, device=lidar.device).unsqueeze(1)
    values = {
        "raw_lidar": lidar,
        "processed_lidar": processed,
        "previous_speed": speed,
        "speed_embedding": embedding,
        "gru_feature": feature,
        "gru_output": gru_output,
        "next_hidden": gru_output,
        **postprocess(model, gru_output, epsilon),
    }
    for name in STAGES:
        output[name] = torch.where(valid.unsqueeze(-1), values[name], output[name])
    return output, final_hidden


def compare(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], lengths: list[int]) -> dict[str, Any]:
    valid = torch.arange(next(iter(reference.values())).shape[1]).unsqueeze(0) < torch.as_tensor(lengths).unsqueeze(1)
    return {name: diff_metrics(reference[name].cpu()[valid], candidate[name].cpu()[valid]) for name in STAGES}


def run_case(device_name: str, dtype: torch.dtype, tf32_off: bool, seed: int = 7319) -> dict[str, Any]:
    device = torch.device(device_name)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch, steps = 16, 12
    lidar = (torch.rand((batch, steps, 360), generator=generator, dtype=torch.float64) * 30.0).to(device=device, dtype=dtype)
    speed = (torch.rand((batch, steps, 1), generator=generator, dtype=torch.float64) * 8.0).to(device=device, dtype=dtype)
    initial_hidden = (torch.randn((1, batch, 1680), generator=generator, dtype=torch.float64) * 0.1).to(device=device, dtype=dtype)
    epsilon = torch.randn((batch, steps, 2), generator=generator, dtype=torch.float64).to(device=device, dtype=dtype)
    lengths = [12, 11, 10, 9, 8, 7, 6, 5, 12, 10, 8, 6, 11, 9, 7, 5]
    starts_b = torch.zeros((batch, steps), device=device, dtype=dtype)
    starts_b[[0, 3, 9], 0] = 1.0
    starts_b[[2, 8], 4] = 1.0
    starts_c = torch.zeros_like(starts_b)
    starts_c[[0, 3, 9], 0] = 1.0
    model = End2Race(mask_prob=0.0, hidden_scale=4)
    model.load_state_dict(torch.load(PROJECT_ROOT / "pretrained" / "end2race.pth", map_location="cpu", weights_only=True), strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    with backend_flags(tf32_off) as flags, torch.no_grad():
        reference_a, hidden_ref_a = scalar_reference(model, lidar[:, :1], speed[:, :1], initial_hidden, starts_b[:, :1], [1] * batch, epsilon[:, :1])
        candidate_a, hidden_a = timestep_batched(model, lidar[:, :1], speed[:, :1], initial_hidden, starts_b[:, :1], [1] * batch, epsilon[:, :1])
        reference_b, hidden_ref_b = scalar_reference(model, lidar, speed, initial_hidden, starts_b, [steps] * batch, epsilon)
        candidate_b, hidden_b = timestep_batched(model, lidar, speed, initial_hidden, starts_b, [steps] * batch, epsilon)
        reference_c, hidden_ref_c = scalar_reference(model, lidar, speed, initial_hidden, starts_c, lengths, epsilon)
        candidate_c, hidden_c = packed_candidate(model, lidar, speed, initial_hidden, starts_c, lengths, epsilon)
        result = {
            "flags": flags,
            "device": str(device),
            "dtype": str(dtype),
            "A": {"layers": compare(reference_a, candidate_a, [1] * batch), "final_hidden": diff_metrics(hidden_ref_a, hidden_a)},
            "B": {"layers": compare(reference_b, candidate_b, [steps] * batch), "final_hidden": diff_metrics(hidden_ref_b, hidden_b)},
            "C": {"layers": compare(reference_c, candidate_c, lengths), "final_hidden": diff_metrics(hidden_ref_c, hidden_c)},
            "sequence_metadata": {"batch": batch, "steps": steps, "packed_lengths": lengths, "B_internal_resets": [[2, 4], [8, 4]]},
        }
    return result


def main() -> None:
    assert_locked_sources()
    cases = {
        "cpu_float64": run_case("cpu", torch.float64, True),
        "gpu_float64": run_case("cuda", torch.float64, True),
        "gpu_float32_tf32_on": run_case("cuda", torch.float32, False),
        "gpu_float32_tf32_off": run_case("cuda", torch.float32, True),
    }
    cpu_max = max(
        metric["max_abs"]
        for candidate in ("A", "B", "C")
        for metric in cases["cpu_float64"][candidate]["layers"].values()
    )
    cpu_max = max(cpu_max, *(cases["cpu_float64"][candidate]["final_hidden"]["max_abs"] for candidate in ("A", "B", "C")))
    causal = {}
    causal_stages = ("gru_output", "next_hidden", "raw_physical_mean", "transformed_latent_steering_mean", "log_probability")
    for candidate in ("A", "B", "C"):
        causal[candidate] = {}
        for stage in causal_stages:
            on = cases["gpu_float32_tf32_on"][candidate]["layers"][stage]["max_abs"]
            off = cases["gpu_float32_tf32_off"][candidate]["layers"][stage]["max_abs"]
            causal[candidate][stage] = {"tf32_on_max_abs": on, "tf32_off_max_abs": off, "shrink_factor": float("inf") if off == 0.0 and on > 0.0 else (1.0 if off == 0.0 else on / off)}
    core_tf32_gate = all(
        causal[candidate][stage]["shrink_factor"] >= 100.0
        for candidate in ("A", "B", "C")
        for stage in ("gru_output", "next_hidden", "raw_physical_mean", "transformed_latent_steering_mean")
    )
    logp_absolute = {
        candidate: cases["gpu_float32_tf32_off"][candidate]["layers"]["log_probability"]
        for candidate in ("A", "B", "C")
    }
    logp_gate = all(record["max_abs"] <= 1e-5 for record in logp_absolute.values())
    tf32_gate = core_tf32_gate and logp_gate
    semantic_pass = cpu_max <= 1e-9
    bundle = json.loads((HERE / "REFERENCE_BUNDLE.json").read_text())
    result = {
        "schema_version": 2,
        **provenance("Stage 0 A/B/C layerwise oracle", {"A": 16, "B": "all active", "C": "packed"}, cases["gpu_float32_tf32_off"]["flags"]),
        "model_initial_hash": bundle["model_initial_hash"],
        "optimizer_initial_hash": bundle["optimizer_initial_hash"],
        "rng_initial_hash": {**bundle["initial_rng_hashes"], "oracle_input_generator_seed": 7319},
        "rollout_hash": bundle["rollout_hash"],
        "cases": cases,
        "cpu_float64_max_abs_all_candidates_and_layers": cpu_max,
        "semantic_gate_threshold": 1e-9,
        "semantic_gate_pass": semantic_pass,
        "tf32_causal_metrics": causal,
        "tf32_core_shrink_gate_threshold": 100.0,
        "tf32_core_causal_gate_pass": core_tf32_gate,
        "logp_absolute_error_gate": {
            "threshold_max_abs": 1e-5,
            "passed": logp_gate,
            **logp_absolute,
        },
        "tf32_causal_gate_pass": tf32_gate,
        "numerical_metrics": {"cpu_float64_max_abs": cpu_max, "tf32_causal": causal},
        "timing_metrics": {},
        "checkpoint_hash": None,
        "verdict": (
            "TF32_DOMINANT_FOR_CORE_FORWARD_NUMERICS / LOGP_ABSOLUTE_ERROR_PASS"
            if semantic_pass and tf32_gate
            else "STOP_PHASE5_SEMANTIC_BUG" if not semantic_pass else "TF32_CORE_OR_LOGP_GATE_FAIL"
        ),
    }
    write_json(HERE / "SEMANTIC_ORACLE.json", result)
    write_json(
        HERE / "TF32_CAUSAL_AUDIT.json",
        {
            **provenance("TF32 causal audit", {"A": 16, "B": "all active", "C": "packed"}, cases["gpu_float32_tf32_off"]["flags"]),
            "model_initial_hash": bundle["model_initial_hash"],
            "optimizer_initial_hash": bundle["optimizer_initial_hash"],
            "rng_initial_hash": {**bundle["initial_rng_hashes"], "oracle_input_generator_seed": 7319},
            "rollout_hash": bundle["rollout_hash"],
            "numerical_metrics": causal,
            "timing_metrics": {},
            "checkpoint_hash": None,
            "tf32_dominant_cause_confirmed": tf32_gate,
            "tf32_core_causal_gate_pass": core_tf32_gate,
            "logp_absolute_error_gate": {"threshold_max_abs": 1e-5, "passed": logp_gate, **logp_absolute},
            "owner_override": "Ratio-based logp shrink gate withdrawn near FP32 floor; absolute max <= 1e-5 is authoritative.",
            "verdict": (
                "TF32_DOMINANT_FOR_CORE_FORWARD_NUMERICS / LOGP_ABSOLUTE_ERROR_PASS"
                if tf32_gate
                else "TF32_CORE_OR_LOGP_GATE_FAIL"
            ),
        },
    )
    assert_locked_sources()
    print(json.dumps({"semantic_pass": semantic_pass, "cpu_max": cpu_max, "tf32_causal_pass": tf32_gate}, sort_keys=True))
    if not semantic_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

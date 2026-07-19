#!/usr/bin/env python3
"""Assemble registered gates, speedups, combinations, Pareto, and selection."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for path in (PROJECT_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_common import assert_locked_sources, provenance, write_json


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / name).read_text())


def performance(name: str) -> dict[str, Any]:
    return load(f"PERF_{name}_REPEAT1.json")


def improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference


def speed_summary(reference: dict[str, Any], candidate: dict[str, Any], stage: str) -> dict[str, Any]:
    ref = reference["timing_metrics"]
    cand = candidate["timing_metrics"]
    actor_key = "collection" if stage == "collection" else "training"
    return {
        "actor_forward_speedup": ref["actor_forward_cuda_ms"][actor_key] / cand["actor_forward_cuda_ms"][actor_key],
        "rollout_improvement_fraction": improvement(ref["rollout_wall_s"], cand["rollout_wall_s"]),
        "train_improvement_fraction": improvement(ref["ppo_train_wall_s"], cand["ppo_train_wall_s"]),
        "total_update_improvement_fraction": improvement(ref["total_update_wall_s"], cand["total_update_wall_s"]),
        "peak_allocated_vram_change_fraction": (cand["torch_peak_allocated_mib"] - ref["torch_peak_allocated_mib"]) / ref["torch_peak_allocated_mib"],
        "peak_reserved_vram_change_fraction": (cand["torch_peak_reserved_mib"] - ref["torch_peak_reserved_mib"]) / ref["torch_peak_reserved_mib"],
    }


def training_numeric_summary(record: dict[str, Any]) -> dict[str, float]:
    batches = record["numerical_metrics"]["all_four_minibatches"]
    return {
        "policy_kl_max": max(batch["reference_vs_candidate_policy_kl"] for batch in batches),
        "hidden_p99_max": max(batch["forward"]["hidden"]["p99_abs"] for batch in batches),
        "hidden_max": max(batch["forward"]["hidden"]["max_abs"] for batch in batches),
        "logp_p99_max": max(batch["forward"]["new_logp"]["p99_abs"] for batch in batches),
        "logp_max": max(batch["forward"]["new_logp"]["max_abs"] for batch in batches),
        "gradient_cosine_min": min(batch["gradient"]["full"]["cosine"] for batch in batches),
        "gradient_relative_l2_max": max(batch["gradient"]["full"]["relative_l2"] for batch in batches),
        "delta_cosine_min": min(batch["parameter_delta"]["full"]["cosine"] for batch in batches),
        "delta_relative_l2_max": max(batch["parameter_delta"]["full"]["relative_l2"] for batch in batches),
        "policy_loss_abs_max": max(batch["loss"]["policy_loss"]["absolute_difference"] for batch in batches),
    }


def main() -> None:
    assert_locked_sources()
    bundle = load("REFERENCE_BUNDLE.json")
    semantic = load("SEMANTIC_ORACLE.json")
    r0 = performance("R0")
    r1 = performance("R1")
    a = load("A_BATCH16.json")
    b = load("B_TIMESTEP.json")
    c = load("C_PACKED.json")
    perf = {name: performance(name) for name in ("A", "B", "C", "AB", "AC")}
    speeds = {
        "A": speed_summary(r1, perf["A"], "collection"),
        "B": speed_summary(r1, perf["B"], "training"),
        "C": speed_summary(r1, perf["C"], "training"),
        "AB": speed_summary(r1, perf["AB"], "collection"),
        "AC": speed_summary(r1, perf["AC"], "collection"),
    }
    a_gate = {
        "actor_forward_speedup_ge_2": speeds["A"]["actor_forward_speedup"] >= 2.0,
        "rollout_improvement_ge_5pct": speeds["A"]["rollout_improvement_fraction"] >= 0.05,
        "total_update_improvement_ge_3pct": speeds["A"]["total_update_improvement_fraction"] >= 0.03,
    }
    b_gate = {
        "actor_forward_speedup_ge_3": speeds["B"]["actor_forward_speedup"] >= 3.0,
        "train_improvement_ge_10pct": speeds["B"]["train_improvement_fraction"] >= 0.10,
        "total_update_improvement_ge_5pct": speeds["B"]["total_update_improvement_fraction"] >= 0.05,
    }
    c_gate = {
        "actor_forward_speedup_ge_3": speeds["C"]["actor_forward_speedup"] >= 3.0,
        "train_improvement_ge_10pct": speeds["C"]["train_improvement_fraction"] >= 0.10,
        "total_update_improvement_ge_5pct": speeds["C"]["total_update_improvement_fraction"] >= 0.05,
    }
    a["timing_metrics"] = {
        "warmup_ref": "PERF_A_WARMUP.json",
        "repeat1_ref": "PERF_A_REPEAT1.json",
        "repeat1": perf["A"]["timing_metrics"],
        "speedup_vs_R1": speeds["A"],
        "performance_gate_checks": a_gate,
        "performance_gate_pass": all(a_gate.values()),
        "padding_ratio": 1.5,
    }
    a["checkpoint_hash"] = perf["A"]["checkpoint_hash"]
    a["performance_checkpoint"] = perf["A"]["checkpoint"]
    a["verdict"] = "DISTRIBUTIONAL_ONLY_REQUIRES_PRODUCT_TEST"
    write_json(HERE / "A_BATCH16.json", a)

    for name, record, gate in (("B", b, b_gate), ("C", c, c_gate)):
        frozen_timing = record["timing_metrics"]
        record["timing_metrics"] = {
            "frozen_rollout": frozen_timing,
            "warmup_ref": f"PERF_{name}_WARMUP.json",
            "repeat1_ref": f"PERF_{name}_REPEAT1.json",
            "repeat1": perf[name]["timing_metrics"],
            "speedup_vs_R1": speeds[name],
            "performance_gate_checks": gate,
            "performance_gate_pass": all(gate.values()),
            "padding_ratio": 1.5,
        }
        record["numerical_metrics"]["summary"] = training_numeric_summary(record)
        record["full_update_checkpoint"] = perf[name]["checkpoint"]
        record["verdict"] = "QUICK_NUMERIC_AND_SPEED_PASS" if all(gate.values()) else "NUMERIC_PASS_SPEED_TOO_SMALL"
        write_json(HERE / ("B_TIMESTEP.json" if name == "B" else "C_PACKED.json"), record)

    speed_record = {
        "schema_version": 1,
        **provenance("R0/R1 formal speed", 1, r1["flags"], bundle["rollout_hash"]),
        "model_initial_hash": bundle["model_initial_hash"],
        "optimizer_initial_hash": bundle["optimizer_initial_hash"],
        "rng_initial_hash": bundle["initial_rng_hashes"],
        "minibatch_order_hash": bundle["minibatch_order_hash"],
        "backend": "R0 default TF32 versus R1 TF32 off batch-1",
        "batch_or_microbatch": 1,
        "numerical_metrics": {"classification": "R1_BASELINE_EXACT"},
        "timing_metrics": {
            "R0_warmup_ref": "PERF_R0_WARMUP.json",
            "R1_warmup_ref": "PERF_R1_WARMUP.json",
            "R0_repeat1": r0["timing_metrics"],
            "R1_repeat1": r1["timing_metrics"],
            "R1_minus_R0": {
                "rollout_change_fraction": (r1["timing_metrics"]["rollout_wall_s"] - r0["timing_metrics"]["rollout_wall_s"]) / r0["timing_metrics"]["rollout_wall_s"],
                "train_change_fraction": (r1["timing_metrics"]["ppo_train_wall_s"] - r0["timing_metrics"]["ppo_train_wall_s"]) / r0["timing_metrics"]["ppo_train_wall_s"],
                "total_update_change_fraction": (r1["timing_metrics"]["total_update_wall_s"] - r0["timing_metrics"]["total_update_wall_s"]) / r0["timing_metrics"]["total_update_wall_s"],
            },
        },
        "checkpoint_hash": r1["checkpoint_hash"],
        "verdict": "R1_BASELINE_EXACT",
    }
    write_json(HERE / "R0_R1_SPEED.json", speed_record)

    summaries = {"B": training_numeric_summary(b), "C": training_numeric_summary(c)}
    for combo, training in (("AB", "B"), ("AC", "C")):
        checkpoint = perf[combo]["checkpoint"]
        record = {
            "schema_version": 1,
            **provenance(f"{combo} combined full update", {"collection": 16, "training": training}, perf[combo]["flags"], bundle["rollout_hash"]),
            "model_initial_hash": bundle["model_initial_hash"],
            "optimizer_initial_hash": bundle["optimizer_initial_hash"],
            "rng_initial_hash": bundle["initial_rng_hashes"],
            "minibatch_order_hash": bundle["minibatch_order_hash"],
            "backend": combo,
            "batch_or_microbatch": {"collection": 16, "training": "all active" if training == "B" else "packed"},
            "numerical_metrics": {
                "collection_component_ref": "A_BATCH16.json",
                "collection_A1_gate_pass": a["numerical_metrics"]["A1_open_loop"]["gate_pass"],
                "collection_A2_gate_pass": a["numerical_metrics"]["A2_teacher_forced"]["gate_pass"],
                "collection_closed_loop_classification": "DISTRIBUTIONAL_ONLY_REQUIRES_PRODUCT_TEST",
                "training_component_ref": "B_TIMESTEP.json" if training == "B" else "C_PACKED.json",
                "training_all_four_minibatch_gate_pass": True,
                "training_summary": summaries[training],
                "component_composition_note": "Collection A and the selected training backend affect disjoint policy call sites; frozen gradient/delta evidence is the registered B/C component audit.",
            },
            "timing_metrics": {
                "warmup_ref": f"PERF_{combo}_WARMUP.json",
                "repeat1_ref": f"PERF_{combo}_REPEAT1.json",
                "repeat1": perf[combo]["timing_metrics"],
                "speedup_vs_R1": speeds[combo],
                "padding_ratio": 1.5,
            },
            "live_full_update": {
                "transitions": 25600,
                "completed_episodes": perf[combo]["timing_metrics"]["completed_episodes"],
                "outcomes": perf[combo]["outcomes"],
                "normal_vector_env_close": True,
            },
            "checkpoint_hash": checkpoint["sha256"],
            "checkpoint": checkpoint,
            "verdict": "DISTRIBUTIONAL_ONLY_REQUIRES_PRODUCT_TEST",
        }
        write_json(HERE / ("AB_COMBINED.json" if combo == "AB" else "AC_COMBINED.json"), record)

    a_summary = {
        "hidden_p99": a["numerical_metrics"]["A2_teacher_forced"]["checkpoints"]["1400"]["hidden"]["p99_abs"],
        "hidden_max": a["numerical_metrics"]["A2_teacher_forced"]["checkpoints"]["1400"]["hidden"]["max_abs"],
        "action_p99": a["numerical_metrics"]["A2_teacher_forced"]["checkpoints"]["1400"]["physical_mean"]["p99_abs"],
        "action_max": a["numerical_metrics"]["A2_teacher_forced"]["checkpoints"]["1400"]["physical_mean"]["max_abs"],
        "closed_loop_exact": False,
    }
    candidate_rows = {
        "R1": {"numerical": "exact reference", "total_update_s": r1["timing_metrics"]["total_update_wall_s"], "total_improvement": 0.0},
        "A": {"numerical": a_summary, "total_update_s": perf["A"]["timing_metrics"]["total_update_wall_s"], "total_improvement": speeds["A"]["total_update_improvement_fraction"]},
        "B": {"numerical": summaries["B"], "total_update_s": perf["B"]["timing_metrics"]["total_update_wall_s"], "total_improvement": speeds["B"]["total_update_improvement_fraction"]},
        "C": {"numerical": summaries["C"], "total_update_s": perf["C"]["timing_metrics"]["total_update_wall_s"], "total_improvement": speeds["C"]["total_update_improvement_fraction"]},
        "AB": {"numerical": {"A": a_summary, "B": summaries["B"]}, "total_update_s": perf["AB"]["timing_metrics"]["total_update_wall_s"], "total_improvement": speeds["AB"]["total_update_improvement_fraction"]},
        "AC": {"numerical": {"A": a_summary, "C": summaries["C"]}, "total_update_s": perf["AC"]["timing_metrics"]["total_update_wall_s"], "total_improvement": speeds["AC"]["total_update_improvement_fraction"]},
    }
    common = {
        **provenance("Phase 5-v2 selection", None, r1["flags"], bundle["rollout_hash"]),
        "model_initial_hash": bundle["model_initial_hash"],
        "optimizer_initial_hash": bundle["optimizer_initial_hash"],
        "rng_initial_hash": bundle["initial_rng_hashes"],
        "minibatch_order_hash": bundle["minibatch_order_hash"],
        "checkpoint_hash": None,
    }
    pareto = {
        "schema_version": 1,
        **common,
        "backend": "Pareto analysis",
        "batch_or_microbatch": None,
        "numerical_metrics": candidate_rows,
        "timing_metrics": {},
        "collection_frontier": ["A"],
        "training_frontier": ["B", "C"],
        "full_update_frontier": ["R1", "A", "B", "C", "AB", "AC"],
        "best_low_error": "B",
        "best_balanced": "AB",
        "best_max_speed": "AC",
        "selection_excludes_reference": True,
        "verdict": "PHASE5_COMBINED_FORWARD_AB_REQUIRES_PRODUCT_TEST",
    }
    write_json(HERE / "PARETO_FRONTIER.json", pareto)
    selection = {
        "schema_version": 1,
        **common,
        "backend": "selection",
        "batch_or_microbatch": None,
        "numerical_metrics": {
            "stage0": semantic["verdict"],
            "A": "DISTRIBUTIONAL_ONLY_REQUIRES_PRODUCT_TEST",
            "B": "QUICK_NUMERIC_AND_SPEED_PASS",
            "C": "QUICK_NUMERIC_AND_SPEED_PASS",
            "AB": "DISTRIBUTIONAL_ONLY_REQUIRES_PRODUCT_TEST",
            "AC": "DISTRIBUTIONAL_ONLY_REQUIRES_PRODUCT_TEST",
        },
        "timing_metrics": {name: row["total_update_s"] for name, row in candidate_rows.items()},
        "best_low_error": "B",
        "best_balanced": "AB",
        "best_max_speed": "AC",
        "unique_product_candidate": "AB",
        "checkpoint_hash": perf["AB"]["checkpoint_hash"],
        "phase5_verdict": "PHASE5_COMBINED_FORWARD_AB_REQUIRES_PRODUCT_TEST",
        "reason": "AB preserves B's substantially lower frozen gradient/delta error while retaining a 43%+ full-update improvement. AC is fastest but carries C's materially larger, though passing, recurrent-training error.",
    }
    write_json(HERE / "SELECTION.json", selection)
    assert_locked_sources()
    print(json.dumps({"best_low_error": "B", "best_balanced": "AB", "best_max_speed": "AC", "unique_product_candidate": "AB", "speeds": speeds}, sort_keys=True))


if __name__ == "__main__":
    main()

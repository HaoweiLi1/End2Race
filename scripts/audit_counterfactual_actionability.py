#!/usr/bin/env python3
"""P2: bounded counterfactual pulse, actionability, reward, and exploration audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

try:
    from audit_rl_direction_common import (
        EXPERIMENT_DIR,
        GAMMA,
        PREREGISTRATION_PATH,
        ROOT,
        RUN_DIR,
        SPEED_PHYSICAL_STD,
        STEERING_LATENT_STD,
        TIMESTEP,
        FixedScenarioProvider,
        _poses,
        assert_frozen_contract,
        load_actor,
        make_env,
        oriented_rectangle_clearance,
        read_json,
        set_determinism,
        sha256_file,
        write_json_atomic,
    )
except ModuleNotFoundError:
    from scripts.audit_rl_direction_common import (
        EXPERIMENT_DIR,
        GAMMA,
        PREREGISTRATION_PATH,
        ROOT,
        RUN_DIR,
        SPEED_PHYSICAL_STD,
        STEERING_LATENT_STD,
        TIMESTEP,
        FixedScenarioProvider,
        _poses,
        assert_frozen_contract,
        load_actor,
        make_env,
        oriented_rectangle_clearance,
        read_json,
        set_determinism,
        sha256_file,
        write_json_atomic,
    )
from ppo.policy import EVALUATOR_STEER_BOUND
from ppo.scenarios import load_hard_pool, scenario_from_dict


SEED = 20260717
BRANCH_SECONDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
DURATION_SECONDS = (0.25, 0.50)
PULSES = (
    ("STEER_M040", -0.04, 0.0),
    ("STEER_M020", -0.02, 0.0),
    ("STEER_P020", 0.02, 0.0),
    ("STEER_P040", 0.04, 0.0),
    ("SPEED_M050", 0.0, -0.50),
    ("SPEED_M025", 0.0, -0.25),
    ("SPEED_P025", 0.0, 0.25),
    ("COMBINED_M020_M025", -0.02, -0.25),
    ("COMBINED_P020_M025", 0.02, -0.25),
)


def _atanh(value: float) -> float:
    return 0.5 * (math.log1p(value) - math.log1p(-value))


def _one_sided_tail(abs_z: float) -> float:
    return max(0.5 * math.erfc(abs_z / math.sqrt(2.0)), np.finfo(np.float64).tiny)


def _pulse_class(steering: float, speed: float) -> str:
    if steering != 0.0 and speed != 0.0:
        return "combined"
    if steering != 0.0:
        return "steering_only"
    return "speed_only"


def _run_episode(
    env,
    provider: FixedScenarioProvider,
    actor,
    scenario,
    device: torch.device,
    *,
    branch_start_step: int | None,
    duration_steps: int,
    pulse_id: str,
    steering_offset: float,
    speed_offset: float,
) -> dict[str, Any]:
    provider.set(scenario, sampler_branch="p2_counterfactual", hard_pool_id="P2")
    observation, _reset_info = env.reset(seed=SEED)
    hidden = torch.zeros((1, 1, actor.gru.hidden_size), dtype=torch.float32, device=device)
    discounted_return = 0.0
    minimum_clearance = float("inf")
    step_index = 0
    pulse_steps_executed = 0
    pulse_touched_steering_bound = False
    mahalanobis: list[float] = []
    steering_sigmas: list[float] = []
    speed_sigmas: list[float] = []
    log_directional_probability = 0.0
    info: dict[str, Any] | None = None
    while True:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        with torch.inference_mode():
            raw_action, hidden = actor(
                tensor[:360].reshape(1, 1, 360),
                tensor[360:].reshape(1, 1, 1),
                hidden,
            )
        raw = raw_action[0, 0].detach().cpu().numpy().astype(np.float64)
        base = raw.copy()
        base[0] = np.clip(base[0], -EVALUATOR_STEER_BOUND, EVALUATOR_STEER_BOUND)
        action = base.copy()
        in_pulse = (
            branch_start_step is not None
            and branch_start_step <= step_index < branch_start_step + duration_steps
        )
        if in_pulse:
            requested_steering = float(base[0] + steering_offset)
            action[0] = np.clip(requested_steering, -EVALUATOR_STEER_BOUND, EVALUATOR_STEER_BOUND)
            action[1] = float(base[1] + speed_offset)
            pulse_touched_steering_bound |= not math.isclose(
                requested_steering,
                float(action[0]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            mean_normalized = float(np.clip(raw[0] / EVALUATOR_STEER_BOUND, -1.0 + 1.0e-6, 1.0 - 1.0e-6))
            action_normalized = float(
                np.clip(
                    action[0] / EVALUATOR_STEER_BOUND,
                    -1.0 + np.finfo(np.float32).eps,
                    1.0 - np.finfo(np.float32).eps,
                )
            )
            steering_sigma = (_atanh(action_normalized) - _atanh(mean_normalized)) / STEERING_LATENT_STD
            speed_sigma = (float(action[1]) - float(raw[1])) / SPEED_PHYSICAL_STD
            steering_sigmas.append(float(steering_sigma))
            speed_sigmas.append(float(speed_sigma))
            mahalanobis.append(float(math.hypot(steering_sigma, speed_sigma)))
            step_probability = 1.0
            if steering_offset != 0.0:
                step_probability *= _one_sided_tail(abs(steering_sigma))
            if speed_offset != 0.0:
                step_probability *= _one_sided_tail(abs(speed_sigma))
            log_directional_probability += math.log(step_probability)
            pulse_steps_executed += 1
        observation, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        first_pose, second_pose = _poses(env._raw_observation)
        minimum_clearance = min(minimum_clearance, oriented_rectangle_clearance(first_pose, second_pose))
        discounted_return += (GAMMA**step_index) * float(reward)
        step_index += 1
        if terminated or truncated:
            break
        if step_index > 1000:
            raise RuntimeError(f"P2 episode exceeded 1000 steps: {scenario.scenario_id}")
    if info is None:
        raise RuntimeError("P2 episode produced no transition")
    collision = bool(info["ego_collision"])
    relative = float(info["relative_position_m"])
    outcome = "ego_collision" if collision else ("overtake" if relative > 0.0 else "follow")
    max_single_dimension_sigma = max(
        [abs(value) for value in steering_sigmas + speed_sigmas],
        default=0.0,
    )
    sequence_probability = float(math.exp(max(log_directional_probability, math.log(np.finfo(np.float64).tiny))))
    return {
        "scenario_id": scenario.scenario_id,
        "pulse_id": pulse_id,
        "pulse_class": "no_op" if pulse_id == "NO_OP" else _pulse_class(steering_offset, speed_offset),
        "branch_start_step": branch_start_step,
        "duration_steps": duration_steps,
        "steering_offset_rad": steering_offset,
        "speed_offset_mps": speed_offset,
        "pulse_steps_executed": pulse_steps_executed,
        "pulse_touched_steering_bound": pulse_touched_steering_bound,
        "outcome": outcome,
        "ego_collision": collision,
        "opponent_collision": bool(info["opponent_collision"]),
        "steps": step_index,
        "elapsed_time": float(info["elapsed_time"]),
        "discounted_return": float(discounted_return),
        "min_oriented_clearance_m": float(minimum_clearance),
        "final_relative_progress_m": relative,
        "pulse_mahalanobis_mean": float(np.mean(mahalanobis)) if mahalanobis else 0.0,
        "pulse_mahalanobis_max": float(np.max(mahalanobis)) if mahalanobis else 0.0,
        "max_single_dimension_sigma": float(max_single_dimension_sigma),
        "sequence_log_directional_tail_probability": float(log_directional_probability),
        "sequence_directional_tail_probability": sequence_probability if mahalanobis else 1.0,
    }


def _safe_harm_scenarios(safe_reference: dict[str, Any]) -> list[Any]:
    overtakes = [row for row in safe_reference["scenarios"] if row["selection_group"] == "safe_overtake"][:12]
    follows = [row for row in safe_reference["scenarios"] if row["selection_group"] == "safe_follow"][:12]
    rows = overtakes + follows
    if len(rows) != 24:
        raise RuntimeError("P2 safe harm panel must contain 12 safe-overtake and 12 safe-follow cases")
    return [scenario_from_dict(row["scenario"]) for row in rows]


def _quantile_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
    }


def _pulse_template(row: dict[str, Any]) -> tuple[str, float]:
    """A pulse template includes both its physical offset and fixed duration."""

    return str(row["pulse_id"]), float(row["duration_seconds"])


def _pulse_template_id(template: tuple[str, float]) -> str:
    pulse_id, duration_seconds = template
    return f"{pulse_id}__D{int(round(duration_seconds * 1000)):03d}MS"


def main() -> None:
    started = time.monotonic()
    frozen_hashes = assert_frozen_contract()
    preregistration = read_json(PREREGISTRATION_PATH)
    safe_reference = read_json(EXPERIMENT_DIR / "SAFE_REFERENCE.json")
    raw_path = RUN_DIR / "p2" / "counterfactual_branches.json"
    result_path = EXPERIMENT_DIR / "P2_COUNTERFACTUAL_ACTIONABILITY.json"
    previous_result = read_json(result_path) if result_path.is_file() else None
    previous_result_sha256 = sha256_file(result_path) if result_path.is_file() else None
    if raw_path.is_file():
        if (
            previous_result is not None
            and sha256_file(raw_path) != previous_result["raw_branches"]["sha256"]
        ):
            raise RuntimeError("P2 resumable raw record hash does not match the previous result")
        raw_record = read_json(raw_path)
        baseline_rows = raw_record["h0_baselines"]
        collision_rows = raw_record["collision_branches"]
        safe_rows = raw_record["safe_harm_branches"]
        if len(baseline_rows) != 24 or len(safe_rows) != 2592:
            raise RuntimeError("P2 resumable raw record has invalid panel counts")
        print(
            f"P2_RAW_RESUME collision_rows={len(collision_rows)} safe_rows={len(safe_rows)}",
            flush=True,
        )
    else:
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("P2 requires CUDA")
        set_determinism(SEED)
        actor = load_actor(device)
        provider = FixedScenarioProvider()
        env = make_env(provider, SEED)
        h0_scenarios, _h0_ids, _h0_manifest = load_hard_pool("h0_current_det")
        baseline_rows = {}
        collision_rows = []
        safe_rows = []
        try:
            for index, scenario in enumerate(h0_scenarios, start=1):
                baseline = _run_episode(
                    env,
                    provider,
                    actor,
                    scenario,
                    device,
                    branch_start_step=None,
                    duration_steps=0,
                    pulse_id="NO_OP",
                    steering_offset=0.0,
                    speed_offset=0.0,
                )
                baseline_rows[scenario.scenario_id] = baseline
                print(f"P2_H0_BASELINE {index}/24 outcome={baseline['outcome']}", flush=True)
                if not baseline["ego_collision"]:
                    continue
                collision_steps = int(baseline["steps"])
                for branch_seconds in BRANCH_SECONDS:
                    branch_start_step = collision_steps - int(round(branch_seconds / TIMESTEP))
                    if branch_start_step < 0:
                        collision_rows.append(
                            {
                                "scenario_id": scenario.scenario_id,
                                "branch_seconds_before_collision": branch_seconds,
                                "status": "BRANCH_BEFORE_EPISODE_START",
                            }
                        )
                        continue
                    for duration_seconds in DURATION_SECONDS:
                        duration_steps = int(round(duration_seconds / TIMESTEP))
                        for pulse_id, steering_offset, speed_offset in PULSES:
                            row = _run_episode(
                                env,
                                provider,
                                actor,
                                scenario,
                                device,
                                branch_start_step=branch_start_step,
                                duration_steps=duration_steps,
                                pulse_id=pulse_id,
                                steering_offset=steering_offset,
                                speed_offset=speed_offset,
                            )
                            row["branch_seconds_before_collision"] = branch_seconds
                            row["duration_seconds"] = duration_seconds
                            row["status"] = "COMPLETED"
                            collision_rows.append(row)
                    print(
                        f"P2_COLLISION_BRANCH scenario={scenario.scenario_id} "
                        f"branch_seconds={branch_seconds}",
                        flush=True,
                    )

            harm_scenarios = _safe_harm_scenarios(safe_reference)
            for scenario_index, scenario in enumerate(harm_scenarios, start=1):
                for branch_seconds in BRANCH_SECONDS:
                    branch_start_step = int(round((8.0 - branch_seconds) / TIMESTEP))
                    for duration_seconds in DURATION_SECONDS:
                        duration_steps = int(round(duration_seconds / TIMESTEP))
                        for pulse_id, steering_offset, speed_offset in PULSES:
                            row = _run_episode(
                                env,
                                provider,
                                actor,
                                scenario,
                                device,
                                branch_start_step=branch_start_step,
                                duration_steps=duration_steps,
                                pulse_id=pulse_id,
                                steering_offset=steering_offset,
                                speed_offset=speed_offset,
                            )
                            row["branch_seconds_before_timeout"] = branch_seconds
                            row["duration_seconds"] = duration_seconds
                            safe_rows.append(row)
                print(f"P2_SAFE_HARM scenario={scenario_index}/24 id={scenario.scenario_id}", flush=True)
        finally:
            env.close()
        raw_record = {
            "schema_version": 1,
            "record": "P2_COUNTERFACTUAL_RAW_BRANCHES",
            "h0_baselines": baseline_rows,
            "collision_branches": collision_rows,
            "safe_harm_branches": safe_rows,
        }
        write_json_atomic(raw_path, raw_record)

    safe_by_template: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in safe_rows:
        safe_by_template[_pulse_template(row)].append(row)
    template_harm: dict[str, Any] = {}
    usable_templates: set[tuple[str, float]] = set()
    for template, rows in sorted(safe_by_template.items()):
        pulse_id, duration_seconds = template
        harmed_scenarios = sorted({row["scenario_id"] for row in rows if row["ego_collision"]})
        harm_rate = len(harmed_scenarios) / 24.0
        usable = harm_rate < 0.05
        if usable:
            usable_templates.add(template)
        template_harm[_pulse_template_id(template)] = {
            "pulse_id": pulse_id,
            "duration_seconds": duration_seconds,
            "trials": len(rows),
            "harmed_scenario_count": len(harmed_scenarios),
            "harmed_scenario_ids": harmed_scenarios,
            "case_level_collision_rate": harm_rate,
            "usable_pulse": usable,
        }

    completed_collision_rows = [row for row in collision_rows if row.get("status") == "COMPLETED"]
    by_collision_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed_collision_rows:
        by_collision_case[row["scenario_id"]].append(row)
    reproduced_ids = sorted(scenario_id for scenario_id, row in baseline_rows.items() if row["ego_collision"])
    case_summaries: dict[str, Any] = {}
    earliest_actionable: list[float] = []
    last_actionable: list[float] = []
    repairable_count = 0
    best_return_above_noop_count = 0
    best_pulses: list[dict[str, Any]] = []
    repair_counts_by_class: Counter[str] = Counter()
    for scenario_id in reproduced_ids:
        baseline = baseline_rows[scenario_id]
        usable_safe = [
            row
            for row in by_collision_case[scenario_id]
            if _pulse_template(row) in usable_templates and not row["ego_collision"]
        ]
        repairs = bool(usable_safe)
        if repairs:
            repairable_count += 1
            times = [float(row["branch_seconds_before_collision"]) for row in usable_safe]
            earliest_actionable.append(max(times))
            last_actionable.append(min(times))
            for pulse_class in {str(row["pulse_class"]) for row in usable_safe}:
                repair_counts_by_class[pulse_class] += 1
            best = max(usable_safe, key=lambda row: (float(row["discounted_return"]), row["pulse_id"]))
            best["return_exceeds_no_op"] = float(best["discounted_return"]) > float(baseline["discounted_return"])
            best_return_above_noop_count += int(best["return_exceeds_no_op"])
            best_pulses.append(best)
        else:
            best = None
        case_summaries[scenario_id] = {
            "baseline": baseline,
            "usable_safe_repair_found": repairs,
            "usable_safe_pass_repair_found": any(row["outcome"] == "overtake" for row in usable_safe),
            "usable_safe_repair_count": len(usable_safe),
            "earliest_actionable_seconds_before_collision": max(
                [float(row["branch_seconds_before_collision"]) for row in usable_safe],
                default=None,
            ),
            "last_actionable_seconds_before_collision": min(
                [float(row["branch_seconds_before_collision"]) for row in usable_safe],
                default=None,
            ),
            "best_safe_branch": best,
        }

    reproduced_count = len(reproduced_ids)
    repairable_fraction = repairable_count / reproduced_count if reproduced_count else 0.0
    reward_better_fraction = best_return_above_noop_count / repairable_count if repairable_count else 0.0
    misaligned_count = repairable_count - best_return_above_noop_count
    misaligned_fraction = misaligned_count / repairable_count if repairable_count else 0.0
    reward_direction_ok = repairable_fraction >= 2.0 / 3.0 and reward_better_fraction >= 0.80
    reward_misaligned = repairable_count > 0 and misaligned_fraction >= 0.30
    exploration_bad = [
        row
        for row in best_pulses
        if float(row["max_single_dimension_sigma"]) > 3.0
        or float(row["sequence_directional_tail_probability"]) < 0.01
    ]
    exploration_insufficient = reward_direction_ok and len(exploration_bad) > len(best_pulses) / 2.0
    local_action_not_found = repairable_fraction < 0.50
    if reward_misaligned:
        verdict = "REWARD_MISALIGNED"
    elif local_action_not_found:
        verdict = "LOCAL_ACTION_NOT_FOUND"
    elif exploration_insufficient:
        verdict = "EXPLORATION_COVERAGE_INSUFFICIENT"
    elif reward_direction_ok:
        verdict = "REWARD_DIRECTION_OK"
    else:
        verdict = "INCONCLUSIVE"

    result = {
        "schema_version": 1,
        "record": "P2_COUNTERFACTUAL_ACTIONABILITY_AND_REWARD_RANKING",
        "status": "COMPLETED",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_head": preregistration["source"]["head"],
        "device": "cuda",
        "optimizer_steps": 0,
        "frozen_hashes": frozen_hashes,
        "h0_count": 24,
        "h0_collision_reproduced": reproduced_count,
        "h0_not_reproduced_ids": sorted(set(baseline_rows) - set(reproduced_ids)),
        "safe_harm_panel_ids": [scenario.scenario_id for scenario in _safe_harm_scenarios(safe_reference)],
        "safe_harm_panel_contract": "first 12 preregistered safe_overtake plus first 12 preregistered safe_follow rows",
        "pulse_template_definition": (
            "physical offset identity plus fixed duration; branch placement is a repeated trial of the same template"
        ),
        "safe_template_harm": template_harm,
        "usable_pulse_ids": sorted({pulse_id for pulse_id, _duration in usable_templates}),
        "usable_pulse_templates": [
            {
                "template_id": _pulse_template_id(template),
                "pulse_id": template[0],
                "duration_seconds": template[1],
            }
            for template in sorted(usable_templates)
        ],
        "repairable_case_count": repairable_count,
        "repairable_fraction": repairable_fraction,
        "best_safe_return_above_noop_count": best_return_above_noop_count,
        "best_safe_return_above_noop_fraction": reward_better_fraction,
        "reward_misaligned_case_count": misaligned_count,
        "reward_misaligned_fraction": misaligned_fraction,
        "best_safe_pulse_exploration_bad_count": len(exploration_bad),
        "best_safe_pulse_count": len(best_pulses),
        "actionability_window": {
            "earliest_actionable_seconds_before_collision": _quantile_summary(earliest_actionable),
            "last_actionable_seconds_before_collision": _quantile_summary(last_actionable),
            "repairable_cases_by_action_class": dict(sorted(repair_counts_by_class.items())),
        },
        "case_summaries": case_summaries,
        "gates": {
            "reward_direction_ok": reward_direction_ok,
            "reward_misaligned": reward_misaligned,
            "exploration_coverage_insufficient": exploration_insufficient,
            "local_action_not_found": local_action_not_found,
        },
        "verdict": verdict,
        "raw_branches": {
            "path": str(raw_path.relative_to(ROOT)),
            "sha256": sha256_file(raw_path),
            "collision_branch_rows": len(collision_rows),
            "safe_harm_branch_rows": len(safe_rows),
        },
        "probability_definition": "For each nonzero pulse dimension, use the one-sided Gaussian tail at the achieved signed latent/physical z magnitude; multiply dimensions and iid pulse steps. This is an at-least-as-extreme directional occurrence probability, not a continuous-action point probability.",
        "aggregation_revision": {
            "revision": 2,
            "superseded_result_sha256": previous_result_sha256,
            "superseded_verdict": None if previous_result is None else previous_result.get("verdict"),
            "reason": (
                "Completion audit found that revision 1 collapsed 0.25 s and 0.50 s pulses into one "
                "family, although duration is part of the guide's pulse specification. Revision 2 "
                "uses offset plus duration as the template; raw branches and all thresholds are unchanged."
            ),
            "raw_branches_reused": previous_result is not None,
        },
        "collection_elapsed_seconds": (
            float(previous_result["elapsed_seconds"])
            if previous_result is not None
            else float(time.monotonic() - started)
        ),
        "aggregation_elapsed_seconds": float(time.monotonic() - started),
    }
    result["elapsed_seconds"] = (
        result["collection_elapsed_seconds"] + result["aggregation_elapsed_seconds"]
        if previous_result is not None
        else result["collection_elapsed_seconds"]
    )
    write_json_atomic(EXPERIMENT_DIR / "P2_COUNTERFACTUAL_ACTIONABILITY.json", result)
    print(
        f"P2_COMPLETE verdict={verdict} reproduced={reproduced_count} "
        f"repairable={repairable_count} elapsed_seconds={result['elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

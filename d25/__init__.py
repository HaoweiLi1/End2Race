"""Locked D2.5 counterfactual branch library."""

from __future__ import annotations

from dataclasses import dataclass


LEADS_SECONDS = (3.0, 2.0, 1.0)
DURATIONS_SECONDS = (0.5, 0.3, 0.1)
MACRO_STEPS = 10
SIM_DT = 0.01


@dataclass(frozen=True)
class Intervention:
    intervention_id: str
    brake_mps: float
    steer_rad: float


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str
    requested_lead_s: float
    actual_lead_s: float
    start_step: int
    duration_steps: int
    intervention: Intervention


INTERVENTIONS = (
    Intervention("brake100_steer_m010", 1.0, -0.1),
    Intervention("brake100_steer_p010", 1.0, +0.1),
    Intervention("brake050_steer_m010", 0.5, -0.1),
    Intervention("brake050_steer_p010", 0.5, +0.1),
    Intervention("steer_m020", 0.0, -0.2),
    Intervention("steer_p020", 0.0, +0.2),
    Intervention("steer_m010", 0.0, -0.1),
    Intervention("steer_p010", 0.0, +0.1),
    Intervention("brake100", 1.0, 0.0),
    Intervention("brake050", 0.5, 0.0),
)


def build_branch_specs(impact_step: int) -> tuple[BranchSpec, ...]:
    impact_step = int(impact_step)
    if impact_step <= 0:
        raise ValueError("impact_step must be positive")
    specs = []
    for lead in LEADS_SECONDS:
        raw_start = impact_step - int(round(lead / SIM_DT))
        if raw_start < 0:
            continue
        start = (raw_start // MACRO_STEPS) * MACRO_STEPS
        actual_lead = (impact_step - start) * SIM_DT
        for duration in DURATIONS_SECONDS:
            duration_steps = int(round(duration / SIM_DT))
            if duration_steps % MACRO_STEPS != 0:
                raise AssertionError("duration is not macro aligned")
            for intervention in INTERVENTIONS:
                branch_id = (
                    f"lead{int(round(lead * 100)):03d}_"
                    f"start{start:04d}_dur{duration_steps:03d}_"
                    f"{intervention.intervention_id}"
                )
                specs.append(
                    BranchSpec(
                        branch_id=branch_id,
                        requested_lead_s=lead,
                        actual_lead_s=actual_lead,
                        start_step=start,
                        duration_steps=duration_steps,
                        intervention=intervention,
                    )
                )
    return tuple(specs)


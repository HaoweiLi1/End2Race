"""D0.1 canonical audit package."""

from __future__ import annotations

import copy


ANALYSIS_VERSION = "d0.1"
CLASSIFIER_VERSION = "d0.1-traj-1"
RUNCONFIG_SCHEMA = "d0.1-runconfig-1"


_RUNCONFIG = {
    "schema": RUNCONFIG_SCHEMA,
    "analysis_version": ANALYSIS_VERSION,
    "classifier_version": CLASSIFIER_VERSION,
    "source_run_id": "20260710_121955",
    "repository_root": "/home/haowei/Documents/End2Race",
    "eval_root": "eval_results",
    "assets_root": "f1tenth_racetracks",
    "goal_root": "Experiments/A0_project_registry",
    "opened_registry": "Experiments/A0_project_registry/opened_registry.tsv",
    "opened_at_utc": "2026-07-10T23:02:12+08:00",
    "tag_template": "p1v_{run}_{model}_{map}_off{offset}",
    "result_dir_template": "eval_results/{tag}_{map}",
    "models": {
        "bc": {
            "path": "pretrained/end2race.pth",
            "sha256": "b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4",
        },
        "cand160": {
            "path": "Experiments/A1_p1_validation/models/end2race_ppo_full_disc_r8192_seed1_20260709_210827_iter0160.pth",
            "sha256": "77cd79904f0f57c1e7a4914dd0b52384628dce225f9222e4e2274e0eda3b5aa6",
        },
        "cand120": {
            "path": "Experiments/A1_p1_validation/models/end2race_ppo_full_disc_r8192_seed1_20260709_210827_iter0120.pth",
            "sha256": "9f2f47bf46363946ba29c1fe5fcada3a3d5fe514ece6eb160c03b25d8f82b3b3",
        },
        "cand040": {
            "path": "Experiments/A1_p1_validation/models/end2race_ppo_full_disc_r8192_seed0_20260709_210827_iter0040.pth",
            "sha256": "c7a72f5564a191e103d319a7f66167e6969fb3528534b90bafba77ceb598d7e1",
        },
    },
    "grids": [
        ["Austin", 21],
        ["Austin", 42],
        ["Austin", 63],
        ["Austin", 84],
        ["Nuerburgring", 0],
        ["MoscowRaceway", 0],
        ["Hockenheim", 0],
    ],
    "offset_start_formula": "(i*max_wc//50+offset)%max_wc for i=0..49",
    "zero_start_formula": "i*max_wc//49 for i=0..49",
    "dev_start_formula": "i*max_wc//49 for i=0..49 on Austin raceline1",
    "max_wc_convention": "line count of raceline CSV minus 2",
    "ego_raceline": "raceline1",
    "opponent_racelines": ["raceline0", "raceline1", "raceline2"],
    "opponent_speedscales": [0.5, 0.6, 0.7, 0.8],
    "interval_idx": 15,
    "sim_dt": 0.01,
    "duration_s": 8.0,
    "duration_ticks": 800,
    "noise": 0.0,
    "noise_seed": 42,
    "expected_occurrences": {"smoke": 1200, "full": 16800},
    "bootstrap": {"B": 10000, "seed": 20260710},
    "reconciliation_targets": {
        "reviewer_predictions": {
            "source": ".agents/HANDOFF.md section 6 Tier 3",
            "modes": ["full"],
            "values": {
                "geometry_reconciliation.exact_N": 3072,
                "geometry_reconciliation.primary_N": 3036,
                "geometry_reconciliation.sensA_N": 3024,
                "geometry_reconciliation.sensB_N": 2772,
                "geometry_reconciliation.excluded_from_exact": 300,
                "geometry_reconciliation.already_excluded_by_primary": 36,
                "geometry_reconciliation.additional_vs_primary": 264,
                "geometry_reconciliation.sensitivityA_pairs": 12,
                "estimands.primary.bc.N": 3036,
                "estimands.primary.bc.collision": 170,
                "estimands.primary.bc.overtake": 1792,
                "estimands.primary.cand160.collision": 154,
                "estimands.primary.cand120.collision": 168,
                "estimands.primary.cand040.collision": 166,
                "collision_phases.primary.bc.opponent_raceline1.pre": 27,
                "collision_phases.primary.bc.opponent_raceline1.alongside": 50,
                "collision_phases.primary.bc.opponent_raceline1.post": 1,
                "collision_phases.primary.bc.opponent_raceline1.total": 78,
                "opponent_only_floor.primary.bc_count": 17,
                "opponent_only_floor.primary.identical_across_all_models": True,
            },
        },
        "d0_v1_sensitivity_reference": {
            "source": "Experiments/A2_d0_canonical_audit/d0_summary.json",
            "source_sha256": "c697cd498b9b9a7eb3e443df5f7027a34571315ae26bc63737a32c2d90af70f0",
            "modes": ["full"],
            "values": {
                "geometry_reconciliation.sensA_N": 3024,
                "estimands.sensA.bc.N": 3024,
                "estimands.sensA.bc.collision": 169,
                "estimands.sensA.bc.overtake": 1792,
                "estimands.sensA.cand160.collision": 154,
                "estimands.sensA.cand160.overtake": 1799,
                "estimands.sensA.cand120.collision": 167,
                "estimands.sensA.cand120.overtake": 1797,
                "estimands.sensA.cand040.collision": 166,
                "estimands.sensA.cand040.overtake": 1787,
                "strata.sensA.skill_F.N": 504,
                "strata.sensA.skill_F.bc.collision": 56,
                "strata.sensA.skill_F.bc.ego_collision": 56,
                "strata.sensA.skill_F.bc.overtake": 7,
                "strata.sensA.skill_S.N": 1008,
                "strata.sensA.skill_S.bc.collision": 75,
                "strata.sensA.skill_S.bc.ego_collision": 64,
                "strata.sensA.skill_S.bc.overtake": 792,
                "strata.sensA.other.N": 1512,
                "strata.sensA.other.bc.collision": 38,
                "strata.sensA.other.bc.ego_collision": 32,
                "strata.sensA.other.bc.overtake": 993,
                "collision_phases.sensA.bc.opponent_raceline1.pre": 26,
                "collision_phases.sensA.bc.opponent_raceline1.alongside": 50,
                "collision_phases.sensA.bc.opponent_raceline1.post": 1,
                "collision_phases.sensA.bc.opponent_raceline1.total": 77,
                "opponent_only_floor.sensA.bc_count": 17,
                "opponent_only_floor.sensA.identical_across_all_models": True,
            },
        },
    },
    "classifier": {
        "attempt_m": 0.6,
        "confirmed_lead_m": 2.0,
        "confirmed_hold_s": 0.7,
        "car_distance_m": 1.0,
        "alongside_strict_m": 0.6,
    },
}


def default_runconfig() -> dict:
    """Return an isolated copy of the frozen D0.1 RunConfig."""

    return copy.deepcopy(_RUNCONFIG)

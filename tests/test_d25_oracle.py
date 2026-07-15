#!/usr/bin/env python3
"""Unit tests for D2.5 macro branches and outcome classification."""

import numpy as np

from d25 import INTERVENTIONS, build_branch_specs
from d25.oracle import classify_trajectory, compose_branch_action


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def make_arrays(collision=False, ego_collision=False, opp_collision=False, pass_hold=False):
    n = 801
    time = np.arange(n, dtype=np.float32) * np.float32(0.01)
    rel = np.linspace(-3.0, -1.0, n + 1, dtype=np.float32)
    label = "following"
    if pass_hold:
        rel = np.linspace(-3.0, 3.0, n + 1, dtype=np.float32)
        rel[-71:] = 2.1
        label = "overtaking"
    if collision:
        label = "collision"
    zeros = np.zeros(n, dtype=np.float32)
    poses = np.zeros((n, 3), dtype=np.float32)
    return {
        "time": time,
        "ego_progress": rel[:-1],
        "opp_progress": zeros,
        "ego_collision": np.array(ego_collision, dtype=bool),
        "opp_collision": np.array(opp_collision, dtype=bool),
        "collision": np.array(collision, dtype=bool),
        "final_time": np.float32(8.01),
        "final_ego_pose": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "final_opp_pose": np.array([0.5 if collision else 3.0, 0.0, 0.0], dtype=np.float32),
        "final_ego_progress": np.float32(rel[-1]),
        "final_opp_progress": np.float32(0.0),
        "state_label": np.array(label),
    }


def main():
    specs = build_branch_specs(800)
    check("full-library-90", len(specs) == 90)
    check("fixed-first", specs[0].intervention == INTERVENTIONS[0])
    check("macro-start", all(spec.start_step % 10 == 0 for spec in specs))
    check("macro-duration", all(spec.duration_steps % 10 == 0 for spec in specs))
    check("actual-lead-not-shorter", all(spec.actual_lead_s >= spec.requested_lead_s for spec in specs))
    check("short-impact-30", len(build_branch_specs(150)) == 30)

    branch = specs[0]
    before = compose_branch_action(0.2, 5.0, branch, branch.start_step - 1)
    active = compose_branch_action(0.2, 5.0, branch, branch.start_step)
    after = compose_branch_action(0.2, 5.0, branch, branch.start_step + branch.duration_steps)
    check("inactive-before", before[:2] == (0.2, 5.0) and not before[3])
    check("active-residual", np.allclose(active[:2], [0.1, 4.0]) and active[3])
    check("inactive-after", after[:2] == (0.2, 5.0) and not after[3])
    clipped = compose_branch_action(0.5, 0.2, branch, branch.start_step)
    check("clipping-detected", clipped[2] and clipped[0] == 0.4 and clipped[1] == 0.0)

    safe_pass = classify_trajectory(make_arrays(pass_hold=True), "Austin")
    check("confirmed-pass", safe_pass.four_state == "confirmed_pass")
    safe_follow = classify_trajectory(make_arrays(), "Austin")
    check("safe-follow", safe_follow.four_state == "safe_follow")
    collision = classify_trajectory(
        make_arrays(collision=True, ego_collision=True), "Austin"
    )
    check("collision", collision.four_state == "collision")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

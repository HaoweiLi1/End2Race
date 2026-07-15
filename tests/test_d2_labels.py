#!/usr/bin/env python3
"""Synthetic tests for privileged D2 labels and horizon censoring."""

import numpy as np

from d2.labels import LabelConfig, ReferenceProjector, build_episode_labels


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def fixture(ego_collision=False, opp_collision=False, lateral=0.0):
    time = np.arange(5, dtype=np.float64)
    ego_x = np.arange(5, dtype=np.float64)
    opp_x = ego_x + 3.0
    zeros = np.zeros(5, dtype=np.float64)
    return {
        "time": time,
        "ego_actual_speed": np.full(5, 4.0),
        "opp_actual_speed": np.full(5, 2.0),
        "ego_pose": np.column_stack([ego_x, zeros, zeros]),
        "opp_pose": np.column_stack([opp_x, np.full(5, lateral), zeros]),
        "ego_progress": ego_x,
        "opp_progress": opp_x,
        "final_time": np.float64(5.0),
        "final_ego_progress": np.float64(5.0),
        "final_opp_progress": np.float64(8.0),
        "ego_collision": np.array(ego_collision),
        "opp_collision": np.array(opp_collision),
    }


def main():
    projector = ReferenceProjector.from_arrays(
        s=np.array([0.0, 10.0, 20.0]),
        xy=np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]),
    )
    config = LabelConfig(horizons=(0.5, 1.0, 2.0), ttc_cap_s=5.0)

    event = build_episode_labels(fixture(ego_collision=True), projector, 100.0, config)
    check("alignment-ok", event["alignment_status"] == "ok")
    check("signed-closing", np.array_equal(event["closing_rate"], np.full(5, 2.0, dtype=np.float32)))
    check("ttc-front-corridor", np.allclose(event["corridor_ttc"], 1.21, atol=1e-6))
    check("event-all-valid", np.all(event["ego_valid_200"]))
    check("exact-two-second-positive", event["ego_target_200"].tolist() == [False, False, False, True, True])
    check("exact-one-second-positive", event["ego_target_100"].tolist() == [False, False, False, False, True])
    check("half-second-none-at-one-hz", not np.any(event["ego_target_050"]))
    check("any-matches-ego", np.array_equal(event["ego_target_200"], event["any_target_200"]))

    safe = build_episode_labels(fixture(), projector, 100.0, config)
    check("normal-end-censored", safe["ego_valid_200"].tolist() == [True, True, True, True, False])
    check("normal-end-exact-boundary-negative", safe["ego_valid_200"][3] and not safe["ego_target_200"][3])
    check("safe-no-positive", not np.any(safe["any_target_200"]))

    competing = build_episode_labels(fixture(opp_collision=True), projector, 100.0, config)
    check("competing-ego-censored", competing["ego_valid_200"].tolist() == [True, True, True, True, False])
    check("competing-not-ego-positive", not np.any(competing["ego_target_200"]))
    check("competing-any-positive", competing["any_target_200"].tolist() == [False, False, False, True, True])

    outside = build_episode_labels(fixture(lateral=0.510001), projector, 100.0, config)
    check("outside-corridor-capped", np.array_equal(outside["corridor_ttc"], np.full(5, 5.0, dtype=np.float32)))

    contact_data = fixture()
    contact_data["opp_progress"] = contact_data["ego_progress"] + 0.58
    contact_data["final_opp_progress"] = np.float64(5.58)
    contact = build_episode_labels(contact_data, projector, 100.0, config)
    check("contact-zero-ttc", np.array_equal(contact["corridor_ttc"], np.zeros(5, dtype=np.float32)))

    reverse = fixture()
    reverse["ego_actual_speed"][:] = 1.0
    reverse["opp_actual_speed"][:] = 2.0
    receding = build_episode_labels(reverse, projector, 100.0, config)
    check("signed-receding", np.array_equal(receding["closing_rate"], np.full(5, -1.0, dtype=np.float32)))
    check("receding-capped", np.array_equal(receding["corridor_ttc"], np.full(5, 5.0, dtype=np.float32)))

    s, d, theta = projector.project_many(np.array([[2.0, 1.0], [12.0, -2.0]]))
    check("project-s", np.allclose(s, [2.0, 12.0]))
    check("project-signed-d", np.allclose(d, [1.0, -2.0]))
    check("project-theta", np.allclose(theta, 0.0))
    exhaustive = projector.project_many_exhaustive(np.array([[2.0, 1.0], [12.0, -2.0]]))
    check("project-fast-exhaustive", all(np.array_equal(left, right) for left, right in zip((s, d, theta), exhaustive)))

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

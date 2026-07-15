#!/usr/bin/env python3
"""Tests for D2 frame and episode-level probe metrics."""

import numpy as np

from d2.metrics import (
    binary_metrics,
    evaluate_alarm_threshold,
    select_alarm_threshold,
    ttc_mae,
)


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def main():
    y = np.array([0, 1, 0, 1], dtype=bool)
    p = np.array([0.1, 0.9, 0.4, 0.8], dtype=np.float64)
    metric = binary_metrics(y, p, prevalence_reference=0.5, bins=10)
    check("aucpr-perfect", metric["aucpr"] == 1.0)
    check("brier", abs(metric["brier"] - 0.055) < 1e-12)
    check("brier-skill", abs(metric["brier_skill"] - 0.78) < 1e-12)
    check("ece", abs(metric["ece"] - 0.2) < 1e-12)
    check("reliability-accounting", sum(row["count"] for row in metric["reliability"]) == 4)

    # Four episodes x three frames. Episodes 0/1 are safe; 2/3 are ego
    # collisions at time 3. Positive-window maxima are 0.8 and 0.7.
    episode_index = np.repeat(np.arange(4), 3)
    time = np.tile(np.array([0.0, 1.0, 2.0]), 4)
    probability = np.array(
        [
            0.1, 0.9, 0.3,  # safe max .9
            0.1, 0.2, 0.1,  # safe max .2
            0.1, 0.4, 0.8,  # event max .8
            0.1, 0.3, 0.7,  # event max .7
        ]
    )
    valid = np.ones(12, dtype=bool)
    positive_window = np.tile(np.array([False, False, True]), 4)
    episode_any_collision = np.array([False, False, True, True])
    episode_ego_collision = np.array([False, False, True, True])
    final_time = np.full(4, 3.0)

    selected = select_alarm_threshold(
        probability,
        valid,
        positive_window,
        episode_index,
        episode_any_collision,
        episode_ego_collision,
        time,
        final_time,
        false_alarm_limit=0.5,
    )
    check("threshold-max-recall", selected["threshold"] == 0.7)
    check("event-recall", selected["event_recall"] == 1.0)
    check("safe-fa", selected["safe_episode_false_alarm_rate"] == 0.5)
    check("event-count", selected["event_episode_count"] == 2)

    evaluated = evaluate_alarm_threshold(
        probability,
        valid,
        positive_window,
        episode_index,
        episode_any_collision,
        episode_ego_collision,
        time,
        final_time,
        threshold=selected["threshold"],
    )
    check("frozen-threshold-match", evaluated["event_recall"] == 1.0 and evaluated["safe_episode_false_alarm_rate"] == 0.5)
    check("lead-times", evaluated["earliest_lead_seconds"] == [1.0, 1.0])
    check("warned-at-one", evaluated["warned_at_least_1s"] == 1.0)
    check("warned-at-two", evaluated["warned_at_least_2s"] == 0.0)

    tie_probability = probability.copy()
    tie_probability[-1] = 0.8
    tied = select_alarm_threshold(
        tie_probability,
        valid,
        positive_window,
        episode_index,
        episode_any_collision,
        episode_ego_collision,
        time,
        final_time,
        false_alarm_limit=0.5,
    )
    check("higher-threshold-tie-break", tied["threshold"] == 0.8)

    mae = ttc_mae(
        np.array([0.2, 1.5, 2.0, 5.0]),
        np.array([0.3, 1.2, 0.0, 4.0]),
    )
    check("ttc-region-strict", abs(mae["mae"] - 0.2) < 1e-12 and mae["count"] == 2)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

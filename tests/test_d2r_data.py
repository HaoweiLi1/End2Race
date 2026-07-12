#!/usr/bin/env python3
"""Unit tests for D2R causal history, TTC bins, and locked sampling."""

import numpy as np

from d2r import HISTORY_OFFSETS
from d2r.data import (
    SIGNALS_MANIFEST_SHA256,
    causal_history_indices,
    deterministic_fit_indices,
    inverse_sampling_weights,
    ttc_bin_centers,
    ttc_bin_indices,
)


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def main():
    check(
        "signals-manifest-hash",
        SIGNALS_MANIFEST_SHA256
        == "d653d77dcf270ce9b9e714d23a9b5600b15dfb90996e627610153be1763b513a",
    )
    episode_index = np.repeat(np.arange(2), [120, 130]).astype(np.int64)
    starts = np.array([0, 120], dtype=np.int64)
    frames = np.array([0, 3, 100, 120, 125, 249], dtype=np.int64)
    history = causal_history_indices(frames, episode_index, starts)
    check("history-shape", history.shape == (len(frames), len(HISTORY_OFFSETS)))
    check("current-first", np.array_equal(history[:, 0], frames))
    check("causal", np.all(history <= frames[:, None]))
    check("first-clamped", np.all(history[0] == 0) and np.all(history[3] == 120))
    check("no-cross", np.all(episode_index[history] == episode_index[frames, None]))
    check("one-second", history[2, -1] == 0 and history[-1, -1] == 149)

    values = np.array([0.0, 0.099999, 0.1, 1.9999, 2.0, 4.999, 5.0])
    bins = ttc_bin_indices(values)
    check("ttc-bins", np.array_equal(bins, [0, 0, 1, 19, 20, 49, 49]))
    centers = ttc_bin_centers()
    check("centers", len(centers) == 50 and np.isclose(centers[0], 0.05) and np.isclose(centers[-1], 4.95))

    n = 80
    frame_episode = np.repeat(np.arange(4), 20)
    train = np.array([True, True, False, False])
    target = np.zeros(n, dtype=bool)
    target[[3, 23, 43]] = True
    ttc = np.full(n, 5.0, dtype=np.float32)
    ttc[[7, 27, 47]] = 1.0
    selected = deterministic_fit_indices(frame_episode, train, target, ttc)
    check("heldout-excluded", np.all(frame_episode[selected] < 2))
    check("forced-events", {3, 7, 23, 27}.issubset(set(selected.tolist())))
    check("background-stride", {0, 20}.issubset(set(selected.tolist())))
    weights = inverse_sampling_weights(selected, target, ttc)
    by_index = dict(zip(selected.tolist(), weights.tolist()))
    check("forced-weight-one", all(by_index[index] == 1.0 for index in (3, 7, 23, 27)))
    check("background-weight-20", by_index[0] == 20.0 and by_index[20] == 20.0)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

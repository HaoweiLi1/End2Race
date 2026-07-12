#!/usr/bin/env python3
"""Synthetic contract tests for D0.1 corrected trajectory outcomes."""

import numpy as np

from d0.outcomes import (
    EQUALITY_FIELDS,
    align_rel,
    classify_collision,
    classify_outcome,
    equality_vector,
    normalize_archived_outcome,
    unwrap_progress,
)


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def expect_raises(name, exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"FAIL {name}: expected {exc_type.__name__}")


def episode(rel, dt=0.01, collision=False, ego_collision=False, opp_collision=False,
            final_gap=None, final_dist=2.0):
    """Build recorded-plus-terminal progress from a desired raw rel series."""
    rel = np.asarray(rel, dtype=np.float64)
    if len(rel) < 2:
        raise ValueError("need recorded and terminal frames")
    n = len(rel) - 1
    time = np.arange(n, dtype=np.float64) * dt
    final_time = float(time[-1] + (dt if final_gap is None else final_gap))
    ego_pose = np.zeros((n, 3), dtype=np.float64)
    opp_pose = np.zeros((n, 3), dtype=np.float64)
    final_ego_pose = np.array([0.0, 0.0, 0.0])
    final_opp_pose = np.array([final_dist, 0.0, 0.0])
    return {
        "time": time,
        "ego_progress": rel[:-1].copy(),
        "opp_progress": np.zeros(n, dtype=np.float64),
        "ego_pose": ego_pose,
        "opp_pose": opp_pose,
        "collision": np.array(collision, dtype=bool),
        "ego_collision": np.array(ego_collision, dtype=bool),
        "opp_collision": np.array(opp_collision, dtype=bool),
        "final_time": np.float64(final_time),
        "final_ego_progress": np.float64(rel[-1]),
        "final_opp_progress": np.float64(0.0),
        "final_ego_pose": final_ego_pose,
        "final_opp_pose": final_opp_pose,
        "state_label": np.array("collision" if collision else ("overtaking" if rel[-1] > 0 else "following")),
    }


def meta(raw, ego_collision=False, opp_collision=False):
    return {
        "outcome": raw,
        "state_label": raw,
        "ego_collision": bool(ego_collision),
        "opp_collision": bool(opp_collision),
    }


def main():
    check("raw-follow", normalize_archived_outcome("following") == "follow")
    check("raw-overtake", normalize_archived_outcome("overtaking") == "overtake")
    check("raw-collision", normalize_archived_outcome("collision") == "collision")
    expect_raises("raw-alias-follow", ValueError, lambda: normalize_archived_outcome("follow"))
    expect_raises("raw-alias-overtake", ValueError, lambda: normalize_archived_outcome("overtake"))

    unwrapped = unwrap_progress(np.array([418.0, 419.0, 1.0, 2.0]), 420.0)
    check("unwrap-continuity", np.allclose(unwrapped.values, [418.0, 419.0, 421.0, 422.0]))
    check("unwrap-count", unwrapped.wrap_count == 1)

    # Mid-track pass with a full 0.7 s terminal hold at >=2 m.
    rel = np.concatenate([np.linspace(-2.0, 1.9, 29), np.full(71, 2.0), [2.1]])
    rec = classify_outcome(episode(rel), meta("overtaking"), 420.0)
    check("clean-pass-outcome", rec.archived_outcome3 == rec.corrected_outcome3 == "overtake")
    check("clean-pass-confirmed", rec.confirmed_safe_pass is True and rec.four_state == "confirmed_pass")

    # Reviewer seam class: raw terminal lead is actually 2 m behind.
    seam_npz = episode([415.0, 416.0, 417.0, 418.0])
    seam = classify_outcome(seam_npz, meta("overtaking"), 420.0)
    check("seam-k", seam.alignment_k == -1)
    check("seam-corrected-follow", seam.corrected_outcome3 == "follow")
    check("seam-mismatch", seam.archived_outcome3 != seam.corrected_outcome3)
    check("whole-series-k", np.allclose(seam.rel_series.values - seam.rel_series.raw_values, -420.0))

    # Both cars cross the seam; corrected relative progress remains continuous.
    aligned = align_rel(
        np.array([416.0, 419.0, 2.0]),
        np.array([418.0, 1.0, 4.0]),
        5.0,
        7.0,
        420.0,
    )
    check("dual-seam-status", aligned.status == "ok")
    check("dual-seam-continuity", np.max(np.abs(np.diff(aligned.values))) < 10.0)

    genuine = classify_outcome(episode([-2.0, -1.0, 0.5, 1.0]), meta("overtaking"), 420.0)
    check("genuine-pass", genuine.corrected_outcome3 == "overtake")
    check("terminal-only", genuine.four_state == "terminal_overtake_only")

    anchor = classify_outcome(episode([0.0, 0.1, 0.2]), meta("overtaking"), 420.0)
    check("anchor-failure", anchor.alignment_status == "alignment_failure")
    check("anchor-unknown", anchor.corrected_outcome3 == "unknown" and anchor.interaction_attempt == "unknown")

    collision_rel = np.linspace(-2.0, -0.2, 312)
    coll_npz = episode(collision_rel, collision=True, ego_collision=True, opp_collision=True, final_dist=0.8)
    coll = classify_outcome(coll_npz, meta("collision", True, True), 420.0)
    check("collision-valid", coll.physics_status == "ok" and coll.censored is False)
    check("collision-no-pass", coll.confirmed_safe_pass is False and coll.four_state == "collision")
    check("collision-direct", coll.collision_involvement == "both")

    short = classify_outcome(episode(np.linspace(-1.0, -0.5, 51)), meta("following"), 420.0)
    check("short-censored", short.censored is True)

    # Lead threshold is inclusive at 2.0.
    rel199 = np.concatenate([[-1.0], np.full(71, 1.99), [1.99]])
    rel200 = np.concatenate([[-1.0], np.full(71, 2.0), [2.0]])
    r199 = classify_outcome(episode(rel199), meta("overtaking"), 420.0)
    r200 = classify_outcome(episode(rel200), meta("overtaking"), 420.0)
    check("lead-199", r199.confirmed_safe_pass is False)
    check("lead-200", r200.confirmed_safe_pass is True)

    # The terminal (71st nominal window frame) is decisive.
    terminal_bad = np.concatenate([[-1.0], np.full(70, 2.1), [1.9]])
    terminal_good = np.concatenate([[-1.0], np.full(70, 2.0), [2.0]])
    check("terminal-window-bad", classify_outcome(episode(terminal_bad), meta("overtaking"), 420.0).confirmed_safe_pass is False)
    check("terminal-window-good", classify_outcome(episode(terminal_good), meta("overtaking"), 420.0).confirmed_safe_pass is True)

    for span, expected in ((0.69, True), (0.70, False), (0.71, False)):
        n = int(round(span / 0.01))
        values = np.linspace(-1.0, -0.5, n + 1)
        record = classify_outcome(episode(values), meta("following"), 420.0)
        check(f"censor-span-{span}", record.censored is expected)

    attempt60 = classify_outcome(episode([-1.0, -0.8, -0.6]), meta("following"), 420.0)
    attempt61 = classify_outcome(episode([-1.0, -0.8, -0.61]), meta("following"), 420.0)
    check("attempt-060", attempt60.interaction_attempt is True)
    check("attempt-061", attempt61.interaction_attempt is False)
    check("confirmed-subset", not r200.confirmed_safe_pass or r200.corrected_outcome3 == "overtake")

    for gap, expected in ((0.004, "invalid"), (0.010, "ok"), (0.016, "invalid")):
        record = classify_outcome(episode([-1.0, -0.9, -0.8], final_gap=gap), meta("following"), 420.0)
        check(f"gap-{gap}", record.physics_status == expected)

    for dt, expected in ((0.005, "ok"), (0.015, "ok"), (0.0049, "invalid"), (0.0151, "invalid")):
        record = classify_outcome(episode([-1.0, -0.9, -0.8], dt=dt, final_gap=0.01), meta("following"), 420.0)
        check(f"dt-{dt}", record.frame_spacing_status == expected)

    for dist, cause in ((0.99, "car"), (1.0, "car"), (1.01, "wall")):
        npz = episode([-1.0, -0.8, -0.2], collision=True, ego_collision=True, final_dist=dist)
        rs = align_rel(npz["ego_progress"], npz["opp_progress"], npz["final_ego_progress"], npz["final_opp_progress"], 420.0)
        event = classify_collision(npz, rs)
        check(f"cause-{dist}", event.cause == cause and event.ego_collision and not event.opp_collision)

    for terminal, phase in ((-0.60, "pre"), (-0.59, "alongside"), (0.59, "alongside"), (0.60, "post")):
        npz = episode([-1.0, -0.8, terminal], collision=True, ego_collision=True, final_dist=0.5)
        rs = align_rel(npz["ego_progress"], npz["opp_progress"], npz["final_ego_progress"], npz["final_opp_progress"], 420.0)
        event = classify_collision(npz, rs)
        check(f"phase-{terminal}", event.phase == phase)

    seam_collision = episode([415.0, 417.0, 419.0], collision=True, ego_collision=True, final_dist=0.5)
    seam_rs = align_rel(seam_collision["ego_progress"], seam_collision["opp_progress"], seam_collision["final_ego_progress"], seam_collision["final_opp_progress"], 420.0)
    seam_event = classify_collision(seam_collision, seam_rs)
    check("seam-collision-corrected-phase", seam_event.phase == "pre")

    mismatch_meta = meta("following")
    mismatch_meta["state_label"] = "overtaking"
    expect_raises("raw-json-disagreement", ValueError, lambda: classify_outcome(episode([-1.0, -0.9]), mismatch_meta, 420.0))

    vector = equality_vector(rec)
    check("equality-vector-complete", len(vector) == len(EQUALITY_FIELDS))
    check("equality-vector-order", vector[0] == rec.archived_outcome_raw and vector[-1] == rec.collision_final_dist_hex)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

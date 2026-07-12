#!/usr/bin/env python3
"""Regression tests for aggregate_eval.py completeness validation.

The anchor case reproduces the full_disc_r8192 seed1 iter300 failure
(2026-07-09 sweep): 600 workers launched, 112 crashed with Python's
uncaught-exception exit code 1 and wrote no metrics JSON. The legacy
exit-code aggregation counted them as "following" and reported a clean
600-episode result. The validated aggregator must reject that run.

Run: python tests/test_eval_aggregation.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def make_episode(sandbox, tmp_dir, i, outcome, exit_code=0, write_json=True,
                 episode_key=None, write_npz=True):
    with open(os.path.join(tmp_dir, f"{i}.exit"), "w") as f:
        f.write(f"{exit_code}\n")
    if not write_json:
        return
    npz_dir = f"npz_{os.path.basename(tmp_dir)}"
    npz_path = os.path.join(npz_dir, f"{i}.npz")
    if write_npz:
        os.makedirs(os.path.join(sandbox, npz_dir), exist_ok=True)
        with open(os.path.join(sandbox, npz_path), "wb") as f:
            f.write(b"fake-npz-bytes")
    metric = {
        "episode_key": episode_key or f"ol1_e{i}_o{i + 15}_s0.5",
        "outcome": outcome,
        "state_label": outcome,
        "state": {"following": 1, "overtaking": 2, "collision": 3}[outcome],
        "ego_collision": outcome == "collision",
        "opp_collision": False,
        "npz_path": npz_path,
        "avg_speed": 3.0,
        "speed_variance": 0.1,
        "total_distance": 20.0,
    }
    with open(os.path.join(tmp_dir, f"{i}.json"), "w") as f:
        json.dump(metric, f)


def run_aggregate(sandbox, tmp_dir, expected_total, result_tag):
    env = dict(os.environ, PYTHONPATH=REPO)
    return subprocess.run(
        [PY, os.path.join(REPO, "aggregate_eval.py"),
         "--tmp_dir", tmp_dir,
         "--expected_total", str(expected_total),
         "--model_path", "pretrained/fake.pth",
         "--map_name", "Austin",
         "--result_tag", result_tag,
         "--require_npz"],
        cwd=sandbox, env=env, capture_output=True, text=True)


def check(name, cond, detail=""):
    if not cond:
        print(f"FAIL {name}: {detail}")
        sys.exit(1)
    print(f"ok   {name}")


def results_json(sandbox, tag):
    return os.path.join(sandbox, "eval_results", f"{tag}_Austin", "results.json")


def test_silent_failure_488_of_600(sandbox):
    """112 crashed workers (exit 1, no JSON) must fail aggregation."""
    tmp = os.path.join(sandbox, "tmp_488")
    os.makedirs(tmp)
    outcomes = ["collision"] * 25 + ["overtaking"] * 279 + ["following"] * 184
    for i, oc in enumerate(outcomes):  # 488 valid episodes
        make_episode(sandbox, tmp, i, oc)
    for i in range(488, 600):  # crashed: uncaught exception -> exit 1, no JSON
        make_episode(sandbox, tmp, i, "following", exit_code=1, write_json=False)
    r = run_aggregate(sandbox, tmp, 600, "case488")
    check("488/600 rejected", r.returncode == 2, f"rc={r.returncode} err={r.stderr[-500:]}")
    check("488/600 names missing JSONs", "missing or empty metrics JSON" in r.stderr, r.stderr[-500:])
    check("488/600 no results.json", not os.path.exists(results_json(sandbox, "case488")))


def test_complete_run_passes(sandbox):
    tmp = os.path.join(sandbox, "tmp_ok")
    os.makedirs(tmp)
    outcomes = ["collision"] * 25 + ["overtaking"] * 279 + ["following"] * 296
    for i, oc in enumerate(outcomes):
        make_episode(sandbox, tmp, i, oc)
    r = run_aggregate(sandbox, tmp, 600, "caseok")
    check("complete run accepted", r.returncode == 0, f"rc={r.returncode} err={r.stderr[-500:]}")
    check("counts in RESULT line",
          "collision=25 overtake=279 follow=296 error=0 ego_collision=25" in r.stdout, r.stdout)
    with open(results_json(sandbox, "caseok")) as f:
        data = json.load(f)
    check("600 episodes written", len(data["episodes"]) == 600, str(len(data.get("episodes", {}))))
    check("final counts", data["final"]["collision_count"] == 25
          and data["final"]["overtaking_count"] == 279
          and data["final"]["error_count"] == 0
          and data["final"]["ego_collision_count"] == 25
          and data["final"]["validated"] is True, json.dumps(data["final"]))


def test_duplicate_key_rejected(sandbox):
    tmp = os.path.join(sandbox, "tmp_dup")
    os.makedirs(tmp)
    for i in range(4):
        make_episode(sandbox, tmp, i, "following", episode_key="ol1_e0_o15_s0.5" if i == 3 else None)
    r = run_aggregate(sandbox, tmp, 4, "casedup")
    check("duplicate key rejected", r.returncode == 2 and "duplicate episode_key" in r.stderr,
          f"rc={r.returncode} err={r.stderr[-300:]}")


def test_extra_artifact_rejected(sandbox):
    tmp = os.path.join(sandbox, "tmp_extra")
    os.makedirs(tmp)
    for i in range(4):
        make_episode(sandbox, tmp, i, "following")
    make_episode(sandbox, tmp, 4, "following")  # stale extra job beyond 0..3
    r = run_aggregate(sandbox, tmp, 4, "caseextra")
    check("extra artifact rejected", r.returncode == 2 and "unexpected extra artifact" in r.stderr,
          f"rc={r.returncode} err={r.stderr[-300:]}")


def test_nonzero_exit_with_json_rejected(sandbox):
    tmp = os.path.join(sandbox, "tmp_exit")
    os.makedirs(tmp)
    for i in range(4):
        make_episode(sandbox, tmp, i, "following", exit_code=137 if i == 2 else 0)
    r = run_aggregate(sandbox, tmp, 4, "caseexit")
    check("nonzero worker exit rejected", r.returncode == 2 and "worker exit code 137" in r.stderr,
          f"rc={r.returncode} err={r.stderr[-300:]}")


def test_missing_npz_rejected(sandbox):
    tmp = os.path.join(sandbox, "tmp_npz")
    os.makedirs(tmp)
    for i in range(4):
        make_episode(sandbox, tmp, i, "following", write_npz=i != 1)
    r = run_aggregate(sandbox, tmp, 4, "casenpz")
    check("missing npz rejected", r.returncode == 2 and "missing or empty NPZ" in r.stderr,
          f"rc={r.returncode} err={r.stderr[-300:]}")


def test_stale_results_merge_rejected(sandbox):
    tmp = os.path.join(sandbox, "tmp_stale")
    os.makedirs(tmp)
    for i in range(4):
        make_episode(sandbox, tmp, i, "following")
    stale_dir = os.path.join(sandbox, "eval_results", "casestale_Austin")
    os.makedirs(stale_dir)
    with open(os.path.join(stale_dir, "results.json"), "w") as f:
        json.dump({"episodes": {"ol9_e999_o0_s0.5": {"outcome": "collision"}}}, f)
    r = run_aggregate(sandbox, tmp, 4, "casestale")
    check("stale merge rejected", r.returncode == 2 and "stale pre-existing results" in r.stderr,
          f"rc={r.returncode} err={r.stderr[-300:]}")


def main():
    sandbox = tempfile.mkdtemp(prefix="eval_agg_test_")
    try:
        test_silent_failure_488_of_600(sandbox)
        test_complete_run_passes(sandbox)
        test_duplicate_key_rejected(sandbox)
        test_extra_artifact_rejected(sandbox)
        test_nonzero_exit_with_json_rejected(sandbox)
        test_missing_npz_rejected(sandbox)
        test_stale_results_merge_rejected(sandbox)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

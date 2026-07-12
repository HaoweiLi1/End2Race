# D1-b Local Startup Hang Handoff

Date: 2026-07-05

This document explains why the D1-b local runs were not used for KPI judgment and what should be checked in a fresh debugging chat.

## Executive Summary

The local machine did not produce a valid D1-b training run. Two local attempts started `train_ppo.py`, passed preflight, and then failed to emit even the first `iter=` training line.

This is a startup/first-rollout hang, not a PPO result.

Valid KPI data came only from remote runs:

- remote `seed1`: complete training + 3-grid eval + collision analysis
- remote `seed2r`: complete training + 3-grid eval + collision analysis

The local failed runs must not be counted as D1-b training outcomes.

## Machines

Local:

- hostname: `wsl2-ubuntu2204`
- Python: `/home/haowei/miniconda3/envs/end2race/bin/python`, version `3.10.20`
- script preflight `DISPLAY`: `:1`

Remote:

- hostname: `haowei-MSI`
- Python: `/home/haowei/miniconda3/envs/end2race/bin/python`, version `3.10.19`
- script preflight `DISPLAY`: `:1`

Important: this was not a "forgot DISPLAY" issue. The local D1-b pipeline preflight recorded `display=:1` for both failed local attempts.

## Code State

The relevant training files matched between local and remote for D1-b:

```text
train_ppo.py        bef8fff1f9535e9ed3767b1666a95cf1f533ba021b3cbebf94f78c9c466b31c4
ppo_utils.py        aff20febb5e630de03380941d5675be7cecc53f563b608ccb576e9610b3225d1
analyze_collisions.py ca8d0f9e4f8343acf7d0f87709c48c10bdd1ce1b646f3590a923fb521de66ad0
```

`evaluate_ol1.sh` differed because the remote copy contains `export DISPLAY=:1`. That difference does not explain the local training hang, because training does not call `evaluate_ol1.sh`.

## Failed Local Runs

### Local seed0

Log dir:

```text
logs/d1b_seed0_20260705_092447/
```

Command:

```bash
/home/haowei/miniconda3/envs/end2race/bin/python train_ppo.py \
  --total_iterations 300 \
  --gae_lambda 0.99 \
  --critic_lr 5e-4 \
  --time_gap 0.8 \
  --front_base_margin 0.9 \
  --freeze_speed \
  --lateral_offset_prob 0.5 \
  --lateral_offset_min 0.3 \
  --lateral_offset_max 0.8 \
  --train_seed 0 \
  --save_every 50 \
  --save_actor_path pretrained/end2race_ppo_d1b_seed0_20260705_092447.pth \
  --save_full_path pretrained/end2race_ppo_d1b_seed0_20260705_092447_full.pt
```

Status:

```text
09:24:52 train_d1b_seed0 START
09:33:22 train_d1b_seed0 FAIL 143
09:33:28 train_d1b_seed0 MANUAL_KILL 137
```

Gate log:

```text
09:25:22 iter=0 rows=0
...
09:33:22 iter=0 rows=0
```

Observation during run:

- `train_ppo.py` stayed at about 100% CPU.
- No `iter=00001` line was ever printed.
- `SIGINT` did not produce a Python traceback.
- The process had to be terminated to avoid indefinite resource burn.

### Local seed2

Log dir:

```text
logs/d1b_seed2_20260705_092447/
```

Same training command, except:

```text
--train_seed 2
--save_actor_path pretrained/end2race_ppo_d1b_seed2_20260705_092447.pth
--save_full_path pretrained/end2race_ppo_d1b_seed2_20260705_092447_full.pt
```

Status:

```text
09:37:45 train_d1b_seed2 START
09:42:48 train_d1b_seed2 STARTUP_HANG 21
```

Gate log:

```text
09:38:15 iter=0 rows=0
...
09:42:45 iter=0 rows=0
```

The startup guard terminated this run after 300 seconds without any `iter=` line.

Note: the first version of the pipeline script had a reporting bug for `startup_hang` mode (`ZeroDivisionError` when writing a no-eval report). That was fixed afterward. The reporting bug is separate from the training startup hang.

## Successful Remote Contrast

Remote seed1:

```text
logs/d1b_seed1_20260705_092447/
09:25:00 train START
09:53:31 train DONE
10:02:18 full pipeline COMPLETE
```

Remote seed2r:

```text
logs/d1b_seed2r_20260705_092447/
10:03:02 train START
10:31:33 train DONE
10:40:15 full pipeline COMPLETE
```

Remote seed2r used the same numeric seed as the failed local seed2 (`--train_seed 2`) and completed. This strongly suggests the local failure is environment/runtime specific rather than a deterministic seed-2 logic bug in the D1-b code.

## What Is Known

Known facts:

- Local training processes launched successfully.
- Preflight passed locally.
- The local process reached at least `f110_gym` initialization, because the RK4 warning appears in `train_300.log`.
- No first PPO iteration completed locally.
- No checkpoint was saved locally.
- No local eval was run.
- The remote machine completed equivalent D1-b jobs with `DISPLAY=:1`.

Known not to be the cause:

- Not missing D1-b code sync: hashes matched for `train_ppo.py`, `ppo_utils.py`, and `analyze_collisions.py`.
- Not missing `DISPLAY` in the pipeline: local preflight showed `display=:1`.
- Not a PPO KPI failure: no local PPO iteration completed.

## What Is Not Known

The true root cause is not proven yet.

No Python traceback was captured. `strace`/`gdb` attach was blocked by ptrace permissions during the original run, so the exact stack location is unknown.

The hang could be before, during, or just after the first rollout. Since `train_ppo.py` only logs after an iteration finishes, the absence of `iter=` does not distinguish:

- slow or stuck first environment rollout;
- stuck opponent lattice planner call;
- stuck lateral-offset reset / collision-check loop;
- WSL2-specific runtime issue in `f110_gym`, numpy, threading, map/collision code, or native libraries;
- GPU/CPU thread scheduling issue.

## Most Likely Hypothesis

The strongest current hypothesis is:

> On WSL2 local, the first D1-b rollout can enter a very long or stuck native/simulation/planner path before the first PPO iteration logs.

Why this hypothesis fits:

- Both failed local runs are on `wsl2-ubuntu2204`.
- The remote native Linux machine completed the same D1-b code path.
- The local process consumed 100% CPU instead of idling.
- The log stops after Gym/RK4 initialization, before first training summary.
- D1-b added lateral-offset reset and more off-raceline starting states, which can stress simulator/planner geometry paths that D1-a did not stress as much.

This is still a hypothesis, not a proven root cause.

## Minimal Reproduction Commands

Run these locally with explicit timeout so the machine does not burn indefinitely.

### D1-b reproduction

```bash
cd /home/haowei/Documents/End2Race
export DISPLAY=:1
timeout 420 /home/haowei/miniconda3/envs/end2race/bin/python train_ppo.py \
  --total_iterations 1 \
  --gae_lambda 0.99 \
  --critic_lr 5e-4 \
  --time_gap 0.8 \
  --front_base_margin 0.9 \
  --freeze_speed \
  --lateral_offset_prob 0.5 \
  --lateral_offset_min 0.3 \
  --lateral_offset_max 0.8 \
  --train_seed 0 \
  --save_actor_path /tmp/d1b_debug_seed0.pth \
  --save_full_path /tmp/d1b_debug_seed0_full.pt
```

Expected bad behavior if reproduced:

- no `iter=00001`;
- process runs at high CPU until `timeout` kills it.

### D1-a control

Same command with no lateral-offset curriculum:

```bash
cd /home/haowei/Documents/End2Race
export DISPLAY=:1
timeout 420 /home/haowei/miniconda3/envs/end2race/bin/python train_ppo.py \
  --total_iterations 1 \
  --gae_lambda 0.99 \
  --critic_lr 5e-4 \
  --time_gap 0.8 \
  --front_base_margin 0.9 \
  --freeze_speed \
  --lateral_offset_prob 0.0 \
  --train_seed 0 \
  --save_actor_path /tmp/d1a_control_seed0.pth \
  --save_full_path /tmp/d1a_control_seed0_full.pt
```

Interpretation:

- If D1-a control completes but D1-b hangs, focus on `_reset_sim_with_offset` and off-raceline rollout/planner interactions.
- If both hang, the local environment is broadly unsuitable for this training path and the issue is not D1-b-specific.

### Lateral-offset stress controls

Force every episode offset:

```bash
timeout 420 /home/haowei/miniconda3/envs/end2race/bin/python train_ppo.py \
  --total_iterations 1 \
  --gae_lambda 0.99 \
  --critic_lr 5e-4 \
  --time_gap 0.8 \
  --front_base_margin 0.9 \
  --freeze_speed \
  --lateral_offset_prob 1.0 \
  --lateral_offset_min 0.3 \
  --lateral_offset_max 0.8 \
  --train_seed 0 \
  --save_actor_path /tmp/d1b_debug_prob1_seed0.pth \
  --save_full_path /tmp/d1b_debug_prob1_seed0_full.pt
```

Narrow offset:

```bash
timeout 420 /home/haowei/miniconda3/envs/end2race/bin/python train_ppo.py \
  --total_iterations 1 \
  --gae_lambda 0.99 \
  --critic_lr 5e-4 \
  --time_gap 0.8 \
  --front_base_margin 0.9 \
  --freeze_speed \
  --lateral_offset_prob 1.0 \
  --lateral_offset_min 0.3 \
  --lateral_offset_max 0.3 \
  --train_seed 0 \
  --save_actor_path /tmp/d1b_debug_offset03_seed0.pth \
  --save_full_path /tmp/d1b_debug_offset03_full.pt
```

## Suggested Instrumentation

Add temporary debug logging in these places:

1. `End2RacePPOEnv.reset()`
   - print scenario fields: `ego_idx`, `opp_idx`, `interval_idx`, `opp_speedscale`
   - print whether lateral offset was attempted

2. `End2RacePPOEnv._reset_sim_with_offset()`
   - print each sampled offset and retry count
   - print `min(scan)` after reset
   - print fallback-to-native-reset event

3. `collect_rollout()`
   - print every 100 environment steps for the first iteration only
   - print episode termination/truncation reason

4. `End2RacePPOEnv._opponent_action()`
   - print when opponent planner `plan()` starts and returns
   - include `tracker_count`, opponent pose, and speed

5. Add Python faulthandler early in `train_ppo.py` during debug:

```python
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)
```

This may capture a Python stack if the process is not stuck inside native code.

## Recommended Debugging Order

1. Run D1-a control with `--lateral_offset_prob 0.0` for `--total_iterations 1`.
2. Run D1-b reproduction with `--lateral_offset_prob 0.5` for `--total_iterations 1`.
3. If only D1-b hangs, instrument `_reset_sim_with_offset()` and `_opponent_action()`.
4. If both hang, compare local vs remote package/runtime environment:
   - Python 3.10.20 local vs 3.10.19 remote
   - WSL2 local vs native Linux remote
   - `numpy`, `f110_gym`, `numba`, `torch`, OpenMP/MKL thread settings
5. Add `faulthandler.dump_traceback_later` before trying long runs again.

## Final Interpretation for the D1-b Experiment

The local failure is a runtime/execution failure before training iteration 1, not evidence about PPO quality.

For D1-b KPI, use only:

- `logs/d1b_seed1_20260705_092447/report.md`
- `logs/d1b_seed2r_20260705_092447/report.md`
- `logs/d1b_summary_20260705_092447.md`

Do not use:

- `logs/d1b_seed0_20260705_092447/`
- `logs/d1b_seed2_20260705_092447/`

except as local startup-hang diagnostics.

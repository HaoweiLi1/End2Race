# End2Race PPO V1.3-D: infrastructure-stable replication

**Status:** implementation/precheck in progress  
**Date:** 2026-07-17  
**Formal config:** `v1_3_d`  
**Formal seeds:** `20260735, 20260736, 20260737`  
**Only candidate checkpoint:** U8

## 1. Purpose

V1.3-D completes the product question left unanswered by V1.3-C. It is not a
new PPO hyperparameter arm.

V1.3-C removed the singular steering `atanh` likelihood and its first formal
seed produced six controlled updates:

```text
KL = 0.006702, 0.001966, 0.002458, 0.002440, 0.002573, 0.002905
```

The process then received SIGSEGV in PyTorch thread `pt_autograd_0`, inside
`libc10.so`. Kernel evidence showed no OOM, NVIDIA Xid, or memory pressure.
V1.3-C was correctly closed as `INVALID_INFRASTRUCTURE`; it has no U8 or
performance conclusion.

A nonformal recovery probe repeated the same seed with PyTorch autograd
multithreading disabled. U1-U6 were bitwise identical to the failed formal
run, and the probe then completed U7/U8:

```text
KL U7/U8 = 0.003178 / 0.002299
8-update max KL = 0.006702
target-window hits = 7/8
```

Therefore V1.3-D changes only execution scheduling:

```text
autograd_multithreading: true -> false
```

All RL mathematics and data remain V1.3-C.

## 2. Frozen RL configuration

```text
n_envs=16, n_steps=1600, batch_size=1600
n_epochs=1, updates=8, checkpoint_updates=(8,)
GRU LR=3e-6, head LR=3e-5, critic LR=3e-4
target_kl=0.010, update guardrail=0.020
steering_distribution=physical_gaussian
effective physical steering std=0.026 rad
speed std=0.15 m/s
critic=C0_RAW_SINGLE_FRAME
H0 current deterministic, p=0.50 with replacement
margin=0, sim duration=8 s
```

No reward, actor schema, critic, sampler, scenario, evaluator, or product gate
changes are allowed. The canonical actor remains 12 keys.

## 3. Why three fresh seeds

V1.3-D is a recovery replication, not an exploratory sweep. Three fresh seeds
are the minimum useful cross-seed test and were locked before any formal run:

```text
20260735
20260736
20260737
```

The failed formal seed `20260729` and all probe/smoke seeds are excluded from
formal gates. There is no retry or replacement seed.

## 4. Precheck

Before training:

1. assert V1.3-D differs from C only in `name` and
   `autograd_multithreading`;
2. assert all V1.3-C action-distribution identity/curvature tests still pass;
3. compile all touched code and run all available tests;
4. run a one-update `/tmp` smoke with seed `20260738`;
5. require finite metrics, KL `<=0.020`, `1..16` optimizer steps, nonzero
   GRU/head movement, zero frozen/log-std drift, and strict 12-key load;
6. lock hashes, implementation commit, seeds, gates, and paths in a clean
   preregistration commit.

## 5. Formal training

Run all three seeds serially, one GPU, exact order. For each:

```bash
python train_ppo.py \
  --config v1_3_d \
  --seed <seed> \
  --output_dir runs/ppo/v1_3_d_seed<seed>
```

Use shell `pipefail` and `runs/ppo/v1_3_d_logs/train_seed<seed>.log`.

Fail-fast for any KL `>0.020`, non-finite metric, native/runtime crash, frozen
drift, or checkpoint failure. No retry. A second infrastructure failure closes
the entire recovery line; no further recovery arm is authorized.

## 6. Process gate

Every seed must have:

```text
COMPLETED U1..U8
all KL <= 0.020
at least 6/8 KL in [0.002,0.010]
each actual optimizer step count in [1,16]
nonzero GRU/head delta every update
zero frozen actor and log_std drift
one strict-loadable 12-key U8 checkpoint
```

If any completed seed misses the target-window count, train all three but do
not evaluate products; verdict `FAIL_UPDATE_WINDOW_NOT_REACHED`.

## 7. Product evaluation

Only after 3/3 process pass:

1. freshly pair-evaluate canonical BC on Austin development 600 with 4 workers,
   ego index 0, noise 0, no render/trace, 8-second duration;
2. require exact BC `21 collision / 233 follow / 346 overtake / 0 error`;
3. evaluate the fixed U8 checkpoint for each seed once.

Every seed must satisfy:

```text
G = BC_fixed_collision - candidate_collision >= 5
candidate collision <= 16
candidate overtake >= 340
speed ratio to BC >= 0.99
distance ratio to BC >= 0.99
```

Success requires 3/3 product passes. No averaging, median override, best seed,
or alternate checkpoint.

## 8. Verdicts

```text
PASS_STABLE_PHYSICAL_GAUSSIAN_DEVELOPMENT
FAIL_KL_UNSTABLE
FAIL_UPDATE_WINDOW_NOT_REACHED
FAIL_NO_STABLE_IMPROVEMENT
STOP_PROTOCOL_DRIFT
INVALID_INFRASTRUCTURE
```

A pass is development-only and does not authorize holdout, promotion, or
modification of `posttrained/`.

## 9. Records

```text
ppo_experiments/v1_3_d/
  IMPLEMENTATION_GUIDE.md
  RECOVERY_PROBE.json
  PRECHECK.json
  PREREGISTRATION.json
  STATUS.json
  aggregate_results.py
  RESULTS.json
  FINAL_REPORT.md
```

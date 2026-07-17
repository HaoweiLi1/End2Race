# End2Race PPO V1.3-C: steering action-distribution repair

**Status:** implementation and precheck in progress  
**Date:** 2026-07-17  
**Formal config:** `v1_3_c`  
**Formal seeds:** `20260729, 20260730, 20260731, 20260732, 20260733`  
**Only candidate checkpoint:** U8 for every seed

## 1. Question

Can the existing one-epoch, 3x-actor-LR PPO update become controlled and
cross-seed useful after removing the singular steering likelihood geometry,
without changing the 12-key End2Race actor, reward, critic, scenarios, or
deterministic deployment behavior?

This is not another LR sweep. V1.3-C is V1.3-A with one implementation axis
changed: `steering_distribution=squashed_latent -> physical_gaussian`.

## 2. Evidence motivating the change

V1.3-A used one epoch and 3x actor LR. Its first formal seed stopped at U5:

```text
KL:    0.007839, 0.006706, 0.002609, 0.001792, 0.096712
steps: 1,        16,       16,       16,       7
```

V1.3-B used nominal LR and four epochs. Its first formal seed also stopped at
U5 with `approx_kl=0.119444`. Therefore neither increasing epochs nor raising
LR is a controlled way to strengthen the current update.

The current steering distribution maps the actor's physical raw mean through:

```text
latent_mean = atanh(raw_steer / 0.52)
```

Its derivative is:

```text
d latent_mean / d raw_steer = 1 / (0.52 * (1 - (raw_steer/0.52)^2))
```

It diverges at the evaluator steering boundary. At `raw_steer=0.519`, a raw
mean shift of about `2e-5 rad` already implies steering KL near `0.02` for the
frozen latent std `0.05`. For raw means outside the physical bound, the clamp
makes the steering policy gradient zero. The unchanged BC actor is known to
emit raw steering outside the evaluator bound on canonical hard-case states.

An implementation-only paired probe used V1.3-A seed `20260718` and changed
only the distribution. It completed all eight updates:

```text
KL:    0.010797, 0.005871, 0.003679, 0.003372,
       0.002141, 0.002147, 0.005171, 0.003001
steps: 1, 16, 16, 16, 16, 16, 14, 16
```

It achieved 7/8 updates in `[0.002,0.010]`, maximum KL `0.010797`, nonzero
GRU/head displacement, zero frozen/log-std drift, and a strict-loadable 12-key
U8 checkpoint. This probe is mechanism evidence only and is excluded from all
formal gates.

## 3. Frozen configuration

V1.3-C equals V1.3-A in every field except name and distribution mode:

```text
n_envs=16
n_steps=1600
batch_size=1600
n_epochs=1
updates=8
checkpoint_updates=(8,)
GRU LR=3e-6
head LR=3e-5
critic LR=3e-4
target_kl=0.010
update_kl_guardrail=0.020
critic=C0_RAW_SINGLE_FRAME
hard pool=H0 current deterministic
hard probability=0.50 with replacement
margin=0
latent steering std field=0.05
effective physical steering std=0.52*0.05=0.026 rad
speed physical std=0.15 m/s
sim duration=8.0 s
```

The physical Gaussian emits an unbounded steering sample. SB3 stores that
unclipped sample and its exact likelihood in the rollout buffer, while the
existing Box action-space clip sends only `[-0.52,0.52]` to the simulator.
Deterministic raw actor output is clipped exactly as in the evaluator.

## 4. Hard invariants

- The actor checkpoint remains the original 12-key `End2Race` schema.
- BC weights and all frozen actor tensors remain bitwise unchanged unless they
  are one of the existing trainable GRU/output-layer tensors.
- `log_std` remains frozen.
- Reward, critic, environment, scenarios, hard pool, evaluator, and deployment
  checkpoint loader are unchanged.
- No margin arm is run: V1.3-M inherits V1.3-B's already-failed four-epoch
  intensity and its own guide requires it to stop.
- No retries, replacement seeds, extra LR arms, checkpoint lottery, or holdout.

## 5. Precheck

Before formal training:

1. compile `train_ppo.py`, `ppo`, the aggregator, and tests;
2. run all available PPO tests;
3. assert V1.3-C differs from V1.3-A only in `name` and
   `steering_distribution`;
4. assert deterministic action identity after the existing Box clip;
5. assert unchanged-parameter likelihood replay ratio is one;
6. run one nonformal one-update smoke with seed `20260734` in `/tmp`;
7. verify finite metrics, `1..16` optimizer steps, KL `<=0.020`, nonzero
   GRU/head displacement, zero frozen/log-std drift, and strict 12-key load;
8. record all hashes in `PRECHECK.json` and lock `PREREGISTRATION.json` in a
   clean commit before any formal directory exists.

## 6. Formal training protocol

Run the five seeds serially on one GPU in this exact order:

```text
20260729
20260730
20260731
20260732
20260733
```

Each command uses:

```bash
python train_ppo.py \
  --config v1_3_c \
  --seed <seed> \
  --output_dir runs/ppo/v1_3_c_seed<seed>
```

Use shell `pipefail` and log to `runs/ppo/v1_3_c_logs/train_seed<seed>.log`.
Do not evaluate candidates until all five training runs reach their registered
terminal state.

The only online stop is an existing hard failure: update-level
`approx_kl>0.020`, non-finite data, frozen-state drift, runtime failure, or
checkpoint schema failure. Any such failure stops the whole line; later seeds
are not started.

## 7. Process gate

Every seed must satisfy all of:

```text
status=COMPLETED at U8
exactly eight metric rows U1..U8
all update approx_kl <= 0.020
at least 6/8 updates have approx_kl in [0.002,0.010]
actual optimizer steps per update in [1,16]
GRU/head update displacement > 0
frozen actor drift = 0
log_std drift = 0
exactly one strict-loadable 12-key U8 checkpoint
```

If all training completes but any seed misses the KL-window count, the verdict
is `FAIL_UPDATE_WINDOW_NOT_REACHED` and no development evaluation is consumed.

## 8. Paired development evaluation

Only after 5/5 process pass:

1. copy the canonical BC to
   `runs/ppo/v1_3_c_baseline/end2race_bc_v1_3_c.pth`;
2. evaluate it on Austin canonical development 600 with 4 workers,
   `EGO_IDX_OFFSET=0`, noise 0, no render, no trace, 8-second duration;
3. require exact BC counts `21 collision / 233 follow / 346 overtake / 0 error`;
4. evaluate each fixed U8 candidate once under the identical contract.

Per seed, define `G = fixed_BC_collision - candidate_collision`. Every one of
the five candidates must meet:

```text
G >= 5
collision <= 16
overtake >= 340
mean speed / BC mean speed >= 0.99
mean distance / BC mean distance >= 0.99
```

The cross-seed gate is 5/5 process pass and 5/5 product pass. Means, medians,
or a best seed cannot override a failed seed.

## 9. Allowed verdicts

```text
PASS_STABLE_PHYSICAL_GAUSSIAN_DEVELOPMENT
FAIL_KL_UNSTABLE
FAIL_UPDATE_WINDOW_NOT_REACHED
FAIL_NO_STABLE_IMPROVEMENT
STOP_PROTOCOL_DRIFT
INVALID_INFRASTRUCTURE
```

A pass is development evidence only. It does not authorize holdout use,
promotion, or replacement of `posttrained/`.

## 10. Records

```text
ppo_experiments/v1_3_c/
  IMPLEMENTATION_GUIDE.md
  MECHANISM_PROBE.json
  PRECHECK.json
  PREREGISTRATION.json
  STATUS.json
  aggregate_results.py
  RESULTS.json             # terminal
  FINAL_REPORT.md          # terminal
```

Raw runs stay under `runs/ppo/` and are not committed. Result records contain
their exact paths and SHA-256 hashes.

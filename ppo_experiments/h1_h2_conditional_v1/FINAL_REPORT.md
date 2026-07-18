# H1/H2 Conditional Exploration Final Report

## Registered outcome

- Overall status: `FORWARD_SIGNAL`.
- Forward config: `N1-H1F-p50`.
- H1: full-pool 50% arm selected; product direction passed 2/3 seeds; short U1-to-U2 retention did not pass.
- H2: matched pool stopped at the registered size gate (primary 1, fallback 2, required 24); no H2 arm, seed, checkpoint, or paired-training telemetry was created.
- This result is an experiment forward signal, not a deployment or held-out-performance claim.

## Preflight and baseline

- Starting repository reference: `eb2ecef661e63dcf0a12fb7e7a9ffa8caa782ce3`.
- Formal preflight commit: `245e87f27cdd3325742263c4054e3192a1a224a1`.
- Support validation: `PASS`; strict BC checkpoint load: `True`.
- Current CPU full-600 BC: collision 22, follow 233, overtake 345, error 0, 600 unique scenarios.
- Formal evaluation contract: CPU, ego collision scope, 8 persistent workers, one Torch thread per worker.

## Post-hoc A1 diagnostic

The existing A1 update-2 checkpoint was evaluated once and was not eligible for formal selection.

| Seed | Update | Collision | Follow | Overtake | Fixed | New | deltaC | Repair rate [Wilson 95%] | Damage rate [Wilson 95%] |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 20260718 | 2 | 23 | 231 | 346 | 7 | 8 | 1 | 31.818% [16.361%, 52.681%] | 1.384% [0.703%, 2.707%] |

## H1 U1 screen

| Arm | Seed | Update | Collision | Follow | Overtake | Fixed | New | deltaC | Repair rate [Wilson 95%] | Damage rate [Wilson 95%] | Legal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| N1-H1F-p50 | 20260719 | U1 | 18 | 237 | 345 | 8 | 4 | -4 | 36.364% [19.733%, 57.048%] | 0.692% [0.269%, 1.766%] | PASS |
| N1-H1F-p25 | 20260719 | U1 | 24 | 233 | 343 | 7 | 9 | 2 | 31.818% [16.361%, 52.681%] | 1.557% [0.821%, 2.933%] | PASS |
| N1-H1E-p50 | 20260719 | U1 | 24 | 233 | 343 | 6 | 8 | 2 | 27.273% [13.151%, 48.152%] | 1.384% [0.703%, 2.707%] | PASS |
| N1-H1E-p25 | 20260719 | U1 | 27 | 230 | 343 | 3 | 8 | 5 | 13.636% [4.749%, 33.335%] | 1.384% [0.703%, 2.707%] | FAIL |

Full winner: `N1-H1F-p50`. Early winner: `N1-H1E-p50`. The full-pool 25% ratio was not supported; the early-pool ratio comparison was inconclusive.

## H1 U2 retention

| Arm | Seed | Update | Collision | Follow | Overtake | Fixed | New | deltaC | Repair rate [Wilson 95%] | Damage rate [Wilson 95%] | Legal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| N1-H1F-p50 | 20260719 | U2 | 19 | 237 | 344 | 10 | 7 | -3 | 45.455% [26.920%, 65.340%] | 1.211% [0.588%, 2.479%] | PASS |
| N1-H1E-p50 | 20260719 | U2 | 24 | 232 | 344 | 5 | 7 | 2 | 22.727% [10.123%, 43.440%] | 1.211% [0.588%, 2.479%] | PASS |

Pool preference: `H1_FULL_PREFERRED`. Selected config: `N1-H1F-p50`. Selected-arm short retention: `False`.

## H1 repeatability

| Arm | Seed | Update | Collision | Follow | Overtake | Fixed | New | deltaC | Repair rate [Wilson 95%] | Damage rate [Wilson 95%] | Legal | Product gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1-H1F-p50 | 20260719 | U2 | 19 | 237 | 344 | 10 | 7 | -3 | 45.455% [26.920%, 65.340%] | 1.211% [0.588%, 2.479%] | PASS | True |
| N1-H1F-p50 | 20260720 | U2 | 19 | 234 | 347 | 6 | 3 | -3 | 27.273% [13.151%, 48.152%] | 0.519% [0.177%, 1.515%] | PASS | True |
| N1-H1F-p50 | 20260721 | U2 | 25 | 232 | 343 | 3 | 6 | 3 | 13.636% [4.749%, 33.335%] | 1.038% [0.477%, 2.246%] | FAIL | False |

Registered repeat result: `True` (2/3 product-pass seeds, median collision 19, median deltaC -3).

## H2 matched-pool reconstruction

- Source interval-8 bases: 199.
- Preflight valid/invalid: I7 199/0; I8 199/0.
- Deterministic 8-second safe: I7 122; I8 83.
- Stochastic trial-count distribution: I7 {0: 77, 4: 111, 8: 11}; I8 {0: 116, 4: 58, 8: 25}.
- Collision K=0..8 distribution: I7 {'0': 185, '1': 2, '2': 3, '3': 2, '4': 6, '5': 1, '6': 0, '7': 0, '8': 0}; I8 {'0': 173, '1': 2, '2': 6, '3': 6, '4': 6, '5': 4, '6': 0, '7': 2, '8': 0}.
- Collision-time distribution (seconds): I7 {'count': 43, 'max': 3.9599999999999596, 'mean': 2.661860465116263, 'median': 2.669999999999987, 'min': 1.0000000000000007}; I8 {'count': 90, 'max': 3.979999999999959, 'mean': 2.275333333333321, 'median': 2.1449999999999982, 'min': 0.9000000000000006}.
- Primary matched: 1; fallback matched: 2; selected: 0.
- Selected tier/status: `STOP_H2_MATCHED_POOL_TOO_SMALL` / `STOP_H2_MATCHED_POOL_TOO_SMALL`.
- Matched manifest SHA-256: `b2c1e4b2ee0fc3d84e6f03685809a01b12b5ef5c4b29d4489d613fdc045c804d`.

## H2 conditional gate and training stages

| Stage | Status | Arms | Checkpoints | Evaluations |
|---|---|---:|---:|---:|
| Conditional exploration gate | NOT_RUN_H2_MATCHED_POOL_TOO_SMALL | 0 | 0 | 0 |
| Screen | NOT_RUN_H2_MATCHED_POOL_TOO_SMALL | 0 | 0 | 0 |
| Selection | NOT_RUN_H2_MATCHED_POOL_TOO_SMALL | 0 | 0 | 0 |
| Retention | NOT_RUN_H2_MATCHED_POOL_TOO_SMALL | 0 | 0 | 0 |
| Repeatability | NOT_RUN_H2_MATCHED_POOL_TOO_SMALL | 0 | 0 | 0 |

No discordant-pair mechanism table can be computed because the registered matched-pool size gate selected zero bases. The eight-trial classification rows were retained; unmatched I7/I8 pools were not used.

## Training stability diagnostics

| Run | Updates | Max KL | Max clip fraction | Frozen actor max delta | log_std max delta | Optimizer steps |
|---|---:|---:|---:|---:|---:|---|
| N1-H1E-p25_seed20260719 | 1 | 0.001801539 | 0.095976561 | 0.0 | 0.0 | U1 4/4 |
| N1-H1E-p50_seed20260719 | 2 | 0.001410168 | 0.068906249 | 0.0 | 0.0 | U1 4/4, U2 4/4 |
| N1-H1F-p25_seed20260719 | 1 | 0.001852492 | 0.101718746 | 0.0 | 0.0 | U1 4/4 |
| N1-H1F-p50_seed20260719 | 2 | 0.001364535 | 0.061796873 | 0.0 | 0.0 | U1 4/4, U2 4/4 |
| N1-H1F-p50_seed20260720 | 2 | 0.001895908 | 0.096367185 | 0.0 | 0.0 | U1 4/4, U2 4/4 |
| N1-H1F-p50_seed20260721 | 2 | 0.001879443 | 0.103320311 | 0.0 | 0.0 | U1 4/4, U2 4/4 |

Across 10 recorded H1 updates, max KL was 0.001895908, max clip fraction was 0.103320311, frozen-actor delta was zero, log_std delta was zero, and every optimizer-step count matched the plan.

## Source and manifest hashes

### Starting reference hashes

| Path | SHA-256 |
|---|---|
| `eval_multiagent.py` | `5ca6329513fd46b2216b84a2f7a154e470a83c674670628d94373e9cca891580` |
| `evaluate.sh` | `be774b398d63725a4ac9329816c79c68f673e3d60f11793b53cfff5fcfceee4a` |
| `ppo/config.py` | `7464a0260498705242189e09528c6b1925522911ddd5e5a4fae34e154921f295` |
| `ppo/environment.py` | `49c484064d4a4c1a95e25128420ed419e73ac36acc6d0661a64bd82e6085bb0c` |
| `ppo/policy.py` | `7b14ccb60ce7494c611cab2df729df562895137af4694aa805c5a57e8e4f64d8` |
| `ppo/reward.py` | `bd895a131e1eb04a187a29781082ce729628912446fcc5b796be6982a0570aaf` |
| `ppo/scenarios.py` | `f3103d1cb3ba4601db46ab419e6776c535183182f2b8fec2ff72136746dc56d1` |
| `ppo_experiments/quick_pool_3s_v2/FULL600_BC_RESULTS.json` | `dd690ac2241ff5269ef1ed4df6f428fdb931b740e7cc95dbdfb5eaa224da89f9` |
| `pretrained/end2race.pth` | `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4` |
| `train_ppo.py` | `817bdac4e9e4dfb43c813942fe688c5dfffd30eacf1a9c2d25dcb8402d55a5c2` |
| `utils.py` | `77459621bf6af7fa8726ba4182fe28f3c4f4c5ebabf0f876553534dfcb95edfe` |

### Post-implementation frozen hashes

| Path | SHA-256 |
|---|---|
| `eval_multiagent.py` | `5ca6329513fd46b2216b84a2f7a154e470a83c674670628d94373e9cca891580` |
| `evaluate.sh` | `be774b398d63725a4ac9329816c79c68f673e3d60f11793b53cfff5fcfceee4a` |
| `ppo/config.py` | `572216c14d43ef29640cafa7fff4f1121f155d2bc11bf93018dfeb2cd453dd4d` |
| `ppo/environment.py` | `25f6f289455c86197b657945b099c220f11b1f56a47a652692a979b9aeccb742` |
| `ppo/policy.py` | `7b14ccb60ce7494c611cab2df729df562895137af4694aa805c5a57e8e4f64d8` |
| `ppo/reward.py` | `bd895a131e1eb04a187a29781082ce729628912446fcc5b796be6982a0570aaf` |
| `ppo/scenarios.py` | `6bdfd480be4cae8984204567da33d41130d9b73849c2378a35a75304880a5a42` |
| `pretrained/end2race.pth` | `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4` |
| `train_ppo.py` | `080a797f1300ae90cd51c12d0429efc61eed0fb4c62f117e348d33dc3f6eaaca` |
| `utils.py` | `77459621bf6af7fa8726ba4182fe28f3c4f4c5ebabf0f876553534dfcb95edfe` |

H2 source manifest SHA-256: `97b20441984a2305ebbd5b0a2771786029d83b39474c94dbeef9db4d64110a8b`. H2 generated base manifest SHA-256: `56f235c150d712bf0aa8d7313a65cfbfbd37475b6c60153c7acbea361c2f83ba`. H2 selected matched manifest SHA-256: `b2c1e4b2ee0fc3d84e6f03685809a01b12b5ef5c4b29d4489d613fdc045c804d`.

All 10 H1 checkpoint paths, hashes, local-presence checks, hash verification results, and evaluation status are in `GLOBAL_CHECKPOINTS.tsv`. Checkpoint binaries remain local.

## Exact recorded commands

1. Evaluate existing A1 H1-full update-2 checkpoint once on current CPU full-600

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A1_H1FULL_8S_seed20260718/checkpoints/end2race_ppo_QP3_A1_H1FULL_8S_u0002_s20260718.pth --output /home/haowei/Documents/End2Race/ppo_experiments/h1_h2_conditional_v1/posthoc_a1_u2_full600_raw.json --workers 8 --sim-duration 8.0
```

2. Train N1-H1F-p50 seed 20260719 through U1 and pause

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/train_ppo.py --config N1-H1F-p50 --seed 20260719 --output_dir /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1F-p50_seed20260719 --screen-pause
```

3. Train N1-H1F-p25 seed 20260719 through U1 and pause

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/train_ppo.py --config N1-H1F-p25 --seed 20260719 --output_dir /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1F-p25_seed20260719 --screen-pause
```

4. Train N1-H1E-p50 seed 20260719 through U1 and pause

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/train_ppo.py --config N1-H1E-p50 --seed 20260719 --output_dir /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1E-p50_seed20260719 --screen-pause
```

5. Train N1-H1E-p25 seed 20260719 through U1 and pause

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/train_ppo.py --config N1-H1E-p25 --seed 20260719 --output_dir /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1E-p25_seed20260719 --screen-pause
```

6. Evaluate N1-H1F-p50 seed 20260719 U1 on current CPU full-600

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1F-p50_seed20260719/checkpoints/end2race_ppo_N1-H1F-p50_u0001_s20260719.pth --output /home/haowei/Documents/End2Race/ppo_experiments/h1_h2_conditional_v1/h1_u1_eval_N1-H1F-p50_seed20260719.json --workers 8 --sim-duration 8.0
```

7. Evaluate N1-H1F-p25 seed 20260719 U1 on current CPU full-600

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1F-p25_seed20260719/checkpoints/end2race_ppo_N1-H1F-p25_u0001_s20260719.pth --output /home/haowei/Documents/End2Race/ppo_experiments/h1_h2_conditional_v1/h1_u1_eval_N1-H1F-p25_seed20260719.json --workers 8 --sim-duration 8.0
```

8. Evaluate N1-H1E-p50 seed 20260719 U1 on current CPU full-600

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1E-p50_seed20260719/checkpoints/end2race_ppo_N1-H1E-p50_u0001_s20260719.pth --output /home/haowei/Documents/End2Race/ppo_experiments/h1_h2_conditional_v1/h1_u1_eval_N1-H1E-p50_seed20260719.json --workers 8 --sim-duration 8.0
```

9. Evaluate N1-H1E-p25 seed 20260719 U1 on current CPU full-600

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1E-p25_seed20260719/checkpoints/end2race_ppo_N1-H1E-p25_u0001_s20260719.pth --output /home/haowei/Documents/End2Race/ppo_experiments/h1_h2_conditional_v1/h1_u1_eval_N1-H1E-p25_seed20260719.json --workers 8 --sim-duration 8.0
```

10. Evaluate N1-H1F-p50 seed 20260719 U2 on current CPU full-600

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1F-p50_seed20260719/checkpoints/end2race_ppo_N1-H1F-p50_u0002_s20260719.pth --output /home/haowei/Documents/End2Race/ppo_experiments/h1_h2_conditional_v1/h1_u2_eval_N1-H1F-p50_seed20260719.json --workers 8 --sim-duration 8.0
```

11. Evaluate N1-H1E-p50 seed 20260719 U2 on current CPU full-600

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1E-p50_seed20260719/checkpoints/end2race_ppo_N1-H1E-p50_u0002_s20260719.pth --output /home/haowei/Documents/End2Race/ppo_experiments/h1_h2_conditional_v1/h1_u2_eval_N1-H1E-p50_seed20260719.json --workers 8 --sim-duration 8.0
```

12. Train N1-H1F-p50 seed 20260720 through U2

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/train_ppo.py --config N1-H1F-p50 --seed 20260720 --output_dir /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1F-p50_seed20260720
```

13. Train N1-H1F-p50 seed 20260721 through U2

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/train_ppo.py --config N1-H1F-p50 --seed 20260721 --output_dir /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1F-p50_seed20260721
```

14. Evaluate N1-H1F-p50 seed 20260720 U2 on current CPU full-600

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1F-p50_seed20260720/checkpoints/end2race_ppo_N1-H1F-p50_u0002_s20260720.pth --output /home/haowei/Documents/End2Race/ppo_experiments/h1_h2_conditional_v1/h1_u2_eval_N1-H1F-p50_seed20260720.json --workers 8 --sim-duration 8.0
```

15. Evaluate N1-H1F-p50 seed 20260721 U2 on current CPU full-600

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/h1_h2_conditional_v1/N1-H1F-p50_seed20260721/checkpoints/end2race_ppo_N1-H1F-p50_u0002_s20260721.pth --output /home/haowei/Documents/End2Race/ppo_experiments/h1_h2_conditional_v1/h1_u2_eval_N1-H1F-p50_seed20260721.json --workers 8 --sim-duration 8.0
```

16. Build and classify the matched H2 interval-7/interval-8 contrast pool

```sh
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/h1_h2_conditional_v1/build_h2_pool.py
```

## Completion inventory

All registered stage outputs exist. Every started H1 arm reached its registered terminal update; all eight formal H1 evaluations and the post-hoc diagnostic have 600 unique rows, zero errors, and finite observations/actions. H2 reached its registered pool-size kill gate without starting training. Final process and Git-worktree checks are performed after committing this report.

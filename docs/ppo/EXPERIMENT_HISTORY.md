# End2Race PPO Experiment History

Updated: 2026-07-12. This ledger summarizes substantive results while keeping
the original reports and canonical artifacts as evidence.

## 1. Historical PPO and evaluation repair

Early PPO experiments produced only small or unstable changes. The important
evaluation repair was P0: worker exit codes no longer encode outcomes;
outcomes and per-agent collisions are read from strict per-episode JSON/NPZ
records. The 488/600 silent-completeness failure is covered by regression
tests. Offset-grid duplication was also corrected.

P1 evaluated BC and three historical candidates over 16,800 episodes. The
owner designated `cand160` as the deployed baseline because it was equivalent
on Austin and transfer-favorable. Canonical D0.1 later showed that it did not
meet the new B+ target: primary collision RR was about 0.906 overall, and its
Austin overtake count was below BC.

Canonical checkpoint set retained locally:

| role | file | SHA256 |
|---|---|---|
| BC | `pretrained/end2race.pth` | `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4` |
| cand160 | `pretrained/end2race_ppo_full_disc_r8192_seed1_20260709_210827_iter0160.pth` | `77cd79904f0f57c1e7a4914dd0b52384628dce225f9222e4e2274e0eda3b5aa6` |
| cand120 | `pretrained/end2race_ppo_full_disc_r8192_seed1_20260709_210827_iter0120.pth` | `9f2f47bf46363946ba29c1fe5fcada3a3d5fe514ece6eb160c03b25d8f82b3b3` |
| cand040 | `pretrained/end2race_ppo_full_disc_r8192_seed0_20260709_210827_iter0040.pth` | `c7a72f5564a191e103d319a7f66167e6969fb3528534b90bafba77ceb598d7e1` |

## 2. D0.1 canonical audit

The canonical D0.1 release regenerated the complete 16,800-occurrence
analysis and corrected inference/accounting boundaries. Primary N=3,036:

| model | collisions | overtakes |
|---|---:|---:|
| BC | 170 | 1,792 |
| cand160 | 154 | 1,799 |
| cand120 | 168 | 1,797 |
| cand040 | 166 | 1,787 |

For cand160, clustered all-pool RR was 0.906 with two-sided 95% CI
approximately `[0.818, 0.994]`. It is favorable historical evidence but not
the required `RR <= 0.70` result. The canonical release is
`logs/ppo_next_unattended_20260710_230212/artifacts/d01_full_reconcile_20260711_170200_a`.

## 3. D2 representation probe

D2 replayed the non-test BC population and evaluated four grouped-OOF probe
families. All failed the original complete gate, principally the frozen
`TTC<2 MAE <= 0.30s` requirement. Temporal T2 nevertheless passed its
collision-classification sub-gates. The grouped test was never opened.

Key OOF results:

| family | 1s recall | 1s safe FA | Brier skill | TTC<2 MAE |
|---|---:|---:|---:|---:|
| frozen temporal T1 | 0.648 | 0.075 | 0.078 | 1.073s |
| raw-history T2 | 0.725 | 0.095 | 0.105 | 1.068s |
| D2R-G | 0.868 | 0.099 | 0.130 | 0.800s |

The monotone improvement supports useful risk information in deployable raw
observations and a limitation of the tested frozen-feature route. It does not
isolate freezing causally because architecture and supervision also changed.

## 4. D2.5 recoverability

D2.5 performed no learning. On 91 non-test BC ego-collision episodes it
searched a fixed bounded library of brake/steer residuals. It found 67
confirmed-safe-pass witnesses and exhausted the valid library for 24 cases.
No witness required clipping or positive speed residual.

Correct claim: the library recovered 67/91 tested cases. Incorrect claims:
“74% theoretical ceiling”, whole-population recovery rate, or expected PPO
collision RR.

## 5. D2R-G and project-owner TTC override

D2R-G passed the 1s recall, 1s false-alarm, and Brier-skill requirements but
failed 2s false alarm (`0.1026 > 0.10`) and TTC MAE (`0.800s > 0.300s`). Its
immutable result remains FAILED.

The owner then made a prospective decision: TTC is diagnostic-only for the
policy phase, while the old D2/D2R results remain failed under their original
rules. This did not rewrite any result.

## 6. B+ v2.2 supervised warm-start

Three arms shared a frozen BC driving backbone:

- A: adapter over frozen BC features;
- B: frozen pretrained risk sidecar;
- C: the same sidecar initialization, fine-tuned at a smaller learning rate.

Fresh zero residual reproduced BC bitwise. A supervised witness warm-start
then learned a high intervention prior. Its step-level diagnostic passed, but
Task 10 exposed the deployment failure on 288 development scenarios:

| arm | collisions | fixed/new collisions | gained/lost overtakes | episodes braked |
|---|---:|---:|---:|---:|
| A | 91 | 11 / 78 | 26 / 31 | 206/288 |
| B | 54 | 14 / 44 | 15 / 71 | 287/288 |
| C | 67 | 13 / 56 | 28 / 54 | 287/288 |

BC had 24 collisions and 138 overtakes on this mechanism population. Every
arm created more collisions than it fixed and lost more overtakes than it
gained. Steering was always active, and physical action composition clipped
after BC moved within a macro. Task 10 therefore FAILED and no PPO began.

The measured D2R safe-episode alarm rate (9.87%) closely matched Task-10's
26/264 = 9.85% alarm exposure, confirming that false interventions were a
real overtake risk.

## 7. Hierarchical action remediation

The replacement architecture introduced an explicit intervention gate,
gated both steering and braking, held a latent macro action, and recomputed a
bound-preserving physical residual at each 100 Hz micro-step. Fresh identity
passed 16/16.

Replacement Task 6 then used 542 fit episodes / 43,902 macros and a separate
fold-4 calibration set with nine positive witness episodes and 75 negative
confirmed-safe-pass episodes. The release was internally valid but failed all
positive/type acceptance bars:

- A/B/C positive episode recall: `0/9`;
- positive macro recall: `0/39`;
- steer-only episode recall: `0/4`;
- brake-containing episode recall: `0/5`;
- conditional brake specificity: `0/14`.

The natural schedule yielded only 1,502 intervention occurrences in 262,144
draws; 231/1,024 batches contained none. This supports an exposure/optimization
failure but does not rank sidecars or establish generalization.

Canonical release:
`logs/bplus_v22_d3r2_20260711/artifacts/hierarchical_warmstart_20260712_160212`,
output-manifest SHA256
`ffd3c59cbbfe39931930d88e5f7d2781b5c44174ab4a635e30bd490665c2a0d9`.

## 8. What has and has not been learned

Established:

- collision risk is partly observable from deployable inputs;
- bounded corrective actions can recover confirmed passes in many observed
  cases;
- fixed alarm-to-brake and uncalibrated witness imitation damage overtaking;
- action/log-probability consistency and bound-preserving composition are
  mandatory before PPO.

Not established:

- no current result demonstrates `RR <= 0.70` with overtake noninferiority;
- no Task-6/Task-10 result is policy-generalization evidence;
- no A/B/C arm has been selected;
- no v2.2 PPO iteration has run;
- the sealed D2 test and fresh/final pool have not been opened.


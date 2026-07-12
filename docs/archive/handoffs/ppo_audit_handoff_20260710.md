# SUPERSEDED ENTRY POINT — READ ../CURRENT_HANDOFF.md FIRST

This document remains the full audit and experiment ledger, but its original
work order predates P1 completion, the invalid first D0 estimand, the closed-
track outcome defect, and the D0.1 v2.1 plan. Start a new chat from the
repository-root CURRENT_HANDOFF.md and use this file only for detailed history.

# End2Race PPO Audit Handoff

Generated: 2026-07-10 (Asia/Singapore)  
Repository: `/home/haowei/Documents/End2Race`  
Remote: `haowei@100.95.251.103:~/Documents/End2Race`  
Audience: a new coding agent continuing the work, and external reviewers auditing the code and experiments.

This was the authoritative audit handoff at generation time. It remains the
detailed ledger, but `../CURRENT_HANDOFF.md` now governs whenever conclusions
or work orders conflict.

## 0. How To Use This Handoff

For a new chat or coding agent, use this opening instruction:

> Read `logs/ppo_audit_handoff_20260710.md` first. Verify the live local and remote state before acting. Treat Section 6 as confirmed audit findings, Section 7 as the experiment ledger, Section 8 as the conclusion-confidence boundary, and Section 10 as the required work order. Do not launch another long training sweep before repairing evaluation integrity and independently validating the three existing Austin candidates.

For an external reviewer:

- audit claims against the named source files and experiment paths rather than older conversational summaries;
- distinguish direct code verification, descriptive point estimates, paired development-set evidence, and untouched-test evidence;
- challenge any causal claim that relies on a run where multiple variables changed together;
- report disagreements against the numbered finding or experiment subsection so they can be resolved precisely.

## 1. Project Goal

End2Race starts from a recurrent behavior-cloning (BC) policy. PPO post-RL fine-tuning is intended to:

1. reduce collision rate relative to BC (primary KPI);
2. preserve or improve overtake rate (secondary KPI);
3. keep the deployed actor observation unchanged: 360-beam LiDAR, previous ego speed, and recurrent hidden state;
4. train only on Austin, while retaining useful behavior on unseen maps;
5. remain theoretically defensible and simple enough to explain in a paper.

Smoothness is not a primary optimization target. The original motivating failure was unsafe overtaking/merging, but the experiments later showed that most measured failures are pre-overtake or alongside contacts rather than post-overtake collisions.

## 2. Current Executive Status

- The core PPO implementation runs and the main recurrent/PPO data flow is connected correctly.
- The D2 residual actor is currently the main architecture:
  - frozen BC backbone;
  - zero-initialized residual head;
  - bounded steering residual;
  - asymmetric speed residual, usually `[-1.0, 0.0] m/s` in the latest runs;
  - separate privileged critic used only during training.
- Branch B is the best checkpoint with completed multi-map evaluation. It is broadly BC-equivalent with small favorable point differences, not statistically proven superior.
- The latest Austin full-distribution sweep found stronger Austin development-set candidates, but those candidates have not passed independent holdout or cross-map validation.
- The full sweep is stopped, not running. It failed at the start of `full_cont_r8192 seed0` because the shared Numba cache was corrupted.
- The evaluation pipeline has a confirmed exit-code bug that can silently count failed workers as `following`. This must be fixed before trusting future unattended evaluation.
- No active `train_ppo.py`, `eval_multiagent.py`, or full-sweep process was present at the final audit check.

## 3. Source State

### 3.1 Local checkout

At the audit point:

```text
branch: main
HEAD: d22b974fe29fa772d2cacff9d89bba36b761722b

 M ppo_utils.py
 M train_ppo.py
?? analyze_paired_eval.py
```

The uncommitted changes add the latest full-distribution sampler features:

- multiple ego/opponent raceline choices;
- discrete opponent speedscale choices;
- hard-start index sampling;
- canonical-like scenario support and validation.

Do not revert these changes. They were used by the latest full-distribution experiments.

### 3.2 Remote checkout

The remote core files have the same content as local, but the remote Git state is older:

```text
remote HEAD: bae93dd7ea5fd002a2a28f093baded7ed0b1d19d
```

The remote carries D2/D4 and sampler development as uncommitted/untracked files. This is a reproducibility risk: matching file content exists now, but Git history alone cannot reconstruct the remote experiment checkout.

### 3.3 Core-file hashes verified equal locally and remotely

```text
train_ppo.py          224f1a5d974a43001fcfb4d87faffc61b328a9fa02834f0d56854265a20cb6b6
ppo_utils.py           94a65a91b74e72f04564ecd12f5f560b40638a5388f0d3d81abdb0b950461ad7
model.py               ef8281a9fc2bf584f9fcb210502bd8baf787cfb7d8dbcb557f26a4e1be3d221c
eval_multiagent.py     52586aacb61ac6026f124339b797f56a33fd74233b310b2260a78e0918c4ca09
analyze_paired_eval.py 9aac05772bead1f1093cf24fd6fa4a4800c3d97a3d70b5dfc941d317ba968d4b
```

The latest full-sweep archive stored `git_head.txt`, `git_status.txt`, and a partial source diff, but the diff command omitted `model.py` and cannot include untracked files. The run archive is therefore not self-contained.

## 4. Current PPO Pipeline

### 4.1 Scenario and environment

Entry point: `train_ppo.py:main()`.

For each episode:

```text
sample scenario
-> place ego/opponent
-> reset lattice-planner opponent
-> reset F110 two-agent simulator
-> actor observes LiDAR + previous ego speed
-> privileged critic observes simulator geometry
```

The latest stage sampler can select:

- ego raceline choices;
- opponent raceline choices;
- uniform or discrete opponent speedscale;
- interval range;
- uniform start or hard-start-location sampling.

The completed latest run used:

```text
map                    Austin
ego raceline            raceline1
opponent racelines      raceline0/raceline1/raceline2
opponent speedscale     {0.5, 0.6, 0.7, 0.8}
interval_idx            integer 10..20
start index             uniform over Austin raceline1
lateral offset          disabled
```

### 4.2 Actor

Relevant class: `model.py:End2RaceResidual`.

```text
BC LiDAR/speed preprocessing
-> frozen GRU features
-> frozen BC output head gives base [steer, speed]
-> trainable residual MLP gives latent r
-> bounded residual_delta(r)
-> composed physical action
```

Latest configuration:

```text
steer = BC_steer + 0.2 * tanh(r_steer)
speed = BC_speed + asymmetric_delta(r_speed)
speed up budget = 0.0
speed down budget = 1.0
```

Only the residual head, `log_std`, and critic are optimized. BC preprocessing, GRU, and BC output head are frozen.

### 4.3 Privileged critic

The critic is a separate MLP:

```text
12 privileged features -> 128 -> 128 -> scalar V(s)
```

Features include relative progress, lateral gap, ego/opponent longitudinal speed, track offsets, overtake/hold state, opponent speedscale, and track phase. Critic gradients do not touch the actor.

This is a standard asymmetric actor-critic construction and is deployable because the critic is discarded. It has not been independently ablated, so its isolated performance benefit remains unproven.

### 4.4 Reward

Current reward components:

```text
ego progress
relative progress (positive component gated by front/side risk)
dense opponent clearance risk
ego collision penalty -120
safe overtake bonus after 0.7 s hold
optional closing-potential shaping
```

The latest full sweep used:

```text
front_base_margin       0.9
time_gap                0.8
side_gate_edge_margin   0.10
closing potential       disabled
```

There is no dense wall-clearance term. Wall safety is learned only from BC inheritance, LiDAR, and terminal collision feedback.

### 4.5 Rollout, GAE, and PPO

- Serial recurrent rollout buffer.
- Episode-start masks reset GRU hidden state during replay.
- Ego collision is a true termination with zero bootstrap.
- Timeout and opponent-only collision are treated as truncations with privileged-critic bootstrap.
- GAE computes returns backward across each episode, stopping at episode boundaries.
- PPO uses clipped surrogate loss, value loss, latent Gaussian entropy, gradient clipping, and KL early stop.
- Latest runs use running advantage scale (`EMA decay 0.99`) and full-sequence PPO updates without minibatches.
- In residual runs, `beta_bc=0`; the BC anchor values in logs are diagnostics only. Structural freezing and residual budgets, not anchor loss, constrain the actor.

### 4.6 Checkpoints and evaluation

- Actor-only snapshots contain the BC backbone, residual head, and residual budgets.
- Full checkpoints additionally contain critic, optimizer, iteration, config, log standard deviation, and running advantage state.
- `eval_multiagent.py` identifies residual checkpoints by `res_head.*` keys and deploys the actor deterministically.
- Current evaluation outcome is:
  - collision if either vehicle collision flag is set;
  - otherwise overtake if ego progress is ahead at the final step;
  - otherwise follow.

## 5. Properties Re-verified During This Audit

The following statements are supported by fresh checks on the current checkout:

1. `py_compile` passed for:
   - `train_ppo.py`
   - `model.py`
   - `ppo_utils.py`
   - `eval_multiagent.py`
   - `analyze_paired_eval.py`
   - `analyze_collisions.py`
2. Loading the BC checkpoint into a zero-initialized residual actor gives exact deterministic equality:
   - max action difference: `0.0`
   - max recurrent hidden difference: `0.0`
3. Frozen BC parameters receive no gradients under the current residual setup.
4. The residual final layer receives gradients; its first layer receives none on the first update because the final layer is zero initialized, then becomes trainable after the final layer moves away from zero.
5. Runtime recurrent replay-identity validation passed throughout the completed training logs.
6. The full sampler actually produced all three opponent racelines, all four speedscale values, and intervals from 10 through 20.
7. Residual composed-action clipping was rare in the completed full-distribution runs:
   - mean clip rate approximately `0.04%`;
   - maximum logged iteration approximately `0.95%` for seed0 and `0.83%` for seed1.
8. The latest top-three result directories each contain all 600 episode records.

## 6. Confirmed Errors and Omissions

### 6.1 Critical: evaluator exit code collides with Python error status

`eval_multiagent.py` exits with:

```text
1 = following
2 = overtaking
3 = collision
```

Uncaught Python exceptions also normally return exit code `1`. `evaluate.sh`, `evaluate_ol1.sh`, and the full-sweep wrapper therefore count a crashed worker as a valid following episode.

This happened in the latest run:

```text
full_disc_r8192 seed1 iter300
reported final: 600 episodes, 296 follow, 279 overtake, 25 collision, 0 error
actual stored episodes: 488
missing records: 112
```

The missing workers were silently counted as follow. This snapshot is invalid. The top-three snapshots are complete and are not affected by this specific occurrence.

Required correction:

- process success must use exit code 0;
- outcome must be read from the metrics JSON;
- aggregation must fail unless exactly the expected number of nonempty metrics files exists;
- selection must reject any evaluation with `error_count != 0` or `episode_count != total`.

### 6.2 High: training and evaluation use different collision semantics

Training:

```text
collision = obs['collisions'][0]
```

Evaluation:

```text
collision = np.any(obs['collisions'])
```

An opponent-only crash is unpenalized by PPO but counted against the model in evaluation. Existing NPZ files save only one aggregate collision boolean, so the discrepancy cannot be repaired offline.

Required correction:

- save `ego_collision` and `opp_collision` separately;
- report both;
- explicitly define the primary KPI before changing either training or evaluation semantics.

### 6.3 High: snapshot selection and final claim share the same Austin 600 set

The latest run evaluated 14 snapshots on Austin full 600 and ranked the same results to identify the top three. Their apparent improvement is therefore a development-set result subject to snapshot-selection bias.

Independent paired recomputation against BC:

| candidate | collision | overtake | fixed collision | new collision | exact paired p(collision) |
|---|---:|---:|---:|---:|---:|
| seed1 iter160 | 20/600 | 341/600 | 11 | 6 | 0.332 |
| seed1 iter120 | 21/600 | 346/600 | 13 | 9 | 0.524 |
| seed0 iter40 | 22/600 | 345/600 | 9 | 6 | 0.607 |

All three have favorable point estimates. None is statistically established on the selection set, and none has completed independent holdout/cross-map evaluation.

### 6.4 High: latest experiment does not isolate the raceline or rollout hypothesis

Relative to Branch B, the completed full-distribution condition changes several variables together:

- opponent racelines: OL1 only -> OL0/OL1/OL2;
- rollout size: approximately 4096 -> 8192;
- start sampler: canonical-like -> uniform full raceline;
- interval range: Branch B range -> 10..20;
- snapshot selection population.

The result is consistent with broader training diversity fixing directional bias. It does not prove that opponent-raceline expansion alone caused the gain, nor that 8192 is better than 4096.

### 6.5 High: asymmetric speed residual has a stochastic-train/deterministic-eval mismatch

With `speed_up_budget=0`, positive latent speed residuals map to zero while negative residuals map to braking. At zero mean and standard deviation 0.25:

```text
deterministic compose(mean): 0.000 m/s residual
training sampled expectation: approximately -0.096 m/s
probability of a negative residual: approximately 50%
```

Thus zero initialization is exactly BC only for deterministic inference, not for stochastic training rollouts. Deployment uses the deterministic latent mean.

Current `dspeed` and `dspd_c` log the transformed mean residual, not the sampled residual actually executed. Therefore `dspeed == 0` alone does not prove that training rollouts never braked.

The latest completed runs did show small negative mean residuals early in training, but both selected seed1 snapshots had returned to zero mean speed residual. The current candidates are empirically close to steer-only modifications at deployment.

### 6.6 High: historical statistical tests are often not valid for the claimed conclusion

Several historical reports:

- used an unpaired two-proportion z-test although BC and PPO ran the same deterministic episode keys;
- repeated the same BC 600 outcomes once per training seed and treated the repeated rows as independent observations;
- treated “candidate lies inside the BC Wilson interval” as evidence of equivalence or passing;
- reported p-values after adaptive seed/snapshot selection without an untouched final test set.

These procedures are useful descriptively but do not prove equality or superiority.

Preferred protocol:

1. use episode-paired transition counts and McNemar/exact sign tests;
2. select checkpoints on a development set;
3. evaluate the chosen checkpoint once on untouched holdout scenarios;
4. report seed variability separately instead of cloning the same BC cases into a larger binomial sample.

### 6.7 Medium: hard-case replay is only location oversampling

The generated hard-start file contains one `ego_idx` per location plus comments such as `weight=3`. `_load_index_file()` removes comments and deduplicates values, and the sampler draws uniformly.

It does not retain:

- opponent raceline;
- opponent speedscale;
- interval;
- BC/candidate outcome;
- severity weight.

The not-yet-run hard groups should be described as hard-location oversampling, not hard-episode replay. Before using them for a causal experiment, either rename the mechanism or preserve complete scenario tuples and weights.

### 6.8 Medium: collision/trajectory analysis is approximate

`eval_multiagent.py` records trajectory data before `env.step()`. If the step collides, the post-step collision pose and per-agent collision flags are not appended.

`analyze_collisions.py` then classifies the last pre-impact frame using:

- ego-opponent distance <= 1 m => car collision;
- otherwise => wall collision;
- final relative progress => pre/alongside/post.

`analyze_paired_eval.py` reconstructs lateral separation using Euclidean distance and centerline progress rather than the exact training `project_to_reference()` geometry. Its action differences compare policies at the same timestep after trajectories have diverged, so they combine policy and state-distribution effects.

These tools are useful diagnostics but should not be treated as ground-truth collision causality.

### 6.9 Medium: the frozen-feature probe does not settle the unfreeze question

The probe report shows high test R2 for front/side/rear risk, but:

- the probe is a nonlinear MLP, not a linear probe;
- train/test rows are randomly split timesteps from the same serial trajectories;
- adjacent states can appear on both sides of the split;
- positive-risk states are rare (roughly 1-3%).

The result supports “risk information is probably recoverable from frozen features.” It does not prove that the frozen representation generalizes to unseen episodes/maps or that unfreezing can never help.

For a decisive probe, split by whole episode/start region and report positive-state recall/calibration in addition to MSE/R2.

### 6.10 Medium: training logs are timestep-level, not episode-level

`coll` is mean collision flag per step. `succ` is the fraction of timesteps after the success flag becomes true. They are not episode collision and success rates.

Consequences:

- training plots cannot be directly compared to evaluation episode rates;
- `succ` is affected by how long an episode continues after success;
- stop-loss gates work only because a collision is usually a one-step event, so `coll * rollout_steps` approximates collision event count.

The buffer records completed episodes but does not log episode outcome counts by scenario/raceline/speedscale.

### 6.11 Medium: reward/evaluation success definitions differ

Training success requires:

- starting behind;
- crossing ahead;
- reaching a 2 m lead;
- low clearance risk;
- holding for 0.7 seconds.

Evaluation overtake only checks whether ego is ahead at the final episode step. This can count transient leads that would not receive the training success bonus, and can miss a safe overtake followed by being repassed.

Both metrics may be retained, but reports must distinguish `safe_hold_success` from `final_lead_overtake`.

### 6.12 Medium: the motivating post-overtake failure is not well represented by the current benchmark

The measured collision buckets in recent Austin runs are mostly car/alongside, with no car/post collisions under the approximate classifier:

| model | total collision | car/pre | car/alongside | wall | car/post |
|---|---:|---:|---:|---:|---:|
| BC Austin full | 25 | 3 | 18 | 4 | 0 |
| full-disc seed1 iter160 | 20 | 2 | 14 | 4 | 0 |
| full-disc seed1 iter120 | 21 | 3 | 13 | 5 | 0 |
| full-disc seed0 iter40 | 22 | 2 | 16 | 4 | 0 |

This suggests the current gain is mainly reduced alongside contact. It does not demonstrate that post-overtake merge/tail collision has been solved. A dedicated post-pass metric or scenario family is required for that claim.

### 6.13 Medium: resume configuration is not validated

A full checkpoint loads residual budgets and optimizer state, but CLI/config compatibility is not checked. Resuming with different residual budgets or normalization settings can silently produce a run whose logged CLI does not describe the loaded actor.

### 6.14 Infrastructure: Numba cache failure and fragile sweep control

The latest long sweep failed when Numba loaded a corrupted `.nbi` cache. The latticeplanner cache directory contained tens of thousands of `.nbc` entries and a roughly 52 MB index after parallel evaluation. The likely failure boundary is concurrent worker writes to a shared Numba cache.

The unattended script exits the entire sweep on the first failed group. Future runs should:

- use a fresh per-run `NUMBA_CACHE_DIR`;
- optionally warm up the planner once before launching workers;
- preserve failed-group status and continue independent groups;
- never select a result without episode-completeness validation.

## 7. Experiment Ledger

This section records the main experiment sequence, the relevant result, and the evidence location. Historical reports contain more detailed per-iteration tables.

### 7.1 Privileged critic initial full run

Run:

```text
logs/ppo_privcritic_20260703_235433/
```

Key evidence:

- `eval_summary.log`
- `analyze_ppo.log`
- `analyze_bc.log`
- `train_100.log`
- `train_1000.log`
- `status.tsv`

Reported result:

```text
BC:  collision 4.0%, overtake 57.2%
PPO: collision 16.3%, overtake 62.8%
```

Interpretation after audit:

- code ran and critic EV improved on ordinary rollouts;
- PPO increased overtake attempts but caused many new pre-overtake contacts;
- this run does not isolate privileged critic because there is no otherwise-identical non-privileged critic control.

### 7.2 OL1 positive-progress risk gate

Run:

```text
logs/ppo_ol1gate_20260704_090713/
```

Key files:

- `README.md`
- `eval_summary.log`
- `analyze_ppo.log`
- `train_300.log`

Result:

```text
BC OL1 collision: 2.5%
gated PPO collision: 58.5%
gated PPO overtake: 6.0%
```

Conclusion: the soft front-risk gate was too weak to prevent unsafe close-follow behavior. It did not establish that reward gating in general is ineffective.

### 7.3 Credit/critic experiment 1b

Run:

```text
logs/ppo_ol1credit_20260704_104608/
```

Change:

```text
gae_lambda 0.99
critic_lr 5e-4
```

Result:

```text
collision 25.0%
overtake 7.0%
```

Conclusion: longer credit horizon and faster critic learning materially improved the failed gate run but did not restore BC safety.

### 7.4 Continue 1b to 600 iterations

Run:

```text
logs/ppo_ol1credit_resume600_20260704_130847/
```

Key files:

- `analysis_report.md`
- `decision_criteria.md`
- `eval_summary.log`
- `train_resume_600.log`

Result:

```text
OL1 collision 38.5%
```

Conclusion: the apparent 300-iteration improvement was not stable convergence. More training made the deterministic policy worse.

### 7.5 Pre-overtake BC anchor multiplier

Run:

```text
logs/ppo_ol1prebc10_20260704_144139/
```

Result:

```text
OL1 collision approximately 22.5%
overtake approximately 3.0%
```

Conclusion: stronger pre-overtake anchoring reduced the collapse but remained far worse than BC and suppressed policy change.

### 7.6 D1-a: frozen BC speed plus PPO steering

Primary report:

```text
logs/all_results_summary_remote.md
logs/d1a_seed1_confirmation_20260705_0846/report.md
```

Seed1 three-grid result:

```text
collision 21/600 = 3.50%
overtake 11/600 = 1.83%
BC        24/600 = 4.00%, 8/600 = 1.33%
```

Conclusion:

- useful safety decomposition and evidence that speed-path drift mattered;
- not a final architecture proof;
- point improvement is small and not a strict per-grid pass.

### 7.7 D1-b: lateral-offset curriculum

Report:

```text
logs/d1b_summary_20260705_092447.md
```

Configuration:

```text
freeze BC speed
lateral_offset_prob 0.5
offset magnitude 0.3..0.8 m
```

Two complete seeds:

```text
seed1  collision 17/600, overtake 5/600
seed2r collision 42/600, overtake 11/600
combined collision 59/1200 = 4.92%
combined overtake 16/1200 = 1.33%
```

Conclusion: curriculum exposure did not robustly transfer to safe evaluation behavior; alongside contact became dominant.

### 7.8 Strong global BC-anchor baseline

Report:

```text
logs/anchor_baseline_summary_20260705_113917.md
```

Configuration:

```text
single network
beta_bc 5.0
anchor_speed_scale 7.5
no frozen execution speed
```

Result:

```text
seed0 collision 49/600
seed1 collision 27/600
combined collision 76/1200 = 6.33%
combined overtake 22/1200 = 1.83%
```

Conclusion: global output anchoring did not reliably preserve BC safety. Historical confidence-interval comparisons should not be interpreted as an equivalence test.

### 7.9 D2: frozen BC backbone plus bounded residual head

Design/result report:

```text
logs/d2_design_20260705.md
logs/d2_summary_20260705_164511.md
```

Result:

```text
seed0 collision 38/600, overtake 6/600
seed1 collision 33/600, overtake 7/600
```

Mechanism observed: positive speed residual drift moved toward the `+0.2 m/s` budget and correlated with new collisions.

### 7.10 D2-b: remove positive speed budget

Report:

```text
logs/d2b_summary_20260705_183533.md
```

Change:

```text
residual_speed_up_budget 0.0
```

Result:

```text
seed1 collision 19/600, overtake 9/600
seed0 collision 176/600, overtake 80/600
```

Conclusion: positive speed budget was a real risk source, but removing it did not prevent a steering-driven collapse. One seed passed descriptively; one failed catastrophically.

### 7.11 D2-c: running advantage scale and rollout 4096

Report:

```text
logs/d2c_5seed_summary_20260706.md
```

Five-seed results:

```text
seed0 23/600 collision, 11/600 overtake
seed1 32/600, 6/600
seed42 24/600, 6/600
seed2 24/600, 8/600
seed3 21/600, 8/600
```

Conclusion:

- catastrophic D2-b-style failures were reduced;
- aggregate point performance stayed near BC;
- the historical five-seed binomial aggregation repeats the same BC cases and should be read descriptively, not as 3000 independent tests.

### 7.12 D3: closing-potential shaping

Report:

```text
logs/d3_summary_20260706.md
```

Results:

```text
k=0.5 seed0 collision 11.33%, overtake 2.83%
k=0.5 seed1 collision 2.00%,  overtake 2.00%
k=1.0 seed0 collision 4.33%, overtake 1.83%
k=1.0 seed1 collision 6.50%, overtake 2.17%
```

Conclusion:

- closing shaping increased overtaking attempts;
- safety was highly variable and failed as a configuration family;
- the report's “制动通道死亡” conclusion is stronger than the logged mean-residual evidence supports because sampled residual braking was not logged;
- potential-based policy-invariance language should be qualified because the actor is partially observed/restricted and the physical transform is asymmetric/non-invertible when up budget is zero.

### 7.13 D4-A initial paired analysis, margin 0.20

Report:

```text
logs/d4a_paired_summary_20260707.md
```

Finding:

- selected early snapshots mostly swapped a few episode outcomes rather than producing systematic gains;
- BC OL1 overtakes used narrow lateral corridors;
- side-risk geometry penalized many feasible passes.

Important correction: the report calls the risk probe “linear”; the implemented probe is nonlinear and temporally leaked, so it does not conclusively eliminate representation limits.

### 7.14 D4 margin 0.04

Report:

```text
logs/d4b_margin004_summary_20260707.md
```

Result:

- no snapshot Pareto-dominated BC in the desired direction;
- paired example gained three overtakes but introduced seven new collisions.

Conclusion: lowering the margin encouraged more pass attempts, but “reward repair succeeded and only execution precision remains” is too strong. The run does not isolate execution precision from sampling, policy capacity, residual parameterization, or optimization variance.

### 7.15 D4-C margin 0.10 and exact-canonical training

Reports:

```text
logs/d4c_final_summary_20260708.md
logs/d4c_holdout_paired_audit_20260708.md
```

Best D4-C candidate:

```text
pretrained/end2race_ppo_d4c_best_canon_seed0_iter0040.pth
```

Three-grid result:

```text
candidate collision 18/600, overtake 14/600
BC        collision 24/600, overtake 8/600
```

Fresh offsets showed the safety advantage did not hold cleanly:

```text
off63: candidate new collisions > fixed collisions
fresh off63+off84: candidate collision 20 vs BC 18
```

Conclusion: useful near-BC candidate and a reason to try canonical-like diversity; not a final generalized improvement.

### 7.16 Branch B canonical-like sampler

Primary report:

```text
logs/branch_b_canonlike_final_report_20260709.md
logs/branch_b_canonlike_summary_20260708_164416.md
```

Training sampler:

```text
scenario_mode canonical_ol1
canonical_jitter 21
interval range [12,19)
lateral_offset_prob 0
side_gate_edge_margin 0.10
```

Checkpoint:

```text
pretrained/end2race_ppo_branchb_best_canonlike_seed0_iter0080.pth
sha256 66117f2388e941edb74091973452be74c1ebf17a91ae2b453d26b9518cf8489f
```

Five-grid OL1 result:

```text
Branch B collision 28/1000, overtake 22/1000
BC       collision 42/1000, overtake 12/1000
```

Paired five-grid result:

```text
fixed collisions 19
new collisions 5
gained overtakes 10
lost overtakes 0
```

This was the strongest OL1 result. It was selected from many snapshots/seeds and then evaluated more broadly.

### 7.17 Branch B broader evaluation

Remote report:

```text
logs/broader_eval_20260709_065523/summary.md
```

Local checkpoint/report references:

```text
logs/branch_b_canonlike_final_report_20260709.md
pretrained/end2race_ppo_branchb_best_canonlike_seed0_iter0080.pth
```

Evaluation per map:

```text
50 starts x 3 opponent racelines x 4 speedscales = 600 episodes
```

| map | BC collision/overtake | Branch B collision/overtake |
|---|---:|---:|
| Austin | 25/341 | 22/342 |
| Nuerburgring | 38/384 | 36/388 |
| MoscowRaceway | 38/381 | 39/383 |
| Hockenheim | 37/330 | 36/329 |

Paired totals across the four reports:

```text
collision fixed/new = 37/32, exact p approximately 0.630
overtake gained/lost = 36/30, exact p approximately 0.539
```

Conclusion: Branch B transfers without a broad collapse and is the best fully evaluated candidate. Its advantage over BC is small and not statistically established.

### 7.18 Latest Austin full-distribution sweep

Run id:

```text
20260709_210827
```

Remote run root:

```text
logs/full_sweep_20260709_210827/
```

Local progress report:

```text
logs/full_sweep_progress_report_20260710.md
```

Unattended script:

```text
logs/full_sweep_unattended/run_austin_full_sweep.sh
```

Completed only:

```text
full_disc_r8192 seed0: train + 7 snapshot evals
full_disc_r8192 seed1: train + 7 snapshot evals
```

Top Austin development candidates:

| candidate | collision | overtake | paired fixed/new |
|---|---:|---:|---:|
| seed1 iter160 | 20/600 | 341/600 | 11/6 |
| seed1 iter120 | 21/600 | 346/600 | 13/9 |
| seed0 iter40 | 22/600 | 345/600 | 9/6 |

Remote-only checkpoint paths:

```text
pretrained/end2race_ppo_full_disc_r8192_seed1_20260709_210827_iter0160.pth
pretrained/end2race_ppo_full_disc_r8192_seed1_20260709_210827_iter0120.pth
pretrained/end2race_ppo_full_disc_r8192_seed0_20260709_210827_iter0040.pth
```

Remote hashes:

```text
iter160 77cd79904f0f57c1e7a4914dd0b52384628dce225f9222e4e2274e0eda3b5aa6
iter120 9f2f47bf46363946ba29c1fe5fcada3a3d5fe514ece6eb160c03b25d8f82b3b3
seed0/40 c7a72f5564a191e103d319a7f66167e6969fb3528534b90bafba77ceb598d7e1
```

Not run:

```text
full_cont_r8192 seed0/1
hard_cont_r8192 seed0/1
hard_cont_r4096 seed0
hard_cont_r12288 seed0
selected-candidate Austin holdouts
selected-candidate cross-map evaluation
final_summary.md
```

Failure:

```text
step=train_full_cont_r8192_seed0
status=failed
UnicodeDecodeError while loading Numba cache
```

## 8. Conclusion Confidence Map

### 8.1 Established by code or direct verification

- Zero-initialized deterministic residual actor equals BC exactly.
- BC backbone is frozen and gradient isolated in residual mode.
- Recurrent log-prob replay is internally consistent in completed runs.
- Positive speed budget was associated with unsafe upward speed drift in D2.
- `speed_up_budget=0` prevents deterministic positive speed residual.
- Global BC anchor alone did not reliably preserve safety.
- Lateral-offset curriculum at probability 0.5 was not robust.
- Branch B is broadly deployable on the tested maps without catastrophic degradation.
- Latest full-distribution top candidates are complete 600-episode Austin evaluations.
- Evaluator exit-code ambiguity has already corrupted at least one snapshot result.

### 8.2 Supported but not isolated

- Training on all opponent racelines probably helps Austin directional coverage.
- Running advantage scale plus larger rollout reduces catastrophic training waves.
- Side-gate margin 0.10 is a viable operating point.
- Frozen GRU features probably contain useful side/rear-risk information.
- Latest full-distribution candidates may improve Austin safety.

### 8.3 Not yet proven

- `rollout_steps=8192` is better than 4096 or 12288.
- Discrete speedscale sampling is better than continuous sampling.
- Hard-location replay improves safety.
- Privileged critic improves final policy relative to an otherwise-identical ordinary critic.
- Any current model is statistically superior to BC on an untouched final test set.
- Current PPO solves post-overtake merge/tail collisions.
- Frozen backbone is definitively sufficient for unseen maps.

### 8.4 Historical claims that should be withdrawn or softened

- “Inside the BC Wilson interval means equivalent/pass.”
- “Five seeds x the same 600 BC episodes gives 3000 independent BC trials.”
- “The frozen-feature probe is linear and rules out representation limits.”
- “Margin 0.04 proves only execution precision remains.”
- “The latest experiment proves raceline expansion caused the improvement.”
- “8192 has already been shown to reduce collision.”
- “`dspeed=0` proves the stochastic training policy never brakes.”
- “Top Austin snapshot already beats BC” without the qualifier “on the snapshot-selection development set.”

## 9. Current Candidate Ranking

### Candidate A: Branch B, best validated

```text
pretrained/end2race_ppo_branchb_best_canonlike_seed0_iter0080.pth
```

Pros:

- strong five-grid OL1 result;
- complete Austin/Nuerburgring/Moscow/Hockenheim evaluation;
- no broad catastrophic regression;
- available locally and remotely.

Cons:

- broad gains are small and paired tests are not significant;
- selected from multiple seeds/snapshots;
- evaluator collision semantics and error protocol still need correction for final publication-quality claims.

### Candidate B: full-disc seed1 iter160, best Austin safety candidate

```text
remote: pretrained/end2race_ppo_full_disc_r8192_seed1_20260709_210827_iter0160.pth
```

Pros:

- Austin full collision 20/600 vs BC 25/600;
- balanced across opponent racelines and speed bins;
- fixed/new collision 11/6.

Cons:

- chosen on the same Austin 600 used for reporting;
- overtake equals BC;
- no holdout or cross-map evaluation;
- remote-only artifact at audit time.

### Candidate C: full-disc seed1 iter120, Austin safety/overtake tradeoff

```text
remote: pretrained/end2race_ppo_full_disc_r8192_seed1_20260709_210827_iter0120.pth
```

Pros:

- Austin 21/600 collision and 346/600 overtake;
- fixed/new collision 13/9.

Cons:

- OL1 collision is worse than BC;
- speedscale performance is uneven;
- no independent validation.

### Candidate D: D4-C historical candidate

```text
pretrained/end2race_ppo_d4c_best_canon_seed0_iter0040.pth
```

Keep as an ablation/historical artifact. Branch B supersedes it on fresh OL1 offsets.

## 10. Mandatory Next-Agent Work Order

Do not start another large training sweep before completing Priority 0 and Priority 1.

### Priority 0: repair evaluation integrity

1. Remove outcome-as-process-exit-code semantics.
2. Require one valid metrics JSON and NPZ per requested episode.
3. Save ego/opponent collision flags and the post-step terminal pose.
4. Define whether primary collision KPI means ego collision or any-agent collision.
5. Make aggregation reject stale, missing, duplicate, and extra episodes.
6. Add a regression test reproducing the 488/600 silent-failure case.

### Priority 1: validate existing top three before more training

Use a fresh per-run Numba cache and unique result tags. Evaluate each top candidate against matched BC on:

1. Austin holdout start offsets 21/42/63/84, full 600 each;
2. Nuerburgring, MoscowRaceway, Hockenheim, full 600 each;
3. episode-paired transition reports;
4. per-opponent-raceline and per-speedscale breakdown;
5. exact completeness checks.

Selection must occur before opening the holdout results. Do not choose a different snapshot per map.

### Priority 2: archive reproducible source and candidates

For every new run, store:

```text
git HEAD
git status
complete git diff including model.py
copies or hashes of all untracked source scripts
core-file SHA256 values
exact command line
environment/package versions
checkpoint SHA256 values
evaluation manifest with expected keys
```

Sync the remote top-three checkpoints to a durable local/archive location after validation.

### Priority 3: decide whether more training is justified

If one current top candidate passes holdout and cross-map checks, stop the broad sweep and write the final model report.

If all fail:

1. repair the sweep wrapper so independent groups continue after infrastructure failures;
2. isolate Numba cache per run;
3. decide whether the next single-variable comparison is:
   - full discrete 4096 vs 8192, or
   - discrete vs continuous speedscale at fixed rollout;
4. do not call the current hard-start mechanism hard-episode replay until it preserves complete scenario tuples.

### Priority 4: algorithm changes only after validation

Potential later work, in evidence order:

1. log sampled and mean physical residual actions separately;
2. eliminate or explicitly model the asymmetric stochastic/deterministic speed mismatch;
3. add episode-level training outcomes by raceline/speed/start bucket;
4. add precise wall and collision-type diagnostics before adding more reward terms;
5. run an episode-held-out frozen-feature probe;
6. ablate privileged critic only if the user needs a paper-level causal claim about it.

## 11. Useful Commands for a New Chat

### Local source status

```bash
cd /home/haowei/Documents/End2Race
git status --short
git rev-parse HEAD
/home/haowei/miniconda3/envs/end2race/bin/python -m py_compile \
  train_ppo.py model.py ppo_utils.py eval_multiagent.py \
  analyze_paired_eval.py analyze_collisions.py
```

### Remote sweep state

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 haowei@100.95.251.103 '
  cd ~/Documents/End2Race
  cat logs/full_sweep_20260709_210827/current_status.txt
  cat logs/full_sweep_20260709_210827/status.tsv
  pgrep -fa "train_ppo.py|eval_multiagent.py|run_austin_full_sweep" | grep -v grep || true
'
```

### Current run report

```bash
sed -n '1,320p' logs/full_sweep_progress_report_20260710.md
```

### Branch B final report

```bash
sed -n '1,260p' logs/branch_b_canonlike_final_report_20260709.md
```

### Paired analyzer

```bash
python analyze_paired_eval.py <bc_eval_dir> <candidate_eval_dir> \
  --side_gate_edge_margin 0.10 \
  --front_base_margin 0.9 \
  --time_gap 0.8 \
  --out <report.md>
```

Do not run paired analysis until confirming that both directories contain the exact expected episode-key set.

## 12. External Reviewer Questions

An external reviewer should focus on these questions:

1. Is latent-space PPO with the current asymmetric/non-invertible speed transform the desired deployed-policy objective?
2. Should deterministic evaluation use transformed latent mean, transformed-mode, or expected transformed action?
3. Should opponent-only collisions be a primary model failure, secondary interaction metric, or truncation-only event?
4. Is the privileged critic's track-phase feature introducing Austin-specific advantage shaping that harms map transfer?
5. Can the claimed side/rear representation sufficiency survive episode-level and map-level heldout probes?
6. Is final-lead overtake an acceptable KPI, or should safe-hold success be the primary overtake metric?
7. Are Austin startpoints and speed/raceline combinations a fixed benchmark or samples from a target distribution? This determines whether confidence intervals and p-values are meaningful.
8. Does the current reward optimize the user's original post-merge failure, given that recent measured collisions are overwhelmingly alongside?

## 13. Final Handoff Position

The project has made real progress:

- catastrophic unconstrained PPO drift was diagnosed;
- structural residual constraints made training safer;
- side-risk geometry and scenario distribution were improved;
- Branch B produced a model that transfers approximately at BC level across four maps;
- full Austin opponent-raceline training produced promising new candidates.

The remaining bottleneck is no longer simply “find a better PPO hyperparameter.” The immediate bottleneck is experimental integrity: evaluation error handling, collision semantics, snapshot selection, and independent validation.

Until those are repaired, the defensible statement is:

> End2Race residual PPO has produced candidate policies with favorable point estimates and no broad catastrophic regression. Branch B is the best fully evaluated candidate. The latest full-distribution candidates are promising on the Austin development benchmark, but no current model has yet been shown on an untouched final test set to be statistically superior to BC in collision rate while preserving or improving overtaking.

## 14. Addendum 2026-07-10: Priority 0 repaired, Priority 1 launched

Status update after the audit above; the audit body is kept unchanged as the reference point.

### 14.1 Priority 0 implemented and verified

- `eval_multiagent.py` now exits 0 on success; the outcome lives only in the metrics JSON (`outcome`, plus `ego_collision`/`opp_collision`, scenario identity fields, `npz_path`). NPZ additionally stores per-agent collision flags and the post-step terminal pose/progress/time. Fixes findings 6.1 (partially: worker side), 6.2 (recording side), 6.8 (terminal pose).
- New `aggregate_eval.py` performs validated aggregation: requires exit code 0 plus one valid JSON (and, with `--require_npz`, one non-empty NPZ) per requested episode; rejects missing, duplicate-key, extra, and stale-merged episodes; writes `results.json` with `ego_collision_count`/`opp_collision_count` and `validated: true`. `evaluate.sh` and `evaluate_ol1.sh` now use it; exit-code outcome counting is removed. Note: the historical sweep wrapper `logs/full_sweep_unattended/run_austin_full_sweep.sh` is now protocol-incompatible by design (it would loudly report 600 errors, never silently pass).
- Regression test `tests/test_eval_aggregation.py` reproduces the 488/600 silent-failure case plus duplicate/extra/stale/nonzero-exit/missing-NPZ cases; 12 assertions pass locally and remotely.
- KPI definition (finding 6.2 decision): primary collision KPI stays any-agent collision for comparability with all historical numbers; ego-only collision is reported as a secondary breakdown from the new fields.

### 14.2 New finding: offset grids historically contained duplicate episode keys

The start-index formula `idx = i * max_wp / (N-1)` spans the closed racing loop inclusively, so start i=0 and i=N-1 coincide physically. With a nonzero offset wrap they also collide as episode keys: historical offset grids (for example D4-C off63/off84) actually contained 588/600 (or 196/200) unique episodes while counts were reported as 600 (200). Caught by the new duplicate-key validation during the P1 smoke run. Offset grids now use open-interval spacing `idx = (i * max_wp / N + offset) % max_wp`; offset-0 grids keep the historical formula so their keys remain pairable with broader_eval/Branch B results.

### 14.3 Priority 1 validation running unattended on remote

- Run: `logs/p1_validation_20260710_121955/` on `haowei@100.95.251.103`, launched detached (`nohup setsid`), survives SSH/local shutdown.
- Script: `logs/p1_validation_unattended/run_p1_validation.sh` (see its header and `logs/p1_validation_unattended/README.md`): BC + cand160/cand120/cand040 (sha256-pinned) x Austin holdout offsets 21/42/63/84 + Nuerburgring/MoscowRaceway/Hockenheim, 600 episodes each, fresh per-run `NUMBA_CACHE_DIR` with serial warmup, per-grid completeness validation, failed grids recorded and skipped by paired analysis while the run continues, `final_summary.md` with paired exact sign tests.
- Pre-registered before any holdout result was opened: primary candidate cand160 (seed1 iter0160); pass rule = pooled Austin holdout collision <= BC and overtake >= BC with no cross-map collapse.

### 14.4 Outcome (completed 2026-07-10 16:07, zero failures, 28/28 grids validated)

Full analysis: `logs/p1_final_report_20260710.md`. Summary:

- cand160 (primary) formally failed the pass rule by 4/2400 overtakes; both holdout KPIs are statistical ties. Cross-map it beats BC on all three maps (overtake gain p=0.043, collision p=0.053), never significantly worse anywhere.
- cand120 met the pass rule with the first statistically significant double improvement on untouched data in the project: holdout collision 91 vs 111 (p=0.010), overtake 1381 vs 1358 (p=0.003); both survive ×3 Bonferroni. Cross-map collision point-worse (124 vs 113, n.s.), concentrated in ol1 close-following on unseen maps.
- cand040 met the rule (holdout collision p=0.029 uncorrected; overtake tie; cross-map tie).
- Per §10 Priority 3 the remaining sweep groups stay cancelled. Top-3 checkpoints archived locally and remotely.
- Decision (user, 2026-07-10): **cand160 is the sole final deployed model** (`logs/final_model_report_20260710.md`); cand120's Austin holdout improvement is recorded as an Austin-specialized exploratory finding requiring a fresh independent test set before any upgrade to a final conclusion. Candidate ranking in §9 is superseded accordingly: cand160 (Candidate B) is now the project's final model, Branch B (Candidate A) becomes the historical baseline.

### 14.5 New confirmed structural finding (2026-07-10, code-verified): BC train/deploy time-base and speed-input mismatch

- `demonstration.py:216,286`: BC demonstrations are recorded at `sample_interval = 0.1` s (10 Hz) and the recorded speed is the tracker's **desired** speed command.
- `train.py:86-87`: BC's speed input channel is the **previous frame's desired_speed** (its own previous action), sequences built at 10 Hz.
- `eval_multiagent.py` and `train_ppo.py` drive the same GRU at **100 Hz** and feed **actual** speed (`obs['linear_vels_x'][0]`).
- Calibration (2026-07-10, user correction accepted): this mismatch does NOT explain the PPO-vs-BC comparison — BC baseline, PPO rollouts, and PPO checkpoints all run at the same 100 Hz + actual-speed operating point, so comparisons are fair and BC-as-measured already includes the mismatch. The only surviving (weak) hypothesis is a representation ceiling: the frozen GRU may not encode the closing-rate/TTC/future-risk information a residual head would need for anticipatory longitudinal control. This must be decided by an episode-held-out probe, not asserted from the 10/100 Hz discrepancy. Treat the finding as (a) a probe-decidable ceiling hypothesis and (b) a hygiene item (align rates and speed-input semantics) if the backbone is ever retrained — not as a current performance bottleneck. The confirmed PPO-side temporal problems are separate: credit decay (γλ = 0.997 × 0.99 ≈ 0.987 per 10 ms step → a terminal collision reaches an action 1 s earlier with weight ≈ 0.27, 2 s ≈ 0.073, 3 s ≈ 0.020) and 100 Hz iid exploration.
- P1 data cross-tab (BC pooled holdout+cross-map, 224 collisions): failure modes are bimodal — **ol1 × speedscale {0.5, 0.6} = 72 collisions (76% of ol1's 95)** = fast-closing rear-end/following contacts, and **ol0/ol2 × {0.7, 0.8} = 103 collisions** = prolonged alongside passes at small speed differential. Two distinct skills; curriculum and reward design must treat them separately (earlier suggestion "oversample ol1 × {0.7, 0.8}" is corrected to ol1 × {0.5, 0.6}). Note: these are occurrence counts; §14.6 supersedes them with canonical de-duplicated denominators.

### 14.6 D0 canonical audit complete (2026-07-10): corrected denominators are now authoritative

D0 (stage 1 of `docs/superpowers/specs/2026-07-10-ppo-safety-first-bplus-design.md`) passed all stop rules. Deliverables: `logs/d0_canonical_audit_20260710_121955/` (provenance, occurrences, canonical table, collision events, summary, validation).

- Canonical identity = resolved (map, racelines, exact ego pose, exact opponent pose, speedscale, interval, duration, noise). The opponent pose is identity-relevant: under ol1 the raw wrap index shifts the opponent one waypoint at the same ego pose.
- 16,800 P1 occurrences reduce to 3072 canonical scenarios (12 shadow clones from raceline endpoint near-duplication, 36 dev-grid overlaps); usable dev-disjoint N = 3024 (Austin 1260, cross-map 1764). The four P1 Austin offsets contain only 108 unique starts per 200 occurrences — offset spacing collides with grid spacing, so P1 pooled tests double-counted duplicated deterministic scenarios.
- BC canonical: 169 any-agent / 152 ego-involved / 17 opponent-only collisions, 1792 overtakes. The 17 opponent-only collisions occur on identical canonical keys for all four models (empirical floor confirmed).
- Strata (exposure, not causal classes): skill_F (ol1 × 0.5/0.6) N=504, 56 collisions (all ego-involved), 7 overtakes; skill_S (ol0/2 × 0.7/0.8) N=1008, 75 collisions (64 ego), 792 overtakes; together 77.5% of BC collisions. OL1 phase decomposition: 26 pre / 50 alongside / 1 post — a front-TTC brake directly targets at most ~1/3 of OL1; "early braking prevents later alongside contact" stays a D2.5 hypothesis.
- Corrected paired statistics (supersede §14.4 and the P1/final-report p-values): cand120 Austin collision 19/6 p=0.015 (survives ×3 Bonferroni at 0.044), overtake 20/8 p=0.036 (0.107 after correction — "double-significant" withdrawn); cand160 cross-map favorable trends only (collision 25/13 p=0.073, overtake 20/9 p=0.061; the previously reported 0.053/0.043 were occurrence-inflated); cand040 Austin collision 12/5 p=0.14 (previous 0.029 withdrawn). Deployment decision (cand160 final) unaffected.
- Determinism validated at scale: every exact-duplicate scenario replay agrees on outcome across all four models; 3 near-duplicate cases (opponent shifted one waypoint) flip outcomes — boundary chaos sensitivity, supporting clustered statistics (spec §10.3).
- Future Austin final pool: off11/32/75/86 verified (200 unique starts, zero overlap with all opened history); off10/31/52/73 rejected (108 unique, 2 overlaps).

### 14.7 D0.1 reviewed replacement (2026-07-11)

Section 14.6 is retained as D0 v1/Sensitivity-A provenance, but its statement
that D0 was complete is superseded. The reviewed D0.1 release is
`logs/ppo_next_unattended_20260710_230212/artifacts/d01_full_reconcile_20260711_170200_a`;
the sibling `_b` output is byte-identical confirmation.

- 16,800/16,800 occurrence inventory; exact/primary/SensA/SensB
  `3072/3036/3024/2772`; G1–G8 and independent emitted validation pass.
- Primary `(collision/overtake)`: BC `170/1792`, cand160 `154/1799`, cand120
  `168/1797`, cand040 `166/1787`.
- cand160 clustered bootstrap: all RR 0.906 [0.818,0.994], Austin 0.947
  [0.776,1.146] with overtake `716 vs 720`, cross 0.885 [0.787,0.985] with
  overtake `1083 vs 1072`. This is favorable historical evidence but not the
  B+ RR<=0.70/Austin-overtake product gate.
- BC primary breakdown: 170 any-agent, 153 ego-involved, 17 opponent-only;
  OL1 phase 27/50/1; skill_F N=510/57 collisions/7 overtakes; skill_S
  N=1008/75/792.
- Sensitivity A exactly reproduces §14.6; zero P1 outcome corrections were
  required; G4 reports 13/384 matched adjacent-L3 outcome disagreements.

Full audit history, rejected-intermediate explanation, hashes, and claim
boundaries are in
`logs/ppo_next_unattended_20260710_230212/D01_EVIDENCE_REPORT_20260711.md`.

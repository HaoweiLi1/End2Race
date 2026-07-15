# End2Race Safety-First B+ PPO Design

Date: 2026-07-10

Status: Approved design; implementation and experiments are not authorized by this document

Scope: PPO-side changes only; the deployed BC backbone remains frozen and runs at 100 Hz

## 1. Decision summary

The next End2Race research phase will optimize a lexicographic objective:

1. reduce any-agent collision rate materially below BC;
2. keep overtake rate no lower than BC;
3. only after the safety floor is established, optionally improve overtake rate.

Safety gains must not come only from suppressing interaction or abandoning passes. The evaluation must therefore preserve the historical terminal-overtake KPI and add behavior-mechanism diagnostics that distinguish:

- collision converted to a safe overtake;
- collision converted to a justified safe abort/follow;
- a previously safe overtake lost;
- a new collision introduced.

The selected technical family is **B+**:

- keep the BC actor, recurrent state update, observation contract, and 100 Hz deployment unchanged;
- make all behavioral changes in a PPO residual policy;
- replace 100 Hz independent residual exploration with temporally coherent macro residual decisions;
- repair the longitudinal residual distribution so that no-op and braking are identifiable;
- use a constrained safety-first PPO objective rather than relying on one fixed scalar reward;
- add a small causal temporal safety module only if a leakage-free probe shows that frozen BC features lack actionable risk information.

The work is gated. D0, D2, and D2.5 must complete before any PPO training. D3 changes are introduced one at a time. A failed mechanism gate stops the branch; it does not trigger a hyperparameter sweep.

## 2. Goals and non-goals

### 2.1 Primary goals

Let `C(pi)` be any-agent collision rate and `O(pi)` terminal safe-overtake rate under a pre-registered scenario distribution. The first-stage optimization target is:

```text
minimize C(pi)
subject to O(pi) >= O(BC) - delta_O
```

Use `delta_O = 1 percentage point` as the training non-inferiority tolerance. Final product selection remains stricter: the candidate overtake point estimate must be no lower than BC.

The final product gates are:

- hard safety effect: collision relative risk `RR <= 0.70` versus BC;
- target safety effect: collision relative risk `RR <= 0.60` versus BC;
- collision uncertainty: the one-sided 95% upper confidence bound excludes `RR = 1`;
- overtake count: candidate point estimate is no lower than BC in both the Austin pool and the known-cross-map pool;
- paired overtake non-inferiority: one-sided 95% lower confidence bound for risk difference is greater than `-1 percentage point`;
- no material map or skill-stratum collapse;
- demonstrable `collision -> safe overtake` transitions on cases shown to be recoverable within the allowed action space.

### 2.2 Secondary goals

- Preserve the historical any-agent collision KPI for comparability.
- Report ego-involved and opponent-only collisions separately.
- Preserve or improve confirmed passes and BC-safe-overtake retention.
- Produce interpretable longitudinal behavior: braking should activate in risk corridors and remain near no-op outside them.
- Keep all deployment inputs sensor-compatible: LiDAR, ego speed, previous command, and recurrent state only.

### 2.3 Non-goals

- Do not retrain, unfreeze, or change the BC backbone during B+.
- Do not change the BC evaluator from its established 100 Hz operating point.
- Do not claim that 10 Hz BC demonstration collection is a PPO train/eval mismatch.
- Do not start with SAC, TD3, a large parallel-environment rewrite, or an end-to-end planner replacement.
- Do not use aggregate overtake preservation alone as proof that the policy did not become conservative.
- Do not treat a single successful seed or snapshot as a result.
- Do not reuse an opened final set for model or architecture selection.

## 3. Evidence baseline and corrected denominators

All design calculations use canonical scenario identity, not grid occurrence count.

After de-duplicating the four Austin P1 offset grids, removing 36 episode keys that overlap the Austin development grid, and combining the three P1 cross-map grids:

```text
N = 3060 unique, development-disjoint episode keys
BC any-agent collisions = 170
BC ego-involved collisions = 153
BC terminal overtakes = 1815
BC opponent-only collisions = 17
```

The 17 opponent-only collisions occur on the same episode keys for BC, cand040, cand120, and cand160. They are an empirical policy-invariant floor in the currently explored policy neighborhood, not a proof of absolute uncontrollability.

Two high-exposure skill strata cover most collisions:

| Stratum | Definition | N | Any collision | Ego-involved collision | Overtake |
|---|---|---:|---:|---:|---:|
| `skill_F` | opponent raceline1, speedscale 0.5 or 0.6 | 510 | 57 | 57 | 7 |
| `skill_S` | opponent raceline0/2, speedscale 0.7 or 0.8 | 1020 | 75 | 64 | 803 |
| other | remaining distribution | 1530 | 38 | 32 | 1005 |

`skill_F + skill_S` contains 77.6% of any-agent collisions and 79.1% of ego-involved collisions. These are exposure strata, not established causal classes:

- the 78 total OL1 collisions contain approximately 34 pre-pass, 43 alongside, and 1 post-pass terminal phase under the current diagnostic classifier;
- a front-TTC intervention directly targets only part of this set;
- whether early braking can prevent later alongside contact is a hypothesis to test in D2.5, not an assumed fact.

## 4. System boundary and architecture

### 4.1 Frozen BC path

At every 10 ms simulator step:

1. the current 360-ray LiDAR and actual ego speed are passed through the existing BC actor;
2. the BC GRU hidden state updates at 100 Hz exactly as in current evaluation;
3. the BC action remains the base action;
4. the PPO residual composes with the BC action.

With a zero/no-op residual, the deployed action sequence must be exactly the current BC action sequence. This equality is a hard regression test.

The historical 10 Hz demonstration cadence and previous-desired-speed training input are retained only as a representation-ceiling hypothesis. They do not alter the BC/PPO comparison because BC, PPO rollout, and PPO evaluation all operate at 100 Hz with actual speed.

### 4.2 Macro residual path

The B+ residual policy makes one decision every `K = 10` simulator steps, or 10 Hz. The BC actor still updates every simulator step.

At a macro boundary, the residual policy reads the current frozen BC feature and chooses:

- a continuous bounded steering residual;
- a longitudinal no-op/brake decision;
- conditional brake magnitude when braking is selected.

The residual decision is held for up to 10 micro-steps. A terminal event can end the macro transition early.

One macro decision produces exactly:

- one stored policy action;
- one policy log-probability;
- one macro reward/cost transition;
- one PPO ratio during replay.

The implementation must not store the same macro log-probability as ten independent actions.

### 4.3 Longitudinal hurdle distribution

The current speed residual maps every positive latent sample to the same physical no-op when `speed_up_budget = 0`. B+ replaces it with a mixed distribution:

```text
gate g in {NO_OP, BRAKE}

if g = NO_OP:
    delta_speed = 0
    log_prob = log P(g = NO_OP)

if g = BRAKE:
    latent z ~ p(z | BRAKE)
    magnitude b in (0, B)
    delta_speed = -b
    log_prob = log P(g = BRAKE) + log p(z | BRAKE)
```

The initial deterministic gate selects `NO_OP`, so deployment is exactly BC. In the brake branch, sample a Gaussian latent `z` and use `b = B * sigmoid(z)` with `B = 1.0 m/s`; the stored brake-branch action and PPO log-probability remain in latent space. This makes every finite brake latent correspond to one brake magnitude in `(0, B)`. Positive speed residual is not allowed in B+.

The policy must log, separately:

- gate probability and deterministic gate;
- sampled brake magnitude;
- conditional mean magnitude;
- expected physical speed residual;
- executed speed residual after composition and clipping;
- the same values inside and outside risk corridors.

### 4.4 Optional temporal safety module

The default B+ residual reads the frozen BC feature. A small causal temporal safety module is added only if D2 proves both of the following:

1. a capacity-matched MLP cannot recover actionable risk from the frozen feature;
2. a causal short-history model can recover it on an independent grouped test split.

The optional module may read only deployable signals:

- frozen BC feature history;
- actual ego speed history;
- previous command history;
- LiDAR or compact LiDAR-sector history.

It must not read Frenet geometry, opponent pose, map phase, or simulator state at deployment.

### 4.5 Training-only critics

B+ uses separate training-only value functions for:

- collision cost;
- overtake/confirmed-pass return;
- optional shaped task return.

Privileged simulator features may be used by critics and probe labels, but never by the deployed actor. Actor gradients, collision-critic gradients, and other critic gradients are clipped separately so that a large collision-batch value loss cannot determine the actor's global clipping coefficient.

## 5. Candidate method routes

### 5.1 Route R1: pure B+ residual PPO

Use the macro residual, hurdle brake distribution, constrained objective, and staged curriculum without demonstration warm-start.

Advantages:

- closest to the current architecture;
- clearest PPO attribution;
- lowest engineering scope.

Risk:

- rare positive examples may lead the policy to learn safe following or aborting without learning recoverable safe passes.

R1 is suitable for a short mechanism pilot, not an immediate full training run.

### 5.2 Route R2: counterfactual recovery warm-start plus B+ PPO

Use D2.5 to find residual macro-action sequences that convert BC collisions into confirmed safe passes. Supervise the residual gate/magnitude on those positive branches before constrained PPO fine-tunes timing and generalization.

Advantages:

- directly supplies the rare action sequences that pure on-policy exploration is unlikely to discover;
- preserves BC as the base policy;
- lets PPO optimize when to intervene instead of inventing every maneuver from terminal collision events.

Risk:

- requires deterministic branch replay and careful separation of privileged search from deployable policy inputs.

R2 is the recommended main route if D2.5 finds a sufficient recoverable set.

### 5.3 Route R3: hierarchical option PPO

If D2.5 shows that recovery requires multi-second coherent maneuvers outside the expressive range of the macro residual, introduce high-level options such as follow, initiate pass, abort/brake, retry, and merge. Low-level options may be distilled from counterfactual search or a planner; PPO learns selection and timing.

R3 has the greatest capability and engineering cost. It is authorized only after R1/R2 feasibility evidence shows that the B+ residual action space is insufficient.

## 6. Experiment ladder

No stage may be merged with the next stage for convenience. Every stage has one causal question and a stop rule.

### 6.1 D0: canonical collision audit and data locking

D0 performs no training. It creates the data foundation for all later work.

Required outputs:

1. `input_provenance.tsv`
   - run/tag;
   - result and NPZ roots;
   - checkpoint SHA256;
   - evaluation protocol/source SHA;
   - validation flag.
2. `episode_occurrences.tsv`
   - one row per original grid occurrence;
   - model, map, grid, offset, raw key, indices, racelines, speedscale, interval, outcome, collision flags, NPZ path/hash.
3. `canonical_episodes.tsv`
   - one row per canonical scenario identity;
   - duplicate group and development-overlap markers;
   - outcome-conflict assertion.
4. `collision_events.tsv`
   - any/ego-involved/opponent-only;
   - car/wall/unknown;
   - pre/alongside/post/unknown terminal phase;
   - direct simulator flags separated from inferred classifications.
5. `d0_summary.json` and `d0_summary.md`
   - map x raceline x speedscale x skill counts;
   - attempt, terminal overtake, confirmed pass;
   - collision transition summaries;
   - opponent-only empirical floor.
6. `d0_validation.json`
   - expected and observed counts;
   - duplicate, missing, extra, stale, conflict, and unknown-classification counts;
   - input and analysis hashes.

Canonical scenario identity must be derived from resolved map/raceline/start/pose/speed/interval/duration/noise fields. Offset and tag are provenance, not identity.

D0 confirms or revises the `skill_F` and `skill_S` manifests. It does not label them as causal skills until trajectory diagnostics support that label.

D0 stops the branch if:

- counts cannot reconcile with validated P1 results;
- duplicate scenarios have conflicting deterministic outcomes;
- either skill manifest contains fewer than 30 ego-involved collision cases for an independent pilot;
- required NPZ terminal fields are missing.

### 6.2 D2: episode-held-out representation probe

D2 performs no PPO training.

#### Feature replay

- Freeze the current BC checkpoint and all parameters.
- Replay each 100 Hz episode causally with hidden reset at episode start.
- Save actor features at time `t` using only observations available through `t`.
- Do not include critic privileged features in probe inputs.

#### Grouping and splits

- Group by canonical scenario and neighboring start-region block.
- Keep every timestep from one episode in one split.
- Keep near-duplicate starts and scenario variants in the same group.
- Use grouped nested cross-validation on the non-test data for model-family selection and threshold setting.
- Reserve one final probe test split that is opened once for the selected model family only.
- Fit normalization, balancing, and thresholds only on training/validation data.
- Mark future-horizon labels crossing an episode boundary as censored; do not label them negative.

#### Targets

- current longitudinal closing rate;
- corridor TTC, capped for numerical stability;
- ego collision within 0.5, 1.0, and 2.0 seconds;
- any-agent collision within 0.5, 1.0, and 2.0 seconds.

A 3-second horizon is diagnostic only because censoring is substantial in an 8-second episode.

#### Probe sequence

1. prevalence/constant baseline;
2. linear probe on frozen features;
3. capacity-matched MLP on frozen features;
4. causal short-history temporal module, only if the MLP fails the gate on grouped cross-validation predictions.

#### Metrics

- AUCPR;
- Brier score;
- expected calibration error and reliability plot;
- recall at a fixed safe-episode false-alarm rate;
- fraction of collision episodes warned at least 0.5/1.0/2.0 seconds before impact;
- safe-episode false-alarm rate;
- TTC MAE in the `TTC < 2 seconds` region;
- performance by map, skill stratum, raceline, and speedscale.

#### Pre-registered gate

The model family passes branch selection when all conditions hold on grouped out-of-fold predictions, and must later pass the same conditions on the unopened grouped test:

- 1-second ego-collision recall at least 60% with safe-episode false-alarm rate at most 10%;
- 2-second ego-collision recall at least 40%;
- Brier skill score `1 - Brier_model / Brier_prevalence` at least 0.10;
- TTC MAE at most 0.3 seconds in the `TTC < 2 seconds` region;
- at least 30 independent held-out ego-collision episodes are present.

Branch selection uses grouped out-of-fold predictions before the final probe test is opened:

- MLP passes the gate: select pure frozen-feature B+;
- MLP fails and the temporal module passes: select the temporal module;
- both fail: stop D3 and revisit observation/action representation instead of adding reward terms.

After branch selection, fit the selected family on all non-test data and open the final probe test once. If the selected family fails the same gate on final probe test, stop D3; do not switch families and reuse that test.

### 6.3 D2.5: counterfactual recoverability oracle

D2.5 performs no learning. It estimates the reachable safety/performance frontier of the proposed residual action space.

For each ego-involved BC collision in the training/development pool:

1. replay the deterministic scenario from episode start;
2. branch at selected points 1-3 seconds before the original impact;
3. apply a bounded library of coherent macro interventions;
4. return to BC/no-op or continue the intervention according to the branch definition;
5. run to the normal 8-second endpoint.

The branch library covers:

- no-op;
- brake magnitudes up to 1.0 m/s;
- small left/right steering residuals within the existing steering budget;
- brake-plus-steer combinations;
- intervention durations from 0.1 to 0.5 seconds.

Every branch is classified as:

- collision to confirmed safe pass;
- collision to terminal overtake without confirmed hold;
- collision to safe abort/follow;
- still collision;
- invalid or action-clipped.

The oracle is privileged and training-only. A final policy may use only deployable observations.

Route R2 is feasible when:

- at least 25 distinct canonical collision cases have a confirmed-safe-pass recovery branch;
- recoveries span at least two maps, at least five start-region blocks, at least five `skill_F` cases, and at least fifteen `skill_S` cases;
- at least 30% of ego-involved `skill_S` collision cases tested have a confirmed-safe-pass branch;
- branches do not require positive speed residual or action clipping.

If safe resolution is dominated by follow/abort and confirmed-pass recovery is rare, R2 does not satisfy the anti-conservative objective. The project must either expand the action space, introduce hierarchical options/teacher behavior, or reduce the claimed target before PPO training.

### 6.4 D3a: macro residual mechanics

D3a changes only residual decision cadence and replay semantics. Speed mapping, reward, objective, and scenario sampler remain at their current baseline settings.

#### D3a-1: coherent exploration with time-equivalent credit

Use `K = 10` micro-steps per macro decision:

```text
Gamma = gamma_micro ^ K = 0.997 ^ 10
Lambda = lambda_micro ^ K = 0.99 ^ 10
R_macro = sum(i=0..K-1) gamma_micro^i * r_(t+i)
```

This preserves the current physical-time reward and GAE decay. It isolates the effect of coherent residual action persistence.

#### D3a-2: extend physical credit

Starting from a passing D3a-1 implementation, change only:

```text
Lambda = 0.99 per macro step
```

Keep `Gamma = 0.997^10`. This changes the one-second GAE trace multiplier from approximately 0.27 to approximately 0.67 while preserving the reward discount time constant.

#### Required tests

- `K = 1` reproduces the current rollout/replay/return fingerprint;
- zero residual at `K = 10` reproduces the current 100 Hz BC action sequence;
- one macro action generates one log-probability and one PPO ratio;
- partial macro transitions terminate correctly on collision;
- true termination and truncation use correct cost/reward bootstrap semantics;
- recurrent hidden reset and replay identity pass;
- macro reward aggregation matches a hand-computed trajectory.

D3a stops immediately on replay identity failure or unintended BC action differences. After the second seed is available, it also stops if either seed has `new collision >= fixed collision` on both pre-registered snapshots.

### 6.5 D3b: longitudinal hurdle policy

D3b changes only the speed residual distribution on the selected D3a parent. TTC shaping and curricula remain disabled.

Mechanism gates:

- deterministic initialization is exact BC;
- the no-op atom and brake branch have correct conditional log-probabilities;
- sampled, expected, deterministic, and executed speed residuals reconcile;
- on frames with `front_risk > 0.1`, the deterministic gate selects brake on at least 10% of frames and the mean expected physical residual is at most `-0.05 m/s`;
- on frames with `front_risk < 0.01` and `side_risk < 0.01`, mean expected brake magnitude is below `0.02 m/s`;
- on the representative set, mean episode speed decreases by no more than 2% and mean progress decreases by no more than 1%;
- two fixed seeds both activate conditional braking;
- no positive speed residual is possible.

If braking remains unused on two pre-registered snapshots, stop. Do not add TTC shaping to conceal a distribution or gradient error.

### 6.6 D3c: lexicographic constrained PPO

D3c replaces the scalar safety trade-off with a constrained objective. It does not add TTC shaping or curriculum.

Training maintains separate episode signals and advantages for:

- any-agent collision cost;
- ego-involved collision cost breakdown;
- terminal overtake/confirmed pass;
- optional dense progress diagnostics.

The primal policy minimizes collision cost. A nonnegative dual variable increases pressure on overtake return only when the running overtake estimate falls below the pre-registered BC floor. Collision and overtake advantages are normalized separately. Actor and critics use separate optimizers or, at minimum, separate clipping operations and logged pre-clip norms.

The final checkpoint selector remains lexicographic even if training uses a smooth dual update:

1. reject any candidate that fails overtake protection;
2. among survivors, minimize the start-cluster bootstrap upper bound on collision relative risk;
3. break ties in favor of the earlier snapshot.

### 6.7 D3d: optional TTC potential

D3d runs only if D3b proves that the brake branch is learnable but D3c shows that recoverable risk states still assign non-helpful brake advantage.

The new signal must be a potential, active only when ego is behind by at least one car length and is longitudinally closing inside the current lateral-overlap corridor. It must not tax the alongside passing corridor already implicated by historical reward mismatch.

Preflight requirements:

- coefficient zero is bit-equivalent to D3c;
- macro shaping uses the same `Gamma` as the return;
- true terminal potential is zero;
- truncation uses the correct next potential;
- telescoping is verified on hand-built and recorded trajectories;
- potential scale is set once from the D0 training split: choose the coefficient so the 95th percentile absolute one-macro-step shaping term on active risk frames equals 25% of the 95th percentile absolute non-collision macro reward. Lock the resulting coefficient before D3d outcomes are opened; do not sweep it.

The mechanism gate requires the brake option to have higher estimated advantage than no-op on at least 60% of D2.5 recoverable pre-impact frames, while retaining the D3b representative-set speed, progress, and overtake guards. Failure stops TTC work.

### 6.8 D3e: two-skill curriculum

Curriculum is added only after macro mechanics, longitudinal action, constrained optimization, and any optional shaping pass their mechanism gates.

From the same D3 parent, run independent pilots:

```text
D3e-F: 50% representative scenarios + 50% skill_F
D3e-S: 50% representative scenarios + 50% skill_S
```

Do not initialize `D3e-S` from `D3e-F` or vice versa. Only after both improve their own skill development sets without harming the preservation set, run:

```text
D3e-FS: 50% representative + 25% skill_F + 25% skill_S
```

Curriculum manifests preserve the complete scenario tuple: map, ego/opp raceline, resolved starts, speedscale, interval, duration, and noise. Location-only hard replay is not permitted.

Each update logs complete episode counts from each sampler component. Timestep shares are not accepted as episode coverage.

## 7. Pilot sizes, promotion, and stopping

Every D3 stage begins as a mechanism pilot:

1. one fixed seed, 20 iterations, one pre-registered snapshot;
2. if the mechanism gate passes, extend to iteration 40 and add a second fixed seed;
3. evaluate only the locked small development manifests;
4. do not claim significance from a mechanism pilot.

Each small development manifest contains 96 episodes for:

- `skill_F`;
- `skill_S`;
- representative/preservation scenarios.

A stage is promoted only if:

- both seeds show more fixed collisions than new collisions;
- lost overtakes do not exceed gained overtakes on the representative set;
- BC-safe-overtake losses do not exceed gained terminal overtakes on the representative set;
- at least one recoverable BC collision converts to a confirmed safe pass;
- speed residual is risk-conditional rather than globally negative;
- replay identity, saturation, clipping, and completeness checks pass.

Immediate stop conditions:

- two consecutive pre-registered snapshots have `new collision >= fixed collision`;
- lost overtake exceeds gained overtake by at least two episodes on a 96-episode development manifest;
- collision improvement is explained entirely by disappearance of interaction attempts;
- two seeds move in opposite safety directions;
- longitudinal hurdle policy never produces deterministic/mode corridor braking;
- evaluation or manifest integrity fails.

Only a complete 20/40-iteration ladder may authorize one 80-100 iteration medium confirmation. No 300-iteration run or broad hyperparameter sweep is authorized by this design.

## 8. Anti-conservatism metrics

Historical outcome metrics remain primary, but the following mechanism metrics are mandatory.

### 8.1 Interaction attempt

An episode contains an interaction attempt when the longitudinal separation satisfies `abs(delta_s) <= 0.6 m` at any timestep. The exact geometry implementation must use the same reference projection on BC and candidate.

Attempt rate is diagnostic, not a hard invariant: refusing an unrecoverable attempt can be correct. A large attempt-rate drop must nevertheless be disclosed and explained.

### 8.2 Terminal overtake

Keep the repaired historical final-lead overtake outcome as the hard non-inferiority KPI.

### 8.3 Confirmed safe pass

Report a stricter secondary pass:

- ego lead at least 2 m;
- lead held throughout the final 0.7 seconds;
- no collision.

### 8.4 Paired transition matrix

For every BC/candidate matched scenario, report:

- BC collision to candidate confirmed pass;
- BC collision to candidate terminal overtake only;
- BC collision to candidate follow/abort;
- BC collision to candidate collision;
- BC safe overtake lost;
- BC safe follow converted to collision;
- new candidate overtake.

Report transitions separately for `skill_F`, `skill_S`, representative scenarios, maps, racelines, and speed bins.

## 9. Dataset registry and split integrity

Create an append-only `opened_registry.tsv` before D2. It registers every scenario used for:

- training;
- probe fitting or architecture choice;
- reward/action/curriculum selection;
- snapshot or seed selection;
- historical P1 analysis;
- final evaluation.

A probe test used to choose pure B+ versus a temporal module is architecture-development data, not final data.

Every evaluation run uses three frozen artifacts:

1. `scenario_manifest.tsv`
   - protocol version;
   - split;
   - canonical scenario ID;
   - map, resolved indices/poses, racelines, speedscale, interval, duration, noise/seed.
2. `models.tsv`
   - BC and candidate paths and SHA256;
   - exactly one final candidate before final results are opened.
3. `jobs.tsv`
   - Cartesian product of scenario and model IDs;
   - expected count and output location.

Preflight assertions:

- scenario IDs are unique across the whole manifest, not merely within each grid;
- BC and candidate scenario sets match exactly;
- development and final manifests have zero intersection;
- manifest rows resolve to the expected episode key;
- source and manifest hashes are archived;
- P0 completeness validation is enabled.

## 10. Final evaluation design

### 10.1 Austin final pool

Do not use `off10/31/52/73`: their 200 start occurrences contain only 108 unique starts and overlap the old development set.

`off11/32/75/86` is an admissible candidate under the currently known history:

- 200 unique starts;
- no intersection with historical off0 or P1 off21/42/63/84 starts.

This offset set is not trusted by name. The resolved 2400-scenario manifest must still pass full uniqueness and zero-intersection assertions before it is locked.

### 10.2 Known-cross-map confirmation pool

Nuerburgring, MoscowRaceway, and Hockenheim have already been opened in P1 and in this design analysis. Future evaluation on these maps must use new canonical starts that exclude every opened key.

Such a result may be described as:

> episode-held-out confirmation on previously opened maps

It must not be described as unseen-map validation.

The current repository has raceline0/1/2 assets only for Austin and these three maps. A genuine new-map full-protocol claim requires separately generating and pre-registering equivalent multi-raceline assets for additional maps; that work is outside B+.

### 10.3 Statistical hierarchy

Use a single frozen candidate and hierarchical gates:

1. any-agent collision effect and paired superiority;
2. overtake non-inferiority;
3. map/skill protection guards;
4. mechanism diagnostics.

Primary confidence intervals use paired bootstrap clustered by `(map, canonical start block)`. Episode-level exact McNemar/sign tests are reported as secondary because the 12 raceline/speed variants from one start are geometrically correlated.

Final safety gate:

- Austin fresh pool `RR <= 0.70` and one-sided 95% upper bound `< 1`;
- known-cross-map fresh pool `RR <= 0.70` and one-sided 95% upper bound `< 1`;
- each map has a lower collision point estimate than BC and satisfies a `+1 percentage point` non-inferiority guard.

Final performance gate:

- candidate overtake count is at least BC in each pooled family;
- paired one-sided 95% lower bound is greater than `-1 percentage point`;
- OL0/OL2 and per-map point estimates do not show an unreported collapse;
- BC-safe-overtake preservation and confirmed-pass metrics are reported.

The final set is opened once. A failed candidate may not be replaced and re-tested on the same final manifest.

## 11. Implementation integrity requirements

Before any training, tests must cover:

- macro action/log-prob one-to-one accounting;
- macro reward and cost aggregation;
- partial macro termination;
- termination/truncation bootstrap for reward and cost critics;
- hurdle distribution log-probability and entropy;
- exact deterministic BC equality at initialization;
- frozen backbone gradient and parameter isolation;
- recurrent replay identity;
- actor/critic separate pre-clip norms and clipping;
- sampled/mean/executed residual logging;
- scenario manifest uniqueness and split isolation;
- evaluation JSON/NPZ completeness under the repaired P0 protocol.

No test or pilot may silently fall back to the historical exit-code outcome protocol.

## 12. Documentation corrections required before experiments

The following project records contain stale or overstated conclusions and must be corrected before they are cited as experimental authority:

1. Claude memory `end2race-ppo-primary-kpi.md`
   - replace occurrence-level P1 denominators with canonical unique, development-disjoint counts;
   - replace "OL1 safety is zero-cost" with a low-overtake/high-collision exposure statement;
   - separate longitudinal-closing and alongside/abort hypotheses;
   - replace deterministic mean `dspeed = 0` claims with the narrower statement that sampled braking was not logged;
   - remove the claim that macro transitions automatically change physical credit decay;
   - remove claims that the old leaked nonlinear probe excludes representation limits;
   - mark the research phase as reopened while retaining cand160 as the current deployed checkpoint.
2. `logs/ppo_audit_handoff_20260710.md`
   - augment sections 14.2-14.4 with cross-grid overlap and development-overlap correction;
   - replace P1 pooled independent claims with the canonical N=1260 Austin analysis;
   - qualify causal collision-family labels in section 14.5.
3. Claude memory `end2race-p1-validation-run.md`
   - retain the 28/28 completeness result;
   - remove the claim that cand120 remains double-significant after three-candidate correction under canonical de-duplication;
   - distinguish occurrence totals from independent scenario totals.
4. `logs/p1_final_report_20260710.md` and `logs/final_model_report_20260710.md`
   - retain the original run record for provenance;
   - append, rather than overwrite, the canonical de-duplicated reanalysis and its corrected claim boundaries.

These documentation repairs do not change the currently deployed cand160 checkpoint. They change only the strength and denominator of historical claims.

## 13. Execution and authority boundaries

- This document authorizes no code changes and no experiment launch.
- Training and batch evaluation, once separately authorized, run only on the remote server.
- Local work remains limited to code/document review and read-only analysis unless the user explicitly expands scope.
- A separate implementation plan is required before modifying `train_ppo.py`, model definitions, evaluators, or scripts.
- Final manifests and checkpoint hashes must be reviewed before any final result is opened.

## 14. Deliverables by gate

| Gate | Required deliverable |
|---|---|
| Design | this approved specification |
| D0 | canonical datasets, collision audit, manifests, provenance, validation report |
| D2 | grouped probe report, locked splits, model comparison, branch decision |
| D2.5 | recoverability matrix, branch library, feasible-action frontier |
| D3a | macro semantics tests and two-step coherence/credit pilot |
| D3b | hurdle-policy tests and conditional-braking mechanism report |
| D3c | constrained-PPO diagnostics and lexicographic dev selection report |
| D3d | optional TTC invariance/mechanism report |
| D3e | independent F/S and combined curriculum reports |
| Medium confirmation | two-seed 80-100 iteration result and stop/go decision |
| Final | locked manifests/models/jobs, repaired aggregation, paired clustered report, model card |

The terminal design principle is:

> Preserve the validated BC behavior by default, intervene coherently only when risk is observable, prove that safe overtaking is reachable before training, and optimize collision reduction under an explicit overtake floor rather than hoping a scalar reward discovers the desired trade-off.

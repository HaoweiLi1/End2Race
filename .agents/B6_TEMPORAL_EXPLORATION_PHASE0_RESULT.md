# B6 temporally coherent exploration phase-0 result

Status: **INTEGRITY PASS; SCIENTIFIC NO-GO; LEARNER UNRUN**

Date: 2026-07-14

## 1. Decision

The no-learning comparison completed all 1,440 frozen training-only episodes.
The intended temporal mechanism was present, but the direct task outcome was
adverse:

```text
AR(1) produced 8 additional collision repairs out of 240 collision pairs,
but also 48 additional safe-to-collision transitions out of 480 safe pairs
and 17 additional lost overtakes out of 240 overtake pairs.
```

The prospective collision-repair effect and cluster-evidence gate failed;
both safe-behavior gates failed strongly. The final decision is:

```text
rho=0.95 equal-marginal-std AR(1) phase-0: NO-GO
AR(1) PPO learner:                         UNRUN
Austin 600 / seed0 / sealed data:          UNTOUCHED
```

This closes the proposed AR(1) learner and, under the external stopping
decision, closes further tuning of the current frozen-feature direct-head
line. It does not prove that all temporally coherent exploration, all plain
`End2Race` PPO, representation adaptation, or bounded macro safety correction
must fail.

## 2. Exact valid boundary

| Item | Identity |
|---|---|
| corrected implementation | `5a4c48f2debb8f4dd58807c966d47635408698d9` |
| immutable execution source | `677ab3a75070f7ef5d685ad34e987f65c99893b3` |
| replacement RunPlan SHA256 | `4a3923dbe2cf87073aa0aadb0bc59d8d8222882c107cb3d31c5e50f275dbbe7f` |
| selection digest | `7224f1f3da6a35febc50392cc35b4844076c77094f508d78dbe7b9b3fafb93fd` |
| selection file SHA256 | `7612d66ecd4708fb20905266c88033e5ffe6dd903aae4cdca310d7d1ae7c2b44` |
| canonical BC SHA256 | `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4` |
| remote root | `/home/haowei/end2race_analysis/b6_temporal_phase0_v3_677ab3a` |

The remote run used `DISPLAY=:1`, one CPU simulator queue, and the RTX 4080
SUPER for canonical BC inference. It produced 1,440 atomic episode JSON files,
720 complete iid/AR(1) pairs, no partial files and one atomic `COMPLETE`.
Local summarization from the copied raw rows reproduced the remote
`summary.json`, `paired_results.tsv`, `report.md` and `COMPLETE` byte for byte.

No candidate actor, optimizer state or learner checkpoint was created.

## 3. Population and estimands

Only the frozen Task-8 1,640-row training population was used. The deterministic
selection found exactly 60 L4 identities containing all three archived BC
outcomes and selected one hash-ranked L2 per `(L4,outcome)`:

```text
60 L4 x (collision + overtake + follow) x 4 innovations x (iid + AR1)
= 1,440 episodes
```

The same keyed standard-Normal innovation was used at each common
`(L2,seed,step)`; arm order was independently hash balanced. Simulator, GRU
and actor mean remained at 100 Hz. The only scientific contrast was:

```text
iid: epsilon_t = sigma * xi_t
AR1: epsilon_t = .95*epsilon_(t-1) + sqrt(1-.95^2)*sigma*xi_t
```

Collision repair, safe-to-collision harm and lost overtake use corrected
terminal outcomes. Repeated innovations/outcomes were aggregated within L4
before sign-flip/bootstrap inference.

## 4. Integrity and mechanism

Every integrity check passed:

| Check | iid | AR(1) | Gate |
|---|---:|---:|---:|
| noise lag-1 steer | `-0.000469` | `0.949869` | iid abs `<=.02`; AR1 `[.93,.97]` |
| noise lag-1 speed | `-0.000310` | `0.948965` | same |
| empirical steer std | `0.029983` | `0.029934` | within 5% of `.03` |
| empirical speed std | `0.199822` | `0.197920` | within 5% of `.20` |
| framewise max `abs(ratio-1)` | — | `0` pooled | `<=1e-6` |
| batched max `abs(ratio-1)` | `3.48e-5` | `1.47e-4` | `<=1e-4` / `<=3.3e-4` |
| speed projection | `0` | `0` | reported |
| steer projection | `679/531391` | `687/512494` | reported |

Temporal coherence was not merely nominal. At a 50-step/0.5-second window,
the RMS mean perturbation was:

| Dimension | iid | AR(1) |
|---|---:|---:|
| steer | `0.004230 rad` | `0.021154 rad` |
| speed | `0.027844 m/s` | `0.138888 m/s` |

Thus equal per-step marginal standard deviation does not mean equal
low-frequency maneuver energy. AR(1) created roughly five times the sustained
0.5-second mean displacement. That is the mechanism being tested, and it was
implemented successfully.

## 5. Direct outcome result

### 5.1 Archived collision scenarios

| Arm | collision | follow | overtake | repaired |
|---|---:|---:|---:|---:|
| iid | `124` | `54` | `62` | `116/240` |
| AR(1) | `116` | `52` | `72` | `124/240` |

AR(1) minus iid repair was `+8/240 = +3.33 pp`, below the prospective `+12`
pair/5 pp floor. Occurrence discordance was `45 positive / 37 negative`;
McNemar two-sided `p=0.4397`. The L4 sign-flip one-sided probability was
`0.2620`, and the cluster bootstrap 95% interval was `[-5.42,+12.08] pp`.

The primary repair gate therefore failed. This is not evidence of a stable
collision-repair advantage.

### 5.2 Archived safe scenarios

Safe-to-collision counts were:

| Archived group | iid collision | AR(1) collision |
|---|---:|---:|
| follow (`n=240`) | `10` | `40` |
| overtake (`n=240`) | `5` | `23` |
| pooled safe (`n=480`) | `15` | `63` |

The paired harm difference was `+48/480 = +10.0 pp`. Occurrence discordance
was `52 adverse / 4 favorable`; McNemar two-sided `p=1.10e-11`. L4 sign-flip
one-sided was `4.19e-9`, and the cluster bootstrap 95% interval was
`[+7.08,+13.13] pp`. The one-sided 90% upper bound was `+12.08 pp`, far above
the `+2 pp` non-inferiority margin.

Overtake retention also worsened. iid retained `230/240`; AR(1) retained
`213/240`. The paired lost-overtake difference was `+17/240 = +7.08 pp`, with
L4 sign-flip one-sided `p=0.000473` and cluster 95% interval
`[+3.33,+11.25] pp`. This fails both the point and 5 pp upper-bound gate.

## 6. Interpretation

The phase-0 prediction was falsified. At fixed per-step marginal std, temporal
correlation did make perturbations persistent and did repair a few additional
collision cases, but it made BC-safe closed loops substantially less safe and
lost overtakes. The similar, rare steering projection rates and zero speed
projection rule out action clipping as the explanation.

The result is narrower than "temporal coherence is bad." It rejects this
specific combination:

```text
canonical frozen-feature BC actor
full-action Gaussian perturbation
rho=0.95
std=(0.03,0.20)
unconditional application at every 100 Hz state
```

Reducing AR(1) marginal std to equalize window energy, conditioning correlated
noise on risk, using bounded macro corrections, changing rho, or learning a
representation are new scientific mechanisms. None may be inferred as a
remediation arm from this result, and none was run.

Under the owner's stated stopping rule, the next legal action is an external
review of this exact packet and a new prospective path decision between
representation adaptation and an explicit bounded macro safety-control
proposal. Do not start an AR(1) learner, tune rho/std, resume B5, run seed0, or
open Austin/sealed data.

## 7. Evidence

Compact evidence is under `docs/ppo/evidence/b6_temporal_phase0_v3/`:

| Artifact | SHA256 |
|---|---|
| `summary.json` | `3c7bf88e3aacc6b7a5516d7141cf3433eb4bbd0d46bde3daf75fe101057742d7` |
| `paired_results.tsv` | `c3e8397affd80c3617a7f5309307d803e37f83f7ac522dfee6499389c359bcfe` |
| `episode_results.jsonl` | `7b562be07f406df6b58a78c871515dd61bcc01e5fccd50fcde1fb7178f214fb7` |
| `report.md` | `5b3c3489bb55338211ce2cf01f662c39f224b02c8bbbe19ffe00b3b48defa18f` |
| `remote_run.log` | `a1317051152ff1fd19eca4ecfe7ec435ee8f9c9de87fa675ff949a3fa2a0fc9e` |

The two incomplete correctness attempts remain documented in the v1/v2
evidence directories and remote quarantine paths. They are not part of any
scientific count.

The exact B6 code-and-evidence content boundary is
`081092987877619e9b84f108f80cbebe3bda847c`. External review should inspect:

```text
9aeddcc3ecd4d7f896e5a00660c545d6176fba17
  ..081092987877619e9b84f108f80cbebe3bda847c
```

The following commit only binds this addressable review boundary.

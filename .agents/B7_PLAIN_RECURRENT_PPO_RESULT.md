# B7 plain recurrent PPO engineering result

Status: **INTEGRITY PASS; EARLY STOP; NO CANDIDATE; PPO LINE CLOSED**

Date: 2026-07-14

## 1. Decision

The remediated authoritative seed1 learner stopped after iteration 9 because
iterations 7, 8 and 9 were three consecutive actor-update rejections under the
prospective current-rollout BC-safe mean-KL cap. No iteration-10 actor was
created. Therefore no 288 evaluation, seed0 replication, Austin 600 evaluation
or sealed/final access was legal or performed.

```text
completed iterations:       9
actor steps attempted:      9
actor steps committed:      4 (iterations 1, 2, 3, 6)
actor steps rolled back:    5 (iterations 4, 5, 7, 8, 9)
terminal status:            EARLY_STOP_NO_CANDIDATE
iter10 deployment actor:    absent
candidate evaluation:       unrun
seed0 / Austin / sealed:    unrun / unopened / unopened
```

This is a valid negative result for the owner-approved B7 engineering protocol:
the configuration could not produce its sole candidate while satisfying its
own BC-preservation contract. It is **not** a numerical collision/overtake
failure, because no candidate reached the pre-registered evaluation boundary.

## 2. Exact valid boundary

| Item | Identity |
|---|---|
| implementation/remediation source | `3e262e2bf00acd8ef9338122a82780e68a825981` |
| documentation boundary before execution | `05d804c` |
| immutable RunPlan SHA256 | `3cd0f801f59609fcf6ab02a674851f49678de6b0fb04dc6a27201ff08c2672ad` |
| source archive SHA256 | `d2e0975b4e70f0f6c3cb407a5fbeba29da12e8914d9c3ec8792c7501a689264e` |
| inputs archive SHA256 | `1dda79806680e8299d3048477276c123291c468d181f2bef411f6524f7d6a18d` |
| canonical BC SHA256 | `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4` |
| remote stage | `/home/haowei/Experiments/staging/b7_seed1_20260714_114132` |
| local full release | `Experiments/B7_plain_recurrent_ppo/runs/b7_seed1_20260714_114132/remote/seed1` |

The clean read-only stage passed its four-map production smoke. The learner ran
under `DISPLAY=:1`, the RTX 4080 SUPER physical-GPU lock, isolated caches and
one-thread CPU-library limits from 19:43:09 to 19:58:51 Asia/Singapore.

The earlier staged source `1c096c2` exposed a stale critic-gradient cleanup bug
after iteration 1 and failed closed before a candidate. It was not resumed or
merged. The remediated source above adds a consecutive-iteration regression;
the authoritative run crossed that boundary normally. See
`docs/ppo/evidence/b7_stale_critic_gradient_20260714/`.

## 3. Iteration result

| iter | transitions | actor | safe KL | rollout KL | head LR before step |
|---:|---:|---|---:|---:|---:|
| 1 | 23,513 | accepted | 0.001707 | 0.001748 | `1.00e-5` |
| 2 | 21,301 | accepted | 0.006384 | 0.001467 | `1.00e-5` |
| 3 | 22,661 | accepted | 0.008799 | 0.000321 | `1.00e-5` |
| 4 | 23,883 | rolled back | 0.010652 | 0.000326 | `1.00e-5` |
| 5 | 23,027 | rolled back | 0.011198 | 0.000149 | `5.00e-6` |
| 6 | 23,056 | accepted | 0.009610 | 0.000030 | `2.50e-6` |
| 7 | 22,428 | rolled back | 0.010376 | 0.000016 | `2.50e-6` |
| 8 | 21,340 | rolled back | 0.010711 | 0.000009 | `1.25e-6` |
| 9 | 22,080 | rolled back; early stop | 0.011441 | 0.000001 | `6.25e-7` |

The run collected 288 complete episodes and 203,289 transitions. Every
iteration attempted exactly one actor Adam step and completed all three critic
epochs; the critic executed 162 minibatch optimizer steps in total. Rejected
actor updates restored the actor and full Adam state before the next-iteration
LR reduction.

Only `actors/iter_0000.pth` exists, and it is the canonical BC actor. There is
no `actors/iter_0010.pth`. The atomic `COMPLETE` content equals the
`summary.json` SHA256:

```text
96cbc9d8b2d04f111a4ed2b68aa18118dd04bc58cbef95b232e300eb2cbd5d0d
```

## 4. Interpretation

B7 corrected the earlier transition-reuse problem: 32 complete unique-L2
episodes per update produced one recurrent actor step, and the original GRU was
genuinely trainable. The run still did not yield a legal candidate.

The binding mechanism was cumulative BC-safe behavior preservation, not the
old-policy PPO trust region. By iteration 9, the attempted step's rollout KL
was only `7.16e-7`, yet mean KL from canonical BC on that iteration's archived
BC-safe observation histories was `0.01144`. This makes it unlikely that the
last step size was the main source of the excess; it is consistent with the
accepted iter6 policy already exceeding the cap on a newly sampled safe-state
set. The ledger does not separately record pre-update BC-safe KL, so that last
mechanistic statement remains an inference rather than a direct measurement.

This supports a bounded conclusion:

> Low-LR GRU adaptation, localized collision credit, more independent
> episodes, one recurrent step and current-policy hard mining did not overcome
> the conflict between the learned update direction and the current-rollout
> BC-safe cap in this protocol.

It does not show whether an unconstrained iter6/iter9 checkpoint would improve
the 288 collision/overtake KPIs, and those checkpoints must not be evaluated
post hoc. Selecting iter6 would violate the sole-candidate rule and create a
checkpoint lottery.

Per the owner-approved stop rule, do not tune the reward window, cap, LR,
episode count, action std or iteration count; do not run seed0. The plain
recurrent PPO engineering line is closed pending a new prospective decision on
deployment-contract relaxation or auxiliary risk-supervised representation
adaptation.

## 5. Evidence

Compact Git-tracked evidence is under
`docs/ppo/evidence/b7_plain_recurrent_negative_20260714/`. The 1.5 GiB full
release, including all recurrent replay and full checkpoints, remains in the
local experiment tree and the remote immutable stage.

| Artifact | SHA256 |
|---|---|
| `config.json` | `943211febbf32e1614cb87a115fab3e72aa1a57b41354a45589821a1edc7e9ef` |
| `iterations.jsonl` | `cc8cb983db27103140546dc2f91a6e8ec38f8fa4f88c6498d698f5c29342dab4` |
| `summary.json` | `96cbc9d8b2d04f111a4ed2b68aa18118dd04bc58cbef95b232e300eb2cbd5d0d` |
| `plumbing_smoke.json` | `54ed5406238746e45e37a7b8c719ae10be22b776dc219287fa249ebf02b0e03d` |
| `train_seed1.log` | `0be17c3026466899a59d5fb1717aa9d320669baaaf332753b9482b1a99508d9a` |

No evaluation rows exist because no candidate existed.

## 6. Later post-hoc evaluation cancellation

The owner later authorized a diagnostic Austin-600 evaluation of the last
accepted iter6 actor, then explicitly terminated it before completion. Local
shard0 stopped at 84/120 without `COMPLETE`; remote shards1-4 completed before
the stop arrived, but were never merged. The outputs are quarantined as
`ABORTED_BY_OWNER`. They are not a B7 evaluation result and do not modify the
`EARLY_STOP_NO_CANDIDATE` verdict.

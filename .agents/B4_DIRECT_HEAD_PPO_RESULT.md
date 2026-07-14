# B4 Plain-End2Race Direct-Head PPO — Execution and Product-Grid Result

Date: 2026-07-14 (Asia/Singapore)

Final verdict: **B4_SUBSTANTIVE_NEGATIVE**

Selected candidate: **none**

This is the externally reviewable numerical record for the single authorized
B4 hypothesis. It does not rewrite the prospective contract in
`.agents/B4_DIRECT_HEAD_PPO_PLAN.md`, and it does not authorize a B3 run or a
new B4/B5 experiment.

## 1. Frozen identities

| Item | Identity |
|---|---|
| B3 baseline | `4b06b7af0d6c84d45e688bd54478705ef021927f` |
| initial B4 implementation boundary | `ac99f5f406561664bb9c400735efd9e1f27591e3` |
| stochastic plumbing remediation | `bc0d81ece46c77a96c001d565e1de3bd8ffa030c` |
| immutable-stage fix and **training source** | `9e5afdc9584343a163c4704597dad87487bd750a` |
| RunPlan ID | `b4_seed1_20260714_003027` |
| RunPlan SHA256 | `08f0fe4275ae60928a6d5a6ce9704679bc91a624258bf5aef7f7a268b2c5e381` |
| product evaluator provenance fix and **evaluation source** | `cd5d4675fcbd9fe545e3684991cb932afa706dff` |
| evaluation source archive SHA256 | `6e94f5f5b6a9b40fd47f8a226bcfee9b1650765e1dbb4a863c99e4e58e17ab37` |
| post-run worker-cwd/provenance hardening | `292ffecc95312b8bbbe9b9dd17a6b1407f409903` |
| single-remote-job collection fix | `241a207cbb6e37222c82f7c300cb7ca5cc4edd0d` |

The learner used only seed1. It ran on the remote RTX 4080 SUPER with
`DISPLAY=:1` and `CUDA_VISIBLE_DEVICES=0`. The product evaluation used exactly
five startpoint shards: shard0 on the local RTX 3080 Laptop GPU and shards1–4
on the remote RTX 4080 SUPER. Thus local execution was exactly 480/2400 product
episodes (1/5), not a duplicate validation run.

## 2. Review-remediation evidence

Before training, both staged hosts passed the hash-bound baseline and CUDA
plumbing markers. The production-shaped stochastic smoke exercised one real
early any-agent collision and two product-horizon episodes end to end:

```text
raw Normal sample -> stored raw latent -> raw old log-prob
-> fixed actuator projection -> live simulator step
-> terminal-only -2*C+O -> complete-episode GAE
-> actor KL stop -> three critic epochs -> strict actor snapshot/full restore
```

Observed smoke facts:

- 2,242 real transitions;
- collision case terminated at 640 steps with reward `-2` and zero bootstrap;
- follow/overtake cases terminated at the literal 801-step product horizon;
- pre-update `max |ratio-1| = 3.0040741e-05 <= 1e-4`;
- a forced actor KL stop completed one actor epoch while the critic completed
  all three epochs;
- a `1e6` shaped-reward sentinel entered neither reward, advantage nor return;
- only `output_layer.*` and critic tensors changed; all frozen actor tensors
  remained exact;
- actor-only output strict-loaded as the canonical 12-key `End2Race`.

## 3. Training result

The remote job ran from 2026-07-14 00:42:47 to 01:06:27 +08:00 and completed
all 30 iterations without resume. The collected release reports
`passed=true`, `integrity_passed=true`, `fresh_pool_opened=false`.

| Training diagnostic | Observed |
|---|---:|
| completed iterations | 30 |
| actor epochs 1 / 2 / 3 | 3 / 1 / 26 iterations |
| critic epochs not equal to 3 | 0 iterations |
| mean / max weighted KL | 0.011416 / 0.082855 |
| stochastic speed projections | 0 |
| stochastic steer projections | 306 |

Actor snapshot file SHA256 values:

| Iteration | SHA256 |
|---:|---|
| 0 | `09b8eda3d38a3a13b389d13516ffba6dfe8739500a627d5dc233803d9273a273` |
| 10 | `271a875a7f8803d6606d1ba18b4d48fa4b1c91043758d8f637d349d87add4183` |
| 20 | `bb7258f056c39f60e798c683c1816afd8361e0611c0d70a1a3eee47c8cfa939c` |
| 30 | `cfbac4c045f58a30ef3199f83929adba7c0e2677bc5c830e5c666a7509e75f32` |

Every deployment snapshot contains exactly the canonical 12 state-dict keys
and 11,301,482 parameter elements. No critic, std, optimizer, residual, sidecar
or gate key is present.

## 4. Frozen product evaluation

The decision evaluation used the literal original BC grid on Austin:

```text
ego raceline: raceline1
opponent racelines: raceline0, raceline1, raceline2
opponent speed scales: 0.5, 0.6, 0.7, 0.8
startpoints: 50 from evaluate.sh's i*max_waypoints/(50-1)
interval index: 15
horizon: 8 s (literal 801-step loop)
episodes per variant: 3*4*50 = 600
variants: BC, seed1_iter10, seed1_iter20, seed1_iter30
```

All 2,400 metric rows and 2,400 NPZ files passed case uniqueness, shard
assignment, model hash, training provenance, outcome, and NPZ hash validation.
The BC overtake count was 342, so the prospective 5% floor was
`ceil(0.95*342) = 325`.

| Variant | Collision | RR | Overtake | Follow | Fixed / new C | Gained / lost O | Speed projection | Feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BC | 24 | 1.000 | 342 | 234 | — | — | 0 | baseline |
| seed1 iter10 | 24 | 1.000 | 332 | 244 | 11 / 11 | 8 / 18 | 0 | no |
| seed1 iter20 | 36 | 1.500 | 294 | 270 | 14 / 26 | 10 / 58 | 0 | no |
| seed1 iter30 | 39 | 1.625 | 296 | 265 | 14 / 29 | 12 / 58 | 0 | no |

Interpretation:

- iter10 passed the overtake floor but produced no net collision improvement;
  it moved which scenarios collided (`fixed=11`, `new=11`);
- iter20 and iter30 increased collisions and failed the overtake floor;
- no snapshot met even feasibility, so none can be called a directional
  survivor or product success;
- the result rejects this frozen-feature, output-head-only PPO configuration.
  It does **not** establish that residual policies work, that PPO is generally
  unsuitable, or that the frozen GRU representation is sufficient/insufficient.

## 5. Problems encountered and disposition

1. The first staging attempt (`...002701`) generated six pyc files during
   verification and was stopped before rollout. It is preserved; the stage
   bootstrap was fixed in `9e5afdc` and a new RunPlan was created.
2. One local BC product attempt used a hand-entered wrong source SHA and was
   stopped. Its partial directory is preserved as
   `BC/shard0.failed_wrong_source_sha_20260714_0041`. The evaluator now derives
   training provenance only from the signed RunPlan (`cd5d467`).
3. The first remote product launch resolved racetracks from the stage root and
   stopped before producing an episode. Four `*.failed_cwd.log` files are
   preserved. The valid run used a private writable work directory linked to
   the read-only source. Commit `292ffec` makes that isolation automatic for
   future runs.
4. The first official collection attempt assumed both staged hosts had learner
   status, although B4 intentionally had only one remote job. The 28 KiB failed
   partial was quarantined under `collection.attempt_failures`. Commit
   `241a207` makes status/output requirements conditional on actual host jobs;
   the second collection completed atomically.

None of these failures produced or selected a candidate, changed a checkpoint,
changed an evaluation case, or caused a completed case to be rerun under a
different model/configuration.

## 6. Evidence locations and hashes

- immutable RunPlan:
  `Experiments/B4_direct_head_ppo/configs/b4_seed1_20260714_003027.json`;
- atomic collected training release:
  `Experiments/B4_direct_head_ppo/runs/b4_seed1_20260714_003027`;
- product rows and per-episode artifacts:
  `Experiments/B4_direct_head_ppo/product_evaluations/b4_product_seed1_20260714_003027`;
- final summary SHA256:
  `2f98df4627e87d4e13ae0f415ceda91fb6104ca1d5ee3fde9eb03a56752f6159`;
- final paired rows SHA256:
  `d1d89d17e82311638742aade04600a1afe517360f809f051f2c99e5965dc1458`;
- final report SHA256:
  `84cddf4e6e338454bf5229452039a34ccc63b57d8db5a3aef772f4769f060ce5`;
- collected training summary SHA256:
  `286b1de828bf20d2fa9775812660098cf89e1c396d4a3062e098ed9fe5ea4e16`.

## 7. Stop decision and external-review question

The frozen stop rule is now active:

- do not run B3 automatically;
- do not add seed0;
- do not unfreeze the GRU;
- do not change reward/std/LR or add an anchor/dual;
- do not extend to 40/60 iterations;
- do not open fresh/final pools.

Before authorizing another experiment, an external reviewer should first use
the existing paired rows, replay files and snapshots to distinguish two facts:
(a) iter10 changed collision identity without net safety gain, and (b) later
updates caused both collision and overtake regression. Any B5 proposal must be
a new prospective owner decision with one isolated hypothesis; it must not be
presented as a continuation or rescue of B4.

## 8. Post-hoc cause analysis

That read-only analysis is now complete in
`.agents/B4_SUBSTANTIVE_NEGATIVE_ANALYSIS.md`, with reproducible compact tables
under `docs/ppo/evidence/b4_substantive_negative/`. It preserves this verdict.
The strongest measured signal is monotonic BC-relative action drift—especially
global slowing—without a cumulative BC trust region. Empirically iid 100 Hz
exploration and a 9.375x collision-prevalence shift are secondary supported
contributors. Frozen representation sufficiency and all counterfactual fixes
remain unresolved; no new run is authorized.

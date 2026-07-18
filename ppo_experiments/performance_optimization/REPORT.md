# End2Race PPO training pipeline performance report

## Decision

Phase 1–4 are retained. They pass the fixed-seed numerical contract and reduce one formal update from **65.271 s to 34.432 s (-47.25%)**, increasing total throughput from **392.21 to 743.49 transitions/s (+89.56%)**. The recurrent rollout buffer falls from **692.29 MiB to 200.10 MiB (-71.10%)**.

Phase 5 A/B/C is **not merged**. Every batched-GRU variant changes actor numerics, action/logp/hidden, and/or training gradients and parameter deltas. It remains an isolated experiment requiring an owner tolerance decision.

All performance numbers below use one unreported warm-up followed by the owner-requested single measured repeat. With one measured sample, the reported value is also the median; IQR is not defined.

## Locked baseline

The machine-readable lock is [BASELINE_LOCK.json](BASELINE_LOCK.json).

- HEAD: `ef8570bc522fb3b4dc6df2636bbe3a6e9afc4da4`; tree `57a9a0796f9088bf0d3da4a92bf5116fd38be54d`; worktree clean when locked.
- Work branch: `perf/ppo-training-pipeline-20260718`.
- Formal command: `/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/train_ppo.py --config N1-H1F-p50 --seed 20260917 --output_dir /home/haowei/Documents/End2Race/ppo_experiments/performance_optimization/formal_N1-H1F-p50_seed20260917`.
- Measured baseline command: `/home/haowei/miniconda3/envs/end2race/bin/python -u ppo_experiments/performance_optimization/profile_pipeline.py --config N1-H1F-p50 --seed 20260917 --mode dummy --label baseline_detailed_repeat1 --output ppo_experiments/performance_optimization/baseline_detailed_repeat1.json`.
- Config: `n_envs=16`, `n_steps=1600`, `batch_size=6400`, `n_epochs=1`, 25,600 transitions/update, C0 feed-forward critic, 8 hard env ranks + 8 ordinary env ranks, 8 s horizons, fixed hard/ordinary role assignment.
- Runtime: Python 3.10.19; PyTorch 2.7.0+cu128; CUDA 12.8; SB3/sb3-contrib 2.7.1; RTX 4080 SUPER 16,376 MiB, driver 580.159.03; Intel i5-14600KF, 14C/20T.
- BC checkpoint: `pretrained/end2race.pth`, SHA-256 `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4`.
- H1 manifest: `ppo/hard_pools/h1_expanded_det.json`, SHA-256 `ad35a6d56dddfe7c5e0460877f3aeb41ecd428d83c61ca1d1dc82c2b8709b0b8`.
- Scenario schema is locked in `BASELINE_LOCK.json`. The parent-visible reset scenario includes `scenario_id`, `env_role`, `sampler_branch`, `hard_pool_id`, `hard_sampling_mode`, map/raceline/start indices, speed scale, simulator parameters, and optional pair metadata. No worker-side fallback for missing `env_role` was added.

## Baseline profile

Top-level shares use total update wall time (65.271 s). Nested simulator and policy rows overlap their parent rows and are not additive.

| Component | Wall time (s) | Total update share |
|---|---:|---:|
| Rollout total | 42.895 | 65.72% |
| Environment step | 30.856 | 47.27% |
| Reset | 5.757 | 8.82% |
| Collection policy forward | 4.919 | 7.54% |
| Buffer add | 0.534 | 0.82% |
| GAE | 0.005 | 0.01% |
| PPO train total | 22.165 | 33.96% |
| PPO evaluate/forward | 6.143 | 9.41% |
| PPO backward | 15.626 | 23.94% |
| Optimizer | 0.020 | 0.03% |
| Sequence generation/padding | 0.182 | 0.28% |
| Checkpoint I/O | 0.125 | 0.19% |
| JSON I/O | 0.00027 | <0.01% |

Simulator detail: physics 1.284 s (1.97%), collision checking 0.537 s (0.82%), LiDAR 11.008 s (16.87%), observation construction 0.481 s (0.74%), progress/raceline projection 3.124 s (4.79%), opponent replan 12.015 s (18.41%), opponent tracking 0.170 s (0.26%), and repeated planner/static asset construction 5.399 s (8.27%).

Policy detail was synchronized around actor/forward/backward/optimizer boundaries. CPU enqueue attribution was 5.920 s for preprocessing/head, 3.125 s for GRU, and 0.082 s for critic; these overlap synchronized actor/train totals. The baseline logical padding ratio was `38,400 / 25,600 = 1.50`.

Baseline resources: process CPU mean 101.61%, system CPU mean 5.67%, GPU SM mean 33.97%, GPU memory activity mean 29.97%, RSS median 3,142.83 MiB, external GPU framebuffer median/max 852/1,962 MiB, PyTorch peak allocated/reserved 1,031.995/1,292 MiB, 57 median threads. Sampled process context-switch counters were voluntary mean/max 23,003/38,050 and involuntary mean/max 103/165; these are cumulative sampled counters, not per-second rates.

Copy estimate per update: collection observation H2D 36.97 MB, action D2H 0.20 MB, value/logp D2H 0.20 MB, four recurrent-state D2H copies 688.13 MB, and padded training observation H2D 55.45 MB. Measured observation H2D was 0.039 s; buffer add including recurrent D2H was 0.534 s.

Startup was 4.195 s: sampler 0.012 s, vector environment 3.411 s, model 0.652 s. Startup is excluded from steady-state update time.

## Independent phase results

PSS is used for subprocess configurations because summed RSS double-counts shared pages. The DummyVecEnv baseline has parent RSS only, so its RSS is not directly comparable to subprocess PSS.

| State | Startup (s) | Rollout (s) | Train (s) | Total (s) | Rollout t/s | Total t/s | Episodes/s | Buffer MiB | PSS / RSS median MiB | Torch peak alloc/reserved MiB | Padding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline DummyVecEnv | 4.195 | 42.895 | 22.165 | 65.271 | 596.80 | 392.21 | 0.816 | 692.29 | n/a / 3,142.83 | 1,031.995 / 1,292 | 1.50 |
| Phase 1, 6 workers | 2.723 | 21.659 | 22.272 | 44.143 | 1,181.98 | 579.94 | 1.616 | 692.29 | 6,751.88 / 8,602.17 | 1,031.995 / 1,292 | 1.50 |
| Phase 2, asset cache | 2.687 | 19.153 | 22.203 | 41.574 | 1,336.57 | 615.77 | 1.827 | 692.29 | 7,005.35 / 8,858.96 | 1,031.995 / 1,292 | 1.50 |
| Phase 3, actor-h buffer | 2.748 | 18.964 | 22.109 | 41.285 | 1,349.92 | 620.08 | 1.846 | 200.10 | 6,685.58 / 8,538.43 | 1,031.995 / 1,292 | 1.50 |
| Phase 4, skip padded actor | 2.777 | 19.024 | 15.182 | 34.432 | 1,345.69 | 743.49 | 1.840 | 200.10 | 6,228.28 / 8,077.74 | 832.715 / 1,092 | 1.50 |

### Phase 1: parent-scheduled subprocess environments — retained

Files: `ppo/config.py`, `ppo/vec_env.py`, `train_ppo.py`, and the external reset-spec path in `ppo/environment.py`.

- Parent retains the only `FixedMixtureScenarioSampler`, per-rank RNG streams, pair ordinals, visit counts, scenario queues, and deterministic rank-order scheduling. Workers contain no sampler and reject unscheduled resets.
- `forkserver` is started before CUDA initialization. Each worker has OMP/MKL/OpenBLAS/Torch intra-op/inter-op threads fixed to 1.
- Workers return all episode/scenario info. Normal close and an injected worker exception both left no live/zombie workers.
- Rollout improved 49.51%; total update improved 32.37%. IPC/serialization was only 0.348 s (1.61% of Phase-1 rollout), so shared memory was not implemented.

Worker sweep, one measured repeat each:

| Workers | Vec startup (s) | Rollout (s) | Total (s) | Total t/s | PSS median MiB |
|---:|---:|---:|---:|---:|---:|
| 6 | 1.889 | 21.659 | 44.143 | 579.94 | 6,751.88 |
| 8 | 2.229 | 23.166 | 45.930 | 557.37 | 7,837.74 |
| 12 | 2.317 | 22.143 | 44.565 | 574.44 | 10,381.56 |
| 14 | 2.435 | 22.009 | 44.617 | 573.78 | 11,508.68 |
| 16 | 2.349 | 22.777 | 45.393 | 563.97 | 12,817.64 |

Recommended configuration: **6 environment workers, `forkserver`, worker OMP/MKL/OpenBLAS/Torch threads=1**. Parent Torch remains at the original 14 intra/inter-op threads and original autograd multithreading setting.

### Phase 2: immutable planner asset cache — retained

File: the planner factory in `ppo/environment.py`.

The per-process template caches only immutable map/raceline/config assets. Every mutable planner and tracker field is reset to the values from a fresh constructor. Reset fell from 3.475 s to 1.046 s (-69.91%), rollout improved 11.57%, and total update improved 5.82%. No local-window nearest search was introduced because exactness risk was not justified after this gain.

### Phase 3: actor-h-only recurrent buffer — retained

Files: `ppo/buffer.py` and C0 buffer selection in `ppo/policy.py`.

Only real actor `h` is stored. Actor dummy `c` and critic `h/c` are lazily created as zero tensors for the unchanged batch API. Logical minibatch generation, split RNG, sequence starts, ordering, mask, advantages, returns, and padding remain stock-equivalent. Buffer memory fell 71.10%; buffer add fell from 0.550 s to 0.332 s; total update changed -0.69%, within the memory optimization threshold.

### Phase 4: skip invalid padded actor calls — retained

File: `ppo/policy.py`.

The unchanged sequencer supplies valid lengths. The policy skips only invalid tail positions while retaining every logical sequence, order, mask, advantage-normalization set, critic call, loss reduction, and optimizer-step boundary. Logical padding remains 1.50 by design; training actor GRU calls fall from 38,400 to 25,600. Train time improves 31.33%; total update improves 16.60%; PyTorch peak allocation falls 19.31%.

Final IPC estimate is 0.348 s, 1.83% of rollout and 1.01% of total update. Recurrent-state D2H estimate falls from 688.13 MB to 172.03 MB; observation copies remain unchanged. Final system CPU mean is 11.85%, GPU SM mean 45.77%, GPU memory activity mean 39.40%, and external framebuffer median/max 852/1,762 MiB.

## Accuracy evidence for Phase 1–4

Reference artifacts are [reference_v2_nonzero.json](reference_v2_nonzero.json), [reference_v2_zero_lr.json](reference_v2_zero_lr.json), and [reference_actions.npy](reference_actions.npy). Candidate artifacts are named `phase1_*`, `phase2_*`, `phase3_*`, and `phase4_*` in this directory.

- All four nonzero candidates match the locked reference exactly for all 51 reset specs/order, per-rank scenario/env_role sequence, sampler visits, every step hash, stochastic action/value/old-logp/actor-hidden/episode-start states, frozen rollout tensors, loss logger, per-parameter gradient hash, per-parameter delta hash, optimizer state, outcomes, role counts, and actor checkpoint.
- The 51 reset specs cover the requested 32–64-scenario stepwise regression range. Reset-order SHA-256 is `e657103581f646e2dc3d289caa886447fc8f93d679ac32acf5d6bff7d5abc059`; reset-spec SHA-256 is `cb08ebc3f35eab4e0736aa263aeda3af48d42459585d10edd709eb1605d4df4b`.
- Saved-action environment replay covers 1,600 vector steps / 25,600 actions and is all-PASS for observation, executed action, reward, terminated, truncated, info/outcome, reset order, and sampler state. Action SHA-256 is `8a361be0f698ee55664b0d137bdcf8f82ebd26ba263a154f35945cd918dd5be8`.
- Action/logp replay is exact (`old_logp_max_abs=0`). Value replay has the same pre-existing maximum difference `5.960464477539062e-7` before and after every phase; cross-version value tensors are exact.
- Episode-start mean/hidden/cell reset identity is exact.
- Every zero-LR candidate has strict zero parameter delta and exactly matches the baseline zero-LR optimizer state. The final optimizer hash is `3c49a0d678cde5f5d5fe038ee48a294087924cd61c4075f6d2f6ffe586b7e09a` in both baseline and candidate.
- The same frozen rollout produces exact logger losses, gradients, parameter deltas, and final optimizer state. Nonzero optimizer hash is `ba7652ec01b59a339082f6a8b3990b51d90f216de73acc7efe81bb94bd4e40ac`.
- A real nonzero update produces a strict-loadable original 12-key actor checkpoint, SHA-256 `44a3033cd75d16d329e43beca2f695e491fa14d49744a15a6ded442c68ed1c29`.
- Episode semantics remain exact: 13 ego-collision outcomes, 13 opponent-collision events, 13 overtakes, 9 follows, 22 timeouts/truncations, 13 terminations, and hard/ordinary transitions 12,800/12,800. Step contracts also include opponent-only collision latch and termination reason fields.
- Lifecycle evidence [phase1_lifecycle.json](phase1_lifecycle.json) passes normal exit, exception propagation, cleanup, and no-live-worker checks.

The cumulative full-600 regression compares the current worktree against a detached worktree at the exact locked HEAD using the same evaluator command and BC checkpoint. All 600 canonical rows are exact; both row sets hash to `950fb9f305a27c76b1d5cdf647f6d58d18d8dbee7825df0454bfe98a63e677ca`. Both summaries are collision 21, follow 233, overtake 346, error 0. The only top-level JSON difference is the absolute `model_path` in the temporary baseline worktree. A pre-existing historical full-600 artifact with a different result was rejected as stale and was not used as the comparator.

## Phase 5 batched-GRU experiment — not merged

Experiment file: `ppo_experiments/performance_optimization/experiment_batched_gru.py`. Evidence: [phase5_collection_a_closed_loop_repeat1.json](phase5_collection_a_closed_loop_repeat1.json) and [phase5_training_b_c_repeat1.json](phase5_training_b_c_repeat1.json).

| Variant | CUDA actor forward, warm-up then repeat-1 | Numerical result |
|---|---:|---|
| A, collection env batch | 0.610 ms vs 2.843 ms reference | sampled action max/mean abs `2.375e-3 / 7.578e-5`; hidden `4.819e-3 / 1.941e-5`; old logp `4.300e-3 / 9.035e-6`; closed-loop rollout/optimizer contract diverged |
| B, training timestep batch | 273.7 ms vs 1,080.8 ms reference first minibatch | mean action max `1.024e-2`; logp max `1.029e-1`; hidden max `5.361e-3`; approx KL `1.124e-5`; gradient max diff `3.716e-4`; parameter-delta max diff `1.215e-5` |
| C, packed full sequence | 31.70 ms vs 1,080.8 ms reference first minibatch | mean action max `1.024e-2`; logp max `1.029e-1`; hidden max `4.492e-3`; approx KL `1.128e-5`; gradient max diff `3.746e-4`; parameter-delta max diff `1.215e-5` |

B/C used the same frozen formal rollout, identical initial model/optimizer/RNG, and the same first logical minibatch: 6,400 valid positions, 8,800 padded positions, 11 sequences, no internal episode boundary. Value loss is exact because C0 critic is unchanged, while policy loss, total loss, gradients, and parameter deltas differ. A used matched RNG for reference/candidate sampled actions and then ran the full fixed-seed closed-loop contract; action/logp/replay identity failed as expected. Per the owner rule, all three variants remain outside the production policy.

## Retained, excluded, and remaining bottlenecks

Retained: Phase 1 six-worker central scheduling, Phase 2 immutable planner asset cache, Phase 3 actor-h-only C0 buffer, and Phase 4 invalid-padding actor skip.

Excluded or not implemented: 8/12/14/16-worker defaults (slower and larger), shared-memory transport (IPC only 1–2%), local-window/progress approximation (unnecessary exactness risk), length bucketing/stratified sampling (forbidden), and all Phase-5 batched/packed GRU paths (numerically non-identical). AMP, BF16, TF32 changes, `torch.compile`, CUDA graphs, and full-GPU rollout storage were not attempted.

Remaining steady-state bottlenecks after Phase 4 are approximately 19.0 s rollout and 15.2 s train. Within rollout, worker critical simulation remains dominated by opponent replanning, LiDAR, and progress projection; collection still performs exact batch-size-one actor GRU calls. Within train, exact batch-size-one GRU recurrence remains the main forward/backward cost. Logical padding remains 1.50, although invalid actor compute is skipped. The subprocess process tree uses about 6.23 GiB median PSS, materially more than DummyVecEnv's single-process memory.

## Final checks

- `python -m py_compile` passed for all modified production and validation modules.
- `python -m unittest discover -s tests -v`: 13/13 PASS.
- `git diff --check`: PASS.
- All generated JSON evidence parses with `jq`.
- No performance/validation/forkserver/resource-tracker process remained after the runs.
- No site-packages or evaluator file was modified.

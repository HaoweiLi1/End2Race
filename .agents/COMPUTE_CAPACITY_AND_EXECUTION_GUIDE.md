# End2Race local/remote compute capacity and execution guide

Updated: 2026-07-14 (Asia/Singapore)

This is an infrastructure-capacity record, not a scientific experiment. The
benchmarks used only opened Task-8 training `follow` scenarios, did not inspect
or compare outcomes, and did not use Austin, seed0, D2 test, fresh or final
data. Every process performed one unmeasured warm-up episode before timing.
The compact measurement ledger is
`docs/ppo/evidence/compute_capacity_20260714/throughput.tsv`.

## 1. Hardware actually available

| host | CPU | memory | GPU | GPU power limit | runtime/storage |
|---|---|---:|---|---:|---|
| local `wsl2-ubuntu2204` | i9-11950H, 8 physical / 16 logical | 47 GiB | RTX 3080 Laptop, 16 GiB | 90.17 W | WSL2, ext4, 819 GiB free |
| remote `haowei@192.168.2.127` | i5-14600KF, 14 physical / 20 logical | 94 GiB | RTX 4080 SUPER, 16 GiB | 320 W | native Ubuntu, NVMe, 850 GiB free |

Both hosts use PyTorch `2.7.0+cu128`, CUDA runtime `12.8`, and cuDNN `90701`
inside the `end2race` environment. The remote CPU topology is important:

```text
P-core hardware-thread pairs: (0,1) (2,3) (4,5) (6,7) (8,9) (10,11)
E-cores:                      12 13 14 15 16 17 18 19
```

For one single-threaded worker per P-core, use CPU IDs
`0,2,4,6,8,10`, not `0-5`.

## 2. Why one learner leaves both machines under-used

The observed low utilization is expected from the current implementation:

1. `bplus_v22/b4_runner.py` and `bplus_v22/b5_runner.py` collect the 16
   episodes of an iteration serially.
2. `bplus_v22/b4_env.py::run_b4_episode` owns one F110 environment and performs
   recurrent actor inference with batch size 1.
3. Every 100 Hz step immediately synchronizes CUDA back to Python through
   `.item()` and `.cpu().numpy()` for the command, feature, latent, log-prob
   and value ledgers.
4. F110 scan, collision and dynamics kernels are single-threaded Numba
   functions. The opponent planner is also driven by the episode process.
5. The PPO update is short relative to collection. The completed B4/B5
   30-iteration, 480-episode learners each took about 1,403-1,414 seconds on
   the remote host; their wall time is almost entirely episode collection.

Consequently, one learner normally occupies roughly one CPU core and sends
short batch-1 kernels to the GPU. Low average GPU utilization is not evidence
that the actor ran on CPU or that CUDA failed.

The current managed runner also prevents capacity use: it takes one exclusive
GPU `flock` for the whole host and runs `_host_jobs` sequentially with
`subprocess.run`. Independent learner jobs cannot be made concurrent by merely
adding more queue entries. Do not bypass this lock with hand-written SSH
commands in an authoritative run.

## 3. Measured capacity

### 3.1 GPU collector scaling

The production-shaped probe used stochastic full-horizon 801-step training
episodes, the canonical B4 collector and one CUDA policy per process. All BLAS,
PyTorch CPU and Numba worker thread counts were fixed to one.

| CUDA processes | local episode/s | local seconds/episode/job | remote episode/s | remote seconds/episode/job |
|---:|---:|---:|---:|---:|
| 1 | 0.140 | 7.15 | 0.329 | 3.04 |
| 2 | 0.269 | 7.34 | 0.585 | 3.41 |
| 4 | 0.447 | 8.46 | 1.026 | 3.76 |
| 6 | 0.529 | 10.22 | **1.323** | 4.33 |
| 8 | **0.578** | 12.87 | 1.227 | 4.96 |

Remote throughput peaks at six processes and regresses at eight. During the
6-8 process probes, remote SM utilization reached roughly 80-94%, while total
framebuffer use remained only about 3.4-4.3 GiB. The limiting resource is
small-kernel/synchronization scheduling and CPU service, not VRAM.

Local aggregate throughput still rises to eight, but individual job latency
is 80% higher than at one or two processes, and the 90 W laptop GPU/CPU reaches
the thermal/power-sharing regime. Eight local learner processes are therefore
not a safe long-running default.

### 3.2 Evaluation-shaped CPU scaling

This probe used deterministic 801-step episodes and additionally compressed
the full trajectory to NPZ and read it back for SHA256, matching the expensive
parts of the product evaluator. It is a capacity probe, not a replacement for
an exact evaluator smoke.

| CPU processes | local episode/s | remote episode/s |
|---:|---:|---:|
| 4 | 0.541 | 1.047 |
| 6 | 0.628 | — |
| 8 | 0.691 | 1.175 |
| 10 | — | 1.323 |
| 12 | — | **1.429** |
| 14 | — | 1.355 |

The remote CPU-only evaluator optimum is 12 workers; 14 workers regress. The
local maximum tested value is 8 workers, but 6 retains most throughput while
leaving headroom.

Historical exact `scripts/b4_product_eval.py`/B5 product shards used two CUDA
workers. A 120-case local shard took 311-335 seconds (0.36-0.39 episode/s),
while remote shards took 147-154 seconds (0.78-0.82 episode/s). This confirms
that the old two-worker topology left substantial CPU capacity unused.

### 3.3 Mixed training plus evaluation

Unpinned same-host concurrency hurts learner latency:

| topology | GPU collector episode/s | CPU eval episode/s | total episode/s |
|---|---:|---:|---:|
| remote, 4 CUDA + 8 CPU | 0.659 | 1.135 | 1.794 |
| local, 2 CUDA + 4 CPU | 0.210 | 0.431 | 0.641 |

Remote CPU affinity materially improves the mix:

| remote pinned topology | GPU collector episode/s | CPU eval episode/s | total episode/s |
|---|---:|---:|---:|
| 4 CUDA on P-cores `0,2,4,6` + 8 CPU on E-cores `12-19` | 0.931 | 0.909 | 1.840 |
| 6 CUDA on P-cores `0,2,4,6,8,10` + 8 CPU on E-cores `12-19` | **1.213** | 0.897 | **2.110** |

The second topology uses all 14 physical remote cores and keeps CUDA learner
throughput within about 8% of the isolated six-process value. It is the
measured maximum-total-throughput mode, not the minimum-latency mode.

Local pinning (`2 CUDA` on `0,2`, `6 CPU` on `4,6,8,10,12,14`) produced total
throughput 0.810 episode/s, but CUDA learner throughput fell to 0.199. This is
useful only when aggregate throughput matters more than learner completion
time.

## 4. Recommended operating profiles

### A. One authorized learner only

- Run it on remote CUDA.
- Accept low GPU utilization as an architectural property.
- Do not duplicate the same seed/job to fill the GPU.
- Do not parallelize the 16 episodes of one current PPO iteration without a
  separately reviewed vectorized on-policy collector.

### B. Several independent learner arms/seeds

| host | stable default | throughput mode | avoid |
|---|---:|---:|---:|
| remote | 4 concurrent learners | 6 concurrent learners | 8 or more |
| local | 2 concurrent learners | 4 concurrent learners | 6-8 for long unattended runs |

These counts apply only to genuinely independent, prospectively authorized
jobs with separate RNG, output, replay, checkpoint and cache paths.

### C. Evaluation only

- Prefer CPU evaluation; reserve GPU for learner/update work.
- Remote: `--device cpu --workers 12` after an exact short smoke; use 8-10 if
  the desktop must remain responsive.
- Local: `--device cpu --workers 8` for an eval-only window, or 6 for headroom.
- For a five-shard 600-case variant, keep the proven four-remote/one-local
  assignment. Run one shard command per host queue with internal workers;
  never launch overlapping processes that own the same shard/output path.
- Freeze device and worker count in the EvalPlan. Do not change them after
  seeing candidate outcomes.

`scripts/b4_product_eval.py` and `scripts/b5_opened_product_eval.py` already
expose `--workers`. The B2/B3 managed evaluator does not yet expose this CPU
parallel contract and requires a versioned runner/evaluator change first.

### D. Training and evaluation at the same time

For shortest learner completion time:

1. use the remote GPU only for learners;
2. send CPU evaluation to the local host;
3. copy immutable snapshots before evaluation and collect results only after
   each atomic shard is complete.

For maximum total throughput when both workloads must share remote:

```text
CUDA learners: taskset -c 0,2,4,6,8,10   (up to 6 independent jobs)
CPU evaluator: taskset -c 12-19           (one evaluator, 8 workers)
```

Use this only through a reviewed managed-runner resource profile. The measured
trade-off is approximately 8% lower learner throughput in exchange for about
0.90 additional evaluation episode/s.

## 5. Required environment and monitoring

Every managed worker process should receive:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1
```

Remote jobs additionally retain the existing contract:

```bash
export DISPLAY=:1
```

Worker initialization should also call:

```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

During a run, inspect actual processes and artifacts rather than trusting a
stale status label:

```bash
nvidia-smi dmon -s pucvmet -d 1
mpstat -P ALL 1
pidstat -u -r -d -C python 1
./run.sh status <plan.json> --all-hosts
```

Interpretation:

- one learner and low GPU use: expected;
- remote six learners and sustained 80-94% SM: saturated;
- increasing process count while episode/s falls: oversubscribed;
- high CPU `%iowait`: keep NPZ/results on the host NVMe and rsync only at
  stage/collect boundaries;
- high per-job latency with unchanged total throughput: reduce concurrency.

## 6. Runner work required before these profiles are authoritative

The next new experiment runner should add a reviewed resource contract rather
than shell-level bypasses:

1. Freeze per job: `device`, `cpu_workers`, `cpu_affinity`, `gpu_slot`, and
   thread-limit environment.
2. Keep one host controller under the existing physical-GPU lock, but allow it
   to launch a bounded number of independent queues concurrently.
3. Use a host capacity semaphore: remote CUDA slots default 4, maximum 6;
   local default 2, maximum 4.
4. Allow CPU evaluation jobs to declare `gpu_exclusive=false` and a disjoint
   affinity set; keep their atomic shard validation unchanged.
5. Preserve unique output directories, Numba/Python caches, RNG seeds,
   iteration ledgers and COMPLETE envelopes per job.
6. Fail closed if two jobs overlap an output path, reuse a seed identity, omit
   an affinity/thread contract, or exceed the frozen host capacity.
7. Benchmark the exact staged source with a short correctness-only smoke before
   a long run; capacity numbers are starting defaults, not timeless constants.

Do not enable CUDA MPS, mixed precision, `torch.compile`, 10 Hz action hold, or
batched recurrent inference in an authoritative run merely to raise GPU
utilization. Each can change execution/numerical semantics and requires its own
identity and determinism review. A future vectorized collector may batch
multiple 100 Hz environments, but it must preserve recurrent hidden masks,
raw-action probability ledgers, episode boundaries, terminal rewards and
episode-equivalent weighting.

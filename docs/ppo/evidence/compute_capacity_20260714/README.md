# Compute-capacity audit evidence

Date: 2026-07-14 (Asia/Singapore)

This directory records infrastructure throughput only. The probe used the
canonical B4 episode collector on opened Task-8 training `follow` scenarios,
discarded outcomes, used no Austin/seed0/sealed data, and wrote no scientific
result artifact. Each process completed one unmeasured warm-up episode before
the measured episodes.

The initial probe source SHA256 was
`5124daa3e73e459068e89972cf7dd51d0016a96c55c4ea010baa03de351452b7`.
It loaded one canonical BC/B4 policy per process, fixed PyTorch and numerical
library thread counts to one, and reported aggregate episode throughput after
synchronizing CUDA. `eval_npz_sha` rows additionally used deterministic
rollouts, `numpy.savez_compressed`, file readback and SHA256.

`throughput.tsv` is the compact aggregate ledger. `mixed=isolated` means no
other probe ran on that host. `mixed=concurrent` means the CUDA and CPU rows
with the same `pair` ran simultaneously. CPU affinity was applied by `taskset`
to the parent process and inherited by all spawned workers.

These measurements are capacity starting points. The exact staged source of a
new experiment still requires a short correctness-only benchmark, and all
resource choices must be frozen prospectively in its RunPlan.

The interpretation and operating profiles are in
`.agents/COMPUTE_CAPACITY_AND_EXECUTION_GUIDE.md`.

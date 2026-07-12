# Local Artifact Retention and Cleanup Record

Cleanup date: 2026-07-12. Scope: local workspace only. No remote path,
process, registry, checkpoint, or artifact is modified.

## Retained evidence

- canonical BC/cand160/cand120/cand040 checkpoints and their SHA256 ledger;
- P0/P1 reports and source archive;
- D0 v1 frozen audit and D0.1 canonical `_a` release;
- D2 canonical report/artifact and sealed-test metadata;
- D2.5 canonical report/full oracle;
- D2R-G canonical report/full grouped OOF artifact;
- all substantive B+ releases needed to explain fresh identity, sidecar
  initialization, warm-start PASS/FAIL history, Task 9, Task 10 FAILED, and
  hierarchical Task 6 FAILED;
- source packages, CLIs, registry, specifications, decisions, experiment
  ledgers, and structural tests.

## Removed generated data

- `eval_results/**`: local raw rollout mirror. Canonical counts, reports,
  manifests, hashes, and higher-level evidence artifacts are retained.
- historical checkpoint snapshots not in the four-checkpoint canonical set;
- Python/Numba test caches and bytecode;
- aborted-run payloads and `.partial` directories;
- explicitly superseded D0.1 duplicate/intermediate releases;
- explicitly superseded B+ source-preflight and Task-8 manifest copies;
- legacy per-run directories whose conclusions are already consolidated in
  P1/D0.1/current experiment records.

Removal is a local storage/organization decision. It does not invalidate,
promote, or rewrite an experiment. Where reproducibility requires a removed
large raw mirror, the original canonical report identifies the authoritative
remote release and hashes.

## Code retention decisions

- P0 evaluation-integrity changes in `eval_multiagent.py`, `evaluate.sh`,
  `evaluate_ol1.sh`, `aggregate_eval.py`, and `utils.py` are retained.
- isolated `d0/`, `d2/`, `d25/`, `d2r/`, and `bplus_v22/` packages are
  retained because canonical validators and future direct-PPO work depend on
  them.
- old D4 scenario-sampling changes in `train_ppo.py` and `ppo_utils.py` are
  removed from the live diff because they are not used by the current B+
  route and obscure the future PPO implementation baseline.
- the speculative generated Superpowers-style direct-PPO implementation plan
  is removed; the concise, owner-readable proposal is
  `docs/ppo/NEXT_PPO_DIRECTION.md`.

## Prevention

Large run products remain ignored by Git. Future experiments should use one
run directory containing `STATUS.md`, `DECISIONS.md`, `EXPERIMENTS.md`, an
`artifacts/` directory, and explicit `canonical`/`superseded` labels. A new
run must not copy full raw evaluation trees locally unless needed for active
analysis.


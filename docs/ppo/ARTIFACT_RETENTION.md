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

## Archived during Tier 3

- historical per-run directories were moved to `logs/archive/legacy_runs/`;
- historical summaries/designs were moved to
  `logs/archive/legacy_reports/`;
- reviewer briefs were moved to `logs/archive/reviews/`;
- historical handoffs were moved to `docs/archive/handoffs/`;
- explicitly superseded D0.1/B+ artifacts were moved to
  `logs/archive/superseded_artifacts/`.

These files were not deleted. Canonical artifact paths were not moved.

## Not yet removed

- `eval_results/**` remains present locally;
- all historical checkpoint snapshots remain present;
- caches and aborted payloads remain present inside the archive;
- no source or test file was removed as part of Tier 3.

## Code retention decisions

- P0 evaluation-integrity changes in `eval_multiagent.py`, `evaluate.sh`,
  `evaluate_ol1.sh`, `aggregate_eval.py`, and `utils.py` are retained.
- isolated `d0/`, `d2/`, `d25/`, `d2r/`, and `bplus_v22/` packages are
  retained because canonical validators and future direct-PPO work depend on
  them.
- old D4 scenario-sampling changes in `train_ppo.py` and `ppo_utils.py` remain
  in the live diff pending a separate code-design decision.
- the speculative generated Superpowers-style direct-PPO implementation plan
  is removed; the concise, owner-readable proposal is
  `docs/ppo/NEXT_PPO_DIRECTION.md`.

## Prevention

Large run products remain ignored by Git. Future experiments should use one
run directory containing `STATUS.md`, `DECISIONS.md`, `EXPERIMENTS.md`, an
`artifacts/` directory, and explicit `canonical`/`superseded` labels. A new
run must not copy full raw evaluation trees locally unless needed for active
analysis.

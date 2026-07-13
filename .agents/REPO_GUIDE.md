# End2Race Repository Guide

Updated: 2026-07-12 after local Tier-3 log organization.

## Entry points

- `.agents/HANDOFF.md` — current authority and chronological provenance.
- `docs/ppo/README.md` — concise project state.
- `docs/EXPERIMENT_RECORD.md` — detailed Chinese cross-stage experiment record.
- `docs/ppo/EXPERIMENT_HISTORY.md` — shorter evidence/claim ledger.
- `docs/ppo/NEXT_PPO_DIRECTION.md` — proposed BC-direct PPO direction.
- `logs/README.md` — canonical/archived log layout.

## Runtime and training code

- `model.py`, `train.py`, `demonstration.py` — original BC model and training.
- `train_ppo.py`, `ppo_utils.py` — historical PPO implementation. This is not
  a v2.2 runner and is not yet wired to the hierarchical action/objective.
- `eval_multiagent.py`, `evaluate.sh`, `evaluate_ol1.sh`,
  `aggregate_eval.py` — evaluator and strict P0 aggregation path.
- `bplus_v22/` — v2.2 policy mechanics, buffers, objectives, sidecar,
  warm-start and closed-loop validators. It contains both historical
  warm-start code and the later hierarchical remediation because current
  modules still share utilities.
- `bplus_v22_cli.py` — v2.2 artifact/preflight/evaluation commands.

## Evidence-generation packages

- `d0/`, `d0_audit.py`, `d0_canonical_audit.py` — canonical episode/outcome
  audit.
- `d2/`, `d2_cli.py` — representation dataset, grouped probes and release.
- `d25/`, `d25_cli.py` — bounded counterfactual branch search.
- `d2r/`, `d2r_cli.py` — spatiotemporal geometry sidecar/probe.
- `tests/` — standalone structural and regression programs. They are run with
  `PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1` in the `end2race` interpreter.

## Data and large generated content

- `Dataset_Austin_0525/` — BC demonstration/training data; do not classify as
  disposable experiment output.
- `pretrained/` — BC, historical candidates and many intermediate
  checkpoints. No checkpoint was removed by Tier 3.
- `eval_results/` — local early raw rollout trees, approximately 33 GB. No
  raw rollout was removed by Tier 3.
- `logs/` — canonical evidence and organized local archive; see
  `logs/README.md`.
- `f1tenth_racetracks/`, `f1tenth_gym/`, `latticeplanner/` — simulator,
  maps and planner dependencies; outside log cleanup scope.

## Log organization

Seven canonical experiment directories retain their original paths. Early
per-run content is under `logs/archive/legacy_runs/`; early summaries are in
`logs/archive/legacy_reports/`; reviewer briefs are in
`logs/archive/reviews/`; superseded artifacts are byte-preserved under
`logs/archive/superseded_artifacts/`.

Use `logs/archive/PATH_MIGRATION.tsv` for old-to-new paths and
`docs/ppo/LOG_FILE_INDEX.tsv` / `DOCUMENT_INDEX.tsv` for exhaustive lookup.

## Current implementation boundary

The repository now contains a managed B+ runner connecting hierarchical
sampling, exact executed-action/log-probability accounting, multi-objective
GAE, dual updates, checkpoint releases and paired KPI evaluation. Historical
B2 used a centered deterministic deployment rule and completed one 20-iteration
run; all candidates failed overtake feasibility. Prospective B3 adds a unified
training/deployment distribution and 40-iteration RunPlan support. B3 code is
CPU-contract-tested but no numerical B3 RunPlan or GPU result exists yet. See
`.agents/B3_PPO_PLAN.md` and `.agents/B3_IMPLEMENTATION_RECORD.md`.

Do not delete “old” `bplus_v22` modules based only on their names: current
hierarchical modules import shared functionality from them. Code refactoring
requires a separate dependency audit and test-preserving change.

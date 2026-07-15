# Analysis and Report Index

Updated: 2026-07-14. The purpose of this index is to locate evidence without
searching dozens of dated run directories.

Complete machine-readable inventories are:

- `docs/ppo/DOCUMENT_INDEX.tsv` — every local Markdown/TSV handoff, report,
  ledger, review, and artifact table found after organization;
- `docs/ppo/LOG_FILE_INDEX.tsv` — every file under `logs/` and the historical
  handoff archive, with classification and byte size.

## Canonical reports

| stage | primary report | result boundary |
|---|---|---|
| P1/deployed baseline | `logs/p1_final_report_20260710.md`, `logs/final_model_report_20260710.md` | cand160 historical baseline, not B+ success |
| D0 v1 | `logs/d0_canonical_audit_20260710_121955/d0_summary.md` | frozen rejected/sensitivity evidence |
| D0.1 | `logs/ppo_next_unattended_20260710_230212/D01_EVIDENCE_REPORT_20260711.md` | canonical 16,800-occurrence audit |
| D2 | `logs/d2_representation_20260711_174039/D2_EVIDENCE_REPORT_20260711.md` | original representation gate failed; test unopened |
| D2.5 | `logs/d25_counterfactual_20260711/D25_EVIDENCE_REPORT_20260711.md` | 67/91 tested cases recovered by fixed library |
| D2R-G | `logs/d2r_geometry_20260711/D2R_EVIDENCE_REPORT_20260711.md` | original TTC/2s-FA gate failed |
| B+ v2.2 | `logs/bplus_v22_d3r2_20260711/{STATUS,DECISIONS,EXPERIMENTS}.md` | Task 10 failed; hierarchical Task 6 failed; no PPO |
| B2/B3 | `.agents/PPO_DEVELOPMENT_REPORT.md`, `.agents/B3_PPO_PLAN.md` | B2 failed overtake direction; B3 reviewed GO but paused unrun |
| B4 | `.agents/B4_DIRECT_HEAD_PPO_RESULT.md` | plain frozen-feature direct head substantive negative |
| B5 | `.agents/B5_SAFE_TRUST_REGION_RESULT.md`, `.agents/B5_POSTHOC_STATISTICS_AND_OBJECTIVE_AUDIT.md` | behavior preserved; safety effect inconclusive; no promotion |
| B6 | `.agents/B6_TEMPORAL_EXPLORATION_PHASE0_RESULT.md` | `rho=.95` unconditional AR(1) no-go; learner unrun |
| B7 | `.agents/B7_PLAIN_RECURRENT_PPO_RESULT.md` | remediated recurrent run early-stopped; no candidate/evaluation |

The concise cross-stage interpretation is
`docs/ppo/EXPERIMENT_HISTORY.md`.
The detailed Chinese chronology and mechanism discussion is
`docs/EXPERIMENT_RECORD.md`; repository layout is `docs/REPO_GUIDE.md`.

## Canonical machine evidence

- D0.1:
  `logs/ppo_next_unattended_20260710_230212/artifacts/d01_full_reconcile_20260711_170200_a`.
- D2: `logs/d2_representation_20260711_174039/artifacts/`.
- D2.5: `logs/d25_counterfactual_20260711/artifacts/`.
- D2R-G: `logs/d2r_geometry_20260711/artifacts/`.
- B+ v2.2: `logs/bplus_v22_d3r2_20260711/artifacts/`.
- B7 compact result:
  `docs/ppo/evidence/b7_plain_recurrent_negative_20260714/`.
- B7 full release:
  `Experiments/B7_plain_recurrent_ppo/runs/b7_seed1_20260714_114132/remote/seed1/`.

Canonical paths were not moved by Tier 3.

## Archived reports

`logs/archive/legacy_reports/` contains dated summaries, launch notes,
external-review synthesis, and early design documents from the pre-canonical
PPO sweeps. Their conclusions are useful for provenance but are superseded by
P1/D0.1 and the current experiment history.

`logs/archive/legacy_runs/` contains per-run command/status/report trees from
early PPO experiments. These are not product evidence and should not be used
to select a current model.

`logs/archive/reviews/` contains Claude/reviewer briefs. These are analysis
inputs, not experimental results or project authority.

`logs/archive/superseded_artifacts/` contains byte-preserved partial,
intermediate, determinism-confirmation, and display-manifest releases that
were explicitly superseded. They are excluded from the canonical path but
were not deleted.

## Claim discipline

- “PASS” for artifact integrity is distinct from policy/mechanism acceptance.
- D2/D2R-G remain failed even though TTC was prospectively downgraded for the
  later policy phase.
- D2.5 67/91 is not a theoretical ceiling.
- Task 6 metrics cannot rank A/B/C.
- Task 10 and the 288-scenario development panel provide no policy
  generalization evidence.
- B7 did not reach its candidate evaluation boundary; do not interpret its
  early stop as a measured collision/overtake result or select iter6 post hoc.

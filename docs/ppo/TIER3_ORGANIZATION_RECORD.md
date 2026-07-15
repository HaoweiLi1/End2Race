# Tier-3 Local Organization Record

Date: 2026-07-12. Scope: local workspace only.

## Actions completed

- Identified 72 local handoff/report/summary/analysis documents.
- Kept seven canonical experiment directories at their original paths.
- Moved all other top-level historical run directories under
  `logs/archive/legacy_runs/`.
- Moved historical top-level summaries/designs under
  `logs/archive/legacy_reports/` while retaining the P1 and deployed-model
  reports at `logs/` root.
- Moved historical handoffs under `docs/archive/handoffs/`.
- Moved reviewer briefs under `logs/archive/reviews/`.
- Moved explicitly superseded D0.1 and B+ artifacts under
  `logs/archive/superseded_artifacts/`.
- Added current authority, report, experiment, and artifact indexes.

## Size of archived classes

- legacy run directories: approximately 141 MB;
- legacy reports: approximately 252 KB;
- reviewer briefs: approximately 32 KB;
- superseded artifacts: approximately 94 MB.

## Explicit non-actions

- no remote access or modification;
- no PPO, Task 6, Task 9, Task 10, D2-test, or final-pool process;
- no checkpoint deletion;
- no raw `eval_results/` deletion;
- no code deletion or tracked-code rollback;
- no registry mutation;
- no canonical or substantive FAILED artifact deletion.

## Post-organization verification

- five canonical output manifests pass `sha256sum -c`: D0.1, D2, D2.5,
  D2R-G, and hierarchical Task 6;
- all 33 local standalone test programs pass in the pinned `end2race`
  interpreter with `PYTHONDONTWRITEBYTECODE=1`;
- the canonical log root contains only the seven canonical run directories,
  two primary reports, navigation, archive, and one compatibility symlink for
  the historically referenced P0/P1 audit handoff;
- `logs/archive/PATH_MIGRATION.tsv` contains the original-to-archive path
  mapping;
- `docs/ppo/LOG_FILE_INDEX.tsv` and `docs/ppo/DOCUMENT_INDEX.tsv` provide
  complete machine-readable inventories.

## Concurrent worktree change

At 20:22 +08:00, while Tier 3 was organizing ignored log paths, another
process in the shared workspace switched to `chore/commit-evidence-pipeline`
and created seven local commits through `a4b0128`. Tier 3 did not create,
reset, amend, push, or reorder those commits. Subsequent documentation work
was based on that observed HEAD and preserved the concurrent changes.

## Pre-Tier-3 draft deletion disclosure

Before the owner paused the earlier cleanup, two untracked local documents
had already been deleted: `docs/codex-ultra-guide.md` and the speculative
`docs/superpowers/plans/2026-07-12-ppo-pilot-bc-direct.md`. Neither contained
numerical experimental evidence. The direct-PPO rationale is retained in
`docs/ppo/NEXT_PPO_DIRECTION.md`; the deletion remains disclosed rather than
being silently omitted from the record.

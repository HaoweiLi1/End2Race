# D2 Episode-Held-Out Probe — Implementation Plan

Version: `d2-plan-1`  
Specification: `../specs/2026-07-11-d2-episode-heldout-probe.md`

## 1. Stage order

1. **D2-S0 split lock**: tests first; create outcome-poison invariance,
   L4-isolation, quota, fold-balance, deterministic-rerun, test-path-redaction,
   source-hash, and registry negative tests. Emit split artifacts only.
2. **D2-S1 replay/labels**: tests first; verify hidden reset, one-frame-lag
   speed, per-frame batch-1 replay, exact action equality, censoring,
   competing-event censoring, branch alignment, velocity projection, and TTC
   boundaries on synthetic episodes. Run a small remote BC smoke across all
   maps and collision/noncollision cases.
3. **D2-S2 non-test extraction**: append non-test registry rows, extract only
   non-test features/labels to fresh atomic storage, validate counts/hashes,
   and deterministically re-extract a locked sample.
4. **D2-S3 grouped probes**: implement metrics and threshold tests, then run
   prevalence, linear, and MLP five-outer/three-inner evaluation. Freeze OOF
   report and branch decision. Implement temporal probe only if MLP fails.
   Temporal T1 is feature-tap-only; if it fails, extract the non-test-only
   deployable signal sidecar and run the single final capacity-matched T2
   family described in the spec. No further temporal family is permitted.
5. **D2-S4 one-open test**: fit the selected family on all non-test data,
   freeze checkpoint/normalization/threshold hashes, create the durable open
   marker and registry rows, then extract/evaluate test exactly once.
6. **D2-S5 closure**: independently validate the release, mirror reviewed
   manifests/reports/checkpoint, update handoff/status/experiments/decisions,
   and state whether D2.5/D3 are allowed.

No stage may rewrite D0.1 artifacts or an existing D2 output directory.

## 2. Planned source/test surface

New package:

- `d2/__init__.py`: schemas and frozen defaults;
- `d2/split.py`: projected D0 reader, hash domains, quotas/folds, seals;
- `d2/labels.py`: privileged geometry and censored horizon labels;
- `d2/replay.py`: exact BC replay and atomic memmap extraction;
- `d2/metrics.py`: AUCPR/Brier/ECE/TTC/episode alarm metrics;
- `d2/models.py`: constant, linear, MLP, optional temporal family;
- `d2/probe.py`: nested grouped evaluation, fit, threshold freeze, test open;
- `d2/validate.py`: emitted-artifact and registry recomputation;
- `d2_cli.py`: stage-explicit command line entry point.

Tests:

- `tests/test_d2_split.py`;
- `tests/test_d2_labels.py`;
- `tests/test_d2_replay.py`;
- `tests/test_d2_metrics.py`;
- `tests/test_d2_probe.py`.

Long remote runs use checked-in wrappers under the active goal's `commands/`
directory with PID, status, log, and exit files. Sync uses explicit file
allowlists only.

## 3. Red-green gates

Each module begins with a failing focused test and records the expected
failure. Required negative cases include:

- modifying labels/outcome directories changes no split byte;
- one L4 in two splits or one episode in two folds is rejected;
- a test source locator appears before open and is rejected;
- a missing/extra/rewritten registry row is rejected;
- current-speed instead of lagged-speed replay fails exact action equality;
- whole-sequence replay is not accepted as the identity oracle;
- a normal terminal horizon and a competing collision are censored;
- a matching collision boundary is positive at exactly the horizon;
- normalization or threshold fitting touches an outer/test row and is
  rejected;
- a second test-open attempt or config/hash mismatch is rejected;
- post-open family, checkpoint, normalization, or threshold mutation is
  rejected.

All test commands use:

`PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /home/haowei/miniconda3/envs/end2race/bin/python`

Run focused suites, combined D2 suites, existing D0/P0 suites, and
`py_compile` before remote execution.

## 4. Remote run gates

Every remote stage must record:

- local and remote source SHA256 equality;
- pinned interpreter/package/CUDA inventory;
- clean process preflight and adequate disk/RAM/GPU capacity;
- input and output manifests;
- start/end time, PID, exit code, and independent validator result;
- no writes below `eval_results/**` or `f1tenth_racetracks/**`;
- explicit artifact pullback and double-end hash comparison.

The test stage additionally records the pre-open registry hash, durable open
marker timestamp/hash, selected-family bundle hash, post-open registry hash,
and proof that no alternate-family test prediction exists.

## 5. Completion decisions

- `MLP_OOF_PASS`: fit/freeze MLP, one-open test; do not run temporal.
- `MLP_OOF_FAIL_TEMPORAL_PASS`: fit/freeze temporal, one-open test.
- `BOTH_OOF_FAIL`: D2 stop; no D3 PPO.
- `SELECTED_TEST_PASS`: D2 complete; proceed to D2.5 before D3.
- `SELECTED_TEST_FAIL`: D2 complete as a negative result; no family switch and
  no D3 PPO.

D2.5 remains a separate causal stage and cannot be merged into probe fitting.

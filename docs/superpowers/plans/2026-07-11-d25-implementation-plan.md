# D2.5 Counterfactual Recoverability — Implementation Plan

Version: `d2.5-plan-1`  
Specification: `../specs/2026-07-11-d25-counterfactual-recoverability.md`

## 1. Red-green order

1. Intervention timing/composition tests: macro rounding, duration, no
   positive speed, clipping detection, fixed-order IDs.
2. Synthetic outcome tests: confirmed hold, terminal-only overtake,
   safe-follow, collision, opponent-only collision, seam alignment.
3. Simulator baseline replay smoke: eight outcome-blind map/skill-stratified
   non-test cases; compare every archived array/terminal field bitwise.
4. Branch smoke: no-op plus a small fixed subset on the same cases; rerun
   byte-identically and benchmark wall time.
5. Full 91-case search only after baseline and branch smoke pass.
6. Independent validation, witness reruns, synthesis, and record sync.

All tests use the pinned end2race interpreter and no bytecode in the worktree.

## 2. Planned files

- `d25/__init__.py`: locked library/config schemas;
- `d25/oracle.py`: exact evaluator replay and branch simulation;
- `d25/search.py`: registry, case manifest, ordered search, atomic release;
- `d25/validate.py`: independent accounting/replay checks;
- `d25_cli.py`: explicit smoke/full/validate commands;
- `tests/test_d25_oracle.py` and `tests/test_d25_search.py`.

Remote wrappers record PID/status/log/exit and use fresh output directories.
Source sync remains an explicit allowlist.

## 3. Stop conditions

- any smoke baseline mismatch;
- source/model/scenario/registry hash mismatch;
- test/final-pool ID leakage;
- non-macro branch start/duration;
- positive residual or unreported clipping;
- incomplete non-recovered case search;
- witness rerun mismatch;
- missing branch/case row or manifest hash failure.

On a stop, preserve the partial output and append-only registry, fix by
red-green evidence, and rerun only to a fresh directory.


# D0.1 Implementation Plan (v2.2, executable)

Date: 2026-07-10
Implements: D0.1 specification v2.2
Execution authority: the user's 2026-07-10 single-Agent unattended Goal
Objective. Stage promotion is automatic only after the preceding technical
gate passes. No push is authorized.

## 0. Boundaries

- Never modify `d0_canonical_audit.py`,
  `logs/d0_canonical_audit_20260710_121955/`, old experiments, checkpoints,
  `eval_results/**`, or `f1tenth_racetracks/**`.
- New D0.1 source is limited to the twelve Stage-0 files in section 2.
- Analysis runtime writes only to a fresh output directory, its sibling
  `.partial`, the canonical opened registry after full G8 passes, and the
  current Goal log root.
- D0.1 never invokes the simulator, evaluator, PPO, or D2.
- Predictions are reconciliation targets, never source data.
- Preserve every pre-existing worktree modification. Never use broad staging,
  destructive Git commands, or whole-repository synchronization.

## 1. Pinned interpreter and import/bytecode contract

```text
PY=/home/haowei/miniconda3/envs/end2race/bin/python
Python=3.10.19
NumPy=1.26.4
ROOT=/home/haowei/Documents/End2Race
GOAL_ROOT=logs/ppo_next_unattended_20260710_230212
PYCACHE=/tmp/d0_stage0_pycache_20260710_230212
```

Every direct Python command is prefixed exactly with:

```text
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/d0_stage0_pycache_20260710_230212
```

Before Stage 0, run and retain this verification:

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/d0_stage0_pycache_20260710_230212 \
  /home/haowei/miniconda3/envs/end2race/bin/python -c \
  "import os,sys,numpy as np; root=os.path.realpath('.'); assert os.environ['PYTHONPATH']=='.'; assert os.environ['PYTHONDONTWRITEBYTECODE']=='1'; assert root in [os.path.realpath(p or '.') for p in sys.path]; assert np.__version__=='1.26.4'; print(sys.version); print('IMPORT_ENV_OK')"
```

`PYTHONDONTWRITEBYTECODE=1` suppresses import caches. `py_compile` is routed to
the unique `/tmp` prefix. One ignored cache predates this Goal and is preserved:
`tests/__pycache__/test_eval_aggregation.cpython-310.pyc`, mtime
`2026-07-10T12:15:13+08:00`, SHA256
`1c13c5663f903b8c355eb9cdf58734b13ebceadc923cca325b670939e133fba7`.
A post-check requires no cache newer than the Stage-0 start and no D0 cache;
the baseline cache must retain the recorded hash and mtime.

## 2. Exact Stage-0 source layout

```text
d0/__init__.py
d0/identity.py
d0/outcomes.py
d0/gates.py
d0/stats.py
d0/scan.py
d0_audit.py
tests/test_d0_identity.py
tests/test_d0_outcomes.py
tests/test_d0_gates.py
tests/test_d0_stats.py
tests/test_d0_scan.py
```

`d0/__init__.py` exports `ANALYSIS_VERSION="d0.1"`,
`CLASSIFIER_VERSION="d0.1-traj-1"`, schema constants, and the frozen
RunConfig factory. No other source/test path is created in Stage 0.

## 3. Frozen RunConfig `d0.1-runconfig-1`

The serialized config has exactly the following semantic values; code emits
canonical sorted JSON and its SHA256:

```json
{
  "schema": "d0.1-runconfig-1",
  "analysis_version": "d0.1",
  "classifier_version": "d0.1-traj-1",
  "source_run_id": "20260710_121955",
  "repository_root": "/home/haowei/Documents/End2Race",
  "eval_root": "eval_results",
  "assets_root": "f1tenth_racetracks",
  "goal_root": "logs/ppo_next_unattended_20260710_230212",
  "opened_registry": "logs/ppo_next_unattended_20260710_230212/opened_registry.tsv",
  "opened_at_utc": "2026-07-10T23:02:12+08:00",
  "tag_template": "p1v_{run}_{model}_{map}_off{offset}",
  "result_dir_template": "eval_results/{tag}_{map}",
  "models": {
    "bc": {
      "path": "pretrained/end2race.pth",
      "sha256": "b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4"
    },
    "cand160": {
      "path": "pretrained/end2race_ppo_full_disc_r8192_seed1_20260709_210827_iter0160.pth",
      "sha256": "77cd79904f0f57c1e7a4914dd0b52384628dce225f9222e4e2274e0eda3b5aa6"
    },
    "cand120": {
      "path": "pretrained/end2race_ppo_full_disc_r8192_seed1_20260709_210827_iter0120.pth",
      "sha256": "9f2f47bf46363946ba29c1fe5fcada3a3d5fe514ece6eb160c03b25d8f82b3b3"
    },
    "cand040": {
      "path": "pretrained/end2race_ppo_full_disc_r8192_seed0_20260709_210827_iter0040.pth",
      "sha256": "c7a72f5564a191e103d319a7f66167e6969fb3528534b90bafba77ceb598d7e1"
    }
  },
  "grids": [
    ["Austin", 21], ["Austin", 42], ["Austin", 63], ["Austin", 84],
    ["Nuerburgring", 0], ["MoscowRaceway", 0], ["Hockenheim", 0]
  ],
  "offset_start_formula": "(i*max_wc//50+offset)%max_wc for i=0..49",
  "zero_start_formula": "i*max_wc//49 for i=0..49",
  "dev_start_formula": "i*max_wc//49 for i=0..49 on Austin raceline1",
  "max_wc_convention": "line count of raceline CSV minus 2",
  "ego_raceline": "raceline1",
  "opponent_racelines": ["raceline0", "raceline1", "raceline2"],
  "opponent_speedscales": [0.5, 0.6, 0.7, 0.8],
  "interval_idx": 15,
  "sim_dt": 0.01,
  "duration_s": 8.0,
  "duration_ticks": 800,
  "noise": 0.0,
  "noise_seed": 42,
  "expected_occurrences": {"smoke": 1200, "full": 16800},
  "bootstrap": {"B": 10000, "seed": 20260710},
  "classifier": {
    "attempt_m": 0.6,
    "confirmed_lead_m": 2.0,
    "confirmed_hold_s": 0.7,
    "car_distance_m": 1.0,
    "alongside_strict_m": 0.6
  }
}
```

The config object is passed into geometry and scan functions; tags, grids,
hashes, formulas, thresholds, and registry location are never re-derived from
ambient defaults.

## 4. Module contracts

### 4.1 `d0/identity.py`

```python
canonical_json(payload) -> bytes
domain_id(layer, payload) -> str
asset_namespace(runconfig) -> AssetNamespace
resolve_scenario(runconfig, map_name, offset, start_ordinal,
                 opponent_raceline, speedscale) -> ResolvedScenario
l1_payload(source_occurrence, resolved_scenario, hashes, runconfig) -> dict
l2_payload(resolved_scenario, runconfig, asset_namespace_sha256) -> dict
l3_payload(resolved_scenario, asset_namespace_sha256) -> dict
build_l4_blocks(exact_l3_nodes, dev_l3_nodes) -> BlockManifest
geometry_manifest(runconfig) -> S0Outputs
append_opened_registry(path, rows) -> RegistryAppendResult
```

All payload key sets, hashing domains, graph construction, SensA exact pattern,
SensB accounting, and registry schema follow spec sections 2, 3, and 8.
Geometry emits six S0 artifacts and a manifest from assets only.

### 4.2 `d0/outcomes.py`

```python
centerline_length(asset_root, map_name) -> float
unwrap_progress(raw, L) -> UnwrapResult
align_rel(ego_recorded, opp_recorded, ego_terminal, opp_terminal, L) -> RelSeries
normalize_archived_outcome(raw) -> str
classify_outcome(npz, json_episode, L) -> OutcomeRecord
classify_collision(npz, rel_series) -> CollisionEvent
equality_vector(record) -> tuple
```

`RelSeries` contains recorded-plus-terminal rel values, one integer k, both
wrap counts, alignment status, and evidence. `OutcomeRecord` contains every
spec section 5.3 equality-vector field. Unknown values are explicit strings,
never `None` fallbacks.

### 4.3 `d0/gates.py`

```python
@dataclass(frozen=True)
class GateResult:
    name: str
    blocking: bool
    passed: bool
    counts: dict
    violations: tuple[str, ...]

run_g1_duplicate_determinism(...)
run_g2_inventory(...)
run_g3_integrity(...)
run_g4_near_duplicate(...)
run_g5_collision_floors(...)
run_g6_record_physics(...)
run_g7_unknown_censoring(...)
run_g8_reconciliation(...)
release_verdict(results) -> GateResult
```

G8 accepts in-memory canonical truth plus emitted artifacts and reparses every
artifact. It must detect missing, extra, and duplicate correction rows,
incorrect collision-event keys, malformed matrices, corrupted summary counts,
estimand drift, and manifest/hash omissions.

### 4.4 `d0/stats.py`

```python
point_estimates(pool, bc_records, candidate_records) -> dict
paired_block_bootstrap(pool, bc_records, candidate_records, blocks_by_map,
                       B, rng) -> dict
run_all_stats(estimands, records, blocks, B=10000, seed=20260710) -> dict
```

`run_all_stats` creates 27 child streams once in the exact spec order. JSON is
canonical and includes observed counts, effects, intervals, B, root seed,
child order/index, zero-denominator fraction, stability verdict, and a
replicate-draw fingerprint without dumping all replicates.

### 4.5 `d0/scan.py`

```python
expected_inventory(runconfig, mode) -> Inventory
scan_occurrence(source, resolved, hashes, centerline_L) -> OccurrenceRecord
collapse_canonical(occurrences) -> CanonicalRecords
build_transition_matrices(...)
build_reconciliation(...)
run_scan(mode, output_dir, runconfig, workers) -> int
```

The worker result order is normalized by L1 ID before reduction. SHA256 and
TSV/JSON output order are deterministic. Multiprocessing workers never write
shared output files.

Full success sequence is: validate fresh destination; create sibling partial;
write config/S0 snapshot; scan; run G1-G8; construct prospective registry
snapshot; validate it in G8; lock and append the canonical registry; copy the
locked registry snapshot; write all outputs and manifest; fsync; atomically
rename partial; write `COMPLETE` last. Failures write stable diagnostics and
never write COMPLETE.

### 4.6 `d0_audit.py`

```text
python d0_audit.py --mode {geometry,smoke,full} --output-dir DIR
                   [--eval-root DIR] [--assets-root DIR] [--workers N]

0 success and COMPLETE
2 blocking gate failure, FAILED retained
3 destination exists and is non-empty, no modification
4 missing/unreadable/invalid input
```

CLI overrides may change roots and workers only. Models, grids, identities,
thresholds, and statistics are frozen by RunConfig.

## 5. Stage 0 exact red-green sequence

Run from repository root. Create each test first with `apply_patch`; do not
create implementation and test in the same step. Record stdout, stderr, and
exit status in the pre-created Goal transcript.

Entry snapshots:

```bash
git status --porcelain=v1 --untracked-files=all > /tmp/d0_stage0_before_20260710_230212.txt
git rev-parse HEAD > /tmp/d0_stage0_head_20260710_230212.txt
```

Environment preflight is the command in section 1 and must print
`IMPORT_ENV_OK`.

Use `E` below as the literal environment prefix from section 1 and `PY` as the
pinned absolute interpreter; the executed transcript expands both.

```text
S0-01 create tests/test_d0_identity.py
S0-02 E PY tests/test_d0_identity.py
      expected nonzero and "ModuleNotFoundError: No module named 'd0'"
S0-03 create d0/__init__.py and d0/identity.py
S0-04 E PY tests/test_d0_identity.py -> ALL TESTS PASSED

S0-05 create tests/test_d0_outcomes.py
S0-06 E PY tests/test_d0_outcomes.py
      expected nonzero and "ModuleNotFoundError: No module named 'd0.outcomes'"
S0-07 create d0/outcomes.py
S0-08 E PY tests/test_d0_outcomes.py -> ALL TESTS PASSED

S0-09 create tests/test_d0_gates.py
S0-10 E PY tests/test_d0_gates.py
      expected nonzero and "ModuleNotFoundError: No module named 'd0.gates'"
S0-11 create d0/gates.py
S0-12 E PY tests/test_d0_gates.py -> ALL TESTS PASSED

S0-13 create tests/test_d0_stats.py
S0-14 E PY tests/test_d0_stats.py
      expected nonzero and "ModuleNotFoundError: No module named 'd0.stats'"
S0-15 create d0/stats.py
S0-16 E PY tests/test_d0_stats.py -> ALL TESTS PASSED

S0-17 create tests/test_d0_scan.py
S0-18 E PY tests/test_d0_scan.py
      expected nonzero and "ModuleNotFoundError: No module named 'd0.scan'"
S0-19 create d0/scan.py and d0_audit.py
S0-20 E PY tests/test_d0_scan.py -> ALL TESTS PASSED
```

Expanded green/compile commands are exactly:

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/d0_stage0_pycache_20260710_230212 \
  /home/haowei/miniconda3/envs/end2race/bin/python -m py_compile \
  d0/__init__.py d0/identity.py d0/outcomes.py d0/gates.py d0/stats.py \
  d0/scan.py d0_audit.py tests/test_d0_identity.py \
  tests/test_d0_outcomes.py tests/test_d0_gates.py tests/test_d0_stats.py \
  tests/test_d0_scan.py

for t in identity outcomes gates stats scan; do
  env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp/d0_stage0_pycache_20260710_230212 \
    /home/haowei/miniconda3/envs/end2race/bin/python tests/test_d0_${t}.py \
    || exit 1
done

find d0 tests \( -type d -name __pycache__ -o -type f -name '*.pyc' \) \
  -newermt '2026-07-10T23:15:24+08:00'
find d0 tests -type f -name 'test_d0_*.pyc'
sha256sum tests/__pycache__/test_eval_aggregation.cpython-310.pyc
```

Both find commands must print nothing, and the preserved cache hash must match
section 1. Then capture:

```bash
git status --porcelain=v1 --untracked-files=all > /tmp/d0_stage0_after_20260710_230212.txt
diff -u /tmp/d0_stage0_before_20260710_230212.txt /tmp/d0_stage0_after_20260710_230212.txt
```

Normalize the diff additions by removing the `?? ` prefix. They must equal
exactly the twelve paths in section 2, sorted bytewise. No tracked
modification or thirteenth path is allowed. Copy both snapshots, diff, full
transcript, and source SHA256 into the Goal log only after this check.

## 6. Required synthetic tests

### 6.1 Identity

- canonical JSON golden bytes and one known SHA256 vector per domain;
- exact payload key-set rejection for missing or extra L1/L2/L3/L4 fields;
- cross-domain hashes differ for identical JSON;
- float-hex distinguishes adjacent binary64 values;
- L1 changes with checkpoint, result JSON, episode key, or NPZ hash;
- waypoint wrap and opponent correspondence match evaluator semantics;
- asset namespace order independence;
- L4 input-order independence and inclusive 1.000 m edge;
- planted exact dev overlap and 1 cm diagnostic separation;
- SensA fixture retains minimum resolved index and emits both IDs;
- full-pattern validator rejects 11 rows, 13 rows, wrong map/raceline/speed,
  swapped retained/excluded, or missing pair;
- first/last ego-pose-and-ego-waypoint-speed equality assertion, plus explicit
  zero initial actual speeds;
- SensB analog validates exact/from-primary/additional arithmetic and rejects
  primary-minus-full-exclusion logic;
- opened registry header, row ID, idempotent append, conflicting duplicate,
  and append-only preservation.

### 6.2 Outcomes

Use synthetic L=420 and recorded-plus-terminal data:

- raw labels preserve `following/overtaking/collision`; normalization is exact;
  aliases `follow/overtake` as raw labels fail;
- mid-track pass agrees archived/corrected;
- named +417 m seam class corrects to follow with k=-1;
- opponent seam crossing unwraps continuously;
- ego seam crossing and genuine pass remains overtake;
- one k shift applies to the whole series;
- positive/zero initial aligned rel triggers alignment failure and G7 unknown;
- collision at 3.1 s is valid and never confirmed pass;
- 0.5 s non-collision is censored;
- lead boundaries 1.99/2.00 and window 0.69/0.70/0.71 seconds;
- terminal frame is decisive in a nominal 71-frame fixture;
- attempt at 0.60 is true and 0.61 false;
- confirmed-pass subset invariant;
- terminal gap 0.004/0.010/0.016 fails/passes/fails;
- recorded dt 0.005 and 0.015 pass, values outside fail;
- collision distance 0.99/1.00/1.01 gives car/car/wall;
- phase -0.60/-0.59/+0.59/+0.60 gives pre/alongside/alongside/post;
- seam-straddling collision phase uses corrected rel;
- direct flags/involvement never change with inferred cause.

### 6.3 Gates

- G1 disagrees separately on every section-5.3 vector field, including
  alignment status and censoring;
- G2 missing, extra, duplicate, unvalidated, and stale inventory;
- G3 empty NPZ, bad hash, root escape, and duplicate L1;
- G4 compares matched conditions only;
- G5 29 fails and 30 passes per skill;
- G6 missing terminal field, JSON/raw mismatch, collision Boolean mismatch,
  unequal length, nonfinite value, and invalid dt/gap;
- G7 unknown/alignment failure/censored counts block;
- G8 happy path plus mandatory negative cases: omit one correction row, add an
  extra row, duplicate a row, corrupt a summary count, corrupt a matrix sum,
  omit a manifest hash, and drift one estimand ID.

### 6.4 Statistics

- hand-computable two-map, three-block fixture validates point RR/RD;
- map-stratified block counts and multiplicity expansion;
- BC/candidate receive the same sampled scenario multiset;
- child stream order is exactly 27 tuples and stable;
- same-seed byte equality and changed-seed fingerprint difference;
- observed BC zero gives undefined RR and authoritative RD;
- bootstrap zero denominators are excluded/reported and >1% is unstable.

### 6.5 Scan

Build a synthetic tree with two models, one grid, eight episodes, an exact
duplicate occurrence, endpoint pair, and dev-overlap start. Verify:

- exact/primary/SensA/SensB memberships;
- L1 provenance binds source JSON and NPZ hashes;
- archived raw labels and corrections ledger completeness;
- one collision row per model/L2 with direct/inferred separation;
- every applicable 4x4 matrix has 16 cells and sums to N;
- G8 reparses emitted files;
- deterministic rerun yields byte-identical TSV/JSON/manifests;
- non-empty output exits 3 unchanged;
- injected failure leaves partial/FAILED and no COMPLETE;
- success promotes atomically and COMPLETE is last;
- canonical registry uses a temp path, appends only after gate pass, and its
  snapshot hash matches.

## 7. Stage 1 — geometry and smoke

After Stage 0 passes, freeze source hashes and run geometry to a new timestamped
directory below the Goal root. Geometry may read only the configured CSV and
checkpoint files (hash verification); it opens no NPZ.

Blocking review checks exact payload schemas, regenerated set counts, the 12
SensA pairs, SensB accounting, L4 determinism, asset/checkpoint hashes, and
output manifest. Prediction mismatches are investigated, not patched around.

Then run smoke on BC/cand160 x Austin off21. Smoke must pass every applicable
G1-G8 condition, exact 1,200 occurrence inventory, corrections-ledger
reconciliation, deterministic rerun, and atomic output tests before full scan.

## 8. Stage 2 — detached full scan

Full scan command uses the pinned interpreter, `PYTHONPATH=.`, disabled source
bytecode, eight workers, a fresh timestamped destination, `nohup setsid`, PID,
exit/status files, and a dedicated stdout/stderr log. It reads all 16,800 NPZ
files and hashes each one. Monitoring continues until COMPLETE or confirmed
failure. A recoverable crash resumes by launching a new fresh destination; it
never mutates the failed partial or old outputs.

Promotion requires all blocking gates and full reconciliation. After success,
append historical-analysis rows to the canonical registry under lock, freeze
its snapshot/hash, and independently rerun the summary/G8 validator from the
emitted files.

## 9. Stage 3 closure and D2 handoff

Stage 3 writes the D0.1 evidence report inside the Goal root, updates stale
historical reports by append-only corrections where authorized, freezes D2
scenario groups/splits without opening the probe test, and records source,
input, output, registry, environment, command, and Git evidence. A local
milestone commit is optional and, if made, uses an explicit Goal-only file
list. No push occurs.

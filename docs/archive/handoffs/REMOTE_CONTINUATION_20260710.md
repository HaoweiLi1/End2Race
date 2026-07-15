# Remote Continuation Brief — D0.1 v2.2 Revision Cycle (2026-07-10)

Written: 2026-07-10, end of the local session that received the Stage-0 GO
relay, prepared the implementation, and was then paused by the user for
migration to the remote machine.

Authority: **subordinate to `CURRENT_HANDOFF.md`** (current version SHA256
`a13c7023505d561523827ced95b538b140d2c8da529e09663c218ccf576cc44c`, includes
the §12 addendum). This brief changes no authority order and no locked
decision. It records (a) what happened in the paused local session, (b) the
prepared draft material for the v2.2 document revision, (c) verified code
facts with file:line citations, and (d) the migration/verification record.
When this brief and CURRENT_HANDOFF.md conflict, CURRENT_HANDOFF.md wins.

Audience: the Claude Code session on the remote machine
(`haowei@100.95.251.103:~/Documents/End2Race`) that continues this work with
zero conversational context and no access to the local machine's memory.

## 0. Opening instruction for the remote session

> Read `CURRENT_HANDOFF.md` fully (especially §8.1 and §12), then this brief,
> then the B+ design spec and the two v2.1 documents (§4 table there; verify
> SHA256 first). The immediate task is **document revision to v2.2 only**
> (spec + plan), per CURRENT_HANDOFF.md §12.1: close the open §8.1 items and
> the two additional findings, make the spec self-contained, sync, report
> hashes, then STOP for the external reviewer's verdict. Do NOT execute
> Stage 0 from the v2.1 documents. Do NOT launch any scan, D2, PPO, or git
> commit/push. Draft resolutions in §3 below are proposals prepared for
> v2.2 — the reviewer approves them by approving v2.2, not by their presence
> here.

## 1. What happened in the paused local session (2026-07-10)

1. Session re-anchored from CURRENT_HANDOFF.md at its then-current version
   (`72e9de7b…39fb`, 327 lines); local state verified unchanged.
2. The user relayed **GO** ("开始执行") under the then-current §9 branch, i.e.
   Stage-0 authorization against the v2.1 plan.
3. Preparation performed (all read-only): spec/plan v2.1 hashes re-verified
   (`34553fec…`, `61462772…`); pinned interpreter verified locally
   (Python 3.10.20, NumPy 1.26.4); full checkpoint hashes read from
   `logs/p1_validation_20260710_121955/source_archive/checkpoint_sha256.txt`;
   Stage-0 entry git-status snapshots taken to local `/tmp`; source-code
   conventions researched (§4 below); §8.1 items mapped to concrete
   implementation resolutions (§3 below).
4. **Before the first Stage-0 file was created** (S0-1 had not produced
   `tests/test_d0_identity.py`), CURRENT_HANDOFF.md was found replaced on
   both ends (`72e9de7b…` → `a13c7023…`, 384 lines) with the §12 addendum:
   the fresh §8.1 preflight is a **technical NO-GO for Stage 0**; the §9
   branch is now "revise to v2.2 only". The user then ordered: pause all
   tasks, write this brief, migrate necessary files to the remote, continue
   in remote Claude Code.
5. Net effect: **Stage 0 was never started.** Zero of the twelve whitelist
   files (`d0/*.py ×6, d0_audit.py, tests/test_d0_*.py ×5`) exist on either
   end. No process was launched, no commit/push, no memory/report edits. The
   only repository writes of the session are this brief and the `.gitignore`
   sync recorded in §5.

## 2. Current authoritative state (restating CURRENT_HANDOFF.md §12)

- v2.1 is technically NO-GO for Stage 0. Items 4 and 5 of §8.1 are closed;
  item 8 is partial; items 1, 2, 3, 6, 7, 9, 10, 11, 12 are open.
- Two additional findings must be closed in v2.2: the v2.1 spec is not
  self-contained ("Unchanged from v2" / "As v2" refer to text that no longer
  exists anywhere — v2 was overwritten in place), and the Stage-0 status-diff
  must use `git status --porcelain=v1 --untracked-files=all` so individual
  untracked files are observable (`tests/` is already untracked).
- Required next step: produce spec+plan **v2.2**, sync, report double-end
  SHA256, stop for review. Nothing else is authorized.
- The earlier "开始执行" Stage-0 authorization is superseded by the §12
  NO-GO: a fresh GO against v2.2 is required before any Stage-0 file is
  created.

## 3. Prepared draft resolutions for v2.2 (proposals, per §8.1 item)

Item numbers refer to CURRENT_HANDOFF.md §8.1.

1. **Raw label preservation.** Evaluator emits `outcome`/`state_label` ∈
   {`following`, `overtaking`, `collision`} (eval_multiagent.py:335,417).
   v2.2: add field `archived_outcome_raw` (verbatim JSON string) and the
   locked total normalization map {following→follow, overtaking→overtake,
   collision→collision} producing `archived_outcome3`; any other raw label is
   a hard G6 failure, never a silent default.
2. **Exact L2 schema + domain separation + L1 provenance.** v2.2 closes L2 to
   an exact field set (draft): `schema="d0.1-L2-1"`, `assets`
   (`f1tenth_racetracks`), `map`, `ego_raceline`, `opp_raceline`, `ego_pose`
   [hex x,y,theta], `opp_pose` [hex x,y,theta], `ego_init_speed` (hex, ego row
   col 5), `opp_init_speed` (hex, opp row col 5, unscaled — `speedscale` is
   its own field), `speedscale` (hex), `interval` (int), `duration_ticks`
   (int), `sim_dt` (hex), `noise` (hex), `noise_seed` (int). Missing or extra
   keys → hard error. Hash payloads domain-prefixed: `"L1:"`, `"L2:"`,
   `"L3:"`, `"L4:"` + canonical JSON. L1 occurrence provenance exact fields:
   model name, checkpoint SHA256, result dir/tag, results.json SHA256,
   npz_path, episode_key, map, offset.
3. **Sensitivity A emission.** Emit both lists: each `sensitivityA_excluded.tsv`
   row carries its `retained_l2` counterpart (or a twin retained TSV).
   The 12-ID pattern assertion (exactly 12; all cross-map × raceline1; each
   excluded member has the larger resolved ego index) is wired as a blocking
   check for the real RunConfig at geometry mode; synthetic fixtures pass
   their own expected pattern so the assertion path itself is tested.
4. Closed per §12.1 (SensB 3072−300=2772; accounting fields
   excluded_from_exact=300 / already_excluded_by_primary=36 /
   additional_vs_primary=264; the 2736 path is structurally rejected).
5. Closed per §12.1 (single integer k·L shift over the entire
   recorded+terminal rel series; evaluator's open-chain chord-sum L).
6. **G1 comparison set.** Duplicate consistency covers every released derived
   field: `archived_outcome_raw`, `archived_outcome3`, `ego_collision`,
   `opp_collision`, `corrected_outcome3`, `four_state`,
   `interaction_attempt`, `confirmed_safe_pass`,
   `attempted_follow_no_collision`, `censored`, `alignment_failure`,
   collision-event `cause`/`phase` when present — so no arbitrary duplicate
   representative can be chosen.
7. **G8 negative tests.** Fixtures: (a) a mismatching occurrence omitted from
   `outcome_corrections.tsv` → G8 fail; (b) an extra ledger row with no
   underlying mismatch → G8 fail; (c) corrupted `d0_summary` counts that do
   not re-derive from `canonical_episodes.tsv` → G8 fail; plus the
   happy-path mismatch fixture.
8. **Phase sign convention (completes the partial item).** With
   rel = ego − opp corrected alignment at the terminal frame:
   |rel| < 0.6 → `alongside`; rel ≤ −0.6 → `pre`; rel ≥ +0.6 → `post`
   (negative = pre, positive = post). Thresholds match analyze_collisions.py
   (car ≤ 1.0 m, alongside 0.6 m). Simulator flags (direct) stay in separate
   columns from inferred `cause`/`phase`.
9. **Import mechanism.** Every test/audit command runs with `PYTHONPATH=.`
   (plus the pinned interpreter); each test file asserts
   `Path(d0.__file__).resolve() == REPO/'d0'/'__init__.py'` so a `tests/`
   shadow can never be imported silently.
10. **Bytecode + observable whitelist.** Every command runs with
    `PYTHONDONTWRITEBYTECODE=1`; both Stage-0 snapshots use
    `git status --porcelain=v1 --untracked-files=all` (per §12.1) so the diff
    shows exactly the twelve authorized files and nothing else.
11. **Full checkpoint hashes** for the frozen RunConfig (verified identical
    both ends, source file `268e1cae…`):
    - bc `pretrained/end2race.pth`
      `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4`
    - cand160 `…iter0160.pth`
      `77cd79904f0f57c1e7a4914dd0b52384628dce225f9222e4e2274e0eda3b5aa6`
    - cand120 `…iter0120.pth`
      `9f2f47bf46363946ba29c1fe5fcada3a3d5fe514ece6eb160c03b25d8f82b3b3`
    - cand040 `…seed0…iter0040.pth`
      `c7a72f5564a191e103d319a7f66167e6969fb3528534b90bafba77ceb598d7e1`
12. **opened_registry.tsv.** Draft locked schema (columns):
    `schema_version` (`d0.1-registry-1`), `analysis_version`, `mode`,
    `npz_path`, `npz_sha256`, `purpose`. Deterministic content only — no
    wallclock timestamps, so rerun byte-equality holds; append-only
    semantics (rows are never rewritten). Per-run copy lives in
    `--output-dir`; the canonical append-only location is proposed as
    `logs/opened_registry.tsv`, created only at a later authorized stage
    (D0.1 itself writes nothing outside `--output-dir`). v2.2 must state
    both the schema and this location explicitly.

**Self-containment (additional finding).** v2.2 spec must inline everything
it currently cites as "Unchanged from v2" / "As v2". The v2 text no longer
exists (overwritten in place), so the G1–G8 base definitions below are a
**reconstruction** from the plan §5 fixture list and the parent B+ spec
§6.1 stop rules — the reviewer must confirm they match the intended v2
semantics:

- G1 (blocking): same-L2 duplicate occurrences agree on the full §3-item-6
  field set.
- G2 (blocking): occurrence file-set equals the RunConfig manifest —
  missing / extra / stale files all fail.
- G3 (blocking): NPZ integrity — file exists, non-empty, loads, required
  keys present; repeated references to one path record one SHA256.
- G4 (informational): near-duplicate outcome agreement computed only on
  matched-condition adjacent-L3 pairs within an L4 block (same map,
  racelines, speedscale, interval); cross-variant comparisons are not
  computed.
- G5 (blocking): skill-manifest floors — each of skill_F and skill_S must
  contain ≥ 30 ego-involved collision cases (parent spec §6.1 stop rule;
  the 29-vs-30 fixture probes the boundary).
- G6 (blocking): per-occurrence physics/record validity — finite values,
  equal array lengths, recorded frame spacing within ±50% of 0.01 s,
  terminal-gap rule gap > 0 ∧ |gap − 0.01| ≤ 0.005, `final_*` presence,
  `collision == ego_collision or opp_collision`, JSON↔NPZ label
  consistency, raw-label domain check (item 1).
- G7 (blocking): unknown budget — zero `unknown`/`alignment_failure` required
  to publish primary statistics; censored counts reported, censored > 0
  blocks the affected metric pending user decision.
- G8 (blocking): corrections-ledger completeness and summary reconciliation —
  `outcome_corrections.tsv` rows are exactly the archived≠corrected set, and
  every published summary count re-derives from `canonical_episodes.tsv`.

One known spec-text tension for v2.2 to resolve explicitly: spec §3.3 states
the gate as `gap > 0` and `|gap − 0.01| ≤ 0.005` but glosses it as the open
interval `(0.005, 0.015)`; the formula (closed at 0.005/0.015) should be
declared normative, or the gloss corrected — the planned fixtures
(0.004 / 0.01 / 0.016) do not distinguish the two readings.

## 4. Verified code facts from this session (cite, don't re-derive)

All line numbers refer to the synced working-tree versions (§5 hash table).

- Waypoint CSV parsing (evaluator convention): `lines[1:]` (one header),
  `split(';')`, rows with ≥ 6 parts, columns `[1,2,3,5]` = x, y, theta,
  speed — eval_multiagent.py:82-89, utils.py:450-457.
- Start-index resolution is modulo row count: `ego_idx % len(waypoints)`,
  `opp_idx % len(waypoints)` — eval_multiagent.py:102-106,
  utils.py:474-477. Same-raceline opponents: `opp_idx = (ego_idx +
  interval_idx) % len(ego_waypoints)` (eval_multiagent.py:106) — this is the
  arithmetic that produces the OL1 15→14 wrap pair when the last data row is
  bitwise-equal to row 0 (e.g. n_rows=101: (100+15)%101 = 14 vs (0+15)%101 =
  15, identical ego pose).
- Cross-raceline mapping: `find_corresponding_waypoint` = argmin of
  Euclidean distance on (x, y), NumPy first-minimum tie-break —
  utils.py:438-442; then `opp_idx = (ego_map_idx + interval_idx) %
  len(opp_waypoints)` (eval_multiagent.py:104). Consequence: the OL0/OL2
  analogue of the wrap pair collapses to an exact duplicate (same opp idx),
  which is why Sensitivity A is an OL1-only phenomenon.
- Centerline length: `np.loadtxt(path, delimiter=';', skiprows=1)`, columns
  1,2, open-chain chord sum (no closing segment) —
  eval_multiagent.py:152-155.
- NPZ keys (complete set): time, ego_lidar, opp_lidar, ego_desired_steer,
  ego_desired_speed, ego_actual_speed, ego_pose, ego_progress,
  opp_desired_steer, opp_desired_speed, opp_actual_speed, opp_pose,
  opp_progress, collision, ego_collision, opp_collision, final_time,
  final_ego_pose, final_opp_pose, final_ego_progress, final_opp_progress,
  state_label — eval_multiagent.py:356-380. Per-step arrays record pre-step
  state; the post-step terminal frame exists only in the `final_*` fields
  (eval_multiagent.py:319-323).
- Recorded progress arrays already carry the recorder's broken one-directional
  wrap (`if progress < initial − L/2: progress += L`, per car vs own initial)
  — eval_multiagent.py:246-249 (in-loop) and 299-302 (terminal). The D0.1
  unwrap-and-shift must treat them as raw input.
- Metrics JSON fields (complete set): episode_key, state (1/2/3),
  state_label, outcome, ego_collision, opp_collision, map_name,
  ego_raceline, opp_raceline, ego_idx, opp_idx, interval_idx,
  opp_speedscale, sim_duration, noise, npz_path, avg_speed, speed_variance,
  total_distance, collision_occurred, proximity/steering quality fields —
  eval_multiagent.py:413-441.
- Result layout: `eval_results/<tag>_<map>/<state_dir>/{c|o|f}_<episode_key>.npz`
  with state_dir ∈ {collision, overtake, follow} and episode_key like
  `ol1_e2095_o13_s0.5` — eval_multiagent.py:325-340. Aggregated
  `results.json` = {"episodes": {episode_key: metric…}, "final":
  {counts…, "validated": true}} (aggregate_eval.py;
  tests/test_eval_aggregation.py:100-108).
- Test harness convention: plain python, `check(name, cond, detail)` printing
  `FAIL <name>` + exit 1 on first failure, final line `ALL TESTS PASSED`,
  tempfile sandbox with try/finally cleanup — tests/test_eval_aggregation.py.
- G5 floor provenance: parent B+ spec §6.1 D0 stop rule — "either skill
  manifest contains fewer than 30 ego-involved collision cases".

## 5. Migration and double-end verification record (2026-07-10)

Environment: remote `/home/haowei/miniconda3/envs/end2race/bin/python` is
**Python 3.10.19**, NumPy 1.26.4 (local is 3.10.20/1.26.4). Plan §1 pins
"3.10.20"; since execution now moves to the remote box, **v2.2 must re-pin
the interpreter line to the remote reality** (same path, 3.10.19/1.26.4) —
flagged for the reviewer rather than silently absorbed.

Git: local HEAD `32661d2`, remote HEAD `bae93dd` (older; carries the work as
uncommitted files — known caveat, unchanged). Pre-existing remote-only
untracked artifacts left untouched: `HANDOFF.md` (superseded),
`run_d4a_pipeline.sh`, two `end2race_anchor_baseline_*` checkpoints.

File verification (SHA256, both ends, this session): the following were
found byte-identical **before** this migration and were not rewritten —
CURRENT_HANDOFF.md `a13c7023…`; D0.1 spec v2.1 `34553fec…`; D0.1 plan v2.1
`61462772…`; B+ design spec `a3b362a3…`; eval_multiagent.py `130f45dd…`;
evaluate.sh `1c42c315…`; evaluate_ol1.sh `527fdae1…`; ppo_utils.py
`94a65a91…`; train_ppo.py `224f1a5d…`; utils.py `4403af9b…`;
aggregate_eval.py `9d538dcd…`; analyze_paired_eval.py `9aac0577…`;
d0_canonical_audit.py (v1, frozen) `cc18d28c…`; tests/test_eval_aggregation.py
`a8778940…`; model.py `ef8281a9…`; analyze_collisions.py `ca8d0f9e…`;
probe_side_rear_risk.py `278ddeb9…`; logs/ppo_audit_handoff_20260710.md
`4c9e1c64…`; logs/p1_final_report_20260710.md `938ec0ba…`;
logs/final_model_report_20260710.md `d687052e…`;
logs/p1_validation_20260710_121955/source_archive/checkpoint_sha256.txt
`268e1cae…`.

Synced by this migration (explicit file list, no whole-repo operation):
1. `.gitignore` — local `941127fc…` (= commit 32661d2 content, adds
   `pretrained/end2race_anchor*` and `HANDOFF*`) replaced remote stale
   `23f03e55…`. Ignore rules affect `git status` output, which the Stage-0
   whitelist check depends on, so both ends must agree.
2. `REMOTE_CONTINUATION_20260710.md` (this brief) — hash reported in the
   session log at sync time (a file cannot contain its own hash).

Data that already lives on the remote natively and was NOT copied:
`eval_results/**` (incl. the 16,800-episode `p1v_20260710_121955_*` NPZ+JSON
trees), `f1tenth_racetracks/**`, all four checkpoints,
`logs/p1_validation_20260710_121955/`, `logs/d0_canonical_audit_20260710_121955/`
(D0 v1, frozen evidence). The local machine retains its mirror and becomes
read-only for this project after migration.

## 6. Operating mode after migration (for the user/reviewer to confirm)

Work continues on the remote box inside remote Claude Code. Consequences the
v2.2 revision should state explicitly (proposals):

- "Local implementation + synthetic tests" (Stage 0) and all later stages
  now run on the same machine that holds `eval_results/**`. The plan's
  `/tmp/d0_stage0_before.txt` snapshot paths remain valid on the remote.
- The review loop's "sync to remote → double-end SHA256" step reverses
  direction: after each document/code change on the remote, sync the
  explicit allowlist **back to the local mirror** and report double-end
  hashes, OR the reviewer explicitly accepts remote-single-end hashing with
  periodic mirror syncs. Decision belongs to the user/reviewer, not the
  agent.
- Remote worktree remains dirty/uncommitted by design; commits stay deferred
  to Stage 3 with an explicit file list (never `git add -A`), on whichever
  end the reviewer designates.

## 7. Authorization state (unchanged)

Currently authorized: **document revision toward v2.2 only** (spec + plan),
plus allowlist sync + hash reporting for that revision. Explicitly NOT
authorized: creating any of the twelve Stage-0 files, any scan (geometry/
smoke/full), D2/D2.5, any PPO training, any git commit/push, any memory or
report §12 numeric replacement. This migration itself is information
transfer only and confers no execution authority (CURRENT_HANDOFF.md §12.2).

## 8. Suggested first actions for the remote session (checklist)

1. `sha256sum` the §5 table files on the remote; every value must match.
2. Confirm no running D0/D2/eval/PPO processes; confirm none of the twelve
   Stage-0 files exist; confirm HEAD `bae93dd` and the expected dirty status
   (`git status --porcelain=v1 --untracked-files=all`).
3. Draft v2.2 spec+plan: fold in §3 resolutions, inline the reconstructed
   G1–G8 base definitions (marked for reviewer confirmation), swap every
   status-snapshot command to `--porcelain=v1 --untracked-files=all`, re-pin
   the interpreter line (3.10.19 remote), embed the four full checkpoint
   hashes, resolve the §3.3 terminal-gap open/closed wording, and bump both
   revision logs v2.1 → v2.2 with an itemized change list.
4. Sync the two revised documents to the local mirror (or per the §6
   decision), report double-end SHA256, and STOP for the reviewer's verdict.
5. On GO only: execute Stage 0 per plan §4 on the remote box, then stop and
   request Stage 1.

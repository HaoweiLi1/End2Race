# End2Race Session Handoff — B+ v2.2 Objective-Aligned Policy Phase

Generated: 2026-07-10; current checkpoint updated 2026-07-12T20:28:00+08:00.
Repository: `/home/haowei/Documents/End2Race` (local) ↔ `haowei@192.168.2.127:~/Documents/End2Race` (active remote, host `haowei-MSI`). The historical `100.95.251.103` address is retired unless the user changes it again.
Audience: a new chat/agent continuing this work with zero conversational context.

Authority: this file is the current new-chat entry point. Newer numbered
sections supersede older state/authorization text. Current execution authority
is §13; current technical state and proposed next direction are §19, with
§§17–18 and the B+ v2.2 spec/plan retained as technical provenance.
`docs/archive/handoffs/HANDOFF.md`, old remote-goal text, first D0 artifacts, unqualified
P1/final-report bodies, and older Claude memories are historical evidence only.

Concise project navigation is now `docs/ppo/README.md`. The full historical
ledger below is intentionally retained for provenance; use newest §19 for the
post-cleanup state, and use `docs/ppo/EXPERIMENT_HISTORY.md` for the readable
result summary.

## 0. Opening instruction for a new chat

> Read `docs/ppo/README.md`, §19, then §§13 and 17–18. Consult
> `docs/superpowers/specs/2026-07-11-ppo-safety-first-bplus-v2.2.md` only as
> historical technical provenance. Verify live local/remote state before acting. The
> remote Codex goal is revoked and must not be resumed. D2/D2R retain their
> original failed gates; the project owner prospectively made TTC diagnostic
> only for v2.2. Full-non-test sidecar initialization, fitted-checkpoint
> identity and the amended Task 6 empirical-prior remediation have passed.
> The first smoke remains FAILED and immutable; the new three-arm checkpoints
> pass all recall/BCE/specificity bars and independent validation. Stop and
> report before Tasks 7/8 or PPO, as the owner required. No arm has been
> selected, no sixth probe is authorized, and no commit or push is implied.
> Latest numerical state is §18.13: the hierarchical replacement fresh identity passed,
> but replacement Task 6 failed its unchanged recall/type bars. Task 9/10/PPO
> were not run. §19 documents the proposed BC-direct PPO redirect, but does
> not itself authorize a numerical run. Do not use older checkpoint
> eligibility text to proceed.

## 1. Historical one-paragraph state (2026-07-10; superseded by §§14–18)

The PPO post-training project closed its first arc on 2026-07-10: evaluation
integrity was repaired (P0), the three sweep candidates were independently
validated on untouched grids (P1, 16,800 episodes, 28/28 validated, zero
failures), and the user designated **cand160 as the sole final deployed
model** (equivalence-on-Austin + transfer-favorable; cand120 downgraded to an
Austin-specialized exploratory finding). The project then pivoted to a new
lexicographic objective (hold overtake at BC, cut collisions hard) with an
approved design spec ("B+"), and is now in a strict, externally-reviewed
rebuild of the analysis foundation: the D0 canonical audit v1 was run and
**rejected** (estimand inconsistency + missing deliverables), and the D0.1
spec/plan pair is at **v2.1, awaiting the reviewer's final approval check**
before Stage 0 (local implementation + synthetic tests) may begin.

## 2. Objective (user-fixed, lexicographic — final wording 2026-07-10)

1. Hard constraint: overtake rate not below BC (Austin and cross maps).
2. Under that constraint: reduce any-agent collision rate materially
   (target family: RR ≤ 0.70 vs BC, see design spec §2.1).
3. Optional afterwards: slowly improve overtake (not required).

Primary KPI remains any-agent collision (historical comparability); ego-only
reported as breakdown. Data lever: BC collisions are bimodal —
`skill_F` = OL1 × speed {0.5, 0.6} (fast-closing rear-end/following,
~0.4% of overtakes live there) and `skill_S` = OL0/2 × {0.7, 0.8}
(alongside passes). Two skills, trained/curriculum'd separately.

## 3. What happened this session (chronological, with artifacts)

1. **Audit verification** of `logs/ppo_audit_handoff_20260710.md` — all
   claims matched live state (hashes, remote sweep failure, 488/600 case).
2. **P0 repair** (implemented + regression-tested both ends):
   `eval_multiagent.py` exits 0 on success, outcome lives in metrics JSON
   (`outcome`, `ego_collision`, `opp_collision`, scenario fields,
   `npz_path`); NPZ gains per-agent flags + post-step terminal frame;
   new `aggregate_eval.py` strict completeness validation;
   `evaluate.sh`/`evaluate_ol1.sh` switched; `tests/test_eval_aggregation.py`
   reproduces the 488/600 silent failure (12 assertions green local+remote).
   KPI decision: any-agent primary. Offset-grid duplicate-key defect found
   by smoke and fixed (open-interval spacing for offset grids).
3. **P1 validation run `20260710_121955`** (remote, nohup/setsid, ~3h47m):
   BC+cand160/120/040 × Austin off21/42/63/84 + Nuerburgring/MoscowRaceway/
   Hockenheim, 600 eps each. Zero failures. Pre-registered verdict:
   cand160 (primary) missed the pass rule by 4/2400 overtakes (statistical
   tie both KPIs on holdout; favorable on all three cross maps); cand120
   passed with occurrence-level significance (see §6 tier-1); cand040 passed
   weakly. Artifacts: `logs/p1_validation_20260710_121955/` (mirrored),
   `logs/p1_final_report_20260710.md`.
4. **User decision**: cand160 = sole final model
   (`logs/final_model_report_20260710.md`, checkpoint sha `77cd7990…`,
   archived both ends); cand120 = exploratory, upgrade requires a fresh
   pre-specified test set. Remaining sweep groups permanently cancelled.
5. **Diagnosis discussion** (why only ±1pp): safety-by-structural-freezing
   caps the reachable set; cm-scale pass corridors + sparse terminal
   penalties break naive exploration/credit. Old logs showed transformed
   **mean** speed residual at zero, but sampled residuals were not logged;
   therefore "11/11 runs never braked" is withdrawn. Confirmed PPO-side
   temporal problems: credit decay γλ = 0.997×0.99 ≈ 0.987/step
   (1s/2s/3s → 0.27/0.073/0.020; γ=0.997 verified `train_ppo.py:106`,
   sweep used `--gae_lambda 0.99`) and 100 Hz iid exploration.
6. **Structural finding + calibration**: BC demos recorded at 10 Hz with
   previous **desired** speed as input (`demonstration.py:216,286`,
   `train.py:86-87`); deployment/PPO run 100 Hz with **actual** speed.
   User calibration (accepted): NOT a bottleneck and not a PPO-vs-BC
   confounder (all measured at the same operating point); survives only as a
   representation-ceiling hypothesis → decided by the D2 episode-held-out
   probe. Recorded in audit handoff §14.5 (calibrated wording).
7. **B+ design spec** authored (via user/GPT) and committed as
   `docs/superpowers/specs/2026-07-10-ppo-safety-first-bplus-design.md`
   (HEAD `32661d2`). I audited it: math verified exact (D3a-1
   time-equivalence 0.271=0.271; D3a-2 → 0.67@1s), thresholds match
   `analyze_collisions.py` (car ≤ 1.0 m, alongside 0.6 m), §10.1 offset
   claims machine-verified (off10/31/52/73 REJECTED: 108/200 unique;
   **off11/32/75/86 admissible**: 200 unique, zero history overlap).
8. **D0 v1 run** (`d0_canonical_audit.py` → `logs/d0_canonical_audit_20260710_121955/`,
   local mirror exists): stop rules passed after two fixes (exact-pose
   identity; opponent pose added to identity — the OL1 wrap shifts opp_idx
   15→14, which caused the pseudo-conflicts; Austin cross-grid exact
   duplicates agreed 100% = large-scale determinism confirmation).
   **Externally rejected (NO-GO)**: I deleted 12 real scenarios as "shadow
   clones" (estimand inconsistency: identity included opp pose, the deletion
   rule ignored it), overstated "D0 passed" (stop rules ≠ full §6.1
   deliverables), missed attempt/confirmed-pass/transition-matrix/hashes/
   registry. v1 outputs are now **frozen evidence, never written again**.
9. **Seam mislabel discovery** (reviewer, mechanism code-confirmed by me):
   the recorder's progress wrap is per-car vs own initial
   (`eval_multiagent.py:241-245,294-298`) — cars starting just past the
   seam never trigger it, and the two cars are never branch-aligned →
   rel progress off by ±L (the +417 m → −2 m case class). **Historical
   terminal-overtake labels can be wrong near the seam.** D0.1 therefore
   computes `corrected_outcome3` (integer-k branch shift over the whole
   series) alongside `archived_outcome3`, with a full corrections ledger.
10. **D0.1 spec/plan review loop**: v2 → NO-GO (7 blockers) → **v2.1**
    (current, awaiting final review): Sensitivity-A deterministic
    outcome-blind selection rule (keep min resolved ego index; assert
    exactly 12 IDs); Sensitivity-B accounting 300/36/264 (N: 3072−36=3036
    primary; 3072−300=2772 SensB; 2736 = wrong); three-state vs four-state
    separation; `collision_events.tsv` restored (phase via corrected rel);
    G4 matched-condition pairs only; full bootstrap pre-registration
    (scenario-weighted, map-stratified paired L4 block bootstrap, B=10,000,
    seed 20260710, spawn order fixed, BC-zero handling); executable
    red-green sequence S0-1…22 with pinned interpreter and git-status
    whitelist; RunConfig schema `d0.1-runconfig-1`.

No D0.1, D2, feature-replay, or PPO process was running locally or remotely
when this handoff was finalized. Multi-agent execution was explicitly
cancelled; continue single-agent unless the user later changes that decision.
The user also requested no Superpowers workflow for this handoff.

## 4. Authoritative documents (verify SHA256 both ends before relying)

| Doc | Path | SHA256 (2026-07-10) |
|---|---|---|
| B+ design spec (parent) | `docs/superpowers/specs/2026-07-10-ppo-safety-first-bplus-design.md` | committed at `32661d2` |
| D0.1 spec **v2.1** | `docs/superpowers/specs/2026-07-10-d01-p01-canonical-audit-plan.md` | `34553fec5f8853ec640533b2c495bb59b93d6a74a2e09d132ab18957be4d3858` |
| D0.1 implementation plan **v2.1** | `docs/superpowers/plans/2026-07-10-d01-implementation-plan.md` | `614627724df2b67c2597a751eeee17286112abd2075fd16cb1a28dca68b742bf` |
| Audit handoff (historical arc + §14 addenda) | `logs/ppo_audit_handoff_20260710.md` | — |
| P1 final analysis | `logs/p1_final_report_20260710.md` | — |
| Final model card (cand160) + Appendix A canonical reanalysis | `logs/final_model_report_20260710.md` | — |

## 5. Verified code facts (cite these, don't re-derive)

- `eval_multiagent.py:435→` now `sys.exit(0)`; outcome JSON-only (P0).
- Seam/wrap defect: `eval_multiagent.py:241-245` and `294-298` (see §3.9).
- BC demos 10 Hz + desired speed: `demonstration.py:216,286`; BC speed input
  = previous desired: `train.py:86-87`; deploy/PPO actual@100 Hz:
  `eval_multiagent.py:226`, `train_ppo.py:336`.
- γ default 0.997 (`train_ppo.py:106`); sweep λ=0.99; collision classifier
  thresholds car ≤ 1.0 m / alongside 0.6 m (`analyze_collisions.py`).
- Determinism: identical raw scenarios reproduced identical outcomes across
  all Austin cross-grid duplicates, 4 models (D0 v1, G-level evidence).
- Sub-waypoint sensitivity: 3 of 48 near-dup model-cases flipped outcome on
  a one-waypoint opponent shift (supports clustered statistics).

## 6. Numbers ledger — three tiers, never mix

**Tier 1 — published occurrence-level (P1 reports, superseded for inference
but kept as provenance):** Austin holdout 2400: BC 111 coll/1358 ot;
cand160 106/1354 (ties); cand120 91/1381 (p=0.010/0.003 at occurrence
level); cross 1800: BC 113/1095; cand160 100/1107; cand120 124/1088.
Known defect: the four Austin offsets share physical starts (200→108
unique); duplicated deterministic episodes double-counted.

**Tier 2 — canonical D0 v1 (N=3024 variant; superseded estimand, kept as
sensitivity reference):** BC 169 coll (152 ego / 17 opp-only) / 1792 ot;
cand120 Austin 19/6 p=0.0146 (×3 → 0.044 survives), ot 20/8 p=0.0357
(×3 → 0.107 does NOT survive); cand160 cross 25/13 p=0.073, ot 20/9
p=0.061 (trends, not significant); cand040 Austin 12/5 p=0.14. OL1 phases
26/50/1. opp-only floor: 17 keys identical across all 4 models. Strata:
skill_F 504 N / 56 coll / 7 ot; skill_S 1008 / 75 / 792.
`final_model_report` Appendix A already carries these corrected boundaries.

**Tier 3 — reviewer predictions for D0.1 (reconciliation targets ONLY):**
primary N=3036, BC 170/1792; SensA 12 enumerated IDs → N=3024; SensB
300/36/264 → N=2772; OL1 27/50/1; per-candidate 154/168/166 collisions.
D0.1 must regenerate; mismatches surfaced, never absorbed.

## 7. Locked decisions (do not relitigate)

1. cand160 is the **current deployed baseline checkpoint** (user-designated
   after P1; research has since reopened under B+, so it is a baseline, not
   the research endpoint). Claim boundary = "equivalence on Austin +
   transfer-favorable, never significantly worse"; NO holdout superiority
   claim; occurrence-level significance wording is superseded by the
   canonical reanalysis (§6 tier 2). cand120 exploratory-only.
2. Lexicographic objective (§2). BC backbone stays frozen at 100 Hz;
   B+ = PPO-residual-side changes only.
3. 10 Hz demo cadence is NOT called a bottleneck (probe-decidable
   hypothesis only).
4. D3b gate split (blocking mechanism sub-gate vs post-D3c/D3d behavior
   sub-gate). Macro-transition credit fix: D3a-1 (time-equivalent) vs
   D3a-2 (λ per macro step) as separate stages.
5. Austin final pool candidate = off11/32/75/86 (verified admissible),
   pending manifest assertions; off10/31/52/73 rejected.
6. Estimand discipline: primary N=3036 exact-start dev exclusion; 1 cm =
   diagnostic only; 1.0 m clusters = bootstrap units + SensB only.
7. Single-agent continuation; reviewer gates every stage.

## 8. Review-loop protocol and CURRENT authorization state

- Loop: I revise documents → sync to remote → report double-end SHA256 →
  external reviewer (GPT, relayed by user) verifies hashes, absence of
  processes/commits, and issues GO/NO-GO with itemized blockers.
- History: D0 v1 NO-GO; D0.1 v2 NO-GO (7 items); **v2.1 submitted, awaiting
  the final approval check**.
- **Currently authorized: NOTHING beyond document revision.** Explicitly NOT
  authorized: Stage 0 implementation, any remote execution, full scan, D2,
  any PPO training, git commit/push, memory/report §12 replacements (those
  wait for regenerated D0.1 numbers).
- Stage ladder (each = separate explicit user authorization):
  Stage 0 local implementation + synthetic tests (S0-1…22, pinned
  `$PY=/home/haowei/miniconda3/envs/end2race/bin/python`, git-status
  whitelist of exactly 12 new files) → Stage 1 remote analysis-only
  geometry+smoke → Stage 2 full scan (16,800 NPZ ≈ 40 GB read) → Stage 3
  docs/memory/commit/D2-plan, itemized.
- Guardrails: v1 D0 outputs frozen; fresh `--output-dir` refusing non-empty;
  atomic `.partial`→`COMPLETE`; sync allowlist only (never whole-repo, never
  `git add -A`); remote strictly read-only over `eval_results/**` +
  `f1tenth_racetracks/**`.

### 8.1 Final v2.1 review checklist still open

The v2.1 hashes above were verified, but final approval was interrupted by the
request to create this handoff. A new chat must close these points before
declaring Stage 0 GO:

1. Raw JSON labels are `following`/`overtaking`/`collision`, while the spec's
   three-state enum is `follow`/`overtake`/`collision`. Preserve
   `archived_outcome_raw` and lock the normalization to
   `archived_outcome3`; do not call the normalized value verbatim.
2. Close the L2 canonical schema to an exact field set (v2 used "at minimum")
   and domain-separate L2/L3/L4 hash payloads. L1 occurrence provenance must
   explicitly include model/checkpoint and source-result identity.
3. Sensitivity A is outcome-blind and the checked-in cross-map raceline1
   first/last pose **and speed** rows were verified bitwise equal. Emit both
   retained and excluded IDs and assert the documented 12-ID pattern.
4. Sensitivity B is `3072-300=2772`; 36 of the 300 were already removed by
   primary, so the additional removal from 3036 is 264.
5. Apply the single integer `k*L` shift to the entire recorded+terminal rel
   series. Use the evaluator's open-chain chord-sum L.
6. G1 duplicate consistency should include censored/alignment status and all
   released derived mechanism fields, so no arbitrary duplicate can be chosen.
7. Add negative G8 tests for omitted/extra correction-ledger rows and corrupted
   summary counts, not just a happy-path mismatch.
8. Collision events must keep simulator flags separate from inferred cause and
   phase; phase sign is pre for negative corrected rel and post for positive.
9. The plan's direct `$PY tests/test_d0_*.py` commands may place `tests/`
   first on `sys.path`. Use `PYTHONPATH=.` (and verify it) or another locked
   import mechanism.
10. Use `PYTHONDONTWRITEBYTECODE=1` or explicitly handle `__pycache__` so the
    twelve-file Stage 0 whitelist is meaningful. The no-write rule applies to
    analyzer runtime, not creation of the twelve authorized source/test files.
11. Use the full local checkpoint hashes from
    `logs/p1_validation_20260710_121955/source_archive/checkpoint_sha256.txt`.
12. Lock the `opened_registry.tsv` schema and canonical append-only location so
    D2 can extend it without rewriting D0 history.

The seam regression evidence used by this review is:

    eval_results/d2_c_seed2_local_off42_20260706_005227_Austin/
      overtake/o_ol1_e2095_o13_s0.5.npz
    raw terminal rel = +417.1104679 m
    Austin open-chain L = 419.1146336 m
    corrected wrapped rel = -2.0041657 m

## 9. Immediate next step

1. Receive the reviewer's verdict on v2.1.
2. If GO: execute Stage 0 exactly per plan §4 (S0-1…22): create each test
   file first, observe the expected failure text, implement minimally, green
   per-module, combined run, `py_compile`, git-status before/after diff must
   equal the 12-file whitelist. Deliver transcript + snapshots. Then STOP
   and request Stage 1.
3. If NO-GO: revise documents to v2.2; nothing else.

## 10. Pitfalls the next agent must not repeat

- Don't equate "my stop rules passed" with "the spec's gate passed" —
  enumerate deliverables against the spec section, not against your script.
- Don't invent estimand filters mid-analysis; every inclusion/exclusion rule
  is pre-registered, outcome-blind, and emitted as a frozen ID list (S0).
- Don't trust `ego_progress − opp_progress` from NPZ raw: seam wrap is
  broken two ways; always use the D0.1 corrected alignment.
- Don't transcribe reviewer/GPT numbers into results — regenerate and
  reconcile.
- 0.7 s = 71 frames including the post-step terminal frame; select by time
  window, never array length.
- Offsets differing by 21 collide with the Austin grid spacing — never
  design offset families without machine-verifying uniqueness.
- Use the end2race interpreter explicitly; system `python` has no NumPy.
- For local synthetic Stage 0 only, use
  `PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /home/haowei/miniconda3/envs/end2race/bin/python ...`.
- Local machine may shut down: anything long-running goes on the remote via
  `nohup setsid`, with status files.

## 11. Inventory

Git (local): HEAD `32661d2` ("docs: design safety-first B+ PPO").
Modified, uncommitted: `eval_multiagent.py evaluate.sh evaluate_ol1.sh
ppo_utils.py train_ppo.py utils.py`. Untracked: `aggregate_eval.py
analyze_paired_eval.py d0_canonical_audit.py docs/superpowers/plans/
docs/superpowers/specs/2026-07-10-d01-p01-canonical-audit-plan.md tests/`.
Commit deferred by user instruction (Stage 3, explicit file list, no -A).

Remote: same content for the synced files (docs verified by hash; P0 files
verified earlier); remote git HEAD is older (`bae93dd`) and carries the work
as uncommitted files — known reproducibility caveat.

Key artifacts: `logs/p1_validation_20260710_121955/` (P1 run, both ends);
`logs/d0_canonical_audit_20260710_121955/` (D0 v1, frozen evidence);
`eval_results/p1v_20260710_121955_*` (remote, 16,800 episodes NPZ+JSON);
checkpoints: BC `b5a1360f…`, cand160 `77cd7990…`, cand120 `9f2f47bf…`,
cand040 `c7a72f55…` (both ends). P1 runner:
`logs/p1_validation_unattended/run_p1_validation.sh` (+README).

Memory (auto-loaded each session): the index (`MEMORY.md`) names this file
as the sole entry point. `end2race-ppo-primary-kpi.md` and
`end2race-p1-validation-run.md` carry reviewer-applied "CURRENT STATUS
OVERRIDE" headers (2026-07-10) that withdraw the occurrence-level
significance claims, demote cand160 to deployed baseline, and mark N=3024 as
Sensitivity A only — those overrides are authoritative over the older
narrative below them. Full numeric replacement of historical denominators
still waits for regenerated D0.1 numbers (Stage 3).

## 12. Live review and remote-goal sync addendum (2026-07-10)

This addendum records a fresh, document-led review performed after this
handoff was created. It does not use or elevate older memory, and it does not
change the authority order or any locked decision above.

### 12.1 v2.1 review status

The fresh §8.1 preflight is a **technical NO-GO for Stage 0**; the formal
external-reviewer verdict is still pending. Items 4 and 5 are closed. Item 8
is only partial. Items 1, 2, 3, 6, 7, 9, 10, 11, and 12 remain open:

- add `archived_outcome_raw` and lock raw-to-three-state normalization;
- make L1 provenance and the domain-separated L2/L3/L4 schemas exact;
- emit both retained and excluded Sensitivity-A IDs and assert the complete
  12-ID pattern;
- expand G1 to censored/alignment status and every released derived field;
- add the required negative G8 correction-ledger/summary tests;
- spell out negative corrected rel = pre and positive = post;
- run every direct test command with verified `PYTHONPATH=.`;
- suppress/route `__pycache__`, and use
  `git status --porcelain=v1 --untracked-files=all` so the exact twelve-file
  whitelist is observable even though `tests/` is already untracked;
- put all four complete checkpoint SHA256 values in the frozen RunConfig;
- lock the canonical append-only `opened_registry.tsv` path and exact schema.

Two additional review findings must also be closed in the next document
revision: the current v2.1 spec is not self-contained where it says
"Unchanged from v2" / "As v2", and the Stage-0 status-diff procedure cannot
validate individual untracked files without `--untracked-files=all`.

Therefore the §9 branch is: revise the spec/plan to **v2.2 only**, sync them,
report double-end hashes, then stop for review. Do not execute Stage 0 from
the current v2.1 documents.

### 12.2 Live state and remote Codex goal boundary

The live check found no matching D0/D2/evaluation/PPO process on either end,
no D0.1 output, and none of the twelve Stage-0 source/test files. The two
v2.1 documents and the four checkpoint hashes matched bit-for-bit on both
ends. Local HEAD remains `32661d2`; remote HEAD remains `bae93dd`; both
worktrees remain intentionally dirty and uncommitted.

On the user's request, the authoritative documents, reports, and core code
were verified through an explicit 25-file sync allowlist for a future
unattended Codex goal on the remote machine. The sync changed only this
handoff and the parent B+ design (which was missing remotely at preflight);
the other 23 allowlisted files were already byte-identical and were not
rewritten. All 25 files match by SHA256 on both ends after synchronization.

This sync request is information transfer only: it does not itself authorize
Stage 0, a remote scan, D2/D2.5, PPO training, or a commit/push. A future
Codex goal must begin with this file, then the parent B+ design and the two
current D0.1 documents, and must take any expanded execution authority from
the user's separately stated goal objective rather than inferring it from
file synchronization.

## 13. Authority update — unattended remote Codex revoked (2026-07-11)

This section is the newest user authority and supersedes §8/§9/§12.2 and the
historical remote `GOAL_OBJECTIVE.md` wherever they conflict about who may
execute work or whether the remote goal may continue unattended.

1. The user has **revoked all unattended/autonomous execution authority from
   the remote Codex goal**. Its objective and logs remain immutable historical
   provenance; they are not live authority and must not be resumed by an
   independent remote agent.
2. The current primary assistant is now the single controlling agent. The
   user explicitly authorizes it to continue the previously unfinished work:
   run, debug, analyze, solve, optimize, and record the remote experiments and
   results. Technical gates, frozen-data rules, single-causal-change
   discipline, and final-pool one-open integrity remain binding; stage pauses
   no longer require a separate remote-agent reviewer relay.
3. The active remote address is now
   `haowei@192.168.2.127:~/Documents/End2Race` (host `haowei-MSI`). Do not use
   the historical `100.95.251.103` address unless the user changes it again.
4. Live takeover audit at 2026-07-11 16:38 +08:00 found no Codex, D0, D2,
   evaluation, or PPO process. The remote goal had completed D0.1 v2.2,
   Stage 0, production geometry, and the 1,200-occurrence smoke with G1–G8
   passing. It had only prepared, but never launched, the 16,800-occurrence
   full-scan script: its PID/status/log/exit files and output directory were
   all absent. `STATUS.md`/`status.tsv` therefore contain a stale
   `RUNNING/in_progress` state.
5. Immediate continuation: the primary assistant verifies the frozen source
   snapshot, launches and monitors the prepared D0.1 full scan, audits its
   gates and regenerated results, mirrors the reviewed artifacts, and only
   then advances to D2/D2.5 and the evidence-supported B+ PPO route.

## 14. D0.1 closure and current research position (2026-07-11)

This section supersedes the earlier v2.1/v2.2 waiting language and all D0.1
predictions with regenerated evidence. Authority remains §13: the current
primary assistant controls execution; no independent unattended remote Codex
may resume.

### 14.1 Reviewed canonical release

The first recovered full run (`artifacts/d01_full_20260710_235320`) exited 0
but was rejected by the primary assistant's deliverable audit: its
`d0_summary` and G8 implemented only a counts subset, not the complete v2.2
reconciliation/prospective-registry contract. It is frozen intermediate
evidence.

Red-green fixes added full model/BC/strata/phase/opponent-only/corrections
reconciliation, exact D0-v1/reviewer-target checks, prospective registry G8,
independent emitted-summary/registry recomputation, and fsync promotion. Two
fresh full runs then completed byte-identically:

`logs/ppo_next_unattended_20260710_230212/artifacts/
d01_full_reconcile_20260711_170200_{a,b}`

Use `_a` as canonical and `_b` only as its determinism confirmation.
Evidence report:
`logs/ppo_next_unattended_20260710_230212/D01_EVIDENCE_REPORT_20260711.md`.

Key hashes:

- corrected source list:
  `manifests/d01_full_reconcile_source_sha256_20260711_170200.txt`;
- output manifest A/B:
  `425d62097b1463e72fca33f4e08690385bfbd21e6be3a91db900b92e4664bd89`;
- summary A/B:
  `56c9dcdc4af24afdd8b0f69a10e9b71487c75d23466bdde28a5090a214f92505`;
- validation A/B:
  `cf2a8165419bf49a1c3507eab2ca9cf9f0476aa125854374de62db999f5e4613`;
- live/snapshot registry:
  `2d9138258a88c4037d889ae42946b05ba23a04c459e93cc9332a26e615d7761b`.

All 16,800 occurrences were present and hashed. G1–G8 and RELEASE passed;
zero alignment failures, censored/unknown rows, or P1 outcome corrections.
G4 is informational: 13 disagreements among 384 matched adjacent-L3 pairs.

### 14.2 Regenerated numbers and claim boundaries

Primary N=3036, counts `(collision/overtake)`:

- BC `170/1792`;
- cand160 `154/1799`;
- cand120 `168/1797`;
- cand040 `166/1787`.

Sensitivity A N=3024 exactly reproduces D0 v1: BC `169/1792`, cand160
`154/1799`, cand120 `167/1797`, cand040 `166/1787`. Sensitivity B N=2772:
BC `164/1640`, cand160 `146/1647`, cand120 `163/1641`, cand040 `161/1635`.

Primary cand160 clustered bootstrap: all-pool RR 0.906, two-sided 95% CI
[0.818,0.994], one-sided upper 0.982; Austin RR 0.947 [0.776,1.146] with
overtake 716 vs BC 720; cross RR 0.885 [0.787,0.985] with overtake 1083 vs
1072. Therefore cand160 has stronger favorable historical evidence but does
not meet the B+ RR<=0.70 product target or the Austin overtake point floor.
Deployment status remains baseline, not new research endpoint.

cand120 Austin is favorable (RR 0.772 [0.615,0.936], overtake 732 vs 720)
but cross-map direction is worse (RR 1.097 [0.979,1.247], overtake 1065 vs
1072); it remains exploratory. cand040 remains inconclusive.

Primary BC mechanism baseline: 170 any-agent / 153 ego-involved / 17
opponent-only; the same 17 opponent-only L2 IDs occur for all four models.
OL1 phases are 27 pre / 50 alongside / 1 post. skill_F is N=510, 57
collisions, 7 overtakes; skill_S is N=1008, 75 collisions, 792 overtakes.

### 14.3 Immediate next task

D0.1 Stage 3 evidence integration is complete enough to prepare D2. Freeze
episode/L4 groups and train/validation/test splits without opening the D2
test split; implement causal feature replay and grouped baselines/probes; use
grouped out-of-fold evidence for model-family selection. Then run D2.5 before
any new PPO training. No D2/D2.5/new-PPO result exists at this checkpoint.

## 15. D2 closure — test remains sealed, D3 blocked (2026-07-11)

This section supersedes §14.3's statement that no D2 result exists. Authority
remains §13: the current primary assistant is the controlling agent and no
remote Codex goal may resume autonomously.

### 15.1 Locked data and causal replay

D2 used only the D0.1 BC Primary population. The outcome-blind,
map-stratified L4 split contains 3,036 episodes / 244 blocks:

- non-test: 1,928 episodes / 156 blocks;
- sealed grouped test: 1,108 episodes / 88 blocks;
- split domain hash:
  `8faab61cdabc4271f917641ca5facc134d10944ce57bb8470263f98d92b88db6`;
- test-seal file SHA256:
  `cee71d818bc050b0ca0647ee32ed1b5655e471ea60b39133aed7b37fc9c1a87e`.

Non-test extraction replayed 1,505,848 frames with batch-1, framewise GRU
execution, zero hidden reset, waypoint-speed×0.9 at t=0 and previous actual
speed thereafter. All archived desired actions matched bit-for-bit; fast and
exhaustive label geometry matched exactly. The release has 91 ego-collision
and 105 any-collision episodes. Dataset manifest SHA256:
`36b9640c9ec8407f12573bc3543712573283881b73400856a4b25f294b1f57c4`.

### 15.2 Grouped nested-CV results

Five outer L4 folds and three inner grouped calibration folds were used. The
released results are:

| family | 1s recall | 1s safe FA | 2s recall | 2s safe FA | 1s BSS | TTC<2 MAE | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| linear | 0.769 | 0.106 | 0.626 | 0.095 | 0.155 | 0.976s | FAIL |
| static MLP | 0.703 | 0.083 | 0.659 | 0.105 | 0.131 | 1.099s | FAIL |
| temporal T1, frozen-feature taps | 0.648 | 0.075 | 0.571 | 0.098 | 0.078 | 1.073s | FAIL |
| temporal T2, deployable LiDAR/speed/command history | 0.725 | 0.095 | 0.703 | 0.095 | 0.105 | 1.068s | **FAIL** |

T2 passes every collision-classification sub-gate but fails the frozen TTC
requirement (`<=0.300s`) by a wide margin. `skill_S` is especially weak:
1s recall 0.595, FA 0.136, TTC MAE 1.513s. Therefore no probe family is
selected.

Canonical report:
`logs/d2_representation_20260711_174039/D2_EVIDENCE_REPORT_20260711.md`.
Machine-readable evidence:
`logs/d2_representation_20260711_174039/artifacts/d2_evidence_20260711_182700/`.

### 15.3 Test/registry state and decision

The grouped test was never opened. There is no `TEST_OPENING_STARTED`, test
source dataset, test feature array, or test prediction. The append-only
registry has exactly 1,928 D2 rows, all `probe_fit`; it has zero
`probe_select` and zero D2 final-pool rows. Current registry SHA256:
`220d98a97741324a66c0ee2f185a78145aae9211805222cbef95754925616675`.

Decision:
`STOP_D3_TEST_UNOPENED_CONTINUE_D2P5_DIAGNOSTIC`.

No D3 or new PPO training is evidence-authorized. The immediate next task is
D2.5 counterfactual recoverability on non-test BC ego-collision cases. D2.5
may determine whether the residual action set can recover confirmed safe
passes, but even a positive D2.5 result cannot override the failed D2
representation gate; a future deployable actor requires a revised,
independently locked representation stage.

## 16. D2.5 closure — Route R2 action space feasible, D3 still blocked (2026-07-11)

This section supersedes §15.3's statement that D2.5 is the immediate next
task. Authority remains §13: the current primary assistant is the controlling
agent and no remote Codex goal may resume autonomously.

### 16.1 Locked oracle and replay integrity

D2.5 used exactly the 91 ego-collision episodes in the D2 non-test release
and performed no learning. The fixed library used 3/2/1-second requested
leads, 0.5/0.3/0.1-second durations, and ten bounded brake/steer residuals.
All 91 no-op baselines reproduced the archived evaluator arrays and terminal
fields bit-for-bit. All 24 branch-smoke reruns and all 67 reported witness
reruns were also bit-identical.

The full oracle executed 3,340 branches. It found 67 confirmed-safe-pass
witnesses and exhausted the valid library for 24 cases. Recovery spans all
four maps and 54 L4 blocks: skill_F 31/37, skill_S 29/37 (78.4%), and other
7/17. Ten non-witness branches required clipping and were invalidated; no
witness used clipping or positive speed residual.

Every frozen Route-R2 feasibility condition passes. Decision:

`D25_COMPLETE_ROUTE_R2_FEASIBLE_D2_REDESIGN_REQUIRED`.

Canonical report:
`logs/d25_counterfactual_20260711/D25_EVIDENCE_REPORT_20260711.md`.
Machine-readable/full trajectory evidence:
`logs/d25_counterfactual_20260711/artifacts/full_oracle_20260711_185500/`.
Output-manifest SHA256:
`42a31686a1c654bfe702085d0a7ae4f587e02e4807ae9eba33fae7ad600dcca3`.

### 16.2 Registry, test seal, and next task

The append-only registry has 8,163 data rows. Exactly 91 are D2.5
`oracle_search/action_choice`, all `final_pool=false`. Live and full-artifact
snapshot SHA256:
`02fe1db47ffbab7aad01a35180a7902daec3c640cfaa45c9b9b5f56d5256aa31`.

The grouped D2 test remains sealed: no `TEST_OPENING_STARTED`, test dataset,
test features, test predictions, or `probe_select` row exists. No D3/new-PPO
process has run.

D2.5 proves the bounded macro-residual action space is expressive enough for
Route R2; it does not prove that a deployable policy can choose the witness
action. D2 still failed its complete representation gate and selected no
family. Therefore D3/PPO remains blocked. The immediate next task is a newly
specified and independently locked deployable representation redesign using
episode/L4-grouped OOF evidence. D2.5 witness/action outputs may be privileged
training labels, never deployment inputs. The existing D2 test remains
sealed until the redesigned representation passes its frozen non-test gate.

## 17. D2R-G closure — original gate failed, test sealed (2026-07-11)

This section supersedes §16.2's statement that another representation redesign
is the immediate next task. It records D2R-G under its original locked rules;
the project-owner redirect is separately recorded in §18 and does not rewrite
this result.

### 17.1 Full grouped OOF result

D2R-G tested one beam-local spatiotemporal deployable family on the exact D2
non-test release: 1,928 episodes, 156 L4 blocks, 1,505,848 frames, and 91
ego-collision episodes. It used five outer / three inner grouped folds and
retains 20 model bundles plus complete OOF predictions.

| original gate | result | required | pass |
|---|---:|---:|:---:|
| 1s ego recall | 0.868132 | >=0.600 | yes |
| 1s safe FA | 0.098738 | <=0.100 | yes |
| 2s ego recall | 0.835165 | >=0.400 | yes |
| 2s safe FA | 0.102578 | <=0.100 | **no** |
| 1s Brier skill | 0.129928 | >=0.100 | yes |
| TTC<2 MAE | 0.800232s | <=0.300s | **no** |

Immutable original decision:

`STOP_D3_TEST_UNOPENED_D2R_G_FAILED_TTC_AND_2S_FA`.

D2R-G did not pass its complete pre-registered gate, selected no family, and
must never be relabeled as passing. The D2 grouped test remains sealed and is
retired unused under v2.2.

Canonical report:
`logs/d2r_geometry_20260711/D2R_EVIDENCE_REPORT_20260711.md`.
Full artifact:
`logs/d2r_geometry_20260711/artifacts/full_oof_20260711_210200/`.

Key hashes:

- full output manifest:
  `be7936acc95b9a98a3a97d4248d94b11ea8c4ed8adacc82a3dde513323b7c057`;
- OOF report:
  `67880c87225781be3a10b756d9332f4546df9cf09f71dc4859d418baed7c09ad`;
- OOF predictions:
  `18d69344efd72bbb7b3263e1bc98869f8620f2f6da9fa0cc7bce5afdd75fa63c`;
- slice summary JSON:
  `00851bcf99b1727437afa998dc32293243e141ac660cbdef92a41d1313bd286c`.

The remote independent validator passes with zero violations. The full 157 MB
D2R evidence tree is locally mirrored and its manifests verify. The 11.6 GB
upstream D2 arrays remain remote, so the input-dependent full validator is a
remote claim; local verification covers the complete D2R release artifact.

### 17.2 Representation and objective-alignment findings

The same grouped population shows a monotone empirical improvement from
frozen-feature T1 (1s recall/BSS 0.648/0.078), to deployable-history T2
(0.725/0.105), to D2R-G (0.868/0.130). This supports useful risk information
in deployable raw observations and a limitation of the tested frozen-feature
route. It is not a causal proof that freezing alone explains the difference,
because architecture and auxiliary supervision also changed.

A read-only outer-fold-threshold join with corrected D0.1 outcomes found a 1s
alarm in 180/1,823 safe episodes, including 128/1,139 corrected safe
overtakes and 110/1,061 confirmed safe passes. These are alarm exposures, not
executed brakes or lost overtakes. They make direct warning-to-fixed-brake
control unsupported and motivate policy-level `alarm -> action -> outcome`
measurement.

The append-only registry now has 10,091 data rows. Exactly 1,928 are D2R-G
`probe_fit/representation_choice`, all `final_pool=false`. Live/full-snapshot
SHA256 is
`59c8967034e12dbcbcc57f776b6ff246c5a313c9b1ec58641d7eba151c4b4663`.
No `probe_select`, test, D3, or new PPO result exists at this checkpoint.

## 18. Project-owner B+ v2.2 redirect (2026-07-11)

At 2026-07-11T22:08:00+08:00 the project owner explicitly approved a
prospective override and redirect:

1. TTC is diagnostic-only for the next policy stage and no longer a PPO
   admission gate.
2. D2 and D2R-G remain failed under their original gates; neither may be
   rewritten as a pass.
3. D2.5's 67/91 means the fixed bounded library demonstrated confirmed-safe-
   pass recovery on 67/91 tested non-test BC ego-collision cases. It satisfies
   Route-R2 feasibility; it is not a theoretical ceiling, whole-population
   recovery rate, or product RR estimate.
4. No sixth representation probe is the immediate next experiment.
5. The next causal test is a clean three-arm policy comparison using direct
   closed-loop collision/overtake outcomes.

Current project-level decision:

`OWNER_REDIRECT_BPLUS_V22_DIRECT_POLICY_KPI_TTC_DIAGNOSTIC_ONLY`.

### 18.1 Locked three arms

All arms keep the BC driving backbone frozen and use identical macro/action/
objective/data/seed/evaluation settings:

- A `BC_FROZEN`: current frozen BC feature through a trainable 128-d adapter;
- B `SIDECAR_FROZEN`: full-non-test pretrained D2R sidecar frozen;
- C `SIDECAR_FINETUNE`: byte-identical B initialization, sidecar LR `3e-6`
  versus common action-core LR `3e-5`.

Common policy mechanics are 10 Hz macro residual decisions, bounded
`NO_OP/BRAKE` plus steering, zero positive-speed budget, Route-R2 witness
warm-start, and collision/overtake-separated constrained PPO. TTC and a
frozen D2R warning head are logged only for diagnosis.

Full BC unfreezing, beta sweep, positive speed, TTC shaping, fixed alarm-to-
brake control, and opening the D2 test are outside this comparison.

### 18.2 Direct promotion gate and immediate task

Development promotion requires both seeds to have more fixed than new
collisions; each seed and the pooled population must satisfy
`lost-gained <= floor(0.01*N_gate)` on their complete locked development
manifests; at least one collision must become a confirmed safe pass; and no
unexplained interaction-attempt collapse or integrity violation may occur.
This 1pp development tolerance replaced an accidentally stricter zero-loss
implementation before any policy result. Final selection retains the original
product objective: fresh Austin and cross-map collision RR<=0.70 with the
specified uncertainty guards and corrected overtake point counts no lower
than BC.

Authoritative v2.2 documents:

- spec:
  `docs/superpowers/specs/2026-07-11-ppo-safety-first-bplus-v2.2.md`, SHA256
  `c8a29a96cfd9ebdd9e5877ebf8a9703413b05171cd469e482b852d7641bb5fd6`;
- implementation plan:
  `docs/superpowers/plans/2026-07-11-bplus-v2.2-d3r2-implementation-plan.md`,
  SHA256
  `05d1b40d3d98da8b5a4f674fc01b9ec5063fb66fdbe7ae8dfd6879ac32fc34a1`.

The immediate next task is the prospective D3-R2-v2.2 registry append and one
full-non-test sidecar initialization release, followed by a checkpoint-backed
identity rerun. No PPO job has started. PPO begins only after the remaining
registry/manifests, fitted-sidecar identity, and warm-start gates pass locally
and remotely. Remote unattended Codex remains revoked; the current primary
assistant retains single-agent control under §13.

### 18.3 First v2.2 structural release

The isolated `bplus_v22` package now implements locked constants, exact
micro-to-macro signal accounting, variable-length multi-objective GAE, the
NO_OP/BRAKE hurdle distribution, A/B/C representation isolation, separate
critics, the overtake-floor/collision-first selector, and an atomic source
preflight. Seven structural tests pass locally and remotely.

Current live source artifact:
`logs/bplus_v22_d3r2_20260711/artifacts/source_preflight_20260711_230535/`.
Its output-manifest SHA256 is
`b36f1bce28035e64fde5449bf72bb04847f6beefec04f79fda2d802d30e2c784`;
it validates eight pinned inputs and 30 source/runtime files with zero
violations. Earlier preflights `223400` and `230358` remain immutable frozen
intermediates but are superseded by live source changes.

### 18.4 Objective/dual pre-pilot review closure

The development gate now converts the declared 1pp tolerance to exact paired
counts with `floor(0.01*N_gate)` per seed and pooled. The final product gate is
unchanged and strict. The dual starts at 1.0, is clamped to `[0,3]`, uses LR
0.5 and overtake-rate EMA coefficient 0.2, and waits for 32 completed episodes
before updating. Collision/overtake advantages are normalized separately and
combined as `(-A_collision + lambda*A_overtake)/(1+lambda)`. Expected dual lag,
oscillation, or a transient training-overtake dip is diagnostic-only; scheduled
paired closed-loop outcomes alone stop/select arms.

### 18.5 Structural wiring identity release

Canonical artifact:
`logs/bplus_v22_d3r2_20260711/artifacts/zero_identity_20260711_230554/`.
Its output-manifest SHA256 is
`b4d38b2128081fece69fbc9727b7b21850bbc424b2775300986624bf3baaf850`.

One deterministic archived collision case from each of Austin, Hockenheim,
MoscowRaceway, and Nuerburgring was replayed twice for BC and zero-residual
A/B/C. All 16 case/variant rows match BC, their reruns, and the archived
trajectory bitwise; every residual is exact zero; the 10-step macro ledger and
terminal remainders cover every micro-step; remote and local validators pass;
and the registry/test seal retain their locked hashes.

The first attempt `zero_identity_20260711_230425` encountered a corrupt shared
Numba cache before any complete artifact. Atomic cleanup removed the partial.
The successful release used and recorded an isolated cache.

This release is explicitly `STRUCTURAL_WIRING_IDENTITY_ONLY`: B/C used
byte-identical deterministic `UNFITTED_STRUCTURAL_PLACEHOLDER` sidecars. It
does not claim a pretrained sidecar. There has still been no D3-R2 registry
append, sidecar fit, witness warm-start, PPO iteration, policy checkpoint, or
policy outcome. The next task is the prospective registry append and exactly
one six-epoch full-non-test `SIDECAR_INITIALIZATION_ONLY` fit, followed by a
checkpoint-backed identity rerun. PPO remains blocked.

### 18.6 Full-non-test sidecar initialization and fitted identity

This section supersedes §18.5's pending-sidecar statement.

The live source preflight is now
`logs/bplus_v22_d3r2_20260711/artifacts/source_preflight_20260712_075911/`
with output-manifest SHA256
`96e1a829427c84fb7c3303a7a71265332cb93f93b8e33315635921acc0bf47ab`.
Eight B+ tests plus affected D0/D2 registry regressions pass locally and
remotely, and all 38 source/runtime files match byte-for-byte.

Before fitting, `registry_plan_20260712_075931` froze exactly 1,928
`D3-R2-v2.2/actor_pretrain` rows and the complete expected-after registry.
All rows are opened non-test data, `final_pool=false`, and disjoint from the
1,108 sealed-test L2 IDs. The live registry moved exactly once from
`59c8967034e12dbcbcc57f776b6ff246c5a313c9b1ec58641d7eba151c4b4663`
to
`753c478700a499fa24f1c216f77e810bd1f634ba9cc7d934a2ec707593b1439c`.
Both machines now match the planned expected-after bytes and contain 12,019
data rows.

Exactly one six-epoch, full-non-test D2R-G fit ran on the remote RTX 4080
SUPER. Canonical release:
`logs/bplus_v22_d3r2_20260711/artifacts/sidecar_init_20260712_080012/`.
It is labeled `SIDECAR_INITIALIZATION_ONLY`, used 152,338 deterministic
sampled frames, and emitted one 333,115-parameter checkpoint with state-dict
SHA256
`34158ecba356ec9d524529e0d928e452140f8da2f98c59d491f0a5cf26cd12e5`.
Its output-manifest SHA256 is
`ac9e10661102efb1164aaa7b6d57fdbf0a63be9c1af454ddc9954d30031163a7`.
It is initialization for byte-identical B/C state, not a D2R gate pass,
representation selection, arm selection, or policy result. A fresh remote
process reloaded it and reproduced the 512 fixed predictions bitwise on the
same CUDA device. Local validation intentionally checks portable structure,
finite tensors, normalization, registry, source, and hashes rather than
claiming cross-device floating-point identity.

Canonical fitted-checkpoint identity:
`logs/bplus_v22_d3r2_20260711/artifacts/zero_identity_20260712_080201/`,
output-manifest SHA256
`b4d82bffe58900fb58dddeacd6457491f5a0b08216ddfd615c5ffc0dbc4f2c53`.
It records `FITTED_SIDECAR_CHECKPOINT_IDENTITY` and
`pretrained_sidecar_gate_satisfied=true`; all 16 BC/A/B/C case rows match BC,
rerun, and archive bitwise. The D2 test remains sealed. No warm-start update,
PPO iteration, arm selection, policy checkpoint, or policy outcome exists.

### 18.7 Task 6 witness/preservation lock and fixed-update warm-start

This section supersedes §18.6's pending-warm-start statement.

Canonical live source preflight:
`logs/bplus_v22_d3r2_20260711/artifacts/source_preflight_20260712_091836/`,
output-manifest SHA256
`5736a744481a3cdf456178db212a64c0f305488095478791d72240abbb527a81`.
It covers 41 source/runtime files and eight pinned inputs; the same structural
suite and source bytes pass locally and remotely. Two earlier preflights and
one earlier manifest remain immutable superseded intermediates. They were
created before CUDA synthetic backward checks exposed two launch/kernel
requirements; no live registry row or data fit was touched by those checks.

Canonical pre-fit manifest:
`logs/bplus_v22_d3r2_20260711/artifacts/warmstart_manifest_20260712_091851/`,
output-manifest SHA256
`8b53294f7049d53a0a7261c9daa8acfe9df88857e8ba211aafe09bf05ad915a2`.
It freezes:

- all 67 independently rerun, non-clipped D2.5 confirmed-safe-pass witnesses;
- 291 exact intervention macro labels and 5,136 witness no-op labels;
- 602 preservation episodes selected before fitting as the minimum
  domain-separated SHA256 L2 within each
  `(map, skill, opponent_raceline, L4)` stratum from 1,061 eligible corrected
  BC confirmed-safe-passes;
- 48,762 preservation no-op labels, for 54,189 total examples;
- a 1,024-update shared schedule with every 256-example batch containing
  128 intervention, 64 witness-no-op, and 64 preservation-no-op examples;
- actor inputs limited to deployable BC feature, causal LiDAR, speed, and
  previous-command histories. Privileged witness/outcome fields remain
  separate labels and never enter an actor tensor.

There is zero witness/preservation L2 duplication and zero sealed-test/final
use. The registry appended exactly 669
`D3-R2-v2.2/actor_pretrain/action_choice` rows without rewriting D2.5 or prior
rows. Local and remote registries now have 12,688 data rows and exact SHA256
`aff5f03db06836c6c51ff53944ed2ec2e521fbe777cc7d26228a15a9362d0b0d`.

Canonical smoke release:
`logs/bplus_v22_d3r2_20260711/artifacts/warmstart_smoke_20260712_091950/`,
output-manifest SHA256
`150b41fa68fbec40442741bdc6613355ab41b44cc0fdb4591fa9e455438dc8be`.
On the remote RTX 4080 SUPER, A/B/C each completed exactly 1,024 fixed
updates with the same ordered schedule, no early stopping, and no selection.
All BC and immutable shadow-sidecar hashes remained unchanged; A/B policy
sidecars remained unchanged; only C's trainable policy-sidecar encoder changed.
A fresh remote process passed the full same-device CUDA recomputation; local
artifact/hash validation passed without CPU/GPU float comparison.

CUDA hardening was prospective: the runner fail-closes unless
`CUBLAS_WORKSPACE_CONFIG` is set before process start, and C's locked
360-to-18 beam pool uses a fixed 20-beam `avg_pool1d`. On the remote CUDA
preflight this was bit-identical to the original adaptive-pool forward and
provided the deterministic backward missing from the pinned PyTorch adaptive
kernel.

Warm-start losses are diagnostic only. All arms reduced aggregate diagnostic
loss, but all three still chose NO_OP for every positive brake-gate diagnostic
at the deterministic threshold (`gate_recall=0`, `specificity=1`). This does
not select/reject an arm or authorize an unregistered schedule change; it is a
mechanism finding that must be preserved for the later closed-loop warm-start
evaluation.

Task 6 is complete. No PPO iteration, arm selection, closed-loop policy
evaluation, final-pool access, or D2 test opening occurred. The next work must
follow the remaining prospective v2.2 plan gates (Task 7/8 closure, then the
fresh no-learning/closed-loop preflights); PPO remains blocked until those
gates explicitly pass.

### 18.8 Owner-approved Task 6 remediation (2026-07-12)

This section supersedes §18.7's statement that Task 6 is complete. The owner
approved a prospective mechanics remediation before any PPO or Task 7/8 work.

Read-only code audit established that the two zero-residual identity claims
are load-bearing on the fresh deterministic policy naturally selecting NO_OP:
at each macro boundary `ZeroResidualActor` evaluates the policy distribution
and asserts gate=NO_OP and exact physical zero before the simulator composes a
checked zero tensor. Therefore both 16/16 identity releases depend on the
fresh `INITIAL_BRAKE_LOGIT=-6.0`; that constant remains byte-for-byte
unchanged. The remediation applies a separate empirical bias only after fresh
initialization is recorded and immediately before Task 6 fitting. Both
identity releases and the fitted sidecar remain valid without rerun.

Prospective spec amendment version `bplus-v2.2-owner-redirect-2` is in
`docs/superpowers/specs/2026-07-11-ppo-safety-first-bplus-v2.2.md`, SHA256
`799b40a4ab6680b094c5565573ef61833df38fc5a509351cfa369ed1d1d89b04`.
It freezes
two initialization phases and derives the fit bias solely from the exact
training schedule: 90,089 brake-positive occurrences / 262,144 total,
prevalence `0.3436622619628906`, logit `-0.6470161225499584`. The unchanged
873-example diagnostic subset has 200 positives and constant-marginal BCE
`0.538180595747381`.

Every arm must satisfy `gate_recall>0`, `gate_loss<0.5382`, and
`gate_specificity>0.05`; precision, specificity, and confusion counts are
mandatory. Any arm failure fails Task 6 for all arms. The immutable
`warmstart_smoke_20260712_091950` files and integrity validation remain
unchanged, but its Task-6 decision is now `FAILED` because recall was zero and
gate loss exceeded 0.5382 in all arms. Its checkpoint is not PPO-eligible.

There is not yet a v2.2 PPO entry point, so the earlier spec did not enforce
checkpoint continuity in code. The amendment now requires every future PPO
pilot to load the exact accepted warm-start checkpoint for the same arm and to
verify release/file/state hashes plus `task6_acceptance_passed=true`. Starting
PPO from a fresh `-6.0` policy is forbidden. Warm-started policies are expected
to deviate from BC; near-NO_OP applies only to fresh identity initialization.

Current task: finish the new remediation source release and manifest, rerun
Task 6 once in a new artifact, and report all per-arm bars. Do not start PPO or
Tasks 7/8 until all three arms pass and independent validation succeeds.

### 18.9 Task 6 remediation PASS (2026-07-12)

This section supersedes §18.8's pending-remediation state. No Task 7/8 or PPO
work occurred.

Canonical amended source preflight:
`logs/bplus_v22_d3r2_20260711/artifacts/source_preflight_20260712_100006/`,
output-manifest SHA256
`6921a91a9265ca8cd630dbe68ce90c53c38677e6eedd7b1ff37a03f955b671fe`.
All nine B+ tests pass on both hosts; the unchanged identity test explicitly
retains fresh `INITIAL_BRAKE_LOGIT=-6.0`.

Canonical remediation manifest:
`logs/bplus_v22_d3r2_20260711/artifacts/warmstart_remediation_manifest_20260712_100032/`,
output-manifest SHA256
`72b3ef0e25a41984e256454218e36640bd9e045430671b57af570e7d1896f24e`.
It records 90,089/262,144 fit prevalence, float64 derived bias
`-0.6470161225499584`, applied float32 bias `-0.6470161080360413`, the frozen
200/873 diagnostic marginal BCE, and the first smoke's immutable FAILED
provenance. Registry state was `already_appended`: zero new rows and all 669
action-choice rows reused; live SHA stayed
`aff5f03db06836c6c51ff53944ed2ec2e521fbe777cc7d26228a15a9362d0b0d`.

Canonical passing release:
`logs/bplus_v22_d3r2_20260711/artifacts/warmstart_remediation_20260712_100124/`,
output-manifest SHA256
`57c6f900d57da1c59b46354c1502304576ad2ab352b03a29c8756f4bfce83252`.
Each arm completed the unchanged 1,024-update schedule. Frozen BC/shadow and
A/B sidecar invariants passed; C alone changed its permitted sidecar. The
fixed diagnostic results are:

| Arm | Recall | Gate BCE | Precision | Specificity | TP / FP / TN / FN |
|---|---:|---:|---:|---:|---:|
| `BC_FROZEN` | 0.995 | 0.156037 | 0.884 | 0.961 | 199 / 26 / 647 / 1 |
| `SIDECAR_FROZEN` | 0.735 | 0.361333 | 0.659 | 0.887 | 147 / 76 / 597 / 53 |
| `SIDECAR_FINETUNE` | 0.975 | 0.226790 | 0.762 | 0.909 | 195 / 61 / 612 / 5 |

Every arm passes recall `>0`, gate BCE `<0.5382`, and specificity `>0.05`;
`task6_acceptance_passed=true` and `ppo_checkpoint_eligible=true`. These are
mechanics gates, not an arm ranking or closed-loop policy result.

The run completed remotely from 10:01:45 to 10:02:45 +08:00 with exit 0. A
fresh second process (`validate_warmstart_remediation_20260712_100259`) passed
exact same-device CUDA recomputation from 10:03:14 to 10:03:34; local
artifact/hash-only validation also passed. D2 test/final pools remain sealed,
registry bytes are unchanged, and PPO has not started.

Per owner instruction, stop here and report. Do not proceed to Tasks 7/8 or
PPO without subsequent direction.

### 18.10 Tasks 7–9 structural gates PASS; stopped before closed-loop learning (2026-07-12)

This section supersedes §18.9's stop state. The owner accepted Task 6 as a
shared mechanics gate only and authorized Task 7/8 plus the no-learning
preflight. Task-6 recall/BCE differences were not used to rank, reject, or
promote A/B/C.

The current prospective spec keeps version
`bplus-v2.2-owner-redirect-2` and has SHA256
`c058d41dce01649cd87cf0b1b99e01264ad4fad2cca5059e8f41a4ff672938ac`.
It now distinguishes the fresh natural-NO_OP identity gate from the post-Task-6
checkpoint-continuity gate: Task 9 loads each exact accepted warm-start
checkpoint, records its natural deterministic hurdle decision, and then
forces the physical residual to exact zero before simulator composition.
Natural NO_OP is not required after warm-start. This amendment was landed
before the Task-9 run and does not alter either earlier 16/16 identity result.

Task 7 now has executable contracts for five independently stored episode
signals (any-agent collision, ego collision, terminal overtake, confirmed
safe pass, progress), named critic targets, actor detachment from
critic/privileged inputs, separate advantage normalization, independent
actor/critic pre/post-clip norms, deterministic dual state/transient logs, and
per-seed plus pooled direct-outcome gates. The fresh dual remains locked to
1.0; checkpoint restoration accepts only ordered, range-checked state. EMA
and lambda do not update before the 32-episode boundary. Ten B+ structural
test programs pass on `haowei@192.168.2.127`.

Canonical live source preflight:
`logs/bplus_v22_d3r2_20260711/artifacts/source_preflight_20260712_103501/`,
output-manifest SHA256
`7b29ec64ec0db0a022b36a91e1a6bd28963403b4aab1cc9bbe0421f2d636e05c`.
It validates eight pinned inputs and 44 source/runtime files on both hosts.

Canonical Task-8 release:
`logs/bplus_v22_d3r2_20260711/artifacts/task8_manifests_20260712_103509/`,
output-manifest SHA256
`4a7c3343246166e68f56bb6f48e1d5b44b192696d741f03ba35e372942bfacb8`.
It freezes 288 unique development L2 scenarios (96 representative/
preservation, 96 `skill_F`, 96 `skill_S`), the remaining 1,640 non-test L2
scenarios for training, 67 explicitly non-held-out D2.5 witness-training
cases, an eight-case no-learning smoke, and 18 byte-identical Cartesian
evaluation jobs (three arms x three snapshots x two seeds). The
representative panel is the only distributional development estimate;
`skill_F`, `skill_S`, and witness panels are mechanism-enriched. All 67
recoverable D2.5 cases share an L4 with the witness-training set, so the
L4-disjoint recoverability panel is truthfully recorded as unavailable (zero),
not presented as held-out evidence. The development and training L2 sets are
disjoint, the 1,108 D2 test cases remain sealed, and the registry remains
byte-identical at
`aff5f03db06836c6c51ff53944ed2ec2e521fbe777cc7d26228a15a9362d0b0d`.

The earlier complete Task-8 intermediate
`task8_manifests_20260712_103419` (output-manifest
`76a944b28abf328d0cda5772428c25e2e1aebaa66ea1067eab8c4ed15cb15cce`)
is superseded, not rewritten: its smoke rows omitted three simulator scenario
fields. The first Task-9 command failed before its first replay and atomically
removed its partial directory. No numerical result or complete Task-9
artifact was produced by that attempt.

Canonical Task-9 release:
`logs/bplus_v22_d3r2_20260711/artifacts/task9_checkpoint_preflight_20260712_103517/`,
output-manifest SHA256
`c61be6b70ff31146d3506b09235afbae426d7c9ffa7834bf27eac6a2a5faed4f`.
On the remote RTX 4080 SUPER it replayed eight fixed scenarios twice for BC
and checkpoint-backed forced-zero A/B/C: 32 case/variant cells and 64 total
simulations. Every run-1/run-2 action/state/outcome/terminal array is
bit-identical; every A/B/C trajectory equals BC; all forced residuals are
exact zero; all three arms exercised a short terminal macro; and independent
P0 completeness plus output hashes pass. Natural checkpoint decisions were
non-NO_OP as expected (aggregate brake decisions across the eight cases: A
32, B 266, C 237); these are continuity diagnostics and not arm-ranking
evidence. Remote and local artifact validators both pass.

No closed-loop warm-start policy evaluation, PPO iteration, arm selection,
candidate promotion, D2-test opening, or final-pool access occurred. Stop at
this gate and report. The next prospective action is Task 10 closed-loop
warm-start evaluation; PPO remains blocked until that separate gate is
executed and reviewed.

### 18.11 Task 10 integrity PASS, mechanism gate FAILED (2026-07-12)

The owner authorized deterministic nonzero-residual Task 10. Before running,
the interpretation boundary was tightened prospectively in the v2.2 spec,
current SHA256
`dee6a042309c483fcc792d56b4c08c2fbf055024eb284d4f1886ab123a4fe18d`:
all 288 development rows are `held_out_policy_generalization=false`; Task 10
cannot rank arms; Task 11/12 may perform only internal development selection;
the 80–100 iteration medium stage is still opened-development evidence; and
the first fresh-pool generalization evidence appears only at the one-open
final evaluation.

Canonical current source preflight:
`logs/bplus_v22_d3r2_20260711/artifacts/source_preflight_20260712_113240/`,
output-manifest SHA256
`23c77a9c3fe4b9185db91fa7adc8e5069fd2709a63b0f1f14632ab6077135364`.
It validates eight pinned inputs and 46 source/runtime files.

Task 10 canonical numerical release:
`logs/bplus_v22_d3r2_20260711/artifacts/task10_warmstart_20260712_105740/`,
output-manifest SHA256
`605d3413df35cef8ddd9cdd4769164f52016edeaa7c9e58e1c34ba234fb9ed46`.
The remote RTX 4080 SUPER evaluated all 288 scenarios for BC and all three
accepted warm-start checkpoints. The release has 1,152 complete episode rows
and 62,345 macro-decision rows; checkpoint hashes, ordered L2 sequence, source
hashes, outcome accounting, and output inventory validate on both hosts.

The initial live validator reported a Cartesian mismatch after all numerical
runs had finished. Read-only audit found a non-load-bearing Task-8 display-key
bug: `manifest_order` used dynamic `len(list)` inside `list.extend(generator)`,
creating even-numbered and cross-panel duplicate labels. The actual byte-locked
TSV row order and 288 unique L2 IDs drove every evaluation; each variant has
the identical complete ordered L2 sequence. No episode was omitted, repeated,
or reassigned. The numerical release was not rewritten. Validation now uses
the canonical ordered-L2 x variant key, and the generator requires contiguous
0..287 labels. Corrected future Task-8 release:
`task8_manifests_20260712_113241`, output-manifest SHA256
`84ea20b76a42f87f6a1e6bb25eecc214defbf2123a405b33b7a6c2631afdba9b`.

BC on this mechanism population has 24 collisions and 138 terminal
overtakes. Task-10 outcomes are:

| Arm | collisions | fixed / new | gained / lost overtake | collision->confirmed pass | episodes braked | brake decisions | clipped episodes / microsteps |
|---|---:|---:|---:|---:|---:|---:|---:|
| A `BC_FROZEN` | 91 | 11 / 78 | 26 / 31 | 8 | 206/288 (71.5%) | 6.77% | 13 / 128 |
| B `SIDECAR_FROZEN` | 54 | 14 / 44 | 15 / 71 | 3 | 287/288 (99.7%) | 35.55% | 12 / 181 |
| C `SIDECAR_FINETUNE` | 67 | 13 / 56 | 28 / 54 | 6 | 287/288 (99.7%) | 23.40% | 14 / 199 |

All three create many more new collisions than they fix and all three have
`lost_overtake > gained_overtake`. Clipping is steering-bound clipping; no
recorded macro-boundary request had negative speed. Therefore the locked
shared stop has both `all_arms_net_overtake_loss=true` and
`any_action_clipping=true`; `task10_passed=false`. Task 11/PPO is blocked.

The exact baseline OOF 100 Hz alarm ledger contains 26 false-alarm episodes
among 264 BC-safe episodes (9.85%, matching the earlier 9.9% concern). A
braked 21/26 of them and lost nine overtakes; B and C braked 26/26 and lost
14 and seven overtakes respectively. Task-6 gate recall does not order this
closed-loop behavior: B had the lowest Task-6 recall but the highest Task-10
brake-decision rate. This confirms Task-6 was only a mechanics gate and shows
warm-start action overgeneralization/anti-conservatism, not evidence that B's
sidecar is useless.

No PPO iteration, arm selection, medium/final promotion, D2-test opening, or
fresh-pool access occurred. The next allowed work is prospective remediation
of the shared warm-start/action implementation (especially steering-bound
composition and excessive brake generalization), followed by a new Task-10
release. Do not start Task 11 from these checkpoints.

### 18.12 Owner-approved post-Task-10 remediation in progress (2026-07-12)

This section supersedes §18.11's pending-remediation sentence, not its FAILED
result. The owner approved immediate implementation and execution, while
keeping Task 11/PPO blocked until replacement Task 10 is reported and audited.

The mechanism audit confirmed three distinct defects in the old action path:

- the old class-weighted schedule encoded a 34.37% unconditional brake prior;
- steering was outside the brake hurdle and therefore active at every macro
  decision (A alone introduced 20 no-brake new collisions);
- the actor held a macro-boundary physical delta for ten micro-steps, so BC
  movement inside the macro could push the composed steer outside the
  simulator bound. The 39 clipped arm/episode cells had residual magnitudes
  within the nominal 0.2 budget; the violation was in composition.

The project-owner prospective specification is now
`bplus-v2.2-owner-redirect-3`, SHA256
`7faa0133428cd7d4cbaaf90a4dd9fd7247fd1cff770fd0a3a0630ef458dbe976`;
implementation plan `bplus-v2.2-d3r2-plan-2`, SHA256
`fc24a4da3292cb5dfe7d517a20a128355bfecb6c143a7f178cdb6c2aacabef57`.
The old Task-6 PASS, old Task-9 PASS, and Task-10 FAILED artifacts remain
immutable historical evidence, but the new action-schema decision makes all
three old checkpoints ineligible for future PPO.

The replacement action is a canonical four-coordinate hierarchical latent
`[intervention_gate, steer_latent, conditional_brake_gate, brake_latent]`.
`NO_OP` zeros both physical channels exactly; steer-only, brake-only, and
combined witnesses remain expressible. The held object is the latent. At each
100 Hz micro-step its requested residual is projected against the current
deployed BC steering/braking headroom, and evaluator clipping must be an exact
identity. The unchanged fresh `INITIAL_BRAKE_LOGIT=-6.0` remains intact; a new
top-level intervention gate also starts at `-6.0`. Old three-coordinate
actions/checkpoints fail closed.

Replacement Task 6 uses D2 outer folds 0--3 for the new action-head fit (542
episodes, 43,902 unique macros) and fold 4 for calibration. Thresholds are
chosen independently per arm from exactly 75 strict confirmed-pass negative
episodes that were used by neither the old warm-start selection nor Task 8/10;
positive witnesses cannot tune the threshold. This is L4-held-out action-head
calibration only: it is not historically fresh, representation-held-out, or
policy generalization evidence. Required episode/macro/type recalls and a
maximum 7/75 false-intervention episodes are frozen in spec §6.3.

Load-bearing order: new source release and structural tests; new fitted-sidecar
fresh natural-NO_OP identity; frozen replacement Task-6 manifest; remote CUDA
fit/calibration; replacement checkpoint-backed Task 9; then Task 10 in full,
steer-disabled, and brake-disabled diagnostic modes. The ablations cannot rank
arms. Numerical recomputation remains remote on
`haowei@192.168.2.127`; local work is structural/hash/artifact validation.
The remote unattended Codex authority remains revoked.

### 18.13 Hierarchical replacement Task 6 FAILED (2026-07-12)

This section supersedes §18.12's in-progress state. It does not rewrite the
old Task-6/9 PASS artifacts or either Task-10 result.

The current 55-file source preflight is
`source_preflight_20260712_155907`, output-manifest SHA256
`3bf937b5365c627edb225d1a0c51874add9a7db2f2d7430420c2fea0d15a136c`.
The new natural-NO_OP identity
`hierarchical_zero_identity_20260712_155938` passed all 16 BC/A/B/C rows
bitwise, SHA256
`f6ba2d94da85256a222a71294d31cd05b282745e2eb0548dea0a40c7e05c9e0e`.

The frozen Task-6 manifest
`hierarchical_warmstart_manifest_20260712_160123` contains 542 fit episodes /
43,902 fit macros, nine fold-4 positive witness episodes / 39 intervention
macros, and 75 fold-4 negative confirmed-pass episodes / 6,075 macros. Its
output-manifest SHA256 is
`b70137560938aea79e0b750886797b10431fdb89287453f54495f529839005ff`.
Before fitting, exactly 75 prospective `action_choice` reuse rows were
appended; registry is now 12,763 rows with SHA256
`60caa4175e0aeeee9cb1788293245f5737fe5bce4ae1958da20276ffd3f1b6ac`.

Canonical failed numerical release:
`hierarchical_warmstart_20260712_160212`, output-manifest SHA256
`ffd3c59cbbfe39931930d88e5f7d2781b5c44174ab4a635e30bd490665c2a0d9`.
All arms completed 1,024 updates. Artifact integrity and an independent fresh
same-device CUDA recomputation pass with zero violations, but all-arm
acceptance fails. A/B/C each have false interventions 7/75, positive episode
recall 0/9, positive macro recall 0/39, steer-only episode recall 0/4,
brake-containing episode recall 0/5, conditional brake recall 25/25, and
conditional specificity 0/14. No checkpoint is PPO-eligible.

Read-only fit diagnostics show the failure occurs before L4 generalization:
calibrated fit recall is 0.1508/0.0873/0.0675 for A/B/C and all conditional
specificities are zero. The natural schedule delivered only 1,502
intervention occurrences (1.47 per batch; 231/1,024 batches had none), versus
131,072 in the old class-heavy schedule—87.3x fewer. This is evidence of
positive/type underexposure and coupled gradient clipping, not a valid arm or
sidecar ranking.

Stop here. Replacement Task 9, Task 10, Task 11/PPO, D2 test, and final pools
were not run. The next recommended experiment keeps the 75-negative
calibration population and every acceptance bar unchanged but prospectively
separates learning exposure from deployment prevalence using balanced
top-gate and conditional-gate auxiliary streams. That schedule change needs
explicit project-owner approval; do not infer it from this failed result.

## 19. Local consolidation and PPO-direction clarification (2026-07-12)

This section supersedes §18.13 only as a statement of the proposed next
direction and local workspace organization. It does not change any recorded
result, open a dataset, authorize a remote run, or relabel a failed gate.

The project owner requested a local-only cleanup because historical raw
rollouts, checkpoints, superseded artifacts, and dated plans had obscured the
actual state. Detailed consolidated records now live under `docs/ppo/`.
Canonical evidence and every substantive failed release remain preserved.
Historical runs/reports/handoffs and explicitly superseded artifacts were
locally classified under `logs/archive/` and `docs/archive/handoffs/` according
to `docs/ppo/ARTIFACT_RETENTION.md`; they were not deleted. Raw rollout
mirrors, checkpoints, caches, code, and canonical artifacts were not removed.
No remote content was touched.

The current scientific state is:

1. deployable observations contain useful risk information;
2. a bounded residual action library recovered confirmed safe passes for
   67/91 tested non-test BC ego-collision cases;
3. supervised witness warm-start is not a reliable PPO admission gate: its
   first closed-loop form failed Task 10, and the hierarchical replacement
   failed Task 6;
4. no v2.2 PPO iteration or product-KPI result exists.

The simplified proposed next experiment is a BC-direct PPO mechanism pilot.
Fresh deterministic behavior remains bit-identical to BC and keeps
`INITIAL_INTERVENTION_LOGIT=-6.0`. Exploration is introduced only in the
stochastic training behavior distribution and must be represented exactly in
stored log-probabilities. The only blocking preflights are BC identity,
bound-preserving composition without hidden clipping, and equality between
sampled/executed/logged actions. Direct collision and corrected-overtake KPIs
decide the pilot. TTC and supervised warm-start metrics are diagnostics only.

This is a documented proposal, not numerical-run authorization. No new PPO,
Task 6, Task 9, Task 10, D2-test, or final-pool process was started during
cleanup.

Tier-3 verification passed five canonical output manifests (D0.1, D2, D2.5,
D2R-G, hierarchical Task 6) and all 33 standalone local test programs. Exact
log/document inventories and original-to-archive path mappings are recorded
under `docs/ppo/` and `logs/archive/PATH_MIGRATION.tsv`.

Shared-workspace caveat: at 2026-07-12 20:22 +08:00, a concurrent external
process (not this Tier-3 organization flow) switched the local worktree from
`main` to `chore/commit-evidence-pipeline` and created seven local commits,
ending at HEAD `a4b0128`. The commits capture D0/D2/D2.5/D2R/B+ code, PPO/eval
changes, specs/plans, and a consolidated experiment record. They were not
reset, amended, pushed, or otherwise rewritten by Tier 3. The remaining
Tier-3 indexes/organization documentation is visible in the local worktree.

---
name: handoff-log
description: Record a finished, rejected, abandoned, or interrupted End2Race experiment into .agents/ANALYSIS.md and its decision-grade summary into .agents/HANDOFF.md. Use whenever an experiment arm finishes, a direction is rejected, a run is interrupted, a production default changes, or either document is stale relative to the repo. Also use to audit the two-document split for drift.
---

# Recording End2Race experiments into ANALYSIS.md and HANDOFF.md

The two files have different, non-overlapping jobs:

> **`HANDOFF.md` lets a new agent decide what to do next in under a minute.**
>
> **`ANALYSIS.md` preserves enough design, data and reasoning that no completed
> training, eval or diagnostic must be repeated.**

Write the current production/default, active state, compact verdict, stop/reopen
rule and model registry to `HANDOFF.md`. Write exact controls, panel/cohort
definitions, sample sizes, matched changes, uncertainty, checkpoint bands,
mechanism evidence, failure analysis and limitations to `ANALYSIS.md`.
`EXPERIMENTS.md` owns tests/scripts implementation logic; do not duplicate it.

### Minimal experiment-script rule

Before creating any experiment helper, search the current entrypoints and tools.
Prefer, in order:

1. reuse an existing command or script unchanged by setting its existing parameters;
2. make the smallest parameter or argument change to an existing script;
3. minimally extend an existing script when the new experiment shares its workflow;
4. only when no existing implementation can express the authorized experiment,
   create `scripts/` if it is absent and add one narrowly scoped script there.

Do not create a parallel helper, generic framework, wrapper shell file, automatic
model selector, or extra log file for convenience. Keep a new script limited to
the new component or diagnostic that cannot be represented by current parameters.
Record its basename, CLI contract, input/output schema and deletion consequence in
`EXPERIMENTS.md`; do not copy the source or output paths into `HANDOFF.md` or
`ANALYSIS.md`.

Before authorizing deletion of an experiment tool or regression file, make
`EXPERIMENTS.md` sufficient for functional reconstruction by a coding agent.
For every necessary file, record:

- its purpose and place in the call/import dependency graph;
- every CLI flag and non-path default that changes behavior;
- public symbols imported by another file, including signatures and return keys;
- the core algorithm in ordered steps, with constants, units and numerical tolerances;
- input schemas, stable output basenames and fields consumed downstream;
- deterministic ordering, seed, resume/partial-write and atomic-write behavior;
- fail-closed validation and expected exception conditions;
- the corresponding test names, fixtures, assertions and skip gates;
- runner-level fixed controls, the single changed variable and checkpoint/panel loop.

Audit this record mechanically before cleanup: every source basename, CLI flag,
cross-file import and test function must appear in `EXPERIMENTS.md`. Treat this as
functional equivalence, not byte-for-byte source recovery; say explicitly which
temporary formatting, old default paths and arbitrary future re-analysis capability
will be lost.

This skill is a **procedure, not a project snapshot**. Do not hard-code current
section numbers, baseline metrics, panel sizes, active-run state, or
production choices into `SKILL.md`. Resolve those facts from the current repo and
artifacts each time, then update both documents at their proper level.

Read JSON/CSV/run artifacts to establish the facts, but do **not** turn either file
into an artifact index. No evidence-entry path lists, recomputation-script lists,
or provenance manifests over analysis outputs — those directories may later be
cleaned. Preserve full decision evidence in `ANALYSIS.md`; preserve the compact
verdict and action boundary in `HANDOFF.md`.

Model checkpoints are the exception: they persist and their identity matters, so
they get one hash registry in `HANDOFF.md`. See "Hash models, not products" in §3.

Cost anchor: a full training run plus its eval panels is expensive. A direction
that gets re-explored because the handoff lost its result wastes that. A direction
that gets *wrongly closed* because the handoff overstated a result wastes more.
Both failures come from the same cause — recording conclusions without their
scope.

---

## 0. Before writing anything: verify against the repo

Never write either document from the conversation. Write it from the artifacts.
The conversation contains claims; the repo contains facts, and they drift.

```bash
cd /home/haowei/Documents/End2Race
PY=/home/haowei/miniconda3/envs/end2race/bin/python

sed -n '1,220p' AGENTS.md
git rev-parse HEAD
git status --short | head -40
pgrep -af '[r]un\.sh|[t]rain_ppo\.py|[e]val_multiagent\.py|[e]valuate\.sh' || true

# Inventory only. A row count is not by itself proof that a run completed.
# Two checkpoint layouts coexist: the canonical `update<N>/actor.pth` and an older
# `checkpoints/actor_uNNNN.pth`. Count both, or a canonical run reads as empty.
for d in post-trained/ppo_*/; do
  printf '%-70s metrics=%-5s canonical=%-4s legacy=%s\n' \
    "$d" \
    "$(wc -l < "$d/metrics.jsonl" 2>/dev/null || printf missing)" \
    "$(find "$d" -mindepth 2 -maxdepth 2 -name 'actor.pth' 2>/dev/null | wc -l)" \
    "$(find "$d/checkpoints" -maxdepth 1 -name 'actor_u*.pth' 2>/dev/null | wc -l)"
done

# CLI and live tree, not a copied flag count from an older handoff.
$PY train_ppo.py --help
ls ppo/ scripts/ 2>/dev/null
ls *.sh *.py 2>/dev/null          # root entry points; some may have been retired
```

Resolve the layout before judging completeness, and never hard-code either shape
into a conclusion: directory names, layouts and entry points change across
cleanups, so report what the inventory actually found.

The bracketed `pgrep` patterns deliberately avoid matching the audit command
itself. If a process match is ambiguous, inspect its full command and cwd before
declaring a training/eval active.

To call a run **complete**, verify all applicable signals:

1. `run_config.json` says what update range was intended;
2. `metrics.jsonl` contains the expected warmup/formal rows, all parse and are finite;
3. the expected last actor and critic checkpoint exist **in whichever layout that run
   uses** (`update<N>/actor.pth` + `update<N>/critic.pt`, or the older
   `checkpoints/actor_uNNNN.pth`), plus any final actor artifact;
4. no matching process is still writing the directory;
5. any claimed eval panel has its expected unique scenario count, zero errors,
   result/trace reconciliation, and machine-readable summary.

A run that is complete but whose *guardrail* panel was never evaluated is not
comparable: a later arm cannot compute matched removed/created against it. When
recording a baseline, check that every panel the handoff cites as a decision gate
actually has per-episode results for that baseline, not just an aggregate number.

`metrics rows = 1 + formal updates` is a useful necessary check for the common
fresh-start run, not a universal completion theorem. Resume, extension, and
analysis-only directories need their own contract.

Do not add paths merely to prove that an artifact existed. Implementation paths
belong only in a short searchable call chain when they help the next agent find
the active code; experiment-result paths and deleted-file recovery instructions
do not belong in either document.

---

### 0.1 Separate fact, inference, and unknown

For every important statement, decide which class it belongs to:

- **Verified fact** — directly present in source, run config, checkpoint contents, or
  machine-readable artifact;
- **Interpretation** — a mechanism consistent with the evidence but not directly
  identified; label it as an interpretation/hypothesis;
- **Unknown / not run** — planned, interrupted, missing power, or never evaluated.

Never promote “planned” to “tested”, “one value tested” to “axis swept”, a
configuration-level failure to a single-cause diagnosis, or an interpretation to
an implementation fact.

---

## 1. Where a result goes

| What happened | Where it goes |
|---|---|
| A direction reached a verdict | Full topic section in `ANALYSIS.md`; compact verdict in `HANDOFF.md` |
| A small experiment with a clean verdict | Compact analysis index plus one-line handoff decision if it changes action |
| A run was interrupted | `HANDOFF.md` last-activity block; detailed state/attempt in `ANALYSIS.md` |
| Production default changed | `HANDOFF.md` production, defaults, hash registry and verdict; analysis topic updated |
| New CLI flag / module | `EXPERIMENTS.md` implementation; only production-relevant default in `HANDOFF.md` |
| Nothing changed but time passed | Nothing. Do not restate. |

Resolve section locations by heading; section numbers may change. Never let a
result live only in `HANDOFF.md`: its evidence must exist in an `ANALYSIS.md`
topic section or compact index.

---

## 2. The shape of an ANALYSIS topic section

Six parts, in this order. Skipping any one of them is what makes a handoff
un-actionable later.

### 2.1 Verdict table — first, before any narrative

Every arm, its production decision, and the numbers that drove it. The reader
must be able to stop after this table and still act correctly.

```markdown
| 臂 | 本质 | `<primary panel>` | `<KPI guardrail>` | `<diagnostic panel>` | `<generalization panel>` | 决策 |
|---|---|---:|---:|---:|---:|---|
| B（baseline） | `<verified baseline mechanism>` | **`<c / o>`** | **`<c / o>`** | `<c / o>` | `<c / o>` | **production** |
| T | `<single changed axis>` | `<c / o>` | `<c / o>` | `<c / o>` | `<c / o>` | `<artifact-backed verdict>` |
```

State the units once (`ego collision / overtake`) and the fixed controls once
(seed, updates, what was held constant). Bold the production row. The values
above are placeholders: resolve every number and baseline identity from current
artifacts instead of copying an old example.

### 2.2 What problem it was trying to solve

One paragraph on the *failure mode*, not the method. "Ego 跟在同线 opponent 后面
需要持续减速却做不到" is reusable knowledge. "我们试了时序噪声" is not.

Then the call chain, so the next agent can find the code:

```text
<entrypoint> --<treatment flag> ...
-> <scheduler/vector env> -> <environment> -> <implementation module>
```

Resolve the real symbols from current source; ANALYSIS should contain a
searchable call chain, not a remembered one.

### 2.3 The mechanism result — separately from the acceptance result

**This is the part handoffs most often lose, and the most expensive to lose.**

A rejected intervention often *proved its mechanism* and only failed on
generalization. If you record only "rejected", the next agent re-derives the
mechanism from scratch. Record both, explicitly labelled:

> 不要把这条线读成"`<mechanism>`没用"。`<treatment>`在`<target cohort>`上
> 直接改变了`<mechanism metric>`并化解`<removed>/<total>`个目标；它失败在
> `<guardrail/generalization>`，见下节。

### 2.4 Why it failed — at the mechanism level, not the metric level

"Austin600 变差了" is a measurement. "same-line 碰撞下降、off-line 碰撞上升，
净结果只取决于该面板 same-line 占比" is a *model*, and a model is what lets the
next agent predict whether a new variant will fail the same way.

Push until you can state the failure as a bounded rule. Generic examples:

- *"`<signal>`在失败中是结果传感器而不是起势传感器，所以单纯加剂量不能修复信用时机。"*
- *"`<sampler>`只告诉PPO哪里多采样，没有提供此处应该采取什么安全动作的标签。"*
- *"已测面板中`<target stratum>`下降、`<other stratum>`上升；净结果受面板构成影响。"*

If the best you have is "it got worse", say so plainly and mark the mechanism as
unknown. A stated gap is useful; a fabricated mechanism is not. Avoid words such
as “proves”, “always”, “never”, and “only cause” unless the experiment actually
identifies that claim.

### 2.5 Stop rule — what specifically must not be retried

Be concrete enough to be checkable. Name only axes actually run and supported by
artifacts. A proposed or interrupted arm is not a swept axis.

```markdown
1. production 保持 `<verified baseline>`，不传 `<treatment flag>`；
2. 不再扫 `<axis A/B>`——列出已经运行的取值、关键结果和失败原因；
3. `<axis C>` 只被提出/中断，保持“未测试”，不得写成已否决；
4. 不要用 `<diagnostic panel>` 单独翻案——说明它为何不是验收面板。
```

Include what would *legitimately* reopen it (a materially different task
distribution, a new control). A stop rule with no reopening condition reads as
dogma and gets ignored.

### 2.6 Durable core record — independent of artifact retention

Finish the section with the smallest self-contained record that survives later
artifact cleanup:

- exact experimental axis and fixed controls;
- panel/cohort definitions and denominators;
- primary totals plus matched identity changes and uncertainty;
- mechanism metric and its direction;
- guardrail/generalization result;
- what remains unknown and the precise stop/reopen rule.

Do not append an “evidence paths” section. Do not list JSON/CSV/report files or
hashes. The next agent should not need those files to understand the verdict.

**But distinguish products from inputs.** Analysis and eval outputs are products:
they are cleanable, so their content must be inlined. Some artifacts are *training
inputs* — a classified scenario cache, a frozen pool file, a checkpoint another
experiment initializes from. Deleting those does not just lose a record, it makes a
configuration unreproducible without re-paying the compute that built it. When a
section depends on such an input, name it and state the rebuild cost, so a later
cleanup does not remove it by association with the products around it.

Source code and tests are likewise not evidence directories. Point at the module
that implements a mechanism; that pointer stays valid after artifacts are gone.

---

## 3. Numbers: the non-negotiables

**For matched same-scenario binary eval comparisons, report identity, not just
totals.** Two checkpoints with the same collision count may have different
failures. These comparisons need:

```text
removed / created, and a paired McNemar p
```

For overtakes, use `lost / gained` plus the paired p. Net change alone has caused
wrong conclusions in this project more than once.

Do **not** fabricate McNemar tests for training rollups, unmatched panels,
continuous mechanism metrics, or comparisons whose scenario identities differ.
For those, state the unit of analysis, denominators, uncertainty, and why a paired
test is unavailable.

**Always name the panel and its role, but resolve both dynamically.** Read the
panel manifest/preregistration and the current handoff. Common names include
Austin600, near400, hard subsets, and held-out maps, but their size and role are
not permanent properties and must not be copied from this skill.

For every panel record: scenario count, scenario-identity source, map(s), outcome
definition, whether traces were saved, and whether the panel was used to design
the treatment. A diagnostic panel cannot silently become a production gate.

**Always carry the current baseline.** Resolve its run/model identity, checkpoint
number, configuration, and same-panel metrics, then quote them in the table. A
reader must not have to scroll to interpret a number, and materially different
configurations must not share the same label.

**Identify models by path and update.** Record the canonical initializer, production model,
and each arm's evaluated checkpoint path in one `HANDOFF.md` registry section. Do not
calculate or maintain file hashes for models, analysis products, or eval outputs.

Two things the registry must capture beyond the raw values:

- **Equivalence sets.** One model can live at several paths — a reproduction run, an
  extension run, and an experiment's control arm may hold identical weights. Say so.
  Otherwise a later reader finds the control arm under a directory name that does not
  match the authoritative one and re-evaluates just to check.
- **Naming traps.** An incomplete run beside its `_rerun`, or two checkpoints from the
  same run at different updates, produce different digests for similar-looking names.
  Note which one the reported numbers came from.

Verify every digest against disk before writing it, and state when it was measured.

**Always state the evidence boundary.** Single seed. Fixed panel. Panel reused
from the design phase. Training-distribution effect vs. deployed behavior.
Where an artifact already records a boundary (`evidence_boundary`,
`interpretation.negative_result`), quote it rather than paraphrasing.

**Record checkpoint variance, not just the chosen checkpoint.** A selected value
drawn from a wide, oscillating band means something different from one drawn
from a stable plateau. Report the actual band from the current artifact when you
have it; do not copy a historical band from an example.

---

## 4. Recording an interrupted run

An interrupted run is the single highest-value thing to record, because the next
agent's first question is always "what was running and did it finish?"

Put the compact current state in the `HANDOFF.md` last-activity block, and preserve
the detailed attempt in `ANALYSIS.md`, with all four elements:

1. **Exact state** — dir, intended update range, parsed finite metrics rows,
   last actor/critic checkpoint, final-artifact presence, mtime, and process state;
2. **What it was testing** — the preregistered question, in one sentence;
3. **What is legal to do next** — an explicit short list;
4. **The prohibition** — never continue writing into an existing run directory;
   restart into a new one.

If a preregistration file exists, quote its panel list and flag anything it
excludes that later evidence says should be included.

---

## 5. Maintenance rules

**Every section carries its own scope.** An older dated section inside a current
document must say so in its header, and its status column must be re-verified or
deleted. Status words like 正在训练 / 待运行 rot fastest — check them against
`pgrep` and the full completion contract every time you touch the file, or
replace them with a completed/rejected/interrupted verdict.

**Keep volatile facts out of this skill.** Exact counts, current model identity,
panel roles, and section numbers belong in the current HANDOFF/ANALYSIS snapshot.
If this skill starts naming the current winner or claiming an axis is closed, it
has become stale project documentation and must be generalized.

**Never enumerate a growing directory.** `post-trained/`, `eval_results/`,
`analysis_results/` change every session. Document the *naming convention* and
the command to list them. Listing a small subset of a growing directory is worse
than listing none — it reads as complete.

**Fix contradictions immediately.** If one section says a path is absent and
another cites it as current, resolve the contradiction against the repo. Once a
reader finds one contradiction, they stop trusting the rest of the document,
including the parts that are correct.

**Demote, don't duplicate.** HANDOFF contains the decision and pointer; ANALYSIS
contains the full result; EXPERIMENTS contains implementation logic. Do not copy
full analytical tables back into HANDOFF or implementation inventories into either
analysis document.

**Delete only history, never boundaries.** Compressing an old section is fine.
Dropping the caveat that made its conclusion valid is how a hedged result turns
into a false certainty two sessions later.

---

## 6. Self-check before finishing

Read your own text and answer each of these with a specific line number:

1. Can a new agent tell, in under a minute, what is running and what is not?
2. For each rejected direction — can they tell *why*, at the mechanism level?
3. For each direction — could they mistakenly conclude it is worth retrying?
4. Did an analysis/eval path or evidence-entry list leak in? Are checkpoint hashes
   confined to the HANDOFF registry, verified against disk, and absent elsewhere?
5. Does every claimed number match its source artifact at write time? Spot-check three.
6. Does every matched same-scenario binary eval comparison have removed/created
   (or lost/gained) and a paired p? Are non-paired analyses clearly labelled?
7. Is there any status word that `pgrep` or a metrics row count would contradict?
8. Is there any internal contradiction between two sections?
9. Did you preserve the mechanism result of every rejected arm?
10. Would the next agent need to re-run anything because ANALYSIS omitted its evidence?
11. Did you call any planned/interrupted value “tested”, or infer that an entire
    axis was swept from a single setting?
12. Did any baseline number, panel role, directory count, or section number
    leak into this skill instead of being resolved into HANDOFF/ANALYSIS?

Question 10 is the whole point. If the answer is yes, the missing number is
almost certainly already in an artifact — go find it and write it down.

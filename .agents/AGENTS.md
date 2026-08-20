# End2Race — agent instructions

## Read first

`.agents/`只保留五份职责文档，不得创建或恢复并行的专题HANDOFF、Gate汇编、单方法
预注册或阶段报告。开始工作时先读`HANDOFF.md`与`GUIDE.md`；需要审计历史证据、实现实验
组件或修改Python时，再按职责读取其余文档。

**`.agents/HANDOFF.md` — what is true now.** The single continuity document: current
production configuration and baseline, last-activity block, and one verdict/stop-rule
section per experiment line. Several directions have already been rejected with paired
evidence, so check the verdict sections before proposing anything — the result is
already recorded and does not need re-running. Resolve the current count and section
numbers from the handoff rather than copying them into this instruction file.

**`.agents/GUIDE.md` — how experiments must be designed and run.** Long-lived
methodology, not project state: offline screening before spending a training run,
confound disclosure for continuation runs, evaluation-panel roles (which panels carry
acceptance authority and which do not), required paired statistics, artifact layout,
and naming. Follow it for any new experiment; it deliberately differs from some
historical practice in this repo, and the handoff marks where.

**`.agents/ANALYSIS.md` — why each result is believed.** It contains complete panel
definitions, paired statistics, mechanism analysis, uncertainty and evidence boundaries.
Use its method-family index instead of searching for temporary method numbers.

**`.agents/EXPERIMENTS.md` — how historical experiment components can be rebuilt.**
It records implementation, data-flow and regression contracts, not current verdicts.
Use its method-family index before recreating a deleted tool.

**`.agents/STYLE.md` — how new code must look.** Derived from the user's own
`train.py` / `eval_multiagent.py`: no `from __future__` imports, no module docstrings,
one `add_argument` per physical line with no `help=`, sparse type annotations, long
lines tolerated, orchestration in `if __name__ == "__main__":` rather than `def main()`.
Read it before writing or refactoring any Python in this repo.

## Ground rules

- Verify against the repo, not the conversation. Source of truth order: current
  source and machine-readable artifacts > each run's `run_config.json` >
  `.agents/HANDOFF.md` > `.agents/ANALYSIS.md` > `.agents/EXPERIMENTS.md` > README >
  git history.
- That order applies to **implementation facts**. For finished experiment results the
  HANDOFF verdict is authoritative and ANALYSIS supplies its detailed evidence: analysis
  and eval directories are cleanable, so a missing file never overturns a recorded
  conclusion and is never a reason to re-run.
- Never reset, checkout, or clean the working tree. It holds uncommitted
  user-owned experiment code and results.
- Never write into an existing `post-trained/` run directory. Restart into a new one.
- Never overwrite `pretrained/end2race.pth`.
- Never delete `post-trained/collision-cache/`. It is a training input, not an
  analysis product; rebuilding it costs a full frozen-BC classification pass.
- From 2026-08-13 the project is in reporting and consolidation mode. Do not propose,
  implement, preregister, queue, or add CLI/config for any new technical method. There
  is no current `run.sh` or queued experiment; the unexecuted BC-native fixed-preference
  contract is historical context, not authorization to recreate or run it.
- Use the project interpreter: `/home/haowei/miniconda3/envs/end2race/bin/python`.
- Change one axis per experiment.
- Name every method by its mechanism. Do not use lettered or numbered placeholders as
  method names. Round/Gate IDs are historical locators only and must be accompanied by
  a descriptive name; `A/B` is reserved for an explicit control/treatment comparison.
- Assign every new experiment to one primary family from `.agents/GUIDE.md` §2.0:
  reward/objective, pool/sampling, exploration/curriculum, BC/reference regularization,
  counterfactual action learning, representation/auxiliary supervision, checkpoint
  combination, or diagnostic/engineering Gate. Put its records with that family.

## Recording results

When an experiment finishes, is abandoned, is interrupted, or a production
default changes, update `.agents/HANDOFF.md` following
`.claude/skills/handoff-log/SKILL.md`. That file defines the required shape:
verdict table, the problem being solved, the mechanism result recorded separately
from the acceptance result, a mechanism-level failure explanation, and a stop rule.

Write the detailed design, paired evidence, uncertainty and applicability boundary into
`.agents/ANALYSIS.md`; write only implementation or reconstruction contracts that remain
useful into `.agents/EXPERIMENTS.md`. Temporary preregistration or report documents must
be merged into these responsibility documents and removed when the line closes.

Write the numbers into the handoff itself — **do not leave conclusions as pointers
to `analysis_results/` or `eval_results/`**. Those are cleanable products. Record
checkpoint paths and update numbers directly; do not maintain file hashes.

The standard it enforces: **a new agent must be able to act on the result without
re-running training or eval.**

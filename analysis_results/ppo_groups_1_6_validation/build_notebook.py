#!/usr/bin/env python3
"""Build and execute the Claude-analysis validation notebook."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_results" / "ppo_groups_1_6_validation"
NOTEBOOK = OUT / "claude_analysis_validation.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    markdown(
        """# Claude Groups 1–6 analysis validation

## tl;dr

The analysis is directionally strong but needs revision before it becomes a decision record. The
G3/G5 worker confound and the target-KL failure are real. Material corrections are: use paired
scenario inference instead of a universal ±4.2/8-collision rule; align rollout KPIs to
`rollout_policy_update`, not the same-number post-update checkpoint; change target-KL 0.02 early
stops from 10/20 to 11/20; and describe G5 clip 0.20 as non-monotonic.
"""
    ),
    markdown(
        """## Context & Methods

This notebook is a diagnostic companion. Its source calculation is
`validate_claude_analysis.py`, which reads the recorded `run_config.json`, `metrics.jsonl`,
`episodes.jsonl`, actor checkpoints, and Austin600 `results_multi.json` files.

### Key Assumptions

- Austin600 is a fixed paired scenario panel. Exact paired p-values are descriptive and unadjusted
  for checkpoint selection; they do not establish cross-seed or out-of-panel generalization.
- `ego_collision=true` with `opp_collision=false` is classified as ego/wall-like, matching the
  simulator collision flags; it is not a full geometric root-cause label.
- The worker-count association is verified, while the internal source of process-topology
  sensitivity remains unresolved.
"""
    ),
    code(
        """from pathlib import Path
import json
import pandas as pd

ROOT = Path(globals().get("_REPO_ROOT", Path.cwd()))
if not (ROOT / "analysis_results" / "ppo_groups_1_6_validation").exists():
    raise FileNotFoundError("Run from the End2Race repository root")
OUT = ROOT / "analysis_results" / "ppo_groups_1_6_validation"

summary = json.loads((OUT / "validation_summary.json").read_text())
claims = pd.read_csv(OUT / "claim_review.csv")
target_steps = pd.read_csv(OUT / "target_kl_steps.csv")
target_eval = pd.read_csv(OUT / "target_kl_eval.csv")
paired = pd.read_csv(OUT / "paired_scenario_tests.csv")
actor_diff = pd.read_csv(OUT / "old_vs_long_actor_diff.csv")

print(summary["overall_assessment"])
print(f"Claims reviewed: {len(claims)}; high-severity issues: {(claims.severity == 'high').sum()}")
"""
    ),
    markdown(
        """## Data

### 1. Claim-level audit

The table separates numerical discrepancies from causal or methodological overreach.
"""
    ),
    code(
        """print(claims[["claim", "assessment", "severity", "required_revision"]].to_string(index=False))
"""
    ),
    markdown(
        """## Results

### 2. G3 versus G5 diverges before PPO actor updates

The old G3 and G5 long clip 0.20 configs differ in `env_workers` and total horizon. Total horizon
does not alter the constant LR/clip schedules before U20; the first warm-up rollout already differs,
so the observed model divergence cannot be caused by clip or later checkpoints.
"""
    ),
    code(
        """g3 = summary["g3_vs_g5"]
print("Config differences:", json.dumps(g3["config_differences"], indent=2))
print("Old warm-up:", g3["old_warmup"])
print("Long warm-up:", g3["long_warmup"])
print("First structural difference:", g3["first_structural_difference"])
print("\\nActor checkpoint differences:")
print(actor_diff.to_string(index=False))
"""
    ),
    markdown(
        """### 3. Target-KL changes the optimization path, not just a scalar metric

The gate is evaluated before the current minibatch update. A large KL at the gate was produced by
earlier completed minibatches; stopping cannot undo those steps. Completed steps therefore vary
sharply by update while the critic still receives its five epochs.
"""
    ),
    code(
        """step_summary = target_steps.groupby("label").agg(
    early_stops=("early_stop", "sum"),
    completed_steps=("steps_completed", "sum"),
    planned_steps=("steps_planned", "sum"),
    max_trigger_kl=("trigger_kl", "max"),
)
print(step_summary.to_string())
print("\\nTarget-KL 0.04 update path:")
print(target_steps[target_steps.label == "target-KL 0.04"][["update", "steps_completed", "early_stop", "trigger_kl"]].to_string(index=False))
print("\\nTarget-KL 0.04 evaluation path:")
print(target_eval.round(4).to_string(index=False))
"""
    ),
    markdown(
        """### 4. Paired scenario evidence replaces a universal count threshold

The relevant observation is each scenario's before/after outcome. The p-values below are exact
two-sided McNemar/binomial tests on discordant pairs and are unadjusted for selecting among several
checkpoints.
"""
    ),
    code(
        """print(paired.round(4).to_string(index=False))
"""
    ),
    markdown(
        """## Takeaways

1. Keep the direction of the six-group conclusion, but revise the statistical and temporal-alignment claims.
2. Treat `env_workers` as an experiment parameter until a process-isolation test proves invariance.
3. Use Group 5—not legacy Group 3—as the authoritative clip 0.15 versus 0.20 comparison.
4. Disable target-KL in the selected recipe (`None`), but retain the optional implementation and telemetry.
5. Describe mechanism claims such as harmful advantages, conservative reward gaming, planner shallow-copy
   contamination, and near-impossible scenarios as hypotheses until directly tested.
"""
    ),
]


namespace = {"__name__": "__notebook__", "_REPO_ROOT": ROOT}
execution_count = 0
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    execution_count += 1
    cell["execution_count"] = execution_count
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}:cell-{execution_count}", "exec"), namespace)
    except Exception as exc:
        cell["outputs"] = [{"output_type": "error", "ename": type(exc).__name__, "evalue": str(exc), "traceback": []}]
        raise
    if stream.getvalue():
        cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": stream.getvalue().splitlines(keepends=True)}]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Wrote and executed {NOTEBOOK}")

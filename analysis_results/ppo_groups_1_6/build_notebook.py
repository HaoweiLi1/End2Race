#!/usr/bin/env python3
"""Build and execute the reproducible companion notebook without nbformat.

The End2Race environment does not currently include nbformat/nbclient.  This
builder emits valid notebook v4 JSON, executes every code cell in one clean
namespace, and records stdout in the cell outputs.  It intentionally consumes
the bounded analysis tables produced by analyze_groups.py rather than scanning
the raw 41,999 trace files again.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_results" / "ppo_groups_1_6"
NOTEBOOK = OUT / "ppo_groups_1_6_analysis.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# PPO baseline to Group 6: reproducible evidence companion

## tl;dr

The selected configuration is the 30-update `privilege_gru`, batch 12,800,
actor GRU/head learning rates `3e-6 / 3e-5`, clip `0.20`, and target-KL off.
At U30 it records **11/600 collisions**, **358 overtakes**, and **98.17% success**,
versus BC's 22 collisions. The comparison is single-run and checkpoint-selected,
so it is a strong result on the fixed Austin600 panel, not a multi-seed estimate.
"""
    ),
    markdown(
        """## Context & Methods

- Primary metric: collision count on the same 600-scenario Austin evaluation panel.
- Secondary metrics: overtake/follow counts, per-scenario collision identity, PPO KL/clip telemetry,
  critic fit, actor optimizer steps, and NPZ trace integrity.
- Formal checkpoints are U1/U5/U10/U15/U20, plus U25/U30 for Group 5.
- `mean_collision_u5_plus` excludes U1 warm-start behavior. Invalid panels are excluded.
- Group 3's legacy 0.10/0.20 arms and Group 6's target-KL 0.02 arm used eight evaluation
  workers while the shared baseline used twelve; those baseline comparisons are not single-axis.
- Correlations pool heterogeneous runs and are descriptive, not causal.
"""
    ),
    code(
        """from pathlib import Path
import json
import pandas as pd

ROOT = Path(globals().get("_REPO_ROOT", Path.cwd()))
if not (ROOT / "analysis_results" / "ppo_groups_1_6").exists():
    raise FileNotFoundError("Run this notebook from the End2Race repository root")
OUT = ROOT / "analysis_results" / "ppo_groups_1_6"

summary = json.loads((OUT / "analysis_summary.json").read_text())
groups = pd.read_csv(OUT / "group_summary.csv")
controls = pd.read_csv(OUT / "group_control_audit.csv")
correlations = pd.read_csv(OUT / "training_eval_correlations.csv")
pairwise = pd.read_csv(OUT / "scenario_pairwise.csv")
traces = pd.read_csv(OUT / "trace_summary_selected.csv")
npz = json.loads((OUT / "npz_audit.json").read_text())

print(f"Generated: {summary['generated_at']}")
print(f"Valid panels: {summary['quality']['valid_panels']} / {summary['quality']['panels']}")
print(f"NPZ files audited: {npz['totals']['files']:,}")
"""
    ),
    markdown(
        """## Data

The extraction script joins run configs and training `metrics.jsonl` records to every evaluation
`results_multi.json`, then audits the corresponding NPZ files. It also checks common scenario
coverage, checkpoint presence, final-checkpoint tensor identity, and training/evaluation ego-index
separation. The tables below are the bounded, reviewable outputs of that extraction.
"""
    ),
    code(
        """quality_rows = [
    ("Evaluation panels", summary['quality']['panels']),
    ("Valid panels", summary['quality']['valid_panels']),
    ("Invalid panels", len(summary['quality']['invalid_panels'])),
    ("NPZ numeric", npz['totals']['numeric_True']),
    ("NPZ time-aligned", npz['totals']['aligned_True']),
    ("Legacy pre-terminal-step NPZ", npz['by_format']['legacy_pre_post_step']['files']),
    ("Post-step-v2 terminal-valid NPZ", npz['by_format']['post_step_v2']['terminal_valid_True']),
]
print(pd.DataFrame(quality_rows, columns=["check", "count"]).to_string(index=False))
print("\\nSingle-axis control audit:")
print(controls[["group", "arm_run", "recorded_differences", "confounds", "strict_single_axis"]].to_string(index=False))
"""
    ),
    markdown(
        """## Results

The collision path is the direct evaluation record at successive formal checkpoints. The table also
shows training telemetry, but the checkpoint decision is made from evaluation outcomes rather than
critic loss or KL alone.
"""
    ),
    code(
        """cols = [
    "group", "label", "env_workers", "collision_path", "mean_collision_u5_plus",
    "best_update", "best_collision_count", "final_update", "final_collision_count",
    "mean_approx_kl", "mean_clip_fraction", "final_explained_variance_post",
    "early_stop_updates", "actor_steps_completed", "actor_steps_planned",
]
print(groups[cols].round(4).to_string(index=False))
"""
    ),
    code(
        """print("Scenario-identity comparisons:")
print(pairwise[["comparison", "a_collisions", "b_collisions", "shared", "resolved", "created", "jaccard"]].round(4).to_string(index=False))

print("\\nPooled training/evaluation correlations (diagnostic only):")
print(correlations[["label", "panels", "pearson_with_eval_collision_count", "spearman_with_eval_collision_count"]].round(4).to_string(index=False))

print("\\nSelected trace-level collision summaries:")
print(traces[["label", "collision_count", "collision_with_opponent_count", "median_collision_time_s", "raceline0_count", "raceline1_count", "raceline2_count"]].round(3).to_string(index=False))
"""
    ),
    markdown(
        """## Takeaways

1. `privilege_gru` is the strongest Group 1 critic: 14 U20 collisions versus 25 for the privileged MLP
   and 34 for the independent GRU.
2. Batch 12,800 is retained; larger batches reduce optimizer steps and lose final/stability performance.
3. Group 3 establishes only the eight-worker 0.10-vs-0.20 comparison. Group 5 is the clean 12-worker
   clip experiment and selects clip 0.20 at U30 (11 versus 20 collisions).
4. The middle actor learning rates `3e-6 / 3e-5` are best. The low rate peaks early then drifts; the high
   rate is worse and its U20 evaluation is invalid.
5. Target-KL is not retained. The clean 0.04 arm stops early in 12/20 updates yet reaches 33 U20
   collisions; gating reduced optimizer work but did not ensure safety.
6. Training metrics are useful mechanism diagnostics but weak selectors: all pooled absolute Pearson
   correlations with evaluation collision count are at most 0.114.
7. The final 11-collision result is promising but remains one seed on one fixed panel; repeat-seed
   confirmation is the next evidentiary step.
"""
    ),
]


namespace: dict = {"__name__": "__notebook__", "_REPO_ROOT": ROOT}
execution_count = 0
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    execution_count += 1
    cell["execution_count"] = execution_count
    source = "".join(cell["source"])
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            exec(compile(source, f"{NOTEBOOK.name}:cell-{execution_count}", "exec"), namespace)
    except Exception as exc:
        cell["outputs"] = [
            {
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "output_type": "error",
                "traceback": [],
            }
        ]
        raise
    output = stream.getvalue()
    if output:
        cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": output.splitlines(keepends=True)}]


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

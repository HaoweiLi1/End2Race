#!/usr/bin/env python3
"""Build and execute the clip-extension decision notebook without optional Jupyter packages."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_results" / "clip_extension_decision"
NOTEBOOK = OUT / "clip_extension_decision.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    markdown(
        """# End2Race PPO next experiment roadmap

## tl;dr

Cap clip exploration at 0.25. Run clean 45-update clip 0.20/0.25 arms to close both the clip boundary
and training-horizon questions. Freeze gamma because it also changes risk-potential reward; defer GAE
until advantage telemetry is available. Test hard neighbors later as a separate fixed-cache ablation.
"""
    ),
    markdown(
        """## Context & Methods

The decision uses the two clean 12-worker, 30-update Group 5 runs. Austin600 results are compared
at U1/U5/U10/U15/U20/U25/U30. Collision identities are paired by scenario; training telemetry uses
formal rows U2-U30 to reduce the shared U1 warm-up transient.

### Key Assumptions

- Austin600 is a fixed selection panel, not a new IID sample at every checkpoint.
- Exact paired p-values are unadjusted for checkpoint selection and do not establish cross-seed generalization.
- Clip fraction measures how often the recorded importance ratio is outside the configured PPO interval;
  it is not a direct forecast of the effect of a larger clip.
"""
    ),
    code(
        """from pathlib import Path
import csv
import json

ROOT = Path(globals().get("_REPO_ROOT", Path.cwd()))
OUT = ROOT / "analysis_results" / "clip_extension_decision"
summary = json.loads((OUT / "decision_summary.json").read_text())
with (OUT / "checkpoint_eval.csv").open() as handle:
    checkpoint_eval = list(csv.DictReader(handle))
with (OUT / "paired_by_checkpoint.csv").open() as handle:
    paired = list(csv.DictReader(handle))
with (OUT / "training_telemetry.csv").open() as handle:
    telemetry = list(csv.DictReader(handle))
with (OUT / "late_training_u20_u30.csv").open() as handle:
    late_training = list(csv.DictReader(handle))
with (OUT / "discount_horizons.csv").open() as handle:
    discount_horizons = list(csv.DictReader(handle))
with (OUT / "hard_neighbor_evidence.csv").open() as handle:
    hard_neighbor = list(csv.DictReader(handle))
with (OUT / "experiment_roadmap.csv").open() as handle:
    roadmap = list(csv.DictReader(handle))
with (OUT / "evaluation_panel_audit.csv").open() as handle:
    evaluation_panel_audit = list(csv.DictReader(handle))
with (OUT / "paired_stat_examples.csv").open() as handle:
    paired_stat_examples = list(csv.DictReader(handle))
print(summary["decision"])
print("clip 0.15:", summary["clip015_collision_path"])
print("clip 0.20:", summary["clip020_collision_path"])
"""
    ),
    markdown("""## Data

### 1. Paired checkpoint results

`resolved_by_clip020` counts 0.15 collision scenarios eliminated by 0.20; `created_by_clip020`
counts new collision scenarios introduced by 0.20.
"""),
    code(
        """for row in paired:
    print(row)
"""
    ),
    markdown("""## Review corrections: use paired identities, and audit the holdout constructor

The Austin panel is fixed and paired, so a universal independent-binomial `+/-4.2` noise floor or
an `>8 collisions` detection rule is not valid. Evidence depends on resolved versus created scenario
identities. The current 50-startpoint constructor also aliases its last endpoint to the first after
modulo reduction, yielding 49 unique starts and 588 unique physical scenarios out of 600 slots.
"""),
    code(
        """print("Paired statistical examples:")
for row in paired_stat_examples:
    print(row)
print("Evaluation panel audit:")
for row in evaluation_panel_audit:
    print(row)
"""
    ),
    markdown("""## Results

### 2. The current best result is real as a recorded checkpoint, but not yet a stable trend

U30 improves from 20 to 11 collisions and resolves 15 scenarios while creating 6. The exact paired
p-value is 0.078 before any correction for selecting U30 after looking at seven checkpoints.
Earlier checkpoints are mostly ties or small differences, and clip 0.20 regresses at U25.
"""),
    code(
        """u30 = next(row for row in paired if int(row["update"]) == 30)
print("U30 paired comparison:", u30)
print("Mean collisions across seven checkpoints:",
      round(summary["clip015_mean_collisions_all_checkpoints"], 3),
      round(summary["clip020_mean_collisions_all_checkpoints"], 3))
"""
    ),
    markdown("""### 3. Clip 0.20 is not obviously over-constrained, but its KL tail is already wide

Only about 5.3% of recorded samples are clipped on average at 0.20 over U2-U30. That leaves a
plausible but small region for 0.25 to change directly. At the same time, 15/29 updates already have
`approx_kl_max > 0.5` and five exceed 1.0, so a larger clip should be treated as a safety-sensitive
probe rather than an automatic improvement.
"""),
    code(
        """for row in telemetry:
    print(row)
"""
    ),
    markdown("""### 4. U30 is not an optimization-convergence point

From U20 through U30, actor checkpoint step norms do not shrink toward zero and approximate KL
continues to spike. This supports extending the horizon as a diagnostic, but the eval path
`13 -> 17 -> 11` does not support a monotonic-improvement claim. A 45-update run must be fresh-started
because current checkpoints omit optimizer, RNG, scheduler, and environment-queue state.
"""),
    code(
        """for row in late_training:
    print(row)
"""
    ),
    markdown("""### 5. Gamma and GAE lambda are not equivalent tuning axes

`gamma=0.999` is used both by PPO returns and by potential-based risk shaping, so changing it also
changes the reward. `gae_lambda` only changes advantage estimation. At 100 Hz, the current
`gamma*lambda=0.994005` gives TD residuals a half-life of about 1.15 seconds. Lowering lambda to 0.99
shortens that to about 0.63 seconds; raising it increases delayed credit but also variance.
"""),
    code(
        """for row in discount_horizons:
    print(row)
"""
    ),
    markdown("""### 6. Hard neighbors are promising, but the 11 probe scenarios are not a production pool

The standalone probe produced 11/18 BC collisions versus a 4.45% global candidate rate, but it
deliberately selected three dense collision families and is not integrated. The full outcome lattice
contains 1,042 collision/other adjacent boundary pairs, supporting a general fixed boundary-aware
cache instead of merging or duplicating the 11 examples.
"""),
    code(
        """print(hard_neighbor[0])
print("Experiment order:")
for row in roadmap:
    print(row)
"""
    ),
    markdown("""## Takeaways

1. Fix or explicitly deduplicate the Austin endpoint alias, then verify shifted-holdout index disjointness.
2. Fresh-start 45-update clip0.20; after exact U1-U30 reproduction, reuse seven existing eval panels and evaluate U35/U40/U45.
3. Either test matched clip0.25 before hard-neighbor, or explicitly freeze clip0.20 and cancel the 0.25 question.
4. Keep gamma at 0.999. Defer lambda until non-RNG-consuming advantage telemetry exists.
5. Implement hard neighbors only after clip/horizon freeze, as a fixed schema-2 boundary-aware cache A/B.
"""),
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

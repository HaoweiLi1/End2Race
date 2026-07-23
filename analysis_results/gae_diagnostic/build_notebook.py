#!/usr/bin/env python3
"""Build and execute the reproducible companion notebook for the GAE diagnosis."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_results" / "gae_diagnostic"
NOTEBOOK = OUT / "gae_diagnostic.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    markdown(
        """# End2Race PPO GAE diagnostic

## tl;dr

No GAE correctness defect is visible. Collision-role transitions carry 5.83x the ordinary-role raw
advantage second-moment proxy in U20-U30, but the proxy does not positively track actor KL or gradient
spikes. Keep lambda at 0.995 until transition-level counterfactual telemetry can test 0.99 on the same rollout.
"""
    ),
    markdown(
        """## Context & Methods

The focal run is `ppo_privilege_gru_0722_long_clip020`: gamma 0.999, lambda 0.995, 30 updates,
12 workers, and a 100 Hz simulator. The notebook loads bounded CSVs produced from the persisted run
config, `metrics.jsonl`, and `episodes.jsonl` by `analyze_gae.py`.

### Key Assumptions

- Before critic training, replayed values closely match values stored during collection, so pre-update
  value MSE is a proxy for the raw advantage second moment, not a direct saved distribution.
- Correlations across 29 updates are descriptive and do not identify causality.
- Completed-episode logs omit unfinished episode fragments at rollout boundaries.
"""
    ),
    code(
        """from pathlib import Path
import csv
import json

ROOT = Path(globals().get("_REPO_ROOT", Path.cwd()))
OUT = ROOT / "analysis_results" / "gae_diagnostic"
summary = json.loads((OUT / "diagnosis_summary.json").read_text())

def load_csv(name):
    with (OUT / name).open() as handle:
        return list(csv.DictReader(handle))

windows = load_csv("window_diagnostics.csv")
lambda_sensitivity = load_csv("lambda_sensitivity.csv")
collision_times = load_csv("collision_time_distributions.csv")
correlations = load_csv("proxy_correlations.csv")
availability = load_csv("data_availability.csv")
print(summary["verdict"])
"""
    ),
    markdown("""## Data

### 1. Late-training advantage-energy proxy and critic fit

The collision/ordinary split is at the transition role, not the completed episode outcome.
"""),
    code(
        """for row in windows:
    print(row)
"""
    ),
    markdown(
        """## Results

### 2. Collision-role raw advantage energy is larger, while critic post-fit remains strong

At U20-U30, collision/ordinary second-moment proxies are 0.1435 and 0.0246. Overall post-update
explained variance is 0.925 and collision-role post-update explained variance is 0.913. This is
consistent with harder collision targets that the critic can fit, not with critic failure.
"""
    ),
    code(
        """late = next(row for row in windows if row["window"] == "U20-U30")
print("U20-U30 role ratio:", late["collision_to_ordinary_second_moment_ratio"])
print("U20-U30 overall/collision post EV:", late["overall_ev_post_mean"], late["collision_ev_post_mean"])
"""
    ),
    markdown(
        """### 3. Lambda 0.99 materially shortens the credit path

With gamma fixed at 0.999, changing lambda from 0.995 to 0.99 changes the two-second weight from
0.300 to 0.110 and the four-second weight from 0.090 to 0.012. This can reduce long-tail variance,
but also removes direct collision credit that must then come from critic bootstrap.
"""
    ),
    code(
        """for row in lambda_sensitivity:
    print(row)
"""
    ),
    markdown(
        """### 4. Current proxies do not explain actor instability spikes

The overall second-moment proxy has Pearson correlation -0.204 with mean KL and 0.071 with mean
actor gradient norm; collision-role values are -0.148 and 0.020. Advantage normalization and the
small update count limit interpretation, but there is no positive association supporting an immediate
lambda reduction.
"""
    ),
    code(
        """for row in correlations:
    if row["source"] in {"overall advantage second-moment proxy", "collision advantage second-moment proxy"}:
        print(row)
"""
    ),
    markdown(
        """### 5. Episode timing confirms broad failures but cannot reconstruct collision lag

The collision-time distribution spans early and late failures. Existing episode rows do not contain
the rollout timestep and environment rank needed to align advantages at 0.5/1/2/4 seconds before impact.
"""
    ),
    code(
        """for row in collision_times:
    print(row)
print("Evidence availability:")
for row in availability:
    print(row)
"""
    ),
    markdown(
        """## Takeaways

1. Keep lambda at 0.995 for the 45-update reproduction run.
2. Before any buffer `get()` call, read raw `[time, env]` advantages without model forward or RNG use.
3. Reconstruct current TD residuals and recompute 0.99/0.995/0.9975 advantages on the same rollout.
4. Compare role tails, sign flips, rank/cosine agreement, and collision-lag slices before running 0.99.
5. Do not test 0.9975 or 1.0 now; no current evidence indicates credit is too short.
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

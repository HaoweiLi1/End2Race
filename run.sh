#!/bin/bash
# Single entry point for the immutable experiment control plane.
#
#   ./run.sh list
#   ./run.sh plan ...
#   ./run.sh show PLAN
#   ./run.sh stage PLAN --all-hosts --dry-run
#   ./run.sh baseline-preflight PLAN --dry-run
#   ./run.sh preflight PLAN --all-hosts --dry-run
#   ./run.sh execute PLAN --all-hosts --dry-run
#   ./run.sh resume PLAN --host remote --dry-run
#
# B2 execution always consumes a previously frozen RunPlan.  There is no
# dynamic `run <job>` path and no command may execute from the remote worktree.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=/home/haowei/miniconda3/envs/end2race/bin/python
if [[ ! -x "$PYTHON" ]]; then
    echo "pinned interpreter is missing: $PYTHON" >&2
    exit 2
fi
exec "$PYTHON" Experiments/runner.py "$@"

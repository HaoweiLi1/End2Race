#!/bin/bash
# Single entry point for every batch / unattended run in this project.
#
#   ./run.sh list                 show every registered job
#   ./run.sh show <job>           print the exact command without running it
#   ./run.sh run  <job>           run it on the remote GPU host
#   ./run.sh run  <job> --local   run it here instead
#
# Jobs are declared in Experiments/runner.py. Add new unattended work there,
# never as a loose shell command — a job must be reviewable before it burns GPU.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 Experiments/runner.py "$@"

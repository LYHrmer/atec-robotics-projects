#!/usr/bin/env bash
set -euo pipefail
TASK_B_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_B_PYTHON="${ATEC_PYTHON:-python3}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$TASK_B_PROJECT_ROOT${ATEC_TASK_ROOT:+:$ATEC_TASK_ROOT:$ATEC_TASK_ROOT/source/atec_rl_lab}${PYTHONPATH:+:$PYTHONPATH}"
exec "$TASK_B_PYTHON" -u "$TASK_B_PROJECT_ROOT/task_b/evaluate.py" "$@"

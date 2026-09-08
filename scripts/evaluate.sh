#!/usr/bin/env bash
set -euo pipefail
TASK_E_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_E_PYTHON="${ATEC_PYTHON:-python3}"
if ! command -v "$TASK_E_PYTHON" >/dev/null; then
    echo "Set ATEC_PYTHON to the Python executable in your Isaac Lab environment." >&2
    exit 1
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="$TASK_E_PROJECT_ROOT${ATEC_TASK_ROOT:+:$ATEC_TASK_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == --check ]]; then
    shift
    exec "$TASK_E_PYTHON" "$TASK_E_PROJECT_ROOT/tools/task_e/check_environment.py" "$@"
fi
TASK_E_HAS_EXPERIENCE=0
TASK_E_CAMERA_OPTIONS=()
for TASK_E_ARGUMENT in "$@"; do
    case "$TASK_E_ARGUMENT" in
        --experience|--experience=*) TASK_E_HAS_EXPERIENCE=1 ;;
        --headless) TASK_E_CAMERA_OPTIONS+=(--headless) ;;
    esac
done
"$TASK_E_PYTHON" "$TASK_E_PROJECT_ROOT/tools/task_e/check_environment.py"
TASK_E_EXTRA_ARGS=()
if [[ "$TASK_E_HAS_EXPERIENCE" == 0 ]]; then
    TASK_E_EXPERIENCE="$("$TASK_E_PYTHON" "$TASK_E_PROJECT_ROOT/tools/task_e/check_environment.py" --experience-path "${TASK_E_CAMERA_OPTIONS[@]}")"
    if [[ -n "$TASK_E_EXPERIENCE" ]]; then
        TASK_E_EXTRA_ARGS=(--experience "$TASK_E_EXPERIENCE")
    fi
fi
exec "$TASK_E_PYTHON" -u "$TASK_E_PROJECT_ROOT/tools/eval_task_e.py" "${TASK_E_EXTRA_ARGS[@]}" "$@"

#!/usr/bin/env bash
set -euo pipefail
ATEC_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ATEC_PROJECT_DIR"
case "${1:-help}" in
  task-a-fast)
    shift
    exec bash "$ATEC_PROJECT_DIR/task_a/scripts/run_taska.sh" --speed .70 --rough_speed .50 "$@"
    ;;
  task-a|a)
    shift
    exec bash "$ATEC_PROJECT_DIR/task_a/scripts/run_taska.sh" "$@"
    ;;
  task-e|e)
    shift
    exec bash "$ATEC_PROJECT_DIR/scripts/evaluate.sh" "$@"
    ;;
  task-b|b)
    shift
    exec bash "$ATEC_PROJECT_DIR/task_b/run.sh" "$@"
    ;;
  help|-h|--help)
    echo 'Usage: bash run.sh task-a [Task A options]'
    echo '       bash run.sh task-a-fast [Task A options]'
    echo '       bash run.sh task-e [Task E options]'
    echo '       bash run.sh task-b --mode hold --seed 42 --max_steps 500 --output runs/task_b_hold'
    echo 'Run setup first: task_a/README.md, task_b/README.md or docs/TASK_E.md'
    ;;
  *)
    echo "Unknown task: $1. Use task-a, task-a-fast, task-b or task-e." >&2
    exit 2
    ;;
esac

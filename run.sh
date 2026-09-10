#!/usr/bin/env bash
set -euo pipefail
ATEC_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ATEC_PROJECT_DIR"
case "${1:-help}" in
  task-a|a)
    shift
    exec bash "$ATEC_PROJECT_DIR/task_a/scripts/run_taska.sh" "$@"
    ;;
  task-e|e)
    shift
    exec bash "$ATEC_PROJECT_DIR/scripts/evaluate.sh" "$@"
    ;;
  help|-h|--help)
    echo 'Usage: bash run.sh task-a [Task A options]'
    echo '       bash run.sh task-e [Task E options]'
    echo 'Run setup first: task_a/README.md or docs/TASK_E.md'
    ;;
  *)
    echo "Unknown task: $1. Use task-a or task-e." >&2
    exit 2
    ;;
esac

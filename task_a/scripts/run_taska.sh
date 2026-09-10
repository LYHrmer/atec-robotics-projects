#!/usr/bin/env bash
set -euo pipefail
ATEC_PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ATEC_EXECUTABLE="${ATEC_PYTHON:-python}"
ATEC_LAB_ROOT="${ISAACLAB_PATH:-/home/lybm/IsaacLab}"
ATEC_ASSET_ROOT="${ATEC_D1G2_ASSET_ROOT:-$ATEC_PACKAGE_ROOT/assets/DDT_Lab}"
ATEC_RESIDUAL_PATH="${ATEC_CHECKPOINT:-$ATEC_PACKAGE_ROOT/weights/model_1999_final.pt}"
ATEC_EXPERIENCE_PATH="${ATEC_EXPERIENCE:-$ATEC_LAB_ROOT/apps/isaacsim_4_5/isaaclab.python.headless.rendering.kit}"
if ! (cd "$ATEC_PACKAGE_ROOT" && "$ATEC_EXECUTABLE" "$ATEC_PACKAGE_ROOT/scripts/setup_assets.py" --check --destination "$ATEC_ASSET_ROOT"); then
  echo "Import the original workspace2.zip assets before starting Isaac Sim:" >&2
  printf '  %q %q --from-zip %q --destination %q\n' "$ATEC_EXECUTABLE" "$ATEC_PACKAGE_ROOT/scripts/setup_assets.py" /path/to/workspace2.zip "$ATEC_ASSET_ROOT" >&2
  echo "Or use --from-directory /path/to/DDT_Lab instead of --from-zip." >&2
  exit 2
fi
if [[ ! -f "$ATEC_EXPERIENCE_PATH" ]]; then
  echo "Set ISAACLAB_PATH or ATEC_EXPERIENCE to the Isaac Sim 4.5 rendering experience." >&2
  exit 2
fi
cd "$ATEC_PACKAGE_ROOT"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPATH="$ATEC_PACKAGE_ROOT/source/atec_rl_lab:$ATEC_PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ATEC_EXECUTABLE" -u tools/run_d1g2_taska.py \
  --headless --experience "$ATEC_EXPERIENCE_PATH" \
  --assets_root "$ATEC_ASSET_ROOT" \
  --policy "$ATEC_ASSET_ROOT/ddt_ros2_control/controller/rl_controller/config/d1/flat_lab.onnx" \
  --residual_checkpoint "$ATEC_RESIDUAL_PATH" \
  --navigation rgbd --axis_heading --visual_recovery --camera_pitch_deg 30 \
  --speed .6 --rough_speed .45 --seed 42 \
  --max_steps 60000 --max_wall_seconds 1800 "$@"

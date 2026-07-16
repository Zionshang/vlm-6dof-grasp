#!/usr/bin/env bash
# One-click launcher for the Fast-FoundationStereo RealSense live demo.
# Shows three live panels: RGB | camera hardware depth | FFS model depth (+ FPS).
#
#   ./run_realsense_demo.sh                         # defaults: scale 0.5, valid_iters 4
#   ./run_realsense_demo.sh --scale 1 --valid_iters 8   # higher accuracy
#   ./run_realsense_demo.sh --baseline 0.05         # force baseline (m) if depth looks scaled
#
# Press q or ESC to quit.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/jyx/miniconda3/envs/ffs/bin/python"
# Triton (used by torch.compile) needs a C compiler on first run:
export CC="/home/jyx/miniconda3/envs/ffs/bin/x86_64-conda-linux-gnu-gcc"

cd "$REPO_DIR"
exec "$PYTHON" scripts/run_realsense_demo.py "$@"

#!/usr/bin/env bash
# ============================================================================
# stage_ffi_deps.sh — one-time staging of the venv build tools (cmake, ninja)
# that config/frontera/build_ffi_host.sh uses.  Frontera builds the CPU leg
# only: its rtx GPUs (sm_75) are below the CUDA leg's sm_80 floor.
#
# RUN INSIDE the python:3.12 apptainer container, e.g.:
#   apptainer exec --bind /home1,/work2,/scratch1,/scratch2 \
#       $LORRAX_SIF bash config/frontera/stage_ffi_deps.sh
#
# Idempotent: safe to re-run.
# ============================================================================
set -euo pipefail

: "${LORRAX_VENV:=$WORK/lorrax_env/.venv}"
UVBIN="${UV_BIN:-$WORK/bin/uv}"

PY="$LORRAX_VENV/bin/python"
[ -x "$PY" ] || { echo "[stage] venv python not found at $PY" >&2; exit 2; }

export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/uv_cache}"
export UV_PYTHON_DOWNLOADS=never
"$UVBIN" pip install --python "$PY" cmake ninja
echo "[stage] done: $("$LORRAX_VENV/bin/cmake" --version | head -1); ninja $("$LORRAX_VENV/bin/ninja" --version)"

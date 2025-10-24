#!/usr/bin/env bash
set -euo pipefail

# Perlmutter UV bootstrap script
# - Installs uv for the current user (if missing)
# - Creates a project venv (default: $SCRATCH/isdf-venv or $PM_SOFTWARE_PREFIX/isdf-venv)
# - Prints activation and sync instructions

: "${PM_SOFTWARE_PREFIX:=${SCRATCH:-$HOME}}"
VENV_DIR="${PM_SOFTWARE_PREFIX%/}/isdf-venv"

if ! command -v uv >/dev/null 2>&1; then
	echo "Installing uv (user-local) ..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.local/bin:$PATH"
fi

PY=3.11

# Create venv if not exists
if [ ! -d "$VENV_DIR" ]; then
	echo "Creating venv at: $VENV_DIR (python $PY)"
	uv venv "$VENV_DIR" --python "$PY"
else
	echo "Reusing existing venv at: $VENV_DIR"
fi

cat <<EOF
Perlmutter UV environment is ready.

Activate and sync:
  source "$VENV_DIR/bin/activate"
  uv sync --frozen

GPU JAX (CUDA 12) install:
  uv pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Recommended runtime env:
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  export HDF5_USE_FILE_LOCKING=FALSE
EOF 
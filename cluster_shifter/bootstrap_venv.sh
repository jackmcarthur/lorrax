#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a persistent virtualenv for running isdf_cohsex inside a Shifter container.
#
# Design goals:
# - keep the container image stock (e.g. nvcr.io/nvidia/jax:25.04-py3)
# - create/reuse a venv on a writable filesystem (typically $PSCRATCH mounted at /workspace/output)
# - install the project editable plus dependencies (normal h5py; NOT MPI/parallel HDF5)
#
# Optional offline mode:
# - set WHEELHOUSE to a directory of pre-downloaded wheels and we install with --no-index.
#
# Usage (inside Shifter allocation):
#   export ISDF_CODE_DIR=/workspace/ISDF
#   export ISDF_VENV_ROOT_DIR=/workspace/venvroot
#   bash /workspace/ISDF/cluster_shifter/bootstrap_venv.sh

: "${ISDF_CODE_DIR:=/workspace/ISDF}"
: "${ISDF_VENV_ROOT_DIR:=/workspace/venvroot}"

VENV_DIR="${ISDF_VENV_DIR:-${ISDF_VENV_ROOT_DIR%/}/isdf_cohsex_py311}"
WHEELHOUSE="${WHEELHOUSE:-}"

echo "[bootstrap] code dir:    ${ISDF_CODE_DIR}"
echo "[bootstrap] venv root:   ${ISDF_VENV_ROOT_DIR}"
echo "[bootstrap] venv dir:    ${VENV_DIR}"
if [[ -n "${WHEELHOUSE}" ]]; then
  echo "[bootstrap] wheelhouse:  ${WHEELHOUSE}"
fi

mkdir -p "${VENV_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[bootstrap] creating venv..."
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

python -m pip install -U pip setuptools wheel

PIP_INSTALL_EXTRA_ARGS=()
if [[ -n "${WHEELHOUSE}" ]]; then
  # Offline wheelhouse mode
  PIP_INSTALL_EXTRA_ARGS+=(--no-index --find-links "${WHEELHOUSE}")
fi

# Install project (editable) and dependencies from pyproject.toml.
#
# Note: This does not install JAX (we rely on the base image), and will install normal
# h5py wheels if available. If pip tries to build h5py from source, you likely want
# wheelhouse mode or a different base image with HDF5 dev headers.
python -m pip install "${PIP_INSTALL_EXTRA_ARGS[@]}" -e "${ISDF_CODE_DIR}"

python -c "import sys; print('[bootstrap] python:', sys.executable); import h5py; print('[bootstrap] h5py:', h5py.__version__)"



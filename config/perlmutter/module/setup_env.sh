#!/usr/bin/env bash
# ============================================================================
# Create the module's runtime venv at $LORRAX_MODULE_PREFIX/venv.
#
#   LORRAX_MODULE_PREFIX=/abs/prefix bash setup_env.sh <clean lorrax checkout>
#
# Third-party packages come from the checkout's uv.lock (no workspace members:
# the module puts a source snapshot on PYTHONPATH instead).  JAX is the
# `cuda13-local` build, which uses the site CUDA toolkit and installs no CUDA
# runtime.  cuDNN and nvidia-mathdx install with --no-deps: their dependency
# closures would add a second CUDA runtime that shadows the toolkit's.
# The venv is never refreshed in place: a pin change is a new prefix.
# ============================================================================
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stack.sh"

SRC="$(realpath -e "${1:?usage: setup_env.sh <clean lorrax checkout>}")"
lorrax_module_clean_rev "$SRC" >/dev/null
[[ ! -e "$VENV" ]] || lorrax_module_die "venv exists: $VENV (a pin change is a new prefix)"
UV="$(command -v uv || true)"
[[ -n "$UV" ]] || lorrax_module_die "uv is not on PATH"

mkdir -p "$PREFIX"
UV_PYTHON_INSTALL_DIR="$PREFIX/python" \
    "$UV" venv --quiet --python "$PYTHON_VERSION" --python-preference only-managed "$VENV"

REQ="$PREFIX/venv/lorrax-locked-requirements.txt"
"$UV" export --quiet --project "$SRC" --frozen --no-emit-workspace --no-hashes \
    --format requirements-txt --output-file "$REQ"
echo "[setup_env] installing $(grep -c '==' "$REQ") locked packages + jax[cuda13-local]==$JAX_VERSION"
"$UV" pip install --quiet --python "$VENV/bin/python" -r "$REQ" \
    "jax[cuda13-local]==$JAX_VERSION"
"$UV" pip install --quiet --python "$VENV/bin/python" --no-deps \
    "nvidia-cudnn-cu13==$CUDNN_VERSION" \
    "nvidia-mathdx==$MATHDX_VERSION"

SITE="$(lorrax_module_site_packages)"
for p in "$SITE/nvidia/mathdx/include/cufftdx.hpp" "$SITE/nvidia/cudnn/lib"; do
    [[ -e "$p" ]] || lorrax_module_die "the wheels did not supply $p"
done
if ls -d "$SITE"/nvidia/cuda_runtime "$SITE"/nvidia/cu13/lib/libcudart.so* >/dev/null 2>&1; then
    lorrax_module_die "a pip CUDA runtime is in the venv; it would shadow $TOOLKIT_MODULE"
fi
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
import jax_plugins.xla_cuda13  # noqa: F401  the local-CUDA plugin
print("[setup_env] " + " ".join(f"{n}={md.version(n)}" for n in
      ("jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt",
       "nvidia-cudnn-cu13", "nvidia-mathdx")))
PY
echo "[setup_env] venv: $VENV"

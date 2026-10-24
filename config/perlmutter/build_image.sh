#!/bin/bash -l
# ============================================================================
# Build a node-local runtime image of one LORRAX revision on Perlmutter and
# install it (scripts/runtime_image.sh).  Run on a LOGIN node:
#
#   bash config/perlmutter/build_image.sh <git-ref> <install-dir>
#
# The whole runtime -- uv's Python, the venv with every CUDA wheel, a shallow
# clone of <git-ref> (editable install), the cuSOLVERMp/cuBLASMp stage and both
# FFI legs with their closure records, precompiled .pyc -- is built AT
# /tmp/lorrax-$USER/<rev12>, the path every compute node restores it to, so no
# recorded path ever differs at run time.  <install-dir> may be any persistent
# filesystem; a run reads one archive from it per node (about 2.4 GB).
# Measured 2026-10-23: build about 2 min, pack 22 s (ri_image.md).
# ============================================================================
set -euo pipefail
REF=${1:?git ref}
INSTALL=${2:?install dir}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO=${LORRAX_IMAGE_REPO:-https://github.com/jackmcarthur/lorrax.git}
REV=$(git ls-remote "$REPO" "$REF" | head -n 1 | cut -c1-12)
[ -n "$REV" ] || REV=$(git -C "$HERE" rev-parse --short=12 "$REF")
TREE=/tmp/lorrax-$USER/rt-$REV
[ -e "$INSTALL/images/rt-$REV.tar.zst" ] && { echo "already installed: $INSTALL/images/rt-$REV.tar.zst"; exit 0; }
export UV_CACHE_DIR=${UV_CACHE_DIR:-$SCRATCH/.cache/uv} UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR=$TREE/python UV_PYTHON_PREFERENCE=only-managed
rm -rf "$TREE"; mkdir -p "$TREE"
git clone -q --depth 1 --branch "$REF" "$REPO" "$TREE/lorrax" 2>/dev/null \
  || { git clone -q "$REPO" "$TREE/lorrax" && git -C "$TREE/lorrax" checkout -q "$REV"; }
cd "$TREE/lorrax"
uv sync --extra cuda13
LORRAX_BUILD_JOBS=${LORRAX_BUILD_JOBS:-16} bash config/perlmutter/build_ffi_host.sh
LORRAX_BUILD_JOBS=${LORRAX_BUILD_JOBS:-16} bash config/perlmutter/build_ffi_cuda.sh
.venv/bin/python -m compileall -q -j 16 .venv/lib "$TREE/python" src services tests > /dev/null || true
bash "$HERE/scripts/runtime_image.sh" install "$TREE" "$INSTALL"

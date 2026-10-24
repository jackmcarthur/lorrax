#!/usr/bin/env bash
# ============================================================================
# install_deps.sh: install everything LORRAX needs on NERSC Perlmutter that the
# system does not already provide, under one prefix, and build the native pair.
#
#   bash install_deps.sh [PREFIX] [--ref REF] [--with-xc] [--no-ffi]
#
#   PREFIX     where everything goes (default $HOME/lorrax). A directory under
#              /global/common/software/<project> imports Python faster on
#              compute nodes than $HOME does.
#   --ref REF  the LORRAX branch, tag or commit to clone (default main). Ignored
#              when the script runs from inside a checkout, which it then uses.
#   --with-xc  also install the patched jax_xc 0.0.11 (the nscf driver only).
#   --no-ffi   skip building the two native libraries.
#
# Runs on a login node in a few minutes (about 90 s with warm uv and ccache
# caches); no allocation is needed.
#
# Taken from Perlmutter, not installed (the build scripts load them; the
# versions are pinned once in ffi_mpi.sh):
#   gcc-native, cray-mpich, cray-libsci, cray-hdf5-parallel, cmake, and ninja
#   and ccache from /usr/bin.
#
# Installed under PREFIX:
#   bin/uv                 uv, the Python package manager (a pinned release binary)
#   uv-python/             CPython 3.12, installed by uv
#   lorrax/                the LORRAX source (git clone)
#   lorrax/.venv/          the environment from uv.lock: JAX with its CUDA 13
#                          plugin, the CUDA libraries and nvcc, NCCL, cuSOLVERMp,
#                          cuBLASMp and nvidia-mathdx, all as pip wheels
#   xc-site/               jax_xc 0.0.11, patched (with --with-xc)
#   lorrax/src/ffi/cpp/build_host/liblorrax_ffi_host.so, build/liblorrax_ffi.so
#   lorrax-env.sh          source this before srun
# uv's download cache goes to $SCRATCH/.cache/uv.
# ============================================================================
set -euo pipefail

UV_VERSION=0.13.5
REPO=https://github.com/jackmcarthur/lorrax.git

PREFIX="" REF=main WITH_XC=0 BUILD_FFI=1
while (($#)); do
    case "$1" in
        --ref) REF="$2"; shift 2 ;;
        --with-xc) WITH_XC=1; shift ;;
        --no-ffi) BUILD_FFI=0; shift ;;
        -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1' "$0"; exit 0 ;;
        -*) echo "install_deps: unknown option $1" >&2; exit 2 ;;
        *) PREFIX="$1"; shift ;;
    esac
done
PREFIX="$(mkdir -p "${PREFIX:-$HOME/lorrax}" && cd "${PREFIX:-$HOME/lorrax}" && pwd)"
say() { printf '\n[install_deps] %s\n' "$*"; }

[[ "${NERSC_HOST:-}" == perlmutter ]] || {
    echo "install_deps: this script is for NERSC Perlmutter (NERSC_HOST=${NERSC_HOST:-unset})." >&2
    echo "  Other machines: see config/local/ and the Installation page of the manual." >&2
    exit 2; }

# --- 1. uv -------------------------------------------------------------------
mkdir -p "$PREFIX/bin"
if [[ "$("$PREFIX/bin/uv" --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]]; then
    say "installing uv $UV_VERSION into $PREFIX/bin"
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar -xz -C "$PREFIX/bin" --strip-components=1
fi
export PATH="$PREFIX/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:?SCRATCH is not set}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="$PREFIX/uv-python"
export UV_PYTHON_PREFERENCE=only-managed  # never a cray-python or conda module
export UV_LINK_MODE=copy                  # the cache and PREFIX are on different file systems

# --- 2. the LORRAX source ---------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$HERE/build_ffi_cuda.sh" && -f "$HERE/../../uv.lock" && -d "$HERE/../../src/ffi/cpp" ]]; then
    ROOT="$(cd "$HERE/../.." && pwd)"
    say "using the checkout at $ROOT"
elif [[ -d "$PREFIX/lorrax/.git" ]]; then
    ROOT="$PREFIX/lorrax"
    say "using the existing clone at $ROOT (not updated; git pull there to update)"
else
    ROOT="$PREFIX/lorrax"
    say "cloning LORRAX ($REF) into $ROOT"
    git clone --filter=blob:none "$REPO" "$ROOT"
    git -C "$ROOT" checkout -q "$REF"
fi

# --- 3. the Python environment ----------------------------------------------
say "installing Python 3.12 and the locked environment (uv sync --frozen --extra cuda13)"
(cd "$ROOT" && uv sync --frozen --extra cuda13)
PY="$ROOT/.venv/bin/python"

# --- 4. jax_xc (optional) ---------------------------------------------------
if ((WITH_XC)); then
    say "installing the patched jax_xc into $PREFIX/xc-site"
    uv pip install --python "$PY" --target "$PREFIX/xc-site" --no-deps -r "$ROOT/config/xc/requirements.txt"
    "$PY" "$ROOT/config/xc/patch_jax_xc.py" "$PREFIX/xc-site"
fi

# --- 5. the native library pair ---------------------------------------------
if ((BUILD_FFI)); then
    say "building the host leg"
    bash "$ROOT/config/perlmutter/build_ffi_host.sh" --fresh
    say "building the CUDA leg"
    bash "$ROOT/config/perlmutter/build_ffi_cuda.sh" --fresh
fi

# --- 6. the environment file ------------------------------------------------
ENVF="$PREFIX/lorrax-env.sh"
{
    echo "# LORRAX on Perlmutter, written by install_deps.sh. Source before srun."
    echo "export LORRAX_ROOT=\"$ROOT\""
    echo "export PATH=\"$PREFIX/bin:$ROOT/.venv/bin:\$PATH\""
    ((WITH_XC)) && echo "export PYTHONPATH=\"$PREFIX/xc-site\${PYTHONPATH:+:\$PYTHONPATH}\""
    echo "source \"$ROOT/config/perlmutter/gpu_env.sh\""
} > "$ENVF"

# --- 7. performance self-check ----------------------------------------------
# On a login node it runs only the static checks and prints the compute-node
# command that certifies this install (exit 3); inside a GPU job step it runs
# in full and fails loudly, naming the cause, when a run would be slow.
say "performance self-check (scripts/perf_check.py)"
rc=0
(source "$ENVF" && python "$ROOT/scripts/perf_check.py" --gpus 4) || rc=$?
case $rc in
    0) say "performance self-check: PASS" ;;
    3) say "performance self-check: not certified here; run the command above on a GPU node" ;;
    *) say "performance self-check: FAIL (exit $rc); see the lines above" ;;
esac

say "done. To check the installation on four GPUs:"
cat <<EOF
  salloc -N 1 -C gpu -G 4 -q interactive -t 0:30:00 -A <account>
  source $ENVF
  cd \$LORRAX_ROOT
  srun -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh python -m pytest tests/hsuite -q -p no:cacheprovider
EOF

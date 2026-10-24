#!/usr/bin/env bash
# ============================================================================
# install_deps.sh: install everything LORRAX needs on NERSC Perlmutter that the
# system does not already provide, under one prefix, and build the native pair.
#
#   curl -fsSLO https://raw.githubusercontent.com/jackmcarthur/lorrax/main/config/perlmutter/install_deps.sh
#   bash install_deps.sh [PREFIX] [--ref REF] [--with-xc] [--no-ffi] [--cubins SET]
#
#   PREFIX     where everything goes (default $SCRATCH/lorrax; about 5 GB, plus
#              4.5 GB of uv download cache on $SCRATCH). Compute nodes read
#              the install fastest from $SCRATCH (the same libraries under
#              /global/common/software made each NVRTC kernel build 2.7x
#              slower, 2026-10-23), and a home quota (40 GB) is easily too
#              small. $SCRATCH is purged after 8 weeks without access; rerun
#              this script after a purge.
#   --ref REF  the LORRAX branch, tag or commit to clone (default main). Ignored
#              when the script runs from inside a checkout, which it then uses.
#   --with-xc  also install the patched jax_xc 0.0.11 (the nscf driver only).
#   --no-ffi   skip building the two native libraries.
#   --no-space-check  skip the free-space and quota check (about 5 GB at
#              PREFIX and 4.5 GB for an empty uv cache).
#   --cubins SET  which NVRTC kernel images to build ahead of the first run:
#              hsuite (the test suite's k-grids), all (every grid of
#              scripts/build_cubin_store.py) or none.  The build needs a GPU
#              of the compute nodes' architecture and is too heavy for a
#              shared login node, so it runs only inside a GPU allocation
#              (default there: hsuite; elsewhere: none).  Skipped, each image
#              compiles on first use instead (about 7 s per image per rank).
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

PREFIX="" REF=main WITH_XC=0 BUILD_FFI=1 CUBINS="" SPACE_CHECK=1
while (($#)); do
    case "$1" in
        --no-space-check) SPACE_CHECK=0; shift ;;
        --ref) REF="$2"; shift 2 ;;
        --with-xc) WITH_XC=1; shift ;;
        --no-ffi) BUILD_FFI=0; shift ;;
        --cubins) CUBINS="$2"; shift 2
            [[ "$CUBINS" =~ ^(hsuite|all|none)$ ]] || { echo "install_deps: --cubins hsuite|all|none, not $CUBINS" >&2; exit 2; } ;;
        -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1' "$0"; exit 0 ;;
        -*) echo "install_deps: unknown option $1" >&2; exit 2 ;;
        *) PREFIX="$1"; shift ;;
    esac
done
PREFIX="${PREFIX:-${SCRATCH:-$HOME}/lorrax}"
PREFIX="$(mkdir -p "$PREFIX" && cd "$PREFIX" && pwd)"
say() { printf '\n[install_deps] [%4ds] %s\n' "$SECONDS" "$*"; }

[[ "${NERSC_HOST:-}" == perlmutter ]] || {
    echo "install_deps: this script is for NERSC Perlmutter (NERSC_HOST=${NERSC_HOST:-unset})." >&2
    echo "  Other machines: see config/local/ and the Installation page of the manual." >&2
    exit 2; }

# --- 0. room ----------------------------------------------------------------
# PREFIX takes about 5 GB and an empty uv cache about 4.5 GB of downloads.  Both
# the file system's free space (df) and the user's quota (NERSC myquota or
# showquota, where present) must hold them; a quota shared by both places must
# hold the sum.  An over-quota file system refuses writes partway through, so
# refuse here instead.
UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:?SCRATCH is not set}/.cache/uv}"
_quota_table=""
if ((SPACE_CHECK)); then
    for _q in myquota showquota; do
        command -v "$_q" >/dev/null 2>&1 && { _quota_table="$(timeout 30 "$_q" 2>/dev/null || true)"; break; }
    done
fi
_quota_label() {  # the quota row a path is charged to: home, pscratch, or nothing
    case "$1/" in
        "$HOME"/*) echo home ;;
        "${SCRATCH:-/nonexistent}"/*|/pscratch/*) echo pscratch ;;
    esac
}
_quota_free_gb() {  # free GiB in quota row $1 ("" when unknown); myquota's columns or showquota's | table
    printf '%s\n' "$_quota_table" | awk -v fs="$1" '
        function gib(s,  v, u) { v = s + 0; u = s; gsub(/[0-9. ]/, "", u)
            return u ~ /^T/ ? v * 1024 : u ~ /^M/ ? v / 1024 : u ~ /^K/ ? v / 1048576 : v }
        { gsub(/\|/, " "); $0 = $0 } $1 == fs { printf "%.1f\n", gib($3) - gib($2); exit }'
}
_df_free_gb() { local d=$1; while [[ ! -d "$d" ]]; do d=$(dirname "$d"); done
                df -Pk "$d" | awk 'NR == 2 { printf "%.1f\n", $4 / 1048576 }'; }
_df_fs() { local d=$1; while [[ ! -d "$d" ]]; do d=$(dirname "$d"); done; df -P "$d" | awk 'NR == 2 { print $6 }'; }
if ((SPACE_CHECK)); then
    _need_prefix=5
    _need_uv=$([[ -n "$(ls -A "$UV_CACHE_DIR" 2>/dev/null | head -1)" ]] && echo 1 || echo 4.5)
    declare -A _need=() _where=()
    for _pair in "$PREFIX:$_need_prefix" "$UV_CACHE_DIR:$_need_uv"; do
        _p=${_pair%:*} _n=${_pair##*:}
        _k="df:$(_df_fs "$_p")"
        _need[$_k]=$(awk -v a="${_need[$_k]:-0}" -v b="$_n" 'BEGIN { print a + b }'); _where[$_k]+=" $_p"
        _l=$(_quota_label "$_p")
        if [[ -n "$_l" && -n "$(_quota_free_gb "$_l")" ]]; then
            _k="quota:$_l"
            _need[$_k]=$(awk -v a="${_need[$_k]:-0}" -v b="$_n" 'BEGIN { print a + b }'); _where[$_k]+=" $_p"
        fi
    done
    for _k in "${!_need[@]}"; do
        case "$_k" in
            df:*) _free=$(_df_free_gb "$(awk '{print $1}' <<< "${_where[$_k]}")") ;;
            quota:*) _free=$(_quota_free_gb "${_k#quota:}") ;;
        esac
        if awk -v f="$_free" -v n="${_need[$_k]}" 'BEGIN { exit !(f < n) }'; then
            _have=$(awk -v f="$_free" 'BEGIN { print (f < 0 ? "none: over quota by " (-f) " GB" : "only " f " GB") }')
            echo "install_deps: REFUSED: needs ${_need[$_k]} GB for${_where[$_k]} but has $_have free (${_k/:/ }); choose another PREFIX (bash install_deps.sh /path/with/room) and/or UV_CACHE_DIR, or pass --no-space-check." >&2
            exit 3
        fi
    done
    unset _need _where _pair _p _n _k _l _free _need_prefix _need_uv
fi

# --- 1. uv -------------------------------------------------------------------
mkdir -p "$PREFIX/bin"
if [[ "$("$PREFIX/bin/uv" --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]]; then
    say "installing uv $UV_VERSION into $PREFIX/bin"
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar -xz -C "$PREFIX/bin" --strip-components=1
fi
export PATH="$PREFIX/bin:$PATH"
export UV_CACHE_DIR
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

# --- 6. NVRTC kernel images -------------------------------------------------
# Into the per-user image cache (kconv_mathdx/images under $SCRATCH/.cache/lorrax),
# where every later install and rebuild finds them: an image's key names its
# toolchain, never the install path.
if [[ -z "$CUBINS" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]] && nvidia-smi -L >/dev/null 2>&1; then CUBINS=hsuite; else
        CUBINS=none
        say "NVRTC kernel images not prebuilt (no GPU allocation here); each compiles on first use.
  To build them ahead, inside a 1-node GPU allocation:
    srun -N 1 -n 1 --gpus-per-node=4 $ROOT/.venv/bin/python $ROOT/scripts/build_cubin_store.py --jobs 16"
    fi
fi
if ((BUILD_FFI)) && [[ "$CUBINS" != none ]]; then
    if nvidia-smi -L >/dev/null 2>&1; then
        case "$CUBINS" in
            hsuite) GRIDS="probe na_fixture h2_fixture" ;;
            all) GRIDS="" ;;
        esac
        JOBS=$(( $(nproc) / 8 )); ((JOBS < 1)) && JOBS=1; ((JOBS > 16)) && JOBS=16
        say "building the NVRTC kernel images ($CUBINS) on this node's GPU with $JOBS workers"
        # shellcheck disable=SC2086  # GRIDS is a word list
        (cd "$ROOT" && source config/perlmutter/gpu_env.sh && \
            "$PY" scripts/build_cubin_store.py --jobs "$JOBS" $GRIDS) \
            || say "kernel image build failed (non-fatal): each image compiles on first use instead"
    else
        say "no GPU on this node: NVRTC kernel images will compile on first use (about 7 s each)"
    fi
fi

# --- 7. the environment file ------------------------------------------------
ENVF="$PREFIX/lorrax-env.sh"
{
    echo "# LORRAX on Perlmutter, written by install_deps.sh. Source before srun."
    echo "export LORRAX_ROOT=\"$ROOT\""
    echo "export PATH=\"$PREFIX/bin:$ROOT/.venv/bin:\$PATH\""
    ((WITH_XC)) && echo "export PYTHONPATH=\"$PREFIX/xc-site\${PYTHONPATH:+:\$PYTHONPATH}\""
    echo "source \"$ROOT/config/perlmutter/gpu_env.sh\""
} > "$ENVF"

say "done. To check the installation on four GPUs:"
cat <<EOF
  salloc -N 1 -C gpu -G 4 -q interactive -t 0:30:00 -A <account>
  source $ENVF
  cd \$LORRAX_ROOT
  srun -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh python -m pytest tests/hsuite -q -p no:cacheprovider
EOF

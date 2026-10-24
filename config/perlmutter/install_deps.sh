#!/usr/bin/env bash
# ============================================================================
# install_deps.sh: install everything LORRAX needs on NERSC Perlmutter that the
# system does not already provide, under one prefix, and build the native pair.
#
#   curl -fsSLO https://raw.githubusercontent.com/jackmcarthur/lorrax/main/config/perlmutter/install_deps.sh
#   bash install_deps.sh [PREFIX] [--ref REF] [--with-xc] [--no-ffi] [--no-space-check]
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
#
# Runs on a login node in 1.5-3 minutes (96 s with a warm uv cache, 168 s
# without; login21, 2026-10-23); no allocation is needed for this part.  It
# ends by installing PREFIX/bin/lorrax-postinstall (config/perlmutter/
# postinstall.sh), the one compute-node step: the NVRTC kernel store (about
# 4 min) and the performance self-check.  Inside a GPU allocation it runs it;
# elsewhere it prints the salloc line that does, and the install stays
# unverified (every GPU run's startup block says so) until it has run.
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
#   bin/lorrax-postinstall the compute-node step (kernel store + self-check)
# uv's download cache goes to $SCRATCH/.cache/uv.
# ============================================================================
set -euo pipefail

UV_VERSION=0.13.5
REPO=https://github.com/jackmcarthur/lorrax.git

PREFIX="" REF=main WITH_XC=0 BUILD_FFI=1 SPACE_CHECK=1
while (($#)); do
    case "$1" in
        --no-space-check) SPACE_CHECK=0; shift ;;
        --ref) REF="$2"; shift 2 ;;
        --with-xc) WITH_XC=1; shift ;;
        --no-ffi) BUILD_FFI=0; shift ;;
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
            _have=$(awk -v f="$_free" 'BEGIN { print (f < 0 ? "nothing free (over quota by " (-f) " GB)" : "only " f " GB free") }')
            echo "install_deps: REFUSED: needs ${_need[$_k]} GB for${_where[$_k]} but has $_have (${_k/:/ }); choose another PREFIX (bash install_deps.sh /path/with/room) and/or UV_CACHE_DIR, or pass --no-space-check." >&2
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

# --- 6. the environment file and lorrax-postinstall --------------------------
ENVF="$PREFIX/lorrax-env.sh"
{
    echo "# LORRAX on Perlmutter, written by install_deps.sh. Source before srun."
    echo "export LORRAX_ROOT=\"$ROOT\""
    echo "export PATH=\"$PREFIX/bin:$ROOT/.venv/bin:\$PATH\""
    ((WITH_XC)) && echo "export PYTHONPATH=\"$PREFIX/xc-site\${PYTHONPATH:+:\$PYTHONPATH}\""
    echo "source \"$ROOT/config/perlmutter/gpu_env.sh\""
} > "$ENVF"
POST="$PREFIX/bin/lorrax-postinstall"
{
    echo "#!/usr/bin/env bash"
    echo "# Written by install_deps.sh: the install's compute-node step (config/perlmutter/postinstall.sh)."
    echo "source \"$ENVF\""
    echo "exec bash \"$ROOT/config/perlmutter/postinstall.sh\" \"\$@\""
} > "$POST"
chmod +x "$POST"

say "installed. Every run: source $ENVF, then e.g.
    srun -N 1 -n 4 --gpus-per-node=4 \$LORRAX_ROOT/src/ffi/cpp/select_gpu.sh python -u -m gw.gw_jax -i deck.in"
if ((BUILD_FFI)); then
    # Inside a GPU allocation this builds the kernel store and verifies the install;
    # elsewhere it prints INSTALL NOT VERIFIED and the salloc line that does both.
    "$POST" || true
fi

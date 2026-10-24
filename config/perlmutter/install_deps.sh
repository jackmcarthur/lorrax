#!/usr/bin/env bash
# ============================================================================
# install_deps.sh: install LORRAX on NERSC Perlmutter, everything the system
# does not provide, as ONE runtime image under one prefix, and build the
# native pair.
#
#   curl -fsSLO https://raw.githubusercontent.com/jackmcarthur/lorrax/main/config/perlmutter/install_deps.sh
#   bash install_deps.sh [PREFIX] [--ref REF] [--with-xc] [--no-space-check]
#                        [--node-dir DIR] [--rebuild] [--in-place] [--no-ffi]
#
#   (raw.githubusercontent.com caches a file for about 5 minutes: right after a
#   push, use the commit's SHA in the URL instead of the branch name.)
#
#   PREFIX     where the install goes (default $SCRATCH/lorrax): any persistent
#              filesystem, fast or slow; it holds one 2.56 GB archive, so it
#              needs about 2.5 GB.  Every job reads that archive once per node
#              and runs from a node-local copy (below), so a slow filesystem
#              (CFS, $HOME, /global/common) costs about 7 s per node instead of
#              a 2.2-2.6x slower hsuite run.
#   --ref REF  the LORRAX branch, tag or commit to install (default main).
#              Ignored when the script runs from inside a checkout: the image is
#              then built from that checkout's HEAD commit (not from uncommitted
#              changes; for those, see --in-place).
#   --with-xc  also install the patched jax_xc 0.0.11 (the nscf driver only).
#   --node-dir DIR  the node-local directory the runtime is built in and every
#              job restores it to (default /tmp/lorrax-$USER; /dev/shm and
#              $TMPDIR are tried when /tmp is missing or too small here).  It
#              must exist on every compute node; about 5 GB.
#   --rebuild  build a new image even when PREFIX already holds one of this
#              revision.
#   --in-place the developer layout instead of an image: the clone and its venv
#              under PREFIX (about 5 GB), run from there (the tree's
#              src/ffi/cpp/select_gpu.sh in the srun line).  Fast only when
#              PREFIX is on a fast filesystem (Lustre $SCRATCH); the self-check
#              says when it is not.
#   --no-ffi   skip building the two native libraries.
#   --no-space-check  skip the free-space and quota check.
#
# Runs on a login node in about 2-3 minutes; no allocation is needed for this
# part.  The runtime -- uv's Python, the venv with every CUDA wheel, the LORRAX
# source, the cuSOLVERMp/cuBLASMp stage, both FFI legs with their dependency
# closure, precompiled .pyc -- is built AT the node-local path
# <node-dir>/rt-<revision>-<time>, packed into PREFIX/images/ as one pax
# tar.zst, and the build tree is removed.  A job restores the image at the same
# path on each node (once per node, about 7 s; scripts/runtime_image.sh), so no
# recorded path differs at run time.
#
# It ends by running PREFIX/bin/lorrax-postinstall (config/perlmutter/
# postinstall.sh), the one compute-node step: the NVRTC kernel store (about
# 4 min) and the performance self-check.  Inside a GPU allocation it runs both;
# elsewhere it prints INSTALL NOT VERIFIED and the salloc line that does, and
# every GPU run's startup block says the install is unverified until then.
#
# Taken from Perlmutter, not installed (the build scripts load them; the
# versions are pinned once in ffi_mpi.sh):
#   gcc-native, cray-mpich, cray-libsci, cray-hdf5-parallel, cmake, and ninja,
#   ccache and zstd from /usr/bin.
#
# Installed under PREFIX:
#   images/rt-<id>.tar.zst(.meta)  the runtime image (current -> the one in use)
#   bin/lorrax-run         the srun command: restore once per node, then run
#                          through select_gpu.sh with the image's venv
#   bin/lorrax-postinstall the compute-node step (kernel store + self-check)
#   bin/uv                 uv, the Python package manager (build time only)
#   share/lorrax/          gpu_env.sh and postinstall.sh, copied from the image
#   lorrax-env.sh          source this before srun
# Outside PREFIX: uv's download cache (4.5 GB when empty) and ccache's in
# $SCRATCH/.cache/{uv,ccache} (in <node-dir>/cache without a writable SCRATCH),
# never in $HOME; the build tree in <node-dir> on this login node.
# ============================================================================
set -euo pipefail

UV_VERSION=0.13.5
REPO=https://github.com/jackmcarthur/lorrax.git

PREFIX="" REF=main WITH_XC=0 BUILD_FFI=1 SPACE_CHECK=1 MODE=image NODE_DIR="" REBUILD=0 HAVE_IMAGE=0
while (($#)); do
    case "$1" in
        --no-space-check) SPACE_CHECK=0; shift ;;
        --ref) REF="$2"; shift 2 ;;
        --with-xc) WITH_XC=1; shift ;;
        --no-ffi) BUILD_FFI=0; shift ;;
        --in-place) MODE=in-place; shift ;;
        --node-dir) NODE_DIR="$2"; shift 2 ;;
        --rebuild) REBUILD=1; shift ;;
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

_df_free_gb() { local d=$1; while [[ ! -d "$d" ]]; do d=$(dirname "$d"); done
                df -Pk "$d" | awk 'NR == 2 { printf "%.1f\n", $4 / 1048576 }'; }
_df_fs() { local d=$1; while [[ ! -d "$d" ]]; do d=$(dirname "$d"); done; df -P "$d" | awk 'NR == 2 { print $6 }'; }

# --- 0. where things go -------------------------------------------------------
# The node-local build directory: the image is built at the path every compute
# node restores it to.  /tmp first, then /dev/shm and $TMPDIR where /tmp is
# missing or lacks the ~7 GB the build needs here.
NEED_NODE=7
if [[ "$MODE" == image && -z "$NODE_DIR" ]]; then
    for _r in /tmp /dev/shm ${TMPDIR:-}; do
        [[ -d "$_r" && -w "$_r" ]] || continue
        if awk -v f="$(_df_free_gb "$_r")" -v n="$NEED_NODE" 'BEGIN { exit !(f >= n) }'; then
            NODE_DIR="$_r/lorrax-${USER:-$(id -un)}"; break
        fi
        echo "install_deps: $_r has only $(_df_free_gb "$_r") GB free, the build needs $NEED_NODE GB" >&2
    done
    [[ -n "$NODE_DIR" ]] || { echo "install_deps: REFUSED: no node-local directory (/tmp, /dev/shm, \$TMPDIR) with $NEED_NODE GB free here; pass --node-dir DIR (a directory every compute node has), or --in-place" >&2; exit 3; }
    [[ "$NODE_DIR" == /tmp/* ]] || echo "install_deps: building at $NODE_DIR: every job restores the image there (or links it from another node-local directory with room)" >&2
fi
NODE_DIR="${NODE_DIR:-/tmp/lorrax-${USER:-$(id -un)}}"
# uv's download cache and ccache: $SCRATCH where it is writable, else the
# node-local build directory; never PREFIX or $HOME (a 40 GB home quota is
# easily too small for the 4.5 GB a first download takes).
if [[ -n "${SCRATCH:-}" && -d "$SCRATCH" && -w "$SCRATCH" ]]; then _bcache="$SCRATCH/.cache"
else _bcache="$NODE_DIR/cache"; fi
UV_CACHE_DIR="${UV_CACHE_DIR:-$_bcache/uv}"
export CCACHE_DIR="${CCACHE_DIR:-$_bcache/ccache}"

# --- 1. room ----------------------------------------------------------------
# The image install: PREFIX about 2.5 GB (one archive), the node-local build
# directory about 7 GB here, an empty uv cache about 4.5 GB of downloads.  The
# in-place install: PREFIX about 5 GB plus the uv cache.  Both the file system's
# free space (df) and the user's quota (NERSC myquota or showquota, where
# present) must hold them; a quota shared by several places must hold the sum.
# An over-quota file system refuses writes partway through, so refuse here.
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
if ((SPACE_CHECK)); then
    _need_uv=$([[ -n "$(ls -A "$UV_CACHE_DIR" 2>/dev/null | head -1)" ]] && echo 1 || echo 4.5)
    if [[ "$MODE" == image ]]; then _pairs=("$PREFIX:2.5" "$NODE_DIR:$NEED_NODE" "$UV_CACHE_DIR:$_need_uv")
    else _pairs=("$PREFIX:5" "$UV_CACHE_DIR:$_need_uv"); fi
    declare -A _need=() _where=()
    for _pair in "${_pairs[@]}"; do
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
            echo "install_deps: REFUSED: needs ${_need[$_k]} GB for${_where[$_k]} but has $_have (${_k/:/ }); the image install needs about 2.5 GB at PREFIX, $NEED_NODE GB in the node-local build directory and 4.5 GB for an empty uv cache: choose another PREFIX (bash install_deps.sh /path/with/room) and/or UV_CACHE_DIR, or pass --no-space-check." >&2
            exit 3
        fi
    done
    unset _need _where _pair _p _n _k _l _free _need_uv _pairs
fi

# --- 2. uv -------------------------------------------------------------------
mkdir -p "$PREFIX/bin"
if [[ "$("$PREFIX/bin/uv" --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]]; then
    say "installing uv $UV_VERSION into $PREFIX/bin"
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar -xz -C "$PREFIX/bin" --strip-components=1
fi
export PATH="$PREFIX/bin:$PATH"
export UV_CACHE_DIR
export UV_PYTHON_PREFERENCE=only-managed  # never a cray-python or conda module
export UV_LINK_MODE=copy                  # the cache and the tree are on different file systems

# --- 3. the LORRAX source -----------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT=""
if [[ -f "$HERE/build_ffi_cuda.sh" && -f "$HERE/../../uv.lock" && -d "$HERE/../../src/ffi/cpp" ]]; then
    CHECKOUT="$(cd "$HERE/../.." && pwd)"
fi
if [[ "$MODE" == in-place ]]; then
    if [[ -n "$CHECKOUT" ]]; then
        ROOT="$CHECKOUT"; say "using the checkout at $ROOT"
    elif [[ -d "$PREFIX/lorrax/.git" ]]; then
        ROOT="$PREFIX/lorrax"; say "using the existing clone at $ROOT (not updated; git pull there to update)"
    else
        ROOT="$PREFIX/lorrax"; say "cloning LORRAX ($REF) into $ROOT"
        git clone --filter=blob:none "$REPO" "$ROOT"
        git -C "$ROOT" checkout -q "$REF"
    fi
    export UV_PYTHON_INSTALL_DIR="$PREFIX/uv-python"
else
    if [[ -n "$CHECKOUT" ]]; then
        SRC_REPO="$CHECKOUT"; REV="$(git -C "$CHECKOUT" rev-parse HEAD)"
        [[ -z "$(git -C "$CHECKOUT" status --porcelain --untracked-files=no 2>/dev/null)" ]] \
            || echo "install_deps: $CHECKOUT has uncommitted changes; the image holds its HEAD $REV only (--in-place runs the tree itself)" >&2
    else
        SRC_REPO="$REPO"
        REV="$(git ls-remote "$REPO" "$REF" "refs/heads/$REF" "refs/tags/$REF" 2>/dev/null | awk 'NR == 1 {print $1}')"
        [[ -n "$REV" ]] || REV="$REF"                     # a commit id
    fi
    _opts="$( ((WITH_XC)) && printf -- -xc)$( ((BUILD_FFI)) || printf -- -noffi)"
    _cur="$(readlink -f "$PREFIX/current" 2>/dev/null || true)"
    HAVE_IMAGE=0
    if ((!REBUILD)) && [[ "$(basename "$_cur")" == "rt-${REV:0:12}$_opts-"* && -f "$_cur.meta" ]]; then
        ID="$(basename "${_cur%.tar.zst}" .tar)"; HAVE_IMAGE=1
        say "PREFIX already holds the image $ID of ${REV:0:12} (--rebuild builds a new one)"
    else
        ID="rt-${REV:0:12}$_opts-$(date +%m%d%H%M%S)"
    fi
    TREE="$NODE_DIR/$ID"
    ROOT="$TREE/lorrax"
    export UV_PYTHON_INSTALL_DIR="$TREE/python"   # the interpreter is part of the image
    export CCACHE_BASEDIR="${CCACHE_BASEDIR:-$TREE}"  # hits across image ids
fi

if [[ "$MODE" == image && $HAVE_IMAGE == 0 ]]; then
    trap 'echo "install_deps: FAILED; the build tree is left at $TREE (rm -rf $TREE when done)" >&2' ERR
    say "building the runtime image $ID at $TREE (source ${REV:0:12} from $SRC_REPO)"
    rm -rf "$TREE"; mkdir -p -m 700 "$NODE_DIR"; mkdir -p "$ROOT"
    git -C "$ROOT" init -q
    if git -C "$ROOT" fetch -q --depth 1 "$SRC_REPO" "$REV" 2>/dev/null; then
        git -C "$ROOT" checkout -q FETCH_HEAD
    else                                                 # a short commit id: a full clone
        rm -rf "$ROOT"; git clone -q --filter=blob:none "$SRC_REPO" "$ROOT"
        git -C "$ROOT" checkout -q "$REV"
    fi
    git -C "$ROOT" remote add origin "$REPO" 2>/dev/null || true
fi

# --- 4. the Python environment, jax_xc, the native pair, bytecode -----------
if [[ "$MODE" == in-place || $HAVE_IMAGE == 0 ]]; then
    say "installing Python 3.12 and the locked environment (uv sync --frozen --extra cuda13)"
    (cd "$ROOT" && uv sync --frozen --extra cuda13)
    PY="$ROOT/.venv/bin/python"
    if ((WITH_XC)); then
        if [[ "$MODE" == image ]]; then XC="$TREE/xc-site"; else XC="$PREFIX/xc-site"; fi
        say "installing the patched jax_xc into $XC"
        uv pip install --python "$PY" --target "$XC" --no-deps -r "$ROOT/config/xc/requirements.txt"
        "$PY" "$ROOT/config/xc/patch_jax_xc.py" "$XC"
        # Image: on sys.path through a .pth file (no PYTHONPATH to carry into jobs).
        [[ "$MODE" == image ]] && echo "$XC" > "$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/lorrax_xc_site.pth"
    fi
    if ((BUILD_FFI)); then
        say "building the host leg"
        bash "$ROOT/config/perlmutter/build_ffi_host.sh" --fresh
        say "building the CUDA leg"
        bash "$ROOT/config/perlmutter/build_ffi_cuda.sh" --fresh
    fi
    if [[ "$MODE" == image ]]; then
        # Bytecode for everything a run imports: the image is a release, and a
        # compute node should never byte-compile.
        "$PY" -m compileall -q -j 16 "$ROOT/.venv/lib" "$TREE/python" "$ROOT/src" "$ROOT/services" "$ROOT/tests" \
            > /dev/null || true
        [[ ! -d "$TREE/xc-site" ]] || "$PY" -m compileall -q -j 16 "$TREE/xc-site" > /dev/null || true
    fi
fi

# --- 5. the image, the environment file and lorrax-postinstall ---------------
mkdir -p "$PREFIX/share/lorrax"
ENVF="$PREFIX/lorrax-env.sh"
if [[ "$MODE" == image ]]; then
    if [[ $HAVE_IMAGE == 0 ]]; then
        say "packing the image into $PREFIX/images ($(du -sh "$TREE" | cut -f1) unpacked)"
        bash "$ROOT/scripts/runtime_image.sh" install "$TREE" "$PREFIX"
        cp "$ROOT/config/perlmutter/gpu_env.sh" "$ROOT/config/perlmutter/postinstall.sh" "$PREFIX/share/lorrax/"
        rm -rf "$TREE"                       # every job restores from the archive
        trap - ERR
    fi
    {
        echo "# LORRAX on Perlmutter, written by install_deps.sh: the runtime image $ID."
        echo "# Source before srun; every srun command starts with lorrax-run."
        echo "export LORRAX_PREFIX=\"$PREFIX\" LORRAX_RUNTIME=image"
        echo "export PATH=\"$PREFIX/bin:\$PATH\""
        echo "source \"$PREFIX/share/lorrax/gpu_env.sh\""
    } > "$ENVF"
    RUNLINE="srun -N 1 -n 4 --gpus-per-node=4 lorrax-run python -u -m gw.gw_jax -i input.in"
else
    cp "$ROOT/config/perlmutter/gpu_env.sh" "$ROOT/config/perlmutter/postinstall.sh" "$PREFIX/share/lorrax/"
    {
        echo "# LORRAX on Perlmutter, written by install_deps.sh --in-place. Source before srun."
        echo "export LORRAX_PREFIX=\"$PREFIX\" LORRAX_RUNTIME=in-place LORRAX_ROOT=\"$ROOT\""
        echo "export PATH=\"$PREFIX/bin:$ROOT/.venv/bin:\$PATH\""
        ((WITH_XC)) && echo "export PYTHONPATH=\"$PREFIX/xc-site\${PYTHONPATH:+:\$PYTHONPATH}\""
        echo "source \"$ROOT/config/perlmutter/gpu_env.sh\""
    } > "$ENVF"
    RUNLINE="srun -N 1 -n 4 --gpus-per-node=4 \$LORRAX_ROOT/src/ffi/cpp/select_gpu.sh python -u -m gw.gw_jax -i input.in"
fi
POST="$PREFIX/bin/lorrax-postinstall"
{
    echo "#!/usr/bin/env bash"
    echo "# Written by install_deps.sh: the install's compute-node step (config/perlmutter/postinstall.sh)."
    echo "source \"$ENVF\""
    echo "exec bash \"$PREFIX/share/lorrax/postinstall.sh\" \"\$@\""
} > "$POST"
chmod +x "$POST"

say "installed ($(du -sh "$PREFIX" 2>/dev/null | cut -f1) in $PREFIX). Every run: source $ENVF, then e.g.
    $RUNLINE"
if ((BUILD_FFI)); then
    # Inside a GPU allocation this builds the kernel store and verifies the install;
    # elsewhere it prints INSTALL NOT VERIFIED and the salloc line that does both.
    "$POST" || true
fi

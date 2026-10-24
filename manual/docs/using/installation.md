# Installation

`uv sync --extra cuda13` Python environment, the native FFI pair, and the machine routes for Perlmutter, Frontera and other sites.
{: .page-summary }

A LORRAX installation has three parts: a Python 3.12 environment with the JAX 0.12 series, a pair of native libraries built from `src/ffi/cpp/`, and a launcher that gives each process one device. All three are required at every process count, one process included.

## Requirements {: #requirements }

### Base requirements {: #req-base }

Always required, on CPU and GPU alike.

| Component | Requirement | What it is for |
|---|---|---|
| Python | ≥ 3.12 | The runtime; uv installs its own 3.12 |
| JAX, jaxlib | `>= 0.12.0, < 0.13.0` | Every array operation. Any other minor series refuses at startup, and there is no override |
| Python packages | `h5py >= 3.14`, `numpy >= 2.3.1`, `scipy >= 1.16`, `matplotlib`, `xmlschema`, `xsdata` | File I/O, the QE schema reader and plots; pinned in `pyproject.toml` and `uv.lock` |
| MPI | the site MPI | Communication between processes; both FFI legs link the same `libmpi` |
| HDF5 | parallel HDF5 ≥ 1.12, built with `HDF5_IS_PARALLEL` | Parallel reads and writes of the large files; both legs link the HDF5 the run maps |
| Host FFI leg | `liblorrax_ffi_host.so`, built from `src/ffi/cpp/` ([below](#ffi)) | Loaded by every run, a GPU run included |
| `jax_xc` | the patched 0.0.11 of `config/xc/` | Only for the [`nscf`](../drivers/nscf/index.md) driver and the dense-Hamiltonian route |

### GPU requirements {: #req-gpu }

For runs on NVIDIA GPUs. A GPU run loads both FFI legs.

| Component | Requirement | What it is for |
|---|---|---|
| GPU | NVIDIA sm_80 or newer | The floor of the CUDA FFI leg |
| CUDA extra | `cuda13` (driver ≥ 580) or `cuda12` | Installs `jax[cudaNN]` and `nvidia-mathdx==25.6.0`, whose kernels do the k-convolutions |
| CUDA FFI leg | `liblorrax_ffi.so` with nvcc, cuBLAS, cuFFT, NVRTC and NCCL | The complete NVIDIA stack; no partial CUDA leg is built |

The GPU linear algebra comes from these libraries:

| Library | Status | What it gives, and when LORRAX uses it |
|---|---|---|
| cuSOLVERMp ≥ 0.7 and cuBLASMp | required, part of the CUDA leg | The default distributed eigensolver and LU on a multi-GPU mesh. cuSOLVERMp 0.6.x returns wrong LU results on a 2-D process grid |
| cuSOLVER, through jaxlib | installed with the CUDA extra | The single-device eigensolver and LU, used when one matrix fits on one GPU |

### CPU requirements {: #req-cpu }

For runs on CPUs. A CPU run loads the host leg only.

| Component | Requirement | What it is for |
|---|---|---|
| FFT | any FFTW3-ABI library: FFTW, `cray-fftw`, MKL's FFTW3 interface or AOCL | The lattice FFTs; bound at run time and never linked into the host leg |

The distributed eigensolver and LU on CPU are ScaLAPACK, together with BLACS and a CBLAS GEMM. One of these stacks provides all three:

| Stack | Typical site |
|---|---|
| Cray LibSci | Cray systems |
| Intel MKL (`mkl_scalapack_lp64` with the `mkl_blacs_<mpi>_lp64` that matches the MPI) | Intel systems; a BLACS built for another MPI hangs in the first collective |
| netlib ScaLAPACK with OpenBLAS or BLIS | generic Linux clusters |
| AMD AOCL | AMD CPU systems |

The Cholesky factorization is LORRAX's own, on CPU and GPU alike.

The service packages under `services/` (`distrib_la`, `lxkit`, `minimax`, `symmetry_maps`, `vcoul`, `wfn_loader`, `zeta_loader`) are uv workspace members and install with the project.

## The native FFI pair {: #ffi }

| Leg | File | Contains |
|---|---|---|
| host | `liblorrax_ffi_host.so` | ScaLAPACK, a CBLAS GEMM, the FFTW3-ABI FFT, parallel HDF5. CUDA-free. |
| CUDA | `liblorrax_ffi.so` | cuSOLVERMp, cuBLASMp, cuBLAS, cuSOLVER, cuFFT, NCCL, the NVRTC-built mathdx k-convolution kernels, parallel HDF5 |

A CPU run loads the host leg; a GPU run loads both into one process. There is no pure-JAX fallback for the k-convolution, the distributed linear algebra or the slab I/O, so a missing leg refuses at startup. Each build ends in `scripts/verify_ffi_build.sh` and prints `VERIFY PASSED`; each leg records its dependency closure in `<leg>.closure.json`, and every startup refuses a process that maps any other copy of a recorded library. A private build loads with a `LEGACY-UNSEALED` line and its hash; only a sealed pair is a production provider.

## Routes by machine {: #routes }

| Machine | Route | Entry |
|---|---|---|
| NERSC Perlmutter (A100, CUDA 13.2) | a published module, or a clone that builds its own pair | `config/perlmutter/` |
| TACC Frontera (Cascade Lake CPU) | a clone with the host leg, run inside apptainer | `config/frontera/` |
| one Ubuntu machine with an NVIDIA driver (workstation, WSL2, a rented GPU box) | everything from pip wheels plus the distro OpenMPI | `config/local/` |
| any other cluster | a clone with both legs built from the Perlmutter scripts, re-pinned to the site ([porting](#porting)) | `src/ffi/cpp/` |

The Perlmutter route is the reference build. The sections below say what it takes from the site, what it installs itself and which settings are not obvious, so that the same choices can be made on another cluster.

### Perlmutter: what the site provides {: #perlmutter-site }

LORRAX links against these Perlmutter modules and system files. Everything in one process must use the same MPI, so the versions are pinned as a set in `config/perlmutter/ffi_mpi.sh`.

| Component | Pin | Role, and why this pin |
|---|---|---|
| Compiler, host leg | `PrgEnv-gnu`, the Cray `CC` wrapper | builds `liblorrax_ffi_host.so` |
| Compiler, CUDA leg | `gcc-native/14`, plain `gcc`/`g++` (not the Cray wrapper) | host compiler for nvcc; the wrapper would inject Cray libraries the CUDA leg must not link |
| CUDA toolkit | `cudatoolkit/13.2` | nvcc, cuFFT, cuSOLVER and NVRTC at **build** time only; at run time the CUDA libraries come from the pip wheels |
| MPI | `cray-mpich/9.0.1` (`libmpi_gnu_123.so.12`) | the one `libmpi` both legs link |
| ScaLAPACK, BLAS | `cray-libsci/25.09.0`, threaded (`_mp`) flavour | the CPU distributed eigensolver and LU, and the host GEMM. The site default (26.03.0) is built against MPICH 9.1.0, which would put a second MPI in the process |
| Parallel HDF5 | `cray-hdf5-parallel/1.14.3.7` | the parallel reads and writes of both legs |
| FFTW | `cray-fftw` | found by `dlopen` at run time, never linked |
| NCCL network plugin | `/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib/libnccl-net.so` | NCCL over the Slingshot network (libfabric) for multi-node runs |

Not used on Perlmutter: the site Python or conda (uv installs its own CPython 3.12), CUDA-aware MPI and its GTL library (device collectives go through NCCL), the site cuSOLVERMp module, mpi4py and mpi4jax, and ELPA (a ROCm-leg backend).

### Perlmutter: what LORRAX installs {: #perlmutter-installs }

| Component | Version | How |
|---|---|---|
| JAX, jaxlib, the CUDA 13 plugin and PJRT | 0.12.0.1 | `uv sync --extra cuda13`, from `uv.lock` |
| CUDA runtime libraries | `nvidia-cublas` 13.8, `nvidia-cufft` 12.4, `nvidia-cusolver` 12.3, `nvidia-cuda-runtime` 13.4, `nvidia-nccl-cu13` 2.32.3 | the same `uv sync`; these, not the toolkit's copies, are what runs |
| cuFFTDx headers | `nvidia-mathdx==25.6.0` | the same `uv sync`; the k-convolution kernels are compiled from them at run time |
| cuSOLVERMp, cuBLASMp | `nvidia-cusolvermp-cu13==0.9.1.9318.post1`, `nvidia-cublasmp-cu13==0.10.0.3695` | `build_ffi_cuda.sh` installs the wheels with `uv pip install --no-deps --target <clone>/.build/cusolvermp-<ver>/` and links against them |
| h5py | 3.16, the PyPI wheel with its own serial HDF5 | small serial files; every parallel read and write goes through the FFI's parallel-HDF5 handlers, so h5py is **not** built against the site HDF5 |
| native FFI pair | `src/ffi/cpp/build_host/liblorrax_ffi_host.so`, `src/ffi/cpp/build/liblorrax_ffi.so` | `build_ffi_host.sh`, `build_ffi_cuda.sh`; each writes a `.closure.json` and a provenance stamp beside the library |
| MPIwrapper | upstream eschnett/MPIwrapper, ABI 2.10.0 | `build_mpiwrapper.sh`; only for multi-process **CPU** runs, which use JAX's MPI collectives through MPItrampoline |

### Perlmutter: one script {: #perlmutter-script }

`config/perlmutter/install_deps.sh` installs everything in the table above that Perlmutter does not provide and builds the two native libraries, as **one runtime image** under a directory of your choice. It runs on a login node in about three minutes and needs no allocation. The whole runtime (uv's Python, the environment with every CUDA wheel, the LORRAX source, both native libraries with their dependency records, precompiled bytecode) is built at the node-local path `/tmp/lorrax-$USER/rt-<revision>-<time>`, packed into one 2.4 GB archive in the prefix, and the build directory is removed. Every job restores the archive to that same path on each node before its first rank starts (about 7 s per node) and runs from there, so the prefix can be on any filesystem: CFS, `$HOME` and `/global/common` are 10-30 ms per file on compute nodes, which made an install kept there run up to 3.5x slower.

```bash
curl -fsSLO https://raw.githubusercontent.com/jackmcarthur/lorrax/main/config/perlmutter/install_deps.sh
bash install_deps.sh $SCRATCH/lorrax
salloc -N 1 -C gpu -G 4 -q interactive -t 0:30:00 -A <project>_g $SCRATCH/lorrax/bin/lorrax-postinstall
```

and then, in every shell that launches a run:

```bash
source $SCRATCH/lorrax/lorrax-env.sh
srun -N 1 -n 4 --gpus-per-node=4 lorrax-run python -u -m gw.gw_jax -i gw.in
```

!!! note "Before the next merge to main"
    The script is on the branch `claude/robust-install`. Until it reaches `main`, download from `https://raw.githubusercontent.com/jackmcarthur/lorrax/claude/robust-install/config/perlmutter/install_deps.sh` (GitHub serves a cached copy for about 5 minutes after a push; a commit's SHA in place of the branch name avoids that) and add `--ref claude/robust-install`.

| Option | Effect |
|---|---|
| `PREFIX` | where the install goes; default `$SCRATCH/lorrax`. It needs about 2.5 GB, on any filesystem |
| `--ref REF` | the branch, tag or commit to install; default `main`. Ignored when the script runs from inside a checkout: the image is then built from that checkout's last commit |
| `--with-xc` | also install the patched `jax_xc` 0.0.11, needed only by the [`nscf`](../drivers/nscf/index.md) driver |
| `--node-dir DIR` | the node-local directory the image is built in and restored to; default `/tmp/lorrax-$USER` |
| `--rebuild` | build a new image even when the prefix holds one of this revision |
| `--in-place` | the developer layout instead of an image: clone and environment under the prefix (about 5 GB), run through the tree's `select_gpu.sh`; fast only on `$SCRATCH` |
| `--no-ffi` | skip building the native libraries |
| `--no-space-check` | skip the up-front free-space and quota check |

What it does, in order:

1. Refuses up front, in one line, when the prefix (2.5 GB), the node-local build directory (7 GB) or uv's download cache (4.5 GB when empty) lacks room on disk or in your quota.
2. Downloads a pinned uv release binary into `PREFIX/bin`. uv's download cache and the compiler cache go to `$SCRATCH/.cache`, never to the prefix or `$HOME`.
3. Fetches the requested commit into the build directory and runs `uv sync --frozen --extra cuda13`, which installs CPython 3.12 (never a site Python module) inside the image and every Python dependency at the versions in `uv.lock`, including the CUDA libraries, nvcc, NCCL, cuSOLVERMp, cuBLASMp and nvidia-mathdx, all as wheels.
4. With `--with-xc`, installs and patches `jax_xc` inside the image.
5. Builds the host and CUDA libraries with `build_ffi_host.sh` and `build_ffi_cuda.sh`, which load the site modules themselves; each build ends with `VERIFY PASSED`.
6. Compiles the bytecode, packs the image into `PREFIX/images/` (removing the previous one), and writes `PREFIX/bin/lorrax-run`, `PREFIX/bin/lorrax-postinstall` and `PREFIX/lorrax-env.sh`, which puts `PREFIX/bin` on `PATH` and applies `gpu_env.sh`.
7. Runs `lorrax-postinstall`. Outside a GPU allocation it prints `INSTALL NOT VERIFIED` and the `salloc` line above, with your `_g` account filled in. Inside one it restores the image on the compute node, builds the NVRTC kernel store (about 4 minutes) and runs the [performance self-check](https://github.com/jackmcarthur/lorrax/blob/main/docs/installation/perf_check.md); a pass prints `INSTALL VERIFIED`. Until then every GPU run's startup block warns that the install is unverified.

`lorrax-run` restores the image once per node: the first rank fetches and unpacks it, the other ranks and any later job on the node find it ready. It needs about 9 GB free in `/tmp` during the restore and 4.9 GB after; where `/tmp` lacks room it uses `/dev/shm` or `$TMPDIR`, and a node with no node-local room at all runs from one extracted copy in the prefix and prints one line saying so. Older images no running job holds are removed. Each node of a multi-node job restores the image itself.

The script, as of this page's revision:

```bash title="config/perlmutter/install_deps.sh"
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
#              filesystem, fast or slow; it holds one 2.4 GB archive, so it
#              needs about 2.5 GB.  Every job reads that archive once per node
#              and runs from a node-local copy (below), so a slow filesystem
#              (CFS, $HOME, /global/common) costs about 7 s per node instead of
#              a 2-3.5x slower run.
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
```


### Perlmutter: a clone, by hand {: #perlmutter }

1. Keep uv's cache and interpreters on scratch, then clone and install the Python environment:

    ```bash
    export UV_CACHE_DIR=$SCRATCH/.cache/uv UV_PYTHON_INSTALL_DIR=$SCRATCH/.local/uv-python
    git clone --filter=blob:none https://github.com/jackmcarthur/lorrax.git $SCRATCH/lorrax
    cd $SCRATCH/lorrax
    uv sync --extra cuda13
    ```

2. Take one allocation, for example `salloc -N 1 -C gpu -G 4 -q interactive -t 1:00:00 -A <account>`, and build both legs on its node with no GPU requested. Each build ends with `VERIFY PASSED`.

    ```bash
    srun --jobid=$JOBID -N 1 -n 1 -c 64 --gres=none bash -c '
      bash config/perlmutter/build_ffi_host.sh &&
      bash config/perlmutter/build_ffi_cuda.sh'
    ```

3. Run with `config/perlmutter/gpu_env.sh` sourced in the shell that calls `srun`, one rank per GPU through `select_gpu.sh`:

    ```bash
    source config/perlmutter/gpu_env.sh
    srun -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh .venv/bin/python -u -m gw.gw_jax -i gw.in
    ```

Rebuild both legs after a pull that changes `src/ffi/cpp/`, after moving `.build`, or after a `uv sync` that changes the CUDA wheels; a stale leg refuses.

### Perlmutter: settings that are not obvious {: #perlmutter-settings }

The build scripts and `gpu_env.sh` set the following. Each one fixes a failure that a straightforward build or launch runs into, and each has a counterpart on any Cray or Slurm system.

**Building the host leg** (`build_ffi_host.sh`).

| Setting | Why |
|---|---|
| unload `craype-accel-nvidia80` and `cudatoolkit` first | otherwise the compiler wrapper links `libmpi_gtl_cuda`, and through it `libcuda`, into the CUDA-free leg |
| load `cray-libsci`, record its library path, unload it | otherwise the wrapper also adds the sequential `-lsci_gnu`, and two LibSci copies end up in one library; the recorded path is passed as `LORRAX_CBLAS_DIR` |
| unload `cray-fftw` | otherwise CMake's OpenMP probe adopts `fftw3_omp` as the OpenMP library |
| `-DLORRAX_SCALAPACK_LIBRARIES="-L<libsci> -Wl,--no-as-needed -lsci_gnu_mpi_mp -lsci_gnu_mp -lgomp …"` | the ScaLAPACK probe looks for MKL; on LibSci the link line is given explicitly |
| `-DLORRAX_MPI_INCLUDE_DIR`, `-DLORRAX_MPICH_LIB_DIR`, `-DLORRAX_MPI_LIBRARY`, `-DHDF5_ROOT` | the CMake searches use `NO_DEFAULT_PATH`, so the wrapper's implicit include and library paths are invisible to them |

**Building the CUDA leg** (`build_ffi_cuda.sh`).

| Setting | Why |
|---|---|
| `gcc-native/14` with plain `gcc`/`g++` | see the table above: the Cray wrapper would add site libraries |
| cuSOLVERMp and cuBLASMp from wheels, `-DCUSOLVERMP_INCLUDE_DIR`, `-DCUSOLVERMP_LIB_DIR` | `cudatoolkit/13.2` ships cuSOLVERMp 0.8 under the same SONAME, and cuSOLVERMp 0.6.x returns wrong LU results on a 2-D grid |
| `-DLORRAX_FFI_HAVE_CAL=OFF` | cuSOLVERMp ≥ 0.7 talks to NCCL directly, without the CAL layer |
| `-DNCCL_INCLUDE`, `-DNCCL_LIBRARY` pointing into the venv's `nvidia-nccl-cu13` | the leg must link the NCCL that JAX loads |
| `-DHDF5_PREFER_PARALLEL=ON`, `-DNVHPC_ROOT` | pick the parallel HDF5 and the toolkit root explicitly |
| CUDA architectures `80;86-real;89-real;90-real;100-real;120` (the default) | sm_80 is the floor; `LORRAX_FFI_HAVE_CUBLASMP` and `LORRAX_FFI_HAVE_CUFFT` are always on, and passing `OFF` is a configure error |

**Which libraries a run loads.** Each leg carries `$ORIGIN` as its only `RUNPATH` and exports a fixed symbol list. Because `LD_LIBRARY_PATH` outranks a `RUNPATH`, a module that puts another copy of a library first would win; `cudatoolkit/13.2`, for example, puts its own cuSOLVERMp 0.8 there, which fails with `syevd status=7`. The build therefore records the leg's complete dependency closure in `<leg>.closure.json`, and every startup refuses a process that maps any other copy (`GATE native-closure`). The pinned MPICH, HDF5 and LibSci directories (`LORRAX_PM_RUNTIME_LIB_DIRS`) must precede `/opt/cray/pe/lib64`, whose links point at the site defaults.

**Launching** (`gpu_env.sh` and the runtime).

| Setting | Why |
|---|---|
| `MPICH_GPU_SUPPORT_ENABLED=0` | `craype-accel-nvidia80` sets it to 1, and `MPI_Init` then aborts with `GTL library is not linked`; LORRAX does not use CUDA-aware MPI |
| `libcudart.so` linked to the wheel's runtime and placed first on `LD_LIBRARY_PATH` | libfabric opens the unversioned name; without the link a second CUDA runtime loads and a multi-node step refuses |
| `SLURM_NETWORK=no_vni` (the same as `srun --network=no_vni`) | with a VNI, MPI-IO and NCCL's network initialization fail with `ENOSPC` |
| `NCCL_NET="AWS Libfabric"`, `NCCL_NET_GDR_LEVEL=PHB`, `NCCL_SOCKET_IFNAME=hsn`, `FI_CXI_DISABLE_HOST_REGISTER=1`, `FI_CXI_RDZV_THRESHOLD=0`, `FI_MR_CACHE_MONITOR=kdreg2` | set by the runtime from `config/perlmutter/site.toml` unless already set. The rendezvous threshold prevents a cross-node deadlock, and `kdreg2` replaces the default memory-registration monitor, which deadlocked inside a cuSOLVERMp split |
| `select_gpu.sh`, never `--gpus-per-task=1` | `select_gpu.sh` gives each rank one entry of Slurm's GPU list in `CUDA_VISIBLE_DEVICES`; `--gpus-per-task=1` confines each rank so that the first collective fails with `invalid device ordinal` |
| `CUDA_CACHE_PATH=$SCRATCH/.nv/ComputeCache`; JAX compile cache in `$SCRATCH/.cache/lorrax/jax_compile` | keep the kernel caches off the small home quota |

The runtime also chooses the JAX coordinator (the step's first host, a port derived from the job and step ids; `JAX_COORDINATOR_ADDRESS` overrides it) and merges two `--xla_gpu_*` compilation flags into `XLA_FLAGS`; a value you set wins. Do not set XLA's allocator or memory-fraction variables.

A multi-process CPU run additionally sources `config/perlmutter/cpu_mpi_env.sh`, which sets `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`, `MPITRAMPOLINE_LIB` to the MPIwrapper library, `LD_PRELOAD=/opt/cray/pe/lib64/libpmi.so.0`, `JAX_NUM_CPU_DEVICES=1` and `MPICH_ASYNC_PROGRESS=1`.

### Perlmutter: the module {: #perlmutter-module }

```bash
module use <prefix>/modulefiles
module load lorrax
srun -N 1 -n 4 --gpus-per-node=4 $LORRAX_ROOT/src/ffi/cpp/select_gpu.sh python -u -m gw.gw_jax -i gw.in
```

The module carries one venv, one source snapshot and one sealed FFI bundle, and sets `LORRAX_ROOT`, `LORRAX_FFI_SO`, `LORRAX_FFI_HOST_SO`, `MPICH_GPU_SUPPORT_ENABLED=0`, `SLURM_NETWORK=no_vni`, `JAX_PLATFORMS=cuda,cpu` and `JAX_ENABLE_X64=1`, so `gpu_env.sh` is not needed. Do not add a checkout to `PYTHONPATH` or pin one leg by hand: a mixed source tree or a partial bundle refuses. Maintainers build a module with the scripts in `config/perlmutter/module/`: `setup_env.sh` (the venv, from the same lock), `build.sh` (a fresh clone, both legs, the sealed bundle, the snapshot and the modulefile), `accept.sh` (acceptance runs on 1 and 4 GPUs) and `publish.sh`.

### Porting to another cluster {: #porting }

Copy `config/perlmutter/` and replace the site-specific pieces:

| Perlmutter piece | Replace with |
|---|---|
| `ffi_mpi.sh`: MPICH 9.0.1, LibSci 25.09.0, HDF5 1.14.3.7 | the site's MPI, ScaLAPACK and parallel HDF5 (≥ 1.12), chosen so that all three are built against the **same** MPI. `config/frontier/ffi_mpi.sh` is a second example |
| LibSci | MKL (`mkl_scalapack_lp64` with the `mkl_blacs_<mpi>_lp64` that matches the MPI), netlib ScaLAPACK with OpenBLAS or BLIS, or AOCL; give the link line in `LORRAX_SCALAPACK_LIBRARIES` |
| `cray-hdf5-parallel` | a parallel HDF5 ≥ 1.12; `config/local/` builds 1.14.6 from source when the system has none |
| `cudatoolkit/13.2` | any CUDA 13 toolkit, or the `nvidia/cu13` directory of the `jax[cuda13]` wheels, which includes nvcc (`config/local/` does this) |
| the compiler-wrapper module dance | on a non-Cray system, plain compilers and explicit `-D` paths; on a Cray system, the same unloads |
| `site.toml` (NCCL plugin and libfabric settings) | the site's NCCL network plugin and its recommended variables, or nothing on a single node |
| `gpu_env.sh` (`no_vni`, GTL off, the `libcudart` link) | the site's equivalents; on a single machine, `config/local/launch.sh` with `mpirun` |
| `select_gpu.sh` | keep it; it reads Slurm's variables, or `LORRAX_GPU_DEVICE` elsewhere |
| cuSOLVERMp 0.9.1 and cuBLASMp 0.10.0 wheels | the same wheels, from pypi.org |

Build each leg with a site script that ends the way the Perlmutter and `config/local/` scripts do: CMake with explicit `-D` paths, then `src/ffi/cpp/stage/write_checkout_closure.py` on the library, which writes the `.closure.json` every startup requires (a leg without one refuses with `no closure record`), then `scripts/verify_ffi_build.sh`. On a cluster without Cray compiler wrappers, `config/local/build_ffi.sh` and `build_ffi_host.sh` are the closer template: plain `gcc`/`g++`, nvcc from the wheels, and the system MPI and ScaLAPACK. Check the result with the test suite ([below](#check)).

### Frontera {: #frontera }

Frontera runs the host leg only: its RTX 5000 GPUs (sm_75) are below the CUDA leg's floor. JAX runs inside the apptainer image `py312.sif`, because the host glibc is 2.17. Multi-process CPU collectives run on MPI through MPItrampoline and a patched MPIwrapper.

| Artifact | Script | Runs on |
|---|---|---|
| host leg | `LORRAX_HOST_SCALAPACK=1 config/frontera/build_ffi_host.sh` | login node |
| MPIwrapper | `config/frontera/build_mpiwrapper.sh --fresh` | login node |
| mpi4py + parallel h5py overlay | `config/frontera/build_mpi_overlay.sh fetch`, then `build` | login node, then compute node in the image |
| staged PMI2 library | `config/frontera/stage_host_pmi.sh` | login node |
| node-local runtime bundle | `config/frontera/build_cpu_runtime_bundle.sh` | inside the image |
| launch | `config/frontera/templates/gw_dev.sbatch` | batch job |

The image and the venv `$WORK/lorrax_env/.venv` are not rebuilt by any script in the repository. The launch template refuses when `LORRAX_ROOT` or `LORRAX_RUN_DIR` is unset; edit its `#SBATCH` block (account, nodes, time) before submitting.

### Other sites {: #other }

```bash
uv sync --extra cuda13          # or --extra cuda12; plain `uv sync` for a CPU-only install
```

then build the two legs with a site script modelled on [Porting](#porting). The CMake options a CUDA leg needs are `LORRAX_FFI_PLATFORM=cuda`, `CMAKE_CUDA_COMPILER`, `CUDA_TOOLKIT_ROOT`, `CUSOLVERMP_INCLUDE_DIR` and `CUSOLVERMP_LIB_DIR`, `LORRAX_FFI_HAVE_CAL=OFF`, `NCCL_INCLUDE` and `NCCL_LIBRARY`, `HDF5_ROOT` (a parallel HDF5), and `LORRAX_MPI_INCLUDE_DIR`, `LORRAX_MPICH_LIB_DIR` and `LORRAX_MPI_LIBRARY`; the host leg takes `LORRAX_FFI_PLATFORM=host`, the MPI and HDF5 options and `LORRAX_SCALAPACK_LIBRARIES`. A library outside the in-tree build directories is selected with [`LORRAX_FFI_SO` and `LORRAX_FFI_HOST_SO`](environment.md#ffi). For one machine with only an NVIDIA driver ≥ 580, `config/local/` builds both legs from wheels, and its README walks through it.

## Checking an installation {: #check }

The test suite runs every driver on the bundled magnetic H₂⁻ fixture and compares each stage with `tests/hsuite/reference/`:

```bash
srun -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh \
  .venv/bin/python -m pytest tests/hsuite -q -p no:cacheprovider
```

It ends `1 passed`. `python -m tests.hsuite.chain --out <dir>` runs the same workflow outside pytest and leaves every input file, report and output in `<dir>/run/`.

## Sizing a job {: #sizing }

Process-count rules, the per-GPU memory budget and how many GPUs a calculation needs are on [Running on clusters](clusters.md).

## Startup refusals {: #refusals }

!!! refuses
    - `Could not locate liblorrax_ffi_host.so` or `… liblorrax_ffi.so`: no library for this platform. Build the leg the message names, or load a module.
    - `HANDLER ABI MISMATCH`: the library and the Python tree disagree on the handler ABI. Rebuild both legs from the same tree.
    - `partial sealed-bundle override refused`: only one leg of a sealed pair is pinned. Unset both pins, or pin both legs of one bundle.
    - `mixed native providers`: the two legs come from different bundles. Select both from one bundle.
    - `GATE native-closure: got …; want …; fix: …`: the process maps a library other than the recorded one. Remove what maps it (an `LD_PRELOAD`, a package built against another MPI), or rebuild the leg.
    - `JAX_ENV_REFUSED`, or the `runtime.jax_support` refusal before the first `jit`: JAX or jaxlib outside the 0.12 series. Install the 0.12 series.
    - `GATE mathdx-headers`: a CUDA run without `nvidia-mathdx`. Run `uv sync --extra cuda13` (or `cuda12`).


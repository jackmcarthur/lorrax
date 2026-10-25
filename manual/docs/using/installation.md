# Installation

Python environment, the native FFI pair, and the machine routes for Perlmutter, Frontier, Frontera and other sites.
{: .page-summary }

A LORRAX installation has three parts: a Python 3.12 environment with the JAX 0.12 series, a pair of native libraries built from `src/ffi/cpp/`, and a launcher that gives each process one device. All three are required at every process count, one process included.

## Requirements {: #requirements }

### Base requirements {: #req-base }

Always required, on CPU and GPU alike.

| Component | Requirement | What it is for |
|---|---|---|
| Python | ≥ 3.12 | The runtime; uv installs its own 3.12 |
| JAX, jaxlib | `>= 0.12.0, < 0.13.0`; on AMD GPUs also exactly `0.9.1` | Every array operation. Any other version refuses at startup, and there is no override. 0.9.1 is accepted only with the ROCm plugin installed or selected ([Frontier](#frontier)) |
| Python packages | `h5py >= 3.14`, `numpy >= 2.3.1`, `scipy >= 1.16`, `matplotlib`, `xmlschema`, `xsdata` | File I/O, the QE schema reader and plots; pinned in `pyproject.toml` and `uv.lock` |
| MPI | the site MPI | Communication between processes; both FFI legs link the same `libmpi` |
| HDF5 | parallel HDF5 ≥ 1.12, built with `HDF5_IS_PARALLEL` | Parallel reads and writes of the large files; both legs link the HDF5 the run maps |
| Host FFI leg | `liblorrax_ffi_host.so`, built from `src/ffi/cpp/` ([below](#ffi)) | Loaded by every run, a GPU run included |
| `jax_xc` | the patched 0.0.11 of `config/xc/` | Only for the [`nscf`](../drivers/nscf/index.md) driver and the dense-Hamiltonian route |

### GPU requirements {: #req-gpu }

A GPU run loads the matching device leg and the host leg.

**NVIDIA GPUs**

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

**AMD GPUs on Frontier**

Use the validated JAX 0.9.1 ROCm environment and the native ROCm/host pair,
not a CUDA extra. The ROCm leg uses the same HIP, rocBLAS, rocSOLVER, RCCL
and rocFFT libraries as the JAX plugin; ELPA supply the
distributed operations. The [Frontier route](#frontier) selects the
environment and explains the separate native builds.

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
| ROCm | `liblorrax_ffi_rocm.so` | HIP, rocBLAS/rocSOLVER, ELPA, ROCm FFT kernels and parallel HDF5 |

A CPU run loads the host leg; a GPU run loads the host and its matching
device leg into one process. A missing required leg refuses at startup.
Each leg records its dependency closure in `<leg>.closure.json`, and startup
checks that the process maps those libraries. The acceptance contract is
`scripts/verify_ffi_build.sh`; skipped runtime gates on a login node do not
certify a GPU build. A private checkout build prints `LEGACY-UNSEALED` and
its hash, with its closure still enforced; a published module carries a
sealed pair.

## Routes by machine {: #routes }

| Machine | Route | Entry |
|---|---|---|
| NERSC Perlmutter (A100) | `install_deps.sh`: one runtime image, restored on each node for every job | `config/perlmutter/` |
| OLCF Frontier (MI250X, ROCm) | a validated ROCm environment and a bare-metal host/ROCm pair | `config/frontier/` |
| TACC Frontera (Cascade Lake CPU) | a clone with the host leg, run inside apptainer | `config/frontera/` |
| one Ubuntu machine with an NVIDIA driver (workstation, WSL2, a rented GPU box) | everything from pip wheels plus the distro OpenMPI | `config/local/` |
| any other cluster | the Perlmutter scripts, re-pinned to the site ([porting](#porting)) | `src/ffi/cpp/` |

The Perlmutter route is the reference. The sections below give the commands, then what the install takes from the site, what it installs itself and which settings are not obvious, so that the same choices can be made on another cluster.

### Perlmutter: install {: #perlmutter }

Three steps. The first runs on a login node, the second once on a GPU node, the third is every run:
{: #perlmutter-script }

```bash
# 1. login node, 1.5–3 minutes, no allocation
curl -fsSLO https://raw.githubusercontent.com/jackmcarthur/lorrax/main/config/perlmutter/install_deps.sh
bash install_deps.sh $SCRATCH/lorrax

# 2. once per install, on one GPU node, about 6 minutes
salloc -N 1 -C gpu -G 4 -q interactive -t 0:30:00 -A <project>_g $SCRATCH/lorrax/bin/lorrax-postinstall

# 3. every run
source $SCRATCH/lorrax/lorrax-env.sh
srun -N 1 -n 4 --gpus-per-node=4 lorrax-run python -u -m gw.gw_jax -i gw.in
```

Step 1 prints the exact `salloc` line of step 2 with your `_g` account filled in. Step 2 prints `INSTALL VERIFIED`; until it has, the startup block of every GPU run warns `INSTALL NOT VERIFIED`. Step 3 is how every [driver](../drivers/gw/index.md) is launched from this install: `lorrax-run` in front of `python`.

#### Why a runtime image {: #perlmutter-image }

On Perlmutter's compute nodes, opening a file on `/global/common`, `$HOME` or CFS is a synchronous round trip to a DVS server. A cold LORRAX process opens about 10⁴ installed files: the CUDA and mathdx headers of every NVRTC kernel build, libdevice for every XLA module, every imported Python file. Login nodes read every filesystem at about 0.07 ms per file, so the cost is invisible there.

| Filesystem, from a compute node | Median `open()` |
|---|---|
| `$HOME` | 10.6 ms |
| CFS (`/global/cfs`) | 9.6 ms |
| `/global/common` (DVS) | 3.5 ms |
| Lustre `$SCRATCH` | 0.24 ms |
| node-local `/tmp` | 0.005 ms |

`install_deps.sh` therefore builds the whole runtime once, at a node-local path, and packs it into one archive in the prefix. Every job restores the archive to that same path on each node, in about 7 s per node, and runs from there. The prefix can then live on any filesystem. One commit (JAX 0.9.1, before the kernel store and the thread defaults), run cold through the test suite on one node with four A100s, isolates the effect of where the files are read from: the image saves about 11% against the same tree on `$SCRATCH`, and runs in less than half the time of the module on `/global/common`.

| Install | Suite, cold |
|---|---|
| image, archive on CFS | 523.6 s |
| image, archive on `$SCRATCH` | 526.6 s |
| the same tree, run in place on `$SCRATCH` | 592.0 s |
| the published module, on `/global/common` | 1164–1352 s |

The supported install as shipped (JAX 0.12, the kernel store from `lorrax-postinstall`, the thread defaults of `select_gpu.sh`): fresh installs that follow this page, each with its own empty cache root, on one node with four 80 GB A100s. The verdicts are from the self-check as it now stands, rerun on the same node type.

| Prefix on | Install (login node) | `lorrax-postinstall`: restore / kernel store / self-check | Self-check verdict | Suite, cold: k-means / GN-PPM / BSE / total |
|---|---|---|---|---|
| `/global/common` | 48.2 s | 8.6 / 235 / 41 s | PASS (`INSTALL VERIFIED`) | 12.5 / 37.1 / 10.8 / 328.3 s |
| `$HOME` | 47.9 s | 9.6 / 242 / 42 s | PASS (`INSTALL VERIFIED`) | 12.4 / 37.4 / 10.8 / 329.6 s |
| CFS | 47.3 s | 7.0 / 230 / 40 s | PASS (`INSTALL VERIFIED`) | 12.2 / 37.2 / 11.0 / 329.5 s |
| `$SCRATCH` | 48.3 s | 6.9 / 235 / 43 s | PASS (`INSTALL VERIFIED`) | 12.2 / 37.0 / 10.8 / 328.9 s |
| CFS, `--in-place`, without `lorrax-run` | 63.1 s | — | FAIL: slow filesystem, 6.74 ms per file open | 145.0 / 120.1 / 11.7 s; imports alone 238 s |

Wherever the prefix lives, the cold suite takes the same 328–330 s, against 524 s for the older commit above. The kernel store accounts for about 125 s of that; JAX 0.12 and the launch settings account for the rest. With the Triton GEMM switch above, added after these runs, the same install takes 301.8 s. The same tree run in place from CFS is several times slower before it reaches the first GPU stage.

#### Options {: #perlmutter-options }

```text
bash install_deps.sh [PREFIX] [--ref REF] [--with-xc] [--node-dir DIR]
                     [--rebuild] [--in-place] [--no-ffi] [--no-space-check]
```

| Option | Effect |
|---|---|
| `PREFIX` | where the install goes; default `$SCRATCH/lorrax`. It holds one 2.6 GB archive and needs about 2.5 GiB, on any filesystem |
| `--ref REF` | the branch, tag or commit to install; default `main`. When the script runs from inside a checkout it builds that checkout's last commit instead, and warns about uncommitted changes |
| `--with-xc` | also install the patched `jax_xc` 0.0.11, needed only by the [`nscf`](../drivers/nscf/index.md) driver |
| `--node-dir DIR` | the node-local directory the image is built in and restored to; default `/tmp/lorrax-$USER`. It must exist on every compute node |
| `--rebuild` | build a new image even when the prefix holds one of this revision |
| `--in-place` | the developer layout: the clone and its environment under the prefix, about 5 GB, run in place through the tree's `select_gpu.sh`. Fast only on `$SCRATCH` |
| `--no-ffi` | skip the native libraries |
| `--no-space-check` | skip the up-front free-space and quota check |

#### What `install_deps.sh` does {: #perlmutter-steps }

1. Checks free space and quota, and refuses in one line when the prefix (2.5 GiB), the node-local build directory (7 GB) or uv's download cache (4.5 GB when empty) lacks room.
2. Puts a pinned uv release binary in `PREFIX/bin`. uv's download cache and the compiler cache go to `$SCRATCH/.cache`, never to the prefix or `$HOME`.
3. Fetches the requested commit into `<node-dir>/rt-<revision>-<time>/lorrax` and runs `uv sync --frozen --extra cuda13`. That installs CPython 3.12 inside the image and every Python dependency at the versions in `uv.lock`: JAX with its CUDA 13 plugin, the CUDA libraries with nvcc, NCCL, cuSOLVERMp, cuBLASMp and nvidia-mathdx, all as wheels.
4. With `--with-xc`, installs and patches `jax_xc` inside the image.
5. Builds the host and CUDA libraries ([settings](#perlmutter-settings)); each ends with `VERIFY PASSED`.
6. Precompiles the Python bytecode, packs the tree into `PREFIX/images/` (a pax `tar.zst`, about 2.6 GB from 5.1 GB), removes the previous image and the build tree, and writes `PREFIX/bin/lorrax-run`, `PREFIX/bin/lorrax-postinstall` and `PREFIX/lorrax-env.sh`.
7. Runs `lorrax-postinstall`, which outside a GPU allocation only prints the `salloc` line of step 2.

With a warm uv cache the install takes about 80 s (uv sync 13 s, host library 14 s, CUDA library 27 s, packing 18 s); with an empty one, about 170 s.

#### `lorrax-run` {: #lorrax-run }

`lorrax-run <command>` is the launcher of every rank. It restores the image on the node if it is not there yet, then runs the command through `select_gpu.sh`.

- **One restore per node.** The first rank on a node fetches and unpacks the archive under a lock; the other ranks, and later jobs on the node, find it ready. A restore needs about 9.2 GB free in `/tmp` while it runs and keeps 5.1 GB, which on Perlmutter is RAM, since `/tmp` is a tmpfs.
- **Fallbacks.** Where `/tmp` lacks room it uses `/dev/shm` or `$TMPDIR`, linked at the canonical path. A node with no node-local room at all runs from one extracted copy in the prefix, which needs 5 GB more there once, and prints one line saying so.
- **Cleanup.** Each restore removes your older images on the node that no running job holds.
- **Many nodes.** Each node of a multi-node job restores the image itself, about 7 s each.
- **One GPU and a CPU share per rank.** `select_gpu.sh` gives each rank one GPU, cuts its CPU mask to one hardware thread per physical core, and sets `OMP_NUM_THREADS` to the core count minus 2, with `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. An `OMP_NUM_THREADS` you export first is kept, and then the threading is yours.
- `lorrax-run --all-gpus` skips `select_gpu.sh`, for a single task that drives every GPU of the node.

An image is a release: edits under `/tmp/lorrax-$USER` are lost at the next restore. Develop in an `--in-place` install or a clone.

#### `lorrax-postinstall` and the self-check {: #self-check }

`lorrax-postinstall`, run inside a GPU allocation, restores the image on the compute node, builds the NVRTC kernel store (389 images, 3.1 GB, about 4 minutes; without it the [k-convolution kernels](../developers/kernels.md) compile at their first use), and runs the performance self-check, `scripts/perf_check.py`, on four ranks. A pass prints `INSTALL VERIFIED` and writes a stamp; the startup warning stops. Rerun it after every reinstall, since each image has its own stamp; a complete kernel store is reused.

The self-check judges the install, the node and the launch line together, against measured A100 references, and names a fix for every failure:

| Component | Fails when | Fix |
|---|---|---|
| filesystem | the median open of the install's files is ≥ 1.0 ms (pass ≤ 0.5 ms); the environment walks at < 5000 entries/s; an `LD_LIBRARY_PATH` entry costs ≥ 1 ms per failed lookup; a read-only tree has no `.pyc` | launch with `lorrax-run`, free `/tmp`, or reinstall |
| caches | the compile or kernel cache is slow (≥ 1.0 ms per lookup), not writable, or missed by a second process; the kernel store has fewer than 10 images for this GPU | `export LORRAX_CACHE_ROOT=<fast filesystem>` and rerun `lorrax-postinstall` |
| kernels | the CUDA library has no code for this GPU; `CUDA_FORCE_PTX_JIT` or `JAX_DISABLE_JIT` is set | rebuild, or unset the variable |
| binding | ranks share CPUs, or a thread team is larger than the rank's cores | launch through `lorrax-run` or `select_gpu.sh`; leave `OMP_NUM_THREADS` unset |
| libraries | a native library refuses to load, maps a library other than its record, or a CUDA library is mapped twice | rebuild, or clear `LD_LIBRARY_PATH` |
| benchmark | a cold compile of 36 programs, a GEMM or an FFT is more than 15% slower than the reference; an NVRTC kernel build is more than 2.5 times slower (above 1.5 times it warns, naming whether its headers were read from node-local storage or the install filesystem) | follow the other components' fixes |

Its exit status is 0 for a pass, 1 for a failure, 2 when the check itself broke, and 3 on a login node, where it runs only the static checks and prints the `srun` line to use. Run it again by hand whenever the install, the modules or the launch line change:

```bash
source $SCRATCH/lorrax/lorrax-env.sh
srun -N 1 -n 4 --gpus-per-node=4 lorrax-run bash -c 'exec python $LORRAX_ROOT/scripts/perf_check.py'
```

The caches (compiled XLA programs, NVRTC images, the CUDA JIT cache) live under `LORRAX_CACHE_ROOT`, by default `$SCRATCH/.cache/lorrax`, else `~/.cache/lorrax`. Keep them off `$HOME` and `/global/common`, where every lookup costs milliseconds.

#### Things to know {: #perlmutter-notes }

- **A full `$SCRATCH` quota refuses the install.** A different prefix alone does not help, because uv's download cache (4.5 GB when empty) is on `$SCRATCH` too: also point `UV_CACHE_DIR` at a filesystem with room, as the refusal line says, or pass `--no-space-check` if the space is really there.
- **Your own `PATH`.** Directories of yours on `$HOME` early in `PATH` make the self-check warn: each command lookup that misses there costs a few milliseconds on a compute node.
- **CPU runs with several processes** also source `config/perlmutter/cpu_mpi_env.sh` for JAX's MPI collectives ([below](#perlmutter-settings)).
- **No `/tmp` on the compute nodes** makes the restore refuse with the instruction to reinstall with `--node-dir`.

??? example "The script, `config/perlmutter/install_deps.sh`"

    ```bash
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
    ```

### Perlmutter: what the site provides {: #perlmutter-site }

LORRAX links against these Perlmutter modules and system files. Everything in one process must use the same MPI, so the versions are pinned as a set in `config/perlmutter/ffi_mpi.sh`, and `config/perlmutter/ffi_site.cmake` hands them to CMake.

| Component | Pin | Role, and why this pin |
|---|---|---|
| Compiler | `gcc-native/14`, plain `gcc`/`g++`, for both libraries | no Cray compiler wrapper, so no loaded module decides what is linked |
| MPI | `cray-mpich/9.0.1` (`libmpi_gnu_123.so.12`) | the one `libmpi` both libraries link; the parallel HDF5 below is built against it |
| ScaLAPACK, BLAS | `cray-libsci/25.09.0`, threaded (`_mp`) flavour | the CPU distributed eigensolver and LU, and the host GEMM. The site default, 26.03.0, links MPICH 9.1.0, which would put a second MPI in the process |
| Parallel HDF5 | `cray-hdf5-parallel/1.14.3.7` | the parallel reads and writes of both libraries |
| FFTW | `cray-fftw` 3.3.10.11 | the host FFT |
| NCCL network plugin | `/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib/libnccl-net.so` | NCCL over the Slingshot network for multi-node runs |
| Build tools | the `cmake` module (≥ 3.23), `ninja` and `ccache` from `/usr/bin` | the CMake presets, and fast rebuilds |

Not used: the `cudatoolkit` module (nvcc and every CUDA library come from the wheels), the Cray compiler wrappers, the site Python or conda, CUDA-aware MPI and its GTL library (device collectives go through NCCL), the site cuSOLVERMp, mpi4py and mpi4jax, and ELPA, a ROCm backend.

### Perlmutter: what LORRAX installs {: #perlmutter-installs }

| Component | Version | How |
|---|---|---|
| JAX, jaxlib, the CUDA 13 plugin and PJRT | 0.12.0.1 | `uv sync --extra cuda13`, from `uv.lock` |
| CUDA libraries and nvcc | `nvidia-cuda-runtime` 13.4, `nvidia-cublas` 13.8, `nvidia-cufft` 12.4, `nvidia-cusolver` 12.3, `nvidia-nccl-cu13` 2.32.3 | the same `uv sync`; what runs, and what the CUDA library is compiled with |
| cuSOLVERMp, cuBLASMp | `nvidia-cusolvermp-cu13==0.9.1.9318.post1`, `nvidia-cublasmp-cu13==0.10.0.3695` | the same `uv sync`; the CUDA library links against these wheels |
| cuFFTDx headers | `nvidia-mathdx==25.6.0` | the same `uv sync`; the k-convolution kernels are compiled from them |
| h5py | 3.16, the PyPI wheel with its own serial HDF5 | small serial files; every parallel read and write goes through the native library, so h5py is **not** built against the site HDF5 |
| native library pair | `liblorrax_ffi_host.so`, `liblorrax_ffi.so` | `build_ffi_host.sh` and `build_ffi_cuda.sh`, through the `perlmutter-host` and `perlmutter-cuda` CMake presets; each records its dependency closure and runs the verifier |
| MPIwrapper | upstream eschnett/MPIwrapper, ABI 2.10.0 | `build_mpiwrapper.sh`; only for multi-process **CPU** runs, which use JAX's MPI collectives through MPItrampoline |

### Perlmutter: settings that are not obvious {: #perlmutter-settings }

Each setting below prevents a failure that a straightforward build or launch runs into, and each has a counterpart on any Cray or Slurm system.

**Building the two libraries** (`build_ffi_host.sh`, `build_ffi_cuda.sh`, `ffi_site.cmake`).

| Setting | Why |
|---|---|
| plain `gcc`/`g++` with explicit library paths, read from `ffi_mpi.sh` | the Cray wrapper links whatever modules are loaded: `libmpi_gtl_cuda` with `craype-accel-nvidia80`, a second LibSci, `fftw3_omp` as the OpenMP library |
| MPICH 9.0.1, LibSci 25.09.0 and HDF5 1.14.3.7 as one set | the parallel HDF5 needs `libmpi_gnu_123.so.12`, which is MPICH 9.0.1; the site-default LibSci links 9.1.0. Change all three together, in `ffi_mpi.sh` only |
| nvcc and the CUDA libraries from the `nvidia/cu13` wheel directory | the `cudatoolkit` module puts the HPC SDK's cuSOLVERMp 0.8 on `LD_LIBRARY_PATH` under the same library name, which fails inside the eigensolver with `syevd status=7`; cuSOLVERMp 0.6 returns wrong LU results on a 2-D process grid |
| `LORRAX_FFI_HAVE_CAL=OFF` | cuSOLVERMp ≥ 0.7 talks to NCCL directly |
| the venv's NCCL for the CUDA library | it must be the NCCL that JAX loads |
| `CMAKE_CUDA_ARCHITECTURES=80` and `-march=znver3` | Perlmutter has only A100s and Milan CPUs; the generic default compiles for six GPU architectures |
| `LORRAX_HOST_HAVE_SCALAPACK=OFF` (optional) | a GPU-only install never calls ScaLAPACK, and can skip LibSci's ScaLAPACK |

**Which libraries a run loads.** Each library carries `$ORIGIN` as its only `RUNPATH`. Because `LD_LIBRARY_PATH` outranks a `RUNPATH`, a module that puts another copy of a library first would win. The build therefore records the complete dependency closure in `<library>.closure.json`, and every startup refuses a process that maps any other copy (`GATE native-closure`). The pinned MPICH, HDF5 and LibSci directories precede `/opt/cray/pe/lib64`, whose links point at the site defaults.

**Launching** (`lorrax-env.sh`, which applies `gpu_env.sh`, and `select_gpu.sh`).

| Setting | Why |
|---|---|
| `MPICH_GPU_SUPPORT_ENABLED=0` | `craype-accel-nvidia80` sets it to 1, and `MPI_Init` then aborts with `GTL library is not linked` |
| `SLURM_NETWORK=no_vni` (as `srun --network=no_vni`) | with a VNI, MPI-IO and NCCL's network start-up fail with `ENOSPC` |
| a `libcudart.so` link to the wheel's runtime, first on the library path | libfabric opens the unversioned name; without the link a second CUDA runtime loads and a multi-node step refuses. `lorrax-run` makes the link inside the image |
| `NCCL_NET="AWS Libfabric"`, `NCCL_NET_GDR_LEVEL=PHB`, `NCCL_SOCKET_IFNAME=hsn`, `FI_CXI_DISABLE_HOST_REGISTER=1`, `FI_CXI_RDZV_THRESHOLD=0`, `FI_MR_CACHE_MONITOR=kdreg2` | set by the runtime from `config/perlmutter/site.toml` unless already set. The rendezvous threshold prevents a cross-node deadlock, and `kdreg2` replaces the default memory-registration monitor, which deadlocked inside a cuSOLVERMp split |
| one GPU per rank through `select_gpu.sh`, never `--gpus-per-task=1` | `--gpus-per-task=1` confines each rank so that the first collective fails with `invalid device ordinal` |
| one thread per physical core, `OMP_NUM_THREADS` = cores − 2, BLAS threads 1 | without it each rank's OpenMP team runs on hyperthread siblings and oversubscribes; that cost about 8% of a stage |

The runtime also chooses the JAX coordinator (the step's first host, on a port derived from the job and step ids; `JAX_COORDINATOR_ADDRESS` overrides it) and appends its flags to `XLA_FLAGS`: on NVIDIA GPUs `--xla_gpu_autotune_level=0` and `--xla_gpu_enable_triton_gemm=false`, and the rematerialization pass off. Without the Triton switch, XLA sends real float64 dots to an untuned Triton GEMM that runs about 5 times slower on an A100. A value you set wins. Do not set XLA's allocator or memory-fraction variables.

A multi-process CPU run additionally sources `config/perlmutter/cpu_mpi_env.sh`, which sets `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`, `MPITRAMPOLINE_LIB` to the MPIwrapper library, `LD_PRELOAD=/opt/cray/pe/lib64/libpmi.so.0`, `JAX_NUM_CPU_DEVICES=1` and `MPICH_ASYNC_PROGRESS=1`.

The CPU adapter manifest fingerprints `cpu_mpi_config.sh`, its builder, the
rank prelude and the one-MPI gate. After updating from an older manifest,
run `config/perlmutter/build_mpiwrapper.sh` once on a CPU compute node;
the prelude prints that command if a refresh is required. CUDA-only changes
to site defaults no longer invalidate the CPU adapter.

### Perlmutter: the module {: #perlmutter-module }

```bash
module use <prefix>/modulefiles
module load lorrax
srun -N 1 -n 4 --gpus-per-node=4 $LORRAX_ROOT/src/ffi/cpp/select_gpu.sh python -u -m gw.gw_jax -i gw.in
```

The published module keeps its environment and source on `/global/common`, which makes a cold run 2.2 to 2.6 times slower than an `install_deps.sh` install ([why](#perlmutter-image)), and its self-check fails on the filesystem component. Prefer `install_deps.sh`. The module sets `LORRAX_ROOT`, `LORRAX_FFI_SO`, `LORRAX_FFI_HOST_SO`, `MPICH_GPU_SUPPORT_ENABLED=0`, `SLURM_NETWORK=no_vni`, `JAX_PLATFORMS=cuda,cpu` and `JAX_ENABLE_X64=1`. Do not add a checkout to `PYTHONPATH` or pin one library by hand: a mixed source tree or a partial bundle refuses. Maintainers build it with the scripts in `config/perlmutter/module/`.

### Porting to another cluster {: #porting }

Copy `config/perlmutter/` and replace the site-specific pieces:

| Perlmutter piece | Replace with |
|---|---|
| `ffi_mpi.sh`: MPICH 9.0.1, LibSci 25.09.0, HDF5 1.14.3.7 | the site's MPI, ScaLAPACK and parallel HDF5 (≥ 1.12), all built against the **same** MPI. `config/frontier/ffi_mpi.sh` is a second example |
| LibSci | MKL (`mkl_scalapack_lp64` with the `mkl_blacs_<mpi>_lp64` that matches the MPI), netlib ScaLAPACK with OpenBLAS or BLIS, or AOCL; the link line goes in `LORRAX_SCALAPACK_LIBRARIES` |
| `cray-hdf5-parallel` | a parallel HDF5 ≥ 1.12; `config/local/` builds 1.14.6 from source when the system has none |
| `ffi_site.cmake` | the same file with the site's compiler, paths, GPU architecture and CPU flags |
| `site.toml` (NCCL plugin and libfabric settings) | the site's NCCL network plugin and its recommended variables, or nothing on a single node |
| `gpu_env.sh` (`no_vni`, GTL off, the `libcudart` link) | the site's equivalents; on one machine, `config/local/launch.sh` with `mpirun` |
| `select_gpu.sh` | keep it; it reads Slurm's variables, or `LORRAX_GPU_DEVICE` elsewhere |
| `/tmp` as the image's node-local directory | any node-local disk or tmpfs with about 10 GB free (`--node-dir`); with none, run `--in-place` from the fastest shared filesystem |

Before choosing where things go, measure the file-open latency of each filesystem **from a compute node**: login nodes hide it. The self-check does this as its first component, and with exit status 3 on a login node it prints the `srun` line to run it where it counts. Put the runtime on node-local storage and the caches (`LORRAX_CACHE_ROOT`) on the fastest shared filesystem, then run the self-check on a GPU node.

On a cluster without Cray compiler wrappers, `config/local/build_ffi.sh` and `build_ffi_host.sh` are the closer template: plain `gcc`/`g++`, nvcc from the wheels, the system MPI and ScaLAPACK. Build through CMake, whose post-build step writes the `.closure.json` every startup requires and runs `scripts/verify_ffi_build.sh`.

### Frontier (OLCF) {: #frontier }

Use the existing validated JAX/jaxlib 0.9.1 environment with `rocm-sdk`
7.14.1 wheels. From the checkout, source the setup once:

```bash
source config/frontier/gpu_env.sh
```

It selects `LORRAX_OLCF_VENV`, the active `VIRTUAL_ENV`, or the default
`$HOME/envs/lx-jax091-rocm7141`, in that order. It activates the environment,
checks for the ROCm plugin without starting a GPU client, prepares the
host/ROCm library paths and selects this checkout's source. It does not
create the Python environment or build native libraries. For a fresh Python
3.12 installation, use the [pinned ROCm/JAX recipe](https://github.com/jackmcarthur/lorrax/blob/main/docs/installation/frontier.md#fresh-python-installation):
AMD's multi-architecture wheel index supplies
`rocm[libraries,device-gfx90a]==7.14.1` and
`jax_rocm7_plugin`/`jax_rocm7_pjrt==0.9.1+rocm7.14.1`; PyPI supplies
`jax==0.9.1` and `jaxlib==0.9.1`. That recipe installs the editable workspace
packages from `pyproject.toml`, then the root package, and records the final
package manifest. No CUDA extra is installed. `config/frontier/constraints.txt`
holds these pins: export it as `PIP_CONSTRAINT` for any later `pip install` in
that environment, and do not point `uv sync` at it, since `uv.lock` holds JAX
0.12. JAX 0.9.1 is accepted on AMD GPUs only; the same environment run with
`JAX_PLATFORMS=cpu` still counts as AMD, because its ROCm plugin is installed.

The normal PyPI `h5py` wheel is supported; no MPI-enabled `h5py` build is
needed. Its private serial HDF5 libraries are separate from the one pinned
parallel HDF5 family shared by both native legs. The tested environment
uses `h5py 3.16.0` with private HDF5 2.0.0, while the native pair uses site
HDF5 1.14.3.7. Runtime owns the load order and file-handle separation.

If the shared dependency installs are already present, build only the two
LORRAX legs for this checkout:

```bash
bash config/frontier/build_ffi_host.sh
bash config/frontier/build_ffi_rocm.sh
```

On a first installation, run `bash config/frontier/build_openblas_seq.sh`
before those commands. Its install under `$HOME/software/openblas_seq/install`
and the content-keyed ELPA installs under `$HOME/software/elpa_builds` are
shared across checkouts; the ROCm builder creates a missing matching ELPA prefix. The ROCm builder creates or verifies its content-keyed ELPA
prefix under `$HOME/software/elpa_builds`. Rebuilding LORRAX does not
require deleting or rebuilding these dependencies.

After building both legs, certify them inside an active allocation with
four free GCDs:

```bash
bash config/frontier/verify.sh
```

The command sources `gpu_env.sh`, verifies the recorded host/ROCm closures
and intended host FFTW, then compares native and XLA Fourier transforms at
four MPI processes. It refuses disabled acceptance checks. The host build
cache supplies the FFTW prefix; a copied host leg without that cache needs
the existing `LORRAX_FFTW3_STAGE` set to its verified FFTW prefix. No added
library paths or runtime toolkit module are needed. Skipped runtime gates
on a login node do not certify a build.

Run a calculation inside the allocation:

```bash
source config/frontier/gpu_env.sh
srun -N 1 -n 4 -c 7 --threads-per-core=1 --gpus-per-task=1 \
  python -u -m gw.gw_jax -i /path/to/run/gw.in
```

`--threads-per-core=1` selects one hardware thread per CPU core for each
rank's seven cores. `--gpus-per-task=1` also supplies Slurm's `per_task:1`
binding, assigning one
GCD to each MPI process. Keep that implicit binding; an explicit
`--gpu-bind=closest` replaces it with proximity-based selection
([Slurm reference](https://slurm.schedmd.com/srun.html#OPT_gpus-per-task)).

This is a 2 × 2 mesh on four GCDs, one per MPI process. A Frontier node has
eight GCDs, but eight processes are not a valid square mesh; sixteen form
a 4 × 4 mesh on two complete nodes. The helper and runtime own the ROCm
and RCCL/OFI setup; do not use the NVIDIA `select_gpu.sh` or copy Perlmutter
network settings. No container or runtime ROCm toolkit module is needed.

The records can also be inspected with `python -m lxkit.closure verify <leg.so>`.
Reusing a build from another checkout also requires matching FFI sources,
Frontier build inputs and JAX/jaxlib headers; a matching ABI number or a
closure JSON alone is insufficient. At runtime both closures must be
enforced. Full build pins, paths and reuse rules are in the
[Frontier installation source](https://github.com/jackmcarthur/lorrax/blob/main/docs/installation/frontier.md).

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
source $SCRATCH/lorrax/lorrax-env.sh
srun -N 1 -n 4 --gpus-per-node=4 lorrax-run bash -c \
  'cd $LORRAX_ROOT && exec python -m pytest tests/hsuite -q -p no:cacheprovider'
```

From a clone or an `--in-place` install, run the same `pytest` line through the tree's `src/ffi/cpp/select_gpu.sh .venv/bin/python` instead of `lorrax-run`. On one node with four A100s it takes about 8 minutes cold and 2.5 minutes once the caches are warm.

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
    - `JAX_ENV_REFUSED`, or the `runtime.jax_support` refusal before the first `jit`: JAX or jaxlib outside the 0.12 series (on AMD GPUs, outside the 0.12 series and 0.9.1). Install the 0.12 series.
    - `GATE mathdx-headers`: a CUDA run without `nvidia-mathdx`. Run `uv sync --extra cuda13` (or `cuda12`).


# Environment Setup & Configuration

Dependencies, installation, JAX configuration, cluster usage, and troubleshooting. Read this for build and deployment issues. For code structure see [`CODEBASE_COMPREHENSIVE.md`](architecture/codebase.md). For Perlmutter specifically, [`config/README.md`](../config/README.md) is authoritative — this file summarises and cross-references.

---

## Table of Contents

1. [Dependencies](#1-dependencies)
2. [Installation Methods](#2-installation-methods)
3. [JAX Configuration](#3-jax-configuration)
4. [Perlmutter via Lmod module](#4-perlmutter-via-lmod-module)
5. [FFI stack (SLATE, cuSOLVERMp, phdf5)](#5-ffi-stack-slate-cusolvermp-phdf5)
6. [Multi-Host / Multi-GPU Setup](#6-multi-host--multi-gpu-setup)
7. [Generic SLURM clusters](#7-generic-slurm-clusters)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Dependencies

Authoritative source: [`pyproject.toml`](../pyproject.toml). No docker images, no cupy, no jax-finufft. The Perlmutter Shifter image ships JAX + CUDA; the rest is installed via `uv sync` or bind-mounted from `LORRAX_SITE_PACKAGES`.

### 1.1 Runtime dependencies

| Package | Version | Purpose |
|---|---|---|
| **Python** | ≥3.12 | Language runtime |
| **jax[cuda13]** | ≥0.9 | Array ops, autodiff, GPU acceleration (CUDA-13 wheels bundle the runtime) |
| **jaxlib** | ≥0.9 | JAX backend |
| **numpy** | ≥2.3.1 | Arrays, host-side I/O |
| **scipy** | ≥1.16.0 | Linear algebra, FFTs (host) |
| **h5py** | ≥3.14.0 | HDF5 I/O (host path) |
| **matplotlib** | ≥3.10.3 | Plotting |
| **xmlschema**, **xsdata** | ≥4.1.0, ≥25.7 | UPF pseudopotential parsing |
| **mkdocs** / **mkdocs-material** / **mkdocstrings** | — | Documentation site + API reference (`mkdocs build`) |

### 1.2 Dependency groups

Declared under `[dependency-groups]` in `pyproject.toml`:

| Group | Contents | When |
|---|---|---|
| `dev` | flake8, pytest | Everywhere |
| `jax` | Explicit `jax[cuda13]` + jaxlib pin | If uv resolves without extras |
| `build` | cmake, ninja, nanobind, scikit-build-core | Building the FFI C++ shared object `liblorrax_ffi.so` (§5) |
| `profile` | tensorboard, tensorboard-plugin-profile, xprof | JAX/XProf traces |

### 1.3 Not dependencies

The following appear in older docs but are **removed or never required**:

- **cupy / cufft / cupyx** — gone. All GPU arrays are JAX. `common.gpu_utils` only provides host-side memory detection helpers.
- **jax-finufft / finufft / cufinufft** — not a dep. Any residual NUFFT code paths have been either retired or rewritten onto FFTs.
- **Docker / docker-compose** — no Dockerfiles in-tree. Production uses the NVIDIA JAX Shifter image; local dev uses plain `uv venv`.

---

## 2. Installation Methods

### 2.1 Local dev (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # one-time
cd $LORRAX_ROOT
uv sync                                            # editable install of the project (puts src/ on sys.path)
source .venv/bin/activate
uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in
# or: uv run gw_jax -i tests/regression/cohsex_debug/cohsex_test.in
```

Console scripts (from `pyproject.toml`):

```toml
lorrax-gw        = "gw.gw_jax:main"
gw_jax           = "gw.gw_jax:main"          # alias
lorrax-centroids = "centroid.kmeans_cli:main"
```

One `.venv/` per machine, gitignored. Let uv use its global cache.

### 2.2 Perlmutter (Shifter via Lmod module)

See §4. The module bind-mounts `LORRAX_SITE_PACKAGES` into the container so the JAX image gets h5py / scipy / matplotlib without an in-container pip install. No uv invocation needed inside the container — the python interpreter inside Shifter is the one that runs LORRAX.

---

## 3. JAX Configuration

### 3.1 Environment variables

Set **before `import jax`**. `gw_jax.py` hard-defaults the two X64 / platform vars; everything else is set by `module load lorrax` (see `config/README.md`):

| Variable | Value | Purpose |
|---|---|---|
| `JAX_ENABLE_X64` | `1` | 64-bit precision (required for GW) |
| `JAX_PLATFORMS` | `cuda,cpu` | Prefer CUDA, fall back to CPU |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Don't pre-grab a fixed XLA pool |
| `XLA_PYTHON_CLIENT_ALLOCATOR` | `platform` | Use CUDA async mempool (shared with NCCL / cuSOLVERMp / SLATE) |
| `TF_GPU_ALLOCATOR` | `cuda_malloc_async` | CUDA 12 async allocator (no pipeline stalls) |
| `JAX_COMPILATION_CACHE_DIR` | `$SCRATCH/.jax_cache` | Persistent XLA PTX cache across processes |
| `HDF5_USE_FILE_LOCKING` | `FALSE` | Lustre HDF5 compatibility |

### 3.2 Why the platform allocator, not `MEM_FRACTION=0.95`

Pre-allocating 95 % of A100 VRAM into XLA's BFC pool leaves NCCL only ~2 GB for staging, and cuSOLVERMp's `syevd` surfaces this as `NCCL error 1 unhandled cuda error` → `cusolverMpSyevd: status=7`. Using `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (cudaMallocAsync) lets XLA and NCCL (and CAL / SLATE) share one pool. If you need a hard cap, use `MEM_FRACTION` only with `XLA_PYTHON_CLIENT_ALLOCATOR=default` (BFC) — the platform allocator ignores it.

### 3.3 Device selection

```python
import jax
devices = jax.devices()              # all available GPUs or CPUs
jax.config.update("jax_platform_name", "cpu")    # force CPU
```

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m gw.gw_jax -i cohsex.in   # restrict GPUs
```

### 3.4 Multi-host CPU mock (for sharding tests)

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
```

### 3.5 CPU multi-process MPI runs (production-quality)

Validated end-to-end at Si 4×4×4 μ=384, x_only + full COHSEX, on
Perlmutter Milan (1 node, 4 ranks × 8 threads).  Same `cohsex.in`
works on both GPU and CPU backends; LORRAX auto-routes the FFI flags
based on `jax.default_backend()` (see `gw.gw_config.LorraxConfig.
from_input_file`).

**Required dependencies** (one-time, inside the venv):

```bash
module load cray-hdf5-parallel/1.12.2.9 cray-mpich/9.0.1
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICC=$(which mpicc) CC=$(which mpicc)
export HDF5_MPI=ON HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.12.2.9/gnu/12.3
VIRTUAL_ENV=$LORRAX_VENV uv pip install --link-mode=copy \
    --no-binary=mpi4py mpi4py
VIRTUAL_ENV=$LORRAX_VENV uv pip install --link-mode=copy \
    --no-binary=h5py --force-reinstall --no-deps h5py==3.16.0
```

Verify both have MPI built in:

```bash
$LORRAX_VENV/bin/python -c "
from mpi4py import MPI; print('mpi4py + Cray MPICH OK')
import h5py; assert h5py.get_config().mpi, 'h5py NOT built with MPI'
print('h5py-parallel OK')"
```

**Launch recipe**:

```bash
salloc --nodes=1 --qos=interactive --constraint=cpu --time=04:00:00 \
       --account=m2651 -J "lx-alloc-$USER" bash -c "sleep 100000" &
# wait for RUNNING in squeue, then export SLURM_JOBID

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=0
export PYTHONPATH=$LORRAX_SRC/src

srun --jobid=$SLURM_JOBID -N 1 -n 4 -c 8 --cpu-bind=cores \
     $LORRAX_VENV/bin/python -u -m gw.gw_jax -i cohsex.in
```

The `lxalloc` + `lxrun` shell helpers from the `lorrax_agent` overlay
module support `LORRAX_PARTITION=cpu` to do the same thing more
concisely; see the overlay's source for the GPU vs CPU branches.

**What auto-routes on CPU**:

| `cohsex.in` setting | CPU value | Routes to |
|---|---|---|
| `use_ffi_io = true` | (unchanged) | `_slab_io_mpi_host.py` (per-rank MPI-IO via mpi4py + h5py-parallel) |
| `cusolvermp_charge = auto/on` | forced `off` | in-tree `cholesky_2d.sharded_cholesky` |
| `cusolvermp_lu = auto/on` | forced `off` | per-q `jnp.linalg.solve` |
| `pair_density_slots = 3` (GPU XLA) | 4 (CPU XLA) | `_default_pair_density_slots()` |

The CPU path uses synchronous writes (no async writer thread) because
the FFI's threaded design deadlocks at `H5Fclose` under Cray MPICH's
default `MPI_THREAD_SINGLE`.  See `_slab_io_mpi_host.py` module
docstring for the full rationale.

### 3.5 Memory inspection

```python
import jax
print(jax.local_devices()[0].memory_stats())
```

Also useful: `common.gpu_utils.get_device_memory_info()` returns a dict with `backend / total_gb / available_gb / budget_gb / source`, which `gw_init` uses to size chunking parameters.

---

## 4. Perlmutter via Lmod module

### 4.1 One-time install

```bash
vi config/perlmutter/site_config.sh          # edit account, QoS, paths
bash config/perlmutter/install.sh
```

To run several checkouts side-by-side, install each with a distinct module name:

```bash
LORRAX_MODULE_NAME=<module-name> bash $LORRAX_ROOT/config/perlmutter/install.sh
```

`family("lorrax")` in the modulefile makes variants mutually exclusive within a shell; loading one auto-swaps the other out. Separate shells are fully independent (own `LORRAX_ROOT`, own `lxalloc` allocation).

### 4.2 Every session

```bash
module load lorrax
lxalloc                                 # 1 node / 4 GPUs / 2 h, exports SLURM_JOBID
lxalloc 4                               # 4 nodes / 16 GPUs / 2 h
lxalloc 1 4:00:00                       # custom time

lxpre cohsex.in 640                     # all 3 preprocessing steps (single-GPU)
lxrun python3 -u -m gw.gw_jax -i cohsex.in           # 4-GPU GW
LORRAX_NGPU=1 lxrun python3 -u -m gw.gw_jax -i cohsex.in    # single-GPU override
lxshell                                 # interactive single-rank shell in container

lxkill                                  # cancel allocation, unset SLURM_JOBID
```

Every `lxrun` invocation expands to:

```bash
srun --mpi=cray_shasta --gres=gpu:$NGPU -N $NNODES -n $NGPU \
     select_gpu.sh   \                          # CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
     shifter --module=gpu,mpich --image=... --volume=... --env=... \
     in_container.sh \                          # re-asserts MPICH_GPU_SUPPORT_ENABLED=1
     "$@"
```

Each rank sees exactly one GPU as device 0 — callers that need `jax.distributed.initialize` pass `local_device_ids=[0]` (§6).

### 4.3 Batch submission

Template at [`config/perlmutter/run_gw.slurm`](../config/perlmutter/run_gw.slurm):

```bash
#!/bin/bash -l
#SBATCH -N 1 -C gpu -q regular -t 01:00:00 -A m2651
#SBATCH --ntasks-per-node=4 --gpus-per-node=4

module load lorrax
lxrun python3 -u -m gw.gw_jax -i cohsex.in 2>&1 | tee gw.out
```

`SLURM_JOBID` is not required inside `sbatch` — the allocation is the job itself.

### 4.4 Per-invocation cost

| Phase | Time |
|---|---|
| srun step creation | 2–5 s |
| Shifter namespace bring-up | ~5 s |
| `import jax` + first GPU tensor | ~1.2 s |
| `jax.distributed.initialize` handshake (multi-rank) | 3–5 s |

Single-rank: ~7 s end-to-end. Multi-rank: ~10–15 s. Fast-iteration knobs:

- `lxshell` keeps the container alive across calls (saves ~5 s bring-up per invocation). Still pays Python cold-start; for real back-to-back work keep one REPL alive.
- `JAX_COMPILATION_CACHE_DIR=$SCRATCH/.jax_cache` amortises XLA PTX compile across processes.

---

## 5. FFI stack (SLATE, cuSOLVERMp, phdf5)

LORRAX ships an XLA FFI bridge (`src/ffi/`) that calls into three native libraries not present in the JAX container. A single `liblorrax_ffi.so` exposes all of them; the LORRAX module bind-mounts pre-staged copies of the supporting shared libraries into the container at fixed paths.

### 5.1 Targets

| Subpackage | Library | Process model | Use |
|---|---|---|---|
| `cusolvermp` | cuSOLVERMp + CAL/NCCL | 1 proc per GPU | Distributed `eigh` (syevd) |
| `phdf5` | parallel HDF5 via MPI-IO | 1 proc per GPU | Sharded slab read/write of `zeta_q.h5`, `V_qmunu.h5`, etc. |
| `slate` | SLATE + libsci | p×q GPU grid | Distributed Cholesky, trsm, heev (evaluation) |

Details: [`src/ffi/AGENTS.md`](../src/ffi/AGENTS.md), [`src/ffi/PORTING.md`](../src/ffi/PORTING.md), [`src/ffi/phdf5/ARCHITECTURE.md`](../src/ffi/phdf5/ARCHITECTURE.md).

### 5.2 Bind-mounts

Pre-staged host-side directories bind-mount to fixed container paths:

| Host path (override var) | Container mount | Contents |
|---|---|---|
| `$LORRAX_FFI_NVHPC_DIR` (default `$HOME/software/lorrax_nvhpc`) | `/lorrax_nvhpc` | NVHPC 25.5 subset: `libcusolverMp.so.0`, `libcal` |
| `$LORRAX_FFI_PHDF5_DIR` (default `$HOME/software/lorrax_phdf5_cray/stage`) | `/lorrax_phdf5` | Cray HDF5 1.12 (libmpi_gnu_*.so.12) |
| `$LORRAX_FFI_SLATE_DIR` (default `$HOME/software/lorrax_slate_cray/stage`) | `/lorrax_slate` | Cray libsci + `libmpi_gtl_cuda.so.0` + xpmem + lustreapi |

Container-side `LD_LIBRARY_PATH` (order matters):

```
$LORRAX_SLATE_INSTALL_DIR/lib64 : /lorrax_slate/lib : /lorrax_phdf5/lib :
/lorrax_nvhpc/<nvhpc-subpath>/lib64 : /opt/udiImage/modules/mpich :
/opt/udiImage/modules/mpich/dep [: darshan]
```

`LORRAX_NVHPC_SUBPATH`, `LORRAX_MPICH_CONTAINER_DIR`, and `LORRAX_DARSHAN_LIB_DIR` are cluster-specific and patched from `site_config.sh`.

### 5.3 Staging (one-time per cluster)

```bash
src/ffi/cusolvermp/scripts/stage_nvhpc.sh   # cuSolverMp + CAL (~100 MB)
src/ffi/phdf5/scripts/stage_cray.sh         # Cray HDF5 1.12 — canonical default on Perlmutter
src/ffi/phdf5/scripts/stage_openmpi.sh      # OpenMPI HDF5 — the portable stack for non-Cray clusters
src/ffi/slate/scripts/stage_cray.sh         # libsci + GTL + xpmem
```

Staging copies are mandatory because Shifter's `udiRoot.conf` on Perlmutter forbids `--volume` sources under `/opt/*` or `$HOME` — only `/pscratch` is bind-mountable. All scripts are idempotent and end with a `readelf -d` sanity check.

### 5.4 Building `liblorrax_ffi.so`

> **Prereqs.** Before building, (a) the native stacks must be staged (`stage_nvhpc.sh` for cuSolverMp at minimum — see §5.3), and (b) you must hold a GPU allocation (`lxalloc`). `build.sh` fails loudly if `LORRAX_MPI_INCLUDE_DIR` / `LORRAX_MPICH_LIB_DIR` are unset, which `run_shifter.sh` sets for you.

```bash
src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
```

Output: `src/ffi/common/cpp/build/liblorrax_ffi.so`. CMake logs the resolved HDF5 / MPI paths — eyeball them to confirm the right stack.

To build **outside** Shifter (a non-Cray cluster, native libs obtained independently), drive CMake directly with explicit `-D` overrides instead of `run_shifter.sh` — see [`src/ffi/PORTING.md`](../src/ffi/PORTING.md) and (once available) the Installation → FFI native libraries page.

### 5.5 MPI stack override

```bash
LORRAX_MPI_TYPE=cray_shasta   # default — Cray MPICH PMI
LORRAX_MPI_TYPE=none          # disable --mpi flag (single-rank code)
LORRAX_MPI_TYPE=pmix          # legacy OpenMPI path, not unified; has hung non-FFI workloads
```

`pmix` is opt-in and known to hang some non-FFI workloads — don't set it unconditionally.

### 5.6 GPU-aware MPICH (Cray MPICH stack only)

`module load lorrax` sets:

```
MPICH_GPU_SUPPORT_ENABLED=1
LD_PRELOAD=libmpi_gtl_cuda.so.0      # CUDA-12 copy for the container's MPICH
```

This activates GPU-Direct RDMA for `MPI_*` collectives used by SLATE / CAL. `in_container.sh` re-asserts the env var inside Shifter after the image switch.

These two knobs are **Cray-MPICH-specific** GPU-Direct mechanisms — they do not exist for OpenMPI/UCX. On an OpenMPI cluster, CUDA-awareness comes from CUDA-aware UCX (`UCX_*` / `OMPI_MCA_*` env); none of the `MPICH_*` / `libmpi_gtl_cuda` vars apply.

---

## 6. Multi-Host / Multi-GPU Setup

### 6.1 Distributed init

`gw_jax.py:21` calls `_maybe_init_jax_distributed()` which auto-detects SLURM:

```python
proc_count = int(os.environ.get("SLURM_NTASKS", "1"))
if proc_count > 1:
    jax.distributed.initialize()      # reads SLURM_PROCID, SLURM_NTASKS via Cray PMI
```

A sentinel env var (`_LORRAX_JAX_DISTRIBUTED_DONE`) guards re-entry when `python -m gw.gw_jax` executes as `__main__` and then gets re-imported as `gw.gw_jax`.

### 6.2 One-GPU-per-rank via `select_gpu.sh`

LORRAX deliberately avoids `--gpus-per-task=1` and instead pins GPUs via `select_gpu.sh` (`CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`).

Rationale: `--gpus-per-task=1` breaks JAX's distributed topology sync. Each rank's `local_devices` call returns narrow-view device ordinals, but JAX's coordinator expects all ranks to report from the same global ordinal space. With `select_gpu.sh` each rank sees exactly one GPU as device 0, and the sandbox tests that call `jax.distributed.initialize()` pass `local_device_ids=[0]` (auto-detected via `len(CUDA_VISIBLE_DEVICES.split(",")) == 1`).

### 6.3 Multi-node via the module

```bash
#!/bin/bash -l
#SBATCH -N 2 -C gpu -q regular -t 04:00:00 -A m2651
#SBATCH --ntasks-per-node=4 --gpus-per-node=4

module load lorrax
LORRAX_NNODES=2 LORRAX_NGPU=8 lxrun python3 -u -m gw.gw_jax -i cohsex.in
```

Expected topology inside the job:

```python
import jax
jax.process_index(), jax.process_count()  # e.g. (0, 8)
jax.local_devices()                       # [cuda:0]  — one per rank
len(jax.devices())                        # 8 — global across both nodes
```

### 6.4 Non-SLURM clusters

```python
jax.distributed.initialize(
    coordinator_address="node001:12355",
    num_processes=4,
    process_id=0,
)
```

Or via env vars `JAX_COORDINATOR_ADDRESS`, `JAX_NUM_PROCESSES`, `JAX_PROCESS_INDEX`.

---

## 7. Generic SLURM clusters

Port via `config/<cluster>/` (see [`config/README.md`](../config/README.md) §Porting for the full knob list). Headline edits live in `site_config.sh`:

| Knob | Perlmutter value |
|---|---|
| `LORRAX_SLURM_{ACCOUNT,QOS,CONSTRAINT}` | `m2651` / `interactive` / `gpu` |
| `LORRAX_GPUS_PER_NODE` | 4 |
| `LORRAX_SHIFTER_MODULES` | `gpu,mpich` |
| `LORRAX_MPI_TYPE_DEFAULT` | `cray_shasta` |
| `LORRAX_NVHPC_SUBPATH` | `25.5_cuda12.9/math_libs/12.9/lib64` |
| `LORRAX_MPICH_CONTAINER_DIR` | `/opt/udiImage/modules/mpich` |
| `LORRAX_DARSHAN_LIB_DIR` | (optional; empty to skip) |
| `LORRAX_FFI_{NVHPC,PHDF5,SLATE}_DIR_DEFAULT` | under `$HOME/software` |
| `LORRAX_SLATE_INSTALL_DIR_DEFAULT` | `$HOME/software/slate/install` |

For non-Shifter runtimes (Apptainer, Singularity, bare venv) you need to swap the `shifter` invocation in `lxrun`/`lxshell`/`lxpre`. `select_gpu.sh`, `in_container.sh`, `LD_LIBRARY_PATH` composition, and SLURM defaults are portable.

Bare-venv fallback (no container):

> **Note.** This bare-venv example runs only the **pure-JAX** code path (centroids, load, serial GW). The distributed FFI features (cuSolverMp `eigh`, sharded HDF5, SLATE) additionally require building `liblorrax_ffi.so` against native libs you must obtain — see §5 and (once available) the Installation → FFI native libraries page. The `site_config.sh` stage knobs above assume Cray-PE modules to copy from.

```bash
#!/bin/bash
#SBATCH -N 1 --gres=gpu:4 -t 02:00:00
module load cuda/12.3 python/3.12
source $LORRAX_ROOT/.venv/bin/activate
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async
python -m gw.gw_jax -i cohsex.in
```

---

## 8. Troubleshooting

### 8.1 "No GPU/TPU found"

```
RuntimeError: No GPU/TPU found, falling back to CPU.
```

1. `nvidia-smi` — are GPUs visible at all?
2. `python -c "import jax; print(jax.default_backend())"` — should print `gpu`
3. `echo $CUDA_VISIBLE_DEVICES` — inside Shifter this is set by `select_gpu.sh` to `$SLURM_LOCALID`
4. `pip list | grep jax` — jaxlib must be the CUDA build (`jax[cuda13]`)

### 8.2 Out-of-memory

```
XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory
```

1. Confirm `memory_per_device_gb` in `cohsex.in` matches GPU (28 on A100, 6 on RTX 5070).
2. Reduce `chunk_bands` / `chunk_q` (or unset and let `gw_init.compute_optimal_chunks` auto-size).
3. Inspect bottleneck: `common.gpu_utils.get_device_memory_info()` at a checkpoint, or `jax.local_devices()[0].memory_stats()`.
4. Check `architecture/memory-model.md` for per-stage formulas.

### 8.3 cuSOLVERMp `status=7` or NCCL "unhandled cuda error"

Symptom: eigh FFI fails with `cusolverMpSyevd: status=7` and an NCCL error 1.

Cause: XLA BFC preallocated the GPU pool, leaving NCCL no staging memory.

Fix: confirm `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_ALLOCATOR=platform` are set (the module sets them; a user-level `XLA_PYTHON_CLIENT_MEM_FRACTION` export can override and break things).

### 8.4 `LORRAX_MPI_TYPE=pmix` hangs

Don't set `pmix` unconditionally — it hangs some non-FFI workloads. The unified default (`cray_shasta`) covers SLATE, cuSOLVERMp, and phdf5.

### 8.5 HDF5 "file is already open" on Lustre

Set (or confirm) `HDF5_USE_FILE_LOCKING=FALSE`. The module sets this.

### 8.6 JIT cache misses after module updates

Stale cache entries can produce `KeyError` warnings. Easiest: `rm -rf $JAX_COMPILATION_CACHE_DIR`, or use a per-checkout subdir (the module does this via `common.jax_compile_cache`).

### 8.7 Debug flags

```bash
export JAX_DEBUG_NANS=1              # crash on NaN
export JAX_DISABLE_JIT=1              # disable JIT
export JAX_LOG_COMPILES=1             # log compilations
export TF_CPP_MIN_LOG_LEVEL=0         # verbose XLA logs
```

Profiling:

```python
import jax
jax.profiler.start_trace("/tmp/jax_trace")
# ... run code ...
jax.profiler.stop_trace()
```

Canonical xprof triage: [`reports/ppm_sigma_profiling_2026-04-05/XPROF_TRACE_GUIDE.md`](../../reports/ppm_sigma_profiling_2026-04-05/XPROF_TRACE_GUIDE.md) in the sandbox repo (not in LORRAX src).

---

## Next Steps

- Code structure: [`CODEBASE_COMPREHENSIVE.md`](architecture/codebase.md)
- Physics / theory: [`PHYSICS_COMPREHENSIVE.md`](theory/physics.md)
- Memory model: [`MEMORY_MODEL.md`](architecture/memory-model.md)
- Perlmutter cluster detail: [`../config/README.md`](../config/README.md)
- FFI internals: [`../src/ffi/AGENTS.md`](../src/ffi/AGENTS.md)
- Agent TODOs: see developer notes under `docs/dev/notes/AGENT_TODO.md`

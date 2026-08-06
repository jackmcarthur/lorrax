# LORRAX Configuration

Environment modules, site configuration, and batch scripts for running
LORRAX on HPC clusters.

## Quick Start (Perlmutter)

```bash
# 1. Edit site-specific paths (source dir, site-packages, container image)
vi config/perlmutter/site_config.sh

# 2. Install the module
bash config/perlmutter/install.sh

# 3. Start a new shell, then:
module load lorrax
```

## Usage

### Get a GPU allocation

```bash
lxalloc            # 1 node / 4 GPUs / 2 hours
lxalloc 4          # 4 nodes / 16 GPUs / 2 hours
lxalloc 1 4:00:00  # 1 node / 4 GPUs / 4 hours
```

From a separate terminal (e.g. IDE), find the job ID and export it:
```bash
squeue -u $USER
export SLURM_JOBID=<jobid>
```

### Run LORRAX

```bash
# Preprocessing (single-GPU, all 3 steps)
lxpre cohsex.in 640
#   [1/3] python3 -m centroid.kmeans_cli 640 --seed 42  -> centroids_frac_640.txt
#   [2/3] python3 -m psp.get_dipole_mtxels -i cohsex.in -> dipole.h5
#   [3/3] python3 -m gw.kin_ion_io -i cohsex.in -> kin_ion.h5

# GW calculation (4 GPUs, default)
lxrun python3 -u -m gw.gw_jax -i cohsex.in

# Single-GPU GW
LORRAX_NGPU=1 lxrun python3 -u -m gw.gw_jax -i cohsex.in
```

### Batch submission

```bash
cd /path/to/run_dir   # must contain cohsex.in, WFN.h5, etc.
sbatch $LORRAX_ROOT/config/perlmutter/run_gw.slurm
```

## What `module load lorrax` provides

**Environment variables** (performance-optimal GPU defaults):

| Variable | Value | Purpose |
|---|---|---|
| `HDF5_USE_FILE_LOCKING` | `FALSE` | Lustre filesystem HDF5 compatibility |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Don't pre-grab a fixed XLA pool; let NCCL/cuSOLVERMp share VRAM |
| `XLA_PYTHON_CLIENT_ALLOCATOR` | `cuda_async` | cudaMallocAsync mempool (shared with NCCL) instead of XLA's BFC — **not** `platform`, see below |
| `JAX_COMPILATION_CACHE_DIR` | `$SCRATCH/.jax_cache` | Persistent XLA PTX cache across JAX processes |

> **`platform` is NOT cudaMallocAsync.**  The three values name three
> different allocators in the CUDA plugin, and this table used to conflate
> two of them:
>
> | value | what it actually is | `memory_stats()` |
> |---|---|---|
> | unset / `default` / `bfc` | XLA's BFC pool ("Using BFC allocator.") | fully populated |
> | `platform` | plain `cudaMalloc` ("Using platform allocator.") | `bytes_limit=0`, `peak_bytes_in_use=0` |
> | `cuda_async` | `cudaMallocAsync` (a separate `CudaAsyncAllocator`) | keeps `peak_bytes_in_use` |
>
> Measured on 8× Quadro RTX 5000 across 2 nodes, jobs 7882442/7882447/7882468
> (each cell run twice, rep 2 in reverse order).  `cuda_async` was the best
> of the three — 0.19 GB overhead, 9.20 GB largest creatable cuFFT plan.
> `platform` gives good headroom but blinds every memory report in the
> codebase, because `gw_init`'s high-water report, `gw_output`'s XLA-pool
> banner and `runtime/aot_memory` all read `memory_stats()`.
>
> **`TF_GPU_ALLOCATOR` is a TensorFlow variable and is inert for JAX.**  A
> cell setting only `TF_GPU_ALLOCATOR=cuda_malloc_async` was byte-identical
> to the unset cell on every metric — including an 11.805 GB BFC pool that
> the real `cuda_async` allocator never has (job 7882442).  It was removed
> from this table rather than corrected; do not add it back.
>
> On sm_75 (Frontera `rtx`) `cuda_async` additionally needs the
> command-buffer restriction in `config/frontera/ffi_env.sh` — that file is
> the one place that sets both together.  `runtime.set_default_env()`
> therefore leaves `XLA_PYTHON_CLIENT_ALLOCATOR` unset (= BFC) and only
> pins `XLA_PYTHON_CLIENT_PREALLOCATE=false`, which is a correctness knob
> here: LORRAX's FFI handlers allocate OUTSIDE the XLA allocator.

> **Why not `MEM_FRACTION=0.95`?**  We tried it.  Pre-allocating 95 % of A100
> VRAM into XLA's BFC pool leaves NCCL only ~2 GB for its staging buffers,
> and cuSOLVERMp's `syevd` surfaces that as `NCCL error 1 unhandled cuda
> error` → `cusolverMpSyevd: status=7`.  Switching to the cudaMallocAsync
> allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`) lets XLA and NCCL
> share one pool.  `MEM_FRACTION` is honoured only by BFC.

**Shell functions:**

| Function | Purpose |
|---|---|
| `lxalloc [N] [time]` | Interactive GPU allocation (N nodes, default 1) |
| `lxrun <cmd>` | Run `<cmd>` on `LORRAX_NGPU` ranks (default 4) inside the Shifter container |
| `lxshell` | Single-rank pty shell inside the container — iterate without paying shifter bring-up per invocation |
| `lxpre <input> <N>` | Run all 3 preprocessing steps (single-GPU each) |

**Exported variables** for scripting:
`LORRAX_ROOT`, `LORRAX_SRC`, `LORRAX_SITE`, `LORRAX_IMAGE`, `LORRAX_SHIFTER`,
`LORRAX_FFI_NVHPC_HOST`, `LORRAX_FFI_PHDF5_HOST`, `LORRAX_FFI_SLATE_HOST`,
`LORRAX_SLATE_INSTALL_DIR`, `JAX_COMPILATION_CACHE_DIR`.

### Unified Cray MPICH stack

The module defaults all workloads (SLATE, cuSOLVERMp, phdf5) to a single
stack: **Cray MPICH + one GPU per rank**.  Every `lxrun` invocation
does:

```
srun --mpi=cray_shasta --gres=gpu:$NGPU -N 1 -n $NGPU \
    select_gpu.sh   \    # CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    shifter --module=gpu,mpich --image=... --volume=... --env=... \
    in_container.sh \    # re-assert MPICH_GPU_SUPPORT_ENABLED=1
    "$@"
```

Each rank sees exactly one GPU (JAX callers must use
`jax.distributed.initialize(local_device_ids=[0])` when world > 1 —
the common tests auto-detect this via `CUDA_VISIBLE_DEVICES`).

**Bind-mounts** (pre-staged via `src/ffi/*/scripts/stage_*.sh`, one-time):

| Host path (override)                | Container mount   | Contents |
|-------------------------------------|-------------------|----------|
| `$LORRAX_FFI_NVHPC_DIR` (default `$HOME/software/lorrax_nvhpc`)              | `/lorrax_nvhpc`   | NVHPC 25.5 subset: `libcusolverMp.so.0`, `libcal` |
| `$LORRAX_FFI_PHDF5_DIR` (default `$HOME/software/lorrax_phdf5_cray_1.14.3.7/stage`) | `/lorrax_phdf5`   | Cray HDF5 **1.14.3.7** (SOVERSION 310; `libmpi_gnu_*.so.12` shims). Exactly one HDF5 — both FFI legs bind this SOVERSION, enforced by GATE 7 (`src/ffi/cpp/gate_one_hdf5.sh`). The old `lorrax_phdf5_cray/stage` is 1.12 and is what the host leg could not load in-container (CLAIMS 89). |
| `$LORRAX_FFI_SLATE_DIR` (default `$HOME/software/lorrax_slate_cray/stage`)   | `/lorrax_slate`   | Cray libsci + `libmpi_gtl_cuda.so.0` + xpmem + lustreapi |

Container-side `LD_LIBRARY_PATH` (in order):
```
$LORRAX_SLATE_INSTALL_DIR/lib64 : /lorrax_slate/lib : /lorrax_phdf5/lib :
/lorrax_nvhpc/<nvhpc-subpath>/lib64 : /opt/udiImage/modules/mpich :
/opt/udiImage/modules/mpich/dep [: darshan]
```

The `nvhpc-subpath`, mpich container dir, and optional Darshan dir are
cluster-specific (`LORRAX_NVHPC_SUBPATH`, `LORRAX_MPICH_CONTAINER_DIR`,
`LORRAX_DARSHAN_LIB_DIR` in `site_config.sh`).

**`LORRAX_MPI_TYPE`** override:

```bash
LORRAX_MPI_TYPE=cray_shasta   # default — Cray MPICH PMI (fastest on Perlmutter)
LORRAX_MPI_TYPE=none          # disable --mpi flag (single-rank code)
LORRAX_MPI_TYPE=pmix          # OpenMPI launch protocol — the correct choice on OpenMPI clusters
                              #   (not the default because Cray MPICH is faster on Perlmutter;
                              #    has hung some non-FFI workloads, so set it deliberately)
```

### Per-invocation cost & fast-iteration tips

Measured breakdown of a single `lxrun python3 ...` call:

| Phase | Time |
|---|---|
| srun step creation | 2–5 s |
| Shifter namespace bring-up | ~5 s |
| `import jax` + `jax.devices()` + first GPU tensor | ~1.2 s |
| `jax.distributed.initialize` handshake (multi-rank only) | 3–5 s |

Single-rank: ~7 s end-to-end. Multi-rank: ~10–15 s. The JAX cold start
itself is *not* the bottleneck — everything above it is.

- **`lxshell`**: drop into an interactive container shell, then run
  `python3 tests/bench/slate_batched_test.py`, `python3 tests/bench/cusolvermp_eigh_test.py`,
  etc. back-to-back. Saves the ~5 s shifter bring-up per invocation.
  (Python still cold-starts each call inside the shell — the real 100×
  win is keeping one Python REPL alive.)
- **`JAX_COMPILATION_CACHE_DIR`** defaults to `$SCRATCH/.jax_cache`.
  Amortises XLA PTX compile across JAX processes.
- For multi-rank MPI runs, `lxrun` is still required — shifter's
  MPI integration needs `srun` on the outside (per upstream Shifter
  SLURM integration docs).

## Multiple parallel checkouts

To run several LORRAX checkouts side-by-side, install each with a distinct
module name:

```bash
LORRAX_MODULE_NAME=<module-name> bash $LORRAX_ROOT/config/perlmutter/install.sh
```

`family("lorrax")` in the modulefile makes variants mutually exclusive
within a single shell: loading one variant auto-swaps the other out.
Across separate shells each variant is fully independent (own
`LORRAX_ROOT`, own `lxrun`, own `SLURM_JOBID` from `lxalloc`).

## Shared group installation

Install once to a shared path so all group members can `module load lorrax`:

```bash
# 1. Edit config/perlmutter/site_config.sh:
LORRAX_INSTALL_ROOT="/global/cfs/cdirs/m2651/software/lorrax"
LORRAX_SITE_PACKAGES="/global/cfs/cdirs/m2651/software/lorrax_site"
LORRAX_MODULEFILE_DIR="/global/common/software/m2651/modulefiles"

# 2. Run install.sh
bash config/perlmutter/install.sh

# 3. Each user adds one line to ~/.bashrc:
module use /global/common/software/m2651/modulefiles
```

## Building the site-packages directory

The NVIDIA JAX container ships JAX, NumPy, and CUDA but not h5py, scipy,
or matplotlib. Build the supplemental site-packages directory:

```bash
pip install --target=/path/to/lorrax_site \
    h5py scipy matplotlib contourpy cycler fonttools \
    kiwisolver packaging pillow pyparsing python-dateutil six
```

## File layout

```
config/
├── README.md                  # this file
├── modulefiles/
│   └── lorrax/
│       └── 0.1.0.lua          # Lmod modulefile template
└── perlmutter/
    ├── site_config.sh         # site-specific paths (edit this)
    ├── install.sh             # patches + installs the module
    └── run_gw.slurm           # batch job template
```

## Porting to other clusters

The Lua modulefile at `config/modulefiles/lorrax/0.1.0.lua` is
cluster-agnostic: every cluster-specific value is an `@FOO@` placeholder
patched at install time by `site_config.sh`. See that file for the full
list; headline knobs:

| `site_config.sh` var | Purpose |
|---|---|
| `LORRAX_SLURM_{ACCOUNT,QOS,CONSTRAINT}` | `lxalloc` SLURM defaults |
| `LORRAX_GPUS_PER_NODE` | GPU count per node (Perlmutter: 4) |
| `LORRAX_SHIFTER_MODULES` | Shifter `--module=` list (Perlmutter: `gpu,mpich`) |
| `LORRAX_MPI_TYPE_DEFAULT` | `srun --mpi=` (Perlmutter: `cray_shasta`) |
| `LORRAX_NVHPC_SUBPATH` | NVHPC lib subdir under `/lorrax_nvhpc` |
| `LORRAX_MPICH_CONTAINER_DIR` | Where Shifter bind-mounts MPICH libs |
| `LORRAX_DARSHAN_LIB_DIR` | Optional I/O profiler lib dir (empty to skip) |
| `LORRAX_FFI_{NVHPC,PHDF5,SLATE}_DIR_DEFAULT` | Default host stage-dir roots |
| `LORRAX_SLATE_INSTALL_DIR_DEFAULT` | Host SLATE install prefix |

To port:

1. `cp -r config/perlmutter config/<cluster>/`
2. Edit `config/<cluster>/site_config.sh` with the new cluster's values.
3. `bash config/<cluster>/install.sh`.

For non-Shifter clusters (Apptainer, Singularity, bare venv) the Lua
modulefile's `shifter_args` composition needs adaptation — swap the
`shifter` invocation in the shell functions for the equivalent runtime
wrapper. Everything else (SLURM defaults, LD_LIBRARY_PATH composition,
`select_gpu.sh`, `in_container.sh`) is portable.

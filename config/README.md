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
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.95` | Pre-allocate 95% of GPU into XLA's memory pool |
| `TF_GPU_ALLOCATOR` | `cuda_malloc_async` | CUDA 12 async allocator (no pipeline stalls) |

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

| Host path (override) | Container mount | Contents |
|---|---|---|
| `$LORRAX_FFI_NVHPC_DIR` *(default `/pscratch/sd/${U:0:1}/${U}/lorrax_nvhpc`)* | `/lorrax_nvhpc` | NVHPC 25.5 subset: `libcusolverMp.so.0`, `libcal` |
| `$LORRAX_FFI_PHDF5_DIR` *(default `/pscratch/sd/${U:0:1}/${U}/lorrax_phdf5_cray/stage`)* | `/lorrax_phdf5` | Cray HDF5 1.12 (libmpi_gnu_*.so.12) |
| `$LORRAX_FFI_SLATE_DIR` *(default `/pscratch/sd/${U:0:1}/${U}/lorrax_slate_cray/stage`)* | `/lorrax_slate` | Cray libsci + `libmpi_gtl_cuda.so.0` + xpmem + lustreapi |

Container-side `LD_LIBRARY_PATH` (in order):
```
$LORRAX_SLATE_INSTALL_DIR/lib64 : /lorrax_slate/lib : /lorrax_phdf5/lib :
/lorrax_nvhpc/.../lib64 : /opt/udiImage/modules/mpich : /opt/udiImage/modules/mpich/dep
```

**`LORRAX_MPI_TYPE`** override:

```bash
LORRAX_MPI_TYPE=cray_shasta   # default — Cray MPICH PMI
LORRAX_MPI_TYPE=none          # disable --mpi flag (single-rank code)
LORRAX_MPI_TYPE=pmix          # legacy OpenMPI path (not wired up)
```

### Fast-iteration tips

- **`lxshell`**: drop into an interactive container shell, then run
  `python3 -m common.slate_batched_test`, `python3 -m common.cusolvermp_eigh_test`,
  etc. back-to-back.  Saves the ~5 s shifter bring-up per invocation.
- **`JAX_COMPILATION_CACHE_DIR`** is set to `$SCRATCH/.jax_cache` by
  default.  Amortises XLA PTX compile across JAX processes
  (doesn't cut the ~15-25 s CUDA backend init itself).
- For multi-rank MPI runs, `lxrun` is still required — shifter's
  MPI integration needs `srun` on the outside (per upstream Shifter
  SLURM integration docs).

## Multiple parallel checkouts (A/B/C agent sessions)

To run several LORRAX checkouts side-by-side, install each with a distinct
module name:

```bash
LORRAX_MODULE_NAME=lorrax_A bash /path/to/lorrax_A/config/perlmutter/install.sh
LORRAX_MODULE_NAME=lorrax_B bash /path/to/lorrax_B/config/perlmutter/install.sh
LORRAX_MODULE_NAME=lorrax_C bash /path/to/lorrax_C/config/perlmutter/install.sh
```

`family("lorrax")` in the modulefile makes variants mutually exclusive
within a single shell: `module load lorrax_B` auto-swaps `lorrax_A` out.
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

1. Create `config/<cluster>/site_config.sh` with that cluster's paths
2. Create `config/<cluster>/install.sh` (the Perlmutter one is reusable
   for any Lmod + Shifter site)
3. For non-Shifter clusters (Apptainer/Singularity), the modulefile's
   `shifter_base` variable needs adaptation

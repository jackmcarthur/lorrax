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
| `lxalloc [N] [time]` | Get an interactive GPU allocation (N nodes, default 1) |
| `lxrun <cmd>` | Run `<cmd>` inside the LORRAX Shifter container with srun |
| `lxpre <input> <N>` | Run all 3 preprocessing steps (single-GPU each) |

**Exported variables** for scripting:
`LORRAX_ROOT`, `LORRAX_SRC`, `LORRAX_SITE`, `LORRAX_IMAGE`, `LORRAX_SHIFTER`,
`LORRAX_FFI_NVHPC_HOST`, `LORRAX_FFI_PHDF5_HOST`.

### FFI staged-deps bind-mounts inside `lxrun`

`shifter_base` bind-mounts the staged NVHPC (cuSOLVERMp, libcal) and
parallel-HDF5 (MPI-IO, linked against the container's HPC-X OpenMPI)
trees into the container at stable paths, and adds
`LD_LIBRARY_PATH` so the loader finds them.  End users don't have to
set anything for this — it happens automatically on `module load`.

| Host path (override) | Container mount | Contents |
|---|---|---|
| `$LORRAX_FFI_NVHPC_DIR` *(default `/pscratch/sd/${U:0:1}/${U}/lorrax_nvhpc`)* | `/lorrax_nvhpc` | NVHPC 25.5 subset: `libcusolverMp.so.0`, `libcal`, cuSOLVERMp + NCCL headers |
| `$LORRAX_FFI_PHDF5_DIR` *(default `/pscratch/sd/${U:0:1}/${U}/lorrax_phdf5_openmpi/stage`)* | `/lorrax_phdf5` | conda-forge HDF5 1.14 built against the container's HPC-X OpenMPI (libmpi.so.40) |

Container-side `LD_LIBRARY_PATH` set to:
```
/lorrax_phdf5/lib : /lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64 : /opt/hpcx/ompi/lib
```

**Staging**: run the one-time scripts under `src/ffi/`:
```bash
src/ffi/cusolvermp/scripts/stage_nvhpc.sh      # ~100 MB
src/ffi/phdf5/scripts/stage_openmpi.sh         #  ~40 MB
```

**Running FFI workloads** (`use_ffi_io=true`, `cusolvermp` eigh, etc.) —
opt in with `LORRAX_MPI_TYPE=pmix` so `srun` bootstraps PMIx for the
container's OpenMPI:

```bash
LORRAX_MPI_TYPE=pmix LORRAX_NGPU=4 lxrun python3 -u -m common.phdf5_write_test
LORRAX_MPI_TYPE=pmix LORRAX_NGPU=4 lxrun env CUSOLVERMP_FORCE_NCCL=1 \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    python3 -u -m common.cusolvermp_eigh_test --grid 2 2
```

`--mpi=pmix` is **off by default** because unconditionally enabling it
was observed to hang some non-FFI workloads after the first NCCL
collective.  Setting `LORRAX_MPI_TYPE=none` explicitly disables it.

If you need a different staging path or a non-default container image,
set `LORRAX_FFI_NVHPC_DIR` / `LORRAX_FFI_PHDF5_DIR` *before* loading
the module (they're read at module-load time, not per-call).

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

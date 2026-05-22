# Porting LORRAX's FFI subpackages to a new cluster

Covers `ffi.cusolvermp`, `ffi.phdf5`, and `ffi.slate`. All three link
into one `liblorrax_ffi.so` and share a build system
(`src/ffi/common/cpp/CMakeLists.txt`).

## Hard requirements

| Requirement             | Minimum | Notes |
|-------------------------|---------|-------|
| NVIDIA GPU              | CC ≥ 7.0 | A100 / H100 tested. |
| CUDA toolkit            | 12.0    | 12.9 is what we link against. |
| NCCL                    | 2.18    | Ships with CUDA 12.4+; JAX bundles it. |
| JAX with `jax.ffi`      | 0.5     | Must match CUDA major. `nvcr.io/nvidia/jax:25.04-py3` is our container. |
| NVHPC SDK (cuSOLVERMp)  | 22.7    | 25.5 validated. Only the `libcusolverMp` + `libcal` subset is needed. |
| Parallel HDF5           | 1.12    | Either Cray HDF5 (MPICH ABI) or a MPI-linked conda-forge build. |
| SLATE (+ blaspp/lapackpp)| any     | Built from source against the target MPI + libsci/BLAS. |

MPI is required only for `ffi.phdf5` and `ffi.slate`; `ffi.cusolvermp`
bootstraps via JAX's KV store + NCCL, no MPI.

## Build system ([`common/cpp/CMakeLists.txt`](common/cpp/CMakeLists.txt))

Autodetection probes for each dep, overridable with CMake `-D...` or
env vars:

| Dep          | Auto-probe                                                   | Override                                                 |
|--------------|--------------------------------------------------------------|----------------------------------------------------------|
| cuSOLVERMp   | `$NVHPC_ROOT`, `$HPCSDK_ROOT`, `/opt/nvidia/hpc_sdk/…`, `/lorrax_nvhpc/…`, `/usr/local/cuda-12.X/…` | `-DNVHPC_ROOT=…` or `-DCUSOLVERMP_{INCLUDE,LIB}_DIR=…` |
| Parallel HDF5| `$HDF5_ROOT`, `$HDF5_DIR`, `/lorrax_phdf5`                   | `-DHDF5_ROOT=…`                                          |
| MPI          | `$LORRAX_MPI_INCLUDE_DIR` / `$LORRAX_MPICH_LIB_DIR`          | `-DLORRAX_MPI_INCLUDE_DIR=…`, `-DLORRAX_MPICH_LIB_DIR=…` |
| SLATE        | `/global/homes/<u>/software/slate/install`                   | `-DLORRAX_SLATE_INSTALL_DIR=…` or env                    |
| NCCL header  | `/usr/include/nccl.h`, then `$NVHPC_ROOT/comm_libs/.../nccl` | `-DNCCL_INCLUDE=…`                                       |

Build: `bash src/ffi/common/cpp/build.sh` (calls CMake + ninja). Rebuild
from scratch with `--fresh`.

## Staging vendor deps (one-time per cluster)

Many clusters put system libs under `/opt/...`, which containers can't
bind-mount freely. We copy the minimal subsets into `$SCRATCH` (or
equivalent bindable location) via scripts under `src/ffi/*/scripts/`:

| Script                             | Produces                                     | Bind-mounted to |
|------------------------------------|----------------------------------------------|-----------------|
| [`cusolvermp/scripts/stage_nvhpc.sh`](cusolvermp/scripts/stage_nvhpc.sh) | `libcusolverMp.so.0`, `libcal.so.0`, cuSOLVERMp+NCCL headers (~80 MB) | `/lorrax_nvhpc` |
| [`phdf5/scripts/stage_cray.sh`](phdf5/scripts/stage_cray.sh)         | Cray HDF5 1.12 + MPICH-ABI shim | `/lorrax_phdf5` |
| [`phdf5/scripts/stage_openmpi.sh`](phdf5/scripts/stage_openmpi.sh)   | conda-forge HDF5 (OpenMPI)      | `/lorrax_phdf5` |
| [`slate/scripts/stage_cray.sh`](slate/scripts/stage_cray.sh)         | Cray libsci + `libmpi_gtl_cuda.so.0` + xpmem + lustreapi | `/lorrax_slate` |

SLATE itself is built from source; `stage_cray.sh` only copies the
runtime libs SLATE links against.

## Runtime

The LORRAX Lmod module wires the full `srun + select_gpu + shifter
+ in_container` invocation. On a ported cluster, install the module
via `config/<cluster>/install.sh` (see [`config/README.md`](../../config/README.md)),
then `lxrun <cmd>` — no per-call env juggling.

What the module sets:

```
XLA_PYTHON_CLIENT_PREALLOCATE=false   # share VRAM with NCCL/cuSOLVERMp
XLA_PYTHON_CLIENT_ALLOCATOR=platform  # = cudaMallocAsync (via TF_GPU_ALLOCATOR)
TF_GPU_ALLOCATOR=cuda_malloc_async
HDF5_USE_FILE_LOCKING=FALSE           # Lustre compatibility
MPICH_GPU_SUPPORT_ENABLED=1           # GPU-Direct RDMA
LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0   # CUDA-12 shim; see §Slate-specific below
```

`CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` is set per-rank by
`select_gpu.sh` (invoked from `lxrun`). JAX callers must pass
`local_device_ids=[0]` to `jax.distributed.initialize()` when
`process_count > 1` — the sandbox tests auto-detect via the length
of `CUDA_VISIBLE_DEVICES`.

## Checklist for a new cluster

1. **NVHPC SDK**: `module spider nvhpc`. Run
   [`stage_nvhpc.sh`](cusolvermp/scripts/stage_nvhpc.sh) to copy the
   `libcusolverMp`+`libcal` subset into `$SCRATCH`.
2. **Parallel HDF5**: pick a stack.
   - Cray MPICH + cray-hdf5-parallel → [`stage_cray.sh`](phdf5/scripts/stage_cray.sh).
   - Anything else MPI → [`stage_openmpi.sh`](phdf5/scripts/stage_openmpi.sh)
     (edit the conda-forge URL to match your MPI).
3. **SLATE**: clone [icl-utk-edu/slate](https://github.com/icl-utk-edu/slate),
   build against the target MPI + BLAS, install under
   `$HOME/software/slate/install`. Stage Cray runtime libs via
   [`slate/scripts/stage_cray.sh`](slate/scripts/stage_cray.sh).
4. **Configure and install the module**: copy
   `config/perlmutter/` → `config/<cluster>/`, edit `site_config.sh`
   (especially `LORRAX_SLURM_{ACCOUNT,QOS,CONSTRAINT}`,
   `LORRAX_SHIFTER_MODULES`, `LORRAX_MPI_TYPE_DEFAULT`,
   `LORRAX_NVHPC_SUBPATH`, `LORRAX_MPICH_CONTAINER_DIR`), run
   `bash config/<cluster>/install.sh`.
5. **Build the .so**: `bash src/ffi/common/cpp/build.sh` (inside shifter
   via `src/ffi/common/cpp/run_shifter.sh`).
6. **Verify**:
   ```bash
   lxalloc
   lxrun python3 -u -m common.cusolvermp_eigh_test
   lxrun python3 -u -m common.slate_cholesky_trsm_test -n 256 --dtype c128
   lxrun python3 -u -m common.phdf5_multi_offset_test
   ```
   Expected: all PASS at machine precision (~1e-13 for C128 eigh).

For non-Shifter runtimes (Singularity/Apptainer): swap the
`shifter ...` invocation inside the module's shell functions for
`apptainer exec --nv --bind ... image.sif ...`. Everything else
(SLURM flags, `select_gpu.sh`, `in_container.sh`, LD_LIBRARY_PATH
composition) is runtime-agnostic.

### Container bind-mount paths (`LORRAX_CONTAINER_*_PATH`)

The staged libraries are bind-mounted at `/lorrax_nvhpc`, `/lorrax_phdf5`,
and `/lorrax_slate` inside the container by default.  These paths are baked
into `liblorrax_ffi.so`'s `RPATH` at build time; changing them requires a
rebuild.  On clusters where those paths conflict with existing directory
layout (Apptainer images, Frontier, etc.), override before `cmake`:

```bash
export LORRAX_CONTAINER_NVHPC_PATH=/opt/lorrax/nvhpc
export LORRAX_CONTAINER_PHDF5_PATH=/opt/lorrax/phdf5
export LORRAX_CONTAINER_SLATE_PATH=/opt/lorrax/slate
cmake -B build src/ffi/common/cpp   # picks up env vars automatically
```

The same vars control `--volume=<host>:<ct>` in both the modulefile and
`run_shifter.sh`, so host-to-container mapping and RPATH stay consistent
without extra flags.

### Multi-user shared allocations

The upstream `lorrax` module assumes one user per allocation.  For concurrent
multi-agent or multi-user use of a single shared allocation, load the
`lorrax_agent` sandbox overlay on top of the base module (see
[`config/README.md`](../../config/README.md#multi-user-shared-allocations)).
The overlay adds pool-aware `lxrun`, `lxattach`, `lxreap`, and an enhanced
`lxstatus` — none of which belong in the upstream module because they depend
on sandbox-local `lx_pool.py`.

## Cluster-specific: NERSC Perlmutter

Shifter forbids `--volume` sources outside `/pscratch` and a handful
of other paths, which is why every "stage" script copies to
`$SCRATCH` first. The `nvcr.io/nvidia/jax:25.04-py3` container does
**not** ship NVHPC SDK, Cray MPICH, Cray HDF5, or SLATE — all four
come in via bind-mount.

`lxrun` uses `--mpi=cray_shasta` (not `pmi2` or `pmix`) — both of
those silently give singleton `MPI_COMM_WORLD` with
`shifter --module=mpich`.

`libmpi_gtl_cuda.so.0` is the CUDA GPU-Direct RDMA transport for Cray
MPICH. Shifter's `--module=mpich` bind-mounts a copy built against
CUDA 11, needing `libcudart.so.11.0` not in our container.
`stage_cray.sh` for slate also copies the CUDA-12 version, and
`lxrun` `LD_PRELOAD`s it so the loader binds that one first.

## phdf5 stack choice

The unified default is **Cray MPICH** on Perlmutter (and any other
Cray site). Historical context: we ran on OpenMPI / HPC-X for the
first few months because Cray MPICH's collective-write path
(`ad_cray_write_coll.c:669`) OOMs at ≥ 1 GB/rank. The 2026-04-20 fix
was to flip the FFI's default writes to `H5FD_MPIO_INDEPENDENT` and
disable collective metadata ops, which bypasses the Cray collective
write driver entirely. Result at 4 GPU / 4.29 GB C128:
**3.79 GB/s Cray vs 3.06 GB/s OpenMPI**, and small-write latency
within noise. OpenMPI path is still viable — select via
`LORRAX_PHDF5_MPI_STACK=openmpi` (affects build-time and
`run_shifter.sh`); requires `--mpi=pmix` + container's HPC-X OpenMPI.

## phdf5 tuning knobs

Env vars read at `open_file` time:

| Var                              | Default      | Effect |
|----------------------------------|--------------|--------|
| `LORRAX_PHDF5_CB_WRITE`          | `enable`     | ROMIO collective buffering on writes. |
| `LORRAX_PHDF5_CB_BUFFER_SIZE`    | `67108864`   | Per-aggregator CB buffer size (bytes). |
| `LORRAX_PHDF5_CB_NODES`          | `world_size` | ROMIO aggregator count. |
| `LORRAX_PHDF5_CB_PER_NODE`       | _unset_      | Cray MPICH: aggregators/node (`cb_config_list=*:N`). |
| `LORRAX_PHDF5_STRIPE_COUNT`      | `16`         | Lustre `striping_factor` hint. |
| `LORRAX_PHDF5_STRIPE_SIZE`       | `4194304`    | Lustre `striping_unit` (bytes). |
| `LORRAX_PHDF5_ALIGN_MB`          | `4`          | `H5Pset_alignment` threshold (MiB). |
| `LORRAX_PHDF5_INDEPENDENT`       | `0`          | 1 → also force **reads** to independent (writes already are). |
| `LORRAX_PHDF5_COLLECTIVE_WRITES` | `0`          | 1 → re-enable collective writes. **Do not** on Cray. |
| `LORRAX_PHDF5_COLL_META`         | `0`          | 1 → re-enable collective metadata ops. |

Rule of thumb: bump `STRIPE_COUNT` to 32-64 for writes > 10 GB; drop
`CB_BUFFER_SIZE` to 8 MiB for writes < 100 MB. If the enclosing
directory has an explicit `lfs setstripe` layout, the `striping_*`
hints are no-ops (directory wins).

Baked-in DCPL: `H5D_FILL_TIME_NEVER` + `H5D_ALLOC_TIME_EARLY` +
`H5F_LIBVER_LATEST`.

## Gotchas

- **`Failed to parse ib device list`**: harmless libcal warning on
  Perlmutter; libcal probes InfiniBand transports it won't use
  (we route CAL through NCCL).
- **`NCCL error 1 unhandled cuda error` → `cusolverMpSyevd status=7`**:
  NCCL starved of VRAM. Check `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  is set (the module sets it; don't override with `true` + a fixed
  `MEM_FRACTION`).
- **`MPI_COMM_WORLD` size 1 inside `--module=mpich`**: wrong
  `--mpi=` flavour. Use `cray_shasta`.
- **CUDA driver vs toolkit mismatch**: `nvidia-smi`'s "CUDA Version"
  is the driver's maximum supported toolkit; must be ≥ what we
  linked (12.9 currently).
- **Multi-node NCCL**: needs cluster-specific NCCL env (e.g.
  `NCCL_NET_PLUGIN=ofi` on Perlmutter, `NCCL_IB_HCA=...` on IB
  fabrics). Not validated here.

## References

- [NVIDIA cuSOLVERMp](https://docs.nvidia.com/cuda/cusolvermp/)
- [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk)
- [JAX FFI](https://jax.readthedocs.io/en/latest/ffi.html)
- [NERSC parallel HDF5 tuning](https://docs.nersc.gov/performance/io/library/)
- [ROMIO hints](https://wordpress.cels.anl.gov/romio/2008/09/26/system-hints/)
- [NERSC Shifter mpich module](https://docs.nersc.gov/development/shifter/how-to-use/)

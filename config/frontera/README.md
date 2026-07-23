# LORRAX on TACC Frontera

Frontera-specific enablement for LORRAX. **Additive and self-contained** —
nothing here changes Perlmutter/Cray behaviour. The only edits outside this
directory are three backward-compatible option guards (all default to the
previous behaviour):

* `src/ffi/common/cpp/CMakeLists.txt` — `option(LORRAX_FFI_HAVE_CAL ON)`,
  `option(LORRAX_FFI_HAVE_PHDF5 ON)`.
* `src/ffi/cusolvermp/cpp/{ctx.h,context.cc}` — `#if LORRAX_FFI_HAVE_CAL`
  around the CAL comm path (cuSOLVERMp ≥ 0.7 is NCCL-native).
* `src/ffi/common/cpp/api.cc` — `#if LORRAX_FFI_HAVE_PHDF5` around the phdf5
  lifecycle entry points.
* `src/ffi/common/ffi_loader.py` — skips FFI handler / lifecycle symbols a
  partial build doesn't export.

Kept on the `frontera-ffi` branch so `main` stays pristine.

## Why Frontera is different

| | Perlmutter (upstream) | Frontera |
|---|---|---|
| Container runtime | Shifter | **apptainer** (`tacc-apptainer`, compute nodes only) |
| Base image | `nvcr.io/nvidia/jax` | `python:3.12-bookworm` + `uv`-built venv |
| OS glibc | 2.31+ | **2.17** (CentOS 7) → JAX must run in-container |
| GPU | A100, NVLink | **RTX 5000** (Turing sm_75, 16 GB, **no NVLink**, PCIe P2P intra-socket only) |
| Driver | current | **535.113.01** (CUDA 12.2) → **CUDA 12**, not 13 |
| cuSOLVERMp source | NVHPC SDK (+ CAL) | **pip** `nvidia-cusolvermp-cu12` 0.9 (**NCCL-native, no CAL**) |
| MPI | Cray MPICH | Intel MPI / MVAPICH2-X (host, hybrid-mounted); `module load phdf5` |

## Distributed eigh (cuSOLVERMp) — the built path

cuSOLVERMp bootstraps via JAX's KV-store + NCCL (no MPI), so distributed
`eigh` needs no MPI/IB — single-node 4-GPU aggregates 64 GB for matrices too
big for one card. NCCL runs over PCIe P2P (intra-socket) / host memory
(cross-socket); set `CUSOLVERMP_FORCE_NCCL=1`.

```bash
# On a compute node (apptainer is blocked on login nodes):
export LORRAX_SIF=$SCRATCH/lorrax_setup/py312.sif
EXEC="apptainer exec --bind /home1,/work2,/scratch1,/scratch2 $LORRAX_SIF"

$EXEC bash config/frontera/stage_ffi_deps.sh      # once: pip wheels + CUDA root
$EXEC bash config/frontera/build_ffi.sh --fresh   # build liblorrax_ffi.so
# 4 ranks x 1 GPU, 2x2 mesh:
srun -n 4 apptainer exec --nv --bind /home1,/work2,/scratch1,/scratch2 $LORRAX_SIF \
    bash -lc 'source config/frontera/ffi_env.sh; export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID;
              $LORRAX_VENV/bin/python -m common.cusolvermp_eigh_test --grid 2 2'
```
`$SCRATCH/lorrax_setup/ffi_build_test.sbatch` wraps all three as one job.

Build flags: `-DLORRAX_FFI_HAVE_CAL=OFF -DLORRAX_FFI_HAVE_PHDF5=OFF`,
`-DCMAKE_CUDA_ARCHITECTURES=75`, CUDA toolkit assembled from the venv's pip
`nvidia-*-cu12` packages (see `stage_ffi_deps.sh`).

## Deferred

* **phdf5** — sharded slab I/O. Frontera has `module load phdf5`; build with
  `-DLORRAX_FFI_HAVE_PHDF5=ON` + the host MPI hybrid-mounted into the
  container (TACC pattern: launch via `ibrun apptainer`, `FI_PROVIDER=tcp` on
  rtx due to the ConnectX-3/mlx4 fabric).
* **SLATE**, **cuBLASMp fused kernels beyond the linked lib**, **multi-node**
  (NCCL-over-IB on mlx4 likely falls back to TCP — benchmark first).

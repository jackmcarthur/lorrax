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

Originally staged on the `frontera-ffi` branch; long since merged — the
Frontera FFI stack (CUDA + host SLATE/ScaLAPACK + phdf5) is mainline and
production-certified (2026-07 campaign, scorecard AM-AU).

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

## Multi-process CPU runs: collectives on MPI

LORRAX's CPU collectives run on MPI, not on jax's default gloo — gloo's
`reduce-scatter` silently corrupts ~5% of executions here, and mpi is also
1.4-8.2x faster on the collective-bound stages. That needs an MPIwrapper we
build:

```bash
export LORRAX_ROOT=$PWD
config/frontera/build_mpiwrapper.sh --fresh    # LOGIN NODE (needs gfortran)

# then, in the job's container env:
export JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi
export MPITRAMPOLINE_LIB=$WORK/lorrax_mpiwrapper/install/lib64/libmpiwrapper.so
export LORRAX_MPI_FORCE_THREAD_MAIN=1
export LORRAX_MPI_FINALIZE_FIX=skip_atexit
PYTHONPATH=$WORK/lorrax_env_mpi_overlay/site:$PYTHONPATH
```

All four are load-bearing; omitting the last one makes every successful run
exit rc=1. Full rationale and the rest of the env block:
**`docs/dev/mpi_collectives.md`**.

## Cold start: the node-local run-time bundle

A first run on a fresh node reads the 5.6 GB venv off Lustre through the
container, one mmap page fault at a time. Measured on fresh Frontera compute
nodes (job 7882055), `gw.kin_ion_io` needed **44–88 s** to resolve its import
graph — of which **34–73 s was `jax.devices()` alone**, dlopening a CUDA stack
a CPU run cannot use. Two changes take that to **4.6 s**:

1. `runtime.skip_gpu_plugin_discovery()` — in-tree, automatic, no packaging
   change. 88 s → 11 s on its own.
2. the bundle below — 11 s → 4.6 s.

```bash
# ONCE per venv/source revision, inside the SIF (byte-compiling needs the
# container's python 3.12).  Writes $SCRATCH/lorrax_bundle/lorrax_cpu_bundle.tar
apptainer exec --bind /home1,/work2,/scratch1,/scratch2 $LORRAX_SIF \
    config/frontera/build_cpu_runtime_bundle.sh

# in the job's CONTAINER-SIDE runner, before python:
export LORRAX_BUNDLE=$SCRATCH/lorrax_bundle/lorrax_cpu_bundle.tar
. $LORRAX_ROOT/config/frontera/stage_runtime.sh    # SOURCE it, don't exec it
export PYTHONPATH=$LORRAX_OVERLAY_DIR:$LORRAX_SRC_DIR
$LORRAX_PY -u -m gw.kin_ion_io ...
```

The bundle is venv + MPI overlay + `src/` minus what a CPU run cannot use
(`nvidia/*`, `jax_plugins/`, `jax_cuda12_plugin/`, and the
`jax_cuda12_pjrt-*.dist-info` that advertises the plugin entry point),
byte-compiled: 5.6 GB → 769 MB, striped 12-wide so a whole job can read it at
once. `stage_runtime.sh` unrolls it onto `/tmp` — a real 144 GB local XFS SSD
on CLX, writable inside apptainer — once per node under `flock`, in 1.5–2.2 s.
`LORRAX_STAGE=0` disables it; a missing bundle falls back to the Lustre venv
and **says so on rank 0**.

Nothing here is a patched dependency: every file is a byte-for-byte copy of
what uv installed, and the GPU venv is untouched. Full measurement and the
falsified instruments: **`docs/dev/cold_start.md`**.

## Built since (formerly "Deferred" — updated 2026-07-28)

* **phdf5** — sharded slab I/O is BUILT and is a production write path
  (`slab_io` router: `PHDF5_FFI` → `phdf5_host` → allgather).
  `build_ffi_host.sh` (this dir) builds it with
  `-DLORRAX_FFI_HAVE_PHDF5=ON` against the host Intel MPI hybrid-mounted
  into the container; `ffi_env.sh` stages the runtime (PMI2 lib,
  `FI_PROVIDER_PATH`, the `LORRAX_MPI_PROVIDER` dial, UCX setdefaults).
* **SLATE / ScaLAPACK** — built by `build_ffi_host.sh` into
  `liblorrax_ffi_host.so`; ScaLAPACK `pzheevd`
  (`ScalapackEighHostFfi`) is the permanent CPU distributed eigh behind
  the `ffi/linalg` facade.
* Still open: cuBLASMp fused kernels beyond the linked lib (consumer-less
  package, owner-ledgered for deletion pending one rtx gate), and
  cross-node **GPU** phdf5 bring-up on the rtx mlx4 fabric
  (HANDOFF_2026-07-28 open ledger).

## Transport — do NOT seed `FI_PROVIDER` (post-AU doctrine)

On Frontera CLX leave `FI_PROVIDER` **unset** (`LORRAX_MPI_PROVIDER=auto`,
the `ffi_env.sh` default): Intel MPI then auto-selects the native `mlx`
(UCX/RDMA) provider — measured 1.07 µs / 11.4 GB/s vs the old
`FI_PROVIDER=tcp` seed's 10.9 µs / 2.15 GB/s, which was the root cause of
the 30-minute pzheevd era (n=2448 P=144: ~12 s/q under tcp vs
0.5–0.9 s/q under mlx; scorecard AP, seed deleted by AU).
`LORRAX_MPI_PROVIDER=tcp` remains ONLY as the rtx/mlx4 (ConnectX-3)
escape hatch.  Trust the `I_MPI_DEBUG≥4` `libfabric provider:` banner,
never `fi_info` (it false-negatives on mlx).

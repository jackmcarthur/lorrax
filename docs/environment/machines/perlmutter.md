# Perlmutter (NERSC)

The GPU reference platform: A100 40/80 GB nodes, bare-host CUDA 13.2, JAX and
JAXLIB 0.9.1. The default runtime is the module
([runtime defaults](../../installation/index.md#defaults)); a development
clone builds its own FFI pair ([install page](../../installation/perlmutter.md)). Porting knobs are in [`config/README.md`](../../../config/README.md);
JAX and the GPU memory pool are in the [overview](../overview.md#gpu-pool).

**Certified scope.** GPU: 1–4 nodes × 4 A100 at P=4 and P=16. CPU
`impl=mpi` on Milan (Cray MPICH 9.0.1): collective exactness at P=4 (one and
two nodes) and P=16 (four nodes), and a P=4 GN-PPM GW run matching its
reference Σ cell for cell. Larger CPU runs are not certified.

## 1. Entry point: `srun` {#1-entry-point}

A step runs on a compute node through `srun` inside an allocation, with
`src/ffi/cpp/select_gpu.sh` as the rank wrapper. From a clone
([install page](../../installation/perlmutter.md)):

```bash
source config/perlmutter/gpu_env.sh   # once per shell
srun --jobid=$JOBID -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh \
  .venv/bin/python -u -m gw.gw_jax -i cohsex.in
```

With the module loaded the interpreter and the FFI pair are the module's and
`gpu_env.sh` is not needed
([Perlmutter module](../../installation/perlmutter-module.md#using-the-module)).
`runtime.source_closure` refuses a run whose imported runtime is not the
checkout `LORRAX_CHECKOUT` names ([env vars](../../dev/env_vars.md)).

### Required GPU task geometry {#required-gpu-task-geometry}

One rank per GPU: `-N n -n 4n --gpus-per-node=4`. `-n 1 --gpus-per-node=4`
is a single process over four devices, not P=4 evidence. `src/ffi/cpp/select_gpu.sh`
pins each rank's GPU through `CUDA_VISIBLE_DEVICES`; the runtime passes
`local_device_ids` accordingly, so each rank sees `jax.local_devices() ==
[cuda:0]` and `len(jax.devices())` equals the rank count.

A job uses one GPU model and memory class. A multi-node allocation requests
`-C 'gpu&hbm40g'` or `-C 'gpu&hbm80g'`; the runtime refuses a mixed
40/80 GB set before building the mesh, because independently compiled XLA
programs on mixed nodes can disagree on collective exchange order.

### Network transport at startup {#network-transport-at-startup}

`runtime.network_env` decides the NCCL transport before JAX or NCCL starts,
from the step's node count (`SLURM_STEP_NUM_NODES` or the expanded
`SLURM_STEP_NODELIST`, never the allocation's).

| placement | result |
|---|---|
| one node | no network settings |
| several nodes, CUDA, `SLURM_NETWORK` contains `no_vni` | the site OFI/CXI profile: `NCCL_NET_PLUGIN` = the absolute path of `nccl/2.29.2-cu13`'s plugin, plus that module's settings including `FI_CXI_RDZV_THRESHOLD=0`; a value the caller set is kept |
| several nodes, CUDA, no `no_vni` | **refuses**, naming the relaunch: without a VNI, NCCL falls back to TCP sockets (about 12× slower) and MPI-IO + NCCL initialization fails |
| `NCCL_NET=Socket` on several nodes | **refuses** |
| explicit `NCCL_NET` or `NCCL_NET_PLUGIN` | the caller's configuration, unchanged |

A multi-node GPU `srun` passes `--network=no_vni` (or exports
`SLURM_NETWORK=no_vni`). The setting acts at step creation,
so setting it from Python is too late. `FI_CXI_RDZV_THRESHOLD=0` prevents a
cross-node send/recv deadlock (XLA `all_to_all` / `collective_permute`) in
which libfabric's NCCL proxy thread synchronizes the device while the NCCL
kernel waits on that thread.

## 2. The `lorrax_A` module and the FFI bundle

The module is a descriptor: it sets `LORRAX_ROOT`, `PYTHONPATH` (a
`releases/source-<rev>` snapshot), `LORRAX_FFI_SO` and `LORRAX_FFI_HOST_SO`
(the two legs of one sealed bundle), the CUDA and vendor library paths,
`JAX_PLATFORMS=cuda,cpu` and `JAX_ENABLE_X64=1`. It sets no allocator,
compile-cache, HDF5 or profiling policy; the runtime owns those.

| bundle leg | libraries (private closure in the bundle's `lib/`) | serves |
|---|---|---|
| CUDA `liblorrax_ffi.so` | cuSOLVERMp, cuBLASMp (private); cuFFT, NVRTC, NCCL, parallel HDF5, Cray MPICH | distributed eigh/Cholesky/LU (`cusolvermp`), distributed GEMM (`cublasmp`), the mathdx k-convolution, slab I/O. It carries no SLATE handler (device SLATE is not built), so `slate` on a CUDA mesh refuses at resolve |
| host `liblorrax_ffi_host.so` | SLATE, BLAS++, LAPACK++ (private); ScaLAPACK and CBLAS (LibSci), FFTW (dlopened), parallel HDF5, Cray MPICH | the SLATE handlers on a CPU mesh (eigh, potrf, trsm, batched potrf and trsm), ScaLAPACK eigh and LU, host GEMM and FFT, slab I/O |

The bundle manifest (`lorrax_ffi_bundle.json`) hashes every byte; the loader
refuses a library whose handler ABI differs from the source's and announces
an unsealed library as `LEGACY-UNSEALED`. The mathdx k-convolution also needs
the `nvidia-mathdx` wheel in the venv (`GATE mathdx-headers` otherwise).
Cray MPICH GPU support is off (`MPICH_GPU_SUPPORT_ENABLED=0`); cuSOLVERMp and
cuBLASMp communicate through NCCL. The site's default `craype-accel-nvidia80`
exports `1`, and then `MPI_Init` in either leg aborts ("GTL library is not
linked"). The module sets `0`; a clone's GPU steps source
`config/perlmutter/gpu_env.sh`, which sets it. Each leg carries the pinned HDF5 and MPICH
lib dirs on its RPATH, because `/opt/cray/pe/lib64` points
`libhdf5_parallel_gnu.so.310` at the site-default HDF5, which links the
site-default MPI. The legs build with `config/perlmutter/build_ffi_host.sh`
and `config/perlmutter/build_ffi_cuda.sh`, and seal with
`src/ffi/cpp/stage/seal_bundle.py` ([building_ffi.md](../../building_ffi.md#seal-the-deployable-pair)).
The module's venv, the build and seal, the acceptance (Gate 10 and a P4
run) and the publication are `config/perlmutter/module/`
([Perlmutter module](../../installation/perlmutter-module.md)).

## 3. CPU multi-process runs (Milan)

CPU runs use the same parallel-HDF5 transport as GPU runs and require the
host FFI leg. Build the MPI ABI adapter once on a CPU compute node (pinned,
unmodified upstream MPIwrapper against the versioned Cray wrappers; a failed
rebuild cannot replace the active release):

```bash
config/perlmutter/build_mpiwrapper.sh --fresh
```

Every multi-process CPU step sources `config/perlmutter/cpu_mpi_env.sh` in the
rank shell before Python:

```bash
export LORRAX_CHECKOUT=/path/to/checkout
srun --jobid=$JOBID -N 2 -n 4 -c 16 bash -c '
  set -euo pipefail
  export OMP_NUM_THREADS=14
  . "$LORRAX_CHECKOUT/config/perlmutter/cpu_mpi_env.sh"
  python3 -u -m gw.gw_jax -i gw.in'
```

`cpu_mpi_env.sh` refuses `JAX_PLATFORMS` other than `cpu` and sets:

| setting | why |
|---|---|
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` | gloo corrupts `psum_scatter` silently ([transports](../transports.md)) |
| `MPITRAMPOLINE_LIB` = the MPIwrapper release | the ABI adapter JAX's bundled MPItrampoline loads |
| `LD_PRELOAD=/opt/cray/pe/lib64/libpmi.so.0` | loads Cray PMI before `jax.distributed` starts threads; without it `PMI2_Init` segfaults (`libpmi2.so.0` also segfaults) |
| `MPICH_ASYNC_PROGRESS=1` | promotes XLA's FUNNELED request to `MPI_THREAD_MULTIPLE`, required when XLA threads and native MPI I/O or linear algebra coexist; the native FFI aborts the MPI world on a lower grant |
| `MPICH_GPU_SUPPORT_ENABLED=0` | CPU steps |

**Threads per rank.** `srun -c` sets one affinity mask per rank. The mask is shared by XLA's CPU worker pool (not capped by `OMP_NUM_THREADS`),
LibSci/SLATE OpenMP teams (capped by `OMP_NUM_THREADS` and the handler dials
the startup report prints), the MPICH progress thread and Python's I/O
threads. LORRAX binds none of them to a private CPU subset, so set
`OMP_NUM_THREADS` below `-c` (`14` with `-c16`) to leave CPUs for the XLA
pool, the progress thread and I/O.

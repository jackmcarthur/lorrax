# Perlmutter (NERSC)

This page states the Perlmutter facts a run depends on: how ranks map to
GPUs, which node mixes and network settings refuse, why Cray MPICH's GPU
support is off, and how a CPU-only multi-process run is launched. Install
LORRAX first ([Perlmutter installation](../../installation/perlmutter.md),
which also gives the `srun` lines for a clone and a module); JAX and the GPU
memory pool are in the [environment overview](../overview.md#gpu-pool).

Perlmutter's GPU nodes carry four A100 40 GB or 80 GB cards; LORRAX runs
there bare-host on CUDA 13.2 with JAX and JAXLIB 0.9.1. Certified scope: GPU
runs on 1–4 nodes × 4 A100 at P = 4 and P = 16; CPU runs with `impl=mpi` on
the Milan nodes (Cray MPICH 9.0.1), with collective exactness at P = 4 (one
and two nodes) and P = 16 (four nodes), and a P = 4 GN-PPM GW run matching
its reference Σ cell for cell. Larger CPU runs are not certified.

## GPU task geometry {#required-gpu-task-geometry}

Run one rank per GPU: `-N n -n 4n --gpus-per-node=4`, with
`src/ffi/cpp/select_gpu.sh` as the rank wrapper.

```bash
srun -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh python -u -m gw.gw_jax -i deck.in
```

`select_gpu.sh` sets each rank's `CUDA_VISIBLE_DEVICES` to one GPU, the
rank's entry in the list Slurm assigned the step (`LORRAX_GPU_DEVICE`
overrides the list). The runtime passes `local_device_ids` accordingly, so
each rank sees `jax.local_devices() == [cuda:0]` and `len(jax.devices())`
equals the rank count. The step itself keeps all four GPUs
(`--gpus-per-node=4`), which the intra-node collectives need:
`--gpus-per-task=1` confines each rank to its own device, and the first
collective fails with `invalid device ordinal`. `-n 1 --gpus-per-node=4` is
one process over four devices, which is not a P = 4 run.

A job uses one GPU model and memory class. A multi-node allocation requests
`-C 'gpu&hbm40g'` or `-C 'gpu&hbm80g'`; the runtime refuses a mixed 40/80 GB
set before building the mesh (`GATE heterogeneous_gpu_targets`), because
independently compiled XLA programs on mixed nodes can disagree on
collective exchange order and deadlock.

`runtime.source_closure` refuses a run whose imported `runtime` package is
not in the checkout `LORRAX_CHECKOUT` names, so a launcher cannot run one
tree while claiming another ([env vars](../../reference/env_vars.md)).

## Network transport at startup {#network-transport-at-startup}

`runtime.network_env` decides the NCCL transport before JAX or NCCL starts,
from the step's node count (`SLURM_STEP_NUM_NODES` or the expanded
`SLURM_STEP_NODELIST`, never the allocation's).

| placement | result |
|---|---|
| one node | no network settings |
| several nodes, CUDA, `SLURM_NETWORK` contains `no_vni` | the site OFI/CXI profile: `NCCL_NET_PLUGIN` = the absolute path of `nccl/2.29.2-cu13`'s plugin, plus that module's settings including `FI_CXI_RDZV_THRESHOLD=0`; a value the caller set is kept |
| several nodes, CUDA, no `no_vni` | **refuses**, naming the relaunch: without a VNI, NCCL falls back to TCP sockets (about 12× slower) and MPI-IO with NCCL initialization fails |
| `NCCL_NET=Socket` on several nodes | **refuses** |
| explicit `NCCL_NET` or `NCCL_NET_PLUGIN` | the caller's configuration, unchanged |

A multi-node GPU `srun` therefore passes `--network=no_vni` (or exports
`SLURM_NETWORK=no_vni`). The setting acts when SLURM creates the step, so
setting it from Python is too late. `FI_CXI_RDZV_THRESHOLD=0` prevents a
cross-node send/receive deadlock (in XLA's `all_to_all` and
`collective_permute`) in which libfabric's NCCL proxy thread synchronizes the
device while the NCCL kernel waits on that thread.

## Cray MPICH GPU support is off {#mpich-gpu-support}

Every GPU run sets `MPICH_GPU_SUPPORT_ENABLED=0` (the module sets it; a
clone's GPU steps source `config/perlmutter/gpu_env.sh`). The site's default
`craype-accel-nvidia80` module exports `1`, and then `MPI_Init` in either FFI
leg aborts with "GTL library is not linked": neither leg links Cray's GTL,
which is built for CUDA 12. Nothing is lost, because no LORRAX path hands a
device buffer to MPI: parallel HDF5 moves host buffers, and cuSOLVERMp and
cuBLASMp communicate through NCCL ([transports](../transports.md#ffi-mpi)).

## CPU multi-process runs (Milan) {#cpu}

A CPU run uses the same parallel-HDF5 I/O as a GPU run and needs the host
FFI leg. Its JAX collectives run on MPI through an ABI adapter, for the
reasons in [Collective transports](../transports.md). Build the adapter once,
on a CPU compute node; it is pinned, unmodified upstream MPIwrapper built
against the versioned Cray wrappers, and a failed rebuild cannot replace the
active release:

```bash
config/perlmutter/build_mpiwrapper.sh --fresh
```

Every multi-process CPU step sources `config/perlmutter/cpu_mpi_env.sh` in
the rank shell, before Python:

```bash
export LORRAX_CHECKOUT=/path/to/checkout
srun --jobid=$JOBID -N 2 -n 4 -c 16 bash -c '
  set -euo pipefail
  export OMP_NUM_THREADS=14
  . "$LORRAX_CHECKOUT/config/perlmutter/cpu_mpi_env.sh"
  python3 -u "$LORRAX_CHECKOUT/tools/require_jax09.py"
  python3 -u -m gw.gw_jax -i gw.in'
```

`cpu_mpi_env.sh` checks the adapter's source pin, MPI ABI and SHA-256
manifest; refuses a stale Frontera overlay, a conflicting MPI or PMI
preload, `JAX_PLATFORMS` other than `cpu` and more than one CPU device per
process; and sets:

| setting | why |
|---|---|
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` | gloo's CPU reduce-scatter corrupts silently ([transports](../transports.md#why-mpi)) |
| `MPITRAMPOLINE_LIB` = the MPIwrapper release | the ABI adapter JAX's bundled MPItrampoline loads |
| `LD_PRELOAD=/opt/cray/pe/lib64/libpmi.so.0` | Cray PMI must be resident before `jax.distributed` starts its threads; without it `PMI2_Init` segfaults (`libpmi2.so.0` also segfaults) |
| `MPICH_ASYNC_PROGRESS=1` | makes Cray MPICH grant `MPI_THREAD_MULTIPLE` to XLA's FUNNELED request ([thread level](../transports.md#thread-level)) |
| `JAX_NUM_CPU_DEVICES=1` | one JAX device per rank, as the collectives require |
| `MPICH_GPU_SUPPORT_ENABLED=0` | a CPU step has no GPU |

**Threads per rank.** `srun -c` sets one affinity mask per rank. XLA's CPU
worker pool (which ignores `OMP_NUM_THREADS`), the LibSci and SLATE OpenMP
teams (capped by `OMP_NUM_THREADS`), the MPICH progress thread and Python's
I/O threads all share that mask, and LORRAX binds none of them to a private
subset. Set `OMP_NUM_THREADS` below `-c` (`14` with `-c 16`) to leave CPUs
for the XLA pool, the progress thread and I/O.

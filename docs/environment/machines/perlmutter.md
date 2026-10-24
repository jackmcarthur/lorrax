# Perlmutter (NERSC)

This page states the Perlmutter facts a run depends on: how ranks map to
GPUs, which node mixes and network settings refuse, why Cray MPICH's GPU
support is off, and how a CPU-only multi-process run is launched. Install
LORRAX first ([Perlmutter installation](../../installation/perlmutter.md),
which also gives the `srun` lines for a clone and a module); JAX and the GPU
memory pool are in the [environment overview](../overview.md#gpu-pool).

Perlmutter's GPU nodes carry four A100 40 GB or 80 GB cards. The 2026-10-24
JAX/JAXLIB 0.12.0.1 qualification covers the full GW/BSE hsuite at P = 4 on
one A100-80 node and one Milan CPU node (`impl=mpi`, Cray MPICH 9.0.1),
with cold and warm caches and unchanged references. The GPU source was
`496985a8a`; the CPU source carried the same topology-key and live-buffer
accounting fixes (jobs 60203773 and 60203776).

Earlier qualifications include P = 16 on four GPU nodes, CPU collective
exactness at P = 4 on one and two nodes and P = 16 on four nodes, and a P = 4
GN-PPM run matching its reference Σ cell for cell. Those runs do not renew
multi-node certification for the 0.12 upgrade.

On released 0.12.0.1, a real shared-pole constructor at P = 64 on two CPU
nodes (8 × 8 mesh, four physical cores per rank) still aborts in
`MPI_Allgather` with inconsistent panel counts. The failing run used a fresh
compile and authenticated inputs; this is not cleared by the P = 4 tests.
Larger CPU runs, including the former P = 144 compiler/VMA limit, remain
uncertified.

The official CPU nightly pair `jax/jaxlib==0.12.1.dev20261023` passes that
same P = 64 constructor from both a fresh compile and a fresh-process cache
reload. Its XLA pin is `4dc8a9012437`; this is a bounded upstream-build
comparison, not certification of a full P = 64 GW run or a default-lock
change.

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
one process over four devices, which is not a P = 4 run: centroid selection
runs that way, and every driver that opens SlabIO refuses it there. Several
processes that each address several GPUs refuse at startup
(`GATE one_gpu_per_process`).

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
`SLURM_STEP_NODELIST`, never the allocation's). Perlmutter's values live in
its site profile, `config/perlmutter/site.toml`, selected by
`NERSC_HOST=perlmutter` (or `LORRAX_SITE_PROFILE=perlmutter`): the Slurm
network option a step needs (`step_network`), the site NCCL module's settings
(`[nccl.env]`) and the collective cost model receipts price with
(`[comm_model]`). The runtime owns when they apply:

| placement | result |
|---|---|
| one node | no network settings |
| several nodes, CUDA, `SLURM_NETWORK` contains `no_vni` | the profile's `[nccl.env]`: `NCCL_NET_PLUGIN` = the absolute path of `nccl/2.29.2-cu13`'s plugin, plus that module's settings including `FI_CXI_RDZV_THRESHOLD=0`; a value the caller set is kept |
| several nodes, CUDA, no `no_vni` | **refuses**, naming the relaunch: the supported MPI-IO/OFI combination needs a VNI-free step; a controlled 16-GPU run without it failed during MPI/libfabric endpoint creation (`ENOSPC`) |
| `NCCL_NET=Socket` on several nodes | **refuses** |
| explicit `NCCL_NET` or `NCCL_NET_PLUGIN` | preserves the caller's transport choice; the site launch requirements still apply |

Source `config/perlmutter/gpu_env.sh` or load the LORRAX module before `srun`.
Both default an unset `SLURM_NETWORK` to `no_vni` in a launch shell, so ordinary
`srun` needs no network flag. An explicit value, including empty, is kept;
`--network=...` on `srun` still chooses the step's network. Setup does not add
this default inside an active numeric `SLURM_STEP_ID` (or legacy
`SLURM_STEPID`), because the setting cannot repair a step already created.
The Python runtime validates the actual step and names the relaunch if needed. `FI_CXI_RDZV_THRESHOLD=0` prevents a
cross-node send/receive deadlock (in XLA's `all_to_all` and
`collective_permute`) in which libfabric's NCCL proxy thread synchronizes the
device while the NCCL kernel waits on that thread.

Another Slingshot machine (a node with `/sys/class/cxi`) without a profile
gets only `FI_CXI_DISABLE_HOST_REGISTER=1` and `FI_CXI_RDZV_THRESHOLD=0` and
a startup warning naming the missing profile; any multi-node CUDA step with
neither `NCCL_NET` nor a net plugin (`NCCL_NET_PLUGIN`, or `libnccl-net.so`
on `LD_LIBRARY_PATH`) warns that NCCL may fall back to sockets. A new site
adds `config/<NERSC_HOST or LORRAX_SITE_PROFILE>/site.toml` with the same
tables.

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
the rank shell, before Python. The prelude selects the backend and MPI
adapter, not Python or its wheels. Select a CPU-only environment for this
checkout explicitly: the tracked lock and standard module use stable
0.12.0.1; the nightly tests described above use a separate environment.

```bash
export LORRAX_CHECKOUT=/path/to/checkout
export CPU_PYTHON=/path/to/cpu-only-venv/bin/python
srun --jobid=$JOBID -N 2 -n 4 -c 16 --threads-per-core=1 --cpu-bind=threads bash -c '
  set -euo pipefail
  export OMP_NUM_THREADS=14
  export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS=1
  . "$LORRAX_CHECKOUT/config/perlmutter/cpu_mpi_env.sh"
  "$CPU_PYTHON" -u "$LORRAX_CHECKOUT/tools/require_jax.py"
  "$CPU_PYTHON" -u -m gw.gw_jax -i gw.in'
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

**Threads per rank.** Use `--threads-per-core=1 --cpu-bind=threads` so `-c`
assigns physical cores and each rank's affinity mask excludes SMT siblings.
`--cpu-bind=cores` expands the mask to sibling hardware threads, even with
`--hint=nomultithread`: a measured `-c 16` step had 16 logical CPUs on only
eight physical cores. Check the actual mask with `taskset -pc $$` inside a
rank shell when comparing launch configurations.

XLA's CPU worker pool ignores `OMP_NUM_THREADS` and uses the rank's affinity
mask; the LibSci and host FFT OpenMP teams honor `OMP_NUM_THREADS`.
These teams, the MPICH progress thread and Python's I/O threads share the
mask, and LORRAX binds none of them to a private subset. Set
`OMP_NUM_THREADS` below `-c` (`14` with `-c 16`) and allow at least two CPUs
per rank: the required asynchronous MPI progress thread can consume a full
CPU even while the solver is idle, so `-c 1` makes progress compete with
computation.

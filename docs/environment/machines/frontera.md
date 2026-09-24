# Frontera (TACC)

Frontera is LORRAX's CPU target: Cascade Lake nodes, apptainer, Intel MPI and
the layered CPU stack of [the overview](../overview.md#layer-stack). Build
scripts live in [`config/frontera/`](../../../config/frontera/README.md); that
README is the per-script inventory, this page the machine reference.

## 1. Machine facts

| | |
|---|---|
| CPU nodes (CLX) | 56-core Cascade Lake; `/tmp` is a local SSD (224 GB, ~144 GB free, writable inside apptainer) |
| Host OS | CentOS 7, glibc 2.17: JAX runs in the container (`py312.sif`, python:3.12-bookworm base) |
| Container runtime | `tacc-apptainer`, compute nodes only |
| Filesystems | `/work2` streams at 70 MB/s, `/scratch2` at 560 MB/s, node `/tmp` is local XFS |
| Login nodes | RLIMIT_NPROC 300 (`make -j4` at most); no containers or `srun`; `sbatch` allowed |
| dev queue | at most 2 jobs / 40 nodes; `sbatch --parsable` prints the job id on its last line |
| MPI | Intel MPI 2020.4 on the host, hybrid-mounted into the container; provider policy in [transports §3](../transports.md#3-the-intel-mpi-provider-layer-frontera) |

## 2. What gets built, and where

All scripts are in `config/frontera/`; the order is the layer table in the
[overview](../overview.md#layer-stack).

| artifact | script | runs on |
|---|---|---|
| host FFI `liblorrax_ffi_host.so` (parallel HDF5, ScaLAPACK, SLATE, MKL FFT/GEMM) | `build_ffi_host.sh` | login node (host toolchain) |
| MPIwrapper `libmpiwrapper.so` | `build_mpiwrapper.sh --fresh` | login node (needs gfortran; verifies the thread patch in the disassembly) |
| mpi4py + parallel-h5py overlay | `build_mpi_overlay.sh fetch` (login, network) then `build` (SIF, compute) | two phases: compute nodes have no network, login nodes no apptainer |
| staged PMI2 library `$WORK/host_pmi/libpmi2.so.0` | `stage_host_pmi.sh` | login node |
| CPU runtime bundle tar | `build_cpu_runtime_bundle.sh` | inside the SIF |
| per-node staging | `stage_runtime.sh` (source it) | in the job, before Python |
| launch | `templates/gw_dev.sbatch` | the multi-node CPU job |

`mpi_transport_env.sh` applies the Intel MPI transport settings
unconditionally; the launch template sources it. The CPU distributed eigh is
ScaLAPACK `pzheevd` in the host `.so`, reached through the `distrib_la` door
([services](../../architecture/services.md#ffilinalg)).

## 3. Cold start

A cold CPU run pays two costs before physics: jax's CUDA plugin discovery
(34–73 s of `jax.devices()` dlopening a CUDA stack the run cannot use) and
import-graph resolution from Lustre. Resolving the driver's import graph on a
fresh node takes 44–88 s as shipped. `runtime.skip_gpu_plugin_discovery()`
removes the first cost ([overview §2.2](../overview.md#22-the-cpu-run-plugin-skip)),
leaving 11–20 s; the node-local runtime bundle removes most of the second,
leaving 4.6 s, with bit-identical outputs:

```bash
# once per venv/source revision, inside the SIF:
apptainer exec --bind /home1,/work2,/scratch1,/scratch2 $LORRAX_SIF \
    config/frontera/build_cpu_runtime_bundle.sh    # -> $SCRATCH/lorrax_bundle/

# in the job's container-side runner, before python:
export LORRAX_BUNDLE=$SCRATCH/lorrax_bundle/lorrax_cpu_bundle.tar
. $LORRAX_ROOT/config/frontera/stage_runtime.sh    # source it
export PYTHONPATH=$LORRAX_OVERLAY_DIR:$LORRAX_SRC_DIR
$LORRAX_PY -u -m gw.kin_ion_io ...
```

The bundle is the venv, the MPI overlay and `src/` without what a CPU run
cannot use (`nvidia/*`, `jax_plugins/`, `jax_cuda12_plugin/`, the PJRT
dist-info), byte-compiled: 5.6 GB → 769 MB, striped wide so every rank of a
job reads it concurrently. `stage_runtime.sh` unpacks it onto `/tmp` once per
node under `flock` (1.5–2.2 s), keeps the newest two extracts, and falls back
to the Lustre venv with a rank-0 announcement. `LORRAX_STAGE=0` disables it.
Multi-process startup is then flat in P (17 s wall at P=16 and P=64). Do not
distribute the tar with `sbcast`: it is 40–60× slower than the concurrent
striped read and grows linearly with the node count.

## 4. Artifacts outside the repository {#not-yet-vendored}

A cold start of a new allocation depends on artifacts no repo script
rebuilds:

1. the container image `py312.sif` (no `.def` recipe in the repo);
2. the venv `$WORK/lorrax_env/.venv` (jax 0.9.1; no lockfile-to-venv script);
3. the SLATE host install `$WORK/slate_builds/cpu/install`, which
   `build_ffi_host.sh` consumes;
4. an end-to-end multi-rank MPI-IO smoke; the overlay's first multi-rank use
   in a job is the real gate.

"Cold start" therefore means a node that already sees the `/work2` artifacts.

# Frontera (TACC)

*The fully-worked non-Shifter port: apptainer, Intel MPI, the layered CPU
stack of [the overview](../overview.md#1-the-layered-dependency-tree), and
the rtx GPU leg. Build recipes live in
[`config/frontera/`](../../../config/frontera/README.md) — this page is the
machine reference; that README is the per-script inventory.*

## 1. Machine facts

| | |
|---|---|
| CPU nodes (CLX) | 56-core Cascade Lake, `/tmp` is a real local SSD (224 GB, ~144 GB free, writable inside apptainer — job 7882047) |
| GPU partition (`rtx`) | Quadro RTX 5000, Turing **sm_75**, 16 GB, no NVLink, PCIe P2P intra-socket only; driver 535.113.01 → **CUDA 12**, not 13 |
| Host OS | CentOS 7, **glibc 2.17** → JAX must run in-container (`py312.sif`, python:3.12-bookworm base) |
| Container runtime | `tacc-apptainer` — **compute nodes only**; login nodes cannot run it |
| Filesystems (measured, job 7882047) | `/work2` streams at 70 MB/s; `/scratch2` at 560 MB/s; node `/tmp` is local XFS |
| Login-node rules | RLIMIT_NPROC 300 (no process storms, `make -j4` max), no containers/`srun`; `sbatch` submission is allowed |
| dev queue | max 2 jobs / 40 nodes; `sbatch --parsable` output = last line |
| MPI | Intel MPI 2020.4 (host, hybrid-mounted into the container); transport policy in [transports §3](../transports.md#3-the-intel-mpi-provider-layer-frontera) |

## 2. What gets built, and where each build runs

The layer tree is in the [overview](../overview.md#1-the-layered-dependency-tree);
this is the operational summary. All scripts are in `config/frontera/`.

| artifact | script | runs where |
|---|---|---|
| GPU FFI `liblorrax_ffi.so` | `stage_ffi_deps.sh` then `build_ffi.sh --fresh` | inside the SIF, compute node |
| host FFI `liblorrax_ffi_host.so` (phdf5 + ScaLAPACK + SLATE + MKL FFT/GEMM) | `build_ffi_host.sh` | login node (host toolchain) |
| MPIwrapper `libmpiwrapper.so` | `build_mpiwrapper.sh --fresh` | **login node** (needs gfortran; verifies the patch in machine code) |
| mpi4py + parallel-h5py overlay | `build_mpi_overlay.sh fetch` (login: network) then `build` (SIF, compute) | two-phase by necessity: compute nodes have no network, login has no apptainer |
| staged PMI2 lib `$WORK/host_pmi/libpmi2.so.0` | `stage_host_pmi.sh` | login node (provenance + checksum recorded in the script) |
| CPU runtime bundle tar | `build_cpu_runtime_bundle.sh` | inside the SIF (byte-compiles with the container python) |
| per-node staging | `stage_runtime.sh` (**source** it) | in the job, before python |
| certified launch | `templates/gw_dev.sbatch` | the canonical multi-node CPU job |

Environment glue: `gpu_env.sh` (rtx CUDA env — FFI `.so` path, venv nvidia
libs, `CUSOLVERMP_FORCE_NCCL=1`, and the `cuda_async` + sm_75 `XLA_FLAGS`
**matched pair**) and `mpi_transport_env.sh` (unconditional Intel-MPI
transport hygiene). `ffi_env.sh` is a deprecated back-compat shim that
sources both.

## 3. Cold start — the operative recipe

A first run on a fresh node used to pay **44–88 s** resolving its import
graph, of which 34–73 s was `jax.devices()` dlopening a CUDA stack a CPU
run cannot use (job 7882055). Two fixes take a cold `gw.kin_ion_io` to
**4.6 s** import-graph / 28 s wall (job 7882076), bit-identical outputs:

1. `runtime.skip_gpu_plugin_discovery()` — in-tree, arms automatically on
   CPU runs ([overview §2.2](../overview.md#22-the-cpu-run-plugin-skip)).
   88 s → 11 s alone.
2. The node-local runtime bundle — 11 s → 4.6 s:

```bash
# once per venv/source revision, inside the SIF:
apptainer exec --bind /home1,/work2,/scratch1,/scratch2 $LORRAX_SIF \
    config/frontera/build_cpu_runtime_bundle.sh    # -> $SCRATCH/lorrax_bundle/

# in the job's container-side runner, before python:
export LORRAX_BUNDLE=$SCRATCH/lorrax_bundle/lorrax_cpu_bundle.tar
. $LORRAX_ROOT/config/frontera/stage_runtime.sh    # SOURCE it
export PYTHONPATH=$LORRAX_OVERLAY_DIR:$LORRAX_SRC_DIR
$LORRAX_PY -u -m gw.kin_ion_io ...
```

The bundle is venv + MPI overlay + `src/` minus what a CPU run cannot use
(`nvidia/*`, `jax_plugins/`, `jax_cuda12_plugin/`, the pjrt dist-info),
byte-compiled: 5.6 GB → **769 MB**, striped wide so a whole job reads it
concurrently. `stage_runtime.sh` unrolls it onto `/tmp` once per node
under `flock` (1.5–2.2 s), evicts stale extracts (keeps the newest 2), and
falls back to the Lustre venv **loudly**. `LORRAX_STAGE=0` disables.
Multi-process steady state is flat: 17 s wall at both P=16 and P=64
(jobs 7882128 / 7882121).

Do **not** reach for `sbcast` to distribute the bundle: measured 40–60×
slower than the concurrent Lustre read at N=8/N=32, growing linearly in N
(jobs 7882128 / 7882121). Stripe the tar wide instead.

Full measurement record, instruments and falsification protocol:
`docs/dev/archive/cold_start_2026-07.md`.

## 4. Distributed eigh on rtx (GPU leg)

cuSOLVERMp from pip (`nvidia-cusolvermp-cu12` 0.9, NCCL-native, no CAL)
bootstraps via JAX's KV-store + NCCL — no MPI/IB needed. Build flags:
`-DLORRAX_FFI_HAVE_CAL=OFF -DLORRAX_FFI_HAVE_PHDF5=OFF`
`-DCMAKE_CUDA_ARCHITECTURES=75`; CUDA toolkit assembled from the venv's
pip `nvidia-*-cu12` wheels (`stage_ffi_deps.sh`). Set
`CUSOLVERMP_FORCE_NCCL=1` (`gpu_env.sh` does). Sanity run:
`tests/bench/cusolvermp_eigh_test.py --grid 2 2` under
`srun -n 4 apptainer exec --nv …`.

The permanent CPU distributed eigh is ScaLAPACK `pzheevd` in the host
`.so`, behind the `distrib_la` door
([services](../../architecture/services.md#ffilinalg)).

## 5. NOT yet vendored {#not-yet-vendored}

A cold start of a **new allocation** still depends on artifacts outside
the repo (2026-07-31 env-audit ledger — stating the gaps, not implying
completeness):

1. **The container image.** `py312.sif` exists only as a built image
   (`/scratch2/.../lorrax_setup/py312.sif`); no `.def` recipe in the repo.
2. **The venv build.** `$WORK/lorrax_env/.venv` (jax 0.9.1 CPU+CUDA-12
   wheels) has no in-repo lockfile-to-venv script; `pyproject.toml` did
   not build it.
3. **The SLATE host build.** `build_ffi_host.sh` consumes
   `$WORK/slate_builds/cpu/install`, whose build script lives on a
   purgeable filesystem outside the repo.
4. **The end-to-end MPI-IO smoke.** The overlay's first multi-rank use in
   a job is still the real gate; the original 4-rank smoke is not vendored.

Until 1–3 are vendored, "cold start" means "cold start of a node that
already has the `/work2` artifacts".

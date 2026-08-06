# Environment: the runtime stack

*What LORRAX runs on, as a **layered dependency tree**: which script builds
each layer, what each layer depends on, and what breaks — with its failure
signature — when a layer is missing. Plus the JAX configuration that every
platform shares.*

This section replaces the old `ENVIRONMENT_COMPREHENSIVE.md`. The pages:

| page | contents |
|---|---|
| this page | the layered runtime stack, JAX configuration, troubleshooting |
| [Collective transports](transports.md) | gloo vs `impl=mpi` vs NCCL, with the measured verdicts |
| [Frontera (TACC)](machines/frontera.md) | machine facts, cold start, build recipes, vendoring ledger |
| [Perlmutter (NERSC)](machines/perlmutter.md) | Shifter module, FFI staging, what is and is not tested there |

Three references deliberately stay **outside** this section — see the
[register](../index.md#register) for the full ownership map:

* **`docs/dev/env_vars.md`** — the environment-variable **registry**, every
  `LORRAX_*` read machine-enforced by `tests/test_env_registry.py` +
  `tools/env_audit.py`. Rows are never copied into prose here; the registry
  is the single source for spellings, defaults, classes and parse grammar.
* **`docs/architecture/ffi_layout.md`** — everything about *how* a native
  library is reached: the build legs, which nvhpc stage selects which
  communication path, which FFT engine a given `.so` links, and the
  native-layer failure triage. This page owns only *whether the layer is
  present on this machine and what it depends on*.
* **The run's own startup report.** What a particular run *resolved* is
  printed in one rank-0 block by `runtime.initialize_communicator_stack()`.
  After backend init, `os.environ` is a false witness (measured, job
  7882443: two byte-identical environments, `bytes_limit` 11.805 GB vs
  0.000 GB) — read the block, not the env.

---

## 1. The layered dependency tree

The full stack, bottom to top. Layers 3–8 are the multi-process CPU stack
and are Frontera's; Perlmutter's GPU stack uses layers 1–2 plus its own
Shifter module and staged native libraries
([Perlmutter](machines/perlmutter.md)). Every build script named below is
tracked in `config/frontera/`.

```
9  launch template          config/frontera/templates/gw_dev.sbatch
   │                        (the certified composition of everything below)
8  staged runtime bundle    stage_runtime.sh  (per node, in-job)
   │  └── bundle tar        build_cpu_runtime_bundle.sh  (once per revision)
7  env glue                 gpu_env.sh · mpi_transport_env.sh
   │                        └── staged PMI2 lib   stage_host_pmi.sh
6  host FFI .so             build_ffi_host.sh   → liblorrax_ffi_host.so
   │  (GPU twin:            stage_ffi_deps.sh + build_ffi.sh → liblorrax_ffi.so)
5  MPIwrapper               build_mpiwrapper.sh → libmpiwrapper.so
   │                        (thread patch; MPITRAMPOLINE_LIB points here)
4  mpi4py/h5py overlay      build_mpi_overlay.sh (fetch → build)
   │                        + sitecustomize.py teardown fix
3  shared uv venv           NOT vendored (ledger, machines/frontera.md)
2  container image (.sif)   NOT vendored (ledger)
1  host OS + SLURM + MPI    the machine
```

Layer by layer — what it is, what it needs, and the signature when it is
missing or wrong:

| # | layer | built / staged by | depends on | failure signature if missing/wrong |
|---|---|---|---|---|
| 2 | **container image** (`py312.sif` on Frontera; NVIDIA JAX image on Perlmutter) | *not vendored* — see the [ledger](machines/frontera.md#not-yet-vendored) | host OS | on a glibc-2.17 host (Frontera), jax wheels fail outside the container with `GLIBC_2.28 not found` at import |
| 3 | **venv** (`$WORK/lorrax_env/.venv`, uv-built, jax 0.9.1 CPU+CUDA-12 wheels) | *not vendored* — `pyproject.toml` is the dependency authority but did not build this exact wheel set | 2 | `ModuleNotFoundError` at the first import; a CUDA-13 wheel set on the 535.x driver fails at backend init |
| 4 | **mpi4py + parallel-h5py overlay** (`$WORK/lorrax_env_mpi_overlay/site`, mpi4py 4.1.2 + h5py 3.16.0 `HDF5_MPI=ON` + `sitecustomize.py`) | `build_mpi_overlay.sh` (`fetch` on login, `build` in the SIF on a compute node; sha256-pinned sdists) | 2, 3, host Intel MPI 2020.4, host parallel HDF5 1.14.6 | `h5py.get_config().mpi` is False → the `PHDF5_HOST` slab-IO tier is unavailable; without the overlay `sitecustomize`: **every run exits rc=1 after succeeding** ("MPI routine after finalizing MPICH") |
| 5 | **MPIwrapper** (upstream v2.11.1 + `mpiwrapper/lorrax_thread.patch`; the `MPI_THREAD_MULTIPLE` upgrade) | `build_mpiwrapper.sh --fresh` (**login node** — needs gfortran; machine-code verification of the patch) | host gcc/gfortran + Intel MPI (not the container) | `MPITRAMPOLINE_LIB` unset/missing → MPItrampoline refuses loudly at startup; an **unpatched** wrapper loads fine and reintroduces the measured ~29% multi-node segfault/hang class (AS.4b) — which is why the build script verifies the patch in the disassembly |
| 6 | **host FFI `.so`** (`liblorrax_ffi_host.so`: phdf5 + ScaLAPACK + SLATE + FFT/GEMM legs) | `build_ffi_host.sh`; GPU twin `stage_ffi_deps.sh` + `build_ffi.sh` | 2, host MPI, a CBLAS/ScaLAPACK vendor, a SLATE install (*not vendored*) | **fatal at startup.** Since the 2026-08-01 ruling the FFI layer is *required*: a missing or unloadable library refuses in `Gate.enforce`, naming the `.so`. The row that stood here — "never fatal by design … falls back to XLA lowering / lower slab-IO tiers" — described the pre-ruling world and is withdrawn, not kept alongside. Corrected 2026-08-06 during doc consolidation; the ruling is [`decisions.md`](../architecture/decisions.md) 2026-08-01, and the layer itself is [`ffi_layout.md`](../architecture/ffi_layout.md) |
| 7 | **env glue** (`gpu_env.sh`, `mpi_transport_env.sh`; staged PMI2 lib from `stage_host_pmi.sh`) | sourced per job | 5, 6, host SLURM | without the staged PMI2 lib, `srun --mpi=pmi2` bootstraps against TACC's PMI-1 `libpmi.so` → `MPIR_pmi_init` fails; without `mpi_transport_env.sh`, the login shell's leaked `FI_PROVIDER`/`I_MPI_PMI_LIBRARY` win silently |
| 8 | **staged runtime bundle** (`lorrax_cpu_bundle.tar` → node-local `/tmp` SSD) | `build_cpu_runtime_bundle.sh` (in the SIF, once per venv/src revision) then `stage_runtime.sh` (sourced per job, `flock`-once per node) | 3, 4, `src/` | not fatal, **loud**: falls back to the Lustre venv and says so on rank 0; cold import returns to the 44–88 s class from the 4.6 s staged path |
| 9 | **launch template** (`templates/gw_dev.sbatch`) | vendored | all of the above | — the certified composition; edit the `#SBATCH` block and deck variables, leave the env block alone |

Reading the table downward answers "what do I need to build first";
reading the last column answers "which layer is broken" from a job log.

---

## 2. JAX configuration (all platforms)

> ### The two machines are not on the same JAX generation
>
> **Measured 2026-08-06.** Frontera's venv is **jax 0.9.1**. Perlmutter's GPU
> container (`nvcr.io/nvidia/jax:25.04-py3`) ships **0.5.3.dev20260806**.
> `pyproject.toml` declares `jax>=0.9.0` and `jaxlib>=0.9.0` for both, and
> **nothing enforces it** — there is no runtime version check anywhere under
> `src/`, so Perlmutter runs several minor versions below the declared floor
> and says nothing.
>
> This is not a footnote; it decides what you can believe on which machine:
>
> * `tests/test_orbit_syms.py:241` fails **only** under 0.5.3. A red run there
>   on Perlmutter is a JAX artifact, not a defect in the code under test.
> * `common/jax_compile_cache.py:595` `_canonical_accelerator` is called with
>   the wrong arity against 0.5.3's signature, which kills **every P>1 run**
>   that has a cache directory set.
> * `jax.memory_stats()` returns `None` in the container, so device peak memory
>   is **unmeasurable** there. Every allocator figure in §2.1 below was taken
>   on Frontera hardware and is carried forward, not re-measured.
>
> Treat "measured on one machine" as "unknown on the other" until the pin is
> reconciled. Which of the two generations LORRAX is going to support is an
> open owner decision, not something these docs can settle.

Set **before `import jax`**. `runtime.initialize_communicator_stack()` /
`bootstrap()` set the hard defaults; cluster modules and env scripts set the
rest.

| variable | value | purpose |
|---|---|---|
| `JAX_ENABLE_X64` | `1` | 64-bit precision (required for GW) |
| `JAX_PLATFORMS` | `cuda,cpu` (GPU) / `cpu` (CPU runs) | an explicit `cpu` also arms the CUDA-plugin-skip (below) |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | don't pre-grab a fixed XLA pool (set by `runtime.set_default_env()`) |
| `HDF5_USE_FILE_LOCKING` | `FALSE` | Lustre HDF5 compatibility |

**There are two compile-cache variables and they are not the same knob.**
This page used to list only the first, and `env_vars.md` only the second,
which is how "just clear the cache directory" came to name the wrong
directory.

| variable | whose | what it is |
|---|---|---|
| `JAX_COMPILATION_CACHE_DIR` | **jax's own**, default `$SCRATCH/.jax_cache` | set by `config/modulefiles/lorrax/0.1.0.lua` and `config/README.md`; nothing under `src/` reads it. |
| `ISDF_JAX_CACHE_DIR` | **LORRAX's**, `$SCRATCH/lorrax_jax_cache` → `$XDG_CACHE_HOME/isdf_jax_compilation` | the knob `common/jax_compile_cache.py` actually acts on, together with the whole `LORRAX_JAX_CACHE_*` family. Registry: [`env_vars.md`](../dev/env_vars.md) §2b. |

Either way the key includes **every array shape**, so a new system size always
misses.

### 2.1 The three allocators

`XLA_PYTHON_CLIENT_ALLOCATOR` selects between **three distinct allocators**
in the CUDA plugin; the difference decides whether every memory report in
the codebase works:

| value | what it actually is | `memory_stats()` |
|---|---|---|
| unset / `default` / `bfc` | XLA's BFC pool | fully populated |
| `platform` | plain `cudaMalloc` | `bytes_limit=0`, `peak_bytes_in_use=0` — blinds `gw_init` / `gw_output` / `runtime.aot_memory` |
| `cuda_async` | `cudaMallocAsync` mempool | keeps `peak_bytes_in_use` |

Measured on 8× Quadro RTX 5000 across 2 nodes (jobs 7882442 / 7882447 /
7882468, every cell run twice): `cuda_async` is the best of the three
(0.19 GB overhead, largest creatable cuFFT plan 9.20 GB) **and** keeps
`memory_stats()` alive. It also lets XLA, NCCL, CAL and SLATE share one
pool — pre-grabbing 95 % into BFC (`MEM_FRACTION=0.95`) starves NCCL and
surfaces as `cusolverMpSyevd: status=7`.

Three standing corrections:

* `runtime.set_default_env()` deliberately leaves the allocator **unset**
  (= BFC). On sm_75 (Frontera rtx) `cuda_async` additionally needs the
  command-buffer `XLA_FLAGS` restriction — `config/frontera/gpu_env.sh`
  sets the **pair**; never promote one half alone.
* An unrecognised allocator spelling is refused up front by
  `runtime._check_allocator_env()` — left to jaxlib it surfaces as
  `Backend 'cuda' is not in the list of known backends`, which reads as
  missing hardware.
* `TF_GPU_ALLOCATOR` is a TensorFlow variable and is **inert for JAX**
  (byte-identical run with and without it, job 7882442). Do not add it to
  any table.

The memory-fraction cap is read new-spelling-first:
`XLA_CLIENT_MEM_FRACTION`, then the deprecated
`XLA_PYTHON_CLIENT_MEM_FRACTION` (flagged in the startup report) —
`runtime/xla_memory.py`.

### 2.2 The CPU-run plugin skip

On any run that resolves to CPU, jax 0.9.1 still dlopens the full CUDA
library stack during plugin discovery — measured at **76.9 s** on a cold
Frontera node (job 7882076) for libraries that are then discarded.
`runtime.skip_gpu_plugin_discovery()` (armed automatically by
`bootstrap()` / `set_default_env()` when `JAX_PLATFORMS=cpu` or no GPU
device node is visible) answers the discovery with a stub; no jax file is
modified and the same venv still runs GPU jobs.
`LORRAX_CPU_SKIP_GPU_PLUGINS=0` restores the old behaviour, and says so.
Full measurement record: `docs/dev/archive/cold_start_2026-07.md`.

### 2.3 Device selection and multi-host

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m gw.gw_jax -i cohsex.in    # restrict GPUs
export XLA_FLAGS="--xla_force_host_platform_device_count=4"  # CPU mock mesh
```

Multi-process bring-up is owned by `runtime.initialize_communicator_stack()`
(see the [service reference](../architecture/services.md#runtime)): SLURM is
auto-detected (`SLURM_NTASKS > 1` → `jax.distributed.initialize()`), a
sentinel env var guards re-entry, and every rank must call it. Off SLURM,
set `JAX_COORDINATOR_ADDRESS` / `JAX_NUM_PROCESSES` / `JAX_PROCESS_INDEX`
or pass the same to `jax.distributed.initialize()` directly. One GPU per
rank is pinned via `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`, **not**
`--gpus-per-task=1`, which breaks JAX's distributed topology sync
(each rank then sees its GPU as device 0 and passes
`local_device_ids=[0]`).

`jax.distributed` bring-up itself is flat in P and costs about a second to
P=64 (jobs 7882070 / 7882139); a slow "distributed init" is almost always
the CUDA plugin cold load hiding inside the first `jax.devices()`.

---

## 3. Troubleshooting

| symptom | cause / fix |
|---|---|
| `No GPU/TPU found, falling back to CPU` | `nvidia-smi`; `CUDA_VISIBLE_DEVICES`; jaxlib must be the CUDA build |
| `RESOURCE_EXHAUSTED: Out of memory` | check `memory_per_device_gb` in the deck; reduce `chunk_bands`/`chunk_q` or let `gw_init.compute_optimal_chunks` auto-size; per-stage formulas in [memory-model](../architecture/memory-model.md) |
| `cusolverMpSyevd: status=7` + NCCL error 1 | XLA pre-allocated the pool — confirm `XLA_PYTHON_CLIENT_PREALLOCATE=false` and no user `MEM_FRACTION` override (§2.1) |
| every run exits rc=1 **after** succeeding (CPU/MPI) | the overlay `sitecustomize` + `LORRAX_MPI_FINALIZE_FIX=skip_atexit` are not on the path ([transports](transports.md)) |
| HDF5 "file is already open" on Lustre | `HDF5_USE_FILE_LOCKING=FALSE` |
| wrong data from `psum_scatter` on CPU, rc=0 | you are on gloo — see [transports](transports.md); this is the corruption that moved LORRAX to `impl=mpi` |
| stale JIT cache `KeyError` warnings | clear the directory `common.jax_compile_cache` actually used — that is `$ISDF_JAX_CACHE_DIR` (see §2), **not** `$JAX_COMPILATION_CACHE_DIR`. The run's startup block prints the resolved path; use that. |
| `LORRAX_MPI_TYPE=pmix` hangs (Perlmutter) | opt-in legacy path; the unified default `cray_shasta` covers SLATE, cuSOLVERMp and phdf5 |

Debug flags: `JAX_DEBUG_NANS=1`, `JAX_DISABLE_JIT=1`, `JAX_LOG_COMPILES=1`,
`TF_CPP_MIN_LOG_LEVEL=0`. Profiling via `jax.profiler.start_trace` /
`stop_trace` (see `common.jax_profile`).

---

## 4. Dependencies and other clusters

The dependency authority is [`pyproject.toml`](../../pyproject.toml)
(runtime deps, `[dependency-groups]`: `dev`, `jax`, `build`, `profile`).
Not dependencies, despite older prose: cupy, jax-finufft, Docker.

Porting to another SLURM cluster goes through `config/<cluster>/` —
[`config/README.md`](../../config/README.md) §Porting is the knob list.
Frontera is the fully-worked non-Shifter (apptainer) port; Perlmutter is
the Shifter reference. A bare venv runs the pure-JAX path (centroids,
load, serial GW); everything distributed needs the FFI stack
([installation/ffi-native-libs](../installation/ffi-native-libs.md)).

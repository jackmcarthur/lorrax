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
  printed in one rank-0 block by `runtime.initialize_communicator_stack()`
  — [§2.0 below shows a real one, verbatim](#startup-block).
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

## 2.0 What a run actually resolved — the startup block {#startup-block}

Several pages in this tree, including the [register](../index.md#register),
tell you that the run's own rank-0 startup block outranks every documented
default. None of them showed one. This is one, verbatim, so you know what
you are looking for before you need it.

`runtime.initialize_communicator_stack()` prints it once on rank 0, ahead of
the first stage line. Captured **2026-08-06 on Perlmutter, job 56393848**,
`python3 -u -m gw.gw_jax -i cohsex_test.in` on the bundled COHSEX fixture,
**1 process / 1 A100**, launched with `lx run`
([Perlmutter §1](machines/perlmutter.md#1-entry-point-lx)):

```text
==============================================================================
  LORRAX runtime — resolved startup configuration
==============================================================================
  This is a single-process run with 1 addressable device(s).
  jax.distributed.initialize() was not called because this is a single-process run.
  The JAX platform resolved to 'gpu' on devices of kind 'NVIDIA A100-SXM4-40GB', from JAX_PLATFORMS='cuda,cpu', under jax 0.5.3.dev20260806 with 64-bit values enabled.
  The run's device mesh is 1x1 over axes ('x', 'y'), and its communicator cliques were warmed before the first physics jit.
  Bringing this stack up took 2.5 s in total: 1.0 s for the environment, jax.distributed and backend init, 1.4 s to build the mesh and warm its communicators, 0.0 s to arm the compile cache and 0.1 s to measure everything in this block.
  There are no cross-process collectives in this run, so JAX_CPU_COLLECTIVES_IMPLEMENTATION does not apply.
  The live client reports no arena accounting at all, so there is no XLA pool figure for this run and any memory number it prints later came from an nvidia-smi sample of the whole GPU.
  XLA_PYTHON_CLIENT_PREALLOCATE resolved to false (raw 'false') and XLA_PYTHON_CLIENT_ALLOCATOR resolved to 'platform' (raw 'platform') — NOT LORRAX's canonical pair, which is preallocate=false with the allocator left unset (BFC); a caller overrode it.
  TF_GPU_ALLOCATOR='cuda_malloc_async' is set but is INERT for jax; it selects nothing here.
  The CUDA FFI library loaded from /global/u2/j/jackm/software/lorrax_P/src/ffi/cpp/build/liblorrax_ffi.so, which is the in-tree default because LORRAX_FFI_SO is unset.
  FFI build provenance: /global/u2/j/jackm/software/lorrax_P/src/ffi/cpp/build/liblorrax_ffi.so | rev 886139f8e000 | sha 5e2eaa7a30e82a85 | built 2026-08-06T08:10:39Z | slate=?
  The LORRAX_BANDS_GEMM_FFI dial is unset and resolved to on, so the contract_bands right-GEMM contraction rides the platform's native lowering — the dial exists on cpu only and this run's backend is 'gpu', where startup enforcement skips it by the gate's declared platform policy (the native lowering IS the required path there).
  The LORRAX_FFT_FFI dial is unset and resolved to on, so the flat-k 3-D FFT helper path is routed through the FFI handler (the required layer).
  The LORRAX_FFT_FFI_FUSED dial is unset and resolved to on, so the fused IFFT-multiply-FFT tau kernel is routed through the FFI handler (the required layer).
  The distributed backends available for eigh on this mesh are cusolvermp, distributed, native, slate; which one runs is the input-file key, not an environment variable.
  The distributed backends available for cholesky on this mesh are native, slate; which one runs is the input-file key, not an environment variable.
  The distributed backends available for solve_lu on this mesh are native; which one runs is the input-file key, not an environment variable.
  This process is pinned to 128 schedulable CPUs, with OMP_NUM_THREADS=None, MKL_NUM_THREADS=None and OPENBLAS_NUM_THREADS=None.
  Inside the FFI handlers the BLAS team size is LORRAX_MKLBLAS_THREADS=None for the batched GEMM, where unset means the ambient thread count, and LORRAX_SCALAPACK_MKL_THREADS=None for the ScaLAPACK handlers, where unset means a cap of 4 because pzheevd measured 11.28 s per q at 14 MKL threads against 0.463 s at 4.
  The JAX persistent compile cache is enabled at /pscratch/sd/j/jackm/lorrax_jax_cache/np1, used by this single rank.
  The cache key includes every array shape, so a system size this machine has not run before misses every entry no matter how warm the cache looks.
  The fail-fast excepthook is not installed because a single-process run already fails with a traceback and rc=1.
  glibc malloc tuning is in force with M_MMAP_THRESHOLD=1 MB and M_TRIM_THRESHOLD=128 MB, which is the mitigation for the per-r-chunk RSS ramp on long XLA:CPU runs.
==============================================================================
```

### The four lines to read first

**1. Platform and JAX generation.** `The JAX platform resolved to 'gpu' …
under jax 0.5.3.dev20260806`. This is the line that tells you which of the
two generations you are on (§2) — and note that the container restamps its
display string to the *run* date, so `0.5.3.dev20260806` means "a 0.5.3-line
build, looked at on 2026-08-06", not a build from that day. It is also the
line that explains the warning printed immediately *after* the block on this
run:

```text
/opt/jax/jax/_src/compiler.py:723: UserWarning: Error reading persistent compilation cache entry for
'jit_convert_element_type': AttributeError: module 'jax._src.config' has no attribute
'compilation_cache_check_contents'
```

That is the 0.5.3-vs-0.9 private-API divergence of §2 surfacing as a dead
compile cache — the block says the cache is *enabled*, and the warning says
nothing is being read from it. Believe the warning.

**2. The mesh.** `The run's device mesh is 1x1 over axes ('x', 'y')`. The
axis names are the ones every `PartitionSpec` in the codebase refers to; a
mesh that is not the shape you asked for is the first thing to suspect when
a distributed run is slow or wrong.

**3. The allocator / preallocate pair.** On this run:

```text
XLA_PYTHON_CLIENT_PREALLOCATE resolved to false (raw 'false') and XLA_PYTHON_CLIENT_ALLOCATOR
resolved to 'platform' (raw 'platform') — NOT LORRAX's canonical pair …; a caller overrode it.
```

This is the pair §2.1 is about, and it is the reason to read the block rather
than the environment: after backend init `os.environ` is a false witness.
**The sanctioned launcher resolves `platform`,** which is plain `cudaMalloc`
— so `bytes_limit` and `peak_bytes_in_use` both read 0 and every memory
figure the run prints later comes from an `nvidia-smi` sample of the whole
GPU, not from LORRAX's own accounting. That is why the very next line of the
block says there is no XLA pool figure, and why `[gpu_utils] WARNING:
bytes_limit=None, falling back to nvidia-smi` appears further down. The
adjacent `TF_GPU_ALLOCATOR='cuda_malloc_async'` is **inert** — it is a
TensorFlow variable and selects nothing here, which the block says outright
so that nobody credits it with the allocator behaviour.

**4. The FFI `.so` provenance.**

```text
FFI build provenance: …/liblorrax_ffi.so | rev 886139f8e000 | sha 5e2eaa7a30e82a85 | built 2026-08-06T08:10:39Z
```

Path, source revision, content hash and build time of the native library
actually loaded. Because the path is printed, this line is also how you catch
a launcher that put a *different checkout* on `PYTHONPATH` than the one you
are editing — the run above loaded from `lorrax_P` while being launched from
a different worktree, which the path makes obvious and nothing else would.
`LORRAX_FFI_SO` overrides it.

> **Read the block, not the env, and not this page.** Where the block and any
> documented default disagree, the block is what ran.

---

## 2. JAX configuration (all platforms)

> ### The two machines are not on the same JAX generation
>
> **Measured 2026-08-06.** Frontera's venv is **jax 0.9.1**. Perlmutter's GPU
> container (`nvcr.io/nvidia/jax:25.04-py3`) ships **0.5.3.dev20260806**.
> `pyproject.toml` declares `jax>=0.9.0` / `jaxlib>=0.9.0` for both, and
> **nothing enforces it** — grepping `__version__` across `src/` finds exactly
> one hit, `runtime/__init__.py:1589`, and it *records* the version into the
> run fingerprint rather than checking it. So Perlmutter runs below the
> declared floor and says nothing.
>
> **What the straddle does *not* threaten**, because this was the load-bearing
> worry and it was measured clean:
>
> * **The FFI surface is the modern one on both generations.** `jax.ffi`
>   exports the full set on each; `jax.extend.ffi.ffi_call` is the *same
>   object* reached through a deprecation shim; and
>   `register_ffi_target(api_version=1)` / `ffi_call(custom_call_api_version=4)`
>   are byte-identical across the two. LORRAX's ~30 `ffi_call` sites and 2
>   `register_ffi_target` sites are **not** part of the divergence.
> * **No measured numerical result is undermined.** A 2×2 probe
>   ({0.5.3, 0.9.1} × {GPU, CPU}, one script, one node) matched every cell to
>   its version-partner to the last printed digit, including
>   `jax_use_shardy_partitioner=False` on both. *Scope: single-device
>   arithmetic only. Multi-rank reduction order was **not** measured.*
>
> **Two real divergences, not one.** *(This paragraph said "the entire real
> divergence is `jax._src`" until 2026-08-06. The second one below was
> measured that day and is the one that actually blocks the move.)*
>
> 1. **`jax._src` private-API arities**, reached from
>    `common/jax_compile_cache.py`, which monkeypatches jax internals. Every
>    *public* symbol LORRAX touches does exist with a compatible signature on
>    both generations, so this half of the old claim stands.
> 2. **`shard_map` behaviour, from jax 0.7.0** — a public-API divergence that
>    a signature comparison cannot see. From 0.7.0 `shard_map` tracks
>    **varying manual axes** (VMA): every value in the body carries the set of
>    mesh axes it may differ over, and `scan` / `fori_loop` / `while_loop`
>    require that set to be **equal** at carry input and output. An
>    accumulator built by `jnp.zeros` has an empty set, so the first
>    `A = A + <sharded data>` makes the output varying and the loop is
>    **rejected at trace time**. Tracking starts at **0.7.0, not 0.9** — a
>    guard written as "only 0.9 needs this" is wrong across the whole
>    0.7–0.8 range. Worse, `lax.pcast` does not exist on 0.7.x at all, so the
>    obvious `try: lax.pcast / except AttributeError: identity` shim installs
>    a **no-op on exactly the versions that enforce the rule**; that defect
>    was live in `common/cholesky_2d.py`.
>
> **The blast radius of (2) is small and was counted, not estimated.** An AST
> census over `src/` — `shard_map` entry points → within-module call-graph
> closure → loop carries, classified by how the init is built — finds 21
> loop-carry sites reachable from a `shard_map` body, of which **8 need
> marking, all in `src/bse/bse_ring_comm.py`**. Marking is not free either
> way: a carry that is invariant by construction and leaves through a
> *replicated* `out_specs` entry **fails** if it is marked, so "mark all the
> mesh axes to be safe" is its own bug.
>
> **Status.** The port is done on `agent/jax-070-land-2026-08-06`
> (**unmerged** — not an ancestor of `integration/2026-08-06` or of
> `origin/main`): `src/common/vma.py` owns the version decision once and
> refuses at import on a jax that tracks VMA but offers neither spelling;
> the container moves to `ghcr.io/nvidia/jax:jax-2025-07-21` (jax 0.7.0,
> CUDA 12.9); `common/jax_support.py` gates the window at
> `[0.7.0, 0.10.0)`; and four of the five compile-cache shims are deleted.
> The startup version gate itself is already on this branch
> (`agent/jax-version-gate-2026-08-06`, merged).
>
> **"Just move to a newer container" is not available, and this is the first
> thing to check before planning around it.** Measured 2026-08-06: **no NVIDIA
> JAX container exists with both JAX ≥ 0.9 and CUDA 12.** The earliest tag
> carrying JAX ≥ 0.9 is `26.02-py3`, and it ships **CUDA 13.1** — the
> CUDA 12 → 13 flip lands three minors *before* JAX reaches 0.9, so the two
> requirements never overlap in any published image. Satisfying the declared
> `jax>=0.9.0` on Perlmutter therefore means taking a CUDA major bump with it,
> against everything staged under `/lorrax_nvhpc` for CUDA 12.9 (§4 of
> [`ffi_layout.md`](../architecture/ffi_layout.md)). That is a port, not a pin
> change.
>
> **But the exit is not to 0.9 — it is to 0.7.0, and that image does exist.**
> `ghcr.io/nvidia/jax:jax-2025-07-21` ships jax 0.7.0 on CUDA 12.9 and is the
> **last CUDA-12 image**. It clears the VMA divergence above and lets the
> supported window be `[0.7.0, 0.10.0)` — a window both machines can meet —
> without touching the CUDA-12 stage. Taken on
> `agent/jax-070-land-2026-08-06`, unmerged.
>
> **A version string read from a container is not evidence.** Every container
> JAX is a dev build that restamps its display string to the *run* date: all
> ten images probed on 2026-08-06 printed `.dev20260806`, whatever they
> actually contain. So `jax.__version__` from inside a container tells you
> when you looked, not what you have. **Read `jax.__version_info__`**, which
> carries the real tuple, and record that instead — including in any gate or
> provenance record that currently captures the string.
>
> ### `memory_stats()` returns `None` because the modulefile asks for it
>
> **Measured across ten container images, 2026-08-06: `memory_stats()` returns
> a full dict on all ten, including today's `25.04`.** It is not the JAX
> version and it is not the PJRT plugin. Adding the Perlmutter modulefile's
> allocator environment is what turns it to `None`.
>
> `config/modulefiles/lorrax/0.1.0.lua:130` sets
> `XLA_PYTHON_CLIENT_ALLOCATOR=platform` on the host and `:175` passes the
> same value into the container. **§2.1 below already says what `platform`
> costs** — it is plain `cudaMalloc`, so `bytes_limit` and
> `peak_bytes_in_use` both read 0 — and `config/README.md:66` already says
> the value should be `cuda_async`, "**not** `platform`". The modulefile sets
> the one allocator both documents warn against.
>
> So this is **recoverable, and it is a trade rather than a limitation**: the
> allocator setting buys whatever `platform` is there for and pays for it
> with every memory report in the codebase (`gw_init`, `gw_output`,
> `runtime.aot_memory`). Change the allocator and the numbers come back.
>
> It fails in the worst available way: the `hasattr` guard passes, the caller
> receives `None`, and nothing announces it. Until the modulefile changes,
> the allocator figures in §2.1 are **Frontera** measurements carried
> forward. (There is also no top-level `jax.memory_stats` symbol on either
> generation — the `src/` references to that spelling are a docstring and an
> f-string label, not calls.)
>
> *Recorded because this cause was stated wrongly twice before it was
> measured — first as version-linked, then as a PJRT-plugin limitation that
> upgrading could not fix. Both were wrong, and the second would have sent
> someone to rebuild a container. The measurement is ten images, one probe.*

> ### f32 does not mean the same thing on the two machines
>
> Unrelated to the straddle, identical under either JAX generation, and worth
> knowing before you compare numbers across machines: an f32 result on
> Perlmutter GPU carries **TF32** error, not true f32 — measured **3.08e-04**
> against CPU's **5.66e-07**. That is the hardware, not a bug and not a
> regression. The GW/BSE production path is complex128 throughout, so it does
> not bite there; a cross-machine f32 comparison that has not accounted for it
> will look like a defect and is not one.

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

> **On the GPU leg the persistent compile cache is currently dead at P=1, and
> LORRAX's own patch is what kills it.** Measured 2026-08-06. With
> `ISDF_JAX_CACHE_DIR` unset, JAX's native cache works and writes entries.
> With LORRAX's patch armed, `common.jax_compile_cache` reports
> `enabled=True` and **zero entries are written** — so the run pays full
> compile time on every launch while the startup block says the cache is on.
>
> This is not the container: the native path works in the same image. Any
> statement anywhere promising compile-cache behaviour on the GPU leg —
> including the `LORRAX_JAX_CACHE_*` rows in
> [`env_vars.md`](../dev/env_vars.md) §2b, which describe the multi-process
> design — is currently describing the CPU leg only.
>
> `enabled=True` with zero entries written is the exact shape QUALITY_PATTERNS
> #7 is about: a status field reporting intent rather than outcome. Treat the
> entry count, not the flag, as the observable.

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

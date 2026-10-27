# Start-up, environment and caches

This page owns the start-up contracts for the process environment, CPU malloc
arenas, concurrent on-disk caches, and the pairing of a checkout with its FFI
libraries. The start-up steps themselves
and their costs are in [Compilation](compilation.md#3-xla-programs). The
variables are in the [registry](../reference/env_vars.md).

## 1 The environment freezes at step 7b {#env-freeze}

**Rule.** Every LORRAX write to `os.environ` happens before
`runtime.initialize_communicator_stack` calls
`runtime.env_guard.freeze_process_environment()` at step 7b, after resolving
the vendor kernel stores and final BLAS thread defaults. After the freeze:
- A write from LORRAX's own code (`src/`, `services/*/src`) that **adds** a key
  refuses with `GATE env-frozen`.
- An adding write from any other code is announced once per key and caller,
  then performed. The residual risk is named, not hidden.
- Overwrites of existing keys and unsets pass. Reading is always safe.

Values that later code needs travel in-process (function arguments,
`common.sanity.forced_level`), never through the environment. Treat any
post-freeze warning in a log as a bug (INT7 release notes, issue 5).

**Why** (the historical rc 139 start-up SIGSEGV). glibc's `getenv` takes no lock. A `setenv`
of a **new** key reallocates the `environ` array and frees the old one, so a
native thread scanning `environ` reads freed memory. Overwrites swap one pointer
and unsets shift in place, so neither frees what a reader holds.
- In the former asynchronous Frontier warm-up, the faulting thread was
  `lorrax-blas-warmup`:
  `rocblas_dgemm → hipModuleLoadData → libhsa-runtime64 → getenv`.
- The writer was the main thread: the first `SlabIO` open exported two new keys,
  `LORRAX_PHDF5_STRIPE_COUNT` and `_SIZE_FS`.
- Under load the Tensile code-object load is slow, so the window widened. The
  crash was intermittent, about 10 s in, on five nodes and any GCD set (lane
  crash, `crash_w8`; native backtraces under
  `/lustre/orion/mat280/scratch/jackmc/crash`).

The fix made the striping policy a returned value (`_resolve_striping`; C++
phdf5 derives the same policy) and armed the freeze. The same stress then gave
0 crashes in 48 start-ups, with eqp unchanged bit for bit
([Compilation §3](compilation.md#3-xla-programs)). HSA, MPI-IO, HDF5 and XLA
all read the environment from their own threads, so the race is reachable on
any leg; the rule is leg-neutral. Tests: `tests/test_env_guard.py`.

### ROCm code-object preload {#rocm-preload}

ROCm now calls the public `rocblas_initialize(void)` synchronously on the
main thread at startup step 5e, before the first GEMM or collective. It uses
the already-attested provider's dependency scope and restores the caller's
HIP device. The asynchronous small-GEMM warm-up is gone. CPU and CUDA have
no such preload. The environment freeze remains at step 7b because HSA,
MPI-IO, HDF5 and XLA still read environment variables from native threads.

Preloading moves a real cost to startup. In the 2026-10-27 Frontier MI250X
qualification with SDK 7.14.1, a direct P1 API call took 27.9 s; P4 rank-0
preload clocks ranged from 24.3 to 118.7 s. These are observed process-local
clocks, not fixed startup estimates. The P1 probe also observed about 1.0 GB
of additional device residency and 2.7 GB of host RSS. Those increments are
outside XLA's executable buffers, are not full-job peaks, and are not
portable constants. Startup also includes other work: the same P4 jobs'
small hipRTC gate build ranged from about 0.9 to 90 s per rank. An XLA
cache hit therefore does not establish a warm vendor startup. See
[Compilation](compilation.md#3-xla-programs) and the
[memory budget](memory-model.md#budget) for their separate coverage.

### Stable ROCm communicator ownership {#rocm-communicators}

The shared ROCm leg disables XLA's communicator-splitting optimization at
startup, before the first communicator is acquired. XLA can otherwise
replace a warmed direct subgroup with a child sharing a larger parent's
resources. Actual 144-rank logs showed this replacement before the first
post-ELPA eigenvector exchange stalled; disabling splitting completed the
same checked cold solve and two repeats. This preserves the bounded array
algorithms and their numerical checks. It adds no application warm-up tier.

The runtime supplies this default automatically; Frontier run scripts need
no extra flag. CPU and CUDA leg defaults are unchanged. This Python policy
does not change the native ABI or require another build. Independent
communicators may retain additional resources outside XLA's executable
buffers, so use the reported memory budget and outside-pool observations.
See [the XLA flag table](compilation.md#3-xla-programs).

## 2 Shared caches: atomic, keyed, validated, concurrent-safe {#caches}

A cache that several processes, nodes or checkouts share must meet four rules.
A miss is always correct; a wrong hit is not.
- **(a) Atomic publication:** a reader sees the old file or the new one, never a
  partial one.
- **(b) A full content key:** everything the payload depends on is in the
  identity, not a hand-bumped version number.
- **(c) A validated read:** framing, a payload hash and a type/range check.
  Any mismatch is a miss.
- **(d) Concurrent safety:** writers on different nodes of one tree never
  collide on a temporary name.

`lxkit.cache.atomic_write_bytes(path, data)` implements (a) and (d). It writes a
temporary beside the target named for host, pid and a random token, created
with `O_EXCL`, then fsyncs it and calls `os.replace`. On failure the temporary
is removed and the target is untouched. Use it for every new cache file.

| cache | key | read check | publication |
|---|---|---|---|
| hipRTC / NVRTC kernel images (`common/rtc_build.cc`: kdft, active_gemm, active_subspace, kconv) | FNV over source, headers, defines, version files, RTC version and path; the kconv store is one directory per native closure identity (`kconv_mathdx/ffi-<12 hex>`) | magic + key + payload hash; a refused image is deleted and rebuilt | unique tmp + rename |
| `wfn_trs` stamps (`symmetry_maps.density_symmetry_check`, in `lxkit.user_cache_dir("wfn_trs")`) | WFN identity, a module-source digest, algorithm version, nocc, tolerance; schema `wfn-trs-stamp-v2` | SHA-256 of the report, then a type and range check | `atomic_write_bytes` |
| rocFFT kernel snapshot | rocFFT's, in one store per native closure identity (`rocfft/ffi-<12 hex>`) | rocFFT's | copy + merge + fsync + `os.replace` |
| JAX compile cache | namespace keyed by the jax version and the location-free native closure identity (content hashes, not install paths: `…_ffi-<12 hex>_k3`); an explicit `ISDF_JAX_CACHE_DIR` is split by the same tag | JAX's | prune by atomic rename into `.trash` |

A `wfn_trs` stamp holds only the time-reversal verdict, but a hit skips the
measurement and hands `trs_holds` to `WfnLoader`. A stale verdict would
therefore change k completion silently. That is why (b) and (c) matter for a
small file.

## 3 The FFI source key {#ffi-source-key}

The FFI libraries of each leg are built from `src/ffi/cpp`. The loader checks
the handler ABI number (each `lxkit.LEGS` row's `abi_symbol`,
`lorrax_ffi_{host,cuda,rocm}_abi_version`). That check does not catch a library
built from an older tree with the same ABI number. The SlabIO lane hit this: a
handler signature changed without an ABI bump, so an older build passed the
version check and then failed XLA's argument-count check, not by name. A site recipe must therefore pair a checkout with libraries built
from **its own** FFI sources.

The Frontier pool recipe does it with a source key (`lxffi.sh`, site tooling,
not in the repo). Use the same pattern for a new site:
1. **Key.** A 16-hex SHA-1 over the tracked index, the uncommitted diff and the
   untracked files under `src/ffi/cpp` and `config/<site>`, plus the names of
   the venv's jax and jaxlib dist-info directories.
2. **Stamp.** A build writes the key it was built from to `.lx_srckey` beside
   its `.so`, and checks the key again after the build (sources changed during
   a build are an error). The build also writes `<leg>.closure.json`, the
   dependency closure it resolved
   ([the dependency closure](../installation/ffi-build.md#closure)).
3. **Resolve.** Export the leg pins (`LORRAX_FFI_ROCM_SO`, `LORRAX_FFI_HOST_SO`;
   CUDA: `LORRAX_FFI_SO`) to a build whose stamp equals the checkout's key
   **and whose closure record still verifies on this machine**
   (`python -m lxkit.closure verify <leg.so>`, exit 0): its own build
   first, then any sibling worktree. The key covers the sources; the record
   covers the libraries (ELPA, OpenBLAS, the ROCm or CUDA wheels, MPI, HDF5,
   LibSci, by content hash). A rebuilt ELPA install or a changed wheel
   therefore fails the match and forces a rebuild, rather than being reused.
   The loader would refuse such a build at start-up anyway. Optionally build what is missing,
   under a per-worktree `flock`. **Keep locks and logs outside the worktree**,
   so a `git add -A` cannot commit them. With no match, refuse (exit 126);
   never fall back to another build.
4. **Override.** A manually exported path still wins, and is reported as an
   override with its key status (matching or not). Its closure is enforced at
   start-up like any other.

The key changes when any FFI source, the site config or the jax/jaxlib pin
changes, which is exactly when a rebuild is needed. It does not change for
Python-only commits, so Python-only lanes share one build.

## 4 CPU malloc arenas {#cpu-arenas}

After resolving a CPU platform and before creating JAX/MPI workers,
`runtime.set_default_env()` calls `runtime.blas_env.tune_cpu_malloc_arenas()`.
On glibc it limits future malloc arena creation to two through
`mallopt(M_ARENA_MAX, 2)`. Each rank otherwise accumulates arenas for its XLA,
OpenMP and native callback workers; free space in those arenas can remain
resident after preprocessing. This limit controls allocator fragmentation,
not the number of compute threads. It complements the existing mmap/trim
threshold policy and does not clear JAX executable caches.

The launcher's explicit `MALLOC_ARENA_MAX` or
`GLIBC_TUNABLES=glibc.malloc.arena_max=...` wins. The existing
`LORRAX_MALLOC_TUNE=0` opts out. Accelerator clients and non-glibc hosts are
left alone. The operation writes no environment variables, and its result
is recorded in the startup `malloc_tune.cpu_arenas` facts. Import/runtime
startup should precede numerical worker creation: changing the cap cannot
destroy arenas that were created earlier.

The Perlmutter CPU pre-release profile (2026-10-22) verified the cap with
`malloc_info`: a warmed four-rank FFT/GEMM run used a median 156.5 arenas per
rank with the default policy and two after the change. At 64 ranks the
counts were 41 and two. Warmed GEMM time changed by less than one percent;
the FFT had no measured slowdown. The focused integration check is
`tests/test_runtime_cpu_arenas.py`.

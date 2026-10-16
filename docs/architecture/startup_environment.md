# Start-up, environment and caches

This page owns three start-up contracts: when the process environment may be
written, what makes an on-disk cache safe to share, and how a site recipe pairs
a checkout with the FFI libraries built from it. The start-up steps themselves
and their costs are in [Compilation](compilation.md#3-xla-programs). The
variables are in the [registry](../reference/env_vars.md).

## 1 The environment freezes at step 7b {#env-freeze}

**Rule.** Every LORRAX write to `os.environ` happens before
`runtime.initialize_communicator_stack` starts the first native helper thread
(step 7b, the rocBLAS/BLAS warm-up, `runtime.vendor_kernels`). At that point it
calls `runtime.env_guard.freeze_process_environment()`. After the freeze:
- A write from LORRAX's own code (`src/`, `services/*/src`) that **adds** a key
  refuses with `GATE env-frozen`.
- An adding write from any other code is announced once per key and caller,
  then performed. The residual risk is named, not hidden.
- Overwrites of existing keys and unsets pass. Reading is always safe.

Values that later code needs travel in-process (function arguments,
`common.sanity.forced_level`), never through the environment. Treat any
post-freeze warning in a log as a bug (INT7 release notes, issue 5).

**Why** (the rc 139 start-up SIGSEGV). glibc's `getenv` takes no lock. A `setenv`
of a **new** key reallocates the `environ` array and frees the old one, so a
native thread scanning `environ` reads freed memory. Overwrites swap one pointer
and unsets shift in place, so neither frees what a reader holds.
- On Frontier the faulting thread was always `lorrax-blas-warmup`:
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
| hipRTC / NVRTC kernel images (`common/rtc_build.cc`: kdft, active_gemm, active_subspace, kconv) | FNV over source, headers, defines, version files, RTC version and path | magic + key + payload hash; a refused image is deleted and rebuilt | unique tmp + rename |
| `wfn_trs` stamps (`symmetry_maps.density_symmetry_check`, in `lxkit.user_cache_dir("wfn_trs")`) | WFN identity, a module-source digest, algorithm version, nocc, tolerance; schema `wfn-trs-stamp-v2` | SHA-256 of the report, then a type and range check | `atomic_write_bytes` |
| rocFFT kernel snapshot | rocFFT's | rocFFT's | copy + merge + fsync + `os.replace` |
| JAX compile cache | namespace keyed by the jax version and the FFI identity | JAX's | prune by atomic rename into `.trash` |

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
   a build are an error).
3. **Resolve.** Export the leg pins (`LORRAX_FFI_ROCM_SO`, `LORRAX_FFI_HOST_SO`;
   CUDA: `LORRAX_FFI_SO`) to a build whose stamp equals the checkout's key: its
   own build first, then any sibling worktree. Optionally build what is missing,
   under a per-worktree `flock`. **Keep locks and logs outside the worktree**,
   so a `git add -A` cannot commit them. With no match, refuse (exit 126);
   never fall back to another build.
4. **Override.** A manually exported path still wins, and is reported as an
   override with its key status (matching or not).

The key changes when any FFI source, the site config or the jax/jaxlib pin
changes, which is exactly when a rebuild is needed. It does not change for
Python-only commits, so Python-only lanes share one build.

# `ffi.phdf5` — architecture, pitfalls, and why things look odd

**Audience**: the next engineer or agent who touches this FFI and needs
a durable explanation before "fixing" something that turns out to be
load-bearing. Companion to `cpp/`, `write.py`, `read.py`, `context.py`,
and `file_io/_slab_io_ffi.py`. Timings are MoS2 3×3, 4 × A100 Perlmutter,
2026-04-19 build — absolute numbers drift, the *shapes* of the problems
don't.

**MPI stack**: unified on Cray MPICH as of 2026-04-20 (was OpenMPI
earlier in the investigation). Section 2.6 below refers to the first
`MPI_Init_thread` cost; the number scales similarly on both stacks.
Stack-specific tuning is in [`PORTING.md`](../PORTING.md).

**CPU: this FFI is NOT CUDA-only any more.**  It was, until workstreams
A (read, 2026-07-25) and AE (write, 2026-07-26) made every phdf5 TU
compile into BOTH platform libraries from the same source under one
flag, `LORRAX_FFI_NO_CUDA`:

    liblorrax_ffi.so       CUDA   PhdfRead*Ffi     / PhdfWriteFfi
    liblorrax_ffi_host.so  cpu    PhdfRead*HostFfi / PhdfWriteHostFfi

The collective MPI-IO core — hyperslab arithmetic, `valid_shape`
clipping, the FIFO writer thread, `H5Dread`/`H5Dwrite` — is byte-identical
on both.  Only three seams switch, and they all live in
`cpp/platform_seam.h`: (1) the handler binding (host handlers take no
`PlatformStream` Ctx and get distinct symbol names so both `.so`s can be
`RTLD_GLOBAL`-loaded in one process), (2) the small index buffers
(`cudaMemcpy` D2H vs a plain host read), (3) the payload staging — and on
the host build the write side of (3) is *nothing at all*: XLA's CPU
buffer already IS host memory, so `H5Dwrite` reads the local shard in
place, with no pinned allocation, no copy and no event.

`src/file_io/_slab_io_mpi_host.py` (mpi4py + h5py-parallel) remains as the
**second** CPU tier, for a deployed host lib built before the write port;
it needs an extra Python environment that the FFI does not.  All three
writers produce byte-identical output.  Selection is automatic in
`LorraxConfig.from_input_file` → `_route_cpu_slab_io`, which
capability-probes the lib rather than guessing from the platform.

The mpi4py path is synchronous (no Python worker thread) because there's
no D2H to overlap and Cray MPICH's default `MPI_THREAD_SINGLE` deadlocks
on cross-thread MPI-IO at `H5Fclose`; see its module docstring.  The FFI
path keeps its writer thread on both platforms — one thread per ctx is
what guarantees every rank enters the MPI-IO collectives in the same
order.

------------------------------------------------------------------------

## 1. The two-layer async write, in one diagram

Writing a sharded JAX array to an HDF5 hyperslab looks like this,
reading top-to-bottom:

```
Python main thread                                Python worker thread               C++ writer thread
(user code)                                       (per-SlabIO)                       (per-PhdfCtx)
───────────────────────────────                   ──────────────────────             ──────────────────────
write_slab(A, offset)                             queue.get()  ─── waits
  ├─ compute sharding / cache key                  ↓
  ├─ _sm_cache lookup  (jit(sm) )                  [pops task]
  ├─ jnp.asarray(offset)                           sm(A, offset_arr)                   queue_cv.wait() ── waits
  ├─ put(task) on queue  ────────────────►          │
  ▼                                                 │  (XLA dispatch — compile cached
returns in ~0.2 ms                                  │   via sm_cache + jax.jit wrap)
                                                    │
                                                    ▼
                                                   FFI handler:
                                                     ├─ ensure_pinned(ctx, bytes)
                                                     ├─ cudaMemcpyAsync D2H
                                                     │    on ctx->stream
                                                     ├─ cudaEventRecord
                                                     │    on ctx->d2h_event     (POOLED)
                                                     ├─ enqueue task ───────────►  [pops task]
                                                     └─ return ffi::Future
                                                    ↓                                    │
                                                   XLA waits on Future                   cudaEventSync(d2h_event)
                                                    ↓                                    H5Dwrite (MPI-IO collective)
                                                   jit(sm)(A) returns when Future.SetAvailable                       │
                                                    │                                                                ▼
                                                    ▼                                                          promise.SetAvailable()
                                                   block_until_ready()   ◄───────────── Future fires ──────────
                                                    ↓
                                                   done; loop to next task
```

Three layers of async coordination, **each doing exactly one job**:

| layer | freed when | costs | why we can't collapse it |
|---|---|---|---|
| Python main thread → Python worker queue | ~0.2 ms per call | Python queue is cheap | lets user's compute continue during write |
| Python worker → XLA FFI handler | after handler enqueues C++ task | handler itself is ~ms | serialises jit dispatch across ranks (needed for MPI collective order) |
| C++ writer thread → ffi::Future | after H5Dwrite completes | 770 ms/chunk is the MPI-IO floor | one thread per ctx → MPI collective rendezvous in order |

Reads use the same architecture, minus the Python worker: handler
does the H5Dread inline, async H2D onto `ctx->stream`, records
`ctx->h2d_event`, makes `xla_stream` wait on it, returns.

------------------------------------------------------------------------

## 2. Design decisions that look odd, and why

### 2.1. `ffi::Future` alone does **not** free the Python main thread

The natural read of XLA's async FFI docs is: "return a Future and the
caller doesn't block." **Not true for Python.** Measured by injecting
a 300 ms sleep *after* `promise.SetAvailable()` in the writer thread:
`jit(ffi_call)(A)` wall-time grew from 45 ms → 350 ms in lockstep.

**What `ffi::Future` actually buys you**: XLA's scheduler can overlap
*downstream XLA ops* with the handler's work. It does **not** release
`jit(...)(A)` on the Python caller — that still waits for the Future
to resolve.

**Why this matters**: to free the Python main thread you *also* need
a Python-side worker thread that owns the `jit(sm)(A).block_until_ready()`
call. The C++ `ffi::Future` + bg worker thread is correct but
insufficient on its own. Both layers are needed; each does its own job.

This is not documented in JAX's FFI tutorial as of 0.6; confirmed
via agent research against OpenXLA's AsyncStart/Done and HDF5's
`vol-async` connector — both use the same pattern.

### 2.2. Pooled `cudaEvent_t` on the ctx (NOT per-call)

Earlier versions of this FFI created one `cudaEvent_t` per call
with `cudaEventCreateWithFlags(cudaEventDisableTiming)`, used it
for cross-stream coordination, then `cudaEventDestroy`'d it in the
worker thread. **Correct logically; catastrophic in practice.**

Measured:
```
rank  cudaEventSynchronize   cudaEventDestroy
 r0          +10 ms                 +8 μs
 r1          +10 ms              +753 ms   ◀
 r2          +10 ms              +775 ms
 r3          +10 ms              +775 ms
```

`cudaEventDestroy` blocks **~750 ms on non-rank-0 processes** when
`xla_stream` has a pending backlog of work. Confirmed by running
with `LORRAX_PHDF5_SKIP_DESTROY=1` (leaks events) — total wall drops
from 31 s to 25 s.

**Hypothesis** (not 100% confirmed against CUDA driver source):
CUDA's `cuda_async` stream-ordered allocator keeps per-event
bookkeeping that it serialises against the recording-stream's pending
work on destroy. rank 0's `xla_stream` happens to be idle at the
critical moment on each chunk; ranks 1/2/3's is 700-800 ms ahead on
next-chunk compute.

**Fix**: `PhdfCtx::d2h_event` and `PhdfCtx::h2d_event` — **one event
per ctx**, created once in `open_ctx`, destroyed once in `close_ctx`.
`cudaEventRecord` on an already-recorded event just updates the
record point; subsequent `cudaEventSynchronize` waits for the new
record. Zero per-call event lifecycle, zero 800 ms stalls.

**Rule for future edits**: do not introduce a per-call
`cudaEventCreate` / `cudaEventDestroy` pair on the hot path under
`cuda_async`. Pool every event on the ctx.

### 2.3. `offset_base` is a runtime `Buffer<S64>`, not an FFI `Attr`

Natural reading: offsets are small (ndim × 8 bytes), and XLA's FFI
Attrs are the obvious place for compile-time-constant metadata like
that. **Don't do this for anything that varies per call.**

FFI Attrs are compile-time constants. Every distinct value → a
fresh XLA compile. At MoS2 3×3 this meant 4 distinct zeta-chunk
offsets compiled 4 identical-signature-but-different-attr HLO modules
(~400 ms each); the read path did the same for its many read
offsets. Visible in the HLO dump as
`jit__unnamed_wrapped_function_` × 10 with identical arg signatures.

**Fix**: change the FFI handler signature from
`.Attr<Span<const int64_t>>("offset_base")` to
`.Arg<Buffer<S64>>()`. Handler does a ~10 μs `cudaMemcpy` D2H of the
tiny offset buffer and proceeds identically. On the Python side pass
`jnp.asarray(off, dtype=jnp.int64)`.

Side effect: the ffi_call's trace signature becomes shape-only, the
shard_map closure can compile once per
(dataset, ndim, dtype, sharding) combination and re-dispatch across
offsets.

**Rule**: anything that the *caller* varies per call belongs as a
runtime `Arg`, not a compile-time `Attr`. Attrs are for things that
are genuinely constant for the life of the compiled module (e.g.
`ctx_handle`, `ds_id`, `mesh_shape` — at least until we add
dynamically-created datasets).

### 2.4. `jax.jit(shard_map(...))` wrap around the cached `sm`

Even after the runtime-offset fix, we cached `sm = shard_map(_per_rank, …)`
in `_sm_cache` by signature — but `sm` alone still produced 4 distinct
HLO modules across 4 calls. shard_map in eager mode re-traces on each
invocation despite the caller holding the same `sm` object.

**Fix**: wrap once at cache-insertion time: `sm = jax.jit(sm_bare)`.
`jax.jit`'s own cache then correctly dedups by traced signature.

After this combined with 2.3: 4 zeta writes → 1 compiled HLO,
shared across all 4 calls. Same for reads.

### 2.5. Single `SlabIO` handle for create+write, not two

Before 0a41f42, `zeta_fit_chunked` did:

```python
with SlabIO(path, mode='w') as io_create:      # H5Fcreate + H5Dcreate
    io_create.create_dataset('zeta_q', ...)
    # __exit__ → H5Fclose
zeta_io = SlabIO(path, mode='a')                # H5Fopen AGAIN
# ... many writes through zeta_io ...
zeta_io.close()                                  # H5Fclose AGAIN
```

Each `H5Fcreate` / `H5Fopen` / `H5Fclose` is an MPI-IO collective.
Two open/close cycles, ~1.9 s total at MoS2 3×3 (including the first-
call MPI_Init_thread cost of ~400 ms).

**Fix for the FFI path**: a single `SlabIO(mode='w')` that stays open
for the subsequent write loop. Shared `_sm_cache` on the same backend.
~220 ms saved (just the second open/close; MPI_Init was hoisted
separately — see 2.6). The allgather backend keeps its original
create-then-reopen pattern because rank-0 h5py doesn't need a
long-lived collective handle.

### 2.6. Eager `MPI_Init_thread` at program startup

Unrelated to the FFI itself but turned up in the open audit: the first
`MPI_Init_thread(THREAD_MULTIPLE)` costs ~400 ms (measured on Perlmutter
HPC-X OpenMPI; Cray MPICH is the same order). Previously paid by the
first `open_file` inside `zeta_fit_chunked`, on the critical path.

**Fix**: `lrx_phdf5_init_mpi()` extern-C entry + Python wrapper + eager
call in `gw_jax.main()` right after mesh setup, before any jit. 400 ms
moves off the critical path and overlaps with JAX compile. Same trick
applies for SLATE (`lrx_slate_init_mpi()`).

### 2.7. File path must be identical on every rank for `H5Fcreate`

Early integration bug that deadlocks in an insidious way: every rank
independently generated its HDF5 path with `tempfile.mktemp(...)`.
Each rank got a distinct `/tmp/foo_<pid>.h5`. The collective
`H5Fcreate` then hung — each rank tried to create a *different*
file in its MPI-IO group.

**Fix**: broadcast the path from rank 0 to the others before
`SlabIO.__enter__`. `ffi.common.broadcast.broadcast_bytes` does this
via the JAX KV store. See `phdf5_write_test.py` for the pattern.

**Rule**: any collective HDF5 op takes the filename as input. Paths
must agree across all ranks. If any rank-local state (`os.getpid()`,
`tempfile.mkstemp()`, etc.) feeds into the path, you need a broadcast.

### 2.8. Persistent compile cache was silently broken

Not an FFI issue per se, but turned up in the audit. The LORRAX
codebase had `_ensure_compilation_cache()` in `w_isdf.py` using the
old `jax.experimental.compilation_cache.set_cache_dir` API — which
is soft-deprecated on recent jaxlib and **silently no-ops**. The
cache directory stayed at 0 entries across runs. Also activated
only inside `w_isdf` / `ppm_sigma`, so `zeta_fit_chunked` and
`V_q_compute` (the two most compile-heavy regions) never benefited.

**Fix**: `common/jax_compile_cache.py` with the modern
`jax.config.update('jax_compilation_cache_dir', …)` API, activated
from each driver's `main()` before any jit. Cache dir partitioned
by `np{n_proc}` so a 1-GPU debug run doesn't collide with 4-GPU
prod. Opt-out via `ISDF_JAX_CACHE_DIR=""`.

Warm-run savings:
- `gw.gw_jax`: 24.4 s → 20.0 s (−4.3 s)
- `psp.run_nscf`: 19.6 s → 18.4 s (−1.2 s, by warm run 2)

Cache footprint: ~3 MB total across drivers for a MoS2 3×3 run.

------------------------------------------------------------------------

## 3. Sequenced performance wins over the investigation

Commits on `agent/C-sigma-ppm-cleanup`, MoS2 3×3 total gw_jax wall:

| commit | change | Total |
|---|---|---:|
| pre-session `async broken`    | ffi::Future + per-call event                | 35.2 s |
| `5eb6293`                      | pooled `ctx->d2h_event` (the mystery)      | 25.1 s |
| `0a41f42`                      | single SlabIO + eager MPI_Init             | 24.8 s |
| `e31970d`                      | writes: `offset_base` Buffer + jit-wrap    | 24.7 s |
| `a5e404f`                      | reads: same                                 | 24.0 s |
| `6169e1b` / `e380ec8`          | persistent compile cache fix + audit        | 20.0 s *warm* |

Total improvement from the initial async-broken state: **35.2 s →
20.0 s = −43 %**. vs a sync-FFI baseline (no async, no cache):
**25.3 s → 20.0 s = −21 %**. `eqp0.dat` bit-identical at every step.

------------------------------------------------------------------------

## 4. Knobs, with defaults

Environment variables, loosely in order of how often you'd touch them:

| var | default | effect |
|---|---|---|
| `ISDF_JAX_CACHE_DIR` | `~/.cache/isdf_jax_compilation` | persistent compile-cache path; set to `""` to disable |
| `LORRAX_PHDF5_INDEPENDENT` | `0` | if `1`, also force **reads** to independent (writes are already independent by default) — rarely helpful on OpenMPI; neutral on Cray |
| `LORRAX_PHDF5_COLLECTIVE_WRITES` | `0` | if `1`, force writes back to collective — **do not set on Cray MPICH** (triggers `ad_cray_write_coll.c:669` OOM at ≥ 1 GB/rank); on OpenMPI the default independent write path matches collective within noise |
| `LORRAX_PHDF5_COLL_META` | `0` | if `1`, re-enable collective metadata ops — default is non-collective (faster everywhere: +100 ms on OpenMPI small writes, required on Cray for n ≥ 16384 C128 writes) |
| `LORRAX_PHDF5_ALIGN_MB` | `4` | H5Pset_alignment threshold/stride in MiB; `0` disables |
| `LORRAX_PHDF5_STRIPE_COUNT` | `16` | Lustre stripe count hint via MPI_Info |
| `LORRAX_PHDF5_STRIPE_SIZE` | `4194304` | stripe size (bytes) via MPI_Info |
| `LORRAX_PHDF5_CB_NODES` | `world_size` | ROMIO aggregator count |
| `LORRAX_PHDF5_CB_BUFFER_SIZE` | `67108864` (64 MiB) | ROMIO collective buffer per aggregator |
| `LORRAX_PHDF5_CB_WRITE` | `enable` | ROMIO collective buffering on writes |
| `LORRAX_PHDF5_DS_WRITE` | `disable` | ROMIO "data sieving" on writes |

At MoS2 3×3 / 4 GPU we tried the full MPI-IO tuning sweep: all flat
within ±200 ms. At larger scale these will matter more.

------------------------------------------------------------------------

## 5. Testing

| test | coverage | `-m <target>` |
|---|---|---|
| `phdf5_write_test` | single-offset 4-GPU write + serial + collective read round-trip | `common.phdf5_write_test` |
| `phdf5_multi_offset_test` | **multi-offset write + read round-trip — guards the runtime-offset code path** (otherwise a regression would only surface at zeta scale) | `common.phdf5_multi_offset_test` |
| `phdf5_read_bench` | read-bandwidth bench | `common.phdf5_read_bench` |
| `phdf5_loop_test` | overlap / pipelining sanity | `common.phdf5_loop_test` |
| `pytest -q` (top-level) | 14 LORRAX-wide correctness tests; passes throughout | — |

Run locally with `lxalloc` + `lxrun` (module's defaults cover the
full stack env):

```bash
lxalloc
lxrun python3 -u -m common.phdf5_multi_offset_test
```

------------------------------------------------------------------------

## 6. What's *not* done, and why — for honest triage later

1. **Double-buffered writes**: pool of 2 pinned buffers to pipeline
   the D2H of write N+1 with the H5Dwrite of write N. ~90 ms gain
   at current scale, changes the Python-worker serialisation
   contract. Not worth disturbing the single-buffer invariants.

2. **V_q_compute read prefetch**: issue read N+1 while processing
   read N. Requires refactoring `compute_vcoul.py`'s inner loop.
   The reads are bounded by H5Dread collective rendezvous anyway,
   so the prefetch mostly hides ~50 ms of H2D per read. Low ROI
   for the blast radius.

3. **GPU Direct Storage (cuFile VFD)**: skip D2H entirely, write
   direct GPU → Lustre. Requires HDF5 build with GDS enabled and
   a site that supports it (Perlmutter does on some FSs). Worth
   revisiting when per-rank shards grow ≥10×.

4. **Queue backpressure**: `queue.Queue(maxsize=N)` on the Python
   worker queue. At MoS2 3×3 the depth stays ≤ 4; at larger chunk
   counts the in-flight `jax.Array` refs on device could pressure
   HBM. Cheap fix; add when it bites.

5. **HDF5 async VOL connector**: an upstream equivalent to our
   custom bg writer thread. Same architecture, more packaging,
   extra dependency to vet. Keep a note for if/when we want to
   align with the HDF5 ecosystem.

6. **ALLOC_TIME_EARLY ablation**: currently enabled in `ctx->dcpl_id`
   alongside `FILL_TIME_NEVER` to avoid implicit zero-fill. At the
   largest dataset sizes the early alloc (file-extent reservation)
   may itself cost noticeable wall. Worth ablating at e.g. MoS2 6×6.

------------------------------------------------------------------------

## 7. Files, so the mental model is grounded

```
src/ffi/phdf5/
├── ARCHITECTURE.md        (this file)
├── __init__.py            (re-exports write_sharded_slab, read_sharded_slab, open_file, close_file)
├── context.py             (Python-side open/close wrappers)
├── write.py               (ffi_write_call, write_sharded_slab)
├── read.py                (ffi_read_call, read_sharded_slab)
├── cpp/
│   ├── ctx.h              (PhdfCtx: fapl/dxpl plists, pooled events, writer_thread, task_queue)
│   ├── context.cc         (open_ctx, close_ctx, ensure_dataset, ensure_mpi_initialized)
│   ├── write_ffi.cc       (WriteDispatch — runtime offset, async H5Dwrite task)
│   ├── read_ffi.cc        (ReadDispatch — runtime offset, sync H5Dread + async H2D)
│   └── phdf5_interface.h  (HDF5 dtype helpers)
└── scripts/
    ├── stage_cray.sh      (copy cray-hdf5-parallel + MPICH-ABI shim to $SCRATCH)
    └── stage_openmpi.sh   (copy conda-forge HDF5+OpenMPI to $SCRATCH)

src/file_io/_slab_io_ffi.py         (_FfiBackend: Python worker thread, _sm_cache, wraps shard_map+jit)
src/file_io/slab_io.py              (unified SlabIO front-end: dispatches to FFI or rank-0-allgather)
src/common/jax_compile_cache.py     (shared JAX persistent compile-cache activator)

src/common/phdf5_write_test.py          (single-offset smoke)
src/common/phdf5_multi_offset_test.py   (multi-offset smoke — guards §2.3 regression)
src/common/phdf5_read_bench.py          (read bandwidth bench)
src/common/phdf5_loop_test.py           (pipelining/overlap bench)
```

## 8. Anti-patterns — things you'll probably want to do and shouldn't

- ❌ `cudaEventCreate` + `cudaEventDestroy` per call in the handler.
  Use `ctx->d2h_event` / `ctx->h2d_event`. See §2.2.
- ❌ `Attr<Span<int64_t>>("per_call_thing")` for anything that varies
  per invocation. Use `Arg<Buffer<S64>>`. See §2.3.
- ❌ `shard_map(_per_rank, …)(A)` in a hot path without `jax.jit`
  wrap and without an outer cache. You'll get N distinct HLO modules
  for N calls. See §2.4.
- ❌ Two `SlabIO` handles on the same file in the FFI path
  (create-then-append). Use one. See §2.5.
- ❌ Rank-local path generation for collective `H5Fcreate`. Broadcast
  the path. See §2.7.
- ❌ `jax.experimental.compilation_cache.set_cache_dir(...)`. Use
  `jax.config.update('jax_compilation_cache_dir', ...)`. See §2.8.

If you're about to do any of these, read the corresponding section
first; they're non-obvious and each one cost us measurable time to
figure out.

# Host–device synchronization and transfers

This page owns the rules for when host code may wait on a device, read a device
value, or place a host value on a device. It also owns the tools that count
those events and the switch that makes stage clocks wait. It does not own the
SlabIO transport ([SlabIO](slab_io.md)), the reduction order of collectives
([Determinism](determinism.md)) or the compile cache ([Compilation](compilation.md)).

## 0 Why it matters {#why}

A host wait drains the stream. Until the host has the value, nothing behind it is
enqueued, so the device idles for the round trip plus the relaunch. One such
round trip costs ≈14 µs on MI250X: the `lb2.py` loop micro-benchmark, a traced
`fori` bound against a static one, gave 30.1 against 16.5 µs per trip (lane
lsync, `lsync_lb2a.out`). One trip is cheap. What costs is having one per
window, per eigh or per read on a path that runs thousands of times. A host
value placed on the device is the mirror case: it is one transfer per call, and
it is also a host value the program keys on.

## 1 The rules {#rules}

1. **No host decision on a device value in a hot path.** Decide on the device
   (a mask, a `where`, a count carried as an array), or read one small word per
   stage boundary that carries every bit the host needs. Never read the bulk
   array to compute the decision on the host ([§6](#device-decisions)).
2. **No `device_put` in a hot path.** A control operand that is a pure function
   of host plan data is placed once per distinct value
   ([placement cache, §5](#placement-cache)) or computed in the program from a
   device-carried counter. Host payloads enter through SlabIO or a sanctioned
   placement, never as jit arguments.
3. **Static trip counts where the count is static.** XLA skips the predicate
   read of a loop with a known trip count. A loop whose bound is device data
   reads its predicate to the host on every trip, and every executed `cond`
   reads its predicate once, on **both** legs ([§7](#loops)).
4. **Timing-only waits are opt-in.** A `block_until_ready` whose only purpose is
   to charge device work to a stage row runs under `LORRAX_STAGE_SYNC=1` only
   ([§3](#stage-sync)). A wait that a host decision needs (a value read, a buffer
   reused, memory released to a ledger, an HDF5 call that must not overlap)
   stays unconditional at its site, and its comment names the decision.
5. **A program a host decision needs is compiled on every rank.** The
   cross-rank compile agreement (`GATE cross_rank_compile_agreement`,
   INVARIANTS 21) refuses a program only some ranks compile. A word that only
   rank 0 prints is still computed and read on every rank, each on its own
   replica (`distrib_la._result_check._read_status`).
6. **No new resident copies.** A batched read, a deferred verdict or a cached
   placement holds kilobytes, transiently or in a bounded cache. It never keeps
   a second layout or an operand-sized buffer alive, and it never lowers the
   largest calculation that fits on a fixed number of GPUs. A change reports the
   per-rank device peak of any new path against its baseline.

## 2 The census tool {#census}

`tools/transfer_census.py` counts, per rank, every host materialization of a
`jax.Array`, every host-to-device placement, every explicit wait, every compile
and every Python warning. It keys each event by stage path
(`common.timing.current_path()`) and the innermost LORRAX frames. Nothing in
`src/` imports it.

```bash
python tools/transfer_census.py run OUT_DIR SCRIPT.py [args...]   # or: run OUT_DIR -m gw.gw_jax -i cohsex.in
python tools/transfer_census.py summarize OUT_DIR --rank 0        # -> OUT_DIR/census_rank0.md
```

| kind | what it counts |
|---|---|
| `d2h`, `d2h_explicit` | `ArrayImpl._value`: `np.asarray`, `float`/`bool`/`int`, `device_get` of an array with no cached host copy |
| `h2d_device_put`, `h2d_jnp_array`, `h2d_jit_arg` | numpy entries of `pxla.shard_arg_handlers`, classified by the JAX frames on the stack |
| `sync:block_until_ready`, `sync:jax.block_until_ready` | the array method and the function |
| `compile:*` | `jax_log_compiles` records: trace, executable-cache miss, persistent-cache hit |

The hooks are JAX 0.9.1's. `XCENSUS_NO_FASTPATH=1` (the default) routes every jit
call through Python dispatch, so repeated host arguments are seen; results stay
bit-identical and the wall grows ~2–3 %, so a census wall is not a stage wall.
JAX 0.9.1 does not read `JAX_TRANSFER_GUARD_*` from the environment;
`XCENSUS_GUARD=log` sets the C++ guard through `jax.config`, as a count
cross-check. The census does not see `make_array_from_callback`, XLA's own While
and cond predicate reads, or syncs inside FFI handlers. Count those with a
vendor API trace (rocprofv3 on ROCm; on CUDA, nsys with one NVTX range per
stage).

Fe V2, P=4, rank 0, int6 → int7 (INT7 release notes; `xsync/cen2` → `int7/cen5`):

| kind | int6 | int7 |
|---|---:|---:|
| H2D `device_put` / `jit_arg` / `jnp_array` | 7545 / 657 / 217 | 713 / 63 / 160 |
| D2H / explicit D2H | 2089 / 461 | 425 / 71 |
| `block_until_ready` (array / `jax.`; the `jax.` count is timing-dependent, 6–17 per run) | 907 / 117 | 119 / 6 |
| census wall (s) | 133.3 | 95.4 |

## 3 `LORRAX_STAGE_SYNC` {#stage-sync}

`common.timing.stage_sync()` reads `LORRAX_STAGE_SYNC` once per process (*bool*,
default off). Off is production. No timing-only wait runs. A section's clock
stops when its host work, dispatch included, returns. Device work it enqueued
is charged to the first later section that needs a value from it. **The run's
total wall is exact; a stage row is host-attributed.** With `=1`, every
timing-only wait runs again: the section watchers (`timed(watch=True)`,
`TimingSection.watch`), the `common.timing.stage_wait` boundaries and the
progress milestones. Each row is then its stage's own device time, which is
the setting for profiling. The stage table prints which attribution is in
force. `fence` and `fenced_section` stay unconditional: they are explicit
profiling boundaries.

On Fe V2 P=4 the switch took explicit waits from 1024 to 126 per rank
(907 + 117 → 119 + 7; `es/ab3/bcen` → `es/ab4/ncen`). eqp was byte-identical on
the same node, with walls 106.98 → 106.94 s. The waits were cheap; the rule
exists so that later ones are not added.

`common.collectives.barrier` orders host work only (rank-0 HDF5 writes, closed
readers, timing bands). It uses the coordination-service barrier
(`wait_at_barrier`, one id per use, 1800 s deadline), not a device all-gather.
Without a distributed client it falls back to the device barrier and prints
`GATE barrier_coordination` once.

## 4 `transfer_guard` gates {#transfer-guard}

A conversion that claims "no transfer here" is proved by running the stage under
`jax.transfer_guard("disallow")`: any implicit or explicit transfer then raises
at its site. For a hot loop body, `"disallow_explicit"` admits the planned
implicit ones and refuses the rest. The gate is a harness around named
functions, not a pytest:

| scope | guard | status (Fe V2 P=4) |
|---|---|---|
| `response_bank.produce_sample_bank`, `_integrate_sigma_batches` | `disallow` | passes, eqp exact (dput/w2 g5b; dput/w3 g8a with `LORRAX_CHECK_COUNTS=1`) |
| Σ per-window `window_pick`, `run_window`, `window_arguments`, selector | `disallow_explicit` | passes (same runs) |
| `construct_shared_poles` | `disallow` | trips on the per-round round table (dput/w2 g5d) |

## 5 The placement cache {#placement-cache}

`lxkit.placement.device_put_process_local_cached(host_array, sharding)` places a
small host control value once per distinct (sharding, dtype, shape, bytes) and
returns the same device array afterwards. `common.collectives` and
`distrib_la._collectives` re-export it.
- Entries are ≤ 64 KiB; larger values take the uncached `device_put_process_local`.
- The cache is a bounded LRU (≤ 8 MiB per device) and thread-safe. Device
  arrays and tracers pass through.
- **Never donate the result.** It is shared; JAX refuses a deleted buffer by
  name.
- `placement_cache_stats()` reports hits, misses and peak bytes. On Fe the
  cache holds about 32 KB per device.

`device_put_process_local` itself is the sanctioned placement of a
process-replicated host table. It declares each process's shards instead of
letting `jax.device_put` assert cross-process equality with an all-gather
(`LORRAX_CHECK_REPLICA=1` restores the assertion).

## 6 Decisions on the device, one word to the host {#device-decisions}

| site | what the host needs | how it gets it |
|---|---|---|
| checked solves (`distrib_la._result_check`) | failed? any notice to print? | `_read_status`: one int32 word `[failed, notice_0, …]` per status, on every rank. The error table is read only when a notice prints. |
| the reciprocity census (`response_bank`) | receipt rows | scalars kept on the device and read once, at `_finish_receipt` |
| the shared-pole model digest (`shared_pole_store._model_digest`) | refuse if any panel is invalid; hash bytes | per-panel verdicts read once after the loop; one `psum_replicate` of all q-batches' row hashes |
| the Coulomb sub-span start, Σ window tables, omega-slab starts | loop control | device-carried counters (dput/w2, dput/w3) |
| the bank's rule pencils | the shift pencil of a geometry | one read per consumed geometry, the next one prefetched ([shared-pole model §2](shared_pole_model.md)) |

Fe V2 P=4, rank 0: checked-solve
status 164 → 82; the reciprocity census 64 → 1; digest verdicts and reductions
33 + 18 → 2 + 3. eqp was byte-identical in every case (lsync D2H
report). The counts that remain set static shapes of the next programs, so the
host must have them.

## 7 Loops and conditionals, both legs {#loops}

XLA's GPU thunks are the same on both legs (XLA 3cc8846c, the jax-v0.9.1 pin).
`WhileThunk` with a known trip count loops on the host without reading the
predicate. Otherwise every trip copies the predicate to the host and calls
`BlockHostUntilDone`. `ConditionalThunk` does the same once per execution.

Command buffers (CUDA graphs) do not change this by default. The default
`xla_gpu_enable_command_buffer` at that pin is FUSION, CUBLAS, CUBLASLT,
CUSTOM_CALL, CUDNN, DYNAMIC_SLICE_FUSION: WHILE and CONDITIONAL are not in it
(`xla/debug_options_flags.cc`; checked on the installed jaxlib 0.9.1 through
the dumped `.debug_options`). No LORRAX FFI handler declares command-buffer
compatibility, so a LORRAX custom call also ends a captured graph. On CUDA the
loop predicates therefore go to the host exactly as on ROCm, unless a deck adds
WHILE and CONDITIONAL to the flag and the body is capturable.

Fe V2 measured at P=1 on ROCm (rocprofv3, lane lsync):
- The Σ window node loop: 1650 τ nodes, 1676 DtoH / 1664 syncs, 1.01 per trip.
  A static count removed them and saved nothing, because the ceiling is
  1650 × 14 µs ≈ 0.02 s.
- The χ₀ contour accumulate: its output blocks are the one device-bounded loop
  and its groups an inner loop of static trip count
  (`gw.contour_accumulator.contour_block_accumulate_local`). That took χ₀
  direct dispatch from 12 930 to 5 798 syncs and the whole run from 27 987 to
  20 561 stream syncs, bit for bit.
- Removing a `cond` around heavy work made χ₀ slower (+4.2 s at P=1). A cond
  costs one round trip; computing the branch it skips can cost more. Remove
  conds that guard cheap work, not every cond.

CUDA pairs two τ nodes per Σ window trip (`gw.ppm_accumulators.WINDOW_PAIRING`),
so its node loop makes half the trips. ROCm keeps one node per trip because
async collectives are disabled there.

## 8 SlabIO {#slab-io}

SlabIO's reads and writes run on its collective lane. The rules above apply
there in this form ([SlabIO](slab_io.md#api)):
- `read_slab` and `read_multi` return before the read lands. They register
  with `_CollectiveLane.track_read`, and the next HDF5 entry point of any handle
  waits only for reads still in flight (`_finish_reads`: `is_ready()` filter).
  That wait serves HDF5's no-overlap rule. Its count depends on timing (2 to
  12 per Fe run).
- `read_multi`: N requests in one `PhdfReadMulti` dispatch, with no Python host
  sync. Fe constructor host syncs went 836 → 292 and `sample_batch_read`
  2.80–2.87 → 2.12–2.16 s (slabmr/w8).
- The phdf5 handlers copy their small control vectors to the host with one D2H
  and an `xla_stream` sync per call (`phdf5/platform_seam.h`
  `copy_index_to_host`, both device legs).
- The ≥ 1 GiB-file read heartbeat polls `is_ready()` on its own thread and never
  waits on the caller.

## 9 Leg-specific host reads {#legs}

| site | CUDA | ROCm |
|---|---|---|
| Σ active-range GEMM (`ACTIVE_LOCAL_ROUTE`) | `native`: bounds D2H + stream sync per call (`cublas/local_active_gemm_ffi.cc`), two guard conds per call | `device`: bounds read on the device, 0 ([ROCm leg](rocm_leg.md#active-gemm)) |
| distributed eigh / LU at small n (`distrib_la.resolve`) | cuSOLVERMp at every n: `info` D2H + sync per call, per stack member in the batched handlers | one-device route (eigh n ≤ 16384, LU n ≤ 8192); no in-handler sync |
| active subspace (`NATIVE_HANDLERS`) | `blas`: descriptor D2H + sync per call | `devsize`: descriptors read on the device |

# SlabIO — the sharded-slab HDF5 transport

*Verified against `src/` and against measurements taken inside the Shifter
container on Perlmutter, 2026-08-05/06 (allocations 56389339, 56405696,
4 × A100-40G nodes), and re-verified 2026-08-15 on
`integ/metal-mpa-qsgw-2026-08-15` against the metallic MPA-QSGW campaign
(JIDs 57031206, 57038615) for everything in
[One HDF5 library per file](#one-owner), [The operation
journal](#journal) and [Failure modes](#failures). Where this page and
older prose disagree, this page wins. Every number below names the command
that produced it and the filesystem it ran on.*

**This page owns SlabIO**: the tile contract, the caller-facing API, the
**one-owner-per-file rule and the refusal that enforces it**, the launcher
requirement, the striping measurements, the certification, and the three
measured HDF5 failure signatures. It does not own the owner rulings behind
them ([`decisions.md`](decisions.md)), the native layer underneath
([`ffi_layout.md`](ffi_layout.md)), or knob spellings
([`../dev/env_vars.md`](../dev/env_vars.md)) — see the
[register](../index.md#register).

Conventions used here, same as [`services.md`](services.md):

- **Level / imports.** `file_io.slab_io` is L3 and imports only downhill.
  It used to reach *up* into `gw.gw_config` for the backend enum; that
  import is gone with the enum (2026-08-06).
- **Announce-or-refuse.** A deployment that cannot run says so, naming the
  probe that declined. Nothing about the I/O path is ever silent.
- **Shapes are logical.** Files store the logical shape; padding for mesh
  divisibility never reaches disk.

---

## The contract {#contract}

**One rank writes one tile. Nothing larger than one rank's tile is ever
materialised.**

That is the whole design constraint, and it is a hard one, not a
preference. The production envelope is arrays that need hundreds of GPUs
to hold at all — ζ(q, μ, G) and V_q for a converged cell do not fit in one
node's host RAM, let alone one device's. Any code path that gathers a
whole array to one rank is therefore not a slow path, it is a path that
**cannot run the workload it exists for**.

The owner's statement of it, verbatim:

> there should always exist a valid path that does not materialize
> N_mu^2 on any proc because it is a guaranteed OOM.

Consequences that follow directly, and that callers must not work around:

- `write_slab` takes the array the rank already owns, in its existing
  sharding. It does not reshard, gather, or replicate.
- `read_slab` returns a sharded array. The rank reads its own hyperslab
  and no one else's.
- Peak host memory attributable to SlabIO is one rank's tile plus the
  pinned staging buffer, independent of process count. **Read
  "attributable to SlabIO" strictly: it is the staging above whatever the
  caller asked for.** On the collective transport the two nearly coincide,
  because a rank's result *is* one tile. On the emulated-mesh tier they do
  not: one process holds every shard, so the returned array is
  global-sized by construction and only the staging — measured at one
  shard — is this layer's. A number quoted for that tier has to say which
  of the two it is; `file_io._slab_io_serial`'s docstring carries both,
  with the probe and its scope.

**Since 2026-08-06 the contract is enforced by construction rather than by
a check: there is one transport, so there is nothing else to select.**
One geometry is served by a second backend and it is still not a choice:
on an EMULATED mesh (`common.collectives.mesh_is_emulated` — `P == 1` with
more mesh cells than processes) `ffi.io.open_file` refuses, correctly, and
`SlabIO` constructs `file_io._slab_io_serial._SerialBackend` instead. It
moves one shard at a time, takes no argument, refuses above one process,
and announces itself; see that module's docstring for why it is not the
tier deleted below.

### What a caller may assume, and what it may not {#may-assume}

The tile rule above is the *design* constraint. This is the *call-site*
contract — every row is something a caller has actually got wrong.

**A caller MAY assume:**

| assumption | enforced by |
|---|---|
| logical shapes reach disk; padding never does | `write_slab` derives the written extent from the dataset ([API](#api)) |
| `read_slab` with no `shape` returns a legal, mesh-divisible, zero-filled buffer | `_FfiBackend.padded_shape_for` |
| a deployment that cannot serve the tile path REFUSES at open, naming the probe | `assert_available()` ([Availability](#availability)) |
| the same logical array is byte-identical on disk through any mesh geometry | [Certification](#certification), within-platform md5 match |
| a refusal computed from the operands is raised on **every** rank, not one | every `fail`/`fail_read` path in `read_ffi.cc` / `write_ffi.cc` is computed from replicated inputs |

**A caller MAY NOT assume:**

| ...that | because |
|---|---|
| it can hold an h5py handle and a `SlabIO` handle on the same file at once | **REFUSED by name.** Two HDF5 library instances are mapped in this process; either side being a writer is undefined. See [One HDF5 library per file](#one-owner) |
| a scalar (rank-0) dataset can be read with `read_slab` | it cannot, structurally: a scalar dataspace has **no hyperslab**, and the request refuses at `_normalize_slab_request` before HDF5 sees it. `read_small` is the door — see [The API](#api) |
| program order at the Python call site serializes two HDF5 calls | the FFI writer is **asynchronous**. Two handles on the same ranks need `SlabIO.sync_writes()` between them, or the second enters HDF5 while the first is still draining |
| a `create_dataset(attrs=…)` or `stamp_dataset_attrs` stamp is readable before `close()` | attrs are **deferred** to one rank-0 h5py reopen after `H5Fclose` — the transport cannot stamp while collective MPI-IO holds the file |
| `close()` is a local operation | it drains pending writes on every rank, then rank 0 reopens the file serially. `close()` is where three of this page's failure modes surface |
| the caller may close, free, or reuse the underlying `PhdfCtx` | it does not own it. `SlabIO` opens it in `__init__` and closes it in `close()`; there is no other lifetime |
| a `SlabIO` object is re-openable, thread-safe, or fork-safe | it is a single collective handle over one mesh. Use `with`, one handle at a time, and re-construct rather than reopen |
| control operands (offset, valid_shape, handle) are safe to pass as traced device values | true **only since the stream fix** — see [S3](#s3) for what happened before it, and note the host leg was never affected |

**Collectivity, stated once.** `create_dataset`, `write_slab`, `read_slab`,
`read_slabs`, `sync_writes` and `close` are **collective over the mesh
passed to the constructor** and must be called by every rank in the same
order with the same dataset name. `write_attr` and `stamp_dataset_attrs`
are the two exceptions: they queue rank-replicated metadata that only rank
0 writes, at close. There is no per-rank subset call in this API, and a
rank that skips one call deadlocks the rest with no traceback
(`ffi_layout.md` §7d).

---

## One HDF5 library per file {#one-owner}

*Measured 2026-08-15, JID 57038615, metallic MPA-QSGW on 4 A100. On branch
`integ/metal-mpa-qsgw-2026-08-15` as `c1563a25` + `81c99c95`. Owner:
`src/file_io/hdf5_owner.py`; hazard class recorded by the sandbox as
claims/0110, by the audits as A1.*

**A LORRAX process maps two independent HDF5 library instances.** Read off
the deployed artifacts' `NEEDED` entries and confirmed at run time from
`/proc/self/maps`:

| instance | who pulls it in | generation |
|---|---|---|
| `h5py.libs/libhdf5-9e18f0c6.so.320.0.0` | the h5py wheel (3.16.0), serial | HDF5 **2.0.0** |
| `/opt/cray/pe/lib64/libhdf5_parallel_gnu_123.so.200` | `NEEDED` of `liblorrax_ffi.so` | HDF5 **1.12.x**, parallel |

Different sonames, so both load, and **neither is a mistake to be
removed** — SlabIO needs the parallel build, and h5py is how every small
metadata read in this tree happens. What is not allowed is one *file*
reaching both.

### The condition, stated precisely {#metadata-cache-condition}

Each instance keeps its **own metadata cache, its own open-file table and
its own free-space manager**, and neither can see the other's. That is the
whole mechanism. Three consequences, in increasing order of nastiness:

1. **Concurrent, either side writing → undefined.** One instance's dirty
   cache against the other's stale one. The writer's flush lands on top of
   metadata that never saw the other's changes. Measured symptom on this
   stack: h5py reading "file signature not found" off a superblock that is
   not durable yet (job 7888644).
2. **Concurrent, both read-only → safe.** Two libraries reading a file
   nobody is mutating cannot diverge. `wfn_loader` holds an h5py handle
   and a SlabIO handle on `WFN.h5` read-only for the whole run. It is
   legal *and it is counted*.

   `_FfiBackend._introspect_dataset` used to be the second example here,
   and **it no longer is** (2026-08-22). It asks the FFI for a dataset's
   geometry (`lrx_phdf5_dataset_geometry`), i.e. the library that already
   holds the file, so there is no second instance to count. That mattered
   for more than tidiness: on a handle opened `'a'` the h5py introspect is
   not merely counted, it is **refused** — correctly — and a real caller
   walked into it (`--parallel-transport-out`, [below](#pt-introspect)).
   A library built before that date exports no geometry entry point; the
   old route is then taken on a read-only handle, **announced once per
   path on rank 0**, and refused by name on a writable one.
3. **Sequential alternation with a write → survives, mostly.** Every close
   flushes, so open/close/open/close through different libraries usually
   works. It is still two caches taking turns on one file, it is the
   condition the crash signatures correlate with, and it is what audit A1
   wants gone.

**The pairing is cross-MAJOR (2.0.0 against 1.12.x), which is not a
detail:** the superblock file-consistency flags are exactly what changed
across that boundary. The `env_prelude_v2` experiment exists to narrow
that to 1.14.6-against-1.12.x, and `v3` moved the FFI leg to 1.14.3.7.

### The measurement — 1027 alternations on one file, in one iteration {#alternations}

The driver measures what is mapped and what has been shared at startup and
after every SC store cycle (`hdf5_owner.probe`). Safe probes are silent in a
production run; an `UNSAFE-BY-A1` condition is always printed. Set
`LORRAX_H5_JOURNAL=1` to print the full inventory as well. Diagnostic output
from JID 57038615, iteration 2:

```
[hdf5-probe startup] libhdf5 2 core mapped: libhdf5-9e18f0c6.so.320.0.0,
  libhdf5_parallel_gnu_123.so.200 (+1 companion:
  libhdf5_hl-69da89c9.so.320.0.0); HDF5_USE_FILE_LOCKING=FALSE
[hdf5-probe sc_0002] 11 path(s) opened through BOTH stacks, 7 of them
  with a write; worst: mpa_samples_sc_0000.h5 (h5py 533 opens, FFI 514,
  1027 alternations)
[hdf5-probe sc_0002] UNSAFE-BY-A1 ... Sequential only (no live overlap
  was allowed), which is why the run proceeds
```

**1027 cross-library alternations on ONE file in ONE iteration**, against
the static audit's estimate of ~25 — the audit understated it by ~50×,
and it understated it because it counted the Σ-side reads and missed the
former *writer*: `write_fit_block_collective` did five opens per block (four
of them h5py: ledger, `diagnostic_keys`, the rank-0 commit, the ledger again),
and the R6 deck ran 464 blocks per iteration. The dual-stack written-file
count also grew every iteration (3 → 5 → 7), which is the "gets worse with
iteration count" property A1 predicted.

The production fit walk now uses `mpa_store.FitWriter`: one collective
payload handle across every block, three dataset handles (`Omega_p`, `B_p`,
and `fit_condition`) opened before the first transfer, and one bulk ledger
commit only after payload close.  The three writes of each block drain before
the next W-sample read; this is the required
cross-handle ownership transfer from the asynchronous writer, not a file
reopen.  The single-block helper retains its one-block transaction for
surgical resume callers; it is not the production loop.  This removes the
measured per-block alternation mechanism without changing the stripe policy
or holding pole tensors across blocks.  Backward error remains a certification
metric: its per-block maximum is retained in the small ledger, while its full
element map, sample residual and valid-pole count are deliberately ephemeral.

The probe counts **library instances, not shared objects**: `libhdf5_hl`
and friends are thin wrappers over the core object they were built
against, and counting them would make a perfectly safe single-stack
process report two (`81c99c95`).

### What the registry enforces, and the refusal's name {#registry}

`file_io.hdf5_owner` is a per-process, in-memory registry — not a lock,
and it says nothing about other ranks. It answers the one question neither
library can answer for itself: *has this path already been opened through
the other HDF5 library in THIS process, and is that handle still live?*

| condition | verdict |
|---|---|
| cross-stack **live overlap** where either side can write | **REFUSED, always**, by `note_open` — *"LORRAX HDF5 one-owner-per-file refusal (audit A1; sandbox claims/0110)"*, naming the file, the incoming stack and mode, every live foreign handle with its `where=`, and the fix |
| cross-stack live overlap, **all read-only** | allowed, counted |
| **sequential** cross-stack alternation after a write | counted; refused under `LORRAX_HDF5_ONE_OWNER=strict` as *"LORRAX HDF5 one-owner-per-file refusal, strict policy"* |
| two instances mapped **and** a file written through both | printed as `UNSAFE-BY-A1` every cycle; raises under `strict` from `probe` |

`LORRAX_HDF5_ONE_OWNER` is `measure` (default) or `strict`; any other
value is a refusal at `policy()`, not a silent fallback.

Every h5py open in `mpa_store` goes through one door, `mpa_store._h5`,
which declares to the registry, flushes on write modes and releases in a
`finally`. `SlabIO.close` declares its own claim and **releases it between
`H5Fclose` and the rank-0 deferred-attr reopen** — that reopen is the
other library touching the same path, legal precisely because MPI-IO has
let go, and a claim still held across it would make `close` refuse itself.

### The sibling split — kept, re-scoped {#sibling-split}

The permanent fix is one owner per file: give a payload `X.h5` (FFI only)
a sibling `X.meta.h5` (h5py only). The spec is
`reports/metal_mpa_plan_2026-08-15/METALLIC_INVARIANT_AUDIT.md`.

**Read the re-scoping before you schedule it.** The root-cause audit
proved S3 independent of h5py entirely, and S1/S2 most plausibly the same
stream race — so the split is *not* the fix for any of the three crash
signatures. It retires the cohabitation class and cuts the alternation
count, and its correct place in the queue is **after** the stream fix
([B1](#s3)) and the ctx-handle validation ([B2](#s1)). It is also **not
small**: `symmetry_maps.QirrDest`'s "a group, or a path to open one"
contract cannot honour two files from one group, and the paths this
process touches through both stacks are not only the MPA pair (`v_q_bispinor`'s
output, `downfold_run`'s tensors, `tagged_arrays`' restart file,
`parallel_transport.h5`, `sigma_mnk.h5`, `WFN.h5`, `isdf_fitting`'s zeta
file). **Done when `LORRAX_HDF5_ONE_OWNER=strict` runs a 3-iteration R6
clean** — that acceptance test exists today.

---

## The operation journal {#journal}

*Spec: `reports/metal_mpa_plan_2026-08-15/SLAB_IO_ROOT_CAUSE_AUDIT.md` §C.
Implementation `src/file_io/h5_journal.py`, on branch
`integ/metal-mpa-qsgw-2026-08-15` as `18bb48ea`. Knob spellings and defaults are the
[env registry](../dev/env_vars.md)'s; they are repeated here because a
debugger reading this page at 3am should not have to open a second one.
The fingerprints in [Failure modes](#failures) below name the lines it
emits.*

The journal is the **black box for a segfault**: one choke-pointed,
line-buffered, per-rank log of every HDF5 operation. It is **opt-in**:
normal production runs create only the consolidated driver log and physics
outputs, while an incident run sets `LORRAX_H5_JOURNAL=1`. Line-buffered
writes then survive the process, so a native death leaves a file current to
its last line, which is the one thing a Python traceback cannot give you
when rank 0 dies inside `close_ctx`.

| | |
|---|---|
| module | `src/file_io/h5_journal.py` |
| toggle | `LORRAX_H5_JOURNAL` — `0` off (**default**), `1` on, `sync` = fsync after every line |
| output dir | `LORRAX_H5_JOURNAL_DIR`, default the process's cwd (the run directory, for every LORRAX driver launch) |
| files | `h5_journal.rank<R>.log`, and `h5_journal_crash.rank<R>.txt` for the ring |
| crash ring | last 256 lines |
| overhead | ~1–2 µs per line; ~1030 HDF5 ops per SC iteration → order 1e-5 of a ~280 s iteration. `sync` costs ~1 ms/line — opt-in only |

**Line format** (frozen; fixed key order, so `awk`/`grep` on it is stable):

```
t=<monotonic.6f> rank=<R> stack={h5py|ffi} op={open|close|create|read|write|attr_r|attr_w}
  path=<abs> handle=<id-or-ctxptr> ds=<name|-> off=<t> cnt=<t> mode=<r|w|a|->
  owner=<live-verdict> rc={ok|refused:<first 40 chars>}
```

Two fields repay a second read:

* **`owner` is a state, not an outcome.** It is
  `hdf5_owner.live_verdict(path)` — which stacks hold a live handle on
  this path *at this instant* (`free`, `ffi:1w`, `h5py+ffi:2r`). The
  outcome of the registry's check is `rc`. Confusing the two is the
  easiest way to misread [S2](#s2).
* **`handle` is `-` on an `open` line.** The line is written **at issue,
  before the call**, which is exactly what makes the file a black box —
  a process that dies inside HDF5 leaves its last line naming the op that
  killed it. An open has not returned its handle yet, so the handle
  appears on the completion line the SlabIO seam writes next.

So one slab read or write is **one** line, and one `SlabIO` file open is
**three** — the registry's claim, the FFI's `H5Fopen`, and SlabIO's
completion line carrying the ctx handle. That is deliberate: opens are
where all three signatures live, and those are three different facts about
one open. An op that raises appends a second line with `rc=refused:…` and
dumps the crash ring.

**Hook points are existing choke points; no new seams.**
`hdf5_owner.note_open`/`note_close` (the one place that sees h5py and the
FFI in a single stream, whichever library is opening);
`SlabIO.__init__/close/create_dataset/write_slab/read_slab/read_slabs/
write_attr/stamp_dataset_attrs`; and `_slab_io_ffi`'s lifecycle calls plus
and its ONE remaining serial-h5py touch on an FFI-driven path
(`close`'s deferred-attr reopen; `_introspect_dataset`'s metadata read
left that list on 2026-08-22 and is journaled `stack=ffi` now).
Journaling is Python-side only — every FFI op transits `_slab_io_ffi.py` —
and the journal never journals itself.

The crash ring is dumped from the registry's refusal path, from any
exception crossing a `SlabIO` method, and from an `atexit` handler on
abnormal flags. Segfaults need nothing extra.

**The journal never takes a run down.** An I/O error on the journal file
disables it with one warning and lets the caller proceed: a log that kills
a 40-node job because its directory went read-only is worse than no log.
A bad `op`/`stack` spelling *does* refuse — those are in-tree constants,
and a mis-spelled op in a log is a fact nobody can recover later.

**How to read one at 3am:** `tail` the rank-0 file. The last line before a
native death names the file, the op and the ctx pointer. Then
`grep "handle=<that ctxptr>"` across all ranks: if the pointer appears
with two different `path=` values, or appears after its own
`op=close`, you are looking at [S1](#s1) route (ii) — a stale ctx — and
not at an HDF5 bug.

---

## The API — everything a caller must know {#api}

```python
from file_io.slab_io import SlabIO

with SlabIO(path, mode="w", mesh=mesh) as io:
    io.create_dataset("V_qmunu", shape=(n_q, n_mu, n_mu), dtype=c128)
    io.write_slab("V_qmunu", V)                    # V may be padded
    W = io.read_slab("V_qmunu", partition_spec=P(None, "x", "y"))
```

That is the entire surface. A caller states **a path, a mode, a mesh, and
logical shapes**. A caller does *not*:

| ...need to know | because |
|---|---|
| which backend | there is one |
| Lustre stripe count / size | resolved from the rank count by `_stripe_policy` and exported before `H5Fcreate`; see [Tuning](#tuning) |
| ROMIO / MPI-IO hints | set by the C++ context; read them back with `MPI_File_get_info`, never assume |
| whether the MPI world matches the JAX world | asserted at the first collective open, unconditionally ([Failure modes](#failures)) |
| the mesh-divisible extent | omit `shape` and `read_slab` rounds up for you (below) |
| whether the deployment can do parallel I/O | `assert_available()` runs at open and refuses naming the probe |
| an HDF5 chunk layout | SlabIO's native collective create is contiguous and exposes no no-op `chunks=` argument; use a format-specific serial writer only where chunking is load-bearing |
| a second HDF5 library, to learn a shape or read a stamp | `read_slab` with no `shape`, and `read_small`, both answer from the FFI (`lrx_phdf5_dataset_geometry` / `lrx_phdf5_read_whole`, 2026-08-22) |

**Padding is SlabIO's business, not the caller's** (decisions.md
2026-08-04):

- `write_slab(name, A, offset=...)` accepts any `A`. What reaches the file
  is `min(A.shape, dataset - offset)` per dim, derived from the dataset —
  a buffer padded for mesh divisibility needs no argument at all.
- `read_small(name)` reads a WHOLE small dataset into a host
  `np.ndarray` on **every** rank, through `lrx_phdf5_read_whole`. It is a
  separate method rather than a `shape=()` case of `read_slab` because a
  scalar dataspace has no hyperslab to select: `read_slab` refuses such a
  request before a byte moves. Every stamp `write_attr` publishes is a
  scalar or a short vector, and this is their reader. The payload must be
  O(1) in the design envelope — every rank materialises all of it.
  On a `.so` built before 2026-08-22 the entry point does not exist; on a
  **read-only** handle it then takes the announced serial-h5py fallback —
  the same one `_introspect_dataset` takes, counted by the same registry,
  and the one cross-stack overlap `hdf5_owner` allows — and on a
  **writable** handle it refuses by name and points at the rebuild, because
  two HDF5 instances over one file with a writer among them is audit A1.
- `read_slab(name, partition_spec=spec)` **with no `shape`** returns the
  dataset rounded UP to the mesh-divisible extent under `spec`,
  zero-filled past the dataset. That is the padded consumer buffer.
  This is new in 2026-08-06 and it is the point: `shape=None` used to mean
  "the dataset's own shape", which *refuses* whenever that shape is not
  mesh-divisible — the normal case, since N_mu is a physics number and not
  a multiple of the device count. Every caller therefore computed the
  round-up itself. The easy call is now the correct call.
- `shape` may still be stated exactly and is returned exactly. It must be
  mesh-divisible under `partition_spec`: the return value is a `jax.Array`
  of that shape sharded that way, and JAX will not build one at a
  non-divisible extent, so there is nothing to trim to.
- `valid_shape` survives **only** as the ragged-chunk override — a chunk
  buffer whose tail is genuinely not part of this write. Routine calls do
  not pass it.

Two rounding rules exist and must not be confused. `mesh_divisible_shape`
rounds each dim by the product of the mesh axes that shard *that dim*;
`runtime.padding.padded_mu_extent` rounds the μ extent by the **total**
device count, which is the separate in-memory convention `Meta.n_rmu_padded`
carries. The restart path deliberately uses the latter.

---

## Availability — one probe, one refusal {#availability}

`SlabIO.__init__` calls `assert_available()` before touching the inode. It
checks two things, which fail for unrelated reasons and are reported
separately:

1. this platform's FFI library exports `lorrax_phdf5_write`, probed with
   `ffi_loader.probe_target` — whose three states ("unknown target" /
   "could not be loaded" / "does not export") are three *different*
   repairs, which is why the refusal quotes its reason verbatim instead of
   reducing it to a bool;
2. MPI can bootstrap here. Handler presence is not capability: the write
   calls `MPI_Init_thread`, and on a bare launch with no PMI environment
   Intel MPI aborts inside `MPIR_pmi_init` rather than returning an error
   (job 7884926), so the probe runs in a throwaway subprocess.

The probe is cached per process. **Nothing here keys on process count.**
The old refusals all did, because the tier they guarded was legal at
exactly one process; this one guards the only path there is, so it either
works or the deployment is broken.

### There used to be three tiers and a router {#tiers-history}

Kept because the deletion is the design decision, and because the shape of
the mistake recurs.

`PHDF5_FFI` (this one), `PHDF5_HOST` (the same collective MPI-IO driven
from Python by mpi4py + h5py-parallel) and `H5PY_ALLGATHER` (gather the
whole global array onto rank 0, write it with serial h5py), selected by a
`slab_io` deck key with an `auto` platform router and a deprecated
`use_ffi_io` boolean.

**`H5PY_ALLGATHER` was refused at SEVEN separate doors**, each closure
landed and reported as complete, and an eighth door kept being found. The
doors were: the `auto` router running out of tiers; the deck naming the
tier (or `use_ffi_io = false`) post-checked after the whole precedence
chain; three separate entries in `SlabIO.__init__`'s argument normaliser
(named enum, unstated default, legacy boolean); the shared gate those
three funnel through; and a resolve-time check in the sharded-σ layout
path. An eighth, entirely ungated route survived all of them:
`gw/gw_init.py` imported `_to_host` **directly from the allgather backend
module**, bypassing every refusal.

A tier that must be refused at seven doors is not a tier; it is dead code
wearing a safety label. Deleted 2026-08-06 — the module, the enum member,
the deck value, all seven refusals, and the direct import (repointed at
`common.collectives.gather_to_host`, the sanctioned L3 gather it was a
private copy of).

**`PHDF5_HOST` was deleted with it**, and the evidence for that is worth
stating because it is a different argument. Its *only* selection condition
was a host FFI library built before workstream AE, which exports the phdf5
read symbols and not `PhdfWriteHostFfi`. Measured 2026-08-06 with `nm -D`:

| library | `PhdfWriteHostFfi` | dated |
|---|---|---|
| Perlmutter `lorrax_P/.../build_host/liblorrax_ffi_host.so` | **present** | live |
| Perlmutter `lorrax_P/.../build/liblorrax_ffi.so` (`PhdfWriteFfi`) | **present** | live |
| Frontera `$WORK/lorrax_ffi_unified/build_host_W/` | **present** | 2026-07-26 |
| Frontera `$WORK/lorrax_ffi_unified/build_host_V/` | absent | 2026-07-26 |
| Frontera `$WORK/lorrax_ffi_unified/build_host/` | absent | 2026-07-25 |

So the tier's condition is false on every deployed library; the two that
fail the probe are dated A/B control builds in a staging directory, and a
post-AE build sits beside them. Frontera's live tree (`$WORK/lorrax`) has
**no compiled `.so` at all**, so nothing there resolves either tier today.
And `PHDF5_HOST` additionally needs an mpi4py + `HDF5_MPI=ON` h5py overlay
that the FFI path does not — on Frontera's default python, `h5py` does not
even import (`libhdf5.so.103: cannot open shared object file`).

A tier that requires *more* to do the *same* thing, selected only by a
stale artifact, is not a fallback. The correct response to a stale `.so`
is a refusal naming it, which is the repo-wide contract (CLAIMS 81) and is
what `assert_available` now does.

**Not deleted, and not a tier:** `bse_loading`'s serial h5py readers. They
hyperslab exactly one rank's (μ, ν) tile, allgather nothing, and are
memory-correct at any process count — they are simply ~17× slower
(0.17 GiB/s at P=4, CLAIMS 76, vs 2.919 GiB/s for the tile path at 16
ranks, CLAIMS 69) because they issue `nq × μ/px` short row-runs one at a
time with no collective buffering. `bse_loading` falls back to them, loudly,
when `probe_availability()` declines. Slow and correct is a legitimate
fallback; the tier that was deleted was neither.

---

## Restart at P>1 {#restart}

`restart = true` reads `V_qmunu`, `S_qmunu`, `V0_noG0_munu`,
`psi_full_y` and (bispinor) `psi_full_y_transverse` back from
`isdf_tensors_<n_rmu>.h5`. Until 2026-08-06
`tagged_arrays.read_restart_state_from_h5` read every one of them with
`[:]` — the whole `(nq, μ, μ)` tensor **on every rank** — and only then
applied `jnp.pad` on both μ axes and `with_sharding_constraint`. Measured
at P=4 (job 56389339, MoS2 6×6, N_mu=1496, nq=36): **+1.53 GiB VmHWM per
rank**, silently. At the envelope (N_mu=20000, nq=64) the same read is
**381.47 GiB per rank**.

It was therefore *guarded off* above one process, which removed a
capability that had worked at deck scale and left `restart = true` with no
P>1 story. It is now on the tile path, following
`bse_loading.load_bse_data_from_restart_sharded`:

1. one serial-h5py pass reads **shapes** and the small replicated arrays
   (`enk_full`, `G0_mu_nu`, the stamps), then closes — two live handles on
   one file is a hazard nobody needs;
2. SlabIO reads the N_mu²-class tensors and ψ as per-rank tiles, asked for
   at the **padded** extent so the zero-fill past the dataset *is* the pad.
   No `jnp.pad`, and no sharding constraint applied to an already-resident
   global array.

What stays on serial h5py is μ-class or smaller (`G0` is 320 KB at the
envelope) and is needed whole on every rank. The doctrine forbids
materialising an N_mu²-class object, not reading a vector.

**Spinor and bispinor.** `nspinor` is read from the ψ dataset and carried
through as a replicated axis at its on-disk extent — 2 for spinor, 4 for
bispinor — and is **never padded**. It is gated to `{1, 2, 4}`: an
unexpected extent would otherwise sail through as a perfectly shardable
replicated axis and misindex every downstream ψ contraction with no shape
error. The bispinor `psi_full_y_transverse` is read at its **own** μ
extent, since the transverse centroid count differs from the charge one
and carries its own pad; the stamped `n_rmu_transverse_logical` is
cross-checked against the dataset's extent *before any bytes move*.

**Parity is bit equality**, not a tolerance: this is an element-*selection*
change, not a reduction-order one. `tests/multi_device/restart_sharded_parity.py`
asserts every returned element equals the serial-h5py read of the same
file at the same index, at `RESTART_NS` ∈ {1, 2, 4}, and additionally that
the pad rows are exact zeros and that no rank's shard is the whole array.
`tests/test_restart_pad_roundtrip.py` covers the same round trip at 1×1.

---

## Launcher requirements {#launcher}

### `srun --mpi=cray_shasta` is mandatory

Shifter's `--module=mpich` bind-mounts Cray MPICH. Cray MPICH speaks
`cray_shasta` PMI. Launch it under `--mpi=pmi2` or `--mpi=pmix` and MPI
initialises **as a singleton**: every rank gets its own `MPI_COMM_WORLD` of
size 1.

This is the single most dangerous failure mode in the whole subsystem,
because **it can produce a correct-looking file**. Under independent I/O
each rank writes its own hyperslab with no collective handshake, so 16
singleton ranks write 16 disjoint regions of one file and the result looks
right. Under collective I/O (the current default) it does not survive, but
the historical record includes runs that "worked" on pmi2 for exactly this
reason.

**Therefore: a phdf5 success is not evidence of working MPI unless the run
asserted its world size.** The assertion is three lines and there is no
excuse for omitting it from a gate:

```python
import ctypes
mpi = ctypes.CDLL("libmpi.so.12")          # MPICH: MPI_COMM_WORLD == 0x44000000
sz = ctypes.c_int(-1)
mpi.MPI_Comm_size(ctypes.c_int(0x44000000), ctypes.byref(sz))
assert sz.value == jax.process_count(), f"SINGLETON MPI: {sz.value}"
```

### The rest of the launch

```
srun --jobid=$JID --overlap --mpi=cray_shasta \
     --gres=gpu:4 -N $NNODES -n $NTASKS --cpu-bind=cores \
     src/ffi/cpp/select_gpu.sh \
     shifter --image=... --module=gpu,mpich --volume=... --env=... \
     src/ffi/cpp/in_container.sh \
     <command>
```

- `select_gpu.sh` sets `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`. It is a
  **script file**, not an inline `bash -c`, so srun execs it as a single
  process image and the `PMI_*` variables reach `MPI_Init_thread` intact.
- `in_container.sh` re-exports `MPICH_GPU_SUPPORT_ENABLED=1`, which
  Shifter's mpich module explicitly unsets.
- `MPICH_GPU_SUPPORT_ENABLED` is **irrelevant to phdf5** and was never
  A/B'd against it. phdf5 writes from host pinned buffers
  (`cudaMallocHost`, `src/ffi/cpp/phdf5/context.cc`). The variable exists
  for SLATE / cuSOLVERMp GPU-Direct RDMA.
- The module's `lxrun` helper hardcodes `-N 1`. **It is not a multi-node
  launcher.** For multi-node, write the `srun` line above.

---

## Tuning {#tuning}

All figures: 2.000 GiB complex128 (32768 × 4096), 2-D mesh, `/pscratch`,
`slab_io=phdf5_ffi` forced, two reps per configuration, best of two,
world size asserted. Produced by one process per geometry that sets
`os.environ` and opens a fresh path per configuration (the C++ context
reads its knobs at `open_file` time), so all rows in a column share one
container bring-up.

Two geometries:

- **N1** — 1 node, 4 ranks, 2×2 mesh, 512 MiB per rank.
- **N4** — 4 nodes, 16 ranks, 4×4 mesh, 128 MiB per rank.

| configuration | N1 write | N1 read | N4 write | N4 read |
|---|---|---|---|---|
| **default** (collective W+R, non-collective meta, 16 × 4 MiB) | 0.638 | 1.212 | 1.988 | 3.147 |
| default, repeated (noise) | 0.640 | 1.209 | 2.045 | 3.208 |
| `COLLECTIVE_WRITES=0` (independent writes) | **0.943** | — | **0.068** | — |
| `COLL_META=1` (collective metadata) | 0.644 | 1.215 | 1.963 | 3.177 |
| `INDEPENDENT=1` (independent reads) | — | **3.744** | — | 1.708 |
| stripe 1 × 1 MiB (the `/pscratch` default) | 0.616 | 1.006 | 0.695 | 0.761 |
| stripe 4 × 4 MiB | 0.688 | 1.209 | 1.377 | 1.487 |
| stripe 8 × 4 MiB | 0.660 | 1.211 | 1.735 | 2.332 |
| stripe 48 × 4 MiB | 0.568 | 1.220 | 1.829 | 3.452 |
| **stripe 16 × 1 MiB** | **0.802** | **2.240** | **2.941** | **4.267** |
| stripe 16 × 16 MiB | 0.777 | 1.633 | 2.532 | 4.042 |
| `CB_NODES=4`, `CB_WRITE=enable` | 0.641 | 1.206 | 1.397 | 1.854 |
| `ALIGN_MB=0` (alignment off) | 0.662 | 1.204 | 2.101 | 3.196 |

Units: GiB/s aggregate for the whole file. Repeat rows bracket the noise
at **±1.5 % or better**, so differences below ~5 % are not real.

**Reads are warm-cache.** Each read follows the write of the same file in
the same process, so the data is in the client page cache. The read column
therefore measures the MPI-IO path, not Lustre read bandwidth; use it to
compare configurations, not to size a cold restart.

### What the numbers say

**Collective writes stay on, and the reason is variance, not the mean.**
The headline number is that independent writes measured **0.068 GiB/s at
N4** — a 29× cliff, a run that never finishes. But the honest version is
more interesting, and a second sweep found it: that cliff is
*striping-dependent*. At 16 × 1 MiB, independent writes measured 2.927
GiB/s at N4 and 0.915 at N1 — level with collective, or slightly better.
The catastrophe was independent writes **at 4 MiB striping** specifically.

So the case for collective is not "always faster". It is:

| | collective | independent |
|---|---|---|
| N1, 16×4M | 0.654 | 0.943 |
| N4, 16×4M | 2.07 | **0.068** |
| N1, 16×1M | 0.818 | 0.915 |
| N4, 16×1M | 2.93 | 2.927 |

Collective never varies by more than ~10 % from the best available at its
geometry. Independent varies by **43×** across a single unrelated knob,
and one of those corners is unusable. With hundreds of ranks in the
envelope and no way to sweep every deployment, take the configuration
whose worst case is bounded. Two-phase aggregation is doing exactly what
it is for: making the strided 2-D tile pattern insensitive to how the
filesystem underneath is laid out.

> Note for anyone holding an older brief: `use_collective_write` was
> flipped to `true` on 2026-07-27 (`ctx.h`, `context.cc`). Any statement
> that "the GPU write path is independent" describes a configuration that
> was abandoned, and any "4.45 GB/s collective" figure predates the
> current one. The measurements above are on the current default.

**Collective reads stay on** for the same directional reason (1.84×
faster at N4, though slower at N1), and because the read column is
cache-warm and so is the weaker evidence of the two. Not changed.

**Collective metadata stays off.** Within noise at both geometries
(≤ 1.3 % either way). No measured reason to turn it on; keep the simpler
default.

**ROMIO collective-buffering hints stay unset.** `cb_nodes` + `cb_write`
cost 30 % at N4 and were neutral at N1. Cray MPICH's own heuristics beat
hand-set hints here. Leave them alone.

**Alignment is not load-bearing** at this size — turning it off is within
noise at both geometries.

**Striping is the largest single lever, and the folklore about it was
wrong in an instructive way.** `lxrun`'s comment claimed the `/pscratch`
default of 1 × 1 MiB "caps ~30 MB/s/rank". The default is confirmed
(`lfs getstripe -d /pscratch/sd/j/jackm` → `stripe_count: 1
stripe_size: 1048576`), but the cap is **not per-rank**: a 1-stripe file
measured 0.616 GiB/s at 4 ranks and 0.695 GiB/s at 16 ranks. Quadrupling
the ranks bought 13 %. It is a **per-file, single-OST ceiling of roughly
0.65 GiB/s**, and adding ranks does not move it. Stated per-rank the
number looks like something more parallelism can fix; stated per file it
is obviously a striping problem.

Stripe layouts were verified applied, not assumed — `lfs getstripe -c -S`
on each swept file returned exactly the requested count and size, so the
`LORRAX_PHDF5_STRIPE_*` hints do reach Lustre through ROMIO.

### Striping is not a constant: the stripe count is the aggregator count

Everything above this heading was measured at **16 ranks**, where the old
default of `16 × 1 MiB` happens to equal `nranks × 1 MiB`. That coincidence
is what made a constant look like a tuning.

The mechanism, **read back from ROMIO rather than assumed**
(`LORRAX_DEBUG_PRINT=1`, which dumps `MPI_File_get_info` after
`H5Fcreate` — the hints ROMIO *retained*, not the ones we asked for):

| requested `striping_factor` | 4 ranks → `cb_nodes` | 16 ranks → `cb_nodes` |
|---|---|---|
| 1 | — | 1 |
| 4 | **4** | 4 |
| 16 | **4** | 16 |
| 64 | — | 16 |
| 128 | **4** | 16 |
| 366 (`-1`, every OST) | — | 16 |

`cb_nodes = min(striping_factor, nranks)`, exactly, at every point measured
— and we never set `cb_nodes` ourselves (`romio_cb_write = automatic`,
`cb_config_list = *:*` in the same dump). **So the stripe count IS the
collective-buffering aggregator count.** A fixed 16 pins `cb_nodes = 16`
forever, and every rank past the sixteenth is a rank that does not
aggregate. Per-rank throughput rolls over just past 16 ranks for that
reason and no other.

Write bandwidth, GiB/s aggregate, `/pscratch`, phase-separated harness:

| ranks | payload | `16 × 1 MiB` | `nranks` stripes | gain | artifact |
|---|---|---|---|---|---|
| 4 | 2.00 GiB | 0.803 | **0.873** (4 × 1M) | +8.6 % | `w_P4.json` |
| 4 | 2.00 GiB | 0.813 | **0.906** (4 × 1M) | +11 % | `w_SC4.json` |
| 16 | 8.00 GiB | 3.152 | 3.152 (16 × 1M) | *null control* | `w_SC16.json` |
| 64 | 32.00 GiB | 5.189 | **10.630** (64 × 4M) | +105 % | `w_SC64.json` |
| 64 | **381.47 GiB** | 7.403 | **13.222** (64 × 4M) | **+79 %** | `w_EV64.json` |
| 100 | 50.00 GiB | 7.872 | **15.216** (100 × 4M) | +93 % | `w_R100.json` |

The 16-rank row is the **null control**, not a win: there the two policies
are literally the same configuration, and a policy that changed that row
would be a policy with a bug. The 381 GiB row is the real envelope payload
— `V_qmunu`, `(nq=64, 20000, 20000)` complex128 — where the old default
leaves **44 % of the write bandwidth on the floor**. `stripe = nranks` won
or tied at every rank count measured.

### The read advantage in the table above was the page cache

The `16 × 4M → 16 × 1M` change was justified partly on a **76 % read
improvement**. That number was warm: `slabio_sweep.py` read the file it had
just written, in the same process, so every byte came from the writing
node's page cache. A production restart reads files nobody just wrote.

Re-measured with write and read as separate `--phase` invocations on
**disjoint node sets** (write `nid003837`, read `nid003840`; `lctl` is
absent from the compute image and `drop_caches` needs root, so a disjoint
node set is the only honest cold-read instrument here):

| 4 ranks, 2.00 GiB | warm read | cold read | write |
|---|---|---|---|
| `16 × 1 MiB` | 2.29 | 0.932 | 0.803 |
| `4 × 1 MiB` (policy) | — | 0.932 | 0.873 |

Cold read ≈ write throughput, and the layout advantage shrinks from 76 % to
**5.7 %**. Reads are not where the lever is. **Any benchmark on this
filesystem must use disjoint node sets or it is measuring the cache**, and
a read number quoted without its node sets is not a measurement.

### What the policy is

`file_io._slab_io_ffi._stripe_policy(nranks)` — one pure function of the
rank count, no env, no MPI, no filesystem, so it is testable at rank counts
no allocation on this machine can reach:

```
striping_factor = min(max(nranks, 4), 128)
striping_unit   = the power of two NEAREST IN LOG2 to (nranks/16) MiB,
                  clamped to [1 MiB, 4 MiB]
```

which is `1 MiB` below 22.6 ranks, `2 MiB` to 45.3, `4 MiB` above.

**The unit is a ramp, not a step at 64.** Both ends are measured
(`16 × 4M` = 2.07 vs `16 × 1M` = 2.93 at 16 ranks; `64 × 1M` = 7.068 vs
`64 × 4M` = 10.630 at 64), and between them the rule is the geometric
interpolation those two anchors imply, with no free parameter to pick.
It has to be argued rather than measured, because **the midpoint of that
range cannot be run here**: `resolve_mesh` requires a perfect-square device
count, so 32 ranks is not a legal geometry, and `16 < n < 64` contains no
power-of-two rank count that is also a legal mesh. The legal points in the
gap are 25, 36 and 49.

**4 MiB is a ceiling, not a trend to extrapolate.** The per-rank tile knee
is 4 MiB — at 16 ranks, tiles of 0.06 / 0.25 / 1 / 4 / 16 MiB gave
0.126 / 0.609 / 1.779 / 3.165 / 3.551 GiB/s: flat above 4 MiB, a cliff
below (`w_TS16.json`). `V_qmunu` at `N_mu = 20000` on 1024 ranks is
**~6.1 MiB per rank**, only 1.5× above that knee, so the envelope sits on
the edge of the cliff and a stripe unit above the per-rank tile would
starve aggregators.

**`-1` is refused, not passed through.** It reads like a maximum and
measures like a failure: on `/pscratch` (370 OSTs) it expands to 366
stripes and gave **0.105 GiB/s** at 64 ranks / 32 GiB and 1.118 at 16 ranks
/ 8 GiB, against 10.63 and 3.15 for the policy. Striping across every OST
is the maximum-*contention* layout, not the maximum-bandwidth one.

**Layouts are verified granted, not assumed.** `lfs getstripe -c -S` from a
login node (the compute image has no `lfs`) returned exactly the requested
`4 × 1M`, `16 × 1M` and `128 × 4M` on the files this branch wrote.

#### The upper clamp is NOT verified in-policy {#clamp-open}

`striping_factor = 128` is emitted only at `nranks >= 128`, and **no run at
that scale has been made**: the interactive QOS maximum is 4 nodes / 16
GPUs, so the largest legal mesh reachable here is 16 ranks. What *is*
measured is that 128 stripes behave badly when the rank count is *below*
128 — 1.014 GiB/s at 64 ranks and 0.825 at 100 ranks with `128 × 4M`,
against 10.63 and 15.22 for `nranks × 4M`, and 0.501 vs 0.869 at 4 ranks.
Those points are **off-policy** (the policy never emits a count above
`nranks`), so they do not condemn the clamp — but they do mean the clamp
endpoint has never been observed in the regime where the policy actually
uses it, and 128 is anomalous even among its neighbours: at 100 ranks
`sf = 150` and `sf = 200` measured 14.07 and 12.96 while `sf = 128`
measured 0.416. **Measure `nranks = 128` (or 121, the nearest legal mesh)
before an envelope run leans on it.**

---

## Defaults {#defaults}

| knob | default | basis |
|---|---|---|
| `LORRAX_PHDF5_COLLECTIVE_WRITES` | `1` | 29× cliff at N4 when off; the asymmetry decides |
| `LORRAX_PHDF5_INDEPENDENT` (reads) | `0` (collective) | 1.84× at N4; read evidence is cache-warm, so no change on it |
| `LORRAX_PHDF5_COLL_META` | `0` | within noise both geometries — keep the simpler one |
| `LORRAX_PHDF5_STRIPE_COUNT` | **`nranks`, clamped to [4, 128]** | **changed from `16`** — the stripe count is the aggregator count (`cb_nodes = min(striping_factor, nranks)`, read back from ROMIO); a constant pins the aggregator count. +79 % on the 381 GiB envelope payload at 64 ranks. `-1` is refused |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | **`1M`/`2M`/`4M` by rank count** | **now a ramp** — 1 MiB at <=16 ranks, 4 MiB at >=64, the power of two nearest in log2 to `nranks/16` MiB between; both ends measured, the midpoint is not runnable (32 ranks is not a legal mesh) |
| `LORRAX_PHDF5_ALIGN_MB` | `4` | unchanged; `4`/`1`/`0` all inside repeat noise, so not worth a second knob to keep in sync |
| `LORRAX_PHDF5_CB_*` | unset | hand-set hints cost 30 % at N4 |
| `lxrun` pre-stripe | `-c 16 -S 1M` | same basis as `STRIPE_SIZE_FS` |

Both striping rows are resolved by ONE pure function, `file_io._slab_io_ffi._stripe_policy(nranks)`, and an explicit `LORRAX_PHDF5_STRIPE_*` still wins over it. The FFI writer's hints are built in C++ whose only input is `getenv`, so the resolved values are exported back into `os.environ` before `open_file` — which is why a run log's environment remains a complete description of the layout, and why the `lfs getstripe` readback means anything.

> **Why the stripe count is `nranks` and not a constant.** The reason 16
> won the older sweep is that the sweep ran at 4 and 16 ranks. ROMIO sets
> `cb_nodes = min(striping_factor, nranks)`, so **the stripe count IS the
> aggregator count** -- pinning it to 16 caps aggregation at 16 however many
> ranks write, which is exactly backwards for a design envelope of hundreds.
> The owner approved `stripe_count = nranks` on 2026-08-05.
>
> **Status, verified 2026-08-06 in `integration/2026-08-06`:** implemented
> and IN this tree, via `feat/slab-io-stripe-nranks-2026-08-06` (`e5c9618`,
> `252b80d`), which reaches here as an ancestor of
> `feat/bse-slabio-2026-08-06`. `_stripe_policy(nranks)` clamps the count to
> [4, 128], ramps the unit 1 -> 4 MiB, and **refuses a negative count**
> (a negative `striping_factor` means "every OST on the filesystem", the
> maximum-contention layout). The measurements behind it are recorded above
> `_stripe_policy` in `src/file_io/_slab_io_ffi.py`.
>
> **The policy is now on BOTH sides, and the C++ side is UNBUILT.** It
> used to be Python-only: `context.cc` carried the literals `"16"` and
> 1 MiB as the fallback for an unset `LORRAX_PHDF5_STRIPE_*`, and on every
> LORRAX path the variables are set before C++ can read them
> (`_FfiBackend.__init__` calls `_export_striping_env()` immediately before
> `_open_file()`, which is the call in which `context.cc` builds the
> `MPI_Info` and `H5Fcreate` applies the layout). That masked, rather than
> removed, a real disagreement: a caller reaching `ffi.io.open_file`
> **without** going through `_FfiBackend` got 16 × 1 MiB and no warning,
> at any rank count. Two writers requesting *different* layouts is worse
> than both being wrong the same way, so `context.cc` now derives the
> layout from `ctx->world_size` via `stripe_policy_count` /
> `stripe_policy_unit`, which transcribe `_stripe_policy` line for line.
> `tests/test_slab_io_routing.py::test_cpp_stripe_policy_transcribes_the_python_one`
> extracts those two functions, compiles them with the host `g++` and diffs
> them against the Python over 0..4100 ranks.
>
> **Not verified, and here is the exact gap:** that test compiles two
> `static long` functions in isolation. `liblorrax_ffi*.so` has **not been
> rebuilt** with this change — no cluster, no toolchain, on the branch that
> made it — so nothing yet shows a deployed library requesting the policy
> layout. Required before leaning on it: rebuild both libs, then one
> `lfs getstripe -c -S` readback of a file written through
> `ffi.io.open_file` with `LORRAX_PHDF5_STRIPE_COUNT` **unset** at a rank
> count away from 16 (the old literal), which is the only geometry where
> the two answers differ.

The stripe-size choice, across both geometries (GiB/s write / read):

| layout | 1 node / 4 ranks | 4 nodes / 16 ranks |
|---|---|---|
| 16 × 4 MiB *(previous default)* | 0.654 / 1.30 | 2.07 / 3.23 |
| **16 × 1 MiB** *(new default)* | **0.818 / 2.29** | **2.93 / 4.74** |
| 16 × 2 MiB | 0.626 / 1.30 | **3.12 / 5.06** |
| 32 × 1 MiB | 0.754 / 2.28 | 2.78 / **5.52** |
| 8 × 1 MiB | **0.867** / 2.28 | 2.52 / 3.53 |

2 MiB is the fastest configuration measured at 4 nodes and the *worst*
measured at 1 node. 32-wide has the best read at 4 nodes and a poor write
at 1. 8-wide has the best write at 1 node and a poor read at 4. Only
16 × 1 MiB is at or near the top of every column, so it is the default —
elegance here means one layout that does not have to be tuned per job,
not the maximum of any single cell.

Changed in three places that must agree: `src/ffi/cpp/phdf5/context.cc`
(the C++ writer's `striping_unit`), `src/file_io/_slab_io_mpi_host.py`
(the Python writer's, so one environment means one layout in every
writer), and `lxrun`'s `lfs setstripe` pre-stripe in the modulefile.

---

## Certification {#certification}

Scope of every claim below: **inside the Shifter container, on compute
nodes**, launched by raw `srun … select_gpu.sh … shifter … in_container.sh`
— not `lxrun` (which is `-N 1` only), and not from a login node. Job
56389339, branch `fix/run-shifter-nvhpc-subpath-2026-08-05`.

All four cells, `slab_io=phdf5_ffi` **forced** so the router cannot mask
the result, each asserting `MPI_Comm_size()` against `jax.process_count()`
before writing anything:

| cell | geometry | world size | round-trip | payload md5 |
|---|---|---|---|---|
| **GPU** write + read | 1 node, 4 ranks, 2×2 | 4, asserted | bit-exact | `b738748c4803951a7dc5230121e1e5b5` |
| **GPU** write + read | 4 nodes, 16 ranks, 4×4 | 16, asserted | bit-exact | `b738748c4803951a7dc5230121e1e5b5` |
| **CPU** write + read | 1 node, 4 ranks, 2×2 | 4, asserted | bit-exact | `27a17944d643fad537a0b98e2774bec7` |
| **CPU** write + read | 4 nodes, 16 ranks, 4×4 | 16, asserted | bit-exact | `27a17944d643fad537a0b98e2774bec7` |

The md5 is of the **dataset payload**, re-read serially with plain h5py,
for the same logical 8192 × 4096 complex128 array. Within each platform
the 4-node/16-rank and 1-node/4-rank digests are **equal**: the same
logical array written through a 4×4 mesh and a 2×2 mesh is byte-identical
on disk. That is the multi-node write certification.

The GPU and CPU digests differ from each other, and that is not an I/O
result — the test array is built with `jnp.sin`, which differs in the last
ulp between the CUDA and host backends. The I/O claim is the round-trip
column and the within-platform digest match.

(The raw HDF5 *files* differ in metadata/layout bytes even when payloads
match, which is why the payload is hashed rather than the file.)

The routing decision was verified separately at 4 nodes after the fix —
`slab_io=auto` on 16 GPUs now prints:

```
[config] slab_io=auto on GPU backend: CUDA FFI exports the collective phdf5
write handler and MPI can bootstrap (launcher PMI environment present
(PMI_CONTROL_PORT)).  Routing SlabIO through PHDF5_FFI [processes=16,
devices=16, local_devices=1, SLURM_NNODES=4, SLURM_NTASKS=16].
```

### Re-verified against the rebuilt library

The stripe-size default change means a rebuilt `liblorrax_ffi.so`
(sha256 `323f4f0e…`, stamped `PROVENANCE`). With **no tuning environment
set at all**, that binary produced 16 × 1 MiB files (`lfs getstripe`
confirmed) and:

| geometry | write | read | round-trip |
|---|---|---|---|
| 1 node, 4 ranks | 0.812 GiB/s | 2.261 GiB/s | bit-exact |
| 4 nodes, 16 ranks | 2.655 GiB/s | 4.717 GiB/s | bit-exact |

against 0.654 / 1.30 and 2.07 / 3.23 for the old default — **+24 % / +74 %
and +28 % / +46 %**, for a one-word change.

---

## Failure modes {#failures}

*Three of these — S1, S2, S3 — were measured on the metallic MPA-QSGW
campaign of 2026-08-15 and share one mechanism. They are first because
they are the ones you will meet, and each carries the journal fingerprint
that tells it apart from its look-alikes. Root-cause analysis:
`reports/metal_mpa_plan_2026-08-15/SLAB_IO_ROOT_CAUSE_AUDIT.md`.*

### The one mechanism behind S1–S3: the control operands raced the XLA stream {#stream-race}

**VERIFIED by reading, and it is not "two HDF5 libraries are spooky".**

The phdf5 handlers receive `offset_base`, `valid_shape` and the ctx
`handle` as **device buffers** on the CUDA platform (`src/ffi/io.py`
passes them as `jax.Array` operands to `ffi_call`). Every one of them is
brought to the host by `copy_index_to_host` in
`src/ffi/cpp/phdf5/platform_seam.h`, whose CUDA arm was a **synchronous
`cudaMemcpy` on the legacy default stream**.

XLA creates its compute streams and the ctx stream with
`cudaStreamNonBlocking` (`context.cc`), and **the legacy default stream
does not order against a non-blocking stream.** There was no
`cudaStreamSynchronize` anywhere under `src/ffi/cpp/phdf5/` (grep: zero
hits) and the handlers never waited on the operand-producing stream. So
the host could read an operand buffer *before the XLA stream had written
it*, harvesting whatever bytes previously occupied that device allocation.

The **output** side was already stream-correct — `stage_host_to_output` in
`read_ffi.cc` does `cudaMemcpyAsync` on `ctx->stream`, then an event, then
`cudaStreamWaitEvent(xla_stream)`. The discipline was known. The small
index copies are what missed it.

Three properties of all three signatures fall straight out, and no other
hypothesis explains all three:

* **identical on every rank** — every rank runs the same schedule over the
  same stale allocator layout, and the bounds check then refuses
  collectively, by design;
* **intermittent across runs** — allocator layout and stream timing differ
  per launch;
* **worse at later iterations** — buffer churn and deeper async pipelines
  widen the window, and this pipeline's churn is measured at [1027
  cross-library alternations per iteration](#alternations).

**Fix (B1), on branch `integ/metal-mpa-qsgw-2026-08-15` as `ef98d47f`:** `copy_index_to_host` takes the XLA
stream the handler already receives (`LRX_STREAM_PARAM`); the CUDA arm
becomes `cudaMemcpyAsync(dst, src, n, D2H, xla_stream)` followed by
`cudaStreamSynchronize(xla_stream)`. Eleven call sites (eight read, three
write). **The host arm is unchanged and the host leg was never affected**
— there the buffer is already host-resident, so `copy_index_to_host`
degrades to a `memcpy` and there is no stream to race.

**Fix (B2), on branch `integ/metal-mpa-qsgw-2026-08-15` as `15eef55f`:** a mutex-guarded registry of live
`PhdfCtx*` in `context.cc` (`register_live_ctx` at open,
`unregister_live_ctx` at close), consulted by `shard_index.h`'s
`check_live_ctx` on the value that came off the copy and **before anything
dereferences it** — including before the `fail` lambdas, which announce
through `ctx->rank`. An address that is not in the registry becomes a
named refusal:

```
phdf5 read: stale or foreign ctx handle -- stream race or double close.
  got=0x3df1… (not in this library's live-ctx registry; 1 context(s) currently open).
  want= a handle returned by lrx_phdf5_open on THIS platform leg and not yet closed.
  fix= …
```

This does not remove the race; it converts any residual one — and any
handle minted by the *other* platform leg, which is the
[ODR](#odr-host-so) symptom — into something attributable. The registry
is per library instance, which is the point: a handle from the host leg
is not in the CUDA leg's set and vice versa.

> **The sibling-file split would NOT have prevented any of S1–S3.** No
> h5py involvement is required: this is a single-library, single-file race
> in the FFI marshal. The split is still worth doing, for the reasons and
> in the order given in [Sibling split](#sibling-split).

!!! danger "B1 and B2 are in the SOURCE. They are not in any deployed `.so`."
    Checked 2026-08-15, after both commits: **no `liblorrax_ffi*.so` on
    this filesystem carries either fix.** The newest CUDA library
    (`lorrax_cuda13_module_20260814/.../build_cuda13_phdf5/`, built 14:24)
    predates `ef98d47f` (16:58), and `strings -a <so> | grep 'stale or
    foreign ctx handle'` is empty on every library present, including both
    halves of the pinned Aug-7 pair.

    This is the same trap the [striping policy](#defaults) fell into: a
    correct C++ change that no run has loaded. **A run is protected from
    S1/S2/S3 only if the `.so` it pinned was built after these commits.**
    Cheap check, and it costs nothing to put in a prelude:

    ```bash
    strings -a "$LORRAX_FFI_SO" | grep -q 'stale or foreign ctx handle' \
      || echo "WARNING: this .so predates SLAB_IO B1/B2 — S1/S2/S3 are live"
    ```

    Until a rebuild lands, the [journal](#journal) is the instrument and
    the fingerprints below are how you use it.

### S3 — a no-offset read arrives with a garbage offset {#s3}

*Looks like* (measured 2026-08-15 15:52:53, JID 57038615, P=4 one node,
5th SC iteration of the damped R6 arm):

```
[phdf5 ERROR rank=2] phdf5 read: logical slab out of bounds extent=[16]
  offset_base=[4462667732332943029] valid_shape=[16] rank=2
  -- refused identically on every rank
pjrt_stream_executor_client.cc:2939 Execution of replica 0 failed:
  INVALID_ARGUMENT
```

*Read the line carefully.* `extent=[16]` is `2·n_p` at `n_p=8` — one of
the 1-D `mpa_head/{sample_z,sample_Wc,Omega_p,B_p}` vectors that
`read_head_fit_collective` reads with `partition_spec=P(None)` **and no
offset argument**, so the offset reaching the FFI must be zero.
`4.46e18` is not an arithmetic mistake, it is uninitialised memory. The
apparent inconsistency between `extent=[16]` (one dim) and `rank=2` is
**benign**: `ctx->rank` is the MPI rank, not the array rank.

*Mechanism:* [the stream race](#stream-race), on the `offset_base`
operand. VERIFIED.

*Fix:* B1. *Not* a store-format problem, not a caller bug, and not
something the [sibling split](#sibling-split) would have touched.

*Journal fingerprint:* the Python-side line and the C++ refusal
**disagree**. Look for

```
… stack=ffi op=read path=…/mpa_fit_sc_0004.h5 ds=mpa_head__sample_z off=(0,) cnt=(16,) …
```

immediately before a refusal quoting a nonzero `offset_base`. Python said
0; C++ saw 4.46e18. That disagreement localises the corruption to the
marshal and rules out every caller.

*Honest, not silent.* The bounds check refused on every rank and no wrong
number was written. This is a crash, not a corruption — which is why the
check earns its keep even after B1 lands.

### S1 — SIGSEGV in `SlabIO.close` {#s1}

*Looks like:* exit 139, **native, no Python traceback**, rank 0 first (it
is the only serial-h5py writer), in SC iteration 3, immediately after a
fit-store `SlabIO.close`. The last thing on stdout is the close banner:

```
[SlabIO.close] draining 1 pending writes for mpa_fit_sc_0002.h5 …
```

*Mechanism:* SUSPECTED — the same [stream race](#stream-race), two routes.
The **handle** operand travels the same seam (`read_ffi.cc`,
`write_ffi.cc`), and the handler immediately
`reinterpret_cast<PhdfCtx*>(handle_host[0])` and dereferences it. A raced
handle read gives either

* **(i)** a garbage pointer → segfault on the spot, or
* **(ii)** a *stale* pointer to a previous iteration's freed ctx →
  use-after-free whose corruption surfaces later, e.g. at the next
  `close`'s writer-thread join, which is where this died.

A raced **write-path offset** is a third route: it lands collective writes
at wrong file offsets, corrupting HDF5 internal structures whose failure
also surfaces at close. Not witnessed in a debugger, hence suspected — but
every route is reachable from reads that are verified racy, and the
per-iteration open/write/close churn is exactly the allocator churn that
widens the window.

*Secondary contributor, real but a different class:* the [two-instance
metadata-cache cohabitation](#metadata-cache-condition). The registry
reduced it and the split retires it. It cannot explain S3 and need not be
invoked for S1.

*Fix:* B1, then B2 — after which route (i) and route (ii) both become a
named refusal instead of a dereference.

*Journal fingerprint:* take the last `handle=` on rank 0's journal and
grep it across **all** ranks and all iterations.

* the same `handle=` appearing under two different `path=` values, or
  appearing **after its own `op=close rc=ok`**, is route (ii) — a stale
  ctx, and the answer is B2, not HDF5;
* a `handle=` that appears exactly once, in the dying op, with a value
  that is not a plausible pointer, is route (i);
* a rank-0 journal whose last line is `stack=h5py op=open mode=a` in the
  deferred-attr reopen means you are in the *serial* half of close, not
  the collective half — a different investigation, and the registry line
  beside it tells you whether the FFI claim had been released.

*Look-alike that is NOT S1:* a death in the same drain reading
`Fatal glibc error: tpp.c:83 (__pthread_tpp_change_priority): assertion
failed` is the ODR collision, [below](#odr-host-so), and it is fixed by
changing which `.so` you pinned, not by anything on this page.

### S2 — "file is already open for write" on a fresh store {#s2}

*Looks like:* an `OSError` out of `h5py.File(...)` at
`qirr_store.py:470`, reached through `mpa_store._h5` → `QirrDest`, on a
store that this run created moments earlier. Single occurrence, on a cold
pipeline, consistent with the race's rarity there.

*Mechanism:* SUSPECTED, downstream of [S1](#s1) route (ii). **The guard
is correct and there is no defect in it** — it is h5py's own same-process
open-state check. A mis-fired or wrong-ctx `close` (raced handle) leaves
the FFI's write handle *genuinely* open; h5py's later open then refuses,
accurately, about a file that really is still open.

*Fix:* B1/B2. Nothing to change in `qirr_store`; it is the messenger.

*Journal fingerprint:* the discriminator is the registry verdict on the
refused line. `mpa_store._h5` declares every h5py open, so:

* `owner=refused` → the registry caught it first and you are looking at a
  **caller ordering bug**, with the FFI claim still live. Read the
  refusal; it names the live handle's `where=`.
* `owner=ok` **and** h5py still refuses → the Python side believed the
  file was closed and the C++ side had not actually closed it. That is
  S2, and it is B1/B2.

*Do not confuse this with the multi-node variant.* At 16 ranks / 4 nodes,
the 12 ranks **not on rank 0's node** die with the same message in
`gw/mpa/fit_driver.py`'s collective finalize, at the
`fit_completion_ledger` h5py open, after all 464 fit blocks of iteration 0
are on disk. That one is **PRE-EXISTING and reproducible** — an A/B at
`bf57701b`, before the registry, the `_h5` door and `PoleReader` existed,
fails identically on the same 12 ranks at the same line. It is not a
locking problem (`HDF5_USE_FILE_LOCKING=FALSE` was set on the passing
single-node run too). It is registered in `KNOWN_LORRAX_ISSUES` and
**every accepted sodium number on this campaign was produced at 4 ranks
on one node.**

### The `--parallel-transport-out` self-refusal — fixed by removing the second stack {#pt-introspect}

*Measured 2026-08-15 on `integ/metal-mpa-qsgw-2026-08-15` @ `814278cd`,
P=4, sodium 48-band SOC deck, reproducible on both smearing rungs.*

*Looked like:* every rank dying on the one-owner refusal, naming
`h5py open(mode='r') from _FfiBackend._introspect_dataset('links_ibz')`
while the FFI held the same path `mode='a'`. **The guard was right and
the caller was wrong**, and the price was paid at the worst moment: the
refusal fires on the velocity-validation stamping, i.e. AFTER the
expensive PT tensor is complete and BEFORE `dipole.h5` is written, so the
next stage failed with `head_correction.py:341`
("Failed to resolve q=0 Coulomb head"), naming neither file.

*Fix, 2026-08-22:* `_introspect_dataset` asks the FFI. There is no second
HDF5 instance to refuse. The two caller-side mitigations already in the
tree stay and are still the right shape for anyone on an older library:
pre-register the geometry with `create_dataset` (idempotent for an
identical existing dataset — `file_io/parallel_transport.py`'s connection
stage does exactly this), and never straddle an h5py open with a live
writable FFI handle.

*Related, same commit:* `gw.qsgw_head`'s two head loaders no longer open
`parallel_transport.h5` with h5py at all. Their stamps come through
`read_small` in the same read-only handle as the payload.

### The pre-ODR-fix host `.so` — a loaded gun for any host-leg run {#odr-host-so}

*Not what failed on this campaign — the failing runs' logs show only the
CUDA library loaded — but it is live in three of the four preludes, and
its documented signatures are precisely a garbage `offset_base` and death
in the close drain, i.e. it is an S1/S3 impostor.*

`tests/KNOWN_FAILURES.md` **L1** records that the two platform libraries
cross-wired their phdf5 through `RTLD_GLOBAL`: both are dlopened
`RTLD_GLOBAL`, ld.so answers a name from the first object that defined
it — for the whole process, including for the second library's own
internal calls — and `PhdfCtx` compiles to **two different struct layouts**
under `#ifndef LORRAX_FFI_NO_CUDA` while both libraries export one
`open_ctx(...) -> PhdfCtx*`. Fixed in-source by
`fix/ffi-odr-2026-08-08`: `exports_{cuda,host}.map`, the `_host`-suffixed
C ABI in `src/ffi/cpp/common/c_abi.h`, and split struct tags.

**The fix is in the source. It is not in the deployed host library.**
Re-measured 2026-08-15:

| pair | shared dynamic names LORRAX defines | C-linkage among them |
|---|---|---|
| the deployed Aug-7 pair (`~/software/lorrax_ffi_2026-08-07/`) | **25** | 9 |
| Aug-7 host `.so` × a current post-fix CUDA `.so` | **5** | **5** — `lrx_phdf5_{open,close,ensure_dataset,init_mpi,open_dataset_ro}` |

The 25 include `PhdfCtx::~PhdfCtx` (D1 **and** D2), `open_ctx`,
`close_ctx`, `ensure_pinned`, `env_flag` and the `dt::` HDF5-type
singletons with their guard variables. The destructor is the worst of
them: a `std::thread` join and a `std::mutex` destruction executed at
whichever build's field offsets answered — which is exactly the glibc
`tpp.c:83` death GATE 10 reproduces.

**Rebuilding the CUDA leg alone does not fix it** (KNOWN_FAILURES L1,
"rebuilt device + pre-fix host": still 9 LORRAX-own C-linkage names
shared). **The host rebuild alone IS sufficient** ("pre-fix device +
rebuilt host": 0 shared, 0 C-linkage).

**Where the safe build lives: nowhere yet.** Swept 2026-08-15 — the only
`liblorrax_ffi_host.so` anywhere under `$HOME`, `$CFS` or `$SCRATCH` is
the 2026-08-07 pre-fix one, and preludes v1, v2 and the `AB_bf57701b`
control all pin it. `env_prelude_v3` leaves `LORRAX_FFI_HOST_SO` **unset**,
which is why v3 is the safe prelude today: with no host library pinned
there is nothing to collide with, and a host-leg request refuses by name
rather than resolving to the wrong object.

*So:* build one from this tree — `src/ffi/cpp/build_host.sh`, which
carries `c_abi.h` and `exports_host.map` — before any run touches the host
leg in a CUDA process. **Acceptance, both parts required:**

```bash
nm -D "$HOST_SO" | grep -q lrx_phdf5_open_host          # the suffixed ABI exists
comm -12 <(nm -D --defined-only "$HOST_SO" | awk '{print $NF}' | sort -u) \
         <(nm -D --defined-only "$CUDA_SO" | awk '{print $NF}' | sort -u) \
  | grep -E 'lrx_|lorrax_ffi'                            # must print NOTHING
```

`src/ffi/cpp/gate_one_odr.py` (GATE 10) is the live version of the same
check: a real CUDA process doing host phdf5 work.

### The older failure modes {#older-failures}

**Singleton MPI.** *Looks like:* everything succeeds; output is plausible;
at 16 ranks the file even has the right size. *Detect:* the world-size
assertion above. *Cause:* `--mpi=pmi2` / `--mpi=pmix` against Shifter's
Cray MPICH. *Fix:* `--mpi=cray_shasta`.

**Swallowed MPI bring-up.** *Looks like:* one line,
`[phdf5 init_mpi] skipped: …`, then a run that appears to work. *Fixed
2026-08-05* — `_setup_runtime` now raises, naming `--mpi=cray_shasta`.
This is how a silent bring-up failure could be mistaken for a
platform-level "phdf5 doesn't work multi-node" conclusion.

**FFI library will not load.** *Looks like:* the router demotes with
*"the … FFI library could not be loaded: OSError: lib….so: cannot open
shared object file"*. **This says nothing about whether the handler is
compiled** — the three-state probe exists precisely to keep you from
rebuilding a library that is fine. It is a `LD_LIBRARY_PATH` / staging
problem. Run `ldd` on the `.so` *inside the container*.

Worked example, the CPU leg on 2026-08-05. `config/perlmutter/build_ffi_host.sh`
links the host lib against `cray-hdf5-parallel/1.14.3.7` and `cray-fftw`,
and against the `gpu_backend=none` SLATE build. None of those are in the
container by default:

| missing SONAME | where it lives | how to expose it |
|---|---|---|
| `libhdf5_parallel_gnu.so.310` | cray-hdf5-parallel **1.14** | **DONE 2026-08-06**, and it was not enough on its own: the stage is now `$HOME/software/lorrax_phdf5_cray_1.14.3.7/stage` (`site_config.sh`), but the DEVICE leg linked the old 1.12 stage, so staging 1.14 *beside* it would have mapped two HDF5 majors into one process instead of failing loudly. Both legs were moved together and GATE 7 (`src/ffi/cpp/gate_one_hdf5.sh`) now refuses the skew and the two-HDF5 "repair" alike |
| `libfftw3.so.mpi31.3` (+ `f`, `_omp` variants) | `/opt/cray/pe/fftw/*/x86_milan/lib` | `/opt/cray` is **not** a valid Shifter `--volume` source; copy under `$HOME/software` and bind-mount |
| `libslate.so.2`, `libblaspp`, `liblapackpp` | `$HOME/software/slate_builds/cpu/install/lib64` | `/global/homes` is siteFs-visible, so put it on `LD_LIBRARY_PATH` directly |

`phdf5_stage_cray.sh` used to fall back to a hardcoded 1.12 path when
`HDF5_DIR` was unset, so `module load cray-hdf5-parallel/1.14.3.7` inside a
non-login shell that never initialised Lmod produced a 1.12 stage with no
warning. **Since 2026-08-06 it refuses instead**, naming `HDF5_DIR` and the
`module load` that sets it (same for `MPICH_DIR`). `CRAY_HDF5_PATH` /
`CRAY_MPICH_PATH` still take precedence — an explicit path is a stated
fact, not a guess.

**`MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is
not linked`.** *Looks like:* an immediate MPI abort (rank 0 rc=255, peers
segfault) the moment anything calls `MPI_Init_thread`. *Cause (fixed
2026-08-06):* `in_container.sh` exported `MPICH_GPU_SUPPORT_ENABLED=1`
**unconditionally**, which is right for the GPU leg (it is paired with an
`LD_PRELOAD` of the CUDA-12 `libmpi_gtl_cuda.so.0`) and wrong for a
CPU-platform launch that has no GTL to preload — so the CPU FFI leg was
unusable in-container without a manual override. The variable is now
**per platform**: `run_shifter.sh` resolves `LORRAX_PLATFORM` (inferred
from `JAX_PLATFORMS` when unset) and `in_container.sh` exports `1` on
`gpu`, `0` on `cpu`. It is re-derived inside the container rather than
passed through because shifter's `--module=mpich` *unsets*
`MPICH_GPU_SUPPORT_ENABLED` on the way in; the two `LORRAX_*` variables
survive that and carry the intent.

So the CPU leg is now simply:

```
JAX_PLATFORMS=cpu src/ffi/cpp/run_shifter.sh python3 …
```

*Escape hatch:* `LORRAX_MPICH_GPU_SUPPORT=0|1` overrides the inference
outright. (The CPU cells in [Certification](#certification) predate the fix
and were run with the old manual
`… in_container.sh env MPICH_GPU_SUPPORT_ENABLED=0 python3 …`.)

**`ldd` on the login node lies about the MPI closure.** *Looks like:*
`libmpi_gnu_91` and `libmpi_gnu_123` resolving to two distinct real files,
i.e. an apparent two-MPI-ABI defect. **There is no such defect** — the second
name is a deliberate SONAME alias that only resolves inside the container.
This inspection produced a false defect report on 2026-08-05, retracted
2026-08-06. *Rule: any claim about the library closure must be measured
inside the container, on a compute node.* Mechanism, the alias, and the
`gate_one_mpi.sh` in-container gate: [`ffi_layout.md`](ffi_layout.md) §7c,
which owns it.

**A gather at scale.** *Used to look like:* a routing banner naming
`H5PY_ALLGATHER`, then a run that OOMs rank 0 or takes forever. That tier
no longer exists ([history](#tiers-history)); if you see a whole global
array on one rank now, it is a caller doing its own `process_allgather`,
not SlabIO.

**`ad_cray_write_coll.c:669` OOM.** *Looks like:* an MPI-IO abort inside
the collective write at ≳ 1 GiB per rank on Cray MPICH. Historical, from
before the current defaults; not reproduced at 512 MiB/rank in this
campaign. `LORRAX_PHDF5_COLLECTIVE_WRITES=0` is the escape hatch, at the
cost documented above — prefer splitting the write.

**Slow first write with no other symptom.** Check the file's Lustre
layout: `lfs getstripe -c -S <file>`. A 1-stripe file is capped near
0.65 GiB/s no matter how many ranks write it.

---

## Build provenance {#provenance}

Both Perlmutter build legs now stamp a `PROVENANCE` file beside the `.so`
(`src/ffi/cpp/stage/stamp_provenance.sh`, called from
`src/ffi/cpp/build.sh` and `config/perlmutter/build_ffi_host.sh`). It
records the git rev, branch, dirty flag, sha256, build time, host,
exported-symbol count, and the leg's resolved vendor paths.
`ffi_loader.build_provenance()` prints it in every run's startup report.

This is not bookkeeping. On 2026-08-05 a 4-node log was analysed by
inspecting the on-disk `liblorrax_ffi.so`, which had been rebuilt **seven
minutes after that log's last write** — so the ELF being read was provably
not the ELF that produced the numbers, and nothing on disk said so. An
unstamped build makes "which source produced this artifact" permanently
unanswerable, and this project asks that question constantly.

---

## Reproducing {#reproducing}

The gate and sweep drivers used for everything on this page are
`slabio_gate.py` and `slabio_sweep.py` (world-size assertion, forced
backend, bit-exact round-trip, geometry-invariant payload digest;
`--router-only` prints what `auto` would choose without doing I/O). Both
must be launched with the raw `srun` line in
[Launcher requirements](#launcher) — a measurement taken in the wrong
environment is worse than no measurement.

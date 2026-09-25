# SlabIO — the sharded-slab HDF5 transport

`file_io.slab_io.SlabIO` moves mesh-sharded arrays between device memory and
HDF5 files, one hyperslab per rank, through collective MPI-IO in the phdf5
FFI. This page owns the tile contract, the caller-facing API and what a call
site may and may not assume of it, close-time error agreement and the commit
receipt, the one-owner-per-file rule, the launcher requirement, the striping
and collective-I/O rules, the restart-read path, the HDF5 operation journal,
and SlabIO's refusals. It does not own the rulings behind them
([`decisions.md`](decisions.md) 2026-08-04, 2026-08-05), the native layer and
each knob's effective default ([`ffi_layout.md`](ffi_layout.md) §5–§7), knob
spellings ([`../dev/env_vars.md`](../dev/env_vars.md)), padded-axis receipts
([`padding.md`](padding.md)), or the restart bundle's contents
([`../drivers.md`](../drivers.md)).

`file_io.slab_io` is L3 and imports only downhill. The FFI is imported
lazily, so `file_io` imports without a built library.

---

## The tile contract {#contract}

**Each rank reads and writes only its own hyperslab. SlabIO never
materialises more than one rank's tile.** The design envelope does not fit
one node: `V_qmunu` at `(nq, N_μ, N_μ) = (64, 20000, 20000)` complex128 is
381 GiB, so a route that gathers a global array onto one rank cannot run the
workload it exists for.

- `write_slab` takes the operand in its existing sharding and writes each
  rank's local shard. It does not reshard, gather or replicate an operand
  sharded on the handle's mesh.
- `read_slab` with a `partition_spec` returns a `jax.Array` sharded that
  way; each rank reads only its block.
- The file holds the logical array. Its bytes do not depend on the mesh or
  rank count that wrote it, and pad rows never reach disk.

**Host memory**, per open handle:

- **Write staging (CUDA leg).** One pinned buffer grows to the largest
  local tile written. `read_slabs` also uses this buffer, as at most three
  chunks of max(one output row, 256 MiB) each.
- **Read staging.** A second buffer is sized to the local tile being read.
  A read buffer larger than 32 MiB is freed once its copy completes.
- **Host leg.** Writes read the XLA buffer in place, so only reads stage.

Buffers round up to 2 MiB and are freed at `close`. Pinned memory does not
show in `/proc`; `ffi.io.staging_totals()` returns
`(live_contexts, staged_bytes)`.

**Device memory.** `write_slab` holds a reference to its operand until that
operand's `H5Dwrite` completes. The process's one writer queue holds at most
two queued writes plus the one in flight, and a further `write_slab` blocks.

**Writes leave in file order.** `_file_order_plan` decides how each
`write_slab` leaves, from the operand's shape, valid extent and sharding and
the dataset's extent.
- **Row-block pieces, written independently.** The operand is cut into pieces of
  one index of each leading axis by `rows` rows of a split axis `k` by every later
  axis whole. Each piece is redistributed on the device so that rank r holds rows
  `[r·rows/P, (r+1)·rows/P)`, then written with `lorrax_phdf5_write_independent`
  (independent `H5Dwrite`). Each rank then writes one contiguous file run per
  leading index. A piece is at most `_FILE_ORDER_PIECE_BYTES` (64 MiB) per rank,
  so with the queue below the redistribution adds about 256 MiB of staging per
  rank. Only unsharded axes are cut or indexed; a sharded split axis is taken
  whole, because cutting a sharded axis would make XLA gather it.
- **As-is, independent.** An operand that already is such row blocks is written
  independently, with no copy.
- **As-is, collective.** Everything else keeps two-phase collective `H5Dwrite`
  (`lorrax_phdf5_write`): P = 1, a tiny slab, a layout that could only be cut by
  slicing a sharded axis (a face `(q, μ_X, ν_Y)` tile too large to take whole),
  or pieces whose per-rank runs would be shorter than 1 MiB.
- **Why:** at P16 on Lustre, the shared-pole bank shape runs at 2.8 GB/s as row
  blocks against 2.0 GB/s collective, and a WFN_qp G window at 4.2 against 2.6.
  Independent writes of short runs lose (76 kB runs: 0.8 GB/s).
- **Unchanged:** the bytes in the file, the dataset layout and every call
  signature.

**One collective lane per process.** Every handle's asynchronous writes go
through one queue and one worker (`_slab_io_ffi._CollectiveLane`), so
collective HDF5 calls leave in program order, which is the same on every
rank. Before any handle's next HDF5 call (open, create, read, write, close),
every handle's queued writes and in-flight `read_slabs` finish, including
reads on that same handle. An async union read can overlap compute until the
next HDF5 call; a same-handle metadata call also needs the native reader to
leave HDF5 first. With one writer thread
per handle, three files with queued writes deadlocked a CrI3 run: each rank
matched the three files' collectives in its own order.

**Two calls are not tile-bounded, and cost O(global array) per process:**

- `read_slab` **without** `partition_spec` reads the whole requested extent,
  replicated, on every rank.
- `write_slab` of an operand that every process already holds whole (host
  numpy, a single-device or process-local array, a replicated array) is
  staged replicated from each process's own copy, with no collective.

Use them only for O(1)-sized data. Bulk arrays go in on the handle's mesh
and come out with a `partition_spec`.

A sharded operand on another mesh is never gathered. On a mesh over the same
devices in the same order that differs only in axis names, it is relabelled
onto the handle's mesh and each device keeps its shard. Anything else refuses
with `GATE slab_io_foreign_mesh`, because writing it would gather the whole
array to every process's host (decisions 2026-08-05).

## One transport per geometry {#one-transport}

<a id="tiers-history"></a>
No caller selects a transport: there is no backend argument, env var or
deck key. The deck keys `slab_io` and `use_ffi_io` refuse at parse time
("must be removed"). The backend follows from the mesh:

| mesh | backend | library and data path |
|---|---|---|
| GPU mesh, `px·py = jax.process_count()` | `_slab_io_ffi._FfiBackend` | `liblorrax_ffi.so`: D2H/H2D through pinned staging on a private non-blocking CUDA stream |
| CPU mesh, `px·py = jax.process_count()` | `_slab_io_ffi._FfiBackend` | `liblorrax_ffi_host.so`: same C++ handlers; `H5Dwrite` reads the XLA buffer in place |
| emulated CPU mesh (`common.collectives.mesh_is_emulated`: one process owning a multi-device mesh) | `_slab_io_serial._SerialBackend` | h5py, one shard at a time |

There is no fallback. A deployment that cannot run `_FfiBackend` refuses at
open ([Availability](#availability)), and nothing demotes to a gathering
path (decisions 2026-08-05).

`_SerialBackend` refuses `process_count() > 1`, a non-emulated mesh, a
non-CPU mesh, and a mode outside `w`/`a`/`r`, all at construction. Its
staging is one shard; the result of an emulated read is global-sized because
this process owns every shard. It writes no commit receipt.

---

## The API {#api}

```python
from file_io.slab_io import SlabIO

with SlabIO(path, mode="w", mesh=mesh) as io:
    io.create_dataset("V_qmunu", shape=(n_q, n_mu, n_mu), dtype=np.complex128)
    io.write_slab("V_qmunu", V)                   # V may carry pad rows
with SlabIO(path, mode="r", mesh=mesh) as io:
    V = io.read_slab("V_qmunu", partition_spec=P(None, "x", "y"))
```

A caller states a path, a mode, a mesh and logical shapes. It does not state
a backend, striping, ROMIO hints, or a mesh-divisible extent.

| call | collective | effect |
|---|---|---|
| `SlabIO(path, *, mode, mesh)` | yes | Opens collectively. `mode` is required: `"w"` replaces the inode, `"a"` appends, `"r"` reads. `mesh` is required. `path` may be a `Path`. |
| `create_dataset(name, *, shape, dtype, attrs=None)` | yes | Creates a dataset at its logical shape. An identical existing dataset is reused; a different shape or dtype refuses on every rank. Drains queued writes first. `attrs` are stamped at close. |
| `write_slab(name, A, *, offset=None, global_shape=None, valid_shape=None)` | yes, **asynchronous** | Enqueues and returns. Bytes are on disk after `sync_writes()` or `close()`. |
| `read_slab(name, *, shape=None, dtype=None, offset=None, valid_shape=None, mesh=None, partition_spec=None, as_numpy=False)` | yes, synchronous | Returns a `jax.Array` sharded `partition_spec` on `mesh` (default: the handle's). `as_numpy=True` returns host numpy through `device_get`, which is valid for a replicated read. |
| `read_slabs(name, *, shape, offsets, valid_shapes, partition_spec, window_axis, dtype=None, mesh=None)` | yes (every rank calls it; a band-block transfer is independent) | n windows of one slab shape, stacked on a new axis at `window_axis`, read in chunks of output rows ([tuning](#tuning)). Returns an async result that the caller's next op sequences. |
| `read_small(name, *, dtype=None)` | yes | A whole small dataset as host numpy on every rank, scalars included. |
| `write_attr(name, value)` | no (queues) | A small replicated dataset, written by rank 0 at close. It replaces any existing dataset of that name. |
| `stamp_dataset_attrs(name, attrs)` | no (queues) | HDF5 attributes on an existing dataset, stamped by rank 0 at close. |
| `sync_writes()` | effectively | Waits for this rank's queued collective writes. Call it on every rank or on none. |
| `close()` / `__exit__` | yes | See [Close](#commit). |

The collective calls must be made by every rank, in the same order, with the
same dataset name and replicated arguments. A rank that skips one leaves its
peers blocked in the collective with no traceback.

### Shapes, padding and offsets

"Padding is SlabIO's business" (decisions 2026-08-04):

- **Datasets** are N-D with N ≥ 1, in `float32`, `float64`, `int32`, `int64`,
  `complex64` or `complex128`. Complex values are stored as the h5py
  compound `{r, i}`. A scalar goes through `write_attr` and `read_small`.
- **Write extent.** `A`'s sharding is read from `A.sharding`. Each sharded
  dim must divide by the product of the mesh axes that shard it, which is
  JAX's own constraint. The written extent is `min(A.shape, dataset − offset)`
  per dim, so pad rows past the dataset are dropped with no argument. A slab
  that starts past the end is an empty collective (every rank selects
  nothing).
- **Replicas.** A mesh axis that shards no dim of `A` is a replica axis. Only
  the rank at coordinate 0 on it writes; the others enter the collective with
  an empty selection. Overlapping selections are undefined under collective
  MPI-IO, so `LORRAX_PHDF5_DEDUP_REPLICAS=0` is for debugging only.
- **Dataset extent** is taken, in order, from this handle's `create_dataset`
  record, then `global_shape`, then `A.shape`. A padded `A` written to a
  dataset this handle has not created therefore creates a padded dataset:
  call `create_dataset` with the logical shape first. `global_shape` that
  contradicts a known extent refuses.
- **Read extent.** `shape=None` with a `partition_spec` returns the dataset
  rounded up by `mesh_divisible_shape`: each dim goes to a multiple of the
  product of the mesh axes that shard it, and everything past the dataset is
  zero. An explicit `shape` is returned exactly. It must be mesh-divisible
  under `partition_spec` and may exceed the dataset (the overhang is zero).
  The μ-carrier convention, `runtime.padding.padded_mu_extent`, rounds to a
  multiple of the total device count and belongs to [`padding.md`](padding.md).
  A caller that wants it passes it as an explicit `shape`.
- **`valid_shape`** is only the ragged-chunk override: a buffer whose tail is
  not part of this write or read. An override that runs past the dataset
  refuses on every rank.
- **`read_slabs` windows** share `shape`. `offsets` and `valid_shapes` are
  `(n, ndim)` with `n ≥ 1`. Each explicit valid extent must fit its slab
  shape and dataset extent; the same admission rule as `read_slab` checks
  it before native I/O. The caller guarantees that windows are pairwise
  disjoint and sorted ascending in row-major file order, and `window_axis` sits
  immediately before the dim that varies across windows. Production
  consumer: [`wfn_loader`](../services/wfn_loader.md).
- **Geometry of an existing dataset.** A dataset this handle did not create
  costs one collective FFI query (`lrx_phdf5_dataset_geometry`), cached per
  name.
- **On disk.** Datasets are contiguous, with space allocated at create
  (`H5D_ALLOC_TIME_EARLY`) and no fill (`H5D_FILL_TIME_NEVER`). Bytes that no
  write covered are undefined. Files carry `H5F_LIBVER_LATEST` bounds, so a
  reader needs an HDF5 at least as new as the FFI's.
- **Control operands.** `offset`, `valid_shape` and the handle vector are
  int64 device operands, so `jax_enable_x64` must be on
  (`runtime.bootstrap()` sets it).

### What a call site may not assume {#may-assume}

| a caller may not assume that… | because |
|---|---|
| `write_slab`'s data is on disk, or its failure visible, when it returns | a writer error is sticky; it is raised at `close`, or at the first read after the write (`read_slab`, `read_slabs`, `read_whole`, `padded_shape_for` on a write handle), agreed across ranks either way |
| `write_attr`, `stamp_dataset_attrs` or `create_dataset(attrs=…)` output is readable before `close()` | all three land in one rank-0 h5py reopen after `H5Fclose` |
| a rank other than 0 contributes deferred metadata | every rank queues, only rank 0's copy is written |
| `close()` is local | it drains, closes collectively, and agrees on errors across ranks |
| `mode="w"` keeps anything at `path` | rank 0 unlinks the path first |
| it can hold an h5py handle and a writable `SlabIO` on one path | refused by name ([one owner](#one-owner)) |
| two `SlabIO` handles may be open on one path in one process | refused unless both are `"r"`; identical read-only opens share one native context |
| a handle is re-openable, thread-safe or fork-safe | it is one collective handle over one mesh; use `with`, one at a time |
| it may raise between collectives on one rank | peers stay in the collective; an error is agreed only when every rank reaches `close` or a read after writes |

---

## Close, error agreement and the commit receipt {#commit}

`close()` runs the same sequence on every rank:

1. Drain this handle's queued writes from the collective lane. A writer
   error is recorded, not raised, so this rank still issues every collective
   its peers do.
2. `H5Fclose`, collectively.
3. Release the [one-owner](#one-owner) claim.
4. `common.collectives.agree_io_error(stage="SlabIO.data_close")`. Each rank
   contributes a 4096-byte receipt through the distributed host control
   store, with no device collective. If any rank failed (a writer error, a
   close error, an exception inside the `with` block of a `"w"`/`"a"`
   handle, or deferred metadata on a `"r"` handle), every rank raises the
   same
   `GATE io_global_commit: path=…; stage=…; failing rank=R; …`, naming the
   lowest failing rank, and nothing is published.
5. `"r"` handles return here.
6. `rank0_transaction(stage="SlabIO.metadata_commit")`. Every rank pulls the
   deferred values to host. Rank 0 then reopens the file with h5py `"a"`,
   writes the `write_attr` datasets and the dataset attributes, flushes, and
   sets the receipt to 1 last. Its verdict is agreed across ranks.

**The receipt** is the dataset `lorrax_io_committed`, int32, shape `(1,)`,
owned by `file_io.commit_state`:

- A `"w"` or `"a"` open creates it at 0 with an explicit collective write
  before any caller data. The explicit write is needed because datasets have
  no fill. If that write fails, the handle closes collectively before
  raising.
- `"a"` refuses an existing uncommitted file and sets the receipt to 0 before
  any mutation. `"r"` refuses an uncommitted file. The check runs on rank 0
  and is agreed across ranks.
- A file without the dataset (legacy, or written by `_SerialBackend`) is
  accepted.
- A caller that mutates a committed artifact outside SlabIO does it inside
  `rank0_transaction`, clearing the receipt first and setting it last.
  `tagged_arrays` does this for readiness flags and head scalars.

**Read after write.** On a `"w"`/`"a"` handle, a read door that follows
writes drains the queue and then calls
`agree_io_error(stage="SlabIO.read_after_write")` before it touches the file,
so no rank reads bytes that another rank failed to write. Writes are
collective, so every rank takes this agreement or none does; it is taken
once per batch of writes followed by a read.

Cost per write-mode close: two bounded host control exchanges and one rank-0
serial reopen. This is error agreement, not crash recovery. A process that
dies inside an MPI collective leaves its peers hung, and that case belongs to
the process supervisor.

---

## One HDF5 library per file {#one-owner}

<a id="metadata-cache-condition"></a>
A LORRAX process maps two independent HDF5 library instances: h5py's bundled
serial libhdf5 and the Cray parallel libhdf5 that `liblorrax_ffi*.so` links.
Each has its own metadata cache, open-file table and free-space manager, and
neither sees the other's. So a file held live by both, with either side able
to write, is undefined: one instance's flush lands on metadata the other
never saw.

`file_io.hdf5_owner` is a per-process, in-memory registry, not a lock. It
says nothing about other ranks and sees only opens that are declared to it
(`note_open` / `open_scope`). The declared opens are:

- every `SlabIO` open (`_FfiBackend` under `ffi`, `_SerialBackend` under
  `h5py`);
- the commit-state preflight;
- `mpa_store._h5`;
- the `open_scope` sites in `restart_bundle` and `sigma_output`;
- `close`'s rank-0 reopen. `close` releases its FFI claim between `H5Fclose`
  and that reopen.

A bare `h5py.File` elsewhere is invisible to the registry.

| condition | verdict |
|---|---|
| cross-stack **live overlap** where either side can write | **refused, always**: *"LORRAX HDF5 one-owner-per-file refusal"*, naming the file, the incoming open, every live foreign handle with its `where=`, and the fix (close the other handle first) |
| cross-stack live overlap, all read-only | allowed, counted |
| **sequential** cross-stack alternation after a write | counted; refused under `LORRAX_HDF5_ONE_OWNER=strict` |
| two libhdf5 mapped **and** a file written through both | `hdf5_owner.probe` (called at `gw_jax` startup and after each SC iteration) prints `UNSAFE-BY-A1`; raises under `strict` |

`LORRAX_HDF5_ONE_OWNER` is `measure` (default) or `strict`; any other value
refuses.

**Caller rule:** close your h5py handle on a path before constructing
`SlabIO` on it. Metadata goes through the library that holds the file: the
dataset-geometry query and `read_small` (`lrx_phdf5_read_whole`) are FFI
calls. A library built without those entry points takes a serial-h5py read
on a `"r"` handle (counted, and printed under `LORRAX_DEBUG_PRINT`) and
refuses on a writable handle. The fix is to rebuild the library, or to
pre-register the geometry with `create_dataset`.

---

## Availability {#availability}

`_FfiBackend` calls `assert_available()` before it touches the inode. The
check is two probes, cached per process:

1. **Library.** This platform's FFI library must export `lorrax_phdf5_write`
   (`ffi_loader.probe_target`). The probe reports one of three reasons, and
   each has a different fix: "unknown target" (rebuild: `src/ffi/cpp/build.sh`
   for the CUDA leg, `config/perlmutter/build_ffi_host.sh` for the host leg),
   "could not be loaded" (fix `LD_LIBRARY_PATH` or the bind-mounts, checked
   inside the container on a compute node), "does not export" (a stale `.so`
   is first on the path).
2. **MPI bootstrap.** Either a launcher PMI/PMIx variable is present, or
   `lrx_phdf5_init_mpi` succeeds in a throwaway subprocess. The probe runs in
   a subprocess because Intel MPI aborts, rather than returning an error, on
   a PMI-less init.

A failure raises *"SlabIO REFUSED: this stack cannot write one tile per
rank"*, naming the stage (`loader`, `probe` or `mpi`), the reason, the run
geometry and the fix. `probe_availability()` is the same check as a
non-raising pre-flight. `probe_read_availability(platform)` probes the
union-read target and is not cached. Nothing here keys on process count.

---

## Launcher requirements {#launcher}

- **One process per device.** The mesh is 2-D, `('x', 'y')`, with
  `px·py = jax.process_count()`; `ffi.io.open_file` refuses otherwise. The
  path is on a filesystem every rank shares.
- **MPI world = JAX world.** At the first collective open, `_assert_mpi_world`
  compares `MPI_Comm_size(MPI_COMM_WORLD)` (probed through ctypes, with the
  MPICH or Open MPI ABI detected by symbol) to `jax.process_count()`. A
  mismatch refuses on every rank, with no downgrade. An unprobeable world
  refuses unless `LORRAX_PHDF5_REQUIRE_MPI_WORLD=0`, which downgrades it to a
  warning. `LORRAX_PHDF5_SKIP_MPI_WORLD_CHECK=1` disables the check, for
  debugging only. The check exists because a PMI flavour that does not match
  the MPI library gives every rank a singleton `MPI_COMM_WORLD`, and
  "collective" I/O then runs as P unsynchronised writers
  ([`ffi_layout.md`](ffi_layout.md) §7a). A gate that needs the check without
  opening a file compares `file_io._slab_io_ffi._probe_mpi_world_size()[0]`
  with `jax.process_count()` after MPI is initialised.
- **`MPI_THREAD_MULTIPLE`.** The collective lane's worker drives MPI-IO
  while other threads use MPI. A lower granted level aborts the whole world before any
  collective (`src/ffi/cpp/common/mpi_thread_guard.h`).
- **The launch line** belongs to the machine page. On Perlmutter, Shifter's
  Cray MPICH needs `srun --mpi=cray_shasta`, which is the `run_shifter.sh` /
  `LORRAX_MPI_TYPE` default; `pmi2` and `pmix` give singleton MPI there
  ([perlmutter](../environment/machines/perlmutter.md)). Frontera's line is on
  [frontera](../environment/machines/frontera.md).

---

## Striping and collective I/O {#tuning}

<a id="defaults"></a>
**Transfers are collective (`H5FD_MPIO_COLLECTIVE`), except a
`read_slabs` whose windows are band blocks. Metadata operations are
independent, and ROMIO's collective-buffering hints are left to ROMIO.**
Two-phase aggregation turns each rank's strided 2-D tile into a few large
contiguous writes per aggregator. Independent I/O issues one short write per
row-run of the tile, so its throughput depends on how the stripe layout
happens to line up with those rows.

`read_slabs` decides per call, the same way on every rank. If every window
is whole along each dim after the one the windows vary in (a WFN band block:
long contiguous file runs), the read is independent. There, two-phase
aggregation only re-ships every byte through 16 MiB collective-buffer
rounds: at VI3 12×12 P16, bands 0:360 (120.2 GB,
`runs/runtime/p2d_io_20260924`), it took 56.8 s cold and 44.4 s warm, against
17.8 s and 7.6 s independent. Short strided windows (shared-pole bank
rectangles) stay collective.

Either way the handler reads each chunk of output rows into packed pinned
staging, with the memory side contiguous. It holds two chunks, plus one
padded chunk on CUDA, each at most max(one row, 256 MiB). It then places the
data on the device run by run, or through the padded chunk when the runs are
short. Pad cells are zeroed. The bytes are identical to a one-shot union
read, because HDF5 maps the n-th selected file element to the n-th selected
memory element in both. HDF5's own selection iterator costs about 1 ms per
run on an OR'd selection, so the runs are enumerated directly. The
overrides
`LORRAX_PHDF5_COLLECTIVE_WRITES=0`, `LORRAX_PHDF5_INDEPENDENT=1` (reads),
`LORRAX_PHDF5_COLL_META=1` and `LORRAX_PHDF5_CB_*` exist for A/B runs.
[`ffi_layout.md`](ffi_layout.md) §6 lists every field's effective default,
including the 4 MiB `H5Pset_alignment` (`LORRAX_PHDF5_ALIGN_MB`).

**The stripe count is the aggregator count.** ROMIO sets
`cb_nodes = min(striping_factor, nranks)`, read back from
`MPI_File_get_info` (sandbox CLAIMS 75), so a fixed stripe count caps the
number of aggregators. The policy is one pure function,
`file_io._slab_io_ffi._stripe_policy(nranks)`:

```
striping_factor = clamp(nranks, 4, 128)
striping_unit   = the power of two nearest in log2 to (nranks/16) MiB, clamped to [1, 4] MiB
                  → 1 MiB below 22.6 ranks, 2 MiB to 45.3, 4 MiB above
```

- The lower clamp exists because a one-stripe file is a single-aggregator
  ceiling that more ranks cannot raise.
- The unit is capped at 4 MiB because a stripe unit larger than the
  per-rank tile starves aggregators, and at the envelope the per-rank tile
  of one q-slice of `V_qmunu` is about 6 MiB at 1024 ranks.
- No run at `nranks ≥ 128` has exercised the upper clamp (sandbox CLAIMS 77).

**Cost model** (sandbox CLAIMS 69): at 64 ranks, the 381 GiB envelope
`V_qmunu` writes at 13.2 GiB/s under the policy (64 × 4 MiB) against
7.4 GiB/s at a fixed 16 × 1 MiB. Cold reads run at about write throughput
(2.9 GiB/s at 16 ranks).

**One policy, two writers.** The C++ `stripe_policy_count` and
`stripe_policy_unit` (`context.cc`) transcribe `_stripe_policy`, and
`tests/test_slab_io_routing.py::test_cpp_stripe_policy_transcribes_the_python_one`
compiles and diffs them. `_FfiBackend` exports the resolved values into
`LORRAX_PHDF5_STRIPE_COUNT` and `LORRAX_PHDF5_STRIPE_SIZE_FS` before opening,
so a run's environment records its layout. An explicit value overrides the
policy. Both writers refuse a malformed value and a negative count (a
negative count means every OST, which is the maximum-contention layout).
`_SIZE_FS` takes `<number>[k|M|G]` with a single suffix letter; a bare
number is bytes.

**A layout belongs to the inode and is fixed at creation.**

- For `"w"`, rank 0 unlinks the path and all ranks barrier, so `H5Fcreate`
  makes a fresh inode that takes the hints. If the unlink fails, the open
  refuses, because truncation would keep the old layout.
- `"a"` and `"r"` keep the existing layout.
- After a `"w"` create, rank 0 reads the `lustre.lov` xattr and warns if the
  granted layout differs from the request.

**Reads are governed by the file's own layout**:
`cb_nodes = min(file stripe count, nranks)`. On a `"r"` open of a file of
1 GiB or more, rank 0 prints the actual layout and warns on a one-stripe
inode, which gives one read aggregator at any rank count. A `pw2bgw`
`WFN.h5` produced in a default 1 × 1 MiB directory is that case. The fix is
to `lfs setstripe` the directory before the file is produced, or to use a
migrated copy. A `read_slab` on a file of 1 GiB or more also prints `START`,
then `PROGRESS` every 60 s, then `COMPLETE`.

**Measuring.**

- `LORRAX_DEBUG_PRINT=1` dumps, on rank 0, the hints ROMIO retained.
- `LORRAX_SLAB_IO_TIMING=1` records API and native `H5D` wall times. It
  needs a library that exports `lrx_phdf5_close_timed`.
- A read timed on the node set that wrote the file measures the page cache.
  Write and read on disjoint node sets
  (`tests/bench/slabio_scaling_bench.py --phase write|read`).

---

## Restart bundles {#restart}

`tmp/isdf_tensors_<N_μ>.h5` is written by
`tagged_arrays.write_restart_state_to_h5` through SlabIO at the logical μ
extent, so a bundle reads at any device count. Its contents are listed in
[`drivers.md`](../drivers.md). `restart_bundle.read_restart_state_from_h5`
reads it in two passes, and never holds an h5py handle and a SlabIO handle on
the file at once:

1. **One read-only h5py pass.** It runs admission (`_require_current`). An
   uncommitted file refuses at the commit gate. A missing raw-parent
   dataset, any `psi_full_*`, or a stale band-window schema refuses with
   "regenerate it with gwjax at main >= 891047f4". The same pass reads the
   dataset shapes and the small replicated data (`enk_full`, `G0_mu_nu`,
   stamps, q-IBZ unfold tables).
2. **One `SlabIO(mode="r")`.** It reads two groups as per-rank tiles:
   - the N_μ²-class tensors `V_qmunu`, `S_qmunu` and `V0_noG0_munu`, as
     `P(…, "x", "y")`, with q-IBZ datasets unfolded to full q after the
     read;
   - the parent ψ faces `psi_parent_y` `(nk, n, s, μ)` and
     `psi_parent_y_mun` `(nk, s, μ, n)`, plus the transverse pair at its own
     μ extent.

   μ is read at its padded carrier and bands at the band carrier. The zero
   pad rows come from the read itself.

Refused before any tensor bytes move: a spinor extent outside {1, 2, 4}, a
torn transverse pair, charge and transverse faces with different
parent-row counts, and a stamped `n_rmu_transverse_logical` that differs
from the dataset's μ extent. The read is an element selection into the
band-distributed face specs.

---

## Control operands and handle validation {#stream-race}

`(ctx handle, ds_id)`, `offset` and `valid_shape` travel as replicated int64
device operands rather than FFI attributes. As a result, one compiled
`shard_map` serves every file, dataset and process, and the persistent
compile cache hits. A non-int64 operand refuses (`ffi.io.require_control_i64`).

- **CUDA leg.** The handler copies the operands to host with
  `cudaMemcpyAsync` on the XLA stream followed by `cudaStreamSynchronize`
  (`platform_seam.h::copy_index_to_host`). The host leg uses `memcpy`.
- **Handle check.** Every handle is checked against the library's
  live-context registry before it is dereferenced. An unknown handle refuses
  with *"stale or foreign ctx handle -- stream race or double close"*.
  <a id="s1"></a>A second close of the same handle prints the same diagnosis
  and tears nothing down.
- <a id="s3"></a>**Stale library.** Two refusals mean the operands were read
  before XLA wrote them, which is what a `.so` built before the
  stream-ordered copy does: a *"logical slab out of bounds"* refusal quoting
  a nonzero `offset_base` on a call that passed no offset, and a
  `[descriptor forensics]` line reading a small-magnitude double. Check that
  `strings -a "$LORRAX_FFI_SO" | grep -q 'stale or foreign ctx handle'`
  succeeds.
- <a id="odr-host-so"></a>**No shared symbols between legs.** The CUDA and
  host libraries share no dynamic symbol: the host leg's C ABI carries a
  `_host` suffix, and each leg exports through `exports_{cuda,host}.map`. A
  host library built without this, loaded beside the CUDA one, aliases
  `PhdfCtx` across two struct layouts and shows the same garbage-descriptor
  symptoms. To check, `comm -12` of `nm -D --defined-only` on the two legs,
  filtered to `lrx_|lorrax_ffi`, must print nothing;
  `src/ffi/cpp/gate_one_odr.py` checks a live process.
  `tests/KNOWN_FAILURES.md` L1 owns the history.

---

## The operation journal {#journal}

The journal is an opt-in, per-rank, line-buffered log of every HDF5
operation, written **before** each call. A rank that dies inside native HDF5
therefore leaves a last line naming the operation, the file and the ctx
handle.

| | |
|---|---|
| module | `src/file_io/h5_journal.py` |
| toggle | `LORRAX_H5_JOURNAL`: `0` off (default), `1` on, `sync` fsync after every line (segfault capture only); any other value refuses |
| output | `LORRAX_H5_JOURNAL_DIR`, default the working directory: `h5_journal.rank<R>.log` |
| crash ring | the last 256 lines, dumped to `h5_journal_crash.rank<R>.txt` on a registry refusal, on an exception crossing a SlabIO method, and at exit after any journaled refusal |

Line format (fixed key order):

```
t=<monotonic.6f> rank=<R> stack={h5py|ffi} op={open|close|create|read|write|attr_r|attr_w}
  path=<abs> handle=<id-or-ctxptr> ds=<name|-> off=<t> cnt=<t> mode=<r|w|a|->
  owner=<live-verdict> rc={ok|refused:<first 40 chars>}
```

- `owner` is `hdf5_owner.live_verdict(path)`: which stacks hold a live handle
  at that instant (`free`, `ffi:1w`, `h5py+ffi:2r`). The outcome of a check
  is `rc`.
- An `open` line carries `handle=-`. The handle appears on SlabIO's
  completion line.
- One slab read or write is one line. One SlabIO open writes three lines:
  the registry's claim, the FFI's issue-time open, and the completion line.
  The commit-state preflight adds its own lines (on `"a"`/`"r"` of an
  existing file), and so does the receipt's `create` (on `"w"`/`"a"`).
- Hooks are the existing choke points: `hdf5_owner.note_open`/`note_close`,
  every `SlabIO` method, and `_slab_io_ffi`'s lifecycle calls and h5py
  touches.
- An I/O error on the journal file disables it with one warning. It never
  stops a run.

**Reading one.** `tail` rank 0's file, then `grep "handle=<ctxptr>"` across
all ranks. A handle that appears under two `path=` values, or after its own
`op=close`, is a stale context, not an HDF5 bug.

---

## Refusals {#failures}

| message (abridged) | cause | fix |
|---|---|---|
| `SlabIO REFUSED: this stack cannot write one tile per rank` | [availability](#availability) probe failed at stage `loader`, `probe` or `mpi` | the stage's fix, printed in the refusal |
| `MPI_Comm_size(MPI_COMM_WORLD)=… but jax.process_count()=…` | launcher PMI flavour does not match the MPI library | [launch line](#launcher) |
| `could not verify the MPI world size` | no probeable libmpi | fix the MPI closure; `LORRAX_PHDF5_REQUIRE_MPI_WORLD=0` only if the launch is known good |
| `mesh p×q=… != jax.process_count()` | a non-emulated mesh that is not one device per process | one process per device |
| `MPI granted thread level … < MPI_THREAD_MULTIPLE` (abort) | MPI initialised below `THREAD_MULTIPLE` | the machine page's certified launch |
| `phdf5 …: control operand must be int64` | `jax_enable_x64` off | `runtime.bootstrap()` |
| `LORRAX HDF5 one-owner-per-file refusal` | live h5py and FFI handles on one path, one writable | close the other handle first |
| `this path is already open with mode=…` / `this path already has a live context` | a second `SlabIO` on a path open in this process, not both `"r"` | close the first |
| `SlabIO mode='w': could not replace existing file` | rank-0 unlink failed | delete the file or fix permissions |
| `phdf5 ensure_dataset: dataset '…' already exists with shape …` | shape or dtype differs from the existing dataset | `mode="w"`, a new name, or delete the file |
| `GATE slab_io_foreign_mesh: write_slab …` | a sharded operand on a mesh that is not the handle's and not the same devices in the same order ([contract](#contract)) | produce it on the run's `mesh_xy`; an O(1) array may be gathered at the call site with `gather_to_host` |
| `write_slab …: global_shape=… contradicts the dataset's extent` | `global_shape` on a known dataset | drop `global_shape` |
| `slab shape must be non-empty` | `read_slab` on a scalar dataset | `read_small` |
| `logical slab out of bounds … refused identically on every rank` | offset or `valid_shape` past the dataset; with a nonzero `offset_base` on a no-offset call, a [stale library](#s3) | fix the request, or rebuild |
| `stale or foreign ctx handle` | double close, a handle from the other platform leg, or a [stale library](#stream-race) | close once; rebuild |
| `GATE io_global_commit: … failing rank=R` | some rank's write, close or metadata publication failed | read rank R's error; rebuild in a new run directory |
| `GATE io_global_commit … artifact is not globally committed` | opening an incomplete artifact `"r"` or `"a"`, or restart admission | rebuild in a new run directory |
| `SlabIO mode='r' acquired deferred write metadata` | `write_attr` or `stamp_dataset_attrs` on a `"r"` handle | reopen with `mode="a"` |
| `LORRAX_PHDF5_STRIPE_COUNT=… is refused` / `is not a valid stripe count` | negative or malformed override | a positive integer, or unset |
| `LORRAX_PHDF5_STRIPE_SIZE_FS=… is not a valid stripe size` | wrong grammar (e.g. `4MiB`) | a number with one `k`, `M` or `G` suffix (e.g. `4M`), or unset |
| `learning the geometry of dataset … needs either the FFI metadata entry points …` | library without `lrx_phdf5_dataset_geometry`, writable handle | rebuild, or `create_dataset` first |
| `LORRAX_SLAB_IO_TIMING=1 requires a rebuilt PHDF5 provider` | library without `lrx_phdf5_close_timed` | rebuild, or unset |
| `Input key 'slab_io'` / `'use_ffi_io'` `… must be removed` | retired deck key | delete the line |

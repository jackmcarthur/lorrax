"""Parallel-HDF5 (MPI-IO) FFI service — the ONE io module of the facade.

Wave-2 merge (docs/architecture/ffi_layout.md §3/§6, executed 2026-08-01):
the former ``ffi.phdf5`` package — ``context.py`` (file/handle lifecycle),
``read.py`` (sharded slab / k-chunk readers) and ``write.py`` (sharded slab
writer) — concatenated verbatim into one module, imports de-duplicated,
NOTHING renamed.  ``ffi.phdf5`` remains as a re-export shim until its
consumers (file_io/_slab_io_ffi.py, file_io/wfn_loader.py) migrate; deleting
the shim is the gate that the migration is complete.

Each process reads/writes its local shard directly to a hyperslab of the
shared HDF5 file via MPI-IO — no gather through rank 0.  See
``src/ffi/AGENTS.md`` for required environment and ``src/ffi/PORTING.md``
for per-cluster setup.

Section docstrings from the merged modules follow inline.
"""
from __future__ import annotations

import atexit
import functools
import threading
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from common.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .common import ffi_loader
from .common.ffi_loader import get_lib

__all__ = [
    # lifecycle (was phdf5/context.py)
    "open_file", "close_file", "platform_for_handle", "validate_mesh_2d",
    # readers (was phdf5/read.py)
    "handle_vector",
    "ffi_read_call", "ffi_read_kchunk_call", "ffi_read_kchunk_union_call",
    "read_sharded_slab", "read_kchunk_union_sharded",
    # writer (was phdf5/write.py)
    "write_sharded_slab", "ffi_write_call",
]

# =========================================================================
#  Lifecycle  (merged from phdf5/context.py; original docstring below)
# =========================================================================
"""Python-side lifecycle for the parallel-HDF5 FFI.

``open_file`` is collective: the underlying ``lrx_phdf5_open`` calls
``MPI_Init_thread(MPI_THREAD_MULTIPLE)`` if needed, duplicates
``MPI_COMM_WORLD``, and invokes ``H5Fcreate``/``H5Fopen`` with
parallel-IO property lists cached on the context.  The returned
``int64`` handle is the address of a C++ ``PhdfCtx`` struct.

File handles are cached per-process so repeated ``open_file(path)``
calls return the same handle until ``close_file`` is called.
"""
_LOCK = threading.Lock()
# path -> (int64 ctx handle, owning platform "CUDA"|"cpu", open mode).
# The platform is recorded at open and used for EVERY lifecycle call on
# the handle: a
# PhdfCtx* allocated by one platform's .so (its heap, its HDF5/MPI state)
# must never be handed to the other library — the dual-platform
# (JAX_PLATFORMS=cuda,cpu) hazard the mesh routing exists for.  Routing
# only the open by mesh platform while close/ensure_dataset followed the
# JAX default backend was exactly that bug (audit fix/zq 2026-07-28).
_FILE_CTXS: Dict[str, Tuple[int, str, str]] = {}


def validate_mesh_2d(mesh: Mesh) -> tuple[int, int]:
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"mesh must have axes ('x','y'); got {mesh.axis_names}")
    return int(mesh.shape["x"]), int(mesh.shape["y"])


def _platform_for_mesh(mesh: Mesh) -> str:
    """Which FFI library owns the collective context for ``mesh`` — "CUDA"
    for a GPU mesh, "cpu" for a CPU mesh.  The phdf5 read handlers are
    registered under both platform strings (same jax.ffi target names), and
    the collective open/dataset lifecycle must go through the matching .so.
    """
    try:
        plat = mesh.devices.flat[0].platform
    except Exception:
        plat = ""
    return "CUDA" if plat in ("gpu", "cuda") else "cpu"


_MODE_FLAGS = {"w": 0, "a": 1, "r": 2}


def open_file(path: str, *, mesh: Mesh, mode: str = "w") -> int:
    """Collective open/create of a parallel-HDF5 file.

    Parameters
    ----------
    path
        Absolute filesystem path for the HDF5 file.  Must be reachable
        by every JAX process (shared FS — Lustre/GPFS/NFS).
    mesh
        2-D ``Mesh`` labelled ``('x','y')``.  Shape must equal
        ``jax.process_count()``.
    mode
        ``'w'`` truncate+create, ``'a'`` append-or-create, ``'r'``
        read-only.

    Returns
    -------
    int
        Opaque handle (int64 address of the C++ ``PhdfCtx``).  Pass to
        ``write_sharded_slab`` and ``close_file``.
    """
    if mode not in _MODE_FLAGS:
        raise ValueError(f"mode must be w/a/r; got {mode!r}")
    platform = _platform_for_mesh(mesh)
    ffi_loader.get_lib(platform)
    p, q = validate_mesh_2d(mesh)
    if p * q != jax.process_count():
        raise ValueError(
            f"mesh {p}×{q}={p*q} != jax.process_count()={jax.process_count()}")

    with _LOCK:
        if path in _FILE_CTXS:
            # Handle reuse is deliberate (see the module docstring), but it
            # is only sound when the second caller wants the SAME access.
            # Silently handing back a 'r' context to a 'w' caller loses the
            # write; handing back a 'w' context to a 'w' caller is worse,
            # because ``_slab_io_ffi._replace_inode_for_write`` has already
            # unlinked the path on rank 0 — the cached ctx still points at
            # the ORPHANED inode, so every subsequent write lands in a file
            # with no name and the run finishes rc=0 with nothing on disk.
            # Rank-invariant (every rank performs the same opens), so this
            # refuses on every rank or on none.
            prev_ctx, prev_plat, prev_mode = _FILE_CTXS[path]
            if prev_mode != mode:
                raise RuntimeError(
                    f"phdf5 open_file({path!r}, mode={mode!r}): this path is "
                    f"already open in this process with mode={prev_mode!r} "
                    f"(handle {prev_ctx}).  Handles are cached per path, so "
                    f"you would get the mode={prev_mode!r} context back — "
                    f"and on mode='w' the target inode has already been "
                    f"unlinked, so the writes would go to an orphaned file. "
                    f"Close the existing SlabIO/handle first.")
            return prev_ctx
        ctx = ffi_loader.phdf5_open(
            path, p, q,
            int(jax.process_index()), int(jax.process_count()),
            _MODE_FLAGS[mode], platform=platform,
        )
        _FILE_CTXS[path] = (ctx, platform, mode)
        return ctx


def platform_for_handle(ctx_handle: int) -> Optional[str]:
    """The platform ("CUDA"/"cpu") whose library allocated ``ctx_handle``,
    or None for an unknown handle.  Every lifecycle call on a handle
    (``phdf5_close`` / ``phdf5_ensure_dataset`` / ``phdf5_open_dataset_ro``)
    must go through the owning platform's .so — this is the lookup the
    write-side lifecycle sites use to route theirs."""
    with _LOCK:
        for _ctx, _plat, _mode in _FILE_CTXS.values():
            if _ctx == int(ctx_handle):
                return _plat
    return None


def close_file(path_or_handle) -> None:
    """Collective close.  Accepts either a path (the original open_file
    argument) or the int handle returned from open_file.

    Routed to the platform library that OPENED the handle (recorded in
    ``_FILE_CTXS`` at open) — not the JAX default backend, which in a
    dual-platform process could be the other library and would then free
    a foreign PhdfCtx* against foreign HDF5/MPI state.  An unknown handle
    (not opened through this module) falls back to the default-backend
    library, the pre-existing best guess."""
    global _FILE_CTXS
    with _LOCK:
        platform = None
        if isinstance(path_or_handle, str):
            entry = _FILE_CTXS.pop(path_or_handle, None)
            ctx = entry[0] if entry is not None else None
            platform = entry[1] if entry is not None else None
        else:
            ctx = int(path_or_handle)
            # Drop any path entries pointing at this ctx, keeping its
            # recorded platform.
            for k in [k for k, v in _FILE_CTXS.items() if v[0] == ctx]:
                platform = _FILE_CTXS[k][1]
                _FILE_CTXS.pop(k, None)
        if ctx is not None and ctx != 0:
            # Drop this ctx's memoised dataset ids FIRST.  H5Fclose
            # invalidates every hid_t opened against the file, and the
            # ctx address is about to become reusable — a memo entry that
            # outlives the close is a stale ``ds_id`` waiting for the next
            # ``open_file`` that lands on the same address
            # (:func:`_forget_datasets_for_ctx`).
            _forget_datasets_for_ctx(int(ctx))
            # platform=None (unknown handle) follows the JAX default
            # backend inside ffi_loader — hardcoding "CUDA" would fail on
            # a CPU node where only the host lib exists.
            ffi_loader.phdf5_close(int(ctx), platform=platform)


def _atexit_close_all() -> None:
    """Close any files still open at process exit — catches forgotten
    close_file calls.  Runs on every process.  Each handle closes through
    its own recorded platform library."""
    with _LOCK:
        for path, (ctx, platform, _mode) in list(_FILE_CTXS.items()):
            try:
                _forget_datasets_for_ctx(int(ctx))
                ffi_loader.phdf5_close(int(ctx), platform=platform)
            except Exception:
                pass
        _FILE_CTXS.clear()
        _DS_ID_MEMO.clear()


atexit.register(_atexit_close_all)


# =========================================================================
#  Readers  (merged from phdf5/read.py; original docstring below)
# =========================================================================
"""Sharded-slab FFI readers for parallel-HDF5 datasets.

Two entry points, each wrapping a C++ handler defined in
``cpp/read_ffi.cc``:

===========================  ===========================================
``read_sharded_slab``         2-D dataset → ``P('x','y')``-sharded array.
                              Thinnest convenience wrapper.
``read_kchunk_union_sharded`` N-D dataset, ``n_kchunk`` **disjoint**
                              windows via ``H5S_SELECT_OR`` compound
                              hyperslab, **one H5Dread**.  Use when the
                              caller can provide per-k variable counts
                              so the windows are disjoint by
                              construction.
===========================  ===========================================

There were THREE until 2026-08-07.  ``read_kchunk_sharded`` wrapped
``PhdfReadKchunk``, which does n_kchunk SEQUENTIAL H5Dreads inside one
handler invocation, and it had zero callers in the tree — every reader of
variable-ngk WFN windows takes the union path, whose per-k counts make the
windows disjoint by construction and cost one collective instead of n.
The measurement that settled the question (2026-08-07, ``_measure_fold``):
n separate reads cost 1.4-3.6x the union call at every deck measured, and
the gap grows with n.  A wrapper with no callers and a losing measurement
behind it is not an option, it is a second spelling; DELETED.  Its C++
handler is still in the deployed ``.so`` pair and its removal is
registered to the owner, so ``ffi_read_kchunk_call`` below stays as the
1:1 binding of a handler that still exists.

Each high-level function returns a jitted ``shard_map`` closure: the
caller dispatches against it with runtime buffer arguments (offsets,
counts), and the same compiled module handles any offsets/counts
combination at the same shapes/dtypes.

Preferred public entry point remains :mod:`file_io.slab_io`, whose
``read_slab`` / ``read_slabs`` are the door onto these two.
"""

_TARGET_READ = "lorrax_phdf5_read"
_TARGET_READ_KCHUNK = "lorrax_phdf5_read_kchunk"
_TARGET_READ_KCHUNK_UNION = "lorrax_phdf5_read_kchunk_union"


# =============================================================================
#  Helpers shared by the three wrappers
# =============================================================================
def _encode_sharding_axes(
    mesh: Mesh, file_partition_spec: P, ndim_file: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(axis_count_per_dim, axis_flat)`` tuples the C++ handler
    consumes.  Defers to the shared encoder in
    ``file_io._slab_io_ffi`` so there's exactly one source of truth.
    """
    # Lazy import: file_io may not be on sys.path in some minimal test
    # harnesses that import this module directly.
    from file_io._slab_io_ffi import _sharding_to_axis_info
    return _sharding_to_axis_info(
        NamedSharding(mesh, file_partition_spec), ndim_file)


def _insert_at(seq: Sequence, axis: int, value) -> tuple:
    """Insert ``value`` at position ``axis`` in ``seq``, returning a tuple."""
    items = list(seq)
    items.insert(axis, value)
    return tuple(items)


def _register_and_open_dataset(fh: int, ds_name: str,
                               mesh: Mesh | None = None) -> int:
    # Drive the collective H5Dopen through the SAME platform library that
    # owns the context (CUDA on a GPU mesh, cpu on a CPU mesh) — so a
    # dual-platform process (JAX_PLATFORMS=cuda,cpu) with a CPU mesh doesn't
    # route the lifecycle call to the CUDA lib's HDF5/MPI state.  The read
    # ffi_call itself still resolves by lowering platform automatically.
    platform = _platform_for_mesh(mesh) if mesh is not None else None
    get_lib(platform)
    return ffi_loader.phdf5_open_dataset_ro(fh, ds_name, platform=platform)


# =============================================================================
#  Low-level FFI call wrappers (one per handler)
# =============================================================================
# Low-level padding contract: out_struct is the physical equal-block
# shard; valid_shape is the logical file prefix that C++ reads.
def handle_vector(ctx_handle: int, ds_id: int, mesh=None) -> jax.Array:
    """The ``(2,)`` int64 ``[ctx_handle, ds_id]`` buffer the FFI expects.

    Passed with ``P()`` so it is replicated — every rank sees the same
    pair, which is what lets the C++ dispatch keep treating its refusals
    as every-rank-or-none.  Defined here, next to the calls that consume
    it, so the read and write paths cannot disagree about the layout.

    ``mesh`` is accepted and ignored: the shard_map's ``in_specs``
    already does the replication, and materialising this through an
    explicit ``NamedSharding`` would run JAX's hidden ``assert_equal``
    all-gather on every call (scorecard AA.1).  ``file_io/_slab_io_ffi``
    builds the same vector through ``_replicated_i64_vector``, which uses
    ``device_put_process_local`` for exactly that reason.
    """
    del mesh
    return jnp.asarray([int(ctx_handle), int(ds_id)], dtype=jnp.int64)


def ffi_read_call(
    out_struct: jax.ShapeDtypeStruct,
    handle: jax.Array,
    offset_base: jax.Array,
    valid_shape: jax.Array,
    *,
    mesh_shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
) -> jax.Array:
    """Single-hyperslab read — one H5Dread of one rectangle.

    ``handle`` (``(2,)`` == ``[ctx_handle, ds_id]``), ``offset_base`` and
    ``valid_shape`` are int64 ``jax.Array`` buffers passed at runtime (not
    as FFI Attrs), so the shard_map closure compiles ONCE per ``(ndim,
    dtype, sharding)`` tuple and re-dispatches across chunks, datasets,
    files and processes.  See :func:`ffi_write_call` for the measurement
    that moved ctx_handle/ds_id out of the Attrs.
    """
    return jax.ffi.ffi_call(_TARGET_READ, out_struct)(
        handle,
        offset_base,
        valid_shape,
        mesh_shape=np.asarray(mesh_shape, dtype=np.int64),
        axis_count_per_dim=np.asarray(axis_count_per_dim, dtype=np.int64),
        axis_flat=np.asarray(axis_flat, dtype=np.int64),
    )


def ffi_read_kchunk_call(
    out_struct: jax.ShapeDtypeStruct,
    handle: jax.Array,               # (2,) == [ctx_handle, ds_id]
    offset_base: jax.Array,          # (n_kchunk, ndim_file) int64
    *,
    mesh_shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
    n_kchunk: int,
) -> jax.Array:
    """Sequential-reads kchunk — one handler invocation doing
    ``n_kchunk`` H5Dread calls into a packed pinned buffer.  Each of the
    per-k rectangles has the same shape; only its file offset varies.

    NO PYTHON CALLER since 2026-08-07: ``read_kchunk_sharded``, the only
    one, was deleted on the measurement above.  Kept because the
    ``PhdfReadKchunk`` handler is still exported by the deployed ``.so``
    pair, and a binding is how anyone reaches or debugs one; it goes out
    WITH the handler, which is registered to the owner.
    """
    return jax.ffi.ffi_call(_TARGET_READ_KCHUNK, out_struct)(
        handle,
        offset_base,
        mesh_shape=np.asarray(mesh_shape, dtype=np.int64),
        axis_count_per_dim=np.asarray(axis_count_per_dim, dtype=np.int64),
        axis_flat=np.asarray(axis_flat, dtype=np.int64),
        n_kchunk=int(n_kchunk),
    )


def ffi_read_kchunk_union_call(
    out_struct: jax.ShapeDtypeStruct,
    handle: jax.Array,               # (2,) == [ctx_handle, ds_id]
    offset_base: jax.Array,          # (n_kchunk, ndim_file) int64
    count_base: jax.Array,           # (n_kchunk, ndim_file) int64
    *,
    mesh_shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
    n_kchunk: int,
    kchunk_axis: int,
) -> jax.Array:
    """Compound-hyperslab kchunk — ONE H5Dread pulls ``n_kchunk`` disjoint
    windows via ``H5S_SELECT_OR``.  Caller supplies per-k counts so the
    per-k file rectangles are disjoint by construction.  ``kchunk_axis``
    locates the k axis inside ``out_struct`` (required so memspace
    iteration order matches filespace iteration order — see
    ``read_kchunk_union_sharded`` docstring).
    """
    return jax.ffi.ffi_call(_TARGET_READ_KCHUNK_UNION, out_struct)(
        handle,
        offset_base,
        count_base,
        mesh_shape=np.asarray(mesh_shape, dtype=np.int64),
        axis_count_per_dim=np.asarray(axis_count_per_dim, dtype=np.int64),
        axis_flat=np.asarray(axis_flat, dtype=np.int64),
        n_kchunk=int(n_kchunk),
        kchunk_axis=int(kchunk_axis),
    )


# =============================================================================
#  High-level convenience wrappers (build a jitted shard_map closure)
# =============================================================================
def read_sharded_slab(
    fh: int,
    ds_name: str,
    *,
    global_shape: tuple[int, int],
    dtype,
    mesh: Mesh,
) -> jax.Array:
    """Read a 2-D dataset into a ``P('x','y')``-sharded ``jax.Array``.

    Simplest entry point: one rectangular read of the whole dataset,
    block-partitioned across the 2-D mesh.  Returns the array directly
    (not a closure — there are no per-call parameters to vary).
    """
    p, q = validate_mesh_2d(mesh)
    n_rows, n_cols = int(global_shape[0]), int(global_shape[1])
    if n_rows % p or n_cols % q:
        raise ValueError(
            f"global shape ({n_rows},{n_cols}) must divide mesh ({p},{q})")

    ds_id = _register_and_open_dataset(fh, ds_name, mesh)
    local_shape = (n_rows // p, n_cols // q)
    out_struct = jax.ShapeDtypeStruct(local_shape, jnp.dtype(dtype))
    offset_zero = jnp.zeros((2,), dtype=jnp.int64)
    valid_shape = jnp.asarray((n_rows, n_cols), dtype=jnp.int64)

    handle = handle_vector(fh, ds_id, mesh)

    def _per_rank(handle_local, offset_local, valid_shape_local):
        return ffi_read_call(
            out_struct, handle_local, offset_local, valid_shape_local,
            mesh_shape=(p, q),
            axis_count_per_dim=(1, 1),
            axis_flat=(0, 1),
        )

    return shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(), P(), P()), out_specs=P("x", "y"),
        check_vma=False,
    )(handle, offset_zero, valid_shape)


def read_kchunk_union_sharded(
    fh: int,
    ds_name: str,
    *,
    n_kchunk: int,
    kchunk_axis: int,
    file_global_shape: Sequence[int],
    per_rank_file_shape: Sequence[int],
    dtype,
    mesh: Mesh,
    file_partition_spec: P,
    count_partition_spec: P | None = None,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """See ``_read_kchunk_union_sharded_cached`` for impl; this wrapper
    normalises Sequence args to tuples for lru_cache hashability.

    ``count_partition_spec`` (optional): partition spec for the
    ``count_base`` argument.  Defaults to ``P()`` (fully replicated; all
    ranks see the same global counts table, as the original API).  Pass
    ``P(('x','y'), None)`` (or any other sharding on the leading
    n_kchunk axis) when the caller wants per-rank counts — useful when
    a rank's read window must be clamped to the on-disk file extent
    (e.g. ``mnband`` not divisible by world_size).  The ``count_base``
    global shape must then be ``(world * n_kchunk, ndim_file)``; each
    rank receives its ``(n_kchunk, ndim_file)`` slice.
    """
    if count_partition_spec is None:
        count_partition_spec = P()
    # ``fh``/``ds_name`` resolve to the runtime handle HERE, outside the
    # compiled-closure cache.  That is the whole fix: the jitted closure
    # below is keyed on GEOMETRY only, so two files, two datasets, or two
    # processes with different heap addresses now share one compiled
    # module instead of forking one each.  The dataset open stays memoised
    # on ``(fh, ds_name)`` so the collective H5Dopen count does not change
    # (this wrapper runs on every ``read_slabs`` call).
    ds_id = _open_dataset_memo(int(fh), str(ds_name), mesh)
    handle = handle_vector(int(fh), ds_id, mesh)
    inner = _read_kchunk_union_sharded_cached(
        n_kchunk=int(n_kchunk), kchunk_axis=int(kchunk_axis),
        file_global_shape=tuple(int(s) for s in file_global_shape),
        per_rank_file_shape=tuple(int(s) for s in per_rank_file_shape),
        dtype=jnp.dtype(dtype),
        mesh=mesh, file_partition_spec=file_partition_spec,
        count_partition_spec=count_partition_spec,
    )

    # The handle is HIDDEN: callers keep the two-argument
    # ``reader(offset_base, count_base)`` shape they already have.
    def _reader(offset_base: jax.Array, count_base: jax.Array) -> jax.Array:
        return inner(handle, offset_base, count_base)

    return _reader


#: ``(ctx_handle, ds_name, mesh) -> hid_t`` for the k-chunk union reader.
#: A PLAIN DICT, cleared by :func:`_forget_datasets_for_ctx` at every
#: ``close_file``.  It used to be an ``lru_cache`` and that is a defect,
#: not a style: see :func:`_open_dataset_memo`.
_DS_ID_MEMO: Dict[Tuple[int, str, Mesh], int] = {}


def _forget_datasets_for_ctx(ctx: int) -> None:
    """Drop every memoised ``ds_id`` belonging to ``ctx``.

    THE CTX HANDLE IS A HEAP ADDRESS, AND HEAP ADDRESSES ARE REUSED.
    ``open_file`` returns ``reinterpret_cast<int64_t>(ctx)``; ``close_ctx``
    ``delete``s that object.  A second ``open_file`` of the same path in
    the same process — same allocation size, freed moments earlier —
    very often gets the SAME address back from the allocator.  Every
    ``hid_t`` minted against the first context was invalidated by its
    ``H5Fclose``, so a cache keyed on the address alone hands the second
    context the first one's dead dataset ids and the read handler refuses
    with ``phdf5 read_kchunk_union: ds_id is invalid``.

    MEASURED, and this is the failure it explains: JID 57269074, a second
    ``bandstructure.htransform.streaming_galerkin_solve`` in one exciton
    process, rank 3 raising exactly that message after the first 24x24
    load had completed and closed; an identical FRESH-PROCESS retry passed
    the same site.  Fresh process, fresh allocator, no address reuse — so
    the "intermittent" in that register row is the allocator's freedom to
    hand back a different address, not a race.

    Called from ``close_file`` (and the atexit sweep), which is the only
    place a ctx dies, so the memo cannot outlive the handle it describes.
    """
    for key in [k for k in _DS_ID_MEMO if k[0] == int(ctx)]:
        _DS_ID_MEMO.pop(key, None)


def _open_dataset_memo(fh: int, ds_name: str, mesh: Mesh) -> int:
    """Memoised collective ``H5Dopen``, keyed to a LIVE ctx.

    Split out of :func:`_read_kchunk_union_sharded_cached` when
    ``fh``/``ds_name`` left that function's key.  It is kept as its own
    cache — and NOT folded into the compiled-closure cache — because the
    two have genuinely different keys: a dataset id depends on the file
    and the dataset, a compiled module depends on the geometry.  Keeping
    it here holds the collective H5Dopen count at exactly one per
    ``(fh, ds_name, mesh)``, which is what the un-split code did.

    IT IS NOT AN ``lru_cache``, deliberately.  An ``lru_cache`` has no
    invalidation hook, and the key's first component is a reused heap
    address — see :func:`_forget_datasets_for_ctx` for the measured
    failure that combination produces.  The entry is dropped when the ctx
    it belongs to is closed, so a stale id is unreachable rather than
    merely unlikely.
    """
    # ``mesh`` is part of the key exactly as it was under ``lru_cache``:
    # ``Mesh`` is hashable, and the collective open is routed by the mesh's
    # platform.  Keep the OBJECT, not ``id(mesh)`` — an id is a reused
    # address too, which is the bug this function is about.
    key = (int(fh), str(ds_name), mesh)
    got = _DS_ID_MEMO.get(key)
    if got is not None:
        return got
    ds_id = _register_and_open_dataset(fh, ds_name, mesh)
    _DS_ID_MEMO[key] = int(ds_id)
    return int(ds_id)


@functools.lru_cache(maxsize=None)
def _read_kchunk_union_sharded_cached(
    *,
    n_kchunk: int,
    kchunk_axis: int,
    file_global_shape: tuple,
    per_rank_file_shape: tuple,
    dtype,
    mesh: Mesh,
    file_partition_spec: P,
    count_partition_spec: P = P(),
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Build a jitted ``f(handle, offset_base, count_base) → array`` callable
    that issues **ONE** ``H5Dread`` for ``n_kchunk`` per-k windows via
    ``H5S_SELECT_OR``.  Correctness preconditions — see below.

    The per-k file rectangles must be:
      1. **pairwise disjoint** (caller supplies per-k ``count[k]`` tight
         enough that the windows don't overlap), and
      2. **sorted in ascending row-major file order** over the varying
         file dim.  If the physical k-indices aren't sorted that way,
         the caller should argsort the offset table before dispatch and
         permute the output's kchunk axis back afterward (both cheap).

    Parameters
    ----------
    n_kchunk : int
        Number of windows (compile-time).
    kchunk_axis : int
        Position in the output shape at which the n_kchunk axis lives.
        For correct row-major iteration-order matching between memspace
        and filespace, this must be placed **immediately before the file
        dim that varies across k** (the G axis for WFN-style reads).
        Passing ``kchunk_axis=2`` for a ``(nb, ns, ngkmax, 2)``-shaped
        per-rank file shape produces the output
        ``(nb, ns, n_kchunk, ngkmax, 2)``.
    file_global_shape : length ``ndim_file``
        Dataset shape on disk (for sanity checks only).
    per_rank_file_shape : length ``ndim_file``
        One rank's portion of a single window.
    dtype : numpy dtype
    mesh : Mesh
        2-D ``('x','y')`` mesh.
    file_partition_spec : PartitionSpec, length ``ndim_file``
        How the file dims are sharded on ``mesh``.

    Returns
    -------
    callable
        ``f(handle, offset_base, count_base)``; ``handle`` is the ``(2,)``
        ``[ctx_handle, ds_id]`` vector (replicated, ``P()``) and the other
        two are ``(n_kchunk, ndim_file) int64``.  Callers should reach this
        through :func:`read_kchunk_union_sharded`, which supplies the
        handle and preserves the two-argument call shape.
        Output shape is ``per_rank_file_shape`` with ``n_kchunk`` inserted
        at ``kchunk_axis``; partition spec gets ``None`` inserted at the
        same position.
    """
    ndim_file = len(per_rank_file_shape)
    if len(file_global_shape) != ndim_file:
        raise ValueError(
            f"file_global_shape ndim {len(file_global_shape)} != "
            f"per_rank_file_shape ndim {ndim_file}")
    if not (0 <= kchunk_axis <= ndim_file):
        raise ValueError(
            f"kchunk_axis {kchunk_axis} must be in [0, {ndim_file}]")
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])

    axis_count_per_dim, axis_flat = _encode_sharding_axes(
        mesh, file_partition_spec, ndim_file)

    out_local_shape = _insert_at(
        [int(s) for s in per_rank_file_shape], kchunk_axis, n_kchunk)
    out_struct = jax.ShapeDtypeStruct(out_local_shape, jnp.dtype(dtype))
    out_partition_spec = P(*_insert_at(
        list(file_partition_spec), kchunk_axis, None))

    def _per_rank(handle_local, offset_base_local, count_base_local):
        return ffi_read_kchunk_union_call(
            out_struct,
            handle_local,
            offset_base_local, count_base_local,
            mesh_shape=(p, q),
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
            n_kchunk=int(n_kchunk),
            kchunk_axis=int(kchunk_axis),
        )

    return jax.jit(shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(), P(), count_partition_spec),
        out_specs=out_partition_spec,
        check_vma=False,
    ))


# =========================================================================
#  Writer  (merged from phdf5/write.py; original docstring below)
# =========================================================================
"""Sharded-slab FFI writer — thin shard_map wrapper around PhdfWriteFfi.

Preferred public entry point for gw_jax / isdf / etc. is
:mod:`file_io.slab_io`, which has exactly one transport — this one.  The
sentence that used to sit here still offered a choice between "the
allgather-and-rank-0-h5py backend (default)" and the FFI backend; both
the router and that backend were deleted at 233a830d, so the choice it
described has not existed since.  Call this module directly only for 2-D
block-partitioned writes where you know you want the raw FFI path.

The underlying C++ handler is N-D and derives per-rank hyperslab
offsets from ``ctx->rank`` + the mesh_shape / axis_for_dim attrs.
"""
_FFI_TARGET = "lorrax_phdf5_write"


# Low-level padding contract: A_local is the physical equal-block shard;
# valid_shape is the logical global slab prefix that C++ clips against.
def ffi_write_call(
    A_local: jax.Array,
    handle: jax.Array,
    offset_base: jax.Array,
    valid_shape: jax.Array,
    *,
    mesh_shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
) -> jax.Array:
    """Low-level FFI call for one rank's local shard.  Returns token.

    ``handle``, ``offset_base`` and ``valid_shape`` are jax.Arrays of
    dtype int64 — passed as traced Buffer inputs (not FFI Attrs) so that
    shard_map closures compile ONCE per ndim-dtype-sharding tuple and
    re-dispatch across chunks, datasets and processes.

    ``handle`` is ``(2,)`` == ``[ctx_handle, ds_id]``; see
    :func:`handle_vector`.  These were Attrs until 2026-08-07, and
    ctx_handle is a heap address, so the compiled module differed in
    every process and the JAX persistent compile cache could never hit
    one — it rewrote the entry on every run instead.  MEASURED with a
    byte-identical workload into a private cache dir: ``jit__per_rank``
    entries went 4 -> 8 -> 12 over three runs while a plain jit control
    stayed at 1; the shared ``np1`` cache had 6813 such dead entries out
    of 14443.  ds_id moved with it because as an Attr it forked a module
    per (file, dataset) pair, making the module count scale with the deck.

    ``offset_base``/``valid_shape`` were made Buffers earlier for the same
    class of reason: each distinct chunk offset otherwise triggered a
    fresh ~400 ms XLA compile of the FFI body (measured at MoS2 3x3, see
    reports/zeta_offset_runtime_2026-04-19/).

    The mesh/axis attrs remain compile-time attrs — they are geometry, so
    a change in them genuinely IS a different module.

    Use inside a ``shard_map`` body.
    """
    token_spec = jax.ShapeDtypeStruct((1,), jnp.int32)
    return jax.ffi.ffi_call(_FFI_TARGET, token_spec, has_side_effect=True)(
        A_local,
        handle,
        offset_base,
        valid_shape,
        mesh_shape=np.asarray(mesh_shape, dtype=np.int64),
        axis_count_per_dim=np.asarray(axis_count_per_dim, dtype=np.int64),
        axis_flat=np.asarray(axis_flat, dtype=np.int64),
    )


def write_sharded_slab(
    fh: int,
    ds_name: str,
    A: jax.Array,
    *,
    mesh: Mesh,
    global_shape: tuple[int, int] | None = None,
    valid_shape: tuple[int, int] | None = None,
) -> jax.Array:
    """Write a 2-D P('x','y')-sharded JAX array to an open HDF5 dataset.

    Thin wrapper on top of the N-D FFI: sets
    ``mesh_shape=(p,q)`` and ``axis_for_dim=(0,1)`` so each rank writes
    its (rank/q, rank%q) hyperslab.  Shipped as the simplest 2-D entry
    point; for N-D writes use :mod:`file_io.slab_io`.
    """
    if A.ndim != 2:
        raise NotImplementedError(
            "write_sharded_slab is 2-D only; use file_io.slab_io for N-D")
    p, q = validate_mesh_2d(mesh)
    phys_rows, phys_cols = (int(A.shape[0]), int(A.shape[1]))
    valid_rows, valid_cols = (
        tuple(int(s) for s in valid_shape)
        if valid_shape is not None else (phys_rows, phys_cols)
    )
    n_rows, n_cols = (
        tuple(int(s) for s in global_shape)
        if global_shape is not None else (valid_rows, valid_cols)
    )
    if phys_rows % p or phys_cols % q:
        raise ValueError(
            f"physical shape ({phys_rows},{phys_cols}) must divide mesh ({p},{q})")
    if valid_rows > phys_rows or valid_cols > phys_cols:
        raise ValueError(
            f"valid_shape ({valid_rows},{valid_cols}) exceeds physical "
            f"shape ({phys_rows},{phys_cols})")
    if valid_rows > n_rows or valid_cols > n_cols:
        raise ValueError(
            f"valid_shape ({valid_rows},{valid_cols}) exceeds dataset "
            f"shape ({n_rows},{n_cols})")
    get_lib()

    ds_id = ffi_loader.phdf5_ensure_dataset(
        fh, ds_name, (int(n_rows), int(n_cols)),
        str(jnp.dtype(A.dtype).name))

    offset_array = jnp.zeros((2,), dtype=jnp.int64)
    valid_shape_arr = jnp.asarray((valid_rows, valid_cols), dtype=jnp.int64)

    handle = handle_vector(fh, ds_id, mesh)

    def _per_rank(A_local, handle_local, offset_local, valid_shape_local):
        return ffi_write_call(
            A_local, handle_local, offset_local, valid_shape_local,
            mesh_shape=(p, q),
            axis_count_per_dim=(1, 1),
            axis_flat=(0, 1),
        )

    return shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P("x", "y"), P(), P(), P()), out_specs=P(),
        check_vma=False,
    )(A, handle, offset_array, valid_shape_arr)

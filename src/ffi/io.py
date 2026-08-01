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
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .common import ffi_loader
from .common.ffi_loader import get_lib

__all__ = [
    # lifecycle (was phdf5/context.py)
    "open_file", "close_file", "platform_for_handle", "validate_mesh_2d",
    # readers (was phdf5/read.py)
    "ffi_read_call", "ffi_read_kchunk_call", "ffi_read_kchunk_union_call",
    "read_sharded_slab", "read_kchunk_sharded", "read_kchunk_union_sharded",
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
# path -> (int64 ctx handle, owning platform "CUDA"|"cpu").  The platform
# is recorded at open and used for EVERY lifecycle call on the handle: a
# PhdfCtx* allocated by one platform's .so (its heap, its HDF5/MPI state)
# must never be handed to the other library — the dual-platform
# (JAX_PLATFORMS=cuda,cpu) hazard the mesh routing exists for.  Routing
# only the open by mesh platform while close/ensure_dataset followed the
# JAX default backend was exactly that bug (audit fix/zq 2026-07-28).
_FILE_CTXS: Dict[str, Tuple[int, str]] = {}


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
            return _FILE_CTXS[path][0]
        ctx = ffi_loader.phdf5_open(
            path, p, q,
            int(jax.process_index()), int(jax.process_count()),
            _MODE_FLAGS[mode], platform=platform,
        )
        _FILE_CTXS[path] = (ctx, platform)
        return ctx


def platform_for_handle(ctx_handle: int) -> Optional[str]:
    """The platform ("CUDA"/"cpu") whose library allocated ``ctx_handle``,
    or None for an unknown handle.  Every lifecycle call on a handle
    (``phdf5_close`` / ``phdf5_ensure_dataset`` / ``phdf5_open_dataset_ro``)
    must go through the owning platform's .so — this is the lookup the
    write-side lifecycle sites use to route theirs."""
    with _LOCK:
        for _ctx, _plat in _FILE_CTXS.values():
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
            # platform=None (unknown handle) follows the JAX default
            # backend inside ffi_loader — hardcoding "CUDA" would fail on
            # a CPU node where only the host lib exists.
            ffi_loader.phdf5_close(int(ctx), platform=platform)


def _atexit_close_all() -> None:
    """Close any files still open at process exit — catches forgotten
    close_file calls.  Runs on every process.  Each handle closes through
    its own recorded platform library."""
    with _LOCK:
        for path, (ctx, platform) in list(_FILE_CTXS.items()):
            try:
                ffi_loader.phdf5_close(int(ctx), platform=platform)
            except Exception:
                pass
        _FILE_CTXS.clear()


atexit.register(_atexit_close_all)


# =========================================================================
#  Readers  (merged from phdf5/read.py; original docstring below)
# =========================================================================
"""Sharded-slab FFI readers for parallel-HDF5 datasets.

Three entry points, each wrapping a C++ handler defined in
``cpp/read_ffi.cc``:

===========================  ===========================================
``read_sharded_slab``         2-D dataset → ``P('x','y')``-sharded array.
                              Thinnest convenience wrapper.
``read_kchunk_sharded``       N-D dataset, ``n_kchunk`` independent
                              windows, **one handler invocation doing
                              n_kchunk sequential H5Dreads**.  Use when
                              the per-k windows might overlap in the
                              file (e.g. ngkmax slabs at variable-ngk
                              WFN files).
``read_kchunk_union_sharded`` N-D dataset, ``n_kchunk`` **disjoint**
                              windows via ``H5S_SELECT_OR`` compound
                              hyperslab, **one H5Dread**.  Use when the
                              caller can provide per-k variable counts
                              so the windows are disjoint by
                              construction.
===========================  ===========================================

Each high-level function returns a jitted ``shard_map`` closure: the
caller dispatches against it with runtime buffer arguments (offsets,
counts), and the same compiled module handles any offsets/counts
combination at the same shapes/dtypes.

Preferred public entry point for simple cases remains
:mod:`file_io.slab_io`, which auto-dispatches between this FFI and an
h5py+gather fallback.  The three functions here are the low-level
building blocks.
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
def ffi_read_call(
    out_struct: jax.ShapeDtypeStruct,
    offset_base: jax.Array,
    valid_shape: jax.Array,
    *,
    ctx_handle: int,
    ds_id: int,
    mesh_shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
) -> jax.Array:
    """Single-hyperslab read — one H5Dread of one rectangle.

    ``offset_base`` and ``valid_shape`` are ``(ndim,)`` int64
    ``jax.Array`` buffers passed at runtime (not as FFI Attrs), so the
    shard_map closure compiles ONCE per ``(dataset, ndim, dtype,
    sharding)`` tuple and re-dispatches across chunks with different
    offsets or logical extents.
    """
    return jax.ffi.ffi_call(_TARGET_READ, out_struct)(
        offset_base,
        valid_shape,
        ctx_handle=int(ctx_handle),
        ds_id=int(ds_id),
        mesh_shape=np.asarray(mesh_shape, dtype=np.int64),
        axis_count_per_dim=np.asarray(axis_count_per_dim, dtype=np.int64),
        axis_flat=np.asarray(axis_flat, dtype=np.int64),
    )


def ffi_read_kchunk_call(
    out_struct: jax.ShapeDtypeStruct,
    offset_base: jax.Array,          # (n_kchunk, ndim_file) int64
    *,
    ctx_handle: int,
    ds_id: int,
    mesh_shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
    n_kchunk: int,
) -> jax.Array:
    """Sequential-reads kchunk — one handler invocation doing
    ``n_kchunk`` H5Dread calls into a packed pinned buffer.  Each of the
    per-k rectangles has the same shape; only its file offset varies.
    """
    return jax.ffi.ffi_call(_TARGET_READ_KCHUNK, out_struct)(
        offset_base,
        ctx_handle=int(ctx_handle),
        ds_id=int(ds_id),
        mesh_shape=np.asarray(mesh_shape, dtype=np.int64),
        axis_count_per_dim=np.asarray(axis_count_per_dim, dtype=np.int64),
        axis_flat=np.asarray(axis_flat, dtype=np.int64),
        n_kchunk=int(n_kchunk),
    )


def ffi_read_kchunk_union_call(
    out_struct: jax.ShapeDtypeStruct,
    offset_base: jax.Array,          # (n_kchunk, ndim_file) int64
    count_base: jax.Array,           # (n_kchunk, ndim_file) int64
    *,
    ctx_handle: int,
    ds_id: int,
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
        offset_base,
        count_base,
        ctx_handle=int(ctx_handle),
        ds_id=int(ds_id),
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

    def _per_rank(offset_local, valid_shape_local):
        return ffi_read_call(
            out_struct, offset_local, valid_shape_local,
            ctx_handle=int(fh), ds_id=int(ds_id),
            mesh_shape=(p, q),
            axis_count_per_dim=(1, 1),
            axis_flat=(0, 1),
        )

    return shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(), P()), out_specs=P("x", "y"),
        check_rep=False,
    )(offset_zero, valid_shape)


def read_kchunk_sharded(
    fh: int,
    ds_name: str,
    *,
    n_kchunk: int,
    file_global_shape: Sequence[int],
    per_rank_file_shape: Sequence[int],
    dtype,
    mesh: Mesh,
    file_partition_spec: P,
) -> Callable[[jax.Array], jax.Array]:
    """Build a jitted ``f(offset_base) → array`` callable for
    ``n_kchunk`` **same-shape** hyperslab windows.

    Use when the per-k windows may overlap in the file (e.g. ngkmax
    slabs at variable-ngk WFN files): the handler does n_kchunk
    sequential H5Dreads under the hood, each reading into a distinct
    stripe of the output.  One XLA op, n_kchunk MPI-IO collectives.

    Parameters
    ----------
    fh : int
        Context handle from :func:`ffi.phdf5.open_file`.
    ds_name : str
        HDF5 dataset path (e.g. ``"wfns/coeffs"``).
    n_kchunk : int
        Number of windows (compile-time constant).
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
        ``f(offset_base: (n_kchunk, ndim_file) int64) → (n_kchunk, *file_shape_sharded)``.
        The leading n_kchunk axis is always replicated across ranks;
        only the trailing file dims are sharded per ``file_partition_spec``.
    """
    ndim_file = len(per_rank_file_shape)
    if len(file_global_shape) != ndim_file:
        raise ValueError(
            f"file_global_shape ndim {len(file_global_shape)} != "
            f"per_rank_file_shape ndim {ndim_file}")
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])

    axis_count_per_dim, axis_flat = _encode_sharding_axes(
        mesh, file_partition_spec, ndim_file)
    ds_id = _register_and_open_dataset(fh, ds_name, mesh)

    out_local_shape = (n_kchunk,) + tuple(int(s) for s in per_rank_file_shape)
    out_struct = jax.ShapeDtypeStruct(out_local_shape, jnp.dtype(dtype))
    out_partition_spec = P(None, *tuple(file_partition_spec))

    def _per_rank(offset_base_local):
        return ffi_read_kchunk_call(
            out_struct, offset_base_local,
            ctx_handle=int(fh), ds_id=int(ds_id),
            mesh_shape=(p, q),
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
            n_kchunk=int(n_kchunk),
        )

    return jax.jit(shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(),), out_specs=out_partition_spec,
        check_rep=False,
    ))


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
    return _read_kchunk_union_sharded_cached(
        int(fh), str(ds_name),
        n_kchunk=int(n_kchunk), kchunk_axis=int(kchunk_axis),
        file_global_shape=tuple(int(s) for s in file_global_shape),
        per_rank_file_shape=tuple(int(s) for s in per_rank_file_shape),
        dtype=jnp.dtype(dtype),
        mesh=mesh, file_partition_spec=file_partition_spec,
        count_partition_spec=count_partition_spec,
    )


@functools.lru_cache(maxsize=None)
def _read_kchunk_union_sharded_cached(
    fh: int,
    ds_name: str,
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
    """Build a jitted ``f(offset_base, count_base) → array`` callable
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
    fh, ds_name :
        As :func:`read_kchunk_sharded`.
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
    file_global_shape, per_rank_file_shape, dtype, mesh,
    file_partition_spec :
        As :func:`read_kchunk_sharded`.

    Returns
    -------
    callable
        ``f(offset_base, count_base)`` both ``(n_kchunk, ndim_file) int64``.
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
    ds_id = _register_and_open_dataset(fh, ds_name, mesh)

    out_local_shape = _insert_at(
        [int(s) for s in per_rank_file_shape], kchunk_axis, n_kchunk)
    out_struct = jax.ShapeDtypeStruct(out_local_shape, jnp.dtype(dtype))
    out_partition_spec = P(*_insert_at(
        list(file_partition_spec), kchunk_axis, None))

    def _per_rank(offset_base_local, count_base_local):
        return ffi_read_kchunk_union_call(
            out_struct,
            offset_base_local, count_base_local,
            ctx_handle=int(fh), ds_id=int(ds_id),
            mesh_shape=(p, q),
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
            n_kchunk=int(n_kchunk),
            kchunk_axis=int(kchunk_axis),
        )

    return jax.jit(shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(), count_partition_spec),
        out_specs=out_partition_spec,
        check_rep=False,
    ))


# =========================================================================
#  Writer  (merged from phdf5/write.py; original docstring below)
# =========================================================================
"""Sharded-slab FFI writer — thin shard_map wrapper around PhdfWriteFfi.

Preferred public entry point for gw_jax / isdf / etc. is
:mod:`file_io.slab_io`, which dispatches on ``use_ffi_io`` between the
allgather-and-rank-0-h5py backend (default) and this FFI backend.
Call this module directly only for 2-D block-partitioned writes where
you know you want the FFI path.

The underlying C++ handler is N-D and derives per-rank hyperslab
offsets from ``ctx->rank`` + the mesh_shape / axis_for_dim attrs.
"""
_FFI_TARGET = "lorrax_phdf5_write"


# Low-level padding contract: A_local is the physical equal-block shard;
# valid_shape is the logical global slab prefix that C++ clips against.
def ffi_write_call(
    A_local: jax.Array,
    offset_base: jax.Array,
    valid_shape: jax.Array,
    *,
    ctx_handle: int,
    ds_id: int,
    mesh_shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
) -> jax.Array:
    """Low-level FFI call for one rank's local shard.  Returns token.

    ``offset_base`` and ``valid_shape`` are jax.Arrays of shape (ndim,)
    dtype int64 — passed
    as a traced Buffer input (not an FFI Attr) so that shard_map closures
    compile ONCE per dataset-ndim-dtype-sharding tuple and re-dispatch
    across chunks with different offsets or logical extents.  Without this, each chunk
    triggers a fresh ~400 ms XLA compile for the FFI body (measured at
    MoS2 3x3 scale, see reports/zeta_offset_runtime_2026-04-19/).

    The other mesh/axis attrs ARE compile-time attrs — they don't change
    across chunks of a given dataset, so no recompile happens.

    Use inside a ``shard_map`` body.
    """
    token_spec = jax.ShapeDtypeStruct((1,), jnp.int32)
    return jax.ffi.ffi_call(_FFI_TARGET, token_spec, has_side_effect=True)(
        A_local,
        offset_base,
        valid_shape,
        ctx_handle=int(ctx_handle),
        ds_id=int(ds_id),
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

    def _per_rank(A_local, offset_local, valid_shape_local):
        return ffi_write_call(
            A_local, offset_local, valid_shape_local,
            ctx_handle=int(fh),
            ds_id=int(ds_id),
            mesh_shape=(p, q),
            axis_count_per_dim=(1, 1),
            axis_flat=(0, 1),
        )

    return shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P("x", "y"), P(), P()), out_specs=P(),
        check_rep=False,
    )(A, offset_array, valid_shape_arr)

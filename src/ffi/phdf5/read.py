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
from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..common import ffi_loader
from ..common.ffi_loader import get_lib
from .context import validate_mesh_2d


__all__ = [
    "ffi_read_call",
    "ffi_read_kchunk_call",
    "ffi_read_kchunk_union_call",
    "read_sharded_slab",
    "read_kchunk_sharded",
    "read_kchunk_union_sharded",
]


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


def _register_and_open_dataset(fh: int, ds_name: str) -> int:
    get_lib()
    return ffi_loader.phdf5_open_dataset_ro(fh, ds_name)


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

    ds_id = _register_and_open_dataset(fh, ds_name)
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
    ds_id = _register_and_open_dataset(fh, ds_name)

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
    ds_id = _register_and_open_dataset(fh, ds_name)

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
        in_specs=(P(), P()), out_specs=out_partition_spec,
        check_rep=False,
    ))

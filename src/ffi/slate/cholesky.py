"""``distributed_cholesky`` — JAX FFI wrapper around ``slate::potrf``.

Returns a :class:`SlateLowerL` opaque handle, not a plain JAX array.
The underlying buffer is in SLATE's distributed-tile / col-major-tile
layout, which JAX cannot interpret as a "row-major lower-triangular L"
without an explicit transform.  Two ways to consume it:

    * Pass it directly to :func:`ffi.slate.distributed_trsm` — the trsm
      FFI knows how to feed SLATE's own layout straight back without
      going through any layout massaging.
    * Call :meth:`SlateLowerL.to_jax_lower` to get the standard
      lower-triangular L in JAX's row-major view (equivalent to
      ``jnp.tril(raw.conj().T)``).

Only ``Uplo::Lower`` is supported (SLATE's potrf).  dtype F64 or C128.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common.ffi_loader import get_lib
from .context import get_or_init_context

__all__ = ["SlateLowerL", "distributed_cholesky"]

_FFI_TARGET = "lorrax_slate_potrf"


@dataclass(frozen=True)
class SlateLowerL:
    """Opaque handle to a SLATE-format Cholesky lower factor.

    Not interpretable as a dense L in JAX's row-major view — the buffer
    holds SLATE's distributed col-major-tile layout.  The strict-upper
    tiles are explicitly zero (potrf FFI guarantees this for the
    default ``nb = n/p`` block size) so chaining into ``distributed_trsm``
    works correctly.

    Attributes
    ----------
    raw : jax.Array
        Raw FFI output buffer, P('x','y')-sharded on ``mesh``.  Shape
        ``(n, n)``, dtype matching the input ``A``.
    mesh : Mesh
        2-D mesh the factor was computed on.
    n : int
        Side length of the matrix.
    nb : int
        Tile block size SLATE used.
    """
    raw: jax.Array
    mesh: Mesh
    n: int
    nb: int

    def to_jax_lower(self) -> jax.Array:
        """Return the standard lower-triangular L in JAX's row-major view.

        Equivalent to ``jnp.tril(raw.conj().T)``.  Useful if the caller
        wants to do operations on L outside of SLATE.  For chaining into
        SLATE's trsm, pass the handle directly instead.
        """
        return jnp.tril(self.raw.conj().T)


def distributed_cholesky(
    A: jax.Array,
    *,
    mesh: Mesh,
    block_size: Optional[int] = None,
) -> SlateLowerL:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"distributed_cholesky: expected square A; got {A.shape}")
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"mesh must have axes ('x','y'); got {mesh.axis_names}")
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    # SLATE potrf supports any p x q grid in principle, but our FFI
    # passes row-major JAX shards as if they were col-major SLATE tiles.
    # For square (p==q), per-rank shard is (n/p, n/p) and the
    # reinterpretation is just a transpose, which a Hermitian A absorbs.
    # For p != q the per-rank shape is rectangular (n/p, n/q) and the
    # SLATE col-major view ends up at shape (n/q, n/p) — mismatched, so
    # SLATE assembles something that isn't actually A and potrf reports
    # info != 0.  Fix would be a local D2D transpose per shard before
    # SLATE sees it; not implemented yet.
    if p != q:
        raise ValueError(
            f"distributed_cholesky: only square meshes (p==q) are supported "
            f"by this FFI; got {p}x{q}.  See cholesky.py for layout details.")
    if p * q != jax.process_count():
        raise ValueError(
            f"mesh {p}x{q} != jax.process_count()={jax.process_count()}")
    n = int(A.shape[0])
    if n % p != 0:
        raise ValueError(f"n={n} must be divisible by mesh axis size {p}")

    get_lib()
    ctx_handle = get_or_init_context(mesh)

    # Default tile size: divide by the larger grid axis so each rank
    # holds at least 1 tile along both directions.  For square p==q this
    # collapses to n//p (the original).  For non-square, n//p alone gave
    # tile = n on p=1 grids → one giant tile, SLATE complains.
    nb = n // max(p, q) if block_size is None else int(block_size)
    L_local = jax.ShapeDtypeStruct((n // p, n // q), A.dtype)

    attrs = dict(n=n, nb=nb, ctx_handle=int(ctx_handle))

    @partial(shard_map, mesh=mesh,
             in_specs=P("x", "y"), out_specs=P("x", "y"),
             check_rep=False)
    def _call(local_A):
        return jax.ffi.ffi_call(_FFI_TARGET, L_local)(local_A, **attrs)

    L_raw = _call(A)
    return SlateLowerL(raw=L_raw, mesh=mesh, n=n, nb=nb)

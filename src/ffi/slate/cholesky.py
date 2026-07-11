"""``distributed_cholesky`` — JAX FFI wrapper around ``slate::potrf``.

Layout
------
JAX is row-major; SLATE tiles are column-major.  Each rank locally
transposes its own shard via ``shard_map`` + ``jnp.transpose`` (no
inter-rank comm) so the bytes SLATE reads are correctly col-major.
Combined with the C++-side MPI rank remap (``cpp/context.cc``), SLATE
sees the user's matrix correctly assembled on **any p × q mesh**.

Returns a :class:`SlateLowerL` opaque handle.  Two ways to consume it:

  * Pass the handle to :func:`ffi.slate.distributed_trsm` — trsm knows
    the handle's layout and feeds it straight back to SLATE.
  * Call :meth:`SlateLowerL.to_jax_lower` for a conventional
    row-major lower-triangular L.

Only ``Uplo::Lower`` (SLATE's potrf), dtypes F64 / C128.  See
``src/ffi/slate/README.md`` for the wider design notes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from .context import (ensure_registered, get_or_init_context, validate_mesh,
                      validate_tile_layout)

__all__ = ["SlateLowerL", "distributed_cholesky"]

_FFI_TARGET = "lorrax_slate_potrf"


@dataclass(frozen=True)
class SlateLowerL:
    """Opaque handle to a SLATE-format Cholesky lower factor.

    Attributes
    ----------
    raw : jax.Array
        Shape ``(n, n)``, sharded ``P('y', 'x')`` on ``mesh``.  Bytes
        are SLATE's col-major L tiles; JAX reading this directly would
        give ``L.T`` (or ``L.conj().T`` for complex).
    mesh : Mesh
        2-D mesh the factor was computed on.
    n : int
        Matrix side length.
    nb : int
        Tile block size SLATE used.
    """
    raw: jax.Array
    mesh: Mesh
    n: int
    nb: int

    def to_jax_lower(self) -> jax.Array:
        """Return L in the conventional JAX row-major lower-triangular form.

        Local-transposes the per-rank buffer (P('y','x') → P('x','y'),
        no inter-rank comm) and strips the (zeroed) strict-upper.
        """
        @partial(shard_map, mesh=self.mesh,
                 in_specs=P("y", "x"), out_specs=P("x", "y"),
                 check_rep=False)
        def _local_T(local):
            return jnp.transpose(local, (1, 0))
        return jnp.tril(_local_T(self.raw))


def distributed_cholesky(
    A: jax.Array,
    *,
    mesh: Mesh,
    block_size: Optional[int] = None,
) -> SlateLowerL:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"distributed_cholesky: expected square A; got {A.shape}")
    p, q = validate_mesh(mesh)
    n = int(A.shape[0])
    if n % p != 0 or n % q != 0:
        raise ValueError(
            f"distributed_cholesky: n={n} must be divisible by both mesh "
            f"axes ({p},{q}).")

    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)

    # Default tile size divides by the larger grid axis so each rank
    # still holds >=1 tile along both directions.
    nb = n // max(p, q) if block_size is None else int(block_size)
    validate_tile_layout(n, nb, p, q, what="distributed_cholesky")

    # Local transpose via shard_map: each rank flips its own (n/p, n/q)
    # row-major shard to (n/q, n/p) row-major.  Bytes are the same set
    # the rank already had — no inter-rank communication.  After the
    # transpose, those bytes equal the original block in col-major
    # layout, which SLATE reads correctly.  Combined with the comm-remap
    # on the C++ side (context.cc), SLATE assembles the global A
    # correctly for any p x q mesh.
    L_local_T = jax.ShapeDtypeStruct((n // q, n // p), A.dtype)
    attrs = dict(n=n, nb=nb, ctx_handle=int(ctx_handle))

    @partial(shard_map, mesh=mesh,
             in_specs=P("x", "y"), out_specs=P("y", "x"),
             check_rep=False)
    def _potrf(local_A):
        local_A_T = jnp.transpose(local_A, (1, 0))
        return jax.ffi.ffi_call(_FFI_TARGET, L_local_T)(local_A_T, **attrs)

    L_raw_T = _potrf(A)
    return SlateLowerL(raw=L_raw_T, mesh=mesh, n=n, nb=nb)

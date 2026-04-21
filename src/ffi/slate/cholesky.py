"""``distributed_cholesky`` — JAX FFI wrapper around ``slate::potrf``.

Layout convention
-----------------
JAX stores arrays row-major; SLATE tiles are column-major with
``lda = nb``.  We materialise the transpose (``jnp.transpose(A, (1,0))``
+ ``with_sharding_constraint``) before handing the buffer to SLATE:
the bytes of that transposed array are the original ``A`` in
column-major layout, and SLATE reads them directly.  This works for
**any p×q mesh** — including rectangular ones — because the transpose
is a local (per-rank) operation, no inter-rank communication.

Returns a :class:`SlateLowerL` opaque handle.  The underlying buffer
is in SLATE's col-major-in-row-major layout (i.e. the transpose of the
row-major L).  Two ways to consume it:

  * Pass the handle to :func:`ffi.slate.distributed_trsm` — trsm knows
    the handle's layout and feeds it straight back to SLATE.
  * Call :meth:`SlateLowerL.to_jax_lower` for a conventional
    row-major lower-triangular L.

Only ``Uplo::Lower`` (SLATE's potrf) and dtypes F64 / C128.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..common.ffi_loader import get_lib
from .context import get_or_init_context

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
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"mesh must have axes ('x','y'); got {mesh.axis_names}")
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    if p * q != jax.process_count():
        raise ValueError(
            f"mesh {p}x{q} != jax.process_count()={jax.process_count()}")
    n = int(A.shape[0])
    if n % p != 0 or n % q != 0:
        raise ValueError(
            f"n={n} must be divisible by both mesh axes ({p},{q}).")

    get_lib()
    ctx_handle = get_or_init_context(mesh)

    # Default tile size divides by the larger grid axis so each rank
    # still holds >=1 tile along both directions.
    nb = n // max(p, q) if block_size is None else int(block_size)

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

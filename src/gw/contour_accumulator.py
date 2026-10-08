"""The contour accumulator: ``A[o,q,m,n] += p[o] · c[q,m,n]``, in XLA.

The response bank's Laplace/KMS streams (``gw.w_isdf``) accumulate one shared
contour correlation into many outputs on each device's tile, and the streamed
row passes (``gw.subtile_stream``) add a block of it at a row offset.  Both are
elementwise products and sums that XLA fuses into the stream; a CUDA kernel for
them had no measured gain (``docs/architecture/decisions.md#xla-reference``).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

__all__ = ["contour_accumulator", "contour_block_accumulate_local"]


def contour_accumulator(mesh):
    """``fn(accumulator, contribution, projection) -> accumulator``.

    ``accumulator`` ``[output, q, m, n]`` at ``P(None, None, 'x', 'y')``,
    ``contribution`` ``[q, m, n]`` at ``P(None, 'x', 'y')`` and the replicated
    weights ``projection`` ``[output]``, complex128.  The weights carry the
    contour owner's units (time quadrature and value or d/d(z_Ry**2)
    projection); the caller forms the Keldysh difference -i(A - conj(A)) once,
    before any output loop.  No collectives.
    """
    @partial(jax.shard_map, mesh=mesh,
             in_specs=(P(None, None, 'x', 'y'), P(None, 'x', 'y'), P()),
             out_specs=P(None, None, 'x', 'y'), check_vma=False)
    def accumulate(accumulator, contribution, projection):
        if any(a.dtype != jnp.complex128 for a in (accumulator, contribution, projection)):
            raise TypeError("contour accumulator requires complex128 operands")
        if (accumulator.ndim != 4 or contribution.ndim != 3 or projection.ndim != 1
                or accumulator.shape != (projection.shape[0], *contribution.shape)
                or any(n < 1 for n in accumulator.shape)):
            raise ValueError("contour accumulator shape mismatch")
        return accumulator + projection[:, None, None, None] * contribution[None]

    return accumulate


def contour_block_accumulate_local(accumulator, contribution, projection, valid, *, m0, n0):
    """One device tile: ``A[o, q, m0+m, n0+n] += sum_s projection[s,o] contribution[s,q,m,n]``.

    For code already inside ``shard_map``.  ``accumulator`` ``[output,q,M,N]``,
    ``contribution`` ``[terms,q,bm,bn]``, ``projection`` ``[terms,output]``
    complex128, ``valid`` ``[2]`` int32: the block's rows and columns that are
    not padding (the rest are not touched).  The terms add in order with the
    full accumulator's rounding, so terms ``(a, b)`` give the bytes of two
    :func:`contour_accumulator` calls.  ``m0``/``n0`` are Python ints or traced
    ints (a scanned pass's row offset).
    """
    if any(a.dtype != jnp.complex128 for a in (accumulator, contribution, projection)):
        raise TypeError("contour block accumulator requires complex128 operands")
    # One index dtype: a traced int32 offset beside x64 Python ints is a TypeError.
    start = tuple(jnp.asarray(v, jnp.int32) for v in (0, 0, m0, n0))
    bm, bn = contribution.shape[2:]
    block = jax.lax.dynamic_slice(accumulator, start, (*accumulator.shape[:2], bm, bn))
    new = block
    for s in range(contribution.shape[0]):
        new = new + projection[s][:, None, None, None] * contribution[s][None]
    live = (jnp.arange(bm)[:, None] < valid[0]) & (jnp.arange(bn)[None, :] < valid[1])
    return jax.lax.dynamic_update_slice(accumulator, jnp.where(live, new, block), start)

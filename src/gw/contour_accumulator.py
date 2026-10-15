"""The contour accumulator: ``A[o,q,m,n] += p[o] · c[q,m,n]``, in XLA.

The response bank's Laplace/KMS streams (``gw.w_isdf``) accumulate one shared
contour correlation into many outputs on each device's tile, and the streamed
row passes (``gw.subtile_stream``) add a block of it at a row offset.  Both are
elementwise products and sums in XLA; the block form adds one term per loop
step in place, which holds the deleted CUDA kernel's peak at 1.4-1.5x its time
on that stage (``docs/architecture/decisions.md#xla-reference``).
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


# Output rows per accumulator pass: a node's update reads and writes OUT_BLOCK
# output rows of the block at a time, over the live output prefix only.  Two
# rows are one bank member (its value and ds rows); see the docstring.
OUT_BLOCK = 2


def contour_block_accumulate_local(accumulator, contribution, projection, valid, *, m0, n0,
                                   sets=1, terms=None, outputs=None, out_block=None):
    """One device tile: ``A[o, q, m0+m, n0+n] += sum_s projection[s,o] contribution[s,q,m,n]``.

    For code already inside ``shard_map``.  ``accumulator`` ``[output,q,M,N]``,
    ``contribution`` ``[terms,q,bm,bn]``, ``projection`` ``[terms,output]``
    complex128, ``valid`` ``[2]`` int32: the block's rows and columns that are
    not padding (the rest are not touched).  ``m0``/``n0`` are Python ints or
    traced ints (a scanned pass's row offset).

    The terms come in groups of ``sets`` consecutive terms (one rule node's
    planes: the direct stream's forward and reverse rows).  A group is added in
    ONE read-modify-write of the outputs it touches, its terms in order, so
    every element gets ``((A + p_0 c_0) + p_1 c_1) + ...`` with the full
    accumulator's rounding: the bytes of one :func:`contour_accumulator` call
    per term.  Only one group's planes are read per step; nothing larger than
    ``out_block`` output rows of the block is formed.

    ``terms`` (traced int, default all): only the first ``terms`` terms are
    added; ``outputs`` (traced int, default all): only outputs ``[0, outputs)``
    are read or written.  The caller guarantees the skipped terms and outputs
    carry zero weights, so skipping them is the exact zeros they would add
    (the bank's padded member slots and a chunk's nodes past the live count).
    The step count is ``ceil(terms/sets) * ceil(outputs/out_block)``: traffic
    scales with the live outputs, not the planned slots.
    """
    if any(a.dtype != jnp.complex128 for a in (accumulator, contribution, projection)):
        raise TypeError("contour block accumulator requires complex128 operands")
    n_terms, n_out = int(contribution.shape[0]), int(accumulator.shape[0])
    sets = int(sets)
    if sets < 1 or n_terms % sets or projection.shape != (n_terms, n_out):
        raise ValueError("contour block accumulator: terms must be whole groups of sets "
                         "and projection [terms, output]")
    B = min(int(OUT_BLOCK if out_block is None else out_block), n_out)
    i32 = lambda v: jnp.asarray(v, jnp.int32)
    groups = i32(n_terms // sets if terms is None else (i32(terms) + sets - 1) // sets)
    live_out = i32(n_out if outputs is None else jnp.clip(i32(outputs), 0, n_out))
    n_blocks = (live_out + B - 1) // B
    # One index dtype: a traced int32 offset beside x64 Python ints is a TypeError.
    m0, n0 = i32(m0), i32(n0)
    bm, bn = contribution.shape[2:]
    live = (jnp.arange(bm)[:, None] < valid[0]) & (jnp.arange(bn)[None, :] < valid[1])

    # One (group, output block) per step, updated in place on the accumulator itself:
    # an unrolled chain lets XLA materialize every term's [output, q, bm, bn] product at
    # once (Fe 4^3 forced face, P4: 9.87 -> 45.23 GB), and a separate block carry costs a
    # block more.  The last block is clamped to the accumulator's end; its rows below the
    # block's own first output (already done) and from ``outputs`` on are rewritten as read.
    def step(i, acc):
        g, b = i // n_blocks, i % n_blocks
        first = b * B
        o0 = jnp.minimum(first, n_out - B)
        start = (o0, i32(0), m0, n0)
        block = jax.lax.dynamic_slice(acc, start, (B, acc.shape[1], bm, bn))
        c = jax.lax.dynamic_slice_in_dim(contribution, g * sets, sets, axis=0)
        p = jax.lax.dynamic_slice(projection, (g * sets, o0), (sets, B))
        new = block
        for s in range(sets):
            new = new + p[s][:, None, None, None] * c[s][None]
        rows = o0 + jnp.arange(B, dtype=jnp.int32)
        keep = ((rows >= first) & (rows < live_out))[:, None, None, None] & live[None, None]
        return jax.lax.dynamic_update_slice(acc, jnp.where(keep, new, block), start)
    return jax.lax.fori_loop(i32(0), groups * n_blocks, step, accumulator)

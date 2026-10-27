"""Block-cyclic relabeling of a ``P('x','y')`` face operand.

Shared by the distributed-eigh FFI wrappers (cuSOLVERMp and ELPA); the
block POLICY (which block edge, when to pad) stays with each wrapper, since it
was measured per library.  Moved here unchanged from ``_cusolvermp`` (the
helpers had one caller there); the only generalisation is the mesh axis of
:func:`cyclic_to_block` (cuSOLVERMp un-permutes rows over ``'x'``, ELPA the
columns of its true-column Q, which run over ``'y'``).

The relabeling: a ``P('x','y')`` operand is one ``(n/p, n/p)`` tile per rank.
Handing a block-cyclic library the SAME local buffers with a smaller square
block ``mb`` (``mb | n/p``) makes it read them as ``Pi A Pi^T``, ``Pi``
sending rank ``r``'s local block ``c`` to global block ``c*p + r``.  Rows and
columns get the same ``Pi`` (square grid), so the eigenvalues are ``A``'s and
the eigenvector components read back in ``A``'s order with no data movement.
Only the eigenvalue index comes back block-cyclic; one ``all_to_all``
(:func:`cyclic_to_block`, ``n^2/P`` per rank) puts it in ascending order.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["largest_divisor", "pad_with_sentinels", "cyclic_to_block"]


def largest_divisor(local: int, cap: int) -> int:
    """The largest divisor of a tile edge ``local`` that is at most ``cap``."""
    return max(d for d in range(1, min(local, cap) + 1) if local % d == 0)


def pad_with_sentinels(a, side: int):
    """``a`` (n x n) padded to ``side``, the padded diagonal below its spectrum.

    The padded rows and columns are zero except their diagonal, which holds
    distinct sentinels ``-2 b (1 + j/side)``, ``b`` the Gershgorin bound of
    ``a``: decoupled eigenpairs below every eigenvalue of ``a``.
    """
    n = int(a.shape[-1])
    bound = jnp.max(jnp.sum(jnp.abs(a), axis=-1))
    bound = jnp.where(bound > 0, bound, 1).astype(jnp.float64)
    row = jax.lax.broadcasted_iota(jnp.int32, (side, side), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (side, side), 1)
    sentinel = -2 * bound * (1 + (col - n).astype(jnp.float64) / side)
    whole = jnp.pad(a, ((0, side - n), (0, side - n)))
    return whole + jnp.where((row == col) & (col >= n), sentinel, 0).astype(a.dtype)


def cyclic_to_block(Q, *, n: int, p: int, mb: int, axis: str = "x"):
    """Inside a shard_map: put rows held block-cyclically over ``axis`` in order.

    Local row ``c*mb + t`` of ``axis`` rank ``a`` is global row
    ``(c*p + a)*mb + t``; it moves to rank ``g // C``, slot ``g % C`` of
    the contiguous layout (``g = c*p + a``, ``C = n/(p*mb)``).  Each pair of
    ranks exchanges a contiguous run of at most ``J`` blocks, padded to
    ``J`` for one ``all_to_all``.
    """
    C = n // (p * mb)
    runs = [[(max(0, -(-(t * C - a) // p)),
              min(C, -(-((t + 1) * C - a) // p))) for t in range(p)]
            for a in range(p)]
    J = max(hi - lo for row in runs for lo, hi in row)
    send = np.zeros((p, p * J), np.int32)       # [a, t*J + j] -> local block
    recv = np.zeros((p, C), np.int32)           # [t, slot] -> s*J + j
    for a in range(p):
        for t, (lo, hi) in enumerate(runs[a]):
            for j, c in enumerate(range(lo, hi)):
                send[a, t * J + j] = c
                recv[t, (c * p + a) - t * C] = a * J + j
    a = jax.lax.axis_index(axis)
    blocks = Q.reshape(C, mb, Q.shape[-1])
    out = jnp.take(blocks, jnp.asarray(send)[a], axis=0)
    out = jax.lax.all_to_all(out, axis, 0, 0, tiled=True)
    out = jnp.take(out, jnp.asarray(recv)[a], axis=0)
    return out.reshape(Q.shape)


def face_crop(Q, n: int, *, offsets=(0, 0), mesh=None):
    """Crop a face to ``n x n`` through two dependent uniform axis transfers.

    Each axis first trades spectator columns for all rows. The resulting
    product-owned bank has no more elements than the padded local face, so
    the contiguous row crop is local. The reverse transfer restores row
    ownership. Only local spectator columns are padded, by at most p-1.
    """
    from jax import shard_map
    from jax.sharding import PartitionSpec as P

    n, offsets = int(n), tuple(map(int, offsets))
    if n < 0 or len(offsets) != 2 or any(
            start < 0 or start+n > extent for start, extent in zip(offsets, Q.shape[-2:])):
        raise ValueError("face crop is outside the input extents")
    if mesh is None:
        return Q[..., offsets[0]:offsets[0]+n, offsets[1]:offsets[1]+n]
    if any(n % int(mesh.shape[axis]) for axis in ("x", "y")):
        raise ValueError("face crop output must divide both mesh axes")
    if Q.shape[-2:] == (n, n) and offsets == (0, 0):
        return Q
    spec = P(*((None,)*(Q.ndim-2)), "x", "y")

    def rows(current, start, axis):
        parts = int(mesh.shape[axis])
        if parts == 1:
            return current[..., start:start+n, :]
        columns = int(current.shape[-1])
        pad = (-columns) % parts
        current = jnp.pad(current, ((0, 0),)*(current.ndim-1)+((0, pad),))
        bank = jax.lax.all_to_all(current, axis, current.ndim-1, current.ndim-2, tiled=True)
        bank = bank[..., start:start+n, :]
        face = jax.lax.all_to_all(bank, axis, bank.ndim-2, bank.ndim-1, tiled=True)
        return face[..., :columns]

    @partial(shard_map, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False)
    def local(q):
        if n == 0:
            return jnp.zeros(q.shape[:-2]+(0, 0), q.dtype)
        q = rows(q, offsets[0], "x")
        q = rows(jnp.swapaxes(q, -1, -2), offsets[1], "y")
        return jnp.swapaxes(q, -1, -2)

    return local(Q)

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

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["largest_divisor", "pad_with_sentinels", "cyclic_to_block", "face_crop"]


def face_crop(q, n: int, *, offsets, mesh):
    """Crop a distributed matrix face through two dependent local-tile rings.

    Output ``[..., i, j]`` is input ``[..., i+offsets[0], j+offsets[1]]``.
    The row ring feeds the column ring inside one manual region; no implicit
    pad-to-logical reshard or bank of all columns is required. Local callers
    retain the ordinary slice. Both input and output extents divide the mesh.
    """
    row, column = offsets
    if n < 0 or row < 0 or column < 0 or row+n > q.shape[-2] or column+n > q.shape[-1]:
        raise ValueError("matrix crop lies outside the input")
    if mesh is None:
        return q[..., row:row+n, column:column+n]
    from jax.sharding import PartitionSpec as P
    from lxkit import permute_rows_local
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    if n % px or n % py:
        raise ValueError("matrix crop extent must divide both mesh axes")
    if q.shape[-2:] == (n, n) and offsets == (0, 0):
        return q
    spec = P(*((None,) * (q.ndim-2)), 'x', 'y')
    @jax.shard_map(mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False)
    def crop(tile):
        if n == 0:
            return jnp.zeros(tile.shape[:-2] + (0, 0), tile.dtype)
        def rows(block, offset, axis, parts):
            shape = block.shape
            packed = block.reshape((-1,) + shape[-2:])
            source = jnp.broadcast_to((jnp.arange(n, dtype=jnp.int32)+offset)[None, :],
                                      (packed.shape[0], n))
            selected = permute_rows_local(packed, source, mesh_axis=axis,
                                          parts=parts, is_local=parts == 1)
            return selected.reshape(shape[:-2] + (n//parts, shape[-1]))
        cropped = rows(tile, row, 'x', px)
        cropped = rows(jnp.swapaxes(cropped, -1, -2), column, 'y', py)
        return jnp.swapaxes(cropped, -1, -2)
    return crop(q)


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

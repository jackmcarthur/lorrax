"""Block-cyclic relabeling of a ``P('x','y')`` face operand.

Shared by the two distributed-eigh FFI wrappers (cuSOLVERMp and SLATE); the
block POLICY (which block edge, when to pad) stays with each wrapper, since it
was measured per library.  Moved here unchanged from ``_cusolvermp`` (the
helpers had one caller there); the only generalisation is the mesh axis of
:func:`cyclic_to_block` (cuSOLVERMp un-permutes rows over ``'x'``, SLATE the
columns of its true-column Q, which run over ``'y'``).

The relabeling: a ``P('x','y')`` operand is one ``(n/p, n/p)`` tile per rank.
Handing a block-cyclic library the SAME local buffers with a smaller square
block ``mb`` (``mb | n/p``) makes it read them as ``Pi A Pi^T``, ``Pi``
sending rank ``r``'s local block ``c`` to global block ``c*p + r``.  Rows and
columns get the same ``Pi`` (square grid), so the eigenvalues are ``A``'s and
the eigenvector components read back in ``A``'s order with no data movement.
Only the eigenvalue index comes back block-cyclic; :func:`cyclic_to_block`
(``n^2/P`` per rank moved, in rounds of small pieces) puts it in ascending order.
"""
from __future__ import annotations

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


#: The largest piece one exchange step of :func:`cyclic_to_block` moves, as a
#: fraction of the tile: steps run in a loop, so a rank holds the output tile
#: and one piece sent and one received beside its input (EIGHMEM: the one-shot
#: all_to_all held 2.08 n^2/P at 4x4).
PIECE_FRACTION = 8


def _runs(n: int, p: int, mb: int):
    """``(C, J, send, recv)`` of the relabeling's exchange, as the one-shot ``all_to_all`` lays it.

    ``send[a, t*J + j]`` is the local block rank ``a`` sends to rank ``t`` in slot
    ``j``; ``recv[t, slot]`` is where rank ``t``'s output block ``slot`` sits in
    the received ``(p*J)`` blocks (``s*J + j``, from rank ``s`` slot ``j``).
    """
    C = n // (p * mb)
    runs = [[(max(0, -(-(t * C - a) // p)),
              min(C, -(-((t + 1) * C - a) // p))) for t in range(p)]
            for a in range(p)]
    J = max(hi - lo for row in runs for lo, hi in row)
    send = np.zeros((p, p * J), np.int32)       # [a, t*J + j] -> local block
    recv = np.zeros((p, C), np.int32)           # [t, slot] -> s*J + j
    count = np.zeros((p, p), np.int32)          # [a, t] -> blocks a sends t
    for a in range(p):
        for t, (lo, hi) in enumerate(runs[a]):
            count[a, t] = hi - lo
            for j, c in enumerate(range(lo, hi)):
                send[a, t * J + j] = c
                recv[t, (c * p + a) - t * C] = a * J + j
    return C, J, send, recv, count


def cyclic_to_block(Q, *, n: int, p: int, mb: int, axis: str = "x"):
    """Inside a shard_map: put rows held block-cyclically over ``axis`` in order.

    Local row ``c*mb + t`` of ``axis`` rank ``a`` is global row
    ``(c*p + a)*mb + t``; it moves to rank ``g // C``, slot ``g % C`` of
    the contiguous layout (``g = c*p + a``, ``C = n/(p*mb)``).  Each pair of
    ranks exchanges a contiguous run of at most ``J`` blocks.

    The exchange runs in rounds, not one ``all_to_all``: round ``s`` sends
    rank ``a``'s run for rank ``a+s`` (a ``ppermute``; round 0 is the local
    run) in pieces of at most ``C/PIECE_FRACTION`` blocks, one per step of a
    ``fori_loop``, each scattered into the output tile as it arrives. A loop
    keeps one piece in flight whatever the scheduler (an optimization barrier
    did not: XLA:CPU hoisted every piece's gather ahead of the chain), so a
    rank holds ``Q``, the output and two pieces: ``n^2/P (1 + 2/PIECE_FRACTION)``
    beside ``Q`` where the one-shot exchange held ``2 J p/C n^2/P``. The same
    blocks land in the same slots: bitwise the ``all_to_all``'s result.
    """
    C, J, send, recv, count = _runs(n, p, mb)
    piece = max(1, C // PIECE_FRACTION)
    pieces = -(-J // piece)
    Jp = pieces * piece
    # sends[a, t, j]: the local block rank a sends rank t in slot j (block 0, dropped on
    # arrival, past the run); place[a, u, j]: the output slot of the block rank a receives
    # from rank u in slot j (C: out of range, dropped, past u's run for a).
    sends = np.zeros((p, p, Jp), np.int32)
    sends[:, :, :J] = send.reshape(p, p, J)
    place = np.full((p, p, Jp), C, np.int32)
    for t in range(p):
        for slot in range(C):
            u, j = divmod(int(recv[t, slot]), J)
            place[t, u, j] = slot
    a = jax.lax.axis_index(axis)
    blocks = Q.reshape(C, mb, Q.shape[-1])
    # Every slot is written exactly once below, so the initial value is never read; it is
    # derived from Q (not a constant) so it cannot be scheduled before the solve that
    # produces Q, where it would hold the output's buffer during the library call (the
    # buffer XLA otherwise lends the solver's destroyed operand).
    out = blocks * 0
    for s in range(p):
        # Round s: rank a sends its run for rank (a+s) % p and receives rank (a-s) % p's
        # run for it, one piece per loop step (a loop, so one piece is live at a time
        # whatever the scheduler; round 0 is the local run, no exchange).
        mine = jnp.asarray(sends)[a, (a + s) % p]
        theirs = jnp.asarray(place)[a, (a - s) % p]
        perm = [(i, (i + s) % p) for i in range(p)]

        def step(k, out, mine=mine, theirs=theirs, perm=perm, s=s):
            chunk = jnp.take(blocks, jax.lax.dynamic_slice_in_dim(mine, k * piece, piece), axis=0)
            if s:
                chunk = jax.lax.ppermute(chunk, axis, perm)
            slots = jax.lax.dynamic_slice_in_dim(theirs, k * piece, piece)
            return out.at[slots].set(chunk, mode="drop")
        out = jax.lax.fori_loop(0, pieces, step, out)
    return out.reshape(Q.shape)


def cyclic_to_block_all_to_all(Q, *, n: int, p: int, mb: int, axis: str = "x"):
    """The one-shot form :func:`cyclic_to_block` replaced, kept as its bitwise reference (tests)."""
    C, J, send, recv, _ = _runs(n, p, mb)
    a = jax.lax.axis_index(axis)
    blocks = Q.reshape(C, mb, Q.shape[-1])
    out = jnp.take(blocks, jnp.asarray(send)[a], axis=0)
    out = jax.lax.all_to_all(out, axis, 0, 0, tiled=True)
    out = jnp.take(out, jnp.asarray(recv)[a], axis=0)
    return out.reshape(Q.shape)

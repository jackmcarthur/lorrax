"""Resolvent-identity pencil columns for the shared-pole construction.

The even pencil in s = z**2 (W 18, W 19) and the ordered particle-hole pencil
(z sigma_3 - M) (W 25, W 26) of docs/theory/shared-pole-w-model.md. Every
[b, R, R] block is assembled on the x/y face through distrib_la.blocks; eager
concatenation and a + a^H of face-sharded operands come out replicated.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from distrib_la import (face_sharding, hermitian_block, hermitian_part,
                        join_columns, on_face)


def _matrix_layout(a, matrix_sharding):
    """Pin a known matrix intermediate only for an explicit execution layout.

    Local shard_map equations pass None. Distributed callers pass the full
    XY matrix layout; no array dtype or metadata heuristic selects it.
    """
    return a if matrix_sharding is None else jax.lax.with_sharding_constraint(a, matrix_sharding)


def _matrix_concat(arrays, axis, matrix_sharding):
    """Join matrix blocks through the common bounded face redistribution."""
    if matrix_sharding is None:
        return jnp.concatenate(arrays, axis=axis)
    from common.staged_reshard import concatenate_sharded_axis
    return concatenate_sharded_axis(arrays, axis, matrix_sharding.mesh, matrix_sharding.spec)


def _matrix_take_columns(a, order, matrix_sharding):
    """Per-parent columns, using the shared bounded permutation on a face."""
    if matrix_sharding is None:
        return jnp.take_along_axis(a, order[:, None, :], axis=-1)
    from common.staged_reshard import permute_sharded_axis
    return permute_sharded_axis(a, -1, order, matrix_sharding.mesh, matrix_sharding.spec)


def _adjoint(a):
    return jnp.conj(jnp.swapaxes(a, -1, -2))


# ---- the tile-interleaved block order of a face pencil ----
#
# A pencil axis is a run of logical blocks: the originals and mirrors of the finite
# states, then the k0 and k1 infinity directions; or the w and v halves of the paired
# basis. On a p x p face a matrix holds each axis in the tile-interleaved order: the
# axis block of rank row (or column) b is [piece b of block 1 | piece b of block 2 | ...],
# piece b being the block's b-th 1/p. Joining or splitting logical blocks along an axis
# is then every rank's own concatenation or slice of its tiles, and no byte moves; the
# paired-basis congruence pairs columns that sit on the same rank. With whole matrices
# (p = 1) the order is the plain concatenation, so one set of equations serves the
# q-local and the face routes. Every logical block is a multiple of p (round_tables
# rounds the extent to the carrier grain; the infinity width is a port carrier).
# Replicated per-column vectors (nodes, masks, inverse nodes) follow the same order
# (``join_vectors``, ``split_vectors``); a round's host tables enter it once
# (``interleave_tables``).


def tile_count(matrix_sharding):
    """p of the tile-interleaved order: 1 for whole matrices, the face's mesh side otherwise."""
    return 1 if matrix_sharding is None else int(matrix_sharding.mesh.shape['y'])


def on_tiles(fn, matrix_sharding, arrays, *, rows=(), cols=(), outputs=1):
    """``fn(*tiles, *row_pieces, *column_pieces)`` on every rank's tiles of face matrices
    [b, R, C]; ``rows``/``cols`` are replicated per-row/per-column vectors [b, R] / [b, C],
    each rank receiving its own pieces. Returns ``outputs`` face matrices."""
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    face, row, col = P(None, 'x', 'y'), P(None, 'x'), P(None, 'y')
    specs = (face,) * len(arrays) + (row,) * len(rows) + (col,) * len(cols)
    out = face if outputs == 1 else (face,) * outputs
    return shard_map(fn, mesh=matrix_sharding.mesh, in_specs=specs, out_specs=out,
                     check_vma=False)(*arrays, *rows, *cols)


def tile_join(arrays, axis, matrix_sharding):
    """Join matrix blocks along ``axis`` (-1 columns, -2 rows) in the tile-interleaved order:
    each rank concatenates its own tiles. Whole matrices: the plain concatenation."""
    if matrix_sharding is None:
        return jnp.concatenate(arrays, axis=axis)
    return on_tiles(lambda *t: jnp.concatenate(t, axis=axis), matrix_sharding, arrays)


def tile_split(a, sizes, axis, matrix_sharding):
    """The inverse of ``tile_join``: ``a``'s logical blocks of ``sizes`` along ``axis``."""
    import numpy as np
    p = tile_count(matrix_sharding)
    edges = np.concatenate(([0], np.cumsum([int(s) // p for s in sizes])))

    def pieces(t):
        return tuple(jax.lax.slice_in_dim(t, int(lo), int(hi), axis=axis) for lo, hi in zip(edges[:-1], edges[1:]))
    if matrix_sharding is None:
        return pieces(a)
    return on_tiles(pieces, matrix_sharding, (a,), outputs=len(sizes))


def tile_adjoint(a, matrix_sharding):
    """``a^H`` of a face matrix: the tile at the mirrored grid position, conjugate-transposed.
    On the square mesh that is one ``ppermute`` across the grid's diagonal (the tiles keep the
    tile-interleaved order, which rows and columns share)."""
    if matrix_sharding is None:
        return _adjoint(a)
    p = tile_count(matrix_sharding)
    across = tuple((x * p + y, y * p + x) for x in range(p) for y in range(p))
    return on_tiles(lambda t: _adjoint(jax.lax.ppermute(t, ('x', 'y'), perm=across)), matrix_sharding, (a,))


def tile_hermitian(a, matrix_sharding):
    """``(a + a^H) / 2`` on the face (``distrib_la.hermitian_part``'s arithmetic)."""
    if matrix_sharding is None:
        return hermitian_part(a)
    return (a + tile_adjoint(a, matrix_sharding)) * 0.5


def tile_block(rows, matrix_sharding):
    """The block matrix ``[[A, B, ...], [C, D, ...], ...]`` in the tile-interleaved order."""
    return tile_join([tile_join(row, -1, matrix_sharding) for row in rows], -2, matrix_sharding)


def join_vectors(vectors, p):
    """Replicated per-column vectors [..., s_i] joined in the tile-interleaved order of ``p``."""
    xp = jnp if any(isinstance(v, jax.Array) or isinstance(v, jax.core.Tracer) for v in vectors) else __import__('numpy')
    if p == 1:
        return xp.concatenate(vectors, axis=-1)
    lead = vectors[0].shape[:-1]
    parts = [v.reshape(*lead, p, v.shape[-1] // p) for v in vectors]
    return xp.concatenate(parts, axis=-1).reshape(*lead, -1)


def split_vectors(v, sizes, p):
    """The inverse of ``join_vectors``: ``v``'s logical blocks of ``sizes``."""
    import numpy as np
    edges = np.concatenate(([0], np.cumsum([int(s) // p for s in sizes])))
    lead = v.shape[:-1]
    pieces = v.reshape(*lead, p, v.shape[-1] // p)
    return tuple(pieces[..., int(lo):int(hi)].reshape(*lead, -1) for lo, hi in zip(edges[:-1], edges[1:]))


def interleave_tables(tables, p):
    """A round's ordered column tables (``round_tables``) in the face's tile-interleaved
    pencil order: ``order`` and ``points`` [P, F] are the originals and mirrors of the finite
    states (F/2 each), ``active`` [P, side] those and then the k0 and k1 infinity blocks."""
    if p == 1:
        return tables
    order, active = tables['order'], tables['active']
    half, n_inf = order.shape[-1] // 2, (active.shape[-1] - order.shape[-1]) // 2
    if half % p or n_inf % p:
        raise ValueError(f"GATE shared_pole_tile_order: got: finite half {half} and infinity width {n_inf}; "
                         f"want: multiples of the mesh side {p}; why: each rank's tile holds 1/p of every block")
    out = dict(tables)
    for key in ('order', 'points'):
        out[key] = join_vectors(split_vectors(tables[key], (half, half), 1), p)
    out['active'] = join_vectors(split_vectors(active, (half, half, n_inf, n_inf), 1), p)
    return out


def _scale_rows(s, g):
    return s[:, None, :] * g


def _subtract(x, y):
    return x - y


def finite_pencil_column(left, right, *, matmul, matrix_sharding=None):
    """Form one block column of the resolvent-identity pencil.

    Parameters
    ----------
    left : tuple
        ``(s, Q, O)`` with s [R] or [b,R] in Ry**2, Q/O [b,n,R] complex128
        face tiles and O_a = W(s_a) Q_a. Repeated entries of s label
        separate tangential columns, including conjugate and role states.
    right : tuple
        ``(s_b, Q_b, O_b, D_b)`` with panels [b,n,r] in the same layout,
        s_b scalar or [b,r], O_b = W(s_b) Q_b and D_b = dW(s_b)/ds Q_b.
    matmul : callable
        Resolved service GEMM; accepts ``transa='C'`` for the adjoint.

    Returns
    -------
    g, h : arrays
        [b,R,r] complex128 face tiles. They are X.H X and X.H T X
        blocks for X_b = (s_b-T)^-1 b.H Q_b (algorithm guide, section 4).
    """
    sa, qa, oa = left
    sb, qb, ob, db = right
    a = matmul(oa, qb, transa="C")
    # Production assembles the complete square finite block with shared
    # panels. For distinct left/right blocks the second product is needed.
    shared = qa is qb and oa is ob
    b = None if shared else matmul(qa, ob, transa="C")
    derivative = matmul(qa, db, transa="C")
    sa = jnp.broadcast_to(sa, (qa.shape[0], qa.shape[-1]))
    sb = jnp.broadcast_to(sb, (qb.shape[0], qb.shape[-1]))
    # On the face the [b,R,r] intermediates (denominator, confluent mask, a - a^H)
    # stay per-rank tiles; eager broadcasting of the replicated supports and
    # a - a^H would replicate them. Multiply and subtract stay separate programs.
    face = face_sharding(a)
    g = _matrix_layout(on_face(_finite_column_g, face, a, b, derivative, sa, sb), matrix_sharding)
    return g, _matrix_layout(on_face(_subtract, face, on_face(_scale_rows, face, sb, g), a), matrix_sharding)


def _finite_column_g(a, b, derivative, sa, sb):
    """G block of one resolvent-identity column; b = a^H when None (shared panels)."""
    b = _adjoint(a) if b is None else b
    denominator = sb[:, None, :] - jnp.conj(sa[:, :, None])
    # This is the inherited floating-point equality test for confluent s,
    # not a physical support-merging tolerance. Roles are never merged.
    scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(sa[:, :, None]),
                                        jnp.abs(sb[:, None, :])))
    confluent = jnp.abs(denominator) <= 8 * jnp.finfo(jnp.float64).eps * scale
    safe = jnp.where(confluent, 1.0 + 0j, denominator)
    return jnp.where(confluent, -derivative, (a - b) / safe)


def infinity_pencil_column(finite, infinity, *, matmul):
    """Form infinity rows using physical M1 and M3 (guide section 4).

    ``finite=(s,Q,O)`` carries [R] or [b,R], [b,n,R], [b,n,R].
    ``infinity=(Q_inf,M1_Q_inf,M3_Q_inf)`` carries three [b,n,r_inf]
    complex128 face panels. Returns G_inf,finite, H_inf,finite,
    G_inf,inf, H_inf,inf and O_inf. No full moment matrix is retained.
    """
    s, q, output = finite
    qi, m1qi, m3qi = infinity
    gi = matmul(qi, output, transa="C")
    s = jnp.broadcast_to(s, (q.shape[0], q.shape[-1]))
    hi = gi * s[:, None, :] - 2 * matmul(m1qi, q, transa="C")
    gii = 2 * matmul(qi, m1qi, transa="C")
    hii = 2 * matmul(qi, m3qi, transa="C")
    return gi, hi, gii, hii, 2 * m1qi


def assemble_shared_pole_pencil(states, infinity, *, matmul):
    """Assemble G, H and O from bounded-sample action panels.

    ``states`` is an ordered sequence of ``(s,Q,WQ,dWQ)`` tuples, where
    s is scalar or [b,r_a] Ry**2; panels are complex128 [b,n,r_a] face tiles.
    Each state preserves its own role, even when samples are shared.
    ``infinity`` contains Q_inf, M1 Q_inf, M3 Q_inf [b,n,r_inf].
    Only narrow panels and the dense pencil are resident here; full W
    samples must already have been released by their bounded producer.
    """
    q = jnp.concatenate([state[1] for state in states], axis=-1)
    output = jnp.concatenate([state[2] for state in states], axis=-1)
    derivative = jnp.concatenate([state[3] for state in states], axis=-1)
    s = jnp.concatenate([jnp.broadcast_to(jnp.asarray(state[0], jnp.complex128),
                         (state[1].shape[0], state[1].shape[-1]))
                         for state in states], axis=-1)
    finite = (s, q, output)
    # All finite columns share three products. The confluent derivative and
    # support coordinates remain column-specific, including repeated roles.
    g, h = finite_pencil_column(finite, (s, q, output, derivative), matmul=matmul)
    gi, hi, gii, hii, oi = infinity_pencil_column(finite, infinity, matmul=matmul)
    # Face blocks: eager concatenation with the adjoint (y/x) infinity rows and eager
    # a + a^H would return a replicated [b,side,side] G and H on every rank.
    g, h = hermitian_block(g, gi, gii), hermitian_block(h, hi, hii)
    return hermitian_part(g), hermitian_part(h), join_columns(output, oi)


def ordered_infinity_pencil_column(finite, infinity, *, matmul, matrix_sharding=None):
    """Infinity rows of the linear particle-hole pencil (z s3 - M).

    ``finite=(z,Q,O)`` carries complex z in Ry ([R] or [b,R]) and [b,n,R]
    face panels. ``infinity=(Q_inf, M0 Q_inf, M1 Q_inf, M2 Q_inf, M3 Q_inf)``
    with physical z-moments Wc(z) = sum_n 2 M_n z^-(n+1); M0 and M2 are odd
    under time reversal. States are k0 = s3 C^H Q and k1 = s3 M s3 C^H Q.
    Returns G/H infinity-finite rows [b,2r,R], the [b,2r,2r] infinity
    blocks and the outputs C k [b,n,2r], the k0 and k1 blocks joined in the
    tile-interleaved order of ``matrix_sharding``. No full moment matrix is retained.
    """
    z, q, output = finite
    qi, m0qi, m1qi, m2qi, m3qi = infinity
    z = jnp.broadcast_to(z, (q.shape[0], q.shape[-1]))[:, None, :]
    fi = matmul(qi, output, transa="C")
    q0 = 2 * matmul(m0qi, q, transa="C")
    q1 = 2 * matmul(m1qi, q, transa="C")
    ms = matrix_sharding
    g = tile_join((fi, fi * z - q0), -2, ms)
    h = tile_join((fi * z - q0, fi * z * z - q0 * z - q1), -2, ms)
    p0, p1, p2, p3 = (2 * matmul(qi, m, transa="C") for m in (m0qi, m1qi, m2qi, m3qi))
    gii = tile_block(((p0, p1), (p1, p2)), ms)
    hii = tile_block(((p1, p2), (p2, p3)), ms)
    return tuple(_matrix_layout(a, ms) for a in (g, h, gii, hii, 2 * tile_join((m0qi, m1qi), -1, ms)))


def assemble_ordered_shared_pole_pencil(states, infinity, *, matmul, matrix_sharding=None):
    """Assemble G=X^H s3 X, H=X^H M X and O=C X for time-reversal-broken data.

    ``states`` holds ``(z,Q,WQ,dW/dz Q)`` with complex z in Ry, not s=z**2,
    in the paired layout of ``select_round_states(ordered=True)``: every state
    X(z) followed, after all originals, by its mirror X(-z) on the same
    directions. The resolvent-identity column of ``finite_pencil_column`` is
    exact for the linear pencil (z s3 - M) as written. ``infinity`` is None for
    a finite-state bank, else the five panels of
    ``ordered_infinity_pencil_column``. Returns Hermitian G, H, O and the
    finite nodes z [b,R_finite] that the paired reduction needs, in the
    tile-interleaved order of ``matrix_sharding`` (whole matrices: finite columns,
    then k0, then k1).
    """
    q = jnp.concatenate([state[1] for state in states], axis=-1)
    output = jnp.concatenate([state[2] for state in states], axis=-1)
    derivative = jnp.concatenate([state[3] for state in states], axis=-1)
    z = jnp.concatenate([jnp.broadcast_to(jnp.asarray(state[0], jnp.complex128),
                         (state[1].shape[0], state[1].shape[-1]))
                         for state in states], axis=-1)
    q, output, derivative = (_matrix_layout(a, matrix_sharding) for a in (q, output, derivative))
    finite = (z, q, output)
    g, h = finite_pencil_column(finite, (z, q, output, derivative), matmul=matmul, matrix_sharding=matrix_sharding)
    del derivative
    # The finite and infinity blocks join on every rank's own tiles (the infinity rows'
    # adjoint is one exchange across the grid's diagonal), so no rank holds more than its tile.
    ms = matrix_sharding
    if infinity is not None:
        gi, hi, gii, hii, oi = ordered_infinity_pencil_column(finite, infinity, matmul=matmul, matrix_sharding=ms)
        g = tile_block(((g, tile_adjoint(gi, ms)), (gi, gii)), ms)
        h = tile_block(((h, tile_adjoint(hi, ms)), (hi, hii)), ms)
        del gi, hi, gii, hii
        output = tile_join((output, oi), -1, ms)
    return (*(_matrix_layout(a, ms) for a in (tile_hermitian(g, ms), tile_hermitian(h, ms), output)), z)

"""Resolvent-identity pencil columns for the shared-pole construction.

The even pencil in s = z**2 (W 18, W 19) and the ordered particle-hole pencil
(z sigma_3 - M) (W 25, W 26) of docs/theory/shared-pole-w-model.md. Every
[b, R, R] block is assembled on the x/y face through distrib_la.blocks; eager
concatenation and a + a^H of face-sharded operands come out replicated.
"""

from __future__ import annotations

import jax.numpy as jnp
from distrib_la import (face_sharding, hermitian_block, hermitian_part,
                        join_columns, on_face)


def _adjoint(a):
    return jnp.conj(jnp.swapaxes(a, -1, -2))


def _scale_rows(s, g):
    return s[:, None, :] * g


def _subtract(x, y):
    return x - y


def finite_pencil_column(left, right, *, matmul):
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
    g = on_face(_finite_column_g, face, a, b, derivative, sa, sb)
    return g, on_face(_subtract, face, on_face(_scale_rows, face, sb, g), a)


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


def ordered_infinity_pencil_column(finite, infinity, *, matmul):
    """Infinity rows of the linear particle-hole pencil (z s3 - M).

    ``finite=(z,Q,O)`` carries complex z in Ry ([R] or [b,R]) and [b,n,R]
    face panels. ``infinity=(Q_inf, M0 Q_inf, M1 Q_inf, M2 Q_inf, M3 Q_inf)``
    with physical z-moments Wc(z) = sum_n 2 M_n z^-(n+1); M0 and M2 are odd
    under time reversal. States are k0 = s3 C^H Q and k1 = s3 M s3 C^H Q.
    Returns G/H infinity-finite rows [b,2r,R], the [b,2r,2r] infinity
    blocks and the outputs C k [b,n,2r]. No full moment matrix is retained.
    """
    z, q, output = finite
    qi, m0qi, m1qi, m2qi, m3qi = infinity
    z = jnp.broadcast_to(z, (q.shape[0], q.shape[-1]))[:, None, :]
    fi = matmul(qi, output, transa="C")
    q0 = 2 * matmul(m0qi, q, transa="C")
    q1 = 2 * matmul(m1qi, q, transa="C")
    g = jnp.concatenate((fi, fi * z - q0), axis=-2)
    h = jnp.concatenate((fi * z - q0, fi * z * z - q0 * z - q1), axis=-2)
    p0, p1, p2, p3 = (2 * matmul(qi, m, transa="C") for m in (m0qi, m1qi, m2qi, m3qi))
    gii = jnp.concatenate((jnp.concatenate((p0, p1), axis=-1),
                           jnp.concatenate((p1, p2), axis=-1)), axis=-2)
    hii = jnp.concatenate((jnp.concatenate((p1, p2), axis=-1),
                           jnp.concatenate((p2, p3), axis=-1)), axis=-2)
    return g, h, gii, hii, 2 * jnp.concatenate((m0qi, m1qi), axis=-1)


def assemble_ordered_shared_pole_pencil(states, infinity, *, matmul):
    """Assemble G=X^H s3 X, H=X^H M X and O=C X for time-reversal-broken data.

    ``states`` holds ``(z,Q,WQ,dW/dz Q)`` with complex z in Ry, not s=z**2,
    in the paired layout of ``select_round_states(ordered=True)``: every state
    X(z) followed, after all originals, by its mirror X(-z) on the same
    directions. The resolvent-identity column of ``finite_pencil_column`` is
    exact for the linear pencil (z s3 - M) as written. ``infinity`` is None for
    a finite-state bank, else the five panels of
    ``ordered_infinity_pencil_column``. Returns Hermitian G, H, O and the
    finite nodes z [b,R_finite] that the paired reduction needs.
    """
    q = jnp.concatenate([state[1] for state in states], axis=-1)
    output = jnp.concatenate([state[2] for state in states], axis=-1)
    derivative = jnp.concatenate([state[3] for state in states], axis=-1)
    z = jnp.concatenate([jnp.broadcast_to(jnp.asarray(state[0], jnp.complex128),
                         (state[1].shape[0], state[1].shape[-1]))
                         for state in states], axis=-1)
    finite = (z, q, output)
    g, h = finite_pencil_column(finite, (z, q, output, derivative), matmul=matmul)
    del derivative
    # Pinned to the face: eager concatenation with the adjoint (y/x) infinity rows and
    # eager a + a^H return a replicated [b,side,side] G and H resident on every rank.
    if infinity is not None:
        gi, hi, gii, hii, oi = ordered_infinity_pencil_column(finite, infinity, matmul=matmul)
        g, h = hermitian_block(g, gi, gii), hermitian_block(h, hi, hii)
        del gi, hi, gii, hii
        output = join_columns(output, oi)
    return hermitian_part(g), hermitian_part(h), output, z

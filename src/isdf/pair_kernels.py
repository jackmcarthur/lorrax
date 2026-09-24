"""Pair-projector GEMM of the μ-batch ζ fit; see docs/architecture/zeta_fit_mubatch.md.

For one μ batch ``B`` and one r sub-block (rank-local, no collective)

    D^X_{k̄,cd}(μ, r) = Σ_n w^X_n ψ_{n k̄ c}(r_μ) ψ*_{n k̄ d}(r)        (X = L, R)

is evaluated as ONE complex GEMM per band chunk,

    conj D^{L|R}_{k̄}[(c, X, μ), (d, r)] = Σ_n conj(w^X_n X_B[k̄, n, c, μ]) · ψ[k̄, n, d, r],

with the conjugate on the small centroid operand ``X_B``, the L and R
weights stacked along the GEMM's M axis so both share one read of ψ, and
ψ k-leading ``(n_parent, nb, ns, r_s)`` so the batch (k̄) and contraction
(n) axes need no operand transpose.  Band chunks accumulate in one
``lax.scan``; the conjugate of the result and the L/R split are one
elementwise pass after the last chunk.  The consumer is
:func:`isdf.core.parent_projector_kconv` (native ``conv_kparent`` arm on
CUDA; see :func:`ffi.fft.conv_kpair_plan` for the arm rule).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def pair_projectors_lr(x_b, psi_chunk, w_l, w_r):
    """D^L, D^R for one μ batch on one r sub-block, band chunks in one scan.

    Manual mode: call inside the caller's ``shard_map`` (or on local
    arrays); every operand is rank-local and nothing is communicated.

    Parameters
    ----------
    x_b : (n_bc, n_parent, bc_w, ns, b) complex128
        ``X_B = ψ_{n k̄ s}(r_μ)`` of the batch centroids, per band chunk, in
        the plan's raw-parent k̄ order.  Pad bands and pad centroid slots
        are zero (or carry zero weight).
    psi_chunk : callable ``bc -> (n_parent, bc_w, ns, r_s) complex128``
        ψ of band chunk ``bc`` (a traced int32) on the sub-block's r slots,
        k-leading and contiguous.  A resident source passes
        ``lambda bc: psi[bc]``; the plane route passes its regenerator.
    w_l, w_r : (n_bc, bc_w) float64
        Band weights of the two projectors; zero on pad band slots.

    Returns
    -------
    D_l, D_r : (n_parent, ns, b, ns, r_s) complex128
        The operands of :func:`isdf.core.parent_projector_kconv`.

    Transient bytes: the stacked accumulator ``2·n_parent·ns²·b·r_s·16``
    plus the returned pair of the same size, live together only in the
    final conjugate/split pass (ψ is dead by then).
    """
    n_bc = int(x_b.shape[0])

    def lhs(bc):
        # (n_parent, bc_w, ns, 2, b): conj on the small operand, L|R stacked.
        x = x_b[bc]
        return jnp.conj(jnp.stack(
            [x * w_l[bc][None, :, None, None], x * w_r[bc][None, :, None, None]],
            axis=3))

    def gemm(bc):
        # ponytail: one GEMM form for every shape (no per-shape arm choice).
        return jnp.einsum('knaxm,kndr->kaxmdr', lhs(bc), psi_chunk(bc))

    acc = gemm(0)
    if n_bc > 1:
        acc, _ = jax.lax.scan(lambda a, bc: (a + gemm(bc), None), acc,
                              jnp.arange(1, n_bc, dtype=jnp.int32), unroll=1)
    # ponytail: the conj/split is one extra pass over D (~10% of the GEMM at
    # VI3); removing it needs a conv_kparent that reads conj(D), a C++ change.
    D = jnp.conj(acc)
    return D[:, :, 0], D[:, :, 1]

"""Pair-projector GEMM of the μ-batch ζ fit; see docs/architecture/zeta_fit_mubatch.md.

For one μ batch ``B`` on one rank-local column block (route G: the rank's
G slice of the ψ sphere), with no collective,

    D^X_{k̄,cd}(μ, j) = Σ_n w^X_n ψ_{n k̄ c}(r_μ) ψ̄_{n k̄ d}(j)        (X = L, R)

where ``ψ̄ = conj ψ`` is the column operand AS STORED: the fit conjugates
its resident ψ(G) slice once, so no batch pays a conjugate pass over the
large operand.  Each side is one batched complex GEMM over the raw parents
k̄ (batch k̄, M = ns·b, K = bands, N = ns·cols) on a k-leading contiguous
ψ̄ ``(n_parent, nb, ns, cols)``, which needs no operand transpose.  The
consumer is route G's k-convolution (:func:`ffi.fft.make_fused_conv_kplane`).
"""
from __future__ import annotations

import jax.numpy as jnp


def pair_projectors_lr(x_b, psi_bar, w_l, w_r):
    """D^L, D^R of one μ batch on one column block.

    Manual mode: call inside the caller's ``shard_map`` (or on local
    arrays); every operand is rank-local.

    Parameters
    ----------
    x_b : (n_parent, nb, ns, b) complex128
        ``X_B = ψ_{n k̄ s}(r_μ)`` of the batch centroids, raw parent k̄
        order.  Pad centroid slots and pad bands are zero.
    psi_bar : (n_parent, nb, ns, cols) complex128
        ``conj ψ`` on this rank's columns, k-leading and contiguous (route G:
        ``conj c_{n k̄ s}(G)`` on the G slice).
    w_l, w_r : (nb,) float64
        Band weights; zero on pad bands.

    Returns
    -------
    D_l, D_r : (n_parent, ns, b, ns, cols) complex128
        The operands of :func:`ffi.fft.make_fused_conv_kplane`.
        Bytes ``2·n_parent·ns²·b·cols·16``.
    """
    # ponytail: one form for every shape -- a GEMM per side on the stored
    # conj(psi); the stacked L|R GEMM ran no faster and its split pass cost more.
    x = jnp.transpose(x_b, (0, 2, 3, 1))                          # (k, a, m, n)
    return (jnp.einsum('kamn,kndr->kamdr', x * w_l, psi_bar),
            jnp.einsum('kamn,kndr->kamdr', x * w_r, psi_bar))

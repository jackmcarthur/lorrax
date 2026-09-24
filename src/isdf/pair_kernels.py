"""Pair-projector GEMM of the μ-batch ζ fit; see docs/architecture/zeta_fit_mubatch.md.

For one μ batch ``B`` on one rank-local column block (route G: the rank's
G slice of the ψ sphere; r route: an r sub-block), with no collective,

    D^X_{k̄,cd}(μ, j) = Σ_n w^X_n ψ_{n k̄ c}(r_μ) ψ̄_{n k̄ d}(j)        (X = L, R)

where ``ψ̄ = conj ψ`` is the column operand AS STORED: the fit conjugates
its resident ψ(G) slice once, so no batch pays a conjugate pass over the
large operand.  Each side is one batched complex GEMM over the raw parents
k̄ (batch k̄, M = ns·b, K = bands, N = ns·cols) on a k-leading contiguous
ψ̄ ``(n_parent, nb, ns, cols)``, which needs no operand transpose.  Band
chunks accumulate in one ``lax.scan``.  The consumer is
:func:`isdf.core.parent_projector_kconv`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def pair_projectors_lr(x_b, psi_bar_chunk, w_l, w_r):
    """D^L, D^R of one μ batch on one column block; band chunks in one scan.

    Manual mode: call inside the caller's ``shard_map`` (or on local
    arrays); every operand is rank-local.

    Parameters
    ----------
    x_b : (n_bc, n_parent, bc_w, ns, b) complex128
        ``X_B = ψ_{n k̄ s}(r_μ)`` of the batch centroids per band chunk, raw
        parent k̄ order.  Pad centroid slots and pad bands are zero.
    psi_bar_chunk : callable ``bc -> (n_parent, bc_w, ns, cols) complex128``
        ``conj ψ`` of band chunk ``bc`` (a traced int32) on this rank's
        columns, k-leading and contiguous: ``lambda bc: psi_bar[bc]`` for a
        resident store (route G: ``conj c_{n k̄ s}(G)`` on the G slice).
    w_l, w_r : (n_bc, bc_w) float64
        Band weights; zero on pad band slots.

    Returns
    -------
    D_l, D_r : (n_parent, ns, b, ns, cols) complex128
        The operands of :func:`isdf.core.parent_projector_kconv`.
        Bytes ``2·n_parent·ns²·b·cols·16`` (plus one chunk's pair while a
        scan adds it).
    """
    def gemm(bc):
        # ponytail: one form for every shape -- a GEMM per side on the stored
        # conj(psi); the stacked L|R GEMM ran no faster and its split pass cost more.
        x = jnp.transpose(x_b[bc], (0, 2, 3, 1))              # (k, a, m, n): small
        y = psi_bar_chunk(bc)
        return (jnp.einsum('kamn,kndr->kamdr', x * w_l[bc], y),
                jnp.einsum('kamn,kndr->kamdr', x * w_r[bc], y))

    acc = gemm(0)
    if int(x_b.shape[0]) > 1:
        # ponytail: plain add of each chunk (no in-place GEMM accumulate); the
        # resident route-G slice is one chunk, so the scan runs only for streamed bands.
        acc, _ = jax.lax.scan(
            lambda a, bc: (jax.tree.map(jnp.add, a, gemm(bc)), None), acc,
            jnp.arange(1, int(x_b.shape[0]), dtype=jnp.int32), unroll=1)
    return acc

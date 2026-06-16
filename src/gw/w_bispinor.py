"""Bispinor screened-W via the channel-blocked Dyson supermatrix.

    W^{μ_L,ν_L}(q) = [ (δ − V·χ)^{-1} · V ]^{μ_L,ν_L}

The compound vertex index is (Lorentz channel μ_L ∈ {0,1,2,3}) ⊗ (centroid r_μ);
V and χ are 4×4-Lorentz-block supermatrices over it, size ``N = n_C + 3·n_T``
(charge centroids n_C, three transverse channels sharing n_T each), laid out as

    [ charge n_C | T1 n_T | T2 n_T | T3 n_T ].

The inversion is :func:`gw.w_isdf.solve_w` *verbatim*: once V and χ are assembled
as ``(nq, N, N)`` the bispinor solve is the existing scalar per-q LU at a larger
matrix, inheriting its sharding, solver dispatch, and χ-prefactor.  Only the block
assembly and W-tile extraction are bispinor-specific.

Milestone A (charge screening + bare Breit) calls :func:`solve_w_bispinor` with χ
populated only in the (0,0) charge block.  V·χ then has only its (0,0) block
nonzero, so the result collapses to W⁰⁰=(I−V⁰⁰χ⁰⁰)^{-1}V⁰⁰ (scalar screened W),
W^{ij}=V^{ij} (bare Breit), W^{0i}=W^{i0}=0.  Milestone B fills the zero χ-tiles
with the un-traced channel χ (W^{0i} then ≠ 0); assembly/solve/extract unchanged.
Gauge: V^{0i}=V^{i0}=0 (``v_q_bispinor.ZERO_TILES``), so those V blocks stay zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def assemble_supermatrix(blocks: dict, n_C: int, n_T: int) -> jax.Array:
    """Assemble a ``(nq, N, N)`` Lorentz supermatrix from per-channel blocks.

    ``blocks`` maps ``(μ_L, ν_L) → (nq, sz[μ_L], sz[ν_L])``; missing/None blocks
    fill with zeros (gauge-zero V tiles; every non-(0,0) χ tile in milestone A).
    """
    sizes = [int(n_C), int(n_T), int(n_T), int(n_T)]
    ref = next((b for b in blocks.values() if b is not None), None)
    if ref is None:
        raise ValueError("assemble_supermatrix: no non-None blocks provided")
    nq, dtype = int(ref.shape[0]), ref.dtype
    rows = []
    for mu in range(4):
        cols = [b if (b := blocks.get((mu, nu))) is not None
                else jnp.zeros((nq, sizes[mu], sizes[nu]), dtype)
                for nu in range(4)]
        rows.append(jnp.concatenate(cols, axis=-1))
    return jnp.concatenate(rows, axis=-2)             # (nq, N, N)


def extract_blocks(super_mat: jax.Array, n_C: int, n_T: int) -> dict:
    """Inverse of :func:`assemble_supermatrix`: all 16 Lorentz blocks as
    ``{(μ_L, ν_L): (nq, sz[μ_L], sz[ν_L])}`` (cheap slice views)."""
    sizes = [int(n_C), int(n_T), int(n_T), int(n_T)]
    off = [0, sizes[0], sizes[0] + sizes[1], sizes[0] + sizes[1] + sizes[2]]
    return {(mu, nu): super_mat[:, off[mu]:off[mu] + sizes[mu],
                                off[nu]:off[nu] + sizes[nu]]
            for mu in range(4) for nu in range(4)}


# Transverse χ channels: 6 unique (diagonal + upper) + 3 Hermitian companions.
_TT_UNIQUE = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))
_TT_HERM = {(2, 1): (1, 2), (3, 1): (1, 3), (3, 2): (2, 3)}


def build_chi_blocks(chi00, wfns_charge, wfns_transverse, quad, e_ref,
                     meta, mesh_xy, *, milestone_a: bool = False) -> dict:
    """Channel-resolved χ tiles for the screened bispinor W (milestone B).

    Charge χ⁰⁰ is the scalar polarizability (passed in).  The 6 unique transverse
    χ^{ij} fold γ̃^i/γ̃^j into the conduction ψ on the transverse bundle (square
    n_T×n_T); the 3 charge–current cross χ^{0i} mix bundles — charge m-vertex (xn),
    transverse n-vertex (yr) — giving rectangular n_C×n_T blocks.  The 3+3 Hermitian
    companions (χ^{ji}, χ^{i0}) are conj-swapaxes fills (χ^{νμ}=conj(χ^{μν})ᵀ, exact
    for the real static Laplace quad), so the V·χ supermatrix is Hermitian.

    ``milestone_a=True`` skips the charge–current cross χ^{0i}/χ^{i0} blocks
    (returns CC + TT only).  This is required for the bispinor IBZ→full-BZ
    cascade: the cross-block IBZ unfold (rectangular two-centroid-set permute
    + single-R + net-TRS(−1)) is underived (``unfold_bispinor_tiles`` handles
    only CC + TT).  W^{0i} is then identically zero, matching the Milestone-A
    physics (charge screening + bare Breit).
    """
    from .w_isdf import compute_chi0
    chi = {(0, 0): chi00}
    for (i, j) in _TT_UNIQUE:
        chi[(i, j)] = compute_chi0(wfns_transverse, quad, meta, mesh_xy,
                                   energy_reference=e_ref, mu_L=i, nu_L=j)
    for (j, i), src in _TT_HERM.items():
        chi[(j, i)] = jnp.conj(jnp.swapaxes(chi[src], -1, -2))
    if not milestone_a:
        for i in (1, 2, 3):
            chi[(0, i)] = compute_chi0(wfns_charge, quad, meta, mesh_xy,
                                       energy_reference=e_ref, mu_L=0, nu_L=i,
                                       wfns_n=wfns_transverse)
            chi[(i, 0)] = jnp.conj(jnp.swapaxes(chi[(0, i)], -1, -2))
    return chi


def solve_w_bispinor(v_blocks: dict, chi_blocks: dict, meta, mesh_xy,
                     *, solver=None) -> dict:
    """Screened bispinor W tiles via the channel-blocked Dyson supermatrix.

    Assembles V and χ as ``(nq, N, N)`` (missing blocks → zero), runs
    :func:`gw.w_isdf.solve_w`, returns all 16 W blocks.  Milestone A: pass
    ``chi_blocks={(0,0): chi00}`` only.
    """
    from .w_isdf import solve_w
    n_C, n_T = v_blocks[(0, 0)].shape[-1], v_blocks[(1, 1)].shape[-1]
    v_super = assemble_supermatrix(v_blocks, n_C, n_T)
    chi_super = assemble_supermatrix(chi_blocks, n_C, n_T)
    return extract_blocks(solve_w(v_super, chi_super, meta, mesh_xy, solver=solver),
                          n_C, n_T)

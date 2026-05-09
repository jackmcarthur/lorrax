"""Bispinor bare exchange self-energy: Σ^B from the transverse V_q^{i,j} tiles.

Per the Phase-1 DHF + bare-Breit design (``BISPINOR_DHFB_DESIGN.md`` §3),
the bispinor bare exchange splits into two pieces:

* **Σ^C-bare** — charge channel, uses ``V_qmunu_CC`` (== V_00).  This is
  identical to today's scalar ``compute_cohsex_sigma(..., compute_bare_x=
  True)``; the bispinor pipeline doesn't change anything here, just runs
  on 4-spinor wavefunctions instead of 2-spinor.

* **Σ^B** — transverse-only, summed over (i, j) ∈ {1, 2, 3}²:

    Σ^B_{αβ}(12) = -Σ_{i,j} γ̃^i_{αγ} G^0_{γδ}(12) γ̃^j_{δβ} D^{ij}_bare(12)

  where γ̃^μ ≡ γ^0 γ^μ (so γ̃^0 = I_4 and γ̃^i = α^i in the LORRAX
  convention), and D^{ij}_bare = V_qmunu_TT_{ij} (the transverse
  projector-weighted Coulomb).

This module computes Σ^B by reusing the existing scalar
``sigma_sx`` kernel.  The trick:

* **γ̃^i insertion at the LEFT vertex** = left-multiply ``psi_xn`` (the
  "internal-band" wavefunction) on its spin axis by γ̃^i.
* **γ̃^j insertion at the RIGHT vertex** = left-multiply ``psi_yr`` on
  its spin axis by γ̃^j.

After those two ψ rewrites, the existing kernel chain
``build_G → _convolve → project`` evaluates the formula above
verbatim — no changes to the kernel itself.

The 9 (i, j) tiles in the sum decompose as 6 unique kernel calls
(3 diagonal + 3 upper-triangular) + 3 Hermitian-conjugate fills
delivered for free by :class:`gw.v_q_bispinor.BispinorVqReader`.

Public surface
--------------
* :func:`compute_sigma_x_bispinor` — orchestrator that opens
  ``v_q_bispinor.h5`` via ``BispinorVqReader``, loops the 9 transverse
  (i, j) pairs, and returns the summed Σ^B as a (nk, nb_sigma,
  nb_sigma) array on the same sharding as today's scalar Σ_X.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .v_q_bispinor import BispinorVqReader


# Lorentz indices that contribute to Σ^B (transverse only).  The 6
# (0, i) / (i, 0) tiles vanish by Coulomb gauge and are skipped; the
# (0, 0) bare-CC tile is handled by the existing scalar
# compute_cohsex_sigma path.
_TRANSVERSE_INDICES = (1, 2, 3)


def _apply_gamma_left_to_xn(gamma: jax.Array, psi_xn: jax.Array) -> jax.Array:
    """Left-multiply ``γ`` on the spin axis of ``psi_xn``.

    psi_xn shape (nk, s, μ_X, n) with ns = 4 (bispinor).
    Output ``γ ψ`` keeps the same layout; spin axis 1 is rewritten.

        out[k, β, x, n] = Σ_s γ[β, s] · ψ_xn[k, s, x, n]
    """
    return jnp.einsum('bs,ksxn->kbxn', gamma, psi_xn, optimize=True)


def _apply_gamma_left_to_yr(gamma: jax.Array, psi_yr: jax.Array) -> jax.Array:
    """Left-multiply ``γ`` on the spin axis of ``psi_yr``.

    psi_yr shape (nk, n, s, μ_Y) with ns = 4 (bispinor).
    Spin axis is axis 2; the einsum below rewrites it.

        out[k, n, β, μ] = Σ_s γ[β, s] · ψ_yr[k, n, s, μ]

    Note: ``build_G`` internally applies ``jnp.conj(psi_yr)`` before
    contracting on the spin axis ``t``.  Folding γ̃^j (Hermitian) into
    psi_yr here yields ``conj(γ̃^j ψ) = γ̃^j^T ψ*`` which, when
    contracted with ``psi_yn[k, t, μ_Y, n]`` on axis t, gives the
    correct ``ψ†_{yr} γ̃^j ψ_{yn}`` Hermitian sandwich.  γ̃^j Hermitian
    is essential for this identity; α^i and I_4 both qualify.
    """
    return jnp.einsum('bs,knsx->knbx', gamma, psi_yr, optimize=True)


def _wfns_with_lorentz_vertices(wfns, mu_L: int, nu_L: int):
    """Return a Wavefunctions clone with γ̃^{μ_L} folded into psi_xn
    (left vertex) and γ̃^{ν_L} folded into psi_yr (right vertex).

    For (μ_L, ν_L) = (0, 0) this is a trivial copy (γ̃^0 = I_4).

    psi_xr / psi_yn are passed through unchanged; the vertex sits on
    the build_G side of the kernel chain and the project side
    contracts the OUTPUT bands without further γ̃ insertions (the
    internal-band γ̃ matrices are absorbed into G already).
    """
    from common.isdf_fitting import _gamma_tilde_matrix
    gamma_mu = _gamma_tilde_matrix(mu_L)
    gamma_nu = _gamma_tilde_matrix(nu_L)

    psi_xn_new = (wfns.psi_xn if mu_L == 0
                  else _apply_gamma_left_to_xn(gamma_mu, wfns.psi_xn))
    psi_yr_new = (wfns.psi_yr if nu_L == 0
                  else _apply_gamma_left_to_yr(gamma_nu, wfns.psi_yr))

    return dataclasses.replace(wfns, psi_xn=psi_xn_new, psi_yr=psi_yr_new)


def compute_sigma_x_bispinor(
    *,
    wfns_transverse,
    Gij: jax.Array,
    bispinor_v_q_path: Path | str,
    meta,
    mesh_xy: Mesh,
    backend=None,
    use_ffi_io: bool | None = None,
    print_fn=print,
    verbose: bool = True,
) -> jax.Array:
    """Compute Σ^B (transverse-only bispinor bare exchange).

    Sums

        Σ^B[k, m, n] = -Σ_{i, j ∈ {1, 2, 3}} ⟨m, k| γ̃^i V^{i,j}_{q=k-k'}
                         γ̃^j |n, k⟩ ⟨m', k'| ⟨m, k|

    over the 9 transverse Lorentz pairs by reusing the existing
    ``sigma_sx`` kernel (with γ̃ folded into ψ; see module docstring).

    Parameters
    ----------
    wfns_transverse
        :class:`gw.wavefunction_bundle.Wavefunctions` sampled at the
        **transverse** centroid set ``r_{μ_T}``.  All 9 transverse
        tiles share these centroids.  See :meth:`__init__` of the
        next agent's bispinor wfns plumbing for how this is built —
        load_wfns currently only emits one wfns bundle per centroid
        set, so ``prepare_isdf_and_wavefunctions`` needs a small
        extension to emit a second bundle on ``cents_curr_idx`` when
        ``cfg.bispinor`` and ``cfg.paths.centroids_file_current`` are
        set.

    Gij
        Band-space occupation projector (nk, nb_sigma, nb_sigma) —
        same operator the scalar Σ_X uses.

    bispinor_v_q_path
        Path to ``v_q_bispinor.h5`` written by
        :func:`gw.v_q_bispinor.compute_V_q_bispinor_to_h5`.

    meta, mesh_xy, backend, use_ffi_io
        Standard plumbing.

    Returns
    -------
    jax.Array
        Σ^B[k, m, n] on the same sharding as the scalar Σ_X.
    """
    from .cohsex_sigma import _make_cohsex_kernels
    nk_tot = int(meta.nk_tot)
    sigma_sx_k, _, _ = _make_cohsex_kernels(mesh_xy, meta.kgrid, nk_tot)

    sig_x_b = None
    with BispinorVqReader(bispinor_v_q_path, mesh_xy,
                          backend=backend, use_ffi_io=use_ffi_io) as reader:
        for i in _TRANSVERSE_INDICES:
            for j in _TRANSVERSE_INDICES:
                if verbose and jax.process_index() == 0:
                    print_fn(f"  Σ^B tile (μ_L={i}, ν_L={j})")
                wfns_ij = _wfns_with_lorentz_vertices(wfns_transverse, i, j)
                V_ij = reader.get_tile(i, j)
                contrib = sigma_sx_k(wfns_ij, Gij, V_ij)
                sig_x_b = contrib if sig_x_b is None else sig_x_b + contrib

    if sig_x_b is None:  # pragma: no cover — should never happen
        raise RuntimeError("compute_sigma_x_bispinor produced no contributions")
    sig_x_b.block_until_ready()
    return sig_x_b

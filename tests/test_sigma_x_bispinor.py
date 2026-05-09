"""Unit tests for ``gw.sigma_x_bispinor``.

CPU-only (no GPU required).  These tests exercise the γ̃-vertex
algebra at the per-tile level on small synthetic wavefunctions; the
end-to-end smoke (real V_q tiles + transverse centroids) belongs in
a later integration test once the wfns_transverse plumbing lands.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

import jax
import jax.numpy as jnp


def test_gamma0_is_identity_so_left_fold_is_noop():
    """γ̃^0 = I_4 (per the LORRAX storage convention).  Folding γ̃^0
    into psi_xn must return the input array unchanged."""
    from src.gw.sigma_x_bispinor import _apply_gamma_left_to_xn
    from src.common.isdf_fitting import _gamma_tilde_matrix

    gamma0 = _gamma_tilde_matrix(0)
    np.testing.assert_array_equal(np.asarray(gamma0), np.eye(4))

    rng = np.random.default_rng(0)
    psi_xn = jnp.asarray(
        (rng.standard_normal((2, 4, 5, 3)) + 1j * rng.standard_normal((2, 4, 5, 3)))
        .astype(np.complex128))
    out = _apply_gamma_left_to_xn(gamma0, psi_xn)
    np.testing.assert_allclose(np.asarray(out), np.asarray(psi_xn), atol=1e-15)


def test_gamma_left_fold_is_matmul_on_spin_axis():
    """``_apply_gamma_left_to_xn(γ, ψ)[k, β, μ, n]`` must equal
    ``Σ_α γ[β, α] ψ[k, α, μ, n]``."""
    from src.gw.sigma_x_bispinor import _apply_gamma_left_to_xn

    rng = np.random.default_rng(1)
    gamma = jnp.asarray((rng.standard_normal((4, 4))
                         + 1j * rng.standard_normal((4, 4))).astype(np.complex128))
    psi = jnp.asarray((rng.standard_normal((2, 4, 3, 7))
                       + 1j * rng.standard_normal((2, 4, 3, 7))).astype(np.complex128))

    out = _apply_gamma_left_to_xn(gamma, psi)
    # Reference: matmul on the spin axis (axis 1 of psi).
    ref = np.einsum('bs,ksxn->kbxn', np.asarray(gamma), np.asarray(psi))
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-14)


def test_gamma_yr_fold_is_matmul_on_spin_axis():
    """psi_yr has shape (nk, n, s, μ_Y); γ̃ folds via axis 2."""
    from src.gw.sigma_x_bispinor import _apply_gamma_left_to_yr

    rng = np.random.default_rng(2)
    gamma = jnp.asarray((rng.standard_normal((4, 4))
                         + 1j * rng.standard_normal((4, 4))).astype(np.complex128))
    psi = jnp.asarray((rng.standard_normal((2, 7, 4, 3))
                       + 1j * rng.standard_normal((2, 7, 4, 3))).astype(np.complex128))

    out = _apply_gamma_left_to_yr(gamma, psi)
    ref = np.einsum('bs,knsx->knbx', np.asarray(gamma), np.asarray(psi))
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-14)


def test_wfns_replace_no_op_for_00():
    """For (μ_L, ν_L) = (0, 0), ``_wfns_with_lorentz_vertices`` must
    return a Wavefunctions whose psi_xn / psi_yr are the *same*
    arrays as the input (bit-identical, not just equal-valued).

    This is the bispinor → scalar reduction safeguard: a downstream
    σ_X^B at (0,0) is byte-equivalent to today's scalar Σ_X.
    """
    from src.gw.sigma_x_bispinor import _wfns_with_lorentz_vertices

    rng = np.random.default_rng(3)
    nk, ns, mu_x, nb, mu_y = 2, 4, 5, 3, 6

    @dataclass_with_replace_fixture()
    class WfnsLike:
        psi_xn: jax.Array
        psi_xr: jax.Array
        psi_yr: jax.Array
        psi_yn: jax.Array

    wfns = WfnsLike(
        psi_xn=jnp.asarray(rng.standard_normal((nk, ns, mu_x, nb)).astype(np.complex128)),
        psi_xr=jnp.asarray(rng.standard_normal((nk, nb, ns, mu_x)).astype(np.complex128)),
        psi_yr=jnp.asarray(rng.standard_normal((nk, nb, ns, mu_y)).astype(np.complex128)),
        psi_yn=jnp.asarray(rng.standard_normal((nk, ns, mu_y, nb)).astype(np.complex128)),
    )

    out = _wfns_with_lorentz_vertices(wfns, 0, 0)
    # No-op short-circuit: the same array objects come back.
    assert out.psi_xn is wfns.psi_xn
    assert out.psi_yr is wfns.psi_yr
    # The other two are passed through dataclasses.replace so they're
    # still the same array objects too.
    assert out.psi_xr is wfns.psi_xr
    assert out.psi_yn is wfns.psi_yn


def dataclass_with_replace_fixture():
    """Minimal ``dataclass`` decorator that supports ``dataclasses.replace``.

    The real ``Wavefunctions`` class has many fields we don't need
    here; this fixture builds the minimal surface so we can exercise
    ``_wfns_with_lorentz_vertices``.
    """
    from dataclasses import dataclass
    return dataclass


def test_unique_transverse_pairs_correct():
    """Sanity: σ^B sums over (i, j) ∈ {1, 2, 3}² → 9 pairs total.
    The orchestrator iterates exactly that range."""
    from src.gw.sigma_x_bispinor import _TRANSVERSE_INDICES
    pairs = [(i, j) for i in _TRANSVERSE_INDICES for j in _TRANSVERSE_INDICES]
    assert len(pairs) == 9
    assert (0, 0) not in pairs       # CC handled separately by Σ^C
    for (i, j) in pairs:
        assert i in (1, 2, 3) and j in (1, 2, 3)

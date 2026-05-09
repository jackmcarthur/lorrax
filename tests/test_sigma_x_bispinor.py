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


def test_gamma0_perm_phase_is_identity():
    """γ̃^0 = I_4: perm = (0,1,2,3), phase = (1,1,1,1).  Applying it
    via gamma_apply must leave any tensor unchanged on its spin axis."""
    from src.common.gamma_matrices import gamma_perm_phase, gamma_apply

    perm, phase = gamma_perm_phase(0)
    np.testing.assert_array_equal(np.asarray(perm), np.arange(4))
    np.testing.assert_array_equal(np.asarray(phase), np.ones(4, dtype=np.complex128))

    rng = np.random.default_rng(0)
    psi_xn = jnp.asarray(
        (rng.standard_normal((2, 4, 5, 3)) + 1j * rng.standard_normal((2, 4, 5, 3)))
        .astype(np.complex128))
    out = gamma_apply(psi_xn, perm, phase, axis=1)
    np.testing.assert_allclose(np.asarray(out), np.asarray(psi_xn), atol=1e-15)


def test_gamma_apply_matches_dense_matmul_xn_axis():
    """``gamma_apply(ψ, perm, phase, axis=1)`` must equal
    ``Σ_α γ̃[β, α] ψ[k, α, μ, n]`` for every γ̃^μ ∈ {γ̃^0..3}."""
    from src.common.gamma_matrices import (
        gamma_perm_phase, gamma_apply, gamma0, gamma1, gamma2, gamma3,
    )

    rng = np.random.default_rng(1)
    psi = jnp.asarray((rng.standard_normal((2, 4, 3, 7))
                       + 1j * rng.standard_normal((2, 4, 3, 7))).astype(np.complex128))
    for mu, gamma_dense in enumerate([gamma0, gamma1, gamma2, gamma3]):
        perm, phase = gamma_perm_phase(mu)
        out = gamma_apply(psi, perm, phase, axis=1)
        ref = np.einsum('bs,ksxn->kbxn', np.asarray(gamma_dense), np.asarray(psi))
        np.testing.assert_allclose(np.asarray(out), ref, atol=1e-14,
                                   err_msg=f"γ̃^{mu} mismatch on psi_xn axis")


def test_gamma_apply_matches_dense_matmul_yr_axis():
    """psi_yr has shape (nk, n, s, μ_Y); γ̃ applies on axis 2."""
    from src.common.gamma_matrices import (
        gamma_perm_phase, gamma_apply, gamma0, gamma1, gamma2, gamma3,
    )

    rng = np.random.default_rng(2)
    psi = jnp.asarray((rng.standard_normal((2, 7, 4, 3))
                       + 1j * rng.standard_normal((2, 7, 4, 3))).astype(np.complex128))
    for mu, gamma_dense in enumerate([gamma0, gamma1, gamma2, gamma3]):
        perm, phase = gamma_perm_phase(mu)
        out = gamma_apply(psi, perm, phase, axis=2)
        ref = np.einsum('bs,knsx->knbx', np.asarray(gamma_dense), np.asarray(psi))
        np.testing.assert_allclose(np.asarray(out), ref, atol=1e-14,
                                   err_msg=f"γ̃^{mu} mismatch on psi_yr axis")


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

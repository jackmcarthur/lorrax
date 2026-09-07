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

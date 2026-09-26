"""CPU gate: the recycled spectral deflation of the shifted GMRES engine.

``bse_feast.harvest_spectral_deflation`` finds the Ritz vectors of the
diagonally preconditioned operator ``A = (z - H) diag(z - D)^{-1}`` that sit
far from 1, and ``apply_spectral_deflation`` maps them to 1 inside the right
preconditioner.  Three properties, on a dense toy operator with a handful of
outlying eigenvalues (the shape the ladder's bound excitons give):

* the deflated solve reaches the SAME solution to the solver tolerance (the
  preconditioner changes the iterates, never the system);
* it takes fewer iterations than the diagonal-only engine;
* ``deflation=None`` is the historical program, bit for bit.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from bse.bse_feast import (_gmres_solve_core, apply_spectral_deflation,
                           harvest_spectral_deflation)

jax.config.update("jax_enable_x64", True)

N = 96


def _toy_operator(seed=0, n_out=6):
    """``H = D + K``: D spread on [1, 3], K a dense perturbation that pushes
    ``n_out`` eigenvalues of ``H D^{-1}`` far below 1."""
    rng = np.random.default_rng(seed)
    d = np.linspace(1.0, 3.0, N)
    K = 0.02 * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    Qo, _ = np.linalg.qr(rng.standard_normal((N, n_out))
                         + 1j * rng.standard_normal((N, n_out)))
    K = K - Qo @ np.diag(np.linspace(0.6, 0.95, n_out)) @ Qo.conj().T * d[:, None]
    H = np.diag(d) + K
    return jnp.asarray(H), jnp.asarray(d.astype(np.complex128))


def _matvec(x, H):
    return jnp.einsum("ij,abj->abi", H, x)


def _solve(H, d, b, deflation):
    diag = jnp.broadcast_to(d, b.shape)
    z = jnp.asarray(0.05j, dtype=jnp.complex128)
    return _gmres_solve_core(_matvec, b, diag, z, (H,), 120, 1e-10,
                             deflation=deflation)


def test_apply_spectral_deflation_maps_invariant_subspace_to_one():
    rng = np.random.default_rng(3)
    A = np.diag(np.linspace(0.2, 2.0, 8)).astype(np.complex128)
    U = np.zeros((2, 8), np.complex128); U[0, 0] = U[1, 1] = 1.0
    T = U.conj() @ A @ U.T
    C = np.linalg.inv(T) - np.eye(2)
    v = jnp.asarray(rng.standard_normal(8) + 0j)
    Pv = apply_spectral_deflation(v, (jnp.asarray(U), jnp.asarray(C)))
    # A P e_0 = e_0: the deflated eigenvalue 0.2 is mapped to 1.
    e0 = jnp.zeros(8, jnp.complex128).at[0].set(1.0)
    APe0 = A @ np.asarray(apply_spectral_deflation(e0, (jnp.asarray(U), jnp.asarray(C))))
    np.testing.assert_allclose(APe0, np.asarray(e0), atol=1e-14)
    # Off the subspace P is the identity.
    np.testing.assert_allclose(np.asarray(Pv)[2:], np.asarray(v)[2:], atol=0)


def test_deflated_gmres_same_solution_fewer_iterations():
    H, d = _toy_operator()
    rng = np.random.default_rng(1)
    b = jnp.asarray((rng.standard_normal((2, 1, N))
                     + 1j * rng.standard_normal((2, 1, N))).astype(np.complex128))
    x0, k0 = _solve(H, d, b, None)
    diag = jnp.broadcast_to(d, b.shape)
    z = jnp.asarray(0.05j, dtype=jnp.complex128)
    defl, ritz = harvest_spectral_deflation(
        _matvec, diag, z, (H,), b, n_arnoldi=24, rank=8)
    assert defl[0].shape == (8,) + b.shape
    x1, k1 = _solve(H, d, b, defl)
    rel = float(jnp.linalg.norm(x1 - x0) / jnp.linalg.norm(x0))
    assert rel < 1e-8, rel
    assert int(k1) < int(k0), (int(k0), int(k1))


def test_deflation_none_is_the_historical_program():
    H, d = _toy_operator(seed=2)
    rng = np.random.default_rng(5)
    b = jnp.asarray((rng.standard_normal((2, 1, N))
                     + 1j * rng.standard_normal((2, 1, N))).astype(np.complex128))
    diag = jnp.broadcast_to(d, b.shape)
    z = jnp.asarray(0.05j, dtype=jnp.complex128)
    xa, ka = _gmres_solve_core(_matvec, b, diag, z, (H,), 120, 1e-10)
    xb, kb = _gmres_solve_core(_matvec, b, diag, z, (H,), 120, 1e-10,
                               deflation=None)
    assert int(ka) == int(kb)
    assert bool(jnp.all(xa == xb))

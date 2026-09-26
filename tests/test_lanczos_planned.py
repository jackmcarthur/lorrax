"""Planned Lanczos parity, partial windows, and early-exit residuals.

These are arithmetic tests of the public numerical entrypoints. Distributed
placement and collective/HLO checks belong to the combined P4 run alongside
the service's poisoned-storage tests.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from distrib_la import plan_subspace
from solvers import lanczos as lz


def _operator(n, seed=29, separated=False):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, _ = np.linalg.qr(raw)
    energies = (np.concatenate([np.arange(1, 5) / 10, np.linspace(8, 9, n - 4)])
                if separated else np.linspace(0.2, 8, n))
    return (q * energies) @ q.conj().T, energies


def _solve(matrix, *, block_size, steps, planned, n_reorth=-1, early=False):
    n, roots = matrix.shape[0], 4
    plan = (plan_subspace(capacity=(steps + 1) * block_size, n_eig=roots)
            if planned else False)

    @jax.jit
    def run(h):
        # The report is data in production. Avoid a host callback in this
        # small arithmetic gate, whose Hermitian input is constructed above.
        with lz.alpha_herm_sink():
            if early:
                return lz.block_lanczos_eig_jit_converged(
                    lambda x: x @ h.T, n, roots, block_size, steps,
                    n_reorth=n_reorth, rtol=1e-11, check_every=2,
                    subspace_plan=plan)
            return lz.block_lanczos_eig_jit(
                lambda x: x @ h.T, n, roots, block_size, steps,
                n_reorth=n_reorth, subspace_plan=plan)

    return jax.device_get(run(jnp.asarray(matrix)))


@pytest.mark.parametrize("block_size", [1, 2])
@pytest.mark.parametrize("n_reorth", [-1, 3])
def test_planned_matches_reference_windows(block_size, n_reorth):
    matrix, _ = _operator(96)
    expected = _solve(matrix, block_size=block_size, steps=24,
                      planned=False, n_reorth=n_reorth)
    actual = _solve(matrix, block_size=block_size, steps=24,
                    planned=True, n_reorth=n_reorth)
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-10, rtol=2e-10)
    # Projectors remove the arbitrary eigenvector phases. A wrong active
    # interval changes the finite-depth eigenspace, even at partial reorth.
    vectors = actual[1].reshape(4, 96)
    np.testing.assert_allclose(vectors.conj().T @ vectors,
                               expected[1].conj().T @ expected[1], atol=2e-8)


def test_planned_early_exit_uses_completed_basis():
    matrix, exact = _operator(160, separated=True)
    expected = _solve(matrix, block_size=2, steps=60, planned=False, early=True)
    actual = _solve(matrix, block_size=2, steps=60, planned=True, early=True)
    values, vectors, steps = actual
    assert int(steps) < 60, "fixture must leave an inactive capacity tail"
    assert int(steps) == int(expected[2])
    np.testing.assert_allclose(values, exact[:4], atol=2e-10)
    residual = vectors @ matrix.T - values[:, None] * vectors
    assert np.linalg.norm(residual, axis=1).max() < 2e-7
    np.testing.assert_allclose(vectors.conj() @ vectors.T, np.eye(4), atol=2e-10)


def test_plan_geometry_is_checked_before_iteration():
    plan = plan_subspace(capacity=30, n_eig=4)
    with pytest.raises(ValueError, match="capacity=26"):
        lz.block_lanczos_eig_jit(lambda x: x, 64, 4, 2, 12,
                                 subspace_plan=plan)


def test_projected_builder_ignores_unwritten_blocks():
    rng = np.random.default_rng(91)
    alpha = np.full((12, 2, 2), np.nan + 1j * np.nan)
    beta = np.full_like(alpha, np.nan + 1j * np.nan)
    alpha[:3] = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    beta[:2] = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    expected = np.zeros((26, 26), np.complex128)
    for i in range(3):
        s = 2 * i
        expected[s:s + 2, s:s + 2] = (alpha[i] + alpha[i].conj().T) / 2
        if i < 2:
            expected[s + 2:s + 4, s:s + 2] = beta[i]
            expected[s:s + 2, s + 2:s + 4] = beta[i].conj().T
    build = jax.jit(lambda a, b, count: lz._build_block_tridiag(
        a, b, 12, 2, capacity=26, active_blocks=count))
    actual = build(jnp.asarray(alpha), jnp.asarray(beta), jnp.int32(3))
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_thick_restart_planned_preserves_arrowhead():
    from solvers.thick_restart_lanczos import thick_restart_lanczos_eig
    matrix, exact = _operator(96)

    def run(planned, drop=False):
        plan = plan_subspace(capacity=33, n_eig=12) if planned else False
        solve = jax.jit(lambda h: thick_restart_lanczos_eig(
            lambda x: x @ h.T, (96,), n_eig=4, n_keep=12, m_max=32,
            n_restarts=5, subspace_plan=plan, _drop_arrowhead=drop))
        return jax.device_get(solve(jnp.asarray(matrix)))

    expected = run(False)
    values, vectors, alpha_defect = run(True)
    np.testing.assert_allclose(values, expected[0], atol=2e-10)
    np.testing.assert_allclose(values, exact[:4], atol=2e-9)
    assert np.linalg.norm(vectors @ matrix.T - values[:, None] * vectors,
                          axis=1).max() < 2e-7
    np.testing.assert_allclose(vectors.conj() @ vectors.T, np.eye(4), atol=2e-10)
    assert float(alpha_defect) < 1e-12
    broken = run(True, drop=True)
    broken_residual = np.linalg.norm(
        broken[1] @ matrix.T - broken[0][:, None] * broken[1], axis=1).max()
    assert broken_residual > 1e-5, "fixture must detect lost restart coupling"


def test_block_basis_retains_structured_vector_axes():
    matrix, _ = _operator(96)
    plan = plan_subspace(capacity=50, n_eig=4)

    @jax.jit
    def run(h):
        with lz.alpha_herm_sink():
            return lz.block_lanczos_eig_jit(
                lambda x: (x.reshape(2, 96) @ h.T).reshape(x.shape), 96, 4, 2, 24,
                subspace_plan=plan, vector_shape=(8, 12), structured_vectors=True)

    expected = _solve(matrix, block_size=2, steps=24, planned=True)
    actual = jax.device_get(run(jnp.asarray(matrix)))
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-10)
    vectors = actual[1].reshape(4, 96)
    np.testing.assert_allclose(vectors.conj().T @ vectors,
                               expected[1].conj().T @ expected[1], atol=2e-9)

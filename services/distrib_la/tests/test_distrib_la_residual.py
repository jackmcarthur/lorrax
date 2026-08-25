"""Direct definition checks for distributed solve backward error."""
from __future__ import annotations

import numpy as np
from lxkit.testing import require_devices


def _mesh():
    import jax
    from jax.sharding import Mesh

    require_devices(4, "cpu")
    return Mesh(
        np.asarray(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))


def _put(a, mesh):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    spec = P(*((None,) * (a.ndim - 2)), "x", "y")
    return jax.device_put(a, NamedSharding(mesh, spec))


def test_zero_denominator_is_zero_only_for_an_exact_zero_residual():
    import jax.numpy as jnp
    from distrib_la.residual import _backward_error_ratio

    got = np.asarray(_backward_error_ratio(
        jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 0.0])))
    assert got[0] == 0.0
    assert np.isinf(got[1])


def test_distributed_value_matches_the_direct_frobenius_formula():
    import distrib_la as D

    mesh = _mesh()
    rng = np.random.default_rng(20260825)
    batch, n, nrhs = (2, 3), 8, 4
    A = (rng.standard_normal(batch + (n, n))
         + 1j * rng.standard_normal(batch + (n, n)))
    X = (rng.standard_normal(batch + (n, nrhs))
         + 1j * rng.standard_normal(batch + (n, nrhs)))
    # Keep the residual well above GEMM roundoff: this cell checks the
    # definition and prefactor, not bit identity between NumPy BLAS and the
    # distributed JAX contraction.
    perturbation = 1.0e-3 * (
        rng.standard_normal(batch + (n, nrhs))
        + 1j * rng.standard_normal(batch + (n, nrhs)))
    B = A @ X + perturbation
    residual = A @ X - B
    want = (
        np.linalg.norm(residual, axis=(-2, -1))
        / (np.linalg.norm(A, axis=(-2, -1))
           * np.linalg.norm(X, axis=(-2, -1))
           + np.linalg.norm(B, axis=(-2, -1))))

    A_dist = _put(A, mesh)
    got = D.solve_backward_error(
        A_dist, _put(X, mesh), _put(B, mesh),
        mesh=mesh, backend="off", batched_route="batch_reshard",
        norm_a=D.frobenius_norm(A_dist))

    assert got.shape == batch
    np.testing.assert_allclose(np.asarray(got), want, rtol=5e-12, atol=0.0)

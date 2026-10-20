"""The transverse ζ factor's kappa gate is one estimator on both routes (isdf.core._kappa_1_lower):
the local hoisted LU and the mesh route's C^-1 give the same [eta, kappa] figures, and kappa is a
lower bound on ||C||_1 ||C^-1||_1 of the ridged logical block. P = 4 host devices; the mesh LU
provider is stood in by a dense solve (the route's own code runs around it)."""
import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402


def test_both_routes_read_one_kappa(monkeypatch):
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    import distrib_la
    from isdf import core
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    nq, n, n_log = 2, 32, 30
    rng = np.random.default_rng(31)
    c = np.zeros((nq, n, n), complex)
    for q in range(nq):
        u, _ = np.linalg.qr(rng.standard_normal((n_log, n_log)) + 1j * rng.standard_normal((n_log, n_log)))
        w = np.r_[-np.logspace(0, 3 + q, n_log // 2), np.logspace(-1, 1, n_log - n_log // 2)]
        c[q, :n_log, :n_log] = (u * w) @ u.conj().T
    face = NamedSharding(mesh, P(None, "x", "y"))
    cq = core._identity_pad_block_diagonal(jax.device_put(c, face), n_rmu_logical=n_log, mesh_xy=mesh)
    _, _, local = core._factor_c_q_transverse_lu(cq, mesh, n_log)
    monkeypatch.setattr(distrib_la, "plan", lambda *a, **k: types.SimpleNamespace(
        batched=lambda A, B: jnp.linalg.solve(A, B)))
    _, mesh_figs = core._factor_c_q_transverse_inverse(cq, mesh, n_log)
    local, mesh_figs = np.asarray(local), np.asarray(mesh_figs)
    assert local.shape == mesh_figs.shape == (nq, 2)
    np.testing.assert_allclose(mesh_figs[:, 1], local[:, 1], rtol=1e-8)
    assert np.all(local[:, 0] < 1e-12) and np.all(mesh_figs[:, 0] < 1e-12)
    for q in range(nq):
        cl = c[q, :n_log, :n_log]
        cl = cl + 1e-12 * np.sign(np.trace(cl).real) * abs(np.trace(cl)) / n_log * np.eye(n_log)
        k1 = np.linalg.norm(cl, 1) * np.linalg.norm(np.linalg.inv(cl), 1)
        assert 0.5 * k1 <= local[q, 1] <= k1 * (1 + 1e-10)

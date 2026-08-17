"""WP3 frozen-static crossing-block algebra and compression gates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.mpa import intraband_block as IB


def _mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _put(mesh, value, spec):
    return jax.device_put(jnp.asarray(value), NamedSharding(mesh, spec))


def _fixture(n_pair=6):
    mesh = _mesh()
    rng = np.random.default_rng(6217)
    n_mu = 4
    raw = (rng.normal(size=(n_mu, n_mu))
           + 1j * rng.normal(size=(n_mu, n_mu)))
    W0 = 0.015 * (raw + raw.conj().T) + 0.18 * np.eye(n_mu)
    vertices = (rng.normal(size=(n_pair, n_mu))
                + 1j * rng.normal(size=(n_pair, n_mu))) / 3.0
    u = np.linspace(0.04, 0.19, n_pair)
    w = np.linspace(2.0e-4, 1.1e-3, n_pair)
    block = (
        _put(mesh, u, P(None)),
        _put(mesh, w, P(None)),
        (_put(mesh, vertices, P(None, "x")),
         _put(mesh, vertices, P(None, "y"))),
    )
    return mesh, W0, vertices, u, w, block


def _direct(W0, vertices, u, w, z):
    d = -2.0 * w / (u * u - complex(z) ** 2)
    chi1 = vertices.T @ (d[:, None] * vertices.conj())
    return np.linalg.solve(np.eye(W0.shape[0]) - W0 @ chi1,
                           W0 @ chi1 @ W0)


def test_six_mode_limit_is_the_exact_frozen_static_pole_sum():
    mesh, W0, vertices, u, w, block = _fixture()
    # The production contract certifies all 24 stamped samples.  Keep the
    # synthetic oracle at that same cardinality, including the static anchor.
    z = np.concatenate((
        np.asarray([0.0j]),
        np.linspace(0.01, 0.60, 23) + 0.2j,
    ))
    row = IB.build_row(
        _put(mesh, W0, P("x", "y")), block, z,
        sample_rel_tol=1.0e-11)
    assert row.n_modes == row.n_poles == 6
    assert row.certified
    assert row.folded_modes == row.dropped_modes == 0
    for value in z:
        got = np.asarray(IB.evaluate_pole_sum(
            row.Omega_p, row.B_p, value))
        np.testing.assert_allclose(
            got, _direct(W0, vertices, u, w, value),
            rtol=2.0e-10, atol=2.0e-12)


def test_clustered_block_preserves_the_static_anchor_elementwise():
    mesh, W0, _vertices, _u, _w, block = _fixture()
    z = np.asarray([0.0j, 0.04 + 0.2j, 0.15 + 0.2j, 0.6 + 0.2j])
    exact = IB.build_row(
        _put(mesh, W0, P("x", "y")), block, z,
        sample_rel_tol=1.0e-11)
    clustered = IB.build_row(
        _put(mesh, W0, P("x", "y")), block, z,
        sample_rel_tol=1.0)
    assert clustered.n_poles == 3
    np.testing.assert_allclose(
        np.asarray(IB.evaluate_pole_sum(
            clustered.Omega_p, clustered.B_p, 0.0j)),
        np.asarray(IB.evaluate_pole_sum(
            exact.Omega_p, exact.B_p, 0.0j)),
        rtol=2.0e-11, atol=2.0e-12)
    live = np.abs(np.asarray(clustered.B_p)) != 0.0
    omega = np.asarray(clustered.Omega_p)
    assert np.all(omega.real[live] > 0.0)
    assert np.all(omega.imag[live] <= 0.0)


def test_padding_is_causal_and_exactly_dark():
    mesh, W0, _vertices, _u, _w, block = _fixture()
    row = IB.build_row(
        _put(mesh, W0, P("x", "y")), block,
        np.asarray([0.04 + 0.2j]), sample_rel_tol=1.0)
    Omega, Bp = IB.pad_row(row, 5)
    assert Omega.shape == Bp.shape == (5, 4, 4)
    np.testing.assert_array_equal(np.asarray(Bp[row.n_poles:]), 0.0)
    np.testing.assert_array_equal(np.asarray(Omega[row.n_poles:]), 1.0)


def test_dense_eigenproblem_refuses_above_the_design_mode_ceiling():
    mesh, W0, _vertices, _u, _w, _block = _fixture()
    count = IB.MAX_DENSE_MODES + 1
    vertices = np.ones((count, W0.shape[0]), np.complex128)
    block = (
        _put(mesh, np.linspace(0.04, 0.2, count), P(None)),
        _put(mesh, np.full(count, 1.0e-4), P(None)),
        (_put(mesh, vertices, P(None, "x")),
         _put(mesh, vertices, P(None, "y"))),
    )
    with pytest.raises(
            ValueError, match=r"dense_eigenproblem_size.*4097.*4096"):
        IB.build_row(
            _put(mesh, W0, P("x", "y")), block,
            np.asarray([0.04 + 0.2j]))

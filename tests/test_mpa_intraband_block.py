"""WP3/3-A frozen-static crossing-block algebra and compression gates."""

import inspect

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


def test_5598_mode_row_builds_without_pair_square_allocation(capsys, monkeypatch):
    mesh, W0, _vertices, _u, _w, _block = _fixture()
    count = 5598
    rng = np.random.default_rng(317)
    vertices = (
        rng.normal(size=(count, W0.shape[0]))
        + 1j * rng.normal(size=(count, W0.shape[0]))) * 1.0e-3
    block = (
        _put(mesh, np.linspace(0.08, 0.081, count), P(None)),
        _put(mesh, np.full(count, 1.0e-8), P(None)),
        (_put(mesh, vertices, P(None, "x")),
         _put(mesh, vertices, P(None, "y"))),
    )
    monkeypatch.setattr(
        IB, "_dense_reference_modes",
        lambda *_args, **_kwargs: pytest.fail("production called dense oracle"))
    row = IB.build_row(
        _put(mesh, W0, P("x", "y")), block,
        np.asarray([0.0j, 0.04 + 0.2j]), sample_rel_tol=1.0)
    assert row.n_modes == 5598
    assert row.certified
    output = capsys.readouterr().out
    assert "n_pair=5598" in output
    assert "ns_squared_arrays=0" in output
    assert "_dense_reference_modes" not in inspect.getsource(IB.build_row)


def _dense_partition_moments(mesh, W0, block, intervals):
    eigenvalues, left, right, _weights = IB._dense_reference_modes(W0, block)
    lam = np.asarray(eigenvalues, dtype=np.complex128)
    _left_bound, _zero_gap, _right_bound, height = IB._contour_geometry(
        block, W0)
    rows = [[], [], [], []]
    for lo, hi in intervals:
        select = ((lam.real >= lo) & (lam.real <= hi)
                  & (np.abs(lam.imag) < height))
        # Adjacent intervals share only a measure-zero boundary.  These
        # synthetic modes are separated from every constructed edge.
        idx = jnp.asarray(np.flatnonzero(select), dtype=jnp.int32)
        l = left[:, idx]
        r = right[idx, :]
        lc = jnp.asarray(lam[select])
        C = l @ r
        rows[0].append(-C)
        rows[1].append((l / lc[None, :]) @ r)
        roots = jnp.asarray([
            IB._retarded_sqrt_on_interval(value, (lo, hi))
            for value in lam[select]
        ])
        rows[2].append(-(l * roots[None, :]) @ r)
        rows[3].append(-(l * lc[None, :]) @ r)
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    return tuple(jax.device_put(jnp.stack(values), sharding) for values in rows)


def test_contour_moments_and_compression_match_dense_reference():
    mesh, W0, _vertices, _u, _w, block = _fixture(n_pair=18)
    W0j = _put(mesh, W0, P("x", "y"))
    intervals = IB._initial_intervals(block, W0j)
    contour = IB._cluster_moment_matrices(
        block, W0j, intervals, moment_rel_tol=1.0e-10)
    dense = _dense_partition_moments(mesh, W0j, block, intervals)
    for got, expected in zip(contour, dense):
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(expected), rtol=1.0e-10, atol=1.0e-12)

    contour_poles = IB._compress_moments(mesh, *contour)
    dense_poles = IB._compress_moments(mesh, *dense)
    np.testing.assert_allclose(
        np.asarray(contour_poles[0]), np.asarray(dense_poles[0]),
        rtol=1.0e-10, atol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(contour_poles[1]), np.asarray(dense_poles[1]),
        rtol=1.0e-10, atol=1.0e-12)


def test_signed_mp1_weights_cover_the_negative_screened_strip():
    mesh, W0, vertices, u, w, _block = _fixture(n_pair=18)
    signed_w = w.copy()
    signed_w[::2] *= -4.0
    block = (
        _put(mesh, u, P(None)),
        _put(mesh, signed_w, P(None)),
        (_put(mesh, vertices, P(None, "x")),
         _put(mesh, vertices, P(None, "y"))),
    )
    W0j = _put(mesh, W0, P("x", "y"))
    intervals = IB._initial_intervals(block, W0j)
    assert intervals[0][1] < 0.0 < intervals[1][0]
    contour = IB._cluster_moment_matrices(block, W0j, intervals)
    dense = _dense_partition_moments(mesh, W0j, block, intervals)
    for got, expected in zip(contour, dense):
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(expected),
            rtol=1.0e-10, atol=1.0e-10)


def test_contour_sum_rules_close_to_1e_minus_12_relative():
    mesh, W0, _vertices, _u, _w, block = _fixture(n_pair=21)
    W0j = _put(mesh, W0, P("x", "y"))
    moments = IB._cluster_moment_matrices(
        block, W0j, IB._initial_intervals(block, W0j))
    M_total, V_total = IB._exact_moment_totals(block, W0j)
    assert IB._relative_error(jnp.sum(moments[0], axis=0), M_total) <= 1.0e-12
    assert IB._relative_error(jnp.sum(moments[1], axis=0), V_total) <= 1.0e-12


@pytest.mark.parametrize("moment_index", [0, 1], ids=["M", "V"])
def test_contour_sum_rule_mismatch_is_a_refusal(monkeypatch, moment_index):
    mesh, W0, _vertices, _u, _w, block = _fixture(n_pair=9)
    W0j = _put(mesh, W0, P("x", "y"))
    intervals = IB._initial_intervals(block, W0j)
    original = IB._moments_at_order

    def broken(*args, **kwargs):
        values = list(original(*args, **kwargs))
        values[moment_index] = values[moment_index] * 0.99
        return tuple(values)

    monkeypatch.setattr(IB, "_moments_at_order", broken)
    with pytest.raises(ValueError, match=r"intraband_contour_sum_rule"):
        IB._cluster_moment_matrices(block, W0j, intervals)

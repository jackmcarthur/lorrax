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
        block, W0j, intervals, moment_rel_tol=1.0e-10)[:4]
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
    contour = IB._cluster_moment_matrices(block, W0j, intervals)[:4]
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
    # A clean row has no zero mode to deflate; the merge is then a no-op.
    assert moments[4].zero_mode_weight <= 1.0e-12
    assert moments[4].v_closure_after_merge <= 1.0e-12


def test_asymptotic_sum_rule_mismatch_is_a_refusal(monkeypatch):
    """An open D_M is a missed finite mode: refuse, never merge."""
    mesh, W0, _vertices, _u, _w, block = _fixture(n_pair=9)
    W0j = _put(mesh, W0, P("x", "y"))
    intervals = IB._initial_intervals(block, W0j)
    original = IB._moments_at_order

    def broken(*args, **kwargs):
        values = list(original(*args, **kwargs))
        values[0] = values[0] * 0.99
        return tuple(values)

    monkeypatch.setattr(IB, "_moments_at_order", broken)
    with pytest.raises(ValueError, match=r"intraband_contour_sum_rule"):
        IB._cluster_moment_matrices(block, W0j, intervals)


def test_deleting_one_finite_contour_refuses_on_dm_not_a_silent_merge(
        monkeypatch):
    """WP3-A2 acceptance (c): drop a finite cluster's contour entirely.

    The deleted cluster carries asymptotic weight, so the dichotomy must
    resolve to the refusal arm rather than absorbing the whole cluster as
    zero-mode static weight.
    """
    mesh, W0, _vertices, _u, _w, block = _fixture(n_pair=12)
    W0j = _put(mesh, W0, P("x", "y"))
    intervals = IB._initial_intervals(block, W0j)
    original = IB._moments_at_order

    def deleted(*args, **kwargs):
        values = list(original(*args, **kwargs))
        mask = jnp.asarray(
            [0.0] + [1.0] * (int(values[0].shape[0]) - 1))[:, None, None]
        return tuple(value * mask for value in values)

    monkeypatch.setattr(IB, "_moments_at_order", deleted)
    with pytest.raises(ValueError) as excinfo:
        IB._cluster_moment_matrices(block, W0j, intervals)
    message = str(excinfo.value)
    assert "intraband_contour_sum_rule" in message
    assert "masquerade" in message


def test_static_deficiency_is_merged_not_refused(monkeypatch):
    """WP3-A2: closed D_M with a refinement-invariant D_V is the zero mode.

    A uniform static defect is indistinguishable from zero-mode weight by
    the sum rule alone -- that is the ruling, and the guard against a
    tautology is the post-merge sample/static certification in build_row,
    not a second sum rule.
    """
    mesh, W0, _vertices, _u, _w, block = _fixture(n_pair=9)
    W0j = _put(mesh, W0, P("x", "y"))
    intervals = IB._initial_intervals(block, W0j)
    original = IB._moments_at_order

    def broken(*args, **kwargs):
        values = list(original(*args, **kwargs))
        values[1] = values[1] * 0.99
        return tuple(values)

    monkeypatch.setattr(IB, "_moments_at_order", broken)
    M, V, _T1, _T2, closure = IB._cluster_moment_matrices(
        block, W0j, intervals)
    M_total, V_total = IB._exact_moment_totals(block, W0j)
    assert closure.zero_mode_weight == pytest.approx(1.0e-2, rel=2.0e-2)
    assert closure.v_closure_after_merge <= 1.0e-12
    assert closure.m_closure <= 1.0e-12
    assert 0 <= closure.zero_mode_cluster < int(V.shape[0])
    assert IB._relative_error(jnp.sum(V, axis=0), V_total) <= 1.0e-12
    assert IB._relative_error(jnp.sum(M, axis=0), M_total) <= 1.0e-12


def _screening_channel_fixture(k=0.25, eps=1.0e-9, n_pair=18):
    """A perfectly-screened relaxational channel beside a signed continuum.

    One crossing pair carries a machine-small transition energy with an MP1
    weight that vanishes with it as ``w ~ k u**2`` -- the physical edge of
    the intraband continuum, where ``f_a - f_b`` vanishes with ``Delta``.
    ``chihat(0)`` keeps that channel's finite static screening while its mode
    sits at ``lambda ~ u**2`` inside the excluded zero gap with an asymptotic
    weight proportional to the same ``lambda``.  That is exactly the claim
    0319 production signature: ``M`` closes, ``V`` does not.
    """
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
    w[::2] *= -4.0
    u = np.concatenate(([eps], u))
    w = np.concatenate(([k * eps * eps], w))
    vertices = np.vstack([0.7 * vertices[0:1], vertices])
    block = (
        _put(mesh, u, P(None)),
        _put(mesh, w, P(None)),
        (_put(mesh, vertices, P(None, "x")),
         _put(mesh, vertices, P(None, "y"))),
    )
    return mesh, W0, vertices, u, w, block


def test_all_positive_bulk_zero_channel_refuses_on_the_open_m_closure():
    """The other arm of the same construction, on the same fixture family.

    With an all-positive bulk the derived strip's largest-gap tiling does not
    enclose every finite mode, so ``D_M`` opens.  The dichotomy must resolve
    to the refusal even though ``D_V`` is large -- large static deficiency is
    never on its own a licence to merge.
    """
    mesh, W0, vertices, u, w, _block = _screening_channel_fixture(n_pair=9)
    w = w.copy()
    w[1::2] = np.abs(w[1::2])
    w[2::2] = np.abs(w[2::2])
    w[0] = -abs(w[0])
    block = (
        _put(mesh, u, P(None)),
        _put(mesh, w, P(None)),
        (_put(mesh, vertices, P(None, "x")),
         _put(mesh, vertices, P(None, "y"))),
    )
    W0j = _put(mesh, W0, P("x", "y"))
    with pytest.raises(ValueError) as excinfo:
        IB._cluster_moment_matrices(
            block, W0j, IB._initial_intervals(block, W0j))
    message = str(excinfo.value)
    assert "intraband_contour_sum_rule" in message
    assert "masquerade" in message
    assert "NOT merged" in message


def test_perfect_screening_channel_merges_and_anchors_z0_exactly():
    """WP3-A2 acceptance (b): dense-oracle A/B on a constructed zero mode."""
    mesh, W0, vertices, u, w, block = _screening_channel_fixture()
    W0j = _put(mesh, W0, P("x", "y"))

    # Dense oracle: localize the deficiency on the excluded machine-zero mode.
    eigenvalues, left, right, _weights = IB._dense_reference_modes(W0j, block)
    lam = np.asarray(eigenvalues, dtype=np.complex128)
    C = np.einsum("im,mj->mij", np.asarray(left), np.asarray(right))
    _lo, zero_gap, _hi, _height = IB._contour_geometry(block, W0j)
    inside = np.abs(lam) < zero_gap
    assert int(inside.sum()) == 1
    V_dense = (C / lam[:, None, None]).sum(axis=0)
    dV_dense = (C[inside] / lam[inside, None, None]).sum(axis=0)
    dM_dense = -C[inside].sum(axis=0)
    assert (np.linalg.norm(dM_dense) / np.linalg.norm(C.sum(axis=0))
            <= 1.0e-12)
    dense_weight = np.linalg.norm(dV_dense) / np.linalg.norm(V_dense)
    assert dense_weight > 1.0e-3

    # The contour construction, at the fixed initial tiling: the sum rule
    # prices exactly the weight the dense oracle localizes, and both closures
    # become identities after the merge.
    intervals = IB._initial_intervals(block, W0j)
    M, V, T1, T2, closure = IB._cluster_moment_matrices(block, W0j, intervals)
    M_total, V_total = IB._exact_moment_totals(block, W0j)
    assert closure.zero_mode_weight == pytest.approx(dense_weight, rel=1.0e-4)
    assert closure.zero_mode_cluster >= 0
    assert closure.zero_mode_pole_shift > 0.0
    assert closure.m_closure <= IB.SUM_RULE_REL_TOL
    assert closure.v_closure_before_merge > 1.0e-3
    assert closure.v_closure_after_merge <= IB.SUM_RULE_REL_TOL
    assert IB._relative_error(jnp.sum(V, axis=0), V_total) <= 1.0e-12
    assert IB._relative_error(jnp.sum(M, axis=0), M_total) <= 1.0e-12

    # z=0 is a parameter-free identity after the merge, not a tolerance:
    # the compressed poles reproduce the dense `lambda -> 0` static limit.
    Om, Bp, _fold, dropped, _width = IB._compress_moments(mesh, M, V, T1, T2)
    assert dropped == 0
    got = np.asarray(IB.evaluate_pole_sum(Om, Bp, 0.0j))
    exact = _direct(W0, vertices, u, w, 0.0)
    assert (np.linalg.norm(got - exact) / np.linalg.norm(exact)) <= 1.0e-12
    # ... and the dense modal sum is that same static limit, so "matching the
    # dense lambda -> 0 limit at z = 0" is one identity, not two tolerances.
    np.testing.assert_allclose(
        np.asarray(V_dense), exact, rtol=1.0e-10, atol=1.0e-12)

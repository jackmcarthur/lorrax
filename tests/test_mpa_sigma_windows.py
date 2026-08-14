from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from gw.mpa import sigma_windows as SW
from gw.ppm_tau_kernel import build_shared_w_tau
from gw.ppm_windows import _SigmaBranch


def _branches():
    omega = np.asarray([0.0, 0.5, 1.0])
    idx = np.arange(omega.size)
    cond = jnp.asarray([[0.2, 1.15, 1.5]])
    # A slightly negative occupied/empty edge is allowed when the actual
    # denominator remains sign-definite; occupation policy is separate.
    val = jnp.asarray([[-0.02, 1.1, 1.45]])
    mask = jnp.ones_like(cond, dtype=bool)
    return [
        _SigmaBranch("pos_cond", cond, mask, "cond", False, omega, idx),
        _SigmaBranch("pos_val", val, mask, "val", False, omega, idx),
        _SigmaBranch("neg_cond", cond, mask, "cond", True, omega, idx),
        _SigmaBranch("neg_val", val, mask, "val", True, omega, idx),
    ]


def _poles():
    one = np.asarray([[[0.45 - 0.10j, 0.55 - 0.30j],
                       [1.45 - 0.10j, 1.55 - 0.30j]]])
    return (jnp.asarray(one), jnp.asarray(one + 0.02))


def test_actual_windows_match_all_four_causal_denominators():
    """End-to-end scalar oracle for planner signs and pole selection.

    The spatial contraction is deliberately factored out: its scalar input
    is the real production ``build_shared_w_tau`` result.  The residues are
    non-Hermitian so the near-axis result can only pass through the explicit
    +t/-t construction; an inferred band adjoint gives a different answer.
    """
    energy = 0.1
    omega = 0.45
    E_A = jnp.asarray([[energy]])
    mask = jnp.ones_like(E_A, dtype=bool)
    omega_vec = np.asarray([omega])
    omega_idx = np.asarray([0])
    branches = [
        _SigmaBranch("pos_cond", E_A, mask, "cond", False,
                     omega_vec, omega_idx),
        _SigmaBranch("pos_val", E_A, mask, "val", False,
                     omega_vec, omega_idx),
        _SigmaBranch("neg_cond", E_A, mask, "cond", True,
                     omega_vec, omega_idx),
        _SigmaBranch("neg_val", E_A, mask, "val", True,
                     omega_vec, omega_idx),
    ]
    Omega = jnp.stack((
        jnp.full((1, 2, 2), 0.3 - 0.05j),
        jnp.full((1, 2, 2), 0.3 - 0.25j),
    ))
    B = jnp.asarray([
        [[[1.0 + 0.2j, 0.3 + 0.7j],
          [-0.4 + 0.1j, 0.8 - 0.2j]]],
        [[[0.4 - 0.5j, -0.2 + 0.6j],
          [0.9 + 0.3j, -0.1 - 0.4j]]],
    ])
    plan, geometry = SW.build_shared_sigma_windows(
        tuple(Omega), branches, regularization_width_ry=0.2,
        edge_factor=1.5, target_error=1.0e-6, max_rank=96)
    assert [(row.window.name, row.window.prefactor,
             row.window.omega_sign) for row in plan] == [
        ("single", -1.0, -1), ("single", 1.0, -1),
        ("core", -1.0, 1), ("core", 1.0, 1),
    ]

    B_host = np.asarray(B)
    eta = geometry["eta_ry"]
    poles = tuple(complex(Omega[p, 0, 0, 0]) - 1j * eta
                  for p in range(Omega.shape[0]))
    for row in plan:
        window = row.window
        pole_indices = jnp.asarray(row.pole_indices)
        bounds = jnp.asarray(row.bounds)
        phase_real = jnp.asarray(row.phase_real)

        # One compiled batch evaluates the exact production W(t) builder at
        # every node without turning this oracle into a Python dispatch test.
        build_all = jax.jit(jax.vmap(
            lambda t: build_shared_w_tau(
                B, Omega, pole_indices, bounds, phase_real,
                window.E_ref_B, t)))
        W_t = np.asarray(build_all(window.nodes.t))
        t = np.asarray(window.nodes.t)
        alpha = np.asarray(window.nodes.alpha)
        G_t = np.exp(-1j * (energy - window.E_ref_A) * t)
        coeff = (window.prefactor * alpha
                 * np.exp(-1j * (window.E_ref_A + window.E_ref_B
                                  - window.omega_sign * omega) * t))
        got = np.sum(coeff[:, None, None, None]
                     * G_t[:, None, None, None] * W_t, axis=0)

        if window.name == "single":
            want = window.prefactor * sum(
                B_host[p] / (omega + energy + pole)
                for p, pole in enumerate(poles))
        else:
            want = window.prefactor * sum(
                B_host[p] / (omega - energy - pole)
                for p, pole in enumerate(poles))
        np.testing.assert_allclose(got, want, rtol=2.0e-6, atol=3.0e-6)


def test_window_plan_partitions_widths_and_shares_rules(monkeypatch):
    def fake_fit(rectangles, **_kwargs):
        assert np.all(np.asarray(rectangles)[:, 0] > 0.0)
        return SimpleNamespace(
            nodes=np.asarray([0.2 + 0.1j, 0.7 + 0.2j]),
            weights=np.asarray([0.4 + 0.1j, 0.3 + 0.2j]),
            sampled_max_error=8e-5)

    def fake_damped(*_args, **_kwargs):
        return {"t": np.asarray([0.3, 0.8]),
                "h": np.asarray([0.4, 0.6]),
                "sampled_max_error": 7e-5,
                "kappa0": 0.9999}

    monkeypatch.setattr(SW.minimax, "fit_damped_reciprocal", fake_fit)
    monkeypatch.setattr(SW, "damped_rectangle_gauss_rule", fake_damped)

    plan, report = SW.build_shared_sigma_windows(
        _poles(), _branches(), regularization_width_ry=0.2,
        edge_factor=1.5)
    summaries = []
    for p, pole in enumerate(_poles()):
        summaries.extend(SW.summarize_sigma_poles(
            pole[None, ...], _branches(), regularization_width_ry=0.2,
            edge_factor=1.5, pole_offset=p))
    batched = SW.summarize_sigma_poles(
        jnp.stack(_poles()), _branches(), regularization_width_ry=0.2,
        edge_factor=1.5)
    assert batched == tuple(summaries)
    streamed, streamed_report = SW.build_shared_sigma_windows(
        None, _branches(), regularization_width_ry=0.2,
        edge_factor=1.5, pole_summaries=summaries)
    assert streamed_report == report
    assert [(r.window.name, r.pole_indices.tolist(), r.bounds.tolist())
            for r in streamed] == [
                (r.window.name, r.pole_indices.tolist(), r.bounds.tolist())
                for r in plan]
    names = [row.window.name for row in plan]
    assert names.count("single") == 2
    assert names.count("b_slab") == 2
    assert names.count("a_stripe") == 2
    assert names.count("core") == 2
    assert report["n_windows"] == 8

    single = next(row for row in plan if row.window.name == "single")
    assert single.pole_indices.tolist() == [0, 1]
    assert single.phase_real.tolist() == [False, False]
    exact = [row for row in plan if row.window.name == "core"]
    assert all(not np.any(row.phase_real) for row in exact)
    assert report["eta_ry"] == 0.2

    # Eta is a single scalar factor on every family, including complex-time
    # sector nodes.  The pole builder itself still sees the fitted Gamma.
    expected = ((np.asarray([0.4 + 0.1j, 0.3 + 0.2j]))
                * np.exp(-report["eta_ry"]
                         * np.asarray([-1j * (0.2 + 0.1j),
                                       -1j * (0.7 + 0.2j)])))
    np.testing.assert_allclose(np.asarray(single.window.nodes.alpha), expected)


def test_a_truly_crossing_sign_definite_cell_refuses(monkeypatch):
    monkeypatch.setattr(
        SW.minimax, "fit_damped_reciprocal",
        lambda *_a, **_k: SimpleNamespace(
            nodes=np.asarray([0.2 + 0.1j]),
            weights=np.asarray([0.4 + 0.1j]),
            sampled_max_error=1e-4))
    branches = _branches()
    # On pos_val, E_A=-0.02 and a=0.01 make E_A+a negative.  The method must
    # not hide that by flooring the lower bound to a small positive number.
    bad = (jnp.asarray([[[0.01 - 0.1j]]]),)
    try:
        SW.build_shared_sigma_windows(
            bad, branches, regularization_width_ry=0.2, edge_factor=1.5)
    except ValueError as exc:
        assert "crosses zero" in str(exc)
    else:
        raise AssertionError("a non-sign-definite rectangle was accepted")


def test_live_negative_real_pole_refuses():
    poles = (jnp.asarray([[[-0.2 - 0.1j]]]),)
    try:
        SW.build_shared_sigma_windows(
            poles, _branches(), regularization_width_ry=0.2,
            edge_factor=1.5)
    except ValueError as exc:
        assert "Re Omega <= 0" in str(exc)
    else:
        raise AssertionError("an unsupported negative-real pole was omitted")

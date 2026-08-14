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
    non-Hermitian so a residue-adjoint shortcut would give a different answer.
    The causal core must retain the same stored residue in its scalar time
    representation.
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
        tuple(Omega), branches, B_poles=tuple(B),
        regularization_width_ry=0.2,
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


def test_actual_stripe_and_slab_match_direct_complex_denominators():
    """The two sign-definite leftovers retain their exact pole widths."""
    omega = 0.45
    energies = np.asarray([0.1, 1.2])
    E_A = jnp.asarray(energies[None, :])
    branch = _SigmaBranch(
        "pos_cond", E_A, jnp.ones_like(E_A, dtype=bool), "cond", False,
        np.asarray([omega]), np.asarray([0]))
    Omega = jnp.asarray([
        [[[0.3 - 0.08j]]],   # shallow: core for E=0.1, stripe for E=1.2
        [[[1.3 - 0.35j]]],   # deep: slab for both electronic energies
    ])
    B = jnp.asarray([[[[0.7 + 0.2j]]], [[[-0.3 + 0.6j]]]])
    plan, geometry = SW.build_shared_sigma_windows(
        tuple(Omega), [branch], B_poles=tuple(B),
        regularization_width_ry=0.2, edge_factor=1.5,
        target_error=1.0e-6, max_rank=96)
    assert [row.window.name for row in plan] == [
        "b_slab", "a_stripe", "core"]

    eta = geometry["eta_ry"]
    poles = np.asarray(Omega[:, 0, 0, 0]) - 1j * eta
    residues = np.asarray(B[:, 0, 0, 0])
    expected = {
        "b_slab": residues[1] * np.sum(
            1.0 / (energies + poles[1] - omega)),
        "a_stripe": residues[0] / (energies[1] + poles[0] - omega),
        "core": residues[0] / (omega - energies[0] - poles[0]),
    }
    for row in plan:
        window = row.window
        build_all = jax.jit(jax.vmap(
            lambda t: build_shared_w_tau(
                B, Omega, jnp.asarray(row.pole_indices),
                jnp.asarray(row.bounds), jnp.asarray(row.phase_real),
                window.E_ref_B, t)))
        W_t = np.asarray(build_all(window.nodes.t))[:, 0, 0, 0]
        t = np.asarray(window.nodes.t)
        mask = np.asarray(window.mask_A[0])
        G_t = np.sum(np.exp(
            -1j * (energies[mask, None] - window.E_ref_A)
            * t[None, :]), axis=0)
        coeff = (window.prefactor * np.asarray(window.nodes.alpha)
                 * np.exp(-1j * (window.E_ref_A + window.E_ref_B
                                  - window.omega_sign * omega) * t))
        got = np.sum(coeff * G_t * W_t)
        np.testing.assert_allclose(
            got, window.prefactor * expected[window.name],
            rtol=2.0e-6, atol=3.0e-6)


def test_window_plan_partitions_widths_and_shares_rules(monkeypatch):
    seen = {}

    def fake_fit(rectangles, **_kwargs):
        assert np.all(np.asarray(rectangles)[:, 0] > 0.0)
        seen.setdefault("sector", []).append(_kwargs["target_error"])
        return SimpleNamespace(
            nodes=np.asarray([0.2 + 0.1j, 0.7 + 0.2j]),
            weights=np.asarray([0.4 + 0.1j, 0.3 + 0.2j]),
            sampled_max_error=8e-5)

    def fake_damped(*_args, **_kwargs):
        seen["crossing"] = _kwargs["rel_tol"]
        return {"t": np.asarray([0.3, 0.8]),
                "h": np.asarray([0.4, 0.6]),
                "sampled_max_error": 7e-5,
                "continuum_error_bound": 9e-5,
                "tail_budget_fraction": 0.5,
                "kappa0": 0.9999,
                "rule_type": "test"}

    monkeypatch.setattr(SW.minimax, "fit_damped_reciprocal", fake_fit)
    monkeypatch.setattr(SW, "damped_rectangle_positive_rule", fake_damped)

    plan, report = SW.build_shared_sigma_windows(
        _poles(), _branches(),
        B_poles=tuple(jnp.ones_like(pole) for pole in _poles()),
        regularization_width_ry=0.2,
        edge_factor=1.5, target_error=6.5e-4,
        crossing_target_error=2.0e-3)
    summaries = []
    for p, pole in enumerate(_poles()):
        summaries.extend(SW.summarize_sigma_poles(
            pole[None, ...], jnp.ones_like(pole[None, ...]), _branches(),
            regularization_width_ry=0.2,
            edge_factor=1.5, pole_offset=p))
    batched = SW.summarize_sigma_poles(
        jnp.stack(_poles()), jnp.ones_like(jnp.stack(_poles())), _branches(),
        regularization_width_ry=0.2,
        edge_factor=1.5)
    assert batched == tuple(summaries)
    streamed, streamed_report = SW.build_shared_sigma_windows(
        None, _branches(), regularization_width_ry=0.2,
        edge_factor=1.5, pole_summaries=summaries,
        target_error=6.5e-4, crossing_target_error=2.0e-3)
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
    assert report["sector_target_error"] == 6.5e-4
    assert report["crossing_target_error"] == 2.0e-3
    assert set(seen["sector"]) == {6.5e-4}
    assert seen["crossing"] == 2.0e-3

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
            bad, branches, B_poles=(jnp.ones_like(bad[0]),),
            regularization_width_ry=0.2, edge_factor=1.5)
    except ValueError as exc:
        assert "crosses zero" in str(exc)
    else:
        raise AssertionError("a non-sign-definite rectangle was accepted")


def test_live_negative_real_pole_refuses():
    poles = (jnp.asarray([[[-0.2 - 0.1j]]]),)
    try:
        SW.build_shared_sigma_windows(
            poles, _branches(), B_poles=(jnp.ones_like(poles[0]),),
            regularization_width_ry=0.2,
            edge_factor=1.5)
    except ValueError as exc:
        assert "Re Omega <= 0" in str(exc)
    else:
        raise AssertionError("an unsupported negative-real pole was omitted")


def test_exact_zero_residue_pole_does_not_change_geometry(monkeypatch):
    monkeypatch.setattr(
        SW.minimax, "fit_damped_reciprocal",
        lambda *_a, **_k: SimpleNamespace(
            nodes=np.asarray([0.2 + 0.1j]),
            weights=np.asarray([0.4 + 0.1j]),
            sampled_max_error=1e-4))
    monkeypatch.setattr(
        SW, "damped_rectangle_positive_rule",
        lambda *_a, **_k: {
            "t": np.asarray([0.3]), "h": np.asarray([0.4]),
            "sampled_max_error": 7e-5, "kappa0": 0.9999,
            "continuum_error_bound": 9e-5,
            "tail_budget_fraction": 0.5,
            "rule_type": "test"})
    live = _poles()[0]
    residue = jnp.ones_like(live)
    base, base_report = SW.build_shared_sigma_windows(
        (live,), _branches(), B_poles=(residue,),
        regularization_width_ry=0.2, edge_factor=1.5)
    dead = jnp.full_like(live, 1.0e4 - 1.0e4j)
    extended, extended_report = SW.build_shared_sigma_windows(
        (live, dead), _branches(),
        B_poles=(residue, jnp.zeros_like(dead)),
        regularization_width_ry=0.2, edge_factor=1.5)
    assert extended_report == base_report
    assert [(row.window.name, row.window.n_tau,
             row.pole_indices.tolist(), row.bounds.tolist())
            for row in extended] == [
                (row.window.name, row.window.n_tau,
                 row.pole_indices.tolist(), row.bounds.tolist())
                for row in base]


def test_nonpositive_eta_refuses():
    for eta in (0.0, -0.2, np.nan):
        try:
            SW.build_shared_sigma_windows(
                _poles(), _branches(),
                B_poles=tuple(jnp.ones_like(pole) for pole in _poles()),
                regularization_width_ry=eta,
                edge_factor=1.5)
        except ValueError as exc:
            assert "eta must be finite and positive" in str(exc)
        else:
            raise AssertionError(f"invalid eta {eta!r} was accepted")

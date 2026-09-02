from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from gw.mpa import sigma_windows as SW
from gw.ppm_tau_kernel import build_shared_w_tau
from gw.ppm_windows import _SigmaBranch

#: Step of the ω grid the shared fixtures request.  The planner reads the
#: deck's own sample spacing to tell a patch boundary from an adjacent
#: sample, so a fixture must declare the step its grid actually uses.
_STEP = 0.5
#: The patched fixtures place two isolated samples 3.0 apart; any step below
#: 2.0 resolves them as separate deliveries.
_PATCH_STEP = 0.25


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
    summaries = SW.summarize_sigma_poles(
        Omega, B, branches,
        regularization_width_ry=0.2, edge_factor=1.5)
    plan, geometry = SW.build_shared_sigma_windows(
        summaries, branches,
        regularization_width_ry=0.2,
        edge_factor=1.5, target_error=1.0e-6, max_rank=96,
        crossing_max_nodes=SW.CROSSING_NODE_FLOOR,
        omega_grid_step_ry=_STEP)
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
    summaries = SW.summarize_sigma_poles(
        Omega, B, [branch],
        regularization_width_ry=0.2, edge_factor=1.5)
    plan, geometry = SW.build_shared_sigma_windows(
        summaries, [branch],
        regularization_width_ry=0.2, edge_factor=1.5,
        target_error=1.0e-6, max_rank=96,
        crossing_max_nodes=SW.CROSSING_NODE_FLOOR,
        omega_grid_step_ry=_PATCH_STEP)
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
    plan, report = SW.build_shared_sigma_windows(
        summaries, _branches(), regularization_width_ry=0.2,
        edge_factor=1.5,
        target_error=6.5e-4, max_rank=96, crossing_max_nodes=SW.CROSSING_NODE_FLOOR,
        omega_grid_step_ry=_STEP)
    names = [row.window.name for row in plan]
    assert names.count("single") == 2
    assert names.count("b_slab") == 2
    assert names.count("a_stripe") == 2
    assert names.count("core") == 2
    assert report["n_windows"] == 8
    assert report["target_error"] == 6.5e-4
    assert set(seen["sector"]) == {6.5e-4}
    assert seen["crossing"] == 6.5e-4

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
    bad = jnp.asarray([[[[0.01 - 0.1j]]]])
    summaries = SW.summarize_sigma_poles(
        bad, jnp.ones_like(bad), branches,
        regularization_width_ry=0.2, edge_factor=1.5)
    try:
        SW.build_shared_sigma_windows(
            summaries, branches,
            regularization_width_ry=0.2, edge_factor=1.5,
            target_error=1.0e-4, max_rank=96,
            crossing_max_nodes=SW.CROSSING_NODE_FLOOR,
        omega_grid_step_ry=_STEP)
    except ValueError as exc:
        assert "crosses zero" in str(exc)
    else:
        raise AssertionError("a non-sign-definite rectangle was accepted")


def test_live_negative_real_pole_refuses():
    poles = jnp.asarray([[[[-0.2 - 0.1j]]]])
    try:
        SW.summarize_sigma_poles(
            poles, jnp.ones_like(poles), _branches(),
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
        SW.summarize_sigma_poles(
            live[None, ...], residue[None, ...], _branches(),
            regularization_width_ry=0.2, edge_factor=1.5),
        _branches(),
        regularization_width_ry=0.2, edge_factor=1.5,
        target_error=1.0e-4, max_rank=96,
        crossing_max_nodes=SW.CROSSING_NODE_FLOOR,
        omega_grid_step_ry=_STEP)
    dead = jnp.full_like(live, 1.0e4 - 1.0e4j)
    extended, extended_report = SW.build_shared_sigma_windows(
        SW.summarize_sigma_poles(
            jnp.stack((live, dead)),
            jnp.stack((residue, jnp.zeros_like(dead))), _branches(),
            regularization_width_ry=0.2, edge_factor=1.5),
        _branches(),
        regularization_width_ry=0.2, edge_factor=1.5,
        target_error=1.0e-4, max_rank=96,
        crossing_max_nodes=SW.CROSSING_NODE_FLOOR,
        omega_grid_step_ry=_STEP)
    assert extended_report == base_report
    assert [(row.window.name, row.window.n_tau,
             row.pole_indices.tolist(), row.bounds.tolist())
            for row in extended] == [
                (row.window.name, row.window.n_tau,
                 row.pole_indices.tolist(), row.bounds.tolist())
                for row in base]


def test_nonpositive_eta_refuses():
    summaries = SW.summarize_sigma_poles(
        jnp.stack(_poles()), jnp.ones_like(jnp.stack(_poles())), _branches(),
        regularization_width_ry=0.2, edge_factor=1.5)
    for eta in (0.0, -0.2, np.nan):
        try:
            SW.build_shared_sigma_windows(
                summaries, _branches(),
                regularization_width_ry=eta,
                edge_factor=1.5, target_error=1.0e-4, max_rank=96,
                crossing_max_nodes=SW.CROSSING_NODE_FLOOR,
        omega_grid_step_ry=_STEP)
        except ValueError as exc:
            assert "eta must be finite and positive" in str(exc)
        else:
            raise AssertionError(f"invalid eta {eta!r} was accepted")


# ---------------------------------------------------------------------------
#  Clustered decomposition (docs/dev/crossing-rule-cost-law.md)
# ---------------------------------------------------------------------------

def _oracle_total(plan, B, Omega, n_omega, shape):
    """Sum EVERY plan window through the production scalar algebra.

    The same executor arithmetic as the incumbent oracles above, but
    accumulated per global omega index across all windows, so it fails on
    any orientation, prefactor, omega-slicing, reference-anchor, or
    coverage mistake in a decomposed plan — the whole class of error the
    measured -8x sliver disagreement came from.
    """
    got = np.zeros((n_omega,) + shape, dtype=np.complex128)
    for row in plan:
        window = row.window
        build_all = jax.jit(jax.vmap(
            lambda t: build_shared_w_tau(
                B, Omega, jnp.asarray(row.pole_indices),
                jnp.asarray(row.bounds), jnp.asarray(row.phase_real),
                window.E_ref_B, t)))
        W_t = np.asarray(build_all(window.nodes.t))
        t = np.asarray(window.nodes.t)
        alpha = np.asarray(window.nodes.alpha)
        E_row = np.asarray(row.E_A)[0]
        mask = np.asarray(window.mask_A)[0]
        weight = (np.ones(E_row.shape)
                  if row.band_weight is None
                  else np.asarray(row.band_weight)[0])
        G_t = np.sum(
            weight[mask, None] * np.exp(
                -1j * (E_row[mask, None] - window.E_ref_A) * t[None, :]),
            axis=0)
        for w_abs, w_idx in zip(np.asarray(row.omega_abs),
                                np.asarray(row.omega_idx)):
            coeff = (window.prefactor * alpha * np.exp(
                -1j * (window.E_ref_A + window.E_ref_B
                       - window.omega_sign * float(w_abs)) * t))
            got[int(w_idx)] += np.sum(
                (coeff * G_t)[:, None, None, None] * W_t, axis=0)
    return got


def test_gapped_omega_grid_decomposes_the_core_and_matches_the_exact_sum():
    """Two ω clusters: the core splits into shell + slabs, same physics.

    Every band lies in a different cell for the two clusters (pos slab,
    shell, neg slab), so the sum over the five windows reproduces the
    exact denominator sum only if every orientation, conjugate placement,
    reference anchor, and ω slice is right.
    """
    energies = np.asarray([0.1, 1.2, 3.0, 3.6])
    omega = np.asarray([0.45, 3.45])
    E_A = jnp.asarray(energies[None, :])
    branch = _SigmaBranch(
        "pos_cond", E_A, jnp.ones_like(E_A, dtype=bool), "cond", False,
        omega, np.arange(omega.size))
    Omega = jnp.asarray([[[[0.30 - 0.05j]]]])
    B = jnp.asarray([[[[0.7 + 0.2j]]]])
    summaries = SW.summarize_sigma_poles(
        Omega, B, [branch],
        regularization_width_ry=0.2, edge_factor=1.5)
    plan, geometry = SW.build_shared_sigma_windows(
        summaries, [branch],
        regularization_width_ry=0.2, edge_factor=1.5,
        target_error=1.0e-6, max_rank=96, crossing_max_nodes=SW.CROSSING_NODE_FLOOR,
        omega_grid_step_ry=_STEP)

    names = sorted(row.window.name for row in plan)
    assert names == [
        "c_neg_slab", "c_neg_slab", "c_pos_slab", "core", "core"]
    # The two cluster shells share one cached rule object's node count.
    shells = [row for row in plan if row.window.name == "core"]
    assert shells[0].window.n_tau == shells[1].window.n_tau
    # Each window serves exactly its own cluster's ω subset.
    for row in plan:
        assert row.omega_abs.size == 1

    eta = geometry["eta_ry"]
    pole = complex(Omega[0, 0, 0, 0]) - 1j * eta
    residue = np.asarray(B[0])
    got = _oracle_total(plan, B, Omega, omega.size, residue.shape)
    for i, w in enumerate(omega):
        want = -sum(residue / (w - e - pole) for e in energies)
        np.testing.assert_allclose(got[i], want, rtol=5.0e-6, atol=5.0e-6)


def test_gapped_omega_grid_decomposes_the_metal_sliver_and_matches():
    """Sliver corner + sign-definite slabs reproduce the exact sd sum.

    A fractional val branch (weights ≠ 1) whose wrong-side state makes a
    sliver: with a gapped ω grid the sliver decomposes into the damped
    crossing corner (ω and a both small) plus Laplace cells; summed with
    the branch's ordinary windows it must equal the exact fractional
    denominator sum in this family's +ω orientation.
    """
    energies = np.asarray([-0.02, 1.1])
    weights = np.asarray([0.6, 1.0])
    omega = np.asarray([0.1, 0.45, 3.45])
    E_A = jnp.asarray(energies[None, :])
    branch = _SigmaBranch(
        "pos_val", E_A, jnp.ones_like(E_A, dtype=bool), "val", False,
        omega, np.arange(omega.size),
        band_weight=jnp.asarray(weights[None, :]))
    Omega = jnp.asarray([[[[0.30 - 0.05j]]]])
    B = jnp.asarray([[[[0.7 + 0.2j]]]])
    summaries = SW.summarize_sigma_poles(
        Omega, B, [branch],
        regularization_width_ry=0.2, edge_factor=1.5)
    plan, geometry = SW.build_shared_sigma_windows(
        summaries, [branch],
        regularization_width_ry=0.2, edge_factor=1.5,
        target_error=1.0e-6, max_rank=96, crossing_max_nodes=SW.CROSSING_NODE_FLOOR,
        omega_grid_step_ry=_STEP)

    names = sorted(row.window.name for row in plan)
    assert "sd_core" in names and "sd_shallow_slab" in names

    eta = geometry["eta_ry"]
    pole = complex(Omega[0, 0, 0, 0]) - 1j * eta
    residue = np.asarray(B[0])
    got = _oracle_total(plan, B, Omega, omega.size, residue.shape)
    for i, w in enumerate(omega):
        want = -sum(
            f * residue / (w + e + pole)
            for e, f in zip(energies, weights))
        np.testing.assert_allclose(got[i], want, rtol=5.0e-6, atol=5.0e-6)


def test_contiguous_grid_keeps_the_monolithic_plan_bitwise():
    """A gap-free ω grid must reproduce the incumbent plan exactly."""
    branches = _branches()
    poles = jnp.stack(_poles())
    summaries = SW.summarize_sigma_poles(
        poles, jnp.ones_like(poles), branches,
        regularization_width_ry=0.2, edge_factor=1.5)
    kwargs = dict(
        regularization_width_ry=0.2, edge_factor=1.5,
        target_error=1.0e-4, max_rank=96,
        crossing_max_nodes=SW.CROSSING_NODE_FLOOR,
        omega_grid_step_ry=_STEP)
    base, base_report = SW.build_shared_sigma_windows(
        summaries, branches, **kwargs)
    huge_gap, huge_report = SW.build_shared_sigma_windows(
        summaries, branches, **{**kwargs, "omega_grid_step_ry": 1.0e6})
    assert [(row.window.name, row.window.provenance,
             row.omega_idx.tolist()) for row in base] == [
        (row.window.name, row.window.provenance,
         row.omega_idx.tolist()) for row in huge_gap]
    assert {k: v for k, v in base_report.items()
            if k != "omega_gap_ry"} == {
        k: v for k, v in huge_report.items()
        if k != "omega_gap_ry"}


def test_wide_pole_spread_is_cut_from_each_cluster_shell():
    """A pole far above a cluster's reach must not inflate its shell.

    Two poles 2.7 Ry apart (the sodium store's real hazard: a ~5 Ry
    shallow-pole spatial spread made every cluster shell pay ~90 nodes/Ry
    of pole spread).  The low cluster's shell must exclude the deep pole
    (a > w_hi + margin cannot cross it) and stay small; the excluded
    pairs ride a sign-definite slab; and the summed plan still matches
    the exact two-pole denominator sum at every omega.
    """
    energies = np.asarray([0.1, 1.2, 3.0])
    omega = np.asarray([0.45, 3.45])
    E_A = jnp.asarray(energies[None, :])
    branch = _SigmaBranch(
        "pos_cond", E_A, jnp.ones_like(E_A, dtype=bool), "cond", False,
        omega, np.arange(omega.size))
    Omega = jnp.asarray([[[[0.30 - 0.05j]]], [[[3.00 - 0.10j]]]])
    B = jnp.asarray([[[[0.7 + 0.2j]]], [[[-0.4 + 0.6j]]]])
    summaries = SW.summarize_sigma_poles(
        Omega, B, [branch],
        regularization_width_ry=0.2, edge_factor=1.5)
    plan, geometry = SW.build_shared_sigma_windows(
        summaries, [branch],
        regularization_width_ry=0.2, edge_factor=1.5,
        target_error=1.0e-6, max_rank=96, crossing_max_nodes=SW.CROSSING_NODE_FLOOR,
        omega_grid_step_ry=_STEP)

    shells = [row for row in plan if row.window.name == "core"]
    assert len(shells) == 2
    low = min(shells, key=lambda row: float(np.max(row.omega_abs)))
    high = max(shells, key=lambda row: float(np.max(row.omega_abs)))
    # The low cluster keeps only the near pole; its bandwidth is set by
    # the cluster, not the 2.7 Ry pole spread.
    assert low.pole_indices.tolist() == [0]
    assert float(low.bounds[0][1]) < 3.0        # a_le capped below pole 2
    assert low.window.n_tau < high.window.n_tau
    # The cut pairs live somewhere: at least one sign-definite window
    # carries pole 2 for the low cluster's omega.
    low_omega = float(np.min(omega))
    carriers = [row for row in plan
                if 1 in row.pole_indices.tolist()
                and low_omega in np.asarray(row.omega_abs)
                and row.window.name != "core"]
    assert carriers

    eta = geometry["eta_ry"]
    poles = [complex(Omega[p, 0, 0, 0]) - 1j * eta for p in range(2)]
    residues = [np.asarray(B[p]) for p in range(2)]
    got = _oracle_total(plan, B, Omega, omega.size, residues[0].shape)
    for i, w in enumerate(omega):
        want = -sum(res / (w - e - pole)
                    for e in energies
                    for res, pole in zip(residues, poles))
        np.testing.assert_allclose(got[i], want, rtol=5.0e-6, atol=5.0e-6)

"""Physical-window seams used by the GN/HL-PPM denominator-box planner."""

import numpy as np
import pytest

from minimax import UniformRule


def _fake_rule(box, eps, **_kwargs):
    relative = box[0] > 0.0 or box[1] < 0.0
    return UniformRule(
        times=np.asarray([0.2 + 0.03j, 0.4 + 0.02j]),
        weights=np.asarray([0.6 - 0.1j, 0.3 + 0.05j]),
        box=tuple(box), eps=float(eps), relative=relative,
        theta_deg=5.0, rank=3, sup_error=0.5 * eps,
        kappa_max=1.2, seconds=0.01)


def _patch_rule_certificates(monkeypatch):
    monkeypatch.setattr(
        "gw.sigma_box_plan.build_uniform_rule", _fake_rule)
    monkeypatch.setattr(
        "gw.sigma_box_plan.rule_amplification_p99",
        lambda *_args, **_kwargs: 1.1)
    monkeypatch.setattr(
        "gw.sigma_box_plan.rule_sup_error",
        lambda *_args, **_kwargs: (5.0e-5, 1.2))


def test_deferred_ppm_windows_do_not_request_minimax_tables(monkeypatch):
    """Box planning keeps the incumbent panes but never serves old nodes."""
    from gw import ppm_windows

    def _unexpected_solver(*args, **kwargs):
        raise AssertionError("the table quadrature must be deferred")

    monkeypatch.setattr(
        ppm_windows, "solve_laplace_minimax_interval", _unexpected_solver)
    monkeypatch.setattr(
        ppm_windows, "solve_phase_minimax_bandwidth", _unexpected_solver)

    energies = np.array([[0.1, 0.5]], dtype=np.float64)
    poles = ppm_windows.jnp.asarray(
        np.array([[[0.1, 0.5], [0.2, 0.6]]], dtype=np.float64))
    windows = ppm_windows._build_windows_for_branch(
        omega_nonneg_ry=np.array([0.0, 0.1], dtype=np.float64),
        E_A=ppm_windows.jnp.asarray(energies),
        base_mask_A=ppm_windows.jnp.ones(energies.shape, dtype=bool),
        Omega_q=poles,
        base_mask_B=ppm_windows.jnp.ones(poles.shape, dtype=bool),
        space="cond",
        neg_omega_half=False,
        regularization_width_ry=0.1,
        edge_factor=1.5,
        target_error=1.0e-6,
        max_nodes=64,
        crossing_eps_q=1.0e-3,
        crossing_max_nodes=500,
        use_shipped_minimax_tables=True,
        log_tag="test",
        print_fn=lambda *_args, **_kwargs: None,
        defer_quadrature=True,
    )

    assert windows
    assert {window.project for window in windows} == {"full", "imag"}
    assert all(window.n_tau == 0 for window in windows)
    assert all(window.max_error is None for window in windows)
    assert all(window.provenance == "uniform denominator box pending"
               for window in windows)


def test_sigma_window_product_support_uses_live_half_open_selectors():
    """The support is states x PPM entries x this window's omega indices."""
    from gw import ppm_windows

    energies = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    mask_A = np.array([[True, True, False]])
    poles = ppm_windows.jnp.asarray(
        np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float64))
    nodes = ppm_windows._deferred_box_nodes()
    window = ppm_windows._SigmaWindow(
        name="pane_crossing",
        nodes=nodes,
        mask_A=mask_A,
        E_ref_A=0.0,
        E_ref_B=0.0,
        omega_sign=1,
        project="imag",
        prefactor=-1.0,
        crossing_kind="hgl",
        E_min=0.05,
        E_max=0.25,
        B_lo=0.15,
        B_hi=0.35,
        omega_indices=np.array([0, 2], dtype=np.int64),
    )

    support = ppm_windows.sigma_window_product_support(
        window,
        ppm_windows.jnp.asarray(energies),
        poles,
        ppm_windows.jnp.ones(poles.shape, dtype=bool),
        np.array([0.0, 0.5, 1.0], dtype=np.float64),
    )

    np.testing.assert_array_equal(mask_A, [[True, True, False]])
    np.testing.assert_allclose(support["states"], [0.1, 0.2])
    assert support["state_count"] == 2
    assert support["pole_count"] == 2
    assert support["pole_stats"] == ((0.2, 0.3, 0.0, 0.0),)
    np.testing.assert_array_equal(support["omega_indices"], [0, 2])
    np.testing.assert_allclose(support["omega_abs"], [0.0, 1.0])
    assert support["B_lo"] == 0.15
    assert support["B_hi"] == 0.35


def test_ppm_box_planner_executes_each_product_with_one_full_causal_rule(
        monkeypatch):
    """The final direct-box planner does not revive pane-specific nodes."""
    from gw import ppm_windows
    from gw.sigma_box_plan import plan_sigma_windows

    _patch_rule_certificates(monkeypatch)
    energies = ppm_windows.jnp.asarray(
        np.array([[0.1, 0.5]], dtype=np.float64))
    poles = ppm_windows.jnp.asarray(
        np.array([[[0.1, 0.5], [0.2, 0.6]]], dtype=np.float64))
    residues = ppm_windows.jnp.ones(poles.shape, dtype=complex)
    branch = ppm_windows._SigmaBranch(
        tag="positive conduction",
        E_A=energies,
        base_mask_A=ppm_windows.jnp.ones(energies.shape, dtype=bool),
        space="cond",
        neg_omega_half=False,
        omega_abs=np.array([0.0, 0.1], dtype=np.float64),
        omega_idx=np.array([0, 1], dtype=np.int64),
    )

    def pole_batches():
        yield 0, poles, residues

    planned, geometry = plan_sigma_windows(
        pole_batches, [branch], np.array([0.0, 0.1]), 0.1,
        eps=1.0e-4,
        reduction_seconds=120.0,
        pair_ceiling=64,
        cache_dir=None,
        print_fn=lambda *_args, **_kwargs: None,
        edge_factor=1.5,
        broaden_sign_definite=False,
    )

    windows = [row.window for row in planned]
    assert len(windows) == 3
    assert [window.name for window in windows] == [
        "positive conduction:resonant",
        "positive conduction:state_tail",
        "positive conduction:pole_tail",
    ]
    assert {window.project for window in windows} == {"full"}
    assert all(window.prefactor == -1.0 for window in windows)
    assert all(window.omega_sign == 1 for window in windows)
    assert geometry["planner"] == "uniform_denominator_boxes"
    assert geometry["window_tau_pairs"] == 6
    assert geometry["distinct_tau_count"] == 2
    for window, report in zip(
            windows, geometry["branches"][0]["windows"]):
        raw = _fake_rule(report["rule_box_ry"], 1.0e-4)
        np.testing.assert_allclose(window.nodes.t, raw.times)
        np.testing.assert_allclose(
            window.nodes.alpha,
            raw.weights * np.exp(
                -report["external_regularization_ry"] * raw.times))


@pytest.mark.parametrize(
    ("space", "external_sign"),
    (("cond", 1), ("cond", -1), ("val", 1), ("val", -1)),
)
def test_ppm_executor_nodes_reproduce_each_causal_denominator(
        monkeypatch, space, external_sign):
    """The shared time convention maps exactly onto all four PPM branches."""
    from gw import ppm_windows
    from gw.sigma_box_plan import plan_sigma_windows

    _patch_rule_certificates(monkeypatch)
    eta, omega, state, pole = 0.1, 0.2, 0.7, 1.1
    pole_sign = 1.0 if space == "cond" else -1.0
    frequency = external_sign * omega
    states = ppm_windows.jnp.asarray([[state]])
    poles = ppm_windows.jnp.asarray([[[pole]]])
    residues = ppm_windows.jnp.ones(poles.shape, dtype=complex)
    branch = ppm_windows._SigmaBranch(
        tag=f"{space}:{external_sign}",
        E_A=states,
        base_mask_A=ppm_windows.jnp.ones(states.shape, dtype=bool),
        space=space,
        neg_omega_half=external_sign < 0,
        omega_abs=np.asarray([omega]),
        omega_idx=np.asarray([0]),
    )

    def pole_batches():
        yield 0, poles, residues

    planned, geometry = plan_sigma_windows(
        pole_batches, [branch], np.asarray([frequency]), eta,
        eps=1.0e-4, reduction_seconds=120.0, pair_ceiling=64,
        cache_dir=None, print_fn=lambda *_args, **_kwargs: None,
        broaden_sign_definite=True)
    assert len(planned) == 1
    window = planned[0].window
    assert window.omega_sign == pole_sign * external_sign

    # This is the complete scalar phase assembled by ppm_tau_kernel and the
    # omega projector after the E_ref_A/E_ref_B factors cancel.  The spatial
    # kernel's -1 and the box window's prefactor=-1 cancel as well.
    executor = np.sum(
        np.asarray(window.nodes.alpha)
        * np.exp(1.0j * (window.omega_sign * omega - state - pole)
                 * np.asarray(window.nodes.t)))

    report = geometry["branches"][0]["windows"][0]
    raw = _fake_rule(report["rule_box_ry"], 1.0e-4)
    if space == "cond":
        physical_time = raw.times
        physical_weight = raw.weights
        denominator = frequency - state - pole + 1.0j * eta
    else:
        physical_time = -np.conj(raw.times)
        physical_weight = np.conj(raw.weights)
        denominator = frequency + state + pole - 1.0j * eta
    direct = np.sum(
        physical_weight * np.exp(1.0j * physical_time * denominator))
    np.testing.assert_allclose(executor, direct, rtol=2.0e-15, atol=2.0e-15)

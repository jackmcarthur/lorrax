"""Physical-window seams used by the GN/HL-PPM denominator-box planner."""

import numpy as np


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

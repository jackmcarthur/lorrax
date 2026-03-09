import numpy as np
import jax.numpy as jnp

from gw_isdf.minimax_screening import (
    solve_laplace_minimax_interval,
    build_static_minimax_window_pair,
    extract_gn_ppm_parameters,
)


def test_laplace_minimax_interval_accuracy():
    quad = solve_laplace_minimax_interval(
        0.2,
        8.0,
        target_error=5.0e-4,
        max_nodes=24,
    )
    x = np.exp(np.linspace(np.log(quad.x_min), np.log(quad.x_max), 4000))
    approx = np.exp(-np.outer(x, quad.tau)) @ quad.alpha
    ref = 1.0 / x
    err = float(np.max(np.abs(ref - approx)))
    assert err < 2.0e-3


def test_build_static_minimax_window_pair_shapes():
    enk_v = jnp.asarray([[-4.0, -2.0, -0.5], [-3.8, -1.9, -0.4]])
    enk_c = jnp.asarray([[0.3, 1.1, 2.7], [0.4, 1.0, 2.5]])
    windows, quad = build_static_minimax_window_pair(
        enk_v,
        enk_c,
        target_error=1.0e-4,
        max_nodes=30,
    )
    assert len(windows) == 1
    win = windows[0]
    assert win.tau_i.ndim == 1
    assert win.w_i.shape == win.tau_i.shape
    assert win.alpha_i.shape == win.tau_i.shape
    assert np.all(np.isfinite(win.w_i))
    assert quad.node_count == win.tau_i.shape[0]


def test_extract_gn_ppm_parameters_basic():
    # One q-point, 2x2 ISDF block.
    V = np.zeros((1, 1, 1, 1, 1, 1, 2, 2), dtype=np.complex128)
    V[0, 0, 0, 0, 0, 0] = np.diag([0.6, 0.4])

    chi0 = np.zeros((1, 1, 1, 1, 2, 1, 2), dtype=np.complex128)
    chii = np.zeros((1, 1, 1, 1, 2, 1, 2), dtype=np.complex128)
    chi0[0, 0, 0, 0, :, 0, :] = np.diag([-0.20, -0.10])
    chii[0, 0, 0, 0, :, 0, :] = np.diag([-0.08, -0.04])

    ppm = extract_gn_ppm_parameters(
        jnp.asarray(V),
        jnp.asarray(chi0),
        jnp.asarray(chii),
        omega_p=0.8,
        fallback_omega=1.0,
    )

    omega_q = np.asarray(ppm.omega_qmunu)
    b_q = np.asarray(ppm.b_qmunu)
    assert omega_q.shape == (1, 1, 1, 2, 2)
    assert b_q.shape == (1, 1, 1, 2, 2)
    assert np.all(np.isfinite(omega_q))
    assert np.all(np.real(omega_q.diagonal(axis1=-2, axis2=-1)) > 0.0)
    assert 0.0 <= ppm.unfulfilled_fraction <= 1.0

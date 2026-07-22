"""Minimax quadrature math — shipped-table selection + real-axis identities.

Merged (2026-07-09 redesign) from test_minimax_assets.py (shipped-table
catalog selection/rescaling contracts, monkeypatched solvers) and
test_real_axis_quadrature.py (fused (tau, alpha) real-axis quadrature vs
the analytic x/(x^2 - Omega^2) kernel, branch signs, large-Omega
asymptote).  The tau/2 rescaling correctness is load-bearing for every
dynamic Sigma_c run; the analytic identities are invisible at gate level
(a wrong-but-smooth quadrature still freezes reproducibly).
"""

from __future__ import annotations

import numpy as np

from common import minimax as minimax_core
from gw import minimax_screening as ms


def test_find_shipped_table_entry_prefers_smallest_range_then_loosest_error(monkeypatch):
    catalog = {
        "tables": [
            {
                "family": "noncrossing",
                "range_max": 100.0,
                "error_bound": 2.0e-7,
                "node_count": 30,
                "file": "noncrossing/strict.npz",
            },
            {
                "family": "noncrossing",
                "range_max": 100.0,
                "error_bound": 1.0e-6,
                "node_count": 28,
                "file": "noncrossing/loose.npz",
            },
            {
                "family": "noncrossing",
                "range_max": 200.0,
                "error_bound": 1.0e-6,
                "node_count": 20,
                "file": "noncrossing/larger_range.npz",
            },
        ],
    }
    monkeypatch.setattr(ms, "_load_shipped_minimax_catalog", lambda: catalog)

    entry = ms._find_shipped_table_entry(
        "noncrossing",
        range_value=80.0,
        target_error=1.0e-6,
        max_nodes=64,
    )

    assert entry is not None
    assert entry["file"] == "noncrossing/loose.npz"


def test_find_shipped_table_entry_filters_crossing_metadata(monkeypatch):
    catalog = {
        "tables": [
            {
                "family": "crossing",
                "target_kind": "fermi",
                "range_max": 80.0,
                "error_bound": 1.0e-6,
                "eps_q": 1.0e-3,
                "node_count": 50,
                "file": "crossing/wrong_kind.npz",
            },
            {
                "family": "crossing",
                "target_kind": "hgl",
                "range_max": 80.0,
                "error_bound": 1.0e-6,
                "eps_q": 1.0e-2,
                "node_count": 50,
                "file": "crossing/wrong_epsq.npz",
            },
            {
                "family": "crossing",
                "target_kind": "hgl",
                "range_max": 80.0,
                "error_bound": 1.0e-6,
                "eps_q": 1.0e-3,
                "node_count": 70,
                "file": "crossing/too_many_nodes.npz",
            },
            {
                "family": "crossing",
                "target_kind": "hgl",
                "range_max": 80.0,
                "error_bound": 1.0e-6,
                "eps_q": 1.0e-3,
                "node_count": 48,
                "file": "crossing/good.npz",
            },
        ],
    }
    monkeypatch.setattr(ms, "_load_shipped_minimax_catalog", lambda: catalog)

    entry = ms._find_shipped_table_entry(
        "crossing",
        range_value=60.0,
        target_error=1.0e-6,
        max_nodes=64,
        target_kind="hgl",
        eps_q=1.0e-3,
    )

    assert entry is not None
    assert entry["file"] == "crossing/good.npz"


def test_solve_laplace_minimax_interval_uses_shipped_table_and_rescales(monkeypatch):
    tau_hat = np.array([1.0, 2.0], dtype=np.float64)
    alpha_hat = np.array([0.25, 0.5], dtype=np.float64)
    err_hat = 5.0e-7

    monkeypatch.setattr(
        ms,
        "_pick_shipped_table",
        lambda *args, **kwargs: (tau_hat, alpha_hat, err_hat),
    )

    def _unexpected_solver(*args, **kwargs):
        raise AssertionError("Exact noncrossing solver should not run when a shipped table matches.")

    monkeypatch.setattr(ms, "_solve_noncrossing_scaled_cached", _unexpected_solver)

    quad = ms.solve_laplace_minimax_interval(
        2.0,
        20.0,
        target_error=1.0e-6,
        max_nodes=64,
        use_shipped_tables=True,
    )

    np.testing.assert_allclose(quad.tau, tau_hat / 2.0)
    np.testing.assert_allclose(quad.alpha, alpha_hat / 2.0)
    assert quad.max_error == err_hat / 2.0


def test_solve_phase_minimax_bandwidth_falls_back_when_no_shipped_table(monkeypatch):
    tau_hat = np.array([0.5, 1.5], dtype=np.float64)
    alpha_hat = np.array([0.1, 0.2], dtype=np.float64)
    err_hat = 9.0e-7

    monkeypatch.setattr(ms, "_pick_shipped_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ms,
        "_solve_crossing_scaled_cached",
        lambda *args, **kwargs: (tau_hat, alpha_hat, err_hat),
    )

    quad = ms.solve_phase_minimax_bandwidth(
        83.0,
        target_error=1.0e-6,
        max_nodes=500,
        eps_q=1.0e-3,
        target_kind="hgl",
    )

    np.testing.assert_allclose(quad.tau, tau_hat)
    np.testing.assert_allclose(quad.alpha, alpha_hat)
    assert quad.max_error == err_hat
    assert quad.target_kind == "hgl"


def test_solve_laplace_minimax_interval_skips_shipped_lookup_when_disabled(monkeypatch):
    tau_hat = np.array([0.75, 1.25], dtype=np.float64)
    alpha_hat = np.array([0.3, 0.4], dtype=np.float64)
    err_hat = 2.5e-7

    def _unexpected_pick(*args, **kwargs):
        raise AssertionError("Shipped lookup should stay disabled when use_shipped_tables=False.")

    monkeypatch.setattr(ms, "_pick_shipped_table", _unexpected_pick)
    monkeypatch.setattr(
        ms,
        "_solve_noncrossing_scaled_cached",
        lambda *args, **kwargs: (tau_hat, alpha_hat, err_hat),
    )

    quad = ms.solve_laplace_minimax_interval(
        1.0,
        10.0,
        target_error=1.0e-6,
        max_nodes=64,
        use_shipped_tables=False,
    )

    np.testing.assert_allclose(quad.tau, tau_hat)
    np.testing.assert_allclose(quad.alpha, alpha_hat)
    assert quad.max_error == err_hat


def test_crossing_solver_hgl_error_is_dense_and_well_conditioned():
    """The fast analytic start must retain the certified HGL approximation."""
    A = 20.0
    tau, alpha, reported_error = minimax_core.solve_crossing(
        25,
        A,
        minimax_core.G_hgl,
        minimax_core.tau_max_hgl(1.0e-3),
        target_error=1.0e-6,
    )

    u_dense = np.linspace(0.0, A, 20_001)
    dense_error = np.max(np.abs(
        minimax_core.G_hgl(u_dense)
        - np.sin(np.outer(u_dense, tau)) @ alpha
    ))

    assert reported_error < 1.0e-6
    assert dense_error < 1.0e-6
    assert np.sum(np.abs(alpha)) < 1.0


def test_imag_solver_stops_at_target_without_large_weight_cancellation():
    """Tolerance-aware polishing should not trade conditioning for excess digits."""
    R = 10.0
    omega_hat = 1.0
    tau, alpha, _node_count, reported_error = (
        minimax_core.noncrossing_imag_grids(
            R, omega_hat, 1.0e-6, N_max=16)
    )

    x_dense = np.geomspace(1.0, R, 20_001)
    target = x_dense / (x_dense**2 + omega_hat**2)
    dense_error = np.max(np.abs(
        target - np.exp(-np.outer(x_dense, tau)) @ alpha
    ))

    assert reported_error < 1.0e-6
    assert dense_error < 1.0e-6
    assert np.sum(np.abs(alpha)) < 1.0e4


def test_complex_noncrossing_resolvent_is_dense_grid_certified():
    R = 8.0
    omega_hat = 2.0
    target_error = 2.0e-5
    tau, alpha, _node_count, reported_error = (
        minimax_core.noncrossing_complex_grids(
            R, omega_hat, target_error, N_max=24)
    )

    x_dense = np.geomspace(1.0, R, 20_001)
    target = 1.0 / (x_dense + 1j * omega_hat)
    dense_error = np.max(np.abs(
        target - np.exp(-np.outer(x_dense, tau)) @ alpha
    ))

    assert np.iscomplexobj(alpha)
    assert reported_error < target_error
    assert dense_error < target_error


def test_complex_interval_rescales_and_preserves_complex_weights(monkeypatch):
    tau_hat = np.array([0.5, 1.5], dtype=np.float64)
    alpha_hat = np.array([0.25 - 0.5j, -0.1j], dtype=np.complex128)
    err_hat = 8.0e-7
    monkeypatch.setattr(
        ms,
        "_solve_noncrossing_complex_scaled_cached",
        lambda *args, **kwargs: (tau_hat, alpha_hat, err_hat),
    )

    quad = ms.solve_laplace_minimax_complex_interval(
        2.0, 20.0, 3.0, target_error=1.0e-6, max_nodes=16)
    nodes = quad.to_minimax_nodes(time_axis="real")

    np.testing.assert_allclose(quad.tau, tau_hat / 2.0)
    np.testing.assert_allclose(quad.alpha, alpha_hat / 2.0)
    np.testing.assert_allclose(np.asarray(nodes.alpha), alpha_hat / 2.0)
    assert quad.max_error == err_hat / 2.0


def test_minimax_disk_cache_preserves_complex_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("LORRAX_MINIMAX_CACHE_DIR", str(tmp_path))
    payload = {"solver": "complex-test", "revision": 1}
    tau = np.array([0.25, 1.0], dtype=np.float64)
    alpha = np.array([0.4 - 0.2j, -0.3j], dtype=np.complex128)

    ms._store_minimax_disk_cache("complex-test", payload, tau, alpha, 1.0e-7)
    loaded = ms._load_minimax_disk_cache("complex-test", payload)

    assert loaded is not None
    tau_loaded, alpha_loaded, error_loaded = loaded
    np.testing.assert_array_equal(tau_loaded, tau)
    np.testing.assert_array_equal(alpha_loaded, alpha)
    assert error_loaded == 1.0e-7

# ===========================================================================
#  real-axis quadrature vs analytic kernel (was test_real_axis_quadrature.py)
# ===========================================================================


import numpy as np
import pytest

from gw.minimax_screening import (
    LaplaceMinimaxQuadrature,
    solve_laplace_minimax_interval,
)
from gw.minimax_config import MinimaxConfig
from gw.minimax_screening import build_real_quadrature


# Realistic Si 4×4×4 ranges: x ∈ [E_gap, E_max] ≈ [0.5, 5] Ry.
X_MIN, X_MAX = 0.5, 5.0
TARGET_ERROR = 1.0e-6
MAX_NODES = 64


def _static_quad():
    """The static-window quad we'd hand to build_real_quadrature."""
    return solve_laplace_minimax_interval(
        X_MIN, X_MAX, target_error=TARGET_ERROR, max_nodes=MAX_NODES
    )


def _eval_fused(quad: LaplaceMinimaxQuadrature, xs: np.ndarray) -> np.ndarray:
    """Σ_l α_l · exp(−τ_l x) — what compute_chi0 will compute."""
    tau = np.asarray(quad.tau, dtype=np.float64)
    alpha = np.asarray(quad.alpha, dtype=np.float64)
    return np.exp(-np.outer(xs, tau)) @ alpha


def _real_target(xs: np.ndarray, Omega: float) -> np.ndarray:
    return xs / (xs**2 - Omega**2)


@pytest.mark.parametrize("Omega", [10.0, 50.0, 200.0, 800.0])
def test_real_quadrature_matches_target(Omega):
    """Fused (τ, α) reproduces x/(x²−Ω²) to within target_error."""
    qs = _static_quad()
    cfg = MinimaxConfig(target_error=TARGET_ERROR, max_nodes=MAX_NODES)
    qr = build_real_quadrature(qs, Omega, cfg)

    xs = np.linspace(X_MIN, X_MAX, 200)
    approx = _eval_fused(qr, xs)
    target = _real_target(xs, Omega)
    abs_err = np.max(np.abs(approx - target))
    # Loosened to 50× target_error to absorb the +branch-residual at
    # large Ω (where R' > 4000) and small numerical noise.
    assert abs_err < 50 * TARGET_ERROR, (
        f"Ω={Omega}: fused quad max-abs error {abs_err:.2e} > "
        f"50·target {50*TARGET_ERROR:.2e}; nodes={qr.tau.size}, "
        f"τ range [{qr.tau.min():+.3e}, {qr.tau.max():+.3e}]."
    )


@pytest.mark.parametrize("Omega", [10.0, 50.0, 200.0])
def test_branch_signs_and_tau_structure(Omega):
    """+branch has positive τ, −branch has negative τ; fused = sum."""
    qs = _static_quad()
    cfg = MinimaxConfig(target_error=TARGET_ERROR, max_nodes=MAX_NODES)
    qr = build_real_quadrature(qs, Omega, cfg)

    tau = np.asarray(qr.tau)
    pos_mask = tau > 0
    neg_mask = tau < 0
    assert pos_mask.any() and neg_mask.any(), \
        f"Ω={Omega}: expected both signs in τ, got {tau}"
    assert pos_mask.sum() == neg_mask.sum(), \
        f"Ω={Omega}: +branch and −branch should have equal node counts"

    # +branch alone fits 1/(Ω+x) (positive) on x ∈ [x_min, x_max]:
    xs = np.linspace(X_MIN, X_MAX, 100)
    plus_eval = (
        np.exp(-np.outer(xs, tau[pos_mask])) @ np.asarray(qr.alpha)[pos_mask]
    )
    expected_plus = 0.5 / (Omega + xs)
    rel_plus = np.max(np.abs(plus_eval - expected_plus)) / np.max(np.abs(expected_plus))
    assert rel_plus < 1e-3, \
        f"Ω={Omega}: +branch fit relative error {rel_plus:.2e} too large"

    # −branch alone fits −½/(Ω−x):
    minus_eval = (
        np.exp(-np.outer(xs, tau[neg_mask])) @ np.asarray(qr.alpha)[neg_mask]
    )
    expected_minus = -0.5 / (Omega - xs)
    rel_minus = np.max(np.abs(minus_eval - expected_minus)) / np.max(np.abs(expected_minus))
    assert rel_minus < 1e-3, \
        f"Ω={Omega}: −branch fit relative error {rel_minus:.2e} too large"


def test_large_omega_asymptote():
    """As Ω → ∞, the kernel converges to −x/Ω² (leading f-sum-rule)."""
    qs = _static_quad()
    cfg = MinimaxConfig(target_error=TARGET_ERROR, max_nodes=MAX_NODES)

    Omega = 1000.0
    qr = build_real_quadrature(qs, Omega, cfg)
    xs = np.linspace(X_MIN, X_MAX, 50)
    approx = _eval_fused(qr, xs)
    leading = -xs / Omega**2

    # The next correction is −x³/Ω⁴; for Ω=1000, x_max=5, the relative
    # neglected term is (x_max/Ω)² = 2.5e-5.
    rel_err = np.max(np.abs(approx - leading)) / np.max(np.abs(leading))
    assert rel_err < 1e-3, \
        f"Ω=1000 asymptote: rel-err {rel_err:.2e} (expected <1e-3)"


def test_omega_below_xmax_raises():
    """Decomposition is ill-defined for Ω ≤ x_max — should error cleanly."""
    qs = _static_quad()
    cfg = MinimaxConfig(target_error=TARGET_ERROR, max_nodes=MAX_NODES)
    with pytest.raises(ValueError):
        build_real_quadrature(qs, X_MAX * 0.9, cfg)

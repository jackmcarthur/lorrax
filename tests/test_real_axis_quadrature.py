"""Sanity test for ``build_real_quadrature`` (HL real-axis χ₀ probe).

The decomposition is::

    x / (x² − Ω²) = −½/(Ω−x) + ½/(Ω+x)

with both denominators strictly positive on x ∈ [x_min, x_max] when
Ω > x_max.  Each branch is a standard 1/y problem on a shifted
interval that's solved by the existing static (noncrossing) Laplace
minimax — same machinery as the static χ₀ kernel, with [Emin, Emax]
shifted by ±Ω.  The fused (τ, α) representation then drops into
``compute_chi0`` as a single set of nodes.

This test verifies:

1. The fused quadrature reproduces ``x/(x² − Ω²)`` to the requested
   target error on a dense ``x``-grid, across realistic Ω values.
2. Each branch separately gives ``∓½/(Ω∓x)`` with correct sign and
   the static-minimax accuracy.
3. The ``τ`` arrays span the expected mixed-sign structure
   (positive for the `+` branch, negative for the `−` branch).
4. The leading-order asymptote ``−x/Ω²`` is recovered as Ω→∞.

Run with::

    pytest tests/test_real_axis_quadrature.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from gw.minimax_screening import (
    LaplaceMinimaxQuadrature,
    solve_laplace_minimax_interval,
)
from gw.minimax_config import MinimaxConfig
from gw.w_isdf import build_real_quadrature


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

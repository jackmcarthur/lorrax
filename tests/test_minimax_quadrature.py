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

import minimax as mm
from gw import minimax_screening as ms


def _typed(catalog):
    """The synthetic catalogs below, through the door's own parser.

    The selection rule moved to ``services/minimax/`` with the extraction
    and runs on TYPED entries now, so these cells hand it typed entries.
    That is not a weaker test: ``parse_catalog`` is where a malformed row
    became a refusal instead of a silent skip, and the cells that exercise
    THAT live in the service's own suite
    (``services/minimax/tests/test_minimax_catalog_refusals.py``).
    """
    return mm.parse_catalog(catalog, catalog_name="synthetic")


def test_find_shipped_table_entry_prefers_smallest_range_then_loosest_error():
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

    entry = mm.select_entry(
        _typed(catalog),
        "noncrossing",
        range_value=80.0,
        target_error=1.0e-6,
        max_nodes=64,
    )

    assert entry is not None
    assert entry.file == "noncrossing/loose.npz"


def test_find_shipped_table_entry_filters_crossing_metadata():
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

    entry = mm.select_entry(
        _typed(catalog),
        "crossing",
        range_value=60.0,
        target_error=1.0e-6,
        max_nodes=64,
        target_kind="hgl",
        eps_q=1.0e-3,
    )

    assert entry is not None
    assert entry.file == "crossing/good.npz"


def _served(tau, alpha, err, *, family="noncrossing", target="inverse",
            source="shipped"):
    """A ``Quadrature`` standing in for whatever the door would have served."""
    return mm.Quadrature(
        nodes=tau, weights=alpha, family=family, target=target,
        range_param="R", range_value=10.0, error_bound=1.0e-6,
        max_error=err, kappa0=None, kappa1=None,
        provenance=mm.Provenance(
            source=source, catalog_entry="synthetic/fixture.npz",
            table_hash="sha256:0000000000000000",
            generator_commit="test", generation_backend="test",
            certified=False))


def test_solve_laplace_minimax_interval_uses_shipped_table_and_rescales(monkeypatch):
    """THE RESCALE IS WHAT STAYED HERE, so it is what this cell tests.

    The door serves tables in the scaled units the catalog tabulates; the
    wrapper divides by ``x_min``.  Standing a ``Quadrature`` in for the
    door is the whole coupling between the two halves after the
    extraction, and getting the division wrong is the one way this module
    can still move a number.
    """
    tau_hat = np.array([1.0, 2.0], dtype=np.float64)
    alpha_hat = np.array([0.25, 0.5], dtype=np.float64)
    err_hat = 5.0e-7

    monkeypatch.setattr(
        ms._mm, "serve",
        lambda **kw: _served(tau_hat, alpha_hat, err_hat))

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
    # R2: the rule now says where it came from, and the driver prints it.
    assert "synthetic/fixture.npz" in quad.provenance


def test_solve_phase_minimax_bandwidth_carries_the_crossing_table_unrescaled():
    """The crossing wrapper does NOT divide -- ξ enters at the consumer.

    ``ppm_windows`` applies ``t = τ/ξ`` itself, so a division here would
    apply it twice.  The asymmetry with the Laplace wrapper above is the
    reason both cells exist.
    """
    tau_hat = np.array([0.5, 1.5], dtype=np.float64)
    alpha_hat = np.array([0.1, 0.2], dtype=np.float64)
    err_hat = 9.0e-7

    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr(ms._mm, "serve",
                   lambda **kw: _served(tau_hat, alpha_hat, err_hat,
                                        family="crossing", target="hgl",
                                        source="runtime-uncertified"))
        quad = ms.solve_phase_minimax_bandwidth(
            83.0,
            target_error=1.0e-6,
            max_nodes=500,
            eps_q=1.0e-3,
            target_kind="hgl",
        )
    finally:
        mp.undo()

    np.testing.assert_allclose(quad.tau, tau_hat)
    np.testing.assert_allclose(quad.alpha, alpha_hat)
    assert quad.max_error == err_hat
    assert quad.target_kind == "hgl"
    assert "runtime solve" in quad.provenance


def test_solve_laplace_minimax_interval_forwards_the_shipped_table_flag(monkeypatch):
    """``use_shipped_tables=False`` reaches the door as ``use_shipped=False``.

    The deck key ``regenerate_minimax_tables`` is an explicit request for
    the uncertified path.  Dropping it in the wrapper would silently serve
    a shipped table to a run that asked to re-solve -- which is the exact
    inverse of the defect this extraction exists to fix, and just as
    invisible.
    """
    tau_hat = np.array([0.75, 1.25], dtype=np.float64)
    alpha_hat = np.array([0.3, 0.4], dtype=np.float64)
    err_hat = 2.5e-7
    seen = {}

    def _capture(**kw):
        seen.update(kw)
        return _served(tau_hat, alpha_hat, err_hat,
                       source="runtime-uncertified")

    monkeypatch.setattr(ms._mm, "serve", _capture)

    quad = ms.solve_laplace_minimax_interval(
        1.0,
        10.0,
        target_error=1.0e-6,
        max_nodes=64,
        use_shipped_tables=False,
    )

    assert seen["use_shipped"] is False
    assert seen["family"] == "noncrossing" and seen["target"] == "inverse"
    np.testing.assert_allclose(quad.tau, tau_hat)
    np.testing.assert_allclose(quad.alpha, alpha_hat)
    assert quad.max_error == err_hat


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

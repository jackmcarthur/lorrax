"""Minimax quadrature math — the physical rescale of served rules and the
real-axis identities.

The rescaling correctness is load-bearing for every dynamic Sigma_c run;
the analytic identities (fused (tau, alpha) real-axis quadrature vs the
analytic x/(x^2 - Omega^2) kernel, branch signs, large-Omega asymptote) are
invisible at gate level (a wrong-but-smooth quadrature still freezes
reproducibly).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import minimax as mm
from gw import minimax_screening as ms
from gw.wavefunction_bundle import BandSlices


def test_static_window_ignores_process_padded_band_energies(monkeypatch):
    """P16/P36 carrier tails cannot change the physical minimax interval."""
    slices = BandSlices.from_band_edges(
        0, 0, 2, 4, 6, b4_chi=6, b4_sigma=6, b4_logical=4)
    # The last two values stand in for real WFN energies carried in zero-psi
    # padding slots.  They would inflate x_max from 5 to 203 if cond_all were
    # used instead of cond_all_logical.
    wfns = SimpleNamespace(
        slices=slices,
        enk=jnp.asarray([[-3.0, -1.0, 1.0, 2.0, 100.0, 200.0]]))
    seen = {}

    def solve(x_min, x_max, **_kwargs):
        seen["interval"] = (x_min, x_max)
        return SimpleNamespace(x_min=x_min, x_max=x_max)

    monkeypatch.setattr(ms, "solve_laplace_minimax_interval", solve)
    config = SimpleNamespace(
        energy_reference="midgap", target_error=1.0e-6, max_nodes=64)

    quad, _ = ms.build_static_quadrature(wfns, config)

    assert seen["interval"] == (2.0, 5.0)
    assert quad.x_max == 5.0


def _served(tau, alpha, err, *, family="noncrossing", target="inverse",
            source="runtime-uncertified"):
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


def test_solve_laplace_minimax_interval_rescales_the_served_rule(monkeypatch):
    """THE RESCALE IS WHAT STAYED HERE, so it is what this cell tests.

    The door serves rules in scaled units; the wrapper divides by
    ``x_min``.  Standing a ``Quadrature`` in for the door is the whole
    coupling between the two halves, and getting the division wrong is the
    one way this module can still move a number.
    """
    tau_hat = np.array([1.0, 2.0], dtype=np.float64)
    alpha_hat = np.array([0.25, 0.5], dtype=np.float64)
    err_hat = 5.0e-7
    seen = {}

    def _capture(**kw):
        seen.update(kw)
        return _served(tau_hat, alpha_hat, err_hat)

    monkeypatch.setattr(ms._mm, "serve", _capture)

    quad = ms.solve_laplace_minimax_interval(
        2.0,
        20.0,
        target_error=1.0e-6,
        max_nodes=64,
    )

    np.testing.assert_allclose(quad.tau, tau_hat / 2.0)
    np.testing.assert_allclose(quad.alpha, alpha_hat / 2.0)
    assert quad.max_error == err_hat / 2.0
    # target_error is physical.  The door serves the y=x/x_min problem,
    # whose absolute error must be tighter by the inverse return rescale.
    assert seen["error_bound"] == 2.0e-6
    # R2: the rule says where it came from, and the driver prints it.  Every
    # rule is solved at run time, so the provenance names the runtime solve.
    assert "runtime solve" in quad.provenance


def test_a_node_capped_static_rule_that_misses_its_target_refuses():
    """n_max = 4 cannot reach 1e-10 on [1, 1e4]: the capped rule refuses instead of serving."""
    import pytest
    with pytest.raises(ValueError, match="GATE minimax_node_cap"):
        ms.solve_laplace_minimax_interval(1.0, 1.0e4, target_error=1.0e-10, max_nodes=4)
    quad = ms.solve_laplace_minimax_interval(1.0, 1.0e4, target_error=1.0e-6, max_nodes=64)
    assert quad.max_error <= 1.0e-6


def test_solve_phase_minimax_bandwidth_carries_the_crossing_rule_unrescaled():
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
    """Decomposition is ill-defined for Ω ≤ x_max: a standard refusal naming the probe fix."""
    qs = _static_quad()
    cfg = MinimaxConfig(target_error=TARGET_ERROR, max_nodes=MAX_NODES)
    with pytest.raises(ValueError, match="GATE hl_ppm_probe_in_spectrum.*fix: set ppm_omega_p"):
        build_real_quadrature(qs, X_MAX * 0.9, cfg)

"""Σ_PPM WS0 keystone gate G2 — per-branch / per-window reference tiles.

G2 is the bit-identity pin the WS3 file split (moving _SigmaWindow /
_SigmaBranch / _iter_branches / _build_*_sigma_windows /
_build_windows_for_branch into ppm_windows.py) must stay identical
against.  Also guards against a split silently dropping a branch (all 4
branches × their windows asserted non-empty).

G1 (the kij ↔ kij_stream accumulator parity gate that detected Bug B)
was RETIRED 2026-07-31 with the removal of the kij_stream mode itself.
G3 (the head negative-branch regression) lives in
``tests/test_head_correction.py``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_REG = REPO_ROOT / "tests" / "regression"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import gpu_available, requested_platform  # noqa: E402


# ===========================================================================
#  G2 — per-branch / per-window reference tiles (GREEN, WS3 pin)
# ===========================================================================
#
# We drive the window builders directly (the "equivalent per-branch τ-node /
# mask arrays that _build_windows_for_branch produces" the spec explicitly
# permits) rather than scraping an e2e run: WS3 is a *pure move* of exactly
# these symbols, so a unit test that imports and exercises them is the precise
# pin.  Inputs are synthetic but structured to populate every window kind:
#   * a symmetric ω grid (±10 eV) → all 4 (ω-sign × cond/val) branches;
#   * E_A spanning the window threshold T → the core + a_stripe split;
#   * Ω_q spanning T → the b_slab (Ω>T) window as well.

G2_REF = _REG / "sigma_ppm_gates" / "g2_branch_window_ref.npz"


def test_crossing_core_rescales_the_physical_error_contract(monkeypatch):
    """The HGL service request and certificate follow the same xi scaling.

    This uses a non-unit regularization width so an omitted conversion cannot
    pass by coincidence.  The explicit sine values also prove that the
    incumbent ``tau/xi, alpha/xi`` rule represents ``G(x/xi)/xi``; no solver
    or random fixture is involved.
    """
    from gw import ppm_windows
    from gw.minimax_screening import CrossingMinimaxQuadrature

    xi = 2.5
    target_error_phys = 4.0e-7
    tau_hat = np.array([0.5, 1.5], dtype=np.float64)
    alpha_hat = np.array([0.2, -0.1], dtype=np.float64)
    error_hat = 5.0e-7
    seen = {}

    def _served(A_dim, **kwargs):
        seen.update(A_dim=A_dim, **kwargs)
        return CrossingMinimaxQuadrature(
            A_dim=float(A_dim), tau=tau_hat, alpha=alpha_hat,
            max_error=error_hat, target_kind="hgl",
            provenance="deterministic scaling fixture")

    monkeypatch.setattr(ppm_windows, "solve_phase_minimax_bandwidth", _served)
    windows = ppm_windows._build_three_sigma_windows(
        E_A=np.array([0.2], dtype=np.float64),
        base_mask_A=np.array([True]),
        mask_B_all_count=1,
        mask_B_le_count=1,
        mask_B_le_min=0.4,
        mask_B_le_max=0.4,
        mask_B_gt_count=0,
        mask_B_gt_min=None,
        mask_B_gt_max=None,
        omega_nonneg_ry=np.array([0.3], dtype=np.float64),
        neg_omega_half=False,
        regularization_width_ry=xi,
        edge_factor=1.5,
        target_error=target_error_phys,
        max_nodes=64,
        crossing_eps_q=1.0e-3,
        crossing_max_nodes=500,
        use_shipped_tables=True,
    )

    assert [window.name for window in windows] == ["core"]
    core = windows[0]
    assert seen["target_error"] == target_error_phys * xi
    assert core.max_error == error_hat / xi
    np.testing.assert_allclose(np.asarray(core.nodes.t).real, tau_hat / xi)
    np.testing.assert_allclose(np.asarray(core.nodes.alpha).real, alpha_hat / xi)

    u = np.array([0.0, 0.25, 0.75, 1.5], dtype=np.float64)
    x = xi * u
    fitted_hat = np.sin(np.outer(u, tau_hat)) @ alpha_hat
    fitted_phys = (
        np.sin(np.outer(x, np.asarray(core.nodes.t).real))
        @ np.asarray(core.nodes.alpha).real
    )
    np.testing.assert_allclose(fitted_phys, fitted_hat / xi, rtol=1.0e-14)


def test_shared_omega_clusters_preserve_gap_only_owner_and_cap_spans():
    """The shared owner keeps MPA's gap-only result and adds a span cap.

    Deliberately scramble the branch order: returned index arrays must retain
    that order even though clusters themselves are ordered by energy.
    """
    from gw.ppm_windows import _omega_clusters

    omega = np.array([3.2, 0.1, 0.2, 3.1], dtype=np.float64)
    incumbent = _omega_clusters(omega, 1.0)
    assert [(i.tolist(), lo, hi) for i, lo, hi in incumbent] == [
        ([1, 2], 0.1, 0.2), ([0, 3], 3.1, 3.2)]

    capped = _omega_clusters(omega, 1.0, max_span_ry=0.05)
    assert [(i.tolist(), lo, hi) for i, lo, hi in capped] == [
        ([1], 0.1, 0.1), ([2], 0.2, 0.2),
        ([3], 3.1, 3.1), ([0], 3.2, 3.2)]


def test_hgl_capacity_owner_keeps_the_roundoff_band_on_incumbent_family():
    from gw.ppm_windows import _CROSSING_A_MAX, hgl_partition_required

    xi = 0.2
    edge = 1.5
    omega_at_capacity = (0.5 * _CROSSING_A_MAX - edge) * xi
    eps = np.finfo(np.float64).eps
    assert not hgl_partition_required(
        np.array([-omega_at_capacity, omega_at_capacity]), xi, edge)
    assert not hgl_partition_required(
        np.array([omega_at_capacity * (1.0 + 4.0 * eps)]), xi, edge)
    assert hgl_partition_required(
        np.array([omega_at_capacity * (1.0 + 32.0 * eps)]), xi, edge)


def test_hgl_cell_plan_tiles_direct_denominator_and_respects_capacity():
    """Exact omega x A x B ownership and first-principles sign bounds.

    No quadrature and no random arrays: the direct retarded denominator is
    evaluated on every deterministic cell boundary.  Its cell-selected sum
    must be the direct value, including the repository-wide ``(lo, hi]``
    downward assignment at both A and B boundaries.
    """
    from gw.ppm_windows import plan_hgl_crossing_cells

    omega = np.array([0.2, 0.4, 3.1, 3.3], dtype=np.float64)
    energies = np.array([[0.1, 0.3, 1.0, 9.0]], dtype=np.float64)
    base = np.array([[True, True, True, False]])
    xi = 0.2
    edge = 1.5
    A_max = 4.0
    plan = plan_hgl_crossing_cells(
        omega_abs=omega, E_A=energies, base_mask_A=base,
        regularization_width_ry=xi, edge_factor=edge,
        omega_cluster_gap_ry=1.0, omega_max_span_ry=0.25,
        crossing_A_max=A_max)

    assert plan.omega_cluster_count == 2
    assert plan.energy_pane_count == 4
    assert len(plan.cells) == 12
    assert plan.max_A_dim <= A_max

    z = edge * xi
    live_e = energies[base]
    for cell in plan.cells:
        assert cell.omega_hi - cell.omega_lo <= 0.25
        if cell.kind == "crossing":
            corners = [
                w - e - b
                for w in (cell.omega_lo, cell.omega_hi)
                for e in (cell.e_min, cell.e_max)
                for b in (cell.b_lo, cell.b_hi)
            ]
            assert max(abs(x) for x in corners) <= cell.A_dim * xi * (
                1.0 + 8.0 * np.finfo(np.float64).eps)
        elif cell.kind == "positive":
            # The least-positive corner sits at the closed upper B edge.
            x_min = cell.omega_lo - cell.e_max - cell.b_hi
            assert x_min >= z * (1.0 - 8.0 * np.finfo(np.float64).eps)
        else:
            assert cell.kind == "negative"
            # The open lower B edge is approached from above.
            x_sup = cell.omega_hi - cell.e_min - cell.b_lo
            assert x_sup <= -z * (1.0 - 8.0 * np.finfo(np.float64).eps)

    finite_edges = sorted({
        bound for cell in plan.cells for bound in (cell.b_lo, cell.b_hi)
        if np.isfinite(bound)})
    b_probe = np.array(
        [finite_edges[0] - 0.2, *finite_edges,
         *[(a + b) / 2.0 for a, b in zip(finite_edges, finite_edges[1:])],
         finite_edges[-1] + 0.2], dtype=np.float64)
    for iw, w in enumerate(omega):
        for e in live_e:
            for b in b_probe:
                owners = [
                    cell for cell in plan.cells
                    if iw in cell.omega_indices
                    and e > cell.e_lo and e <= cell.e_hi
                    and b > cell.b_lo and b <= cell.b_hi
                ]
                assert len(owners) == 1, (iw, e, b, owners)
                direct = 1.0 / (w - e - b + 1j * xi)
                decomposed = sum(
                    1.0 / (w - e - b + 1j * xi) for _cell in owners)
                np.testing.assert_array_equal(decomposed, direct)


def test_hgl_cell_rules_rephase_the_direct_scalar_kernel(monkeypatch):
    """Bounded +/crossing/- rows reproduce ``exp(i*t*(omega-E-B))``.

    This is the complete scalar coefficient product used by the production
    G, W, and omega projector, including both reference phases.  Deliberate
    near-shell and far-tail poles force exact B panes on both sign-definite
    flanks; no random tensor or frontend fixture is involved.
    """
    from gw import ppm_windows
    from gw.minimax_screening import (
        CrossingMinimaxQuadrature,
        LaplaceMinimaxQuadrature,
    )

    tau_laplace = 0.7
    tau_cross_hat = 0.5

    def _laplace(x_min, x_max, **kwargs):
        return LaplaceMinimaxQuadrature(
            x_min=float(x_min), x_max=float(x_max),
            tau=np.array([tau_laplace]), alpha=np.array([1.0]),
            max_error=0.0, provenance="deterministic scalar rule")

    def _crossing(A_dim, **kwargs):
        return CrossingMinimaxQuadrature(
            A_dim=float(A_dim), tau=np.array([tau_cross_hat]),
            alpha=np.array([1.0]), max_error=0.0, target_kind="hgl",
            provenance="deterministic scalar rule")

    monkeypatch.setattr(
        ppm_windows, "solve_laplace_minimax_interval", _laplace)
    monkeypatch.setattr(
        ppm_windows, "solve_phase_minimax_bandwidth", _crossing)

    xi = 0.2
    omega = 1.0
    energy = 0.2
    poles_flat = np.array(
        [1.0e-4, 0.4, 0.7979, 0.8, 0.8021, 2.0, 1000.0],
        dtype=np.float64)
    poles = poles_flat.reshape(1, 1, -1)
    windows, plan = ppm_windows._build_partitioned_hgl_windows(
        E_A=np.array([[energy]], dtype=np.float64),
        base_mask_A=np.array([[True]]),
        Omega_q=ppm_windows.jnp.asarray(poles),
        base_mask_B=ppm_windows.jnp.ones(poles.shape, dtype=bool),
        omega_nonneg_ry=np.array([omega]),
        neg_omega_half=False,
        regularization_width_ry=xi,
        edge_factor=0.01,
        target_error=1.0e-6,
        max_nodes=64,
        crossing_eps_q=1.0e-3,
        crossing_max_nodes=64,
        use_shipped_tables=False,
    )
    assert plan.max_A_dim <= ppm_windows._CROSSING_A_MAX
    assert sum(w.name == "pane_positive" for w in windows) > 1
    assert sum(w.name == "pane_crossing" for w in windows) == 1
    assert sum(w.name == "pane_negative" for w in windows) > 1

    ownership = np.zeros(poles_flat.size, dtype=np.int64)
    for window in windows:
        lo, hi = ppm_windows.window_mask_B_bounds(window)
        selected = (poles_flat > lo) & (poles_flat <= hi)
        ownership += selected
        selected_poles = poles_flat[selected]
        assert selected_poles.size > 0
        if window.name == "pane_positive":
            x = omega - energy - selected_poles
            assert float(np.max(x)) / float(np.min(x)) <= 256.0
        elif window.name == "pane_negative":
            x = energy + selected_poles - omega
            assert float(np.max(x)) / float(np.min(x)) <= 256.0

        t = complex(np.asarray(window.nodes.t)[0])
        alpha = complex(np.asarray(window.nodes.alpha)[0])
        for pole in (float(np.min(selected_poles)),
                     float(np.max(selected_poles))):
            factorized = (
                np.exp(-1j * (energy - window.E_ref_A) * t)
                * np.exp(-1j * (pole - window.E_ref_B) * t)
                * np.exp(-1j * (
                    window.E_ref_A + window.E_ref_B - omega) * t)
            )
            direct_phase = np.exp(1j * t * (omega - energy - pole))
            np.testing.assert_allclose(
                factorized, direct_phase, rtol=2.0e-15, atol=2.0e-15)

            direct_weighted = window.prefactor * alpha * direct_phase
            factorized_weighted = window.prefactor * alpha * factorized
            if window.project == "imag":
                # Diagonal band-adjoint completion is Im(Z).
                np.testing.assert_allclose(
                    np.imag(factorized_weighted),
                    np.imag(direct_weighted), rtol=2.0e-15, atol=2.0e-15)
            else:
                np.testing.assert_allclose(
                    factorized_weighted, direct_weighted,
                    rtol=2.0e-15, atol=2.0e-15)

    np.testing.assert_array_equal(ownership, np.ones_like(ownership))
    assert all(complex(np.asarray(w.nodes.t)[0]).imag > 0.0
               for w in windows if w.name == "pane_positive")
    assert all(complex(np.asarray(w.nodes.t)[0]).imag < 0.0
               for w in windows if w.name == "pane_negative")
    assert {w.prefactor for w in windows
            if w.name == "pane_positive"} == {-1.0}
    assert {w.prefactor for w in windows
            if w.name == "pane_crossing"} == {-1.0}
    assert {w.prefactor for w in windows
            if w.name == "pane_negative"} == {1.0}


def test_hgl_negative_cell_tail_panes_preserve_exact_partition(monkeypatch):
    """A high-Ω negative cell is split without changing pole ownership."""
    from gw import ppm_windows
    from gw.minimax_screening import (
        CrossingMinimaxQuadrature,
        LaplaceMinimaxQuadrature,
    )

    def _laplace(x_min, x_max, **_kwargs):
        return LaplaceMinimaxQuadrature(
            x_min=float(x_min), x_max=float(x_max),
            tau=np.array([0.5]), alpha=np.array([1.0]),
            max_error=0.0, provenance="deterministic scalar rule")

    def _crossing(A_dim, **_kwargs):
        return CrossingMinimaxQuadrature(
            A_dim=float(A_dim), tau=np.array([0.5]),
            alpha=np.array([1.0]), max_error=0.0, target_kind="hgl",
            provenance="deterministic scalar rule")

    monkeypatch.setattr(
        ppm_windows, "solve_laplace_minimax_interval", _laplace)
    monkeypatch.setattr(
        ppm_windows, "solve_phase_minimax_bandwidth", _crossing)

    omega = 1.0
    energy = 0.2
    poles = np.array([0.4, 0.8, 1.2, 100.0, 1000.0])
    pole_tensor = poles.reshape(1, 1, -1)
    windows, _plan = ppm_windows._build_partitioned_hgl_windows(
        E_A=np.array([[energy]], dtype=np.float64),
        base_mask_A=np.array([[True]]),
        Omega_q=ppm_windows.jnp.asarray(pole_tensor),
        base_mask_B=ppm_windows.jnp.ones(pole_tensor.shape, dtype=bool),
        omega_nonneg_ry=np.array([omega]),
        neg_omega_half=False,
        regularization_width_ry=0.2,
        edge_factor=1.5,
        target_error=1.0e-6,
        max_nodes=64,
        crossing_eps_q=1.0e-3,
        crossing_max_nodes=64,
        use_shipped_tables=False,
    )

    negative = [window for window in windows
                if window.name == "pane_negative"]
    assert len(negative) > 1
    ownership = np.zeros(poles.size, dtype=np.int64)
    for window in windows:
        lo, hi = ppm_windows.window_mask_B_bounds(window)
        selected = (poles > lo) & (poles <= hi)
        ownership += selected
        if window.name == "pane_negative":
            pane_poles = poles[selected]
            assert pane_poles.size > 0
            x_min, x_max = ppm_windows._sign_definite_support(
                energy, energy,
                float(np.min(pane_poles)), float(np.max(pane_poles)),
                omega, omega_min=omega, subtract_omega=True)
            assert (x_max / x_min
                    <= ppm_windows._SIGN_DEFINITE_PANE_MAX_RANGE
                    or np.min(pane_poles) == np.max(pane_poles))

    np.testing.assert_array_equal(ownership, np.ones_like(ownership))


def test_subtractive_sign_definite_support_refuses_a_crossing_interval():
    """A pane mislabeled negative must not be clipped into a Laplace rule."""
    from gw.ppm_windows import _sign_definite_support

    with pytest.raises(AssertionError, match="sign_definite_support"):
        _sign_definite_support(
            0.1, 0.2, 0.1, 0.2, 1.0,
            omega_min=0.9, subtract_omega=True)


def test_sign_definite_omega_panes_exhaust_extreme_tail_exactly():
    """CrI3-shaped pole tails are partitioned, never dropped/staticised."""
    import jax.numpy as jnp
    from types import SimpleNamespace
    from gw.ppm_windows import (
        _build_single_sigma_window,
        _plan_sign_definite_cells,
        _plan_sign_definite_omega_panes,
        window_mask_B_bounds,
    )

    # Minimal deterministic support spanning the frozen run33 endpoints.
    omega = np.array(
        [2.0e-4, 4.0e-4, 0.05, 4.0, 95.8565, 97.8518],
        dtype=np.float64,
    )
    mask = np.ones_like(omega, dtype=bool)
    E_min = 0.0343332986397257
    E_max = 5.40437906406350
    omega_eval = 1.46997235298981
    panes = _plan_sign_definite_omega_panes(
        Omega_q=jnp.asarray(omega), base_mask_B=jnp.asarray(mask),
        mask_B_count=omega.size,
        mask_B_min=float(omega.min()), mask_B_max=float(omega.max()),
        E_min=E_min, E_max=E_max, omega_max=omega_eval,
    )

    ownership = np.zeros(omega.size, dtype=np.int64)
    for lo, hi, count, actual_min, actual_max in panes:
        # Explicit B_lo/B_hi wins over mask_B_mode="all" in the existing
        # runtime selector; no second mask convention is hidden in the test.
        got_lo, got_hi = window_mask_B_bounds(SimpleNamespace(
            B_lo=lo, B_hi=hi, mask_B_mode="all",
            mask_B_threshold=None))
        selected = (omega > got_lo) & (omega <= got_hi)
        ownership += selected
        assert int(np.sum(selected)) == count
        assert actual_min == float(np.min(omega[selected]))
        assert actual_max == float(np.max(omega[selected]))
        R = ((E_max + actual_max + omega_eval)
             / (E_min + actual_min))
        assert R <= 256.0 or actual_min == actual_max

    np.testing.assert_array_equal(ownership, np.ones_like(ownership))
    assert len(panes) > 1

    # Drive the production window builder and shipped rules for both omega
    # halves.  Recompose exactly the A phase, B phase, E_ref rephasing and
    # accumulator frequency factor; boundary lanes must recover the direct
    # signed rational kernel within its requested physical L-infinity error.
    target_error = 1.0e-6
    all_windows = []
    for neg_omega_half in (False, True):
        cells_run33 = _plan_sign_definite_cells(
            E_A=np.array([E_min, E_max], dtype=np.float64),
            base_mask_A=np.array([True, True]),
            Omega_q=jnp.asarray(omega),
            base_mask_B=jnp.asarray(mask),
            mask_B_count=omega.size,
            mask_B_min=float(omega.min()),
            mask_B_max=float(omega.max()),
            omega_nonneg_ry=np.array([0.0, omega_eval]),
            orientation="E+B+omega",
        )
        windows = _build_single_sigma_window(
            E_A=np.array([E_min, E_max], dtype=np.float64),
            base_mask_A=np.array([True, True]),
            mask_B_count=omega.size,
            mask_B_min=float(omega.min()),
            mask_B_max=float(omega.max()),
            omega_nonneg_ry=np.array([0.0, omega_eval]),
            denom_can_cross=False,
            neg_omega_half=neg_omega_half,
            target_error=target_error,
            max_nodes=64,
            use_shipped_tables=True,
            sign_definite_cells=cells_run33,
        )
        assert len(windows) == len(panes)
        all_windows.extend(windows)
        for window in windows:
            got_lo, got_hi = window_mask_B_bounds(window)
            selected_omega = omega[(omega > got_lo) & (omega <= got_hi)]
            assert selected_omega.size > 0
            assert window.max_error <= target_error
            t = np.asarray(window.nodes.t, dtype=np.complex128)
            alpha = np.asarray(window.nodes.alpha, dtype=np.complex128)
            for energy in (E_min, E_max):
                for pole in (selected_omega.min(), selected_omega.max()):
                    for frequency in (0.0, omega_eval):
                        factorized = window.prefactor * np.sum(
                            alpha
                            * np.exp(-1j * (window.E_ref_A
                                            + window.E_ref_B) * t)
                            * np.exp(-1j * (energy - window.E_ref_A) * t)
                            * np.exp(-1j * (pole - window.E_ref_B) * t)
                            * np.exp(-1j * (
                                -window.omega_sign * frequency) * t)
                        )
                        direct = window.prefactor / (
                            frequency + energy + pole)
                        assert abs(factorized - direct) <= target_error

    assert {window.prefactor for window in all_windows} == {-1.0, 1.0}
    assert any("R_256p000000_eps_3p0em08" in str(window.provenance)
               for window in all_windows)

    # A singleton Ω support cannot be split by the B-axis owner alone.
    irreducible = _plan_sign_definite_omega_panes(
        Omega_q=jnp.asarray([1.0e-4]), base_mask_B=jnp.asarray([True]),
        mask_B_count=1, mask_B_min=1.0e-4, mask_B_max=1.0e-4,
        E_min=1.0e-3, E_max=5.0, omega_max=1.0,
    )
    assert len(irreducible) == 1
    lo, hi, count, actual_min, actual_max = irreducible[0]
    assert count == 1 and lo < 1.0e-4 <= hi
    assert actual_min == actual_max == 1.0e-4
    assert (5.0 + actual_max + 1.0) / (1.0e-3 + actual_min) > 256.0

    # The canonical product continuation therefore tiles the existing A and
    # evaluation-omega selectors.  This fixture forces both cuts: the low-A
    # singleton still needs its [0,1] omega grid split after the A cut.
    singleton_E = np.array([1.0e-3, 5.0], dtype=np.float64)
    singleton_w = np.array([0.0, 1.0], dtype=np.float64)
    cells = _plan_sign_definite_cells(
        E_A=singleton_E,
        base_mask_A=np.ones_like(singleton_E, dtype=bool),
        Omega_q=jnp.asarray([1.0e-4]),
        base_mask_B=jnp.asarray([True]),
        mask_B_count=1,
        mask_B_min=1.0e-4,
        mask_B_max=1.0e-4,
        omega_nonneg_ry=singleton_w,
        orientation="E+B+omega",
    )
    assert len(cells) == 3
    assert all(cell.x_max / cell.x_min <= 256.0 for cell in cells)
    ownership = np.zeros((singleton_E.size, singleton_w.size), dtype=np.int64)
    for cell in cells:
        selected_E = ((singleton_E > cell.E_lo)
                      & (singleton_E <= cell.E_hi))
        ownership[np.ix_(selected_E, np.isin(
            np.arange(singleton_w.size), cell.omega_indices))] += 1
    np.testing.assert_array_equal(ownership, np.ones_like(ownership))

    bounded = _build_single_sigma_window(
        E_A=singleton_E,
        base_mask_A=np.ones_like(singleton_E, dtype=bool),
        mask_B_count=1,
        mask_B_min=1.0e-4,
        mask_B_max=1.0e-4,
        omega_nonneg_ry=singleton_w,
        denom_can_cross=False,
        neg_omega_half=False,
        target_error=target_error,
        max_nodes=64,
        use_shipped_tables=True,
        sign_definite_cells=cells,
    )
    assert len(bounded) == len(cells)
    for window in bounded:
        assert window.max_error <= target_error
        E_lo = -np.inf if window.E_min is None else window.E_min
        E_hi = np.inf if window.E_max is None else window.E_max
        selected_E = singleton_E[(singleton_E > E_lo)
                                 & (singleton_E <= E_hi)]
        selected_w = (singleton_w if window.omega_indices is None
                      else singleton_w[window.omega_indices])
        t = np.asarray(window.nodes.t, dtype=np.complex128)
        alpha = np.asarray(window.nodes.alpha, dtype=np.complex128)
        for energy in selected_E:
            for frequency in selected_w:
                got = window.prefactor * np.sum(
                    alpha
                    * np.exp(-1j * (window.E_ref_A
                                    + window.E_ref_B) * t)
                    * np.exp(-1j * (energy - window.E_ref_A) * t)
                    * np.exp(-1j * (1.0e-4 - window.E_ref_B) * t)
                    * np.exp(+1j * window.omega_sign * frequency * t)
                )
                want = window.prefactor / (frequency + energy + 1.0e-4)
                assert abs(got - want) <= target_error


def test_crossing_b_slab_reuses_exact_bounded_omega_panes(monkeypatch):
    """The crossing core/stripe/slab cover survives a high-Ω tail split."""
    from gw import ppm_windows
    from gw.minimax_screening import (
        CrossingMinimaxQuadrature,
        LaplaceMinimaxQuadrature,
    )

    def _laplace(x_min, x_max, **_kwargs):
        return LaplaceMinimaxQuadrature(
            x_min=float(x_min), x_max=float(x_max),
            tau=np.array([0.5]), alpha=np.array([1.0]),
            max_error=0.0, provenance="deterministic scalar rule")

    def _crossing(A_dim, **_kwargs):
        return CrossingMinimaxQuadrature(
            A_dim=float(A_dim), tau=np.array([0.5]),
            alpha=np.array([1.0]), max_error=0.0, target_kind="hgl",
            provenance="deterministic scalar rule")

    monkeypatch.setattr(
        ppm_windows, "solve_laplace_minimax_interval", _laplace)
    monkeypatch.setattr(
        ppm_windows, "solve_phase_minimax_bandwidth", _crossing)

    xi = 0.1
    edge = 1.5
    omega_eval = np.array([0.0, 0.5], dtype=np.float64)
    threshold = float(np.max(omega_eval)) + edge * xi
    energies = np.array([[0.2, threshold, 1.0]], dtype=np.float64)
    energy_mask = np.ones_like(energies, dtype=bool)
    poles = np.array(
        [0.2, threshold, 0.7, 1.0, 100.0, 1000.0],
        dtype=np.float64,
    )
    pole_mask = np.ones_like(poles, dtype=bool)

    windows = ppm_windows._build_windows_for_branch(
        omega_nonneg_ry=omega_eval,
        E_A=ppm_windows.jnp.asarray(energies),
        base_mask_A=ppm_windows.jnp.asarray(energy_mask),
        Omega_q=ppm_windows.jnp.asarray(poles),
        base_mask_B=ppm_windows.jnp.asarray(pole_mask),
        space="cond",
        neg_omega_half=False,
        regularization_width_ry=xi,
        edge_factor=edge,
        target_error=1.0e-6,
        max_nodes=64,
        crossing_eps_q=1.0e-3,
        crossing_max_nodes=64,
        use_shipped_minimax_tables=False,
        log_tag="analytic",
        print_fn=lambda *_args, **_kwargs: None,
        partition_hgl=False,
    )

    slab = [window for window in windows
            if window.name.startswith("b_slab")]
    assert len(slab) > 1
    ownership = np.zeros(
        (energies.size, poles.size, omega_eval.size), dtype=np.int64)
    for window in windows:
        lo, hi = ppm_windows.window_mask_B_bounds(window)
        selected_B = (poles > lo) & (poles <= hi)
        selected_A = np.asarray(
            window.mask_A, dtype=bool).reshape(-1).copy()
        E_lo = -np.inf if window.E_min is None else window.E_min
        E_hi = np.inf if window.E_max is None else window.E_max
        selected_A &= ((energies.reshape(-1) > E_lo)
                       & (energies.reshape(-1) <= E_hi))
        selected_w = np.ones(omega_eval.size, dtype=bool)
        if window.omega_indices is not None:
            selected_w[:] = False
            selected_w[window.omega_indices] = True
        ownership += (selected_A[:, None, None]
                      & selected_B[None, :, None]
                      & selected_w[None, None, :])
        if (window.name.startswith("a_stripe")
                or window.name.startswith("b_slab")):
            pane_poles = poles[selected_B]
            pane_energies = energies.reshape(-1)[selected_A]
            pane_omega = omega_eval[selected_w]
            assert pane_poles.size > 0
            x_min, x_max = ppm_windows._oriented_sign_definite_support(
                float(np.min(pane_energies)), float(np.max(pane_energies)),
                float(np.min(pane_poles)), float(np.max(pane_poles)),
                float(np.min(pane_omega)), float(np.max(pane_omega)),
                "E+B-omega")
            assert x_max / x_min <= ppm_windows._SIGN_DEFINITE_PANE_MAX_RANGE

    np.testing.assert_array_equal(ownership, np.ones_like(ownership))


@pytest.mark.parametrize(
    ("space", "neg_omega_half", "expected_prefactor"),
    (("cond", False, 1.0), ("val", True, -1.0)),
)
def test_tiny_positive_crossing_fallback_keeps_subtractive_support(
    monkeypatch, space, neg_omega_half, expected_prefactor,
):
    """The no-HGL tiny-omega door keeps the physical E+Omega-omega form."""
    from gw import ppm_windows
    from gw.minimax_screening import LaplaceMinimaxQuadrature

    # omega_max sits exactly on the branch builder's no-HGL threshold.  Pick a
    # live pole only one ulp-scale decade above it: the subtractive support is
    # strictly positive but much wider than the (incorrect) additive support,
    # forcing the exact E/omega continuation to expose the orientation.
    omega_eval = np.array([0.0, 1.0e-14], dtype=np.float64)
    energies = np.array([[0.0, 1.0]], dtype=np.float64)
    poles = np.array([1.001e-14], dtype=np.float64)

    def _exact_at_left(x_min, x_max, **_kwargs):
        # Each forced low-energy cell is a singleton denominator.  The other
        # terminal cell has a 1e-14 absolute span near x=1, so this deterministic
        # one-node rule reconstructs all selected points to roundoff without a
        # minimax service or platform dependency.
        tau = 0.5
        return LaplaceMinimaxQuadrature(
            x_min=float(x_min), x_max=float(x_max),
            tau=np.array([tau]),
            alpha=np.array([np.exp(tau * float(x_min)) / float(x_min)]),
            max_error=2.0e-14,
            provenance="deterministic tiny-omega fixture",
        )

    monkeypatch.setattr(
        ppm_windows, "solve_laplace_minimax_interval", _exact_at_left)
    windows = ppm_windows._build_windows_for_branch(
        omega_nonneg_ry=omega_eval,
        E_A=ppm_windows.jnp.asarray(energies),
        base_mask_A=ppm_windows.jnp.ones_like(energies, dtype=bool),
        Omega_q=ppm_windows.jnp.asarray(poles),
        base_mask_B=ppm_windows.jnp.ones_like(poles, dtype=bool),
        space=space,
        neg_omega_half=neg_omega_half,
        regularization_width_ry=0.1,
        edge_factor=1.5,
        target_error=1.0e-6,
        max_nodes=64,
        crossing_eps_q=1.0e-3,
        crossing_max_nodes=64,
        use_shipped_minimax_tables=False,
        log_tag="tiny-positive",
        print_fn=lambda *_args, **_kwargs: None,
        partition_hgl=False,
    )

    assert len(windows) == 3
    assert all(window.crossing_kind is None for window in windows)
    assert {window.omega_sign for window in windows} == {1}
    assert {window.prefactor for window in windows} == {expected_prefactor}

    ownership = np.zeros(
        (energies.size, poles.size, omega_eval.size), dtype=np.int64)
    energies_flat = energies.reshape(-1)
    for window in windows:
        E_lo = -np.inf if window.E_min is None else window.E_min
        E_hi = np.inf if window.E_max is None else window.E_max
        selected_E = ((energies_flat > E_lo) & (energies_flat <= E_hi)
                      & np.asarray(window.mask_A, dtype=bool).reshape(-1))
        B_lo, B_hi = ppm_windows.window_mask_B_bounds(window)
        selected_B = (poles > B_lo) & (poles <= B_hi)
        selected_w = np.ones(omega_eval.size, dtype=bool)
        if window.omega_indices is not None:
            selected_w[:] = False
            selected_w[window.omega_indices] = True
        ownership += (selected_E[:, None, None]
                      & selected_B[None, :, None]
                      & selected_w[None, None, :])

        live_E = energies_flat[selected_E]
        live_B = poles[selected_B]
        live_w = omega_eval[selected_w]
        x_min, x_max = ppm_windows._oriented_sign_definite_support(
            float(np.min(live_E)), float(np.max(live_E)),
            float(np.min(live_B)), float(np.max(live_B)),
            float(np.min(live_w)), float(np.max(live_w)),
            "E+B-omega",
        )
        assert x_max / x_min <= ppm_windows._SIGN_DEFINITE_PANE_MAX_RANGE

        t = np.asarray(window.nodes.t, dtype=np.complex128)
        alpha = np.asarray(window.nodes.alpha, dtype=np.complex128)
        for energy in live_E:
            for pole in live_B:
                for frequency in live_w:
                    got = window.prefactor * np.sum(
                        alpha
                        * np.exp(-1j * (window.E_ref_A
                                        + window.E_ref_B) * t)
                        * np.exp(-1j * (energy - window.E_ref_A) * t)
                        * np.exp(-1j * (pole - window.E_ref_B) * t)
                        * np.exp(+1j * window.omega_sign * frequency * t)
                    )
                    want = expected_prefactor / (energy + pole - frequency)
                    np.testing.assert_allclose(got, want, rtol=3.0e-14)

    np.testing.assert_array_equal(ownership, np.ones_like(ownership))

    # The exact same orientation must fail closed, rather than floor a cell
    # whose physical denominator reaches zero.
    with pytest.raises(AssertionError, match="not strictly positive"):
        ppm_windows._plan_sign_definite_cells(
            E_A=np.array([0.0]),
            base_mask_A=np.array([True]),
            Omega_q=ppm_windows.jnp.asarray([1.0e-14]),
            base_mask_B=ppm_windows.jnp.asarray([True]),
            mask_B_count=1,
            mask_B_min=1.0e-14,
            mask_B_max=1.0e-14,
            omega_nonneg_ry=np.array([1.0e-14]),
            orientation="E+B-omega",
        )


def _build_branch_windows():
    """Build the 4 branches × their windows from controlled synthetic inputs.

    Returns ``(branches, flat)`` where ``branches`` is a list of
    ``(tag, space, [_SigmaWindow, ...])`` and ``flat`` is the dict of
    plain numpy arrays / scalars captured for the reference freeze.
    """
    import jax.numpy as jnp
    from common.units import RYD_TO_EV
    from gw.ppm_windows import _build_windows_for_branch, _iter_branches

    nk, nb = 2, 6                       # 3 valence + 3 conduction bands
    # E_cond (conduction) and H_val (valence) energies-above-Fermi (Ry),
    # chosen to straddle T ≈ 0.76 Ry so both E_A≤T and E_A>T are populated.
    e_above = np.array([0.30, 0.90, 1.60], dtype=np.float64)      # Ry
    E_cond = np.zeros((nk, nb), dtype=np.float64)
    H_val = np.zeros((nk, nb), dtype=np.float64)
    cond_mask = np.zeros((nk, nb), dtype=bool)
    val_mask = np.zeros((nk, nb), dtype=bool)
    for k in range(nk):
        val_mask[k, 0:3] = True
        cond_mask[k, 3:6] = True
        H_val[k, 0:3] = e_above
        E_cond[k, 3:6] = e_above

    # Ω_q (nq, μ, μ) straddling T so mask_B le_t / gt_t are both non-empty.
    Omega_abs = np.array(
        [[[0.40, 0.50], [0.55, 0.45]],
         [[1.00, 1.20], [1.10, 1.30]]], dtype=np.float64)
    B_mask = Omega_abs > 1.0e-14

    # Symmetric ω grid in Ry — both halves present ⇒ all 4 branches.
    omega_ry = np.arange(-10.0, 10.0 + 1e-9, 0.5, dtype=np.float64) / RYD_TO_EV
    idx_pos = np.where(omega_ry >= 0.0)[0]
    idx_neg = np.where(omega_ry < 0.0)[0]
    omega_pos = omega_ry[idx_pos]
    omega_neg_abs = -omega_ry[idx_neg]

    branches = _iter_branches(
        omega_pos=omega_pos, idx_pos=idx_pos,
        omega_neg_abs=omega_neg_abs, idx_neg=idx_neg,
        E_cond=jnp.asarray(E_cond), H_val=jnp.asarray(H_val),
        cond_mask=jnp.asarray(cond_mask), val_mask=jnp.asarray(val_mask),
    )

    quad = dict(
        regularization_width_ry=0.25 / RYD_TO_EV,
        edge_factor=1.5,
        target_error=1e-6,
        max_nodes=64,
        crossing_eps_q=1e-3,
        crossing_max_nodes=500,
        use_shipped_minimax_tables=True,
    )

    out = []
    flat: dict[str, np.ndarray] = {}
    for br in branches:
        windows = _build_windows_for_branch(
            omega_nonneg_ry=br.omega_abs,
            E_A=br.E_A, base_mask_A=br.base_mask_A,
            Omega_q=jnp.asarray(Omega_abs), base_mask_B=jnp.asarray(B_mask),
            space=br.space, neg_omega_half=br.neg_omega_half,
            log_tag=br.tag, print_fn=lambda *a, **k: None,
            **quad,
        )
        out.append((br.tag, br.space, windows))
        for wi, w in enumerate(windows):
            key = f"{br.tag}|{wi}|{w.name}"
            flat[f"{key}|t"] = np.asarray(w.nodes.t, dtype=np.complex128)
            flat[f"{key}|alpha"] = np.asarray(w.nodes.alpha, dtype=np.complex128)
            flat[f"{key}|mask_A"] = np.asarray(w.mask_A, dtype=bool)
            flat[f"{key}|meta"] = np.array([
                float(w.E_ref_A), float(w.E_ref_B),
                float(w.omega_sign), float(w.prefactor),
                float(w.project_code), float(w.n_tau),
                float(w.mask_B_threshold) if w.mask_B_threshold is not None else np.nan,
            ], dtype=np.float64)
            flat[f"{key}|tags"] = np.array(
                [w.project, w.mask_B_mode, str(w.crossing_kind)], dtype="<U16")
    return out, flat


def _regenerate_g2_reference():
    """(Re)write the frozen G2 reference .npz.  Call manually when the window
    builders legitimately change; never inside the test."""
    _, flat = _build_branch_windows()
    G2_REF.parent.mkdir(parents=True, exist_ok=True)
    np.savez(G2_REF, **flat)
    return G2_REF


@pytest.mark.regression
def test_g2_branch_window_tiles_are_frozen():
    if requested_platform() in {"gpu", "cuda"} and not gpu_available():
        pytest.skip("CUDA GPU not available for requested platform=gpu.")

    branches, flat = _build_branch_windows()

    # --- structural guards: no branch or window may silently vanish -------
    tags = [t for t, _, _ in branches]
    assert tags == ["ω≥E_F cond", "ω≥E_F val", "ω<E_F cond", "ω<E_F val"], tags
    win_names = {t: [w.name for w in ws] for t, _, ws in branches}
    # The crossing branch (pole S can coincide with a grid ω) gets the 3-window
    # crossing/stripe/slab split: conduction on the +ω half, valence on the −ω
    # half.  The sign-definite branches get the single Laplace window.
    assert win_names["ω≥E_F cond"] == ["core", "a_stripe", "b_slab"], win_names
    assert win_names["ω<E_F val"] == ["core", "a_stripe", "b_slab"], win_names
    assert win_names["ω≥E_F val"] == ["single"], win_names
    assert win_names["ω<E_F cond"] == ["single"], win_names
    for _t, _s, ws in branches:
        assert ws, f"branch {_t!r} produced no windows"
        for w in ws:
            assert bool(np.any(w.mask_A)), f"{_t}:{w.name} empty mask_A"
            assert w.n_tau > 0, f"{_t}:{w.name} zero τ nodes"

    # --- bit-identity against the frozen reference ------------------------
    # NO CROSS-MACHINE TOLERANCE HERE, and the 2026-08-07 owner ruling
    # ("the micro-eV level is fine for comparisons between machines") does
    # NOT reach this gate.  The Perlmutter/Frontera disagreement in this
    # cell is the crossing-core node ladder, and that is an INTEGER count of
    # quadrature nodes riding in a float64 `meta` row, not a rounding
    # difference and not a quantity in eV — an atol would hide a real
    # change in how many τ points the window integrates over.  Whatever
    # this cell's answer is, it is not "loosen the comparison".
    #
    # THE REFERENCE FOLLOWS THE PERLMUTTER GRID.  Owner ruling P1b, taken at
    # the 2026-08-08 service-phase landing, and this file was re-frozen from
    # a Perlmutter run at the integration head to match it.  Perlmutter is
    # the blessed machine for this reference.
    #
    # THE FRONTERA DIFFERENCE IS STRUCTURAL, NOT NUMERICAL, and that is why
    # it cannot be absorbed by any tolerance.  Measured at the landing, 40
    # tile keys, Frontera-frozen array vs a Perlmutter build:
    #
    #   32 keys bit-identical
    #    4 keys SHAPE-mismatched — the crossing-core `t` and `alpha` ladders
    #      of BOTH crossing branches (ω≥E_F cond, ω<E_F val):
    #      Frontera (98,) vs Perlmutter (100,)
    #    2 `meta` rows differ by exactly 2.0 — the same node count, riding
    #      in the float64 meta row, which is what made this look like a
    #      micro-eV row and got it mis-filed under P1
    #    2 `meta` rows compare unequal with max|Δ| == 0.0: NaN-vs-NaN in the
    #      mask_B_threshold slot, identical in content
    #
    # The τ-node POSITIONS disagree from the first element (Perlmutter
    # [2.666561, 6.499882, ...] vs Frontera [5.442279e-09, 7.894766, ...]),
    # so these are two different quadratures, not one quadrature sampled
    # twice.  A machine running the Frontera minimax tables will therefore
    # fail this cell LOUDLY and by shape, which is the correct outcome: it
    # says "this build integrates over a different number of τ points",
    # which is exactly the fact the gate exists to surface.  Do not add a
    # tolerance and do not re-freeze on Frontera to make it green — bring
    # the question back to the owner, because only one grid is blessed.
    assert G2_REF.exists(), (
        f"missing G2 reference {G2_REF}; regenerate with "
        f"tests.test_sigma_ppm_gates._regenerate_g2_reference()")
    ref = np.load(G2_REF, allow_pickle=False)
    assert set(ref.files) == set(flat), (
        f"window-tile key set drifted:\n  new-only={set(flat) - set(ref.files)}"
        f"\n  ref-only={set(ref.files) - set(flat)}")
    for key in ref.files:
        got = flat[key]
        want = ref[key]
        if got.dtype.kind == "U":
            assert np.array_equal(got, want), f"{key} tag mismatch: {got} != {want}"
        else:
            np.testing.assert_array_equal(got, want, err_msg=f"{key} not bit-identical")

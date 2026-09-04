"""The patched dynamic-Sigma frequency grid and its uncovered-hole guard.

This module formerly also tested the retired clustered pane planner's
crossing-cost law.  Production MPA now uses ``gw.sigma_box_plan`` and the
pane builder no longer has a separate crossing-error control, so those three
tests were stale assertions about behavior that no shipping route provides.
The frequency-patch contract remains live and is kept here.
"""

import numpy as np

RYD = 13.605693122994
ETA_RY = 0.25 / RYD
CROSSING_TOL = 2.0e-3


# ---------------------------------------------------------------------------
#  The patched ω grid and its hole guard (the config half of the law)
# ---------------------------------------------------------------------------

def _sigma_cfg(patches):
    from gw.gw_config import DynamicSigmaConfig
    return DynamicSigmaConfig(
        omega_min_ev=-5.0, omega_max_ev=5.0, omega_step_ev=0.25,
        regularization_ev=0.25, window_edge_factor=1.5,
        fermi_reference="vbm",
        sigma_at_dft_extrapolate=False, sigma_at_dft_energies=False,
        omega_patches_ev=patches)


def test_omega_patches_parse_and_refuse_malformed_decks():
    import pytest
    assert _sigma_cfg("").parsed_omega_patches_ev() == []
    assert _sigma_cfg("-58:-52, -6:8").parsed_omega_patches_ev() == [
        (-58.0, -52.0), (-6.0, 8.0)]
    for bad in ("-6:8, -58:-52",     # descending
                "-58:-52,-52:8",     # touching
                "junk", "1:0"):      # unparseable / empty
        with pytest.raises(ValueError):
            _sigma_cfg(bad)


def test_patched_grid_is_a_gapped_union_at_the_shared_step():
    cfg = _sigma_cfg("-58:-52, -6:8")

    class _Shim:
        sigma = cfg
    from gw.gw_config import LorraxConfig
    grid = LorraxConfig.omega_grid_ev.fget(_Shim())
    assert grid[0] == -58.0 and grid[-1] == 8.0
    steps = np.diff(grid)
    assert np.isclose(np.median(steps), 0.25)
    assert np.max(steps) > 40.0          # the semicore gap survives
    assert np.all(steps > 0.0)


def test_hole_guard_refuses_qp_energies_inside_the_gap():
    import pytest
    from gw.qsgw_utils import assert_omega_grid_covers
    cfg = _sigma_cfg("-58:-52, -6:8")

    class _Shim:
        sigma = cfg
    from gw.gw_config import LorraxConfig
    grid_ry = LorraxConfig.omega_grid_ev.fget(_Shim()) / RYD
    covered = np.asarray([[0.1, -55.0 / RYD]])
    mask = np.ones_like(covered, dtype=bool)
    assert_omega_grid_covers(covered, mask, grid_ry, context="test")
    in_hole = np.asarray([[0.1, -30.0 / RYD]])
    with pytest.raises(ValueError, match="omega_grid_hole"):
        assert_omega_grid_covers(in_hole, mask, grid_ry, context="test")
    # An energy in the hole that is NOT in-grid-classified (scissored
    # band) is not this guard's business.
    assert_omega_grid_covers(
        in_hole, np.asarray([[True, False]]), grid_ry, context="test")


def test_shell_rule_error_verified_against_an_independent_cloud():
    """Brute-force certification check, independent of the builder.

    The rule's sampled_max_error comes from its own boundary grids; here
    the residual |1 - d·Q(d)| is measured on a fresh interior+boundary
    cloud of the certified rectangle (sign-crossing range included by
    construction) and must stay within a small factor of the target —
    analyticity puts the true max on the boundary, so 2x is generous.
    """
    from gw.mpa.evaluator import damped_rectangle_positive_rule
    rule = damped_rectangle_positive_rule(
        ETA_RY, ETA_RY + 0.05, 2.0, rel_tol=CROSSING_TOL, max_nodes=500)
    t, h = np.asarray(rule["t"]), np.asarray(rule["h"])
    rng = np.random.default_rng(20260816)
    x = rng.uniform(-2.0, 2.0, 4096)
    gamma = rng.uniform(ETA_RY, ETA_RY + 0.05, 4096)
    edge = rng.integers(0, 4, 4096)
    x = np.where(edge == 0, -2.0, np.where(edge == 1, 2.0, x))
    gamma = np.where(edge == 2, ETA_RY,
                     np.where(edge == 3, ETA_RY + 0.05, gamma))
    z = gamma - 1j * x
    residual = np.abs(1.0 - (z[:, None] * np.exp(
        -z[:, None] * t[None, :])) @ h)
    assert float(np.max(residual)) <= 2.0 * CROSSING_TOL
    # And the crossing itself is exercised: the cloud straddles x = 0.
    assert np.any(x > 1.0) and np.any(x < -1.0)


def test_patched_deck_hull_fields_are_the_patch_hull(tmp_path):
    """The SC in-grid partition reads sigma.omega_min/max_ev as the grid
    hull; with patches those fields must be the patch hull, or every
    band outside the DEFAULT [-5, +5] silently takes the scissor while
    Σ is computed on the deep clusters and never consulted (measured on
    arm 21: SC partition 2/48 instead of 10/48)."""
    from gw.gw_config import LorraxConfig
    deck = tmp_path / "mpa.in"
    deck.write_text("[lorrax]\nwfn_file = /dev/null\n"
                    "sigma_omega_patches_ev = -66:-48, -32:-20, -7:7\n")
    cfg = LorraxConfig.from_input_file(str(deck))
    assert cfg.sigma.omega_min_ev == -66.0
    assert cfg.sigma.omega_max_ev == 7.0
    g = cfg.omega_grid_ev
    assert g[0] == -66.0 and g[-1] == 7.0

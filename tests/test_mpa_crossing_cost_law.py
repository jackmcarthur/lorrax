"""The crossing-rule cost law is sub-linear in dynamic range.

The incumbent monolithic core paid n = 87·f_max + 10 nodes (measured,
``runs/Na/02_soc48b_qsgw_mpa/14_semicore_cond_window/rule_cost_scan.out``)
because one positive damped rule had to resolve the eta-scale crossing
feature uniformly over the whole ``[ω_lo, ω_hi] × transitions`` product
set — a Landau-density floor for real-time exponential rules, not a node
selection defect (``docs/dev/crossing-rule-cost-law.md``).

The clustered decomposition keeps the eta-resolved rule only on the
crossing shell of each ω cluster, whose bandwidth is set by the cluster
span and the pole bracket, so the total is set by how many places the
physics evaluates Σ, never by how far apart they are.  These tests pin
that law at the PRODUCTION eta and tolerance, on a synthetic Fe-class
geometry: a valence window plus a semicore cluster 7.7 Ry deep (a 209 eV
evaluation span), then the same geometry twice as deep.

All offline scalar planning — no GPU, no spatial kernel.
"""

import jax.numpy as jnp
import numpy as np

from gw.mpa import sigma_windows as SW
from gw.ppm_windows import _SigmaBranch

RYD = 13.605693122994
ETA_RY = 0.25 / RYD                  # production sigma_regularization_ev
CROSSING_TOL = 2.0e-3                # frozen pane-control crossing target
SECTOR_TOL = 6.5e-4                  # frozen pane-control sector target


def _fe_class_plan(depth_ry):
    """A valence window plus one semicore ω cluster ``depth_ry`` deep."""
    energies = np.asarray([0.05, 0.25, 0.45, depth_ry - 0.25])
    omega = np.concatenate([
        np.linspace(0.0, 0.5, 6),
        np.asarray([depth_ry - 0.05, depth_ry + 0.05]),
    ])
    E_A = jnp.asarray(energies[None, :])
    branch = _SigmaBranch(
        "pos_cond", E_A, jnp.ones_like(E_A, dtype=bool), "cond", False,
        omega, np.arange(omega.size))
    Omega = jnp.asarray([[[[0.30 - 0.05j]]]])
    B = jnp.asarray([[[[0.7 + 0.2j]]]])
    summaries = SW.summarize_sigma_poles(
        Omega, B, [branch],
        regularization_width_ry=ETA_RY, edge_factor=1.5)
    return SW.build_shared_sigma_windows(
        summaries, [branch],
        regularization_width_ry=ETA_RY, edge_factor=1.5,
        target_error=SECTOR_TOL, crossing_target_error=CROSSING_TOL,
        max_rank=96, crossing_max_nodes=SW.CROSSING_NODE_FLOOR)


def test_fe_class_span_certifies_under_300_nodes_per_rule():
    """The Fe-class discriminator: every rule in a 209 eV-span plan is
    certified at the production tolerance with n < 300 nodes — where the
    monolithic core REFUSED at 500 and needed ~87·f_max ≈ 1300+."""
    plan, report = _fe_class_plan(7.7)
    assert plan, "Fe-class plan built no windows"
    for row in plan:
        assert row.window.n_tau < 300, (
            f"{row.window.name}: {row.window.n_tau} nodes")
        assert row.window.max_error is not None
        target = (CROSSING_TOL if row.window.name in ("core", "sd_core")
                  else SECTOR_TOL)
        assert row.window.max_error <= target


def test_fe_class_total_beats_the_linear_law_and_meets_the_budget():
    """Total tau dispatches land in the 150–250 budget class, against
    ~700 for the linear law at this span (87·f_max with f_max ≈ ω_max)."""
    plan, _report = _fe_class_plan(7.7)
    total = sum(row.window.n_tau for row in plan)
    linear_law = 87.0 * 7.7 + 10.0
    assert total < 0.5 * linear_law, (total, linear_law)
    assert total < 300, total


def test_doubling_the_dynamic_range_does_not_grow_the_damped_rules():
    """Sub-linearity, stated as the scan would state it: moving the
    semicore cluster twice as deep leaves the eta-resolved (damped) node
    total EXACTLY unchanged and the full plan within 10%."""
    shallow_plan, _ = _fe_class_plan(7.7)
    deep_plan, _ = _fe_class_plan(15.4)

    def damped_total(plan):
        return sum(row.window.n_tau for row in plan
                   if row.window.name in ("core", "sd_core"))

    def total(plan):
        return sum(row.window.n_tau for row in plan)

    assert damped_total(deep_plan) == damped_total(shallow_plan)
    assert total(deep_plan) <= 1.10 * total(shallow_plan), (
        total(shallow_plan), total(deep_plan))


# ---------------------------------------------------------------------------
#  The patched ω grid and its hole guard (the config half of the law)
# ---------------------------------------------------------------------------

def _sigma_cfg(patches):
    from gw.gw_config import DynamicSigmaConfig
    return DynamicSigmaConfig(
        omega_min_ev=-5.0, omega_max_ev=5.0, omega_step_ev=0.25,
        regularization_ev=0.25, window_edge_factor=1.5,
        omega_layout="replicated", fermi_reference="vbm",
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

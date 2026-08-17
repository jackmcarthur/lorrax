"""The Green's-function band window is cut by an OCCUPANCY threshold.

``sigma_windows._a_space`` used to cut the metallic branch support with the
EXACT rule ``weight != 0.0``.  That excludes only what underflowed to zero,
which for MP1 is ~54 smearing widths out from mu (measured historically as
the -0.53 Ry phantom excursion of the first metallic arm, claim 0196).  The
deck key ``occupation_window_threshold`` replaces it with a physical shell:
a band leaves a branch once its weight magnitude falls below
``1 - threshold``.

Three properties this file pins, each of which a plausible wrong
implementation breaks:

1. **MAGNITUDE, not a one-sided cut.**  ``OccupationState.f_kn`` is never
   clipped, so MP1's negative lobe reaches ``f = -0.0355`` just above mu --
   seven times the 0.005 floor.  ``w > floor`` would discard every
   wrong-side band; ``abs(w) > floor`` keeps them.
2. **Exact zeros stay excluded.**  ``0 > 0.005`` is false, so the reason the
   exact rule existed is preserved rather than traded away.
3. **``threshold = 1.0`` is the exact rule, bit-for-bit.**  Floor 0.0 makes
   ``abs(w) > 0.0`` identical to ``w != 0.0``, which is both the escape
   hatch and the A/B control for the knob.

Scope: this file is a unit file.  It exercises the planner's support rule
and the config plumbing.  It does not run a driver and says nothing about
QP energies; that is the smearing sweep in
``reports/occupation_window_threshold_2026-08-16/``.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from gw.mpa import sigma_windows as SW
from gw.ppm_windows import _SigmaBranch


_OMEGA = np.asarray([0.0, 0.25, 0.5])
_IDX = np.arange(_OMEGA.size)


def _branch(energies, weights, *, tag="pos_val", space="val"):
    E_A = jnp.asarray(np.asarray(energies, dtype=np.float64)[None, :])
    bw = (None if weights is None
          else jnp.asarray(np.asarray(weights, dtype=np.float64)[None, :]))
    return _SigmaBranch(tag, E_A, jnp.ones_like(E_A, dtype=bool), space,
                        False, _OMEGA, _IDX, band_weight=bw)


def _support(branch, threshold):
    """Bands the planner keeps, and the (min, max) E_A it plans against."""
    mask, bounds = SW._a_space(
        branch, lambda E: np.ones(E.shape, bool),
        SW._weight_floor(threshold))
    return np.asarray(mask)[0], bounds


# --------------------------------------------------------------------------
# The occupancy -> weight mapping
# --------------------------------------------------------------------------

def test_the_default_is_the_owners_0995_occupancy():
    assert SW.OCCUPATION_WINDOW_THRESHOLD_DEFAULT == 0.995


def test_occupancy_0995_becomes_a_0005_weight_floor():
    """The deck speaks occupancy; the cut is on the branch weight."""
    assert SW._weight_floor(0.995) == pytest.approx(0.005, abs=1e-15)
    assert SW._weight_floor(0.99) == pytest.approx(0.01, abs=1e-15)
    assert SW._weight_floor(1.0) == 0.0


@pytest.mark.parametrize("bad", [0.0, 0.4, 1.0 + 1e-9, 2.0, -1.0,
                                 float("nan"), float("inf")])
def test_an_out_of_range_threshold_refuses_by_name(bad):
    with pytest.raises(ValueError, match="occupation_window_threshold"):
        SW._weight_floor(bad)


# --------------------------------------------------------------------------
# 1. Magnitude, not a one-sided cut
# --------------------------------------------------------------------------

def test_negative_weights_of_physical_size_are_kept():
    """MP1's lobe minimum (w = -0.0355) is seven floors deep. Keep it.

    A one-sided ``w > 0.005`` drops band 1 here; ``abs(w) > 0.005`` keeps
    it.  This is the cell that separates the two implementations.
    """
    branch = _branch([-0.60, 0.30, 0.90], [0.95, -0.0355, 0.20])
    keep, bounds = _support(branch, 0.995)
    assert keep.tolist() == [True, True, True]
    assert bounds == (-0.60, 0.90)


def test_the_cut_is_symmetric_in_the_sign_of_the_weight():
    """Same magnitude either side of zero -> same decision."""
    branch = _branch([-0.4, -0.2, 0.2, 0.4], [0.006, -0.006, 0.004, -0.004])
    keep, _ = _support(branch, 0.995)
    assert keep.tolist() == [True, True, False, False]


def test_a_one_sided_cut_would_have_been_visible_here():
    """Guard cell: the negative band is the ONLY thing holding the bound.

    If someone re-spells the rule one-sided, ``bounds`` loses its lower
    edge and this fails loudly rather than drifting.
    """
    branch = _branch([-0.75, 0.10], [-0.30, 0.90])
    keep, bounds = _support(branch, 0.995)
    assert keep.tolist() == [True, True]
    assert bounds[0] == -0.75
    one_sided = np.asarray([-0.30, 0.90]) > 0.005
    assert one_sided.tolist() == [False, True], (
        "fixture no longer discriminates the two rules")


# --------------------------------------------------------------------------
# 2. The exact rule's reason survives
# --------------------------------------------------------------------------

@pytest.mark.parametrize("threshold", [0.995, 0.99, 0.9999, 1.0])
def test_exact_zero_weight_is_excluded_at_every_threshold(threshold):
    """The -0.53 Ry phantom: a zero-weight band must never set the bound."""
    branch = _branch([-2.0, -0.10, 0.30], [0.0, 0.90, 0.40])
    keep, bounds = _support(branch, threshold)
    assert keep[0] is np.False_ or not keep[0]
    assert bounds[0] == -0.10


def test_the_subnormal_tail_no_longer_widens_the_geometry():
    """The historical failure, in miniature.

    Band 0 carries the 2.67e-322 subnormal weight measured on the refusing
    metallic arm.  The exact rule (threshold 1.0) keeps it and plans against
    E_A = -0.545 Ry; the 0.995 threshold cuts it and plans against the
    physical shell.  ``_geometry``'s ``excursion`` -- and therefore
    ``crossing_edge`` for EVERY branch -- shrinks accordingly.
    """
    branch = _branch([-0.545, -0.043, 0.20], [2.67e-322, 0.30, 0.95])

    keep_exact, bounds_exact = _support(branch, 1.0)
    keep_thr, bounds_thr = _support(branch, 0.995)
    assert keep_exact.tolist() == [True, True, True]
    assert keep_thr.tolist() == [False, True, True]
    assert bounds_exact[0] == -0.545
    assert bounds_thr[0] == -0.043

    def edge(threshold):
        _om, _eta, crossing_edge, _sel = SW._geometry(
            [branch], 0.05, 1.5, SW._weight_floor(threshold))
        return crossing_edge

    assert edge(0.995) < edge(1.0)
    assert edge(1.0) - edge(0.995) == pytest.approx(0.545 - 0.043, abs=1e-12)


# --------------------------------------------------------------------------
# 3. threshold = 1.0 is the incumbent rule, and insulators never notice
# --------------------------------------------------------------------------

def test_threshold_one_reproduces_the_exact_nonzero_rule():
    weights = np.asarray([0.0, 2.67e-322, 1e-30, -1e-12, 0.5, -0.03, 1.0])
    energies = np.linspace(-1.0, 1.0, weights.size)
    keep, _ = _support(_branch(energies, weights), 1.0)
    np.testing.assert_array_equal(keep, weights != 0.0)


@pytest.mark.parametrize("threshold", [0.995, 0.9, 1.0])
def test_an_insulating_branch_has_no_weight_and_is_untouched(threshold):
    """band_weight=None is the incumbent bool-mask path, bit-for-bit."""
    energies = [-0.6, -0.1, 0.4]
    branch = _branch(energies, None)
    keep, bounds = _support(branch, threshold)
    assert keep.tolist() == [True, True, True]
    assert bounds == (-0.6, 0.4)


# --------------------------------------------------------------------------
# The property the owner actually asked about: smearing-awareness
# --------------------------------------------------------------------------

def test_the_retained_shell_scales_with_the_smearing_width():
    """A FIXED occupancy threshold is a FIXED number of smearing widths.

    MP1's weight depends on ``x = (E-mu)/(2W)`` alone, so the outer edge of
    ``abs(f) > 0.005`` sits at a fixed ``x`` and therefore at an ``E-mu``
    proportional to ``W``.  Doubling the width doubles the retained shell;
    the threshold does not need retuning per width.  (The exact rule scales
    the same way but at ~54 widths instead of ~4.3, which is the whole
    difference.)
    """
    from gw.efermi import mp1_occupations

    edges = {}
    for width in (0.005, 0.010, 0.020):
        energies = np.linspace(-0.6, 0.6, 4001)
        f = np.asarray(mp1_occupations(
            jnp.asarray(energies[None, :]), 0.0, width))[0]
        keep, bounds = _support(_branch(energies, f), 0.995)
        edges[width] = bounds[1]

    ratios = [edges[0.010] / edges[0.005], edges[0.020] / edges[0.010]]
    for r in ratios:
        assert r == pytest.approx(2.0, rel=2e-2), (
            f"retained shell is not proportional to the width: {edges}")
    # ...and it lands where the closed form says: |E-mu| = 4.275 * W.
    assert edges[0.010] / 0.010 == pytest.approx(4.275, rel=2e-2)


# --------------------------------------------------------------------------
# Deck plumbing
# --------------------------------------------------------------------------

def test_the_key_is_a_deck_key_with_the_owners_default():
    from gw.gw_config import _DEFAULTS
    assert _DEFAULTS["occupation_window_threshold"] == 0.995


def test_the_key_reaches_the_planner_through_MPAConfig():
    """One value feeds BOTH planner entry points, so they cannot diverge."""
    import inspect
    from gw.gw_config import MPAConfig
    from gw.mpa.sigma import compute_sigma_c_mpa_omega_grid

    assert "occupation_window_threshold" in MPAConfig.__dataclass_fields__
    for fn in (compute_sigma_c_mpa_omega_grid,
               SW.summarize_sigma_poles,
               SW.build_shared_sigma_windows):
        assert "occupation_window_threshold" in (
            inspect.signature(fn).parameters), fn.__name__

    src = inspect.getsource(compute_sigma_c_mpa_omega_grid)
    assert src.count("occupation_window_threshold=occupation_window_threshold") == 2


def test_the_key_has_a_row_in_the_input_reference():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1]
           / "docs" / "input_reference.md").read_text()
    rows = [ln for ln in doc.splitlines()
            if ln.startswith("| `occupation_window_threshold`")]
    assert len(rows) == 1, "expected exactly one reference row"
    assert "`0.995`" in rows[0]
    # The two facts a future reader must not have to rediscover.
    assert "1 - threshold" in rows[0]
    assert "MAGNITUDE" in rows[0]

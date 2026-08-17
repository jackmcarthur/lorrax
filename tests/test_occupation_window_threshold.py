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
QP energies -- those come from the deck-level A/B whose evidence is
archived at
``/global/cfs/cdirs/m4598/jackm/occupation_window_threshold_2026-08-16/``
(the closed form behind the numbers quoted below is ``analysis/mp1_shell.py``
there, and the real-deck sweep is ``analysis/planner_sweep.py``).
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


def test_a_gapped_deck_never_reaches_the_thresholded_line_at_all():
    """The insulator claim is about EXECUTION, not about equal numbers.

    ``compute_sigma_xc`` passes ``occupation_state=None`` on a gapped deck
    (``sc_iteration.py:1674`` builds ``metal_occ_state`` only under
    ``material_class == 'metal'``), and ``_branches`` then leaves every
    ``band_weight`` None -- so the guarded block inside ``_a_space`` is not
    merely agreeing with the old rule, it does not run.  Asserting that here
    is stronger than an artifact diff, which could match for other reasons.
    """
    from gw.mpa.sigma import _branches

    class _Slices:
        full = slice(0, 4)

    class _Wfns:
        enk = jnp.asarray([[-0.6, -0.2, 0.3, 0.9]])
        occ = jnp.asarray([[1.0, 1.0, 0.0, 0.0]])   # gapped: exactly 0/1
        slices = _Slices()

    branches = _branches(_Wfns(), np.asarray([0.0, 0.25]), 0.0,
                         occupation_state=None)
    assert branches, "fixture produced no branches"
    assert all(b.band_weight is None for b in branches)

    calls = {"n": 0, "weighted": 0}
    real = SW._a_space

    def counting(branch, predicate, weight_floor=0.0):
        calls["n"] += 1
        calls["weighted"] += branch.band_weight is not None
        return real(branch, predicate, weight_floor)

    SW._a_space = counting
    try:
        SW._geometry(branches, 0.05, 1.5, SW._weight_floor(0.995))
    finally:
        SW._a_space = real
    assert calls["n"] == len(branches)
    assert calls["weighted"] == 0, (
        "a gapped deck reached the weighted branch of _a_space")


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
    # Three now, not two: the BRANCH BUILD reads the same value as the pole
    # census and the window build, so the supports the executor masks with
    # and the geometry the planner sizes cannot come from different windows.
    assert src.count("occupation_window_threshold=occupation_window_threshold") == 3


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


# ==========================================================================
#  ALL PATHS.  The rule above lived at ONE site (the MPA Sigma planner's
#  A-space).  Everything below pins the other occupancy-based
#  band-inclusion decisions onto the SAME helper, the same default and the
#  same magnitude predicate -- and pins, by name, the two sites that
#  deliberately do NOT honour it.
# ==========================================================================

def test_there_is_exactly_one_helper_and_one_default():
    """No second key, no second default, no re-spelled predicate.

    ``gw.efermi`` owns the occupancy->weight map and the predicate; the
    planner's ``_weight_floor`` is that same function object, not a copy
    that could drift from it.
    """
    from gw import efermi
    from gw import ppm_windows, w_isdf

    assert SW._weight_floor is efermi.occupation_weight_floor
    assert (SW.OCCUPATION_WINDOW_THRESHOLD_DEFAULT
            is efermi.OCCUPATION_WINDOW_THRESHOLD_DEFAULT)
    for mod in (ppm_windows, w_isdf):
        assert (mod.occupation_weight_floor
                is efermi.occupation_weight_floor), mod.__name__
        assert (mod.band_in_occupation_window
                is efermi.band_in_occupation_window), mod.__name__
        assert (mod.OCCUPATION_WINDOW_THRESHOLD_DEFAULT
                is efermi.OCCUPATION_WINDOW_THRESHOLD_DEFAULT), mod.__name__


def test_the_predicate_is_a_magnitude_at_the_helper_itself():
    from gw.efermi import band_in_occupation_window as keep
    w = np.asarray([0.0, 0.004, -0.004, 0.006, -0.006, -0.0355, 1.0])
    np.testing.assert_array_equal(
        keep(w, 0.005), [False, False, False, True, True, True, True])
    # floor 0.0 is the exact incumbent rule, at the helper, bit-for-bit.
    np.testing.assert_array_equal(keep(w, 0.0), w != 0.0)


# --------------------------------------------------------------------------
# SITE 2: the Sigma BRANCH supports (gw.ppm_windows.branches_for_omega_grid).
# Both drivers build their four branches here, so this is where cond_mask /
# val_mask stop being the exact `f != 1` / `f != 0` rule.
# --------------------------------------------------------------------------

def _branch_masks(f, threshold):
    """cond/val supports the branch builder hands the planner+executor."""
    from gw.ppm_windows import branches_for_omega_grid
    f = jnp.asarray(np.asarray(f, dtype=np.float64)[None, :])
    E = jnp.zeros_like(f)
    branches = branches_for_omega_grid(
        np.asarray([0.0, 0.25]), E_cond=E, H_val=-E,
        cond_mask=(f != 1.0), val_mask=(f != 0.0),
        cond_weight=1.0 - f, val_weight=f,
        occupation_window_threshold=threshold)
    got = {}
    for b in branches:
        got[b.space] = np.asarray(b.base_mask_A)[0]
    return got["cond"], got["val"]


def test_branch_supports_are_cut_by_the_threshold_on_both_sides():
    #        deep occ   near-1      fractional  near-0      empty
    f = [1.0, 0.9990, 0.50, 0.0010, 0.0]
    cond, val = _branch_masks(f, 0.995)
    # cond weight is 1-f: [0, 0.001, 0.5, 0.999, 1.0]
    assert cond.tolist() == [False, False, True, True, True]
    # val weight is f: [1.0, 0.999, 0.5, 0.001, 0]
    assert val.tolist() == [True, True, True, False, False]


def test_branch_supports_keep_the_negative_mp1_lobe():
    """f = -0.0355 is the lobe minimum: seven floors deep, and REAL.

    A one-sided `w > floor` drops it from the val branch.  The exact
    incumbent rule kept it; so must the threshold.
    """
    f = [0.95, -0.0355, 1.0355]
    cond, val = _branch_masks(f, 0.995)
    # val weight f = [0.95, -0.0355, 1.0355] -- all three clear 0.005.
    assert val.tolist() == [True, True, True]
    # cond weight 1-f = [0.05, 1.0355, -0.0355] -- likewise.
    assert cond.tolist() == [True, True, True]
    one_sided = np.asarray([0.95, -0.0355, 1.0355]) > 0.005
    assert one_sided.tolist() == [True, False, True], (
        "fixture no longer discriminates the two rules")


@pytest.mark.parametrize("f", [
    [1.0, 0.9990, 0.50, 0.0010, 0.0],
    [0.95, -0.0355, 1.0355],
    [1.0, 2.67e-322, 0.5, 1.0 - 2.67e-322, 0.0],
])
def test_threshold_one_reproduces_the_exact_branch_supports(f):
    """The A/B control: threshold 1.0 == `f != 1` / `f != 0`, bit-for-bit."""
    cond, val = _branch_masks(f, 1.0)
    fa = np.asarray(f, dtype=np.float64)
    np.testing.assert_array_equal(cond, fa != 1.0)
    np.testing.assert_array_equal(val, fa != 0.0)


def test_a_weightless_branch_never_reaches_the_predicate_at_all():
    """The insulator/GN-PPM claim is about EXECUTION, not equal numbers.

    ``branches_for_omega_grid`` called without weights -- every insulating
    MPA plan, and EVERY GN-PPM branch, since that driver has no fractional
    occupancy to supply -- must not run the thresholded line.  Counting is
    stronger than an artifact diff, which could agree for other reasons.
    """
    from gw import ppm_windows

    calls = {"n": 0}
    real = ppm_windows.band_in_occupation_window

    def counting(weight, floor):
        calls["n"] += 1
        return real(weight, floor)

    occ = jnp.asarray(np.asarray([[1.0, 1.0, 0.0, 0.0]]))
    E = jnp.asarray(np.asarray([[-0.6, -0.2, 0.3, 0.9]]))
    ppm_windows.band_in_occupation_window = counting
    try:
        branches = ppm_windows.branches_for_omega_grid(
            np.asarray([-0.25, 0.0, 0.25]), E_cond=E, H_val=-E,
            cond_mask=(occ <= 0.5), val_mask=(occ > 0.5),
            occupation_window_threshold=0.995)
    finally:
        ppm_windows.band_in_occupation_window = real
    assert branches, "fixture produced no branches"
    assert all(b.band_weight is None for b in branches)
    assert calls["n"] == 0, (
        "a weightless branch reached the occupancy predicate")


def test_the_mpa_branch_builder_forwards_the_deck_value():
    """One value from the deck reaches the branch build AND both planner
    entry points -- the supports cannot diverge from the geometry."""
    import inspect
    from gw.mpa.sigma import _branches, compute_sigma_c_mpa_omega_grid

    assert "occupation_window_threshold" in (
        inspect.signature(_branches).parameters)
    src = inspect.getsource(compute_sigma_c_mpa_omega_grid)
    assert src.count(
        "occupation_window_threshold=occupation_window_threshold") == 3


def test_the_mpa_branch_supports_honour_the_threshold_end_to_end():
    """``_branches`` with a state, through the real entry point."""
    from types import SimpleNamespace
    from gw.mpa.sigma import _branches

    class _Slices:
        full = slice(0, 5)

    wfns = SimpleNamespace(
        enk=jnp.asarray([[-0.6, -0.2, 0.0, 0.2, 0.9]]),
        occ=jnp.asarray([[1.0, 1.0, 0.5, 0.0, 0.0]]),
        slices=_Slices())
    state = SimpleNamespace(
        f_kn=jnp.asarray([[1.0, 0.999, 0.5, 0.001, 0.0]]), mu_ry=0.0)

    thr = {b.space: np.asarray(b.base_mask_A)[0]
           for b in _branches(wfns, np.asarray([0.0, 0.25]), 0.0,
                              occupation_state=state,
                              occupation_window_threshold=0.995)}
    exact = {b.space: np.asarray(b.base_mask_A)[0]
             for b in _branches(wfns, np.asarray([0.0, 0.25]), 0.0,
                                occupation_state=state,
                                occupation_window_threshold=1.0)}
    assert thr["val"].tolist() == [True, True, True, False, False]
    assert exact["val"].tolist() == [True, True, True, True, False]
    assert thr["cond"].tolist() == [False, False, True, True, True]
    assert exact["cond"].tolist() == [False, True, True, True, True]


# --------------------------------------------------------------------------
# SITE 3: the chi0 fractional occupation supports
# (gw.w_isdf._occupation_support_slices).  The one consumer where the cut
# removes bands from a CONTRACTION -- the slices index wfns.xn/yr -- rather
# than masking them, and it also sizes the damped-line rule.
# --------------------------------------------------------------------------

def _slices(occ, threshold):
    from gw.w_isdf import _occupation_support_slices
    return _occupation_support_slices(
        np.asarray(occ, dtype=np.float64), threshold)


def test_chi_supports_narrow_with_the_threshold():
    """A near-empty band leaves the f support; a near-full one leaves u."""
    occ = [[1.0, 0.999, 0.5, 0.001, 0.0]]
    assert _slices(occ, 1.0) == (slice(0, 4), slice(1, 5))
    assert _slices(occ, 0.995) == (slice(0, 3), slice(2, 5))


def test_chi_supports_keep_the_negative_mp1_lobe():
    """Band 0 is held ONLY by a wrong-side negative weight.

    Under `w > floor` the f support would start at band 1 and the whole
    negative lobe would leave chi0's occupied Green's function.
    """
    occ = [[-0.0355, 0.9, 0.0]]
    f_slice, _ = _slices(occ, 0.995)
    assert f_slice == slice(0, 2)
    one_sided = np.asarray([-0.0355, 0.9, 0.0]) > 0.005
    assert one_sided.tolist() == [False, True, False], (
        "fixture no longer discriminates the two rules")


def test_chi_supports_still_exclude_exact_zeros():
    occ = [[0.0, 0.9, 0.4, 0.0]]
    for threshold in (1.0, 0.995, 0.9):
        f_slice, _ = _slices(occ, threshold)
        assert f_slice == slice(1, 3), threshold


@pytest.mark.parametrize("occ", [
    [[1.0, 0.82, 0.10, 0.0], [1.0, 0.61, 0.25, 0.0], [1.0, 0.74, -0.01, 0.0]],
    [[1.0, 0.999, 0.5, 0.001, 0.0]],
    [[1.0, 2.67e-322, 0.0]],
])
def test_chi_threshold_one_reproduces_the_exact_supports(occ):
    """Floor 0.0 is `occ != 0` / `occ != 1`, bit-for-bit."""
    a = np.asarray(occ, dtype=np.float64)
    f_sup = np.flatnonzero(np.any(a != 0.0, axis=0))
    u_sup = np.flatnonzero(np.any(a != 1.0, axis=0))
    want = (slice(int(f_sup[0]), int(f_sup[-1]) + 1),
            slice(int(u_sup[0]), int(u_sup[-1]) + 1))
    assert _slices(occ, 1.0) == want


@pytest.mark.parametrize("threshold", [1.0, 0.9999, 0.995, 0.9, 0.5])
def test_a_gapped_occupation_table_gives_one_support_at_every_threshold(
        threshold):
    """The insulator invariance, argued rather than sampled.

    A gapped table stores exactly 0 or exactly 1, so both branch weights are
    exactly 0 or exactly 1 too, and `abs(w) > floor` for floor in [0, 0.5]
    can only be `w == 1`.  The supports are therefore threshold-independent
    by algebra, and equal to the exact rule's.  This cell is the algebra
    checked, at both ends of the allowed range and in between.
    """
    occ = [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    assert _slices(occ, threshold) == (slice(0, 2), slice(2, 4))


def test_the_rule_bandwidth_reads_the_same_supports_as_the_kernel():
    """`occupation_support_bandwidth` must read the slices the kernel gets.

    MEASURED PROPERTY, and it is not the one a reader expects: on a
    monotone occupation table the bandwidth is threshold-INVARIANT, because
    it spans OUTER edges -- ``min(E over f)`` is the deepest band, whose
    occupancy is ~1 and which therefore never leaves the f support, and
    ``max(E over u)`` is the highest band, whose (1-f) is ~1 and which never
    leaves u.  The threshold narrows the supports at their INNER edges.  So
    the saving here is BANDS IN THE CONTRACTION, not quadrature nodes; the
    reason the threshold is still an argument to this function is that the
    two must read one support, and a non-monotone table (band crossings,
    MP1 overshoot) can move the outer edge too.
    """
    from gw.w_isdf import occupation_support_bandwidth

    e = np.asarray([[-2.0, -0.5, 0.0, 0.4, 1.5]])
    occ = np.asarray([[0.999, 0.9, 0.5, 0.1, 0.001]])
    for threshold in (1.0, 0.995):
        f_slice, u_slice = _slices(occ, threshold)
        assert occupation_support_bandwidth(
            e, occ, occupation_window_threshold=threshold) == pytest.approx(
                float(np.max(e[:, u_slice]) - np.min(e[:, f_slice]))), threshold
    # The supports DO narrow -- that is the cost the threshold buys back.
    assert _slices(occ, 1.0) == (slice(0, 5), slice(0, 5))
    assert _slices(occ, 0.995) == (slice(0, 4), slice(1, 5))
    # ...while the span they subtend does not, on this monotone table.
    assert (occupation_support_bandwidth(e, occ,
                                         occupation_window_threshold=0.995)
            == occupation_support_bandwidth(e, occ,
                                            occupation_window_threshold=1.0))


def test_the_chi_fit_driver_uses_one_window_for_rule_and_kernel():
    """gw.mpa.model reads the deck key ONCE and gives it to every consumer,
    so the rule bandwidth and the band slices cannot disagree."""
    import inspect
    from gw.mpa.model import _evaluate_samples as _fn

    src = inspect.getsource(_fn)
    assert "occ_window = float(config.mpa.occupation_window_threshold)" in src
    assert src.count("occupation_window_threshold=occ_window") == 3


# --------------------------------------------------------------------------
# The two sites that deliberately do NOT honour the threshold.  Pinned so a
# reader sees a decision rather than an omission, and so closing either one
# flips a test rather than passing silently.
# --------------------------------------------------------------------------

def test_the_gn_ppm_sigma_driver_has_no_occupancy_to_threshold():
    """Not an omission: that driver has no fractional occupancy at all.

    ``ppm_sigma._prepare_sigma_state`` splits ``wfns.occ > 0.5`` -- an
    INTEGER step on a table ``wavefunction_bundle._build_occ`` fills as
    ``(enk <= efermi)`` -- and passes no weights to
    ``branches_for_omega_grid``, so every GN-PPM branch has
    ``band_weight=None`` and there is nothing for an occupancy threshold to
    cut.  Closing this means porting fractional occupations into that
    driver first (its own ``TODO(metal-greens)``); the threshold then
    governs it through the existing call with no further change.
    """
    import inspect
    from gw import ppm_sigma
    from gw.wavefunction_bundle import _build_occ

    state_src = inspect.getsource(ppm_sigma._prepare_sigma_state)
    assert "occ_mask = occ_full > 0.5" in state_src
    assert "TODO(metal-greens)" in state_src, (
        "the step-occupation gap lost its marker")
    assert "(enk_host <= float(efermi))" in inspect.getsource(_build_occ)

    driver = inspect.getsource(ppm_sigma.compute_sigma_c_ppm_omega_grid)
    call = driver[driver.index("branches_for_omega_grid("):]
    call = call[:call.index(")")]
    assert "cond_weight" not in call and "val_weight" not in call, (
        "GN-PPM now supplies branch weights -- it must also pass "
        "occupation_window_threshold, and this cell must become an "
        "end-to-end assertion instead of a documented gap")


def test_the_static_occupation_projector_is_not_thresholded():
    """``cohsex_sigma.build_Gij`` weights EVERY Sigma band by f and drops
    none.  Thresholding it would delete electrons from the Hartree density,
    which its own fixed-N guard already refuses -- so the exact diag(f) is
    correct there and stays."""
    import inspect
    from gw.cohsex_sigma import build_Gij

    src = inspect.getsource(build_Gij)
    assert "occupation_window_threshold" not in src
    assert "Gij[:, idx, idx] = f_win.astype(np.complex128)" in src
    assert "hartree density would be missing weight carried by bands" in (
        src.lower())

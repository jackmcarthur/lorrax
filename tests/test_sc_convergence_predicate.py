"""The self-consistency convergence criterion: max|dE|, not RMS.

A stage of QSGW is converged when EVERY band inside the protected energy
range has moved by less than the cutoff.  That is an L-infinity statement
over the non-scissored set, and it is not what the loop used to test.

The load-bearing cell is ``test_rms_passes_where_max_abs_correctly_fails``:
a synthetic case built so the RMS sits comfortably under the cutoff while
one band is 8x above it.  These are different tests, RMS is the looser
one, and that cell is what stops anyone quietly swapping them back.

No WFN, no GPU, no jit -- these run on numpy arrays.
"""
from __future__ import annotations

import numpy as np
import pytest

from gw.sc_iteration import protected_band_convergence


def _energies(nk=3, nb=6):
    rng = np.random.default_rng(0)
    return rng.normal(size=(nk, nb)) * 5.0


def test_max_abs_over_non_scissored_bands_is_the_criterion():
    e_prev = _energies()
    e_new = e_prev.copy()
    keep = np.array([True] * 3 + [False] * 3)
    e_new[0, 0] += 4.0e-3            # inside the set, under a 5 meV cutoff
    v = protected_band_convergence(e_new, e_prev, keep, keep, 5.0e-3)
    assert v.converged
    assert v.max_abs_ev == pytest.approx(4.0e-3)
    assert v.worst_k == 0 and v.worst_band == 0


def test_rms_passes_where_max_abs_correctly_fails():
    """THE cell this change exists for.

    10 k-points x 20 bands, all still except one that moves 40 meV.
    Against a 5 meV cutoff the two tests give OPPOSITE answers:

      * RMS over the set = 40 meV / sqrt(200) = 2.83 meV -> PASSES (wrong)
      * max-abs          = 40 meV = 8x the cutoff        -> FAILS  (right)
    """
    nk, nb = 10, 20
    keep = np.ones(nb, dtype=bool)
    e_prev = np.zeros((nk, nb))
    e_new = np.zeros((nk, nb))
    e_new[0, 0] = 40.0e-3
    cutoff = 5.0e-3

    v = protected_band_convergence(e_new, e_prev, keep, keep, cutoff)

    assert v.rms_protected_ev < cutoff, (
        "the synthetic case must be one where the RMS passes")
    assert v.rms_protected_ev == pytest.approx(
        40.0e-3 / np.sqrt(nk * nb), rel=1e-12)
    assert v.max_abs_ev == pytest.approx(40.0e-3)
    assert not v.converged, (
        "max-abs must fail where a single band sits 8x above the cutoff")


def test_scissored_bands_are_excluded():
    """A large move on a SCISSORED band must not block convergence.

    Scissored diagonals are alpha*E_DFT + beta with the coefficients
    refitted each iteration from the in-range corrections, so counting
    them re-counts in-range drift through the fit.
    """
    e_prev = _energies()
    e_new = e_prev.copy()
    keep = np.array([True] * 3 + [False] * 3)
    e_new[1, 4] += 10.0              # scissored: 10 eV, ignored
    v = protected_band_convergence(e_new, e_prev, keep, keep, 5.0e-3)
    assert v.converged and v.max_abs_ev == 0.0
    # ...but the all-band RMS still SEES it, which is why that number is
    # reported separately and is not the criterion.
    assert v.rms_all_ev > 1.0


def test_non_protected_in_range_bands_still_count():
    """The set is "not scissored", NOT ``protected_mask`` alone.

    ``apply_band_partition`` substitutes a scissor diagonal exactly where
    ``in_range_mask`` is False.  A band in range but not protected keeps
    its own Sigma-derived diagonal and is a genuine degree of freedom.
    Today ``run_sc_driver`` builds both masks equal, so this pins the
    PREDICATE rather than reporting a live defect.
    """
    protected = np.array([True, True, False, False, False, False])
    in_range = np.array([True, True, True, True, False, False])
    e_prev = np.zeros((4, 6))
    e_new = e_prev.copy()
    e_new[2, 3] = 40.0e-3            # in range, NOT protected

    v = protected_band_convergence(e_new, e_prev, protected, in_range, 5.0e-3)
    assert v.n_protected == 4
    assert v.worst_band == 3 and not v.converged

    e_sc = e_prev.copy()
    e_sc[2, 5] = 40.0e-3             # scissored -> correctly ignored
    v2 = protected_band_convergence(e_sc, e_prev, protected, in_range, 5.0e-3)
    assert v2.converged and v2.max_abs_ev == 0.0


def test_summary_labels_which_number_is_the_criterion():
    """A number printed beside a cutoff reads as the thing compared to it."""
    keep = np.ones(4, dtype=bool)
    text = protected_band_convergence(
        np.zeros((2, 4)), np.zeros((2, 4)), keep, keep, 5.0e-3).summary()
    assert "CRITERION" in text and "NOT the criterion" in text


def test_mask_length_must_match_the_active_window():
    """A frozen band window against moved energies is refused, not padded."""
    with pytest.raises(ValueError, match="protected_mask"):
        protected_band_convergence(
            np.zeros((2, 6)), np.zeros((2, 6)),
            np.ones(4, dtype=bool), np.ones(4, dtype=bool), 5.0e-3)


def test_zero_non_scissored_bands_refuses():
    """``max`` over the empty set is vacuously true -- refuse instead."""
    z = np.zeros(6, dtype=bool)
    with pytest.raises(ValueError, match="ZERO"):
        protected_band_convergence(
            np.zeros((2, 6)), np.zeros((2, 6)), z, z, 5.0e-3)


def test_mismatched_energy_shapes_refuse():
    with pytest.raises(ValueError, match="shapes disagree"):
        protected_band_convergence(
            np.zeros((2, 6)), np.zeros((3, 6)),
            np.ones(6, dtype=bool), np.ones(6, dtype=bool), 5.0e-3)

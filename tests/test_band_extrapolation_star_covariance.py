"""The band-extrapolation interior cuts must land on degeneracy-clean edges.

WHY THIS IS THE CHECKABLE FORM OF THE INVARIANT.  The invariant one actually
wants is "the extrapolated Sigma is exactly star-covariant".  MEASURED
2026-08-15, that invariant cannot be evaluated from any artifact: on
``feat/band-extrapolation-sampling-2026-08-15`` @ ``81edc49c`` the fitted
``S_inf`` is printed for VBM/CBM and an envelope and **is written nowhere**.
Running the Si 4x4x4 SOC deck at nband=40 with the feature ON and OFF gives
``sigma_diag.dat`` identical, and every dataset in ``sigma_mnk.h5`` identical
to 8e-15 -- while the log reports an 848 meV correction at the VBM.  So a
star-spread test on the written Sigma passes VACUOUSLY: it is measuring the
un-extrapolated cube.

WHAT WAS ESTABLISHED INSTEAD, and it closes the question:

  1. The fit consumes three partial sums S(N1), S(N2), S(N3), one per bracket
     cut.  Each is exactly the Sigma_c of a run at that nband.
  2. MEASURED on the orbit-closed centroid set, max star spread over the 8
     stars of 64 full-BZ k, bands 0-15:

         cut 32 (clean by   1.7 meV)   sigX 0.0000  sigC 0.0000  sigXC 0.0000
         cut 36 (clean by 153   meV)   sigX 0.0000  sigC 0.0000  sigXC 0.0010
         cut 40 (clean by 818   meV)   sigX 0.0000  sigC 0.0000  sigXC 0.0000

     (0.0010 meV is the DFT-eigenvalue floor between independent NSCF runs.)
  3. A partial sum at a SLICED cut is NOT star-covariant: nband=60 on the
     si_cohsex_debug mean field, min gap 0.000000 eV, gives sigCOH star
     spread 1.957 meV.
  4. ``S(N) = S_inf + A/N`` is a deterministic pointwise function of the three
     per (k, band).  A pointwise function of star-covariant inputs is
     star-covariant.

So S_inf is exactly star-covariant IFF every interior cut is
degeneracy-clean -- which is what this file gates, because it is the part
that is checkable without persisting S_inf.

WHY IT IS NOT AUTOMATIC.  The interior-cut degeneracy snap is a PREFERENCE,
not a constraint: ``plan_band_brackets`` falls back to the unsnapped request
when no clean boundary exists in range, on the reasoning that an interior cut
is "a sampling point on a partial-sum curve, not a Sigma window", so a bad cut
"costs accuracy, not correctness" (<= ~6.4 meV measured).  That reasoning is
sound for the FIT RESIDUAL and silent about star covariance -- the comparison
that justified relaxing it was made against BerkeleyGW's k-resolved curve,
where star covariance is not visible at all.  Item 3 above is the measurement
that reasoning lacked: an unsnapped interior cut costs correctness of the
invariant, not only accuracy.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from common import band_degeneracy as bd

FIXTURE = os.path.join(os.path.dirname(__file__),
                       "regression", "si_cohsex_debug", "WFN.h5")


def _enk_ry():
    h5py = pytest.importorskip("h5py")
    if not os.path.exists(FIXTURE):
        pytest.skip(f"fixture WFN.h5 not present at {FIXTURE}")
    with h5py.File(FIXTURE, "r") as f:
        return np.asarray(f["/mf_header/kpoints/el"])[0]


def _plan(**kw):
    be = pytest.importorskip("gw.band_extrapolation")
    return be, be.plan_band_brackets(**kw)


@pytest.mark.parametrize("nband", [40, 36, 28, 20])
def test_interior_cuts_are_degeneracy_clean_or_say_so(nband):
    """Every interior cut is clean, or the plan RECORDS that it is not.

    Silence is the failure mode this guards: a cut that quietly lands
    mid-multiplet makes the extrapolated Sigma star-dependent, and nothing
    downstream would show it because S_inf is not written anywhere.
    """
    enk = _enk_ry()
    if nband > enk.shape[1]:
        pytest.skip(f"fixture has {enk.shape[1]} bands < nband={nband}")
    try:
        be, plan = _plan(enabled=True, enk_ry=enk, n_occ=8,
                         nb_logical=nband, nb_padded=nband)
    except (bd.BandWindowDegeneracyError, ValueError) as e:
        # REFUSING is a pass.  The property is "never silently slice", and a
        # planner that stops rather than emit a mid-multiplet cut satisfies
        # it.  The two branches differ here on purpose:
        # `feat/band-extrapolation-2026-08-15` REFUSES (this path);
        # `feat/band-extrapolation-sampling-2026-08-15` @ 81edc49c relaxed it
        # to fall back UNSNAPPED, which is the case the assertion below
        # covers.  Both are acceptable to this cell; silence is not.
        assert "clean boundary" in str(e) or "degenerate" in str(e).lower(), (
            f"planner raised, but not about degeneracy: {e}")
        return
    gaps = bd.boundary_min_gaps(enk)
    notes = " ".join(getattr(plan, "notes", ()) or ())
    interior = [int(c) for c in plan.counts[:-1]]
    for cut in interior:
        if 0 < cut < len(gaps) and gaps[cut] <= bd.DEGENERACY_TOL_RY:
            assert "UNSNAPPED" in notes.upper(), (
                f"interior cut {cut} slices a degenerate multiplet (min gap "
                f"{gaps[cut] * 13605.693122994:.3f} meV) and the plan records "
                f"no note about it.  An unsnapped interior cut makes the "
                f"EXTRAPOLATED Sigma star-dependent -- measured 1.957 meV of "
                f"sigCOH star spread at a sliced band edge -- and S_inf is "
                f"written to no artifact, so nothing downstream can see it.")


def test_the_last_count_is_the_deck_band_sum():
    """N3 is the deck's own nband, not a snapped value.

    Pins the one cut that IS a Sigma window: it is governed by
    ``check_zeta_fit_windows`` (which refuses a zero-gap edge since
    2026-08-15), not by the extrapolation's own preference-snap.
    """
    enk = _enk_ry()
    be, plan = _plan(enabled=True, enk_ry=enk, n_occ=8,
                     nb_logical=40, nb_padded=40)
    assert int(plan.counts[-1]) == 40


def test_disabled_plan_is_the_trivial_one():
    """The default must not introduce any cut at all."""
    enk = _enk_ry()
    be, plan = _plan(enabled=False, enk_ry=enk, n_occ=8,
                     nb_logical=40, nb_padded=40)
    assert not plan.enabled
    assert int(plan.counts[-1]) == 40

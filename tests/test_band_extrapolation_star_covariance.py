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


def test_at_least_one_parametrised_arm_actually_plans():
    """The parametrised cell above must not be able to pass on refusals alone.

    THE SECOND HALF OF THE SAME FIX.  Naming the exception classes stops an
    UNRELATED refusal being absorbed, but a degeneracy refusal on every single
    arm would still leave the parametrised cell green while testing nothing --
    it would report four passes for four arms that never reached the property.
    This cell holds all four nbands itself (rather than relying on state shared
    between parametrised runs, which xdist distributes across workers and does
    not share) and asserts that the planner produced a real plan for at least
    one of them.

    It is deliberately weak -- ONE arm, not all four -- because a genuine
    degeneracy refusal on a particular band count is a legitimate outcome that
    this suite must keep tolerating.  What it does not tolerate is the whole
    parametrisation going quiet at once.
    """
    enk = _enk_ry()
    be = pytest.importorskip("gw.band_extrapolation")
    planned, refused = [], []
    for nband in (40, 36, 28, 20):
        if nband > enk.shape[1]:
            continue
        try:
            _, plan = _plan(enabled=True, enk_ry=enk, n_occ=8,
                            nb_logical=nband, nb_padded=nband)
        except (bd.BandWindowDegeneracyError,
                be.BandExtrapolationRefused) as e:
            refused.append((nband, f"{type(e).__name__}: {e}"))
        else:
            planned.append((nband, tuple(int(c) for c in plan.counts)))
    assert planned, (
        "plan_band_brackets refused EVERY parametrised band count "
        f"{[n for n, _ in refused]}, so the star-covariance parametrisation "
        f"above is passing without ever exercising an interior cut.  "
        f"Refusals:\n" + "\n".join(f"  nband={n}: {m}" for n, m in refused))


@pytest.mark.parametrize("nband", [40, 36, 28, 20])
def test_interior_cuts_are_degeneracy_clean_or_say_so(nband, record_property):
    """Every interior cut is clean, or the plan RECORDS that it is not.

    Silence is the failure mode this guards: a cut that quietly lands
    mid-multiplet makes the extrapolated Sigma star-dependent, and nothing
    downstream would show it because S_inf is not written anywhere.
    """
    enk = _enk_ry()
    if nband > enk.shape[1]:
        pytest.skip(f"fixture has {enk.shape[1]} bands < nband={nband}")
    be = pytest.importorskip("gw.band_extrapolation")
    try:
        _, plan = _plan(enabled=True, enk_ry=enk, n_occ=8,
                        nb_logical=nband, nb_padded=nband)
    except (bd.BandWindowDegeneracyError,
            be.BandExtrapolationRefused) as e:
        # REFUSING is a pass FOR THIS PROPERTY, and only for this one.  The
        # property is "never silently slice", and a planner that stops rather
        # than emit a mid-multiplet cut satisfies it.  The two branches differ
        # here on purpose: `feat/band-extrapolation-2026-08-15` REFUSES (this
        # path); `feat/band-extrapolation-sampling-2026-08-15` @ 81edc49c
        # relaxed it to fall back UNSNAPPED, which the assertion below covers.
        #
        # ⚠ THE EXCEPT CLAUSE USED TO NAME `ValueError`, AND THAT MADE THIS
        # CELL STOP BEING A GATE.  `BandExtrapolationRefused` SUBCLASSES
        # `ValueError` (`gw/band_extrapolation.py`), so catching the base
        # swallowed EVERY refusal `plan_band_brackets` can raise -- including
        # ones that have nothing to do with degeneracy, and including refusals
        # added after this test was written.  The 2026-08-16 band-count gate
        # change (`n_cond <= n_occ` -> `n_cond < n_occ`) passed this cell both
        # BEFORE and AFTER for exactly that reason: a test that counts a
        # refusal as success is not a gate.  Naming the two classes explicitly
        # is what makes an unrelated `ValueError` -- a shape error, a typo, a
        # new precondition -- come out as the FAILURE it is instead of a pass.
        msg = str(e)
        assert "clean boundary" in msg or "degenerate" in msg.lower(), (
            f"plan_band_brackets refused at nband={nband}, but NOT about "
            f"degeneracy: {type(e).__name__}: {msg}\n"
            f"This cell accepts a degeneracy refusal as satisfying "
            f"'never silently slice'.  It does not accept any other refusal, "
            f"because a refusal for another reason means the planner never "
            f"reached the property under test and this cell measured nothing.")
        # Reported as a refusal, not silently as a pass: without this the arm
        # is indistinguishable in the run log from one that planned cleanly.
        record_property("outcome", f"REFUSED(degeneracy): {msg}")
        return
    record_property("outcome", f"PLANNED counts={tuple(int(c) for c in plan.counts)}")
    # the deck's untruncated ladder, not a window -- the plan's interior
    # cuts are checked against it
    gaps = bd.boundary_min_gaps(enk, is_full_spectrum=True)
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

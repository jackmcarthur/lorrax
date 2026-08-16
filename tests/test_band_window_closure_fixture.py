"""The BerkeleyGW anchor fixture's band edge slices a degenerate multiplet.

Pins the 2026-08-15 measurement so the finding cannot be lost to a re-cut, and
so the guard that now refuses it cannot be quietly relaxed again.

WHY THESE CELLS ARE CHEAP.  Every assertion here reads the SHIPPED `WFN.h5`
eigenvalues and calls `common.band_degeneracy` — no GPU, no Sigma run, no
BerkeleyGW.  The expensive part (the star-spread measurement that established
*why* this matters) is recorded in the fixture README and in
`tests/known_failures/2026-08-10-ibz-cascade-vs-full-bz-sigma-6x6x6.md`; what
is gated here is the band-window fact those numbers rest on.

WHAT THE MEASUREMENT WAS.  Same exactly-orbit-closed centroid set, same
`zeta_rcond = 1e-10`, P=4 fixed, only the band-sum edge moving; max star
spread over the 8 stars of the 64 full-BZ k, bands 0-15:

    nband=60 (slices, gap 0.000000 eV)  sigSX 0.0270  sigCOH 1.9570  V_H 0.0990
    nband=40 (clean by 818 meV)         sigSX 0.0000  sigCOH 0.0000  V_H 0.0000
    nband=36 (clean by 157 meV)         sigSX 0.0000  sigCOH 0.0000  V_H 0.0000

Exactly zero on every column at a clean edge.  Note which row is nearly clean
even on the slicing edge: sigSX.  That is why this went unnoticed -- the
2026-08-10 investigation decided on Sigma_x, and Sigma_x is the term a sliced
edge barely moves.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from common import band_degeneracy as bd

FIXTURE = os.path.join(os.path.dirname(__file__),
                       "regression", "si_cohsex_debug", "WFN.h5")

#: Measured 2026-08-15 on the shipped WFN.h5.  Both lists are exhaustive over
#: the band range the fixture's decks can reach.
CLEAN_EDGES = (8, 16, 20, 28, 36, 40)
SLICING_EDGES = (24, 32, 44, 48, 52, 56, 60)

#: The production deck's edge.  It is in SLICING_EDGES, and that is the point.
SHIPPED_NBAND = 60


def _enk_ry():
    h5py = pytest.importorskip("h5py")
    if not os.path.exists(FIXTURE):
        pytest.skip(f"fixture WFN.h5 not present at {FIXTURE}")
    with h5py.File(FIXTURE, "r") as f:
        return np.asarray(f["/mf_header/kpoints/el"])[0]


def test_shipped_anchor_band_edge_slices_a_multiplet():
    """`nband = 60` cuts a multiplet at EXACTLY zero gap on this mean field."""
    enk = _enk_ry()
    gaps = bd.boundary_min_gaps(enk)
    gap_ry = float(gaps[SHIPPED_NBAND])
    assert gap_ry <= bd.DEGENERACY_TOL_RY, (
        f"nband={SHIPPED_NBAND} is no longer a slicing edge on this WFN "
        f"(min gap {gap_ry * 13605.693122994:.3f} meV).  If the fixture's "
        f"mean field changed, re-measure CLEAN_EDGES and update the README "
        f"table -- do not just delete this cell.")
    # Not merely "within tolerance": it is a genuine exact degeneracy.
    assert gap_ry < 1e-9, (
        f"expected an EXACT degeneracy at band {SHIPPED_NBAND}, got "
        f"{gap_ry:.3e} Ry")


def test_clean_and_slicing_edges_are_what_was_measured():
    """The two lists in the README are the ones this WFN actually has."""
    enk = _enk_ry()
    gaps = bd.boundary_min_gaps(enk)
    for b in CLEAN_EDGES:
        assert gaps[b] > bd.DEGENERACY_TOL_RY, (
            f"edge {b} is documented CLEAN but its min gap is "
            f"{gaps[b] * 13605.693122994:.3f} meV")
    for b in SLICING_EDGES:
        assert gaps[b] <= bd.DEGENERACY_TOL_RY, (
            f"edge {b} is documented SLICING but its min gap is "
            f"{gaps[b] * 13605.693122994:.3f} meV")


def test_strict_mode_refuses_the_shipped_edge_and_accepts_a_clean_one():
    """The guard REFUSES the shipped edge and passes a clean one.

    This is the cell that would have caught it: before 2026-08-15 the zeta
    seam passed ``snap``, so it printed ``edge 60 min gap 0 meV`` and carried
    on.  ``strict`` is the default at every other seam.
    """
    enk = _enk_ry()
    with pytest.raises(bd.BandWindowDegeneracyError):
        bd.check_band_window(enk, 0, SHIPPED_NBAND, mode="strict",
                             where="shipped anchor deck")
    for b in (36, 40):
        bd.check_band_window(enk, 0, b, mode="strict",
                             where=f"clean edge {b}")


def test_snap_is_a_named_override_not_a_default():
    """``snap`` must warn rather than raise -- it is the escape hatch."""
    enk = _enk_ry()
    seen = []
    bd.check_band_window(enk, 0, SHIPPED_NBAND, mode="snap",
                         where="shipped anchor deck", log=seen.append)
    assert seen, "snap mode produced no diagnostic at all"
    assert "degenerate multiplet" in " ".join(seen)


def test_default_mode_is_strict():
    """If this flips, every seam flips with it -- including the zeta window."""
    assert bd.DEFAULT_MODE == "strict"

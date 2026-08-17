"""The ζ fit's band windows must be point-group-invariant subspaces.

THE INCIDENT.
``tests/known_failures/2026-08-10-ibz-cascade-vs-full-bz-sigma-6x6x6.md``.
The IBZ cascade builds the full BZ by ROTATING the wedge, and a rotation
sends ψ_n(k) into a combination of its degenerate partners at Sk.  A band
window holding half a multiplet therefore has a rotation image that leaves
it, so the pair space the ζ fit represents is not invariant and the k-star
identity fails for every object built on ζ — Σ_x included, which contains
no screening at all.

Si 6×6×6 at ``nband=60`` cut a 4-fold manifold (bands 59..62) at 4 of the
16 wedge k.  Measured: Σ_x k-star spread 0.0640 meV and Σ_c 38.785 meV,
against 0.0000 meV and 0.083 meV at the nearest degeneracy-closed window
``nband=68``.  The guard that catches this already existed in
``common.band_degeneracy`` and the ζ fit simply never called it.

These cells pin the PRIMITIVES the ζ seam relies on, at the two window
shapes the incident produced.  CPU-only, no wavefunction file.
"""
import numpy as np
import pytest

from common.band_degeneracy import (
    BandWindowDegeneracyError,
    boundary_min_gaps,
    check_band_window,
)

_EV_PER_RY = 13.605693122994


def _si666_like():
    """A spectrum with the incident's shape: a 4-fold manifold at 58..61
    (0-indexed) that is exactly degenerate at 4 of 16 k, so an edge at 60
    slices it and an edge at 62 does not."""
    rng = np.random.default_rng(11)
    e = np.sort(rng.uniform(0.0, 5.0, size=(16, 128)), axis=1)
    # Separate the block from its neighbours so 58 and 62 are clean edges.
    for k in range(16):
        e[k, 58:62] = e[k, 58]
        e[k, 57] = e[k, 58] - 0.05
        e[k, 62] = e[k, 61] + 0.05
    return e


def test_the_open_window_is_caught_at_the_zeta_seam():
    """nband=60 slices the 4-fold block -> strict must refuse."""
    e = _si666_like()
    with pytest.raises(BandWindowDegeneracyError, match=r"upper boundary at band 60"):
        check_band_window(e, 0, 60, mode="strict", where="ISDF left window")


def test_the_closed_window_passes():
    """nband=62 ends on the block boundary -> nothing to report."""
    e = _si666_like()
    check_band_window(e, 0, 62, mode="strict", where="ISDF left window")


def test_snap_warns_when_a_caller_explicitly_asks_for_the_primitive():
    """The primitive keeps its debug mode; the production ζ seam is strict."""
    e = _si666_like()
    said = []
    check_band_window(e, 0, 60, mode="snap", log=said.append,
                      where="ISDF left window")
    assert said, "an open window must not pass in silence"
    assert "ISDF left window" in said[0]
    assert "degenerate multiplet" in said[0]


def test_off_is_silent_and_that_is_deliberate():
    e = _si666_like()
    said = []
    check_band_window(e, 0, 60, mode="off", log=said.append)
    assert said == []


def test_boundary_min_gaps_separates_the_sliced_edge_from_the_clean_ones():
    """The measurement the ζ seam prints.  The sliced edge and the clean
    ones must be separated by orders of magnitude, not by a hair — that is
    what makes the tolerance a classification and not a tuned knob."""
    e = _si666_like()
    g = boundary_min_gaps(e)
    _MEV_PER_RY = _EV_PER_RY * 1e3
    assert g[60] == pytest.approx(0.0, abs=1e-15)     # inside the manifold
    assert g[58] * _MEV_PER_RY > 100.0                # a real gap, in meV
    assert g[62] * _MEV_PER_RY > 100.0
    # Orders of magnitude apart, which is the point: no tolerance anywhere
    # between them changes the verdict.
    assert g[58] / max(g[60], 1e-300) > 1e6


def test_outer_boundaries_cut_nothing_and_are_infinite():
    """b=0 and b=nb separate nothing, so they can never be 'open'."""
    e = _si666_like()
    g = boundary_min_gaps(e)
    assert np.isinf(g[0]) and np.isinf(g[e.shape[1]])
    # ...and a window spanning the whole file is therefore always closed.
    check_band_window(e, 0, e.shape[1], mode="strict")


def test_both_isdf_windows_are_checked_not_just_the_outer_one():
    """The ζ fit has TWO windows -- left (b0,b3) and right (b1,b4) -- and
    the right one's LOWER edge is a cut too.  A guard that only looked at
    nband would miss an nval that slices."""
    e = _si666_like()
    # b1 = 60 as a LOWER edge is just as unsafe as 60 as an upper edge.
    with pytest.raises(BandWindowDegeneracyError, match=r"lower boundary at band 60"):
        check_band_window(e, 60, 128, mode="strict", where="ISDF right window")

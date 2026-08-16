"""A per-band star spread is not a symmetry diagnostic inside a multiplet.

Within a degenerate multiplet the band index is arbitrary: any unitary
mixing of the subspace is an equally valid eigenbasis, so the eigensolver
may order or mix members differently at symmetry-equivalent k and nothing
forbids it.  ``Re Σ_bb`` for a single ``b`` inside such a multiplet is
therefore NOT invariant under the point group, and a max−min of it across a
star measures the solver's gauge as much as the physics.  The TRACE over the
whole multiplet IS invariant.

MEASURED on the Si production deck, 2026-08-15 — **60 of 60 bands sit inside
a multiplet** (groups of 4, 4, 8, 8, 8, 8 and one of 20), tolerance-insensitive
from 1 meV down to 13.6 µeV:

    bands   size   per-band max   multiplet trace   ratio
     0-7      8       0.980 meV      0.134 meV       7.3x
     8-15     8       2.611          0.593           4.4x
    16-19     4       4.821          2.302           2.1x
    20-27     8       7.267          2.835           2.6x
    28-35     8      10.020          2.909           3.4x
    36-39     4       9.471          6.734           1.4x
    40-59    20      41.338          3.604          11.5x

The headline **41.338 meV lives in the 20-fold block and is 91 % gauge.**
The worst invariant residual over the whole window is 6.734 meV, at bands
36-39 — a different place entirely.

AND ALL OF IT IS MEASURED AT A SLICED BAND EDGE.  That deck runs
``nband = 60`` on a 62-band WFN and edge 60 has a min gap over k of
**0.000000 meV**; at a clean edge (40 or 36) every Σ channel's within-star
spread is **exactly 0.0000**.  So the table above describes a run that should
not have been made, and the reason to keep the diagnostic is that it tells a
gauge artifact from a real break once the edge is clean.

These cells pin the property, not the numbers: that the multiplet-resolved
spread is defined, that it is bounded by the per-band one on a degenerate
block, and that it is EXACTLY equal to it where a band is isolated.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from gw.gw_output import (_star_spread_of_sigma_diag,                # noqa: E402
                          _star_spread_over_multiplets)


class _Sym:
    """The two rows of a ``SymMaps`` these functions read."""

    def __init__(self, irr_idx_k):
        self.irr_idx_k = np.asarray(irr_idx_k, dtype=np.int32)


def _sigma_from_diag(diag):
    """(nk, nb) real diagonals -> (nk, nb, nb) with those diagonals."""
    diag = np.asarray(diag, dtype=np.float64)
    nk, nb = diag.shape
    out = np.zeros((nk, nb, nb), dtype=np.complex128)
    idx = np.arange(nb)
    out[:, idx, idx] = diag
    return out


#: Two k in ONE star, two bands, and the two bands EXACTLY degenerate.
#: This is the whole hazard in four numbers: the star's two members carry
#: the same subspace (trace 3.0 at both) with the label swapped between
#: them.  The per-band metric reports the swap as a 1.0 eV symmetry break;
#: the trace reports 0.
_SWAPPED = np.array([[1.0, 2.0],
                     [2.0, 1.0]])
_DEGENERATE_E_RY = np.array([[0.5, 0.5],
                             [0.5, 0.5]])
#: The same Σ with the two bands WELL SEPARATED — now the labels mean
#: something and the same numbers ARE a real disagreement.
_SPLIT_E_RY = np.array([[0.5, 1.5],
                        [0.5, 1.5]])


def test_a_label_swap_inside_a_multiplet_reads_as_a_break_per_band():
    """The per-band metric cannot tell a gauge swap from a symmetry break."""
    sym = _Sym([0, 0])
    per_band, worst, n = _star_spread_of_sigma_diag(
        _sigma_from_diag(_SWAPPED), sym)
    assert n == 2
    assert np.allclose(per_band, [1.0, 1.0]), per_band
    assert worst == pytest.approx(1.0)


def test_the_multiplet_trace_sees_the_swap_for_what_it_is():
    """Same Σ, same star — degenerate bands, so the trace says ZERO."""
    sym = _Sym([0, 0])
    got = _star_spread_over_multiplets(
        _sigma_from_diag(_SWAPPED), sym, _DEGENERATE_E_RY)
    assert got is not None
    assert np.allclose(got, 0.0, atol=1e-12), (
        f"a pure label swap inside a degenerate multiplet must be invisible "
        f"to the trace metric; got {got}")


def test_the_same_numbers_on_SPLIT_bands_are_a_real_break():
    """The metric is not simply blind — separate the bands and it fires.

    Without this cell the one above would pass on a function that returned
    zeros unconditionally.  Same Σ, same star, only the ENERGIES differ.
    """
    sym = _Sym([0, 0])
    got = _star_spread_over_multiplets(
        _sigma_from_diag(_SWAPPED), sym, _SPLIT_E_RY)
    assert got is not None
    assert np.allclose(got, [1.0, 1.0]), (
        f"with the bands split, each is its own subspace and the trace "
        f"metric must reproduce the per-band answer exactly; got {got}")


def test_where_bands_are_isolated_the_two_metrics_agree_exactly():
    """Not merely close — the trace over a size-1 subspace IS the band."""
    rng = np.random.default_rng(5)
    nk, nb = 6, 5
    diag = rng.standard_normal((nk, nb))
    e_ry = np.tile(np.arange(nb, dtype=np.float64), (nk, 1))   # all split
    sym = _Sym([0, 0, 0, 1, 1, 1])
    per_band, _, _ = _star_spread_of_sigma_diag(_sigma_from_diag(diag), sym)
    trace = _star_spread_over_multiplets(_sigma_from_diag(diag), sym, e_ry)
    assert np.allclose(per_band, trace, rtol=0, atol=1e-12)


def test_the_trace_metric_never_exceeds_the_per_band_one_on_a_block():
    """Sanity bound: averaging over a subspace cannot manufacture spread.

    max-min of a SUM over ``m`` bands, divided by ``m``, is at most the max
    over those bands of the per-band max-min.  A violation means the two are
    not measuring the same star or the same rows.
    """
    rng = np.random.default_rng(11)
    nk, nb = 8, 6
    diag = rng.standard_normal((nk, nb))
    # bands 2,3,4 degenerate; 0,1,5 isolated
    e_ry = np.tile(np.array([0.0, 1.0, 2.0, 2.0, 2.0, 3.0]), (nk, 1))
    sym = _Sym([0, 0, 0, 0, 1, 1, 1, 1])
    per_band, _, _ = _star_spread_of_sigma_diag(_sigma_from_diag(diag), sym)
    trace = _star_spread_over_multiplets(_sigma_from_diag(diag), sym, e_ry)
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 6)):
        assert trace[lo:hi].max() <= per_band[lo:hi].max() + 1e-12, (
            f"bands [{lo}, {hi}): trace {trace[lo:hi].max()} exceeds "
            f"per-band {per_band[lo:hi].max()}")


def test_it_declines_rather_than_guessing_when_it_has_no_energies():
    """No ladder, no multiplets — return None, never a plausible zero.

    "Not measured" and "measured zero" are the two things this diagnostic
    must never confuse; the writer then emits no header row at all.
    """
    sym = _Sym([0, 0])
    assert _star_spread_over_multiplets(
        _sigma_from_diag(_SWAPPED), sym, None) is None
    assert _star_spread_over_multiplets(
        _sigma_from_diag(_SWAPPED), None, _DEGENERATE_E_RY) is None
    # Wrong band count on the ladder is also a decline, not a crash.
    assert _star_spread_over_multiplets(
        _sigma_from_diag(_SWAPPED), sym, np.zeros((2, 7))) is None


def test_a_singleton_star_contributes_nothing():
    """A star with one member has no spread to report."""
    sym = _Sym([0, 1])
    got = _star_spread_over_multiplets(
        _sigma_from_diag(_SWAPPED), sym, _DEGENERATE_E_RY)
    assert np.allclose(got, 0.0)

"""A protected band's multiplet partners are protected too.

Owner ruling, 2026-08-16: *"I want degenerate spaces degenerate in LORRAX."*

``protected_mask`` comes from ``scissor.classify_bands_in_grid``, an ALL-k
energy-window predicate.  Nothing makes it contiguous and nothing makes its
edges fall between multiplets — bands degenerate at one k need not be
degenerate at another.  When an edge lands inside a degenerate manifold,
``apply_band_partition`` gives one member full off-diagonal Σ and the other a
scalar scissor, and half a multiplet is not a subspace of anything: within a
degenerate manifold the band label is arbitrary, so the answer starts depending
on the eigensolver's ordering, and the result is ``eigh``'d and reported as QP
energies.

MEASURED on the only committed SC deck (``gnppm_debug``): **no boundary splits
a multiplet**, so the promotion is a no-op there and every output is
byte-identical.  These cells therefore carry the whole burden of proving the
promotion works, because no in-tree deck exercises it.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                          # noqa: E402

from gw.band_partition import BandPartition                      # noqa: E402


#: A ladder whose bands 2,3,4 are EXACTLY degenerate at every k and whose
#: other boundaries are wide open.  Absolute indexing; the active window is
#: carved out of it by ``band_offset`` in the cells below.
def _ladder(nk=4):
    e = np.tile(np.array([0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0]), (nk, 1))
    return e


def _partition(mask):
    m = jnp.asarray(np.asarray(mask, dtype=bool))
    return BandPartition(protected_mask=m, in_range_mask=m)


def test_a_split_multiplet_is_reported():
    """The edge falls between bands 3 and 4, inside the 2-3-4 manifold."""
    part = _partition([False, False, True, True, False, False, False])
    lines = []
    n, worst = part.report_multiplet_splits(_ladder(), 0, print_fn=lines.append)
    assert n == 1, (n, lines)
    assert worst == pytest.approx(0.0, abs=1e-9)
    assert any("CUT A DEGENERATE MULTIPLET" in ln for ln in lines)
    assert any("absolute 4" in ln for ln in lines)


def test_a_clean_partition_reports_no_split():
    """RED TWIN: the report must not fire on every mask.

    Bands 2,3,4 all protected — the edges are at 2 and 5, both wide gaps.
    """
    part = _partition([False, False, True, True, True, False, False])
    lines = []
    n, worst = part.report_multiplet_splits(_ladder(), 0, print_fn=lines.append)
    assert n == 0 and worst == 0.0
    assert any("no boundary splits a multiplet" in ln for ln in lines)


def test_the_promotion_grows_the_mask_to_the_whole_multiplet():
    part = _partition([False, False, True, True, False, False, False])
    out = part.promoted_to_multiplets(_ladder(), 0, print_fn=lambda *_: None)
    got = np.asarray(out.protected_mask)
    assert got.tolist() == [False, False, True, True, True, False, False], got


def test_the_promotion_GROWS_and_never_shrinks():
    """Direction matters, and it is not arbitrary.

    Dropping the protected member would remove off-diagonal Σ from a band the
    ω-grid actually covers — a loss of physics to buy the same invariance.
    Growing admits a band whose Σ is edge-clamped, which is visible and is
    what ``warn_if_protected_outside_grid`` is for.
    """
    for mask in ([False, False, True, False, False, False, False],
                 [False, False, False, False, True, False, False],
                 [False, False, True, True, False, False, False]):
        part = _partition(mask)
        out = np.asarray(part.promoted_to_multiplets(
            _ladder(), 0, print_fn=lambda *_: None).protected_mask)
        assert out.sum() >= np.asarray(mask).sum()
        assert np.all(out | np.asarray(mask) == out), "a band was un-protected"


def test_the_promotion_is_idempotent():
    part = _partition([False, False, True, False, False, False, False])
    once = part.promoted_to_multiplets(_ladder(), 0, print_fn=lambda *_: None)
    twice = once.promoted_to_multiplets(_ladder(), 0, print_fn=lambda *_: None)
    assert np.array_equal(np.asarray(once.protected_mask),
                          np.asarray(twice.protected_mask))


def test_a_clean_mask_is_left_exactly_alone():
    """The no-op case — which is what the one committed SC deck measures."""
    mask = [False, False, True, True, True, False, False]
    part = _partition(mask)
    out = part.promoted_to_multiplets(_ladder(), 0, print_fn=lambda *_: None)
    assert np.asarray(out.protected_mask).tolist() == mask


def test_in_range_mask_is_NOT_promoted():
    """It answers a different question, and that answer is per band.

    ``in_range_mask`` says "is this band's Σ on the ω-grid at every k".  That
    is a property of the band's own energies, not of the manifold it sits in,
    so promoting it would assert something the grid does not support.
    """
    part = _partition([False, False, True, False, False, False, False])
    out = part.promoted_to_multiplets(_ladder(), 0, print_fn=lambda *_: None)
    assert np.array_equal(np.asarray(out.in_range_mask),
                          np.asarray(part.in_range_mask))
    assert not np.array_equal(np.asarray(out.protected_mask),
                              np.asarray(out.in_range_mask))


def test_band_offset_maps_the_active_window_onto_absolute_bands():
    """The mask is over the ACTIVE window; the ladder is absolute.

    With ``band_offset=2`` the active window starts at absolute band 2, so
    active band 1 is absolute 3 — inside the manifold.  Getting this wrong
    would test the wrong boundaries and quietly certify anything.
    """
    active = [True, False, False, False, False]          # absolute 2..6
    part = _partition(active)
    lines = []
    n, _ = part.report_multiplet_splits(_ladder(), 2, print_fn=lines.append)
    assert n == 1, lines
    assert any("active band 1 (absolute 3)" in ln for ln in lines)
    out = np.asarray(part.promoted_to_multiplets(
        _ladder(), 2, print_fn=lambda *_: None).protected_mask)
    assert out.tolist() == [True, True, True, False, False], out


def test_the_report_demands_the_untruncated_ladder():
    """It passes ``is_full_spectrum=True``, so a window would be a lie.

    Pinned because the whole reason ``boundary_min_gaps`` grew that argument
    is that handed a window it returns +inf at the outer edge and cannot see
    the cut that produced it.  A caller here that started passing
    ``e_dft_active_kn_ry`` would silently stop checking the top boundary.
    """
    import ast
    import inspect
    src = inspect.getsource(BandPartition.report_multiplet_splits)
    tree = ast.parse(src.lstrip().replace("\n    ", "\n"))
    calls = [ast.unparse(n) for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and "boundary_min_gaps" in ast.unparse(n.func)]
    assert calls and all("is_full_spectrum=True" in c for c in calls), calls

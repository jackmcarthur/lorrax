"""The multiplet guard on BSE band-window selection, and its red twins.

Every cell here carries BOTH sides: the split multiplet that must fire the
guard, and the clean window that must pass in silence.  A guard that fires on
everything is the same defect as a guard that fires on nothing, and only the
pair distinguishes them.

Pure numpy — no jax, no FFI, no deck.  These run anywhere, which is the point:
the guard sits on the path every BSE driver takes, so its gate must never be
the one that gets skipped for want of a GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.band_degeneracy import (
    BandWindowDegeneracyError, DEGENERACY_TOL_RY, boundary_min_gaps,
    check_band_window, resolve_band_window)

RY2EV = 13.6056980659
MEV = 1.0e-3 / RY2EV                      # 1 meV in Ry


def _kramers_spectrum(nk=4, n_pairs=8, seed=3):
    """(nk, 2*n_pairs) energies whose bands come in EXACT Kramers pairs.

    The SOC+TRS case the guard exists for: every eigenvalue is doubled, so
    every ODD band count splits a pair and every EVEN one does not.
    """
    rng = np.random.default_rng(seed)
    e_pair = np.cumsum(0.05 + 0.05 * rng.random((nk, n_pairs)), axis=1)
    return np.repeat(e_pair, 2, axis=1)      # (nk, 2*n_pairs)


def _nondegenerate_spectrum(nk=4, nb=16, seed=5):
    """(nk, nb) energies with every band well separated — no multiplets."""
    rng = np.random.default_rng(seed)
    return np.cumsum(0.05 + 0.05 * rng.random((nk, nb)), axis=1)


# ---------------------------------------------------------------------------
# 1.  the instrument itself
# ---------------------------------------------------------------------------
def test_boundary_min_gaps_is_a_min_over_k_p1():
    """The gap reported for a boundary is the TIGHTEST one in the BZ."""
    e = np.array([[0.0, 1.0],            # k=0: gap 1.0
                  [0.0, 0.5],            # k=1: gap 0.5  <- the min
                  [0.0, 2.0]])           # k=2: gap 2.0
    gaps = boundary_min_gaps(e)
    assert gaps.shape == (3,)
    assert np.isinf(gaps[0]) and np.isinf(gaps[2]), (
        "the outer boundaries cut nothing and must be +inf, else the snap "
        "walks off the end of the spectrum")
    assert gaps[1] == pytest.approx(0.5), (
        f"expected the min over k (0.5), got {gaps[1]} — a mean or a "
        f"per-k-first would let a boundary that is safe at MOST k through")


# ---------------------------------------------------------------------------
# 2.  snap mode — the owner-requested default
# ---------------------------------------------------------------------------
def test_split_kramers_pair_snaps_outward_p1(capsys):
    """GREEN: an odd band count splits a pair and is widened outward."""
    e = _kramers_spectrum(n_pairs=8)          # 16 bands, 8 exact pairs
    n_occ = 8                                  # 4 valence pairs

    # n_val=3 puts the valence boundary at band 5, INSIDE the pair (4, 5).
    # n_cond=3 puts the conduction boundary at band 11, inside pair (10, 11).
    n_val, n_cond = resolve_band_window(e, n_occ, 3, 3, where="test")
    assert (n_val, n_cond) == (4, 4), (
        f"expected the window to widen outward to whole pairs (4, 4), got "
        f"({n_val}, {n_cond})")

    out = capsys.readouterr().out
    assert "SNAPPED OUTWARD" in out, "the snap must be loud, not silent"
    assert "n_val=4" in out and "n_cond=4" in out, (
        f"the warning must name the new counts; got:\n{out}")
    assert "meV" in out and "at k=" in out, (
        f"the warning must name the multiplet gap and the k at which it is "
        f"tightest — that is what makes it actionable; got:\n{out}")


def test_clean_window_passes_in_silence_p1(capsys):
    """RED TWIN of the cell above: the FALSE case must not fire.

    Same spectrum, same guard, an EVEN count that respects every pair.  If
    this fires, the guard is flagging the window rather than the cut, and the
    green cell above proves nothing.
    """
    e = _kramers_spectrum(n_pairs=8)
    n_occ = 8
    n_val, n_cond = resolve_band_window(e, n_occ, 4, 4, where="test")
    assert (n_val, n_cond) == (4, 4), (
        f"a multiplet-safe window must come back UNCHANGED, got "
        f"({n_val}, {n_cond})")
    out = capsys.readouterr().out
    assert out == "", (
        f"RED TWIN DID NOT GO RED: the guard printed on a clean window, so "
        f"the 'it warns' assertion in the green cell is vacuous.  Output "
        f"was:\n{out}")


def test_nondegenerate_spectrum_never_snaps_p1(capsys):
    """RED TWIN: with no multiplets anywhere, EVERY count is safe."""
    e = _nondegenerate_spectrum(nb=16)
    for nv in range(1, 8):
        for nc in range(1, 8):
            got = resolve_band_window(e, 8, nv, nc, where="test")
            assert got == (nv, nc), (
                f"RED TWIN DID NOT GO RED: ({nv}, {nc}) was changed to {got} "
                f"on a spectrum with no degeneracies at all — the guard is "
                f"snapping on band SPACING, not on multiplets")
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# 3.  strict mode
# ---------------------------------------------------------------------------
def test_strict_mode_raises_on_a_split_multiplet_p1():
    """GREEN: strict refuses rather than resizing the problem."""
    e = _kramers_spectrum(n_pairs=8)
    with pytest.raises(BandWindowDegeneracyError) as ei:
        resolve_band_window(e, 8, 3, 3, mode="strict", where="test")
    msg = str(ei.value)
    assert "--n-val 4" in msg and "--n-cond 4" in msg, (
        f"a strict refusal must tell the user the counts that would work; "
        f"got:\n{msg}")


def test_strict_mode_passes_a_clean_window_p1():
    """RED TWIN: strict must not refuse the FALSE case."""
    e = _kramers_spectrum(n_pairs=8)
    assert resolve_band_window(e, 8, 4, 4, mode="strict",
                               where="test") == (4, 4), (
        "RED TWIN DID NOT GO RED: strict mode raised (or altered) a "
        "multiplet-safe window, so it refuses everything and the green cell "
        "above tests nothing")


# ---------------------------------------------------------------------------
# 4.  off mode
# ---------------------------------------------------------------------------
def test_off_mode_is_a_true_no_op_p1(capsys):
    """The off switch returns the request untouched AND says nothing."""
    e = _kramers_spectrum(n_pairs=8)
    assert resolve_band_window(e, 8, 3, 3, mode="off", where="test") == (3, 3)
    assert capsys.readouterr().out == "", (
        "'off' that still prints is not off — it is a warning nobody can "
        "silence, which is how guards get deleted")


def test_unknown_mode_is_refused_p1():
    e = _kramers_spectrum(n_pairs=8)
    with pytest.raises(ValueError, match="mode="):
        resolve_band_window(e, 8, 3, 3, mode="warn", where="test")


# ---------------------------------------------------------------------------
# 5.  the tolerance is load-bearing
# ---------------------------------------------------------------------------
def test_tolerance_separates_a_5p9_meV_boundary_p1():
    """The 5.9 meV Si-deck boundary: cut at the 1 meV default, caught at 10.

    This pins the tolerance as a DECISION rather than a magic number.  The
    default must be tight enough to leave a 5.9 meV boundary alone (it is a
    real gap, not a multiplet) and the knob must be able to reach it.
    """
    nk, nb = 3, 6
    e = np.tile(np.arange(nb, dtype=np.float64) * 0.1, (nk, 1))
    # Boundary 4 (the conduction top for n_occ=2, n_cond=2) gets the 5.9 meV
    # gap.  Deliberately NOT the midline, which is reported and never snapped.
    e[:, 4:] += -0.1 + 5.9 * MEV

    assert boundary_min_gaps(e)[4] == pytest.approx(5.9 * MEV, rel=1e-9)

    # default 1 meV: 5.9 meV is a genuine gap, leave it alone.
    assert resolve_band_window(e, 2, 2, 2, where="test") == (2, 2)
    # 10 meV: now it reads as one multiplet and the window widens.
    n_val, n_cond = resolve_band_window(
        e, 2, 2, 2, tol_ry=10.0 * MEV, mode="snap", where="test")
    assert (n_val, n_cond) == (2, 3), (
        f"RED TWIN DID NOT GO RED: raising the tolerance above the boundary "
        f"gap did not widen the conduction window (got {n_val}, {n_cond}), "
        f"so the tolerance is not wired in")


def test_default_tolerance_is_one_meV_p1():
    assert DEGENERACY_TOL_RY == pytest.approx(1.0e-3 / RY2EV, rel=1e-12), (
        "the documented default is 1 meV; if this moves, the docstring, the "
        "--degeneracy-tol-ry help text and EXCITON_BANDS_FEATURES.md all "
        "state a number that is no longer true")


# ---------------------------------------------------------------------------
# 6.  the cases the snap CANNOT repair, reported rather than hidden
# ---------------------------------------------------------------------------
def test_multiplet_open_at_the_top_is_reported_p1(capsys):
    """Widening runs out of bands: say so, do not return a quiet cut window."""
    nk = 3
    e = np.tile(np.array([0.0, 0.5, 1.0, 1.0, 1.0]), (nk, 1))  # top 3 degenerate
    resolve_band_window(e, 2, 1, 1, where="test")
    out = capsys.readouterr().out
    assert "still open at the TOP" in out, (
        f"a multiplet that runs past the last available band must be named; "
        f"got:\n{out}")


def test_gapless_midline_is_reported_and_not_snapped_p1(capsys):
    """A degenerate v/c split is a different bug and gets its own message."""
    nk = 3
    e = np.tile(np.array([0.0, 0.5, 0.5, 1.0]), (nk, 1))       # bands 1,2 equal
    n_val, n_cond = resolve_band_window(e, 2, 1, 1, where="test")
    out = capsys.readouterr().out
    assert "valence/conduction split" in out and "NOT snapped" in out, (
        f"the midline case must be reported separately — widening cannot fix "
        f"it; got:\n{out}")
    assert (n_val, n_cond) == (1, 1), (
        f"the midline must not be snapped, got ({n_val}, {n_cond})")


# ---------------------------------------------------------------------------
# 7.  the report-only twin used where shapes are already committed
# ---------------------------------------------------------------------------
def test_check_band_window_warns_and_its_red_twin_is_silent_p1(capsys):
    e = _kramers_spectrum(n_pairs=8)
    check_band_window(e, 5, 11, where="test")          # both boundaries split
    out = capsys.readouterr().out
    assert "lower boundary at band 5" in out and "upper boundary at band 11" in out, (
        f"both cut boundaries must be named; got:\n{out}")

    check_band_window(e, 4, 12, where="test")          # RED TWIN: clean
    assert capsys.readouterr().out == "", (
        "RED TWIN DID NOT GO RED: the report-only check fired on a "
        "multiplet-safe window")

    with pytest.raises(BandWindowDegeneracyError):
        check_band_window(e, 5, 11, mode="strict", where="test")
    check_band_window(e, 5, 11, mode="off", where="test")
    assert capsys.readouterr().out == ""

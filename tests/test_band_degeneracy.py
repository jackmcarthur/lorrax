"""The multiplet guard on BSE band-window selection, and its red twins.

Every cell here carries BOTH sides: the split multiplet that must fire the
guard, and the clean window that must pass in silence.  A guard that fires on
everything is the same defect as a guard that fires on nothing, and only the
pair distinguishes them.

WHAT THE DEFAULT IS, AND WHY THESE CELLS SAY SO OUT LOUD.  Every cell below
that calls the guard without a ``mode=`` is asserting the DEFAULT behaviour,
and since the owner's ruling of 2026-08-10 that default is ``strict``: a
window that cuts a multiplet REFUSES, naming the counts that would work.  It
used to be ``snap``, which widened the window instead, and in one day that
silently re-windowed two decks that were being used as measurement standards
(``tests/KNOWN_FAILURES.md``, the ``si_bse_debug`` anchor row, and the
parity deck's false 28.6 meV "regression").  ``snap`` is still here and still
tested — it is now something a run asks for by name, which is why the cells
that exercise it pass ``mode="snap"`` explicitly.

Pure numpy — no jax, no FFI, no deck.  These run anywhere, which is the point:
the guard sits on the path every BSE driver takes, so its gate must never be
the one that gets skipped for want of a GPU.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from common.band_degeneracy import (
    BandWindowDegeneracyError, DEFAULT_MODE, DEGENERACY_TOL_RY,
    boundary_min_gaps, check_band_window, resolve_band_window)

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
# 2.  snap mode — an explicit opt-in since 2026-08-10, no longer the default
# ---------------------------------------------------------------------------
def test_split_kramers_pair_snaps_outward_when_snap_is_asked_for_p1(capsys):
    """GREEN: an odd band count splits a pair and ``snap`` widens it outward.

    ``mode="snap"`` is passed EXPLICITLY, and that spelling is the assertion.
    Widening is opt-in since the owner's 2026-08-10 ruling; if this cell ever
    goes back to relying on the default it stops testing the opt-in and
    starts testing whatever the default happens to be that week.
    """
    e = _kramers_spectrum(n_pairs=8)          # 16 bands, 8 exact pairs
    n_occ = 8                                  # 4 valence pairs

    # n_val=3 puts the valence boundary at band 5, INSIDE the pair (4, 5).
    # n_cond=3 puts the conduction boundary at band 11, inside pair (10, 11).
    n_val, n_cond = resolve_band_window(e, n_occ, 3, 3, mode="snap",
                                        where="test")
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


def test_the_same_split_REFUSES_when_nobody_says_a_mode_p1(capsys):
    """THE RULING, as a cell: the same call with no ``mode=`` must refuse.

    Owner, 2026-08-10 ("do strict"): a window that cuts a degenerate multiplet
    refuses with an actionable message; it never silently widens the
    calculation.  This is the exact call the cell above makes, minus the
    opt-in, and it is the difference between the two that is the ruling.
    """
    e = _kramers_spectrum(n_pairs=8)
    with pytest.raises(BandWindowDegeneracyError) as ei:
        resolve_band_window(e, 8, 3, 3, where="test")
    msg = str(ei.value)
    assert "--n-val 4" in msg and "--n-cond 4" in msg, (
        f"the DEFAULT refusal has to name the counts that would work — that "
        f"is what makes refusing better than widening; got:\n{msg}")
    assert "SNAPPED" not in capsys.readouterr().out, (
        "the default must not widen anything on its way to refusing")


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
# 3.  strict mode — THE DEFAULT since 2026-08-10
# ---------------------------------------------------------------------------
def test_the_default_mode_is_strict_p1():
    """The owner's ruling of 2026-08-10, pinned as one number.

    "do strict": the band-degeneracy guard's default flips from ``snap`` to
    ``strict``, because ``snap`` silently re-windowed two BGW-parity decks in
    one day (``tests/KNOWN_FAILURES.md``: the ``si_bse_debug`` anchor's
    0.0906 eV false regression, and the parity deck's false 28.6 meV one).
    ``snap`` stays available as an explicit opt-in; ``off`` is unchanged.

    If this moves, every driver's ``--band-degeneracy`` default moves with it
    — which is the point of there being one name — and the module docstring,
    both drivers' help text, ``docs/drivers.md`` and the KNOWN_FAILURES row
    all state a default that is no longer true.
    """
    assert DEFAULT_MODE == "strict", (
        f"DEFAULT_MODE is {DEFAULT_MODE!r}; the owner ruled 'strict' on "
        f"2026-08-10 and nothing in this tree records a later ruling")


def test_no_second_default_lurks_anywhere_p1():
    """One authoritative default, spelled once, read everywhere.

    ``snap`` survived a day as an unwanted default because "the default" was
    a bare string literal in three files and flipping it meant finding all of
    them.  Both drivers' argparse defaults, and every keyword default on the
    choke-point loaders, must now be the NAME ``DEFAULT_MODE`` — a literal
    anywhere in that set is a second default waiting to disagree with the
    first.

    SOURCE-LEVEL BY NECESSITY, and that is why it has a partner.  This file
    imports nothing but numpy on purpose (see the module docstring), and
    ``bse.exciton_bands`` cannot be imported without the FFI stack, so the
    same flip is checked through the driver's real argparse in
    ``tests/test_exciton_bands_rerun_default.py`` — the file that already
    owns "a default flip on this driver, pinned so it cannot rot".
    """
    src = Path(__file__).resolve().parents[1] / "src"
    seen, offenders = 0, []

    def _flag(node, what):
        """Record a default that is a bare mode string rather than a name."""
        nonlocal seen
        seen += 1
        if isinstance(node, ast.Constant) and node.value in ("snap", "strict",
                                                             "off"):
            offenders.append(f"{what} -> {node.value!r} (line {node.lineno})")

    for rel in ("common/band_degeneracy.py", "bse/bse_loading.py",
                "bse/bse_window.py", "bse/bse_jax.py",
                "bse/exciton_bands.py"):
        tree = ast.parse((src / rel).read_text())
        for node in ast.walk(tree):
            # keyword defaults on the guard and its choke-point callers
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                pairs = list(zip(args.args[len(args.args)
                                           - len(args.defaults):],
                                 args.defaults))
                pairs += [(a, d) for a, d in zip(args.kwonlyargs,
                                                 args.kw_defaults)
                          if d is not None]
                for a, d in pairs:
                    if a.arg in ("mode", "degeneracy_mode"):
                        _flag(d, f"{rel}::{node.name}({a.arg}=)")
            # argparse: the --band-degeneracy flag's own default
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    and any(isinstance(a, ast.Constant)
                            and a.value == "--band-degeneracy"
                            for a in node.args)):
                for kw in node.keywords:
                    if kw.arg == "default":
                        _flag(kw.value, f"{rel}::--band-degeneracy default=")

    assert seen >= 8, (
        f"only found {seen} mode defaults across the guard and its callers — "
        f"the choke point moved and this cell is now looking at nothing")
    assert not offenders, (
        "a mode default is spelled as a literal instead of DEFAULT_MODE, so "
        "flipping the guard's default would leave this one behind:\n  " +
        "\n  ".join(offenders))


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
    resolve_band_window(e, 2, 1, 1, mode="snap", where="test")
    out = capsys.readouterr().out
    assert "still open at the TOP" in out, (
        f"a multiplet that runs past the last available band must be named; "
        f"got:\n{out}")


def test_gapless_midline_is_reported_and_not_snapped_p1(capsys):
    """A degenerate v/c split is a different bug and gets its own message."""
    nk = 3
    e = np.tile(np.array([0.0, 0.5, 0.5, 1.0]), (nk, 1))       # bands 1,2 equal
    n_val, n_cond = resolve_band_window(e, 2, 1, 1, mode="snap", where="test")
    out = capsys.readouterr().out
    assert "valence/conduction split" in out and "NOT snapped" in out, (
        f"the midline case must be reported separately — widening cannot fix "
        f"it; got:\n{out}")
    assert (n_val, n_cond) == (1, 1), (
        f"the midline must not be snapped, got ({n_val}, {n_cond})")


def test_an_unrepairable_refusal_does_not_offer_a_counts_fix_p1():
    """The default refuses these two as well, and must not fake a fix.

    Both cases above are FOUND by the guard and neither is repairable by
    widening: the midline is the window's own middle, and a multiplet open at
    the top of the input is not completed by any count.  Under the strict
    default the refusal is now the first thing a user sees on such a deck, so
    a ``Fix: use --n-val ...`` line here would send them round the loop —
    they would re-run with the counts it names and get the same refusal.
    """
    nk = 3
    for e, what in (
            (np.tile(np.array([0.0, 0.5, 1.0, 1.0, 1.0]), (nk, 1)), "top"),
            (np.tile(np.array([0.0, 0.5, 0.5, 1.0]), (nk, 1)), "midline")):
        with pytest.raises(BandWindowDegeneracyError) as ei:
            resolve_band_window(e, 2, 1, 1, where="test")
        msg = str(ei.value)
        assert "Fix: use --n-val" not in msg, (
            f"the {what} refusal offers counts that would not fix it:\n{msg}")
        assert "widening cannot clear everything above" in msg, (
            f"the {what} refusal must say that widening is not the answer "
            f"here, or the reader reaches for --band-degeneracy snap and "
            f"gets a window that is still cut; got:\n{msg}")
        assert "nband" in msg or "n_occ" in msg, (
            f"the {what} refusal has to name what DOES have to change; "
            f"got:\n{msg}")


# ---------------------------------------------------------------------------
# 7.  the report-only twin used where shapes are already committed
# ---------------------------------------------------------------------------
def test_check_band_window_warns_and_its_red_twin_is_silent_p1(capsys):
    e = _kramers_spectrum(n_pairs=8)
    # Both boundaries split.  ``snap`` on this twin is a warning by design —
    # there is no window to widen at a seam whose shape is already committed.
    check_band_window(e, 5, 11, mode="snap", where="test")
    out = capsys.readouterr().out
    assert "lower boundary at band 5" in out and "upper boundary at band 11" in out, (
        f"both cut boundaries must be named; got:\n{out}")

    check_band_window(e, 4, 12, mode="snap", where="test")   # RED TWIN: clean
    assert capsys.readouterr().out == "", (
        "RED TWIN DID NOT GO RED: the report-only check fired on a "
        "multiplet-safe window")

    with pytest.raises(BandWindowDegeneracyError):
        check_band_window(e, 5, 11, mode="strict", where="test")
    check_band_window(e, 5, 11, mode="off", where="test")
    assert capsys.readouterr().out == ""


def test_check_band_window_takes_the_same_strict_default_p1(capsys):
    """The report-only twin defaults to refusing too, and its red twin passes.

    One flag, one meaning.  This twin reads ``--band-degeneracy`` from the
    same driver argument as the resolver (``exciton_bands``' htransform
    conduction window, ``bse_io.apply_eqp_and_reslice_bands``' QP-corrected
    spectrum), and a default that refused at one seam while whispering at the
    other would let a user believe a run had checked every boundary when it
    had only checked some.
    """
    e = _kramers_spectrum(n_pairs=8)
    with pytest.raises(BandWindowDegeneracyError):
        check_band_window(e, 5, 11, where="test")            # no mode= given
    check_band_window(e, 4, 12, where="test")                # RED TWIN: clean
    assert capsys.readouterr().out == "", (
        "RED TWIN DID NOT GO RED: the default report-only check fired (or "
        "raised) on a multiplet-safe window")

"""The band extrapolation brackets the **Σ** count.  Never the χ count.

THE CLAIM.  ``sigma_band_extrapolation`` samples the Σ_c band sum at 0.80,
0.90 and 1.00 of the TOTAL band count and fits ``S(N) = S_∞ + A/N``.  Before
the χ/Σ split there was one total and the sentence was unambiguous.  After it
there are three candidate numbers in scope at the call site —
``number_bands_chi``, ``number_bands_sigma``, and the LOADED extent
``max(chi, sigma)`` (which is what ``meta.b_id_4_user`` and
``BandSlices.full`` both carry) — and two of them are wrong.

WHY IT MATTERS, not just as bookkeeping.  On a deck running χ at 248 and Σ at
100, bracketing against the χ count would sample at ~(198, 223, 248): three
points on a partial-sum curve **this run never evaluates**, because the Σ sum
stops at 100.  The brackets would then reach past the last band Σ owns, and
the "extrapolation" would be a fit to a curve read out of the wrong window.
That failure is silent — the run completes, the fit reports an S_∞, and the
diagnostics all look ordinary.

WHY IT IS EASY TO GET WRONG.  ``BandSlices.full`` reads like "all the bands"
and was the Σ band sum for the whole life of the code before 2026-08-16; it
is now the ALLOCATION extent.  ``meta.b_id_4_user`` reads like "the user's
nband" and is now the max.  Either name reaches for the χ count on a
χ-dominant deck without looking wrong.

WHAT THIS FILE PINS.  The planner arithmetic against a Σ count that DIFFERS
from the χ count (§1), and the two production call sites reading the Σ-side
fields rather than the loaded-extent ones (§2) — read out of the source text,
so it fails if someone reverts a call site to ``s.full`` / ``b_id_4_user``
even when no fixture in the suite happens to run a split deck.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from gw.band_extrapolation import (
    BRACKET_FRACTIONS,
    plan_band_brackets,
    trivial_plan,
)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src")

#: The split the owner's motivating measurement describes: χ at the full band
#: count, Σ short and extrapolated.
CHI, SIGMA, N_OCC = 248, 100, 8


def _flat_spectrum(nb, nk=4):
    """Non-degenerate, evenly spaced: every boundary is clean, so the snap
    never moves a cut and the counts are exactly the fractions."""
    return np.tile(np.linspace(1.0, 20.0, nb), (nk, 1))


# ---------------------------------------------------------------------------
# (1) THE PLANNER ARITHMETIC
# ---------------------------------------------------------------------------

def test_the_brackets_are_fractions_of_the_sigma_count():
    """~(80, 90, 100), not ~(198, 223, 248).

    The fixture is the load-bearing part: χ and Σ differ by 148 bands, so a
    planner fed the wrong one is off by more than the whole Σ sum and no
    rounding tolerance can hide it.
    """
    plan = plan_band_brackets(
        enabled=True, enk_ry=_flat_spectrum(SIGMA), n_occ=N_OCC,
        nb_logical=SIGMA, nb_padded=SIGMA, fractions=BRACKET_FRACTIONS)
    assert plan.counts == (80, 90, 100), plan.counts

    chi_rule = tuple(int(round(f * CHI)) for f in BRACKET_FRACTIONS) + (CHI,)
    assert chi_rule == (198, 223, 248)
    assert plan.counts != chi_rule, "brackets must be of the SIGMA count"
    assert max(plan.counts) <= SIGMA, (
        "no bracket may reach past the last band the Sigma sum owns")


def test_the_last_bracket_stops_at_the_sigma_count():
    """N₃ is the band sum the run actually reports, and the last bracket's
    upper bound is the last band Σ sums.  A bound at the loaded extent would
    sum 148 bands of χ-only ψ into Σ."""
    plan = plan_band_brackets(
        enabled=True, enk_ry=_flat_spectrum(SIGMA), n_occ=N_OCC,
        nb_logical=SIGMA, nb_padded=SIGMA)
    assert plan.counts[-1] == SIGMA
    assert plan.bounds[-1][1] == SIGMA
    covered = sum(hi - lo for lo, hi in plan.bounds)
    assert covered == SIGMA, "the brackets must partition [0, N_sigma) exactly"


def test_the_trivial_plan_is_the_sigma_count_too():
    """The default (non-extrapolating) path takes the same seam: one bracket
    over the Σ band sum.  If the trivial plan spanned the loaded extent, a
    split deck's ORDINARY Σ would sum the χ-only bands and the extrapolation
    flag would be the only thing holding the split together."""
    plan = trivial_plan(SIGMA, N_OCC, SIGMA)
    assert plan.bounds == ((0, SIGMA),)
    assert plan.counts == (SIGMA,)
    assert plan.n_cond == SIGMA - N_OCC


def test_the_refusals_are_computed_against_the_sigma_count():
    """``n_cond <= n_occ`` is a statement about the Σ sum.  With χ = 248 and
    Σ = 100 against n_occ = 60, the Σ sum has 40 conduction bands and must
    refuse — a gate that looked at χ would see 188 and sail through."""
    from gw.band_extrapolation import BandExtrapolationRefused
    with pytest.raises(BandExtrapolationRefused) as exc:
        plan_band_brackets(enabled=True, enk_ry=_flat_spectrum(SIGMA),
                           n_occ=60, nb_logical=SIGMA, nb_padded=SIGMA)
    msg = str(exc.value)
    assert "number_bands_sigma" in msg
    assert str(SIGMA) in msg
    assert str(CHI) not in msg, "the chi count is not in scope for this refusal"


# ---------------------------------------------------------------------------
# (2) THE CALL SITES — read out of the source, so a revert fails here
# ---------------------------------------------------------------------------
# A behavioural test needs a split deck to run end to end on a real WFN, which
# no unit fixture in this suite can do.  These two read the production call
# sites' ARGUMENTS out of the AST instead: cheap, exact, and they fail the
# moment someone reaches back for the name that reads like "all the bands".

def _call_kwargs(path, func_name):
    """Every ``func_name(...)`` call in ``path``, as {kwarg: source text}."""
    tree = ast.parse(open(path).read())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name != func_name:
            continue
        out.append({kw.arg: ast.unparse(kw.value) for kw in node.keywords}
                   | {f"_pos{i}": ast.unparse(a)
                      for i, a in enumerate(node.args)})
    return out


#: The names that mean "the loaded extent = max(chi, sigma)".  Any of these in
#: a band-count argument to the planner is the defect this file exists for.
_LOADED_EXTENT_NAMES = ("s.full", "s.nb_full", "wfns.slices.full",
                        "meta.b_id_4_user")


def test_plan_band_brackets_is_called_with_the_sigma_fields():
    calls = _call_kwargs(os.path.join(_SRC, "gw", "ppm_pipeline.py"),
                         "plan_band_brackets")
    assert len(calls) == 1, f"expected exactly one production call, got {calls}"
    kw = calls[0]
    for arg in ("enk_ry", "nb_logical", "nb_padded"):
        src = kw[arg]
        assert "sigma" in src, (
            f"plan_band_brackets({arg}=...) must read a SIGMA-side field, "
            f"got {src!r}")
    assert kw["nb_logical"] == \
        "int(meta.b_id_4_sigma_user or s.b4) - int(s.b0)", kw["nb_logical"]
    assert kw["nb_padded"] == "int(s.nb_sigma_sum)", kw["nb_padded"]
    assert kw["enk_ry"] == "np.asarray(wfns.enk[:, s.sigma_sum])", kw["enk_ry"]


def test_trivial_plan_is_called_with_the_sigma_fields():
    calls = _call_kwargs(os.path.join(_SRC, "gw", "ppm_sigma.py"),
                         "trivial_plan")
    assert len(calls) == 1, f"expected exactly one production call, got {calls}"
    src = " ".join(calls[0].values())
    assert "nb_sigma_sum" in src and "b_id_4_sigma_user" in src, src
    assert "s.nb_full" not in src, (
        "the trivial plan must span the Sigma band sum, not the loaded extent")


@pytest.mark.parametrize("rel,func", [
    ("gw/ppm_pipeline.py", "plan_band_brackets"),
    ("gw/ppm_sigma.py", "trivial_plan"),
])
def test_no_loaded_extent_name_reaches_the_planner(rel, func):
    """The negative form, stated over the whole family of wrong names."""
    for kw in _call_kwargs(os.path.join(_SRC, rel), func):
        for arg, src in kw.items():
            for bad in _LOADED_EXTENT_NAMES:
                assert bad not in src, (
                    f"{rel}: {func}({arg}={src!r}) reads {bad!r}, which is "
                    f"max(chi, sigma) — the extrapolation would bracket a "
                    f"curve this run never evaluates on a split deck")

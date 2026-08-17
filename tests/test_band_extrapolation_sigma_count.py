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

WHY NO DOWNSTREAM CHECK CAN CATCH IT (2026-08-16).  The OLS fit is solved in
``x = 1/N`` and its coefficients depend only on the RATIOS of the abscissae —
and the sampling fractions are the same 0.80/0.90/1.00 of whichever count is
used, so the ratios are the same too.  Measured:

    counts (80, 90, 100)    -> c = [-4.295082, +0.663934, +4.631148]
    counts (198, 223, 248)  -> c = [-4.254729, +0.663885, +4.590844]

0.94 % apart, ``sum(c) == 1`` in both.  A run that brackets the wrong count
applies a nearly-correct operator to the wrong three partial sums, produces an
exactly Hermitian Σ (``c`` is real), converges, and prints entirely ordinary
numbers.  ``trust_verdict`` cannot see it either: it inspects the fit's own
residual structure, which is self-consistent on the wrong curve.

WHAT THIS FILE PINS.  The planner arithmetic against a Σ count that DIFFERS
from the χ count (§1); the two production call sites reading the Σ-side
fields rather than the loaded-extent ones (§2) — read out of the source text,
so it fails if someone reverts a call site to ``s.full`` / ``b_id_4_user``
even when no fixture in the suite happens to run a split deck; and the RUNTIME
refusal that fires when the partition and the abscissae disagree (§3), which
is the part that still holds if someone rewires the plan in a way the AST pins
do not spell out.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from gw.band_extrapolation import (
    BRACKET_FRACTIONS,
    BandBracketCountMismatch,
    BandBracketPlan,
    BandExtrapolationRefused,
    assert_brackets_match_ols_abscissae,
    extrapolation_weights,
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


# ---------------------------------------------------------------------------
# (3) THE RUNTIME REFUSAL
# ---------------------------------------------------------------------------
# The AST pins above fail on the two rewires we can NAME.  This section pins
# the property itself, so a rewire spelled some other way still stops the run.

class _Slices:
    """The two fields the refusal reads off ``BandSlices``, plus the two
    wrong-reach names it reports by name when it recognises them."""

    def __init__(self, nb_sigma_sum, nb_full=None, nb_chi_sum=None, b0=0):
        self.nb_sigma_sum = nb_sigma_sum
        self.nb_full = nb_full if nb_full is not None else nb_sigma_sum
        self.nb_chi_sum = nb_chi_sum if nb_chi_sum is not None else nb_sigma_sum
        self.b0 = b0


class _Meta:
    def __init__(self, b_id_4_sigma_user):
        self.b_id_4_sigma_user = b_id_4_sigma_user


def _sigma_plan():
    """The correct plan for the χ=248 / Σ=100 deck."""
    return plan_band_brackets(
        enabled=True, enk_ry=_flat_spectrum(SIGMA), n_occ=N_OCC,
        nb_logical=SIGMA, nb_padded=SIGMA)


def _chi_plan():
    """What a planner rewired to the χ count / loaded extent produces."""
    return plan_band_brackets(
        enabled=True, enk_ry=_flat_spectrum(CHI), n_occ=N_OCC,
        nb_logical=CHI, nb_padded=CHI)


def test_the_correct_plan_passes():
    assert_brackets_match_ols_abscissae(
        _sigma_plan(), _Slices(SIGMA, nb_full=CHI, nb_chi_sum=CHI),
        meta=_Meta(SIGMA))


def test_the_trivial_plan_passes():
    """The default path takes the same seam and must not be refused."""
    assert_brackets_match_ols_abscissae(
        trivial_plan(SIGMA, N_OCC, SIGMA),
        _Slices(SIGMA, nb_full=CHI, nb_chi_sum=CHI), meta=_Meta(SIGMA))


def test_a_plan_rewired_to_the_loaded_extent_is_refused():
    """THE HEADLINE.  χ=248, Σ=100: a plan built from max(chi, sigma) brackets
    (198, 223, 248) — a curve this run never evaluates — and every weight-level
    diagnostic would look ordinary.  It must not reach the kernel."""
    with pytest.raises(BandBracketCountMismatch) as exc:
        assert_brackets_match_ols_abscissae(
            _chi_plan(), _Slices(SIGMA, nb_full=CHI, nb_chi_sum=CHI),
            meta=_Meta(SIGMA))
    msg = str(exc.value)
    # Both counts, named.
    assert "248" in msg and "100" in msg, msg
    # Both sources, named — the one that is right and the one that is wrong.
    assert "number_bands_sigma" in msg, msg
    assert "number_bands_chi" in msg, msg
    assert "nb_full" in msg or "b_id_4_user" in msg, msg
    # ...and it says raising the chi count does not help, like the others.
    assert "will not clear this" in msg, msg


def test_a_plan_rewired_to_the_chi_count_is_refused():
    """Same rewire, a deck where χ is not the loaded max."""
    with pytest.raises(BandBracketCountMismatch):
        assert_brackets_match_ols_abscissae(
            _chi_plan(), _Slices(SIGMA, nb_full=CHI + 8, nb_chi_sum=CHI),
            meta=_Meta(SIGMA))


def test_abscissae_that_are_not_the_partition_are_refused():
    """The other half of the property: the partition may be right while the
    abscissae are not.  Nothing downstream distinguishes these, because the
    fit never sees the bounds and the kernel never sees the counts."""
    good = _sigma_plan()
    bad = BandBracketPlan(
        bounds=good.bounds,
        counts=tuple(int(round(c * CHI / SIGMA)) for c in good.counts),
        requested=good.requested, n_occ=good.n_occ, n_cond=good.n_cond,
        mean_energy_ev=good.mean_energy_ev, enabled=True)
    with pytest.raises(BandBracketCountMismatch):
        assert_brackets_match_ols_abscissae(
            bad, _Slices(SIGMA, nb_full=CHI, nb_chi_sum=CHI),
            meta=_Meta(SIGMA))


def test_a_gapped_partition_is_refused():
    """A partition with a hole does not deliver S(N_i) at all."""
    good = _sigma_plan()
    lo, hi = good.bounds[1]
    holed = good.bounds[:1] + ((lo + 1, hi),) + good.bounds[2:]
    bad = BandBracketPlan(
        bounds=holed, counts=good.counts, requested=good.requested,
        n_occ=good.n_occ, n_cond=good.n_cond,
        mean_energy_ev=good.mean_energy_ev, enabled=True)
    with pytest.raises(BandBracketCountMismatch):
        assert_brackets_match_ols_abscissae(
            bad, _Slices(SIGMA), meta=_Meta(SIGMA))


def test_the_refusal_is_a_band_extrapolation_refusal():
    """Subclassing is load-bearing: callers that mean 'the extrapolation
    cannot run here' catch the base class."""
    assert issubclass(BandBracketCountMismatch, BandExtrapolationRefused)


def test_the_weights_really_are_ratio_invariant():
    """The measurement the refusal's existence rests on.  If this ever stops
    being true, the refusal is no longer the ONLY thing that can catch a
    wrong-count run, and the message above should be revisited."""
    a = extrapolation_weights((80, 90, 100))
    b = extrapolation_weights((198, 223, 248))
    assert np.isclose(a.sum(), 1.0) and np.isclose(b.sum(), 1.0)
    # Under 1 % apart at every coefficient — i.e. a wrong-count fit is a
    # nearly-correct operator applied to the wrong partial sums.
    assert np.max(np.abs(a - b) / np.abs(a)) < 0.01
    assert np.allclose(a, [-4.295082, 0.663934, 4.631148], atol=1e-6)
    assert np.allclose(b, [-4.254729, 0.663885, 4.590844], atol=1e-6)


def test_both_production_sites_call_the_refusal():
    """The refusal is worthless if it is not wired in.  Read out of the
    source, beside the AST pins above and for the same reason."""
    for rel in ("gw/ppm_pipeline.py", "gw/ppm_sigma.py"):
        calls = _call_kwargs(os.path.join(_SRC, rel),
                             "assert_brackets_match_ols_abscissae")
        assert len(calls) == 1, (
            f"{rel} must call the bracket/abscissae refusal exactly once, "
            f"got {calls}")
